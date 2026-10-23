//! Captured, lossless endpoint exchange. Physical MD and observer boundaries
//! are unchanged. All native producers/recipients rendezvous once; a cached
//! parent-owned WHILE graph drains actual source counts without host batches.
use super::*;
use crate::governor_device::{CudaOwned, CudaRetirementLease};
use crate::graph_capture::{capture_pure_template, AutonomousGraph, CapturedTemplate};
use cudarc::driver::sys;
use std::sync::{Condvar, Mutex};
use std::time::Duration;

/// Existing allocation owners expose borrowed arrays; this trait never
/// transfers, aliases, or substitutes their physical storage.
pub trait EndpointBuffers {
    fn endpoint_stream(&self) -> Arc<CudaStream>;
    fn endpoint_source(&self) -> Result<(&CudaSlice<u8>, &CudaSlice<i32>)>;
    fn endpoint_target(
        &mut self,
    ) -> Result<(&mut CudaSlice<f32>, &CudaSlice<f32>, &mut CudaSlice<u8>)>;
    fn endpoint_grid(&self) -> Result<((i32, i32, i32), (f32, f32, f32), f32)>;
}
impl EndpointBuffers for crate::persistent_engine::PersistentNhsEngine {
    fn endpoint_stream(&self) -> Arc<CudaStream> {
        self.cuda_stream().clone()
    }
    fn endpoint_source(&self) -> Result<(&CudaSlice<u8>, &CudaSlice<i32>)> {
        Ok((
            self.spike_buffer_gpu().context("captured source absent")?,
            self.spike_count_gpu()
                .context("captured source count absent")?,
        ))
    }
    fn endpoint_target(
        &mut self,
    ) -> Result<(&mut CudaSlice<f32>, &CudaSlice<f32>, &mut CudaSlice<u8>)> {
        self.coupling_buffers_mut()
            .context("captured recipient buffers absent")
    }
    fn endpoint_grid(&self) -> Result<((i32, i32, i32), (f32, f32, f32), f32)> {
        let (x, y, z, ox, oy, oz, spacing) =
            self.grid_info().context("captured recipient grid absent")?;
        Ok(((x, y, z), (ox, oy, oz), spacing))
    }
}

struct WorkerTemplates {
    fingerprint: Vec<u64>,
    source_binding: [u64; 5],
    stream: Arc<CudaStream>,
    publish: CapturedTemplate,
    begin: Vec<Option<CapturedTemplate>>,
    accumulate: Vec<Option<CapturedTemplate>>,
    finalize: Vec<Option<CapturedTemplate>>,
    pending_ptr: u64,
    max_batches: u32,
}
/// Exact prior-endpoint counts; the future source counter is never used to
/// validate an already-consumed archive after the next physical window.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ArchivedEndpointReceipt {
    pub endpoint_step: u32,
    pub epoch: u64,
    pub archived_counts: Vec<u32>,
    pub reset_before_next_physics: bool,
}

struct EpochResetResources {
    sources: CudaSlice<u64>,
    rings: CudaSlice<u64>,
    windows: CudaSlice<u64>,
    expected: CudaSlice<u32>,
    receipts: CudaSlice<u32>,
    kernel: CudaFunction,
    n_sources: u32,
    n_rings: u32,
    n_windows: u32,
}
impl EpochResetResources {
    fn stage_expected(
        &mut self,
        stream: &Arc<CudaStream>,
        value: &ArchivedEndpointReceipt,
    ) -> Result<()> {
        anyhow::ensure!(
            value.archived_counts.len() == self.n_sources as usize,
            "archived epoch requires every actual source population"
        );
        let mut expected = vec![
            value.endpoint_step,
            value.epoch as u32,
            (value.epoch >> 32) as u32,
            value.reset_before_next_physics as u32,
        ];
        expected.extend_from_slice(&value.archived_counts);
        stream.memcpy_htod(&expected, &mut self.expected)?;
        Ok(())
    }

    fn append_into(
        &mut self,
        stream: &Arc<CudaStream>,
        parent: &mut CapturedTemplate,
        destination: sys::CUgraph,
        dependencies: &[sys::CUgraphNode],
    ) -> Result<sys::CUgraphNode> {
        let guard = capture_pure_template(stream.clone(), |_| {
            unsafe {
                stream
                    .launch_builder(&self.kernel)
                    .arg(&self.sources)
                    .arg(&self.n_sources)
                    .arg(&self.rings)
                    .arg(&self.n_rings)
                    .arg(&self.windows)
                    .arg(&self.n_windows)
                    .arg(&self.expected)
                    .arg(&mut self.receipts)
                    .launch(LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
            Ok(())
        })?;
        parent.retain_child_owners(&guard);
        Ok(unsafe {
            crate::graph_node::add_child_graph_node_v3(destination, dependencies, guard.cu_graph())
        }?)
    }

    fn verify_receipts(
        &self,
        stream: &Arc<CudaStream>,
        expected: &ArchivedEndpointReceipt,
    ) -> Result<Vec<[u32;8]>> {
        let mut values = vec![0u32; self.receipts.len()];
        stream.memcpy_dtoh(&self.receipts, &mut values)?;
        stream.synchronize()?;
        anyhow::ensure!(
            expected.archived_counts.len() == self.n_sources as usize && values.last() == Some(&0),
            "captured previous-epoch handoff failed"
        );
        for (id, &count) in expected.archived_counts.iter().enumerate() {
            let wanted = [
                expected.endpoint_step,
                expected.endpoint_step,
                count,
                count,
                count,
                0,
                expected.epoch as u32,
                (expected.epoch >> 32) as u32,
            ];
            anyhow::ensure!(
                values[id * 8..id * 8 + 8] == wanted,
                "captured prior-epoch receipt mismatch for physical producer {id}"
            );
        }
        Ok(values[..values.len()-1].chunks_exact(8).map(|row|row.try_into().unwrap()).collect())
    }
}

struct EndpointGraph {
    // Drop graph before any allocation its kernels reference.
    graph: AutonomousGraph,
    handoff: Option<EpochResetResources>,
    _iterations: Vec<CudaSlice<u32>>,
}
#[cfg(feature = "v2_ignition")]
struct SharedParent {
    // Actual executable must retire before every buffer/template it references.
    graph: Box<AutonomousGraph>,
    handoff: EpochResetResources,
    coupling_enabled: CudaSlice<u32>,
    _outer_iterations: CudaSlice<u32>,
    _iterations: Vec<CudaSlice<u32>>,
}
/// Two native retirement boundaries demonstrated by the local-to-shared path.
/// Slots are fixed by lifecycle, never by campaign length or replay count.
#[cfg(feature = "v2_ignition")]
#[derive(Clone,Copy,Debug)]
pub enum LocalObservationRetirementBoundary { LocalRebind, SharedActivation }
#[cfg(feature = "v2_ignition")]
impl LocalObservationRetirementBoundary {
    fn index(self)->usize {match self {Self::LocalRebind=>0,Self::SharedActivation=>1}}
}
#[cfg(feature = "v2_ignition")]
struct SharedResidentState {
    parent: Option<SharedParent>,
    // Shared graph destruction precedes these bounded predecessor roots.
    // Each root owns its original exec/template, predicates and V2 leases.
    retired_local_roots: Vec<[Option<crate::persistent_engine::BoundResidentObservationGraph>;2]>,
    released_local_root_count: usize,
    retirement_completed: bool,
    prepared: Vec<Option<crate::persistent_engine::PreparedResidentObservation>>,
    archived: Vec<Option<(u32, u32)>>,
    activation_step: Option<u32>,
    windows: u64,
    final_drained: bool,
    last_handoff: Option<(ArchivedEndpointReceipt,Vec<[u32;8]>)>,
    continuation_rebind_pending: Vec<bool>,
    prior_process_retired_roots: Vec<RetiredOwnerSnapshotV1>,
}

struct State {
    #[cfg(feature = "v2_ignition")]
    shared: SharedResidentState,
    graph: Option<EndpointGraph>,
    workers: Vec<Option<WorkerTemplates>>,
    ready: Vec<bool>,
    steps: Vec<Option<u32>>,
    epoch: u64,
    launch_count: u64,
    cancelled: Option<String>,
    reconstruction_history: Vec<serde_json::Value>,
}
// SAFETY: captured handles are uniquely owned by State, accessed exclusively
// under its mutex, and every access/destruction binds the shared CUDA context.
// Engine/publication CudaOwned owners retain this coordinator's retirement
// lease BEFORE registering any template; no pointee can drop until retirement
// prevents launches, fences the shared stream and destroys all graph handles.
// Failure of that fence invokes CudaOwned's existing process-life quarantine.
unsafe impl Send for State {}

/// Capability minted only while the coordinator owns the exclusive rendezvous
/// lock and all producers have fenced. It cannot outlive that locked State or
/// authorize a different graph. Ordinary graphs retain their own-stream rule.
pub(crate) struct CoordinatorLaunchPermit<'a> {
    coordinator: &'a CapturedEndpointCoordinator,
    state: &'a State,
    graph: &'a AutonomousGraph,
}
impl CoordinatorLaunchPermit<'_> {
    pub(crate) fn verify_graph(
        &self,
        graph: &AutonomousGraph,
        stream: &Arc<CudaStream>,
    ) -> Result<()> {
        anyhow::ensure!(
            std::ptr::eq(self.graph, graph)
                && Arc::ptr_eq(stream, &self.coordinator.stream)
                && self.state.cancelled.is_none()
                && self.state.ready.iter().all(|ready| *ready)
                && self.state.workers.iter().all(Option::is_some),
            "shared-context launch lacks the exact live coordinator rendezvous capability"
        );
        for worker in self.state.workers.iter().flatten() {
            anyhow::ensure!(
                Arc::ptr_eq(worker.stream.context(), stream.context()),
                "shared parent includes a different CUDA context owner"
            );
        }
        Ok(())
    }
    pub(crate) fn contains_owner_stream(&self, owns: impl Fn(&CudaStream) -> bool) -> bool {
        self.state
            .workers
            .iter()
            .flatten()
            .any(|worker| owns(&worker.stream))
    }
}

/// Allocation identity is stable for the coordinator lifetime. Owners must not
/// replace registered arrays through `CudaOwned::DerefMut`; the retirement lease
/// protects owner destruction, not replacement of fields inside a live owner.
/// Every endpoint calls `prepare_worker` before `exchange` so pointer/grid
/// fingerprints and producer-stream completion are checked before graph reuse.
pub struct CapturedEndpointCoordinator {
    stream: Arc<CudaStream>,
    rings: Vec<Arc<Mutex<TwinRingBuffer>>>,
    n_streams: usize,
    engines_per_group: usize,
    state: Mutex<State>,
    changed: Condvar,
}

impl CapturedEndpointCoordinator {
    pub fn new(
        context: &Arc<CudaContext>,
        rings: Vec<Arc<Mutex<TwinRingBuffer>>>,
        n_streams: usize,
    ) -> Result<Arc<Self>> {
        anyhow::ensure!(
            n_streams >= 4 && n_streams % 4 == 0 && rings.len() == 4,
            "captured lossless exchange requires four complete physical source groups"
        );
        let epg = n_streams / 4;
        for (group, ring) in rings.iter().enumerate() {
            let ring = ring
                .lock()
                .map_err(|_| anyhow::anyhow!("coupling ring poisoned"))?;
            let state = ring
                .lossless
                .as_ref()
                .context("captured endpoint requires lossless ring policy")?;
            let expected: Vec<_> = (0..n_streams).filter(|id| id / epg != group).collect();
            anyhow::ensure!(
                state.participating == expected,
                "captured actual recipient population mismatch"
            );
        }
        Ok(Arc::new(Self {
            stream: context.new_stream()?,
            rings,
            n_streams,
            engines_per_group: epg,
            state: Mutex::new(State {
                #[cfg(feature = "v2_ignition")]
                shared: SharedResidentState {
                    parent: None,
                    retired_local_roots: (0..n_streams).map(|_| [None,None]).collect(),
                    released_local_root_count: 0,
                    retirement_completed: false,
                    prepared: (0..n_streams).map(|_| None).collect(),
                    archived: vec![None; n_streams],
                    activation_step: None,
                    windows: 0,
                    final_drained: false,
                    last_handoff: None,
                    continuation_rebind_pending: vec![false;n_streams],
                    prior_process_retired_roots: Vec::new(),
                },
                graph: None,
                workers: (0..n_streams).map(|_| None).collect(),
                ready: vec![false; n_streams],
                steps: vec![None; n_streams],
                epoch: 0,
                launch_count: 0,
                cancelled: None,
                reconstruction_history: Vec::new(),
            }),
            changed: Condvar::new(),
        }))
    }

    /// Must be installed when the engine owner is created, before any worker
    /// can fail initialization. An early exit cancels waiting peers too.
    pub fn bind_owner<T>(self: &Arc<Self>, owner: &mut CudaOwned<T>) {
        owner.retain_retirement(self.clone());
    }

    pub fn prepare_worker<T: EndpointBuffers>(
        self: &Arc<Self>,
        id: usize,
        owner: &mut CudaOwned<T>,
        publication_owner: &mut CudaOwned<Option<TwinSpikePublication>>,
    ) -> Result<()> {
        anyhow::ensure!(id < self.n_streams, "unregistered captured worker");
        self.bind_owner(owner);
        self.bind_owner(publication_owner);
        let stream = owner.endpoint_stream();
        anyhow::ensure!(
            Arc::ptr_eq(stream.context(), self.stream.context()),
            "captured workers must share the exact CUDA context owner"
        );
        // One physical-boundary fence replaces every per-batch worker fence.
        stream.synchronize()?;
        let publication = publication_owner
            .as_mut()
            .context("source publication owner absent")?;
        let (events, count) = owner.endpoint_source()?;
        publication.check_source(&stream, events, count)?;
        let mut fingerprint = vec![
            events.device_ptr(&stream).0,
            count.device_ptr(&stream).0,
            publication.cursor.device_ptr(&stream).0,
            publication.continuation_pending.device_ptr(&stream).0,
        ];
        let source_failure_ptr = publication.failure.device_ptr(&stream).0;
        let grid = owner.endpoint_grid()?;
        let (thresholds, base, protocol) = owner.endpoint_target()?;
        fingerprint.extend([
            thresholds.device_ptr(&stream).0,
            base.device_ptr(&stream).0,
            protocol.device_ptr(&stream).0,
            thresholds.len() as u64,
            base.len() as u64,
            protocol.len() as u64,
            grid.0 .0 as u64,
            grid.0 .1 as u64,
            grid.0 .2 as u64,
            grid.1 .0.to_bits() as u64,
            grid.1 .1.to_bits() as u64,
            grid.1 .2.to_bits() as u64,
            grid.2.to_bits() as u64,
        ]);
        let source_binding = [
            fingerprint[1],
            fingerprint[2],
            source_failure_ptr,
            fingerprint[6],
            publication.source_capacity as u64,
        ];
        fingerprint.push(source_failure_ptr);
        {
            let state = self
                .state
                .lock()
                .map_err(|_| anyhow::anyhow!("captured coordinator poisoned"))?;
            anyhow::ensure!(
                state.cancelled.is_none(),
                "captured coordinator cancelled: {:?}",
                state.cancelled
            );
            if let Some(old) = &state.workers[id] {
                anyhow::ensure!(
                    old.fingerprint == fingerprint,
                    "captured worker allocation/geometry identity changed"
                );
                return Ok(());
            }
        }
        let group = id / self.engines_per_group;
        let pending_ptr = publication.continuation_pending.device_ptr(&stream).0;
        let ring_capacity = self.rings[group]
            .lock()
            .map_err(|_| anyhow::anyhow!("ring poisoned"))?
            .capacity;
        let max_batches = publication.source_capacity.div_ceil(ring_capacity).max(1);
        let publish = capture_pure_template(stream.clone(), |_| {
            let (events, count) = owner.endpoint_source()?;
            self.rings[group]
                .lock()
                .map_err(|_| anyhow::anyhow!("ring poisoned"))?
                .push_device_delta_captured(&stream, events, count, publication)
        })?;
        let mut begin: Vec<_> = (0..4).map(|_| None).collect();
        let mut accumulate: Vec<_> = (0..4).map(|_| None).collect();
        let mut finalize: Vec<_> = (0..4).map(|_| None).collect();
        for source_group in 0..4 {
            if source_group == group {
                continue;
            }
            for (operation, destination) in
                [(0, &mut begin), (1, &mut accumulate), (2, &mut finalize)]
            {
                destination[source_group] = Some(capture_pure_template(stream.clone(), |_| {
                    let (thresholds, base, protocol) = owner.endpoint_target()?;
                    self.rings[source_group]
                        .lock()
                        .map_err(|_| anyhow::anyhow!("ring poisoned"))?
                        .adapt_lossless_window(
                            &stream, id, operation, thresholds, base, grid.0, grid.1, grid.2, 0.3,
                            0.2, 500., protocol,
                        )
                })?);
            }
        }
        let worker = WorkerTemplates {
            fingerprint,
            source_binding,
            stream,
            publish,
            begin,
            accumulate,
            finalize,
            pending_ptr,
            max_batches,
        };
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("captured coordinator poisoned"))?;
        anyhow::ensure!(
            state.cancelled.is_none() && state.workers[id].is_none(),
            "captured registration cancelled or repeated"
        );
        state.workers[id] = Some(worker);
        Ok(())
    }

    #[cfg(feature = "v2_ignition")]
    fn build_shared_parent(&self, state: &mut State) -> Result<SharedParent> {
        anyhow::ensure!(
            state.shared.prepared.iter().all(Option::is_some),
            "shared physical parent lacks a prepared physical worker"
        );
        self.stream.context().bind_to_thread()?;
        let mut iterations = Vec::with_capacity(self.n_streams);
        for _ in 0..self.n_streams {
            iterations.push(self.stream.alloc_zeros::<u32>(1)?);
        }
        let mut handoff = self.prepare_epoch_reset(state)?;
        let coupling_enabled = self.stream.alloc_zeros::<u32>(1)?;
        let outer_iterations = self.stream.alloc_zeros::<u32>(1)?;
        self.stream.synchronize()?;
        let mut raw: sys::CUgraph = std::ptr::null_mut();
        let rc = unsafe { sys::cuGraphCreate(&mut raw, 0) };
        anyhow::ensure!(
            rc == sys::CUresult::CUDA_SUCCESS,
            "shared parent create {rc:?}"
        );
        let mut parent = CapturedTemplate::from_capture(
            raw,
            std::collections::HashMap::new(),
            self.stream.clone(),
        );
        let pending_ptr = coupling_enabled.device_ptr(&self.stream).0;
        let outer_iter_ptr = outer_iterations.device_ptr(&self.stream).0;
        let reset_outer =
            unsafe { crate::graph_capture::memset_node_u32(raw, &[], outer_iter_ptr, 0) }?;
        let outer_handle = unsafe { crate::graph_node::create_while_handle(raw, 0) }?;
        // The first predicate call initializes this one-iteration ownership
        // wrapper; the second terminates it. Its counter is not an MD clock.
        let outer_predicate = capture_pure_template(self.stream.clone(), |_| unsafe {
            crate::graph_capture::launch_while_predicate_bridge(
                self.stream.cu_stream(),
                pending_ptr,
                outer_iter_ptr,
                outer_handle,
                2,
            )
        })?;
        let entry = parent.add_child_graph_node(&[reset_outer], outer_predicate.cu_graph())?;
        let (exchange, body) =
            unsafe { crate::graph_node::add_while_node_v3(raw, &[entry], outer_handle) }?;
        let frontier = self.append_coupling_nodes(state, &mut parent, body, &[], &iterations)?;
        let guard = handoff.append_into(&self.stream, &mut parent, body, &frontier)?;
        unsafe {
            crate::graph_node::add_child_graph_node_v3(body, &[guard], outer_predicate.cu_graph())
        }?;
        // Diagnostic-only control retains every owner, physical step and observer.
        // Production always retains its existing parallel cell dependencies.
        #[cfg(test)]
        let serial_physical = std::env::var("PRISM_SHARED_PHYSICAL_TEST_SERIAL_BRANCHES").as_deref()==Ok("1");
        #[cfg(not(test))]
        let serial_physical = false;
        let mut physical_dependencies=vec![exchange];
        for prepared in state.shared.prepared.iter_mut().flatten() {
            let completed=prepared.append_into(&mut parent, &physical_dependencies)?;
            if serial_physical {physical_dependencies=vec![completed];}
        }
        Ok(SharedParent {
            graph: Box::new(parent.instantiate()?),
            handoff,
            coupling_enabled,
            _outer_iterations: outer_iterations,
            _iterations: iterations,
        })
    }

    /// Register a source-bound prepared factory. Activation is latched only by
    /// the existing all-worker endpoint rendezvous, preventing mixed old/new
    /// dispatch while cells finish their native cold holds at different steps.
    #[cfg(feature = "v2_ignition")]
    pub fn register_resident_worker(
        &self,
        id: usize,
        prepared: crate::persistent_engine::PreparedResidentObservation,
    ) -> Result<()> {
        anyhow::ensure!(id < self.n_streams, "unknown shared physical worker");
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("coordinator poisoned"))?;
        anyhow::ensure!(
            state.cancelled.is_none()
                && state.shared.activation_step.is_none()
                && state.shared.prepared[id].is_none(),
            "shared physical factory registration unavailable"
        );
        let worker = state.workers[id]
            .as_ref()
            .context("prepare physical allocation owner before registration")?;
        anyhow::ensure!(
            prepared.source_stream_id() as usize == id
                && prepared.raw_stream() == worker.stream.cu_stream(),
            "shared factory source/stream identity mismatch"
        );
        if let Some(other) = state.shared.prepared.iter().flatten().next() {
            anyhow::ensure!(
                other.cadence() == prepared.cadence(),
                "shared worker observation cadences disagree"
            );
        }
        state.shared.prepared[id] = Some(prepared);
        self.changed.notify_all();
        Ok(())
    }

    /// Successful transfer is host ownership only: no CUDA calls, replay,
    /// capture or evidence readback. Two fixed slots cover the native lifecycle.
    /// Validation precedes Option::take. A rejected handoff after activation
    /// or cancellation retires global users before the caller may unwind.
    #[cfg(feature = "v2_ignition")]
    pub fn retain_local_observation_root(&self,id:usize,
        boundary:LocalObservationRetirementBoundary,
        root:&mut Option<crate::persistent_engine::BoundResidentObservationGraph>)->Result<()> {
        let Some(local)=root.as_ref() else {return Ok(());};
        let locked=self.state.lock();
        let poisoned=locked.is_err();
        let mut state=locked.unwrap_or_else(|poison|poison.into_inner());
        let retire_on_error=poisoned || state.cancelled.is_some() || state.shared.parent.is_some();
        let result=(||->Result<()> {
            anyhow::ensure!(!poisoned,"coordinator poisoned during local graph handoff");
            anyhow::ensure!(id<self.n_streams,"retired local graph has unknown physical stream");
            let slot=boundary.index();
            anyhow::ensure!(state.cancelled.is_none() && !state.shared.retirement_completed
                && state.shared.retired_local_roots[id][slot].is_none(),
                "local graph retirement boundary unavailable or already retained");
            let worker=state.workers[id].as_ref().context("local graph owner was not registered")?;
            anyhow::ensure!(local.pipeline().md_stream_raw()==worker.stream.cu_stream(),
                "retired local graph belongs to another physical stream");
            match boundary {
                LocalObservationRetirementBoundary::LocalRebind=> {
                    anyhow::ensure!(state.shared.activation_step.is_none(),
                        "local rebind cannot follow shared activation");
                    // Step101 may intentionally replace its V2 allocation owner.
                    // Preserve that original Arc without asserting final-owner identity.
                }
                LocalObservationRetirementBoundary::SharedActivation=> {
                    anyhow::ensure!(state.shared.activation_step.is_some() && state.shared.parent.is_some(),
                        "shared graph must own its executable before local handoff");
                    let prepared=state.shared.prepared[id].as_ref().context("shared physical worker absent")?;
                    anyhow::ensure!(Arc::ptr_eq(local.pipeline(),prepared.pipeline()?)
                        && local.steps_per_launch()==prepared.cadence(),
                        "shared handoff changed V2 owner or native observation cadence");
                }
            }
            Ok(())
        })();
        if result.is_ok() {
            state.shared.retired_local_roots[id][boundary.index()]=root.take();
        }
        drop(state);
        if let Err(error)=result {
            if retire_on_error {
                // Never recursively acquire state while holding its mutex.
                // Existing lease retirement prevents future launches and fences
                // all shared users before releasing any dependent allocation.
                if let Err(rc)=self.retire() {
                    // On failed fencing the caller must not destroy this input
                    // root before its external CudaOwned leases quarantine too.
                    std::mem::forget(root.take());
                    return Err(error).context(format!("local graph handoff retirement failed rc={rc}; input graph quarantined"));
                }
            }
            return Err(error);
        }
        Ok(())
    }

    #[cfg(feature = "v2_ignition")]
    pub fn shared_resident_active(&self) -> Result<bool> {
        let state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("coordinator poisoned"))?;
        anyhow::ensure!(state.cancelled.is_none(), "coordinator cancelled");
        Ok(state.shared.activation_step.is_some())
    }

    #[cfg(feature = "v2_ignition")]
    pub fn shared_resident_pipeline(
        &self,
        id: usize,
    ) -> Result<Arc<crate::captured_pipeline::CapturedAdjudicationPipeline>> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("coordinator poisoned"))?;
        anyhow::ensure!(
            id < self.n_streams
                && state.cancelled.is_none()
                && state.shared.activation_step.is_some(),
            "shared parent not activated"
        );
        if state.shared.parent.is_none() {
            state.shared.parent = Some(self.build_shared_parent(&mut state)?);
        }
        Ok(state.shared.prepared[id]
            .as_ref()
            .context("shared worker absent")?
            .pipeline()?
            .clone())
    }

    /// Read the already-instantiated shared graph while its owning mutex is held.
    /// Never constructs, replays or synchronizes a graph for the diagnostic.
    #[cfg(all(test,feature="v2_ignition"))]
    pub(crate) fn neighbor_arguments_for_test(&self) -> Result<serde_json::Value> {
        let state=self.state.lock().map_err(|_|anyhow::anyhow!("coordinator poisoned"))?;
        let parent=state.shared.parent.as_ref().context("actual shared graph is not built")?;
        crate::graph_capture::neighbor_arguments_for_test(parent.graph.cu_graph_template()
            .context("shared graph has no actual template")?)
    }

    /// Store the already archived current population after observers/controller.
    /// No reset or publication occurs here; the next shared physical launch owns
    /// that previous-epoch handoff. Final closeout explicitly drains without MD.
    #[cfg(feature = "v2_ignition")]
    pub fn defer_archived_endpoint(&self, id: usize, step: u32, archived: usize) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("coordinator poisoned"))?;
        anyhow::ensure!(
            id < self.n_streams
                && state.cancelled.is_none()
                && state.shared.activation_step.is_some()
                && !state.shared.final_drained
                && state.shared.archived[id].is_none(),
            "deferred source epoch absent/duplicate/unavailable"
        );
        let worker = state.workers[id]
            .as_ref()
            .context("deferred source absent")?;
        anyhow::ensure!(
            archived as u64 <= worker.source_binding[4],
            "archive exceeds bound source capacity"
        );
        state.shared.archived[id] = Some((step, u32::try_from(archived)?));
        Ok(())
    }

    /// One actual shared physical/observer launch. Every caller has completed
    /// host feedback and native preflight. Admission, launch, shared-stream fence
    /// and per-engine receipt commit are bound to the same retained executable.
    #[cfg(feature = "v2_ignition")]
    pub fn execute_shared_resident_window(
        &self,
        id: usize,
        engine: &mut crate::persistent_engine::PersistentNhsEngine,
        launch_path: &str,
        cancelled: impl Fn() -> bool,
    ) -> Result<crate::persistent_engine::ResidentGraphLaddCadenceWindowReceipt> {
        anyhow::ensure!(id < self.n_streams, "unknown shared resident worker");
        let start = engine.device_protocol_current_step()?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("coordinator poisoned"))?;
        anyhow::ensure!(
            state.cancelled.is_none()
                && state.shared.activation_step.is_some()
                && !state.shared.final_drained
                && state.shared.continuation_rebind_pending.iter().all(|v|!*v)
                && !state.ready[id],
            "shared resident arrival unavailable/duplicate"
        );
        if state.shared.parent.is_none() {
            state.shared.parent = Some(self.build_shared_parent(&mut state)?);
        }
        let admission_result = {
            let SharedResidentState {
                parent, prepared, ..
            } = &mut state.shared;
            engine.prepare_shared_resident_observation_window(
                prepared[id]
                    .as_mut()
                    .context("shared prepared worker absent")?,
                parent.as_ref().unwrap().graph.as_ref(),
            )
        };
        let admission = match admission_result {
            Ok(value) => value,
            Err(error) => {
                state.cancelled = Some(format!("shared native admission failed: {error:#}"));
                self.changed.notify_all();
                engine.abort_shared_resident_observation_window();
                return Err(error);
            }
        };
        if let Err(error) = engine.synchronize_stream_raw("shared native preflight publication") {
            state.cancelled = Some(format!("shared worker publication fence failed: {error:#}"));
            self.changed.notify_all();
            engine.abort_shared_resident_observation_window();
            return Err(error);
        }
        #[cfg(feature="v2_ignition")]
        anyhow::ensure!(state.shared.continuation_rebind_pending.iter().all(|v|!*v), "shared continuation native bindings are incomplete");
        let epoch = state.epoch;
        state.ready[id] = true;
        state.steps[id] = Some(start);
        loop {
            if cancelled() && state.epoch == epoch {
                state.cancelled =
                    Some("worker/watchdog cancelled shared physical admission".into());
                self.changed.notify_all();
            }
            if let Some(reason) = &state.cancelled {
                engine.abort_shared_resident_observation_window();
                anyhow::bail!("shared physical window cancelled: {reason}");
            }
            if state.epoch != epoch {
                break;
            }
            if state.ready.iter().all(|ready| *ready) {
                let launched = (|| -> Result<()> {
                    anyhow::ensure!(
                        state.steps.iter().all(|step| *step == Some(start)),
                        "actual physical clocks disagree before shared parent"
                    );
                    let bootstrap = state.shared.windows == 0;
                    let counts = if bootstrap {
                        anyhow::ensure!(
                            state.shared.activation_step == Some(start)
                                && state.shared.archived.iter().all(Option::is_none),
                            "shared bootstrap cannot consume a fabricated prior epoch"
                        );
                        vec![0; self.n_streams]
                    } else {
                        state
                            .shared
                            .archived
                            .iter()
                            .map(|source| {
                                let (step, count) = source
                                    .context("shared launch lacks a complete prior archive")?;
                                anyhow::ensure!(
                                    step == start,
                                    "prior archived source clock differs from physical start"
                                );
                                Ok(count)
                            })
                            .collect::<Result<Vec<_>>>()?
                    };
                    let previous = ArchivedEndpointReceipt {
                        endpoint_step: start,
                        epoch,
                        archived_counts: counts,
                        reset_before_next_physics: true,
                    };
                    {
                        let parent = state.shared.parent.as_mut().unwrap();
                        parent.handoff.stage_expected(&self.stream, &previous)?;
                        self.stream
                            .memcpy_htod(&[u32::from(!bootstrap)], &mut parent.coupling_enabled)?;
                    }
                    let graph = state.shared.parent.as_ref().unwrap().graph.as_ref();
                    let permit = CoordinatorLaunchPermit {
                        coordinator: self,
                        state: &state,
                        graph,
                    };
                    graph.launch_on_coordinator_stream(&self.stream, &permit)?;
                    // This real shared-stream fence happens BEFORE completion is
                    // broadcast. Worker-local stream fences cannot substitute.
                    self.stream.synchronize()?;
                    if !bootstrap {
                        let rows=state.shared.parent.as_ref().unwrap().handoff
                            .verify_receipts(&self.stream,&previous)?;
                        state.shared.last_handoff=Some((previous,rows));
                    }
                    Ok(())
                })();
                if let Err(error) = launched {
                    state.cancelled = Some(format!("shared parent execution failed: {error:#}"));
                    self.changed.notify_all();
                    engine.abort_shared_resident_observation_window();
                    return Err(error);
                }
                state.shared.windows += 1;
                state.shared.archived.fill(None);
                state.launch_count += 1;
                state.epoch += 1;
                state.ready.fill(false);
                state.steps.fill(None);
                self.changed.notify_all();
                break;
            }
            state = self
                .changed
                .wait_timeout(state, Duration::from_millis(100))
                .map_err(|_| anyhow::anyhow!("shared physical rendezvous poisoned"))?
                .0;
        }
        let commit = {
            let SharedResidentState {
                parent, prepared, ..
            } = &mut state.shared;
            engine.commit_shared_resident_observation_window(
                prepared[id]
                    .as_mut()
                    .context("shared prepared worker disappeared")?,
                parent.as_ref().unwrap().graph.as_ref(),
                admission,
                launch_path,
            )
        };
        if let Err(error) = &commit {
            state.cancelled = Some(format!("shared native receipt commit failed: {error:#}"));
            self.changed.notify_all();
            engine.abort_shared_resident_observation_window();
        }
        commit
    }

    /// Native device receipt for the just-consumed archived source. The caller
    /// emits this before it can enter another shared launch, so a later source
    /// epoch cannot be substituted into the chronological transition record.
    #[cfg(feature = "v2_ignition")]
    pub fn verified_handoff_for_worker(&self,id:usize,completed_step:u32)->Result<serde_json::Value> {
        let state=self.state.lock().map_err(|_|anyhow::anyhow!("coordinator poisoned"))?;
        anyhow::ensure!(id<self.n_streams && state.cancelled.is_none(),"verified handoff unavailable");
        let (expected,rows)=state.shared.last_handoff.as_ref().context("no verified native prior epoch")?;
        anyhow::ensure!(expected.endpoint_step==completed_step,"native handoff step differs from requested source epoch");
        let row=rows.get(id).context("native handoff lacks physical producer")?;
        Ok(serde_json::json!({"schema_kind":"prism.native_coupling_source_epoch.v1",
            "physical_source_stream":id,"epoch":expected.epoch,
            "receipt_fields":["expected_step","actual_step","archived_count","resident_count",
                "published_count","failure_bits","epoch_lo","epoch_hi"],
            "native_u32_values":row,"all_sources_and_recipient_windows_verified":true,
            "source_reset_before_next_physics":expected.reset_before_next_physics,
            "record_packet_abi_bytes":96,
            "comparison_domain":"protocol_step_endpoint",
            "native_clock_authority":"per-source physical_time_journal_manifest and exact step record",
            "simultaneous_physical_time_asserted":false,"equal_dt_required":false}))
    }

    fn build_graph(&self, state: &State, verify_archive: bool) -> Result<EndpointGraph> {
        self.stream.context().bind_to_thread()?;
        anyhow::ensure!(
            state.workers.iter().all(Option::is_some),
            "captured endpoint lacks a physical worker"
        );
        // Allocations precede parent ownership so error/unwind drops parent first.
        let mut iterations = Vec::with_capacity(self.n_streams);
        for _ in 0..self.n_streams {
            iterations.push(self.stream.alloc_zeros::<u32>(1)?);
        }
        let mut handoff = if verify_archive {
            Some(self.prepare_epoch_reset(state)?)
        } else {
            None
        };
        self.stream.synchronize()?;
        let mut raw: sys::CUgraph = std::ptr::null_mut();
        let rc = unsafe { sys::cuGraphCreate(&mut raw, 0) };
        anyhow::ensure!(
            rc == sys::CUresult::CUDA_SUCCESS,
            "captured endpoint parent create {rc:?}"
        );
        let mut parent = CapturedTemplate::from_capture(
            raw,
            std::collections::HashMap::new(),
            self.stream.clone(),
        );
        let frontier = self.append_coupling_nodes(state, &mut parent, raw, &[], &iterations)?;
        if let Some(guard) = handoff.as_mut() {
            guard.append_into(&self.stream, &mut parent, raw, &frontier)?;
        }
        Ok(EndpointGraph {
            graph: parent.instantiate()?,
            handoff,
            _iterations: iterations,
        })
    }

    fn prepare_epoch_reset(&self, state: &State) -> Result<EpochResetResources> {
        anyhow::ensure!(
            state.workers.iter().all(Option::is_some),
            "epoch handoff needs every actual source owner"
        );
        let sources: Vec<u64> = state
            .workers
            .iter()
            .flatten()
            .flat_map(|worker| worker.source_binding)
            .collect();
        let mut ring_pointers = Vec::with_capacity(self.rings.len() * 3);
        let mut window_pointers = Vec::with_capacity(self.n_streams * 3);
        let mut kernel = None;
        for ring in &self.rings {
            let ring = ring
                .lock()
                .map_err(|_| anyhow::anyhow!("handoff ring poisoned"))?;
            let lossless = ring
                .lossless
                .as_ref()
                .context("handoff requires lossless ring")?;
            if kernel.is_none() {
                kernel = Some(lossless.epoch_reset.clone());
            }
            ring_pointers.extend([
                ring.head.device_ptr(&self.stream).0,
                ring.tail.device_ptr(&self.stream).0,
                lossless.failure.device_ptr(&self.stream).0,
            ]);
            for &recipient in &lossless.participating {
                window_pointers.push(lossless.windows[recipient].device_ptr(&self.stream).0);
            }
        }
        anyhow::ensure!(
            window_pointers.len() == self.n_streams * 3,
            "handoff lost a physical recipient's source-group window"
        );
        Ok(EpochResetResources {
            sources: self.stream.clone_htod(&sources)?,
            rings: self.stream.clone_htod(&ring_pointers)?,
            windows: self.stream.clone_htod(&window_pointers)?,
            expected: self.stream.alloc_zeros::<u32>(self.n_streams + 4)?,
            receipts: self.stream.alloc_zeros::<u32>(self.n_streams * 8 + 1)?,
            kernel: kernel.context("handoff kernel absent")?,
            n_sources: u32::try_from(self.n_streams)?,
            n_rings: u32::try_from(self.rings.len())?,
            n_windows: u32::try_from(window_pointers.len())?,
        })
    }

    /// Append the existing ordered lossless exchange into the caller-owned
    /// root. Conditional handles are created in that root, never copied as a
    /// conditional child graph. The caller must retain State's worker templates,
    /// all external CudaOwned leases, rings and iteration allocations until the
    /// owning executable is retired. Only coordinator-owned roots may call this.
    ///
    /// The returned frontier is after all four source-group finalizations;
    /// subsequent source reset/physical nodes must depend on every returned node.
    fn append_coupling_nodes(
        &self,
        state: &State,
        parent: &mut CapturedTemplate,
        destination: sys::CUgraph,
        dependencies: &[sys::CUgraphNode],
        iterations: &[CudaSlice<u32>],
    ) -> Result<Vec<sys::CUgraphNode>> {
        anyhow::ensure!(
            state.workers.len() == self.n_streams
                && state.workers.iter().all(Option::is_some)
                && iterations.len() == self.n_streams,
            "coupled parent assembly lacks exact worker/iteration population"
        );
        let raw = destination;
        let mut frontier = dependencies.to_vec();
        for group in 0..4 {
            let mut began = Vec::new();
            for worker in state.workers.iter().flatten() {
                if let Some(template) = &worker.begin[group] {
                    began.push(unsafe {
                        crate::graph_node::add_child_graph_node_v3(
                            raw,
                            &frontier,
                            template.cu_graph(),
                        )
                    }?);
                }
            }
            frontier = began;
            let retirement = capture_pure_template(self.stream.clone(), |_| {
                self.rings[group]
                    .lock()
                    .map_err(|_| anyhow::anyhow!("ring poisoned"))?
                    .retire_participating_recipients(&self.stream)
            })?;
            for id in group * self.engines_per_group..(group + 1) * self.engines_per_group {
                let worker = state.workers[id].as_ref().unwrap();
                let iter_ptr = iterations[id].device_ptr(&self.stream).0;
                let reset_pending = unsafe {
                    crate::graph_capture::memset_node_u32(raw, &frontier, worker.pending_ptr, 1)
                }?;
                let reset_iter =
                    unsafe { crate::graph_capture::memset_node_u32(raw, &frontier, iter_ptr, 0) }?;
                let handle =
                    unsafe { crate::graph_node::create_while_handle(parent.cu_graph(), 1) }
                        .map_err(|rc| anyhow::anyhow!("captured coupling WHILE handle {rc}"))?;
                let (while_node, body) = unsafe {
                    crate::graph_node::add_while_node_v3(raw, &[reset_pending, reset_iter], handle)
                }
                .map_err(|rc| anyhow::anyhow!("captured coupling WHILE body {rc}"))?;
                let publish = unsafe {
                    crate::graph_node::add_child_graph_node_v3(body, &[], worker.publish.cu_graph())
                }?;
                let mut readers = Vec::new();
                for recipient in state.workers.iter().flatten() {
                    if let Some(template) = &recipient.accumulate[group] {
                        readers.push(unsafe {
                            crate::graph_node::add_child_graph_node_v3(
                                body,
                                &[publish],
                                template.cu_graph(),
                            )
                        }?);
                    }
                }
                let retire = unsafe {
                    crate::graph_node::add_child_graph_node_v3(
                        body,
                        &readers,
                        retirement.cu_graph(),
                    )
                }?;
                let predicate = capture_pure_template(self.stream.clone(), |_| unsafe {
                    crate::graph_capture::launch_while_predicate_bridge(
                        self.stream.cu_stream(),
                        worker.pending_ptr,
                        iter_ptr,
                        handle,
                        worker.max_batches,
                    )
                })?;
                unsafe {
                    crate::graph_node::add_child_graph_node_v3(
                        body,
                        &[retire],
                        predicate.cu_graph(),
                    )
                }?;
                frontier = vec![while_node];
            }
            let mut finalized = Vec::new();
            for worker in state.workers.iter().flatten() {
                if let Some(template) = &worker.finalize[group] {
                    finalized.push(unsafe {
                        crate::graph_node::add_child_graph_node_v3(
                            raw,
                            &frontier,
                            template.cu_graph(),
                        )
                    }?);
                }
            }
            frontier = finalized;
        }
        Ok(frontier)
    }

    /// All workers call once at the existing controller-complete boundary.
    /// The last arrival launches once. Waiting peers retain their allocation
    /// owners; no worker can resume MD until the shared graph is fenced.
    pub fn exchange(
        &self,
        id: usize,
        completed_step: u32,
        cancelled: impl Fn() -> bool,
    ) -> Result<u64> {
        self.exchange_kind(id, completed_step, cancelled, false)
    }

    /// Drain the final real archived population with zero physical/observer
    /// advancement. Source bytes/count stay intact for the caller's established
    /// checked release; this never fabricates a next observation endpoint.
    #[cfg(feature = "v2_ignition")]
    pub fn drain_shared_endpoint(
        &self,
        id: usize,
        completed_step: u32,
        cancelled: impl Fn() -> bool,
    ) -> Result<u64> {
        self.exchange_kind(id, completed_step, cancelled, true)
    }

    fn exchange_kind(
        &self,
        id: usize,
        completed_step: u32,
        cancelled: impl Fn() -> bool,
        final_drain: bool,
    ) -> Result<u64> {
        anyhow::ensure!(id < self.n_streams, "unknown captured exchange worker");
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("captured coordinator poisoned"))?;
        #[cfg(feature="v2_ignition")]
        anyhow::ensure!(state.shared.continuation_rebind_pending.iter().all(|v|!*v), "shared continuation native bindings are incomplete");
        let epoch = state.epoch;
        anyhow::ensure!(
            state.cancelled.is_none() && state.workers[id].is_some() && !state.ready[id],
            "captured exchange unavailable or duplicate arrival: {:?}",
            state.cancelled
        );
        state.ready[id] = true;
        state.steps[id] = Some(completed_step);
        loop {
            if state.epoch != epoch {
                return Ok(state.launch_count);
            }
            if cancelled() {
                state.cancelled =
                    Some("worker/watchdog cancellation before endpoint launch".into());
                self.changed.notify_all();
            }
            anyhow::ensure!(
                state.cancelled.is_none(),
                "captured exchange cancelled: {:?}",
                state.cancelled
            );
            if state.epoch != epoch {
                return Ok(state.launch_count);
            }
            if state.ready.iter().all(|x| *x) {
                let result = (|| -> Result<()> {
                    anyhow::ensure!(
                        state.steps.iter().all(|step| *step == Some(completed_step)),
                        "physical endpoint clocks disagree"
                    );
                    #[cfg(feature = "v2_ignition")]
                    let previous = if final_drain {
                        anyhow::ensure!(
                            state.shared.activation_step.is_some()
                                && state.shared.windows > 0
                                && !state.shared.final_drained,
                            "final drain requires actual shared physical execution"
                        );
                        let counts = state
                            .shared
                            .archived
                            .iter()
                            .map(|source| {
                                let (step, count) = source
                                    .context("final drain lacks an archived physical source")?;
                                anyhow::ensure!(
                                    step == completed_step,
                                    "final drain source clock mismatch"
                                );
                                Ok(count)
                            })
                            .collect::<Result<Vec<_>>>()?;
                        Some(ArchivedEndpointReceipt {
                            endpoint_step: completed_step,
                            epoch: state.epoch,
                            archived_counts: counts,
                            reset_before_next_physics: false,
                        })
                    } else {
                        None
                    };
                    if state
                        .graph
                        .as_ref()
                        .is_some_and(|g| g.handoff.is_some() != final_drain)
                    {
                        state.graph.take();
                    }
                    if state.graph.is_none() {
                        state.graph = Some(self.build_graph(&state, final_drain)?);
                    }
                    #[cfg(feature = "v2_ignition")]
                    if let Some(expected) = &previous {
                        state
                            .graph
                            .as_mut()
                            .unwrap()
                            .handoff
                            .as_mut()
                            .unwrap()
                            .stage_expected(&self.stream, expected)?;
                    }
                    self.stream.context().bind_to_thread()?;
                    state
                        .graph
                        .as_ref()
                        .unwrap()
                        .graph
                        .launch_on_stream(self.stream.cu_stream())?;
                    self.stream.synchronize()?;
                    #[cfg(feature = "v2_ignition")]
                    if let Some(expected) = &previous {
                        let rows=state.graph.as_ref().unwrap().handoff.as_ref().unwrap()
                            .verify_receipts(&self.stream,expected)?;
                        state.shared.last_handoff=Some((expected.clone(),rows));
                    }
                    for ring in &self.rings {
                        let ring = ring
                            .lock()
                            .map_err(|_| anyhow::anyhow!("captured ring poisoned"))?;
                        ring.validate_lossless_retirement(&self.stream)?;
                        for &recipient in &ring.lossless.as_ref().unwrap().participating {
                            ring.validate_lossless_window(&self.stream, recipient)?;
                        }
                    }
                    Ok(())
                })();
                if let Err(error) = result {
                    state.cancelled =
                        Some(format!("captured endpoint execution failed: {error:#}"));
                    self.changed.notify_all();
                    return Err(error);
                }
                #[cfg(feature = "v2_ignition")]
                if state.shared.activation_step.is_none()
                    && state.shared.prepared.iter().all(Option::is_some)
                {
                    state.shared.activation_step = Some(completed_step);
                }
                #[cfg(feature = "v2_ignition")]
                if final_drain {
                    state.shared.archived.fill(None);
                    state.shared.final_drained = true;
                }
                state.launch_count += 1;
                state.epoch += 1;
                state.ready.fill(false);
                state.steps.fill(None);
                let launches = state.launch_count;
                self.changed.notify_all();
                return Ok(launches);
            }
            state = self
                .changed
                .wait_timeout(state, Duration::from_millis(100))
                .map_err(|_| anyhow::anyhow!("captured rendezvous poisoned"))?
                .0;
        }
    }

    pub fn execution_telemetry(&self) -> serde_json::Value {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        #[cfg(feature = "v2_ignition")]
        let shared = serde_json::json!({"activation_step":state.shared.activation_step,
            "parent_graph_launches":state.shared.windows,
            "same_root_coupling_physics_observers":state.shared.windows>0,
            "bootstrap_parent_launches":u64::from(state.shared.windows>0),
            "previous_epoch_coupling_parent_launches":state.shared.windows.saturating_sub(1),
            "final_coupling_only_drain_completed":state.shared.final_drained,
            "final_drain_physical_steps":0,
            "registered_physical_cells":state.shared.prepared.iter().filter(|p|p.is_some()).count(),
            "continuation_rebind_pending":state.shared.continuation_rebind_pending,
            "prior_process_retired_root_snapshots":state.shared.prior_process_retired_roots.len(),
            "pending_archived_sources":state.shared.archived.iter().filter(|p|p.is_some()).count(),
            "single_live_v2_owner_per_cell":true,
            "single_live_v2_owner_scope":"one replayed V2 owner per physical cell; retained predecessor leases may reference a superseded allocation epoch",
            "retired_local_roots_replayed":false,
            "retained_local_root_capacity_per_cell":2,
            "retained_local_root_counts":state.shared.retired_local_roots.iter()
                .map(|slots|slots.iter().filter(|slot|slot.is_some()).count()).collect::<Vec<_>>(),
            "released_local_root_count":state.shared.released_local_root_count,
            "graph_retirement_completed":state.shared.retirement_completed,
            "graph_retirement_order":"shared_parent_then_retired_local_roots_then_prepared_then_external_owner_release",
            "observation_cadence":state.shared.prepared.iter().flatten().next().map(|p|p.cadence()),
            "completion_order":"shared_owner_stream_fence_then_per_engine_native_receipt_commit"});
        #[cfg(not(feature = "v2_ignition"))]
        let shared = serde_json::Value::Null;
        serde_json::json!({"schema_kind":"prism.captured_lossless_endpoint_execution.v2",
            "shared_resident_parent":shared,
            "physical_streams":self.n_streams,"completed_endpoints":state.epoch,
            "parent_graph_launches":state.launch_count,"host_per_batch_launches":0,
            "host_per_batch_barriers":0,"physical_md_steps_advanced_by_exchange":0,
            "scope":"native endpoint exchange and explicitly reported shared resident parent; host feedback epoch boundaries preserved",
            "continuation_reconstruction_history":state.reconstruction_history,
            "graph_currently_owned":state.graph.is_some(),"retired_or_cancelled":state.cancelled})
    }

    fn retire(&self) -> Result<(), i32> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        state
            .cancelled
            .get_or_insert_with(|| "allocation owner retiring; no future graph launches".into());
        self.changed.notify_all();
        self.stream
            .context()
            .bind_to_thread()
            .map_err(|e| e.0 as i32)?;
        // Lock excludes every launch, including an unwinding last-arrival thread.
        // Fence precedes destruction; on failure State remains retained by the
        // CudaOwned lease quarantine, along with every external pointee owner.
        self.stream.synchronize().map_err(|e| e.0 as i32)?;
        // Require context-level retirement before destroying conditional child
        // graphs and freeing their stream-ordered allocations. Stream fences
        // alone failed the recorded CUDA allocator diagnostic. This fence is
        // teardown-only; keep all owners quarantined if it fails.
        if state.graph.is_some() || state.workers.iter().any(Option::is_some) {
            self.stream
                .context()
                .synchronize()
                .map_err(|e| e.0 as i32)?;
        }
        #[cfg(feature = "v2_ignition")]
        {
            state.shared.parent.take();
            // Only after the owning shared executable is retired may its
            // demonstrated local predecessor dependencies be destroyed.
            let mut released=0usize;
            for slots in &mut state.shared.retired_local_roots {
                for slot in slots {
                    if slot.is_some() {drop(slot.take());released+=1;}
                }
            }
            state.shared.released_local_root_count+=released;
            for prepared in &mut state.shared.prepared {
                *prepared = None;
            }
            state.shared.retirement_completed=true;
        }
        state.graph.take();
        for worker in &mut state.workers {
            *worker = None;
        }
        Ok(())
    }
}
impl CudaRetirementLease for CapturedEndpointCoordinator {
    fn fence_and_retire(&self) -> Result<(), i32> {
        self.retire()
    }
}

impl Drop for CapturedEndpointCoordinator {
    fn drop(&mut self) {
        if self.retire().is_err() {
            // Normally unreachable because owner quarantine retains an Arc.
            // Preserve graph/iteration/ring allocations if final cleanup fails.
            let state = self
                .state
                .get_mut()
                .unwrap_or_else(|poison| poison.into_inner());
            #[cfg(feature = "v2_ignition")]
            {
                std::mem::forget(state.shared.parent.take());
                std::mem::forget(std::mem::take(&mut state.shared.retired_local_roots));
                std::mem::forget(std::mem::take(&mut state.shared.prepared));
            }
            std::mem::forget(state.graph.take());
            std::mem::forget(std::mem::take(&mut state.workers));
            std::mem::forget(self.rings.clone());
            std::mem::forget(self.stream.clone());
        }
    }
}

#[cfg(all(test, feature = "v2_ignition"))]
mod tests;

// All CUDA graph handles/templates and pointer tables are rebuilt by the
// existing native factories. Original addresses below are provenance only.
#[derive(serde::Serialize,serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkerContinuationV1 {fingerprint:Vec<u64>,source_binding:[u64;5],pending_ptr:u64,max_batches:u32}
#[derive(serde::Serialize,serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct HandoffContinuationV1 {sources:Vec<u64>,rings:Vec<u64>,windows:Vec<u64>,expected:Vec<u32>,receipts:Vec<u32>,shape:[u32;3]}
#[derive(serde::Serialize,serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct EndpointGraphContinuationV1 {
    graph:crate::graph_capture::GraphContinuationV1,handoff:Option<HandoffContinuationV1>,iterations:Vec<Vec<u32>>,
}
#[cfg(feature="v2_ignition")]
#[derive(serde::Serialize,serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SharedParentContinuationV1 {
    graph:crate::graph_capture::GraphContinuationV1,handoff:HandoffContinuationV1,
    coupling_enabled:Vec<u32>,outer_iterations:Vec<u32>,iterations:Vec<Vec<u32>>,
}
#[cfg(feature="v2_ignition")]
#[derive(serde::Serialize,serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RetiredRootContinuationV1 {
    worker:usize,slot:usize,
    graph:crate::persistent_engine::PhysicalObservationContinuationV1,
    pipeline:crate::captured_pipeline::AdjudicationContinuationV1,
}
#[cfg(feature="v2_ignition")]
#[derive(Clone,serde::Serialize,serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RetiredOwnerSnapshotV1 {#[serde(with="crate::checkpoint_codec::bytes")] bytes:Vec<u8>}
#[cfg(feature="v2_ignition")]
#[derive(serde::Serialize,serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SharedContinuationV1 {
    parent:Option<SharedParentContinuationV1>,
    prepared:Vec<Option<crate::persistent_engine::PreparedResidentContinuationV1>>,
    archived:Vec<Option<(u32,u32)>>,activation_step:Option<u32>,windows:u64,final_drained:bool,
    last_handoff:Option<(ArchivedEndpointReceipt,Vec<[u32;8]>)>,
    retired_roots:Vec<RetiredRootContinuationV1>,prior_process_retired_roots:Vec<RetiredOwnerSnapshotV1>,
    released_local_root_count:usize,retirement_completed:bool,
}
#[derive(serde::Serialize,serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EndpointContinuationV1 {
    schema_version:u32,n_streams:usize,engines_per_group:usize,
    workers:Vec<Option<WorkerContinuationV1>>,rings:Vec<super::TwinOwnerContinuationV1>,
    ready:Vec<bool>,steps:Vec<Option<u32>>,epoch:u64,launch_count:u64,cancelled:Option<String>,
    graph:Option<EndpointGraphContinuationV1>,
    #[cfg(feature="v2_ignition")] shared:SharedContinuationV1,
    reconstruction_history:Vec<serde_json::Value>,
}
impl EpochResetResources {
    fn continuation_allocation_plan(&self)->Result<Vec<(String,u64)>> {
        let mut rows=Vec::new();
        for(name,count,width)in [("sources",self.sources.len(),8u64),("rings",self.rings.len(),8),("windows",self.windows.len(),8),
            ("expected",self.expected.len(),4),("receipts",self.receipts.len(),4)] {
            rows.push((name.into(),u64::try_from(count)?.checked_mul(width).context("handoff capture extent overflow")?));
        }Ok(rows)
    }
    fn capture_checkpoint(&self,stream:&Arc<CudaStream>)->Result<HandoffContinuationV1> {
        stream.synchronize()?;
        Ok(HandoffContinuationV1 {sources:stream.clone_dtoh(&self.sources)?,rings:stream.clone_dtoh(&self.rings)?,
            windows:stream.clone_dtoh(&self.windows)?,expected:stream.clone_dtoh(&self.expected)?,
            receipts:stream.clone_dtoh(&self.receipts)?,shape:[self.n_sources,self.n_rings,self.n_windows]})
    }
    fn restore_checkpoint(&mut self,stream:&Arc<CudaStream>,saved:&HandoffContinuationV1)->Result<serde_json::Value> {
        anyhow::ensure!(saved.shape==[self.n_sources,self.n_rings,self.n_windows]
            && saved.sources.len()==self.sources.len() && saved.rings.len()==self.rings.len()
            && saved.windows.len()==self.windows.len() && saved.expected.len()==self.expected.len()
            && saved.receipts.len()==self.receipts.len(),"epoch reset continuation population changed");
        // Source table fifth word is a capacity, not a pointer.
        let sources=stream.clone_dtoh(&self.sources)?;
        for (old,new) in saved.sources.chunks_exact(5).zip(sources.chunks_exact(5)) {
            anyhow::ensure!(old[4]==new[4],"handoff source capacity changed");
        }
        let receipt=serde_json::json!({"old":{"sources":saved.sources,"rings":saved.rings,"windows":saved.windows},
            "fresh":{"sources":sources,"rings":stream.clone_dtoh(&self.rings)?,"windows":stream.clone_dtoh(&self.windows)?}});
        stream.memcpy_htod(&saved.expected,&mut self.expected)?;
        stream.memcpy_htod(&saved.receipts,&mut self.receipts)?;
        Ok(receipt)
    }
}
fn capture_endpoint_iterations(stream:&Arc<CudaStream>,planes:&[CudaSlice<u32>])->Result<Vec<Vec<u32>>> {
    planes.iter().map(|v|stream.clone_dtoh(v).map_err(Into::into)).collect()
}
fn restore_endpoint_iterations(stream:&Arc<CudaStream>,planes:&mut [CudaSlice<u32>],saved:&[Vec<u32>])->Result<()> {
    anyhow::ensure!(planes.len()==saved.len() && planes.iter().zip(saved).all(|(a,b)|a.len()==b.len()),"endpoint iteration owner extent changed");
    for (plane,bytes) in planes.iter_mut().zip(saved){stream.memcpy_htod(bytes,plane)?;}Ok(())
}
impl EndpointContinuationV1 {
    pub fn epoch(&self)->u64{self.epoch}
    pub fn worker_registered(&self,id:usize)->bool{self.workers.get(id).is_some_and(Option::is_some)}
    #[cfg(feature="v2_ignition")]
    pub fn prepared_worker(&self,id:usize)->Option<&crate::persistent_engine::PreparedResidentContinuationV1>{self.shared.prepared.get(id).and_then(Option::as_ref)}
    fn validate_boundary(&self)->Result<()> {
        anyhow::ensure!(self.schema_version==1 && self.n_streams>=4 && self.n_streams%4==0
            && self.engines_per_group==self.n_streams/4 && self.rings.len()==4
            && self.workers.len()==self.n_streams && self.ready.len()==self.n_streams && self.steps.len()==self.n_streams,
            "endpoint checkpoint population mismatch");
        anyhow::ensure!(self.ready.iter().all(|x|!*x) && self.steps.iter().all(Option::is_none) && self.cancelled.is_none(),
            "checkpoint must follow all coordinator arrivals, native commits and consumer drains");
        anyhow::ensure!(self.epoch==self.launch_count,"endpoint epoch/launch count mismatch");
        for worker in self.workers.iter().flatten(){anyhow::ensure!(worker.fingerprint.len()==18 && worker.max_batches>0
            && worker.pending_ptr==worker.fingerprint[3] && worker.source_binding[4]>0,"invalid endpoint worker binding");}
        if self.graph.is_some(){anyhow::ensure!(self.workers.iter().all(Option::is_some),"endpoint graph lost a worker");}
        #[cfg(feature="v2_ignition")]
        {
            let s=&self.shared;
            anyhow::ensure!(s.prepared.len()==self.n_streams && s.archived.len()==self.n_streams && !s.retirement_completed,
                "shared checkpoint population/retirement mismatch");
            anyhow::ensure!(s.windows<=self.launch_count && (s.windows==0||s.parent.is_some())
                && (s.parent.is_none()||s.prepared.iter().all(Option::is_some)),"shared parent/window population mismatch");
            for (id,prepared) in s.prepared.iter().enumerate(){if let Some(p)=prepared{
                anyhow::ensure!(p.source_stream_id as usize==id && p.appended==s.parent.is_some(),"shared prepared parent/source mismatch");}}
            let archived=s.archived.iter().filter(|x|x.is_some()).count();
            anyhow::ensure!(archived==0||s.windows>0,"archived sources precede shared physical execution");
            anyhow::ensure!(archived==0||archived==self.n_streams,"partial archived source population at global checkpoint");
            if s.final_drained {anyhow::ensure!(archived==0 && s.windows>0,"final drain continuation has pending source");}
            let mut clock=None;
            for (id,source) in s.archived.iter().enumerate(){if let Some((step,count))=source{
                anyhow::ensure!(s.activation_step.is_some() && *count as u64<=self.workers[id].as_ref().context("archived source owner absent")?.source_binding[4],"invalid archived source owner/count");
                if let Some(expected)=clock{anyhow::ensure!(*step==expected,"archived endpoint clocks disagree");}else{clock=Some(*step);}
            }}
            if let Some((receipt,rows))=&s.last_handoff {anyhow::ensure!(receipt.archived_counts.len()==self.n_streams && rows.len()==self.n_streams
                && receipt.epoch<self.epoch,"invalid handoff population/epoch");}
            let mut slots=std::collections::BTreeSet::new();
            for r in &s.retired_roots{anyhow::ensure!(r.worker<self.n_streams && r.slot<2 && slots.insert((r.worker,r.slot)),"invalid retired owner population");}
        }
        Ok(())
    }
}
impl CapturedEndpointCoordinator {
    /// Caller holds the global checkpoint barrier AFTER every worker's native
    /// commit and output drain acknowledgement. Internal stream fences alone
    /// cannot prevent a worker from submitting its next physical window.
    pub fn continuation_allocation_plan_quiescent(&self)->Result<Vec<(String,u64)>> {
        let state=self.state.lock().map_err(|_|anyhow::anyhow!("coordinator poisoned"))?;
        anyhow::ensure!(state.ready.iter().all(|v|!*v) && state.steps.iter().all(Option::is_none)
            && state.cancelled.is_none(),"endpoint allocation plan is not quiescent");
        let mut rows=Vec::new();
        for(index,ring)in self.rings.iter().enumerate(){for(name,bytes)in ring.lock().map_err(|_|anyhow::anyhow!("ring poisoned"))?
            .continuation_allocation_plan(&self.stream)?{rows.push((format!("ring_{index}/{name}"),bytes));}}
        if let Some(graph)=&state.graph {
            for(name,bytes)in graph.graph.continuation_allocation_plan()?{rows.push((format!("endpoint_graph/{name}"),bytes));}
            if let Some(handoff)=&graph.handoff{for(name,bytes)in handoff.continuation_allocation_plan()?{rows.push((format!("endpoint_handoff/{name}"),bytes));}}
            for(index,plane)in graph._iterations.iter().enumerate(){rows.push((format!("endpoint_iteration_{index}"),u64::try_from(plane.len())?.checked_mul(4).context("endpoint iteration extent overflow")?));}
        }
        #[cfg(feature="v2_ignition")]
        {
            let shared=&state.shared;
            if let Some(parent)=&shared.parent {
                for(name,bytes)in parent.graph.continuation_allocation_plan()?{rows.push((format!("shared_parent_graph/{name}"),bytes));}
                for(name,bytes)in parent.handoff.continuation_allocation_plan()?{rows.push((format!("shared_parent_handoff/{name}"),bytes));}
                for(name,plane)in [("coupling_enabled",&parent.coupling_enabled),("outer_iteration",&parent._outer_iterations)] {
                    rows.push((name.into(),u64::try_from(plane.len())?.checked_mul(4).context("shared endpoint extent overflow")?));
                }
                for(index,plane)in parent._iterations.iter().enumerate(){rows.push((format!("shared_iteration_{index}"),u64::try_from(plane.len())?.checked_mul(4).context("shared iteration extent overflow")?));}
            }
            for(index,prepared)in shared.prepared.iter().enumerate(){if let Some(prepared)=prepared {
                for(name,bytes)in prepared.continuation_allocation_plan()?{rows.push((format!("prepared_{index}/{name}"),bytes));}
            }}
            for(worker,slots)in shared.retired_local_roots.iter().enumerate(){for(slot,root)in slots.iter().enumerate(){if let Some(root)=root {
                for(name,bytes)in root.continuation_allocation_plan()?{rows.push((format!("retired_{worker}_{slot}/graph/{name}"),bytes));}
                for(name,bytes)in root.pipeline().continuation_allocation_plan()?{rows.push((format!("retired_{worker}_{slot}/pipeline/{name}"),bytes));}
            }}}
            rows.push(("already_owned_shared_histories_wire".into(),crate::checkpoint_codec::encoded_size(
                &(&shared.archived,&shared.last_handoff,&shared.prior_process_retired_roots),u64::MAX)?));
        }
        rows.push(("reconstruction_history_wire".into(),crate::checkpoint_codec::encoded_size(&state.reconstruction_history,u64::MAX)?));
        Ok(rows)
    }

    pub unsafe fn capture_continuation_quiescent(&self,run_root:&std::path::Path)->Result<EndpointContinuationV1> {
        let state=self.state.lock().map_err(|_|anyhow::anyhow!("coordinator poisoned"))?;
        anyhow::ensure!(state.ready.iter().all(|v|!*v)&&state.steps.iter().all(Option::is_none)&&state.cancelled.is_none(),"endpoint not quiescent");
        for w in state.workers.iter().flatten(){w.stream.synchronize()?;}self.stream.synchronize()?;
        #[cfg(feature="v2_ignition")]
        anyhow::ensure!(state.shared.continuation_rebind_pending.iter().all(|x|!*x),"shared reconstruction is incomplete");
        let workers=state.workers.iter().map(|w|w.as_ref().map(|w|WorkerContinuationV1{fingerprint:w.fingerprint.clone(),source_binding:w.source_binding,pending_ptr:w.pending_ptr,max_batches:w.max_batches})).collect();
        let rings=self.rings.iter().map(|r|r.lock().map_err(|_|anyhow::anyhow!("ring poisoned"))?.capture_continuation(&self.stream)).collect::<Result<Vec<_>>>()?;
        let graph=state.graph.as_ref().map(|g|->Result<_>{Ok(EndpointGraphContinuationV1{graph:g.graph.capture_continuation(&self.stream)?,
            handoff:g.handoff.as_ref().map(|h|h.capture_checkpoint(&self.stream)).transpose()?,iterations:capture_endpoint_iterations(&self.stream,&g._iterations)?})}).transpose()?;
        #[cfg(feature="v2_ignition")]
        let shared={
            let s=&state.shared;
            let parent=s.parent.as_ref().map(|p|->Result<_>{Ok(SharedParentContinuationV1 {graph:p.graph.capture_continuation(&self.stream)?,
                handoff:p.handoff.capture_checkpoint(&self.stream)?,coupling_enabled:self.stream.clone_dtoh(&p.coupling_enabled)?,
                outer_iterations:self.stream.clone_dtoh(&p._outer_iterations)?,iterations:capture_endpoint_iterations(&self.stream,&p._iterations)?})}).transpose()?;
            let prepared=s.prepared.iter().map(|p|p.as_ref().map(|p|p.capture_prepared_continuation(run_root)).transpose()).collect::<Result<Vec<_>>>()?;
            let mut retired_roots=Vec::new();
            for (worker,slots) in s.retired_local_roots.iter().enumerate(){for(slot,root)in slots.iter().enumerate(){if let Some(root)=root{
                let stream=&state.workers[worker].as_ref().context("retired root lost worker")?.stream;
                retired_roots.push(RetiredRootContinuationV1{worker,slot,graph:root.capture_continuation(stream)?,pipeline:root.pipeline().capture_continuation(run_root)?});
            }}}
            SharedContinuationV1{parent,prepared,archived:s.archived.clone(),activation_step:s.activation_step,windows:s.windows,
                final_drained:s.final_drained,last_handoff:s.last_handoff.clone(),retired_roots,prior_process_retired_roots:s.prior_process_retired_roots.clone(),
                released_local_root_count:s.released_local_root_count,retirement_completed:s.retirement_completed}
        };
        let result=EndpointContinuationV1{schema_version:1,n_streams:self.n_streams,engines_per_group:self.engines_per_group,
            workers,rings,ready:state.ready.clone(),steps:state.steps.clone(),epoch:state.epoch,launch_count:state.launch_count,
            cancelled:state.cancelled.clone(),graph,#[cfg(feature="v2_ignition")]shared,reconstruction_history:state.reconstruction_history.clone()};
        result.validate_boundary()?;Ok(result)
    }
    /// First restore phase, before prepare_worker captures any ring pointers.
    pub unsafe fn restore_rings_before_prepare(&self,saved:&EndpointContinuationV1)->Result<()> {
        saved.validate_boundary()?;
        let state=self.state.lock().map_err(|_|anyhow::anyhow!("coordinator poisoned"))?;
        anyhow::ensure!(self.n_streams==saved.n_streams && state.workers.iter().all(Option::is_none)
            && state.graph.is_none() && state.epoch==0 && state.launch_count==0,"ring restore requires fresh matching coordinator");
        for(r,s)in self.rings.iter().zip(&saved.rings){r.lock().map_err(|_|anyhow::anyhow!("ring poisoned"))?.restore_continuation_before_capture(&self.stream,s)?;}
        self.stream.synchronize()?;Ok(())
    }
    /// Second phase: all fresh workers/optional prepared parts were registered.
    /// Reconstruct graphs, restore numerical controls and shared output owner,
    /// then keep bound workers gated until each reissues its native binding.
    pub unsafe fn restore_continuation_after_prepare(&self,run_root:&std::path::Path,saved:&EndpointContinuationV1)->Result<()> {
        saved.validate_boundary()?;
        let mut state=self.state.lock().map_err(|_|anyhow::anyhow!("coordinator poisoned"))?;
        let result=(||->Result<()> {
            anyhow::ensure!(self.n_streams==saved.n_streams && self.engines_per_group==saved.engines_per_group
                && state.epoch==0 && state.launch_count==0 && state.graph.is_none() && state.cancelled.is_none()
                && state.ready.iter().all(|x|!*x),"coordinator restore needs fresh matching owners");
            let mut mappings=Vec::new();
            for (id,(live,old)) in state.workers.iter().zip(&saved.workers).enumerate(){match(live,old){
                (None,None)=>{},(Some(w),Some(s))=>{
                    anyhow::ensure!(w.fingerprint.len()==18 && w.fingerprint[7..17]==s.fingerprint[7..17]
                        && w.source_binding[4]==s.source_binding[4] && w.max_batches==s.max_batches,"endpoint worker {id} geometry/capacity changed");
                    w.stream.synchronize()?;
                    mappings.push(serde_json::json!({"worker":id,"old":{"fingerprint":s.fingerprint,"source":s.source_binding},"fresh":{"fingerprint":w.fingerprint,"source":w.source_binding}}));
                },_=>anyhow::bail!("endpoint worker {id} population changed"),}}
            // Reverify complete ring state at the post-prepare seam: native pure
            // graph capture must not change any ring history or publication.
            for(r,s)in self.rings.iter().zip(&saved.rings){let actual=r.lock().map_err(|_|anyhow::anyhow!("ring poisoned"))?.capture_continuation(&self.stream)?;
                anyhow::ensure!(crate::checkpoint_codec::to_bytes(&actual)?==crate::checkpoint_codec::to_bytes(s)?,"ring restore missing or capture mutated ring state");}
            #[cfg(feature="v2_ignition")]
            {
                anyhow::ensure!(state.shared.parent.is_none() && state.shared.activation_step.is_none()
                    && state.shared.retired_local_roots.iter().all(|slots|slots.iter().all(Option::is_none)),"shared restart needs fresh graph owners");
                for (live,old) in state.shared.prepared.iter_mut().zip(&saved.shared.prepared){match(live,old){
                    (Some(live),Some(old))=>live.restore_prepared_before_append(old)?,(None,None)=>{},_=>anyhow::bail!("shared prepared population changed")}}
                if let Some(old)=&saved.shared.parent {
                    let mut parent=self.build_shared_parent(&mut state)?;
                    parent.graph.restore_continuation(&self.stream,&old.graph)?;
                    mappings.push(serde_json::json!({"shared_handoff":parent.handoff.restore_checkpoint(&self.stream,&old.handoff)?}));
                    anyhow::ensure!(old.coupling_enabled.len()==parent.coupling_enabled.len() && old.outer_iterations.len()==parent._outer_iterations.len(),"shared parent control extent changed");
                    self.stream.memcpy_htod(&old.coupling_enabled,&mut parent.coupling_enabled)?;
                    self.stream.memcpy_htod(&old.outer_iterations,&mut parent._outer_iterations)?;
                    restore_endpoint_iterations(&self.stream,&mut parent._iterations,&old.iterations)?;
                    state.shared.parent=Some(parent);
                }
                for (id,(live,old)) in state.shared.prepared.iter().zip(&saved.shared.prepared).enumerate(){if let(Some(live),Some(old))=(live,old){
                    live.restore_prepared_pipeline(run_root,old)?;
                    // Pending gates are set below outside the immutable iteration.
                    let _=id;
                }}
                state.shared.continuation_rebind_pending=saved.shared.prepared.iter().map(|p|p.as_ref().is_some_and(|p|p.requires_binding_reconstruction())).collect();
                state.shared.archived=saved.shared.archived.clone();state.shared.activation_step=saved.shared.activation_step;
                state.shared.windows=saved.shared.windows;state.shared.final_drained=saved.shared.final_drained;
                state.shared.last_handoff=saved.shared.last_handoff.clone();
                state.shared.released_local_root_count=saved.shared.released_local_root_count;
                state.shared.retirement_completed=saved.shared.retirement_completed;
                // These graphs were already inactive and are never replayed.
                // Their complete snapshots survive as prior-process ownership
                // evidence. A dead process cannot retain a live CUDA lease.
                state.shared.prior_process_retired_roots=saved.shared.prior_process_retired_roots.clone();
                for root in &saved.shared.retired_roots{state.shared.prior_process_retired_roots.push(RetiredOwnerSnapshotV1{bytes:crate::checkpoint_codec::to_bytes(root)?});}
            }
            if let Some(old)=&saved.graph {
                let mut graph=self.build_graph(&state,old.handoff.is_some())?;
                graph.graph.restore_continuation(&self.stream,&old.graph)?;
                if let(Some(h),Some(s))=(&mut graph.handoff,&old.handoff){mappings.push(serde_json::json!({"endpoint_handoff":h.restore_checkpoint(&self.stream,s)?}));}
                restore_endpoint_iterations(&self.stream,&mut graph._iterations,&old.iterations)?;state.graph=Some(graph);
            }
            self.stream.synchronize()?;
            state.ready=saved.ready.clone();state.steps=saved.steps.clone();state.epoch=saved.epoch;state.launch_count=saved.launch_count;
            state.reconstruction_history=saved.reconstruction_history.clone();
            state.reconstruction_history.push(serde_json::json!({"schema_kind":"prism.endpoint_coordinator_reconstruction.v1",
                "at_epoch":saved.epoch,"physical_steps_advanced":0,"pointer_rebindings":mappings,
                "retired_prior_process_roots":"complete inactive snapshots retained; no old CUDA handle replayed"}));
            Ok(())
        })();
        if let Err(e)=&result {state.cancelled=Some(format!("coordinator continuation reconstruction failed: {e:#}"));self.changed.notify_all();}
        result
    }
    /// Every worker calls on its own engine while the global reconstruction
    /// barrier is held. The final native barrier release follows all receipts.
    #[cfg(feature="v2_ignition")]
    pub fn rebind_continuation_worker(&self,id:usize,engine:&mut crate::persistent_engine::PersistentNhsEngine,
        saved:&EndpointContinuationV1)->Result<Option<serde_json::Value>> {
        let mut state=self.state.lock().map_err(|_|anyhow::anyhow!("coordinator poisoned"))?;
        anyhow::ensure!(id<self.n_streams && state.cancelled.is_none() && state.epoch==saved.epoch,"invalid shared binding restore worker/epoch");
        if !state.shared.continuation_rebind_pending[id] {return Ok(None);}
        let result=(||->Result<_>{
            let shared=&mut state.shared;
            let graph=&shared.parent.as_ref().context("shared parent missing at native rebind")?.graph;
            let prepared=shared.prepared[id].as_mut().context("shared prepared owner missing at native rebind")?;
            let old=saved.shared.prepared[id].as_ref().context("shared prepared snapshot missing")?;
            engine.rebind_prepared_continuation(prepared,graph,old)
        })();
        match result{Ok(receipt)=>{state.shared.continuation_rebind_pending[id]=false;
            state.reconstruction_history.push(receipt.clone());Ok(Some(receipt))},Err(e)=>{
                state.cancelled=Some(format!("shared worker continuation binding failed: {e:#}"));self.changed.notify_all();Err(e)}}
    }
}

#[cfg(all(test,feature="v2_ignition"))]
mod continuation_contract_tests {
    use super::*;
    // These are serialized admission/refusal unit tests. They execute no MD and
    // do not stand in for native uninterrupted-versus-restart qualification.
    fn admitted_initial_boundary()->EndpointContinuationV1 {
        let rings=(0..4).map(|_|super::super::TwinOwnerContinuationV1{schema_version:1,owner_kind:"ring".into(),
            shape:vec![16,4,1,4],planes:std::collections::BTreeMap::new(),host:Vec::new()}).collect();
        EndpointContinuationV1{schema_version:1,n_streams:4,engines_per_group:1,
            workers:(0..4).map(|_|None).collect(),rings,ready:vec![false;4],steps:vec![None;4],epoch:0,launch_count:0,cancelled:None,graph:None,
            shared:SharedContinuationV1{parent:None,prepared:(0..4).map(|_|None).collect(),archived:vec![None;4],activation_step:None,
                windows:0,final_drained:false,last_handoff:None,retired_roots:Vec::new(),prior_process_retired_roots:Vec::new(),
                released_local_root_count:0,retirement_completed:false},reconstruction_history:Vec::new()}
    }
    #[test]
    fn exact_boundary_codec_and_pending_arrival_refusal() {
        let saved=admitted_initial_boundary();saved.validate_boundary().unwrap();
        let bytes=crate::checkpoint_codec::to_bytes(&saved).unwrap();
        let mut restored:EndpointContinuationV1=crate::checkpoint_codec::from_bytes(&bytes).unwrap();
        assert_eq!(bytes,crate::checkpoint_codec::to_bytes(&restored).unwrap());
        restored.ready[3]=true;restored.steps[3]=Some(500);
        assert!(restored.validate_boundary().unwrap_err().to_string().contains("checkpoint must follow"));
    }
    #[test]
    fn incomplete_population_and_cancelled_execution_are_not_continuations() {
        let mut saved=admitted_initial_boundary();saved.rings.pop();assert!(saved.validate_boundary().is_err());
        let mut saved=admitted_initial_boundary();saved.cancelled=Some("actual native failure".into());assert!(saved.validate_boundary().is_err());
        let mut saved=admitted_initial_boundary();saved.launch_count=1;assert!(saved.validate_boundary().is_err());
        let mut saved=admitted_initial_boundary();saved.shared.archived[0]=Some((500,7));assert!(saved.validate_boundary().is_err());
    }
}
