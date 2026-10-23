//! PRISM-TWIN v3.0 Gate 2 Phase C — CUDA Graph Capture
//!
//! Captures the full per-step kernel sequence as a replayable CUDA Graph:
//!   Director → Physics → multi_lif → CA restraints → heartbeat → coupling clear
//!
//! Conditional nodes (COM removal every 100 steps, heartbeat every 1000 steps)
//! are triggered by the Director kernel writing to CUgraphConditionalHandle values.
//!
//! The host loop replays the graph N times, breaking only for neighbor list rebuild
//! (still CPU-side). This eliminates ~40μs/step kernel launch overhead.

use anyhow::{bail, Context, Result};
use cudarc::driver::safe::CudaGraph;
use cudarc::driver::{
    sys, CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;


// AUTONOMOUS_LAUNCH_DOMAIN_BEGIN: host ownership only, never device chronology.
/// Shared by every physical graph issued by one persistent owner. A publication
/// lease excludes escaped graph replay until the observed endpoint is consumed.
#[derive(Default)]
pub(crate) struct PhysicalGraphLaunchDomain {
    generation: std::sync::Mutex<u64>,
}
impl PhysicalGraphLaunchDomain {
    pub(crate) fn lease(&self) -> Result<std::sync::MutexGuard<'_,u64>, &'static str> {
        self.generation.try_lock().map_err(|_| "physical graph launch domain is borrowed or poisoned")
    }
    pub(crate) fn record_launch(generation:&mut u64) -> Result<(), &'static str> {
        *generation = generation.checked_add(1).ok_or("physical graph launch generation overflow")?;
        Ok(())
    }
    pub(crate) fn verify(generation:u64, observed_generation:Option<u64>) -> Result<(), &'static str> {
        if observed_generation != Some(generation) {
            return Err("physical graph replay invalidated the observed autonomous endpoint");
        }
        Ok(())
    }
}
// AUTONOMOUS_LAUNCH_DOMAIN_END

#[cfg(test)]
mod autonomous_chunk_launch_domain_tests {
    use super::PhysicalGraphLaunchDomain;
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/autonomous_chunk_launch_domain_cases.rs"));
}

#[link(name = "while_drain_bridge", kind = "static")]
extern "C" {
    fn prism_while_drain_launch_predicate(
        stream: sys::CUstream,
        d_drain_pending: *const u32,
        d_iter_count: *mut u32,
        handle: u64,
        max_iterations: u32,
    ) -> i32;
}

// ═══════════════════════════════════════════════════════════════════════════
// Amendment 3.4 — CaptureTagger trait + raw-CUgraph escape hatch
// ═══════════════════════════════════════════════════════════════════════════
//
// The cudarc 0.18.2 safe wrapper `CudaGraph` (driver/safe/graph.rs) hides
// `cu_graph: sys::CUgraph` and `cu_graph_exec: sys::CUgraphExec` as private
// fields with no public accessors. Drop destroys both. This forecloses the
// `cudaGraphAddChildGraphNode` injection path the monolithic-fusion
// directive (Amendment 3.4) requires, since the splice operates on the
// TEMPLATE handle BEFORE instantiation.
//
// Bypass: drop to `cudarc::driver::sys` for the entire capture sequence.
// `cuStreamBeginCapture_v2` + `cuStreamEndCapture` give us a raw
// `CUgraph` template; we instantiate ourselves only AFTER any child-graph
// splicing has been applied.
//
// Tagging: `cuStreamGetCaptureInfo_v2` returns the dependency frontier of
// the in-progress capture. Calling it immediately after a kernel launch
// returns a single-element frontier — the just-recorded node. The
// `CaptureTagger` trait packages this into a label-keyed registry so the
// orchestrator can name nodes ("physical_step_complete", "multi_lif", ...) and look
// them up later as splice anchors.

/// Records named CUgraphNode handles during stream capture.
///
/// Invariants:
/// - Must be called INSIDE an active `cuStreamBeginCapture_v2` window.
/// - Must be called IMMEDIATELY after the kernel launch whose node is
///   to be tagged (any subsequent kernel launch on the same stream will
///   advance the dependency frontier).
/// - The same label may be recorded only once per template.
pub trait CaptureTagger {
    /// Snapshot the current dependency frontier on the captured stream and
    /// store the most-recently-added node under `label`.  Returns the raw
    /// `CUgraphNode` handle so the orchestrator can pass it directly to
    /// `cuGraphAddDependencies` / `cuGraphAddChildGraphNode`.
    fn record_node(&mut self, label: &str) -> Result<sys::CUgraphNode>;
}

/// Capture a pure child template on its owning stream. Launch callbacks may
/// enqueue kernels and tag nodes, but must not synchronize, read back, allocate,
/// or create conditional nodes. External pointees must outlive every replay.
/// Explicit DevicePtr guards must be resolved before this call and released
/// after it; cudarc launch-builder arguments obey the capture-mode event policy.
pub fn capture_pure_template(
    stream: Arc<CudaStream>,
    launch: impl FnOnce(&mut StreamCaptureTagger) -> Result<()>,
) -> Result<CapturedTemplate> {
    stream.context().bind_to_thread()?;
    stream.synchronize()?;
    let raw = stream.cu_stream();
    let rc = unsafe { sys::cuStreamBeginCapture_v2(raw,
        sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) };
    anyhow::ensure!(rc == sys::CUresult::CUDA_SUCCESS, "pure capture begin: {rc:?}");
    struct CaptureMode(sys::CUstream);
    impl Drop for CaptureMode {
        fn drop(&mut self) {
            // End and reclaim an abandoned capture even if the callback unwinds.
            if !self.0.is_null() {
                let mut graph = ptr::null_mut();
                unsafe {
                    let _ = sys::cuStreamEndCapture(self.0, &mut graph);
                    if !graph.is_null() { let _ = sys::cuGraphDestroy(graph); }
                }
            }
            cudarc::driver::set_capture_mode_active(false);
        }
    }
    cudarc::driver::set_capture_mode_active(true);
    let mut guard = CaptureMode(raw);
    let mut tagger = StreamCaptureTagger::new(raw);
    let launched = launch(&mut tagger);
    let mut graph = ptr::null_mut();
    let rc = unsafe { sys::cuStreamEndCapture(raw, &mut graph) };
    guard.0 = ptr::null_mut();
    drop(guard);
    if rc != sys::CUresult::CUDA_SUCCESS || launched.is_err() {
        if !graph.is_null() { unsafe { let _ = sys::cuGraphDestroy(graph); } }
        launched?;
        bail!("pure capture end: {rc:?}");
    }
    anyhow::ensure!(!graph.is_null(), "pure capture returned null template");
    Ok(CapturedTemplate::from_capture(graph, tagger.into_registry(), stream))
}

/// Default `CaptureTagger` implementation using `cuStreamGetCaptureInfo_v2`.
/// One instance per capture session. Owned by the orchestrator and consumed
/// when handing off to `CapturedTemplate::from_capture`.
pub struct StreamCaptureTagger {
    stream: sys::CUstream,
    nodes: HashMap<String, sys::CUgraphNode>,
}

impl StreamCaptureTagger {
    pub fn new(stream: sys::CUstream) -> Self {
        Self {
            stream,
            nodes: HashMap::new(),
        }
    }

    /// Consumes the tagger; returns the recorded node registry.
    pub fn into_registry(self) -> HashMap<String, sys::CUgraphNode> {
        self.nodes
    }
}

impl CaptureTagger for StreamCaptureTagger {
    fn record_node(&mut self, label: &str) -> Result<sys::CUgraphNode> {
        let mut status = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        let mut cap_id: sys::cuuint64_t = 0;
        let mut graph: sys::CUgraph = ptr::null_mut();
        let mut deps_ptr: *const sys::CUgraphNode = ptr::null();
        let mut n_deps: usize = 0;
        let rc = unsafe {
            sys::cuStreamGetCaptureInfo_v2(
                self.stream,
                &mut status as *mut _,
                &mut cap_id as *mut _,
                &mut graph as *mut _,
                &mut deps_ptr as *mut _,
                &mut n_deps as *mut _,
            )
        };
        if !matches!(rc, sys::CUresult::CUDA_SUCCESS) {
            bail!(
                "cuStreamGetCaptureInfo_v2 failed for label '{}': {:?}",
                label,
                rc
            );
        }
        if !matches!(
            status,
            sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE
        ) {
            bail!(
                "CaptureTagger::record_node('{}') called outside active capture",
                label
            );
        }
        if n_deps == 0 {
            bail!(
                "dependency frontier empty at record_node('{}') — no kernel launched yet?",
                label
            );
        }
        // Sequential capture on one stream ⇒ frontier is exactly the most
        // recent node. Driver returns the frontier as the LAST element of
        // the deps array (or only element when n_deps == 1).
        let node = unsafe { *deps_ptr.add(n_deps - 1) };
        if self.nodes.insert(label.to_string(), node).is_some() {
            bail!("CaptureTagger::record_node: duplicate label '{}'", label);
        }
        Ok(node)
    }
}

/// A captured CUgraph template, NOT yet instantiated.  Holds the raw
/// `CUgraph` handle plus the labelled-node registry built by a
/// `CaptureTagger`.  The orchestrator can:
///
///   1. Look up tagged nodes via `node()`.
///   2. Splice in child graphs via `add_child_graph_node()`.
///   3. Add dependency edges via `add_dependency()`.
///   4. Convert to a launchable `AutonomousGraph` via `instantiate()`.
///
/// On drop without instantiation the raw `CUgraph` is destroyed.
pub struct CapturedTemplate {
    physical_launch_domains: Vec<Arc<PhysicalGraphLaunchDomain>>,
    governor_devices: Vec<Arc<crate::governor_device::GovernorDevice>>,
    asc_devices: Vec<Arc<crate::asc_device::AscDevice>>,
    pipeline_leases: Vec<Arc<crate::captured_pipeline::CapturedAdjudicationPipeline>>,
    cu_graph: sys::CUgraph,
    nodes: HashMap<String, sys::CUgraphNode>,
    stream: Arc<CudaStream>,
}

impl CapturedTemplate {
    /// Construct from a raw template + node registry produced by a
    /// completed `cuStreamEndCapture` + `StreamCaptureTagger`.
    pub fn from_capture(
        cu_graph: sys::CUgraph,
        nodes: HashMap<String, sys::CUgraphNode>,
        stream: Arc<CudaStream>,
    ) -> Self {
        Self {
            cu_graph, nodes, stream,
            physical_launch_domains: Vec::new(),
            governor_devices: Vec::new(), asc_devices: Vec::new(), pipeline_leases: Vec::new(),
        }
    }

    pub(crate) fn retain_physical_launch_domain(&mut self, owner:Arc<PhysicalGraphLaunchDomain>) {
        if !self.physical_launch_domains.iter().any(|domain|Arc::ptr_eq(domain,&owner)) {
            self.physical_launch_domains.push(owner);
        }
    }
    pub fn retain_asc(&mut self, owner: Arc<crate::asc_device::AscDevice>) {
        if !self.asc_devices.iter().any(|a|Arc::ptr_eq(a,&owner)) {self.asc_devices.push(owner);}
    }
    pub fn retain_governor(&mut self, owner: Arc<crate::governor_device::GovernorDevice>) {
        if !self.governor_devices.iter().any(|g| Arc::ptr_eq(g, &owner)) { self.governor_devices.push(owner); }
    }
    pub fn retain_pipeline(&mut self, child: Arc<crate::captured_pipeline::CapturedAdjudicationPipeline>) {
        self.retain_governor(child.governor_device().clone());
        if let Some(owner)=child.asc_device() {self.retain_asc(owner.clone());}
        self.pipeline_leases.push(child);
    }
    pub fn retain_child_owners(&mut self, child: &CapturedTemplate) {
        for domain in &child.physical_launch_domains {self.retain_physical_launch_domain(domain.clone());}
        for owner in &child.asc_devices {self.retain_asc(owner.clone());}
        for owner in &child.governor_devices { self.retain_governor(owner.clone()); }
        self.pipeline_leases.extend(child.pipeline_leases.iter().cloned());
    }

    /// Raw template handle — for `cuGraphAddChildGraphNode` splicing.
    pub fn cu_graph(&self) -> sys::CUgraph {
        self.cu_graph
    }

    /// Look up a recorded node by label.  Returns `None` for unknown labels.
    pub fn node(&self, label: &str) -> Option<sys::CUgraphNode> {
        self.nodes.get(label).copied()
    }

    /// Insert `child_template` as a child-graph node downstream of `deps`.
    /// Returns the new child-graph node handle.
    ///
    /// **TIER 7 (2026-05-03) — monolithic-splice migration.**
    /// Was `cuGraphAddChildGraphNode` (legacy CUDA 11.x per-type
    /// adder).  Now routes through the unified CUDA 13.x
    /// `cudaGraphAddNode` via `crate::graph_node::add_child_graph_node_v3`,
    /// which is the actual "monolithic splice" — one
    /// `cudaGraphAddNode(graph, deps, dependencyData, n_deps,
    /// &nodeParams{type=GRAPH, .graph=child})` call replaces the
    /// legacy adder.  The legacy adder mishandles the conditional-
    /// handle metadata that CUDA 13.x attaches to body subgraphs
    /// (the source of the operator's reported ChildGraph-IF
    /// failures) — the unified API is the canonical post-CUDA-13.x
    /// way to splice templates that may carry such metadata.
    ///
    /// Returns `Err` with cudaError_t cast to `i32` (wrapped via
    /// `anyhow::anyhow!`) on failure, plus null-handle defensive
    /// check.  Behaviour at the call-site is otherwise identical to
    /// the legacy path — CUDA still copies the child template by
    /// value; future mutations to `child_template` do NOT propagate.
    pub fn add_child_graph_node(
        &mut self,
        deps: &[sys::CUgraphNode],
        child_template: sys::CUgraph,
    ) -> Result<sys::CUgraphNode> {
        // TIER 7 + TIER 8: routes through the v3 path AND runs the
        // splice-legality preflight before any cudaGraphAddNode call.
        // SpliceError::Illegal carries a SpliceLegalityReport describing
        // exactly which child template failed and why — surfaces well
        // before CUDA's own cudaErrorNotSupported (801).
        let child_node = unsafe {
            crate::graph_node::add_child_graph_node_v3(self.cu_graph, deps, child_template)
        }
        .map_err(|e| anyhow::anyhow!("add_child_graph_node: {}", e))?;
        // The v3 helper already null-checks (SpliceError::NullNode) but
        // re-assert here for defence-in-depth — a null handle past this
        // point would corrupt the caller's dependency frontier.
        if child_node.is_null() {
            bail!("add_child_graph_node_v3 returned null node handle (post-check)");
        }
        Ok(child_node)
    }

    /// Add an explicit dependency edge `from → to` in the template.
    pub fn add_dependency(&mut self, from: sys::CUgraphNode, to: sys::CUgraphNode) -> Result<()> {
        let from_arr = [from];
        let to_arr = [to];
        let rc = unsafe {
            sys::cuGraphAddDependencies(self.cu_graph, from_arr.as_ptr(), to_arr.as_ptr(), 1)
        };
        if !matches!(rc, sys::CUresult::CUDA_SUCCESS) {
            bail!("cuGraphAddDependencies failed: {:?}", rc);
        }
        Ok(())
    }

    /// Instantiate the template into an executable graph.  Consumes self.
    /// On success the returned `AutonomousGraph` owns both the template
    /// (kept for `cuGraphExecUpdate` if ever needed) and the exec handle.
    pub fn instantiate(self) -> Result<AutonomousGraph> {
        self.instantiate_with_runtime_state(1, None)
    }

    /// Instantiate the template with explicit logical-step coverage and
    /// optional device-resident loop state owned by the returned graph.
    pub fn instantiate_with_runtime_state(
        mut self,
        steps_per_launch: u32,
        resident_loop_state: Option<ResidentLoopState>,
    ) -> Result<AutonomousGraph> {
        let mut cu_graph_exec: sys::CUgraphExec = ptr::null_mut();
        let rc = unsafe {
            sys::cuGraphInstantiateWithFlags(
                &mut cu_graph_exec as *mut _,
                self.cu_graph,
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH
                    as u64,
            )
        };
        if !matches!(rc, sys::CUresult::CUDA_SUCCESS) {
            bail!("cuGraphInstantiateWithFlags failed: {:?}", rc);
        }
        let cu_graph = self.cu_graph;
        // Transfer every pointee lease as well as the raw graph. Dropping
        // the emptied template frees its registry rather than leaking it.
        self.cu_graph = ptr::null_mut();
        let governor_devices = std::mem::take(&mut self.governor_devices);
        let asc_devices = std::mem::take(&mut self.asc_devices);
        let pipeline_leases = std::mem::take(&mut self.pipeline_leases);
        let stream = self.stream.clone();
        let mut graph = AutonomousGraph {
            physical_launch_domains: std::mem::take(&mut self.physical_launch_domains),
            graph: None,
            raw: Some(RawGraphExec {
                cu_graph, cu_graph_exec, stream, governor_devices, asc_devices, pipeline_leases,
                captured_structure_sha256: String::new(), resident_structure_sha256: None,
            }),
            steps_per_launch: steps_per_launch.max(1),
            resident_loop_state,
            ncmc_resident_capture_identity: None,
            launch_residue: RefCell::new(ChunkResidueLedger::default()),
        };
        // Query native topology only at this immutable owner boundary. The
        // replay binding still checks domain/nonce and exact driver handles.
        // A new graph/rebuild cannot inherit another owner's cache.
        let captured = graph.compute_captured_structure_fingerprint_sha256()?;
        let resident = if graph.resident_loop_state.is_some() {
            Some(graph.compute_resident_structure_fingerprint_sha256()?)
        } else { None };
        let raw = graph.raw.as_mut().expect("new native graph has raw owner");
        raw.captured_structure_sha256 = captured;
        raw.resident_structure_sha256 = resident;
        Ok(graph)
    }
}

impl Drop for CapturedTemplate {
    fn drop(&mut self) {
        if !self.cu_graph.is_null() {
            unsafe {
                let _ = sys::cuGraphDestroy(self.cu_graph);
            }
            self.cu_graph = ptr::null_mut();
        }
    }
}

/// Raw-handle backing for `AutonomousGraph` when the new sys-bypass capture
/// path is used (Amendment 3.4 monolithic fusion).  Both handles must be
/// destroyed on drop in order: exec first, then template.
struct RawGraphExec {
    stream: Arc<CudaStream>,
    governor_devices: Vec<Arc<crate::governor_device::GovernorDevice>>,
    asc_devices: Vec<Arc<crate::asc_device::AscDevice>>,
    pipeline_leases: Vec<Arc<crate::captured_pipeline::CapturedAdjudicationPipeline>>,
    cu_graph: sys::CUgraph,
    cu_graph_exec: sys::CUgraphExec,
    // These describe immutable instantiated topology, not mutable kernel data.
    captured_structure_sha256: String,
    resident_structure_sha256: Option<String>,
}

impl RawGraphExec {
    fn checked_drain(&self) -> Result<()> {
        for owner in &self.asc_devices { owner.snapshot().map_err(|rc|anyhow::anyhow!("ASC graph drain failed rc={}",rc))?; }
        if self.governor_devices.is_empty() {
            crate::governor_device::drain_stream(&self.stream).map_err(|rc| anyhow::anyhow!("graph drain failed rc={}", rc))?;
        } else {
            for owner in &self.governor_devices {
                owner.snapshot().map_err(|rc| anyhow::anyhow!("governed graph drain failed rc={}", rc))?;
            }
        }
        // Parent-owned conditionals may schedule child work beyond the
        // stream-local owner snapshot. Establish the same context retirement
        // boundary as the native coupled endpoint before destroying handles.
        // Failure is handled by AutonomousGraph's existing owner quarantine.
        self.stream.context().synchronize().map_err(|error|
            anyhow::anyhow!("parent graph context retirement failed: {:?}", error))?;
        Ok(())
    }
}

impl Drop for RawGraphExec {
    fn drop(&mut self) {
        unsafe {
            if !self.cu_graph_exec.is_null() {
                let _ = sys::cuGraphExecDestroy(self.cu_graph_exec);
                self.cu_graph_exec = ptr::null_mut();
            }
            if !self.cu_graph.is_null() {
                let _ = sys::cuGraphDestroy(self.cu_graph);
                self.cu_graph = ptr::null_mut();
            }
        }
    }
}

/// Immutable evidence identity for the separately captured reversible NCMC
/// graph. Node topology alone cannot identify kernel functions, arguments or
/// mutable device allocations.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NcmcResidentGraphCaptureIdentity {
    pub schema_kind: String,
    pub schema_version: u32,
    pub capture_body_kind: String,
    pub capture_source_schedule_sha256: String,
    pub selected_fused_ptx_sha256: String,
    pub kernel_symbol_manifest_sha256: String,
    pub mutable_state_allocation_binding_sha256: String,
    pub constraint_work_plane_sha256: String,
    pub protocol_state_guard_sha256: String,
    pub integrator_splitting: String,
    pub timestep_ps: f64,
    pub friction_gamma_per_ps: f64,
    pub stochastic_thermostat_enabled: bool,
    pub rng_state_read_write_enabled: bool,
    pub adaptive_dt_enabled: bool,
    pub autonomous_rescue_enabled: bool,
    pub force_clamp_enabled: bool,
    pub uv_bias_enabled: bool,
    pub momentum_reversal_required: bool,
    pub identity_sha256: String,
}

impl NcmcResidentGraphCaptureIdentity {
    fn recompute_sha256(&self) -> Result<String> {
        let mut projection = self.clone();
        projection.identity_sha256.clear();
        let bytes = serde_json::to_vec(&projection)?;
        let mut digest = Sha256::new();
        digest.update(b"PRISM_NCMC_RESIDENT_GRAPH_CAPTURE_IDENTITY_V3\0");
        digest.update(bytes);
        Ok(format!("{:x}", digest.finalize()))
    }

    pub fn validate(&self) -> Result<()> {
        let valid_sha =
            |value: &str| value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit());
        if self.schema_kind != "prismos_ncmc_resident_graph_capture_identity"
            || self.schema_version != 3
            || self.capture_body_kind != "NCMC_GAMMA_ZERO_REVERSIBLE_WHILE_V2"
            || !valid_sha(&self.capture_source_schedule_sha256)
            || !valid_sha(&self.selected_fused_ptx_sha256)
            || !valid_sha(&self.kernel_symbol_manifest_sha256)
            || !valid_sha(&self.mutable_state_allocation_binding_sha256)
            || !valid_sha(&self.constraint_work_plane_sha256)
            || !valid_sha(&self.protocol_state_guard_sha256)
            || self.integrator_splitting != "FORCE_KICK_RATTLE_DRIFT_SHAKE_FORCE_KICK_RATTLE_V2"
            || !self.timestep_ps.is_finite()
            || self.timestep_ps <= 0.0
            || self.friction_gamma_per_ps.to_bits() != 0.0f64.to_bits()
            || self.stochastic_thermostat_enabled
            || self.rng_state_read_write_enabled
            || self.adaptive_dt_enabled
            || self.autonomous_rescue_enabled
            || self.force_clamp_enabled
            || self.uv_bias_enabled
            || !self.momentum_reversal_required
            || self.identity_sha256 != self.recompute_sha256()?
        {
            bail!("invalid or non-reversible NCMC resident graph capture identity");
        }
        Ok(())
    }

    /// Seal an executable capture identity from the live graph-builder
    /// projection. Callers provide evidence hashes derived from the selected
    /// PTX, exact kernel/argument sequence, mutable allocations, constraint
    /// work plane and the gamma-zero protocol-state guard. The identity hash
    /// is never accepted from an external caller.
    #[allow(clippy::too_many_arguments)]
    pub fn seal(
        capture_source_schedule_sha256: String,
        selected_fused_ptx_sha256: String,
        kernel_symbol_manifest_sha256: String,
        mutable_state_allocation_binding_sha256: String,
        constraint_work_plane_sha256: String,
        protocol_state_guard_sha256: String,
        timestep_ps: f64,
    ) -> Result<Self> {
        let mut identity = Self {
            schema_kind: "prismos_ncmc_resident_graph_capture_identity".into(),
            schema_version: 3,
            capture_body_kind: "NCMC_GAMMA_ZERO_REVERSIBLE_WHILE_V2".into(),
            capture_source_schedule_sha256,
            selected_fused_ptx_sha256,
            kernel_symbol_manifest_sha256,
            mutable_state_allocation_binding_sha256,
            constraint_work_plane_sha256,
            protocol_state_guard_sha256,
            integrator_splitting: "FORCE_KICK_RATTLE_DRIFT_SHAKE_FORCE_KICK_RATTLE_V2".into(),
            timestep_ps,
            friction_gamma_per_ps: 0.0,
            stochastic_thermostat_enabled: false,
            rng_state_read_write_enabled: false,
            adaptive_dt_enabled: false,
            autonomous_rescue_enabled: false,
            force_clamp_enabled: false,
            uv_bias_enabled: false,
            momentum_reversal_required: true,
            identity_sha256: String::new(),
        };
        identity.identity_sha256 = identity.recompute_sha256()?;
        identity.validate()?;
        Ok(identity)
    }
}

/// Holds the captured graph and associated state for autonomous execution.
///
/// Supports two construction paths:
///   - Legacy: `new(CudaGraph)` — wraps the cudarc safe wrapper. Used by
///     existing TWIN dual-engine path; keeps backward compat.
///   - Amendment 3.4: `from_template_instantiate()` (via `CapturedTemplate`)
///     — raw `CUgraph` + `CUgraphExec` for monolithic-fusion injection.
pub struct AutonomousGraph {
    physical_launch_domains: Vec<Arc<PhysicalGraphLaunchDomain>>,
    /// Legacy cudarc safe wrapper. `Some` for non-fused captures.
    graph: Option<CudaGraph>,
    /// Raw-handle backing. `Some` after `CapturedTemplate::instantiate()`.
    /// Exactly one of `graph` / `raw` is `Some`.
    raw: Option<RawGraphExec>,
    /// Number of steps captured in this graph (always 1 for single-step capture)
    steps_per_launch: u32,
    /// Keeps device-resident WHILE predicate buffers alive for graphs that
    /// execute repeated MD bodies inside CUDA conditional nodes.
    resident_loop_state: Option<ResidentLoopState>,
    /// Absent for generic and Langevin graphs; required by NCMC propagation.
    ncmc_resident_capture_identity: Option<NcmcResidentGraphCaptureIdentity>,
    /// Per-chunk launch-granularity residue ledger. A chunk whose logical-step
    /// count is not a multiple of `steps_per_launch` executes MORE device steps
    /// than the caller asked for, and the excess accumulates across chunks.
    /// The ledger makes every overshoot reconstructable instead of silent.
    launch_residue: RefCell<ChunkResidueLedger>,
}

/// One compacted-chunk launch accounting row.
///
/// `device_steps_executed = launches * steps_per_launch` is what the GPU
/// actually advanced; `logical_steps_requested` is what the caller believed it
/// advanced. `residue_steps` is the difference. Nothing here is estimated —
/// every field is computed from the same integers used to issue the launches.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ChunkResidueRecord {
    /// 0-based index of the compacted-chunk call on this graph.
    pub call_index: u64,
    /// Logical MD steps the caller asked to cover.
    pub logical_steps_requested: u32,
    /// MD steps advanced by one graph replay (captured body granularity).
    pub steps_per_launch: u32,
    /// Host graph launches actually issued (lower than planned if a launch
    /// failed part-way through the chunk).
    pub launches: u32,
    /// MD steps the device actually advanced for this chunk.
    pub device_steps_executed: u64,
    /// `device_steps_executed - logical_steps_requested`. Positive is
    /// launch-granularity overshoot; negative means the chunk aborted before
    /// covering the request.
    pub step_delta: i64,
    /// Overshoot only: `max(step_delta, 0)`.
    pub residue_steps: u32,
    /// Running total of `residue_steps` over every chunk on this graph.
    pub cumulative_residue_steps: u64,
}

/// Bounded ledger of launch-granularity residues.
///
/// Aggregates are exact for the whole run. The per-chunk row buffer is bounded
/// so a long run cannot grow it without limit; when rows are dropped the count
/// is reported explicitly, so a truncated ledger can never be mistaken for a
/// complete one.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct ChunkResidueLedger {
    calls: u64,
    total_logical_steps_requested: u64,
    total_device_steps_executed: u64,
    total_residue_steps: u64,
    total_step_delta: i64,
    max_residue_steps: u32,
    first_residue_call_index: Option<u64>,
    aborted_chunks: u64,
    records: Vec<ChunkResidueRecord>,
    records_dropped: u64,
}

/// Maximum per-chunk rows retained in memory. Aggregates stay exact past this
/// bound; only the row detail is truncated, and `records_dropped` says by how
/// much.
const CHUNK_RESIDUE_RECORD_CAPACITY: usize = 4096;

/// Terminal error tag: a caller demanded exact launch alignment and the chunk
/// was not a multiple of the captured body granularity.
pub const GRAPH_CHUNK_NOT_LAUNCH_ALIGNED: &str = "PRISM_GRAPH_CHUNK_NOT_LAUNCH_ALIGNED";
/// Terminal error tag: `run_chunk` was called on a multi-step body, where its
/// "one launch per step" contract is ambiguous.
pub const GRAPH_CHUNK_UNIT_AMBIGUOUS: &str = "PRISM_GRAPH_CHUNK_UNIT_AMBIGUOUS";

impl ChunkResidueLedger {
    fn record(
        &mut self,
        logical_steps_requested: u32,
        steps_per_launch: u32,
        launches: u32,
    ) -> ChunkResidueRecord {
        let device_steps_executed = u64::from(launches) * u64::from(steps_per_launch);
        let step_delta = device_steps_executed as i64 - i64::from(logical_steps_requested);
        let residue_steps = step_delta.max(0).min(i64::from(u32::MAX)) as u32;
        let call_index = self.calls;
        self.calls += 1;
        self.total_logical_steps_requested += u64::from(logical_steps_requested);
        self.total_device_steps_executed += device_steps_executed;
        self.total_residue_steps += u64::from(residue_steps);
        self.total_step_delta = self.total_step_delta.saturating_add(step_delta);
        if step_delta < 0 {
            self.aborted_chunks += 1;
        }
        if residue_steps > self.max_residue_steps {
            self.max_residue_steps = residue_steps;
        }
        if residue_steps > 0 && self.first_residue_call_index.is_none() {
            self.first_residue_call_index = Some(call_index);
        }
        let record = ChunkResidueRecord {
            call_index,
            logical_steps_requested,
            steps_per_launch,
            launches,
            device_steps_executed,
            step_delta,
            residue_steps,
            cumulative_residue_steps: self.total_residue_steps,
        };
        if self.records.len() < CHUNK_RESIDUE_RECORD_CAPACITY {
            self.records.push(record);
        } else {
            self.records_dropped += 1;
        }
        record
    }
}

impl Drop for AutonomousGraph {
    fn drop(&mut self) {
        if let Some(raw) = &self.raw {
            if let Err(error) = raw.checked_drain() {
                log::error!("PARENT_GRAPH_QUARANTINED: {}", error);
                if let Some(raw) = self.raw.take() { std::mem::forget(raw); }
                if let Some(state) = self.resident_loop_state.take() { std::mem::forget(state); }
                if let Some(graph) = self.graph.take() { std::mem::forget(graph); }
            }
        }
    }
}

impl AutonomousGraph {
    /// Create from an instantiated cudarc-wrapped CUDA graph (legacy path).
    pub fn new(graph: CudaGraph) -> Self {
        Self::new_with_steps_per_launch(graph, 1)
    }

    /// Create from an instantiated cudarc-wrapped CUDA graph and bind the
    /// logical MD-step count advanced by each graph replay.
    pub fn new_with_steps_per_launch(graph: CudaGraph, steps_per_launch: u32) -> Self {
        Self {
            graph: Some(graph),
            raw: None,
            physical_launch_domains: Vec::new(),
            steps_per_launch: steps_per_launch.max(1),
            resident_loop_state: None,
            ncmc_resident_capture_identity: None,
            launch_residue: RefCell::new(ChunkResidueLedger::default()),
        }
    }

    pub(crate) fn retain_physical_launch_domain(&mut self, owner:Arc<PhysicalGraphLaunchDomain>) {
        if !self.physical_launch_domains.iter().any(|domain|Arc::ptr_eq(domain,&owner)) {
            self.physical_launch_domains.push(owner);
        }
    }
    pub(crate) fn contains_physical_launch_domain(&self, owner:&Arc<PhysicalGraphLaunchDomain>) -> bool {
        self.physical_launch_domains.iter().any(|domain|Arc::ptr_eq(domain,owner))
    }
    fn begin_physical_launch(&self) -> Result<Vec<std::sync::MutexGuard<'_,u64>>> {
        // Acquire every worker before changing any generation. try_lock avoids
        // lock-order deadlock in a composed multi-owner graph; failure launches
        // nothing. Advance before CUDA so an uncertain launch poisons receipts.
        let mut leases=Vec::with_capacity(self.physical_launch_domains.len());
        for domain in &self.physical_launch_domains {
            leases.push(domain.lease().map_err(anyhow::Error::msg)?);
        }
        for generation in &mut leases {
            PhysicalGraphLaunchDomain::record_launch(&mut **generation).map_err(anyhow::Error::msg)?;
        }
        Ok(leases)
    }

    /// Raw `CUgraph` template handle — Amendment 3.4 splice path only.
    /// Returns `None` when constructed via the legacy `new(CudaGraph)`
    /// constructor (cudarc hides the template).
    pub fn cu_graph_template(&self) -> Option<sys::CUgraph> {
        self.raw.as_ref().map(|r| r.cu_graph)
    }

    /// Raw `CUgraphExec` — Amendment 3.4 splice path only.
    pub fn cu_graph_exec_raw(&self) -> Option<sys::CUgraphExec> {
        self.raw.as_ref().map(|r| r.cu_graph_exec)
    }

    /// Launch the graph once. The captured body may advance multiple logical
    /// MD steps when the capture was built with fused inner steps and body
    /// unroll enabled.
    pub fn launch(&self) -> Result<()> {
        let _leases=self.begin_physical_launch()?;
        if let Some(ref g) = self.graph {
            g.launch()
                .map_err(|e| anyhow::anyhow!("Graph launch failed: {:?}", e))
        } else if let Some(ref r) = self.raw {
            // Raw launch path — call cuGraphLaunch directly. The orchestrator
            // is responsible for keeping the stream alive for the lifetime of
            // this AutonomousGraph (passed via the original CapturedTemplate).
            // For Amendment 3.4 we currently launch on a stream provided by
            // the caller via `launch_on_stream`.
            bail!(
                "AutonomousGraph::launch() called on raw-handle backed graph — \
                   use launch_on_stream(stream) instead"
            )
        } else {
            bail!("AutonomousGraph in invalid state — neither cudarc nor raw handle present")
        }
    }

    pub fn steps_per_launch(&self) -> u32 {
        self.steps_per_launch.max(1)
    }

    pub fn resident_loop_status(&self) -> Option<ResidentLoopStatus> {
        self.resident_loop_state
            .as_ref()
            .map(|state| ResidentLoopStatus {
                max_iterations: state.max_iterations,
                pending_dev_ptr: state.pending_dev_ptr,
                iter_dev_ptr: state.iter_dev_ptr,
            })
    }

    /// Attach the hash-closed capture authority produced by the dedicated
    /// deterministic NCMC graph builder. A generic captured MD graph cannot
    /// acquire this identity by inference.
    pub fn bind_ncmc_resident_capture_identity(
        &mut self,
        identity: NcmcResidentGraphCaptureIdentity,
    ) -> Result<()> {
        if self.raw.is_none() || self.resident_loop_state.is_none() {
            bail!("NCMC capture identity requires a raw WHILE-resident graph");
        }
        identity.validate()?;
        if self.ncmc_resident_capture_identity.is_some() {
            bail!("NCMC resident graph capture identity is immutable once bound");
        }
        self.ncmc_resident_capture_identity = Some(identity);
        Ok(())
    }

    pub fn ncmc_resident_capture_identity(&self) -> Option<&NcmcResidentGraphCaptureIdentity> {
        self.ncmc_resident_capture_identity.as_ref()
    }

    /// Structural witness of the live raw graph template. This is not a graph
    /// executable identity: kernel symbols/arguments and mutable allocations
    /// are bound by `NcmcResidentGraphCaptureIdentity`.
    /// Instantiation-verified raw topology. No driver enumeration on replay.
    pub fn captured_structure_fingerprint_sha256(&self) -> Result<String> {
        let raw=self.raw.as_ref().ok_or_else(||anyhow::anyhow!("captured fingerprint requires raw backing"))?;
        anyhow::ensure!(!raw.captured_structure_sha256.is_empty(),"native graph identity cache not initialized");
        Ok(raw.captured_structure_sha256.clone())
    }

    pub fn resident_structure_fingerprint_sha256(&self) -> Result<String> {
        self.raw.as_ref().and_then(|raw|raw.resident_structure_sha256.clone())
            .ok_or_else(||anyhow::anyhow!("graph lacks instantiation-verified resident-loop identity"))
    }

    fn captured_structure_parts(&self) -> Result<(Vec<u32>, Vec<(u64,u64)>)> {
        let raw = self
            .raw
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("resident fingerprint requires raw graph backing"))?;
        let mut node_count = 0usize;
        let rc = unsafe { sys::cuGraphGetNodes(raw.cu_graph, ptr::null_mut(), &mut node_count) };
        if !matches!(rc, sys::CUresult::CUDA_SUCCESS) || node_count == 0 {
            bail!(
                "cuGraphGetNodes(count) failed or returned an empty graph: {:?}",
                rc
            );
        }
        let mut nodes = vec![ptr::null_mut(); node_count];
        let rc = unsafe { sys::cuGraphGetNodes(raw.cu_graph, nodes.as_mut_ptr(), &mut node_count) };
        if !matches!(rc, sys::CUresult::CUDA_SUCCESS) {
            bail!("cuGraphGetNodes failed: {:?}", rc);
        }
        nodes.truncate(node_count);
        let indices: HashMap<usize, u64> = nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (*node as usize, index as u64))
            .collect();
        let mut node_types = Vec::with_capacity(nodes.len());
        for node in &nodes {
            let mut node_type: sys::CUgraphNodeType = unsafe { std::mem::zeroed() };
            let rc = unsafe { sys::cuGraphNodeGetType(*node, &mut node_type) };
            if !matches!(rc, sys::CUresult::CUDA_SUCCESS) {
                bail!("cuGraphNodeGetType failed: {:?}", rc);
            }
            node_types.push(node_type as u32);
        }
        let mut edge_count = 0usize;
        let rc = unsafe {
            sys::cuGraphGetEdges(
                raw.cu_graph,
                ptr::null_mut(),
                ptr::null_mut(),
                &mut edge_count,
            )
        };
        if !matches!(rc, sys::CUresult::CUDA_SUCCESS) {
            bail!("cuGraphGetEdges(count) failed: {:?}", rc);
        }
        let mut from = vec![ptr::null_mut(); edge_count];
        let mut to = vec![ptr::null_mut(); edge_count];
        if edge_count > 0 {
            let rc = unsafe {
                sys::cuGraphGetEdges(
                    raw.cu_graph,
                    from.as_mut_ptr(),
                    to.as_mut_ptr(),
                    &mut edge_count,
                )
            };
            if !matches!(rc, sys::CUresult::CUDA_SUCCESS) {
                bail!("cuGraphGetEdges failed: {:?}", rc);
            }
        }
        let mut edges: Vec<(u64, u64)> = from
            .into_iter()
            .zip(to)
            .take(edge_count)
            .map(|(from, to)| {
                Ok((
                    *indices
                        .get(&(from as usize))
                        .ok_or_else(|| anyhow::anyhow!("graph edge source is not in node set"))?,
                    *indices
                        .get(&(to as usize))
                        .ok_or_else(|| anyhow::anyhow!("graph edge target is not in node set"))?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        edges.sort_unstable();
        Ok((node_types,edges))
    }

    /// Real raw graph topology without inventing a singular loop state for
    /// multi-engine parents. Per-engine physical cadence is bound separately.
    fn compute_captured_structure_fingerprint_sha256(&self) -> Result<String> {
        let (node_types,edges) = self.captured_structure_parts()?;
        let mut digest=Sha256::new();
        digest.update(b"PRISM_CAPTURED_GRAPH_STRUCTURE_V1\0");
        digest.update((node_types.len() as u64).to_le_bytes());
        for node_type in node_types {digest.update(node_type.to_le_bytes());}
        digest.update((edges.len() as u64).to_le_bytes());
        for (from,to) in edges {digest.update(from.to_le_bytes());digest.update(to.to_le_bytes());}
        Ok(format!("{:x}",digest.finalize()))
    }

    fn compute_resident_structure_fingerprint_sha256(&self) -> Result<String> {
        let (node_types,edges) = self.captured_structure_parts()?;
        let status = self
            .resident_loop_status()
            .ok_or_else(|| anyhow::anyhow!("graph lacks resident-loop state"))?;
        let mut digest = Sha256::new();
        digest.update(b"PRISM_RESIDENT_GRAPH_STRUCTURE_V1\0");
        digest.update((node_types.len() as u64).to_le_bytes());
        for node_type in node_types {
            digest.update(node_type.to_le_bytes());
        }
        digest.update((edges.len() as u64).to_le_bytes());
        for (from, to) in edges {
            digest.update(from.to_le_bytes());
            digest.update(to.to_le_bytes());
        }
        digest.update(status.max_iterations.to_le_bytes());
        Ok(format!("{:x}", digest.finalize()))
    }

    /// Launch the raw-handle graph on the specified stream (Amendment 3.4).
    /// Returns an error when the legacy cudarc-wrapper backing is used.
    pub fn launch_on_stream(&self, stream: sys::CUstream) -> Result<()> {
        let raw = self
            .raw
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("launch_on_stream requires raw-handle backing"))?;
        if stream != raw.stream.cu_stream() {
            bail!("raw graph launch must use its captured owner stream");
        }
        for owner in &raw.asc_devices { if !owner.owns_raw_stream(stream.cast()) {bail!("ASC graph owner stream mismatch");} }
        for owner in &raw.governor_devices {
            if !owner.owns_raw_stream(stream.cast()) { bail!("governor graph owner stream mismatch"); }
        }
        let _leases=self.begin_physical_launch()?;
        let rc = unsafe { sys::cuGraphLaunch(raw.cu_graph_exec, stream) };
        if !matches!(rc, sys::CUresult::CUDA_SUCCESS) {
            bail!("cuGraphLaunch failed: {:?}", rc);
        }
        Ok(())
    }

    /// Explicit multi-worker parent route. The private coordinator capability
    /// binds the exact graph, exclusive rendezvous and live same-context worker
    /// streams whose engine owners retained retirement leases. This does not
    /// weaken `launch_on_stream` for ordinary captured graphs.
    pub(crate) fn launch_on_coordinator_stream(
        &self,
        stream: &Arc<CudaStream>,
        permit: &crate::twin_kernels::endpoint_capture::CoordinatorLaunchPermit<'_>,
    ) -> Result<()> {
        let raw = self.raw.as_ref().context("shared parent requires raw graph backing")?;
        anyhow::ensure!(Arc::ptr_eq(&raw.stream, stream),
            "shared parent launch stream differs from its actual graph owner");
        permit.verify_graph(self, stream)?;
        for owner in &raw.asc_devices {
            anyhow::ensure!(permit.contains_owner_stream(|s| owner.owns_stream(s)),
                "shared parent ASC owner is not a registered fenced physical worker");
        }
        for owner in &raw.governor_devices {
            anyhow::ensure!(permit.contains_owner_stream(|s| owner.owns_stream(s)),
                "shared parent governor owner is not a registered fenced physical worker");
        }
        stream.context().bind_to_thread()?;
        let _leases=self.begin_physical_launch()?;
        let rc = unsafe {sys::cuGraphLaunch(raw.cu_graph_exec, stream.cu_stream())};
        anyhow::ensure!(rc == sys::CUresult::CUDA_SUCCESS,
            "shared coupled parent cuGraphLaunch failed: {rc:?}");
        Ok(())
    }

    /// Run the graph for `n_steps` in a tight loop, breaking every `chunk_size`
    /// steps to allow the caller to do CPU-side housekeeping (NL rebuild).
    ///
    /// Returns the number of steps actually executed. This entry point equates
    /// one launch with one MD step, which is only true for a single-step body.
    /// On a multi-step body the caller's "steps" and the device's steps differ
    /// by `steps_per_launch`, so the call is refused rather than silently
    /// multiplied — use `run_chunk_compacted*`, which accounts for granularity.
    pub fn run_chunk(&self, n_steps: u32) -> Result<u32> {
        let steps_per_launch = self.steps_per_launch();
        if steps_per_launch != 1 {
            bail!(
                "{}: run_chunk({}) treats one launch as one MD step, but this graph body \
                 advances {} steps per launch; call run_chunk_compacted_on_stream instead",
                GRAPH_CHUNK_UNIT_AMBIGUOUS,
                n_steps,
                steps_per_launch
            );
        }
        let mut issued = 0u32;
        let mut outcome = Ok(());
        while issued < n_steps {
            if let Err(error) = self.launch() {
                outcome = Err(error);
                break;
            }
            issued += 1;
        }
        // Record what the device actually advanced, including on the abort
        // path — a chunk that died half-way still moved physical time.
        self.record_chunk_residue(n_steps, steps_per_launch, issued);
        outcome?;
        Ok(n_steps)
    }

    /// Largest logical-step count `<= requested` that this graph can cover with
    /// no launch-granularity overshoot. Lets a caller plan an exact chunk
    /// schedule instead of discovering the residue afterwards.
    pub fn aligned_logical_steps(&self, requested: u32) -> u32 {
        let steps_per_launch = self.steps_per_launch();
        requested - (requested % steps_per_launch)
    }

    /// Launch-granularity residue a chunk of `logical_steps` would incur.
    pub fn projected_chunk_residue(&self, logical_steps: u32) -> u32 {
        let steps_per_launch = self.steps_per_launch();
        let remainder = logical_steps % steps_per_launch;
        if remainder == 0 {
            0
        } else {
            steps_per_launch - remainder
        }
    }

    /// Run enough graph replays to cover `logical_steps`. Returns the actual
    /// number of host graph launches issued. This is the PrismOS hot-path path:
    /// the MD body stays captured on the GPU, and host launch count is reduced
    /// by `steps_per_launch`.
    ///
    /// A captured body is indivisible: a chunk that is not a multiple of
    /// `steps_per_launch` can only be covered by rounding UP, which advances the
    /// device further than the caller asked. The excess is real physical time,
    /// it is invisible to the caller's step accounting, and it ACCUMULATES over
    /// chunks. It is therefore recorded per chunk in the residue ledger
    /// (`chunk_residue_ledger_json`) so the overshoot is exactly
    /// reconstructable. Callers that require exactness should call
    /// `run_chunk_exact_on_stream`, or size chunks with `aligned_logical_steps`.
    pub fn run_chunk_compacted(&self, logical_steps: u32) -> Result<u32> {
        let steps_per_launch = self.steps_per_launch();
        let launches = logical_steps.div_ceil(steps_per_launch);
        let mut issued = 0u32;
        let mut outcome = Ok(());
        while issued < launches {
            if let Err(error) = self.launch() {
                outcome = Err(error);
                break;
            }
            issued += 1;
        }
        self.record_chunk_residue(logical_steps, steps_per_launch, issued);
        outcome?;
        Ok(launches)
    }

    /// Same as `run_chunk_compacted`, but supports raw-handle graphs by
    /// launching them on the caller-provided CUDA stream.
    pub fn run_chunk_compacted_on_stream(
        &self,
        logical_steps: u32,
        stream: sys::CUstream,
    ) -> Result<u32> {
        let steps_per_launch = self.steps_per_launch();
        let launches = logical_steps.div_ceil(steps_per_launch);
        let mut issued = 0u32;
        let mut outcome = Ok(());
        while issued < launches {
            let launched = if self.raw.is_some() {
                self.launch_on_stream(stream)
            } else {
                self.launch()
            };
            if let Err(error) = launched {
                outcome = Err(error);
                break;
            }
            issued += 1;
        }
        self.record_chunk_residue(logical_steps, steps_per_launch, issued);
        outcome?;
        Ok(launches)
    }

    /// Exact-coverage variant: refuses to launch at all unless `logical_steps`
    /// is an exact multiple of the captured body granularity. Fails before any
    /// GPU work with a named terminal error, so a caller that needs step-exact
    /// accounting can never silently acquire drift.
    pub fn run_chunk_exact_on_stream(
        &self,
        logical_steps: u32,
        stream: sys::CUstream,
    ) -> Result<u32> {
        let steps_per_launch = self.steps_per_launch();
        if logical_steps % steps_per_launch != 0 {
            bail!(
                "{}: chunk of {} logical steps is not a multiple of the captured body \
                 granularity {} (would overshoot by {} steps); align the chunk with \
                 aligned_logical_steps() or capture an exact-tail body",
                GRAPH_CHUNK_NOT_LAUNCH_ALIGNED,
                logical_steps,
                steps_per_launch,
                self.projected_chunk_residue(logical_steps)
            );
        }
        self.run_chunk_compacted_on_stream(logical_steps, stream)
    }

    fn record_chunk_residue(&self, logical_steps: u32, steps_per_launch: u32, launches: u32) {
        let record =
            self.launch_residue
                .borrow_mut()
                .record(logical_steps, steps_per_launch, launches);
        if record.residue_steps > 0 {
            log::warn!(
                "graph chunk {} overshoot: requested {} logical steps, body granularity {}, \
                 {} launches => {} device steps (residue {} this chunk, {} cumulative)",
                record.call_index,
                record.logical_steps_requested,
                record.steps_per_launch,
                record.launches,
                record.device_steps_executed,
                record.residue_steps,
                record.cumulative_residue_steps
            );
        } else if record.step_delta < 0 {
            log::warn!(
                "graph chunk {} aborted after {} of {} launches: device advanced {} of {} \
                 requested logical steps ({} short)",
                record.call_index,
                record.launches,
                record
                    .logical_steps_requested
                    .div_ceil(record.steps_per_launch),
                record.device_steps_executed,
                record.logical_steps_requested,
                -record.step_delta
            );
        }
    }

    /// Total MD steps the device advanced beyond what callers requested, over
    /// every compacted chunk on this graph. `0` means the chunk schedule was
    /// exactly launch-aligned.
    pub fn cumulative_launch_residue_steps(&self) -> u64 {
        self.launch_residue.borrow().total_residue_steps
    }

    /// Per-chunk residue rows retained in memory (bounded — see
    /// `chunk_residue_ledger_json` for the dropped-row count).
    pub fn chunk_residue_records(&self) -> Vec<ChunkResidueRecord> {
        self.launch_residue.borrow().records.clone()
    }

    /// Machine-readable launch-granularity accounting for this graph.
    ///
    /// `total_*` fields are exact for the whole run regardless of row
    /// truncation. `records_dropped` is non-zero only when the run exceeded the
    /// row bound, and it is reported explicitly so a truncated ledger is never
    /// mistaken for a complete one.
    pub fn chunk_residue_ledger_json(&self) -> serde_json::Value {
        let ledger = self.launch_residue.borrow();
        serde_json::json!({
            "schema_kind": "prism_graph_chunk_launch_residue_ledger",
            "schema_version": 1,
            "steps_per_launch": self.steps_per_launch(),
            "chunk_calls": ledger.calls,
            "total_logical_steps_requested": ledger.total_logical_steps_requested,
            "total_device_steps_executed": ledger.total_device_steps_executed,
            "total_residue_steps": ledger.total_residue_steps,
            "total_step_delta": ledger.total_step_delta,
            "max_chunk_residue_steps": ledger.max_residue_steps,
            "first_residue_call_index": ledger.first_residue_call_index,
            "aborted_chunks": ledger.aborted_chunks,
            "launch_aligned": ledger.total_residue_steps == 0,
            "records_retained": ledger.records.len(),
            "records_dropped": ledger.records_dropped,
            "record_capacity": CHUNK_RESIDUE_RECORD_CAPACITY,
            "records": ledger.records,
        })
    }
}

/// Device allocations that must stay alive while a WHILE-resident CUDA graph
/// exists. The graph body reads/writes these addresses from inside the GPU.
pub struct ResidentLoopState {
    #[allow(dead_code)]
    pending: CudaSlice<u32>,
    #[allow(dead_code)]
    iter: CudaSlice<u32>,
    pub pending_dev_ptr: u64,
    pub iter_dev_ptr: u64,
    pub max_iterations: u32,
}

impl ResidentLoopState {
    pub fn new(
        pending: CudaSlice<u32>,
        iter: CudaSlice<u32>,
        pending_dev_ptr: u64,
        iter_dev_ptr: u64,
        max_iterations: u32,
    ) -> Self {
        Self {
            pending,
            iter,
            pending_dev_ptr,
            iter_dev_ptr,
            max_iterations: max_iterations.max(1),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ResidentLoopStatus {
    pub pending_dev_ptr: u64,
    pub iter_dev_ptr: u64,
    pub max_iterations: u32,
}

/// Capture the WHILE predicate bridge as a graph-template kernel node.
/// The predicate writes the CUDA conditional handle from device code.
pub unsafe fn launch_while_predicate_bridge(
    stream: sys::CUstream,
    pending_dev_ptr: u64,
    iter_dev_ptr: u64,
    handle: u64,
    max_iterations: u32,
) -> Result<()> {
    let rc = prism_while_drain_launch_predicate(
        stream,
        pending_dev_ptr as *const u32,
        iter_dev_ptr as *mut u32,
        handle,
        max_iterations,
    );
    if rc != 0 {
        bail!(
            "prism_while_drain_launch_predicate failed: cudaError={}",
            rc
        );
    }
    Ok(())
}

pub(crate) unsafe fn memset_node_u32(
    graph: sys::CUgraph,
    deps: &[sys::CUgraphNode],
    dev_ptr: u64,
    value: u32,
) -> Result<sys::CUgraphNode> {
    crate::graph_node::add_memset_node_v3(
        graph,
        deps,
        dev_ptr as *mut c_void,
        std::mem::size_of::<u32>(),
        value,
        4,
        1,
        1,
    )
    .map_err(|rc| anyhow::anyhow!("add_memset_node_v3(u32) failed: rc={}", rc))
}

/// Parameters needed to capture the physics step as a graph.
/// All device pointers must remain valid for the lifetime of the graph.
pub struct GraphCaptureParams<'a> {
    // Stream to capture on
    pub stream: &'a Arc<CudaStream>,
    pub context: &'a Arc<CudaContext>,

    // Director
    pub director_graph_fn: &'a CudaFunction,
    pub d_protocol_state: &'a mut CudaSlice<u8>,

    // Physics kernel
    pub fused_step_kernel: &'a CudaFunction,
    pub n_atoms: usize,
    pub physics_cfg: LaunchConfig,

    // Multi-LIF kernel
    pub multi_lif_kernel: &'a CudaFunction,
    pub voxel_cfg: LaunchConfig,

    // Housekeeping kernels
    pub ca_restraint_fn: &'a CudaFunction,
    pub heartbeat_fn: &'a CudaFunction,
    pub coupling_clear_fn: &'a CudaFunction,
    pub com_reduce_fn: &'a CudaFunction,
    pub com_correct_fn: &'a CudaFunction,
    pub com_accum_clear_fn: &'a CudaFunction,
    // All the device buffers (passed as kernel args during capture)
    // These are borrowed references — the graph captures the device addresses
}

/// Check if the CUDA driver supports conditional graph nodes (CUDA 12.4+).
/// Returns true if `cuGraphConditionalHandleCreate` is available and functional.
pub fn supports_conditional_graphs(context: &Arc<CudaContext>) -> bool {
    // Check compute capability — conditional nodes require CC 9.0+ (Hopper/Blackwell)
    match context.compute_capability() {
        Ok((major, _minor)) => major >= 9,
        Err(_) => false,
    }
}

/// Read the status_code field from ProtocolState via async memcpy.
/// Returns 0 (OK), 1 (NaN), or 2 (diverged).
/// The memcpy is async — the caller should synchronize if they need the result immediately.
pub fn poll_heartbeat_async(
    stream: &Arc<CudaStream>,
    d_protocol_state: &CudaSlice<u8>,
) -> Result<i32> {
    let mut full_state = vec![0u8; std::mem::size_of::<crate::protocol_state::ProtocolState>()];
    stream.memcpy_dtoh(d_protocol_state, &mut full_state)?;
    // Read status_code at its actual byte offset (use memoffset logic)
    let dummy = unsafe { std::mem::zeroed::<crate::protocol_state::ProtocolState>() };
    let base = &dummy as *const _ as usize;
    let field = &dummy.status_code as *const _ as usize;
    let offset = field - base;
    let status = i32::from_ne_bytes([
        full_state[offset],
        full_state[offset + 1],
        full_state[offset + 2],
        full_state[offset + 3],
    ]);
    Ok(status)
}

/// The main autonomous run loop: captures a graph, replays it in chunks,
/// breaks for NL rebuild, polls heartbeat between chunks.
///
/// # Arguments
/// * `total_steps` - Total number of MD steps to run
/// * `chunk_size` - Steps per graph replay chunk (default 500)
/// * `capture_fn` - Closure that launches one step's kernels (called during capture)
/// * `nl_rebuild_fn` - Closure called between chunks for neighbor list rebuild
///
/// # Returns
/// Number of steps completed (may be less than total_steps if heartbeat triggers abort)
pub fn run_autonomous_loop(
    stream: &Arc<CudaStream>,
    d_protocol_state: &CudaSlice<u8>,
    total_steps: u32,
    chunk_size: u32,
    mut capture_fn: impl FnMut() -> Result<()>,
    mut nl_rebuild_fn: impl FnMut(u32) -> Result<()>,
) -> Result<u32> {
    // ── Step 1: Capture one step as a graph ──
    log::info!("Capturing physics step as CUDA Graph...");
    stream
        .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        .map_err(|e| anyhow::anyhow!("Stream capture begin failed: {:?}", e))?;

    // Launch one step's worth of kernels — they get captured, not executed
    capture_fn()?;

    let graph = stream
        .end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
        .map_err(|e| anyhow::anyhow!("Stream capture end failed: {:?}", e))?
        .ok_or_else(|| anyhow::anyhow!("Stream capture produced null graph"))?;

    log::info!("CUDA Graph captured and instantiated");

    let autonomous = AutonomousGraph {
        graph: Some(graph),
        raw: None,
        physical_launch_domains: Vec::new(),
        steps_per_launch: 1,
        resident_loop_state: None,
        ncmc_resident_capture_identity: None,
        launch_residue: RefCell::new(ChunkResidueLedger::default()),
    };

    // ── Step 2: Replay loop ──
    let mut steps_completed = 0u32;
    let n_chunks = (total_steps + chunk_size - 1) / chunk_size;

    for chunk_idx in 0..n_chunks {
        let steps_this_chunk = chunk_size.min(total_steps - steps_completed);

        // Replay the graph steps_this_chunk times
        autonomous.run_chunk(steps_this_chunk)?;
        steps_completed += steps_this_chunk;

        // Synchronize to read heartbeat and do NL rebuild
        stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("Stream sync failed after chunk: {:?}", e))?;

        // Poll heartbeat
        let status = poll_heartbeat_async(stream, d_protocol_state)?;
        if status != 0 {
            let reason = match status {
                1 => "NaN detected in coordinates",
                2 => "System diverged (coordinates > 1000A)",
                _ => "Unknown error",
            };
            log::error!(
                "HEARTBEAT ABORT at step {}: {} (status={})",
                steps_completed,
                reason,
                status
            );
            return Ok(steps_completed);
        }

        // CPU-side neighbor list rebuild
        nl_rebuild_fn(steps_completed)?;

        if chunk_idx % 10 == 0 {
            log::info!(
                "  Graph chunk {}/{}: {} steps completed",
                chunk_idx + 1,
                n_chunks,
                steps_completed
            );
        }
    }

    log::info!("Autonomous graph loop complete: {} steps", steps_completed);
    Ok(steps_completed)
}

// ═══════════════════════════════════════════════════════════════════════════════
// TWIN Dual-Engine Autonomous Loop
// ═══════════════════════════════════════════════════════════════════════════════
//
// The PRISM-TWIN interferometric observation platform:
//
//   [Director_A] → [Physics_A] → [multi_lif_A] → [compact_push A→ring_a]
//                    ↓ ring_a spikes
//   [read_adapt ring_a → thresholds_B]    ← interferometric coupling A→B
//   [Director_B] → [Physics_B] → [multi_lif_B] → [compact_push B→ring_b]
//                    ↓ ring_b spikes
//   [read_adapt ring_b → thresholds_A]    ← interferometric coupling B→A
//   [recovery_A] [recovery_B]             ← threshold decay
//   [heartbeat_A] [heartbeat_B]           ← NaN/divergence safety
//   [CA_restraints_A] [CA_restraints_B]   ← position restraints
//   [coupling_clear_A] [coupling_clear_B] ← double-buffer housekeeping
//
// All in one graph launch. Both groups communicate through GPU-resident ring
// buffers — spikes from A modulate B's detection thresholds, and vice versa.
// The CPU never touches spike data.

/// Run the full TWIN dual-engine simulation autonomously.
///
/// Each graph launch executes one coupled step for both engines.
/// Host breaks every `chunk_size` steps for neighbor list rebuild only.
///
/// # Arguments
/// * `stream` - CUDA stream for graph capture and execution
/// * `total_steps` - Total MD steps to run
/// * `chunk_size` - Steps per graph replay chunk (default 500)
/// * `step_twin_fn` - Closure that launches one coupled TWIN step (both engines + coupling)
/// * `nl_rebuild_fn` - Closure for CPU-side neighbor list rebuild (both engines)
/// * `d_protocol_a` / `d_protocol_b` - ProtocolState for each engine (for heartbeat polling)
pub fn run_twin_autonomous_loop(
    stream: &Arc<CudaStream>,
    d_protocol_a: &CudaSlice<u8>,
    d_protocol_b: &CudaSlice<u8>,
    total_steps: u32,
    chunk_size: u32,
    mut step_twin_fn: impl FnMut() -> Result<()>,
    mut nl_rebuild_fn: impl FnMut(u32) -> Result<()>,
) -> Result<u32> {
    // ── Capture one coupled TWIN step as a graph ──
    log::info!("Capturing TWIN dual-engine step as CUDA Graph...");
    stream
        .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        .map_err(|e| anyhow::anyhow!("TWIN graph capture begin failed: {:?}", e))?;

    step_twin_fn()?;

    let graph = stream
        .end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
        .map_err(|e| anyhow::anyhow!("TWIN graph capture end failed: {:?}", e))?
        .ok_or_else(|| anyhow::anyhow!("TWIN graph capture produced null graph"))?;

    log::info!("TWIN CUDA Graph captured and instantiated (dual-engine interferometric coupling)");

    let autonomous = AutonomousGraph {
        graph: Some(graph),
        raw: None,
        physical_launch_domains: Vec::new(),
        steps_per_launch: 1,
        resident_loop_state: None,
        ncmc_resident_capture_identity: None,
        launch_residue: RefCell::new(ChunkResidueLedger::default()),
    };

    // ── Replay loop ──
    let mut steps_completed = 0u32;
    let n_chunks = (total_steps + chunk_size - 1) / chunk_size;

    for chunk_idx in 0..n_chunks {
        let steps_this_chunk = chunk_size.min(total_steps - steps_completed);

        // Replay the dual-engine graph
        autonomous.run_chunk(steps_this_chunk)?;
        steps_completed += steps_this_chunk;

        // Synchronize and check both engines' heartbeats
        stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("TWIN stream sync failed: {:?}", e))?;

        for (label, d_ps) in [("Group A", d_protocol_a), ("Group B", d_protocol_b)] {
            let status = poll_heartbeat_async(stream, d_ps)?;
            if status != 0 {
                let reason = match status {
                    1 => "NaN detected",
                    2 => "System diverged",
                    _ => "Unknown",
                };
                log::error!(
                    "TWIN HEARTBEAT ABORT ({}) at step {}: {}",
                    label,
                    steps_completed,
                    reason
                );
                return Ok(steps_completed);
            }
        }

        // CPU-side neighbor list rebuild for both engines
        nl_rebuild_fn(steps_completed)?;

        if chunk_idx % 10 == 0 {
            log::info!(
                "  TWIN chunk {}/{}: {} steps (both engines coupled)",
                chunk_idx + 1,
                n_chunks,
                steps_completed
            );
        }
    }

    log::info!(
        "TWIN autonomous loop complete: {} coupled steps",
        steps_completed
    );
    Ok(steps_completed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(byte: u8) -> String {
        format!("{byte:02x}").repeat(32)
    }

    #[test]
    fn ncmc_capture_identity_is_sealed_and_fail_closed() {
        let identity = NcmcResidentGraphCaptureIdentity::seal(
            sha(1),
            sha(2),
            sha(3),
            sha(4),
            sha(5),
            sha(6),
            0.004,
        )
        .expect("seal reversible capture identity");
        identity.validate().expect("identity must revalidate");
        assert_eq!(identity.schema_version, 3);
        assert_eq!(
            identity.capture_body_kind,
            "NCMC_GAMMA_ZERO_REVERSIBLE_WHILE_V2"
        );
        assert_eq!(
            identity.integrator_splitting,
            "FORCE_KICK_RATTLE_DRIFT_SHAKE_FORCE_KICK_RATTLE_V2"
        );

        let mut rng_tamper = identity.clone();
        rng_tamper.rng_state_read_write_enabled = true;
        assert!(rng_tamper.validate().is_err());

        let mut ptx_tamper = identity.clone();
        ptx_tamper.selected_fused_ptx_sha256 = sha(9);
        assert!(ptx_tamper.validate().is_err());

        let mut gamma_tamper = identity.clone();
        gamma_tamper.friction_gamma_per_ps = f64::MIN_POSITIVE;
        assert!(gamma_tamper.validate().is_err());
    }

    #[test]
    fn test_poll_heartbeat_offset() {
        // Verify poll_heartbeat_async reads status_code correctly
        let ps = crate::protocol_state::ProtocolState {
            status_code: 42,
            ..unsafe { std::mem::zeroed() }
        };
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(&ps as *const _ as *const u8, std::mem::size_of_val(&ps))
        };
        // Compute actual field offset
        let base = &ps as *const _ as usize;
        let field = &ps.status_code as *const _ as usize;
        let offset = field - base;
        let val = i32::from_ne_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);
        assert_eq!(val, 42, "status_code not at expected offset {}", offset);
        assert_eq!(
            std::mem::size_of::<crate::protocol_state::ProtocolState>(),
            716
        );
    }
}

/// Test-only inspection of actual graph-owned neighbor kernel arguments.
/// Queries CUDA host metadata only: no kernel, copy, event or fence is issued.
#[cfg(test)]
pub(crate) fn neighbor_arguments_for_test(graph: sys::CUgraph) -> Result<serde_json::Value> {
    extern "C" {
        fn prism_graph_conditional_children_diagnostic_ffi(node: sys::CUgraphNode,
            children: *mut sys::CUgraph, capacity: usize, count: *mut usize) -> i32;
    }
    fn checked(rc: sys::CUresult, what: &str) -> Result<()> {
        anyhow::ensure!(rc == sys::CUresult::CUDA_SUCCESS, "{what}: {rc:?}"); Ok(())
    }
    fn visit(graph: sys::CUgraph, path: String, records: &mut Vec<serde_json::Value>,
        ancestry: &mut Vec<usize>) -> Result<()> {
        anyhow::ensure!(!graph.is_null() && !ancestry.contains(&(graph as usize)),
            "invalid/cyclic actual graph hierarchy at {path}");
        ancestry.push(graph as usize);
        let mut count=0usize;
        checked(unsafe{sys::cuGraphGetNodes(graph,std::ptr::null_mut(),&mut count)},"graph node count")?;
        // Empty conditional branches are valid graph children. Do not pass
        // Vec's non-null dangling pointer as a zero-capacity output array.
        if count == 0 {
            eprintln!("NEIGHBOR_CAPTURE_EMPTY_GRAPH path={path} graph={:#x}", graph as usize);
            ancestry.pop();
            return Ok(());
        }
        let mut nodes=vec![std::ptr::null_mut();count];
        checked(unsafe{sys::cuGraphGetNodes(graph,nodes.as_mut_ptr(),&mut count)},
            &format!("graph nodes path={path} count={count} graph={:#x}", graph as usize))?;
        nodes.truncate(count);
        for (index,node) in nodes.into_iter().enumerate() {
            let path=format!("{path}/{index}");
            let mut kind=unsafe{std::mem::zeroed()};
            checked(unsafe{sys::cuGraphNodeGetType(node,&mut kind)},"graph node type")?;
            if kind==sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_GRAPH {
                let mut child=std::ptr::null_mut();
                checked(unsafe{sys::cuGraphChildGraphNodeGetGraph(node,&mut child)},"actual child graph")?;
                visit(child,path,records,ancestry)?;
            } else if kind==sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_CONDITIONAL {
                let mut count=0usize;
                let rc=unsafe{prism_graph_conditional_children_diagnostic_ffi(node,std::ptr::null_mut(),0,&mut count)};
                anyhow::ensure!(rc==0,"actual conditional body count: {rc}");
                let mut children=vec![std::ptr::null_mut();count];
                let rc=unsafe{prism_graph_conditional_children_diagnostic_ffi(node,children.as_mut_ptr(),children.len(),&mut count)};
                anyhow::ensure!(rc==0 && count==children.len(),"actual conditional bodies: {rc}");
                for (branch,child) in children.into_iter().enumerate() {
                    visit(child,format!("{path}/conditional{branch}"),records,ancestry)?;
                }
            } else if kind==sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
                let mut params:sys::CUDA_KERNEL_NODE_PARAMS=unsafe{std::mem::zeroed()};
                checked(unsafe{sys::cuGraphKernelNodeGetParams_v2(node,&mut params)},"actual kernel parameters")?;
                let mut name=std::ptr::null();
                checked(unsafe{sys::cuFuncGetName(&mut name,params.func)},"actual kernel name")?;
                anyhow::ensure!(!name.is_null(),"actual kernel name is null");
                if unsafe{std::ffi::CStr::from_ptr(name)}.to_bytes()!=b"nhs_build_neighbor_list_warp" {continue;}
                anyhow::ensure!(!params.kernelParams.is_null() && params.extra.is_null(),
                    "neighbor kernel uses unsupported packed parameters");
                let mut arguments=Vec::with_capacity(13);
                for argument in 0..13usize {
                    let pointer=unsafe{*params.kernelParams.add(argument)};
                    anyhow::ensure!(!pointer.is_null(),"neighbor argument {argument} is null");
                    arguments.push(if argument<8 {unsafe{std::ptr::read_unaligned(pointer.cast::<u64>())}}
                        else {u64::from(unsafe{std::ptr::read_unaligned(pointer.cast::<u32>())})});
                }
                records.push(serde_json::json!({"path":path,"node":node as usize,"graph":graph as usize,
                    "kernel":"nhs_build_neighbor_list_warp","arguments_u64_or_u32_bits":arguments,
                    "grid":[params.gridDimX,params.gridDimY,params.gridDimZ],
                    "block":[params.blockDimX,params.blockDimY,params.blockDimZ],"shared_bytes":params.sharedMemBytes}));
            }
        }
        ancestry.pop();Ok(())
    }
    let mut records=Vec::new();visit(graph,"root".into(),&mut records,&mut Vec::new())?;
    anyhow::ensure!(!records.is_empty(),"actual graph contains no neighbor kernel");
    Ok(serde_json::json!({"schema_kind":"prismos_actual_neighbor_capture_arguments_v1",
        "source":"CUDA_graph_owned_kernel_parameters","device_copies":0,"stream_fences":0,
        "kernels":records}))
}

#[cfg(test)]
pub(crate) fn verify_neighbor_owner_for_test(actual:&serde_json::Value, owner:&serde_json::Value)
    -> Result<usize> {
    let expected=owner["arguments_u64_or_u32_bits"].as_array().context("neighbor owner arguments absent")?;
    anyhow::ensure!(expected.len()==13,"neighbor owner requires exactly13 arguments");
    let kernels=actual["kernels"].as_array().context("actual neighbor kernels absent")?;
    let mut matched=0;
    for kernel in kernels {
        let values=kernel["arguments_u64_or_u32_bits"].as_array().context("actual arguments absent")?;
        if values.first()==expected.first() {
            anyhow::ensure!(values==expected,"captured neighbor arguments differ from physical owner: actual={} owner={}",kernel,owner);
            anyhow::ensure!(kernel["block"]==serde_json::json!([256,1,1]),"neighbor launch block differs");
            matched+=1;
        }
    }
    let expected_count=owner["native_neighbor_enqueues_in_capture"].as_u64().context("native enqueue multiplicity absent")? as usize;
    anyhow::ensure!(expected_count>0 && matched==expected_count,"physical owner graph multiplicity differs: found {matched}, native enqueues {expected_count}; owner={owner}");
    Ok(matched)
}


#[derive(Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphContinuationV1 {
    schema_version:u32,steps_per_launch:u32,ledger:Vec<u8>,
    resident:Option<(u32,Vec<u8>,Vec<u8>)>,
    ncmc_identity:Option<NcmcResidentGraphCaptureIdentity>,
}
impl AutonomousGraph {
    /// Exact live payload extents, without allocating any checkpoint payload.
    pub fn continuation_allocation_plan(&self)->anyhow::Result<Vec<(String,u64)>> {
        let mut rows=vec![("launch_ledger_wire".into(),crate::checkpoint_codec::encoded_size(&*self.launch_residue.borrow(),u64::MAX)?),
            ("ncmc_identity_wire".into(),crate::checkpoint_codec::encoded_size(&self.ncmc_resident_capture_identity,u64::MAX)?)];
        if let Some(state)=&self.resident_loop_state {
            rows.push(("resident_pending".into(),u64::try_from(state.pending.len())?.checked_mul(4).ok_or_else(||anyhow::anyhow!("resident extent overflow"))?));
            rows.push(("resident_iteration".into(),u64::try_from(state.iter.len())?.checked_mul(4).ok_or_else(||anyhow::anyhow!("resident extent overflow"))?));
        }Ok(rows)
    }
    pub fn capture_continuation(&self,stream:&Arc<CudaStream>)->anyhow::Result<GraphContinuationV1> {
        use crate::scientific_checkpoint::device::read_raw;
        stream.synchronize()?;
        let resident=if let Some(state)=&self.resident_loop_state {
            Some((state.max_iterations,unsafe{read_raw(stream,state.pending_dev_ptr,state.pending.len()*4)?},
                unsafe{read_raw(stream,state.iter_dev_ptr,state.iter.len()*4)?}))
        } else {None};
        Ok(GraphContinuationV1{schema_version:1,steps_per_launch:self.steps_per_launch,
            ledger:crate::checkpoint_codec::to_bytes(&*self.launch_residue.borrow())?,resident,
            ncmc_identity:self.ncmc_resident_capture_identity.clone()})
    }
    pub fn restore_continuation(&self,stream:&Arc<CudaStream>,saved:&GraphContinuationV1)->anyhow::Result<()> {
        use crate::scientific_checkpoint::device::write_raw;
        anyhow::ensure!(saved.schema_version==1 && saved.steps_per_launch==self.steps_per_launch,
            "captured graph body shape changed on restart");
        anyhow::ensure!(serde_json::to_value(&saved.ncmc_identity)?==serde_json::to_value(&self.ncmc_resident_capture_identity)?,
            "captured graph NCMC identity changed on restart");
        let ledger:ChunkResidueLedger=crate::checkpoint_codec::from_bytes(&saved.ledger)?;
        match (&saved.resident,&self.resident_loop_state) {
            (None,None)=>{},
            (Some((max,pending,iter)),Some(state))=>{
                anyhow::ensure!(*max==state.max_iterations && pending.len()==state.pending.len()*4
                    && iter.len()==state.iter.len()*4,"resident loop allocation shape changed on restart");
                // Fixed-cadence native WHILE leaves pending=1 and exits when
                // its exact iteration bound is reached. Cached unused tails
                // retain their constructor state pending=1,iter=0. The owner
                // rendezvous separately proves no physical launch is in flight.
                anyhow::ensure!(pending.len()==4 && iter.len()==4,"resident predicate cells changed");
                let pending_word=u32::from_le_bytes(pending.as_slice().try_into()?);
                let iteration_word=u32::from_le_bytes(iter.as_slice().try_into()?);
                anyhow::ensure!(pending_word<=1 && iteration_word<=*max
                    && (pending_word==0 || iteration_word==0 || iteration_word==*max),
                    "resident graph contains a partial fixed-cadence iteration");
                unsafe {write_raw(stream,state.pending_dev_ptr,pending)?;write_raw(stream,state.iter_dev_ptr,iter)?;}
            },
            _=>anyhow::bail!("resident graph owner presence changed"),
        }
        *self.launch_residue.borrow_mut()=ledger;Ok(())
    }
}
