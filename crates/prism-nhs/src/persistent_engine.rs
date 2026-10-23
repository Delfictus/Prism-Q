//! Persistent NHS Engine for High-Throughput Batch Processing
//!
//! Keeps CUDA context, modules, and buffers alive across multiple structures.
//! Hot-swaps topologies without reinitializing GPU state.
//!
//! ## Performance Benefits
//! - Single CUDA context creation (~100ms saved per structure)
//! - Single PTX compilation (~200ms saved per structure)
//! - Buffer reuse for similar-sized structures
//! - Pipelined data transfer during compute
//!
//! ## Usage
//! ```ignore
//! let mut engine = PersistentNhsEngine::new(max_atoms)?;
//! for topology in topologies {
//!     engine.load_topology(&topology)?;
//!     let results = engine.run(steps, config)?;
//! }
//! ```

use anyhow::{bail, Context, Result};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Mutex;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

#[cfg(feature = "gpu")]
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtrMut, LaunchConfig,
    PushKernelArg,
};
#[cfg(feature = "gpu")]
use cudarc::nvrtc::Ptx;

use crate::input::PrismPrepTopology;

#[cfg(feature = "gpu")]
fn tier8_capture_driver_error_text(rc: cudarc::driver::sys::CUresult) -> (String, String) {
    let err = cudarc::driver::DriverError(rc);
    let name = err
        .error_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|e| format!("cuGetErrorName failed: {:?}", e));
    let text = err
        .error_string()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|e| format!("cuGetErrorString failed: {:?}", e));
    (name, text)
}

#[cfg(feature = "gpu")]
static TIER8_CUDA_ERROR_DRAIN_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(feature = "gpu")]
static RESIDENT_GRAPH_REPLAY_BINDING_DOMAIN_COUNTER: AtomicU64 = AtomicU64::new(1);

#[cfg(feature = "gpu")]
fn tier8_capture_stream_slot(raw_stream: usize) -> u32 {
    static NEXT_SLOT: AtomicU32 = AtomicU32::new(0);
    static SLOT_MAP: OnceLock<Mutex<HashMap<usize, u32>>> = OnceLock::new();
    let map = SLOT_MAP.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = map.lock().expect("tier8 stream slot map poisoned");
    *map.entry(raw_stream)
        .or_insert_with(|| NEXT_SLOT.fetch_add(1, Ordering::Relaxed))
}

#[cfg(feature = "gpu")]
fn tier8_capture_verbose_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("PRISM_TIER8_VERBOSE_DIAG")
            .ok()
            .map(|v| {
                let v = v.trim();
                v == "1"
                    || v.eq_ignore_ascii_case("true")
                    || v.eq_ignore_ascii_case("yes")
                    || v.eq_ignore_ascii_case("on")
            })
            .unwrap_or(false)
    })
}

#[cfg(feature = "gpu")]
macro_rules! tier8_capture_verbose {
    ($($arg:tt)*) => {
        if tier8_capture_verbose_enabled() {
            log::info!($($arg)*);
        }
    };
}

#[allow(deprecated)]
use crate::fused_engine::{
    CryoUvProtocol,
    EnsembleSnapshot,
    NhsAmberFusedEngine,
    PrismCodeMdPolicyAdmissionV1,
    RunSummary,
    SpikeEvent,
    StepResult,
    // Deprecated - kept for backward compatibility
    TemperatureProtocol,
    UvProbeConfig,
};
use crate::input::NonbondedExecutionPolicy;

#[cfg(feature = "gpu")]
use crate::fused_engine::ResidentGraphBindingProfile;

/// The only bodies admitted through the raw resident-template capture path.
/// Housekeeping bodies are deliberately separate from the autonomous MD body:
/// CA must precede multi-LIF, while clear→heartbeat must follow coupling build.
#[cfg(feature = "gpu")]
#[derive(Clone, Copy, Debug)]
enum RawResidentTemplateBody {
    AutonomousMd,
    NcmcReversible,
    CaRestraintChild,
    CouplingClearHeartbeatChild,
}

#[cfg(feature = "gpu")]
impl RawResidentTemplateBody {
    fn capture_label(self) -> &'static str {
        match self {
            Self::AutonomousMd => "capture_autonomous_template",
            Self::NcmcReversible => "capture_ncmc_reversible_template",
            Self::CaRestraintChild => "capture_resident_ca_restraint_child_template",
            Self::CouplingClearHeartbeatChild => {
                "capture_resident_coupling_clear_heartbeat_child_template"
            }
        }
    }

    fn kernel_label(self) -> &'static str {
        match self {
            Self::AutonomousMd => "step_autonomous_kernels_tagged",
            Self::NcmcReversible => "step_lambda_ncmc_reversible_kernels_tagged",
            Self::CaRestraintChild => "enqueue_resident_ca_restraint_child",
            Self::CouplingClearHeartbeatChild => {
                "enqueue_resident_coupling_clear_then_heartbeat_child"
            }
        }
    }
}

/// Private receipt capability for one successfully instantiated autonomous
/// resident graph. It never leaves [`BoundResidentGraph`], so it cannot outlive
/// or be paired with a different graph object.
#[cfg(feature = "gpu")]
#[derive(Debug)]
struct ResidentGraphReplayBinding {
    domain: u64,
    capture_nonce: u64,
    resident_structure_fingerprint_sha256: String,
    graph_template_handle: usize,
    graph_exec_handle: usize,
}

/// Opaque owner for one successfully instantiated autonomous resident graph
/// and its private receipt binding. The graph is boxed before binding so its
/// Rust-object address, raw graph handle, raw executable handle, and structure
/// witness remain one live identity through every bound replay. No raw graph
/// handle is exposed through this type.
#[cfg(feature = "gpu")]
pub struct BoundResidentGraph {
    graph: Box<crate::graph_capture::AutonomousGraph>,
    binding: ResidentGraphReplayBinding,
}

#[cfg(feature = "gpu")]
impl BoundResidentGraph {
    pub fn steps_per_launch(&self) -> u32 {
        self.graph.steps_per_launch()
    }

    pub fn resident_loop_status(&self) -> Option<crate::graph_capture::ResidentLoopStatus> {
        self.graph.resident_loop_status()
    }
}

/// One physical observation owner: WHILE -> LADD -> KCC -> V2 -> F1/G26.
/// LADD pointer parities select bodies of one root and cannot split V2 history.
#[cfg(feature = "v2_ignition")]
pub struct BoundResidentObservationGraph {
    bound: BoundResidentGraph,
    pipeline: Arc<crate::captured_pipeline::CapturedAdjudicationPipeline>,
    parity_device: crate::governor_device::CudaOwned<cudarc::driver::CudaSlice<u32>>,
    parity: usize,
    ladd_keys: [Vec<u64>;2],
}
#[cfg(feature = "v2_ignition")]
impl BoundResidentObservationGraph {
    pub fn pipeline(&self) -> &Arc<crate::captured_pipeline::CapturedAdjudicationPipeline> { &self.pipeline }
    pub fn steps_per_launch(&self) -> u32 { self.bound.steps_per_launch() }
    #[cfg(test)]
    pub(crate) fn neighbor_arguments_for_test(&self) -> anyhow::Result<serde_json::Value> {
        crate::graph_capture::neighbor_arguments_for_test(self.bound.graph.cu_graph_template()
            .context("local physical graph has no actual template")?)
    }

    /// Test-only host snapshot of keys already sealed during native capture.
    /// No DevicePtr acquisition, CUDA transfer, event or synchronization occurs.
    #[cfg(test)]
    pub(crate) fn captured_ladd_owner_keys_for_test(&self) -> serde_json::Value {
        serde_json::json!({"schema_kind":"prismos_captured_ladd_host_owner_keys_v1",
            "host_only":true,"cuda_calls":0,"parity_keys":self.ladd_keys,
            "pointer_indices":{"voxel_density":12,"density_reference":13,"density_m2":14,"reference_count":15},
            "scalar_indices":{"grid_dim":26,"n_atoms":27,"cold_hold_steps":29},
            "gaussian_reference_pointer_indices":{"mean":31,"m2":32,"count":33}})
    }
}
#[cfg(feature = "v2_ignition")]
extern "C" {
    fn prism_ladd_parent_launch_selector(stream: cudarc::driver::sys::CUstream,
        step: *const u32, parity: *mut u32, cold_end: i32, handle: u64) -> i32;
}

/// Prepared native physical/observer cells for one coordinator-owned root.
/// External physical, spike, CSR and output owners stay alive under the worker
/// registration lease. The same live V2 Arc carries pre-activation history.
#[cfg(feature="v2_ignition")]
pub struct PreparedResidentObservation {
    stream: Arc<CudaStream>,
    cadence: u32,
    profile: ResidentGraphBindingProfile,
    config: crate::captured_pipeline::PipelineConfig,
    physics: crate::graph_capture::CapturedTemplate,
    snapshot: crate::graph_capture::CapturedTemplate,
    ladd: Vec<crate::graph_capture::CapturedTemplate>,
    kcc: crate::graph_capture::CapturedTemplate,
    parity_device: crate::governor_device::CudaOwned<CudaSlice<u32>>,
    pending_device: crate::governor_device::CudaOwned<CudaSlice<u32>>,
    iterations_device: crate::governor_device::CudaOwned<CudaSlice<u32>>,
    step_ptr: u64,
    cold_end: i32,
    parity: usize,
    keys: [Vec<u64>;2],
    pipeline: Arc<crate::captured_pipeline::CapturedAdjudicationPipeline>,
    appended: bool,
    parent_template_identity: usize,
    binding: Option<ResidentGraphReplayBinding>,
}

#[cfg(feature="v2_ignition")]
pub struct SharedResidentObservationAdmission {
    dispatch: crate::fused_engine::LaddCadenceDispatch,
    start: u32,
    capture_nonce: u64,
    parity: usize,
}

#[cfg(feature="v2_ignition")]
impl PreparedResidentObservation {
    pub fn cadence(&self)->u32 {self.cadence}
    pub fn source_stream_id(&self)->u32 {self.config.source_stream_id}
    pub fn raw_stream(&self)->cudarc::driver::sys::CUstream {self.stream.cu_stream()}
    pub fn pipeline(&self)->anyhow::Result<&Arc<crate::captured_pipeline::CapturedAdjudicationPipeline>> {Ok(&self.pipeline)}
    pub fn append_into(&mut self,parent:&mut crate::graph_capture::CapturedTemplate,
        dependencies:&[cudarc::driver::sys::CUgraphNode])
        -> anyhow::Result<cudarc::driver::sys::CUgraphNode> {
        use cudarc::driver::DevicePtr;
        use crate::graph_capture::capture_pure_template;
        anyhow::ensure!(!self.appended,"prepared physical observation already has an owning parent");
        self.stream.context().bind_to_thread()?;self.stream.synchronize()?;
        // Mark before mutation: partial assembly cannot be retried under a
        // second parent or promoted after an error.
        self.appended=true;
        let graph=parent.cu_graph();
        self.parent_template_identity=graph as usize;
        let pending=self.pending_device.device_ptr(&self.stream).0;
        let iter=self.iterations_device.device_ptr(&self.stream).0;
        let parity=self.parity_device.device_ptr(&self.stream).0;
        parent.retain_child_owners(&self.snapshot);parent.retain_child_owners(&self.physics);
        let snapshot=parent.add_child_graph_node(dependencies,self.snapshot.cu_graph())?;
        let reset_pending=unsafe {crate::graph_capture::memset_node_u32(graph,&[snapshot],pending,1)?};
        let reset_iter=unsafe {crate::graph_capture::memset_node_u32(graph,&[snapshot],iter,0)?};
        let handle=unsafe {crate::graph_node::create_while_handle(graph,1)}
            .map_err(|rc|anyhow::anyhow!("shared physical WHILE handle: {rc}"))?;
        let (while_node,body)=unsafe {crate::graph_node::add_while_node_v3(graph,&[reset_pending,reset_iter],handle)}
            .map_err(|rc|anyhow::anyhow!("shared physical WHILE node: {rc}"))?;
        let physics=unsafe {crate::graph_node::add_child_graph_node_v3(body,&[],self.physics.cu_graph())}
            .map_err(|rc|anyhow::anyhow!("shared physical body: {rc}"))?;
        let predicate=capture_pure_template(self.stream.clone(),|_tagger| unsafe {
            crate::graph_capture::launch_while_predicate_bridge(self.stream.cu_stream(),pending,iter,handle,self.cadence)
        })?;
        parent.retain_child_owners(&predicate);
        unsafe {crate::graph_node::add_child_graph_node_v3(body,&[physics],predicate.cu_graph())}
            .map_err(|rc|anyhow::anyhow!("shared physical predicate: {rc}"))?;
        let lh=unsafe {crate::graph_node::create_while_handle(graph,0)}
            .map_err(|rc|anyhow::anyhow!("shared LADD handle: {rc}"))?;
        let selector=capture_pure_template(self.stream.clone(),|_tagger| {
            let rc=unsafe {prism_ladd_parent_launch_selector(self.stream.cu_stream(),self.step_ptr as *const u32,
                parity as *mut u32,self.cold_end,lh)};
            anyhow::ensure!(rc==0,"shared LADD selector: {rc}");Ok(())
        })?;
        parent.retain_child_owners(&selector);
        let selector_node=parent.add_child_graph_node(&[while_node],selector.cu_graph())?;
        let (ladd_node,bodies)=unsafe {crate::graph_node::add_switch4_node(graph,selector_node,lh)}
            .map_err(|rc|anyhow::anyhow!("shared LADD SWITCH: {rc}"))?;
        for (body,template) in bodies.into_iter().zip(self.ladd.iter()) {
            parent.retain_child_owners(template);
            unsafe {crate::graph_node::add_child_graph_node_v3(body,&[],template.cu_graph())}
                .map_err(|rc|anyhow::anyhow!("shared LADD native body: {rc}"))?;
        }
        self.pipeline.append_existing_kcc_observation_nodes(&self.config,&self.kcc,parent,&[ladd_node])
    }
}

/// One exact logical window to execute through an already-bound resident
/// graph. This describes a planned GPU operation only; it cannot mint a
/// receipt until the persistent engine launches and fences it.
#[cfg(feature = "gpu")]
pub struct ResidentGraphReplaySegment<'graph> {
    bound_graph: &'graph BoundResidentGraph,
    logical_steps: u32,
}

#[cfg(feature = "gpu")]
impl<'graph> ResidentGraphReplaySegment<'graph> {
    pub fn exact(bound_graph: &'graph BoundResidentGraph, logical_steps: u32) -> Self {
        Self {
            bound_graph,
            logical_steps,
        }
    }
}


// AUTONOMOUS_CHUNK_CLOCK_BEGIN: pure acceptance state, tested independently.
// This is an ephemeral readback cache, never restart state or host chronology.
#[derive(Default)]
struct AutonomousChunkClock {
    endpoint: Option<u32>,
}
impl AutonomousChunkClock {
    fn observe(&mut self, actual_device_step: u32) { self.endpoint = Some(actual_device_step); }
    fn invalidate(&mut self) { self.endpoint = None; }
    fn endpoint(&self) -> Result<u32, &'static str> {
        self.endpoint.ok_or("autonomous chunk endpoint is unobserved or invalidated")
    }
    fn begin(&mut self, steps: u32, graph_steps: u32) -> Result<(u32,u32), &'static str> {
        if steps == 0 || steps != graph_steps {
            return Err("autonomous chunk requires one exact nonempty WHILE launch");
        }
        let start = self.endpoint.take().ok_or("autonomous chunk clock needs device admission")?;
        let end = start.checked_add(steps).ok_or("autonomous chunk device-clock overflow")?;
        Ok((start,end))
    }
    fn complete(&mut self, expected: u32, observed: u32) -> Result<(), &'static str> {
        if observed != expected {
            self.invalidate();
            return Err("autonomous chunk device-clock endpoint differs from exact WHILE window");
        }
        self.observe(observed);
        Ok(())
    }
}
// AUTONOMOUS_CHUNK_CLOCK_END

#[cfg(test)]
mod autonomous_chunk_clock_tests {
    use super::AutonomousChunkClock;
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/autonomous_chunk_clock_cases.rs"));
}

/// Borrowed proof of the last synchronized autonomous endpoint. Its private
/// constructor binds the actual native owner; retaining it prevents mutable
/// access to the persistent owner during journal and raw-queue publication.
#[cfg(feature = "gpu")]
pub struct VerifiedAutonomousChunkEndpoint<'owner> {
    native_owner: &'owner NhsAmberFusedEngine,
    completed_step: u32,
    _launch_lease: std::sync::MutexGuard<'owner,u64>,
}
#[cfg(feature = "gpu")]
impl VerifiedAutonomousChunkEndpoint<'_> {
    pub fn native_owner(&self) -> &NhsAmberFusedEngine { self.native_owner }
    pub fn completed_step(&self) -> u32 { self.completed_step }
    pub fn observer_physical_time_records(&self,start:u64,end:u64)
        -> Result<Vec<crate::fused_engine::ObserverPhysicalTimeRecord>> {
        self.native_owner.observer_physical_time_records_at_verified_endpoint(self,start,end)
    }
}

/// Direct-device-clock evidence for one graph replay window that this
/// persistent engine itself launched and stream-fenced.
#[cfg(feature = "gpu")]
#[derive(Clone, Debug)]
pub struct ResidentGraphReplayWindowReceipt {
    pub start_device_step: u32,
    pub end_device_step: u32,
    pub logical_steps: u32,
    pub graph_launches: u32,
    pub steps_per_launch: u32,
}

/// Direct-device-clock evidence for a complete resident graph cadence window
/// followed by native LADD dispatch on the same persistent CUDA stream.
#[cfg(feature = "gpu")]
#[derive(Clone, Debug)]
pub struct ResidentGraphLaddCadenceWindowReceipt {
    pub start_device_step: u32,
    pub end_device_step: u32,
    pub cadence_physical_steps: u32,
    pub graph_windows: Vec<ResidentGraphReplayWindowReceipt>,
}

#[cfg(feature = "gpu")]
#[derive(Clone, Debug)]
struct ResidentGraphReplayBindingState {
    capture_nonce: u64,
    resident_structure_fingerprint_sha256: String,
    graph_object_address: usize,
    graph_template_handle: usize,
    graph_exec_handle: usize,
    profile: ResidentGraphBindingProfile,
}

#[cfg(feature = "gpu")]
fn resolve_persistent_fused_ptx_path() -> Result<PathBuf> {
    for variable in ["PRISM4D_PTX_DIR", "PRISM_PTX_DIR"] {
        if let Ok(raw) = std::env::var(variable) {
            let directory = raw.trim();
            if directory.is_empty() {
                bail!("{} is set but empty", variable);
            }
            let path = PathBuf::from(directory).join("nhs_amber_fused.ptx");
            if !path.is_file() {
                bail!(
                    "{} is set but its authoritative fused PTX is missing: {}",
                    variable,
                    path.display()
                );
            }
            return Ok(path);
        }
    }
    [
        PathBuf::from("../prism-gpu/src/kernels/nhs_amber_fused.ptx"),
        PathBuf::from("crates/prism-gpu/src/kernels/nhs_amber_fused.ptx"),
        PathBuf::from("target/ptx/nhs_amber_fused.ptx"),
    ]
    .into_iter()
    .find(|path| path.is_file())
    .ok_or_else(|| anyhow::anyhow!(
        "nhs_amber_fused.ptx not found; set PRISM4D_PTX_DIR or PRISM_PTX_DIR to the installed PTX authority"
    ))
}

/// Configuration for persistent batch processing
#[derive(Debug, Clone)]
pub struct PersistentBatchConfig {
    /// Maximum atoms to pre-allocate for (prevents reallocation)
    pub max_atoms: usize,
    /// Grid dimension for exclusion field
    pub grid_dim: usize,
    /// Grid spacing in Angstroms
    pub grid_spacing: f32,
    /// Survey phase steps (cryo)
    pub survey_steps: i32,
    /// Convergence phase steps (warming)
    pub convergence_steps: i32,
    /// Precision phase steps (production)
    pub precision_steps: i32,
    /// Target temperature (K)
    pub temperature: f32,
    /// Cryo temperature (K)
    pub cryo_temp: f32,
    /// Cryo hold steps before warming
    pub cryo_hold: i32,
    /// Exact V2 is mandatory by default. Legacy execution requires an explicit
    /// non-production CLI option and caps receipts at IMPLEMENTATION_PARTIAL.
    pub nonbonded_execution_policy: NonbondedExecutionPolicy,
}

impl Default for PersistentBatchConfig {
    fn default() -> Self {
        Self {
            max_atoms: 20000, // Handle large structures (1DLO ~16K atoms)
            grid_dim: 128,
            grid_spacing: 0.75,
            survey_steps: 500000,       // 1ns
            convergence_steps: 1000000, // 2ns
            precision_steps: 1000000,   // 2ns
            temperature: 300.0,
            cryo_temp: 100.0,
            cryo_hold: 100000,
            nonbonded_execution_policy: NonbondedExecutionPolicy::RequireExactV2,
        }
    }
}

/// Result from processing a single structure
#[derive(Debug, Clone)]
pub struct StructureResult {
    pub structure_id: String,
    pub total_steps: i32,
    pub wall_time_ms: u64,
    pub spike_events: Vec<SpikeEvent>,
    pub snapshots: Vec<EnsembleSnapshot>,
    pub final_temperature: f32,
    /// Clustered binding sites from RT-accelerated spatial analysis
    pub clustered_sites: Vec<ClusteredBindingSite>,
    /// RT clustering statistics
    pub clustering_stats: Option<ClusteringStats>,
}

/// A residue lining a binding site pocket
#[derive(Debug, Clone, serde::Serialize)]
pub struct LiningResidue {
    /// Chain identifier (e.g., "A", "B")
    pub chain: String,
    /// Residue sequence number
    pub resid: i32,
    /// Residue name (e.g., "ALA", "PHE")
    pub resname: String,
    /// Minimum distance from any atom to pocket centroid (Å)
    pub min_distance: f32,
    /// Number of atoms within cutoff (geometric — from compute_lining_residues)
    pub n_atoms_in_pocket: usize,
    /// Number of GPU spike events attributed to this residue across all chunks
    /// (dynamics — populated by the multi-stream consensus spike-attribution pass).
    /// Distinct from `n_atoms_in_pocket`: the geometric atom count tells you which
    /// residues line the pocket; the spike attribution count tells you which of
    /// those residues actually carried activity during the run.
    #[serde(default)]
    pub spike_attribution_count: u32,
}

/// A clustered binding site detected from spike spatial patterns.
///
/// # Spatial representation contract (Phase 1 post-05dbc3dc lane)
///
/// Every canonical site carries a typed multi-view localization
/// [`CentroidManifold`][crate::spatial_view::CentroidManifold] in
/// `localization`. Canonical-logic callers MUST access site centroids
/// through [`ClusteredBindingSite::view`] / [`ClusteredBindingSite::distance_to`] /
/// [`ClusteredBindingSite::dcc`] / [`ClusteredBindingSite::distance_to_ligand`]
/// naming the physical representation they want. There is no hidden
/// default; absent views return `None`.
///
/// The `legacy_emission_centroid` field is retained in module-private
/// visibility (`pub(in crate::persistent_engine)`) only for the
/// serialization / logging compatibility path exposed via
/// [`ClusteredBindingSite::emission_compat_centroid`]. It must not be
/// used for merge, attribution, ranking, validation, or any canonical
/// logic — those must call the typed view API.
///
/// Writes to the geometric-voxel-mass centroid must go through
/// [`ClusteredBindingSite::set_geometric_voxel_mass_centroid`], which
/// updates both the legacy emission field and the manifold's
/// GeometricVoxelMass view atomically. Other views are written via
/// `localization.set(SpatialView::...)` directly.
#[derive(Debug, Clone)]
pub struct ClusteredBindingSite {
    /// Cluster ID (unique per analysis run)
    pub cluster_id: i32,
    /// Legacy single-scalar centroid retained solely for emission /
    /// logging / serialization compatibility. **DO NOT read this in
    /// canonical logic** — use [`ClusteredBindingSite::view`] or the
    /// other typed accessors. The field is module-private so that
    /// external consumers (bin/nhs_rt_full.rs, etc.) cannot touch
    /// the ambiguous scalar without going through an explicit shim.
    pub(in crate::persistent_engine) legacy_emission_centroid: [f32; 3],
    /// Multi-view localization manifold. Always contains at least
    /// the `GeometricVoxelMass` view when constructed through the
    /// canonical site-building paths.
    pub localization: crate::spatial_view::CentroidManifold,
    /// Number of spikes in this cluster
    pub spike_count: usize,
    /// Spike indices belonging to this cluster
    pub spike_indices: Vec<usize>,
    /// Average spike intensity in cluster
    pub avg_intensity: f32,
    /// Estimated volume (convex hull approximation) in Å³
    pub estimated_volume: f32,
    /// Bounding box dimensions [dx, dy, dz] in Angstroms
    pub bounding_box: [f32; 3],
    /// Quality score for this binding site (0.0-1.0)
    pub quality_score: f32,
    /// Druggability assessment
    pub druggability: DruggabilityScore,
    /// Site classification
    pub classification: SiteClassification,
    /// Aromatic proximity analysis (if computed)
    pub aromatic_proximity: Option<AromaticProximityInfo>,
    /// Residues lining this binding pocket (within cutoff distance)
    pub lining_residues: Vec<LiningResidue>,
}

/// Aromatic residue proximity information for a binding site
#[derive(Debug, Clone, Default)]
pub struct AromaticProximityInfo {
    /// Distance to nearest TRP residue (Å), None if no TRP in structure
    pub nearest_trp_distance: Option<f32>,
    /// Distance to nearest TYR residue (Å)
    pub nearest_tyr_distance: Option<f32>,
    /// Distance to nearest PHE residue (Å)
    pub nearest_phe_distance: Option<f32>,
    /// Distance to nearest aromatic (any type)
    pub nearest_aromatic_distance: f32,
    /// Number of aromatics within 5Å
    pub aromatics_within_5a: usize,
    /// Number of aromatics within 8Å
    pub aromatics_within_8a: usize,
    /// Aromatic residue indices within 8Å
    pub nearby_aromatic_residues: Vec<u32>,
    /// Aromatic score (0.0-1.0) based on proximity
    pub aromatic_score: f32,
}

/// Druggability score for a binding site
#[derive(Debug, Clone, Default)]
pub struct DruggabilityScore {
    /// Overall druggability (0.0-1.0)
    pub overall: f32,
    /// Volume contribution (0.0-1.0) - sites need 200-800 Å³
    pub volume_score: f32,
    /// Enclosure score (0.0-1.0) - how enclosed the pocket is
    pub enclosure_score: f32,
    /// Hydrophobicity score (0.0-1.0) - dewetting signal strength
    pub hydrophobicity_score: f32,
    /// Aromatic score (0.0-1.0) - proximity to aromatics for pi-stacking
    pub aromatic_score: f32,
    /// Catalytic score (0.0-1.0) - enzyme active site potential
    pub catalytic_score: f32,
    /// Is this likely a druggable pocket?
    pub is_druggable: bool,
}

/// Catalytic residue types for enzyme active site detection
pub const CATALYTIC_RESIDUES: &[&str] = &[
    "GLU", "ASP", "HIS", "HID", "HIE", "HIP", "SER", "CYS", "LYS",
];

/// Compute catalytic score from lining residues
///
/// Enzyme active sites typically have:
/// - Multiple catalytic residues (GLU, ASP, HIS, SER, CYS, LYS)
/// - Catalytic residues clustered within 6Å of pocket center
/// - At least 2-3 catalytic residues for enzymatic activity
pub fn compute_catalytic_score(lining_residues: &[LiningResidue]) -> (f32, usize) {
    if lining_residues.is_empty() {
        return (0.0, 0);
    }

    // Count catalytic residues and their proximity
    let mut catalytic_count = 0;
    let mut close_catalytic = 0; // Within 5Å
    let mut total_catalytic_distance = 0.0f32;

    for res in lining_residues {
        if CATALYTIC_RESIDUES.contains(&res.resname.as_str()) {
            catalytic_count += 1;
            total_catalytic_distance += res.min_distance;
            if res.min_distance <= 5.0 {
                close_catalytic += 1;
            }
        }
    }

    if catalytic_count == 0 {
        return (0.0, 0);
    }

    // Score components:
    // 1. Count score: 2-3 catalytic residues is optimal for enzyme activity
    let count_score: f32 = match catalytic_count {
        0 => 0.0,
        1 => 0.3,
        2 => 0.7,
        3 => 1.0,
        4 => 0.95,
        5 => 0.9,
        _ => 0.85, // Many catalytic residues still good
    };

    // 2. Proximity score: closer catalytic residues = more likely active site
    let avg_distance = total_catalytic_distance / catalytic_count as f32;
    let proximity_score: f32 = if avg_distance <= 3.0 {
        1.0
    } else if avg_distance <= 5.0 {
        0.9
    } else if avg_distance <= 7.0 {
        0.7
    } else {
        0.5
    };

    // 3. Close clustering bonus: multiple catalytic residues within 5Å
    let clustering_bonus: f32 = match close_catalytic {
        0 => 0.0,
        1 => 0.1,
        2 => 0.2,
        _ => 0.3,
    };

    // Combined score
    let score: f32 =
        (0.5 * count_score + 0.3 * proximity_score + 0.2 * clustering_bonus).clamp(0.0, 1.0);

    (score, catalytic_count)
}

impl DruggabilityScore {
    /// Compute druggability from binding site properties (without aromatic info)
    pub fn from_site(volume: f32, avg_intensity: f32, bounding_box: &[f32; 3]) -> Self {
        Self::from_site_with_aromatics(volume, avg_intensity, bounding_box, None)
    }

    /// Compute druggability with aromatic proximity information
    pub fn from_site_with_aromatics(
        volume: f32,
        avg_intensity: f32,
        bounding_box: &[f32; 3],
        aromatic_info: Option<&AromaticProximityInfo>,
    ) -> Self {
        // Volume scoring: optimal range 200-800 Å³
        let volume_score = if volume < 100.0 {
            volume / 100.0 * 0.3 // Too small
        } else if volume < 200.0 {
            0.3 + (volume - 100.0) / 100.0 * 0.4 // Getting better
        } else if volume <= 800.0 {
            0.7 + (1.0 - (volume - 200.0) / 600.0 * 0.3).max(0.7) // Optimal
        } else if volume <= 1500.0 {
            0.7 - (volume - 800.0) / 700.0 * 0.3 // Large but ok
        } else {
            0.4 - (volume - 1500.0) / 2000.0 * 0.2 // Too large (surface area)
        }
        .clamp(0.0, 1.0);

        // Enclosure: ratio of volume to bounding box volume
        let bb_volume = bounding_box[0] * bounding_box[1] * bounding_box[2];
        let enclosure_score = if bb_volume > 0.0 {
            (volume / bb_volume).clamp(0.0, 1.0) * 0.7 + 0.3 // Bias toward enclosed
        } else {
            0.0
        };

        // Hydrophobicity: spike intensity indicates dewetting strength
        let hydrophobicity_score = (avg_intensity / 10.0).clamp(0.0, 1.0);

        // Aromatic score: based on proximity to aromatic residues
        // Aromatics enable pi-stacking with drug molecules
        let aromatic_score = aromatic_info.map(|info| info.aromatic_score).unwrap_or(0.5); // Default to neutral if not computed

        // Overall: weighted combination (with aromatics)
        let overall = if aromatic_info.is_some() {
            // With aromatic info: 30% volume, 20% enclosure, 25% hydrophobicity, 25% aromatic
            0.30 * volume_score
                + 0.20 * enclosure_score
                + 0.25 * hydrophobicity_score
                + 0.25 * aromatic_score
        } else {
            // Without aromatic info: original weights
            0.40 * volume_score + 0.30 * enclosure_score + 0.30 * hydrophobicity_score
        };

        // Druggable threshold: overall >= threshold AND volume in reasonable range
        // Bonus: sites with aromatics nearby get lower threshold (pi-stacking potential)
        let aromatic_bonus = aromatic_info
            .map(|info| info.aromatics_within_5a > 0 || info.aromatic_score > 0.4)
            .unwrap_or(false);
        let threshold = if aromatic_bonus { 0.40 } else { 0.48 };
        let is_druggable = overall >= threshold && volume >= 50.0 && volume <= 3000.0;

        Self {
            overall,
            volume_score,
            enclosure_score,
            hydrophobicity_score,
            aromatic_score,
            catalytic_score: 0.0, // Computed separately with lining residues
            is_druggable,
        }
    }

    /// Compute druggability with full context: aromatics AND catalytic residues
    ///
    /// Enzyme active sites are druggable through:
    /// - Substrate mimics (competitive inhibitors)
    /// - Covalent inhibitors (targeting catalytic Ser/Cys)
    /// - Allosteric modulation
    ///
    /// This method properly scores polar enzyme sites that would fail
    /// traditional hydrophobic druggability metrics.
    pub fn from_site_with_catalytic(
        volume: f32,
        avg_intensity: f32,
        bounding_box: &[f32; 3],
        aromatic_info: Option<&AromaticProximityInfo>,
        lining_residues: &[LiningResidue],
    ) -> Self {
        // Start with base scoring
        let mut score =
            Self::from_site_with_aromatics(volume, avg_intensity, bounding_box, aromatic_info);

        // Compute catalytic score
        let (catalytic_score, catalytic_count) = compute_catalytic_score(lining_residues);
        score.catalytic_score = catalytic_score;

        // For enzyme active sites: adjust scoring to account for polar nature
        if catalytic_count >= 2 && catalytic_score >= 0.5 {
            // This is likely an enzyme active site
            // Recalculate overall with catalytic contribution
            // Enzyme sites: 25% volume, 15% enclosure, 15% hydrophobicity, 20% aromatic, 25% catalytic
            let aromatic = aromatic_info.map(|i| i.aromatic_score).unwrap_or(0.5);
            score.overall = 0.25 * score.volume_score
                + 0.15 * score.enclosure_score
                + 0.15 * score.hydrophobicity_score
                + 0.20 * aromatic
                + 0.25 * catalytic_score;

            // Enzyme active sites with good catalytic score are druggable
            // (substrate mimics, covalent inhibitors, etc.)
            // Note: Large enzyme sites have low volume_score, so use lower threshold
            let enzyme_threshold = 0.35; // Lower threshold for enzyme sites
                                         // Note: Large multi-subunit enzymes (aldolases, etc.) can have
                                         // binding sites up to ~8000 Å³. Use < comparison to avoid
                                         // floating-point boundary issues with clamped volumes.
            score.is_druggable =
                score.overall >= enzyme_threshold && volume >= 50.0 && volume < 8001.0;
            // Enzyme sites can be quite large
        }

        score
    }
}

impl AromaticProximityInfo {
    /// Compute aromatic proximity for a site centroid given aromatic residue positions
    ///
    /// # Arguments
    /// * `site_centroid` - [x, y, z] position of binding site center
    /// * `aromatic_positions` - List of (residue_id, aromatic_type, [x, y, z]) for each aromatic
    ///
    /// # Aromatic Types
    /// * 0 = TRP (tryptophan)
    /// * 1 = TYR (tyrosine)
    /// * 2 = PHE (phenylalanine)
    pub fn compute(site_centroid: &[f32; 3], aromatic_positions: &[(u32, u8, [f32; 3])]) -> Self {
        if aromatic_positions.is_empty() {
            return Self::default();
        }

        let mut nearest_trp: Option<f32> = None;
        let mut nearest_tyr: Option<f32> = None;
        let mut nearest_phe: Option<f32> = None;
        let mut nearest_any = f32::MAX;
        let mut within_5a = 0usize;
        let mut within_8a = 0usize;
        let mut nearby_residues = Vec::new();

        for &(residue_id, aromatic_type, pos) in aromatic_positions {
            let dx = pos[0] - site_centroid[0];
            let dy = pos[1] - site_centroid[1];
            let dz = pos[2] - site_centroid[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();

            // Track nearest by type
            match aromatic_type {
                0 => {
                    // TRP
                    if nearest_trp.map_or(true, |d| dist < d) {
                        nearest_trp = Some(dist);
                    }
                }
                1 => {
                    // TYR
                    if nearest_tyr.map_or(true, |d| dist < d) {
                        nearest_tyr = Some(dist);
                    }
                }
                2 => {
                    // PHE
                    if nearest_phe.map_or(true, |d| dist < d) {
                        nearest_phe = Some(dist);
                    }
                }
                _ => {}
            }

            // Track nearest any
            if dist < nearest_any {
                nearest_any = dist;
            }

            // Count within distance thresholds
            if dist <= 5.0 {
                within_5a += 1;
            }
            if dist <= 8.0 {
                within_8a += 1;
                nearby_residues.push(residue_id);
            }
        }

        // Compute aromatic score based on proximity
        // Higher score for closer aromatics, especially TRP (strongest UV absorber)
        let aromatic_score =
            Self::compute_aromatic_score(nearest_trp, nearest_tyr, nearest_phe, within_5a);

        Self {
            nearest_trp_distance: nearest_trp,
            nearest_tyr_distance: nearest_tyr,
            nearest_phe_distance: nearest_phe,
            nearest_aromatic_distance: if nearest_any < f32::MAX {
                nearest_any
            } else {
                0.0
            },
            aromatics_within_5a: within_5a,
            aromatics_within_8a: within_8a,
            nearby_aromatic_residues: nearby_residues,
            aromatic_score,
        }
    }

    /// Compute aromatic score from distances
    fn compute_aromatic_score(
        nearest_trp: Option<f32>,
        nearest_tyr: Option<f32>,
        nearest_phe: Option<f32>,
        within_5a: usize,
    ) -> f32 {
        // TRP is most important (strongest UV absorber, best for pi-stacking)
        let trp_score = nearest_trp
            .map(|d| Self::distance_to_score(d, 1.5)) // TRP weight 1.5x
            .unwrap_or(0.0);

        // TYR is moderate
        let tyr_score = nearest_tyr
            .map(|d| Self::distance_to_score(d, 1.0))
            .unwrap_or(0.0);

        // PHE is weakest
        let phe_score = nearest_phe
            .map(|d| Self::distance_to_score(d, 0.7))
            .unwrap_or(0.0);

        // Combine scores (take best + bonus for multiple)
        let base_score = trp_score.max(tyr_score).max(phe_score);
        let multi_bonus = (within_5a as f32 * 0.05).min(0.2); // Up to 0.2 bonus

        (base_score + multi_bonus).clamp(0.0, 1.0)
    }

    /// Convert distance to score (closer = higher)
    fn distance_to_score(distance: f32, weight: f32) -> f32 {
        if distance < 3.0 {
            // Direct contact: highest score
            weight * 1.0
        } else if distance < 5.0 {
            // Close proximity: good score
            weight * (1.0 - (distance - 3.0) / 2.0 * 0.3)
        } else if distance < 8.0 {
            // Medium range: moderate score
            weight * (0.7 - (distance - 5.0) / 3.0 * 0.3)
        } else {
            // Distal: low score
            weight * (0.4 - (distance - 8.0) / 10.0 * 0.4).max(0.0)
        }
    }
}

/// Classification of detected binding site
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SiteClassification {
    /// Traditional active site (large, persistent)
    ActiveSite,
    /// Allosteric binding site (remote from active site)
    Allosteric,
    /// Cryptic site (only appears transiently)
    Cryptic,
    /// Protein-protein interaction surface
    PpiSurface,
    /// Membrane interface region
    MembraneInterface,
    /// Unclassified
    Unknown,
}

impl Default for SiteClassification {
    fn default() -> Self {
        Self::Unknown
    }
}

impl SiteClassification {
    /// Classify based on spike patterns and volume
    pub fn from_properties(spike_count: usize, volume: f32, _avg_intensity: f32) -> Self {
        if volume >= 400.0 && spike_count >= 50 {
            Self::ActiveSite
        } else if volume >= 200.0 && volume <= 600.0 && spike_count >= 20 {
            Self::Cryptic // Moderate size, fewer spikes → transient
        } else if volume >= 150.0 && spike_count >= 10 {
            Self::Allosteric
        } else if volume >= 800.0 {
            Self::PpiSurface // Large surface areas
        } else {
            Self::Unknown
        }
    }
}

/// Statistics from RT clustering
#[derive(Debug, Clone)]
pub struct ClusteringStats {
    /// Number of clusters found
    pub num_clusters: usize,
    /// Total neighbor pairs examined
    pub total_neighbors: usize,
    /// GPU time in milliseconds
    pub gpu_time_ms: f64,
    /// Whether RT cores were used (vs fallback)
    pub used_rt_cores: bool,
}

// ═══════════════════════════════════════════════════════════════════════════════
// MULTI-SCALE CLUSTERING
// ═══════════════════════════════════════════════════════════════════════════════

/// Cluster found at a single scale (epsilon value)
#[derive(Debug, Clone)]
pub struct ScaleCluster {
    /// Cluster centroid [x, y, z]
    pub centroid: [f32; 3],
    /// Number of spikes in this cluster
    pub spike_count: usize,
    /// Spike indices belonging to this cluster
    pub spike_indices: Vec<usize>,
    /// Epsilon value (scale) at which this cluster was found
    pub epsilon: f32,
    /// Original cluster ID from DBSCAN at this scale
    pub original_cluster_id: i32,
    /// Persistence score (how many scales this cluster appears at)
    pub persistence: usize,
}

/// Merged cluster from multiple scales
#[derive(Debug, Clone)]
pub struct MergedCluster {
    /// Merged centroid (average across scales)
    pub centroid: [f32; 3],
    /// Total unique spikes across all scales
    pub spike_count: usize,
    /// Union of spike indices from all contributing scale clusters
    pub spike_indices: Vec<usize>,
    /// Persistence: number of scales this cluster appears at
    pub persistence: usize,
    /// List of epsilon values where this cluster was detected
    pub scales: Vec<f32>,
}

/// Result of multi-scale clustering
#[derive(Debug, Clone)]
pub struct MultiScaleClusteringResult {
    /// Merged clusters sorted by confidence (persistence × spike_count)
    pub clusters: Vec<MergedCluster>,
    /// Number of epsilon scales used
    pub total_scales: usize,
    /// Epsilon values tested
    pub epsilon_values: Vec<f32>,
    /// Whether adaptive epsilon was used (vs fixed)
    pub adaptive_epsilon: bool,
    /// k value used for k-NN (if adaptive)
    pub knn_k: Option<usize>,
    /// Number of spikes sampled for k-NN (if adaptive)
    pub num_spikes_sampled: Option<usize>,
}

impl MultiScaleClusteringResult {
    /// Convert to cluster IDs array (for compatibility with single-scale API)
    ///
    /// Returns cluster ID for each spike position, using the merged cluster assignments.
    /// Spikes not in any persistent cluster get -1 (noise).
    pub fn to_cluster_ids(&self, num_spikes: usize) -> Vec<i32> {
        let mut cluster_ids = vec![-1i32; num_spikes];

        for (cluster_idx, cluster) in self.clusters.iter().enumerate() {
            for &spike_idx in &cluster.spike_indices {
                if spike_idx < num_spikes {
                    cluster_ids[spike_idx] = cluster_idx as i32;
                }
            }
        }

        cluster_ids
    }

    /// Get the number of persistent clusters
    pub fn num_clusters(&self) -> usize {
        self.clusters.len()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SITE PERSISTENCE TRACKING
// ═══════════════════════════════════════════════════════════════════════════════

/// Tracks binding site persistence across trajectory frames
///
/// Sites that persist across multiple frames are more reliable drug targets.
/// Transient sites may be cryptic pockets that only appear during conformational changes.
#[derive(Debug, Clone)]
pub struct SitePersistenceTracker {
    /// Tracked sites by unique ID
    tracked_sites: Vec<TrackedSite>,
    /// Spatial matching threshold (Å)
    match_threshold: f32,
    /// Total frames processed
    total_frames: usize,
    /// Next unique site ID
    next_site_id: u64,
}

/// A binding site being tracked across frames
#[derive(Debug, Clone)]
pub struct TrackedSite {
    /// Unique ID for this site across all frames
    pub site_id: u64,
    /// Running average centroid position
    pub avg_centroid: [f32; 3],
    /// Running average volume
    pub avg_volume: f32,
    /// Number of frames this site was detected
    pub frame_count: usize,
    /// First frame this site appeared
    pub first_frame: usize,
    /// Most recent frame this site was seen
    pub last_frame: usize,
    /// Consecutive frames detected (current streak)
    pub consecutive_frames: usize,
    /// Maximum consecutive detection streak
    pub max_consecutive: usize,
    /// Running average spike count
    pub avg_spike_count: f32,
    /// Running average quality score
    pub avg_quality: f32,
    /// Is this site considered persistent? (detected in >50% of frames)
    pub is_persistent: bool,
    /// Site classification (most common across frames)
    pub classification: SiteClassification,
    /// Frame-by-frame spike counts for variability analysis
    spike_history: Vec<usize>,
}

/// Result of persistence analysis
#[derive(Debug, Clone)]
pub struct PersistenceAnalysis {
    /// Total frames analyzed
    pub total_frames: usize,
    /// All tracked sites with persistence info
    pub sites: Vec<TrackedSite>,
    /// Number of persistent sites (>50% frame presence)
    pub persistent_count: usize,
    /// Number of transient sites (<50% frame presence)
    pub transient_count: usize,
    /// Average site lifetime (frames)
    pub avg_lifetime: f32,
}

impl SitePersistenceTracker {
    /// Create a new persistence tracker
    ///
    /// # Arguments
    /// * `match_threshold` - Maximum distance (Å) for sites to be considered the same
    pub fn new(match_threshold: f32) -> Self {
        Self {
            tracked_sites: Vec::new(),
            match_threshold,
            total_frames: 0,
            next_site_id: 0,
        }
    }

    /// Process binding sites from a single frame
    ///
    /// Matches sites to existing tracked sites or creates new ones.
    pub fn process_frame(&mut self, frame_sites: &[ClusteredBindingSite]) {
        self.total_frames += 1;
        let current_frame = self.total_frames;

        // Mark all sites as not-seen-this-frame
        let mut matched = vec![false; self.tracked_sites.len()];

        for site in frame_sites {
            // Find best matching tracked site
            let mut best_match: Option<usize> = None;
            let mut best_dist = f32::MAX;

            for (idx, tracked) in self.tracked_sites.iter().enumerate() {
                if matched[idx] {
                    continue;
                }

                // Site tracker matches on GeometricVoxelMass: the tracked
                // average centroid is computed from successive
                // GeometricVoxelMass views (see update_tracked_site). No
                // hidden default; view is explicit.
                let site_gvm = site
                    .view(crate::spatial_view::SpatialView::GeometricVoxelMass)
                    .expect("ClusteredBindingSite always carries GeometricVoxelMass view");
                let dist = Self::distance(&site_gvm, &tracked.avg_centroid);
                if dist < self.match_threshold && dist < best_dist {
                    best_match = Some(idx);
                    best_dist = dist;
                }
            }

            if let Some(idx) = best_match {
                // Update existing site
                self.update_tracked_site(idx, site, current_frame);
                matched[idx] = true;
            } else {
                // Create new tracked site
                self.create_tracked_site(site, current_frame);
            }
        }

        // Update sites not seen this frame (break consecutive streak)
        for (idx, was_matched) in matched.iter().enumerate() {
            if !was_matched {
                self.tracked_sites[idx].consecutive_frames = 0;
            }
        }

        // Update persistence status for all sites
        self.update_persistence_status();
    }

    /// Update an existing tracked site with new detection
    fn update_tracked_site(&mut self, idx: usize, site: &ClusteredBindingSite, frame: usize) {
        let tracked = &mut self.tracked_sites[idx];

        // Update running averages over the GeometricVoxelMass view —
        // the tracker explicitly averages that representation; no
        // implicit view selection.
        let n = tracked.frame_count as f32;
        let site_gvm = site
            .view(crate::spatial_view::SpatialView::GeometricVoxelMass)
            .expect("ClusteredBindingSite always carries GeometricVoxelMass view");
        tracked.avg_centroid[0] = (tracked.avg_centroid[0] * n + site_gvm[0]) / (n + 1.0);
        tracked.avg_centroid[1] = (tracked.avg_centroid[1] * n + site_gvm[1]) / (n + 1.0);
        tracked.avg_centroid[2] = (tracked.avg_centroid[2] * n + site_gvm[2]) / (n + 1.0);
        tracked.avg_volume = (tracked.avg_volume * n + site.estimated_volume) / (n + 1.0);
        tracked.avg_spike_count =
            (tracked.avg_spike_count * n + site.spike_count as f32) / (n + 1.0);
        tracked.avg_quality = (tracked.avg_quality * n + site.quality_score) / (n + 1.0);

        tracked.frame_count += 1;
        tracked.last_frame = frame;
        tracked.consecutive_frames += 1;
        tracked.max_consecutive = tracked.max_consecutive.max(tracked.consecutive_frames);
        tracked.spike_history.push(site.spike_count);

        // Keep most common classification (or use most recent if better quality)
        if site.quality_score > tracked.avg_quality * 1.2 {
            tracked.classification = site.classification;
        }
    }

    /// Create a new tracked site
    fn create_tracked_site(&mut self, site: &ClusteredBindingSite, frame: usize) {
        let site_id = self.next_site_id;
        self.next_site_id += 1;

        let site_gvm = site
            .view(crate::spatial_view::SpatialView::GeometricVoxelMass)
            .expect("ClusteredBindingSite always carries GeometricVoxelMass view");
        self.tracked_sites.push(TrackedSite {
            site_id,
            // Seed tracker average from the GeometricVoxelMass view.
            avg_centroid: site_gvm,
            avg_volume: site.estimated_volume,
            frame_count: 1,
            first_frame: frame,
            last_frame: frame,
            consecutive_frames: 1,
            max_consecutive: 1,
            avg_spike_count: site.spike_count as f32,
            avg_quality: site.quality_score,
            is_persistent: false,
            classification: site.classification,
            spike_history: vec![site.spike_count],
        });
    }

    /// Update persistence status for all sites
    fn update_persistence_status(&mut self) {
        let threshold = self.total_frames as f32 * 0.5; // 50% of frames

        for site in &mut self.tracked_sites {
            site.is_persistent = site.frame_count as f32 >= threshold;
        }
    }

    /// Get final persistence analysis
    pub fn analyze(&self) -> PersistenceAnalysis {
        let persistent_count = self
            .tracked_sites
            .iter()
            .filter(|s| s.is_persistent)
            .count();
        let transient_count = self.tracked_sites.len() - persistent_count;

        let avg_lifetime = if self.tracked_sites.is_empty() {
            0.0
        } else {
            self.tracked_sites
                .iter()
                .map(|s| s.frame_count as f32)
                .sum::<f32>()
                / self.tracked_sites.len() as f32
        };

        PersistenceAnalysis {
            total_frames: self.total_frames,
            sites: self.tracked_sites.clone(),
            persistent_count,
            transient_count,
            avg_lifetime,
        }
    }

    /// Get tracked sites sorted by persistence (most persistent first)
    pub fn get_persistent_sites(&self) -> Vec<&TrackedSite> {
        let mut sites: Vec<_> = self.tracked_sites.iter().collect();
        sites.sort_by(|a, b| {
            // Sort by frame_count descending, then by avg_quality descending
            b.frame_count.cmp(&a.frame_count).then_with(|| {
                b.avg_quality
                    .partial_cmp(&a.avg_quality)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });
        sites
    }

    /// Calculate Euclidean distance between two positions
    fn distance(a: &[f32; 3], b: &[f32; 3]) -> f32 {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

impl TrackedSite {
    /// Calculate spike count variability (coefficient of variation)
    pub fn spike_variability(&self) -> f32 {
        if self.spike_history.len() < 2 {
            return 0.0;
        }

        let mean = self.avg_spike_count;
        if mean < 0.001 {
            return 0.0;
        }

        let variance: f32 = self
            .spike_history
            .iter()
            .map(|&x| (x as f32 - mean).powi(2))
            .sum::<f32>()
            / self.spike_history.len() as f32;

        variance.sqrt() / mean // CV
    }

    /// Get frame presence ratio (0.0 to 1.0)
    pub fn presence_ratio(&self, total_frames: usize) -> f32 {
        if total_frames == 0 {
            return 0.0;
        }
        self.frame_count as f32 / total_frames as f32
    }
}

/// Persistent engine that keeps GPU state alive across structures
#[cfg(feature = "gpu")]
pub struct PersistentNhsEngine {
    /// Shared CUDA context (kept alive)
    context: Arc<CudaContext>,
    /// Compiled module (kept alive)
    module: Arc<CudaModule>,
    /// Stream for operations
    stream: Arc<CudaStream>,

    /// Currently loaded engine instance
    engine: Option<NhsAmberFusedEngine>,
    prismcode_md_policy_admission: Option<PrismCodeMdPolicyAdmissionV1>,

    /// Optional-housekeeping receipt bindings keyed by their opaque capture
    /// nonce. Exact-tail graphs coexist with their primary graph; all bindings
    /// are invalidated when topology replacement changes engine allocations.
    resident_graph_replay_bindings: BTreeMap<u64, ResidentGraphReplayBindingState>,
    resident_graph_replay_binding_domain: u64,
    resident_graph_capture_nonce: u64,
    /// Derived only from actual readback; invalidated before any alternate clock writer.
    autonomous_chunk_clock: AutonomousChunkClock,
    physical_launch_domain: Arc<crate::graph_capture::PhysicalGraphLaunchDomain>,
    autonomous_chunk_observed_generation: Option<u64>,

    /// RT-accelerated clustering engine (lazy initialized)
    rt_engine: Option<crate::rt_clustering::RtClusteringEngine>,

    /// GPU spatial-hash CCL clustering backend (lazy initialized; Phase P0x).
    /// When present, `cluster_spikes` dispatches here on SM120+ instead of
    /// the `fallback_grid_cluster` CPU path. See `gpu_cluster_backend.rs`.
    gpu_cluster: Option<crate::gpu_cluster_backend::GpuSpatialHashBackend>,
    /// GhostPhaseLattice4D backend — physically-constrained 4D edge
    /// adjudication on Ghost v2 records. Lazy-initialised on the first
    /// call to `cluster_ghost_phase_lattice` (or when `cluster_spikes`
    /// has been routed through `BACKEND_GHOST_LATTICE_4D` and a Ghost v2
    /// payload is available on disk).
    ghost_lattice: Option<crate::ghost_phase_lattice::GhostPhaseLattice4D>,

    /// Pre-allocated buffer capacity
    max_atoms: usize,

    /// Grid configuration
    grid_dim: usize,
    grid_spacing: f32,
    nonbonded_execution_policy: NonbondedExecutionPolicy,
    /// Content-addressed per-stream lambda/no-probe runtime binding. It must
    /// be installed before topology load and is consumed exactly once by the
    /// fused-engine constructor.
    lambda_runtime_binding: Option<crate::lambda_fragment::LambdaRuntimeBinding>,

    /// Current topology ID
    current_topology_id: Option<String>,

    /// Initialization time tracking
    context_init_time_ms: u64,
    module_init_time_ms: u64,

    /// RT engine initialization time (if initialized)
    rt_init_time_ms: Option<u64>,

    /// Cumulative statistics
    structures_processed: usize,
    total_steps_run: i64,
    total_compute_time_ms: u64,
}

#[cfg(feature = "gpu")]
impl PersistentNhsEngine {
    /// Create persistent engine with pre-allocated capacity
    pub fn new(config: &PersistentBatchConfig) -> Result<Self> {
        log::info!(
            "🚀 Initializing Persistent NHS Engine (max_atoms: {})",
            config.max_atoms
        );

        // Time context creation
        let ctx_start = Instant::now();
        let context = CudaContext::new(0).context("Failed to create CUDA context")?;
        let context_init_time_ms = ctx_start.elapsed().as_millis() as u64;
        log::info!("  CUDA context: {}ms", context_init_time_ms);

        // Time module loading
        let mod_start = Instant::now();

        let module = if let Some((module, loaded_path)) =
            crate::fused_engine::load_qnanl_force_kernel_module_if_configured(&context)?
        {
            log::info!("  Force-kernel runtime authority: {}", loaded_path);
            module
        } else {
            let ptx_path = resolve_persistent_fused_ptx_path()?;
            log::info!("  Persistent fused PTX authority: {}", ptx_path.display());
            context
                .load_module(Ptx::from_file(&ptx_path))
                .context("Failed to load NHS-AMBER fused PTX")?
        };
        let module_init_time_ms = mod_start.elapsed().as_millis() as u64;
        log::info!("  PTX module: {}ms", module_init_time_ms);

        let stream = context.default_stream();

        log::info!(
            "✅ Persistent engine ready (total init: {}ms)",
            context_init_time_ms + module_init_time_ms
        );
        let resident_graph_replay_binding_domain =
            RESIDENT_GRAPH_REPLAY_BINDING_DOMAIN_COUNTER.fetch_add(1, Ordering::Relaxed);
        if resident_graph_replay_binding_domain == 0 {
            bail!("resident graph replay-binding domain counter exhausted");
        }

        Ok(Self {
            context,
            module,
            stream,
            engine: None,
            prismcode_md_policy_admission: None,
            resident_graph_replay_bindings: BTreeMap::new(),
            resident_graph_replay_binding_domain,
            resident_graph_capture_nonce: 0,
            autonomous_chunk_clock: AutonomousChunkClock::default(),
            physical_launch_domain: Arc::new(crate::graph_capture::PhysicalGraphLaunchDomain::default()),
            autonomous_chunk_observed_generation: None,
            rt_engine: None,     // Lazy initialized on first use
            gpu_cluster: None,   // Lazy initialized on first cluster_spikes call
            ghost_lattice: None, // Lazy initialized on first cluster_ghost_phase_lattice call
            max_atoms: config.max_atoms,
            grid_dim: config.grid_dim,
            grid_spacing: config.grid_spacing,
            nonbonded_execution_policy: config.nonbonded_execution_policy,
            lambda_runtime_binding: None,
            current_topology_id: None,
            context_init_time_ms,
            module_init_time_ms,
            rt_init_time_ms: None,
            structures_processed: 0,
            total_steps_run: 0,
            total_compute_time_ms: 0,
        })
    }

    /// Construct a single-stream engine from the exact content-addressed PTX
    /// selected by a live lambda/no-probe admission binding.
    pub fn new_with_lambda_runtime(
        config: &PersistentBatchConfig,
        binding: crate::lambda_fragment::LambdaRuntimeBinding,
    ) -> Result<Self> {
        let receipt = binding
            .runtime_capability_receipt
            .as_ref()
            .context("lambda persistent constructor lacks capability receipt")?;
        receipt
            .validate_live()
            .map_err(|error| anyhow::anyhow!("lambda persistent live admission: {error}"))?;
        let context = CudaContext::new(0).context("Failed to create CUDA context")?;
        let module = context
            .load_module(Ptx::from_file(&receipt.selected_fused_ptx_path))
            .with_context(|| {
                format!(
                    "load admitted lambda fused PTX {}",
                    receipt.selected_fused_ptx_path.display()
                )
            })?;
        let stream = context.default_stream();
        let mut engine = Self::new_on_stream(config, context, module, stream)?;
        engine.set_lambda_runtime_binding(binding)?;
        Ok(engine)
    }

    /// Create persistent engine on an explicit CUDA stream (for multi-stream concurrency).
    /// Shares context + module with other engines — each gets its own stream for GPU overlap.
    pub fn new_on_stream(
        config: &PersistentBatchConfig,
        context: Arc<CudaContext>,
        module: Arc<CudaModule>,
        stream: Arc<CudaStream>,
    ) -> Result<Self> {
        log::info!(
            "Persistent NHS Engine on dedicated stream (max_atoms: {})",
            config.max_atoms
        );
        let resident_graph_replay_binding_domain =
            RESIDENT_GRAPH_REPLAY_BINDING_DOMAIN_COUNTER.fetch_add(1, Ordering::Relaxed);
        if resident_graph_replay_binding_domain == 0 {
            bail!("resident graph replay-binding domain counter exhausted");
        }
        Ok(Self {
            context,
            module,
            stream,
            engine: None,
            prismcode_md_policy_admission: None,
            resident_graph_replay_bindings: BTreeMap::new(),
            resident_graph_replay_binding_domain,
            resident_graph_capture_nonce: 0,
            autonomous_chunk_clock: AutonomousChunkClock::default(),
            physical_launch_domain: Arc::new(crate::graph_capture::PhysicalGraphLaunchDomain::default()),
            autonomous_chunk_observed_generation: None,
            rt_engine: None,
            gpu_cluster: None,   // Lazy initialized on first cluster_spikes call
            ghost_lattice: None, // Lazy initialized on first cluster_ghost_phase_lattice call
            max_atoms: config.max_atoms,
            grid_dim: config.grid_dim,
            grid_spacing: config.grid_spacing,
            nonbonded_execution_policy: config.nonbonded_execution_policy,
            lambda_runtime_binding: None,
            current_topology_id: None,
            context_init_time_ms: 0,
            module_init_time_ms: 0,
            rt_init_time_ms: None,
            structures_processed: 0,
            total_steps_run: 0,
            total_compute_time_ms: 0,
        })
    }

    /// Access the shared CUDA context (for creating additional streams)
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Access the compiled PTX module (for sharing across engines)
    pub fn cuda_module(&self) -> &Arc<CudaModule> {
        &self.module
    }

    pub fn set_prismcode_md_policy_admission(
        &mut self,
        admission: PrismCodeMdPolicyAdmissionV1,
    ) -> Result<()> {
        if self.engine.is_some() || self.current_topology_id.is_some() {
            bail!("PrismCode MD policy admission must be installed before topology load");
        }
        if let Some(existing) = self.prismcode_md_policy_admission.as_ref() {
            if existing != &admission {
                bail!("persistent engine PrismCode policy admission cannot be rebound");
            }
            return Ok(());
        }
        self.prismcode_md_policy_admission = Some(admission);
        Ok(())
    }

    /// Install the exact per-stream lambda/no-probe runtime binding before
    /// topology load. Rebinding a loaded engine is forbidden because it would
    /// sever journal and capability lineage from the allocated device state.
    pub fn set_lambda_runtime_binding(
        &mut self,
        binding: crate::lambda_fragment::LambdaRuntimeBinding,
    ) -> Result<()> {
        if self.engine.is_some() || self.current_topology_id.is_some() {
            bail!("lambda runtime binding must be installed before topology load");
        }
        if self.lambda_runtime_binding.is_some() {
            bail!("lambda runtime binding is immutable once installed");
        }
        self.lambda_runtime_binding = Some(binding);
        Ok(())
    }

    /// Load a new topology (hot-swap)
    ///
    /// If the new topology fits in existing buffers, reuses them.
    /// Otherwise, reallocates with appropriate capacity.
    pub fn load_topology(&mut self, topology: &PrismPrepTopology) -> Result<()> {
        self.clear_resident_graph_replay_bindings();
        let topo_id = std::path::Path::new(&topology.source_pdb)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        log::info!(
            "📦 Loading topology: {} ({} atoms)",
            topo_id,
            topology.n_atoms
        );

        let load_start = Instant::now();

        // Check if we need to reallocate
        if topology.n_atoms > self.max_atoms {
            log::warn!(
                "  Structure exceeds max_atoms ({}), reallocating to {}",
                self.max_atoms,
                topology.n_atoms + 1000
            );
            self.max_atoms = topology.n_atoms + 1000;
        }

        // Create new engine instance with shared context
        // Note: In a more optimized version, we would reuse GPU buffers
        // For now, we benefit from shared context + module
        let mut engine = match self.nonbonded_execution_policy {
            NonbondedExecutionPolicy::RequireHistoricalMayModel => {
                if self.lambda_runtime_binding.is_some() || topology.lambda_fragment_system.is_some() {
                    bail!("historical May nonbonded execution forbids lambda runtime binding");
                }
                NhsAmberFusedEngine::new_on_stream_historical_may_model(
                    self.context.clone(),self.stream.clone(),topology,self.grid_dim,self.grid_spacing)?
            }
            NonbondedExecutionPolicy::RequireExactV2 => {
                if let Some(binding) = self.lambda_runtime_binding.clone() {
                    NhsAmberFusedEngine::new_on_stream_with_lambda_runtime(
                        self.context.clone(),
                        self.stream.clone(),
                        topology,
                        self.grid_dim,
                        self.grid_spacing,
                        binding,
                    )?
                } else {
                    if topology.lambda_fragment_system.is_some() {
                        bail!(
                            "lambda-fragment topology lacks an admitted persistent runtime binding"
                        );
                    }
                    NhsAmberFusedEngine::new_on_stream(
                        self.context.clone(),
                        self.stream.clone(),
                        topology,
                        self.grid_dim,
                        self.grid_spacing,
                    )?
                }
            }
            NonbondedExecutionPolicy::AllowNonProductionLegacyUniformFudge => {
                if self.lambda_runtime_binding.is_some()
                    || topology.lambda_fragment_system.is_some()
                {
                    bail!("lambda/no-probe runtime binding is forbidden on legacy nonbonded execution");
                }
                NhsAmberFusedEngine::new_on_stream_non_production_legacy_nonbonded(
                    self.context.clone(),
                    self.stream.clone(),
                    topology,
                    self.grid_dim,
                    self.grid_spacing,
                )?
            }
        };
        if let Some(admission) = self.prismcode_md_policy_admission.clone() {
            engine.bind_prismcode_md_policy_admission(admission)?;
        }

        self.engine = Some(engine);
        self.current_topology_id = Some(topo_id.clone());

        let load_time = load_start.elapsed().as_millis() as u64;
        log::info!("  Topology loaded: {}ms", load_time);

        Ok(())
    }

    /// **Configure unified cryo-UV protocol (RECOMMENDED)**
    ///
    /// Sets the integrated cryo-thermal + UV-LIF protocol for the current topology.
    /// This is the canonical PRISM4D cryptic site detection method.
    pub fn set_cryo_uv_protocol(&mut self, protocol: CryoUvProtocol) -> Result<()> {
        self.autonomous_chunk_clock.invalidate();
        if let Some(ref mut engine) = self.engine {
            engine.set_cryo_uv_protocol(protocol)?;
            Ok(())
        } else {
            bail!("No topology loaded")
        }
    }

    /// Enable adaptive dt.
    pub fn set_adaptive_dt(&mut self, enabled: bool) -> Result<()> {
        if let Some(ref mut engine) = self.engine {
            engine.set_adaptive_dt(enabled);
            Ok(())
        } else {
            bail!("No topology loaded")
        }
    }

    /// Telemetry observer — current host-side integration timestep (ps).
    /// Returns 0.0 if no topology loaded.
    pub fn current_dt_ps(&self) -> f64 {
        self.engine
            .as_ref()
            .map(|e| e.current_dt_ps())
            .unwrap_or(0.0)
    }

    /// Telemetry observer — base (pre-adaptive-scale) integration timestep (ps).
    /// Returns 0.0 if no topology loaded.
    pub fn base_dt_ps(&self) -> f64 {
        self.engine.as_ref().map(|e| e.base_dt_ps()).unwrap_or(0.0)
    }

    /// Telemetry observer — true while host-side adaptive_dt heuristic
    /// is the writer of `self.dt` (i.e., not in V2 gearbox mode).
    pub fn adaptive_dt_enabled(&self) -> bool {
        self.engine
            .as_ref()
            .map(|e| e.adaptive_dt_enabled())
            .unwrap_or(false)
    }

    /// Telemetry observer — true while the V2 captured pipeline owns dt.
    pub fn is_gearbox_active(&self) -> bool {
        self.engine
            .as_ref()
            .map(|e| e.is_gearbox_active())
            .unwrap_or(false)
    }

    /// Enable LADD observation kernel
    pub fn set_ladd_enabled(&mut self, enabled: bool) {
        if let Some(ref mut engine) = self.engine {
            engine.set_ladd_enabled(enabled);
        }
    }

    /// ASC steering: write focus residue to GPU ProtocolState.
    /// The Director kernel will read this and modulate UV targeting.
    pub fn set_steering_focus_residue(&mut self, residue_id: i32) {
        if let Some(ref mut engine) = self.engine {
            engine.set_steering_focus_residue(residue_id);
        }
    }

    /// Force-sync spike data from GPU after short run chunks.
    /// Returns number of new spikes downloaded.
    pub fn force_spike_sync(&mut self) -> anyhow::Result<usize> {
        if let Some(ref mut engine) = self.engine {
            engine.force_spike_sync()
        } else {
            anyhow::bail!("No engine initialized for forced GPU spike readback")
        }
    }

    /// Full spike archive without resetting the device population. Only the
    /// quiescent lossless coupled endpoint may use this split lifecycle.
    pub fn force_spike_sync_preserving_source(&mut self) -> anyhow::Result<usize> {
        self.engine.as_mut().context("No engine for lossless spike archive")?
            .force_spike_sync_preserving_source()
    }

    pub fn release_archived_spike_source(&mut self) -> anyhow::Result<()> {
        self.engine.as_mut().context("No engine for spike source release")?.clear_spike_events()
    }

    /// Synchronize the engine stream through a one-shot checked raw-driver
    /// barrier. This bypasses cudarc's shared deferred-error cell and fails
    /// closed on the first non-success driver result.
    pub fn synchronize_stream_raw(&self, label: &str) -> anyhow::Result<()> {
        if let Some(ref engine) = self.engine {
            engine.synchronize_stream_raw(label)
        } else {
            anyhow::bail!("No engine initialized for raw stream synchronization")
        }
    }

    /// Read the authoritative device-resident ProtocolState clock after a
    /// synchronized native graph campaign.  This is deliberately distinct
    /// from the host scheduling counter: callers that need an exact physical
    /// horizon must compare both values rather than infer device progress from
    /// the launch plan alone.
    pub fn device_protocol_current_step(&self) -> anyhow::Result<u32> {
        if let Some(ref engine) = self.engine {
            engine.device_protocol_current_step()
        } else {
            anyhow::bail!("No engine initialized for device ProtocolState clock readback")
        }
    }

    pub fn set_prismcode_md_native_materialized_readouts_required(&mut self, required: bool) {
        if let Some(ref mut engine) = self.engine {
            engine.set_prismcode_md_native_materialized_readouts_required(required);
        }
    }

    pub fn enable_prismcode_md_native_runtime(
        &mut self,
        stream_id: u32,
        require_complete: bool,
        materialized_readouts_required: bool,
    ) -> anyhow::Result<()> {
        if let Some(ref mut engine) = self.engine {
            engine.enable_prismcode_md_native_runtime(
                stream_id,
                require_complete,
                materialized_readouts_required,
            )
        } else {
            anyhow::bail!("No engine initialized")
        }
    }

    pub fn write_prismcode_md_native_receipt(
        &self,
        path: &std::path::Path,
    ) -> anyhow::Result<Option<prismcode::md_native_runtime::MdNativeRuntimeReceipt>> {
        if let Some(ref engine) = self.engine {
            engine.write_prismcode_md_native_receipt(path)
        } else {
            Ok(None)
        }
    }

    pub fn record_prismcode_md_native_qnanl_inline_seal_verify(
        &mut self,
        step: u64,
        inner_step: u32,
        event_byte_source_sha256: &str,
        qnanl_sealed_from_sha256: &str,
        byte_exact_decode_pass: bool,
    ) -> anyhow::Result<()> {
        if let Some(ref mut engine) = self.engine {
            engine.record_prismcode_md_native_qnanl_inline_seal_verify(
                step,
                inner_step,
                event_byte_source_sha256,
                qnanl_sealed_from_sha256,
                byte_exact_decode_pass,
            )
        } else {
            anyhow::bail!("No engine initialized for inline QNANL evidence")
        }
    }

    /// Record a nonresident graph replay supplied by an existing legacy path.
    /// Resident graph labels are refused here because their receipt must be
    /// emitted by the bound executor after its own launch and device fence.
    pub(crate) fn record_prismcode_md_native_graph_replay(
        &mut self,
        start_step: u64,
        launch_count: u32,
        launch_path: &str,
    ) -> anyhow::Result<()> {
        Self::reject_unbound_resident_graph_receipt(launch_path)?;
        if let Some(ref mut engine) = self.engine {
            engine.record_prismcode_md_native_graph_replay(start_step, launch_count, launch_path);
        }
        Ok(())
    }

    /// Record a nonresident graph replay window supplied by an existing legacy
    /// path. Resident graph labels are refused here because their receipt must
    /// be emitted by the bound executor after its own launch and device fence.
    pub(crate) fn record_prismcode_md_native_graph_replay_window(
        &mut self,
        start_step: u64,
        logical_steps: u64,
        launch_count: u32,
        steps_per_launch: u32,
        launch_path: &str,
    ) -> anyhow::Result<()> {
        Self::reject_unbound_resident_graph_receipt(launch_path)?;
        if let Some(ref mut engine) = self.engine {
            engine.record_prismcode_md_native_graph_replay_window(
                start_step,
                logical_steps,
                launch_count,
                steps_per_launch,
                launch_path,
            );
        }
        Ok(())
    }

    fn reject_unbound_resident_graph_receipt(launch_path: &str) -> anyhow::Result<()> {
        if launch_path == "graph.while_resident" || launch_path.contains("while_resident") {
            anyhow::bail!(
                "unbound resident graph receipt is forbidden; use execute_bound_resident_graph_replay_window or execute_bound_resident_graph_ladd_cadence_window"
            );
        }
        Ok(())
    }

    /// Launch, raw-stream-fence, device-clock-verify, and record one exact
    /// nonresident CUDA graph window.  The graph label and all receipt counts
    /// are engine-owned: callers cannot mint a graph receipt from a proposed
    /// host schedule or relabel a resident execution as a legacy replay.
    ///
    /// This is retained for the established nonresident graph capture path.
    /// Resident graphs must use one of the bound executor methods below so
    /// their optional native-housekeeping profile remains graph-specific.
    pub fn execute_nonresident_graph_replay_window(
        &mut self,
        graph: &crate::graph_capture::AutonomousGraph,
        logical_steps: u32,
    ) -> anyhow::Result<ResidentGraphReplayWindowReceipt> {
        self.autonomous_chunk_clock.invalidate();
        if graph.resident_loop_status().is_some() {
            anyhow::bail!(
                "nonresident graph executor refuses a WHILE-resident graph; use execute_bound_resident_graph_replay_window"
            );
        }
        if logical_steps == 0 {
            anyhow::bail!("nonresident graph replay requires logical_steps > 0");
        }
        let steps_per_launch = graph.steps_per_launch();
        if logical_steps % steps_per_launch != 0 {
            anyhow::bail!(
                "nonresident graph replay requires logical_steps {} to align with graph granularity {}",
                logical_steps,
                steps_per_launch
            );
        }
        let expected_launches = logical_steps / steps_per_launch;
        let start_device_step = self
            .device_protocol_current_step()
            .context("nonresident graph replay cannot read start device clock")?;
        let graph_launches = graph
            .run_chunk_exact_on_stream(logical_steps, self.stream.cu_stream())
            .context("nonresident graph launch failed")?;
        if graph_launches == 0 || graph_launches != expected_launches {
            anyhow::bail!(
                "nonresident graph replay launch mismatch: expected {}, observed {}",
                expected_launches,
                graph_launches
            );
        }
        self.synchronize_stream_raw("nonresident graph replay completion")?;
        let end_device_step = self
            .device_protocol_current_step()
            .context("nonresident graph replay cannot read end device clock")?;
        let expected_end = start_device_step
            .checked_add(logical_steps)
            .context("nonresident graph replay device-clock overflow")?;
        if end_device_step != expected_end {
            anyhow::bail!(
                "nonresident graph replay device-clock mismatch: expected {}, observed {}",
                expected_end,
                end_device_step
            );
        }
        self.engine
            .as_mut()
            .context("nonresident graph replay has no loaded fused engine")?
            .record_prismcode_md_native_graph_replay_window(
                u64::from(start_device_step),
                u64::from(logical_steps),
                graph_launches,
                steps_per_launch,
                "graph.run_chunk",
            );
        Ok(ResidentGraphReplayWindowReceipt {
            start_device_step,
            end_device_step,
            logical_steps,
            graph_launches,
            steps_per_launch,
        })
    }

    /// Establish the resident clock from this owner's synchronized device
    /// leaf after warmup/restore. Never accepts a caller-proposed start step.
    pub fn admit_autonomous_chunk_clock(&mut self) -> anyhow::Result<u32> {
        self.autonomous_chunk_clock.invalidate();
        let lease=self.physical_launch_domain.lease().map_err(anyhow::Error::msg)?;
        self.synchronize_stream_raw("autonomous clock admission")?;
        let observed = self.device_protocol_current_step()?;
        self.autonomous_chunk_clock.observe(observed);
        self.autonomous_chunk_observed_generation=Some(*lease);
        Ok(observed)
    }

    pub fn verified_autonomous_chunk_endpoint(&self)
        -> anyhow::Result<VerifiedAutonomousChunkEndpoint<'_>> {
        let lease=self.physical_launch_domain.lease().map_err(anyhow::Error::msg)?;
        crate::graph_capture::PhysicalGraphLaunchDomain::verify(*lease,self.autonomous_chunk_observed_generation)
            .map_err(anyhow::Error::msg)?;
        Ok(VerifiedAutonomousChunkEndpoint {
            _launch_lease:lease,
            native_owner: self.engine.as_ref().context("autonomous endpoint lost native owner")?,
            completed_step: self.autonomous_chunk_clock.endpoint().map_err(anyhow::Error::msg)?,
        })
    }

    /// One native WHILE launch, one synchronized 4-byte endpoint readback.
    /// Start authority is the previous observed endpoint on this exclusive
    /// owner. Alternate physical execution/mutable owner access invalidates it.
    /// A failed launch, fence or readback leaves the cache unusable.
    pub fn execute_autonomous_chunk_while(
        &mut self, bound_graph: &BoundResidentGraph, logical_steps: u32,
    ) -> anyhow::Result<ResidentGraphReplayWindowReceipt> {
        let graph = bound_graph.graph.as_ref();
        let (profile, structure) = self.resolve_resident_graph_replay_binding(bound_graph)?;
        let loop_status = graph.resident_loop_status().context("autonomous chunk lacks a WHILE body")?;
        anyhow::ensure!(loop_status.max_iterations == graph.steps_per_launch(),
            "autonomous chunk loop bound and physical body extent disagree");
        anyhow::ensure!(graph.contains_physical_launch_domain(&self.physical_launch_domain),
            "autonomous graph lost its physical launch domain");
        let expected_generation={
            let lease=self.physical_launch_domain.lease().map_err(anyhow::Error::msg)?;
            crate::graph_capture::PhysicalGraphLaunchDomain::verify(*lease,self.autonomous_chunk_observed_generation)
                .map_err(anyhow::Error::msg)?;
            lease.checked_add(1).context("autonomous graph launch generation overflow")?
        };
        let (start_device_step, expected_end) = self.autonomous_chunk_clock
            .begin(logical_steps, graph.steps_per_launch()).map_err(anyhow::Error::msg)?;
        let graph_launches = graph.run_chunk_exact_on_stream(logical_steps, self.stream.cu_stream())
            .context("autonomous chunk WHILE launch failed")?;
        anyhow::ensure!(graph_launches == 1, "autonomous chunk issued multiple host graph launches");
        // Exclude another escaped physical graph from the fence/readback until
        // the endpoint and its generation have been recorded together.
        let domain=self.physical_launch_domain.clone();
        let lease=domain.lease().map_err(anyhow::Error::msg)?;
        anyhow::ensure!(*lease==expected_generation,"another physical graph replay overlapped the autonomous chunk");
        self.synchronize_stream_raw("autonomous chunk WHILE completion")?;
        let end_device_step = self.device_protocol_current_step()
            .context("autonomous chunk endpoint readback failed")?;
        self.autonomous_chunk_clock.complete(expected_end, end_device_step).map_err(anyhow::Error::msg)?;
        self.autonomous_chunk_observed_generation=Some(*lease);
        let receipt = ResidentGraphReplayWindowReceipt {
            start_device_step, end_device_step, logical_steps, graph_launches,
            steps_per_launch: graph.steps_per_launch(),
        };
        if let Err(error) = self.record_bound_resident_graph_replay_window(
            profile, &structure, &receipt, "graph.while_resident") {
            self.autonomous_chunk_clock.invalidate();
            return Err(error);
        }
        Ok(receipt)
    }

    /// Exact existing journal retirements preserve the device protocol clock.
    /// Keep them inside the owner instead of exposing unrestricted mutable state.
    pub fn commit_conservative_constraint_window(&mut self, begin:u64, end:u64) -> Result<()> {
        self.engine.as_mut().context("constraint journal lost fused owner")?
            .commit_conservative_constraint_window(begin,end)
    }
    pub fn commit_ca_intervention_atom_window(&mut self, begin:u64, end:u64) -> Result<()> {
        self.engine.as_mut().context("CA work lost fused owner")?
            .commit_ca_intervention_atom_window(begin,end)
    }
    pub fn commit_nma_work_window(&mut self, begin:u64, end:u64) -> Result<()> {
        self.engine.as_mut().context("NMA journal lost fused owner")?.commit_nma_work_window(begin,end)
    }

    /// The historical ring kernel writes threshold/steering fields only. The
    /// algorithmic control tick remains distinct from the physical step leaf.
    pub fn read_may_ring_and_adapt(
        &mut self, ring:&mut crate::twin_kernels::TwinRingBuffer,
        grid_dims:(i32,i32,i32), grid_origin:(f32,f32,f32), voxel_size:f32,
        sensitivity_boost:f32, max_reduction:f32, algorithmic_tick:u32, decay_constant:f32,
    ) -> Result<()> {
        let native = self.engine.as_mut().context("May coupling recipient owner absent")?;
        let (thresholds,base,protocol) = native.coupling_buffers_for_twin()
            .context("May coupling recipient buffers absent")?;
        ring.read_and_adapt_with_steering(&self.stream,thresholds,base,
            grid_dims,grid_origin,voxel_size,sensitivity_boost,max_reduction,
            algorithmic_tick,decay_constant,Some(protocol))
    }

    /// Launch, raw-stream-fence, device-clock-verify, and record one exact
    /// bound resident graph window. Optional housekeeping opcodes can enter a
    /// receipt only through this engine-owned operation; callers cannot record
    /// a completed resident window without a real `cuGraphLaunch`.
    pub fn execute_bound_resident_graph_replay_window(
        &mut self,
        bound_graph: &BoundResidentGraph,
        logical_steps: u32,
        launch_path: &str,
    ) -> anyhow::Result<ResidentGraphReplayWindowReceipt> {
        self.autonomous_chunk_clock.invalidate();
        let graph = bound_graph.graph.as_ref();
        if logical_steps == 0 {
            anyhow::bail!("bound resident graph replay requires logical_steps > 0");
        }
        if logical_steps % graph.steps_per_launch() != 0 {
            anyhow::bail!(
                "bound resident graph replay requires logical_steps {} to align with graph granularity {}",
                logical_steps,
                graph.steps_per_launch()
            );
        }
        let (profile, structure) = self.resolve_resident_graph_replay_binding(bound_graph)?;
        let start_device_step = self
            .device_protocol_current_step()
            .context("bound resident graph replay cannot read start device clock")?;
        let graph_launches = graph
            .run_chunk_exact_on_stream(logical_steps, self.stream.cu_stream())
            .context("bound resident graph launch failed")?;
        if graph_launches == 0 {
            anyhow::bail!("bound resident graph replay issued zero launches");
        }
        self.synchronize_stream_raw("bound resident graph replay completion")?;
        let end_device_step = self
            .device_protocol_current_step()
            .context("bound resident graph replay cannot read end device clock")?;
        let expected_end = start_device_step
            .checked_add(logical_steps)
            .context("bound resident graph replay device-clock overflow")?;
        if end_device_step != expected_end {
            anyhow::bail!(
                "bound resident graph replay device-clock mismatch: expected {}, observed {}",
                expected_end,
                end_device_step
            );
        }
        let receipt = ResidentGraphReplayWindowReceipt {
            start_device_step,
            end_device_step,
            logical_steps,
            graph_launches,
            steps_per_launch: graph.steps_per_launch(),
        };
        self.record_bound_resident_graph_replay_window(profile, &structure, &receipt, launch_path)?;
        Ok(receipt)
    }

    /// Execute an exact cadence window as bound resident CUDA graphs followed
    /// by native LADD on the same persistent stream. The method owns the full
    /// graph → raw fence → native LADD → raw fence → receipt sequence, so an
    /// ordinary same-engine MD run cannot mint a resident-graph LADD record.
    pub fn execute_bound_resident_graph_ladd_cadence_window(
        &mut self,
        segments: &[ResidentGraphReplaySegment<'_>],
        cadence_physical_steps: u32,
        launch_path: &str,
    ) -> anyhow::Result<ResidentGraphLaddCadenceWindowReceipt> {
        self.autonomous_chunk_clock.invalidate();
        if cadence_physical_steps == 0 {
            anyhow::bail!("bound resident LADD cadence requires cadence_physical_steps > 0");
        }
        if segments.is_empty() {
            anyhow::bail!("bound resident LADD cadence requires at least one graph segment");
        }

        let mut prepared = Vec::with_capacity(segments.len());
        let mut total_logical_steps = 0u32;
        for segment in segments {
            if segment.logical_steps == 0 {
                anyhow::bail!("bound resident LADD cadence rejects an empty graph segment");
            }
            let graph = segment.bound_graph.graph.as_ref();
            if segment.logical_steps % graph.steps_per_launch() != 0 {
                anyhow::bail!(
                    "bound resident LADD cadence segment of {} steps is not aligned to graph granularity {}",
                    segment.logical_steps,
                    graph.steps_per_launch()
                );
            }
            let (profile, structure) =
                self.resolve_resident_graph_replay_binding(segment.bound_graph)?;
            total_logical_steps = total_logical_steps
                .checked_add(segment.logical_steps)
                .context("bound resident LADD cadence logical-step total overflow")?;
            prepared.push((graph, segment.logical_steps, profile, structure));
        }
        if total_logical_steps != cadence_physical_steps {
            anyhow::bail!(
                "bound resident LADD cadence requires exactly {} physical graph steps, but segments cover {}",
                cadence_physical_steps,
                total_logical_steps
            );
        }

        let start_device_step = self
            .device_protocol_current_step()
            .context("bound resident LADD cadence cannot read start device clock")?;
        let raw_stream = self.stream.cu_stream();
        let window_result = (|| -> anyhow::Result<ResidentGraphLaddCadenceWindowReceipt> {
            let boundary = self
                .engine
                .as_mut()
                .context("bound resident LADD cadence has no loaded fused engine")?
                .begin_ladd_cadence_boundary(start_device_step, cadence_physical_steps)?;

            let mut next_device_step = start_device_step;
            let mut completed = Vec::with_capacity(prepared.len());
            for (graph, logical_steps, profile, structure) in &prepared {
                let graph_launches = graph
                    .run_chunk_exact_on_stream(*logical_steps, raw_stream)
                    .context("bound resident LADD cadence graph launch failed")?;
                if graph_launches == 0 {
                    anyhow::bail!("bound resident LADD cadence issued zero graph launches");
                }
                let end_device_step = next_device_step
                    .checked_add(*logical_steps)
                    .context("bound resident LADD cadence device-clock overflow")?;
                completed.push((
                    *profile,
                    structure.clone(),
                    ResidentGraphReplayWindowReceipt {
                        start_device_step: next_device_step,
                        end_device_step,
                        logical_steps: *logical_steps,
                        graph_launches,
                        steps_per_launch: graph.steps_per_launch(),
                    },
                ));
                next_device_step = end_device_step;
            }

            self.synchronize_stream_raw("bound resident LADD cadence graph completion")?;
            let observed_after_graph = self
                .device_protocol_current_step()
                .context("bound resident LADD cadence cannot read post-graph device clock")?;
            if observed_after_graph != next_device_step {
                anyhow::bail!(
                    "bound resident LADD cadence graph device-clock mismatch: expected {}, observed {}",
                    next_device_step,
                    observed_after_graph
                );
            }

            let dispatch = self
                .engine
                .as_mut()
                .context("bound resident LADD cadence engine disappeared before dispatch")?
                .enqueue_ladd_cadence_dispatch(boundary)?;
            self.synchronize_stream_raw("bound resident LADD cadence native LADD completion")?;
            self.engine
                .as_mut()
                .context("bound resident LADD cadence engine disappeared before receipt")?
                .commit_ladd_cadence_dispatch_after_sync(dispatch)?;

            let end_device_step = self
                .device_protocol_current_step()
                .context("bound resident LADD cadence cannot read final device clock")?;
            if end_device_step != next_device_step {
                anyhow::bail!(
                    "bound resident LADD cadence post-dispatch device clock changed: expected {}, observed {}",
                    next_device_step,
                    end_device_step
                );
            }

            let mut graph_windows = Vec::with_capacity(completed.len());
            for (profile, structure, receipt) in completed {
                self.record_bound_resident_graph_replay_window(
                    profile,
                    &structure,
                    &receipt,
                    launch_path,
                )?;
                graph_windows.push(receipt);
            }
            Ok(ResidentGraphLaddCadenceWindowReceipt {
                start_device_step,
                end_device_step,
                cadence_physical_steps,
                graph_windows,
            })
        })();
        if window_result.is_err() {
            if let Some(engine) = self.engine.as_mut() {
                engine.poison_ladd_cadence_boundary(
                    "bound resident graph LADD cadence did not reach a verified completion",
                );
            }
        }
        window_result
    }

    fn resolve_resident_graph_replay_binding(
        &self,
        bound_graph: &BoundResidentGraph,
    ) -> anyhow::Result<(ResidentGraphBindingProfile, String)> {
        self.resolve_resident_graph_replay_binding_parts(bound_graph.graph.as_ref(),&bound_graph.binding,false)
    }

    fn resolve_resident_graph_replay_binding_parts(&self,graph:&crate::graph_capture::AutonomousGraph,
        binding:&ResidentGraphReplayBinding,shared:bool)->anyhow::Result<(ResidentGraphBindingProfile,String)> {
        if binding.domain != self.resident_graph_replay_binding_domain {
            anyhow::bail!("resident graph replay binding belongs to a different persistent engine");
        }
        let active = self
            .resident_graph_replay_bindings
            .get(&binding.capture_nonce)
            .context("resident graph replay has no binding for this capture nonce")?;
        if active.capture_nonce != binding.capture_nonce
            || active.resident_structure_fingerprint_sha256
                != binding.resident_structure_fingerprint_sha256
            || active.graph_template_handle != binding.graph_template_handle
            || active.graph_exec_handle != binding.graph_exec_handle
        {
            anyhow::bail!(
                "resident graph replay binding does not match the active post-instantiate graph binding"
            );
        }
        let observed_structure = if shared {graph.captured_structure_fingerprint_sha256()?}
            else {graph.resident_structure_fingerprint_sha256()?};
        let (observed_template_handle, observed_exec_handle) =
            Self::resident_graph_driver_handle_identity(graph)?;
        let observed_graph_object_address = graph as *const _ as usize;
        if observed_structure != active.resident_structure_fingerprint_sha256
            || observed_graph_object_address != active.graph_object_address
            || observed_template_handle != active.graph_template_handle
            || observed_exec_handle != active.graph_exec_handle
        {
            anyhow::bail!(
                "resident graph replay identity mismatch: a same-shape or stale graph cannot substitute for the bound driver graph instance"
            );
        }
        Ok((active.profile, observed_structure))
    }

    fn resident_graph_driver_handle_identity(
        graph: &crate::graph_capture::AutonomousGraph,
    ) -> anyhow::Result<(usize, usize)> {
        let template = graph
            .cu_graph_template()
            .context("resident graph binding requires a raw CUDA graph template")?;
        let exec = graph
            .cu_graph_exec_raw()
            .context("resident graph binding requires a raw CUDA graph executable")?;
        if template.is_null() || exec.is_null() {
            anyhow::bail!("resident graph binding encountered a null CUDA graph handle");
        }
        Ok((template as usize, exec as usize))
    }

    fn record_bound_resident_graph_replay_window(
        &mut self,
        profile: ResidentGraphBindingProfile,
        structure: &str,
        receipt: &ResidentGraphReplayWindowReceipt,
        launch_path: &str,
    ) -> anyhow::Result<()> {
        self.engine
            .as_mut()
            .context("resident graph replay has no loaded fused engine")?
            .record_prismcode_md_native_graph_replay_window_with_resident_binding(
                receipt.start_device_step as u64,
                receipt.logical_steps as u64,
                receipt.graph_launches,
                receipt.steps_per_launch,
                launch_path,
                Some(profile),
                Some(structure),
            );
        Ok(())
    }

    fn clear_resident_graph_replay_bindings(&mut self) {
        self.autonomous_chunk_clock.invalidate();
        self.resident_graph_replay_bindings.clear();
    }

    fn issue_resident_graph_replay_binding(
        &mut self,
        graph: &crate::graph_capture::AutonomousGraph,
        profile: ResidentGraphBindingProfile,
    ) -> anyhow::Result<ResidentGraphReplayBinding> {
        self.issue_resident_graph_replay_binding_parts(graph,profile,false)
    }

    fn issue_resident_graph_replay_binding_parts(&mut self,graph:&crate::graph_capture::AutonomousGraph,
        profile:ResidentGraphBindingProfile,shared:bool)->anyhow::Result<ResidentGraphReplayBinding> {
        self.engine
            .as_ref()
            .context("resident graph binding has no loaded fused engine")?
            .validate_resident_graph_binding_profile(profile)
            .context("validate native resident housekeeping graph profile")?;
        let resident_structure_fingerprint_sha256 = if shared {graph.captured_structure_fingerprint_sha256()?}
            else {graph.resident_structure_fingerprint_sha256()?};
        let (graph_template_handle, graph_exec_handle) =
            Self::resident_graph_driver_handle_identity(graph)?;
        let graph_object_address = graph as *const _ as usize;
        let capture_nonce = self
            .resident_graph_capture_nonce
            .checked_add(1)
            .context("resident graph replay-binding nonce exhausted")?;
        if self
            .resident_graph_replay_bindings
            .contains_key(&capture_nonce)
        {
            anyhow::bail!("resident graph replay-binding nonce collision");
        }
        self.resident_graph_capture_nonce = capture_nonce;
        self.resident_graph_replay_bindings.insert(
            capture_nonce,
            ResidentGraphReplayBindingState {
                capture_nonce,
                resident_structure_fingerprint_sha256: resident_structure_fingerprint_sha256
                    .clone(),
                graph_object_address,
                graph_template_handle,
                graph_exec_handle,
                profile,
            },
        );
        Ok(ResidentGraphReplayBinding {
            domain: self.resident_graph_replay_binding_domain,
            capture_nonce,
            resident_structure_fingerprint_sha256,
            graph_template_handle,
            graph_exec_handle,
        })
    }

    pub fn record_prismcode_md_native_graph_capture(&mut self, capture_path: &str) {
        if let Some(ref mut engine) = self.engine {
            engine.record_prismcode_md_native_graph_capture(capture_path);
        }
    }

    /// Load NMA modes from JSON and upload to GPU for NMA-biased perturbation.
    pub fn load_nma_modes(&mut self, path: &str) -> Result<()> {
        if let Some(ref mut engine) = self.engine {
            engine.load_nma_modes(path)
        } else {
            bail!("No topology loaded")
        }
    }

    /// Set NMA amplification factor.
    pub fn set_nma_amplification(&mut self, amp: f32) -> Result<()> {
        self.engine.as_mut().context("NMA parameter publication requires loaded topology")?
            .set_nma_amplification(amp)
    }

    /// Set NMA scan fraction.
    pub fn set_nma_scan_fraction(&mut self, frac: f32) -> Result<()> {
        self.engine.as_mut().context("NMA parameter publication requires loaded topology")?
            .set_nma_scan_fraction(frac)
    }

    /// Get the current spike count from the device.
    pub fn get_spike_count(&self) -> Result<u32> {
        if let Some(ref engine) = self.engine {
            engine.get_spike_count()
        } else {
            Ok(0)
        }
    }

    /// Get the number of accumulated spikes without cloning the vector.
    pub fn accumulated_spike_count(&self) -> usize {
        if let Some(ref engine) = self.engine {
            engine.get_accumulated_spikes().len()
        } else {
            0
        }
    }

    // ── PRISM-TWIN threshold access (Step 2) ──

    /// Get mutable reference to per-voxel neuron threshold buffer.
    pub fn threshold_buffer_mut(&mut self) -> Option<&mut CudaSlice<f32>> {
        self.engine.as_mut().map(|e| e.threshold_buffer_mut())
    }

    /// Get reference to base (initial) threshold buffer.
    pub fn base_threshold_buffer(&self) -> Option<&CudaSlice<f32>> {
        self.engine.as_ref().map(|e| e.base_threshold_buffer())
    }

    /// Get both threshold buffers together (avoids double-borrow).
    /// Returns (mutable_threshold, base_threshold) if engine is loaded.
    pub fn threshold_buffers_mut(&mut self) -> Option<(&mut CudaSlice<f32>, &CudaSlice<f32>)> {
        self.engine.as_mut().map(|e| e.threshold_buffers_mut())
    }

    /// Get grid geometry (dim_x, dim_y, dim_z, origin_x, origin_y, origin_z, spacing).
    pub fn grid_info(&self) -> Option<(i32, i32, i32, f32, f32, f32, f32)> {
        self.engine.as_ref().map(|e| e.grid_info())
    }

    /// Total voxels (dim³).
    pub fn total_voxels(&self) -> usize {
        self.engine.as_ref().map(|e| e.total_voxels()).unwrap_or(0)
    }

    /// Get reference to raw spike buffer on GPU.
    pub fn spike_buffer_gpu(&self) -> Option<&CudaSlice<u8>> {
        self.engine.as_ref().map(|e| e.spike_buffer_gpu())
    }

    /// Get reference to spike count buffer on GPU (for persistent coupling kernel).
    pub fn spike_count_gpu(&self) -> Option<&CudaSlice<i32>> {
        self.engine.as_ref().map(|e| e.spike_count_gpu())
    }

    /// Get a reference to the engine's CUDA stream.
    /// Used by CUDA Graph stream capture to record kernel launches.
    pub fn cuda_stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// T6: raw device pointer to the `d_forces` buffer (n_atoms × 3 f32, AoS).
    /// Returns 0 if no topology is loaded. Pointer-stable for the campaign.
    pub fn d_forces_dev_ptr(&self) -> u64 {
        self.engine
            .as_ref()
            .map(|e| e.d_forces_dev_ptr(&self.stream))
            .unwrap_or(0)
    }

    /// **M1.2.20.C-A / Ruling 6** — raw device pointer to the per-atom
    /// AMBER mass buffer (`f32 [n_atoms]`).  Returns 0 if no topology
    /// is loaded.  Pointer-stable for the campaign.  Consumed by the
    /// gradient gasp kernel and the Phase-3 Momentum Guard.
    pub fn d_masses_dev_ptr(&self) -> u64 {
        self.engine
            .as_ref()
            .map(|e| e.d_masses_dev_ptr(&self.stream))
            .unwrap_or(0)
    }

    /// T6: raw device pointer to the `d_positions` buffer (n_atoms × 3 f32, AoS).
    /// Returns 0 if no topology is loaded. Pointer-stable for the campaign.
    pub fn d_positions_dev_ptr(&self) -> u64 {
        self.engine
            .as_ref()
            .map(|e| e.d_positions_dev_ptr(&self.stream))
            .unwrap_or(0)
    }

    /// Number of atoms in the currently loaded topology. Returns 0 if none.
    pub fn n_atoms_loaded(&self) -> usize {
        self.engine.as_ref().map(|e| e.n_atoms()).unwrap_or(0)
    }

    /// Number of aromatics in the currently loaded topology. Returns 0
    /// if no topology has been loaded.
    pub fn n_aromatics_loaded(&self) -> usize {
        self.engine.as_ref().map(|e| e.n_aromatics()).unwrap_or(0)
    }

    // ─────────────────────────────────────────────────────────────────────
    // Phase A teardown DtoH forwarders — read-only device-pointer + size
    // accessors for the four VRAM-resident outputs flagged by the audit
    // (d_warp_matrix, d_forces final, d_aromatic_centroids final,
    // ProtocolState).  Used by nhs_rt_full.rs's per-stream cleanup block
    // to issue cuMemcpyDtoH before stream teardown.
    // ─────────────────────────────────────────────────────────────────────

    /// Raw device pointer to `d_warp_matrix`. Returns 0 if no topology.
    pub fn d_warp_matrix_dev_ptr(&self) -> u64 {
        self.engine
            .as_ref()
            .map(|e| e.d_warp_matrix_dev_ptr(&self.stream))
            .unwrap_or(0)
    }

    /// Total byte size of `d_warp_matrix`.
    pub fn d_warp_matrix_n_bytes(&self) -> usize {
        self.engine
            .as_ref()
            .map(|e| e.d_warp_matrix_n_bytes())
            .unwrap_or(0)
    }

    /// Raw device pointer to `d_aromatic_centroids`. Returns 0 if no
    /// topology.
    pub fn d_aromatic_centroids_dev_ptr(&self) -> u64 {
        self.engine
            .as_ref()
            .map(|e| e.d_aromatic_centroids_dev_ptr(&self.stream))
            .unwrap_or(0)
    }

    /// Number of f32 elements in `d_aromatic_centroids` (n_aromatics×3).
    pub fn d_aromatic_centroids_n_floats(&self) -> usize {
        self.engine
            .as_ref()
            .map(|e| e.d_aromatic_centroids_n_floats())
            .unwrap_or(0)
    }

    // ── M1.2.18-P1 — Forwarders for the V2 captured-pipeline build ──
    //
    // The five methods below are simple `engine.as_ref()/as_mut()` forwarders
    // that bridge the production `nhs_rt_full --features=v2_ignition` build
    // (which uses `PersistentNhsEngine`) to the inner `NhsAmberFusedEngine`
    // accessors added in the B.3 / M1.2.17 waves.  Without these, the
    // captured-pipeline wire-up at nhs_rt_full.rs:5010+ fails to link.

    /// **B.3** raw device pointer to the `d_velocities` buffer.
    /// Forwards to inner `NhsAmberFusedEngine::d_velocities_dev_ptr`.
    pub fn d_velocities_dev_ptr(&self) -> u64 {
        self.engine
            .as_ref()
            .map(|e| e.d_velocities_dev_ptr(&self.stream))
            .unwrap_or(0)
    }

    /// **B.3** raw device pointer to `d_protocol->dt` within the
    /// GPU-resident ProtocolState.  Forwarder.
    pub fn d_protocol_dt_dev_ptr(&self) -> u64 {
        self.engine
            .as_ref()
            .map(|e| e.d_protocol_dt_dev_ptr(&self.stream))
            .unwrap_or(0)
    }

    /// Device clock owned by this engine; zero means no initialized engine.
    pub fn d_protocol_current_step_dev_ptr(&self) -> u64 {
        self.engine.as_ref()
            .map(|engine| engine.d_protocol_current_step_dev_ptr(&self.stream))
            .unwrap_or(0)
    }


    /// **M1.2.17** raw device pointer to the per-atom potential-energy
    /// components buffer (n_atoms × f64).  CUB DeviceReduce::Sum reads
    /// from this address.  Forwarder.
    pub fn d_potential_energy_components_dev_ptr(&self) -> u64 {
        self.engine
            .as_ref()
            .map(|e| e.d_potential_energy_components_dev_ptr(&self.stream))
            .unwrap_or(0)
    }

    /// **M1.2.17** atom count for the per-atom PE buffer (= length of
    /// d_potential_energy_components / sizeof(f64)).  Forwarder.
    pub fn d_potential_energy_n_atoms(&self) -> usize {
        self.engine
            .as_ref()
            .map(|e| e.d_potential_energy_n_atoms())
            .unwrap_or(0)
    }

    /// **M1.2.18.5** raw device pointer to the 1 × f64 W_ext accumulator.
    /// The captured pipeline writes this address into the FFI struct's
    /// `d_external_work` field at offset 128 pre-capture and emits a
    /// cuMemsetD8Async at the head of every replay window so each chunk
    /// starts with `*d_external_work == 0.0`.  Forwarder to inner
    /// `NhsAmberFusedEngine::allocate_external_work_buffer`.
    pub fn allocate_external_work_buffer(&self) -> u64 {
        self.engine
            .as_ref()
            .map(|e| e.allocate_external_work_buffer(&self.stream))
            .unwrap_or(0)
    }

    /// **B.3.1** Gearbox > Adaptive-DT hierarchy gate.  When the V2
    /// captured pipeline is live the legacy `--adaptive-dt` host write
    /// path is bypassed.  Forwarder.
    pub fn set_gearbox_active(&mut self, active: bool) {
        if let Some(ref mut e) = self.engine {
            e.set_gearbox_active(active);
        }
    }

    /// Get ALL GPU buffer references needed by the persistent coupling kernel.
    pub fn twin_coupling_gpu_state(
        &mut self,
    ) -> Option<(
        &mut CudaSlice<f32>,
        &CudaSlice<f32>,
        &CudaSlice<u8>,
        &CudaSlice<i32>,
    )> {
        self.autonomous_chunk_clock.invalidate();
        self.engine.as_mut().map(|e| e.twin_coupling_gpu_state())
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PRISM-TWIN v3.0 Gate 3: Autonomous GPU execution delegation
    // ═══════════════════════════════════════════════════════════════════════

    /// Launch one complete step's GPU kernels with zero CPU memcpy.
    /// Director → Physics → CA restraints → multi_lif → heartbeat → coupling clear.
    /// This is the graph-capturable step function.
    pub fn step_autonomous_kernels(&mut self) -> Result<()> {
        self.autonomous_chunk_clock.invalidate();
        let stream = self.stream.clone();
        if let Some(ref mut engine) = self.engine {
            engine.step_autonomous_kernels(&stream)
        } else {
            bail!("No topology loaded — call load_topology() first")
        }
    }

    /// Bind runtime-inline lambda/no-probe evidence to the exact admitted
    /// protocol phase before executing a resident window that starts at step.
    pub fn sync_lambda_runtime_phase(&mut self, step: u64) -> Result<()> {
        if let Some(ref mut engine) = self.engine {
            engine.sync_lambda_runtime_phase(step)
        } else {
            bail!("No topology loaded — call load_topology() first")
        }
    }

    /// Explicit terminal lambda/no-probe closeout; never called from Drop.
    pub fn close_lambda_runtime(&mut self, completed_end_step_exclusive: u64) -> Result<()> {
        if let Some(ref mut engine) = self.engine {
            engine.close_lambda_runtime(completed_end_step_exclusive)
        } else {
            bail!("No topology loaded — call load_topology() first")
        }
    }

    /// Get reference to GPU-resident ProtocolState (684 bytes after Stage 2).
    /// Used for heartbeat polling and graph capture.
    pub fn protocol_state_buffer(&self) -> Option<&CudaSlice<u8>> {
        self.engine.as_ref().map(|e| &e.d_protocol_state)
    }

    /// Mutable protocol state buffer for GPU-side ASC steering writes.
    pub fn protocol_state_buffer_mut(&mut self) -> Option<&mut CudaSlice<u8>> {
        self.autonomous_chunk_clock.invalidate();
        self.engine.as_mut().map(|e| &mut e.d_protocol_state)
    }

    /// Get BOTH threshold buffers AND protocol state in one borrow — for GPU ring buffer coupling.
    /// Returns (osc_thresholds, base_thresholds, protocol_state) all as mutable.
    pub fn coupling_buffers_mut(
        &mut self,
    ) -> Option<(&mut CudaSlice<f32>, &CudaSlice<f32>, &mut CudaSlice<u8>)> {
        self.autonomous_chunk_clock.invalidate();
        if let Some(ref mut engine) = self.engine {
            engine.coupling_buffers_for_twin()
        } else {
            None
        }
    }

    /// Capture the autonomous physics step as a CUDA Graph for replay.
    /// Pre-synchronizes the stream to flush all prior work (module loads, memcpy)
    /// before entering capture mode — this prevents CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED.
    /// Amendment 3.4 monolithic-fusion variant. Captures the autonomous
    /// kernel sequence into a raw `CUgraph` template (NOT instantiated)
    /// with named node handles for "director", "physical_step_complete", "multi_lif"
    /// recorded by a `StreamCaptureTagger`. The caller can splice in child
    /// graphs via `CapturedTemplate::add_child_graph_node` before
    /// instantiating with `template.instantiate()`.
    ///
    /// Bypasses `cudarc::driver::safe::CudaGraph` entirely; uses raw
    /// `cuStreamBeginCapture_v2` / `cuStreamEndCapture` from
    /// `cudarc::driver::sys` so the template handle stays alive
    /// post-capture.
    pub fn capture_autonomous_template(
        &mut self,
    ) -> anyhow::Result<crate::graph_capture::CapturedTemplate> {
        self.capture_raw_resident_template(RawResidentTemplateBody::AutonomousMd)
    }

    /// Capture the dedicated intervention-free gamma-zero NCMC integrator
    /// body. It cannot fall back to the ordinary autonomous/Langevin body.
    fn capture_ncmc_reversible_template(
        &mut self,
    ) -> anyhow::Result<crate::graph_capture::CapturedTemplate> {
        self.capture_raw_resident_template(RawResidentTemplateBody::NcmcReversible)
    }

    /// Capture the native CA-restraint launch by itself so it can be spliced
    /// after the autonomous physical step without re-capturing across PTX
    /// module boundaries.
    fn capture_resident_ca_restraint_child_template(
        &mut self,
    ) -> anyhow::Result<crate::graph_capture::CapturedTemplate> {
        self.capture_raw_resident_template(RawResidentTemplateBody::CaRestraintChild)
    }

    /// Capture native coupling clear followed by heartbeat as an ordered,
    /// pure child template.  The helper rejects the old CPU memset fallback.
    fn capture_resident_coupling_clear_heartbeat_child_template(
        &mut self,
    ) -> anyhow::Result<crate::graph_capture::CapturedTemplate> {
        self.capture_raw_resident_template(RawResidentTemplateBody::CouplingClearHeartbeatChild)
    }

    fn capture_raw_resident_template(
        &mut self,
        body: RawResidentTemplateBody,
    ) -> anyhow::Result<crate::graph_capture::CapturedTemplate> {
        use crate::graph_capture::{CapturedTemplate, StreamCaptureTagger};
        use cudarc::driver::sys;
        let stream = self.stream.clone();

        // Same pre-capture posture as the legacy path, but use raw CUDA
        // context ownership here. The focused TIER 8 gate showed
        // ctx.bind_to_thread() can report CUDA_ERROR_INVALID_VALUE before
        // the monolithic parent template capture; keep this replacement
        // scoped to that proven failing gate.
        let ctx = stream.context();
        let raw_stream = stream.cu_stream();
        let stream_slot = tier8_capture_stream_slot(raw_stream as usize);
        let expected_ctx = ctx.cu_ctx();
        let diag_thread_id = format!("{:?}", std::thread::current().id());
        let mut cuda_error_drain_context: Option<crate::fused_engine::Tier8CudaErrorDrainContext> =
            None;

        // Cudarc launch_builder() still calls bind_to_thread() internally.
        // In capture mode that call skips cuCtxGetCurrent/cuCtxSetCurrent,
        // but it still checks and clears the wrapper's CUDA error state.
        // Drain exactly one CUDA error state here, with evidence, so the raw
        // context guard remains the context authority for this gate.
        match ctx.check_err() {
            Ok(()) => {
                tier8_capture_verbose!(
                    "[TIER8-CAPTURE cuda-error-drain] call=ctx.check_err \
                     result=OK stream_id={} protocol_group=monolithic_parent_template \
                     raw_stream={:p} expected_ctx={:p} thread_id={}",
                    stream_slot,
                    raw_stream,
                    expected_ctx,
                    diag_thread_id
                );
            }
            Err(first) => {
                let drain_count = TIER8_CUDA_ERROR_DRAIN_COUNT
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                    + 1;
                let (name, text) = tier8_capture_driver_error_text(first.0);
                log::warn!(
                    "[TIER8-CAPTURE cuda-error-drain] call=ctx.check_err \
                     result=DRAINED stream_id={} protocol_group=monolithic_parent_template \
                     rc={} cuda_name={} cuda_string=\"{}\" raw_stream={:p} \
                     expected_ctx={:p} thread_id={} cuda_error_drain_count={} \
                     call_site=capture_autonomous_template::pre_capture_ctx_check",
                    stream_slot,
                    first.0 as i32,
                    name,
                    text,
                    raw_stream,
                    expected_ctx,
                    diag_thread_id,
                    drain_count
                );
                cuda_error_drain_context = Some(crate::fused_engine::Tier8CudaErrorDrainContext {
                    stream_slot,
                    raw_stream: raw_stream as usize,
                    call_site: "capture_autonomous_template::pre_capture_ctx_check",
                    drain_count,
                    rc: first.0 as i32,
                    cuda_name: name.clone(),
                    cuda_string: text.clone(),
                });
                if let Err(second) = ctx.check_err() {
                    let (second_name, second_text) = tier8_capture_driver_error_text(second.0);
                    log::error!(
                        "[TIER8-CAPTURE cuda-error-drain] call=ctx.check_err(second) \
                         result=FAIL stream_id={} protocol_group=monolithic_parent_template \
                         rc={} cuda_name={} cuda_string=\"{}\" raw_stream={:p} \
                         expected_ctx={:p} thread_id={}",
                        stream_slot,
                        second.0 as i32,
                        second_name,
                        second_text,
                        raw_stream,
                        expected_ctx,
                        diag_thread_id
                    );
                    anyhow::bail!(
                        "TIER8 pre-capture cudarc CUDA error persisted: first={:?}, second={:?}",
                        first,
                        second
                    );
                }
            }
        }

        let mut before_ctx: sys::CUcontext = std::ptr::null_mut();
        let rc_get_before = unsafe { sys::cuCtxGetCurrent(&mut before_ctx as *mut _) };
        let (get_before_name, get_before_text) = tier8_capture_driver_error_text(rc_get_before);
        if !matches!(rc_get_before, sys::CUresult::CUDA_SUCCESS) {
            log::error!(
                "[TIER8-CAPTURE raw-context-guard] call=cuCtxGetCurrent(before) \
                 result=FAIL stream_id=unavailable protocol_group=monolithic_parent_template \
                 rc={} cuda_name={} cuda_string=\"{}\" expected_ctx={:p} \
                 before_ctx={:p} raw_stream={:p} thread_id={}",
                rc_get_before as i32,
                get_before_name,
                get_before_text,
                expected_ctx,
                before_ctx,
                raw_stream,
                diag_thread_id
            );
            anyhow::bail!(
                "TIER8 pre-capture cuCtxGetCurrent(before) failed: rc={} expected={:p} before={:p}",
                rc_get_before as i32,
                expected_ctx,
                before_ctx
            );
        }

        let mut did_set_current = false;
        let mut rc_set_current = sys::CUresult::CUDA_SUCCESS;
        let mut set_current_name = "CUDA_SUCCESS".to_string();
        let mut set_current_text = "no error".to_string();
        if before_ctx.is_null() || before_ctx != expected_ctx {
            did_set_current = true;
            rc_set_current = unsafe { sys::cuCtxSetCurrent(expected_ctx) };
            let (name, text) = tier8_capture_driver_error_text(rc_set_current);
            set_current_name = name;
            set_current_text = text;
            if !matches!(rc_set_current, sys::CUresult::CUDA_SUCCESS) {
                log::error!(
                    "[TIER8-CAPTURE raw-context-guard] call=cuCtxSetCurrent \
                     result=FAIL stream_id=unavailable protocol_group=monolithic_parent_template \
                     rc={} cuda_name={} cuda_string=\"{}\" expected_ctx={:p} \
                     before_ctx={:p} raw_stream={:p} thread_id={}",
                    rc_set_current as i32,
                    set_current_name,
                    set_current_text,
                    expected_ctx,
                    before_ctx,
                    raw_stream,
                    diag_thread_id
                );
                anyhow::bail!(
                    "TIER8 pre-capture cuCtxSetCurrent failed: rc={} expected={:p} before={:p}",
                    rc_set_current as i32,
                    expected_ctx,
                    before_ctx
                );
            }
        }

        let mut after_ctx: sys::CUcontext = std::ptr::null_mut();
        let rc_get_after = unsafe { sys::cuCtxGetCurrent(&mut after_ctx as *mut _) };
        let (get_after_name, get_after_text) = tier8_capture_driver_error_text(rc_get_after);
        if !matches!(rc_get_after, sys::CUresult::CUDA_SUCCESS) {
            log::error!(
                "[TIER8-CAPTURE raw-context-guard] call=cuCtxGetCurrent(after) \
                 result=FAIL stream_id=unavailable protocol_group=monolithic_parent_template \
                 rc={} cuda_name={} cuda_string=\"{}\" expected_ctx={:p} \
                 before_ctx={:p} after_ctx={:p} did_set_current={} \
                 set_rc={} set_name={} set_string=\"{}\" raw_stream={:p} \
                 thread_id={}",
                rc_get_after as i32,
                get_after_name,
                get_after_text,
                expected_ctx,
                before_ctx,
                after_ctx,
                did_set_current,
                rc_set_current as i32,
                set_current_name,
                set_current_text,
                raw_stream,
                diag_thread_id
            );
            anyhow::bail!(
                "TIER8 pre-capture cuCtxGetCurrent(after) failed: rc={} expected={:p} before={:p} after={:p}",
                rc_get_after as i32,
                expected_ctx,
                before_ctx,
                after_ctx
            );
        }

        if after_ctx != expected_ctx {
            log::error!(
                "[TIER8-CAPTURE raw-context-guard] call=raw_context_guard \
                 result=FAIL stream_id=unavailable protocol_group=monolithic_parent_template \
                 reason=context_mismatch expected_ctx={:p} before_ctx={:p} \
                 after_ctx={:p} did_set_current={} get_before_rc={} \
                 get_before_name={} get_before_string=\"{}\" set_rc={} \
                 set_name={} set_string=\"{}\" get_after_rc={} \
                 get_after_name={} get_after_string=\"{}\" raw_stream={:p} \
                 thread_id={}",
                expected_ctx,
                before_ctx,
                after_ctx,
                did_set_current,
                rc_get_before as i32,
                get_before_name,
                get_before_text,
                rc_set_current as i32,
                set_current_name,
                set_current_text,
                rc_get_after as i32,
                get_after_name,
                get_after_text,
                raw_stream,
                diag_thread_id
            );
            anyhow::bail!(
                "TIER8 pre-capture context mismatch: expected={:p} before={:p} after={:p}",
                expected_ctx,
                before_ctx,
                after_ctx
            );
        }
        tier8_capture_verbose!(
            "[TIER8-CAPTURE raw-context-guard] call=raw_context_guard \
             result=OK stream_id=unavailable protocol_group=monolithic_parent_template \
             expected_ctx={:p} before_ctx={:p} after_ctx={:p} did_set_current={} \
             get_before_rc={} get_before_name={} get_before_string=\"{}\" \
             set_rc={} set_name={} set_string=\"{}\" get_after_rc={} \
             get_after_name={} get_after_string=\"{}\" raw_stream={:p} thread_id={}",
            expected_ctx,
            before_ctx,
            after_ctx,
            did_set_current,
            rc_get_before as i32,
            get_before_name,
            get_before_text,
            rc_set_current as i32,
            set_current_name,
            set_current_text,
            rc_get_after as i32,
            get_after_name,
            get_after_text,
            raw_stream,
            diag_thread_id
        );

        let rc_sync = unsafe { sys::cuStreamSynchronize(raw_stream) };
        if !matches!(rc_sync, sys::CUresult::CUDA_SUCCESS) {
            let (sync_name, sync_text) = tier8_capture_driver_error_text(rc_sync);
            let mut current_ctx: sys::CUcontext = std::ptr::null_mut();
            let rc_get_current = unsafe { sys::cuCtxGetCurrent(&mut current_ctx as *mut _) };
            log::error!(
                "[TIER8-CAPTURE raw-context-guard] call=cuStreamSynchronize \
                 result=FAIL stream_id=unavailable protocol_group=monolithic_parent_template \
                 rc={} cuda_name={} cuda_string=\"{}\" raw_stream={:p} \
                 expected_ctx={:p} before_ctx={:p} after_ctx={:p} \
                 did_set_current={} current_ctx={:p} current_ctx_rc={} \
                 thread_id={}",
                rc_sync as i32,
                sync_name,
                sync_text,
                raw_stream,
                expected_ctx,
                before_ctx,
                after_ctx,
                did_set_current,
                current_ctx,
                rc_get_current as i32,
                diag_thread_id
            );
            anyhow::bail!(
                "TIER8 pre-capture raw cuStreamSynchronize failed: rc={} expected={:p} before={:p} after={:p}",
                rc_sync as i32,
                expected_ctx,
                before_ctx,
                after_ctx
            );
        }
        tier8_capture_verbose!(
            "[TIER8-CAPTURE raw-context-guard] call=cuStreamSynchronize \
             result=OK stream_id=unavailable protocol_group=monolithic_parent_template \
             rc=0 raw_stream={:p} expected_ctx={:p} before_ctx={:p} \
             after_ctx={:p} did_set_current={} thread_id={}",
            raw_stream,
            expected_ctx,
            before_ctx,
            after_ctx,
            did_set_current,
            diag_thread_id
        );

        let engine = self
            .engine
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("No engine loaded"))?;

        // Capture-mode flag for cudarc's bind_to_thread no-op shim.
        cudarc::driver::set_capture_mode_active(true);
        struct CaptureGuard;
        impl Drop for CaptureGuard {
            fn drop(&mut self) {
                cudarc::driver::set_capture_mode_active(false);
            }
        }
        let _guard = CaptureGuard;

        // ── Begin capture (raw sys, RELAXED mode matches legacy path) ──
        let rc = unsafe {
            sys::cuStreamBeginCapture_v2(
                raw_stream,
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
            )
        };
        if !matches!(rc, sys::CUresult::CUDA_SUCCESS) {
            anyhow::bail!("cuStreamBeginCapture_v2 failed: {:?}", rc);
        }

        // Tagger snapshots the CUgraphNode handle after each kernel.
        let mut tagger = StreamCaptureTagger::new(raw_stream);

        // Launch the captured kernel sequence. On error, abort capture
        // cleanly and propagate.
        crate::fused_engine::tier8_set_cuda_error_drain_context(cuda_error_drain_context.clone());
        let kernel_result = match body {
            RawResidentTemplateBody::AutonomousMd => {
                #[cfg(test)]
                engine.begin_neighbor_capture_diagnostic();
                engine.step_autonomous_kernels_tagged(&stream, Some(&mut tagger))
            }
            RawResidentTemplateBody::NcmcReversible => {
                engine.step_lambda_ncmc_reversible_kernels_tagged(&stream, Some(&mut tagger))
            }
            RawResidentTemplateBody::CaRestraintChild => {
                engine.enqueue_resident_ca_restraint_child(&stream)
            }
            RawResidentTemplateBody::CouplingClearHeartbeatChild => engine
                .enqueue_resident_coupling_clear_then_heartbeat_child(&stream)
                .map(|_| ()),
        };
        crate::fused_engine::tier8_set_cuda_error_drain_context(None);
        if let Some(ref drain) = cuda_error_drain_context {
            let build_succeeded = kernel_result.is_ok();
            let failure_within_n = kernel_result.is_err();
            log::info!(
                "[TIER8-CAPTURE cuda-error-drain-summary] stream_id={} raw_stream=0x{:x} \
                 call_site={} drain_count={} rc={} cuda_name={} cuda_string=\"{}\" \
                 subsequent_build_succeeded={} failure_within_3_milestones={}",
                drain.stream_slot,
                drain.raw_stream,
                drain.call_site,
                drain.drain_count,
                drain.rc,
                drain.cuda_name,
                drain.cuda_string,
                build_succeeded,
                failure_within_n
            );
        }
        if let Err(e) = kernel_result {
            // Abort capture: end and discard the template.
            let mut discard: sys::CUgraph = std::ptr::null_mut();
            unsafe {
                let _ = sys::cuStreamEndCapture(raw_stream, &mut discard);
            }
            if !discard.is_null() {
                unsafe {
                    let _ = sys::cuGraphDestroy(discard);
                }
            }
            return Err(anyhow::anyhow!(
                "{} captured body: {}",
                body.kernel_label(),
                e
            ));
        }

        // ── End capture → raw CUgraph template ─────────────────────────
        let mut cu_graph: sys::CUgraph = std::ptr::null_mut();
        let rc = unsafe { sys::cuStreamEndCapture(raw_stream, &mut cu_graph) };
        if !matches!(rc, sys::CUresult::CUDA_SUCCESS) {
            anyhow::bail!("cuStreamEndCapture failed: {:?}", rc);
        }
        if cu_graph.is_null() {
            anyhow::bail!("cuStreamEndCapture produced null template");
        }
        engine.record_prismcode_md_native_graph_capture(body.capture_label());
        let mut template = CapturedTemplate::from_capture(cu_graph, tagger.into_registry(), stream);
        template.retain_physical_launch_domain(self.physical_launch_domain.clone());
        if let Some(owner) = engine.governor_device() { template.retain_governor(owner.clone()); }
        if let Some(owner) = engine.asc_device() { template.retain_asc(owner.clone()); }
        Ok(template)
    }

    pub fn capture_autonomous_graph(
        &mut self,
    ) -> anyhow::Result<crate::graph_capture::AutonomousGraph> {
        self.capture_autonomous_graph_with_body_unroll(1, 1)
    }

    pub fn capture_autonomous_graph_with_body_unroll(
        &mut self,
        body_unroll: u32,
        steps_per_launch: u32,
    ) -> anyhow::Result<crate::graph_capture::AutonomousGraph> {
        self.capture_original_body_slice_impl(body_unroll.max(1),steps_per_launch.max(1),0,body_unroll.max(1))
    }

    /// Capture a bounded prefix/suffix of the ORIGINAL ordinary graph group.
    /// This is a capture-only operation; execution still uses the production
    /// nonresident graph executor with actual ProtocolState endpoint receipts.
    pub fn capture_original_body_slice(&mut self,body_unroll:u32,steps_per_launch:u32,
        first_original_slot:u32,original_width:u32)->anyhow::Result<crate::graph_capture::AutonomousGraph> {
        anyhow::ensure!(original_width>0 && body_unroll>0 && first_original_slot<original_width
            && body_unroll<=original_width-first_original_slot && steps_per_launch==body_unroll,
            "invalid exact original graph-body slice");
        self.capture_original_body_slice_impl(body_unroll,steps_per_launch,first_original_slot,original_width)
    }
    fn capture_original_body_slice_impl(&mut self,body_unroll:u32,steps_per_launch:u32,
        first_original_slot:u32,original_width:u32)->anyhow::Result<crate::graph_capture::AutonomousGraph> {
        use cudarc::driver::sys;
        let stream=self.stream.clone();

        // CRITICAL: sync the stream and bind the context BEFORE capture to flush all
        // prior module loads, memcpy, and initialization. CUDA capture mode forbids
        // context management calls (cuCtxGetCurrent/cuCtxSetCurrent), so the context
        // must already be bound before we enter capture mode.
        stream
            .context()
            .bind_to_thread()
            .map_err(|e| anyhow::anyhow!("Pre-capture context bind: {:?}", e))?;
        stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("Pre-capture sync: {:?}", e))?;

        // This direct graph body has no resident binding. Existing bound
        // resident graphs remain valid until a topology replacement changes
        // their engine allocations.
        if let Some(ref mut engine) = self.engine {
            // Enable capture mode flag — cudarc's bind_to_thread() becomes a no-op.
            // See vendor/cudarc/src/driver/safe/core.rs for the patch.
            cudarc::driver::set_capture_mode_active(true);

            // Helper to always clear the flag, even on error paths.
            struct CaptureGuard;
            impl Drop for CaptureGuard {
                fn drop(&mut self) {
                    cudarc::driver::set_capture_mode_active(false);
                }
            }
            let _guard = CaptureGuard;

            // Begin stream capture
            stream
                .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                .map_err(|e| anyhow::anyhow!("Stream capture begin: {:?}", e))?;

            // Launch the graph body while capture is active. body_unroll > 1
            // records repeated MD opcode bodies in the CUDA graph template, so
            // one replay advances multiple logical MD steps and removes the
            // host per-step launch loop from the hot path.
            // One exclusive capture group prevents host coordinate/parameter
            // mutation between retained neighbor endpoints. Its first body
            // still rebuilds on every replay, including the first replay after
            // construction, restore, or any host-side state update.
            let capture=if first_original_slot==0 && body_unroll==original_width {
                engine.capture_autonomous_unrolled_endpoint_neighbors(&stream,body_unroll)
            }else{engine.emit_original_autonomous_body_slice(&stream,body_unroll,first_original_slot,original_width)};
            if let Err(e) = capture {
                log::warn!("step_autonomous body_unroll={}: {} — aborting capture", body_unroll, e);
                let _ = stream.end_capture(
                    sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH
                );
                return Err(e);
            }

            // End capture → instantiate
            let graph = stream
                .end_capture(
                    sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                )
                .map_err(|e| anyhow::anyhow!("Stream capture end: {:?}", e))?
                .ok_or_else(|| anyhow::anyhow!("Null graph from capture"))?;
            engine.record_prismcode_md_native_graph_capture(if first_original_slot==0 && body_unroll==original_width {
                "capture_autonomous_graph"
            }else{"capture_original_body_slice"});
            let mut graph=crate::graph_capture::AutonomousGraph::new_with_steps_per_launch(graph,steps_per_launch);
            graph.retain_physical_launch_domain(self.physical_launch_domain.clone());
            Ok(graph)
        } else {
            anyhow::bail!("No engine loaded")
        }
    }

    fn capture_while_predicate_template(
        &mut self,
        pending_dev_ptr: u64,
        iter_dev_ptr: u64,
        handle: u64,
        max_iterations: u32,
    ) -> anyhow::Result<crate::graph_capture::CapturedTemplate> {
        use crate::graph_capture::CapturedTemplate;
        use cudarc::driver::sys;
        use std::collections::HashMap;

        let stream = self.stream.clone();
        stream
            .context()
            .bind_to_thread()
            .map_err(|e| anyhow::anyhow!("predicate pre-capture context bind: {:?}", e))?;
        stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("predicate pre-capture sync: {:?}", e))?;

        cudarc::driver::set_capture_mode_active(true);
        struct CaptureGuard;
        impl Drop for CaptureGuard {
            fn drop(&mut self) {
                cudarc::driver::set_capture_mode_active(false);
            }
        }
        let _guard = CaptureGuard;

        let raw_stream = stream.cu_stream();
        let rc = unsafe {
            sys::cuStreamBeginCapture_v2(
                raw_stream,
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
            )
        };
        if !matches!(rc, sys::CUresult::CUDA_SUCCESS) {
            anyhow::bail!("predicate cuStreamBeginCapture_v2 failed: {:?}", rc);
        }

        let launch_result = unsafe {
            crate::graph_capture::launch_while_predicate_bridge(
                raw_stream,
                pending_dev_ptr,
                iter_dev_ptr,
                handle,
                max_iterations.max(1),
            )
        };
        if let Err(e) = launch_result {
            let mut discard: sys::CUgraph = std::ptr::null_mut();
            unsafe {
                let _ = sys::cuStreamEndCapture(raw_stream, &mut discard);
            }
            if !discard.is_null() {
                unsafe {
                    let _ = sys::cuGraphDestroy(discard);
                }
            }
            return Err(e.context("capture WHILE predicate bridge"));
        }

        let mut cu_graph: sys::CUgraph = std::ptr::null_mut();
        let rc = unsafe { sys::cuStreamEndCapture(raw_stream, &mut cu_graph) };
        if !matches!(rc, sys::CUresult::CUDA_SUCCESS) {
            anyhow::bail!("predicate cuStreamEndCapture failed: {:?}", rc);
        }
        if cu_graph.is_null() {
            anyhow::bail!("predicate cuStreamEndCapture produced null template");
        }
        Ok(CapturedTemplate::from_capture(
            cu_graph,
            HashMap::new(),
            stream,
        ))
    }

    pub fn capture_autonomous_while_resident_graph(
        &mut self,
        max_iterations: u32,
    ) -> anyhow::Result<crate::graph_capture::AutonomousGraph> {
        let (graph, _profile) =
            self.capture_autonomous_while_resident_graph_composed(max_iterations)?;
        Ok(graph)
    }

    /// Capture an autonomous resident WHILE graph into an opaque owner that
    /// retains its private replay binding. The generic capture method remains
    /// available for callers that do not need optional-housekeeping receipts.
    pub fn capture_autonomous_while_resident_graph_with_replay_binding(
        &mut self,
        max_iterations: u32,
    ) -> anyhow::Result<BoundResidentGraph> {
        let (graph, profile) =
            self.capture_autonomous_while_resident_graph_composed(max_iterations)?;
        let graph = Box::new(graph);
        let binding = self.issue_resident_graph_replay_binding(graph.as_ref(), profile)?;
        Ok(BoundResidentGraph { graph, binding })
    }

    fn capture_autonomous_while_resident_graph_composed(
        &mut self,
        max_iterations: u32,
    ) -> anyhow::Result<(
        crate::graph_capture::AutonomousGraph,
        ResidentGraphBindingProfile,
    )> {
        let max_iterations = max_iterations.max(1);
        let stream = self.stream.clone();
        stream
            .context()
            .bind_to_thread()
            .map_err(|e| anyhow::anyhow!("WHILE resident context bind: {:?}", e))?;
        stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("WHILE resident pre-sync: {:?}", e))?;

        // Primary and exact-tail resident graphs may coexist for one campaign;
        // each gets its own opaque binding after instantiation.
        self.engine
            .as_ref()
            .context("WHILE resident capture has no loaded engine")?;

        let (physics_template, resident_binding_profile) = self.capture_composed_physics_body()?;

        let graph = self.assemble_while_resident_graph(
            max_iterations,
            stream,
            physics_template,
            "capture_autonomous_while_resident_graph",
        )?;
        Ok((graph, resident_binding_profile))
    }

    fn capture_composed_physics_body(&mut self) -> anyhow::Result<(
        crate::graph_capture::CapturedTemplate, ResidentGraphBindingProfile)> {
        let mut physics_template = self
            .capture_autonomous_template()
            .context("capture autonomous MD body template for WHILE resident graph")?;

        // Capture housekeeping from its existing module as separate pure child
        // templates.  This avoids the known cross-module stream-capture
        // failure without replacing the MD engine or falling back to host work.
        let available = self
            .engine
            .as_ref()
            .context("resident housekeeping composition has no loaded engine")?
            .resident_housekeeping_kernel_profile();
        if !available.coupling_clear_child_bound {
            bail!(
                "resident housekeeping composition requires coupling_buffer_clear from housekeeping.ptx; CPU memset fallback is not graph-resident"
            );
        }

        let tail_template = self
            .capture_resident_coupling_clear_heartbeat_child_template()
            .context("capture native coupling-clear/heartbeat child template")?;

        let _physical_step_complete = physics_template
            .node("physical_step_complete")
            .context("autonomous MD template lacks physical_step_complete splice anchor")?;
        let _multi_lif = physics_template
            .node("multi_lif")
            .context("autonomous MD template lacks multi_lif splice anchor")?;
        let coupling_build = physics_template
            .node("coupling_build")
            .context("autonomous MD template lacks coupling_build splice anchor")?;

        physics_template.retain_child_owners(&tail_template);
        let _tail_child = physics_template
            .add_child_graph_node(&[coupling_build], tail_template.cu_graph())
            .context("splice native coupling-clear/heartbeat child after coupling build")?;

        let resident_binding_profile = ResidentGraphBindingProfile {
            ca_restraint_child_bound: false,
            ca_coordinate_intervention_bound: available.ca_coordinate_intervention_bound,
            nma_force_phases_bound:available.nma_force_phases_bound,
            nma_controls_device_ptr:available.nma_controls_device_ptr,
            coupling_clear_child_bound: true,
            heartbeat_child_bound: available.heartbeat_child_bound,
        };

        Ok((physics_template, resident_binding_profile))
    }

    /// Prepare allocation-stable native cells and retain the exact live V2
    /// owner; preparation never executes physical or measurement kernels.
    #[cfg(feature="v2_ignition")]
    pub fn prepare_resident_observation_parts(&mut self,cadence:u32,
        cfg:&crate::captured_pipeline::PipelineConfig,
        candidates:&crate::captured_pipeline::KccCausalCandidates,leads:&mut CudaSlice<u32>,
        pipeline:Arc<crate::captured_pipeline::CapturedAdjudicationPipeline>)
        -> anyhow::Result<PreparedResidentObservation> {
        anyhow::ensure!(cadence>0,"shared physical cadence must be positive");
        let stream=self.stream.clone();stream.synchronize()?;
        anyhow::ensure!(pipeline.md_stream_raw()==stream.cu_stream(),"shared V2 owner belongs to another physical stream");
        let (physics,profile)=self.capture_composed_physics_body()?;
        let native=self.engine.as_mut().context("shared physical preparation lacks engine")?;
        let (snapshot,ladd,keys,cold_end)=native.capture_ladd_parent_parts()?;
        let step_ptr=native.d_protocol_current_step_dev_ptr(&stream);
        let kcc=self.capture_kcc_measurement_template(candidates,leads)?;
        let parity=stream.clone_htod(&[0u32])?;
        let pending=stream.clone_htod(&[1u32])?;
        let iterations=stream.clone_htod(&[0u32])?;
        stream.synchronize()?;
        Ok(PreparedResidentObservation {stream:stream.clone(),cadence,profile,config:cfg.clone(),physics,snapshot,ladd,kcc,
            parity_device:crate::governor_device::CudaOwned::new(parity,stream.clone()),
            pending_device:crate::governor_device::CudaOwned::new(pending,stream.clone()),
            iterations_device:crate::governor_device::CudaOwned::new(iterations,stream),
            step_ptr,cold_end,parity:0,keys,pipeline,appended:false,parent_template_identity:0,binding:None})
    }

    #[cfg(feature="v2_ignition")]
    pub fn prepare_shared_resident_observation_window(&mut self,prepared:&mut PreparedResidentObservation,
        graph:&crate::graph_capture::AutonomousGraph)->anyhow::Result<SharedResidentObservationAdmission> {
        self.autonomous_chunk_clock.invalidate();
        anyhow::ensure!(prepared.appended,"shared physical cells were never appended to a parent");
        anyhow::ensure!(graph.cu_graph_template().map(|raw|raw as usize)==Some(prepared.parent_template_identity),
            "shared physical replay graph differs from the actual append destination");
        anyhow::ensure!(prepared.raw_stream()==self.stream.cu_stream(),"shared physical prepared stream mismatch");
        prepared.pipeline.prepare_launch()?;
        if prepared.binding.is_none() {
            // Registration can precede activation while other cells reach their
            // observer boundary. Seed from the actual live pointer permutation,
            // never from elapsed host window counts or the factory's old parity.
            let live=self.engine.as_ref().context("shared activation lacks engine")?.ladd_cadence_capture_key(false);
            prepared.parity=prepared.keys.iter().position(|key|*key==live)
                .context("shared activation LADD allocation/grid/cold-cutoff binding changed")?;
            self.stream.memcpy_htod(&[prepared.parity as u32],&mut *prepared.parity_device)?;
            self.stream.synchronize()?;
            prepared.binding=Some(self.issue_resident_graph_replay_binding_parts(graph,prepared.profile,true)?);
            self.record_prismcode_md_native_graph_capture("shared_resident_observation_parent");
        }
        let binding=prepared.binding.as_ref().context("shared physical binding missing")?;
        self.resolve_resident_graph_replay_binding_parts(graph,binding,true)?;
        let capture_nonce=binding.capture_nonce;
        let start=self.device_protocol_current_step()?;
        let native=self.engine.as_mut().context("shared physical admission lacks engine")?;
        let boundary=native.begin_captured_ladd_cadence_boundary(start,prepared.cadence)?;
        let dispatch=native.prepare_captured_ladd_dispatch(boundary,&prepared.keys[prepared.parity])?;
        Ok(SharedResidentObservationAdmission {dispatch,start,capture_nonce,parity:prepared.parity})
    }

    /// Called only after the coordinator fences the actual global owner stream.
    /// Verify native per-engine clocks/counters before committing any receipt.
    #[cfg(feature="v2_ignition")]
    pub fn commit_shared_resident_observation_window(&mut self,prepared:&mut PreparedResidentObservation,
        graph:&crate::graph_capture::AutonomousGraph,admission:SharedResidentObservationAdmission,
        launch_path:&str)->anyhow::Result<ResidentGraphLaddCadenceWindowReceipt> {
        let result=(|| -> anyhow::Result<_> {
            let binding=prepared.binding.as_ref().context("shared physical commit missing binding")?;
            anyhow::ensure!(binding.capture_nonce==admission.capture_nonce && prepared.parity==admission.parity,
                "shared physical commit has stale capture/parity admission");
            let (profile,structure)=self.resolve_resident_graph_replay_binding_parts(graph,binding,true)?;
            let end=self.device_protocol_current_step()?;
            anyhow::ensure!(Some(end)==admission.start.checked_add(prepared.cadence),"shared physical endpoint mismatch");
            let iterations=self.stream.clone_dtoh(&*prepared.iterations_device)?;
            let parity=self.stream.clone_dtoh(&*prepared.parity_device)?;
            self.stream.synchronize()?;
            anyhow::ensure!(iterations==vec![prepared.cadence],"shared native WHILE iteration count mismatch");
            anyhow::ensure!(parity==vec![(admission.parity^1) as u32],"shared native LADD parity mismatch");
            self.engine.as_mut().context("shared physical commit lacks engine")?
                .commit_captured_ladd_dispatch_after_sync(admission.dispatch)?;
            prepared.parity^=1;
            let receipt=ResidentGraphReplayWindowReceipt {start_device_step:admission.start,end_device_step:end,
                logical_steps:prepared.cadence,graph_launches:1,steps_per_launch:prepared.cadence};
            self.record_bound_resident_graph_replay_window(profile,&structure,&receipt,launch_path)?;
            prepared.pipeline.observe_g26_after_parent_launch(graph.cu_graph_exec_raw()
                .context("shared physical parent lacks executable trace identity")?);
            Ok(ResidentGraphLaddCadenceWindowReceipt {start_device_step:admission.start,end_device_step:end,
                cadence_physical_steps:prepared.cadence,graph_windows:vec![receipt]})
        })();
        if result.is_err(){self.abort_shared_resident_observation_window();}
        result
    }

    #[cfg(feature="v2_ignition")]
    pub fn abort_shared_resident_observation_window(&mut self) {
        if let Some(native)=self.engine.as_mut(){native.poison_ladd_cadence_boundary("shared physical parent epoch failed");}
    }

    /// Capture one complete physical observation endpoint. No capture advances
    /// physical state, native KCC history, reference training or output cursors.
    #[cfg(feature = "v2_ignition")]
    pub fn capture_resident_observation_graph(&mut self, cadence: u32,
        cfg: &crate::captured_pipeline::PipelineConfig,
        candidates: &crate::captured_pipeline::KccCausalCandidates,
        leads: &mut cudarc::driver::CudaSlice<u32>)
        -> anyhow::Result<BoundResidentObservationGraph> {
        self.capture_resident_observation_graph_owner(cadence,cfg,candidates,leads,None,0)
    }

    /// Reassemble the physical observation root around the EXACT live V2
    /// owner. T7/adjudicator/tensor/reference history and output clocks survive.
    /// The existing owner must expose compatible pure parent-predicate nodes;
    /// incompatible standalone roots return an explicit error, never a reset.
    #[cfg(feature="v2_ignition")]
    pub fn capture_resident_observation_graph_with_live_v2(&mut self,cadence:u32,
        cfg:&crate::captured_pipeline::PipelineConfig,
        candidates:&crate::captured_pipeline::KccCausalCandidates,
        leads:&mut cudarc::driver::CudaSlice<u32>,
        pipeline:Arc<crate::captured_pipeline::CapturedAdjudicationPipeline>)
        ->anyhow::Result<BoundResidentObservationGraph> {
        self.capture_resident_observation_graph_owner(cadence,cfg,candidates,leads,Some(pipeline),0)
    }

    #[cfg(feature="v2_ignition")]
    fn capture_resident_observation_graph_owner(&mut self,cadence:u32,
        cfg:&crate::captured_pipeline::PipelineConfig,
        candidates:&crate::captured_pipeline::KccCausalCandidates,
        leads:&mut cudarc::driver::CudaSlice<u32>,
        existing:Option<Arc<crate::captured_pipeline::CapturedAdjudicationPipeline>>,
        binding_parity:usize)
        ->anyhow::Result<BoundResidentObservationGraph> {
        use crate::graph_capture::{CapturedTemplate, ResidentLoopState, capture_pure_template};
        use cudarc::driver::{DevicePtr, DevicePtrMut, sys};
        anyhow::ensure!(cadence > 0, "physical observation cadence must be positive");
        let stream = self.stream.clone();
        stream.synchronize()?;
        let (physics, profile) = self.capture_composed_physics_body()?;
        let engine = self.engine.as_mut().context("physical observation has no fused engine")?;
        let (snapshot, mut ladd, mut keys, cold_end) = engine.capture_ladd_parent_parts()?;
        anyhow::ensure!(binding_parity<2,"invalid checkpoint LADD owner parity");
        if binding_parity==1 {keys.swap(0,1);ladd.swap(0,1);ladd.swap(2,3);}
        let step_ptr = engine.d_protocol_current_step_dev_ptr(&stream);
        let kcc = self.capture_kcc_measurement_template(candidates,leads)?;
        let mut parity = stream.clone_htod(&[binding_parity as u32])?;
        let parity_ptr = parity.device_ptr_mut(&stream).0;
        let mut pending = stream.clone_htod(&[1u32])?;
        let pending_ptr = pending.device_ptr_mut(&stream).0;
        let mut iterations = stream.clone_htod(&[0u32])?;
        let iter_ptr = iterations.device_ptr_mut(&stream).0;
        stream.synchronize()?;
        let parent = crate::captured_pipeline::CapturedAdjudicationPipeline::build_kcc_observation_parent_with_prefix_owner(
            stream.context(), &stream, cfg, kcc, existing, |parent| {
                let graph = parent.cu_graph();
                parent.retain_child_owners(&snapshot);
                parent.retain_child_owners(&physics);
                let snapshot_node = parent.add_child_graph_node(&[],snapshot.cu_graph())?;
                let reset_pending = unsafe { crate::graph_capture::memset_node_u32(graph,&[snapshot_node],pending_ptr,1)? };
                let reset_iter = unsafe { crate::graph_capture::memset_node_u32(graph,&[snapshot_node],iter_ptr,0)? };
                let handle = unsafe { crate::graph_node::create_while_handle(graph,1) }
                    .map_err(|rc|anyhow::anyhow!("physical parent WHILE handle: {rc}"))?;
                let (while_node,body) = unsafe { crate::graph_node::add_while_node_v3(graph,&[reset_pending,reset_iter],handle) }
                    .map_err(|rc|anyhow::anyhow!("physical parent WHILE body: {rc}"))?;
                let physics_node = unsafe { crate::graph_node::add_child_graph_node_v3(body,&[],physics.cu_graph()) }
                    .map_err(|rc|anyhow::anyhow!("physical parent native physics: {rc}"))?;
                let predicate = self.capture_while_predicate_template(pending_ptr,iter_ptr,handle,cadence)?;
                parent.retain_child_owners(&predicate);
                unsafe { crate::graph_node::add_child_graph_node_v3(body,&[physics_node],predicate.cu_graph()) }
                    .map_err(|rc|anyhow::anyhow!("physical parent loop predicate: {rc}"))?;
                let ladd_handle = unsafe { crate::graph_node::create_while_handle(graph,0) }
                    .map_err(|rc|anyhow::anyhow!("physical parent LADD handle: {rc}"))?;
                let selector = capture_pure_template(stream.clone(), |_tagger| {
                    let rc = unsafe { prism_ladd_parent_launch_selector(stream.cu_stream(),step_ptr as *const u32,
                        parity_ptr as *mut u32,cold_end,ladd_handle) };
                    anyhow::ensure!(rc==0,"capture LADD selector: {rc}"); Ok(())
                })?;
                parent.retain_child_owners(&selector);
                let selector_node = parent.add_child_graph_node(&[while_node],selector.cu_graph())?;
                let (ladd_node,bodies) = unsafe { crate::graph_node::add_switch4_node(graph,selector_node,ladd_handle) }
                    .map_err(|rc|anyhow::anyhow!("physical parent LADD SWITCH: {rc}"))?;
                for (body,template) in bodies.into_iter().zip(ladd.iter()) {
                    parent.retain_child_owners(template);
                    unsafe { crate::graph_node::add_child_graph_node_v3(body,&[],template.cu_graph()) }
                        .map_err(|rc|anyhow::anyhow!("physical parent LADD native body: {rc}"))?;
                }
                Ok((vec![ladd_node],Some(ResidentLoopState::new(pending,iterations,pending_ptr,iter_ptr,cadence)),2))
            })?;
        let (graph,pipeline) = parent.into_parts();
        let binding = self.issue_resident_graph_replay_binding(graph.as_ref(),profile)?;
        self.engine.as_mut().context("physical parent engine disappeared")?
            .record_prismcode_md_native_graph_capture("capture_resident_observation_graph");
        Ok(BoundResidentObservationGraph {
            bound: BoundResidentGraph {graph,binding},pipeline,
            parity_device: crate::governor_device::CudaOwned::new(parity,stream),
            parity:binding_parity,ladd_keys:keys,
        })
    }

    /// Exactly one endpoint per replay. Host feedback and durable publication
    /// remain outside this call and precede the next physical endpoint.
    #[cfg(feature = "v2_ignition")]
    pub fn execute_resident_observation_window(&mut self, parent: &mut BoundResidentObservationGraph,
        cadence: u32, launch_path: &str) -> anyhow::Result<ResidentGraphLaddCadenceWindowReceipt> {
        self.autonomous_chunk_clock.invalidate();
        anyhow::ensure!(cadence == parent.steps_per_launch(),
            "physical observation replay must preserve its exact capture cadence");
        parent.pipeline.prepare_launch()?;
        let (profile,structure) = self.resolve_resident_graph_replay_binding(&parent.bound)?;
        let start = self.device_protocol_current_step()?;
        let result = (|| -> anyhow::Result<_> {
            let engine = self.engine.as_mut().context("physical observation has no engine")?;
            let boundary = engine.begin_captured_ladd_cadence_boundary(start,cadence)?;
            let dispatch = engine.prepare_captured_ladd_dispatch(boundary,&parent.ladd_keys[parent.parity])?;
            let launches = parent.bound.graph.run_chunk_exact_on_stream(cadence,self.stream.cu_stream())?;
            anyhow::ensure!(launches==1,"physical observation issued {launches} launches instead of one");
            self.synchronize_stream_raw("physical observation WHILE/LADD/KCC/V2 completion")?;
            let end = self.device_protocol_current_step()?;
            anyhow::ensure!(Some(end)==start.checked_add(cadence),"physical observation device clock mismatch");
            let status = parent.bound.graph.resident_loop_status().context("physical observation missing WHILE status")?;
            let mut iterations = 0u32;
            let rc = unsafe { cudarc::driver::sys::cuMemcpyDtoH_v2(
                (&mut iterations as *mut u32).cast(),status.iter_dev_ptr,std::mem::size_of::<u32>()) };
            anyhow::ensure!(rc == cudarc::driver::sys::CUresult::CUDA_SUCCESS && iterations == cadence,
                "physical observation WHILE iteration count mismatch: {iterations}, rc={rc:?}");
            let parity = self.stream.clone_dtoh(&*parent.parity_device)?;
            anyhow::ensure!(parity == vec![(parent.parity ^ 1) as u32], "physical observation LADD parity mismatch");
            self.engine.as_mut().context("physical observation engine disappeared")?
                .commit_captured_ladd_dispatch_after_sync(dispatch)?;
            parent.parity ^= 1;
            let receipt = ResidentGraphReplayWindowReceipt {start_device_step:start,end_device_step:end,
                logical_steps:cadence,graph_launches:launches,steps_per_launch:cadence};
            self.record_bound_resident_graph_replay_window(profile,&structure,&receipt,launch_path)?;
            parent.pipeline.observe_g26_after_parent_launch(parent.bound.graph.cu_graph_exec_raw()
                .context("physical parent lacks executable trace identity")?);
            Ok(ResidentGraphLaddCadenceWindowReceipt {start_device_step:start,end_device_step:end,
                cadence_physical_steps:cadence,graph_windows:vec![receipt]})
        })();
        if result.is_err() { if let Some(engine)=self.engine.as_mut() {
            engine.poison_ladd_cadence_boundary("physical observation parent did not reach verified endpoint");
        }}
        result
    }

    /// Capture one exact-length resident WHILE graph from the dedicated NCMC
    /// body and attach its immutable executable identity.  The identity must
    /// already match the live scheduler, selected PTX and active gamma-zero
    /// guard; an ordinary autonomous graph cannot pass these checks.
    pub(crate) fn build_lambda_ncmc_capture_identity(
        &mut self,
    ) -> anyhow::Result<crate::graph_capture::NcmcResidentGraphCaptureIdentity> {
        let binding = self
            .lambda_runtime_binding
            .as_ref()
            .context("NCMC capture identity lacks lambda runtime binding")?;
        let schedule_sha256 = binding
            .ncmc_scheduler
            .as_ref()
            .context("NCMC capture identity lacks scheduler authority")?
            .ncmc_protocol_schedule_sha256
            .clone();
        let selected_ptx_sha256 = binding
            .runtime_capability_receipt
            .as_ref()
            .and_then(|receipt| receipt.capabilities.selected_fused_ptx_sha256.clone())
            .context("NCMC capture identity lacks selected fused PTX digest")?;
        let engine = self
            .engine
            .as_mut()
            .context("NCMC capture identity has no loaded engine")?;
        let guard = engine
            .assert_lambda_ncmc_reversible_guard_active(None)?
            .clone();
        let identity = crate::graph_capture::NcmcResidentGraphCaptureIdentity::seal(
            schedule_sha256,
            selected_ptx_sha256,
            engine.lambda_ncmc_kernel_symbol_manifest_sha256()?,
            engine.lambda_ncmc_capture_allocation_binding_sha256()?,
            engine.lambda_ncmc_constraint_work_plane_sha256()?,
            guard.guard_sha256,
            guard.timestep_ps,
        )?;
        identity.validate()?;
        Ok(identity)
    }

    /// Execute the single immutable NCMC opportunity scheduled at `step`.
    /// Every propagation operation is a separately captured exact-length
    /// resident WHILE graph; no ordinary graph replay or host per-step loop is
    /// permitted. The transaction either commits the proposed physical
    /// endpoint or restores the byte-identical checkpoint before returning.
    pub fn execute_scheduled_lambda_ncmc_opportunity(
        &mut self,
        step: u64,
        autonomous_rescue_active: bool,
        ordinary_protocol_steering_active: bool,
    ) -> Result<Option<crate::lambda_fragment::NcmcExecutionResult>> {
        self.autonomous_chunk_clock.invalidate();
        use crate::fused_engine::{LambdaNcmcControlAdmission, LambdaNcmcTransactionClosure};
        use crate::lambda_fragment::{
            adjudicate_ncmc_move, derive_ncmc_balance, ncmc_protocol_direction_for_move,
            EndpointOccupancy, NcmcForceBoundaryIdentity, NcmcForceBoundaryKind,
            NcmcIntegratorPathAccounting, NcmcIntegratorPathProbabilityConvention, NcmcMoveKind,
            NcmcOperationBoundaryLineage, NcmcOrderedWorkTrace, NcmcPlacementMeasureAuthority,
            NcmcProtocolDirection, NcmcProtocolOperation, NcmcRngConsumptionReceipt,
            NcmcRngNonConsumptionReason, NcmcRngRangeDisposition, NcmcWorkTraceEntry,
        };
        use std::collections::BTreeMap;

        let binding = self
            .lambda_runtime_binding
            .as_ref()
            .context("NCMC execution lacks lambda runtime binding")?
            .clone();
        let scheduler = binding
            .ncmc_scheduler
            .as_ref()
            .context("NCMC execution lacks proposal scheduler")?
            .clone();
        let Some(opportunity) = scheduler
            .next_opportunity_at_or_after(step)
            .map_err(anyhow::Error::msg)?
        else {
            return Ok(None);
        };
        if opportunity.step != step {
            return Ok(None);
        }
        let execution_mode = binding
            .ncmc_execution_mode
            .clone()
            .context("NCMC execution lacks admitted execution mode")?;
        binding
            .phase_context_for_step(step)
            .map_err(anyhow::Error::msg)?;
        self.sync_lambda_runtime_phase(step)?;
        let engine = self
            .engine
            .as_mut()
            .context("NCMC execution has no loaded engine")?;
        let endpoint_states = engine.lambda_ncmc_current_endpoint_states()?;
        let mut current_endpoints = BTreeMap::new();
        for state in &endpoint_states {
            if state.endpoint_occupancy == EndpointOccupancy::Switching
                || current_endpoints
                    .insert(
                        state.slot_id,
                        if state.lambda.to_bits() == 0.0f64.to_bits() {
                            state.endpoint_a.clone()
                        } else if state.lambda.to_bits() == 1.0f64.to_bits() {
                            state.endpoint_b.clone()
                        } else {
                            bail!("NCMC opportunity encountered a partial-lambda slot");
                        },
                    )
                    .is_some()
            {
                bail!("NCMC endpoint-state map is partial, switching, or duplicated");
            }
        }
        let proposal = scheduler
            .proposal_for_step(step, &current_endpoints, execution_mode)
            .map_err(anyhow::Error::msg)?;
        scheduler
            .validate_proposal_at_step(step, &proposal, &current_endpoints)
            .map_err(anyhow::Error::msg)?;
        let system = engine.lambda_ncmc_system_snapshot()?;
        let initial = current_endpoints
            .get(&proposal.slot_id)
            .context("NCMC proposal subject slot is absent from endpoint map")?
            .clone();
        let direction = ncmc_protocol_direction_for_move(&system, &proposal, &initial)
            .map_err(anyhow::Error::msg)?;
        let balance =
            derive_ncmc_balance(&system, &proposal, &initial).map_err(anyhow::Error::msg)?;
        let subject_state = endpoint_states
            .iter()
            .find(|state| state.slot_id == proposal.slot_id)
            .context("NCMC subject runtime slot is absent")?
            .clone();
        let endpoint_lambda = |state: &crate::fused_engine::LambdaCurrentEndpointState,
                               endpoint: &crate::lambda_fragment::NcmcEndpointIdentity|
         -> Result<f64> {
            if endpoint == &state.endpoint_a {
                Ok(0.0)
            } else if endpoint == &state.endpoint_b {
                Ok(1.0)
            } else {
                bail!("NCMC endpoint identity is not present on its fixed runtime slot")
            }
        };
        let subject_initial_lambda = endpoint_lambda(&subject_state, &initial)?;
        let proposed_identity = crate::lambda_fragment::NcmcEndpointIdentity {
            species_id: proposal.proposed_species_id,
            chemical_state_id: proposal.proposed_chemical_state_id,
            occupancy: proposal.proposed_occupancy,
        };
        let subject_proposed_lambda = endpoint_lambda(&subject_state, &proposed_identity)?;
        let partner_projection = proposal
            .coupled_charge_partner
            .as_ref()
            .map(|partner| -> Result<_> {
                let state = endpoint_states
                    .iter()
                    .find(|state| state.slot_id == partner.slot_id)
                    .context("NCMC coupled partner runtime slot is absent")?
                    .clone();
                Ok((
                    state.runtime_slot_index,
                    endpoint_lambda(&state, &partner.initial)?,
                    endpoint_lambda(&state, &partner.proposed)?,
                ))
            })
            .transpose()?;
        let coordinates_for = |protocol_lambda: f64| -> Result<Vec<(usize, f64)>> {
            let denominator = subject_proposed_lambda - subject_initial_lambda;
            if denominator.abs() != 1.0 {
                bail!("NCMC subject endpoint coordinate is not an exact 0/1 path");
            }
            let progress = (protocol_lambda - subject_initial_lambda) / denominator;
            if !progress.is_finite() || !(0.0..=1.0).contains(&progress) {
                bail!("NCMC protocol lambda lies outside the directed endpoint path");
            }
            let mut coordinates = vec![(subject_state.runtime_slot_index, protocol_lambda)];
            if let Some((runtime_slot_index, initial_lambda, proposed_lambda)) = partner_projection
            {
                coordinates.push((
                    runtime_slot_index,
                    initial_lambda + progress * (proposed_lambda - initial_lambda),
                ));
            }
            Ok(coordinates)
        };

        let pose_reason = match proposal.move_kind {
            NcmcMoveKind::Insertion | NcmcMoveKind::Deletion => match scheduler.placement_measure {
                NcmcPlacementMeasureAuthority::PersistentDecoupledCoordinateEnsemble { .. } => {
                    NcmcRngNonConsumptionReason::PersistentDecoupledCoordinatesRetained
                }
                NcmcPlacementMeasureAuthority::UniformAccessibleVolume { .. } => {
                    bail!("uniform-volume NCMC is not executable until placement/orientation kernels materialize their reserved draws")
                }
            },
            NcmcMoveKind::Identity | NcmcMoveKind::Tautomer | NcmcMoveKind::Protonation => {
                NcmcRngNonConsumptionReason::StableSlotIdentityTransform
            }
        };
        let rng_consumption = NcmcRngConsumptionReceipt::seal(
            &proposal,
            NcmcRngRangeDisposition::NotApplicable {
                physical_reason: pose_reason,
            },
            NcmcRngRangeDisposition::NotApplicable {
                physical_reason: pose_reason,
            },
            NcmcRngRangeDisposition::NotApplicable {
                physical_reason: NcmcRngNonConsumptionReason::ExistingEquilibriumMomentaRetained,
            },
            NcmcRngRangeDisposition::Forbidden {
                physical_reason: NcmcRngNonConsumptionReason::DeterministicGammaZeroThermostat,
            },
        )
        .map_err(anyhow::Error::msg)?;
        rng_consumption
            .validate_for_scheduler(&proposal, &scheduler)
            .map_err(anyhow::Error::msg)?;

        let guard = engine.enter_lambda_ncmc_reversible_guard(LambdaNcmcControlAdmission {
            dedicated_gamma_zero_propagator: true,
            autonomous_rescue_active,
            force_or_velocity_clamp_active: false,
            ordinary_protocol_steering_active,
        })?;
        let checkpoint_sha256 = match engine.checkpoint_lambda_ncmc_phase_space() {
            Ok(digest) => digest,
            Err(error) => {
                engine.abort_lambda_ncmc_guard_before_checkpoint()?;
                return Err(error.context("lambda NCMC checkpoint failed; guard restored"));
            }
        };
        let operations = match direction {
            NcmcProtocolDirection::Forward => system.ncmc.forward_operations.clone(),
            NcmcProtocolDirection::Reverse => system.ncmc.reverse_operations.clone(),
        };
        let beta = balance.beta_mol_per_kcal;
        let transaction = (|| -> Result<_> {
            let mut entries = Vec::with_capacity(operations.len());
            let mut resident_windows = Vec::new();
            let mut current_protocol_lambda = subject_initial_lambda;
            let mut momentum_reversal_applied = false;
            for (operation_index, operation) in operations.iter().enumerate() {
                let boundary = |kind| {
                    NcmcForceBoundaryIdentity::new(
                        proposal.move_id,
                        direction,
                        operation_index as u32,
                        kind,
                    )
                    .map_err(anyhow::Error::msg)
                };
                match operation {
                    NcmcProtocolOperation::Perturb {
                        lambda_from,
                        lambda_to,
                    } => {
                        if lambda_from.to_bits() != current_protocol_lambda.to_bits() {
                            bail!(
                                "NCMC perturbation does not start at the live directed coordinate"
                            );
                        }
                        let before = self
                            .engine
                            .as_mut()
                            .context("NCMC engine disappeared during perturbation")?
                            .lambda_ncmc_exact_hamiltonian(boundary(
                                NcmcForceBoundaryKind::PerturbPre,
                            )?)?;
                        let coordinates = coordinates_for(*lambda_to)?;
                        self.engine
                            .as_mut()
                            .context("NCMC engine disappeared during coordinate update")?
                            .apply_lambda_slot_coordinates(&coordinates)?;
                        let after = self
                            .engine
                            .as_mut()
                            .context("NCMC engine disappeared after perturbation")?
                            .lambda_ncmc_exact_hamiltonian(boundary(
                                NcmcForceBoundaryKind::PerturbPost,
                            )?)?;
                        entries.push(NcmcWorkTraceEntry::Perturb {
                            operation_index: operation_index as u32,
                            boundary_lineage: NcmcOperationBoundaryLineage {
                                before: before
                                    .force_population_lineage
                                    .clone()
                                    .context("NCMC perturb PRE population was not journaled")?,
                                after: after
                                    .force_population_lineage
                                    .clone()
                                    .context("NCMC perturb POST population was not journaled")?,
                            },
                            lambda_from: *lambda_from,
                            lambda_to: *lambda_to,
                            potential_before_kcal_mol: before.total_potential_kcal_mol,
                            potential_after_kcal_mol: after.total_potential_kcal_mol,
                            reduced_work: beta
                                * (after.total_potential_kcal_mol
                                    - before.total_potential_kcal_mol),
                        });
                        current_protocol_lambda = *lambda_to;
                    }
                    NcmcProtocolOperation::Propagate {
                        lambda,
                        integration_steps,
                    } => {
                        if lambda.to_bits() != current_protocol_lambda.to_bits() {
                            bail!("NCMC propagation Hamiltonian differs from live coordinate");
                        }
                        let before = self
                            .engine
                            .as_mut()
                            .context("NCMC engine disappeared before propagation")?
                            .lambda_ncmc_exact_hamiltonian(boundary(
                                NcmcForceBoundaryKind::PropagatePre,
                            )?)?;
                        self.engine
                            .as_mut()
                            .context("NCMC engine disappeared preparing work plane")?
                            .prepare_lambda_ncmc_resident_window()?;
                        let capture_identity = self.build_lambda_ncmc_capture_identity()?;
                        if capture_identity.protocol_state_guard_sha256 != guard.guard_sha256 {
                            bail!("NCMC capture identity guard changed during transaction");
                        }
                        let graph = self.capture_ncmc_reversible_while_resident_graph(
                            *integration_steps,
                            capture_identity,
                        )?;
                        let window =
                            self.execute_ncmc_resident_window(&graph, step, *integration_steps)?;
                        let constraint_work = self
                            .engine
                            .as_mut()
                            .context("NCMC engine disappeared reducing constraint work")?
                            .lambda_ncmc_constraint_work_kcal_mol()?;
                        let after = self
                            .engine
                            .as_mut()
                            .context("NCMC engine disappeared after propagation")?
                            .lambda_ncmc_exact_hamiltonian(boundary(
                                NcmcForceBoundaryKind::PropagatePost,
                            )?)?;
                        entries.push(NcmcWorkTraceEntry::Propagate {
                            operation_index: operation_index as u32,
                            boundary_lineage: NcmcOperationBoundaryLineage {
                                before: before
                                    .force_population_lineage
                                    .clone()
                                    .context("NCMC propagate PRE population was not journaled")?,
                                after: after
                                    .force_population_lineage
                                    .clone()
                                    .context("NCMC propagate POST population was not journaled")?,
                            },
                            lambda: *lambda,
                            integration_steps: *integration_steps,
                            hamiltonian_before_kcal_mol: before.total_potential_kcal_mol
                                + before.total_kinetic_kcal_mol,
                            hamiltonian_after_kcal_mol: after.total_potential_kcal_mol
                                + after.total_kinetic_kcal_mol,
                            constraint_impulse_work_kcal_mol: constraint_work,
                            reduced_shadow_work: beta
                                * ((after.total_potential_kcal_mol + after.total_kinetic_kcal_mol)
                                    - (before.total_potential_kcal_mol
                                        + before.total_kinetic_kcal_mol)),
                            resident_window: window.clone(),
                        });
                        resident_windows.push(window);
                    }
                    NcmcProtocolOperation::MomentumReverse => {
                        let before = self
                            .engine
                            .as_mut()
                            .context("NCMC engine disappeared before momentum reversal")?
                            .lambda_ncmc_exact_hamiltonian(boundary(
                                NcmcForceBoundaryKind::MomentumReversePre,
                            )?)?;
                        self.engine
                            .as_mut()
                            .context("NCMC engine disappeared during momentum reversal")?
                            .reverse_lambda_ncmc_velocities()?;
                        let after = self
                            .engine
                            .as_mut()
                            .context("NCMC engine disappeared after momentum reversal")?
                            .lambda_ncmc_exact_hamiltonian(boundary(
                                NcmcForceBoundaryKind::MomentumReversePost,
                            )?)?;
                        entries.push(NcmcWorkTraceEntry::MomentumReverse {
                            operation_index: operation_index as u32,
                            boundary_lineage: NcmcOperationBoundaryLineage {
                                before: before
                                    .force_population_lineage
                                    .clone()
                                    .context("NCMC momentum PRE population was not journaled")?,
                                after: after
                                    .force_population_lineage
                                    .clone()
                                    .context("NCMC momentum POST population was not journaled")?,
                            },
                            kinetic_before_kcal_mol: before.total_kinetic_kcal_mol,
                            kinetic_after_kcal_mol: after.total_kinetic_kcal_mol,
                            exact_involution_applied: true,
                        });
                        momentum_reversal_applied = true;
                    }
                }
            }
            if current_protocol_lambda.to_bits() != subject_proposed_lambda.to_bits()
                || !momentum_reversal_applied
            {
                bail!("NCMC operation list did not reach its endpoint with momentum involution");
            }
            let trace = NcmcOrderedWorkTrace::seal(
                &system.ncmc,
                proposal.move_id,
                direction,
                beta,
                entries,
                true,
                true,
            )
            .map_err(anyhow::Error::msg)?;
            let accounting = match system.ncmc.integrator_path_probability_convention {
                NcmcIntegratorPathProbabilityConvention::ShadowWorkVolumePreserving => {
                    NcmcIntegratorPathAccounting::ShadowWorkVolumePreserving {
                        shadow_work_reduced: trace.shadow_work_reduced,
                    }
                }
                NcmcIntegratorPathProbabilityConvention::ShadowWorkWithJacobian => {
                    NcmcIntegratorPathAccounting::ShadowWorkWithJacobian {
                        shadow_work_reduced: trace.shadow_work_reduced,
                        log_abs_phase_space_jacobian: 0.0,
                    }
                }
                NcmcIntegratorPathProbabilityConvention::ExactConditionalPathProbability => {
                    bail!("deterministic resident NCMC cannot fabricate finite conditional path densities")
                }
            };
            let (
                stream_id,
                event_sequence,
                parent_receipt_sha256,
                phase_protocol_sha256,
                endpoint_support_binding_sha256,
                dual_endpoint_union_contract_sha256,
            ) = self
                .engine
                .as_mut()
                .context("NCMC engine disappeared before adjudication")?
                .lambda_ncmc_adjudication_context(step)?;
            let receipt = adjudicate_ncmc_move(
                &system,
                &proposal,
                &balance,
                stream_id,
                step,
                phase_protocol_sha256,
                event_sequence,
                parent_receipt_sha256,
                endpoint_support_binding_sha256,
                dual_endpoint_union_contract_sha256,
                initial.clone(),
                trace.perturbation_work_reduced,
                accounting,
                true,
                true,
                trace,
                rng_consumption,
            )
            .map_err(anyhow::Error::msg)?;
            Ok((receipt, resident_windows))
        })();

        let (receipt, resident_windows) = match transaction {
            Ok(value) => value,
            Err(error) => {
                let engine = self
                    .engine
                    .as_mut()
                    .context("NCMC engine disappeared during failure rollback")?;
                let restored = engine.restore_lambda_ncmc_phase_space()?;
                if restored != checkpoint_sha256 {
                    bail!("NCMC failure rollback digest differs from checkpoint");
                }
                engine
                    .exit_lambda_ncmc_reversible_guard(LambdaNcmcTransactionClosure::RolledBack)?;
                return Err(error.context("scheduled lambda NCMC transaction rolled back"));
            }
        };
        let rollback_verified = if receipt.accepted {
            false
        } else {
            let engine = self
                .engine
                .as_mut()
                .context("NCMC engine disappeared during rejected-move rollback")?;
            let restored = engine.restore_lambda_ncmc_phase_space()?;
            if restored != checkpoint_sha256 || !engine.lambda_ncmc_checkpoint_matches_current()? {
                bail!("rejected NCMC move failed exact checkpoint restoration");
            }
            engine.apply_lambda_ncmc_rejection_momentum_involution()?;
            true
        };
        self.engine
            .as_mut()
            .context("NCMC engine disappeared during transaction close")?
            .exit_lambda_ncmc_reversible_guard(if receipt.accepted {
                LambdaNcmcTransactionClosure::Committed
            } else {
                LambdaNcmcTransactionClosure::RejectedMomentumReversed
            })?;
        self.engine
            .as_mut()
            .context("NCMC engine disappeared during resolution emission")?
            .emit_lambda_ncmc_resolution(&receipt, rollback_verified)?;
        Ok(Some(crate::lambda_fragment::NcmcExecutionResult {
            state_committed: receipt.accepted,
            move_receipt: receipt,
            resident_windows,
        }))
    }

    pub fn capture_ncmc_reversible_while_resident_graph(
        &mut self,
        max_iterations: u32,
        capture_identity: crate::graph_capture::NcmcResidentGraphCaptureIdentity,
    ) -> anyhow::Result<crate::graph_capture::AutonomousGraph> {
        capture_identity.validate()?;
        let binding = self
            .lambda_runtime_binding
            .as_ref()
            .context("NCMC capture lacks lambda runtime binding")?;
        let scheduler = binding
            .ncmc_scheduler
            .as_ref()
            .context("NCMC capture lacks scheduler authority")?;
        let selected_ptx_sha256 = binding
            .runtime_capability_receipt
            .as_ref()
            .and_then(|receipt| receipt.capabilities.selected_fused_ptx_sha256.as_deref())
            .context("NCMC capture lacks selected fused PTX identity")?;
        if capture_identity.capture_source_schedule_sha256
            != scheduler.ncmc_protocol_schedule_sha256
            || capture_identity.selected_fused_ptx_sha256 != selected_ptx_sha256
        {
            bail!("NCMC capture identity differs from live scheduler/PTX authority");
        }
        self.engine
            .as_ref()
            .context("NCMC capture has no loaded engine")?
            .assert_lambda_ncmc_reversible_guard_active(Some(
                &capture_identity.protocol_state_guard_sha256,
            ))?;

        let max_iterations = max_iterations.max(1);
        let stream = self.stream.clone();
        stream
            .context()
            .bind_to_thread()
            .map_err(|e| anyhow::anyhow!("NCMC WHILE resident context bind: {e:?}"))?;
        stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("NCMC WHILE resident pre-sync: {e:?}"))?;
        self.engine
            .as_ref()
            .context("NCMC WHILE resident capture has no loaded engine")?;
        let physics_template = self
            .capture_ncmc_reversible_template()
            .context("capture dedicated gamma-zero NCMC body template")?;
        let mut graph = self.assemble_while_resident_graph(
            max_iterations,
            stream,
            physics_template,
            "capture_ncmc_reversible_while_resident_graph",
        )?;
        graph.bind_ncmc_resident_capture_identity(capture_identity)?;
        Ok(graph)
    }

    fn assemble_while_resident_graph(
        &mut self,
        max_iterations: u32,
        stream: Arc<CudaStream>,
        physics_template: crate::graph_capture::CapturedTemplate,
        capture_label: &'static str,
    ) -> anyhow::Result<crate::graph_capture::AutonomousGraph> {
        use crate::graph_capture::{CapturedTemplate, ResidentLoopState};
        use cudarc::driver::{sys, DevicePtrMut};
        use std::collections::HashMap;

        let mut pending = stream
            .clone_htod(&[1u32])
            .context("allocate WHILE resident pending flag")?;
        let mut iter = stream
            .clone_htod(&[0u32])
            .context("allocate WHILE resident iteration counter")?;
        stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("WHILE resident counter upload sync: {:?}", e))?;

        let pending_dev_ptr = {
            let (ptr, _guard) = pending.device_ptr_mut(&stream);
            ptr
        };
        let iter_dev_ptr = {
            let (ptr, _guard) = iter.device_ptr_mut(&stream);
            ptr
        };

        let mut parent_graph: sys::CUgraph = std::ptr::null_mut();
        let rc = unsafe { sys::cuGraphCreate(&mut parent_graph as *mut _, 0) };
        if !matches!(rc, sys::CUresult::CUDA_SUCCESS) {
            anyhow::bail!("cuGraphCreate parent WHILE graph failed: {:?}", rc);
        }

        let reset_pending =
            unsafe { crate::graph_capture::memset_node_u32(parent_graph, &[], pending_dev_ptr, 1) }
                .context("add WHILE pending reset node")?;
        let reset_iter =
            unsafe { crate::graph_capture::memset_node_u32(parent_graph, &[], iter_dev_ptr, 0) }
                .context("add WHILE iteration reset node")?;

        // The preceding autonomous-template capture can leave a deferred CUDA
        // error in cudarc's shared CudaContext error cell even when the raw
        // parent graph is otherwise valid. Drain once before the host-side
        // WHILE handle call so a stale capture diagnostic is not mis-attributed
        // as cudaGraphConditionalHandleCreate failure. If the error persists,
        // keep strict mode hard-failing here.
        if let Err(first) = stream.context().check_err() {
            let (first_name, first_text) = tier8_capture_driver_error_text(first.0);
            log::warn!(
                "[TIER8-WHILE pre-handle-drain] call=ctx.check_err result=DRAINED \
                 rc={} cuda_name={} cuda_string=\"{}\" parent_graph={:p} \
                 pending_dev_ptr=0x{:x} iter_dev_ptr=0x{:x}",
                first.0 as i32,
                first_name,
                first_text,
                parent_graph,
                pending_dev_ptr,
                iter_dev_ptr
            );
            if let Err(second) = stream.context().check_err() {
                let (second_name, second_text) = tier8_capture_driver_error_text(second.0);
                log::error!(
                    "[TIER8-WHILE pre-handle-drain] call=ctx.check_err(second) result=FAIL \
                     rc={} cuda_name={} cuda_string=\"{}\" parent_graph={:p}",
                    second.0 as i32,
                    second_name,
                    second_text,
                    parent_graph
                );
                anyhow::bail!(
                    "WHILE resident pre-handle CUDA error persisted: first={:?}, second={:?}",
                    first,
                    second
                );
            }
        }

        let while_handle = unsafe { crate::graph_node::create_while_handle(parent_graph, 1) }
            .map_err(|e| anyhow::anyhow!("create WHILE handle: {}", e))?;
        let (_while_node, body_graph) = unsafe {
            crate::graph_node::add_while_node_v3(
                parent_graph,
                &[reset_pending, reset_iter],
                while_handle,
            )
        }
        .map_err(|e| anyhow::anyhow!("add WHILE conditional node: {}", e))?;

        let physics_node = unsafe {
            crate::graph_node::add_child_graph_node_v3(body_graph, &[], physics_template.cu_graph())
        }
        .map_err(|e| anyhow::anyhow!("splice WHILE MD body graph: {}", e))?;

        let predicate_template = self.capture_while_predicate_template(
            pending_dev_ptr,
            iter_dev_ptr,
            while_handle,
            max_iterations,
        )?;
        let _predicate_node = unsafe {
            crate::graph_node::add_child_graph_node_v3(
                body_graph,
                &[physics_node],
                predicate_template.cu_graph(),
            )
        }
        .map_err(|e| anyhow::anyhow!("splice WHILE predicate graph: {}", e))?;

        let resident_state =
            ResidentLoopState::new(pending, iter, pending_dev_ptr, iter_dev_ptr, max_iterations);
        let mut parent_template = CapturedTemplate::from_capture(parent_graph, HashMap::new(), stream);
        parent_template.retain_child_owners(&physics_template);
        parent_template.retain_child_owners(&predicate_template);
        if let Some(ref mut engine) = self.engine {
            engine.record_prismcode_md_native_graph_capture(capture_label);
        }
        parent_template.instantiate_with_runtime_state(max_iterations, Some(resident_state))
    }

    /// Execute one exact NCMC propagation window through the real resident
    /// WHILE graph owner. The graph must have been captured for exactly this
    /// number of steps; ceil/overshoot and legacy replay are rejected.
    pub fn execute_ncmc_resident_window(
        &mut self,
        graph: &crate::graph_capture::AutonomousGraph,
        start_step: u64,
        logical_steps: u32,
    ) -> Result<crate::lambda_fragment::NcmcResidentWindowReceipt> {
        self.autonomous_chunk_clock.invalidate();
        if logical_steps == 0 {
            bail!("NCMC resident propagation window is empty");
        }
        let status = graph
            .resident_loop_status()
            .context("NCMC propagation graph is not WHILE-resident")?;
        if status.max_iterations != logical_steps || graph.steps_per_launch() != logical_steps {
            bail!(
                "NCMC resident graph window mismatch: requested={} resident_max={} steps_per_launch={}",
                logical_steps,
                status.max_iterations,
                graph.steps_per_launch()
            );
        }
        let capture_identity = graph.ncmc_resident_capture_identity().context(
            "NCMC propagation requires a separately captured gamma-zero reversible graph identity",
        )?;
        capture_identity.validate()?;
        let binding = self
            .lambda_runtime_binding
            .as_ref()
            .context("NCMC resident execution lacks lambda runtime binding")?;
        let capability = binding
            .runtime_capability_receipt
            .as_ref()
            .context("NCMC resident execution lacks live capability receipt")?;
        capability
            .validate_live()
            .map_err(|error| anyhow::anyhow!("NCMC live capability recheck: {error}"))?;
        let graph_structure = graph.resident_structure_fingerprint_sha256()?;
        let mut graph_identity = sha2::Sha256::new();
        use sha2::Digest as _;
        graph_identity.update(b"PRISM_NCMC_RESIDENT_GRAPH_EXEC_V1\0");
        graph_identity.update(capability.receipt_sha256.as_bytes());
        graph_identity.update(binding.engine_sha256.as_bytes());
        graph_identity.update(binding.topology_sha256.as_bytes());
        graph_identity.update(
            capability
                .capabilities
                .selected_fused_ptx_sha256
                .as_deref()
                .context("NCMC capability tuple lacks selected PTX hash")?
                .as_bytes(),
        );
        graph_identity.update(binding.stream_id.to_le_bytes());
        graph_identity.update(start_step.to_le_bytes());
        graph_identity.update(logical_steps.to_le_bytes());
        graph_identity.update(status.max_iterations.to_le_bytes());
        graph_identity.update(graph_structure.as_bytes());
        graph_identity.update(capture_identity.identity_sha256.as_bytes());
        graph_identity.update(capture_identity.capture_source_schedule_sha256.as_bytes());
        graph_identity.update(capture_identity.kernel_symbol_manifest_sha256.as_bytes());
        graph_identity.update(
            capture_identity
                .mutable_state_allocation_binding_sha256
                .as_bytes(),
        );
        graph_identity.update(capture_identity.constraint_work_plane_sha256.as_bytes());
        let graph_exec_sha256 = format!("{:x}", graph_identity.finalize());

        self.sync_lambda_runtime_phase(start_step)?;
        let launches =
            graph.run_chunk_compacted_on_stream(logical_steps, self.stream.cu_stream())?;
        if launches != 1 {
            bail!("NCMC resident propagation used {launches} host graph launches");
        }
        self.stream
            .synchronize()
            .map_err(|error| anyhow::anyhow!("NCMC resident window synchronize: {error:?}"))?;
        // NCMC switching substeps are Monte Carlo proposal-path coordinates,
        // not production MD time. Recording them through the ordinary native
        // graph-window ledger would create overlapping coverage at one
        // opportunity step. Residency is proven exclusively by the ordered
        // NCMC trace and this resident-window receipt namespace.
        let receipt = crate::lambda_fragment::NcmcResidentWindowReceipt {
            logical_steps: u64::from(logical_steps),
            host_graph_launches: launches as u64,
            legacy_graph_replay_dispatch_count: 0,
            resident_execution: true,
            capture_identity_sha256: capture_identity.identity_sha256.clone(),
            graph_structure_sha256: graph_structure,
            deterministic_reversible_propagation: true,
            graph_exec_sha256,
        };
        receipt
            .validate(u64::from(logical_steps))
            .map_err(anyhow::Error::msg)?;
        Ok(receipt)
    }

    // rebuild_neighbor_lists_if_needed already exists below (line 1283)

    /// Device-side spike events buffer (for GPU-direct ring buffer push).
    /// Returns the raw d_spike_events allocation — no CPU download.
    pub fn spike_events_device(&self) -> Option<&CudaSlice<u8>> {
        self.engine.as_ref().map(|e| e.spike_events_buffer())
    }

    /// Device-side spike count buffer (for GPU-direct ring buffer push).
    pub fn spike_count_device(&self) -> Option<&CudaSlice<i32>> {
        self.engine.as_ref().map(|e| e.spike_count_buffer())
    }

    /// Rebuild neighbor lists if the rebuild interval has elapsed.
    /// Returns Ok(true) if rebuilt, Ok(false) if not needed.
    pub fn rebuild_neighbor_lists_if_needed(&mut self) -> Result<bool> {
        if let Some(ref mut engine) = self.engine {
            engine.rebuild_neighbor_lists_if_needed()
        } else {
            Ok(false)
        }
    }

    /// Get mutable reference to the inner NhsAmberFusedEngine.
    /// Used when the caller needs direct access (e.g., autonomous TWIN coupling).
    pub fn fused_engine_mut(&mut self) -> Option<&mut crate::fused_engine::NhsAmberFusedEngine> {
        self.autonomous_chunk_clock.invalidate();
        self.engine.as_mut()
    }

    /// Get immutable reference to the inner NhsAmberFusedEngine.
    pub fn fused_engine(&self) -> Option<&crate::fused_engine::NhsAmberFusedEngine> {
        self.engine.as_ref()
    }

    /// Set REST2 solute tempering λ. λ=1.0 = physical, λ<1.0 = softened potential.
    pub fn set_solute_lambda(&mut self, lambda: f32) -> Result<()> {
        if let Some(ref mut engine) = self.engine {
            engine.set_solute_lambda(lambda);
            Ok(())
        } else {
            bail!("No topology loaded")
        }
    }

    /// Focus REST2 λ on spike-frustrated residues identified during cold_hold.
    pub fn apply_focused_lambda(&mut self) -> Result<crate::fused_engine::FocusedLambdaPublication> {
        if let Some(ref mut engine) = self.engine {
            engine.apply_focused_lambda()
        } else {
            bail!("No topology loaded")
        }
    }

    /// Enable spike-guided adaptive bias: closed-loop UV energy modulation
    /// driven by real-time spike activity in each voxel region.
    pub fn set_adaptive_bias(&mut self, enabled: bool) -> Result<()> {
        if let Some(ref mut engine) = self.engine {
            engine.set_adaptive_bias(enabled);
            Ok(())
        } else {
            bail!("No topology loaded")
        }
    }

    /// Set fused multi-step: N AMBER steps per 1 multi-LIF observation.
    pub fn set_fused_inner_steps(&mut self, n: u32) -> Result<()> {
        if let Some(ref mut engine) = self.engine {
            engine.set_fused_inner_steps(n);
            Ok(())
        } else {
            bail!("No topology loaded")
        }
    }

    /// Gate G2 — Configure interval-based ensemble snapshot capture on
    /// the wrapped fused engine. When `n > 0`, the engine's `step()`
    /// end-of-step barrier captures an [`crate::fused_engine::EnsembleSnapshot`]
    /// every `n` MD steps tagged
    /// [`crate::fused_engine::SnapshotTrigger::IntervalScheduled`].
    /// When `n == 0` (default), only the existing activity-driven
    /// trigger paths fire. No-op when no topology has been loaded.
    pub fn set_snapshot_interval(&mut self, n: u32) {
        if let Some(ref mut engine) = self.engine {
            engine.set_snapshot_interval(n);
        }
    }

    /// Set the integration timestep. Use 0.004 (4fs) with HMR masses.
    pub fn set_dt(&mut self, dt: f64) -> Result<()> {
        if let Some(ref mut engine) = self.engine {
            engine.set_dt(dt);
            Ok(())
        } else {
            bail!("No topology loaded")
        }
    }

    /// **DEPRECATED**: Configure temperature protocol separately
    ///
    /// Use `set_cryo_uv_protocol()` instead to configure the unified cryo-UV protocol.
    #[deprecated(since = "1.2.0", note = "Use set_cryo_uv_protocol() instead")]
    pub fn set_temperature_protocol(&mut self, protocol: TemperatureProtocol) -> Result<()> {
        if let Some(ref mut engine) = self.engine {
            #[allow(deprecated)]
            engine.set_temperature_protocol(protocol)?;
            Ok(())
        } else {
            bail!("No topology loaded")
        }
    }

    /// **DEPRECATED**: Configure UV probe separately
    ///
    /// Use `set_cryo_uv_protocol()` instead to configure the unified cryo-UV protocol.
    #[deprecated(since = "1.2.0", note = "Use set_cryo_uv_protocol() instead")]
    pub fn set_uv_config(&mut self, config: UvProbeConfig) -> Result<()> {
        if let Some(ref mut engine) = self.engine {
            #[allow(deprecated)]
            engine.set_uv_config(config);
            Ok(())
        } else {
            bail!("No topology loaded")
        }
    }

    /// Run simulation on current topology
    pub fn run(&mut self, n_steps: i32) -> Result<RunSummary> {
        self.autonomous_chunk_clock.invalidate();
        if let Some(ref mut engine) = self.engine {
            let run_start = Instant::now();
            let summary = engine.run(n_steps)?;
            let run_time = run_start.elapsed().as_millis() as u64;

            self.structures_processed += 1;
            self.total_steps_run += n_steps as i64;
            self.total_compute_time_ms += run_time;

            Ok(summary)
        } else {
            bail!("No topology loaded")
        }
    }

    /// Explicit original startup continuation; a partial result is not a
    /// processed structure or completed run. Runtime fragments are retained in
    /// the same typed pending-group owner and charged only once at completion.
    pub fn run_may_startup_until(&mut self,cursor:&mut crate::native_run_continuation::MayStartupGroupCursorV1,
        stop_after:Option<u32>)->Result<Option<RunSummary>> {
        self.autonomous_chunk_clock.invalidate();
        let engine=self.engine.as_mut().context("No topology loaded")?;
        let begin=Instant::now();let result=engine.run_may_startup_until(cursor,stop_after)?;
        cursor.accumulated_owner_run_ns=cursor.accumulated_owner_run_ns.checked_add(u64::try_from(begin.elapsed().as_nanos())?)
            .context("startup owner runtime fragments overflow")?;
        if result.is_some(){self.structures_processed+=1;self.total_steps_run+=1;
            self.total_compute_time_ms=self.total_compute_time_ms.checked_add(cursor.accumulated_owner_run_ns/1_000_000)
                .context("startup owner compute accounting overflow")?;}
        Ok(result)
    }
    pub fn fresh_may_startup_cursor(&self)->Result<crate::native_run_continuation::MayStartupGroupCursorV1>{
        self.engine.as_ref().context("No topology loaded")?.fresh_may_startup_cursor()
    }

    /// Run simulation and automatically cluster spike events
    ///
    /// Convenience method that:
    /// 1. Runs the simulation for n_steps
    /// 2. Collects spike events
    /// 3. Clusters them using RT cores (or fallback)
    /// 4. Returns structured binding site information
    ///
    /// # Example
    /// ```ignore
    /// let (summary, sites, stats) = engine.run_and_cluster(1_000_000)?;
    /// for site in &sites {
    ///     let c = site.view(prism_nhs::spatial_view::SpatialView::GeometricVoxelMass).unwrap();
    ///     println!("Binding site at {:?} with {} spikes", c, site.spike_count);
    /// }
    /// ```
    pub fn run_and_cluster(
        &mut self,
        n_steps: i32,
    ) -> Result<(
        RunSummary,
        Vec<ClusteredBindingSite>,
        Option<ClusteringStats>,
    )> {
        let summary = self.run(n_steps)?;
        let spike_events = self.get_spike_events();

        if spike_events.is_empty() {
            return Ok((summary, Vec::new(), None));
        }

        // Extract positions
        let positions: Vec<f32> = spike_events
            .iter()
            .flat_map(|s| s.position.iter().copied())
            .collect();

        // Cluster using RT cores or fallback
        let used_rt = self.has_rt_clustering();
        let result = self.cluster_spikes(&positions)?;
        // Phase 2: route the canonical site-construction boundary through
        // the `ClusteringToClusteredSites` transform so the GVM-populated
        // (L1) and emission-compat-matches-GVM (L2) laws are enforced on
        // every output of this path. An abort-routed violation here is a
        // structural-corruption failure of the construction invariant and
        // must surface as a run error, not a silent fallback.
        #[cfg(feature = "gpu")]
        let sites = {
            use crate::transform::clustering_to_clustered_sites::{
                ClusteringInput, ClusteringToClusteredSites,
            };
            use crate::transform::AuditedTransform;
            let outcome = ClusteringToClusteredSites::new().apply(ClusteringInput {
                spike_events: &spike_events,
                clustering_result: &result,
                engine: self.engine.as_ref(),
            });
            let (sites, quarantined) = outcome
                .into_result()
                .map_err(|aborted| anyhow::anyhow!("{aborted}"))?;
            if !quarantined.is_empty() {
                log::warn!(
                    "transform clustering_to_clustered_sites: {} quarantined site(s) in run_and_cluster",
                    quarantined.len()
                );
            }
            sites
        };
        #[cfg(not(feature = "gpu"))]
        let sites = build_clustered_sites(&spike_events, &result, None);
        let stats = ClusteringStats {
            num_clusters: result.num_clusters,
            total_neighbors: result.total_neighbors,
            gpu_time_ms: result.gpu_time_ms,
            used_rt_cores: used_rt,
        };

        Ok((summary, sites, Some(stats)))
    }

    /// Get spike events from current run
    pub fn get_spike_events(&self) -> Vec<SpikeEvent> {
        if let Some(ref engine) = self.engine {
            engine.get_spike_events().to_vec()
        } else {
            Vec::new()
        }
    }

    /// Enable spike accumulation for analysis
    ///
    /// When enabled, spikes are downloaded from the GPU and accumulated
    /// across sync intervals. Use `get_accumulated_spikes()` to retrieve.
    pub fn set_spike_accumulation(&mut self, enabled: bool) {
        if let Some(ref mut engine) = self.engine {
            engine.set_spike_accumulation(enabled);
        }
    }

    /// Adapt internal protocol parameters based on measured cold-hold spike rate.
    ///
    /// Delegates to `NhsAmberFusedEngine::adapt_protocol_from_spike_rate()`.
    /// Returns the detected `FlexibilityClass`.
    pub fn adapt_protocol_from_spike_rate(
        &mut self,
        cold_hold_steps: i32,
    ) -> crate::fused_engine::FlexibilityClass {
        if let Some(ref mut engine) = self.engine {
            engine.adapt_protocol_from_spike_rate(cold_hold_steps)
        } else {
            log::warn!("adapt_protocol_from_spike_rate called with no engine loaded");
            crate::fused_engine::FlexibilityClass::Normal
        }
    }

    /// Get accumulated spike events (GPU format with timestamps)
    ///
    /// Returns all spike events accumulated since spike accumulation was enabled.
    /// Only populated when spike accumulation is enabled via `set_spike_accumulation(true)`.
    pub fn get_accumulated_spikes(&self) -> Vec<crate::fused_engine::GpuSpikeEvent> {
        if let Some(ref engine) = self.engine {
            engine.get_accumulated_spikes().to_vec()
        } else {
            Vec::new()
        }
    }

    /// Borrow the complete append-only native archive; existing cloning getter
    /// retains its nondestructive public semantics for other callers.
    pub fn borrow_accumulated_spikes(&self)->&[crate::fused_engine::GpuSpikeEvent] {
        self.engine.as_ref().map_or(&[],|engine|engine.get_accumulated_spikes())
    }

    /// Move all ordered native records only at the successful terminal owner
    /// boundary, after final capture and PrismCode/engine observations.
    pub fn take_accumulated_spikes_for_terminal_handoff(&mut self)->Result<Vec<crate::fused_engine::GpuSpikeEvent>> {
        Ok(self.engine.as_mut().context("terminal spike handoff has no engine")?
            .take_accumulated_spikes_for_terminal_handoff())
    }

    // ── PACKAGE C: spike clock provenance passthrough ───────────────────
    //
    // Callers that reach the engine only through this wrapper could otherwise
    // not declare or read the spike clock at all — the accessor exists but is
    // unreachable, which is the "built, never activated" failure this repair
    // is about.

    /// Declare the emitting stream identity stamped into every spike record.
    /// Returns false when no engine is loaded, so a silent no-op cannot be
    /// mistaken for a successful declaration.
    pub fn set_spike_stream_id(&mut self, stream_id: u32) -> bool {
        match self.engine {
            Some(ref mut engine) => {
                engine.set_spike_stream_id(stream_id);
                true
            }
            None => {
                log::warn!("set_spike_stream_id called with no engine loaded");
                false
            }
        }
    }

    // NOTE: there are deliberately no `global_md_step()` /
    // `cumulative_simulated_time_ps()` wrappers here. They forwarded to engine
    // accessors that existed only for the reverted 128 B spike record; the
    // sealed 96 B packet has no device clock to read, and the engine methods
    // are gone. The global MD clock is stamped by the per-step governor
    // (`coupled_chronology::GearboxClock`) into the `.ccr` sidecar, which is
    // where a caller must go for it. Re-adding a wrapper here that returned a
    // host step counter would present a host guess as a device measurement.

    /// Machine-readable statement of what the spike clock currently binds.
    pub fn spike_clock_identity_json(&self) -> Option<serde_json::Value> {
        self.engine.as_ref().map(|e| e.spike_clock_identity_json())
    }

    /// Download signal preservation buffers from the GPU engine
    pub fn download_signal_preservation(
        &mut self,
    ) -> anyhow::Result<crate::fused_engine::SignalPreservationData> {
        if let Some(ref mut engine) = self.engine {
            engine.download_signal_preservation()
        } else {
            anyhow::bail!("No engine initialized")
        }
    }

    /// Compute and download KCC v2-full descriptors from GPU
    pub fn compute_and_download_kcc(&mut self) -> anyhow::Result<crate::fused_engine::KccData> {
        if let Some(ref mut engine) = self.engine {
            engine.compute_and_download_kcc()
        } else {
            Ok(crate::fused_engine::KccData::empty())
        }
    }

    /// Read the unmodified native KCC outputs after the composed KCC→V2 replay.
    /// The overlay consumes KCC lead data and does not update its accumulators.
    /// No producer or descriptor recomputation occurs during this readback.
    #[cfg(feature = "v2_ignition")]
    pub fn read_composed_kcc_endpoint(
        &mut self, completed_step: u32, includes_update: bool,
    ) -> anyhow::Result<crate::fused_engine::KccData> {
        anyhow::ensure!(self.device_protocol_current_step()? == completed_step,
            "composed KCC/V2 endpoint differs from physical clock");
        let engine = self.engine.as_mut().context("KCC readback has no engine")?;
        let data = engine.download_kcc_descriptors_and_support()?;
        if includes_update { engine.record_kcc_protocol_update_endpoint(completed_step); }
        engine.record_kcc_descriptor_endpoint(Some(u64::from(completed_step)));
        anyhow::ensure!(self.device_protocol_current_step()? == completed_step,
            "composed KCC/V2 observer advanced physical MD");
        log::info!("[KCC-V2-ENDPOINT-REPLAY] completed_step={} graph_launches=1 residues={} update_in_graph={} archive_readback_complete=true md_steps_advanced=0",
            completed_step, data.n_residues, includes_update);
        Ok(data)
    }

    /// Launch the KCC residue update kernel ONCE outside the captured CUDA Graph.
    #[cfg(feature = "v2_ignition")]
    pub fn prepare_kcc_causal_candidates(&self, candidates: &[Vec<u32>])
        -> anyhow::Result<crate::captured_pipeline::KccCausalCandidates> {
        self.engine.as_ref().ok_or_else(|| anyhow::anyhow!("No engine initialized"))?
            .prepare_kcc_causal_candidates(candidates)
    }

    #[cfg(feature = "v2_ignition")]
    pub fn select_kcc_causal_leads(&self, candidates: &crate::captured_pipeline::KccCausalCandidates,
        leads: &mut cudarc::driver::CudaSlice<u32>) -> anyhow::Result<()> {
        self.engine.as_ref().ok_or_else(|| anyhow::anyhow!("No engine initialized"))?
            .select_kcc_causal_leads(candidates, leads)
    }

    pub fn compute_and_download_kcc_at(&mut self, completed_step: u64)
        -> anyhow::Result<crate::fused_engine::KccData> {
        self.engine.as_mut().ok_or_else(|| anyhow::anyhow!("No engine initialized"))?
            .compute_and_download_kcc_at(completed_step)
    }

    #[cfg(feature = "v2_ignition")]
    pub fn capture_kcc_endpoint_template(&mut self,
        candidates: &crate::captured_pipeline::KccCausalCandidates,
        leads: &mut cudarc::driver::CudaSlice<u32>)
        -> anyhow::Result<crate::graph_capture::CapturedTemplate> {
        self.engine.as_mut().context("KCC capture has no engine")?
            .capture_kcc_endpoint_template(candidates, leads)
    }

    #[cfg(feature = "v2_ignition")]
    pub fn capture_kcc_measurement_template(&mut self,
        candidates: &crate::captured_pipeline::KccCausalCandidates,
        leads: &mut cudarc::driver::CudaSlice<u32>)
        -> anyhow::Result<crate::graph_capture::CapturedTemplate> {
        self.engine.as_mut().context("KCC capture has no engine")?
            .capture_kcc_measurement_template(candidates, leads)
    }

    /// Same device-clock producer for endpoints before V2 capture is available.
    pub fn kcc_step_once_protocol(&mut self, completed_step: u32) -> anyhow::Result<()> {
        let engine = self.engine.as_mut().context("KCC update has no engine")?;
        engine.enqueue_kcc_protocol_update()?;
        engine.record_kcc_protocol_update_endpoint(completed_step);
        Ok(())
    }

    /// Full producer graph; descriptor-only callers retain the separate API.
    #[cfg(feature = "v2_ignition")]
    pub fn execute_kcc_measurement_graph(&mut self,
        graph: &crate::graph_capture::AutonomousGraph, completed_step: u32)
        -> anyhow::Result<crate::fused_engine::KccData> {
        self.execute_kcc_endpoint_graph_inner(graph, completed_step, true)
    }

    /// Execute descriptors and selection on the current physical endpoint,
    /// then read its unchanged native support before any subsequent producer.
    #[cfg(feature = "v2_ignition")]
    pub fn execute_kcc_endpoint_graph(&mut self,
        graph: &crate::graph_capture::AutonomousGraph, completed_step: u32)
        -> anyhow::Result<crate::fused_engine::KccData> {
        self.execute_kcc_endpoint_graph_inner(graph, completed_step, false)
    }

    #[cfg(feature = "v2_ignition")]
    fn execute_kcc_endpoint_graph_inner(&mut self,
        graph: &crate::graph_capture::AutonomousGraph, completed_step: u32, includes_update: bool)
        -> anyhow::Result<crate::fused_engine::KccData> {
        // This public route accepts a generic graph; even launch/download/check
        // failure must retire a previously admitted physical-clock endpoint.
        self.autonomous_chunk_clock.invalidate();
        let start = self.device_protocol_current_step()?;
        anyhow::ensure!(start == completed_step,
            "KCC endpoint differs from physical clock: expected {}, observed {}", completed_step, start);
        graph.launch_on_stream(self.stream.cu_stream())?;
        let engine = self.engine.as_mut().context("KCC replay has no engine")?;
        let data = engine.download_kcc_descriptors_and_support()?;
        if includes_update { engine.record_kcc_protocol_update_endpoint(completed_step); }
        engine.record_kcc_descriptor_endpoint(Some(u64::from(completed_step)));
        anyhow::ensure!(self.device_protocol_current_step()? == start,
            "KCC observer graph advanced physical MD");
        log::info!("[KCC-ENDPOINT-REPLAY] completed_step={} graph_launches=1 residues={} update_in_graph={} archive_readback_complete=true md_steps_advanced=0",
            completed_step, data.n_residues, includes_update);
        Ok(data)
    }

    /// Launch the KCC residue update kernel ONCE outside the captured CUDA Graph.
    ///
    /// Used by the autonomous (multi-stream + multi-differential) chunk loop to
    /// drive KCC streaming reductions and ring buffer events between graph replays.
    /// The captured graph (`step_autonomous_kernels`) does not include KCC because
    /// the kernel needs a fresh `current_step` per call (the ring buffer event
    /// coalescer would otherwise collapse all replays into a single ring slot).
    /// See `NhsAmberFusedEngine::kcc_step_once` for the rationale and design.
    pub fn kcc_step_once(&mut self, current_step: u32, current_phase: i32) -> anyhow::Result<()> {
        if let Some(ref mut engine) = self.engine {
            engine.kcc_step_once(current_step, current_phase)
        } else {
            Ok(())
        }
    }

    /// Stage 2: Write the closed-loop ASC steering focus list into the
    /// device-side ProtocolState.
    ///
    /// `entries` is a slice of `(residue_id, weight)` tuples derived from the
    /// per-chunk GC-PID synergy fraction (Stage 1B-3). At most
    /// `STEERING_FOCUS_MAX = 64` entries are written; any extra are ignored.
    /// The host serializes the new `steering_focus_count` + the array bytes
    /// into a small staging buffer and issues a `memcpy_htod` at the
    /// appropriate offset within ProtocolState. The autonomous CUDA Graph
    /// has captured the *pointer* to ProtocolState; the next graph replay
    /// (and the next launch of `ring_buffer_read_and_adapt`) reads the new
    /// contents automatically — no graph re-capture, no kernel re-launch
    /// configuration change.
    ///
    /// This is the canonical "captured pointer with mutable contents"
    /// pattern, identical to how the existing legacy steering hooks
    /// (`steering_uv_boost`, etc.) are updated. The only difference is the
    /// payload size (516 bytes instead of 4-16 bytes) and the offset within
    /// the struct.
    ///
    /// Used by the autonomous chunk loop in `nhs_rt_full.rs` when
    /// `--closed-loop-steering` is enabled. Without the flag, the
    /// `steering_focus_count` stays at zero and the kernel's per-spike
    /// steering branch is a no-op (the `if (n_focus > 0)` guard short-
    /// circuits) so this method has no effect on default behavior.
    pub fn write_steering_focus(
        &mut self,
        stream: &Arc<CudaStream>,
        entries: &[(i32, f32)],
    ) -> anyhow::Result<()> {
        use crate::protocol_state::{ProtocolState, SteerEntry, STEERING_FOCUS_MAX};

        if !Arc::ptr_eq(stream, &self.stream) {
            anyhow::bail!(
                "write_steering_focus requires the PersistentNhsEngine-owned CUDA stream"
            );
        }

        // Strategy: download the full ProtocolState (684 bytes), update only
        // the steering fields in the host copy, upload the full struct back.
        // The full-struct round trip is ~1.4 KB of PCIe traffic per chunk
        // (~80 chunks per run = ~110 KB total) — negligible compared to the
        // multi-GB spike data downloads. Avoiding partial writes also dodges
        // a cudarc slice-typing issue with `memcpy_htod` on a `CudaView`.
        //
        // We could optimize this to a partial write at the steering field
        // offset using `slice_mut(..).into()` once the cudarc trait
        // resolution is stable, but for Stage 2 v1 the round-trip approach
        // is correct, simple, and fast enough.
        if let Some(ref mut engine) = self.engine {
            // Download
            let mut buf = vec![0u8; std::mem::size_of::<ProtocolState>()];
            stream.memcpy_dtoh(&engine.d_protocol_state, &mut buf)?;
            engine.synchronize_stream_raw("ProtocolState steering readback completion")?;
            // Reinterpret as ProtocolState (matching the device-side layout
            // exactly via #[repr(C)]).
            let mut state: ProtocolState =
                unsafe { std::ptr::read_unaligned(buf.as_ptr() as *const ProtocolState) };

            // Update only the steering fields
            let n_active = entries.len().min(STEERING_FOCUS_MAX) as i32;
            state.steering_focus_count = n_active;
            for i in 0..STEERING_FOCUS_MAX {
                state.steering_focus_residues[i] = if i < entries.len() {
                    SteerEntry {
                        residue_id: entries[i].0,
                        weight: entries[i].1,
                    }
                } else {
                    SteerEntry {
                        residue_id: -1,
                        weight: 0.0,
                    }
                };
            }

            // Upload the modified struct back
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    &state as *const ProtocolState as *const u8,
                    std::mem::size_of::<ProtocolState>(),
                )
            };
            stream.memcpy_htod(bytes, &mut engine.d_protocol_state)?;
        }
        Ok(())
    }

    /// Wall-bound phase scheduler (operator directive 2026-06-15): host-drive the
    /// CCNS phase boundaries in device ProtocolState. The Director kernel
    /// (protocol_director.cu) computes phase + temperature purely from
    /// `current_step` vs the cumulative boundaries `cold_hold_end < ramp_end <
    /// warm_hold_end < ramp_down_end` (cold_return = after ramp_down_end). By
    /// rewriting those four boundaries each chunk — driven by
    /// `WallBoundPhaseController` (wall-clock envelopes + canonical science-codec
    /// events) — the host moves the phase transitions WITHOUT touching the kernel
    /// or the captured graph body. Same proven full-struct round-trip as
    /// `write_steering_focus` (partial `memcpy_htod` on a CudaView trips a cudarc
    /// typing issue; the ~1.4 KB round-trip per chunk is negligible).
    ///
    /// Boundary semantics for event-driven control:
    /// - Set a phase's `end` to a large value (≥ total_steps) so step-count alone
    ///   never advances it; the host advances by snapping `end = current_step`
    ///   when the controller fires the transition.
    /// - For the ACTIVE ramp phase, set its `end` to the wall-envelope-expected
    ///   step (start + live_steps_per_sec × phase_wall_seconds) so the Director's
    ///   step-fraction interpolation tracks wall-time; snap to `current_step` on
    ///   the canonical event. (steps/sec ≈ constant ⇒ step-fraction ≈ wall-fraction.)
    /// Invariant enforced: cold_hold_end ≤ ramp_end ≤ warm_hold_end ≤ ramp_down_end.
    pub fn write_phase_boundaries(
        &mut self,
        stream: &Arc<CudaStream>,
        cold_hold_end: i32,
        ramp_end: i32,
        warm_hold_end: i32,
        ramp_down_end: i32,
    ) -> anyhow::Result<()> {
        use crate::protocol_state::ProtocolState;
        if !Arc::ptr_eq(stream, &self.stream) {
            anyhow::bail!(
                "write_phase_boundaries requires the PersistentNhsEngine-owned CUDA stream"
            );
        }
        if let Some(ref mut engine) = self.engine {
            let mut buf = vec![0u8; std::mem::size_of::<ProtocolState>()];
            stream.memcpy_dtoh(&engine.d_protocol_state, &mut buf)?;
            engine.synchronize_stream_raw("ProtocolState phase-boundary readback completion")?;
            let mut state: ProtocolState =
                unsafe { std::ptr::read_unaligned(buf.as_ptr() as *const ProtocolState) };
            // Enforce monotonic non-decreasing cumulative boundaries (the Director
            // assumes cold_hold_end ≤ ramp_end ≤ warm_hold_end ≤ ramp_down_end).
            let che = cold_hold_end.max(0);
            let re = ramp_end.max(che);
            let whe = warm_hold_end.max(re);
            let rde = ramp_down_end.max(whe);
            state.cold_hold_end = che;
            state.ramp_end = re;
            state.warm_hold_end = whe;
            state.ramp_down_end = rde;
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    &state as *const ProtocolState as *const u8,
                    std::mem::size_of::<ProtocolState>(),
                )
            };
            stream.memcpy_htod(bytes, &mut engine.d_protocol_state)?;
        }
        Ok(())
    }

    /// Stage 2 calibration: download the full ProtocolState from the engine's
    /// device buffer. Used by the chunk loop's end-of-run summary to read the
    /// `focus_match_count` counter that the kernel atomicAdd's whenever a
    /// spike's primary residue id matches an active focus list entry.
    ///
    /// Synchronizes the stream before reading so the dtoh copy sees the
    /// final value (the kernel atomicAdds run on the same stream as the
    /// graph replay, but PCIe traffic from a different stream still needs
    /// the explicit sync).
    pub fn download_protocol_state(
        &self,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<crate::protocol_state::ProtocolState> {
        use crate::protocol_state::ProtocolState;
        if !Arc::ptr_eq(stream, &self.stream) {
            anyhow::bail!(
                "download_protocol_state requires the PersistentNhsEngine-owned CUDA stream"
            );
        }
        if let Some(ref engine) = self.engine {
            let mut buf = vec![0u8; std::mem::size_of::<ProtocolState>()];
            stream.memcpy_dtoh(&engine.d_protocol_state, &mut buf)?;
            engine.synchronize_stream_raw("ProtocolState final download completion")?;
            let state: ProtocolState =
                unsafe { std::ptr::read_unaligned(buf.as_ptr() as *const ProtocolState) };
            Ok(state)
        } else {
            anyhow::bail!("download_protocol_state: engine not initialized")
        }
    }

    /// P2 — MEASURED proof that no aromatic was ever excited in this run.
    ///
    /// Delegates to the owned fused engine, which reads five device planes.
    /// FAIL-CLOSED: with no engine there is nothing to measure, so this returns
    /// an error rather than an empty "clean" verdict — an absent instrument must
    /// never read as a proof of absence.
    pub fn uv_suppression_witness(&self) -> anyhow::Result<serde_json::Value> {
        if let Some(ref engine) = self.engine {
            engine.uv_suppression_witness()
        } else {
            anyhow::bail!(
                "uv_suppression_witness: engine not initialized; the UV-off control \
                 cannot be measured and must not be certified"
            )
        }
    }

    /// Get snapshots from current run
    pub fn get_snapshots(&self) -> Vec<EnsembleSnapshot> {
        if let Some(ref engine) = self.engine {
            engine.get_ensemble_snapshots().to_vec()
        } else {
            Vec::new()
        }
    }

    /// Reset engine state for a new replica
    ///
    /// Clears accumulated spikes, re-initializes velocities with a new seed,
    /// and resets simulation counters. Topology and forces are preserved.
    ///
    /// # Arguments
    /// * `seed` - Random seed for velocity initialization (Maxwell-Boltzmann)
    pub fn reset_for_replica(&mut self, seed: u64) -> Result<()> {
        self.autonomous_chunk_clock.invalidate();
        if let Some(ref mut engine) = self.engine {
            engine.reset_for_replica(seed)
        } else {
            bail!("No topology loaded")
        }
    }

    /// Get current positions
    pub fn get_positions(&self) -> Result<Vec<f32>> {
        if let Some(ref engine) = self.engine {
            engine.get_positions()
        } else {
            bail!("No topology loaded")
        }
    }

    /// Enable UltimateEngine for 2-4x faster MD simulation
    ///
    /// Requires SM86+ GPU (Ampere/Ada/Blackwell) and topology to be loaded.
    /// Uses mixed-precision, SoA layout, and other hyperoptimizations.
    ///
    /// Note: Must be called after load_topology() with the same topology.
    pub fn enable_ultimate_mode(&mut self, topology: &PrismPrepTopology) -> Result<()> {
        if let Some(ref mut engine) = self.engine {
            engine.enable_ultimate_mode(topology)?;
            log::info!("✓ UltimateEngine enabled (2-4x faster MD)");
            Ok(())
        } else {
            bail!("Engine not initialized - call load_topology() first")
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // RT-ACCELERATED CLUSTERING
    // ═══════════════════════════════════════════════════════════════════════════════

    /// Check if RT cores are available for accelerated clustering
    pub fn has_rt_clustering(&self) -> bool {
        crate::rt_utils::has_rt_cores() && crate::rt_utils::is_optix_available()
    }

    /// Ensure the RT clustering pipeline is initialized (lazy init)
    ///
    /// Call this explicitly to pre-warm the pipeline, or let it initialize
    /// lazily on first `cluster_spikes()` call.
    #[allow(unreachable_code)]
    pub fn ensure_rt_pipeline(&mut self) -> Result<bool> {
        if self.rt_engine.is_some() {
            return Ok(true); // Already initialized
        }

        // ════════════════════════════════════════════════════════════
        // E2E UNBLOCK LANE (2026-04-21): OptiX RT clustering disabled.
        // rt_clustering.rs::load_pipeline() at lines 185-246 is
        // structurally incomplete — it never assigns self.pipeline and
        // never constructs a Pipeline, yet returns Ok(()). Routing
        // through it produced `WARN Clustering failed: Pipeline not
        // loaded` followed by SIGSEGV on engine shutdown. Per
        // CLAUDE.md IMMUTABLE_RULE #9 ("OptiX is removed"), we
        // short-circuit here so cluster_spikes falls through to
        // fallback_grid_cluster. No OptiX context is created,
        // eliminating the teardown crash path. To re-enable a future
        // working OptiX path, delete the next two lines.
        // ════════════════════════════════════════════════════════════
        log::info!("RT clustering disabled (OptiX path unavailable); using grid fallback");
        return Ok(false);

        if !self.has_rt_clustering() {
            log::debug!("RT clustering not available (no RT cores or OptiX)");
            return Ok(false);
        }

        log::info!("🔷 Initializing OptiX RT pipeline for clustering...");
        let start = Instant::now();

        // Find the OptiX IR file
        let optixir_path = crate::rt_clustering::find_optixir_path()
            .context("Could not find rt_clustering.optixir")?;

        // Adaptive epsilon: scale inversely with cube root of atom count.
        // Larger proteins need tighter clustering to resolve distinct pockets.
        //   ≤500 atoms  → 3.0 Å  (small, e.g. crambin)
        //   ~2000 atoms → 2.0 Å  (medium)
        //   ~4000 atoms → 1.5 Å  (large, e.g. TEM-1)
        //   ~8000 atoms → 1.2 Å  (very large)
        // Formula: epsilon = 3.0 * (500 / n_atoms)^(1/3), clamped to [1.2, 3.0]
        let n_atoms = self.engine.as_ref().map(|e| e.n_atoms()).unwrap_or(500);
        let adaptive_epsilon = (3.0_f32 * (500.0_f32 / n_atoms as f32).cbrt()).clamp(1.2, 3.0);
        log::info!(
            "  Adaptive epsilon: {:.2}Å (n_atoms={})",
            adaptive_epsilon,
            n_atoms
        );

        // Create RT clustering config with adaptive epsilon
        let rt_config = crate::rt_clustering::RtClusteringConfig {
            epsilon: adaptive_epsilon,
            min_points: 4,        // Minimum 4 neighbors for core point
            min_cluster_size: 20, // Minimum 20 points per cluster (catch smaller pockets)
            rays_per_event: 32,   // 32 rays for neighbor finding
        };

        // Create and initialize the RT engine
        let mut rt_engine =
            crate::rt_clustering::RtClusteringEngine::new(self.context.clone(), rt_config)
                .context("Failed to create RT clustering engine")?;

        rt_engine
            .load_pipeline(&optixir_path)
            .context("Failed to load RT clustering pipeline")?;

        let init_time = start.elapsed().as_millis() as u64;
        self.rt_init_time_ms = Some(init_time);
        self.rt_engine = Some(rt_engine);

        log::info!("  RT pipeline initialized: {}ms", init_time);
        log::info!(
            "  GPU Architecture: {}",
            crate::rt_utils::get_architecture_name()
        );

        Ok(true)
    }

    /// Cluster spike positions using RT-accelerated spatial queries
    ///
    /// Falls back to grid-based clustering if RT cores are unavailable.
    ///
    /// # Arguments
    /// * `spike_positions` - Flat array of [x, y, z, x, y, z, ...] coordinates
    ///
    /// # Returns
    /// Clustering result with cluster assignments and statistics
    pub fn cluster_spikes(
        &mut self,
        spike_positions: &[f32],
    ) -> Result<crate::rt_clustering::RtClusteringResult> {
        use crate::gpu_cluster_backend as gcb;
        let num_spikes = spike_positions.len() / 3;

        // Resolve auto → concrete backend. On SM120 OptiX is disabled
        // (IMMUTABLE_RULE #9), so auto maps to gpu-hash.
        let is_sm120 = !crate::rt_utils::is_optix_available();
        let backend = gcb::resolve_auto(gcb::current_selection(), is_sm120);

        log::info!(
            "POST_MD_CLUSTER_BACKEND_SELECTED backend={} spikes={}",
            gcb::backend_name(backend),
            num_spikes
        );
        log::info!("POST_MD_CLUSTER_START spikes={}", num_spikes);
        let t_dispatch = std::time::Instant::now();

        let result = match backend {
            gcb::BACKEND_GPU_HASH => {
                // Lazy-init the GPU CCL backend once per engine
                if self.gpu_cluster.is_none() {
                    let b =
                        gcb::GpuSpatialHashBackend::new(self.context.clone(), self.stream.clone())
                            .context("init GpuSpatialHashBackend")?;
                    self.gpu_cluster = Some(b);
                }
                // Fixed epsilon semantics match the legacy fallback_grid_cluster.
                // The 5.0 Å value is the established post-MD CCL radius; changing
                // it would be semantic drift and is forbidden in this lane.
                let epsilon = 5.0_f32;
                self.gpu_cluster
                    .as_mut()
                    .unwrap()
                    .cluster(spike_positions, epsilon)?
            }
            gcb::BACKEND_RT_OPTIX => {
                if self.ensure_rt_pipeline()? {
                    if let Some(ref mut rt_engine) = self.rt_engine {
                        log::debug!("Using RT-accelerated clustering for {} spikes", num_spikes);
                        rt_engine.cluster(spike_positions)?
                    } else {
                        anyhow::bail!(
                            "--clustering-backend=optix requested but rt_engine is None; \
                             OptiX unavailable on this device (IMMUTABLE RULE #9 — use auto or gpu-hash)"
                        );
                    }
                } else {
                    anyhow::bail!(
                        "--clustering-backend=optix requested but OptiX is disabled on this device; \
                         use --clustering-backend=auto or --clustering-backend=gpu-hash"
                    );
                }
            }
            gcb::BACKEND_LBVH => {
                anyhow::bail!(
                    "--clustering-backend=lbvh selected but LBVH backend is not yet implemented \
                     (arrives in the Phase-2 LBVH lane). Use auto or gpu-hash."
                );
            }
            gcb::BACKEND_GHOST_LATTICE_4D => {
                // The lattice operates on Ghost v2 records, not raw spike
                // positions. cluster_spikes() only has access to positions,
                // so route the operator to the proper entry point.
                anyhow::bail!(
                    "--clustering-backend=ghost-phase-lattice-4d operates on \
                     Ghost v2 records, not raw spike positions. Invoke \
                     PersistentNhsEngine::cluster_ghost_phase_lattice() \
                     with projected ghost nodes (see \
                     crate::ghost_phase_lattice::load_verified_ghost_segments_from_dir), \
                     or call this from the post-run materializer in \
                     `prism_materialize_sites`. Use \
                     --clustering-backend=auto to fall back to gpu-hash for \
                     spike-only clustering."
                );
            }
            gcb::BACKEND_GRID_DEBUG => {
                log::warn!(
                    "POST_MD_CLUSTER_BACKEND_DEBUG grid fallback active (debug only — degrades at N>>1M)"
                );
                log::debug!("Using fallback grid clustering for {} spikes", num_spikes);
                self.fallback_grid_cluster(spike_positions)?
            }
            other => {
                anyhow::bail!("Unresolved clustering backend code: {}", other);
            }
        };

        log::info!(
            "POST_MD_CLUSTER_DONE clusters={} elapsed_ms={}",
            result.num_clusters,
            t_dispatch.elapsed().as_millis()
        );

        Ok(result)
    }

    /// Run the GhostPhaseLattice4D backend against a pre-projected slice of
    /// [`crate::ghost_phase_lattice::GhostPhaseLatticeNode`]s. The caller is
    /// responsible for projecting Ghost v2 records (see
    /// [`crate::ghost_phase_lattice::load_verified_ghost_segments_from_dir`])
    /// before invoking this method.
    ///
    /// This is the canonical post-MD clustering entry point when
    /// `--clustering-backend=ghost-phase-lattice-4d` is selected, or when
    /// `--clustering-backend=auto` resolves to the lattice (Ghost v2
    /// records present). Returns the full
    /// [`crate::ghost_phase_lattice::GhostPhaseLatticeOutcome`] — the
    /// caller materialises sites via
    /// [`crate::ghost_phase_materializer::materialize_components`].
    pub fn cluster_ghost_phase_lattice(
        &mut self,
        nodes: &[crate::ghost_phase_lattice::GhostPhaseLatticeNode],
        config: crate::ghost_phase_lattice::GhostPhaseLatticeConfig,
    ) -> Result<crate::ghost_phase_lattice::GhostPhaseLatticeOutcome> {
        log::info!(
            "POST_MD_CLUSTER_BACKEND_SELECTED backend=ghost_phase_lattice_4d tiles={} \
             cell={:.1}A max_temporal={} so3_thresh={:.2}",
            nodes.len(),
            config.spatial_cell_size_a,
            config.max_temporal_edge_steps,
            config.so3_threshold
        );
        log::info!("POST_MD_CLUSTER_START tiles={} (lattice)", nodes.len());
        let t = std::time::Instant::now();

        if self.ghost_lattice.is_none() {
            self.ghost_lattice = Some(crate::ghost_phase_lattice::GhostPhaseLattice4D::new(
                self.context.clone(),
                self.stream.clone(),
                config,
            ));
        }
        let backend = self
            .ghost_lattice
            .as_mut()
            .expect("lattice backend just initialised");
        let outcome = backend
            .cluster(nodes)
            .context("ghost-phase-lattice cluster")?;
        log::info!(
            "POST_MD_CLUSTER_DONE backend=ghost_phase_lattice_4d components={} \
             pairs={} edges={} elapsed_ms={}",
            outcome.components.len(),
            outcome.stats.n_pairs_evaluated,
            outcome.stats.n_directed_edges,
            t.elapsed().as_millis()
        );
        Ok(outcome)
    }

    /// Re-cluster spikes at a specific epsilon (for mega-cluster subdivision).
    /// Temporarily overrides the RT engine's epsilon, clusters, then restores it.
    pub fn cluster_spikes_at_epsilon(
        &mut self,
        spike_positions: &[f32],
        epsilon: f32,
    ) -> Result<crate::rt_clustering::RtClusteringResult> {
        if let Some(ref mut rt_engine) = self.rt_engine {
            let saved = rt_engine.config.epsilon;
            rt_engine.config.epsilon = epsilon;
            let result = rt_engine.cluster(spike_positions);
            rt_engine.config.epsilon = saved;
            result
        } else {
            anyhow::bail!("RT clustering engine not initialized")
        }
    }

    /// Get the current adaptive epsilon value
    pub fn current_epsilon(&self) -> Option<f32> {
        self.rt_engine.as_ref().map(|e| e.config.epsilon)
    }

    /// Multi-scale clustering for robust, structure-agnostic binding site detection
    ///
    /// Runs DBSCAN clustering at multiple epsilon values and tracks cluster persistence
    /// across scales. Clusters that appear at multiple scales are more likely to be
    /// real binding sites rather than noise.
    ///
    /// # Algorithm
    /// Compute adaptive epsilon values from k-NN distance distribution
    ///
    /// Samples spike positions and computes the k-th nearest neighbor distance
    /// for each sample. Returns epsilon values at key percentiles that capture
    /// the natural clustering scales in the data.
    ///
    /// # Arguments
    /// * `positions` - Flat array of [x, y, z, x, y, z, ...] coordinates
    /// * `k` - Number of nearest neighbors to consider (default: 4 for DBSCAN min_points)
    /// * `sample_size` - Number of points to sample (default: 1000)
    ///
    /// # Returns
    /// Vector of epsilon values at 25th, 50th, 75th, and 90th percentiles
    pub fn compute_adaptive_epsilon(positions: &[f32], k: usize, sample_size: usize) -> Vec<f32> {
        let n_points = positions.len() / 3;
        if n_points < k + 1 {
            // Not enough points, return default
            return vec![5.0, 7.0, 10.0, 14.0];
        }

        // Sample points (evenly spaced or random)
        let sample_indices: Vec<usize> = if n_points <= sample_size {
            (0..n_points).collect()
        } else {
            let step = n_points / sample_size;
            (0..sample_size).map(|i| i * step).collect()
        };

        // Compute k-NN distance for each sampled point
        let mut knn_distances: Vec<f32> = Vec::with_capacity(sample_indices.len());

        for &i in &sample_indices {
            let xi = positions[i * 3];
            let yi = positions[i * 3 + 1];
            let zi = positions[i * 3 + 2];

            // Compute distances to all other points (brute force for simplicity)
            let mut distances: Vec<f32> = Vec::with_capacity(n_points.min(1000));
            for j in 0..n_points.min(5000) {
                // Limit comparison for performance
                if i == j {
                    continue;
                }
                let xj = positions[j * 3];
                let yj = positions[j * 3 + 1];
                let zj = positions[j * 3 + 2];
                let d = ((xi - xj).powi(2) + (yi - yj).powi(2) + (zi - zj).powi(2)).sqrt();
                distances.push(d);
            }

            // Get k-th smallest distance
            if distances.len() >= k {
                distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                knn_distances.push(distances[k - 1]);
            }
        }

        if knn_distances.is_empty() {
            return vec![5.0, 7.0, 10.0, 14.0];
        }

        // Sort k-NN distances
        knn_distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Extract percentiles: 25th, 50th, 75th, 90th
        let n = knn_distances.len();
        let p25 = knn_distances[n / 4].clamp(3.0, 8.0);
        let p50 = knn_distances[n / 2].clamp(5.0, 12.0);
        let p75 = knn_distances[3 * n / 4].clamp(7.0, 18.0);
        let p90 = knn_distances[9 * n / 10].clamp(10.0, 25.0);

        // Ensure monotonically increasing and minimum spacing
        let mut epsilons = vec![p25];
        for &e in &[p50, p75, p90] {
            if e > epsilons.last().unwrap() + 1.5 {
                epsilons.push(e);
            }
        }

        // Ensure we have at least 3 scales
        while epsilons.len() < 3 {
            let last = *epsilons.last().unwrap();
            epsilons.push((last * 1.4).min(25.0));
        }

        log::info!(
            "Adaptive epsilon: {:?} (from k-NN analysis, k={})",
            epsilons,
            k
        );
        epsilons
    }

    /// Multi-scale clustering with automatic or fixed epsilon selection
    ///
    /// 1. Run clustering at epsilon values: [5.0, 7.0, 10.0, 14.0] Angstroms
    /// 2. For each scale, compute cluster centroids
    /// 3. Find clusters that "persist" (have similar centroids) across ≥2 scales
    /// 4. Merge overlapping clusters and assign persistence scores
    /// 5. Return clusters sorted by persistence × spike_count
    ///
    /// # Arguments
    /// * `spike_positions` - Flat array of [x, y, z, x, y, z, ...] coordinates
    ///
    /// # Returns
    /// MultiScaleClusteringResult with persistence-scored clusters
    pub fn multi_scale_cluster_spikes(
        &mut self,
        spike_positions: &[f32],
    ) -> Result<MultiScaleClusteringResult> {
        self.multi_scale_cluster_spikes_with_epsilon(spike_positions, None)
    }

    /// Multi-scale clustering with optional custom epsilon values
    ///
    /// # Arguments
    /// * `spike_positions` - Flat array of [x, y, z, x, y, z, ...] coordinates
    /// * `custom_epsilon` - If Some, use these values; if None, use adaptive selection
    pub fn multi_scale_cluster_spikes_with_epsilon(
        &mut self,
        spike_positions: &[f32],
        custom_epsilon: Option<Vec<f32>>,
    ) -> Result<MultiScaleClusteringResult> {
        use std::collections::HashMap;

        let num_spikes = spike_positions.len() / 3;

        // Determine epsilon values: custom, adaptive, or default
        // Track whether adaptive was used for output
        let knn_k = 4usize;
        let sample_size = 1000usize;
        let (epsilon_scales, is_adaptive, actual_sample_size): (Vec<f32>, bool, Option<usize>) =
            if let Some(eps) = custom_epsilon {
                log::info!("Using fixed epsilon values: {:?}", eps);
                (eps, false, None)
            } else if num_spikes > 1000 {
                // Use adaptive selection for large datasets
                let actual_samples = num_spikes.min(sample_size);
                let eps = Self::compute_adaptive_epsilon(spike_positions, knn_k, sample_size);
                (eps, true, Some(actual_samples))
            } else {
                // Default values for small datasets
                log::info!("Using default epsilon values (small dataset): [5.0, 7.0, 10.0, 14.0]");
                (vec![5.0, 7.0, 10.0, 14.0], false, None)
            };
        let merge_distance = 12.0f32; // Clusters within 8Å are considered the same site

        log::info!(
            "Running multi-scale clustering on {} spikes at {} scales",
            num_spikes,
            epsilon_scales.len()
        );

        // Track clusters across scales: (centroid, spike_count, epsilon, cluster_id)
        let mut all_clusters: Vec<ScaleCluster> = Vec::new();

        // Run clustering at each scale
        for &epsilon in &epsilon_scales {
            // Create RT engine with this epsilon
            let rt_config = crate::rt_clustering::RtClusteringConfig {
                epsilon,
                min_points: 4,
                min_cluster_size: 30, // Scaled for 128³ grid
                rays_per_event: 32,
            };

            // We need to create a new RT engine for each epsilon
            // (The BVH sphere radii depend on epsilon)
            let optixir_path = crate::rt_clustering::find_optixir_path()
                .context("Could not find rt_clustering.optixir")?;

            let mut rt_engine =
                crate::rt_clustering::RtClusteringEngine::new(self.context.clone(), rt_config)
                    .context("Failed to create RT clustering engine")?;

            rt_engine
                .load_pipeline(&optixir_path)
                .context("Failed to load RT clustering pipeline")?;

            let result = rt_engine.cluster(spike_positions)?;

            log::info!(
                "  Scale ε={:.1}Å: {} clusters, {} neighbors",
                epsilon,
                result.num_clusters,
                result.total_neighbors
            );

            // Compute centroids for each cluster at this scale
            let mut cluster_points: HashMap<i32, Vec<usize>> = HashMap::new();
            for (idx, &cluster_id) in result.cluster_ids.iter().enumerate() {
                if cluster_id >= 0 {
                    cluster_points.entry(cluster_id).or_default().push(idx);
                }
            }

            for (cluster_id, point_indices) in cluster_points {
                if point_indices.len() < 15 {
                    continue; // Skip tiny clusters
                }

                // Compute centroid
                let mut cx = 0.0f32;
                let mut cy = 0.0f32;
                let mut cz = 0.0f32;
                for &idx in &point_indices {
                    cx += spike_positions[idx * 3];
                    cy += spike_positions[idx * 3 + 1];
                    cz += spike_positions[idx * 3 + 2];
                }
                let n = point_indices.len() as f32;
                cx /= n;
                cy /= n;
                cz /= n;

                all_clusters.push(ScaleCluster {
                    centroid: [cx, cy, cz],
                    spike_count: point_indices.len(),
                    spike_indices: point_indices,
                    epsilon,
                    original_cluster_id: cluster_id,
                    persistence: 1, // Will be updated during merge
                });
            }
        }

        log::info!("  Total clusters across all scales: {}", all_clusters.len());

        // Merge clusters that overlap across scales
        let mut merged_clusters: Vec<MergedCluster> = Vec::new();
        let mut used = vec![false; all_clusters.len()];

        for i in 0..all_clusters.len() {
            if used[i] {
                continue;
            }

            // Start a new merged cluster
            let mut merge_group: Vec<usize> = vec![i];
            used[i] = true;

            // Find all clusters within merge_distance of this centroid
            let ci = &all_clusters[i];
            for j in (i + 1)..all_clusters.len() {
                if used[j] {
                    continue;
                }

                let cj = &all_clusters[j];
                let dist = ((ci.centroid[0] - cj.centroid[0]).powi(2)
                    + (ci.centroid[1] - cj.centroid[1]).powi(2)
                    + (ci.centroid[2] - cj.centroid[2]).powi(2))
                .sqrt();

                if dist <= merge_distance {
                    merge_group.push(j);
                    used[j] = true;
                }
            }

            // Compute merged cluster properties
            let scales_present: std::collections::HashSet<u32> = merge_group
                .iter()
                .map(|&idx| (all_clusters[idx].epsilon * 10.0) as u32)
                .collect();
            let persistence = scales_present.len();

            // Merge spike indices (union across scales)
            let mut all_spike_indices: std::collections::HashSet<usize> =
                std::collections::HashSet::new();
            let mut sum_cx = 0.0f32;
            let mut sum_cy = 0.0f32;
            let mut sum_cz = 0.0f32;

            for &idx in &merge_group {
                let c = &all_clusters[idx];
                sum_cx += c.centroid[0];
                sum_cy += c.centroid[1];
                sum_cz += c.centroid[2];
                for &spike_idx in &c.spike_indices {
                    all_spike_indices.insert(spike_idx);
                }
            }

            let n = merge_group.len() as f32;
            let merged_centroid = [sum_cx / n, sum_cy / n, sum_cz / n];
            let merged_spike_count = all_spike_indices.len();

            merged_clusters.push(MergedCluster {
                centroid: merged_centroid,
                spike_count: merged_spike_count,
                spike_indices: all_spike_indices.into_iter().collect(),
                persistence,
                scales: scales_present
                    .into_iter()
                    .map(|s| s as f32 / 10.0)
                    .collect(),
            });
        }

        // Sort by persistence * spike_count (higher = more confident)
        merged_clusters.sort_by(|a, b| {
            let score_a = a.persistence * a.spike_count;
            let score_b = b.persistence * b.spike_count;
            score_b.cmp(&score_a)
        });

        // Filter to clusters with persistence >= 2 (appear at multiple scales)
        let persistent_clusters: Vec<MergedCluster> = merged_clusters
            .into_iter()
            .filter(|c| c.persistence >= 2)
            .collect();

        log::info!(
            "  Multi-scale result: {} persistent clusters (appear at ≥2 scales)",
            persistent_clusters.len()
        );

        for (i, c) in persistent_clusters.iter().take(5).enumerate() {
            log::info!(
                "    #{}: {} spikes, persistence={}, scales={:?}",
                i + 1,
                c.spike_count,
                c.persistence,
                c.scales
            );
        }

        Ok(MultiScaleClusteringResult {
            clusters: persistent_clusters,
            total_scales: epsilon_scales.len(),
            epsilon_values: epsilon_scales,
            adaptive_epsilon: is_adaptive,
            knn_k: if is_adaptive { Some(knn_k) } else { None },
            num_spikes_sampled: actual_sample_size,
        })
    }

    /// Fallback grid-based clustering when RT cores unavailable
    /// CPU grid clustering fallback — public for multi-stream path when RT cores fail
    pub fn cluster_spikes_cpu_fallback(
        &self,
        positions: &[f32],
    ) -> Result<crate::rt_clustering::RtClusteringResult> {
        self.fallback_grid_cluster(positions)
    }

    fn fallback_grid_cluster(
        &self,
        positions: &[f32],
    ) -> Result<crate::rt_clustering::RtClusteringResult> {
        let num_points = positions.len() / 3;
        let start = Instant::now();

        // Simple single-linkage clustering using spatial hashing
        // This is O(N) for sparse data but degrades to O(N²) for dense clusters
        let epsilon = 5.0f32;
        let cell_size = epsilon;

        use std::collections::HashMap;

        // Hash points into cells
        let mut cells: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
        for i in 0..num_points {
            let x = positions[i * 3];
            let y = positions[i * 3 + 1];
            let z = positions[i * 3 + 2];
            let cell = (
                (x / cell_size).floor() as i32,
                (y / cell_size).floor() as i32,
                (z / cell_size).floor() as i32,
            );
            cells.entry(cell).or_default().push(i);
        }

        // Union-find for clustering
        let mut parent: Vec<i32> = (0..num_points as i32).collect();

        fn find(parent: &mut [i32], i: usize) -> i32 {
            if parent[i] != i as i32 {
                parent[i] = find(parent, parent[i] as usize);
            }
            parent[i]
        }

        fn union(parent: &mut [i32], a: usize, b: usize) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra != rb {
                parent[ra as usize] = rb;
            }
        }

        // Find neighbors and union
        let mut total_neighbors = 0usize;
        for (&cell, points) in &cells {
            // Check this cell and 26 neighbors
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let neighbor_cell = (cell.0 + dx, cell.1 + dy, cell.2 + dz);
                        if let Some(neighbors) = cells.get(&neighbor_cell) {
                            for &i in points {
                                let xi = positions[i * 3];
                                let yi = positions[i * 3 + 1];
                                let zi = positions[i * 3 + 2];

                                for &j in neighbors {
                                    if i >= j {
                                        continue;
                                    }
                                    let xj = positions[j * 3];
                                    let yj = positions[j * 3 + 1];
                                    let zj = positions[j * 3 + 2];

                                    let dist_sq =
                                        (xi - xj).powi(2) + (yi - yj).powi(2) + (zi - zj).powi(2);
                                    if dist_sq <= epsilon * epsilon {
                                        union(&mut parent, i, j);
                                        total_neighbors += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Flatten and count clusters
        let mut cluster_ids: Vec<i32> = Vec::with_capacity(num_points);
        let mut cluster_counts: HashMap<i32, usize> = HashMap::new();

        for i in 0..num_points {
            let root = find(&mut parent, i);
            cluster_ids.push(root);
            *cluster_counts.entry(root).or_default() += 1;
        }

        let num_clusters = cluster_counts.len();
        let gpu_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(crate::rt_clustering::RtClusteringResult {
            cluster_ids,
            num_clusters,
            total_neighbors,
            gpu_time_ms,
        })
    }

    /// Report cumulative statistics
    pub fn stats(&self) -> PersistentEngineStats {
        PersistentEngineStats {
            structures_processed: self.structures_processed,
            total_steps_run: self.total_steps_run,
            total_compute_time_ms: self.total_compute_time_ms,
            context_init_time_ms: self.context_init_time_ms,
            module_init_time_ms: self.module_init_time_ms,
            overhead_saved_ms: self.structures_processed.saturating_sub(1) as u64
                * (self.context_init_time_ms + self.module_init_time_ms),
        }
    }
}

/// Statistics from persistent engine
#[derive(Debug, Clone)]
pub struct PersistentEngineStats {
    pub structures_processed: usize,
    pub total_steps_run: i64,
    pub total_compute_time_ms: u64,
    pub context_init_time_ms: u64,
    pub module_init_time_ms: u64,
    /// Estimated overhead saved by reusing context/module
    pub overhead_saved_ms: u64,
}

/// Convert RT clustering results into binding site structures.
///
/// Crate-visible so `crate::transform::clustering_to_clustered_sites`
/// can wrap this function under the Phase 2 audit spine. Not `pub` —
/// external callers go through the transform, not directly.
#[cfg(feature = "gpu")]
pub(crate) fn build_clustered_sites(
    spike_events: &[SpikeEvent],
    clustering_result: &crate::rt_clustering::RtClusteringResult,
    #[cfg(feature = "gpu")] gpu_engine: Option<&NhsAmberFusedEngine>,
    #[cfg(not(feature = "gpu"))] _gpu_engine: Option<&()>,
) -> Vec<ClusteredBindingSite> {
    use std::collections::HashMap;

    if spike_events.is_empty() {
        return Vec::new();
    }

    // Group spikes by cluster
    let mut cluster_spikes: HashMap<i32, Vec<(usize, &SpikeEvent)>> = HashMap::new();
    for (idx, (spike, &cluster_id)) in spike_events
        .iter()
        .zip(clustering_result.cluster_ids.iter())
        .enumerate()
    {
        if cluster_id >= 0 {
            // Skip noise points (-1)
            cluster_spikes
                .entry(cluster_id)
                .or_default()
                .push((idx, spike));
        }
    }

    // Build site structures for each cluster
    let mut sites = Vec::with_capacity(cluster_spikes.len());
    for (cluster_id, spikes) in cluster_spikes {
        if spikes.is_empty() {
            continue;
        }

        // Compute centroid using burial-weighted mean + density peak refinement.
        //
        // Two-stage centroid:
        //   Stage 1: Burial-weighted mean — spikes with more nearby residues
        //   (n_residues) are deeper in the protein and closer to the true
        //   binding cavity. Weight = n_residues² so buried spikes dominate.
        //
        //   Stage 2: Density peak — find the local maximum of spike density
        //   using a 3D Gaussian KDE. The peak is typically 1-3A closer to the
        //   true ligand center than the arithmetic mean because it ignores
        //   outlier surface spikes. Final centroid is 70% density peak +
        //   30% burial-weighted mean (peak for accuracy, mean for stability).
        let mut centroid = [0.0f32; 3];
        let mut sum_intensity = 0.0f32;
        let mut min_pos = [f32::MAX; 3];
        let mut max_pos = [f32::MIN; 3];

        // Stage 1: Burial-weighted centroid
        let mut weighted_centroid = [0.0f32; 3];
        let mut total_weight = 0.0f32;

        for (_, spike) in &spikes {
            // Burial weight: nearby_residues_count² (buried spikes count more)
            // Minimum weight of 1.0 so surface spikes aren't zeroed out
            let burial_weight = (spike.nearby_residues.len() as f32).max(1.0).powi(2);
            weighted_centroid[0] += spike.position[0] * burial_weight;
            weighted_centroid[1] += spike.position[1] * burial_weight;
            weighted_centroid[2] += spike.position[2] * burial_weight;
            total_weight += burial_weight;

            // Unweighted sum for fallback
            centroid[0] += spike.position[0];
            centroid[1] += spike.position[1];
            centroid[2] += spike.position[2];
            sum_intensity += spike.intensity;

            // Update bounding box
            for i in 0..3 {
                min_pos[i] = min_pos[i].min(spike.position[i]);
                max_pos[i] = max_pos[i].max(spike.position[i]);
            }
        }

        let n = spikes.len() as f32;

        // Use burial-weighted centroid if we have burial data, else arithmetic mean
        if total_weight > n {
            // Burial weights are contributing (not all n_residues=1)
            centroid[0] = weighted_centroid[0] / total_weight;
            centroid[1] = weighted_centroid[1] / total_weight;
            centroid[2] = weighted_centroid[2] / total_weight;
        } else {
            centroid[0] /= n;
            centroid[1] /= n;
            centroid[2] /= n;
        }

        // Stage 2: Density peak refinement via 3D Gaussian KDE
        // Try GPU path first, fall back to CPU.
        if spikes.len() >= 20 {
            let bandwidth = 2.0f32;
            let search_radius = 5.0f32;

            let mut gpu_done = false;

            // GPU path: use the fused engine's KDE kernel if available
            #[cfg(feature = "gpu")]
            if let Some(engine) = gpu_engine {
                let positions: Vec<[f32; 3]> = spikes.iter().map(|(_, s)| s.position).collect();
                let n_residues: Vec<i32> = spikes
                    .iter()
                    .map(|(_, s)| s.nearby_residues.len() as i32)
                    .collect();

                match engine.compute_kde_centroid(
                    &positions,
                    &n_residues,
                    centroid,
                    search_radius,
                    bandwidth,
                ) {
                    Ok(refined) => {
                        centroid = refined;
                        gpu_done = true;
                    }
                    Err(e) => {
                        log::warn!("GPU KDE failed, falling back to CPU: {}", e);
                    }
                }
            }

            // CPU fallback
            if !gpu_done {
                let bw2 = bandwidth * bandwidth;
                let grid_step = 1.0f32;
                let mut best_density = 0.0f32;
                let mut peak_pos = centroid;

                let n_steps = (search_radius / grid_step) as i32;
                for ix in -n_steps..=n_steps {
                    for iy in -n_steps..=n_steps {
                        for iz in -n_steps..=n_steps {
                            let px = centroid[0] + ix as f32 * grid_step;
                            let py = centroid[1] + iy as f32 * grid_step;
                            let pz = centroid[2] + iz as f32 * grid_step;

                            let mut density = 0.0f32;
                            for (_, spike) in &spikes {
                                let dx = px - spike.position[0];
                                let dy = py - spike.position[1];
                                let dz = pz - spike.position[2];
                                let r2 = dx * dx + dy * dy + dz * dz;
                                if r2 < 9.0 * bw2 {
                                    density += (-r2 / (2.0 * bw2)).exp();
                                }
                            }

                            if density > best_density {
                                best_density = density;
                                peak_pos = [px, py, pz];
                            }
                        }
                    }
                }

                centroid[0] = 0.7 * peak_pos[0] + 0.3 * centroid[0];
                centroid[1] = 0.7 * peak_pos[1] + 0.3 * centroid[1];
                centroid[2] = 0.7 * peak_pos[2] + 0.3 * centroid[2];
            }
        }

        let bounding_box = [
            max_pos[0] - min_pos[0],
            max_pos[1] - min_pos[1],
            max_pos[2] - min_pos[2],
        ];

        // Estimate pocket volume using voxel density method
        // Grid the space at 2Å resolution and count occupied voxels
        let voxel_size = 2.0f32;
        let estimated_volume =
            if bounding_box[0] > 0.0 && bounding_box[1] > 0.0 && bounding_box[2] > 0.0 {
                let nx = ((bounding_box[0] / voxel_size).ceil() as usize).max(1);
                let ny = ((bounding_box[1] / voxel_size).ceil() as usize).max(1);
                let nz = ((bounding_box[2] / voxel_size).ceil() as usize).max(1);

                // Use HashSet to count unique voxels occupied by spikes
                let mut occupied_voxels = std::collections::HashSet::new();
                for (_, spike) in &spikes {
                    let vx = ((spike.position[0] - min_pos[0]) / voxel_size) as i32;
                    let vy = ((spike.position[1] - min_pos[1]) / voxel_size) as i32;
                    let vz = ((spike.position[2] - min_pos[2]) / voxel_size) as i32;

                    // Mark this voxel and its immediate neighbors (small neighborhood)
                    for dx in -1..=1 {
                        for dy in -1..=1 {
                            for dz in -1..=1 {
                                let key = (vx + dx, vy + dy, vz + dz);
                                occupied_voxels.insert(key);
                            }
                        }
                    }
                }

                // Volume = occupied voxels * voxel volume
                let voxel_volume = voxel_size.powi(3);
                let raw_volume = occupied_voxels.len() as f32 * voxel_volume;

                // Apply packing efficiency correction (sphere packing ~74% efficiency)
                let pocket_volume = raw_volume * 0.74;

                // Sanity bounds: typical pockets 100-2000 Å³, large enzyme sites up to 8000 Å³
                pocket_volume.clamp(50.0, 8000.0)
            } else {
                // Degenerate case: estimate from spike count
                (spikes.len() as f32 * 15.0).clamp(50.0, 2000.0)
            };

        let avg_intensity = sum_intensity / n;
        let spike_count = spikes.len();

        // Compute druggability score
        let druggability =
            DruggabilityScore::from_site(estimated_volume, avg_intensity, &bounding_box);

        // Classify site
        let classification =
            SiteClassification::from_properties(spike_count, estimated_volume, avg_intensity);

        // Overall quality score: combines spike count, intensity, and druggability
        let spike_quality = (spike_count as f32 / 100.0).clamp(0.0, 1.0);
        let intensity_quality = (avg_intensity / 10.0).clamp(0.0, 1.0);
        let quality_score =
            0.3 * spike_quality + 0.3 * intensity_quality + 0.4 * druggability.overall;

        sites.push(ClusteredBindingSite::new_with_geometric_voxel_mass(
            cluster_id,
            centroid,
            spike_count,
            spikes.iter().map(|(idx, _)| *idx).collect(),
            avg_intensity,
            estimated_volume,
            bounding_box,
            quality_score,
            druggability,
            classification,
        ));
        // aromatic_proximity: None, lining_residues: Vec::new() are
        // set by new_with_geometric_voxel_mass and populated later
        // by compute_aromatic_proximity / compute_lining_residues.
    }

    // Sort by spike count (most significant first)
    sites.sort_by(|a, b| b.spike_count.cmp(&a.spike_count));
    sites
}

impl ClusteredBindingSite {
    // ================================================================
    // Phase 1 canonical spatial-view API — post-05dbc3dc lane.
    //
    // Every canonical reader of a site centroid must use one of the
    // typed methods below, naming the physical representation it wants.
    // The `legacy_emission_centroid` field is module-private; external
    // crates and binary targets must go through `emission_compat_centroid`
    // for emission/logging or through `view` / `distance_to` / `dcc` /
    // `distance_to_ligand` for canonical logic.
    //
    // No method in this group admits a default. `view(v)` returns
    // `None` when the view is not derivable for this site, and the
    // caller must decide explicitly — silent fallback is forbidden
    // (Rule A of the lane doctrine).
    // ================================================================

    /// Canonical external constructor. Initializes both the legacy
    /// emission-compat scalar and the `GeometricVoxelMass` view of
    /// the localization manifold from the same input. All other
    /// views start `None`; later pipeline stages populate them.
    ///
    /// `geometric_voxel_mass_centroid` is the intensity²- or
    /// voxel-mass-weighted centroid produced by the clustering stage.
    /// Callers that need to track a different representation should
    /// first build the site here and then call `localization.set(...)`
    /// for each additional view.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_geometric_voxel_mass(
        cluster_id: i32,
        geometric_voxel_mass_centroid: [f32; 3],
        spike_count: usize,
        spike_indices: Vec<usize>,
        avg_intensity: f32,
        estimated_volume: f32,
        bounding_box: [f32; 3],
        quality_score: f32,
        druggability: DruggabilityScore,
        classification: SiteClassification,
    ) -> Self {
        Self {
            cluster_id,
            legacy_emission_centroid: geometric_voxel_mass_centroid,
            localization: crate::spatial_view::CentroidManifold::with_geometric_voxel_mass(
                geometric_voxel_mass_centroid,
            ),
            spike_count,
            spike_indices,
            avg_intensity,
            estimated_volume,
            bounding_box,
            quality_score,
            druggability,
            classification,
            aromatic_proximity: None,
            lining_residues: Vec::new(),
        }
    }

    /// Retrieve the requested spatial view. Returns `None` if the
    /// view is not derivable for this site — no silent fallback.
    #[inline]
    pub fn view(&self, which: crate::spatial_view::SpatialView) -> Option<[f32; 3]> {
        self.localization.view(which)
    }

    /// Euclidean distance from this site's named view to a point.
    /// `None` when the view is unpopulated.
    #[inline]
    pub fn distance_to(
        &self,
        target: [f32; 3],
        view: crate::spatial_view::SpatialView,
    ) -> Option<f32> {
        self.localization.distance_to(target, view)
    }

    /// DCC — distance to a ground-truth reference point under the
    /// named view. Typed alias for [`distance_to`] that documents
    /// intent at the call site. `None` when the view is unpopulated.
    #[inline]
    pub fn dcc(
        &self,
        ground_truth: [f32; 3],
        view: crate::spatial_view::SpatialView,
    ) -> Option<f32> {
        self.localization.dcc(ground_truth, view)
    }

    /// Distance to a ligand center under the named view. Typed alias
    /// for [`distance_to`] that documents intent. `None` when the
    /// view is unpopulated.
    #[inline]
    pub fn distance_to_ligand(
        &self,
        ligand: [f32; 3],
        view: crate::spatial_view::SpatialView,
    ) -> Option<f32> {
        self.localization.distance_to_ligand(ligand, view)
    }

    /// Compatibility scalar accessor for serialization, logging, and
    /// external report generators that need a single 3-vector to
    /// write into a PDB/PyMOL/ChimeraX/Markdown/JSON field. **This is
    /// not a canonical-logic accessor.** Calling this from merge,
    /// attribution, refinement, ranking, or validation code is a
    /// scalar-collapse violation (Rule A). Use [`view`] (named
    /// representation) or [`geometric_voxel_mass_centroid`] (named
    /// shortcut for the most common canonical-logic case).
    ///
    /// The name is intentionally long and lane-specific — it should
    /// stand out in code review when misused.
    #[inline]
    pub fn emission_compat_centroid(&self) -> [f32; 3] {
        self.legacy_emission_centroid
    }

    /// Explicit named accessor for the `GeometricVoxelMass` view.
    /// Equivalent to `site.view(SpatialView::GeometricVoxelMass).expect(...)`
    /// but far more readable at canonical call sites that index the
    /// result as `[0] / [1] / [2]` (merge, grid-bin, spike attribution,
    /// refinement).
    ///
    /// This is NOT a default or fallback — the method name states the
    /// view. Using this method at a site where a different physical
    /// representation is the right one (e.g., LiningResidues for
    /// lining-focused math) is a semantic bug and must be caught in
    /// code review.
    ///
    /// Panics if the GeometricVoxelMass view is somehow absent
    /// (`ClusteredBindingSite` construction paths always populate it,
    /// so this should be unreachable).
    #[inline]
    pub fn geometric_voxel_mass_centroid(&self) -> [f32; 3] {
        self.view(crate::spatial_view::SpatialView::GeometricVoxelMass)
            .expect(
                "ClusteredBindingSite invariant violated: \
                 GeometricVoxelMass view must always be populated by \
                 new_with_geometric_voxel_mass / set_geometric_voxel_mass_centroid",
            )
    }

    /// Canonical write path for the geometric-voxel-mass centroid.
    /// Updates both the legacy emission scalar and the manifold's
    /// `GeometricVoxelMass` view in one atomic operation so the two
    /// can never drift.
    ///
    /// Callers that need to set a *different* view (LiningResidues,
    /// HotPhase, etc.) must write to `self.localization.set(view, c)`
    /// directly — this method deliberately does not take a view
    /// parameter, because the legacy emission-compat scalar
    /// represents the geometric-voxel-mass view only.
    pub fn set_geometric_voxel_mass_centroid(&mut self, centroid: [f32; 3]) {
        self.legacy_emission_centroid = centroid;
        self.localization.set(
            crate::spatial_view::SpatialView::GeometricVoxelMass,
            centroid,
        );
    }

    // ================================================================
    // End Phase 1 canonical spatial-view API. Legacy methods below.
    // ================================================================

    /// Enhance this binding site with aromatic proximity analysis
    ///
    /// Updates the aromatic_proximity field and recalculates druggability score.
    ///
    /// # Arguments
    /// * `aromatic_positions` - List of (residue_id, aromatic_type, [x, y, z])
    ///   - aromatic_type: 0=TRP, 1=TYR, 2=PHE
    pub fn compute_aromatic_proximity(&mut self, aromatic_positions: &[(u32, u8, [f32; 3])]) {
        // Aromatic proximity is computed against the GeometricVoxelMass
        // view — the spike-weighted center of the cluster is the
        // appropriate reference for "aromatics near this pocket."
        let gvm = self
            .view(crate::spatial_view::SpatialView::GeometricVoxelMass)
            .expect("ClusteredBindingSite always carries GeometricVoxelMass view");
        let info = AromaticProximityInfo::compute(&gvm, aromatic_positions);

        // Recalculate druggability with aromatic info
        self.druggability = DruggabilityScore::from_site_with_aromatics(
            self.estimated_volume,
            self.avg_intensity,
            &self.bounding_box,
            Some(&info),
        );

        // Update quality score
        let spike_quality = (self.spike_count as f32 / 100.0).clamp(0.0, 1.0);
        let intensity_quality = (self.avg_intensity / 10.0).clamp(0.0, 1.0);
        self.quality_score = 0.25 * spike_quality
            + 0.25 * intensity_quality
            + 0.3 * self.druggability.overall
            + 0.2 * info.aromatic_score;

        self.aromatic_proximity = Some(info);
    }

    /// Check if this site has been analyzed for aromatic proximity
    pub fn has_aromatic_analysis(&self) -> bool {
        self.aromatic_proximity.is_some()
    }

    /// Get the aromatic score (0.0 if not analyzed)
    pub fn aromatic_score(&self) -> f32 {
        self.aromatic_proximity
            .as_ref()
            .map(|p| p.aromatic_score)
            .unwrap_or(0.0)
    }

    /// Compute residues lining this binding pocket
    ///
    /// Finds all residues with at least one atom within `cutoff` distance
    /// of the pocket centroid.
    ///
    /// # Arguments
    /// * `positions` - Atom positions as flat [x0,y0,z0,x1,y1,z1,...] array
    /// * `residue_ids` - Residue ID for each atom
    /// * `residue_names` - Name of each residue (indexed by residue_id)
    /// * `chain_ids` - Chain ID for each atom
    /// * `cutoff` - Distance cutoff in Angstroms (default: 5.0)
    pub fn compute_lining_residues(
        &mut self,
        positions: &[f32],
        residue_ids: &[usize],
        residue_names: &[String],
        chain_ids: &[String],
        residue_pdb_ids: &[i32],
        cutoff: f32,
    ) {
        use std::collections::HashMap;

        let n_atoms = positions.len() / 3;
        // Lining residues are computed in the neighborhood of the
        // site's GeometricVoxelMass view. Using a different view here
        // would change the definition of "lining" and must be an
        // explicit, reviewed decision — not a silent default.
        let gvm = self
            .view(crate::spatial_view::SpatialView::GeometricVoxelMass)
            .expect("ClusteredBindingSite always carries GeometricVoxelMass view");
        let cx = gvm[0];
        let cy = gvm[1];
        let cz = gvm[2];
        let cutoff_sq = cutoff * cutoff;

        // Track per-residue: (chain, resname, min_distance, atom_count)
        let mut residue_info: HashMap<(String, i32), (String, f32, usize)> = HashMap::new();

        // Accumulate pocket atom positions for centroid refinement
        let mut pocket_sum = [0.0f64; 3];
        let mut pocket_count = 0u32;

        for i in 0..n_atoms {
            let x = positions[i * 3];
            let y = positions[i * 3 + 1];
            let z = positions[i * 3 + 2];

            let dx = x - cx;
            let dy = y - cy;
            let dz = z - cz;
            let dist_sq = dx * dx + dy * dy + dz * dz;

            if dist_sq <= cutoff_sq {
                // Accumulate for pocket centroid
                pocket_sum[0] += x as f64;
                pocket_sum[1] += y as f64;
                pocket_sum[2] += z as f64;
                pocket_count += 1;

                let internal_id = residue_ids[i];
                let res_id = if internal_id < residue_pdb_ids.len() {
                    residue_pdb_ids[internal_id] + 1
                } else {
                    (internal_id + 1) as i32
                };
                let chain = chain_ids[i].clone();
                let resname = if i < residue_names.len() {
                    residue_names[i].clone()
                } else {
                    "UNK".to_string()
                };
                let dist = dist_sq.sqrt();

                let key = (chain.clone(), res_id as i32);
                residue_info
                    .entry(key)
                    .and_modify(|(_, min_d, count)| {
                        if dist < *min_d {
                            *min_d = dist;
                        }
                        *count += 1;
                    })
                    .or_insert((resname, dist, 1));
            }
        }

        // Refine centroid: shift from aromatic probe position toward pocket interior.
        // The spike centroid sits at the aromatic ring (pocket wall). The actual pocket
        // center is shifted toward the protein interior. We detect the interior direction
        // using a large (15Å) hemisphere count: more atoms on the protein-interior side
        // than the solvent-exposed side gives the correction vector and magnitude.
        if pocket_count >= 10 {
            // Protein center of mass
            let mut com = [0.0f64; 3];
            for j in 0..n_atoms {
                com[0] += positions[j * 3] as f64;
                com[1] += positions[j * 3 + 1] as f64;
                com[2] += positions[j * 3 + 2] as f64;
            }
            let na = n_atoms as f64;
            com[0] /= na;
            com[1] /= na;
            com[2] /= na;

            // Direction from spike centroid toward protein COM
            let dir = [com[0] - cx as f64, com[1] - cy as f64, com[2] - cz as f64];
            let dir_mag = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();

            if dir_mag > 0.5 {
                let ux = dir[0] / dir_mag;
                let uy = dir[1] / dir_mag;
                let uz = dir[2] / dir_mag;

                // Use 15Å sphere for asymmetry detection (much larger than lining cutoff)
                // to capture the surface-vs-interior signal that's invisible at 8Å.
                let asym_radius_sq = 15.0f64 * 15.0;
                let mut toward = 0u32;
                let mut away = 0u32;
                for j in 0..n_atoms {
                    let ax = positions[j * 3] as f64 - cx as f64;
                    let ay = positions[j * 3 + 1] as f64 - cy as f64;
                    let az = positions[j * 3 + 2] as f64 - cz as f64;
                    let d2 = ax * ax + ay * ay + az * az;
                    if d2 <= asym_radius_sq {
                        let dot = ax * ux + ay * uy + az * uz;
                        if dot > 0.0 {
                            toward += 1;
                        } else {
                            away += 1;
                        }
                    }
                }

                let total = (toward + away) as f64;
                if total > 0.0 {
                    let asymmetry = (toward as f64 - away as f64) / total;
                    // Shift proportional to asymmetry; max 4Å.
                    let shift = (asymmetry * 10.0).clamp(0.0, 4.0);
                    if shift > 0.1 {
                        // Refine the GeometricVoxelMass centroid only;
                        // the manifold's GVM view tracks lockstep via
                        // set_geometric_voxel_mass_centroid. Other
                        // views (LiningResidues, HotPhase, ...) are NOT
                        // touched here — they represent different
                        // physical quantities and must be set
                        // separately by their producing stage.
                        let new_gvm = [
                            gvm[0] + (ux * shift) as f32,
                            gvm[1] + (uy * shift) as f32,
                            gvm[2] + (uz * shift) as f32,
                        ];
                        self.set_geometric_voxel_mass_centroid(new_gvm);
                    }
                }
            }
        }

        // Convert to LiningResidue list, sorted by distance from refined centroid.
        // spike_attribution_count is left at 0 here — it is populated by the
        // caller after this function runs (if they have spike attribution data).
        self.lining_residues = residue_info
            .into_iter()
            .map(
                |((chain, resid), (resname, min_distance, n_atoms))| LiningResidue {
                    chain,
                    resid,
                    resname,
                    min_distance,
                    n_atoms_in_pocket: n_atoms,
                    spike_attribution_count: 0,
                },
            )
            .collect();

        self.lining_residues.sort_by(|a, b| {
            a.min_distance
                .partial_cmp(&b.min_distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Recompute druggability with catalytic scoring
        self.recompute_druggability_with_catalytic();
    }

    /// Recompute druggability score including catalytic residue analysis
    ///
    /// Called after lining residues are computed to update druggability
    /// for enzyme active sites that would otherwise fail hydrophobic scoring.
    pub fn recompute_druggability_with_catalytic(&mut self) {
        self.druggability = DruggabilityScore::from_site_with_catalytic(
            self.estimated_volume,
            self.avg_intensity,
            &self.bounding_box,
            self.aromatic_proximity.as_ref(),
            &self.lining_residues,
        );
    }

    /// Get lining residue IDs as a simple list (for validation comparisons)
    pub fn lining_residue_ids(&self) -> Vec<i32> {
        self.lining_residues.iter().map(|r| r.resid).collect()
    }

    /// Get formatted residue list string (e.g., "A:PHE347, A:TRP348, A:GLU349")
    pub fn lining_residues_str(&self) -> String {
        self.lining_residues
            .iter()
            .map(|r| format!("{}:{}{}", r.chain, r.resname, r.resid))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Enhance a list of binding sites with aromatic proximity analysis
#[cfg(feature = "gpu")]
pub fn enhance_sites_with_aromatics(
    sites: &mut [ClusteredBindingSite],
    aromatic_positions: &[(u32, u8, [f32; 3])],
) {
    for site in sites.iter_mut() {
        site.compute_aromatic_proximity(aromatic_positions);
    }

    // Re-sort by quality score after enhancement
    sites.sort_by(|a, b| {
        b.quality_score
            .partial_cmp(&a.quality_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Batch processor using persistent engine
#[cfg(feature = "gpu")]
pub struct BatchProcessor {
    engine: PersistentNhsEngine,
    config: PersistentBatchConfig,
}

#[cfg(feature = "gpu")]
impl BatchProcessor {
    /// Create batch processor
    pub fn new(config: PersistentBatchConfig) -> Result<Self> {
        let engine = PersistentNhsEngine::new(&config)?;
        Ok(Self { engine, config })
    }

    /// Process multiple topology files
    pub fn process_batch<P: AsRef<Path>>(
        &mut self,
        topology_paths: &[P],
    ) -> Result<Vec<StructureResult>> {
        log::info!("═══════════════════════════════════════════════════════════════");
        log::info!(
            "  PERSISTENT BATCH PROCESSING: {} structures",
            topology_paths.len()
        );
        log::info!("═══════════════════════════════════════════════════════════════");

        let batch_start = Instant::now();
        let mut results = Vec::with_capacity(topology_paths.len());

        for (idx, path) in topology_paths.iter().enumerate() {
            let path = path.as_ref();
            log::info!(
                "\n[{}/{}] Processing: {}",
                idx + 1,
                topology_paths.len(),
                path.display()
            );

            // Load topology
            let topology = PrismPrepTopology::load(path)
                .with_context(|| format!("Failed to load topology: {}", path.display()))?;

            let structure_id = path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "unknown".to_string());

            let struct_start = Instant::now();

            // Load into engine
            self.engine.load_topology(&topology)?;

            // Configure unified cryo-UV protocol
            let cryo_uv_protocol = CryoUvProtocol {
                start_temp: self.config.cryo_temp,
                end_temp: self.config.temperature,
                cold_hold_steps: self.config.cryo_hold,
                ramp_steps: self.config.convergence_steps / 2,
                warm_hold_steps: self.config.convergence_steps / 2,
                current_step: 0,
                // UV-LIF coupling (validated parameters)
                uv_burst_energy: 30.0,
                uv_burst_interval: 500,
                uv_burst_duration: 50,
                // Keep batch processing aligned with the production
                // ProtocolState surface: TRP, TYR, PHE, BNZ/HIS shoulder, HIS.
                scan_wavelengths: crate::fused_engine::PRODUCTION_UV_SCAN_WAVELENGTHS.to_vec(),
                wavelength_dwell_steps: 500,
                ramp_down_steps: 0,
                cold_return_steps: 0,
                stepped_holds: vec![],
            };
            self.engine.set_cryo_uv_protocol(cryo_uv_protocol)?;

            // Run all phases
            let total_steps = self.config.survey_steps
                + self.config.convergence_steps
                + self.config.precision_steps;

            let summary = self.engine.run(total_steps)?;

            let wall_time_ms = struct_start.elapsed().as_millis() as u64;

            // Collect spike events
            let spike_events = self.engine.get_spike_events();

            // RT-accelerated clustering of spike positions
            let (clustered_sites, clustering_stats) = if !spike_events.is_empty() {
                // Extract positions for clustering
                let spike_positions: Vec<f32> = spike_events
                    .iter()
                    .flat_map(|s| s.position.iter().copied())
                    .collect();

                // Cluster using RT cores (or fallback)
                let used_rt = self.engine.has_rt_clustering();
                match self.engine.cluster_spikes(&spike_positions) {
                    Ok(result) => {
                        // Phase 2: route batch-mode site construction through
                        // the same audit-spine transform as the single-structure
                        // path above. Abort violations propagate out of
                        // `process_batch` as a run error; quarantine violations
                        // are logged and the output is accepted.
                        #[cfg(feature = "gpu")]
                        let sites = {
                            use crate::transform::clustering_to_clustered_sites::{
                                ClusteringInput, ClusteringToClusteredSites,
                            };
                            use crate::transform::AuditedTransform;
                            let outcome =
                                ClusteringToClusteredSites::new().apply(ClusteringInput {
                                    spike_events: &spike_events,
                                    clustering_result: &result,
                                    engine: self.engine.engine.as_ref(),
                                });
                            let (sites, quarantined) = outcome
                                .into_result()
                                .map_err(|aborted| anyhow::anyhow!("{aborted}"))?;
                            if !quarantined.is_empty() {
                                log::warn!(
                                    "transform clustering_to_clustered_sites: {} quarantined site(s) in process_batch",
                                    quarantined.len()
                                );
                            }
                            sites
                        };
                        #[cfg(not(feature = "gpu"))]
                        let sites = build_clustered_sites(&spike_events, &result, None);
                        let stats = ClusteringStats {
                            num_clusters: result.num_clusters,
                            total_neighbors: result.total_neighbors,
                            gpu_time_ms: result.gpu_time_ms,
                            used_rt_cores: used_rt,
                        };
                        log::info!(
                            "  📊 Clustered {} spikes → {} binding sites ({:.1}ms, {})",
                            spike_events.len(),
                            sites.len(),
                            result.gpu_time_ms,
                            if used_rt { "RT cores" } else { "fallback" }
                        );
                        (sites, Some(stats))
                    }
                    Err(e) => {
                        log::warn!("  ⚠️ Clustering failed: {}", e);
                        (Vec::new(), None)
                    }
                }
            } else {
                (Vec::new(), None)
            };

            results.push(StructureResult {
                structure_id,
                total_steps,
                wall_time_ms,
                spike_events,
                snapshots: self.engine.get_snapshots(),
                final_temperature: summary.end_temperature,
                clustered_sites,
                clustering_stats,
            });

            log::info!(
                "  ✓ Completed in {}ms ({:.1} steps/sec)",
                wall_time_ms,
                total_steps as f64 / (wall_time_ms as f64 / 1000.0)
            );
        }

        let total_time = batch_start.elapsed();
        let stats = self.engine.stats();

        log::info!("\n═══════════════════════════════════════════════════════════════");
        log::info!("  BATCH COMPLETE");
        log::info!("═══════════════════════════════════════════════════════════════");
        log::info!("  Structures processed: {}", stats.structures_processed);
        log::info!("  Total steps: {}", stats.total_steps_run);
        log::info!("  Total wall time: {:.1}s", total_time.as_secs_f64());
        log::info!(
            "  Overhead saved (persistent): {}ms",
            stats.overhead_saved_ms
        );
        log::info!(
            "  Avg throughput: {:.0} steps/sec",
            stats.total_steps_run as f64 / total_time.as_secs_f64()
        );

        Ok(results)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// VISUALIZATION OUTPUT FORMATTERS
// ═══════════════════════════════════════════════════════════════════════════════

/// Output formatter for binding site visualization
pub struct BindingSiteFormatter;

impl BindingSiteFormatter {
    /// Generate PDB file with pseudo-atoms at binding site centroids
    ///
    /// Creates HETATM records with:
    /// - Atom name "BS" (Binding Site)
    /// - Residue name based on classification (DRG=druggable, CRY=cryptic, etc.)
    /// - B-factor = quality score * 100
    /// - Occupancy = druggability score
    pub fn to_pdb(sites: &[ClusteredBindingSite]) -> String {
        let mut pdb = String::new();
        pdb.push_str("REMARK   PRISM4D Binding Site Detection Results\n");
        pdb.push_str("REMARK   Pseudo-atoms at binding site centroids\n");
        pdb.push_str("REMARK   B-factor = quality score * 100\n");
        pdb.push_str("REMARK   Occupancy = druggability score\n");
        pdb.push_str("REMARK\n");

        for (idx, site) in sites.iter().enumerate() {
            let atom_num = idx + 1;
            let res_name = match site.classification {
                SiteClassification::ActiveSite => "ACT",
                SiteClassification::Allosteric => "ALO",
                SiteClassification::Cryptic => "CRY",
                SiteClassification::PpiSurface => "PPI",
                SiteClassification::MembraneInterface => "MEM",
                SiteClassification::Unknown => "UNK",
            };

            // PDB HETATM format
            // HETATM    1  BS  DRG A   1      10.000  20.000  30.000  0.80 50.00
            // Emission-compat path: PDB HETATM record uses the
            // legacy emission centroid scalar. Canonical multi-view
            // adjudication is handled by Phase 4; this serializer
            // stays on the compat accessor for backward compatibility
            // with external PDB consumers.
            let c = site.emission_compat_centroid();
            pdb.push_str(&format!(
                "HETATM{:5}  BS  {} A{:4}    {:8.3}{:8.3}{:8.3}{:6.2}{:6.2}\n",
                atom_num,
                res_name,
                atom_num,
                c[0],
                c[1],
                c[2],
                site.druggability.overall,
                site.quality_score * 100.0,
            ));
        }

        pdb.push_str("END\n");
        pdb
    }

    /// Generate PyMOL script for visualization
    ///
    /// Creates industry-standard pocket visualization with:
    /// - Lining residues shown as sticks
    /// - Pocket surface (transparent)
    /// - Residue type coloring (catalytic=magenta, aromatic=green, hydrophobic=yellow)
    /// - Centroid marker (small sphere)
    pub fn to_pymol(sites: &[ClusteredBindingSite], structure_name: &str) -> String {
        let mut script = String::new();

        script.push_str("# PRISM4D Binding Site Visualization\n");
        script.push_str("# Generated by prism-nhs\n");
        script.push_str("# Industry-standard pocket visualization\n\n");

        // Setup commands
        script.push_str("# Setup\n");
        script.push_str("bg_color white\n");
        script.push_str("set cartoon_fancy_helices, 1\n");
        script.push_str("set cartoon_side_chain_helper, 1\n");
        script.push_str("set surface_quality, 1\n\n");

        // Load structure placeholder
        if !structure_name.is_empty() {
            script.push_str(&format!("# Load your structure (adjust path as needed)\n"));
            script.push_str(&format!("# load {}.pdb, protein\n", structure_name));
            script.push_str("# show cartoon, protein\n");
            script.push_str("# color gray80, protein\n\n");
        }

        // Process each binding site — emission layer (PyMOL/ChimeraX).
        for (idx, site) in sites.iter().enumerate() {
            let site_num = idx + 1;
            // Emission-compat scalar for pseudoatom placement.
            let [x, y, z] = site.emission_compat_centroid();
            let druggable_tag = if site.druggability.is_druggable {
                " [DRUGGABLE]"
            } else {
                ""
            };

            script.push_str(&format!(
                "# ========== Site {} ({:?}){} ==========\n",
                site_num, site.classification, druggable_tag
            ));

            // Skip if no lining residues
            if site.lining_residues.is_empty() {
                script.push_str(&format!("# No lining residues for site {}\n", site_num));
                script.push_str(&format!(
                    "pseudoatom site_{}_center, pos=[{:.3}, {:.3}, {:.3}]\n",
                    site_num, x, y, z
                ));
                script.push_str(&format!("show spheres, site_{}_center\n", site_num));
                script.push_str(&format!(
                    "set sphere_scale, 1.0, site_{}_center\n",
                    site_num
                ));
                script.push_str(&format!("color red, site_{}_center\n\n", site_num));
                continue;
            }

            // Build residue selection string
            let residue_sel: Vec<String> = site
                .lining_residues
                .iter()
                .map(|r| format!("(chain {} and resi {})", r.chain, r.resid))
                .collect();
            let sel_str = residue_sel.join(" or ");

            // Create selection for pocket lining residues
            script.push_str(&format!("select pocket_{}_lining, {}\n", site_num, sel_str));

            // Show lining residues as sticks
            script.push_str(&format!("show sticks, pocket_{}_lining\n", site_num));
            script.push_str(&format!(
                "set stick_radius, 0.15, pocket_{}_lining\n",
                site_num
            ));

            // Color by residue type
            let catalytic = ["GLU", "ASP", "HIS", "SER", "CYS", "LYS"];
            let aromatic = ["PHE", "TYR", "TRP"];
            let hydrophobic = ["ALA", "VAL", "LEU", "ILE", "MET", "PRO"];

            // Categorize residues
            let cat_residues: Vec<_> = site
                .lining_residues
                .iter()
                .filter(|r| catalytic.contains(&r.resname.as_str()))
                .collect();
            let aro_residues: Vec<_> = site
                .lining_residues
                .iter()
                .filter(|r| aromatic.contains(&r.resname.as_str()))
                .collect();
            let hydro_residues: Vec<_> = site
                .lining_residues
                .iter()
                .filter(|r| hydrophobic.contains(&r.resname.as_str()))
                .collect();

            // Color catalytic residues (magenta/hotpink)
            if !cat_residues.is_empty() {
                let cat_sel: Vec<String> = cat_residues
                    .iter()
                    .map(|r| format!("(chain {} and resi {})", r.chain, r.resid))
                    .collect();
                script.push_str(&format!(
                    "select pocket_{}_catalytic, {}\n",
                    site_num,
                    cat_sel.join(" or ")
                ));
                script.push_str(&format!("color magenta, pocket_{}_catalytic\n", site_num));
            }

            // Color aromatic residues (green)
            if !aro_residues.is_empty() {
                let aro_sel: Vec<String> = aro_residues
                    .iter()
                    .map(|r| format!("(chain {} and resi {})", r.chain, r.resid))
                    .collect();
                script.push_str(&format!(
                    "select pocket_{}_aromatic, {}\n",
                    site_num,
                    aro_sel.join(" or ")
                ));
                script.push_str(&format!("color forest, pocket_{}_aromatic\n", site_num));
            }

            // Color hydrophobic residues (yellow)
            if !hydro_residues.is_empty() {
                let hydro_sel: Vec<String> = hydro_residues
                    .iter()
                    .map(|r| format!("(chain {} and resi {})", r.chain, r.resid))
                    .collect();
                script.push_str(&format!(
                    "select pocket_{}_hydrophobic, {}\n",
                    site_num,
                    hydro_sel.join(" or ")
                ));
                script.push_str(&format!(
                    "color tv_yellow, pocket_{}_hydrophobic\n",
                    site_num
                ));
            }

            // Create pocket surface (transparent)
            script.push_str(&format!(
                "create pocket_{}_surface, pocket_{}_lining\n",
                site_num, site_num
            ));
            script.push_str(&format!("show surface, pocket_{}_surface\n", site_num));
            script.push_str(&format!(
                "set surface_color, slate, pocket_{}_surface\n",
                site_num
            ));
            script.push_str(&format!(
                "set transparency, 0.7, pocket_{}_surface\n",
                site_num
            ));

            // Small centroid marker
            script.push_str(&format!(
                "pseudoatom pocket_{}_center, pos=[{:.3}, {:.3}, {:.3}]\n",
                site_num, x, y, z
            ));
            script.push_str(&format!("show spheres, pocket_{}_center\n", site_num));
            script.push_str(&format!(
                "set sphere_scale, 0.5, pocket_{}_center\n",
                site_num
            ));
            let center_color = if site.druggability.is_druggable {
                "red"
            } else {
                "gray50"
            };
            script.push_str(&format!(
                "color {}, pocket_{}_center\n",
                center_color, site_num
            ));

            // Group all pocket objects
            script.push_str(&format!(
                "group pocket_{}, pocket_{}_lining pocket_{}_surface pocket_{}_center",
                site_num, site_num, site_num, site_num
            ));
            if !cat_residues.is_empty() {
                script.push_str(&format!(" pocket_{}_catalytic", site_num));
            }
            if !aro_residues.is_empty() {
                script.push_str(&format!(" pocket_{}_aromatic", site_num));
            }
            if !hydro_residues.is_empty() {
                script.push_str(&format!(" pocket_{}_hydrophobic", site_num));
            }
            script.push_str("\n\n");
        }

        // Group all pockets
        if !sites.is_empty() {
            let pocket_groups: Vec<String> =
                (1..=sites.len()).map(|i| format!("pocket_{}", i)).collect();
            script.push_str(&format!(
                "group all_pockets, {}\n\n",
                pocket_groups.join(" ")
            ));
        }

        // Add legend and usage tips
        script.push_str("# ========== Color Legend ==========\n");
        script.push_str("# Magenta = Catalytic residues (GLU, ASP, HIS, SER, CYS, LYS)\n");
        script.push_str("# Green = Aromatic residues (PHE, TYR, TRP) - pi-stacking\n");
        script.push_str("# Yellow = Hydrophobic residues (ALA, VAL, LEU, ILE, MET, PRO)\n");
        script.push_str("# Slate surface = Pocket cavity\n");
        script.push_str("# Red sphere = Druggable site centroid\n");
        script.push_str("# Gray sphere = Non-druggable site centroid\n\n");

        script.push_str("# ========== Usage Tips ==========\n");
        script.push_str("# To view specific pocket: disable all_pockets, enable pocket_N\n");
        script.push_str("# To zoom on pocket: zoom pocket_N_lining\n");
        script.push_str("# To show H-bonds: select donors, pocket_N_lining; distance hbonds, donors, acceptors, 3.5\n");
        script.push_str(
            "# To label residues: label pocket_N_lining and name CA, \"%s%s\" % (resn, resi)\n",
        );

        script
    }

    /// Generate ChimeraX script for visualization
    ///
    /// Creates industry-standard pocket visualization with:
    /// - Lining residues shown as sticks
    /// - Pocket surface (transparent)
    /// - Residue type coloring
    pub fn to_chimerax(sites: &[ClusteredBindingSite], structure_name: &str) -> String {
        let mut script = String::new();

        script.push_str("# PRISM4D Binding Site Visualization\n");
        script.push_str("# Generated by prism-nhs\n");
        script.push_str("# Industry-standard pocket visualization\n\n");

        // Setup
        script.push_str("# Setup\n");
        script.push_str("set bgColor white\n");
        script.push_str("lighting soft\n\n");

        // Load structure placeholder
        if !structure_name.is_empty() {
            script.push_str("# Load your structure (adjust path as needed)\n");
            script.push_str(&format!("# open {}.pdb\n", structure_name));
            script.push_str("# cartoon\n");
            script.push_str("# color #1 gray80\n\n");
        }

        // Process each binding site — emission layer (PyMOL/ChimeraX).
        for (idx, site) in sites.iter().enumerate() {
            let site_num = idx + 1;
            // Emission-compat scalar for pseudoatom placement.
            let [x, y, z] = site.emission_compat_centroid();
            let druggable_tag = if site.druggability.is_druggable {
                " [DRUGGABLE]"
            } else {
                ""
            };

            script.push_str(&format!(
                "# ========== Site {} ({:?}){} ==========\n",
                site_num, site.classification, druggable_tag
            ));

            // Skip if no lining residues
            if site.lining_residues.is_empty() {
                script.push_str(&format!("# No lining residues for site {}\n", site_num));
                script.push_str(&format!(
                    "marker #10{} position {:.3},{:.3},{:.3} color red radius 2.0\n\n",
                    site_num, x, y, z
                ));
                continue;
            }

            // Build residue selection string for ChimeraX
            let residue_sel: Vec<String> = site
                .lining_residues
                .iter()
                .map(|r| format!("/{}:{}", r.chain, r.resid))
                .collect();
            let sel_str = residue_sel.join(",");

            // Select and show lining residues
            script.push_str(&format!("# Pocket {} lining residues\n", site_num));
            script.push_str(&format!("name pocket{}_lining #1{}\n", site_num, sel_str));
            script.push_str(&format!("show pocket{}_lining atoms\n", site_num));
            script.push_str(&format!("style pocket{}_lining stick\n", site_num));

            // Categorize and color residues
            let catalytic = ["GLU", "ASP", "HIS", "SER", "CYS", "LYS"];
            let aromatic = ["PHE", "TYR", "TRP"];
            let hydrophobic = ["ALA", "VAL", "LEU", "ILE", "MET", "PRO"];

            // Color catalytic residues (magenta)
            let cat_residues: Vec<_> = site
                .lining_residues
                .iter()
                .filter(|r| catalytic.contains(&r.resname.as_str()))
                .collect();
            if !cat_residues.is_empty() {
                let cat_sel: Vec<String> = cat_residues
                    .iter()
                    .map(|r| format!("/{}:{}", r.chain, r.resid))
                    .collect();
                script.push_str(&format!("color #1{} magenta\n", cat_sel.join(",")));
            }

            // Color aromatic residues (forest green)
            let aro_residues: Vec<_> = site
                .lining_residues
                .iter()
                .filter(|r| aromatic.contains(&r.resname.as_str()))
                .collect();
            if !aro_residues.is_empty() {
                let aro_sel: Vec<String> = aro_residues
                    .iter()
                    .map(|r| format!("/{}:{}", r.chain, r.resid))
                    .collect();
                script.push_str(&format!("color #1{} forest green\n", aro_sel.join(",")));
            }

            // Color hydrophobic residues (gold)
            let hydro_residues: Vec<_> = site
                .lining_residues
                .iter()
                .filter(|r| hydrophobic.contains(&r.resname.as_str()))
                .collect();
            if !hydro_residues.is_empty() {
                let hydro_sel: Vec<String> = hydro_residues
                    .iter()
                    .map(|r| format!("/{}:{}", r.chain, r.resid))
                    .collect();
                script.push_str(&format!("color #1{} gold\n", hydro_sel.join(",")));
            }

            // Create pocket surface
            script.push_str(&format!("surface pocket{}_lining\n", site_num));
            script.push_str(&format!("transparency pocket{}_lining 70\n", site_num));
            script.push_str(&format!("color pocket{}_lining slate gray\n", site_num));

            // Centroid marker
            let marker_color = if site.druggability.is_druggable {
                "red"
            } else {
                "gray"
            };
            script.push_str(&format!(
                "marker #10{} position {:.3},{:.3},{:.3} color {} radius 1.0\n",
                site_num, x, y, z, marker_color
            ));

            // Label
            script.push_str(&format!("2dlabels text \"Pocket {} ({:.0}% druggable){}\" xpos 0.02 ypos {:.2} color black size 14\n",
                site_num,
                site.druggability.overall * 100.0,
                if site.druggability.is_druggable { " *" } else { "" },
                0.95 - idx as f32 * 0.04
            ));

            script.push_str("\n");
        }

        // Legend
        script.push_str("# ========== Color Legend ==========\n");
        script.push_str("# Magenta = Catalytic residues (GLU, ASP, HIS, SER, CYS, LYS)\n");
        script.push_str("# Forest Green = Aromatic residues (PHE, TYR, TRP)\n");
        script.push_str("# Gold = Hydrophobic residues (ALA, VAL, LEU, ILE, MET, PRO)\n");
        script.push_str("# Slate surface = Pocket cavity\n");
        script.push_str("# Red marker = Druggable site centroid\n");
        script.push_str("# Gray marker = Non-druggable site centroid\n\n");

        script.push_str("# ========== Usage ==========\n");
        script.push_str("# To view pocket: view pocket1_lining\n");
        script.push_str("# To hide surface: hide pocket1_lining surface\n");
        script.push_str("# To show H-bonds: hbonds pocket1_lining\n");

        script
    }

    /// Generate summary report in Markdown format
    pub fn to_markdown_report(
        sites: &[ClusteredBindingSite],
        structure_name: &str,
        persistence: Option<&PersistenceAnalysis>,
    ) -> String {
        let mut report = String::new();

        report.push_str(&format!(
            "# PRISM4D Binding Site Analysis: {}\n\n",
            structure_name
        ));

        // Summary statistics
        report.push_str("## Summary\n\n");
        report.push_str(&format!("- **Total Sites Detected:** {}\n", sites.len()));

        let druggable_count = sites.iter().filter(|s| s.druggability.is_druggable).count();
        report.push_str(&format!("- **Druggable Sites:** {}\n", druggable_count));

        // Classification breakdown
        let mut class_counts = std::collections::HashMap::new();
        for site in sites {
            *class_counts
                .entry(format!("{:?}", site.classification))
                .or_insert(0) += 1;
        }
        report.push_str("\n### Classification Breakdown\n\n");
        for (class, count) in class_counts {
            report.push_str(&format!("- {}: {}\n", class, count));
        }

        // Persistence info if available
        if let Some(pers) = persistence {
            report.push_str("\n## Persistence Analysis\n\n");
            report.push_str(&format!(
                "- **Total Frames Analyzed:** {}\n",
                pers.total_frames
            ));
            report.push_str(&format!(
                "- **Persistent Sites (>50% frames):** {}\n",
                pers.persistent_count
            ));
            report.push_str(&format!(
                "- **Transient Sites (<50% frames):** {}\n",
                pers.transient_count
            ));
            report.push_str(&format!(
                "- **Average Site Lifetime:** {:.1} frames\n",
                pers.avg_lifetime
            ));
        }

        // Top sites table
        report.push_str("\n## Top Binding Sites\n\n");
        report.push_str(
            "| Rank | Position (Å) | Volume (Å³) | Spikes | Quality | Druggable | Class |\n",
        );
        report.push_str(
            "|------|--------------|-------------|--------|---------|-----------|-------|\n",
        );

        for (idx, site) in sites.iter().take(10).enumerate() {
            let c = site.emission_compat_centroid();
            report.push_str(&format!(
                "| {} | ({:.1}, {:.1}, {:.1}) | {:.0} | {} | {:.2} | {} | {:?} |\n",
                idx + 1,
                c[0],
                c[1],
                c[2],
                site.estimated_volume,
                site.spike_count,
                site.quality_score,
                if site.druggability.is_druggable {
                    "✓"
                } else {
                    "✗"
                },
                site.classification,
            ));
        }

        // Lining residues for top sites
        report.push_str("\n## Binding Site Residues (5Å cutoff)\n\n");
        for (idx, site) in sites.iter().take(10).enumerate() {
            if !site.lining_residues.is_empty() {
                report.push_str(&format!(
                    "### Site {} ({:?})\n\n",
                    idx + 1,
                    site.classification
                ));
                report.push_str("| Chain | ResID | ResName | Distance (Å) | Atoms |\n");
                report.push_str("|-------|-------|---------|--------------|-------|\n");
                for res in site.lining_residues.iter().take(20) {
                    report.push_str(&format!(
                        "| {} | {} | {} | {:.2} | {} |\n",
                        res.chain, res.resid, res.resname, res.min_distance, res.n_atoms_in_pocket
                    ));
                }
                if site.lining_residues.len() > 20 {
                    report.push_str(&format!(
                        "| ... | {} more residues | | | |\n",
                        site.lining_residues.len() - 20
                    ));
                }
                report.push_str("\n");
            }
        }

        report
    }
}

/// Write binding sites to multiple visualization formats
pub fn write_binding_site_visualizations(
    sites: &[ClusteredBindingSite],
    base_path: &std::path::Path,
    structure_name: &str,
) -> Result<()> {
    use std::fs;
    use std::io::Write;

    // Write PDB
    let pdb_path = base_path.with_extension("binding_sites.pdb");
    let mut pdb_file = fs::File::create(&pdb_path)?;
    pdb_file.write_all(BindingSiteFormatter::to_pdb(sites).as_bytes())?;
    log::info!("Wrote binding sites PDB: {}", pdb_path.display());

    // Write PyMOL script
    let pml_path = base_path.with_extension("binding_sites.pml");
    let mut pml_file = fs::File::create(&pml_path)?;
    pml_file.write_all(BindingSiteFormatter::to_pymol(sites, structure_name).as_bytes())?;
    log::info!("Wrote PyMOL script: {}", pml_path.display());

    // Write ChimeraX script
    let cxc_path = base_path.with_extension("binding_sites.cxc");
    let mut cxc_file = fs::File::create(&cxc_path)?;
    cxc_file.write_all(BindingSiteFormatter::to_chimerax(sites, structure_name).as_bytes())?;
    log::info!("Wrote ChimeraX script: {}", cxc_path.display());

    // Write Markdown report
    let md_path = base_path.with_extension("binding_sites.md");
    let mut md_file = fs::File::create(&md_path)?;
    md_file.write_all(
        BindingSiteFormatter::to_markdown_report(sites, structure_name, None).as_bytes(),
    )?;
    log::info!("Wrote Markdown report: {}", md_path.display());

    Ok(())
}

// Stub for non-GPU builds
#[cfg(not(feature = "gpu"))]
pub struct PersistentNhsEngine;

#[cfg(not(feature = "gpu"))]
impl PersistentNhsEngine {
    pub fn new(_config: &PersistentBatchConfig) -> Result<Self> {
        bail!("GPU feature required for PersistentNhsEngine")
    }
}

#[cfg(not(feature = "gpu"))]
pub struct BatchProcessor;

#[cfg(not(feature = "gpu"))]
impl BatchProcessor {
    pub fn new(_config: PersistentBatchConfig) -> Result<Self> {
        bail!("GPU feature required for BatchProcessor")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(feature="gpu",feature="v2_ignition"))]
    #[test]
    #[ignore = "explicit GPU, current native PTX and pinned topology required"]
    fn resident_physics_ladd_kcc_v2_parent_preserves_exact_endpoints() {
        run_resident_physical_parent_fixture(false, false, false);
    }

    #[cfg(all(feature="gpu",feature="v2_ignition"))]
    #[test]
    #[ignore = "explicit GPU and native PTX; active native ASC and armed G26 owners on pinned topology"]
    fn resident_physics_parent_active_asc_g26_preserves_owner_epochs() {
        run_resident_physical_parent_fixture(true, false, false);
    }

    /// Diagnostic intervention isolation only: all physical guards and active
    /// owners remain unchanged; the legacy coordinate projection mask is empty.
    #[test]
    #[ignore = "requires actual native GPU/PTX and pinned molecular topology"]
    #[cfg(all(feature="gpu",feature="v2_ignition"))]
    fn resident_physical_active_controls_without_ca_projection_diagnostic() {
        run_resident_physical_parent_fixture(true, true, false);
    }

    #[cfg(all(feature="gpu",feature="v2_ignition"))]
    #[test]
    #[ignore="actual pinned NMA modes, molecular topology and native captured GPU physical phases"]
    fn resident_physical_parent_native_nma_force_work_population() {
        run_resident_physical_parent_fixture(false,false,true);
    }

    #[cfg(all(feature="gpu",feature="v2_ignition"))]
    #[test]
    #[ignore="actual pinned NMA modes with active ASC and armed physical guard"]
    fn resident_physical_parent_native_nma_active_controls() {
        run_resident_physical_parent_fixture(true,false,true);
    }

    #[cfg(all(feature="gpu",feature="v2_ignition"))]
    fn run_resident_physical_parent_fixture(active_controls: bool, isolate_ca_projection: bool, native_nma:bool) {
        use cudarc::driver::DevicePtr;
        use crate::captured_pipeline::{PipelineConfig,measured_kcc_causal_leads};
        use crate::rich_spike::RichSpike;
        use sha2::{Digest,Sha256};
        let path = std::env::var("PRISM_KCC_TEST_TOPOLOGY").unwrap();
        assert_eq!(format!("{:x}",Sha256::digest(std::fs::read(&path).unwrap())),
            "861be5f35178fa5b5f15022bda7f6b6ffa9febf679d4ff1932669a4accf0b4d3");
        let topology = PrismPrepTopology::load(&path).unwrap();
        let config = PersistentBatchConfig {
            max_atoms:topology.n_atoms,grid_dim:32,grid_spacing:2.0,..Default::default()
        };
        // Match the actual multi-stream production constructor. CUDA cannot
        // capture the legacy NULL stream returned by the convenience owner.
        let module_owner = PersistentNhsEngine::new(&config).unwrap();
        let stream = module_owner.context.new_stream().unwrap();
        let mut engine = PersistentNhsEngine::new_on_stream(&config,
            module_owner.context.clone(), module_owner.module.clone(), stream).unwrap();
        drop(module_owner);
        engine.load_topology(&topology).unwrap();
        if isolate_ca_projection {
            assert!(active_controls);
            engine.fused_engine_mut().unwrap().disable_ca_projection_for_diagnostic().unwrap();
            eprintln!("PRISM_DIAGNOSTIC_CA_PROJECTION_OFF physical_guards=unchanged controls=active production_policy_unchanged=true");
        }
        let mut protocol = CryoUvProtocol::standard();
        protocol.cold_hold_steps=30;protocol.ramp_steps=10;protocol.warm_hold_steps=10;
        protocol.ramp_down_steps=10;protocol.cold_return_steps=10;
        engine.set_cryo_uv_protocol(protocol).unwrap();
        let mut nma_records=Vec::new();let mut nma_nonzero=0u64;
        let mut nma_modes_json=serde_json::Value::Null;
        let nma_path=std::env::temp_dir().join(format!("prism_nma_native_{}_{}.f64le",std::process::id(),active_controls));
        let mut nma_file=if native_nma {
            let modes=std::env::var("PRISM_NMA_TEST_MODES").unwrap();
            assert_eq!(format!("{:x}",Sha256::digest(std::fs::read(&modes).unwrap())),
                "a612184a8917056bbe2a20f422b0649d679b10849cf50658cf14876c5950200d");
            nma_modes_json=serde_json::from_slice(&std::fs::read(&modes).unwrap()).unwrap();
            engine.load_nma_modes(&modes).unwrap();
            engine.fused_engine_mut().unwrap().bind_nma_work_journal(70,10).unwrap();
            Some(std::fs::File::create(&nma_path).unwrap())
        } else {None};

        engine.fused_engine_mut().unwrap().bind_ca_intervention_journal(70,10).unwrap();
        let ca_fixture_path=std::env::var("PRISM_CA_WORK_TEST_OUTPUT").map(std::path::PathBuf::from)
            .unwrap_or_else(|_|std::env::temp_dir().join(format!("prism_ca_work_{}_{}_{}_{}.f64le",std::process::id(),active_controls,isolate_ca_projection,native_nma)));
        let mut ca_fixture=std::fs::File::create(&ca_fixture_path).unwrap();
        let mut ca_fixture_index=std::fs::File::create(ca_fixture_path.with_extension("index.jsonl")).unwrap();

        engine.set_ladd_enabled(true);
        let stream = engine.cuda_stream().clone();
        // Use the same pre-capture owner attachment as nhs_rt_full. Policy
        // resolution retains unknown topology permissions; no HMR or solvent
        // authority is invented by this scheduling regression.
        let (governor, asc) = if active_controls {
            // Physical guard decisions run at every completed physical step.
            // This cadence is independent of the ten-step V2 observer window.
            let resolved = crate::gearbox::resolve_governor_policy(
                crate::gearbox::TopologyPermissions::default(), 1,
                &crate::gearbox::default_gearbox_table()).unwrap();
            assert_eq!(resolved.policy.armed, 1);
            assert_eq!(resolved.policy.chunk_span_steps, 1);
            let clusters = engine.fused_engine().unwrap().n_hydrogen_constraint_clusters();
            let governor = crate::governor_device::GovernorDevice::new(
                stream.clone(), &resolved.policy, 0, clusters).unwrap();
            engine.fused_engine_mut().unwrap().set_governor_device(governor.clone()).unwrap();
            let clock=engine.fused_engine().unwrap().d_protocol_current_step_dev_ptr(&stream);
            assert!(unsafe{governor.bind_physical_time(clock,0)}.is_err());
            unsafe{governor.bind_physical_time(clock,70)}.unwrap();
            assert!(unsafe{governor.bind_physical_time(clock,70)}.is_err(),"cannot rebind a live journal");
            assert!(governor.physical_time_records(0,1).is_err(),"capture setup cannot fabricate elapsed time");
            let asc = crate::asc_device::AscDevice::new_legacy_whole_system(
                stream.clone(), 0, topology.n_atoms,
                "861be5f35178fa5b5f15022bda7f6b6ffa9febf679d4ff1932669a4accf0b4d3",
                0.005).unwrap();
            engine.fused_engine_mut().unwrap().set_asc_device(asc.clone()).unwrap();
            engine.set_gearbox_active(true);
            (Some(governor),Some(asc))
        } else { (None,None) };
        let asc_mask = stream.clone_htod(&vec![1u32;topology.n_atoms]).unwrap();
        let mut control_rows = Vec::new();
        let mut physical_time_rows:Vec<crate::governor_device::PhysicalTimeRecord>=Vec::new();
        let mut exact_cumulative_ps=0.0;
        let positions = engine.get_positions().unwrap();
        let initial_positions = positions.clone();
        // Fixed cluster membership is regression input, not a site detection.
        // Physics, LADD, KCC, descriptors, V2 and Ghost execute their native code.
        let spikes: Vec<RichSpike> = (0..16usize).map(|atom| {
            let mut s=RichSpike::zero();s.x=positions[atom*3];s.y=positions[atom*3+1];s.z=positions[atom*3+2];
            s.cluster_id=(atom/8) as i32;s.residue_id=atom as i32;s
        }).collect();
        let bytes=unsafe {std::slice::from_raw_parts(spikes.as_ptr().cast::<u8>(),spikes.len()*std::mem::size_of::<RichSpike>())};
        let spike_storage=stream.clone_htod(bytes).unwrap();
        let offsets=stream.clone_htod(&[0u32,8,16]).unwrap();
        let candidate_ids=vec![(0..16u32).collect(),Vec::new()];
        let csr=engine.prepare_kcc_causal_candidates(&candidate_ids).unwrap();
        let mut leads=stream.clone_htod(&[u32::MAX;2]).unwrap();
        let mut external_work=stream.clone_htod(&[0f64]).unwrap();
        let ghost=crate::ghost_tile::GhostTileRing::allocate(14).unwrap();
        assert_eq!(unsafe {crate::sh_basis::ffi::prism_sh_basis_init(stream.cu_stream().cast())},0);
        stream.synchronize().unwrap();
        let cfg=PipelineConfig {
            seed_layout: crate::rich_spike::NativeV2SeedLayout::default(),
            d_spikes:spike_storage.device_ptr(&stream).0 as *const RichSpike,
            d_cluster_offsets:offsets.device_ptr(&stream).0 as *const u32,
            n_clusters:2,d_k_lm:crate::sh_basis::k_lm_device_ptr().unwrap(),
            initial_frame_id:0,source_stream_id:0,diagnostic_stream_id:Some(0),
            asc:asc.as_ref().map(|owner| crate::captured_pipeline::AscConfig {
                decision_owner:Some(owner.clone()),d_forces:engine.d_forces_dev_ptr() as *mut f32,
                d_atom_positions:engine.fused_engine().unwrap().d_positions_dev_ptr(&stream) as *const f32,
                d_atom_in_cluster:asc_mask.device_ptr(&stream).0 as *const u32,
                n_atoms:topology.n_atoms as i32,steering_gain_alpha:0.005,
                d_asc_applied_atoms:std::ptr::null_mut(),d_asc_applied_sum_f:std::ptr::null_mut(),
            }),
            d_force_norm_mirror:governor.as_ref().map_or(0,|owner|owner.force_value_ptr()),
            governor_device:governor.clone(),zstr:None,sisr:None,
            noise_floor_override:None,mar_v2:None,
            d_dt:if active_controls {engine.d_protocol_dt_dev_ptr() as *mut f32}else{std::ptr::null_mut()},
            d_velocities:if active_controls {engine.d_velocities_dev_ptr() as *mut f32}else{std::ptr::null_mut()},
            g26_parent_cond_handle:0,f1_parent_cond_handle:0,
            d_pe_components:engine.d_potential_energy_components_dev_ptr() as *const f64,
            n_atoms_for_pe:topology.n_atoms as u32,
            d_external_work:if active_controls {engine.allocate_external_work_buffer() as *mut f64}
                else {external_work.device_ptr(&stream).0 as *mut f64},
            ghost_tile_ring_dev:ghost.device_base,ghost_tile_max_records:ghost.max_records,
            firehose_enable:1,body_unroll:1,d_kcc_lead:leads.device_ptr(&stream).0,
            d_forces_anchor:engine.d_forces_dev_ptr(),d_masses:engine.d_masses_dev_ptr(),
            force_burst_step:None,n_spikes:16,d_current_md_step:engine.d_protocol_current_step_dev_ptr() as *const u32,
            branch_trace_dev:0,
        };
        let ladd_initial=engine.fused_engine().unwrap().ladd_reference_snapshot_for_test().unwrap();
        assert_eq!(ladd_initial["cold_hold_steps"],30,"native protocol cold phase did not reach LADD");
        assert_eq!(ladd_initial["maximum_reference_count"],0);
        let gaussian_initial=engine.fused_engine().unwrap().multilif_reference_snapshot_for_test().unwrap();
        assert_eq!(gaussian_initial["maximum_reference_count"],0);
        assert_eq!(gaussian_initial["native_observation_inner_steps"],1);
        let mut ladd_phase_rows=Vec::new();let mut ladd_frozen=None;
        let mut gaussian_phase_rows=Vec::new();let mut gaussian_frozen=None;
        let mut parent=engine.capture_resident_observation_graph(10,&cfg,&csr,&mut leads).unwrap();
        assert_eq!(engine.fused_engine().unwrap().ladd_reference_snapshot_for_test().unwrap(),ladd_initial,
            "initial capture changed live LADD reference");
        assert_eq!(engine.fused_engine().unwrap().multilif_reference_snapshot_for_test().unwrap(),gaussian_initial,
            "initial capture changed live Gaussian reference");
        assert_eq!(engine.device_protocol_current_step().unwrap(),0);
        assert_eq!(ghost.n_frames_written(),0);
        assert_eq!(engine.get_positions().unwrap(),initial_positions,"capture advanced positions");
        let reference=parent.pipeline().frozen_baseline_readback().unwrap();
        assert_eq!(reference["byte_identical"],true);
        assert_eq!(reference["adjudicator_relaxed_binding_is_frozen"],true);
        for endpoint in 1..=7u32 {
            let receipt=engine.execute_resident_observation_window(&mut parent,10,"native_test_complete_parent").unwrap();
            assert_eq!(receipt.start_device_step,(endpoint-1)*10);
            assert_eq!(receipt.end_device_step,endpoint*10);
            assert_eq!(receipt.graph_windows[0].graph_launches,1);
            assert_eq!(parent.parity,(endpoint%2) as usize);
            let ladd=engine.fused_engine().unwrap().ladd_reference_snapshot_for_test().unwrap();
            let gaussian=engine.fused_engine().unwrap().multilif_reference_snapshot_for_test().unwrap();
            assert_eq!(gaussian["maximum_reference_count"],(endpoint*10).min(29),
                "Gaussian reference cadence differs from actual per-step native observations");
            assert_ne!(gaussian["mean_sha256"],ladd["mean_sha256"],"distinct native measured densities unexpectedly coincide");
            let expected_count=endpoint.min(2);
            assert_eq!(ladd["minimum_reference_count"],expected_count);
            assert_eq!(ladd["maximum_reference_count"],expected_count);
            if endpoint==2 {
                // Reuse the same live V2/reference owners after real cold data.
                // This even observation endpoint preserves the native parity.
                let live=parent.pipeline().clone();let v2_before=live.rebind_state_sha256().unwrap();
                let ghost_before=ghost.n_frames_written();
                let replacement=engine.capture_resident_observation_graph_with_live_v2(10,&cfg,&csr,&mut leads,live.clone()).unwrap();
                assert_eq!(engine.fused_engine().unwrap().ladd_reference_snapshot_for_test().unwrap(),ladd,
                    "recapture reset or changed LADD reference bytes");
                assert_eq!(engine.fused_engine().unwrap().multilif_reference_snapshot_for_test().unwrap(),gaussian,
                    "recapture reset or changed Gaussian reference bytes");
                assert_eq!(live.rebind_state_sha256().unwrap(),v2_before);
                assert_eq!(ghost.n_frames_written(),ghost_before);
                parent=replacement;ladd_frozen=Some(ladd.clone());
            }
            if endpoint>=3 {assert_eq!(Some(&ladd),ladd_frozen.as_ref(),
                "LADD reference changed outside configured cold phase");}
            if endpoint==3 {gaussian_frozen=Some(gaussian.clone());}
            if endpoint>=4 {assert_eq!(Some(&gaussian),gaussian_frozen.as_ref(),
                "Gaussian reference changed outside configured cold phase");}
            ladd_phase_rows.push(serde_json::json!({"completed_step":endpoint*10,"reference":ladd}));
            gaussian_phase_rows.push(serde_json::json!({"completed_step":endpoint*10,"reference":gaussian}));

            if let Some(file)=nma_file.as_mut() {
                let begin=u64::from((endpoint-1)*10);let end=u64::from(endpoint*10);
                let (receipts,rows)=engine.fused_engine().unwrap().read_nma_work_window(begin,end).unwrap();
                assert_eq!(rows.len(),10*300*12);
                for (record_index,row) in rows.chunks_exact(12).enumerate() {
                    let step=row[11] as u64;
                    let residue=record_index%300;
                    if step>=40 && step<50 {
                        let into=step as i32-40;let per_mode=3/10;
                        let mode=if into<3 {(into/per_mode.max(1)).min(9)}else{0};
                        assert_eq!(row[9],mode as f64);
                        let mode_start=40+mode*per_mode;let mode_end=mode_start+per_mode;
                        let gate=if (step as i32)<mode_start+100 {
                            0.5f32*(1.0-(((step as i32-mode_start) as f32)*std::f32::consts::PI/100.0).cos())
                        } else if (step as i32)>mode_end-100 {
                            0.5f32*(1.0+(((step as i32-(mode_end-100)) as f32)*std::f32::consts::PI/100.0).cos())
                        } else {1.0};
                        let amplification=if step<=40 {3.0f32}else{2.0f32};
                        assert!((row[10]-(amplification*gate) as f64).abs()<1e-6);
                        let entry=&nma_modes_json["modes"][mode as usize];
                        let scale=entry["force_scale"].as_f64().unwrap() as f32;
                        let mut expected=[0f32;3];
                        for axis in 0..3 {expected[axis]=(row[10] as f32)*scale*(entry["displacements"][residue][axis].as_f64().unwrap() as f32);}
                        let norm=(expected.iter().map(|v|v*v).sum::<f32>()).sqrt();
                        if norm>10.0 {for component in &mut expected {*component*=10.0/norm;}}
                        for axis in 0..3 {assert!((row[axis]-expected[axis] as f64).abs()<1e-5);}
                    }
                    if step<40 || step>=50 {assert!(row[..6].iter().all(|v|*v==0.0));}
                    if row[..3].iter().any(|v|*v!=0.0){nma_nonzero+=1;}
                    assert_eq!(&row[..3],&row[3..6]);
                }
                let bytes=unsafe{std::slice::from_raw_parts(rows.as_ptr().cast::<u8>(),rows.len()*8)};
                std::io::Write::write_all(file,bytes).unwrap();std::io::Write::flush(file).unwrap();
                nma_records.push(serde_json::json!({"start":begin,"end":end,"sha256":format!("{:x}",Sha256::digest(bytes)),"rows":receipts,
                    "native_parameters":engine.fused_engine().unwrap().nma_parameter_state()}));
                engine.fused_engine_mut().unwrap().commit_nma_work_window(begin,end).unwrap();
                // Live feedback after a completed endpoint must reach the SAME
                // captured graph without rebuilding or changing the current step.
                if endpoint==4 {engine.set_nma_amplification(2.0).unwrap();}
            }
            let kcc=engine.read_composed_kcc_endpoint(endpoint*10,true).unwrap();
            let expected=measured_kcc_causal_leads(&candidate_ids,&kcc,(endpoint*10) as u64).unwrap()
                .iter().map(|row|row.causal_lead_residue.unwrap_or(u32::MAX)).collect::<Vec<_>>();
            assert_eq!(stream.clone_dtoh(&leads).unwrap(),expected);
            assert_eq!(ghost.n_frames_written(),endpoint*2,"lost/duplicate Ghost endpoint");
            let ca_work=engine.fused_engine().unwrap().ca_coordinate_intervention_readback().unwrap();
            assert_eq!(ca_work["completed_interventions"].as_u64(),Some(u64::from(endpoint*10)));
            let ca_rows=ca_work["records"].as_array().unwrap();
            assert_eq!(ca_rows.len(),(endpoint*10) as usize);
            let mut ca_sum=0.0;
            for (ordinal,row) in ca_rows.iter().enumerate() {
                assert_eq!(row["physical_step_ordinal"].as_u64(),Some(ordinal as u64+1));
                ca_sum+=row["coordinate_intervention_work_kcal_mol"].as_f64().unwrap();
            }
            assert_eq!(ca_sum,ca_work["cumulative_coordinate_intervention_work_kcal_mol"].as_f64().unwrap());
            let ca_begin=u64::from((endpoint-1)*10);let ca_end=u64::from(endpoint*10);
            let (ca_step_rows,ca_atoms)=engine.fused_engine().unwrap().read_ca_intervention_atom_window(ca_begin,ca_end).unwrap();
            assert_eq!(ca_atoms.len(),10*topology.n_atoms*3);
            let ca_bytes=unsafe{std::slice::from_raw_parts(ca_atoms.as_ptr().cast::<u8>(),ca_atoms.len()*8)};
            std::io::Write::write_all(&mut ca_fixture,ca_bytes).unwrap();
            std::io::Write::flush(&mut ca_fixture).unwrap();
            serde_json::to_writer(&mut ca_fixture_index,&serde_json::json!({"first_step":ca_begin+1,"last_step":ca_end,
                "sha256":format!("{:x}",Sha256::digest(ca_bytes)),"rows":ca_step_rows})).unwrap();
            std::io::Write::write_all(&mut ca_fixture_index,b"\n").unwrap();
            std::io::Write::flush(&mut ca_fixture_index).unwrap();
            engine.fused_engine_mut().unwrap().commit_ca_intervention_atom_window(ca_begin,ca_end).unwrap();

            assert!(engine.get_positions().unwrap().iter().all(|value|value.is_finite()));
            if let (Some(governor),Some(asc)) = (&governor,&asc) {
                let step=u64::from(endpoint*10);
                let time_rows=governor.physical_time_records(physical_time_rows.len() as u64,step).unwrap();
                assert_eq!(time_rows.len(),10);
                for row in time_rows {
                    assert_eq!(row.step,physical_time_rows.len() as u64+1);
                    assert!(row.applied_dt_ps.is_finite()&&row.applied_dt_ps>0.0);
                    exact_cumulative_ps+=row.applied_dt_ps;
                    assert_eq!(row.cumulative_simulated_time_ps,exact_cumulative_ps);
                    physical_time_rows.push(row);
                }
                assert!(governor.physical_time_records(step,step+1).is_err(),"future unexecuted time must remain absent");
                let observed_asc=asc.snapshot().unwrap();
                assert_eq!(observed_asc.completed_epoch,step);
                assert_eq!(observed_asc.published_epoch,step);
                assert_eq!(observed_asc.decisions_produced,u64::from(endpoint));
                assert_eq!(observed_asc.coverage_failures,0);
                let (observed_governor,_)=governor.snapshot().unwrap();
                assert_eq!(observed_governor.state.governor_armed,1);
                assert_eq!(observed_governor.last_physical_epoch,step);
                let guard=governor.physical_guard_snapshot().unwrap().unwrap();
                assert_eq!(guard.physical_eval_epoch,step);
                assert_eq!(guard.science_epoch,step);
                assert_eq!(guard.science_publications,u64::from(endpoint));
                assert_eq!(guard.max_evaluation_gap,1);
                // Preserve measured control vectors/work and status, including
                // legitimate zero-strength decisions. No constructed event or
                // imposed adjudication code substitutes for the native result.
                control_rows.push(serde_json::json!({"physical_step":step,
                    "asc":asc.receipt().unwrap(),"governor":governor.receipt().unwrap()}));
            }
        }
        if native_nma {assert!(nma_nonzero>0,"NMA phase nodes never applied actual force");}
        assert_ne!(engine.get_positions().unwrap(),initial_positions,"physical graph never evolved positions");
        assert_eq!(parent.pipeline().frozen_baseline_readback().unwrap()["frozen_sha256"],reference["frozen_sha256"]);
        println!("{}",serde_json::json!({"schema":"prism.resident_physical_observation_parent.v1",
            "scope":"native 70-step scheduling regression on pinned topology; no binding-site inference",
            "physical_steps":70,"root_replays":7,"cadence_steps":10,"ghost_records":14,
            "native_nma_enabled":native_nma,"native_nma_work_windows":nma_records,
            "native_nma_nonzero_residue_steps":nma_nonzero,"native_nma_atom_population_path":if native_nma {Some(nma_path)}else{None},
            "single_v2_owner":true,"ladd_parities":2,"cold_boundary_steps":30,
            "ladd_reference_phase_rows":ladd_phase_rows,"ladd_live_recapture_after_step":20,
            "multilif_gaussian_reference_phase_rows":gaussian_phase_rows,"native_reference_populations_separate":true,
            "device_clock_and_ladd_receipts_verified":true,"native_kcc_selection_equal":true,
            "active_asc_and_armed_g26":active_controls,"native_control_rows":control_rows,
            "ca_coordinate_intervention_work":engine.fused_engine().unwrap().ca_coordinate_intervention_readback().unwrap(),
            "ca_atom_population_path":ca_fixture_path,"ca_atom_population_bytes":std::fs::metadata(&ca_fixture_path).unwrap().len(),
            "physical_time_records":physical_time_rows,"exact_cumulative_time_ps":if active_controls {Some(exact_cumulative_ps)}else{None}}));
        // Existing setters can mutate this scalar. A root captured under the
        // old cutoff must reject before physical replay instead of selecting a
        // stale branch. This negative check advances zero additional MD steps.
        let before_stale_check=engine.device_protocol_current_step().unwrap();
        engine.fused_engine_mut().unwrap().set_ladd_cold_hold(31);
        let stale=engine.execute_resident_observation_window(&mut parent,10,"stale_cold_cutoff_rejected");
        let error=stale.expect_err("changed cold cutoff must reject before replay");
        assert!(format!("{error:#}").contains("captured LADD allocation/grid/parity binding changed"));
        assert_eq!(engine.device_protocol_current_step().unwrap(),before_stale_check);
        stream.synchronize().unwrap();drop(parent);drop(csr);drop(leads);drop(external_work);
        drop(offsets);drop(spike_storage);drop(ghost);drop(engine);
        stream.synchronize().unwrap();stream.context().check_err().unwrap();
    }

    #[test]
    fn test_default_config() {
        let config = PersistentBatchConfig::default();
        assert_eq!(config.max_atoms, 20000);
        assert_eq!(config.temperature, 300.0);
    }
}


#[cfg(feature="gpu")]
impl PersistentNhsEngine {
    pub fn enable_process_checkpoint_trace(&mut self)->Result<()> {
        self.engine.as_mut().context("checkpoint trace requires canonical loaded engine")?.enable_process_checkpoint_trace()
    }
    /// Whole native physical owner and persistent wrapper, before post-MD
    /// consumers start. CUDA allocation/binding identities are reconstructed.
    pub fn capture_process_checkpoint_fields(&self,owner:&str)->Result<Vec<crate::scientific_checkpoint::CapturedField>> {
        use crate::scientific_checkpoint::capture_host;
        anyhow::ensure!(self.lambda_runtime_binding.is_none(),"unconsumed lambda constructor binding cannot be checkpointed");
        let mut fields=self.engine.as_ref().context("checkpoint requires canonical loaded engine")?
            .capture_process_checkpoint_fields(&format!("{owner}/fused"))?;
        fields.push(capture_host(owner,"max_atoms",&self.max_atoms)?);
        fields.push(capture_host(owner,"grid_dim",&self.grid_dim)?);
        fields.push(capture_host(owner,"grid_spacing",&self.grid_spacing)?);
        fields.push(capture_host(owner,"nonbonded_execution_policy",&self.nonbonded_execution_policy)?);
        fields.push(capture_host(owner,"current_topology_id",&self.current_topology_id)?);
        fields.push(capture_host(owner,"prismcode_md_policy_admission",&self.prismcode_md_policy_admission)?);
        fields.push(capture_host(owner,"context_init_time_ms",&self.context_init_time_ms)?);
        fields.push(capture_host(owner,"module_init_time_ms",&self.module_init_time_ms)?);
        fields.push(capture_host(owner,"rt_init_time_ms",&self.rt_init_time_ms)?);
        fields.push(capture_host(owner,"structures_processed",&self.structures_processed)?);
        fields.push(capture_host(owner,"total_steps_run",&self.total_steps_run)?);
        fields.push(capture_host(owner,"total_compute_time_ms",&self.total_compute_time_ms)?);
        fields.push(capture_host(owner,"resident_graph_capture_nonce",&self.resident_graph_capture_nonce)?);
        // Clustering allocations are algorithm scratch. Their scientific inputs
        // and outputs live in fused spike histories and native output owners.
        // They are rebuilt lazily by the same compiled backend on continuation.
        Ok(fields)
    }
    /// Rebuild the wrapper on its original admitted topology/configuration,
    /// restore before any graph binding becomes visible, then recapture graphs.
    pub fn restore_process_checkpoint_before_capture(&mut self,owner:&str,fields:&[crate::scientific_checkpoint::CapturedField])->Result<()> {
        self.autonomous_chunk_clock.invalidate();
        use crate::scientific_checkpoint::{capture_host,restore_host};
        anyhow::ensure!(self.resident_graph_replay_bindings.is_empty(),"persistent restore must precede fresh graph registration");
        anyhow::ensure!(self.lambda_runtime_binding.is_none(),"unconsumed lambda constructor binding on restore");
        let field=|name:&str|fields.iter().find(|v|v.contract.owner==owner && v.contract.name==name)
            .with_context(||format!("persistent checkpoint missing {owner}/{name}"));
        anyhow::ensure!(fields.iter().filter(|v|v.contract.owner==owner).count()==13,"persistent host field population changed");
        anyhow::ensure!(field("max_atoms")?.bytes==capture_host(owner,"max_atoms",&self.max_atoms)?.bytes,"persistent authority changed: max_atoms");
        anyhow::ensure!(field("grid_dim")?.bytes==capture_host(owner,"grid_dim",&self.grid_dim)?.bytes,"persistent authority changed: grid_dim");
        anyhow::ensure!(field("grid_spacing")?.bytes==capture_host(owner,"grid_spacing",&self.grid_spacing)?.bytes,"persistent authority changed: grid_spacing");
        anyhow::ensure!(field("nonbonded_execution_policy")?.bytes==capture_host(owner,"nonbonded_execution_policy",&self.nonbonded_execution_policy)?.bytes,"persistent authority changed: nonbonded_execution_policy");
        anyhow::ensure!(field("current_topology_id")?.bytes==capture_host(owner,"current_topology_id",&self.current_topology_id)?.bytes,"persistent authority changed: current_topology_id");
        anyhow::ensure!(field("prismcode_md_policy_admission")?.bytes==capture_host(owner,"prismcode_md_policy_admission",&self.prismcode_md_policy_admission)?.bytes,"persistent authority changed: prismcode_md_policy_admission");
        let context_init_time_ms=restore_host(field("context_init_time_ms")?)?;
        let module_init_time_ms=restore_host(field("module_init_time_ms")?)?;
        let rt_init_time_ms=restore_host(field("rt_init_time_ms")?)?;
        let structures_processed=restore_host(field("structures_processed")?)?;
        let total_steps_run=restore_host(field("total_steps_run")?)?;
        let total_compute_time_ms=restore_host(field("total_compute_time_ms")?)?;
        let resident_graph_capture_nonce=restore_host(field("resident_graph_capture_nonce")?)?;
        let fused_owner=format!("{owner}/fused");
        let fused_fields=fields.iter().filter(|v|v.contract.owner==fused_owner).cloned().collect::<Vec<_>>();
        self.engine.as_mut().context("persistent restore requires loaded canonical engine")?
            .restore_process_checkpoint_fields(&fused_owner,&fused_fields)?;
        self.context_init_time_ms=context_init_time_ms;
        self.module_init_time_ms=module_init_time_ms;
        self.rt_init_time_ms=rt_init_time_ms;
        self.structures_processed=structures_processed;
        self.total_steps_run=total_steps_run;
        self.total_compute_time_ms=total_compute_time_ms;
        self.resident_graph_capture_nonce=resident_graph_capture_nonce;
        Ok(())
    }
}

#[cfg(feature="gpu")]
impl BoundResidentGraph {
    pub fn continuation_allocation_plan(&self)->Result<Vec<(String,u64)>>{self.graph.continuation_allocation_plan()}
    pub fn capture_continuation(&self,stream:&Arc<CudaStream>)->Result<crate::graph_capture::GraphContinuationV1>{self.graph.capture_continuation(stream)}
    pub fn restore_continuation(&self,stream:&Arc<CudaStream>,saved:&crate::graph_capture::GraphContinuationV1)->Result<()>{self.graph.restore_continuation(stream,saved)}
}
#[cfg(feature="v2_ignition")]
#[derive(serde::Serialize,serde::Deserialize)]
pub struct PhysicalObservationContinuationV1 {
    schema_version:u32,parity:usize,parity_plane:Vec<u8>,graph:crate::graph_capture::GraphContinuationV1,
}
#[cfg(feature="v2_ignition")]
impl BoundResidentObservationGraph {
    pub fn continuation_allocation_plan(&self)->Result<Vec<(String,u64)>> {
        let mut rows=self.bound.continuation_allocation_plan()?;
        rows.push(("physical_parent_parity".into(),u64::try_from(self.parity_device.len())?.checked_mul(4).context("physical parity extent overflow")?));Ok(rows)
    }
    pub fn capture_continuation(&self,stream:&Arc<CudaStream>)->Result<PhysicalObservationContinuationV1>{
        use cudarc::driver::DevicePtr;
        Ok(PhysicalObservationContinuationV1{schema_version:1,parity:self.parity,
            parity_plane:unsafe{crate::scientific_checkpoint::device::read_raw(stream,self.parity_device.device_ptr(stream).0,self.parity_device.len()*4)?},
            graph:self.bound.capture_continuation(stream)?})
    }
    pub fn restore_continuation(&mut self,stream:&Arc<CudaStream>,saved:&PhysicalObservationContinuationV1)->Result<()>{
        use cudarc::driver::DevicePtr;
        anyhow::ensure!(saved.schema_version==1 && saved.parity==self.parity && saved.parity_plane.len()==self.parity_device.len()*4,
            "physical observation parity owner shape changed");
        self.bound.restore_continuation(stream,&saved.graph)?;
        unsafe{crate::scientific_checkpoint::device::write_raw(stream,self.parity_device.device_ptr(stream).0,&saved.parity_plane)?;}
        // ladd_keys bind this process's engine allocations. They deliberately
        // remain the keys produced by native fresh capture, not old addresses.
        self.parity=saved.parity;Ok(())
    }
}

#[cfg(feature="v2_ignition")]
impl PersistentNhsEngine {
    /// Native reconstruction aligns the captured LADD owner permutations to
    /// the saved selector parity before a single graph is instantiated.
    pub fn capture_resident_observation_graph_for_continuation(&mut self,cadence:u32,
        cfg:&crate::captured_pipeline::PipelineConfig,candidates:&crate::captured_pipeline::KccCausalCandidates,
        leads:&mut cudarc::driver::CudaSlice<u32>,saved:&PhysicalObservationContinuationV1)
        ->Result<BoundResidentObservationGraph>{
        anyhow::ensure!(saved.schema_version==1 && saved.parity<2,"invalid physical observation continuation");
        let mut parent=self.capture_resident_observation_graph_owner(cadence,cfg,candidates,leads,None,saved.parity)?;
        parent.restore_continuation(&self.stream,saved)?;Ok(parent)
    }
}

// Prepared shared-resident continuation. Merge into persistent_engine.rs.
#[cfg(feature = "v2_ignition")]
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedResidentContinuationV1 {
    schema_version: u32,
    pub cadence: u32,
    pub source_stream_id: u32,
    #[serde(with = "crate::checkpoint_codec::bytes")]
    configuration: Vec<u8>,
    profile: [bool; 6],
    cold_end: i32,
    parity: usize,
    keys: [Vec<u64>; 2],
    step_binding: u64,
    #[serde(with = "crate::checkpoint_codec::bytes")]
    parity_plane: Vec<u8>,
    #[serde(with = "crate::checkpoint_codec::bytes")]
    pending_plane: Vec<u8>,
    #[serde(with = "crate::checkpoint_codec::bytes")]
    iterations_plane: Vec<u8>,
    pub appended: bool,
    old_parent_template: usize,
    binding: Option<(u64, u64, String, usize, usize)>,
    pub pipeline: crate::captured_pipeline::AdjudicationContinuationV1,
}
#[cfg(feature = "v2_ignition")]
impl PreparedResidentObservation {
    fn continuation_profile(&self) -> [bool; 6] {
        let p = self.profile;
        [
            p.ca_restraint_child_bound,
            p.ca_coordinate_intervention_bound,
            p.nma_force_phases_bound,
            p.nma_controls_device_ptr != 0,
            p.coupling_clear_child_bound,
            p.heartbeat_child_bound,
        ]
    }
    pub fn continuation_allocation_plan(&self)->anyhow::Result<Vec<(String,u64)>> {
        let mut rows=self.pipeline.continuation_allocation_plan()?;
        for(name,count)in [("prepared_parity",self.parity_device.len()),("prepared_pending",self.pending_device.len()),("prepared_iterations",self.iterations_device.len())] {
            rows.push((name.into(),u64::try_from(count)?.checked_mul(4).ok_or_else(||anyhow::anyhow!("prepared extent overflow"))?));
        }
        rows.push(("prepared_keys_wire".into(),crate::checkpoint_codec::encoded_size(&self.keys,u64::MAX)?));
        rows.push(("prepared_configuration_wire".into(),crate::checkpoint_codec::encoded_size(&self.config.checkpoint_configuration(),u64::MAX)?));Ok(rows)
    }
    pub fn capture_prepared_continuation(
        &self,
        run_root: &std::path::Path,
    ) -> anyhow::Result<PreparedResidentContinuationV1> {
        use crate::scientific_checkpoint::device::read_raw;
        use cudarc::driver::DevicePtr;
        self.stream.synchronize()?;
        anyhow::ensure!(
            self.parity < 2 && self.keys.iter().all(|k| k.len() == 34),
            "invalid shared LADD owner shape"
        );
        let read = |plane: &CudaSlice<u32>| unsafe {
            read_raw(
                &self.stream,
                plane.device_ptr(&self.stream).0,
                plane.len() * 4,
            )
        };
        let saved = PreparedResidentContinuationV1 {
            schema_version: 1,
            cadence: self.cadence,
            source_stream_id: self.source_stream_id(),
            configuration: crate::checkpoint_codec::to_bytes(
                &self.config.checkpoint_configuration(),
            )?,
            profile: self.continuation_profile(),
            cold_end: self.cold_end,
            parity: self.parity,
            keys: self.keys.clone(),
            step_binding: self.step_ptr,
            parity_plane: read(&self.parity_device)?,
            pending_plane: read(&self.pending_device)?,
            iterations_plane: read(&self.iterations_device)?,
            appended: self.appended,
            old_parent_template: self.parent_template_identity,
            binding: self.binding.as_ref().map(|b| {
                (
                    b.domain,
                    b.capture_nonce,
                    b.resident_structure_fingerprint_sha256.clone(),
                    b.graph_template_handle,
                    b.graph_exec_handle,
                )
            }),
            pipeline: self.pipeline.capture_continuation(run_root)?,
        };
        saved.validate()?;
        Ok(saved)
    }
    /// Fresh parts are produced from the already-restored engine. Rotate the
    /// fresh pointer permutation BEFORE append_into embeds its four bodies.
    pub fn restore_prepared_before_append(
        &mut self,
        saved: &PreparedResidentContinuationV1,
    ) -> anyhow::Result<()> {
        saved.validate()?;
        anyhow::ensure!(
            !self.appended && self.binding.is_none(),
            "prepared restore requires an unappended fresh owner"
        );
        anyhow::ensure!(
            self.cadence == saved.cadence
                && self.source_stream_id() == saved.source_stream_id
                && self.cold_end == saved.cold_end
                && self.continuation_profile() == saved.profile
                && crate::checkpoint_codec::to_bytes(&self.config.checkpoint_configuration())?
                    == saved.configuration,
            "prepared native configuration changed"
        );
        anyhow::ensure!(
            self.keys.iter().all(|k| k.len() == 34) && self.ladd.len() == 4,
            "fresh LADD template population changed"
        );
        for (fresh, old) in self.keys.iter().zip(&saved.keys) {
            anyhow::ensure!(
                fresh[22..31] == old[22..31],
                "prepared LADD grid/population/cold-phase binding changed"
            );
        }
        if saved.binding.is_some() && saved.parity == 1 {
            self.keys.swap(0, 1);
            self.ladd.swap(0, 1);
            self.ladd.swap(2, 3);
        }
        use crate::scientific_checkpoint::device::write_raw;
        use cudarc::driver::DevicePtr;
        for (plane, bytes) in [
            (&*self.parity_device, &saved.parity_plane),
            (&*self.pending_device, &saved.pending_plane),
            (&*self.iterations_device, &saved.iterations_plane),
        ] {
            anyhow::ensure!(
                plane.len() * 4 == bytes.len(),
                "prepared control plane extent changed"
            );
            unsafe {
                write_raw(&self.stream, plane.device_ptr(&self.stream).0, bytes)?;
            }
        }
        self.parity = saved.parity;
        Ok(())
    }
    pub fn restore_prepared_pipeline(
        &self,
        run_root: &std::path::Path,
        saved: &PreparedResidentContinuationV1,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.appended == saved.appended,
            "prepared owning-parent presence changed"
        );
        self.pipeline
            .restore_continuation(run_root, &saved.pipeline)
    }
}
#[cfg(feature = "v2_ignition")]
impl PreparedResidentContinuationV1 {
    pub fn configuration(
        &self,
    ) -> anyhow::Result<crate::captured_pipeline::PipelineContinuationConfigV1> {
        Ok(crate::checkpoint_codec::from_bytes(&self.configuration)?)
    }
    pub fn requires_binding_reconstruction(&self) -> bool {
        self.binding.is_some()
    }
    fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.schema_version == 1
                && self.cadence > 0
                && self.parity < 2
                && self.keys.iter().all(|k| k.len() == 34)
                && self.step_binding != 0
                && self.parity_plane.len() == 4
                && self.pending_plane.len() == 4
                && self.iterations_plane.len() == 4,
            "invalid prepared continuation population"
        );
        anyhow::ensure!(
            self.appended == (self.old_parent_template != 0)
                && (self.binding.is_none() || self.appended),
            "prepared continuation parent/binding mismatch"
        );
        if self.binding.is_some() {
            anyhow::ensure!(
                u32::from_le_bytes(self.parity_plane.as_slice().try_into()?) == self.parity as u32
                    && u32::from_le_bytes(self.pending_plane.as_slice().try_into()?) <= 1
                    && u32::from_le_bytes(self.iterations_plane.as_slice().try_into()?)
                        == self.cadence,
                "prepared checkpoint is not after a committed physical window"
            );
        }
        Ok(())
    }
}
#[cfg(feature = "v2_ignition")]
impl PersistentNhsEngine {
    /// Run on this engine's worker after global reconstruction and before the
    /// rendezvous is released. A fresh binding is actual reconstruction work;
    /// retain its nonce and trace separately from prior physical replay history.
    pub fn rebind_prepared_continuation(
        &mut self,
        prepared: &mut PreparedResidentObservation,
        graph: &crate::graph_capture::AutonomousGraph,
        saved: &PreparedResidentContinuationV1,
    ) -> anyhow::Result<serde_json::Value> {
        saved.validate()?;
        anyhow::ensure!(
            prepared.raw_stream() == self.stream.cu_stream()
                && prepared.binding.is_none()
                && prepared.appended
                && prepared.parity == saved.parity
                && graph.cu_graph_template().map(|p| p as usize)
                    == Some(prepared.parent_template_identity),
            "shared continuation binding is not the fresh owning parent"
        );
        let old = saved
            .binding
            .as_ref()
            .context("shared continuation had no prior live binding")?;
        let live = self
            .engine
            .as_ref()
            .context("prepared restart lacks native engine")?
            .ladd_cadence_capture_key(false);
        anyhow::ensure!(
            live == prepared.keys[prepared.parity],
            "restored shared LADD permutation does not match native engine"
        );
        let new = self.issue_resident_graph_replay_binding_parts(graph, prepared.profile, true)?;
        // Graph structural witnesses may include fresh pointer identities. Both
        // witnesses are retained; scalar/configuration and native keys were
        // checked above rather than equating process-specific identities.
        let receipt = serde_json::json!({"schema_kind":"prism.shared_resident_binding_reconstruction.v1",
            "source_stream_id":saved.source_stream_id,"physical_steps_advanced":0,
            "old":{"domain":old.0,"capture_nonce":old.1,"structure":old.2,"template":old.3,"exec":old.4,
                "step_pointer":saved.step_binding,"ladd_keys":saved.keys},
            "fresh":{"domain":new.domain,"capture_nonce":new.capture_nonce,
                "structure":new.resident_structure_fingerprint_sha256,"template":new.graph_template_handle,
                "exec":new.graph_exec_handle,"step_pointer":prepared.step_ptr,"ladd_keys":prepared.keys},
            "preserved_parity":prepared.parity});
        prepared.binding = Some(new);
        self.record_prismcode_md_native_graph_capture(
            "restart_reconstruct_shared_resident_observation_parent",
        );
        Ok(receipt)
    }
}

// Append to persistent_engine.rs. This is the existing production monolithic
// graph recipe, factored for fresh native construction during continuation.
#[cfg(feature = "v2_ignition")]
impl PersistentNhsEngine {
    pub fn reconstruct_monolithic_observation_for_continuation(
        &mut self,
        fresh_config: &crate::captured_pipeline::PipelineConfig,
    ) -> anyhow::Result<(
        crate::graph_capture::AutonomousGraph,
        Arc<crate::captured_pipeline::CapturedAdjudicationPipeline>,
        crate::captured_pipeline::PipelineConfig,
    )> {
        use crate::captured_pipeline::CapturedAdjudicationPipeline;
        let before_step = self.device_protocol_current_step()?;
        let mut parent = self
            .capture_autonomous_template()
            .context("reconstruct exact native monolithic physical template")?;
        let raw = parent.cu_graph();
        unsafe { crate::graph_node::splice_legality_check(raw) }
            .map_err(|rc| anyhow::anyhow!("monolithic parent splice inspection failed: {rc}"))?;
        let g26 = unsafe { CapturedAdjudicationPipeline::create_parent_g26_cond_handle(raw) }
            .map_err(|e| anyhow::anyhow!("monolithic parent G26 handle: {e}"))?;
        let f1 = unsafe { CapturedAdjudicationPipeline::create_parent_f1_cond_handle(raw) }
            .map_err(|e| anyhow::anyhow!("monolithic parent F1 handle: {e}"))?;
        let mut config = fresh_config.clone();
        config.body_unroll = 1;
        config.g26_parent_cond_handle = g26;
        config.f1_parent_cond_handle = f1;
        let pipeline = Arc::new(
            CapturedAdjudicationPipeline::build(self.cuda_context(), self.cuda_stream(), &config)
                .map_err(|e| anyhow::anyhow!("native monolithic observer reconstruction: {e}"))?,
        );
        // The owning graph retains its native pipeline, including every V2,
        // governor and ASC allocation lease, through destruction/unwind.
        parent.retain_pipeline(pipeline.clone());
        let child = pipeline.cu_graph_raw();
        unsafe { crate::graph_node::splice_legality_check(child) }
            .map_err(|rc| anyhow::anyhow!("monolithic child splice inspection failed: {rc}"))?;
        let physical = parent
            .node("physical_step_complete")
            .context("native monolithic physical completion tag absent")?;
        let multi_lif = parent
            .node("multi_lif")
            .context("native monolithic multi_lif tag absent")?;
        let child_node = parent
            .add_child_graph_node(&[physical], child)
            .context("splice existing observer immediately after physical completion")?;
        let n_floats = config
            .asc
            .as_ref()
            .map(|a| (a.n_atoms.max(0) as u32).saturating_mul(3))
            .unwrap_or(0);
        let f1_node = unsafe { pipeline.wire_parent_f1_switch(raw, f1, child_node) }
            .map_err(|e| anyhow::anyhow!("wire native monolithic F1 before G26: {e}"))?;
        let g26_node =
            unsafe { pipeline.wire_parent_g26_switch(raw, f1_node, config.d_velocities, n_floats) }
                .map_err(|e| anyhow::anyhow!("wire native monolithic G26 after F1: {e}"))?;
        unsafe { crate::graph_node::splice_legality_check(raw) }
            .map_err(|rc| anyhow::anyhow!("monolithic composed splice inspection failed: {rc}"))?;
        parent
            .add_dependency(g26_node, multi_lif)
            .context("wire native G26 completion to multi_lif")?;
        let graph = parent
            .instantiate()
            .context("instantiate native monolithic continuation graph")?;
        // The native constructor's seed projection/freeze is actual preparation.
        // The caller MUST subsequently restore the saved full pipeline planes,
        // graph ledger, borrowed controller state and output watermarks before
        // admitting replay; this fresh baseline can never stand in for history.
        pipeline
            .freeze_baseline()
            .map_err(|e| anyhow::anyhow!("freeze native monolithic constructor baseline: {e}"))?;
        self.cuda_stream().synchronize()?;
        anyhow::ensure!(
            self.device_protocol_current_step()? == before_step,
            "monolithic reconstruction advanced the native physical clock"
        );
        self.record_prismcode_md_native_graph_capture(
            "restart_reconstruct_monolithic_observation_parent",
        );
        Ok((graph, pipeline, config))
    }
}


#[cfg(feature="gpu")]
impl PersistentNhsEngine {
    pub fn scientific_restart_field_plan(&self,owner:&str,max_field_bytes:u64)->Result<Vec<crate::fused_engine::ScientificFieldPlan>> {
        anyhow::ensure!(self.lambda_runtime_binding.is_none(),"unconsumed lambda runtime binding");
        let plans=self.engine.as_ref().context("scientific restart requires loaded engine")?.scientific_restart_field_plan(&format!("{owner}/fused"),max_field_bytes)?;
        self.append_persistent_scientific_plans(owner,max_field_bytes,plans)
    }
    fn append_persistent_scientific_plans(&self,owner:&str,max_field_bytes:u64,mut plans:Vec<crate::fused_engine::ScientificFieldPlan>)->Result<Vec<crate::fused_engine::ScientificFieldPlan>> {
        let (n,heap)=crate::checkpoint_codec::allocation_plan(&self.max_atoms,max_field_bytes)?;
        plans.push(crate::fused_engine::ScientificFieldPlan{contract:crate::scientific_checkpoint::FieldContract{owner:owner.into(),name:"max_atoms".into(),encoding:"prism.typed-host-state.v1".into(),expected_bytes:None,may_be_absent:false},present_bytes:Some(n),temporary_bytes:heap});
        let (n,heap)=crate::checkpoint_codec::allocation_plan(&self.grid_dim,max_field_bytes)?;
        plans.push(crate::fused_engine::ScientificFieldPlan{contract:crate::scientific_checkpoint::FieldContract{owner:owner.into(),name:"grid_dim".into(),encoding:"prism.typed-host-state.v1".into(),expected_bytes:None,may_be_absent:false},present_bytes:Some(n),temporary_bytes:heap});
        let (n,heap)=crate::checkpoint_codec::allocation_plan(&self.grid_spacing,max_field_bytes)?;
        plans.push(crate::fused_engine::ScientificFieldPlan{contract:crate::scientific_checkpoint::FieldContract{owner:owner.into(),name:"grid_spacing".into(),encoding:"prism.typed-host-state.v1".into(),expected_bytes:None,may_be_absent:false},present_bytes:Some(n),temporary_bytes:heap});
        let (n,heap)=crate::checkpoint_codec::allocation_plan(&self.nonbonded_execution_policy,max_field_bytes)?;
        plans.push(crate::fused_engine::ScientificFieldPlan{contract:crate::scientific_checkpoint::FieldContract{owner:owner.into(),name:"nonbonded_execution_policy".into(),encoding:"prism.typed-host-state.v1".into(),expected_bytes:None,may_be_absent:false},present_bytes:Some(n),temporary_bytes:heap});
        let (n,heap)=crate::checkpoint_codec::allocation_plan(&self.current_topology_id,max_field_bytes)?;
        plans.push(crate::fused_engine::ScientificFieldPlan{contract:crate::scientific_checkpoint::FieldContract{owner:owner.into(),name:"current_topology_id".into(),encoding:"prism.typed-host-state.v1".into(),expected_bytes:None,may_be_absent:false},present_bytes:Some(n),temporary_bytes:heap});
        let (n,heap)=crate::checkpoint_codec::allocation_plan(&self.prismcode_md_policy_admission,max_field_bytes)?;
        plans.push(crate::fused_engine::ScientificFieldPlan{contract:crate::scientific_checkpoint::FieldContract{owner:owner.into(),name:"prismcode_md_policy_admission".into(),encoding:"prism.typed-host-state.v1".into(),expected_bytes:None,may_be_absent:false},present_bytes:Some(n),temporary_bytes:heap});
        let (n,heap)=crate::checkpoint_codec::allocation_plan(&self.context_init_time_ms,max_field_bytes)?;
        plans.push(crate::fused_engine::ScientificFieldPlan{contract:crate::scientific_checkpoint::FieldContract{owner:owner.into(),name:"context_init_time_ms".into(),encoding:"prism.typed-host-state.v1".into(),expected_bytes:None,may_be_absent:false},present_bytes:Some(n),temporary_bytes:heap});
        let (n,heap)=crate::checkpoint_codec::allocation_plan(&self.module_init_time_ms,max_field_bytes)?;
        plans.push(crate::fused_engine::ScientificFieldPlan{contract:crate::scientific_checkpoint::FieldContract{owner:owner.into(),name:"module_init_time_ms".into(),encoding:"prism.typed-host-state.v1".into(),expected_bytes:None,may_be_absent:false},present_bytes:Some(n),temporary_bytes:heap});
        let (n,heap)=crate::checkpoint_codec::allocation_plan(&self.rt_init_time_ms,max_field_bytes)?;
        plans.push(crate::fused_engine::ScientificFieldPlan{contract:crate::scientific_checkpoint::FieldContract{owner:owner.into(),name:"rt_init_time_ms".into(),encoding:"prism.typed-host-state.v1".into(),expected_bytes:None,may_be_absent:false},present_bytes:Some(n),temporary_bytes:heap});
        let (n,heap)=crate::checkpoint_codec::allocation_plan(&self.structures_processed,max_field_bytes)?;
        plans.push(crate::fused_engine::ScientificFieldPlan{contract:crate::scientific_checkpoint::FieldContract{owner:owner.into(),name:"structures_processed".into(),encoding:"prism.typed-host-state.v1".into(),expected_bytes:None,may_be_absent:false},present_bytes:Some(n),temporary_bytes:heap});
        let (n,heap)=crate::checkpoint_codec::allocation_plan(&self.total_steps_run,max_field_bytes)?;
        plans.push(crate::fused_engine::ScientificFieldPlan{contract:crate::scientific_checkpoint::FieldContract{owner:owner.into(),name:"total_steps_run".into(),encoding:"prism.typed-host-state.v1".into(),expected_bytes:None,may_be_absent:false},present_bytes:Some(n),temporary_bytes:heap});
        let (n,heap)=crate::checkpoint_codec::allocation_plan(&self.total_compute_time_ms,max_field_bytes)?;
        plans.push(crate::fused_engine::ScientificFieldPlan{contract:crate::scientific_checkpoint::FieldContract{owner:owner.into(),name:"total_compute_time_ms".into(),encoding:"prism.typed-host-state.v1".into(),expected_bytes:None,may_be_absent:false},present_bytes:Some(n),temporary_bytes:heap});
        let (n,heap)=crate::checkpoint_codec::allocation_plan(&self.resident_graph_capture_nonce,max_field_bytes)?;
        plans.push(crate::fused_engine::ScientificFieldPlan{contract:crate::scientific_checkpoint::FieldContract{owner:owner.into(),name:"resident_graph_capture_nonce".into(),encoding:"prism.typed-host-state.v1".into(),expected_bytes:None,may_be_absent:false},present_bytes:Some(n),temporary_bytes:heap});
        Ok(plans)
    }
    pub fn visit_process_checkpoint_fields(&self,owner:&str,max_field_bytes:u64,
        mut emit:impl FnMut(crate::scientific_checkpoint::CapturedField)->Result<()>)->Result<()> {
        self.scientific_restart_field_plan(owner,max_field_bytes)?;
        self.engine.as_ref().context("scientific restart requires loaded engine")?.visit_process_checkpoint_fields(&format!("{owner}/fused"),max_field_bytes,&mut emit)?;
        emit(crate::scientific_checkpoint::capture_host_bounded(owner,"max_atoms",&self.max_atoms,max_field_bytes)?)?;
        emit(crate::scientific_checkpoint::capture_host_bounded(owner,"grid_dim",&self.grid_dim,max_field_bytes)?)?;
        emit(crate::scientific_checkpoint::capture_host_bounded(owner,"grid_spacing",&self.grid_spacing,max_field_bytes)?)?;
        emit(crate::scientific_checkpoint::capture_host_bounded(owner,"nonbonded_execution_policy",&self.nonbonded_execution_policy,max_field_bytes)?)?;
        emit(crate::scientific_checkpoint::capture_host_bounded(owner,"current_topology_id",&self.current_topology_id,max_field_bytes)?)?;
        emit(crate::scientific_checkpoint::capture_host_bounded(owner,"prismcode_md_policy_admission",&self.prismcode_md_policy_admission,max_field_bytes)?)?;
        emit(crate::scientific_checkpoint::capture_host_bounded(owner,"context_init_time_ms",&self.context_init_time_ms,max_field_bytes)?)?;
        emit(crate::scientific_checkpoint::capture_host_bounded(owner,"module_init_time_ms",&self.module_init_time_ms,max_field_bytes)?)?;
        emit(crate::scientific_checkpoint::capture_host_bounded(owner,"rt_init_time_ms",&self.rt_init_time_ms,max_field_bytes)?)?;
        emit(crate::scientific_checkpoint::capture_host_bounded(owner,"structures_processed",&self.structures_processed,max_field_bytes)?)?;
        emit(crate::scientific_checkpoint::capture_host_bounded(owner,"total_steps_run",&self.total_steps_run,max_field_bytes)?)?;
        emit(crate::scientific_checkpoint::capture_host_bounded(owner,"total_compute_time_ms",&self.total_compute_time_ms,max_field_bytes)?)?;
        emit(crate::scientific_checkpoint::capture_host_bounded(owner,"resident_graph_capture_nonce",&self.resident_graph_capture_nonce,max_field_bytes)?)?;
        Ok(())
    }
    pub fn restore_streamed_process_checkpoint_before_capture(&mut self,owner:&str,
        fields:&[crate::scientific_checkpoint::FieldContract],max_field_bytes:u64,
        reader:&crate::scientific_checkpoint::restart_transaction::ScientificGenerationReader<'_>)->Result<()> {
        self.autonomous_chunk_clock.invalidate();
        use crate::scientific_checkpoint::{capture_host,restore_host};
        anyhow::ensure!(self.resident_graph_replay_bindings.is_empty(),"persistent restore must precede fresh graph registration");
        anyhow::ensure!(self.lambda_runtime_binding.is_none(),"unconsumed lambda constructor binding on restore");
        let field=|name:&str|reader.field(owner,name);
        anyhow::ensure!(fields.iter().filter(|v|v.owner==owner).count()==13,"persistent host field population changed");
        anyhow::ensure!(field("max_atoms")?.bytes==capture_host(owner,"max_atoms",&self.max_atoms)?.bytes,"persistent authority changed: max_atoms");
        anyhow::ensure!(field("grid_dim")?.bytes==capture_host(owner,"grid_dim",&self.grid_dim)?.bytes,"persistent authority changed: grid_dim");
        anyhow::ensure!(field("grid_spacing")?.bytes==capture_host(owner,"grid_spacing",&self.grid_spacing)?.bytes,"persistent authority changed: grid_spacing");
        anyhow::ensure!(field("nonbonded_execution_policy")?.bytes==capture_host(owner,"nonbonded_execution_policy",&self.nonbonded_execution_policy)?.bytes,"persistent authority changed: nonbonded_execution_policy");
        anyhow::ensure!(field("current_topology_id")?.bytes==capture_host(owner,"current_topology_id",&self.current_topology_id)?.bytes,"persistent authority changed: current_topology_id");
        anyhow::ensure!(field("prismcode_md_policy_admission")?.bytes==capture_host(owner,"prismcode_md_policy_admission",&self.prismcode_md_policy_admission)?.bytes,"persistent authority changed: prismcode_md_policy_admission");
        let context_init_time_ms=restore_host(&field("context_init_time_ms")?)?;
        let module_init_time_ms=restore_host(&field("module_init_time_ms")?)?;
        let rt_init_time_ms=restore_host(&field("rt_init_time_ms")?)?;
        let structures_processed=restore_host(&field("structures_processed")?)?;
        let total_steps_run=restore_host(&field("total_steps_run")?)?;
        let total_compute_time_ms=restore_host(&field("total_compute_time_ms")?)?;
        let resident_graph_capture_nonce=restore_host(&field("resident_graph_capture_nonce")?)?;
        let fused_owner=format!("{owner}/fused");
        self.engine.as_mut().context("persistent restore requires loaded canonical engine")?
            .restore_streamed_process_checkpoint_fields(&fused_owner,fields,max_field_bytes,reader)?;
        self.context_init_time_ms=context_init_time_ms;
        self.module_init_time_ms=module_init_time_ms;
        self.rt_init_time_ms=rt_init_time_ms;
        self.structures_processed=structures_processed;
        self.total_steps_run=total_steps_run;
        self.total_compute_time_ms=total_compute_time_ms;
        self.resident_graph_capture_nonce=resident_graph_capture_nonce;
        Ok(())
    }
}


#[cfg(feature="gpu")]
pub struct PreparedPersistentScientificFieldPlan<'a> {
    engine: &'a PersistentNhsEngine,
    owner: String,
    maximum_logical_bytes: u64,
    fused: crate::fused_engine::PreparedScientificFieldPlan<'a>,
    fields: Vec<crate::fused_engine::ScientificFieldPlan>,
}
#[cfg(feature="gpu")]
impl PreparedPersistentScientificFieldPlan<'_> {
    pub fn fields(&self)->&[crate::fused_engine::ScientificFieldPlan] { &self.fields }
}
#[cfg(feature="gpu")]
impl PersistentNhsEngine {
    pub fn stream_process_checkpoint_fields(&self,owner:&str,max_device_bytes:u64,max_logical_field_bytes:u64,
        emit:&mut dyn FnMut(&crate::fused_engine::ScientificFieldPlan,&mut dyn FnMut(&mut dyn std::io::Write)->Result<()>)->Result<()>)->Result<()> {
        let prepared=self.prepare_process_checkpoint_plan(owner,max_logical_field_bytes)?;
        self.stream_prepared_process_checkpoint_fields(&prepared,max_device_bytes,emit)
    }
    pub fn prepare_process_checkpoint_plan(&self,owner:&str,max_logical_field_bytes:u64)->Result<PreparedPersistentScientificFieldPlan<'_>> {
        anyhow::ensure!(self.lambda_runtime_binding.is_none(),"unconsumed lambda runtime binding");
        let fused=self.engine.as_ref().context("scientific streaming requires loaded engine")?
            .prepare_process_checkpoint_plan(&format!("{owner}/fused"),max_logical_field_bytes)?;
        let fields=self.append_persistent_scientific_plans(owner,max_logical_field_bytes,fused.fields().to_vec())?;
        anyhow::ensure!(fields.len()==351 && fields.iter().filter(|p|p.contract.owner==owner).count()==13,"prepared persistent field population differs");
        let mut keys=std::collections::BTreeSet::new();
        for plan in &fields {
            plan.contract.validate()?;
            anyhow::ensure!(keys.insert(plan.contract.key()) && plan.present_bytes.unwrap_or(0)<=max_logical_field_bytes,"prepared persistent field duplicate/extent differs");
        }
        Ok(PreparedPersistentScientificFieldPlan{engine:self,owner:owner.into(),maximum_logical_bytes:max_logical_field_bytes,fused,fields})
    }
    pub fn stream_prepared_process_checkpoint_fields(&self,prepared:&PreparedPersistentScientificFieldPlan<'_>,max_device_bytes:u64,
        emit:&mut dyn FnMut(&crate::fused_engine::ScientificFieldPlan,&mut dyn FnMut(&mut dyn std::io::Write)->Result<()>)->Result<()>)->Result<()> {
        anyhow::ensure!(std::ptr::eq(self,prepared.engine),"prepared scientific fields belong to a different persistent engine");
        let owner=prepared.owner.as_str();let max_logical_field_bytes=prepared.maximum_logical_bytes;let plans=prepared.fields();
        self.engine.as_ref().context("scientific streaming requires loaded engine")?.stream_prepared_process_checkpoint_fields(&prepared.fused,max_device_bytes,emit)?;
        {let plan=plans.iter().find(|p|p.contract.owner==owner && p.contract.name=="max_atoms").context("persistent stream field plan missing")?;let mut producer=|out:&mut dyn std::io::Write|->Result<()>{crate::checkpoint_codec::write_to(&self.max_atoms,out,max_logical_field_bytes)?;Ok(())};emit(plan,&mut producer)?;}
        {let plan=plans.iter().find(|p|p.contract.owner==owner && p.contract.name=="grid_dim").context("persistent stream field plan missing")?;let mut producer=|out:&mut dyn std::io::Write|->Result<()>{crate::checkpoint_codec::write_to(&self.grid_dim,out,max_logical_field_bytes)?;Ok(())};emit(plan,&mut producer)?;}
        {let plan=plans.iter().find(|p|p.contract.owner==owner && p.contract.name=="grid_spacing").context("persistent stream field plan missing")?;let mut producer=|out:&mut dyn std::io::Write|->Result<()>{crate::checkpoint_codec::write_to(&self.grid_spacing,out,max_logical_field_bytes)?;Ok(())};emit(plan,&mut producer)?;}
        {let plan=plans.iter().find(|p|p.contract.owner==owner && p.contract.name=="nonbonded_execution_policy").context("persistent stream field plan missing")?;let mut producer=|out:&mut dyn std::io::Write|->Result<()>{crate::checkpoint_codec::write_to(&self.nonbonded_execution_policy,out,max_logical_field_bytes)?;Ok(())};emit(plan,&mut producer)?;}
        {let plan=plans.iter().find(|p|p.contract.owner==owner && p.contract.name=="current_topology_id").context("persistent stream field plan missing")?;let mut producer=|out:&mut dyn std::io::Write|->Result<()>{crate::checkpoint_codec::write_to(&self.current_topology_id,out,max_logical_field_bytes)?;Ok(())};emit(plan,&mut producer)?;}
        {let plan=plans.iter().find(|p|p.contract.owner==owner && p.contract.name=="prismcode_md_policy_admission").context("persistent stream field plan missing")?;let mut producer=|out:&mut dyn std::io::Write|->Result<()>{crate::checkpoint_codec::write_to(&self.prismcode_md_policy_admission,out,max_logical_field_bytes)?;Ok(())};emit(plan,&mut producer)?;}
        {let plan=plans.iter().find(|p|p.contract.owner==owner && p.contract.name=="context_init_time_ms").context("persistent stream field plan missing")?;let mut producer=|out:&mut dyn std::io::Write|->Result<()>{crate::checkpoint_codec::write_to(&self.context_init_time_ms,out,max_logical_field_bytes)?;Ok(())};emit(plan,&mut producer)?;}
        {let plan=plans.iter().find(|p|p.contract.owner==owner && p.contract.name=="module_init_time_ms").context("persistent stream field plan missing")?;let mut producer=|out:&mut dyn std::io::Write|->Result<()>{crate::checkpoint_codec::write_to(&self.module_init_time_ms,out,max_logical_field_bytes)?;Ok(())};emit(plan,&mut producer)?;}
        {let plan=plans.iter().find(|p|p.contract.owner==owner && p.contract.name=="rt_init_time_ms").context("persistent stream field plan missing")?;let mut producer=|out:&mut dyn std::io::Write|->Result<()>{crate::checkpoint_codec::write_to(&self.rt_init_time_ms,out,max_logical_field_bytes)?;Ok(())};emit(plan,&mut producer)?;}
        {let plan=plans.iter().find(|p|p.contract.owner==owner && p.contract.name=="structures_processed").context("persistent stream field plan missing")?;let mut producer=|out:&mut dyn std::io::Write|->Result<()>{crate::checkpoint_codec::write_to(&self.structures_processed,out,max_logical_field_bytes)?;Ok(())};emit(plan,&mut producer)?;}
        {let plan=plans.iter().find(|p|p.contract.owner==owner && p.contract.name=="total_steps_run").context("persistent stream field plan missing")?;let mut producer=|out:&mut dyn std::io::Write|->Result<()>{crate::checkpoint_codec::write_to(&self.total_steps_run,out,max_logical_field_bytes)?;Ok(())};emit(plan,&mut producer)?;}
        {let plan=plans.iter().find(|p|p.contract.owner==owner && p.contract.name=="total_compute_time_ms").context("persistent stream field plan missing")?;let mut producer=|out:&mut dyn std::io::Write|->Result<()>{crate::checkpoint_codec::write_to(&self.total_compute_time_ms,out,max_logical_field_bytes)?;Ok(())};emit(plan,&mut producer)?;}
        {let plan=plans.iter().find(|p|p.contract.owner==owner && p.contract.name=="resident_graph_capture_nonce").context("persistent stream field plan missing")?;let mut producer=|out:&mut dyn std::io::Write|->Result<()>{crate::checkpoint_codec::write_to(&self.resident_graph_capture_nonce,out,max_logical_field_bytes)?;Ok(())};emit(plan,&mut producer)?;}
        Ok(())
    }
}
