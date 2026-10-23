use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

pub const MD_NATIVE_SCHEMA_KIND: &str = "prismcode_native_md_runtime_receipt";
pub const MD_NATIVE_SCHEMA_VERSION: u32 = 2;

/// Exact native bound-WHILE dispatch labels. A name merely containing
/// "resident" or "while" never admits an unrecognized/legacy execution path.
pub fn is_resident_md_graph_launch_path(path: &str) -> bool {
    matches!(path, "graph.while_resident"
        | "graph.while_ladd_kcc_v2_parent"
        | "graph.shared_coupling_while_ladd_kcc_v2_parent")
}


pub const OP_MD_PROTOCOL_DIRECTOR: &str = "MD.PROTOCOL_DIRECTOR";
pub const OP_MD_COM_REMOVE: &str = "MD.COM_REMOVE";
pub const OP_MD_NEIGHBOR_REBUILD: &str = "MD.NEIGHBOR_REBUILD";
pub const OP_MD_ACCUMULATOR_ZERO: &str = "MD.ACCUMULATOR_ZERO";
pub const OP_MD_FORCE_BONDED: &str = "MD.FORCE_BONDED";
pub const OP_MD_FORCE_NONBONDED: &str = "MD.FORCE_NONBONDED";
pub const OP_MD_INTEGRATE_VELOCITY_VERLET: &str = "MD.INTEGRATE_VELOCITY_VERLET";
pub const OP_MD_SHAKE_CONSTRAIN: &str = "MD.SHAKE_CONSTRAIN";
pub const OP_MD_EXCLUSION_FIELD: &str = "MD.EXCLUSION_FIELD";
pub const OP_MD_WATER_DENSITY: &str = "MD.WATER_DENSITY";
pub const OP_MD_PHOTO_UV_PUMP: &str = "MD.PHOTO_UV_PUMP";
/// P2 — emitted in place of `OP_MD_PHOTO_UV_PUMP` when the run is a declared
/// UV-off control. The pump kernel was NOT launched; the phase still reports, so
/// a gap in the op stream is never ambiguous between "suppressed" and "did not
/// run". Registered in the binary ISA as `MD_PHOTO_UV_PUMP_SUPPRESSED` and in
/// the prismos / tokenizer MD instruction-set vocabularies.
pub const OP_MD_PHOTO_UV_PUMP_SUPPRESSED: &str = "MD.PHOTO_UV_PUMP_SUPPRESSED";
pub const OP_MD_EXTERNAL_WORK_REDUCE: &str = "MD.EXTERNAL_WORK_REDUCE";
pub const OP_MD_VELOCITY_SECOND_HALF: &str = "MD.VELOCITY_SECOND_HALF";
pub const OP_MD_VELOCITY_CLAMP: &str = "MD.VELOCITY_CLAMP";
pub const OP_MD_CA_RESTRAINT: &str = "MD.CA_RESTRAINT";
pub const OP_MD_LIF_OBSERVE: &str = "MD.LIF_OBSERVE";
pub const OP_MD_COUPLING_BUILD: &str = "MD.COUPLING_BUILD";
pub const OP_MD_COUPLING_CLEAR: &str = "MD.COUPLING_CLEAR";
pub const OP_MD_HEARTBEAT_ASSERT: &str = "MD.HEARTBEAT_ASSERT";
pub const OP_MD_LADD_UPDATE: &str = "MD.LADD_UPDATE";
pub const OP_MD_KCC_UPDATE: &str = "MD.KCC_UPDATE";
pub const OP_MD_NMA_PERTURB: &str = "MD.NMA_PERTURB";
pub const OP_MD_SPIKE_EMIT: &str = "MD.SPIKE_EMIT";
pub const OP_MD_SPIKE_READOUT: &str = "MD.SPIKE_READOUT";
pub const OP_MD_SIGNAL_GRID_READOUT: &str = "MD.SIGNAL_GRID_READOUT";
pub const OP_MD_KCC_DESCRIPTOR: &str = "MD.KCC_DESCRIPTOR";
// Always-On QNANL Spike Bus (Phase 3): chunk-commit / sensor / QNANL seal+verify / manifest ops.
// These extend the SAME receipt context as the spike/graph/runtime ops above.
pub const OP_MD_SPIKE_CHUNK_COMMIT: &str = "MD.SPIKE_CHUNK_COMMIT";
pub const OP_MD_SENSOR_EMIT: &str = "MD.SENSOR_EMIT";
pub const OP_MD_SENSOR_CHUNK_COMMIT: &str = "MD.SENSOR_CHUNK_COMMIT";
pub const OP_MD_QNANL_SEAL: &str = "MD.QNANL_SEAL";
pub const OP_MD_QNANL_VERIFY: &str = "MD.QNANL_VERIFY";
pub const OP_MD_EVENT_STREAM_MANIFEST_COMMIT: &str = "MD.EVENT_STREAM_MANIFEST_COMMIT";
// Directive v2 A3: TWIN16 spike-stream + condition-edge receipt ops.
pub const OP_MD_SPIKE_MANIFEST_COMMIT: &str = "MD.SPIKE_MANIFEST_COMMIT";
pub const OP_MD_TWIN16_CELL_COMMIT: &str = "MD.TWIN16_CELL_COMMIT";
pub const OP_MD_CONDITION_EDGE_COMMIT: &str = "MD.CONDITION_EDGE_COMMIT";

const SAMPLE_LIMIT: usize = 256;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MdNativeRuntimeConfig {
    pub stream_id: u32,
    pub require_complete: bool,
    pub materialized_readouts_required: bool,
    pub n_atoms: u64,
    pub n_residues: u64,
    pub grid_dim: u64,
    pub fused_inner_steps: u32,
    pub use_neighbor_list: bool,
    pub ladd_enabled: bool,
    pub heartbeat_enabled: bool,
    pub ca_restraint_enabled: bool,
    pub nma_enabled: bool,
    /// P2 — the run is a declared UV-off control: the pump kernel is never
    /// launched, so the required-opcode contract must require
    /// `MD.PHOTO_UV_PUMP_SUPPRESSED` instead of `MD.PHOTO_UV_PUMP`.
    pub uv_burst_suppressed: bool,
    pub nonbonded_execution_contract: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MdNativeKernelDispatch {
    pub step: u64,
    pub inner_step: u32,
    pub phase: String,
    pub kernel: String,
    pub opcodes: Vec<String>,
    pub grid_dim: [u32; 3],
    pub block_dim: [u32; 3],
    pub shared_mem_bytes: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MdNativeRuntimeReceipt {
    pub schema_kind: String,
    pub schema_version: u32,
    pub status: String,
    pub execution_mode: String,
    pub stream_id: u32,
    pub require_complete: bool,
    pub materialized_readouts_required: bool,
    pub materialized_readout_policy: String,
    pub n_atoms: u64,
    pub n_residues: u64,
    pub grid_dim: u64,
    pub fused_inner_steps: u32,
    pub use_neighbor_list: bool,
    pub ladd_enabled: bool,
    pub heartbeat_enabled: bool,
    pub ca_restraint_enabled: bool,
    pub nma_enabled: bool,
    /// P2 — declared UV-off control. Consumed by the prismos MD ISA governor so
    /// a control run's required-core contract swaps MD.PHOTO_UV_PUMP for
    /// MD.PHOTO_UV_PUMP_SUPPRESSED instead of reporting a missing opcode.
    pub uv_burst_suppressed: bool,
    pub nonbonded_execution_contract: serde_json::Value,
    pub implementation_status_language: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exact_nonbonded_exception_upload_contract: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub harmonic_parameter_upload_contract: Option<serde_json::Value>,
    pub nma_perturb_required_by_observed_window: bool,
    pub dispatch_count: u64,
    pub logical_step_count: u64,
    pub inner_step_count: u64,
    pub opcode_counts: BTreeMap<String, u64>,
    /// Actual in-loop seal/verify opcode counts, derived from dispatches.
    pub qnanl_inline_seal_count: u64,
    pub qnanl_inline_verify_count: u64,
    /// PASS only when at least one paired in-loop seal/verify dispatch was
    /// recorded. Absence remains explicit EVIDENCE_NOT_FOUND.
    pub qnanl_inline_pair_status: String,
    pub kernel_counts: BTreeMap<String, u64>,
    pub phase_counts: BTreeMap<String, u64>,
    pub standard_step_dispatch_count: u64,
    pub cuda_graph_capture_dispatch_count: u64,
    pub cuda_graph_replay_dispatch_count: u64,
    pub resident_graph_replay_dispatch_count: u64,
    pub legacy_graph_replay_dispatch_count: u64,
    pub graph_logical_step_count: u64,
    pub graph_steps_per_launch_max: u64,
    /// Number of graph-window observations presented to the receipt builder.
    #[serde(default)]
    pub graph_window_count: u64,
    /// PASS only when every observed window has nonzero inputs, exact launch
    /// arithmetic, non-overflowing bounds, and contiguous start/end ordering.
    #[serde(default)]
    pub graph_window_consistency_status: String,
    /// Exact count of discontinuities where a new graph window started after
    /// the prior counted window ended.
    #[serde(default)]
    pub graph_window_gap_count: u64,
    /// Exact count of discontinuities where a new graph window started before
    /// the prior counted window ended.
    #[serde(default)]
    pub graph_window_overlap_count: u64,
    /// Raw whole-run average, including an exact terminal tail launch.
    #[serde(default)]
    pub graph_host_launch_collapse_ratio_raw: f64,
    /// Logical steps represented by launches of the maximum captured body.
    #[serde(default)]
    pub graph_sustained_body_logical_step_count: u64,
    /// Host launches of the maximum captured body.
    #[serde(default)]
    pub graph_sustained_body_host_launch_count: u64,
    /// Steps per launch of the modal repeated resident body shape.
    #[serde(default)]
    pub graph_sustained_body_steps_per_launch: u64,
    /// Logical steps represented by the single smaller exact-tail capture.
    #[serde(default)]
    pub graph_exact_tail_logical_step_count: u64,
    /// Smaller exact-tail host launches; valid production closure permits at most one.
    #[serde(default)]
    pub graph_exact_tail_host_launch_count: u64,
    /// NONE, PRESENT, or INVALID_NONTERMINAL_OR_MULTIPLE.
    #[serde(default)]
    pub graph_exact_tail_status: String,
    /// Sustained full-body collapse used by the hotpath contract. The raw
    /// whole-run average remains available above and is never discarded.
    pub graph_host_launch_collapse_ratio: f64,
    pub hotpath_acceleration_status: String,
    pub hotpath_acceleration_contract: String,
    pub required_core_opcodes: Vec<String>,
    pub required_conditional_opcodes: Vec<String>,
    pub deferred_materialized_readout_opcodes: Vec<String>,
    pub missing_required_opcodes: Vec<String>,
    pub coverage_status: String,
    pub materialized_readout_status: String,
    pub coverage_hash_blake3: String,
    pub dispatch_trace_hash_blake3: String,
    pub sample_dispatches: Vec<MdNativeKernelDispatch>,
    pub macro_fused_cuda_kernel: bool,
    pub force_kernel_opcode_coverage: String,
    pub integration_opcode_coverage: String,
    pub observation_opcode_coverage: String,
    pub readout_opcode_coverage: String,
}

#[derive(Clone, Debug)]
pub struct MdNativeRuntimeState {
    config: MdNativeRuntimeConfig,
    opcode_counts: BTreeMap<String, u64>,
    kernel_counts: BTreeMap<String, u64>,
    phase_counts: BTreeMap<String, u64>,
    sample_dispatches: Vec<MdNativeKernelDispatch>,
    dispatch_count: u64,
    logical_steps: BTreeSet<u64>,
    inner_steps: u64,
    graph_logical_steps: u64,
    graph_host_launches: u64,
    graph_steps_per_launch_max: u64,
    graph_shape_logical_steps: BTreeMap<u64, u64>,
    graph_shape_host_launches: BTreeMap<u64, u64>,
    graph_window_count: u64,
    graph_window_consistent: bool,
    graph_window_gap_count: u64,
    graph_window_overlap_count: u64,
    last_graph_window_end_step: Option<u64>,
    last_graph_window_logical_steps: u64,
    last_graph_window_host_launches: u64,
    last_graph_window_steps_per_launch: u64,
    resident_graph_launches: u64,
    legacy_graph_launches: u64,
    trace_hasher: blake3::Hasher,
    // Exact hash inputs are retained only when process continuation is enabled
    // before the first dispatch. Hasher private memory is never serialized.
    trace_initial_config_bytes: Vec<u8>,
    trace_restart_inputs: Option<Vec<Vec<u8>>>,
    exact_nonbonded_exception_upload_contract: Option<serde_json::Value>,
    harmonic_parameter_upload_contract: Option<serde_json::Value>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MdNativeRuntimeCheckpointV1 {
    schema_version: u32,
    trace_digest: String,
    config: MdNativeRuntimeConfig,
    opcode_counts: BTreeMap<String, u64>,
    kernel_counts: BTreeMap<String, u64>,
    phase_counts: BTreeMap<String, u64>,
    sample_dispatches: Vec<MdNativeKernelDispatch>,
    dispatch_count: u64,
    logical_steps: BTreeSet<u64>,
    inner_steps: u64,
    graph_logical_steps: u64,
    graph_host_launches: u64,
    graph_steps_per_launch_max: u64,
    graph_shape_logical_steps: BTreeMap<u64, u64>,
    graph_shape_host_launches: BTreeMap<u64, u64>,
    graph_window_count: u64,
    graph_window_consistent: bool,
    graph_window_gap_count: u64,
    graph_window_overlap_count: u64,
    last_graph_window_end_step: Option<u64>,
    last_graph_window_logical_steps: u64,
    last_graph_window_host_launches: u64,
    last_graph_window_steps_per_launch: u64,
    resident_graph_launches: u64,
    legacy_graph_launches: u64,
    trace_initial_config_bytes: Vec<u8>,
    trace_restart_inputs: Option<Vec<Vec<u8>>>,
    exact_nonbonded_exception_upload_contract: Option<serde_json::Value>,
    #[serde(default)]
    harmonic_parameter_upload_contract: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct MdNativeRuntimeCheckpointView<'a> {
    schema_version:u32,
    trace_digest:&'a str,
    config:&'a MdNativeRuntimeConfig,
    opcode_counts:&'a BTreeMap<String, u64>,
    kernel_counts:&'a BTreeMap<String, u64>,
    phase_counts:&'a BTreeMap<String, u64>,
    sample_dispatches:&'a Vec<MdNativeKernelDispatch>,
    dispatch_count:&'a u64,
    logical_steps:&'a BTreeSet<u64>,
    inner_steps:&'a u64,
    graph_logical_steps:&'a u64,
    graph_host_launches:&'a u64,
    graph_steps_per_launch_max:&'a u64,
    graph_shape_logical_steps:&'a BTreeMap<u64, u64>,
    graph_shape_host_launches:&'a BTreeMap<u64, u64>,
    graph_window_count:&'a u64,
    graph_window_consistent:&'a bool,
    graph_window_gap_count:&'a u64,
    graph_window_overlap_count:&'a u64,
    last_graph_window_end_step:&'a Option<u64>,
    last_graph_window_logical_steps:&'a u64,
    last_graph_window_host_launches:&'a u64,
    last_graph_window_steps_per_launch:&'a u64,
    resident_graph_launches:&'a u64,
    legacy_graph_launches:&'a u64,
    trace_initial_config_bytes:&'a Vec<u8>,
    trace_restart_inputs:&'a Option<Vec<Vec<u8>>>,
    exact_nonbonded_exception_upload_contract:&'a Option<serde_json::Value>,
    harmonic_parameter_upload_contract:&'a Option<serde_json::Value>,
}

struct NativeCheckpointSizeWriter {bytes:u64,limit:u64}
impl std::io::Write for NativeCheckpointSizeWriter {
    fn write(&mut self,bytes:&[u8])->std::io::Result<usize> {
        self.bytes=self.bytes.checked_add(bytes.len() as u64).ok_or_else(||std::io::Error::new(std::io::ErrorKind::InvalidData,"native checkpoint size overflow"))?;
        if self.bytes>self.limit{return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,"native checkpoint exceeds admitted size"));}
        Ok(bytes.len())
    }
    fn flush(&mut self)->std::io::Result<()>{Ok(())}
}

impl MdNativeRuntimeState {
    /// Enable exact trace reconstruction before the first native dispatch.
    /// Later activation would lose history and is explicitly refused.
    pub fn enable_process_checkpoint_trace(&mut self) -> std::io::Result<()> {
        if self.dispatch_count != 0 { return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "checkpoint trace must be enabled before native dispatch")); }
        if self.trace_restart_inputs.is_none() { self.trace_restart_inputs = Some(Vec::new()); }
        Ok(())
    }
    /// Serialize all counters, coverage, chronology and exact hash inputs. This
    /// owner state alone is not a full-engine scientific checkpoint.
    /// Same JSON field order/values as V1, borrowing all history; no clone.
    pub fn write_process_checkpoint(&self,writer:&mut (impl std::io::Write+?Sized))->std::io::Result<()> {
        if self.trace_restart_inputs.is_none(){return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput,"native trace history was not retained from run start"));}
        let trace_digest=self.trace_hasher.clone().finalize().to_hex().to_string();
        let checkpoint=MdNativeRuntimeCheckpointView {
            schema_version:1,
            trace_digest:&trace_digest,
            config:&self.config,
            opcode_counts:&self.opcode_counts,
            kernel_counts:&self.kernel_counts,
            phase_counts:&self.phase_counts,
            sample_dispatches:&self.sample_dispatches,
            dispatch_count:&self.dispatch_count,
            logical_steps:&self.logical_steps,
            inner_steps:&self.inner_steps,
            graph_logical_steps:&self.graph_logical_steps,
            graph_host_launches:&self.graph_host_launches,
            graph_steps_per_launch_max:&self.graph_steps_per_launch_max,
            graph_shape_logical_steps:&self.graph_shape_logical_steps,
            graph_shape_host_launches:&self.graph_shape_host_launches,
            graph_window_count:&self.graph_window_count,
            graph_window_consistent:&self.graph_window_consistent,
            graph_window_gap_count:&self.graph_window_gap_count,
            graph_window_overlap_count:&self.graph_window_overlap_count,
            last_graph_window_end_step:&self.last_graph_window_end_step,
            last_graph_window_logical_steps:&self.last_graph_window_logical_steps,
            last_graph_window_host_launches:&self.last_graph_window_host_launches,
            last_graph_window_steps_per_launch:&self.last_graph_window_steps_per_launch,
            resident_graph_launches:&self.resident_graph_launches,
            legacy_graph_launches:&self.legacy_graph_launches,
            trace_initial_config_bytes:&self.trace_initial_config_bytes,
            trace_restart_inputs:&self.trace_restart_inputs,
            exact_nonbonded_exception_upload_contract:&self.exact_nonbonded_exception_upload_contract,
            harmonic_parameter_upload_contract:&self.harmonic_parameter_upload_contract,
        };
        serde_json::to_writer(writer,&checkpoint)?;Ok(())
    }
    pub fn process_checkpoint_size_bytes(&self,limit:u64)->std::io::Result<u64> {
        let mut writer=NativeCheckpointSizeWriter{bytes:0,limit};
        self.write_process_checkpoint(&mut writer)?;Ok(writer.bytes)
    }
    pub fn process_checkpoint_bytes_bounded(&self,limit:u64)->std::io::Result<Vec<u8>> {
        let size=self.process_checkpoint_size_bytes(limit)?;
        let capacity=usize::try_from(size).map_err(|e|std::io::Error::new(std::io::ErrorKind::InvalidData,e))?;
        let mut bytes=Vec::new();bytes.try_reserve_exact(capacity).map_err(|e|std::io::Error::new(std::io::ErrorKind::OutOfMemory,e))?;
        self.write_process_checkpoint(&mut bytes)?;
        if bytes.len()!=capacity{return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,"native trace changed between count and serialization"));}Ok(bytes)
    }
    pub fn process_checkpoint_bytes(&self)->std::io::Result<Vec<u8>> {
        self.process_checkpoint_bytes_bounded(512*1024*1024)
    }
    /// Rebuild BLAKE3 using its supported update API; no private hasher bytes or
    /// newly computed replacement digest are accepted as the old trace state.
    pub fn from_process_checkpoint_bytes(bytes: &[u8]) -> std::io::Result<Self> {
        if bytes.len() > 512 * 1024 * 1024 { return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "native trace checkpoint exceeds 512 MiB admission")); }
        let checkpoint: MdNativeRuntimeCheckpointV1 = serde_json::from_slice(bytes)?;
        if checkpoint.schema_version != 1 { return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "unsupported native trace checkpoint schema")); }
        let inputs = checkpoint.trace_restart_inputs.as_ref()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "native trace checkpoint lacks exact historical hash inputs"))?;
        if inputs.len() as u64 != checkpoint.dispatch_count { return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "native trace input population differs from dispatch count")); }
        let mut trace_hasher = blake3::Hasher::new();
        trace_hasher.update(b"PRISMCODE:NATIVE_MD_RUNTIME:V0_1");
        trace_hasher.update(&checkpoint.trace_initial_config_bytes);
        for bytes in inputs {
            // The exact byte representation, including historical float
            // formatting, is hashed. Parsing verifies that each input is an
            // actual dispatch without reserializing it into different bytes.
            let _: MdNativeKernelDispatch = serde_json::from_slice(bytes)?;
            trace_hasher.update(bytes);
        }
        if trace_hasher.clone().finalize().to_hex().as_str() != checkpoint.trace_digest { return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "native trace checkpoint hash-input reconstruction mismatch")); }
        Ok(Self {
            trace_hasher,
            config: checkpoint.config,
            opcode_counts: checkpoint.opcode_counts,
            kernel_counts: checkpoint.kernel_counts,
            phase_counts: checkpoint.phase_counts,
            sample_dispatches: checkpoint.sample_dispatches,
            dispatch_count: checkpoint.dispatch_count,
            logical_steps: checkpoint.logical_steps,
            inner_steps: checkpoint.inner_steps,
            graph_logical_steps: checkpoint.graph_logical_steps,
            graph_host_launches: checkpoint.graph_host_launches,
            graph_steps_per_launch_max: checkpoint.graph_steps_per_launch_max,
            graph_shape_logical_steps: checkpoint.graph_shape_logical_steps,
            graph_shape_host_launches: checkpoint.graph_shape_host_launches,
            graph_window_count: checkpoint.graph_window_count,
            graph_window_consistent: checkpoint.graph_window_consistent,
            graph_window_gap_count: checkpoint.graph_window_gap_count,
            graph_window_overlap_count: checkpoint.graph_window_overlap_count,
            last_graph_window_end_step: checkpoint.last_graph_window_end_step,
            last_graph_window_logical_steps: checkpoint.last_graph_window_logical_steps,
            last_graph_window_host_launches: checkpoint.last_graph_window_host_launches,
            last_graph_window_steps_per_launch: checkpoint.last_graph_window_steps_per_launch,
            resident_graph_launches: checkpoint.resident_graph_launches,
            legacy_graph_launches: checkpoint.legacy_graph_launches,
            trace_initial_config_bytes: checkpoint.trace_initial_config_bytes,
            trace_restart_inputs: checkpoint.trace_restart_inputs,
            exact_nonbonded_exception_upload_contract: checkpoint.exact_nonbonded_exception_upload_contract,
            harmonic_parameter_upload_contract: checkpoint.harmonic_parameter_upload_contract,
        })
    }

    pub fn new(config: MdNativeRuntimeConfig) -> Self {
        let mut trace_hasher = blake3::Hasher::new();
        trace_hasher.update(b"PRISMCODE:NATIVE_MD_RUNTIME:V0_1");
        let trace_initial_config_bytes = serde_json::to_vec(&config).unwrap_or_default();
        trace_hasher.update(&trace_initial_config_bytes);
        Self {
            config,
            opcode_counts: BTreeMap::new(),
            kernel_counts: BTreeMap::new(),
            phase_counts: BTreeMap::new(),
            sample_dispatches: Vec::new(),
            dispatch_count: 0,
            logical_steps: BTreeSet::new(),
            inner_steps: 0,
            graph_logical_steps: 0,
            graph_host_launches: 0,
            graph_steps_per_launch_max: 0,
            graph_shape_logical_steps: BTreeMap::new(),
            graph_shape_host_launches: BTreeMap::new(),
            graph_window_count: 0,
            graph_window_consistent: true,
            graph_window_gap_count: 0,
            graph_window_overlap_count: 0,
            last_graph_window_end_step: None,
            last_graph_window_logical_steps: 0,
            last_graph_window_host_launches: 0,
            last_graph_window_steps_per_launch: 0,
            resident_graph_launches: 0,
            legacy_graph_launches: 0,
            trace_hasher,
            trace_initial_config_bytes,
            trace_restart_inputs: None,
            exact_nonbonded_exception_upload_contract: None,
            harmonic_parameter_upload_contract: None,
        }
    }

    pub fn set_exact_nonbonded_exception_upload_contract(
        &mut self,
        contract: Option<serde_json::Value>,
    ) {
        self.exact_nonbonded_exception_upload_contract = contract;
    }

    pub fn set_harmonic_parameter_upload_contract(&mut self, contract: Option<serde_json::Value>) {
        self.harmonic_parameter_upload_contract = contract;
    }

    pub fn harmonic_parameter_upload_contract(&self) -> Option<&serde_json::Value> {
        self.harmonic_parameter_upload_contract.as_ref()
    }

    pub fn record_dispatch(&mut self, dispatch: MdNativeKernelDispatch) {
        self.dispatch_count += 1;
        self.logical_steps.insert(dispatch.step);
        if dispatch.inner_step > 0 {
            self.inner_steps += 1;
        }
        *self.phase_counts.entry(dispatch.phase.clone()).or_insert(0) += 1;
        *self
            .kernel_counts
            .entry(dispatch.kernel.clone())
            .or_insert(0) += 1;
        for opcode in &dispatch.opcodes {
            *self.opcode_counts.entry(opcode.clone()).or_insert(0) += 1;
        }
        let trace_bytes = serde_json::to_vec(&dispatch).unwrap_or_default();
        self.trace_hasher.update(&trace_bytes);
        if let Some(inputs) = &mut self.trace_restart_inputs { inputs.push(trace_bytes); }
        if self.sample_dispatches.len() < SAMPLE_LIMIT {
            self.sample_dispatches.push(dispatch);
        }
    }

    pub fn set_materialized_readouts_required(&mut self, required: bool) {
        self.config.materialized_readouts_required = required;
    }

    pub fn record(
        &mut self,
        step: u64,
        inner_step: u32,
        phase: &str,
        kernel: &str,
        opcodes: &[&str],
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem_bytes: u32,
        note: Option<&str>,
    ) {
        self.record_dispatch(MdNativeKernelDispatch {
            step,
            inner_step,
            phase: phase.to_string(),
            kernel: kernel.to_string(),
            opcodes: opcodes.iter().map(|opcode| opcode.to_string()).collect(),
            grid_dim: [grid_dim.0, grid_dim.1, grid_dim.2],
            block_dim: [block_dim.0, block_dim.1, block_dim.2],
            shared_mem_bytes,
            note: note.map(str::to_string),
        });
    }

    /// Record a QNANL seal and byte-exact verify operation at the MD step that
    /// produced the event bytes. The two SHA-256 values must be the same
    /// observed byte identity; a caller cannot record a post-run
    /// reinterpretation as an inline pair.
    pub fn record_qnanl_inline_seal_verify(
        &mut self,
        step: u64,
        inner_step: u32,
        event_byte_source_sha256: &str,
        qnanl_sealed_from_sha256: &str,
        byte_exact_decode_pass: bool,
    ) -> Result<(), String> {
        if !is_lower_sha256(event_byte_source_sha256)
            || !is_lower_sha256(qnanl_sealed_from_sha256)
            || event_byte_source_sha256 != qnanl_sealed_from_sha256
            || !byte_exact_decode_pass
        {
            return Err(
                "EVIDENCE_NOT_FOUND: inline QNANL event-byte/seal/decode binding failed"
                    .to_string(),
            );
        }
        let note = json!({
            "runtime_origin": "RUNTIME_INLINE",
            "event_byte_source_sha256": event_byte_source_sha256,
            "qnanl_sealed_from_sha256": qnanl_sealed_from_sha256,
            "byte_exact_decode_pass": byte_exact_decode_pass,
        })
        .to_string();
        self.record(
            step,
            inner_step,
            "step_qnanl_inline",
            "prismcode_qnanl_inline_seal_verify",
            &[OP_MD_QNANL_SEAL, OP_MD_QNANL_VERIFY],
            (1, 1, 1),
            (1, 1, 1),
            0,
            Some(&note),
        );
        Ok(())
    }

    pub fn record_graph_window(
        &mut self,
        start_step: u64,
        logical_steps: u64,
        host_launch_count: u32,
        steps_per_launch: u32,
        launch_path: &str,
        opcodes: &[&str],
        note: Option<&str>,
    ) {
        self.graph_window_count = self.graph_window_count.saturating_add(1);
        if host_launch_count == 0 || steps_per_launch == 0 || logical_steps == 0 {
            self.graph_window_consistent = false;
            return;
        }
        let expected_logical_steps = u64::from(host_launch_count) * u64::from(steps_per_launch);
        let Some(window_end_step) = start_step.checked_add(logical_steps) else {
            self.graph_window_consistent = false;
            return;
        };
        if let Some(expected_start) = self.last_graph_window_end_step {
            if start_step > expected_start {
                self.graph_window_gap_count = self.graph_window_gap_count.saturating_add(1);
            } else if start_step < expected_start {
                self.graph_window_overlap_count = self.graph_window_overlap_count.saturating_add(1);
            }
        }
        if logical_steps != expected_logical_steps
            || self
                .last_graph_window_end_step
                .is_some_and(|expected_start| expected_start != start_step)
        {
            self.graph_window_consistent = false;
            return;
        }
        self.last_graph_window_end_step = Some(window_end_step);
        self.graph_logical_steps = self.graph_logical_steps.saturating_add(logical_steps);
        self.graph_host_launches = self
            .graph_host_launches
            .saturating_add(host_launch_count as u64);
        self.graph_steps_per_launch_max =
            self.graph_steps_per_launch_max.max(steps_per_launch as u64);
        let shape_logical_steps = self
            .graph_shape_logical_steps
            .entry(steps_per_launch as u64)
            .or_insert(0);
        *shape_logical_steps = shape_logical_steps.saturating_add(logical_steps);
        let shape_host_launches = self
            .graph_shape_host_launches
            .entry(steps_per_launch as u64)
            .or_insert(0);
        *shape_host_launches = shape_host_launches.saturating_add(host_launch_count as u64);
        // Retain the last recorded graph window so that an exact-tail exemption
        // can only apply to a demonstrably terminal, one-launch capture. Merely
        // observing one smaller graph shape somewhere in the run is insufficient.
        self.last_graph_window_logical_steps = logical_steps;
        self.last_graph_window_host_launches = host_launch_count as u64;
        self.last_graph_window_steps_per_launch = steps_per_launch as u64;
        if is_resident_md_graph_launch_path(launch_path) {
            self.resident_graph_launches = self
                .resident_graph_launches
                .saturating_add(host_launch_count as u64);
        } else {
            self.legacy_graph_launches = self
                .legacy_graph_launches
                .saturating_add(host_launch_count as u64);
        }

        for idx in 0..host_launch_count {
            let step =
                start_step.saturating_add((idx as u64).saturating_mul(steps_per_launch as u64));
            self.record(
                step,
                0,
                "cuda_graph_replay",
                launch_path,
                opcodes,
                (1, 1, 1),
                (1, 1, 1),
                0,
                note,
            );
        }
    }

    pub fn receipt(&self) -> MdNativeRuntimeReceipt {
        let mut required = core_required_opcodes(
            self.config.materialized_readouts_required,
            self.config.uv_burst_suppressed,
        );
        let mut conditional = Vec::new();
        if self.config.use_neighbor_list {
            conditional.push(OP_MD_NEIGHBOR_REBUILD.to_string());
        }
        if self.config.ladd_enabled {
            conditional.push(OP_MD_LADD_UPDATE.to_string());
        }
        if self.config.heartbeat_enabled {
            conditional.push(OP_MD_HEARTBEAT_ASSERT.to_string());
        }
        if self.config.ca_restraint_enabled {
            conditional.push(OP_MD_CA_RESTRAINT.to_string());
        }
        let nma_perturb_required_by_observed_window =
            self.nma_perturb_required_by_observed_window();
        if nma_perturb_required_by_observed_window {
            conditional.push(OP_MD_NMA_PERTURB.to_string());
        }
        required.extend(conditional.iter().cloned());

        let missing: Vec<String> = required
            .iter()
            .filter(|opcode| !self.opcode_counts.contains_key(*opcode))
            .cloned()
            .collect();
        let materialized_readout_opcodes = [
            OP_MD_SIGNAL_GRID_READOUT.to_string(),
            OP_MD_KCC_DESCRIPTOR.to_string(),
        ];
        let deferred_materialized_readout_opcodes: Vec<String> =
            if self.config.materialized_readouts_required {
                Vec::new()
            } else {
                materialized_readout_opcodes
                    .iter()
                    .filter(|opcode| !self.opcode_counts.contains_key(*opcode))
                    .cloned()
                    .collect()
            };
        let materialized_readout_status = if materialized_readout_opcodes
            .iter()
            .all(|opcode| self.opcode_counts.contains_key(opcode))
        {
            "MATERIALIZED"
        } else if !self.config.materialized_readouts_required
            && !deferred_materialized_readout_opcodes.is_empty()
        {
            "DEFERRED_BOUNDED_TRANSITION_VIEW"
        } else {
            "MISSING_REQUIRED_MATERIALIZATION"
        };
        let readout_opcode_coverage = if coverage_for(
            &self.opcode_counts,
            &[
                OP_MD_SPIKE_READOUT,
                OP_MD_SIGNAL_GRID_READOUT,
                OP_MD_KCC_DESCRIPTOR,
            ],
        ) == "PASS"
        {
            "PASS"
        } else if !self.config.materialized_readouts_required
            && self.opcode_counts.contains_key(OP_MD_SPIKE_READOUT)
        {
            "BOUNDED_MATERIALIZATION_DEFERRED"
        } else {
            "FAIL"
        };
        let coverage_status = if missing.is_empty() && materialized_readout_status == "MATERIALIZED"
        {
            "PASS"
        } else if missing.is_empty()
            && readout_opcode_coverage == "BOUNDED_MATERIALIZATION_DEFERRED"
        {
            "BOUNDED_MATERIALIZATION_DEFERRED"
        } else {
            "FAIL"
        };
        let legacy_nonbonded = self
            .config
            .nonbonded_execution_contract
            .get("mode")
            .and_then(serde_json::Value::as_str)
            == Some("NON_PRODUCTION_LEGACY_UNIFORM_FUDGE");
        let status = if legacy_nonbonded {
            "PARTIAL"
        } else if coverage_status == "PASS" || !self.config.require_complete {
            "PASS"
        } else if coverage_status == "BOUNDED_MATERIALIZATION_DEFERRED" {
            "PARTIAL"
        } else {
            "FAIL"
        };
        let coverage_hash_blake3 = blake3::hash(
            &serde_json::to_vec(&json!({
                "required": required,
                "missing": missing,
                "materialized_readouts_required": self.config.materialized_readouts_required,
                "deferred_materialized_readout_opcodes": deferred_materialized_readout_opcodes,
                "nma_perturb_required_by_observed_window": nma_perturb_required_by_observed_window,
                "opcode_counts": self.opcode_counts,
                "kernel_counts": self.kernel_counts,
            }))
            .unwrap_or_default(),
        )
        .to_hex()
        .to_string();
        let dispatch_trace_hash_blake3 = self.trace_hasher.clone().finalize().to_hex().to_string();
        let standard_step_dispatch_count = self.phase_count_prefix("step")
            + self.phase_count_prefix("step_batch")
            + self.phase_count_prefix("post_fused")
            + self.phase_count_prefix("teardown")
            + self.phase_count_prefix("sync");
        let cuda_graph_capture_dispatch_count = self.phase_count_prefix("cuda_graph_capture");
        let cuda_graph_replay_dispatch_count = self.phase_count_prefix("cuda_graph_replay");
        let graph_host_launch_collapse_ratio_raw = if self.graph_host_launches > 0 {
            self.graph_logical_steps as f64 / self.graph_host_launches as f64
        } else {
            0.0
        };
        // The sustained body is the modal launch shape, not the largest shape.
        // This prevents a one-off oversized/malformed observation from becoming
        // the body contract. Ties prefer the larger shape but cannot pass the
        // sustained predicate below unless repeated.
        let graph_sustained_body_steps_per_launch = self
            .graph_shape_host_launches
            .iter()
            .max_by_key(|(steps_per_launch, host_launches)| (**host_launches, **steps_per_launch))
            .map(|(steps_per_launch, _)| *steps_per_launch)
            .unwrap_or(0);
        let graph_sustained_body_logical_step_count = self
            .graph_shape_logical_steps
            .get(&graph_sustained_body_steps_per_launch)
            .copied()
            .unwrap_or(0);
        let graph_sustained_body_host_launch_count = self
            .graph_shape_host_launches
            .get(&graph_sustained_body_steps_per_launch)
            .copied()
            .unwrap_or(0);
        let graph_exact_tail_logical_step_count = self
            .graph_logical_steps
            .saturating_sub(graph_sustained_body_logical_step_count);
        let graph_exact_tail_host_launch_count = self
            .graph_host_launches
            .saturating_sub(graph_sustained_body_host_launch_count);
        let graph_exact_tail_status = if graph_exact_tail_host_launch_count == 0 {
            "NONE"
        } else if graph_exact_tail_host_launch_count == 1
            && graph_exact_tail_logical_step_count > 0
            && graph_exact_tail_logical_step_count < graph_sustained_body_steps_per_launch
            && self.last_graph_window_host_launches == 1
            && self.last_graph_window_steps_per_launch == graph_exact_tail_logical_step_count
            && self.last_graph_window_logical_steps == graph_exact_tail_logical_step_count
        {
            "PRESENT"
        } else {
            "INVALID_NONTERMINAL_OR_MULTIPLE"
        };
        let graph_sustained_body_collapse_ratio = if graph_sustained_body_host_launch_count > 0 {
            graph_sustained_body_logical_step_count as f64
                / graph_sustained_body_host_launch_count as f64
        } else {
            0.0
        };
        let graph_host_launch_collapse_ratio =
            if matches!(graph_exact_tail_status, "NONE" | "PRESENT") {
                graph_sustained_body_collapse_ratio
            } else {
                graph_host_launch_collapse_ratio_raw
            };
        let graph_window_consistency_status =
            if self.graph_window_count > 0 && self.graph_window_consistent {
                "PASS"
            } else {
                "FAIL"
            };
        let hotpath_acceleration_status = if graph_window_consistency_status == "PASS"
            && matches!(graph_exact_tail_status, "NONE" | "PRESENT")
            && graph_sustained_body_host_launch_count >= 2
            && self.resident_graph_launches > 0
            && self.legacy_graph_launches == 0
            && graph_host_launch_collapse_ratio > 1.0
        {
            "PASS"
        } else {
            "FAIL"
        };
        let status = if self.graph_window_count > 0 && hotpath_acceleration_status != "PASS" {
            "FAIL"
        } else {
            status
        };
        let qnanl_inline_seal_count = self
            .opcode_counts
            .get(OP_MD_QNANL_SEAL)
            .copied()
            .unwrap_or(0);
        let qnanl_inline_verify_count = self
            .opcode_counts
            .get(OP_MD_QNANL_VERIFY)
            .copied()
            .unwrap_or(0);
        let qnanl_inline_pair_status = if qnanl_inline_seal_count > 0
            && qnanl_inline_seal_count == qnanl_inline_verify_count
        {
            "PASS"
        } else {
            "EVIDENCE_NOT_FOUND"
        };

        MdNativeRuntimeReceipt {
            schema_kind: MD_NATIVE_SCHEMA_KIND.to_string(),
            schema_version: MD_NATIVE_SCHEMA_VERSION,
            status: status.to_string(),
            execution_mode: "native_prismcode_md_opcode_dispatch_cuda_macro_fused".to_string(),
            stream_id: self.config.stream_id,
            require_complete: self.config.require_complete,
            materialized_readouts_required: self.config.materialized_readouts_required,
            materialized_readout_policy: if self.config.materialized_readouts_required {
                "full_teardown_materialization_required".to_string()
            } else {
                "bounded_transition_view_materialization_deferred".to_string()
            },
            n_atoms: self.config.n_atoms,
            n_residues: self.config.n_residues,
            grid_dim: self.config.grid_dim,
            fused_inner_steps: self.config.fused_inner_steps,
            use_neighbor_list: self.config.use_neighbor_list,
            ladd_enabled: self.config.ladd_enabled,
            heartbeat_enabled: self.config.heartbeat_enabled,
            ca_restraint_enabled: self.config.ca_restraint_enabled,
            nma_enabled: self.config.nma_enabled,
            uv_burst_suppressed: self.config.uv_burst_suppressed,
            nonbonded_execution_contract: self.config.nonbonded_execution_contract.clone(),
            implementation_status_language: if legacy_nonbonded {
                "IMPLEMENTATION_PARTIAL".to_string()
            } else if status == "PASS" {
                "IMPLEMENTATION_VERIFIED".to_string()
            } else {
                "IMPLEMENTATION_BLOCKED".to_string()
            },
            exact_nonbonded_exception_upload_contract: self
                .exact_nonbonded_exception_upload_contract
                .clone(),
            harmonic_parameter_upload_contract: self.harmonic_parameter_upload_contract.clone(),
            nma_perturb_required_by_observed_window,
            dispatch_count: self.dispatch_count,
            logical_step_count: self.logical_steps.len() as u64,
            inner_step_count: self.inner_steps,
            opcode_counts: self.opcode_counts.clone(),
            qnanl_inline_seal_count,
            qnanl_inline_verify_count,
            qnanl_inline_pair_status: qnanl_inline_pair_status.to_string(),
            kernel_counts: self.kernel_counts.clone(),
            phase_counts: self.phase_counts.clone(),
            standard_step_dispatch_count,
            cuda_graph_capture_dispatch_count,
            cuda_graph_replay_dispatch_count,
            resident_graph_replay_dispatch_count: self.resident_graph_launches,
            legacy_graph_replay_dispatch_count: self.legacy_graph_launches,
            graph_logical_step_count: self.graph_logical_steps,
            graph_steps_per_launch_max: self.graph_steps_per_launch_max,
            graph_window_count: self.graph_window_count,
            graph_window_consistency_status: graph_window_consistency_status.to_string(),
            graph_window_gap_count: self.graph_window_gap_count,
            graph_window_overlap_count: self.graph_window_overlap_count,
            graph_host_launch_collapse_ratio_raw,
            graph_sustained_body_logical_step_count,
            graph_sustained_body_host_launch_count,
            graph_sustained_body_steps_per_launch,
            graph_exact_tail_logical_step_count,
            graph_exact_tail_host_launch_count,
            graph_exact_tail_status: graph_exact_tail_status.to_string(),
            graph_host_launch_collapse_ratio,
            hotpath_acceleration_status: hotpath_acceleration_status.to_string(),
            hotpath_acceleration_contract:
                "resident WHILE CUDA graph only; schema-v2 requires exact contiguous window arithmetic and a repeated modal full-body shape; one exact terminal tail is preserved separately and excluded from sustained collapse"
                    .to_string(),
            required_core_opcodes: core_required_opcodes(
                self.config.materialized_readouts_required,
                self.config.uv_burst_suppressed,
            ),
            required_conditional_opcodes: conditional,
            deferred_materialized_readout_opcodes,
            missing_required_opcodes: missing,
            coverage_status: coverage_status.to_string(),
            materialized_readout_status: materialized_readout_status.to_string(),
            coverage_hash_blake3,
            dispatch_trace_hash_blake3,
            sample_dispatches: self.sample_dispatches.clone(),
            macro_fused_cuda_kernel: true,
            force_kernel_opcode_coverage: coverage_for(
                &self.opcode_counts,
                &[OP_MD_FORCE_BONDED, OP_MD_FORCE_NONBONDED],
            ),
            integration_opcode_coverage: coverage_for(
                &self.opcode_counts,
                &[
                    OP_MD_INTEGRATE_VELOCITY_VERLET,
                    OP_MD_SHAKE_CONSTRAIN,
                    OP_MD_VELOCITY_SECOND_HALF,
                    OP_MD_VELOCITY_CLAMP,
                ],
            ),
            observation_opcode_coverage: coverage_for(
                &self.opcode_counts,
                &[
                    OP_MD_EXCLUSION_FIELD,
                    OP_MD_WATER_DENSITY,
                    OP_MD_LIF_OBSERVE,
                    OP_MD_SPIKE_EMIT,
                    OP_MD_KCC_UPDATE,
                ],
            ),
            readout_opcode_coverage: readout_opcode_coverage.to_string(),
        }
    }

    fn phase_count_prefix(&self, prefix: &str) -> u64 {
        self.phase_counts
            .iter()
            .filter(|(phase, _)| phase.starts_with(prefix))
            .map(|(_, count)| *count)
            .sum()
    }

    fn nma_perturb_required_by_observed_window(&self) -> bool {
        if self.opcode_counts.contains_key(OP_MD_NMA_PERTURB) {
            return true;
        }
        if !self.config.nma_enabled {
            return false;
        }
        self.sample_dispatches.iter().any(|dispatch| {
            dispatch
                .note
                .as_deref()
                .and_then(note_declares_nma_window_overlap)
                .unwrap_or(false)
        })
    }

    pub fn write_receipt(&self, path: &Path) -> std::io::Result<MdNativeRuntimeReceipt> {
        let receipt = self.receipt();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_vec_pretty(&receipt)?)?;
        Ok(receipt)
    }
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn note_declares_nma_window_overlap(note: &str) -> Option<bool> {
    if note.contains("\"nma_window_overlap\":true") || note.contains("nma_window_overlap=true") {
        return Some(true);
    }
    if note.contains("\"nma_window_overlap\":false") || note.contains("nma_window_overlap=false") {
        return Some(false);
    }
    let value = serde_json::from_str::<serde_json::Value>(note).ok()?;
    value
        .get("nma_window_overlap")
        .and_then(serde_json::Value::as_bool)
        .or_else(|| {
            value
                .get("nma_window")
                .and_then(|nested| nested.get("overlap"))
                .and_then(serde_json::Value::as_bool)
        })
        .or_else(|| {
            value
                .get("phase_window")
                .and_then(|nested| nested.get("nma_overlap"))
                .and_then(serde_json::Value::as_bool)
        })
}

fn core_required_opcodes(
    materialized_readouts_required: bool,
    uv_burst_suppressed: bool,
) -> Vec<String> {
    let mut opcodes = vec![
        OP_MD_PROTOCOL_DIRECTOR,
        OP_MD_ACCUMULATOR_ZERO,
        OP_MD_FORCE_BONDED,
        OP_MD_FORCE_NONBONDED,
        OP_MD_INTEGRATE_VELOCITY_VERLET,
        OP_MD_SHAKE_CONSTRAIN,
        OP_MD_EXCLUSION_FIELD,
        OP_MD_WATER_DENSITY,
        // P2 — a declared UV-off control never launches the pump kernel, so the
        // contract requires the suppressed variant. The swap is applied
        // identically at every producer, so coverage stays PASS in both arms
        // while the op stream stops asserting a pump that did not run.
        if uv_burst_suppressed {
            OP_MD_PHOTO_UV_PUMP_SUPPRESSED
        } else {
            OP_MD_PHOTO_UV_PUMP
        },
        OP_MD_EXTERNAL_WORK_REDUCE,
        OP_MD_VELOCITY_SECOND_HALF,
        OP_MD_VELOCITY_CLAMP,
        OP_MD_LIF_OBSERVE,
        OP_MD_COUPLING_BUILD,
        OP_MD_KCC_UPDATE,
        OP_MD_SPIKE_EMIT,
        OP_MD_SPIKE_READOUT,
    ];
    if materialized_readouts_required {
        opcodes.push(OP_MD_SIGNAL_GRID_READOUT);
        opcodes.push(OP_MD_KCC_DESCRIPTOR);
    }
    opcodes.iter().map(|opcode| opcode.to_string()).collect()
}

fn coverage_for(counts: &BTreeMap<String, u64>, opcodes: &[&str]) -> String {
    if opcodes.iter().all(|opcode| counts.contains_key(*opcode)) {
        "PASS".to_string()
    } else {
        "FAIL".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bound_parent_paths_preserve_resident_counts_and_exact_terminal_tail() {
        let mut state=MdNativeRuntimeState::new(config(false));
        record_core_without_materialized_readouts(&mut state);
        for (ordinal,path) in ["graph.while_resident","graph.while_resident",
            "graph.while_ladd_kcc_v2_parent","graph.while_ladd_kcc_v2_parent",
            "graph.shared_coupling_while_ladd_kcc_v2_parent",
            "graph.shared_coupling_while_ladd_kcc_v2_parent"].into_iter().enumerate() {
            state.record_graph_window(ordinal as u64*10,10,1,10,path,&[OP_MD_PROTOCOL_DIRECTOR],None);
        }
        state.record_graph_window(60,5,1,5,"graph.while_ladd_kcc_v2_parent",&[OP_MD_PROTOCOL_DIRECTOR],None);
        let receipt=state.receipt();
        assert_eq!(receipt.resident_graph_replay_dispatch_count,7);
        assert_eq!(receipt.legacy_graph_replay_dispatch_count,0);
        assert_eq!(receipt.graph_sustained_body_steps_per_launch,10);
        assert_eq!(receipt.graph_sustained_body_host_launch_count,6);
        assert_eq!(receipt.graph_exact_tail_status,"PRESENT");
        assert_eq!(receipt.graph_window_consistency_status,"PASS");
        assert_eq!(receipt.hotpath_acceleration_status,"PASS");
        assert_eq!(receipt.graph_host_launch_collapse_ratio,10.0);
    }

    #[test]
    fn unknown_or_legacy_graph_labels_are_not_resident() {
        for path in ["graph.run_chunk","mono.launch_on_stream","graph.while_resident_unknown",
            "graph.shared_coupling_while_ladd_kcc_v2_parent.unverified"] {
            assert!(!is_resident_md_graph_launch_path(path));
            let mut state=MdNativeRuntimeState::new(config(false));
            record_core_without_materialized_readouts(&mut state);
            state.record_graph_window(0,20,2,10,path,&[OP_MD_PROTOCOL_DIRECTOR],None);
            let receipt=state.receipt();
            assert_eq!(receipt.resident_graph_replay_dispatch_count,0);
            assert_eq!(receipt.legacy_graph_replay_dispatch_count,2);
            assert_eq!(receipt.hotpath_acceleration_status,"FAIL");
        }
    }

    #[test]
    fn combined_resident_paths_do_not_exempt_nonterminal_tail_or_clock_gap() {
        for gap in [false,true] {
            let mut state=MdNativeRuntimeState::new(config(false));
            record_core_without_materialized_readouts(&mut state);
            let path="graph.shared_coupling_while_ladd_kcc_v2_parent";
            state.record_graph_window(0,20,2,10,path,&[OP_MD_PROTOCOL_DIRECTOR],None);
            state.record_graph_window(if gap {21}else{20},5,1,5,path,&[OP_MD_PROTOCOL_DIRECTOR],None);
            state.record_graph_window(25,10,1,10,path,&[OP_MD_PROTOCOL_DIRECTOR],None);
            assert_eq!(state.receipt().hotpath_acceleration_status,"FAIL");
        }
    }


    fn config(materialized_readouts_required: bool) -> MdNativeRuntimeConfig {
        MdNativeRuntimeConfig {
            stream_id: 0,
            require_complete: true,
            materialized_readouts_required,
            n_atoms: 16,
            n_residues: 4,
            grid_dim: 8,
            fused_inner_steps: 6,
            use_neighbor_list: false,
            ladd_enabled: false,
            heartbeat_enabled: false,
            ca_restraint_enabled: false,
            nma_enabled: false,
            uv_burst_suppressed: false,
            nonbonded_execution_contract: serde_json::json!({
                "schema_kind": "prismos_nonbonded_execution_contract",
                "schema_version": 1,
                "mode": "OPENMM_EXACT_V2",
                "explicit_nonproduction_option": false,
                "fallback_used": false,
                "scientific_status": "PASS",
                "implementation_status_language": "IMPLEMENTATION_VERIFIED",
            }),
        }
    }

    fn record_core_without_materialized_readouts(state: &mut MdNativeRuntimeState) {
        state.record(
            1,
            1,
            "bounded_transition_view",
            "nhs_amber_fused_step",
            &[
                OP_MD_PROTOCOL_DIRECTOR,
                OP_MD_ACCUMULATOR_ZERO,
                OP_MD_FORCE_BONDED,
                OP_MD_FORCE_NONBONDED,
                OP_MD_INTEGRATE_VELOCITY_VERLET,
                OP_MD_SHAKE_CONSTRAIN,
                OP_MD_EXCLUSION_FIELD,
                OP_MD_WATER_DENSITY,
                OP_MD_PHOTO_UV_PUMP,
                OP_MD_EXTERNAL_WORK_REDUCE,
                OP_MD_VELOCITY_SECOND_HALF,
                OP_MD_VELOCITY_CLAMP,
                OP_MD_LIF_OBSERVE,
                OP_MD_COUPLING_BUILD,
                OP_MD_KCC_UPDATE,
                OP_MD_SPIKE_EMIT,
                OP_MD_SPIKE_READOUT,
            ],
            (1, 1, 1),
            (1, 1, 1),
            0,
            Some("test"),
        );
    }

    #[test]
    fn bounded_materialization_defers_signal_kcc_only_after_spike_readout() {
        let mut state = MdNativeRuntimeState::new(config(false));
        record_core_without_materialized_readouts(&mut state);

        let receipt = state.receipt();

        assert_eq!(receipt.status, "PARTIAL");
        assert_eq!(receipt.coverage_status, "BOUNDED_MATERIALIZATION_DEFERRED");
        assert_eq!(
            receipt.readout_opcode_coverage,
            "BOUNDED_MATERIALIZATION_DEFERRED"
        );
        assert_eq!(
            receipt.materialized_readout_status,
            "DEFERRED_BOUNDED_TRANSITION_VIEW"
        );
        assert_eq!(
            receipt.deferred_materialized_readout_opcodes,
            vec![
                OP_MD_SIGNAL_GRID_READOUT.to_string(),
                OP_MD_KCC_DESCRIPTOR.to_string()
            ]
        );
        assert!(receipt
            .required_core_opcodes
            .contains(&OP_MD_SPIKE_READOUT.to_string()));
        assert!(!receipt
            .missing_required_opcodes
            .contains(&OP_MD_SPIKE_READOUT.to_string()));
    }

    #[test]
    fn explicit_legacy_nonbonded_caps_receipt_at_partial() {
        let mut cfg = config(false);
        cfg.require_complete = false;
        cfg.nonbonded_execution_contract = serde_json::json!({
            "schema_kind": "prismos_nonbonded_execution_contract",
            "schema_version": 1,
            "mode": "NON_PRODUCTION_LEGACY_UNIFORM_FUDGE",
            "explicit_nonproduction_option": true,
            "fallback_used": true,
            "scientific_status": "PARTIAL",
            "implementation_status_language": "IMPLEMENTATION_PARTIAL",
        });
        let receipt = MdNativeRuntimeState::new(cfg).receipt();
        assert_eq!(receipt.status, "PARTIAL");
        assert_eq!(
            receipt.implementation_status_language,
            "IMPLEMENTATION_PARTIAL"
        );
        assert_eq!(
            receipt.nonbonded_execution_contract["fallback_used"],
            serde_json::Value::Bool(true)
        );
    }

    #[test]
    fn bounded_materialization_without_spike_readout_still_fails() {
        let mut state = MdNativeRuntimeState::new(config(false));
        state.record(
            1,
            1,
            "bounded_transition_view",
            "nhs_amber_fused_step",
            &[
                OP_MD_PROTOCOL_DIRECTOR,
                OP_MD_ACCUMULATOR_ZERO,
                OP_MD_FORCE_BONDED,
                OP_MD_FORCE_NONBONDED,
                OP_MD_INTEGRATE_VELOCITY_VERLET,
                OP_MD_SHAKE_CONSTRAIN,
                OP_MD_EXCLUSION_FIELD,
                OP_MD_WATER_DENSITY,
                OP_MD_PHOTO_UV_PUMP,
                OP_MD_EXTERNAL_WORK_REDUCE,
                OP_MD_VELOCITY_SECOND_HALF,
                OP_MD_VELOCITY_CLAMP,
                OP_MD_LIF_OBSERVE,
                OP_MD_COUPLING_BUILD,
                OP_MD_KCC_UPDATE,
                OP_MD_SPIKE_EMIT,
            ],
            (1, 1, 1),
            (1, 1, 1),
            0,
            Some("test"),
        );

        let receipt = state.receipt();

        assert_eq!(receipt.status, "FAIL");
        assert_eq!(receipt.coverage_status, "FAIL");
        assert_eq!(receipt.readout_opcode_coverage, "FAIL");
        assert!(receipt
            .missing_required_opcodes
            .contains(&OP_MD_SPIKE_READOUT.to_string()));
    }

    #[test]
    fn strict_materialization_requires_signal_grid_and_kcc_readouts() {
        let mut state = MdNativeRuntimeState::new(config(true));
        record_core_without_materialized_readouts(&mut state);

        let receipt = state.receipt();

        assert_eq!(receipt.status, "FAIL");
        assert_eq!(receipt.coverage_status, "FAIL");
        assert_eq!(receipt.readout_opcode_coverage, "FAIL");
        assert_eq!(
            receipt.materialized_readout_status,
            "MISSING_REQUIRED_MATERIALIZATION"
        );
        assert!(receipt
            .missing_required_opcodes
            .contains(&OP_MD_SIGNAL_GRID_READOUT.to_string()));
        assert!(receipt
            .missing_required_opcodes
            .contains(&OP_MD_KCC_DESCRIPTOR.to_string()));
    }

    #[test]
    fn nma_flag_without_window_overlap_does_not_require_perturb_opcode() {
        let mut cfg = config(false);
        cfg.nma_enabled = true;
        let mut state = MdNativeRuntimeState::new(cfg);
        record_core_without_materialized_readouts(&mut state);
        state.record_graph_window(
            1,
            8,
            1,
            8,
            "graph.while_resident",
            &[OP_MD_PROTOCOL_DIRECTOR, OP_MD_FORCE_BONDED],
            Some(r#"{"nma_window_overlap":false}"#),
        );

        let receipt = state.receipt();

        assert!(!receipt.nma_perturb_required_by_observed_window);
        assert!(!receipt
            .required_conditional_opcodes
            .contains(&OP_MD_NMA_PERTURB.to_string()));
        assert!(!receipt
            .missing_required_opcodes
            .contains(&OP_MD_NMA_PERTURB.to_string()));
    }

    #[test]
    fn nma_window_overlap_requires_perturb_opcode() {
        let mut cfg = config(false);
        cfg.nma_enabled = true;
        let mut state = MdNativeRuntimeState::new(cfg);
        record_core_without_materialized_readouts(&mut state);
        state.record_graph_window(
            1,
            8,
            1,
            8,
            "graph.while_resident",
            &[OP_MD_PROTOCOL_DIRECTOR, OP_MD_FORCE_BONDED],
            Some(r#"{"nma_window_overlap":true}"#),
        );

        let receipt = state.receipt();

        assert!(receipt.nma_perturb_required_by_observed_window);
        assert!(receipt
            .required_conditional_opcodes
            .contains(&OP_MD_NMA_PERTURB.to_string()));
        assert!(receipt
            .missing_required_opcodes
            .contains(&OP_MD_NMA_PERTURB.to_string()));
    }

    #[test]
    fn nma_window_overlap_passes_when_perturb_opcode_recorded() {
        let mut cfg = config(false);
        cfg.nma_enabled = true;
        let mut state = MdNativeRuntimeState::new(cfg);
        record_core_without_materialized_readouts(&mut state);
        state.record_graph_window(
            1,
            8,
            1,
            8,
            "graph.while_resident",
            &[
                OP_MD_PROTOCOL_DIRECTOR,
                OP_MD_FORCE_BONDED,
                OP_MD_NMA_PERTURB,
            ],
            Some(r#"{"nma_window_overlap":true}"#),
        );

        let receipt = state.receipt();

        assert!(receipt.nma_perturb_required_by_observed_window);
        assert!(receipt
            .required_conditional_opcodes
            .contains(&OP_MD_NMA_PERTURB.to_string()));
        assert!(!receipt
            .missing_required_opcodes
            .contains(&OP_MD_NMA_PERTURB.to_string()));
    }

    #[test]
    fn sustained_hotpath_collapse_preserves_one_exact_tail_without_threshold_demotion() {
        let mut state = MdNativeRuntimeState::new(config(false));
        state.record_graph_window(
            0,
            115 * 64,
            115,
            64,
            "graph.while_resident",
            &[OP_MD_PROTOCOL_DIRECTOR],
            Some("full resident body"),
        );
        state.record_graph_window(
            115 * 64,
            39,
            1,
            39,
            "graph.while_resident",
            &[OP_MD_PROTOCOL_DIRECTOR],
            Some("exact terminal tail"),
        );

        let receipt = state.receipt();

        assert_eq!(receipt.schema_version, 2);
        assert_eq!(receipt.graph_window_consistency_status, "PASS");
        assert_eq!(receipt.graph_logical_step_count, 7_399);
        assert_eq!(receipt.cuda_graph_replay_dispatch_count, 116);
        assert_eq!(receipt.graph_steps_per_launch_max, 64);
        assert_eq!(receipt.graph_sustained_body_logical_step_count, 7_360);
        assert_eq!(receipt.graph_sustained_body_host_launch_count, 115);
        assert_eq!(receipt.graph_exact_tail_logical_step_count, 39);
        assert_eq!(receipt.graph_exact_tail_host_launch_count, 1);
        assert_eq!(receipt.graph_exact_tail_status, "PRESENT");
        assert_eq!(receipt.resident_graph_replay_dispatch_count, 116);
        assert_eq!(receipt.legacy_graph_replay_dispatch_count, 0);
        assert!((receipt.graph_host_launch_collapse_ratio_raw - 63.78448275862069).abs() < 1e-12);
        assert_eq!(receipt.graph_host_launch_collapse_ratio, 64.0);
        assert_eq!(receipt.hotpath_acceleration_status, "PASS");
    }

    #[test]
    fn any_governed_legacy_window_is_preserved_and_fails_hotpath() {
        let mut state = MdNativeRuntimeState::new(config(false));
        state.record_graph_window(
            0,
            128,
            2,
            64,
            "graph.while_resident",
            &[OP_MD_PROTOCOL_DIRECTOR],
            Some("resident body"),
        );
        state.record_graph_window(
            128,
            64,
            1,
            64,
            "mono.launch_on_stream",
            &[OP_MD_PROTOCOL_DIRECTOR],
            Some("truthfully non-resident monolithic MD window"),
        );

        let receipt = state.receipt();

        assert_eq!(receipt.graph_window_consistency_status, "PASS");
        assert_eq!(receipt.resident_graph_replay_dispatch_count, 2);
        assert_eq!(receipt.legacy_graph_replay_dispatch_count, 1);
        assert_eq!(receipt.graph_logical_step_count, 192);
        assert_eq!(receipt.cuda_graph_replay_dispatch_count, 3);
        assert_eq!(receipt.hotpath_acceleration_status, "FAIL");
        assert_eq!(receipt.status, "FAIL");
        assert_eq!(
            receipt.implementation_status_language,
            "IMPLEMENTATION_BLOCKED"
        );
    }

    #[test]
    fn nonterminal_smaller_graph_window_cannot_claim_exact_tail_exemption() {
        let mut state = MdNativeRuntimeState::new(config(false));
        state.record_graph_window(
            0,
            39,
            1,
            39,
            "graph.while_resident",
            &[OP_MD_PROTOCOL_DIRECTOR],
            Some("smaller nonterminal window"),
        );
        state.record_graph_window(
            39,
            64,
            1,
            64,
            "graph.while_resident",
            &[OP_MD_PROTOCOL_DIRECTOR],
            Some("later full resident body"),
        );

        let receipt = state.receipt();

        assert_eq!(
            receipt.graph_exact_tail_status,
            "INVALID_NONTERMINAL_OR_MULTIPLE"
        );
        assert_eq!(receipt.graph_host_launch_collapse_ratio_raw, 51.5);
        assert_eq!(receipt.graph_host_launch_collapse_ratio, 51.5);
        assert_eq!(receipt.hotpath_acceleration_status, "FAIL");
    }

    #[test]
    fn zero_or_malformed_graph_window_inputs_fail_closed() {
        for (logical_steps, host_launches, steps_per_launch) in
            [(64, 0, 64), (64, 1, 0), (0, 1, 64), (63, 1, 64)]
        {
            let mut state = MdNativeRuntimeState::new(config(false));
            state.record_graph_window(
                0,
                logical_steps,
                host_launches,
                steps_per_launch,
                "graph.while_resident",
                &[OP_MD_PROTOCOL_DIRECTOR],
                Some("invalid graph window"),
            );
            let receipt = state.receipt();
            assert_eq!(receipt.graph_window_count, 1);
            assert_eq!(receipt.graph_window_consistency_status, "FAIL");
            assert_eq!(receipt.hotpath_acceleration_status, "FAIL");
        }
    }

    #[test]
    fn graph_window_gap_and_overlap_fail_closed() {
        for second_start in [127, 129] {
            let mut state = MdNativeRuntimeState::new(config(false));
            state.record_graph_window(
                0,
                128,
                2,
                64,
                "graph.while_resident",
                &[OP_MD_PROTOCOL_DIRECTOR],
                Some("repeated resident body"),
            );
            state.record_graph_window(
                second_start,
                39,
                1,
                39,
                "graph.while_resident",
                &[OP_MD_PROTOCOL_DIRECTOR],
                Some("non-contiguous tail"),
            );
            let receipt = state.receipt();
            assert_eq!(receipt.graph_window_consistency_status, "FAIL");
            assert_eq!(receipt.hotpath_acceleration_status, "FAIL");
            if second_start < 128 {
                assert_eq!(receipt.graph_window_overlap_count, 1);
                assert_eq!(receipt.graph_window_gap_count, 0);
            } else {
                assert_eq!(receipt.graph_window_overlap_count, 0);
                assert_eq!(receipt.graph_window_gap_count, 1);
            }
        }
    }

    #[test]
    fn multiple_or_oversized_tail_shapes_fail_closed() {
        let mut multiple = MdNativeRuntimeState::new(config(false));
        multiple.record_graph_window(
            0,
            115 * 64,
            115,
            64,
            "graph.while_resident",
            &[OP_MD_PROTOCOL_DIRECTOR],
            Some("full resident body"),
        );
        multiple.record_graph_window(
            115 * 64,
            39,
            1,
            39,
            "graph.while_resident",
            &[OP_MD_PROTOCOL_DIRECTOR],
            Some("first smaller window"),
        );
        multiple.record_graph_window(
            115 * 64 + 39,
            20,
            1,
            20,
            "graph.while_resident",
            &[OP_MD_PROTOCOL_DIRECTOR],
            Some("second smaller window"),
        );
        let multiple_receipt = multiple.receipt();
        assert_eq!(
            multiple_receipt.graph_exact_tail_status,
            "INVALID_NONTERMINAL_OR_MULTIPLE"
        );
        assert_eq!(multiple_receipt.hotpath_acceleration_status, "FAIL");

        let mut oversized = MdNativeRuntimeState::new(config(false));
        oversized.record_graph_window(
            0,
            115 * 64,
            115,
            64,
            "graph.while_resident",
            &[OP_MD_PROTOCOL_DIRECTOR],
            Some("full resident body"),
        );
        oversized.record_graph_window(
            115 * 64,
            128,
            1,
            128,
            "graph.while_resident",
            &[OP_MD_PROTOCOL_DIRECTOR],
            Some("one-off oversized shape"),
        );
        let oversized_receipt = oversized.receipt();
        assert_eq!(oversized_receipt.graph_sustained_body_steps_per_launch, 64);
        assert_eq!(
            oversized_receipt.graph_exact_tail_status,
            "INVALID_NONTERMINAL_OR_MULTIPLE"
        );
        assert!(oversized_receipt.graph_host_launch_collapse_ratio_raw > 64.0);
        assert_eq!(oversized_receipt.hotpath_acceleration_status, "FAIL");
    }
}

#[cfg(test)]
mod process_continuation_tests {
    use super::*;
    fn initial()->MdNativeRuntimeState {
        MdNativeRuntimeState::new(MdNativeRuntimeConfig{stream_id:3,require_complete:true,
            materialized_readouts_required:true,n_atoms:100,n_residues:10,grid_dim:32,fused_inner_steps:6,
            use_neighbor_list:true,ladd_enabled:true,heartbeat_enabled:true,ca_restraint_enabled:false,
            nma_enabled:false,uv_burst_suppressed:false,nonbonded_execution_contract:json!({"mode":"OPENMM_EXACT_V2"})})
    }
    fn record(state:&mut MdNativeRuntimeState,step:u64) {
        state.record(step,1,"native_test_dispatch","existing_kernel",&[OP_MD_FORCE_BONDED],(2,1,1),(256,1,1),0,None);
    }
    #[test]
    fn actual_runtime_state_receipt_and_hash_continue_exactly() {
        let mut uninterrupted=initial();uninterrupted.enable_process_checkpoint_trace().unwrap();
        for step in 0..300 {record(&mut uninterrupted,step);}
        // Config can legitimately change while the original hash seed remains
        // the exact configuration serialized at constructor time.
        uninterrupted.set_materialized_readouts_required(false);
        let bytes=uninterrupted.process_checkpoint_bytes().unwrap();
        let mut restored=MdNativeRuntimeState::from_process_checkpoint_bytes(&bytes).unwrap();
        assert_eq!(serde_json::to_value(uninterrupted.receipt()).unwrap(),serde_json::to_value(restored.receipt()).unwrap());
        for step in 300..600 {record(&mut uninterrupted,step);record(&mut restored,step);}
        assert_eq!(serde_json::to_value(uninterrupted.receipt()).unwrap(),serde_json::to_value(restored.receipt()).unwrap());
        assert_eq!(uninterrupted.process_checkpoint_bytes().unwrap(),restored.process_checkpoint_bytes().unwrap());
    }
    #[test]
    fn missing_early_trace_and_corrupt_historical_hash_inputs_refuse() {
        let mut state=initial();record(&mut state,1);
        assert!(state.enable_process_checkpoint_trace().is_err());assert!(state.process_checkpoint_bytes().is_err());
        let mut state=initial();state.enable_process_checkpoint_trace().unwrap();record(&mut state,1);
        let mut checkpoint:serde_json::Value=serde_json::from_slice(&state.process_checkpoint_bytes().unwrap()).unwrap();
        checkpoint["trace_restart_inputs"][0][2]=serde_json::Value::from(0);
        assert!(MdNativeRuntimeState::from_process_checkpoint_bytes(&serde_json::to_vec(&checkpoint).unwrap()).is_err());
    }
}
