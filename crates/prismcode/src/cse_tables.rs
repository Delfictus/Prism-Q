use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScientificStateSource {
    pub source_id: String,
    pub synthetic: bool,
    pub generator_version: String,
    pub seed: u64,
    pub molecule: MoleculeSource,
    pub protein: ProteinSource,
    #[serde(default)]
    pub waters: Vec<WaterSource>,
    pub timebase: TimebaseSource,
    #[serde(default)]
    pub variants: Vec<VariantSource>,
    pub events: Vec<EventSource>,
    #[serde(default)]
    pub uncertainties: Vec<UncertaintySource>,
    pub thermo: Vec<ThermoSource>,
    #[serde(default)]
    pub teacher_outputs: Vec<TeacherOutputSource>,
    #[serde(default)]
    pub gflownet_traces: Vec<GFlowNetTraceSource>,
    #[serde(default)]
    pub nanl_symbols: Vec<String>,
    pub provenance: ProvenanceSource,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MoleculeSource {
    pub entity_id: u64,
    pub name: String,
    pub topology_hash: String,
    pub atom_types: Vec<String>,
    pub bonds: Vec<[u32; 2]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProteinSource {
    pub entity_id: u64,
    pub receptor: String,
    pub chain_id: String,
    pub residues: Vec<String>,
    pub atom_site_count: u64,
    pub topology_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WaterSource {
    pub site_id: u64,
    pub shell: String,
    pub gate_state: String,
    pub time_ref: u64,
    pub provenance_ref: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TimebaseSource {
    pub time_id: u64,
    pub unit: String,
    pub start: f64,
    pub step: f64,
    pub count: u64,
    pub timebase_hash: String,
    pub forcefield_parameter_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VariantSource {
    pub variant_id: u64,
    pub reference_state_ref: u64,
    pub alternate_state_ref: u64,
    pub delta_kind: String,
    pub provenance_ref: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventSource {
    pub event_id: u64,
    pub event_kind: String,
    pub source_ref: u64,
    pub target_ref: u64,
    pub time_ref: u64,
    pub time_delta: i64,
    pub motif: String,
    pub amplitude: f64,
    pub unit: String,
    #[serde(default)]
    pub causal_parent_ref: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UncertaintySource {
    pub uncertainty_id: u64,
    pub target_ref: u64,
    pub distribution: String,
    pub calibration: String,
    pub value: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThermoSource {
    pub thermo_id: u64,
    pub target_ref: u64,
    pub quantity_kind: String,
    pub value: f64,
    pub unit_ref: String,
    pub uncertainty_ref: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TeacherOutputSource {
    pub output_id: u64,
    pub model_id: String,
    pub label: String,
    pub confidence: f64,
    pub uncertainty_ref: u64,
    pub provenance_ref: u64,
    pub airlock_state: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GFlowNetTraceSource {
    pub trace_id: u64,
    pub state_ref: u64,
    pub action_sequence: Vec<String>,
    pub reward: f64,
    pub flow: f64,
    pub policy_hash: String,
    pub provenance_ref: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProvenanceSource {
    pub provenance_id: u64,
    pub label: String,
    pub synthetic: bool,
    pub reason: String,
    pub source_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CseComponentTables {
    pub schema_version: String,
    pub source_id: String,
    pub source_hash: String,
    pub canonical_hash: String,
    pub dictionary_hash: String,
    pub entity_table: Vec<CseEntityRow>,
    pub event_table: Vec<CseEventRow>,
    pub thermo_table: Vec<CseThermoRow>,
    pub timebase_table: Vec<CseTimebaseRow>,
    pub teacher_output_table: Vec<CseTeacherRow>,
    pub gflownet_trace_table: Vec<CseGFlowNetRow>,
    pub provenance_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CseEntityRow {
    pub entity_id: u64,
    pub entity_kind: String,
    pub dictionary_ref: u32,
    pub topology_hash: String,
    pub component_span_start: u32,
    pub component_span_len: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CseTimebaseRow {
    pub time_id: u64,
    pub unit_ref: String,
    pub start: f64,
    pub step: f64,
    pub count: u64,
    pub timebase_hash: String,
    pub forcefield_parameter_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CseEventRow {
    pub event_id: u64,
    pub event_kind_ref: u32,
    pub source_ref: u64,
    pub target_ref: u64,
    pub time_ref: u64,
    pub time_delta: i64,
    pub motif_ref: u32,
    pub amplitude: f64,
    pub unit_ref: String,
    pub causal_parent_ref: Option<u64>,
    pub event_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CseThermoRow {
    pub thermo_id: u64,
    pub target_ref: u64,
    pub quantity_ref: u32,
    pub value: f64,
    pub unit_ref: String,
    pub uncertainty_ref: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CseTeacherRow {
    pub output_id: u64,
    pub model_ref: u32,
    pub label_ref: u32,
    pub confidence: f64,
    pub uncertainty_ref: u64,
    pub airlock_state: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CseGFlowNetRow {
    pub trace_id: u64,
    pub state_ref: u64,
    pub action_refs: Vec<u32>,
    pub reward: f64,
    pub flow: f64,
    pub policy_hash: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CseSourceDocument {
    pub schema_version: String,
    pub state_id: String,
    pub classification: String,
    pub synthetic: bool,
    pub unit_registry: Vec<CseUnitSource>,
    pub provenance: Vec<CseProvenanceRecordSource>,
    pub timebases: Vec<CseHdTimebaseSource>,
    pub molecules: Vec<CseHdMoleculeSource>,
    pub proteins: Vec<CseHdProteinSource>,
    pub water_sites: Vec<CseHdWaterSiteSource>,
    pub variants: Vec<CseHdVariantSource>,
    pub events: Vec<CseHdEventSource>,
    pub uncertainties: Vec<CseHdUncertaintySource>,
    pub thermo: Vec<CseHdThermoSource>,
    pub teacher_outputs: Vec<CseHdTeacherOutputSource>,
    pub gflownet_traces: Vec<CseHdGFlowNetTraceSource>,
    pub nanl_symbols: Vec<String>,
    pub nrf_frames: Vec<CseHdNrfFrameSource>,
    pub drf_states: Vec<CseHdDrfStateSource>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CseUnitSource {
    pub id: String,
    pub quantity_kind: String,
    pub unit: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CseProvenanceRecordSource {
    pub id: String,
    pub source_uri: String,
    pub source_hash: String,
    pub classification: String,
    pub synthetic: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CseHdTimebaseSource {
    pub id: String,
    pub timebase_hash: String,
    pub unit_ref: String,
    pub frame_start: u64,
    pub frame_end: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CseHdMoleculeSource {
    pub id: String,
    pub topology_hash: String,
    pub provenance_ref: String,
    pub atoms: Vec<CseHdAtomSource>,
    pub bonds: Vec<CseHdBondSource>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CseHdAtomSource {
    pub id: String,
    pub element: String,
    pub atom_type: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CseHdBondSource {
    pub id: String,
    pub source_atom: String,
    pub target_atom: String,
    pub order: u8,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CseHdProteinSource {
    pub id: String,
    pub topology_hash: String,
    pub provenance_ref: String,
    pub chains: Vec<CseHdProteinChainSource>,
    pub atom_sites: Vec<CseHdAtomSiteSource>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CseHdProteinChainSource {
    pub id: String,
    pub residues: Vec<CseHdResidueSource>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CseHdResidueSource {
    pub id: String,
    pub name: String,
    pub sequence_index: u32,
    pub atom_site_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CseHdAtomSiteSource {
    pub id: String,
    pub atom_name: String,
    pub atom_type: String,
    pub chain_ref: String,
    pub residue_ref: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CseHdWaterSiteSource {
    pub id: String,
    pub gate_kind: String,
    pub event_bound: bool,
    pub time_ref: Option<String>,
    pub provenance_ref: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CseHdVariantSource {
    pub id: String,
    pub reference_state_ref: String,
    pub alternate_state_ref: String,
    pub delta_hash: String,
    pub provenance_ref: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CseHdEventSource {
    pub id: String,
    pub kind: String,
    pub source_ref: String,
    pub target_ref: String,
    pub time_ref: String,
    pub order_index: u64,
    pub motif: String,
    pub provenance_ref: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CseHdUncertaintySource {
    pub id: String,
    pub target_ref: String,
    pub model: String,
    pub calibration_hash: String,
    pub provenance_ref: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CseHdThermoSource {
    pub id: String,
    pub target_ref: String,
    pub quantity_kind: String,
    pub unit_ref: String,
    pub value_payload_ref: String,
    pub uncertainty_ref: String,
    pub provenance_ref: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CseHdTeacherOutputSource {
    pub id: String,
    pub target_ref: String,
    pub model_ref: String,
    pub label: String,
    pub confidence: f64,
    pub airlock_state: String,
    pub provenance_ref: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CseHdGFlowNetTraceSource {
    pub id: String,
    pub state_ref: String,
    pub action: String,
    pub reward_payload_ref: String,
    pub lineage_hash: String,
    pub provenance_ref: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CseHdNrfFrameSource {
    pub id: String,
    pub timebase_ref: String,
    pub tile_kind: String,
    pub source_event_refs: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CseHdDrfStateSource {
    pub id: String,
    pub source_nrf_ref: String,
    pub source_cse_ref: String,
    pub channel_kind: String,
    pub tensor_shape: Vec<u64>,
    pub provenance_ref: String,
    pub canonical_or_derived: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CseHdComponentTables {
    pub schema_rows: Vec<CseHdSchemaRow>,
    pub unit_rows: Vec<CseHdUnitRow>,
    pub entity_rows: Vec<CseHdEntityRow>,
    pub atom_rows: Vec<CseHdAtomRow>,
    pub bond_rows: Vec<CseHdBondRow>,
    pub protein_chain_rows: Vec<CseHdProteinChainRow>,
    pub residue_rows: Vec<CseHdResidueRow>,
    pub atom_site_rows: Vec<CseHdAtomSiteRow>,
    pub water_site_rows: Vec<CseHdWaterSiteRow>,
    pub timebase_rows: Vec<CseHdTimebaseRow>,
    pub variant_rows: Vec<CseHdVariantRow>,
    pub event_rows: Vec<CseHdEventRow>,
    pub uncertainty_rows: Vec<CseHdUncertaintyRow>,
    pub thermo_rows: Vec<CseHdThermoRow>,
    pub teacher_output_rows: Vec<CseHdTeacherOutputRow>,
    pub gflownet_trace_rows: Vec<CseHdGFlowNetTraceRow>,
    pub nrf_frame_rows: Vec<CseHdNrfFrameRow>,
    pub drf_state_rows: Vec<CseHdDrfStateRow>,
    pub payload_blob_rows: Vec<CseHdPayloadBlobRow>,
    pub merkle_rows: Vec<CseHdMerkleRow>,
    pub provenance_rows: Vec<CseHdProvenanceRow>,
    pub seal_rows: Vec<CseHdSealRow>,
    pub nanl_symbol_rows: Vec<CseHdNanlSymbolRow>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CseHdSchemaRow {
    pub schema_version: String,
    pub state_id: String,
    pub classification: String,
    pub synthetic: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CseHdUnitRow {
    pub unit_id: String,
    pub quantity_kind: String,
    pub unit: String,
    pub unit_dict_ref: u32,
    pub quantity_dict_ref: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CseHdEntityRow {
    pub entity_id: String,
    pub entity_kind: String,
    pub topology_hash: String,
    pub provenance_ref: String,
    pub entity_dict_ref: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CseHdAtomRow {
    pub molecule_ref: String,
    pub atom_id: String,
    pub element: String,
    pub atom_type: String,
    pub atom_type_dict_ref: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CseHdBondRow {
    pub molecule_ref: String,
    pub bond_id: String,
    pub source_atom: String,
    pub target_atom: String,
    pub order: u8,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CseHdProteinChainRow {
    pub protein_ref: String,
    pub chain_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CseHdResidueRow {
    pub protein_ref: String,
    pub chain_ref: String,
    pub residue_id: String,
    pub name: String,
    pub residue_dict_ref: u32,
    pub sequence_index: u32,
    pub atom_site_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CseHdAtomSiteRow {
    pub protein_ref: String,
    pub site_id: String,
    pub atom_name: String,
    pub atom_type: String,
    pub atom_type_dict_ref: u32,
    pub chain_ref: String,
    pub residue_ref: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CseHdWaterSiteRow {
    pub water_id: String,
    pub gate_kind: String,
    pub water_gate_dict_ref: u32,
    pub event_bound: bool,
    pub time_ref: Option<String>,
    pub provenance_ref: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CseHdTimebaseRow {
    pub timebase_id: String,
    pub timebase_hash: String,
    pub unit_ref: String,
    pub frame_start: u64,
    pub frame_end: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CseHdVariantRow {
    pub variant_id: String,
    pub variant_dict_ref: u32,
    pub reference_state_ref: String,
    pub alternate_state_ref: String,
    pub delta_hash: String,
    pub provenance_ref: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CseHdEventRow {
    pub event_id: String,
    pub event_kind: String,
    pub event_kind_dict_ref: u32,
    pub source_ref: String,
    pub target_ref: String,
    pub time_ref: String,
    pub order_index: u64,
    pub motif: String,
    pub event_motif_dict_ref: u32,
    pub provenance_ref: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CseHdUncertaintyRow {
    pub uncertainty_id: String,
    pub target_ref: String,
    pub model: String,
    pub uncertainty_model_dict_ref: u32,
    pub calibration_hash: String,
    pub provenance_ref: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CseHdThermoRow {
    pub thermo_id: String,
    pub target_ref: String,
    pub quantity_kind: String,
    pub quantity_dict_ref: u32,
    pub unit_ref: String,
    pub unit_dict_ref: u32,
    pub value_payload_ref: String,
    pub uncertainty_ref: String,
    pub provenance_ref: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CseHdTeacherOutputRow {
    pub teacher_output_id: String,
    pub target_ref: String,
    pub model_ref: String,
    pub label: String,
    pub label_dict_ref: u32,
    pub confidence: f64,
    pub airlock_state: String,
    pub provenance_ref: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CseHdGFlowNetTraceRow {
    pub trace_id: String,
    pub state_ref: String,
    pub action: String,
    pub action_dict_ref: u32,
    pub reward_payload_ref: String,
    pub lineage_hash: String,
    pub provenance_ref: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CseHdNrfFrameRow {
    pub frame_id: String,
    pub timebase_ref: String,
    pub tile_kind: String,
    pub tile_dict_ref: u32,
    pub source_event_refs: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CseHdDrfStateRow {
    pub drf_state_id: String,
    pub source_nrf_ref: String,
    pub source_cse_ref: String,
    pub channel_kind: String,
    pub channel_dict_ref: u32,
    pub tensor_shape: Vec<u64>,
    pub provenance_ref: String,
    pub canonical_or_derived: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CseHdPayloadBlobRow {
    pub payload_id: String,
    pub payload_kind: String,
    pub payload_len: u64,
    pub payload_hash: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CseHdMerkleRow {
    pub merkle_id: String,
    pub root_hash: String,
    pub leaf_count: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CseHdProvenanceRow {
    pub provenance_id: String,
    pub source_uri: String,
    pub source_hash: String,
    pub classification: String,
    pub synthetic: bool,
    pub provenance_dict_ref: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CseHdSealRow {
    pub seal_id: String,
    pub previous_seal_hash: String,
    pub current_seal_hash: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CseHdNanlSymbolRow {
    pub symbol_id: u32,
    pub symbol: String,
}
