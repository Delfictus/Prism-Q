//! §4 — Core type system.
//!
//! The packed token is a single `u64` so the stream is cache-dense and the epistemic
//! bits (INV-2/3/6) travel with every token at zero overhead.
//!
//! Token layout (MSB→LSB):
//! ```text
//!  [63..58] leverage_tier : 6   epistemic ceiling class; 0 = source-of-truth
//!  [57]     verified      : 1   1 = seal integrity_verified; 0 = quarantined (INV-3/6)
//!  [56]     provenance    : 1   1 = sidecar entry present
//!  [55..43] namespace     : 13  ArtifactFamily × field-slot (INV-2)
//!  [42]     authenticated : 1   1 = keyed-signature verified ORIGIN (INV-9); default 0; NEVER
//!                               set from a hash result. verified ≠ authenticated.
//!  [41..00] field_token   : 42  quantizer codebook index OR opcode OR join-record id
//! ```

/// A packed PRISM token.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Token(pub u64);

impl Token {
    pub const LEVERAGE_BITS: u32 = 6;
    pub const NAMESPACE_BITS: u32 = 13;
    pub const FIELD_BITS: u32 = 42;

    const LEVERAGE_SHIFT: u32 = 58;
    const VERIFIED_SHIFT: u32 = 57;
    const PROVENANCE_SHIFT: u32 = 56;
    const NAMESPACE_SHIFT: u32 = 43;
    const AUTH_SHIFT: u32 = 42;

    pub const LEVERAGE_MASK: u64 = (1 << Self::LEVERAGE_BITS) - 1; // 0x3F
    pub const NAMESPACE_MASK: u64 = (1 << Self::NAMESPACE_BITS) - 1; // 0x1FFF
    pub const FIELD_MASK: u64 = (1 << Self::FIELD_BITS) - 1; // 42 ones
    /// INV-7 escape code: the field_token value used when incurred distortion exceeds the field's
    /// declared bound. The exact value lives in the sidecar; quantizer codes never reach this
    /// reserved top value for any realistic budget (≤ ~32 bits), so there is no collision.
    pub const ESCAPE_CODE: u64 = Self::FIELD_MASK;

    /// Build a packed token. Inputs that overflow their field width are a programming
    /// error (debug_assert) and are masked in release so a token is always well-formed.
    #[inline]
    pub fn new(
        leverage_tier: u8,
        verified: bool,
        provenance: bool,
        namespace: u16,
        field_token: u64,
    ) -> Self {
        debug_assert!(
            (leverage_tier as u64) <= Self::LEVERAGE_MASK,
            "leverage_tier {leverage_tier} overflows 6 bits"
        );
        debug_assert!(
            (namespace as u64) <= Self::NAMESPACE_MASK,
            "namespace {namespace} overflows 13 bits"
        );
        debug_assert!(
            field_token <= Self::FIELD_MASK,
            "field_token {field_token} overflows 42 bits"
        );
        let v = ((leverage_tier as u64 & Self::LEVERAGE_MASK) << Self::LEVERAGE_SHIFT)
            | ((verified as u64) << Self::VERIFIED_SHIFT)
            | ((provenance as u64) << Self::PROVENANCE_SHIFT)
            | ((namespace as u64 & Self::NAMESPACE_MASK) << Self::NAMESPACE_SHIFT)
            | (field_token & Self::FIELD_MASK);
        Token(v)
    }

    #[inline]
    pub fn leverage_tier(self) -> u8 {
        ((self.0 >> Self::LEVERAGE_SHIFT) & Self::LEVERAGE_MASK) as u8
    }
    /// INV-3/6: seal integrity verified. A `false` token is quarantined and must never
    /// reach a canonical path.
    #[inline]
    pub fn verified(self) -> bool {
        (self.0 >> Self::VERIFIED_SHIFT) & 1 == 1
    }
    /// INV-4: a sidecar entry holds this token's exact decoded value + originating seal sha.
    #[inline]
    pub fn provenance(self) -> bool {
        (self.0 >> Self::PROVENANCE_SHIFT) & 1 == 1
    }
    #[inline]
    pub fn namespace(self) -> u16 {
        ((self.0 >> Self::NAMESPACE_SHIFT) & Self::NAMESPACE_MASK) as u16
    }
    #[inline]
    pub fn field_token(self) -> u64 {
        self.0 & Self::FIELD_MASK
    }

    /// INV-9: keyed-signature-verified ORIGIN. Distinct from `verified` (integrity). Default 0;
    /// only `with_authenticated` (driven by a keyed-signature check) may set it — never a hash.
    #[inline]
    pub fn authenticated(self) -> bool {
        (self.0 >> Self::AUTH_SHIFT) & 1 == 1
    }

    /// Return a copy with the authenticated bit set/cleared. The ONLY way to set it; callers must
    /// gate this on a keyed-signature result (INV-9 / §7 A-DISTINCT), never on a hash.
    #[inline]
    pub fn with_authenticated(self, authenticated: bool) -> Token {
        let cleared = self.0 & !(1u64 << Self::AUTH_SHIFT);
        Token(cleared | ((authenticated as u64) << Self::AUTH_SHIFT))
    }
}

impl core::fmt::Debug for Token {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Token")
            .field("leverage_tier", &self.leverage_tier())
            .field("verified", &self.verified())
            .field("provenance", &self.provenance())
            .field("namespace", &self.namespace())
            .field("field_token", &self.field_token())
            .finish()
    }
}

// --- Factored namespace (INV-2): namespace = (family_slot, field_slot) --------------
pub const FAMILY_SLOT_BITS: u32 = 7;
pub const FIELD_SLOT_BITS: u32 = 6;
pub const FAMILY_SLOT_MASK: u16 = (1 << FAMILY_SLOT_BITS) - 1; // 0x7F
pub const FIELD_SLOT_MASK: u16 = (1 << FIELD_SLOT_BITS) - 1; // 0x3F

/// Pack a `(family_slot, field_slot)` pair into the 13-bit namespace. Same numeric value
/// in two families therefore yields two distinct tokens (INV-2).
#[inline]
pub fn pack_namespace(family_slot: u8, field_slot: u8) -> u16 {
    (((family_slot as u16) & FAMILY_SLOT_MASK) << FIELD_SLOT_BITS)
        | ((field_slot as u16) & FIELD_SLOT_MASK)
}

#[inline]
pub fn unpack_namespace(ns: u16) -> (u8, u8) {
    (
        ((ns >> FIELD_SLOT_BITS) & FAMILY_SLOT_MASK) as u8,
        (ns & FIELD_SLOT_MASK) as u8,
    )
}

/// Artifact families (A1–A14 token classes + the engine-SOURCE generators + fallbacks).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u16)]
pub enum ArtifactFamily {
    SpikeEvent = 0,
    NanlSpikePacket,
    MsmKineticSummary,
    TeacherStudentConsensus,
    FiberBundleDstw,
    BocpdEvent,
    GflownetTrajectory,
    QnanlArtifact,
    GpuTelemetry,
    ReceiptStatus,
    HoloBoundState,
    CapabilityGate,
    TwinCoupled,
    MoleculeGeneration,
    FalsifiablePrediction,
    RankerRow,
    KccDescriptor,
    SrcCuda,
    SrcEngine,
    SrcCodec,
    SrcTwin,
    SrcGraph,
    SrcGflownet,
    /// Manifest-gap / unknown family fallback. INV-1 preserved via full sidecar.
    Opaque,
    /// KCC consensus SITE (`kcc_visualization.json` `sites[]`): a RELATION over residues plus
    /// spatial / temporal / causal / rank structure — NOT a widened residue row, and NOT
    /// `binding_sites.json` (different schema). Distinct namespace from [`Self::KccDescriptor`].
    /// Appended last so every existing family keeps its discriminant (= namespace slot).
    KccSite,
    /// SPIKE_TRANSITION_EVENT (engine writer `nhs_rt_full.rs` ~7616): a protocol/phase transition
    /// view-anchor over the append-only spike stream. Distinct from [`Self::BocpdEvent`] (the
    /// `bocpd.jsonl` per-frame BOCPD log is a DIFFERENT artifact). Appended last to preserve every
    /// existing family discriminant.
    SpikeTransitionEvent,
    /// `bocpd.jsonl` per-frame BOCPD changepoint LOG (12 fields: frame_idx/spike_delta/observation/
    /// posterior_max/map_run_length/recent_p/…). Explicitly SEPARATE from
    /// [`Self::SpikeTransitionEvent`] and [`Self::BocpdEvent`]. Appended last (discriminant-stable).
    BocpdLog,
    /// `protocol_state.json` per-stream control/protocol state (temperature/phase/step/gamma/dt/…).
    /// Control evidence, not a measurement. Appended last (discriminant-stable).
    ProtocolState,
    /// `noise_floor.json` per-stream QC calibration (target/n_samples/phase + mu/sigma/samples
    /// arrays). Appended last (discriminant-stable).
    NoiseFloor,
    /// `TWIN16_INTERFEROMETRIC_TRANSITION_VIEW_RESIDUALS.json`: 16 per-cell bounded thermo/temporal
    /// CCNS/SDST aggregates + cell-pair `transition_view_l1_residual` values. This is the
    /// interferometric transition-view surface — DISTINCT from a pairwise residue_i↔residue_j TE
    /// matrix. Appended last (discriminant-stable).
    TwinTransitionResidual,
    /// `CONDITION_EDGE_READOUT/per_cellpair_condition_edges.jsonl`: the TWIN16-native
    /// DirectedPerturbationDifferential edges (cell_a→cell_b, axis, `transition_view_l1_residual`,
    /// hamming, material). DPD substrate = the 2×2×2×2 thermodynamic factorial. Appended last.
    TwinConditionEdge,
    /// AUXILIARY coupled-A/B NMA perturbation differential — the `--coupled-twin` path's
    /// `CONDITION_EDGE_READOUT/per_residue_condition_edges.jsonl` (thermal_only vs thermal_plus_nma,
    /// per residue: b_over_a_ratio, nma/thermal_exclusive_count, barrier, edge_score). A SEPARATE
    /// evidence class from the canonical TWIN16 `[[TwinConditionEdge]]` — NMA is NOT a TWIN16 axis,
    /// and this must never be relabelled as / merged into TWIN16. Appended last.
    NmaCoupledEdge,
    /// DSTW-T4 scientific-state VERTEX (`crates/prismcode/src/dstw_state_graph.rs`
    /// `DSTW_T4_VERTEX_SCHEMA_KIND`): one lifecycle state of an MD evidence run — pre_md / post_md
    /// apo vertices (runtime_origin RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS), the BLOCKED holo_gdp
    /// bound-state vertex (runtime_origin BLOCKED), and the conditional EXTERNAL_DERIVED downstream
    /// (post_pathb / post_tokenizer) vertices. Carries the ligand_state (state/feasibility/
    /// gdp_present/VALUE-verified input_sha256/blocker), detection posture, per-tensor feasibility,
    /// QNANL receipts, and the MANDATORY honest `runtime_origin`. Appended last (discriminant-stable).
    DstwStateVertex,
    /// DSTW-T4 scientific-state TRANSITION (`DSTW_T4_TRANSITION_SCHEMA_KIND`): one operator applied
    /// between two vertices — the MD.INTEGRATE_VELOCITY_VERLET integration edge (RUNTIME_SUBPROCESS),
    /// the BLOCKED STATE.CLASSIFY_BLOCKED GDP-forcefield precondition edge, and the EXTERNAL_DERIVED
    /// downstream edges. Carries changed/unchanged/failed-tensor counts, blocked_preconditions,
    /// opcode-COUNT aggregates (never null-padded operands), and `runtime_origin`. Appended last.
    DstwStateTransition,
    /// DSTW-T4 NEXT-OPERATOR recommendation (`DSTW_T4_NEXT_OPERATOR_SCHEMA_KIND`): the rank-ordered
    /// engine-owner work list prismos authors in-process (runtime_origin RUNTIME_INLINE). Advisory
    /// guidance (NOT an observable) — hence a higher epistemic ceiling. Appended last.
    DstwNextOperator,
    /// DSTW-T4 Step h5 DENSE-TENSOR SEMANTIC SUMMARY (`crates/prismcode/src/dstw_state_graph.rs`
    /// `DSTW_T4_DENSE_TENSOR_SUMMARY_SCHEMA_KIND`): a POST_RUN_DERIVED semantic record over ONE
    /// dense engine tensor (signal_grid / warp / DSTW adaptive_dt / superresolution / TWIN16 /
    /// neuromorphic spike sensor). Dense tensors must NOT enter training raw — this record carries
    /// the tensor's identity + reduced semantics (tensor_role, modality, real `source_hash` sha256,
    /// `on_disk_bytes`, decoded-summary `row_count`/`column_count` dimensions, finite/NaN/Inf checks,
    /// `stream_id`, `parent_vertex_id`, downstream `tokenizer_family`/`source_kind`) so the raw bytes
    /// never reach the model. Dimension/finite fields derive ONLY from a decoded reducer summary
    /// (dispatch_analysis_common SignalGridSummary/WarpSummary) — never from raw tensor bytes; when no
    /// decoded summary exists those fields carry an explicit absent_reason (never null-padded). The
    /// inline-at-teardown native reduction is engine-owner; until then origin=POST_RUN_DERIVED.
    /// Appended last (discriminant-stable).
    DenseTensorSemanticSummary,
    /// PRISMOS-authored OPCODE/OPERAND EXECUTION LEDGER (`opcode_operand_ledger.json`,
    /// `schema_kind == "prismcode_typed_operand_ledger"`): one row per executed ISA opcode
    /// (`MD.*`) or kernel-dispatch label (`KERNEL::*`) or post-run QNANL seal (`MD.QNANL_SEAL`),
    /// with a by-value tie — `count` equals the source `*_md_native_receipt.json`
    /// `opcode_counts[name]` (MD.*) / `kernel_counts[name[8:]]` (KERNEL::) — plus the typed operand
    /// posture (operator_class Md/Kcc/Qnanl, operand_target Tensor/Artifact, runtime_origin
    /// RuntimeSubprocessOrchestratedByPrismos/PostRunSealOnly, disassembly/validation/replay status)
    /// and provenance join keys (source_receipt, and for seal rows input/output/qnanl_packet sha256).
    /// The nested `value_proof` + seal-only fields escape losslessly to the sidecar (INV-4).
    /// Appended last (discriminant 36, discriminant-stable).
    OpcodeExecutionLedger,
    /// `prismos_prismcode_isa_execution_verification.json` (schema_kind
    /// `prismos_prismcode_isa_execution_verification`, emitter prismos.rs
    /// `build_prismos_isa_execution_verification`): the AGGREGATE MD-ISA execution VERDICT over one
    /// native-run row — the top-level object carries status/substrate_status/
    /// implementation_status_language enums, the `native_md_receipt_count == len(streams)` aggregate,
    /// the all_streams_pass/all_streams_bounded_or_passed/cuda_opcode_coverage_exists/
    /// hotpath_acceleration_required/metal_resident_md_required verdict booleans,
    /// min_hotpath_launch_collapse, and the plan_hash_blake3 join key. The data-dense `streams[]`
    /// (per-stream dispatch counts + opcode observations) descends losslessly to the sidecar (INV-1).
    /// Fields DIFFER from [`Self::ReceiptStatus`] (status/structural_verified only) — its OWN family,
    /// NEVER relabelled ReceiptStatus. Appended last (discriminant 37, discriminant-stable; slot ≤ 63).
    ValidationState,
    /// CAPTURED-GRAPH GPU TELEMETRY — Path-B ghost temporal/lattice-routing evidence. TWO carriers
    /// fold into this one family: (1) `ghost_time_map.json` (schema_kind `pathb_ghost_time_map`) — a
    /// wrapper whose `streams[]` array carries one per-multi-stream-lane captured-graph temporal frame
    /// (dt_fs / timestep_min / timestep_max / frame_idx_at_teardown / n_spikes / physical_time_fs /
    /// physical_time_at_min_fs / stream_id / gear_id), routed by descending `/streams`; and (2)
    /// `ghost_lattice_routing_status.json` — a FLAT status object that IS one record (n_components /
    /// lattice_kernel_ms / v2_was_live / v2_capture_requested / auto_request / explicit_request /
    /// m1_monolithic_discovery / mar_v2_telemetry). DISTINCT from [`Self::GpuTelemetry`] (whose specs
    /// are GFlowNet-training receipt scalars cuda_graph_replay_ms / tb_loss / gpu_solve_count — field
    /// names DISJOINT from every ghost key, so routing ghost artifacts there gives matched_fields=0).
    /// Its OWN namespace; never relabelled GpuTelemetry. Appended last (discriminant 38, slot ≤ 63).
    CapturedGraphGpuTelemetry,
    /// PATH-B FLAGSHIP materialized binding SITE (`pathb_runtime_stage/binding_sites.materialized.json`,
    /// `schema_kind == "pathb_binding_sites_materialized"`): one record per PROMOTED site in the
    /// wrapper's `binding_sites[]` array (site_count=N). Each per-site record carries the
    /// lexicographic materialization scalars (site_id / rank / new_prism_rank / old_density_rank /
    /// n_spikes / n_raw_peaks_merged / n_active_non_density_dimensions / intensity_sum /
    /// centroid_spread_a / region_radius_a / promoted / materialization_status /
    /// materialization_level) plus the lining/driver/representative residue arrays + centroid_xyz
    /// (residue arrays escape LOSSLESS to the §10 sidecar; the by-value INVARIANT cross-checks them).
    /// DISTINCT from [`Self::KccSite`] (the `kcc_visualization.json` `sites[]` schema keyed on
    /// gtck_rank+rank_C+kcc / gtck_rank+rank_score+site_local_cov+centroid) — this schema has NO
    /// gtck_rank/kcc and MUST NOT be relabelled KccSite; and NOT the string `binding_sites_path`
    /// join field attached to the KccSite manifest entry (that never routes the 20 per-site records).
    /// By-value descent route (route each per-site object, not a filename wrap). Appended last
    /// (discriminant 39, slot ≤ 63; every existing discriminant/namespace stays stable).
    PathBBindingSiteMaterialized,
    /// DSTW-T4 merge-manifest lineage DAG carrier (PRISM_NATIVE_STATE_GRAPH.jsonl; emitter
    /// crates/prismcode/src/merge_manifest.rs fn emit_state_graph_jsonl). One heterogeneous
    /// `state_graph_node` envelope record per emitted artifact vertex/edge + a final
    /// `orphan_check` summary row. Routed by-value as jsonl_records (one record per line) with
    /// the parent→child referential-integrity invariant (every non-null source/target/parent
    /// vertex id resolves to an existing vertex_id). Appended last (discriminant 40, slot ≤ 63;
    /// every existing discriminant/namespace stays stable). Distinct from DstwStateVertex/
    /// DstwStateTransition which own the FLAT-PROJECTED standalone *_state_vertex.json artifacts.
    LineageDag,
    /// ENGINE-authored bounded thermo/temporal CCNS/SDST AGGREGATE SUMMARY
    /// (`bounded_thermo_temporal_ccns_sdst_summary.json`, `schema_kind ==
    /// "bounded_thermo_temporal_ccns_sdst_summary"`, emitter `nhs_rt_full.rs` ~8250): ONE top-level
    /// aggregate record over the SPIKE_TRANSITION_EVENT always-on view anchors of a whole target.
    /// The by-value payload — event_count, per-phase CCNS/SDST fractions, phase-bucket / trigger
    /// counts — lives in NESTED rollup dicts (ccns/sdst/phase_bucket_counts/trigger_counts) that
    /// `route_object` treats as opaque; a flatten helper lifts them to flat scalar keys so they
    /// value-match, while the un-lifted phase_edge_counts/temporal_extent/condition_ledger/cells[]/
    /// event_sources[] (with sha256) ride the lossless sidecar (INV-1/INV-4). BY-VALUE invariant:
    /// event_count == Σ cells[].event_count == Σ event_sources[].event_count == Σ phase_bucket_counts.
    /// DISTINCT from [`Self::TwinTransitionResidual`] (the TWIN16 interferometric RESIDUALS file whose
    /// `/cells` descent DROPS this summary-level rollup) — its OWN namespace, never relabelled.
    /// Appended last (discriminant 41, slot ≤ 63; every existing discriminant/namespace stays stable).
    ThermoPhaseManifold,
    /// PRISMOS-authored ISA REGISTRY SUMMARY — a POST_RUN_DERIVED flat cross-product of the MD-ISA
    /// execution PLAN (`prismos_prismcode_isa_execution_plan.json`, the bare-string ISA registry) ×
    /// the execution VERIFICATION (`..._isa_execution_verification.json`, the per-stream nested
    /// `known_md_prismop_observation[]`). The two RAW artifacts cannot be routed by value at opcode
    /// granularity as-is: the plan's registry is bare STRING ARRAYS (route_object needs object
    /// records) and the verification's opcode observations sit TWO levels deep
    /// (`streams[].known_md_prismop_observation[]`) while `resolve_path` descends only ONE `/key`
    /// level. The DERIVED wrapper `prismos_prismcode_isa_registry_summary.json`
    /// (`schema_kind == "prismos_prismcode_isa_registry_summary"`, emitter prismos.rs
    /// `build_prismos_isa_registry_summary`) exposes a top-level `records[]` FLAT ARRAY of one
    /// per-(stream,opcode) observation each = {stream_id, opcode, opcode_index, count, observed,
    /// planned, required_core, registry_size, required_core_size, native_md_receipt_count,
    /// stream_status, stream_coverage_status}. Routed by descending `/records` (each record is a
    /// genuine by-value object). BY-VALUE invariants: per stream distinct(opcode)==count(records)==
    /// registry_size==len(md_prismop_known_instruction_set); every required_core opcode has
    /// observed==true; distinct(stream_id)==native_md_receipt_count. `opcode` tokenizes as a genuine
    /// categorical (registered ISA_MD_OPCODES vocab), NOT a filename/family label. DISTINCT from
    /// [`Self::ValidationState`] (the aggregate top-level verdict object) and
    /// [`Self::OpcodeExecutionLedger`] (the per-executed-opcode ledger keyed on count-vs-receipt) —
    /// this is the registry×receipt coverage matrix, its OWN namespace. Appended last
    /// (discriminant 42, slot ≤ 63; every existing discriminant/namespace stays stable).
    IsaRegistrySummary,
    /// RAW PRKCC001 `kcc_v2full.bin` per-residue KCC descriptor carrier (schema_kind
    /// `prismos_kcc_descriptor_native_summary`, decoder prismos.rs `decode_prkcc001_native_records`).
    /// The self-describing PRKCC001 envelope holds 14 named f32/u32 columns (temporal_corr,
    /// direction_score, motion_efficiency, burst_motion, phase_shift, causal_lag, lag_corr_peak,
    /// local_cov, net_dx/dy/dz, sum_m, residue_count, active_causal); the decoder TRANSPOSES them into
    /// one per-residue record (+ synthesized residue_id/stream_id JoinKeys) and routes by descending
    /// `/records`. DISTINCT from [`Self::KccDescriptor`], which is bound to the
    /// `kcc_visualization.json residues[]` JSON VIEW (a DIFFERENT field set: residue_name/ca_position/
    /// kcc_score/total_steps + a `vector_field_definition` detector). This is the by-value binary
    /// closure the DenseTensorSemanticSummary min/max/mean reduction cannot provide. BY-VALUE
    /// invariants: len(records)==n_residues(payload)==record_count(envelope)==topology n_residues;
    /// 0<=local_cov,direction_score<=1; burst_motion>=0; motion_efficiency>0; -1<=temporal_corr,
    /// phase_shift<=1; residue_count constant (coherence window); active_causal<=residue_count;
    /// causal_lag/lag_corr_peak == null in md-only mode (NaN sentinel). Appended last (discriminant 43,
    /// slot ≤ 63; every existing discriminant/namespace stays stable).
    KccDescriptorNative,
    /// RAW PRSPK001 `_spikes.bin` per-residue BOUNDED spike aggregate (schema_kind
    /// `prismos_spike_event_native_summary`, decoder prismos.rs `decode_prspk001_spike_summary`).
    /// The PRSPK001 payload holds 96 B `GpuSpikeEvent` records, the width the envelope's
    /// `byte_stride` declares. A reader MUST validate the declared stride and refuse any
    /// other layout; emitting them raw would dwarf the corpus, so the decoder parses EVERY record BY VALUE and
    /// aggregates by the primary nearby residue (`nearby_residues[0]`, 0-indexed into the topology)
    /// into ONE record per residue (~146 for 4LPK_apo stream0). Each record = {residue_id, stream_id,
    /// spike_count, spike_rate, mean_amplitude, max_amplitude, first_frame, last_frame, aromatic_type}
    /// routed by descending `/records`. This is NOT [`Self::DenseTensorSemanticSummary`] (that path is
    /// a min/max/mean tensor reduction): it is a typed per-residue by-value surface whose COUNT
    /// CONSERVATION invariant proves every raw spike was decoded. DISTINCT from [`Self::SpikeEvent`]
    /// (the per-spike `spike_events.json` JSON view / raw per-spike carrier) — this is the bounded
    /// per-residue rollup, its OWN namespace. BY-VALUE invariants: Σ records.spike_count ==
    /// PRSPK001 header record_count == spikes.prismnanl.receipt.json.record_count (1,394,939 on
    /// stream0); every residue_id in [0, n_residues-1]; spike_count >= 1; first_frame <= last_frame;
    /// aromatic_type in {-1(none),0=TRP,1=TYR,2=PHE,3=S-S,4=BNZ,5,6}. Appended last (discriminant 44,
    /// slot ≤ 63; every existing discriminant/namespace stays stable).
    SpikeSummary,
    /// PRISMOS-authored PREDICTION STATE — a FIRST-CLASS prediction record per PROMOTED materialized
    /// binding site (`schema_kind == "prismos_prediction_state"`, emitter prismos.rs
    /// `build_prismos_prediction_state`, top-level `records[]` flat array of one record per site;
    /// 20 records for 4LPK_apo). Operator correction 2026-07-01: a PREDICTION STATE (a hypothesis
    /// ABOUT a binding site with by-value confidence/uncertainty + evidence lineage) is a DISTINCT
    /// epistemic object from [`Self::DstwNextOperator`] (a rank-ordered engine-owner WORK LIST) —
    /// this family SUPERSEDES the earlier DstwNextOperator-alias closure and NEVER aliases it (the
    /// two route independently: PredictionState keys on `schema_kind=="prismos_prediction_state"`,
    /// DstwNextOperator on the next_operator_recommendation carrier). 18 fields, each mapped to a
    /// REAL by-value source (no placeholders): prediction_id/target_id/predicted_surface/
    /// source_vertex_id/target_vertex_id/evidence_parent_ids/operator_recommendation_id/
    /// qnanl_receipt_ref JoinKeys + predicted_delta (== new_prism_rank − old_density_rank per site,
    /// binding_sites.materialized) + confidence (∈[0,1], REAL manifold_consistency.
    /// manifold_consistency_score) + uncertainty (∈[0,1], REAL centroid_spread_a normalized) +
    /// receptor_state/ligand_state/epistemic_status/validation_status/falsification_status/
    /// runtime_origin/tokenizer_family enums. BY-VALUE invariants: n_records ==
    /// site_accuracy_report.n_materialized_sites == len(binding_sites.materialized) (20); confidence
    /// AND uncertainty are REAL decoded numerics that VARY across records (never a constant);
    /// predicted_delta == new_prism_rank − old_density_rank; every evidence_parent_ids/source_vertex_id/
    /// target_vertex_id resolves to a PRISM_NATIVE_STATE_GRAPH.jsonl vertex_id. DISTINCT from
    /// [`Self::PathBBindingSiteMaterialized`] (the raw materialized-site descriptor) and
    /// [`Self::FalsifiablePrediction`] — its OWN namespace. Appended last (discriminant 45, slot ≤ 63;
    /// every existing discriminant/namespace stays stable).
    PredictionState,
    /// GhostPhaseLattice4D PHASE-MANIFOLD EDGE-VALIDATION manifest (`{stem}_binding_sites_lattice.json`,
    /// engine emitter `nhs_rt_full.rs` ~16336, only when `v2_was_live==true` AND the lattice
    /// `cluster()` succeeds): the top-level manifold object over the captured-graph Ghost v2 tiles —
    /// `{schema_version, backend:"ghost_phase_lattice_4d", config:{so3_threshold/spatial_cell_size_a/
    /// max_temporal_edge_steps/step_bucket_size}, stats:{n_nodes/n_lattice_cells/n_pairs_evaluated/
    /// n_pairs_phase_legal/n_pairs_aabb_overlap/n_directed_edges/min_step_idx/max_step_idx/
    /// kernel_elapsed_ms/host_elapsed_ms}, components:[N × {component_id, n_nodes, so3_manifold:
    /// {intra_component_mean_cosine, n_intra_edges, plane_status}, phase_manifold, therm_ccns_lifecycle,
    /// component_aabb_min/max}]}`. Routed as ONE top-level record (pointer "" — the object IS the
    /// record; the data-dense `components[]`/nested blocks descend LOSSLESS to the §10 sidecar). The
    /// exporter's `flatten_phase_manifold_edge_validation` LIFTS the nested `stats.*` + `config.*`
    /// scalars to flat keys AND derives component aggregates (`n_components`, `sum_component_n_nodes`,
    /// `total_intra_edges`, `mean/min/max_intra_component_mean_cosine`) so `route_object` value-matches
    /// the real edge counts + real SO(3) coherence (matched_fields>0); routing the raw wrapper would
    /// make every top-level key an opaque escape. BY-VALUE invariants (recomputed against the artifact):
    /// Σ components[].n_nodes == stats.n_nodes; Σ components[].so3_manifold.n_intra_edges ==
    /// stats.n_directed_edges (edge-count-consistency); every intra_component_mean_cosine ∈ [-1,1] (real
    /// phase/coherence); n_pairs_phase_legal ≤ n_pairs_evaluated; n_pairs_aabb_overlap ≤
    /// n_pairs_evaluated; n_directed_edges ≤ n_pairs_aabb_overlap; min_step_idx ≤ max_step_idx;
    /// len(components) == n_components. DISTINCT from [`Self::CapturedGraphGpuTelemetry`] (the FLAT
    /// `ghost_lattice_routing_status.json` STATUS object keyed on `backend_request`/`v2_was_live` — no
    /// `components[]`, no `stats`) — this is the manifold EDGE-VALIDATION manifest, its OWN namespace,
    /// NEVER relabelled. Appended last (discriminant 46, slot ≤ 63; every existing discriminant stays
    /// stable).
    PhaseManifoldEdgeValidation,
    /// PRISMOS scientific chronology EVENT (`scientific_chronology_events.json`,
    /// wrapper `schema_kind == "prismos_scientific_chronology_events"`, per-record
    /// `schema_kind == "prismos_scientific_chronology_event"`). One first-class chronological event
    /// row keyed by chronology_event_id/epoch_id/event_index, with vertex joins, time bounds,
    /// operator/opcode identity, QNANL/operand refs, state hash, runtime origin/native flag,
    /// perturbation/twin refs, and spike/BOCPD event counts. Appended last (discriminant 47).
    ScientificChronologyEvent,
    /// PRISMOS scientific state EPOCH (`scientific_state_epochs.json`,
    /// wrapper `schema_kind == "prismos_scientific_state_epochs"`, per-record
    /// `schema_kind == "prismos_scientific_state_epoch"`). One first-class state interval keyed by
    /// epoch_id/epoch_index, target and vertex joins, start/end event ids, lifecycle/ligand state,
    /// event/stream counts, timestep and physical-time interval, QNANL/operand refs, state hash,
    /// runtime origin/native flag, closed interval flag, and perturbation/twin refs. Appended last
    /// (discriminant 48).
    ScientificStateEpoch,
    /// DSTW-T4 epoch overlay closeout (`dstw_t4_epoch_overlay.json`,
    /// `schema_kind == "dstw_t4_epoch_overlay"`). A top-level runtime closeout record that binds a
    /// state epoch to the emitted DSTW vertices/transitions, event interval, QNANL/operand refs, and
    /// runtime-origin/native evidence. Appended last (discriminant 49).
    T4EpochOverlay,
    /// Chronology acceptance decision closeout (`chronology_acceptance_decision.json`,
    /// `schema_kind == "dstw_t4_chronology_acceptance_decision"`). A top-level gate result record
    /// keyed by acceptance_decision_id/epoch_overlay_id/epoch_id, with artifact/decode/operand counts,
    /// decision enum, runtime origin, and QNANL/operand refs. Appended last (discriminant 50).
    ChronologyAcceptanceDecision,
    /// Operator-scoped ScientificPocketOpeningEvent. Exact external schema fields are preserved:
    /// epoch_id, frame_event_id, residue_ids, pocket_volume, pocket_surface_area, lining_contacts,
    /// distance_gates, persistence_frames, apo_holo_contrast, confidence. Appended last
    /// (discriminant 51).
    ScientificPocketOpeningEvent,
    /// Operator-scoped LigandCausalActivityEvent. Exact external schema fields are preserved; nested
    /// contacts[] and causal_signals{} children route through documented source-schema aliases.
    /// Appended last (discriminant 52).
    LigandCausalActivityEvent,
    /// Operator-scoped ResidueActivityEvent. Exact external schema fields are preserved; metrics{}
    /// children route through documented source-schema aliases. Appended last (discriminant 53).
    ResidueActivityEvent,
    /// PRISMOS run-specific spectral geometry closure. This is the first-class tokenizer lane for
    /// the multi-step spectral extractor output (`DSTW_RUN_SPECIFIC_SPECTRAL_CLOSURE_MANIFEST.json`):
    /// frame-derived contact graph, physical profile, Laplacian spectrum, site-rank overlay, edge
    /// validation, and TWIN16 residual linkage. It is distinct from [`Self::FiberBundleDstw`] (legacy
    /// fiber telemetry) and from the three causal-event families (operator event rows).
    /// Appended last (discriminant 54).
    SpectralExtractor,
    /// PRISM-native measured scientific uncertainty aggregate tensor. The broader V2 uncertainty
    /// ontology is represented through this family plus the specialized tensor families below.
    /// Appended last (discriminant 55).
    ScientificUncertaintyTensor,
    /// Runtime numerical uncertainty tensor derived from runtime noise-floor / numerical residual
    /// observations. Appended last (discriminant 56).
    RuntimeNumericalUncertaintyTensor,
    /// Thermal uncertainty tensor derived from measured thermal / protocol-state variation.
    /// Appended last (discriminant 57).
    ThermalUncertaintyTensor,
    /// Integrator uncertainty tensor derived from integrator residual, drift, and constraint
    /// residual observations. Appended last (discriminant 58).
    IntegratorUncertaintyTensor,
    /// Observable uncertainty tensor for coordinate, force, contact, pocket, spectral, KCC, spike,
    /// and causal-geometry observables. Appended last (discriminant 59).
    ObservableUncertaintyTensor,
    /// Signal-to-uncertainty tensor that maps scientific signal deltas to propagated uncertainty.
    /// Appended last (discriminant 60).
    SignalToUncertaintyTensor,
    /// Epoch-level uncertainty tensor binding uncertainty back to ScientificStateEpoch /
    /// ScientificChronologyEvent objects. Appended last (discriminant 61).
    EpochUncertaintyTensor,
    /// Multiplexed adversarial scientific falsification lane. Required rejection object names are
    /// preserved as typed values (`negative_evidence_type`, `rejection_class`, `control_family`,
    /// `failed_claim_type`, `motif_status`, `rejection_reason`) instead of consuming one family slot
    /// per rejection object. Appended last (discriminant 62).
    NegativeScientificEvidenceEvent,
    /// Runtime-native TSH salience event emitted by the observational Stage-II observer.
    /// Appended last (discriminant 63).
    ThermodynamicSalienceEvent,
    /// Runtime-native TSH transition motif descriptor. TransitionMotifCandidate rows route here
    /// with schema_kind preserving candidate identity. Appended last (discriminant 64).
    TransitionMotif,
    /// Runtime-native TSH persistent transition object accepted after replicate consensus.
    /// Acceptance remains partial pending negative-control-backed promotion. Appended last
    /// (discriminant 65).
    PersistentTransitionObject,
    /// Runtime-native TSH consensus event joining motif candidates across seeds. Appended last
    /// (discriminant 66).
    TransitionConsensusEvent,
    /// Runtime-native TSH transition evidence ledger, non-interference ledger, and cache-audit rows.
    /// Appended last (discriminant 67).
    TransitionEvidenceLedger,
    /// Corpus-level TSH falsification consensus adjudication event. This is not runtime steering:
    /// it binds runtime motifs to positive evidence, V6 canonical negative evidence, uncertainty,
    /// topology, chronology, spectral recurrence, and consensus/falsification decisions.
    /// Appended last (discriminant 68).
    ScientificTransitionConsensusEvent,
    /// Corpus-level profile explaining how one transition motif intersected the canonical negative
    /// evidence bundle. Appended last (discriminant 69).
    TransitionFalsificationProfile,
    /// Rejected persistent-transition candidate preserved as first-class trainable negative
    /// decision state. Appended last (discriminant 70).
    RejectedPersistentTransitionCandidate,
    /// Global acceptance/rejection ledger for the TSH falsification consensus gate. Appended last
    /// (discriminant 71).
    PersistentTransitionAcceptanceLedger,
    /// Native packed epistemic uncertainty lane for training-time sigma/variance values. This
    /// keeps epistemic scalars in the same u64 stream as physics observations instead of rendering
    /// string tokens such as `[SIGMA_BIN_X]`. Appended last (discriminant 72).
    EpistemicUncertainty,
    /// Native packed teacher-consensus lane for entropy/disagreement/vote-state values. Distinct
    /// from [`Self::TeacherStudentConsensus`], which is a source artifact family; this family is the
    /// compact causal-sequence token lane used by the PLLM dataloader. Appended last
    /// (discriminant 73).
    TeacherConsensus,
    /// Native packed thermodynamic action lane for TSH state opcodes and time-delta values.
    /// Appended last (discriminant 74).
    ThermodynamicAction,
}

impl ArtifactFamily {
    /// Stable 7-bit family slot for the factored namespace.
    #[inline]
    pub fn family_slot(self) -> u8 {
        debug_assert!(
            (self as u16) <= FAMILY_SLOT_MASK,
            "family_slot overflows 7 bits"
        );
        self as u16 as u8
    }
}

/// Field slots for the native epistemic uncertainty lane.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum EpistemicUncertaintyField {
    Sigma = 0,
    Variance = 1,
}

/// Field slots for the compact teacher-consensus lane.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum TeacherConsensusField {
    Entropy = 0,
    Disagreement = 1,
    VoteState = 2,
}

/// Field slots for the native thermodynamic action lane.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum ThermodynamicActionField {
    TshState = 0,
    DeltaTimeFs = 1,
}

/// Compact TSH opcodes stored directly in `field_token`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u64)]
pub enum ThermodynamicActionOpcode {
    Stable = 1,
    SalienceRise = 2,
    SalienceFall = 3,
    Hysteresis = 4,
    Boundary = 5,
    Handoff = 6,
}

#[inline]
pub fn artifact_namespace(family: ArtifactFamily, field_slot: u8) -> u16 {
    pack_namespace(family.family_slot(), field_slot)
}

/// Quantize a bounded scalar into the field payload while reserving ESCAPE_CODE for non-finite
/// values or invalid bounds.
#[inline]
pub fn quantize_bounded_field(value: f64, lo: f64, hi: f64) -> u64 {
    if !value.is_finite() || !lo.is_finite() || !hi.is_finite() || hi <= lo {
        return Token::ESCAPE_CODE;
    }
    let max_code = Token::FIELD_MASK - 1;
    let unit = ((value - lo) / (hi - lo)).clamp(0.0, 1.0);
    (unit * max_code as f64).round() as u64
}

#[inline]
pub fn native_epistemic_uncertainty_token(
    field: EpistemicUncertaintyField,
    value: f64,
    lo: f64,
    hi: f64,
    leverage_tier: u8,
    verified: bool,
    provenance: bool,
) -> Token {
    Token::new(
        leverage_tier,
        verified,
        provenance,
        artifact_namespace(ArtifactFamily::EpistemicUncertainty, field as u8),
        quantize_bounded_field(value, lo, hi),
    )
}

#[inline]
pub fn native_teacher_consensus_token(
    field: TeacherConsensusField,
    value: f64,
    lo: f64,
    hi: f64,
    leverage_tier: u8,
    verified: bool,
    provenance: bool,
) -> Token {
    Token::new(
        leverage_tier,
        verified,
        provenance,
        artifact_namespace(ArtifactFamily::TeacherConsensus, field as u8),
        quantize_bounded_field(value, lo, hi),
    )
}

#[inline]
pub fn native_thermodynamic_action_opcode(
    opcode: ThermodynamicActionOpcode,
    leverage_tier: u8,
    verified: bool,
    provenance: bool,
) -> Token {
    Token::new(
        leverage_tier,
        verified,
        provenance,
        artifact_namespace(
            ArtifactFamily::ThermodynamicAction,
            ThermodynamicActionField::TshState as u8,
        ),
        opcode as u64,
    )
}

#[inline]
pub fn native_thermodynamic_dt_token(
    dt_fs: f64,
    max_dt_fs: f64,
    leverage_tier: u8,
    verified: bool,
    provenance: bool,
) -> Token {
    Token::new(
        leverage_tier,
        verified,
        provenance,
        artifact_namespace(
            ArtifactFamily::ThermodynamicAction,
            ThermodynamicActionField::DeltaTimeFs as u8,
        ),
        quantize_bounded_field(dt_fs, 0.0, max_dt_fs),
    )
}

/// Period for circular (phase / dihedral) quantizers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CircPeriod {
    TwoPi,
    Deg360,
    PhaseBits(u8),
}

impl CircPeriod {
    /// The numeric period in the field's own units.
    #[inline]
    pub fn period(self) -> f64 {
        match self {
            CircPeriod::TwoPi => core::f64::consts::TAU,
            CircPeriod::Deg360 => 360.0,
            CircPeriod::PhaseBits(b) => (1u64 << b) as f64,
        }
    }
}

/// Lattice kind for 3-D coordinate quantization. BCC / A3* are provably optimal 3-D
/// quantizers and require no training (deterministic fallback).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Lattice {
    LearnedVqEma,
    Bcc,
    A3Star,
}

/// The geometry of a field decides its quantizer (§4). This is the platform-native core:
/// other corpora cannot populate this because they lack the field semantics.
#[derive(Clone, Debug)]
pub enum FieldGeometry {
    /// Categorical: spike_source 1..5, status, trigger_kind.
    Enum { cardinality: u32 },
    /// Discrete physics lines (e.g. wavelength_nm 280 = Trp/Tyr) + residual codebook.
    SpectralLine {
        lines: &'static [f32],
        residual_bits: u8,
    },
    /// Phase / dihedral; 0 ≡ period. Linear quantization is wrong here.
    Circular { bits: u8, period: CircPeriod },
    /// Multi-decade magnitudes (intensity a.u., adaptive_dt, rhat, ess).
    LogMagnitude { bits: u8, floor: f32, ceil: f32 },
    /// Bounded linear ([0,1] probabilities, label_confidence).
    LinearBounded { bits: u8, lo: f32, hi: f32 },
    /// Engine-space (x,y,z) Å as a 3-D lattice, not 3 independent scalars.
    LatticeVq3d { codebook_bits: u8, lattice: Lattice },
    /// Sentinel-aware: an exact escape for the sentinel value (−1 unresolved), else `inner`.
    Sentinel {
        value: f32,
        inner: Box<FieldGeometry>,
    },
    /// residue_id / anchor_id / stream_id — relational anchors (§9).
    JoinKey,
    /// is_bimodal / structural_verified / hard_negative_mask / train_mask.
    Boolean,
    /// canonical_smiles — atom/bond grammar, never BPE.
    SmilesGrammar,
    /// PrismCode ISA stream (decode, never execute).
    OpcodeIsa,
    /// Unknown → escape token + full sidecar.
    Opaque,
}

/// Measured-information channel that drives the bit budget (§8).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InfoSource {
    /// Estimate entropy H_f from the corpus.
    Empirical,
    /// DSTW `fisher_info` is a serde-rename of `te_windowed_variance` — never call it Fisher.
    TeWindowedVariance,
    /// GFlowNet trajectory_entropy.
    TrajectoryEntropy,
    /// teacher/student label_confidence.
    LabelConfidence,
    /// Fixed budget (enums, sentinels).
    Fixed(u8),
}

/// INV-7 — the DECLARED maximum distortion for a field, with units. Distinct from the *incurred*
/// [`Distortion`] (a scalar returned by `encode`): this is the contract the encoder must satisfy
/// (or escape to the sidecar). Named `DistortionBound`, not `Distortion`, to avoid conflating the
/// bound with the incurred error.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum DistortionBound {
    /// Å, for coordinates (Euclidean).
    AbsoluteEuclidean(f32),
    /// radians, for circular/dihedral (wrapped angular).
    AbsoluteAngular(f32),
    /// absolute units, for bounded scalars.
    AbsoluteLinear(f32),
    /// fraction (0.05 = ±5%), for decade-spanning log fields.
    RelativeFraction(f32),
    /// spectral lines, enums, sentinels, booleans, join keys — must round-trip exactly.
    Exact,
}

impl DistortionBound {
    /// Does an `incurred_abs` error on a value satisfy this bound? `Exact` requires ~0.
    pub fn allows(&self, incurred_abs: f64, value: f64) -> bool {
        const EPS: f64 = 1e-9;
        match *self {
            DistortionBound::AbsoluteEuclidean(b)
            | DistortionBound::AbsoluteAngular(b)
            | DistortionBound::AbsoluteLinear(b) => incurred_abs <= b as f64 + EPS,
            DistortionBound::RelativeFraction(f) => incurred_abs <= (f as f64) * value.abs() + EPS,
            DistortionBound::Exact => incurred_abs <= EPS,
        }
    }
}

/// INV-7/8 — whether (and how) a decision rides on a field. Drives the critical-region quantizer
/// (§3) and the decision-preservation tests (INV-8).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum DecisionSpec {
    /// No decision rides on this field.
    None,
    /// A decision threshold lives in `region`; quantization must not move a value across `points`.
    Threshold {
        points: &'static [f32],
        region: (f32, f32),
    },
    /// enum/status — exactness already guarantees decision preservation.
    Categorical,
}

/// Declarative per-field spec — the heart of platform-nativeness.
#[derive(Clone, Debug)]
pub struct FieldSpec {
    pub family: ArtifactFamily,
    /// Exact decoded field name from Part I of the catalogue.
    pub field: &'static str,
    pub geometry: FieldGeometry,
    pub info_source: InfoSource,
    /// false ⇒ generative-only; force `verified`-gating (INV-3). e.g. GFlowNet reward.
    pub canonical_safe: bool,
    /// INV-7 — declared maximum distortion (or `Exact`). Required for every field.
    pub max_distortion: DistortionBound,
    /// INV-7/8 — decision relevance. Required for every field.
    pub decision_relevance: DecisionSpec,
}

/// Distortion incurred by a single `encode`, in the field's own metric (e.g. wrapped
/// angular distance for circular, |Δ| for linear). The sidecar (INV-4) holds the exact
/// value; this measures the lossy view's error for the round-trip bound (§14.1).
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Distortion(pub f64);

impl Distortion {
    #[inline]
    pub fn value(self) -> f64 {
        self.0
    }
}

/// One quantizer per ontological field, geometry matched to the field's true distribution.
pub trait Quantizer {
    /// Corpus-fit; EMA for VQ kinds.
    fn fit(&mut self, samples: &[f64]);
    /// Codebook index + incurred distortion.
    fn encode(&self, v: f64) -> (u32, Distortion);
    /// For round-trip / sidecar reconciliation.
    fn decode(&self, t: u32) -> f64;
    fn bit_budget(&self) -> u8;
    /// Codebook health monitor (§13): 2^entropy of the code-usage distribution.
    fn perplexity(&self) -> f64;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_packs_and_unpacks_each_field() {
        let t = Token::new(0b101010, true, false, 0xABC, 0x0F_FFFF_FFFF);
        assert_eq!(t.leverage_tier(), 0b101010);
        assert!(t.verified());
        assert!(!t.provenance());
        assert_eq!(t.namespace(), 0xABC);
        assert_eq!(t.field_token(), 0x0F_FFFF_FFFF);
    }

    #[test]
    fn token_fields_do_not_bleed_at_max_values() {
        let t = Token::new(
            Token::LEVERAGE_MASK as u8,
            true,
            true,
            Token::NAMESPACE_MASK as u16,
            Token::FIELD_MASK,
        );
        assert_eq!(t.leverage_tier() as u64, Token::LEVERAGE_MASK);
        assert!(t.verified());
        assert!(t.provenance());
        assert_eq!(t.namespace() as u64, Token::NAMESPACE_MASK);
        assert_eq!(t.field_token(), Token::FIELD_MASK);
    }

    #[test]
    fn zero_token_is_source_of_truth_quarantined() {
        // leverage_tier 0 = source-of-truth; verified 0 = quarantined (INV-3 default-deny).
        let t = Token(0);
        assert_eq!(t.leverage_tier(), 0);
        assert!(!t.verified());
        assert!(!t.provenance());
    }

    #[test]
    fn namespace_factoring_is_injective_across_families() {
        // INV-2: same field_slot, different family_slot ⇒ different namespace.
        let a = pack_namespace(ArtifactFamily::SpikeEvent.family_slot(), 7);
        let b = pack_namespace(ArtifactFamily::MsmKineticSummary.family_slot(), 7);
        assert_ne!(a, b);
        assert_eq!(unpack_namespace(a).1, 7);
        assert_eq!(
            unpack_namespace(a).0,
            ArtifactFamily::SpikeEvent.family_slot()
        );
    }

    #[test]
    fn scientific_chronology_families_are_append_only_slots() {
        assert_eq!(ArtifactFamily::ScientificChronologyEvent.family_slot(), 47);
        assert_eq!(ArtifactFamily::ScientificStateEpoch.family_slot(), 48);
        assert_eq!(ArtifactFamily::T4EpochOverlay.family_slot(), 49);
        assert_eq!(
            ArtifactFamily::ChronologyAcceptanceDecision.family_slot(),
            50
        );
        assert_eq!(
            ArtifactFamily::ScientificPocketOpeningEvent.family_slot(),
            51
        );
        assert_eq!(ArtifactFamily::LigandCausalActivityEvent.family_slot(), 52);
        assert_eq!(ArtifactFamily::ResidueActivityEvent.family_slot(), 53);
        assert_eq!(ArtifactFamily::SpectralExtractor.family_slot(), 54);
        assert_eq!(
            ArtifactFamily::ScientificUncertaintyTensor.family_slot(),
            55
        );
        assert_eq!(
            ArtifactFamily::RuntimeNumericalUncertaintyTensor.family_slot(),
            56
        );
        assert_eq!(ArtifactFamily::ThermalUncertaintyTensor.family_slot(), 57);
        assert_eq!(
            ArtifactFamily::IntegratorUncertaintyTensor.family_slot(),
            58
        );
        assert_eq!(
            ArtifactFamily::ObservableUncertaintyTensor.family_slot(),
            59
        );
        assert_eq!(ArtifactFamily::SignalToUncertaintyTensor.family_slot(), 60);
        assert_eq!(ArtifactFamily::EpochUncertaintyTensor.family_slot(), 61);
        assert_eq!(
            ArtifactFamily::NegativeScientificEvidenceEvent.family_slot(),
            62
        );
        assert_eq!(ArtifactFamily::ThermodynamicSalienceEvent.family_slot(), 63);
        assert_eq!(ArtifactFamily::TransitionMotif.family_slot(), 64);
        assert_eq!(ArtifactFamily::PersistentTransitionObject.family_slot(), 65);
        assert_eq!(ArtifactFamily::TransitionConsensusEvent.family_slot(), 66);
        assert_eq!(ArtifactFamily::TransitionEvidenceLedger.family_slot(), 67);
        assert_eq!(
            ArtifactFamily::ScientificTransitionConsensusEvent.family_slot(),
            68
        );
        assert_eq!(
            ArtifactFamily::TransitionFalsificationProfile.family_slot(),
            69
        );
        assert_eq!(
            ArtifactFamily::RejectedPersistentTransitionCandidate.family_slot(),
            70
        );
        assert_eq!(
            ArtifactFamily::PersistentTransitionAcceptanceLedger.family_slot(),
            71
        );
        assert_eq!(ArtifactFamily::EpistemicUncertainty.family_slot(), 72);
        assert_eq!(ArtifactFamily::TeacherConsensus.family_slot(), 73);
        assert_eq!(ArtifactFamily::ThermodynamicAction.family_slot(), 74);
        assert_eq!(Token::NAMESPACE_BITS, 13);
        assert_eq!(FAMILY_SLOT_BITS, 7);
        assert_eq!(FIELD_SLOT_BITS, 6);
        assert_ne!(
            pack_namespace(ArtifactFamily::ScientificChronologyEvent.family_slot(), 0),
            pack_namespace(ArtifactFamily::ScientificStateEpoch.family_slot(), 0)
        );
        assert_ne!(
            pack_namespace(ArtifactFamily::T4EpochOverlay.family_slot(), 0),
            pack_namespace(
                ArtifactFamily::ChronologyAcceptanceDecision.family_slot(),
                0
            )
        );
        assert_ne!(
            pack_namespace(
                ArtifactFamily::ScientificPocketOpeningEvent.family_slot(),
                0
            ),
            pack_namespace(ArtifactFamily::LigandCausalActivityEvent.family_slot(), 0)
        );
        assert_ne!(
            pack_namespace(ArtifactFamily::LigandCausalActivityEvent.family_slot(), 0),
            pack_namespace(ArtifactFamily::ResidueActivityEvent.family_slot(), 0)
        );
        assert_ne!(
            pack_namespace(ArtifactFamily::ResidueActivityEvent.family_slot(), 0),
            pack_namespace(ArtifactFamily::SpectralExtractor.family_slot(), 0)
        );
        assert_ne!(
            pack_namespace(ArtifactFamily::SpectralExtractor.family_slot(), 0),
            pack_namespace(ArtifactFamily::ScientificUncertaintyTensor.family_slot(), 0)
        );
        assert_ne!(
            pack_namespace(ArtifactFamily::ScientificUncertaintyTensor.family_slot(), 0),
            pack_namespace(
                ArtifactFamily::RuntimeNumericalUncertaintyTensor.family_slot(),
                0
            )
        );
        assert_ne!(
            pack_namespace(ArtifactFamily::ObservableUncertaintyTensor.family_slot(), 0),
            pack_namespace(ArtifactFamily::SignalToUncertaintyTensor.family_slot(), 0)
        );
        assert_ne!(
            pack_namespace(ArtifactFamily::EpochUncertaintyTensor.family_slot(), 0),
            pack_namespace(
                ArtifactFamily::NegativeScientificEvidenceEvent.family_slot(),
                0
            )
        );
        assert_ne!(
            pack_namespace(
                ArtifactFamily::NegativeScientificEvidenceEvent.family_slot(),
                0
            ),
            pack_namespace(ArtifactFamily::ThermodynamicSalienceEvent.family_slot(), 0)
        );
        assert_ne!(
            pack_namespace(ArtifactFamily::ThermodynamicSalienceEvent.family_slot(), 0),
            pack_namespace(ArtifactFamily::TransitionEvidenceLedger.family_slot(), 0)
        );
        assert_ne!(
            pack_namespace(
                ArtifactFamily::PersistentTransitionAcceptanceLedger.family_slot(),
                0
            ),
            pack_namespace(ArtifactFamily::EpistemicUncertainty.family_slot(), 0)
        );
        assert_ne!(
            pack_namespace(ArtifactFamily::TeacherConsensus.family_slot(), 0),
            pack_namespace(ArtifactFamily::ThermodynamicAction.family_slot(), 0)
        );
    }

    #[test]
    fn native_epistemic_action_tokens_are_packed_u64_namespaces() {
        let sigma = native_epistemic_uncertainty_token(
            EpistemicUncertaintyField::Sigma,
            0.25,
            0.0,
            1.0,
            2,
            true,
            true,
        );
        let tsh = native_thermodynamic_action_opcode(
            ThermodynamicActionOpcode::Hysteresis,
            1,
            true,
            true,
        );
        assert_eq!(
            sigma.namespace(),
            artifact_namespace(
                ArtifactFamily::EpistemicUncertainty,
                EpistemicUncertaintyField::Sigma as u8
            )
        );
        assert_eq!(
            tsh.namespace(),
            artifact_namespace(
                ArtifactFamily::ThermodynamicAction,
                ThermodynamicActionField::TshState as u8
            )
        );
        assert_eq!(
            tsh.field_token(),
            ThermodynamicActionOpcode::Hysteresis as u64
        );
        assert!(sigma.field_token() < Token::ESCAPE_CODE);
    }

    #[test]
    fn circ_period_values() {
        assert_eq!(CircPeriod::Deg360.period(), 360.0);
        assert_eq!(CircPeriod::PhaseBits(8).period(), 256.0);
        assert!((CircPeriod::TwoPi.period() - core::f64::consts::TAU).abs() < 1e-12);
    }
}
