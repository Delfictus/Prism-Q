//! §6 — schema versioning + reserved namespaces for manifest-gap families.
//!
//! A family schema may legitimately exist before its artifacts do (A11 HOLO, A12 BENCH120/gates,
//! A13 TWIN). The registry accepts those families and emits `Opaque` + full sidecar until they
//! are populated (§13 schema-drift guard), so adding artifacts later never breaks the vocabulary.

/// Bumped on any change to the field table or geometry assignments (§13 determinism).
/// v2: added the KccSite family (kcc_visualization.json sites[] relations).
/// v3: added the SpikeTransitionEvent family (full nhs_rt_full SPIKE_TRANSITION_EVENT schema) +
///     KccSite prism_therm TWIN member-join fields.
/// v4: gap-closure — BocpdLog (bocpd.jsonl) + ProtocolState (protocol_state.json) families +
///     KccSite binding_sites join fields.
/// v5: SpikeTransitionEvent enum vocab grounded in a REAL all-on v2_ignition run (trigger_kind/
///     phase/condition_id/protocol_name/spike_count_status) + condition_id/protocol/status typed.
/// v6: TwinTransitionResidual family (TWIN16 interferometric transition-view residuals: per-cell
///     aggregates + cell-pair L1 residuals).
/// v7: TwinCoupled populated with REAL per-residue coupling evidence (twin_per_residue.json:
///     b_over_a_ratio/ccf_*/barrier_class/spikes_a,b/consensus/differential) from a memory-safe
///     8-engine coupled-TWIN run.
/// v8: TwinConditionEdge family (TWIN16-native DirectedPerturbationDifferential per-cell-pair edges:
///     cell_a/cell_b/axis/hamming/transition_view_l1_residual/material).
/// v9: NmaCoupledEdge family (AUXILIARY coupled-A/B NMA per-residue differential — SEPARATE evidence
///     class from TWIN16; never relabelled as / merged into TwinConditionEdge).
/// v10: Directive v6 Agent 3/4 — MoleculeGeneration family populated with its FIRST field block
///     (enamine synthons + V-space products: smiles/ids/charges/counts/reward-components/consensus
///     bonuses/enums; all canonical_safe=false, generative firewall) + GflownetTrajectory extended
///     with the 52-col gflownet_raw_policy_samples reward-component + sampling-control fields
///     (regime/temperature/backward_logprob/u_pose/pathway-sigma telemetry) and the survival_tier +
///     regime enum label tables. No enum discriminants changed; no existing field altered.
/// v11: Directive v6.1 Track B — TeacherStudentConsensus EXTENDED with the per-pair teacher/student
///     BRIDGE training labels (PRISM_V2_ROW_MOLECULE_BRIDGE_TRAINING_TABLE.parquet: pair_label/
///     pair_weight/teacher_target/teacher_loss_weight/distillation_loss_weight/uncertainty|positive|
///     negative_target/model_uncertainty|coverage/evidence_quality/integrated/phase_coherence/
///     gflownet_reward(+fusion_target)/16 fusion_* multimodal evidence fields/5 signed delta_*/
///     evidence_role|phase_evidence_tier|water_class_v2|negative_family|contradiction_flag enums/
///     ood|do_not_discard booleans/pair_id|molecule_id join keys). APPENDED LAST: the v10 per-residue
///     teacher fields keep field-slots 0..10 (INV-2 namespaces stable); no existing field altered.
///     `row__`-prefixed columns resolve via json_key_for(); all canonical_safe=true (label evidence,
///     NOT a generative firewall). MoleculeGeneration/GflownetTrajectory/SpikeEvent UNCHANGED.
/// v12: DSTW-T4 Step h5 — DenseTensorSemanticSummary family (one POST_RUN_DERIVED semantic record per
///     dense engine tensor: signal_grid/warp/DSTW adaptive_dt/superresolution/TWIN16/neuromorphic
///     spike sensor). 18 fields: tensor_role/modality enums + source_kind/source_hash/tokenizer_family/
///     parent_vertex_id join keys + runtime_origin (+POST_RUN_DERIVED added to DSTW_RUNTIME_ORIGIN) +
///     stream_id/on_disk_bytes + artifact_present/summary_present/all_finite booleans + row_count/
///     column_count dimensions + nan_count/inf_count/nonfinite_or_null_count/reducer_scalar finite-scan
///     scalars. Dense tensors must NOT enter training raw — this is the semantic gate. APPENDED LAST:
///     no enum discriminants changed; no existing field altered; the DSTW state-graph families
///     (DstwStateVertex/Transition/NextOperator) keep their field-slots (INV-2 namespaces stable).
/// v13: Phase-1 native-corpus closure batch — OpcodeExecutionLedger family (discriminant 36): one
///     per-executed-opcode/kernel row from the prismos-authored `opcode_operand_ledger.json`
///     (schema_kind=prismcode_typed_operand_ledger, 368 records). 16 fields: opcode_name/opcode_id/
///     blocker_reason/source_receipt/input_sha256/output_sha256/qnanl_packet_sha256 join keys +
///     count (the VALUE field, tied by-value to the md_native_receipt opcode_counts/kernel_counts) +
///     operand_schema_version + operator_class/operand_target/runtime_origin/disassembly_status/
///     validation_status/replay_status enums + runtime_native boolean. APPENDED LAST: no enum
///     discriminants changed; no existing field altered (INV-2 namespaces stable).
///     ValidationState family (discriminant 37): the aggregate MD-ISA execution VERDICT header from
///     the prismos-authored `prismos_prismcode_isa_execution_verification.json` (schema_kind
///     prismos_prismcode_isa_execution_verification). 12 fields: status/substrate_status(+MISSING)/
///     implementation_status_language enums + schema_version/native_md_receipt_count/
///     min_hotpath_launch_collapse counts + all_streams_pass/all_streams_bounded_or_passed/
///     cuda_opcode_coverage_exists/hotpath_acceleration_required/metal_resident_md_required verdict
///     booleans + plan_hash_blake3 join key. BY-VALUE: native_md_receipt_count == len(streams);
///     all_streams_pass == AND(streams[i].status=="PASS"). The data-dense streams[] descends to the
///     sidecar. APPENDED LAST: no enum discriminants changed; no existing field altered.
///     IsaRegistrySummary family (discriminant 42): the POST_RUN_DERIVED registry×receipt coverage
///     matrix (`prismos_prismcode_isa_registry_summary.json`, schema_kind
///     prismos_prismcode_isa_registry_summary, emitter prismos.rs build_prismos_isa_registry_summary),
///     a top-level records[] flat array of one per-(stream,opcode) observation. 12 fields: stream_id/
///     opcode_index/count/registry_size/required_core_size/native_md_receipt_count scalars + observed/
///     planned/required_core booleans + opcode enum (ISA_MD_OPCODES 25-value vocab) + stream_status/
///     stream_coverage_status enums (STATUS). BY-VALUE: per stream distinct(opcode)==count(records)==
///     registry_size==25; every required_core opcode observed; distinct(stream_id)==
///     native_md_receipt_count. APPENDED LAST: no enum discriminants changed; no existing field
///     altered.
/// v14: KccDescriptorNative family (discriminant 43): the RAW PRKCC001 `kcc_v2full.bin` per-residue
///     KCC descriptor carrier (schema_kind prismos_kcc_descriptor_native_summary, decoder prismos.rs
///     decode_prkcc001_native_records), a top-level records[] flat array of one per-residue record
///     (156 residues for 4LPK_apo). 16 fields: residue_id/stream_id JoinKeys + temporal_corr/
///     direction_score/motion_efficiency/burst_motion/phase_shift/causal_lag/lag_corr_peak/local_cov/
///     net_dx/net_dy/net_dz/sum_m scalars + residue_count/active_causal counts. Transposed BY VALUE
///     from the self-describing envelope (NOT tensor min/max/mean stats). BY-VALUE: len(records)==
///     n_residues(payload)==record_count(envelope)==topology n_residues; ranges per field;
///     active_causal<=residue_count; causal_lag/lag_corr_peak == null (md-only NaN sentinel).
///     DISTINCT from KccDescriptor (the kcc_visualization.json residues[] JSON view). APPENDED LAST:
///     no enum discriminants changed; no existing field altered.
/// v15: PredictionState family (discriminant 45): the FIRST-CLASS prediction record per PROMOTED
///     materialized binding site (schema_kind prismos_prediction_state, emitter prismos.rs
///     build_prismos_prediction_state, top-level records[] flat array — 20 records for 4LPK_apo).
///     Operator correction 2026-07-01: a PREDICTION STATE (hypothesis about a site with by-value
///     confidence/uncertainty + evidence lineage) is DISTINCT from DstwNextOperator (an engine-owner
///     work list); this SUPERSEDES the DstwNextOperator-alias closure and routes INDEPENDENTLY
///     (schema_kind=="prismos_prediction_state" => /records). 18 fields: prediction_id/target_id/
///     predicted_surface/source_vertex_id/target_vertex_id/evidence_parent_ids/
///     operator_recommendation_id/qnanl_receipt_ref JoinKeys + predicted_delta (lin, ==
///     new_prism_rank−old_density_rank) + confidence/uncertainty unit(0..1) REAL varying numerics +
///     receptor_state/ligand_state/epistemic_status/validation_status/falsification_status/
///     runtime_origin/tokenizer_family enums. BY-VALUE: n_records==n_materialized_sites==20;
///     confidence==manifold_consistency_score, uncertainty==spread/(1+spread) (both vary);
///     predicted_delta==new_prism_rank−old_density_rank; every vertex id resolves to a
///     PRISM_NATIVE_STATE_GRAPH vertex. APPENDED LAST: no enum discriminants changed; no existing
///     field altered (INV-2 namespaces stable). DstwNextOperator/DstwStateTransition UNCHANGED.
/// v16: PhaseManifoldEdgeValidation family (discriminant 46): the GhostPhaseLattice4D phase-manifold
///     EDGE-VALIDATION manifest ({stem}_binding_sites_lattice.json, engine emitter nhs_rt_full.rs
///     ~16336, emitted ONLY when v2_was_live==true AND lattice cluster() succeeds). Routed as ONE
///     top-level record (pointer ""); flatten_phase_manifold_edge_validation lifts stats.*/config.* +
///     derives component aggregates (n_components, sum_component_n_nodes, total_intra_edges,
///     mean/min/max_intra_component_mean_cosine) so route_object value-matches real edge counts + SO(3)
///     coherence. BY-VALUE: Σ component.n_nodes==stats.n_nodes; Σ component.n_intra_edges==
///     stats.n_directed_edges (edge-count-consistency); every intra_component_mean_cosine∈[-1,1];
///     monotonic edge funnel (phase_legal/aabb_overlap≤evaluated, directed≤aabb_overlap); len(components)
///     ==n_components. DISTINCT from CapturedGraphGpuTelemetry (the flat ghost_lattice_routing_status.json
///     STATUS object — no components[]/stats). RED-FLAG-EXCEPTION corpus-closure lane (non-canonical,
///     tokenizer-family closure only). APPENDED LAST: no enum discriminants changed; no existing field
///     altered (INV-2 namespaces stable).
/// v17: ScientificChronologyEvent (discriminant 47) and ScientificStateEpoch (discriminant 48)
///     first-class lanes. Source wrappers are `scientific_chronology_events.json`
///     (`schema_kind=prismos_scientific_chronology_events`, records[] of
///     `prismos_scientific_chronology_event`) and `scientific_state_epochs.json`
///     (`schema_kind=prismos_scientific_state_epochs`, records[] of
///     `prismos_scientific_state_epoch`). Wrappers route by descending `/records`; they are not
///     tokenizer aliases for DSTW transitions or vertices. APPENDED LAST: no enum discriminants
///     changed; no existing field altered.
/// v18: T4EpochOverlay (discriminant 49) and ChronologyAcceptanceDecision (discriminant 50)
///     first-class closeout lanes. Source artifacts are `dstw_t4_epoch_overlay.json`
///     (`schema_kind=dstw_t4_epoch_overlay`) and `chronology_acceptance_decision.json`
///     (`schema_kind=dstw_t4_chronology_acceptance_decision`). They route as top-level records and
///     carry the DSTW-T4 chronology acceptance gate without aliasing vertices/transitions.
///     APPENDED LAST: no enum discriminants changed; no existing field altered.
/// v19: Operator-scoped causal geometry layer: ScientificPocketOpeningEvent (51),
///     LigandCausalActivityEvent (52), and ResidueActivityEvent (53). No broad event taxonomy is
///     introduced. Nested operator fields are preserved exactly and their source-aligned flat
///     aliases are documented in SCIENTIFIC_SURFACE_FIELD_ALIASES. Spectral extractor and spectral
///     fiber bundle refs are routed inside the same three event schemas as the highest-fidelity
///     geometry authority; Path-B refs route as partial/pending spectral geometry.
/// v20: SpectralExtractor family (discriminant 54). Routes the PRISMOS-owned run-specific spectral
///     closure manifest by value: PHYSICAL_PROFILE/contact-edge input hashes and row counts,
///     Laplacian eigenvalues, ranked-site overlay rows, TWIN16 residual counts, and output artifact
///     hashes. This removes the `PENDING_SPECTRAL_EXTRACTOR_INPUTS` state when the producer has
///     emitted real geometry inputs; corpus compile must see matched_fields>0 or fail.
/// v21: Scientific uncertainty substrate (discriminants 55-61). Adds measured uncertainty tensor
///     families without renumbering chronology/causal/spectral lanes: ScientificUncertaintyTensor,
///     RuntimeNumericalUncertaintyTensor, ThermalUncertaintyTensor, IntegratorUncertaintyTensor,
///     ObservableUncertaintyTensor, SignalToUncertaintyTensor, and EpochUncertaintyTensor. The wider
///     V2 uncertainty ontology (replica/mode/temporal/physics/evidence/signal roles) is represented
///     as typed fields, not extra family slots, to preserve the 6-bit namespace invariant.
/// v22: NegativeScientificEvidenceEvent (discriminant 62). Adds one multiplexed adversarial
///     falsification family without changing the 12-bit namespace or 6-bit family-slot layout.
///     Required rejection object identities remain first-class typed values through
///     negative_evidence_type/rejection_class/control_family/failed_claim_type/motif_status/
///     rejection_reason, with PrismOS opcode / PrismCode instruction / QNANL lineage carried as
///     matched fields; namespace expansion is deliberately deferred.
/// v23: Scoped TSH Stage-II runtime observer ontology. Expands namespace to 13 bits
///     (7-bit family slot + unchanged 6-bit field slot) so ThermodynamicSalienceEvent,
///     TransitionMotif, PersistentTransitionObject, TransitionConsensusEvent, and
///     TransitionEvidenceLedger are append-only first-class runtime-emitted families without
///     aliasing chronology, causal geometry, uncertainty, spectral, or negative evidence lanes.
///     This is limited to the verified observational non-interference scope; PersistentTransitionObject
///     acceptance remains partial pending negative-control-backed promotion.
/// v24: TSH falsification consensus gate. Adds append-only corpus/materialization families
///     ScientificTransitionConsensusEvent, TransitionFalsificationProfile,
///     RejectedPersistentTransitionCandidate, and PersistentTransitionAcceptanceLedger. These
///     bridge Stage-II observer motifs to the V6 canonical negative-evidence bundle without
///     Stage-III steering or runtime mutation.
/// v25: Native PLLM epistemic/action sequence lanes. Adds append-only EpistemicUncertainty,
///     TeacherConsensus, and ThermodynamicAction families for compact sigma/variance,
///     entropy/disagreement, TSH-state, and time-delta tokens. These are causal-sequence u64
///     lanes, not text-token aliases, and preserve the v23 13-bit namespace layout.
/// v26: Append VORONOI_LIF as SpikeEvent source code 6. Source codes 0..5,
///     field slots, and the existing 3-bit categorical width are unchanged.
///     New-source records require this versioned registry; Arrow V5 remains unchanged.
pub const SCHEMA_VERSION: u32 = 26;

/// Families whose artifacts are not yet wired upstream (manifest gaps). Their fields route to
/// `Opaque` + sidecar; reserved here so their namespace slots are stable across versions.
pub const MANIFEST_GAP_FAMILIES: &[crate::types::ArtifactFamily] = &[
    crate::types::ArtifactFamily::HoloBoundState,
    crate::types::ArtifactFamily::CapabilityGate,
    crate::types::ArtifactFamily::TwinCoupled,
];
