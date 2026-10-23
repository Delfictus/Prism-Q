use prismcode::cse_tables::{
    CseComponentTables, CseEntityRow, CseEventRow, CseGFlowNetRow, CseSourceDocument,
    CseTeacherRow, CseThermoRow, CseTimebaseRow, EventSource, GFlowNetTraceSource, MoleculeSource,
    ProteinSource, ProvenanceSource, ScientificStateSource, TeacherOutputSource, ThermoSource,
    TimebaseSource, UncertaintySource, VariantSource,
};
use prismcode::drf_fabric::{build_drf_graph, reconstruct_drf_graph};
use prismcode::drf_validation::{fail_if_invalid, validate_drf_graph};
use prismcode::nrf_frame::build_nrf_frame;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

fn repo_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}

fn load_source() -> ScientificStateSource {
    let bytes = fs::read(repo_path("examples/corpus/cse_hd/drf_graph.pcsrc.json")).unwrap();
    let document: CseSourceDocument = serde_json::from_slice(&bytes).unwrap();
    source_from_cse_document(document, 110, 210, 2, 20)
}

fn source_from_cse_document(
    document: CseSourceDocument,
    molecule_id: u64,
    protein_id: u64,
    time_id: u64,
    event_base: u64,
) -> ScientificStateSource {
    let molecule = document.molecules.first().unwrap();
    let protein = document.proteins.first().unwrap();
    let timebase = document.timebases.first().unwrap();
    let provenance = document.provenance.first().unwrap();
    let mut atom_index = BTreeMap::new();
    for (idx, atom) in molecule.atoms.iter().enumerate() {
        atom_index.insert(atom.id.as_str(), idx as u32);
    }
    let mut state_refs = BTreeMap::new();
    state_refs.insert(molecule.id.as_str(), molecule_id);
    state_refs.insert(protein.id.as_str(), protein_id);
    let mut ordered_events = document.events.clone();
    ordered_events.sort_by_key(|event| event.order_index);

    ScientificStateSource {
        source_id: document.state_id,
        synthetic: document.synthetic,
        generator_version: "prism-cse-hd-drf-fixture-v0.1".to_string(),
        seed: 2026,
        molecule: MoleculeSource {
            entity_id: molecule_id,
            name: molecule.id.clone(),
            topology_hash: molecule.topology_hash.clone(),
            atom_types: molecule
                .atoms
                .iter()
                .map(|atom| atom.atom_type.clone())
                .collect(),
            bonds: molecule
                .bonds
                .iter()
                .map(|bond| {
                    [
                        *atom_index.get(bond.source_atom.as_str()).unwrap(),
                        *atom_index.get(bond.target_atom.as_str()).unwrap(),
                    ]
                })
                .collect(),
        },
        protein: ProteinSource {
            entity_id: protein_id,
            receptor: protein.id.clone(),
            chain_id: protein
                .chains
                .first()
                .map(|chain| chain.id.clone())
                .unwrap_or_default(),
            residues: protein
                .chains
                .iter()
                .flat_map(|chain| chain.residues.iter())
                .map(|residue| format!("{}{}", residue.name, residue.sequence_index))
                .collect(),
            atom_site_count: protein.atom_sites.len() as u64,
            topology_hash: protein.topology_hash.clone(),
        },
        waters: Vec::new(),
        timebase: TimebaseSource {
            time_id,
            unit: timebase.unit_ref.clone(),
            start: timebase.frame_start as f64,
            step: 1.0,
            count: timebase.frame_end.saturating_sub(timebase.frame_start),
            timebase_hash: timebase.timebase_hash.clone(),
            forcefield_parameter_hash: "forcefield_hash_synthetic_drf_v0".to_string(),
        },
        variants: document
            .variants
            .iter()
            .enumerate()
            .map(|(idx, variant)| VariantSource {
                variant_id: 410 + idx as u64,
                reference_state_ref: *state_refs
                    .get(variant.reference_state_ref.as_str())
                    .unwrap_or(&molecule_id),
                alternate_state_ref: *state_refs
                    .get(variant.alternate_state_ref.as_str())
                    .unwrap_or(&protein_id),
                delta_kind: variant.delta_hash.clone(),
                provenance_ref: 910,
            })
            .collect(),
        events: ordered_events
            .iter()
            .enumerate()
            .map(|(idx, event)| EventSource {
                event_id: event_base + idx as u64,
                event_kind: event.kind.clone(),
                source_ref: *state_refs.get(event.source_ref.as_str()).unwrap(),
                target_ref: *state_refs.get(event.target_ref.as_str()).unwrap(),
                time_ref: time_id,
                time_delta: event.order_index as i64,
                motif: event.motif.clone(),
                amplitude: 0.6 - idx as f64 * 0.25,
                unit: "unitless".to_string(),
                causal_parent_ref: (idx > 0).then_some(event_base + idx as u64 - 1),
            })
            .collect(),
        uncertainties: document
            .uncertainties
            .iter()
            .enumerate()
            .map(|(idx, uncertainty)| UncertaintySource {
                uncertainty_id: 510 + idx as u64,
                target_ref: event_base,
                distribution: uncertainty.model.clone(),
                calibration: uncertainty.calibration_hash.clone(),
                value: 0.08,
            })
            .collect(),
        thermo: document
            .thermo
            .iter()
            .enumerate()
            .map(|(idx, thermo)| ThermoSource {
                thermo_id: 610 + idx as u64,
                target_ref: event_base,
                quantity_kind: thermo.quantity_kind.clone(),
                value: 0.35,
                unit_ref: thermo.unit_ref.clone(),
                uncertainty_ref: 510,
            })
            .collect(),
        teacher_outputs: document
            .teacher_outputs
            .iter()
            .enumerate()
            .map(|(idx, teacher)| TeacherOutputSource {
                output_id: 710 + idx as u64,
                model_id: teacher.model_ref.clone(),
                label: teacher.label.clone(),
                confidence: teacher.confidence,
                uncertainty_ref: 510,
                provenance_ref: 910,
                airlock_state: teacher.airlock_state.clone(),
            })
            .collect(),
        gflownet_traces: document
            .gflownet_traces
            .iter()
            .enumerate()
            .map(|(idx, trace)| GFlowNetTraceSource {
                trace_id: 810 + idx as u64,
                state_ref: *state_refs
                    .get(trace.state_ref.as_str())
                    .unwrap_or(&molecule_id),
                action_sequence: vec![trace.action.clone()],
                reward: 0.4,
                flow: 0.2,
                policy_hash: trace.lineage_hash.clone(),
                provenance_ref: 910,
            })
            .collect(),
        nanl_symbols: document.nanl_symbols,
        provenance: ProvenanceSource {
            provenance_id: 910,
            label: provenance.classification.clone(),
            synthetic: provenance.synthetic,
            reason: "deterministic DRF graph/tensor validation fixture".to_string(),
            source_hash: provenance.source_hash.clone(),
        },
    }
}

fn tables_for(source: &ScientificStateSource) -> CseComponentTables {
    let entity_table = vec![
        CseEntityRow {
            entity_id: source.molecule.entity_id,
            entity_kind: "MOLECULE".to_string(),
            dictionary_ref: 0,
            topology_hash: source.molecule.topology_hash.clone(),
            component_span_start: 0,
            component_span_len: source.molecule.atom_types.len() as u32,
        },
        CseEntityRow {
            entity_id: source.protein.entity_id,
            entity_kind: "PROTEIN".to_string(),
            dictionary_ref: 1,
            topology_hash: source.protein.topology_hash.clone(),
            component_span_start: source.molecule.atom_types.len() as u32,
            component_span_len: source.protein.atom_site_count as u32,
        },
    ];
    CseComponentTables {
        schema_version: "PRISM_CSE_HD_TABLES_TEST".to_string(),
        source_id: source.source_id.clone(),
        source_hash: blake3::hash(&serde_json::to_vec(source).unwrap())
            .to_hex()
            .to_string(),
        canonical_hash: blake3::hash(source.source_id.as_bytes())
            .to_hex()
            .to_string(),
        dictionary_hash: blake3::hash(b"drf-dictionary").to_hex().to_string(),
        entity_table,
        event_table: source
            .events
            .iter()
            .enumerate()
            .map(|(idx, event)| CseEventRow {
                event_id: event.event_id,
                event_kind_ref: idx as u32,
                source_ref: event.source_ref,
                target_ref: event.target_ref,
                time_ref: event.time_ref,
                time_delta: event.time_delta,
                motif_ref: idx as u32,
                amplitude: event.amplitude,
                unit_ref: event.unit.clone(),
                causal_parent_ref: event.causal_parent_ref,
                event_hash: blake3::hash(format!("event:{}", event.event_id).as_bytes())
                    .to_hex()
                    .to_string(),
            })
            .collect(),
        thermo_table: source
            .thermo
            .iter()
            .map(|thermo| CseThermoRow {
                thermo_id: thermo.thermo_id,
                target_ref: thermo.target_ref,
                quantity_ref: 0,
                value: thermo.value,
                unit_ref: thermo.unit_ref.clone(),
                uncertainty_ref: thermo.uncertainty_ref,
            })
            .collect(),
        timebase_table: vec![CseTimebaseRow {
            time_id: source.timebase.time_id,
            unit_ref: source.timebase.unit.clone(),
            start: source.timebase.start,
            step: source.timebase.step,
            count: source.timebase.count,
            timebase_hash: source.timebase.timebase_hash.clone(),
            forcefield_parameter_hash: source.timebase.forcefield_parameter_hash.clone(),
        }],
        teacher_output_table: Vec::<CseTeacherRow>::new(),
        gflownet_trace_table: Vec::<CseGFlowNetRow>::new(),
        provenance_hash: source.provenance.source_hash.clone(),
    }
}

#[test]
fn drf_build_validate_and_reconstructs_source_bound_fabric() {
    let source = load_source();
    let tables = tables_for(&source);
    let frame = build_nrf_frame(&source, &tables).unwrap();
    let graph = build_drf_graph(&tables, &frame).unwrap();
    let report = validate_drf_graph(&graph, &tables, &frame).unwrap();
    let reconstruction = reconstruct_drf_graph(&graph).unwrap();

    assert_eq!(report.status, "PASS");
    assert!(report.source_hash_binding_pass);
    assert!(report.nrf_replay_binding_pass);
    assert!(report.canonical_timebase_bound);
    assert!(report.reconstruction_test_pass);
    assert!(report.replay_hash_pass);
    assert_eq!(graph.tensor_count, 2);
    assert_eq!(graph.source_nrf_hash, frame.nrf_frame_hash);
    assert_eq!(graph.source_nrf_replay_hash, frame.replay_hash);
    assert_eq!(reconstruction.status, "PASS");

    let out_dir = repo_path("build/demo");
    fs::create_dir_all(&out_dir).unwrap();
    fs::write(
        out_dir.join("drf_fabric_validation.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
}

#[test]
fn drf_rejects_source_nrf_replay_binding_mismatch() {
    let source = load_source();
    let tables = tables_for(&source);
    let frame = build_nrf_frame(&source, &tables).unwrap();
    let mut graph = build_drf_graph(&tables, &frame).unwrap();
    graph.source_nrf_replay_hash = "tampered".to_string();
    let err = fail_if_invalid(&graph, &tables, &frame)
        .unwrap_err()
        .to_string();
    assert!(err.contains("source NRF replay hash mismatch"));
}

#[test]
fn drf_derived_tensor_cannot_claim_canonical_losslessness() {
    let source = load_source();
    let tables = tables_for(&source);
    let frame = build_nrf_frame(&source, &tables).unwrap();
    let mut graph = build_drf_graph(&tables, &frame).unwrap();
    graph.tensors[1].claims_canonical_lossless = true;
    let err = fail_if_invalid(&graph, &tables, &frame)
        .unwrap_err()
        .to_string();
    assert!(err.contains("cannot claim canonical losslessness"));
}

#[test]
fn drf_rejects_timebase_and_replay_hash_tamper() {
    let source = load_source();
    let tables = tables_for(&source);
    let frame = build_nrf_frame(&source, &tables).unwrap();
    let mut graph = build_drf_graph(&tables, &frame).unwrap();
    graph.canonical_timebase_hash = "tampered_timebase".to_string();
    let err = fail_if_invalid(&graph, &tables, &frame)
        .unwrap_err()
        .to_string();
    assert!(err.contains("canonical timebase hash mismatch"));

    let mut graph = build_drf_graph(&tables, &frame).unwrap();
    graph.replay_hash = "tampered_replay".to_string();
    let err = fail_if_invalid(&graph, &tables, &frame)
        .unwrap_err()
        .to_string();
    assert!(err.contains("DRF replay hash reconstruction failed"));
}
