use prismcode::cse_tables::{
    CseComponentTables, CseEntityRow, CseEventRow, CseGFlowNetRow, CseTeacherRow, CseThermoRow,
    CseTimebaseRow, EventSource, MoleculeSource, ProteinSource, ProvenanceSource,
    ScientificStateSource, ThermoSource, TimebaseSource,
};
use prismcode::drf_active_state::{
    build_drf_active_state, compute_drf_active_state_hash, fail_if_invalid_drf_active_state,
    validate_drf_active_state, DrfActiveStateClass,
};
use prismcode::drf_fabric::build_drf_graph;
use prismcode::nrf_frame::build_nrf_frame;
use prismcode::nrf_sparse::build_nrf_sparse_frame;
use std::fs;
use std::path::PathBuf;

fn repo_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}

fn source_fixture() -> ScientificStateSource {
    ScientificStateSource {
        source_id: "h5-drf-active-fixture".to_string(),
        synthetic: true,
        generator_version: "h5-drf-active-v0.1".to_string(),
        seed: 20260608,
        molecule: MoleculeSource {
            entity_id: 100,
            name: "ligand".to_string(),
            topology_hash: "topology_ligand_hash".to_string(),
            atom_types: vec!["C".to_string(), "N".to_string()],
            bonds: vec![[0, 1]],
        },
        protein: ProteinSource {
            entity_id: 200,
            receptor: "protein".to_string(),
            chain_id: "A".to_string(),
            residues: vec!["GLY1".to_string(), "SER2".to_string()],
            atom_site_count: 2,
            topology_hash: "topology_protein_hash".to_string(),
        },
        waters: Vec::new(),
        timebase: TimebaseSource {
            time_id: 1,
            unit: "frame".to_string(),
            start: 0.0,
            step: 1.0,
            count: 3,
            timebase_hash: "canonical_timebase_hash_h5".to_string(),
            forcefield_parameter_hash: "forcefield_hash_h5".to_string(),
        },
        variants: Vec::new(),
        events: vec![
            EventSource {
                event_id: 10,
                event_kind: "DRF_FIRE".to_string(),
                source_ref: 100,
                target_ref: 200,
                time_ref: 1,
                time_delta: 0,
                motif: "DRF_RESONANCE_MOTIF".to_string(),
                amplitude: 0.25,
                unit: "unitless".to_string(),
                causal_parent_ref: None,
            },
            EventSource {
                event_id: 11,
                event_kind: "DRF_WRITEBACK".to_string(),
                source_ref: 200,
                target_ref: 100,
                time_ref: 1,
                time_delta: 1,
                motif: "UNCERTAINTY_SHIFT_MOTIF".to_string(),
                amplitude: 0.50,
                unit: "unitless".to_string(),
                causal_parent_ref: Some(10),
            },
        ],
        uncertainties: Vec::new(),
        thermo: vec![ThermoSource {
            thermo_id: 600,
            target_ref: 10,
            quantity_kind: "dG".to_string(),
            value: 0.1,
            unit_ref: "kcal_per_mol".to_string(),
            uncertainty_ref: 0,
        }],
        teacher_outputs: Vec::new(),
        gflownet_traces: Vec::new(),
        nanl_symbols: Vec::new(),
        provenance: ProvenanceSource {
            provenance_id: 900,
            label: "SYNTHETIC_H5_TEST".to_string(),
            synthetic: true,
            reason: "H5 DRF active-state boundary fixture".to_string(),
            source_hash: "raw_evidence_hash_h5".to_string(),
        },
    }
}

fn tables_for(source: &ScientificStateSource) -> CseComponentTables {
    CseComponentTables {
        schema_version: "PRISM_CSE_HD_TABLES_H5_TEST".to_string(),
        source_id: source.source_id.clone(),
        source_hash: blake3::hash(&serde_json::to_vec(source).unwrap())
            .to_hex()
            .to_string(),
        canonical_hash: blake3::hash(source.source_id.as_bytes())
            .to_hex()
            .to_string(),
        dictionary_hash: blake3::hash(b"h5-drf-dictionary").to_hex().to_string(),
        entity_table: vec![
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
                component_span_start: 2,
                component_span_len: source.protein.atom_site_count as u32,
            },
        ],
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
fn drf_active_state_validates_mechanism_sufficient_boundary() {
    let source = source_fixture();
    let tables = tables_for(&source);
    let frame = build_nrf_frame(&source, &tables).unwrap();
    let sparse = build_nrf_sparse_frame(&frame, &tables, &tables.provenance_hash).unwrap();
    let graph = build_drf_graph(&tables, &frame).unwrap();
    let state =
        build_drf_active_state(&graph, &sparse, DrfActiveStateClass::DrfMechanismSufficient)
            .unwrap();
    let report = validate_drf_active_state(&state, &graph, &sparse, &frame, &tables).unwrap();

    assert_eq!(report.status, "PASS");
    assert_eq!(report.classification, "DRF_MECHANISM_SUFFICIENT");
    assert!(report.raw_evidence_hash_available);
    assert!(report.cse_source_hash_bound);
    assert!(report.nrf_source_hash_bound);
    assert!(report.drf_hash_bound);
    assert!(report.classification_explicit);
    assert!(report.canonical_vs_derived_explicit);
    assert!(report.reconstruction_fidelity_pass);
    assert!(report.error_bound_pass);
    assert!(report.derived_not_lossless_raw);
    assert!(report.active_state_hash_pass);
    assert!(report.replay_hash_pass);

    let out_dir = repo_path("build/nanl_hd");
    fs::create_dir_all(&out_dir).unwrap();
    fs::write(
        out_dir.join("drf_active_state_report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
}

#[test]
fn drf_active_state_rejects_derived_as_lossless_raw_compression() {
    let source = source_fixture();
    let tables = tables_for(&source);
    let frame = build_nrf_frame(&source, &tables).unwrap();
    let sparse = build_nrf_sparse_frame(&frame, &tables, &tables.provenance_hash).unwrap();
    let graph = build_drf_graph(&tables, &frame).unwrap();
    let mut state =
        build_drf_active_state(&graph, &sparse, DrfActiveStateClass::DrfDerived).unwrap();
    state.lossless_raw_compression_claim = true;
    state.active_state_hash = compute_drf_active_state_hash(&state).unwrap();

    let err = fail_if_invalid_drf_active_state(&state, &graph, &sparse, &frame, &tables)
        .unwrap_err()
        .to_string();
    assert!(err.contains("derived DRF active state cannot claim lossless raw compression"));
}

#[test]
fn drf_active_state_rejects_missing_error_bound_and_raw_hash_mismatch() {
    let source = source_fixture();
    let tables = tables_for(&source);
    let frame = build_nrf_frame(&source, &tables).unwrap();
    let sparse = build_nrf_sparse_frame(&frame, &tables, &tables.provenance_hash).unwrap();
    let graph = build_drf_graph(&tables, &frame).unwrap();

    let mut state =
        build_drf_active_state(&graph, &sparse, DrfActiveStateClass::DrfMechanismSufficient)
            .unwrap();
    state.error_bound = None;
    state.active_state_hash = compute_drf_active_state_hash(&state).unwrap();
    let err = fail_if_invalid_drf_active_state(&state, &graph, &sparse, &frame, &tables)
        .unwrap_err()
        .to_string();
    assert!(err.contains("missing or violated reconstruction error bound"));

    let mut state =
        build_drf_active_state(&graph, &sparse, DrfActiveStateClass::DrfMechanismSufficient)
            .unwrap();
    state.raw_evidence_hash = "tampered_raw_hash".to_string();
    state.active_state_hash = compute_drf_active_state_hash(&state).unwrap();
    let err = fail_if_invalid_drf_active_state(&state, &graph, &sparse, &frame, &tables)
        .unwrap_err()
        .to_string();
    assert!(err.contains("raw evidence hash missing or not bound"));
}

#[test]
fn drf_active_state_rejects_source_hash_and_classification_label_tamper() {
    let source = source_fixture();
    let tables = tables_for(&source);
    let frame = build_nrf_frame(&source, &tables).unwrap();
    let sparse = build_nrf_sparse_frame(&frame, &tables, &tables.provenance_hash).unwrap();
    let graph = build_drf_graph(&tables, &frame).unwrap();

    let mut state =
        build_drf_active_state(&graph, &sparse, DrfActiveStateClass::DrfMechanismSufficient)
            .unwrap();
    state.source_nrf_sparse_hash = "tampered_sparse_hash".to_string();
    state.active_state_hash = compute_drf_active_state_hash(&state).unwrap();
    let err = fail_if_invalid_drf_active_state(&state, &graph, &sparse, &frame, &tables)
        .unwrap_err()
        .to_string();
    assert!(err.contains("source NRF hash mismatch"));

    let mut state =
        build_drf_active_state(&graph, &sparse, DrfActiveStateClass::DrfMechanismSufficient)
            .unwrap();
    state.classification_label = "DRF_DERIVED".to_string();
    state.active_state_hash = compute_drf_active_state_hash(&state).unwrap();
    let err = fail_if_invalid_drf_active_state(&state, &graph, &sparse, &frame, &tables)
        .unwrap_err()
        .to_string();
    assert!(err.contains("classification label mismatch"));
}
