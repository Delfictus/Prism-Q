use prismcode::cse_tables::{
    CseComponentTables, CseEntityRow, CseEventRow, CseGFlowNetRow, CseTeacherRow, CseThermoRow,
    CseTimebaseRow, EventSource, MoleculeSource, ProteinSource, ProvenanceSource,
    ScientificStateSource, ThermoSource, TimebaseSource,
};
use prismcode::nrf_frame::build_nrf_frame;
use prismcode::nrf_sparse::{
    build_nrf_sparse_frame, fail_if_invalid_nrf_sparse, validate_nrf_sparse_frame,
};
use std::fs;
use std::path::PathBuf;

fn repo_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}

fn source_fixture() -> ScientificStateSource {
    ScientificStateSource {
        source_id: "h5-nrf-sparse-fixture".to_string(),
        synthetic: true,
        generator_version: "h5-nrf-sparse-v0.1".to_string(),
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
                event_kind: "WATER_GATE".to_string(),
                source_ref: 100,
                target_ref: 200,
                time_ref: 1,
                time_delta: 0,
                motif: "WATER_GATE_MOTIF".to_string(),
                amplitude: 0.25,
                unit: "unitless".to_string(),
                causal_parent_ref: None,
            },
            EventSource {
                event_id: 11,
                event_kind: "RESIDUE_COUPLING".to_string(),
                source_ref: 200,
                target_ref: 100,
                time_ref: 1,
                time_delta: 1,
                motif: "RESIDUE_COUPLING_MOTIF".to_string(),
                amplitude: 0.50,
                unit: "unitless".to_string(),
                causal_parent_ref: Some(10),
            },
            EventSource {
                event_id: 12,
                event_kind: "THERMO_PULSE".to_string(),
                source_ref: 100,
                target_ref: 200,
                time_ref: 1,
                time_delta: 2,
                motif: "THERMO_PULSE_MOTIF".to_string(),
                amplitude: 0.75,
                unit: "unitless".to_string(),
                causal_parent_ref: Some(11),
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
            reason: "H5 NRF sparse boundary fixture".to_string(),
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
        dictionary_hash: blake3::hash(b"h5-nrf-dictionary").to_hex().to_string(),
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
fn nrf_sparse_build_validate_and_binds_cse_nrf_timebase() {
    let source = source_fixture();
    let tables = tables_for(&source);
    let frame = build_nrf_frame(&source, &tables).unwrap();
    let sparse = build_nrf_sparse_frame(&frame, &tables, &tables.provenance_hash).unwrap();
    let report = validate_nrf_sparse_frame(&sparse, &frame, &tables).unwrap();

    assert_eq!(report.status, "PASS");
    assert!(report.cse_hash_bound);
    assert!(report.raw_evidence_hash_available);
    assert!(report.nrf_hash_bound);
    assert!(report.canonical_timebase_bound);
    assert!(report.event_refs_resolve);
    assert!(report.sparse_index_bounds_checked);
    assert!(report.motif_hashes_bound);
    assert!(report.causal_graph_acyclic);
    assert!(report.event_order_canonical);
    assert!(report.reconstruct_event_hashes_pass);
    assert!(report.replay_hash_pass);

    let out_dir = repo_path("build/nanl_hd");
    fs::create_dir_all(&out_dir).unwrap();
    fs::write(
        out_dir.join("nrf_sparse_report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
}

#[test]
fn nrf_sparse_rejects_missing_raw_evidence_and_timebase_tamper() {
    let source = source_fixture();
    let tables = tables_for(&source);
    let frame = build_nrf_frame(&source, &tables).unwrap();
    let sparse = build_nrf_sparse_frame(&frame, &tables, "").unwrap();
    let err = fail_if_invalid_nrf_sparse(&sparse, &frame, &tables)
        .unwrap_err()
        .to_string();
    assert!(err.contains("raw evidence hash missing"));

    let mut sparse = build_nrf_sparse_frame(&frame, &tables, &tables.provenance_hash).unwrap();
    sparse.canonical_timebase_hash = "tampered_timebase".to_string();
    let err = fail_if_invalid_nrf_sparse(&sparse, &frame, &tables)
        .unwrap_err()
        .to_string();
    assert!(err.contains("canonical timebase binding failed"));
}

#[test]
fn nrf_sparse_rejects_causal_cycle_motif_hash_and_sparse_index_tamper() {
    let source = source_fixture();
    let tables = tables_for(&source);
    let frame = build_nrf_frame(&source, &tables).unwrap();
    let mut sparse = build_nrf_sparse_frame(&frame, &tables, &tables.provenance_hash).unwrap();
    sparse.events[0].causal_parent_ref = Some(12);
    sparse.events[2].causal_parent_ref = Some(10);
    let err = fail_if_invalid_nrf_sparse(&sparse, &frame, &tables)
        .unwrap_err()
        .to_string();
    assert!(err.contains("causal graph contains a cycle"));

    let mut sparse = build_nrf_sparse_frame(&frame, &tables, &tables.provenance_hash).unwrap();
    sparse.motifs[0].motif_hash = "tampered_motif".to_string();
    let err = fail_if_invalid_nrf_sparse(&sparse, &frame, &tables)
        .unwrap_err()
        .to_string();
    assert!(err.contains("motif hash binding failed"));

    let mut sparse = build_nrf_sparse_frame(&frame, &tables, &tables.provenance_hash).unwrap();
    sparse.sparse_index[0].col = 99;
    let err = fail_if_invalid_nrf_sparse(&sparse, &frame, &tables)
        .unwrap_err()
        .to_string();
    assert!(err.contains("sparse index out of bounds"));
}
