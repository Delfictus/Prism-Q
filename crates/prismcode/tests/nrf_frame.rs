use prismcode::cse_tables::{
    CseComponentTables, CseEntityRow, CseEventRow, CseGFlowNetRow, CseSourceDocument,
    CseTeacherRow, CseThermoRow, CseTimebaseRow, EventSource, GFlowNetTraceSource, MoleculeSource,
    ProteinSource, ProvenanceSource, ScientificStateSource, TeacherOutputSource, ThermoSource,
    TimebaseSource, UncertaintySource, VariantSource, WaterSource,
};
use prismcode::nrf_frame::{
    build_nrf_frame, fail_if_invalid, reconstruct_nrf_events, validate_nrf_frame,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

fn repo_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}

fn load_source() -> ScientificStateSource {
    let bytes = fs::read(repo_path(
        "examples/corpus/cse_hd/nrf_event_frames.pcsrc.json",
    ))
    .unwrap();
    let document: CseSourceDocument = serde_json::from_slice(&bytes).unwrap();
    source_from_cse_document(document, 100, 200, 300, 1, 10)
}

fn source_from_cse_document(
    document: CseSourceDocument,
    molecule_id: u64,
    protein_id: u64,
    water_base: u64,
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
    for (idx, water) in document.water_sites.iter().enumerate() {
        state_refs.insert(water.id.as_str(), water_base + idx as u64);
    }
    let mut ordered_events = document.events.clone();
    ordered_events.sort_by_key(|event| event.order_index);

    ScientificStateSource {
        source_id: document.state_id,
        synthetic: document.synthetic,
        generator_version: "prism-cse-hd-nrf-fixture-v0.1".to_string(),
        seed: 1337,
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
        waters: document
            .water_sites
            .iter()
            .enumerate()
            .map(|(idx, water)| WaterSource {
                site_id: water_base + idx as u64,
                shell: water.id.clone(),
                gate_state: water.gate_kind.clone(),
                time_ref: time_id,
                provenance_ref: 900,
            })
            .collect(),
        timebase: TimebaseSource {
            time_id,
            unit: timebase.unit_ref.clone(),
            start: timebase.frame_start as f64,
            step: 1.0,
            count: timebase.frame_end.saturating_sub(timebase.frame_start),
            timebase_hash: timebase.timebase_hash.clone(),
            forcefield_parameter_hash: "forcefield_hash_synthetic_nrf_v0".to_string(),
        },
        variants: document
            .variants
            .iter()
            .enumerate()
            .map(|(idx, variant)| VariantSource {
                variant_id: 400 + idx as u64,
                reference_state_ref: *state_refs
                    .get(variant.reference_state_ref.as_str())
                    .unwrap_or(&molecule_id),
                alternate_state_ref: *state_refs
                    .get(variant.alternate_state_ref.as_str())
                    .unwrap_or(&protein_id),
                delta_kind: variant.delta_hash.clone(),
                provenance_ref: 900,
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
                amplitude: 0.25 + idx as f64 * 0.25,
                unit: "unitless".to_string(),
                causal_parent_ref: (idx > 0).then_some(event_base + idx as u64 - 1),
            })
            .collect(),
        uncertainties: document
            .uncertainties
            .iter()
            .enumerate()
            .map(|(idx, uncertainty)| UncertaintySource {
                uncertainty_id: 500 + idx as u64,
                target_ref: event_base,
                distribution: uncertainty.model.clone(),
                calibration: uncertainty.calibration_hash.clone(),
                value: 0.05,
            })
            .collect(),
        thermo: document
            .thermo
            .iter()
            .enumerate()
            .map(|(idx, thermo)| ThermoSource {
                thermo_id: 600 + idx as u64,
                target_ref: event_base,
                quantity_kind: thermo.quantity_kind.clone(),
                value: 0.15,
                unit_ref: thermo.unit_ref.clone(),
                uncertainty_ref: 500,
            })
            .collect(),
        teacher_outputs: document
            .teacher_outputs
            .iter()
            .enumerate()
            .map(|(idx, teacher)| TeacherOutputSource {
                output_id: 700 + idx as u64,
                model_id: teacher.model_ref.clone(),
                label: teacher.label.clone(),
                confidence: teacher.confidence,
                uncertainty_ref: 500,
                provenance_ref: 900,
                airlock_state: teacher.airlock_state.clone(),
            })
            .collect(),
        gflownet_traces: document
            .gflownet_traces
            .iter()
            .enumerate()
            .map(|(idx, trace)| GFlowNetTraceSource {
                trace_id: 800 + idx as u64,
                state_ref: *state_refs
                    .get(trace.state_ref.as_str())
                    .unwrap_or(&molecule_id),
                action_sequence: vec![trace.action.clone()],
                reward: 0.4,
                flow: 0.2,
                policy_hash: trace.lineage_hash.clone(),
                provenance_ref: 900,
            })
            .collect(),
        nanl_symbols: document.nanl_symbols,
        provenance: ProvenanceSource {
            provenance_id: 900,
            label: provenance.classification.clone(),
            synthetic: provenance.synthetic,
            reason: "deterministic NRF sparse event-frame validation fixture".to_string(),
            source_hash: provenance.source_hash.clone(),
        },
    }
}

fn tables_for(source: &ScientificStateSource) -> CseComponentTables {
    let source_hash = blake3::hash(&serde_json::to_vec(source).unwrap())
        .to_hex()
        .to_string();
    let entity_table = std::iter::once(CseEntityRow {
        entity_id: source.molecule.entity_id,
        entity_kind: "MOLECULE".to_string(),
        dictionary_ref: 0,
        topology_hash: source.molecule.topology_hash.clone(),
        component_span_start: 0,
        component_span_len: source.molecule.atom_types.len() as u32,
    })
    .chain(std::iter::once(CseEntityRow {
        entity_id: source.protein.entity_id,
        entity_kind: "PROTEIN".to_string(),
        dictionary_ref: 1,
        topology_hash: source.protein.topology_hash.clone(),
        component_span_start: source.molecule.atom_types.len() as u32,
        component_span_len: source.protein.atom_site_count as u32,
    }))
    .chain(source.waters.iter().map(|water| CseEntityRow {
        entity_id: water.site_id,
        entity_kind: "WATER".to_string(),
        dictionary_ref: 2,
        topology_hash: format!("water:{}:{}", water.shell, water.gate_state),
        component_span_start: 0,
        component_span_len: 1,
    }))
    .collect::<Vec<_>>();
    CseComponentTables {
        schema_version: "PRISM_CSE_HD_TABLES_TEST".to_string(),
        source_id: source.source_id.clone(),
        source_hash,
        canonical_hash: blake3::hash(source.source_id.as_bytes())
            .to_hex()
            .to_string(),
        dictionary_hash: blake3::hash(b"nrf-drf-dictionary").to_hex().to_string(),
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
        teacher_output_table: source
            .teacher_outputs
            .iter()
            .map(|teacher| CseTeacherRow {
                output_id: teacher.output_id,
                model_ref: 0,
                label_ref: 0,
                confidence: teacher.confidence,
                uncertainty_ref: teacher.uncertainty_ref,
                airlock_state: teacher.airlock_state.clone(),
            })
            .collect(),
        gflownet_trace_table: source
            .gflownet_traces
            .iter()
            .map(|trace| CseGFlowNetRow {
                trace_id: trace.trace_id,
                state_ref: trace.state_ref,
                action_refs: vec![0],
                reward: trace.reward,
                flow: trace.flow,
                policy_hash: trace.policy_hash.clone(),
            })
            .collect(),
        provenance_hash: source.provenance.source_hash.clone(),
    }
}

#[test]
fn nrf_build_validate_and_reconstructs_replay_hash() {
    let source = load_source();
    let tables = tables_for(&source);
    let frame = build_nrf_frame(&source, &tables).unwrap();
    let report = validate_nrf_frame(&frame, &tables).unwrap();
    let reconstruction = reconstruct_nrf_events(&frame, &tables).unwrap();

    assert_eq!(report.status, "PASS");
    assert_eq!(report.event_count, 3);
    assert_eq!(
        report.canonical_timebase_hash,
        source.timebase.timebase_hash
    );
    assert_eq!(report.replay_hash, frame.replay_hash);
    assert!(report.reconstruct_event_hashes_pass);
    assert!(reconstruction.canonical_roundtrip_hash_pass);
    assert!(reconstruction.replay_hash_pass);
    assert_eq!(reconstruction.reconstructed_event_hashes.len(), 3);

    let out_dir = repo_path("build/demo");
    fs::create_dir_all(&out_dir).unwrap();
    fs::write(
        out_dir.join("nrf_frame_validation.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
}

#[test]
fn nrf_rejects_timebase_hash_mismatch() {
    let source = load_source();
    let tables = tables_for(&source);
    let mut frame = build_nrf_frame(&source, &tables).unwrap();
    frame.canonical_timebase_hash = "tampered_timebase".to_string();
    let err = fail_if_invalid(&frame, &tables).unwrap_err().to_string();
    assert!(err.contains("PC_ERR_FORMAT"));
    assert!(err.contains("canonical timebase hash mismatch"));
}

#[test]
fn nrf_rejects_future_causal_parent_and_hash_tamper() {
    let mut source = load_source();
    source.events[0].causal_parent_ref = Some(11);
    let tables = tables_for(&source);
    let frame = build_nrf_frame(&source, &tables).unwrap();
    let err = fail_if_invalid(&frame, &tables).unwrap_err().to_string();
    assert!(err.contains("invalid NRF causal parent"));

    let source = load_source();
    let tables = tables_for(&source);
    let mut frame = build_nrf_frame(&source, &tables).unwrap();
    frame.events[0].event_kind = "TAMPERED".to_string();
    let err = fail_if_invalid(&frame, &tables).unwrap_err().to_string();
    assert!(err.contains("NRF event hash reconstruction failed"));
}
