use prismcode::cse_hd::{compile_cse_hd_from_json_str, validate_roundtrip_reconstruction};
use prismcode::cse_tables::CseSourceDocument;

const REFERENCE: &str = include_str!("../../../examples/corpus/cse_hd/reference.pcsrc.json");
const PROTEIN_WATER_VARIANT: &str =
    include_str!("../../../examples/corpus/cse_hd/protein_water_variant.pcsrc.json");
const NRF_EVENT_FRAMES: &str =
    include_str!("../../../examples/corpus/cse_hd/nrf_event_frames.pcsrc.json");
const DRF_GRAPH: &str = include_str!("../../../examples/corpus/cse_hd/drf_graph.pcsrc.json");

#[test]
fn reference_fixture_builds_typed_cse_hd_component_tables() {
    let source: CseSourceDocument = serde_json::from_str(REFERENCE).unwrap();
    let compilation = compile_cse_hd_from_json_str(REFERENCE).unwrap();
    let tables = &compilation.tables;

    assert_eq!(compilation.validation.status, "PASS");
    assert_eq!(compilation.validation.state_id, "cse_hd_reference_v0_1");
    assert_eq!(tables.schema_rows.len(), 1);
    assert_eq!(tables.unit_rows.len(), 3);
    assert_eq!(tables.provenance_rows.len(), 1);
    assert_eq!(tables.timebase_rows.len(), 1);
    assert_eq!(tables.entity_rows.len(), 3);
    assert_eq!(tables.atom_rows.len(), 3);
    assert_eq!(tables.bond_rows.len(), 2);
    assert_eq!(tables.protein_chain_rows.len(), 1);
    assert_eq!(tables.residue_rows.len(), 2);
    assert_eq!(tables.atom_site_rows.len(), 2);
    assert_eq!(tables.water_site_rows.len(), 1);
    assert_eq!(tables.variant_rows.len(), 1);
    assert_eq!(tables.event_rows.len(), 2);
    assert_eq!(tables.uncertainty_rows.len(), 1);
    assert_eq!(tables.thermo_rows.len(), 1);
    assert_eq!(tables.teacher_output_rows.len(), 1);
    assert_eq!(tables.gflownet_trace_rows.len(), 1);
    assert_eq!(tables.nanl_symbol_rows.len(), 6);
    assert_eq!(tables.nrf_frame_rows.len(), 1);
    assert_eq!(tables.drf_state_rows.len(), 1);

    let water = &tables.water_site_rows[0];
    assert_eq!(water.water_id, "water_gate_1");
    assert_eq!(water.time_ref.as_deref(), Some("timebase_reference"));
    assert_eq!(
        water.provenance_ref.as_deref(),
        Some("prov_reference_synthetic")
    );

    let event_ids = tables
        .event_rows
        .iter()
        .map(|row| row.event_id.as_str())
        .collect::<Vec<_>>();
    assert_eq!(event_ids, ["event_contact_on", "event_water_gate"]);
    assert_eq!(tables.event_rows[1].source_ref, "water_gate_1");

    let drf = &tables.drf_state_rows[0];
    assert_eq!(drf.source_nrf_ref, "nrf_frame_reference_0");
    assert_eq!(drf.source_cse_ref, "cse_hd_reference_v0_1");
    assert_eq!(drf.tensor_shape, [2, 4]);
    assert_eq!(drf.canonical_or_derived, "DRF_CANONICAL");

    let roundtrip = validate_roundtrip_reconstruction(&source).unwrap();
    assert_eq!(roundtrip.status, "PASS");
    assert_eq!(roundtrip.reconstructed_entity_count, 3);
    assert_eq!(roundtrip.reconstructed_event_count, 2);
    assert!(roundtrip.canonical_roundtrip_hash_pass);
    assert_eq!(roundtrip.canonical_hash, compilation.canonical_hash);
}

#[test]
fn cse_hd_corpus_fixtures_compile_to_real_nonempty_tables() {
    for (name, raw, expected_events, expected_drf) in [
        ("reference", REFERENCE, 2, 1),
        ("protein_water_variant", PROTEIN_WATER_VARIANT, 1, 1),
        ("nrf_event_frames", NRF_EVENT_FRAMES, 3, 1),
        ("drf_graph", DRF_GRAPH, 2, 2),
    ] {
        let source: CseSourceDocument = serde_json::from_str(raw).unwrap();
        let compilation = compile_cse_hd_from_json_str(raw).unwrap();

        assert_eq!(compilation.validation.status, "PASS", "{name}");
        assert_eq!(
            compilation.validation.event_count, expected_events,
            "{name}"
        );
        assert_eq!(
            compilation.validation.drf_state_count, expected_drf,
            "{name}"
        );
        assert_eq!(compilation.canonical_hash.len(), 64, "{name}");
        assert_eq!(compilation.dictionaries_hash.len(), 64, "{name}");
        assert_eq!(compilation.tables_hash.len(), 64, "{name}");
        assert!(!compilation.tables.entity_rows.is_empty(), "{name}");
        assert!(!compilation.tables.event_rows.is_empty(), "{name}");
        assert!(!compilation.tables.nrf_frame_rows.is_empty(), "{name}");
        assert!(!compilation.tables.drf_state_rows.is_empty(), "{name}");

        let roundtrip = validate_roundtrip_reconstruction(&source).unwrap();
        assert!(roundtrip.canonical_roundtrip_hash_pass, "{name}");
        assert_eq!(
            roundtrip.canonical_hash, compilation.canonical_hash,
            "{name}"
        );
    }
}
