use prismcode::cse_dictionaries::{CseDictionaries, CseDictionaryKind};
use prismcode::cse_hd::compile_cse_hd_from_json_str;

const REFERENCE: &str = include_str!("../../../examples/corpus/cse_hd/reference.pcsrc.json");
const NRF_EVENT_FRAMES: &str =
    include_str!("../../../examples/corpus/cse_hd/nrf_event_frames.pcsrc.json");
const DRF_GRAPH: &str = include_str!("../../../examples/corpus/cse_hd/drf_graph.pcsrc.json");

#[test]
fn cse_hd_dictionaries_are_sorted_deduped_and_row_refs_resolve() {
    let compilation = compile_cse_hd_from_json_str(REFERENCE).unwrap();
    let dictionaries = &compilation.dictionaries;

    assert_sorted_unique("entity", &dictionaries.entity_dictionary);
    assert_sorted_unique("atom_type", &dictionaries.atom_type_dictionary);
    assert_sorted_unique("residue", &dictionaries.residue_dictionary);
    assert_sorted_unique("event_kind", &dictionaries.event_kind_dictionary);
    assert_sorted_unique("event_motif", &dictionaries.event_motif_dictionary);
    assert_sorted_unique("water_gate", &dictionaries.water_gate_dictionary);
    assert_sorted_unique("variant", &dictionaries.variant_dictionary);
    assert_sorted_unique("thermo_quantity", &dictionaries.thermo_quantity_dictionary);
    assert_sorted_unique("unit", &dictionaries.unit_dictionary);
    assert_sorted_unique(
        "uncertainty_model",
        &dictionaries.uncertainty_model_dictionary,
    );
    assert_sorted_unique("teacher_label", &dictionaries.teacher_label_dictionary);
    assert_sorted_unique("gflownet_action", &dictionaries.gflownet_action_dictionary);
    assert_sorted_unique("nrf_tile", &dictionaries.nrf_tile_dictionary);
    assert_sorted_unique("drf_channel", &dictionaries.drf_channel_dictionary);
    assert_sorted_unique("nanl_symbol", &dictionaries.nanl_symbol_dictionary);
    assert_sorted_unique("provenance", &dictionaries.provenance_dictionary);

    for row in &compilation.tables.entity_rows {
        assert_dict_ref(
            dictionaries,
            CseDictionaryKind::Entity,
            &row.entity_id,
            row.entity_dict_ref,
        );
    }
    for row in &compilation.tables.atom_rows {
        assert_dict_ref(
            dictionaries,
            CseDictionaryKind::AtomType,
            &row.atom_type,
            row.atom_type_dict_ref,
        );
    }
    for row in &compilation.tables.residue_rows {
        assert_dict_ref(
            dictionaries,
            CseDictionaryKind::Residue,
            &row.name,
            row.residue_dict_ref,
        );
    }
    for row in &compilation.tables.event_rows {
        assert_dict_ref(
            dictionaries,
            CseDictionaryKind::EventKind,
            &row.event_kind,
            row.event_kind_dict_ref,
        );
        assert_dict_ref(
            dictionaries,
            CseDictionaryKind::EventMotif,
            &row.motif,
            row.event_motif_dict_ref,
        );
    }
    for row in &compilation.tables.thermo_rows {
        assert_dict_ref(
            dictionaries,
            CseDictionaryKind::ThermoQuantity,
            &row.quantity_kind,
            row.quantity_dict_ref,
        );
    }
}

#[test]
fn cse_hd_dictionaries_cover_nrf_tiles_drf_channels_and_science_labels() {
    let nrf = compile_cse_hd_from_json_str(NRF_EVENT_FRAMES).unwrap();
    assert_eq!(
        nrf.dictionaries
            .id_for(CseDictionaryKind::NrfTile, "phase_tile"),
        Some(nrf.tables.nrf_frame_rows[0].tile_dict_ref)
    );
    assert_eq!(
        nrf.dictionaries
            .id_for(CseDictionaryKind::DrfChannel, "resonance"),
        Some(nrf.tables.drf_state_rows[0].channel_dict_ref)
    );
    assert!(nrf
        .dictionaries
        .id_for(CseDictionaryKind::TeacherLabel, "NRF_EVENT_FIXTURE")
        .is_some());
    assert!(nrf
        .dictionaries
        .id_for(CseDictionaryKind::GFlowNetAction, "PROPOSE_CONTACT")
        .is_some());

    let drf = compile_cse_hd_from_json_str(DRF_GRAPH).unwrap();
    assert!(drf
        .dictionaries
        .id_for(CseDictionaryKind::DrfChannel, "resonance")
        .is_some());
    assert!(drf
        .dictionaries
        .id_for(CseDictionaryKind::DrfChannel, "uncertainty")
        .is_some());
    assert_eq!(drf.tables.drf_state_rows.len(), 2);
    assert_ne!(
        drf.tables.drf_state_rows[0].channel_dict_ref,
        drf.tables.drf_state_rows[1].channel_dict_ref
    );
}

fn assert_sorted_unique(label: &str, values: &[String]) {
    let mut expected = values.to_vec();
    expected.sort();
    expected.dedup();
    assert_eq!(
        expected.as_slice(),
        values,
        "{label} dictionary must be sorted and deduped"
    );
}

fn assert_dict_ref(
    dictionaries: &CseDictionaries,
    kind: CseDictionaryKind,
    symbol: &str,
    actual: u32,
) {
    assert_eq!(
        dictionaries.id_for(kind, symbol),
        Some(actual),
        "{kind:?} dictionary ref for {symbol}"
    );
}
