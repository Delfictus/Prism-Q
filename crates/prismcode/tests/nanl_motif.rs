#[path = "../src/nanl_motif.rs"]
mod nanl_motif;

use nanl_motif::{
    decode_codes, decode_with_motifs, discover_motifs, discover_numeric_motifs,
    encode_observations, encode_with_motifs, motif_to_nrf, validate_motif_book,
    validate_required_motif_classes, MotifDiscoveryPolicy, MotifKind, NanlMotifDiscoveryConfig,
    NanlMotifError, NanlMotifKind, NanlMotifObservation, REQUIRED_MOTIF_KINDS,
};

#[test]
fn nanl_motif_discovers_deterministic_required_classes_and_roundtrips() {
    let observations = required_observations();
    let reversed = {
        let mut items = observations.clone();
        items.reverse();
        items
    };

    let book = discover_motifs(
        "cse-hash-motif-001",
        "event-hash-motif-001",
        &observations,
        NanlMotifDiscoveryConfig::default(),
    )
    .unwrap();
    let book_reordered = discover_motifs(
        "cse-hash-motif-001",
        "event-hash-motif-001",
        &reversed,
        NanlMotifDiscoveryConfig::default(),
    )
    .unwrap();

    let report = validate_motif_book(&book).unwrap();
    assert_eq!(report.status, "PASS");
    assert!(report.stable_ordering);
    assert!(report.deterministic_tie_breaker);
    assert!(report.required_motif_classes_present);
    assert_eq!(book.header.merkle_root, book_reordered.header.merkle_root);
    assert_eq!(
        book.header.motif_dictionary_hash,
        book_reordered.header.motif_dictionary_hash
    );

    validate_required_motif_classes(&book).unwrap();
    let encoded = encode_observations(&book, &observations).unwrap();
    let decoded = decode_codes(&book, &encoded).unwrap();
    assert_eq!(decoded, observations);

    let nrf = motif_to_nrf(&book).unwrap();
    assert_eq!(nrf.len(), REQUIRED_MOTIF_KINDS.len());
    assert!(nrf.iter().all(|item| item.occurrence_count == 2));
}

#[test]
fn nanl_motif_rejects_missing_source_hash() {
    let observations = required_observations();
    let err = discover_motifs(
        "",
        "event-hash-present",
        &observations,
        NanlMotifDiscoveryConfig::default(),
    )
    .unwrap_err();
    assert!(matches!(err, NanlMotifError::Format(_)));
    assert!(err.to_string().contains("source CSE hash"));

    let mut book = discover_motifs(
        "cse-hash-present",
        "event-hash-present",
        &observations,
        NanlMotifDiscoveryConfig::default(),
    )
    .unwrap();
    book.header.source_event_hash.clear();
    let err = book.fail_if_invalid().unwrap_err();
    assert!(err.to_string().contains("source event hash"));
}

#[test]
fn nanl_motif_rejects_bad_motif_hash() {
    let mut book = discover_motifs(
        "cse-hash-bad-motif",
        "event-hash-bad-motif",
        &required_observations(),
        NanlMotifDiscoveryConfig::default(),
    )
    .unwrap();
    book.motifs[0].motif_hash = "bad-hash".to_string();

    let err = book.fail_if_invalid().unwrap_err();
    assert!(matches!(err, NanlMotifError::Hash(_)));
    assert!(err.to_string().contains("bad motif hash"));
}

#[test]
fn nanl_motif_rejects_insufficient_support_and_gain() {
    let mut config = NanlMotifDiscoveryConfig::default();
    config.minimum_support_threshold = 3;
    let err = discover_motifs(
        "cse-hash-support",
        "event-hash-support",
        &required_observations(),
        config,
    )
    .unwrap_err();
    assert!(matches!(err, NanlMotifError::Support(_)));

    let mut config = NanlMotifDiscoveryConfig::default();
    config.minimum_compression_gain_threshold = 10_000.0;
    let err = discover_motifs(
        "cse-hash-gain",
        "event-hash-gain",
        &required_observations(),
        config,
    )
    .unwrap_err();
    assert!(matches!(err, NanlMotifError::Support(_)));
}

#[test]
fn nanl_motif_rejects_unstable_ordering_and_unknown_motif() {
    let book = discover_motifs(
        "cse-hash-order",
        "event-hash-order",
        &required_observations(),
        NanlMotifDiscoveryConfig::default(),
    )
    .unwrap();
    let mut unordered = book.clone();
    unordered.motifs.swap(0, 1);
    let err = unordered.fail_if_invalid().unwrap_err();
    assert!(matches!(err, NanlMotifError::Ordering(_)));
    assert!(err.to_string().contains("stable ordering"));

    let unknown = vec![NanlMotifObservation::new(
        999,
        NanlMotifKind::WaterGateMotif,
        vec!["unseen-water-gate-symbol".to_string()],
        vec![999],
        1,
        2,
        3,
    )];
    let err = encode_observations(&book, &unknown).unwrap_err();
    assert!(matches!(err, NanlMotifError::UnknownMotif(_)));
}

#[test]
fn nanl_motif_numeric_compatibility_api_roundtrips_benchmark_symbols() {
    let symbols = vec![11, 7, 11, 7, 11, 7, 99, 42, 99, 42, 99, 42, 7, 7, 7];
    let motifs = discover_numeric_motifs(
        &symbols,
        MotifKind::WaterGate,
        MotifDiscoveryPolicy {
            minimum_support: 3,
            minimum_compression_gain: 1,
            maximum_motif_length: 4,
            maximum_parameter_count: 8,
        },
    )
    .unwrap();
    let encoded = encode_with_motifs(&symbols, &motifs).unwrap();
    let decoded = decode_with_motifs(&encoded, &motifs).unwrap();

    assert!(!motifs.is_empty());
    assert!(encoded.len() < symbols.len());
    assert_eq!(decoded, symbols);
}

fn required_observations() -> Vec<NanlMotifObservation> {
    let mut observations = Vec::new();
    let mut id = 1;
    for (idx, kind) in REQUIRED_MOTIF_KINDS.iter().enumerate() {
        let symbols = symbols_for(*kind);
        for repeat in 0..2 {
            observations.push(NanlMotifObservation::new(
                id,
                *kind,
                symbols.clone(),
                vec![idx as u64 + 100],
                repeat + 1,
                idx as u64 + 10,
                idx as u64 + 20,
            ));
            id += 1;
        }
    }
    observations
}

fn symbols_for(kind: NanlMotifKind) -> Vec<String> {
    let label = match kind {
        NanlMotifKind::WaterGateMotif => "water_gate_open_close_hydration_shell_cycle",
        NanlMotifKind::ResidueCouplingMotif => "residue_coupling_lys_asp_contact_bridge",
        NanlMotifKind::ThermoPulseMotif => "thermo_pulse_delta_g_entropy_compensation",
        NanlMotifKind::VariantTransitionMotif => "variant_transition_reference_alternate_delta",
        NanlMotifKind::DrfResonanceMotif => "drf_resonance_fire_decay_recover_channel",
        NanlMotifKind::GFlowNetActionMotif => "gflownet_action_propose_contact_reward_flow",
        NanlMotifKind::TeacherDisagreementMotif => "teacher_disagreement_label_uncertainty_airlock",
        NanlMotifKind::UncertaintyShiftMotif => "uncertainty_shift_epistemic_aleatoric_calibration",
    };
    vec![label.to_string(), format!("{label}_phase_1")]
}
