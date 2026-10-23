#[path = "../src/nanl_dict.rs"]
mod nanl_dict;

use nanl_dict::{
    EncodedNanlDictToken, NanlDictError, NanlDictionary, NanlDictionaryClass, NanlDictionarySymbol,
    REQUIRED_DICTIONARY_CLASSES,
};

#[test]
fn nanl_dict_builds_stable_ids_hashes_and_roundtrips() {
    let mut symbols = required_symbols();
    symbols.push(NanlDictionarySymbol::new(
        NanlDictionaryClass::Entity,
        99,
        "ligand-A",
        4,
        99,
    ));
    let mut reordered_symbols = symbols.clone();
    reordered_symbols.reverse();

    let dictionary = NanlDictionary::build("PRISM_NATIVE", "cse-hash-001", 7, &symbols).unwrap();
    let dictionary_reordered =
        NanlDictionary::build("PRISM_NATIVE", "cse-hash-001", 7, &reordered_symbols).unwrap();
    let report = dictionary.validate().unwrap();

    assert_eq!(report.status, "PASS");
    assert!(report.stable_ordering);
    assert!(report.required_classes_present);
    assert!(report.lookup_table_valid);
    assert!(report.reverse_lookup_valid);
    assert_eq!(
        dictionary.header.dictionary_hash,
        dictionary_reordered.header.dictionary_hash
    );

    let encoded = dictionary
        .encode_tokens(&[
            (NanlDictionaryClass::Entity, "ligand-A".to_string()),
            (NanlDictionaryClass::EventKind, "EVENT_BIND".to_string()),
            (
                NanlDictionaryClass::GFlowNetAction,
                "PROPOSE_CONTACT".to_string(),
            ),
        ])
        .unwrap();
    let decoded = dictionary.decode_tokens(&encoded).unwrap();

    assert_eq!(
        decoded,
        vec![
            "ligand-A".to_string(),
            "EVENT_BIND".to_string(),
            "PROPOSE_CONTACT".to_string()
        ]
    );
    assert_eq!(
        dictionary
            .reverse_lookup(encoded[0].symbol_id)
            .unwrap()
            .token,
        "ligand-A"
    );
}

#[test]
fn nanl_dict_rejects_unknown_token() {
    let dictionary =
        NanlDictionary::build("PRISM_NATIVE", "cse-hash-unknown", 1, &required_symbols()).unwrap();
    let err = dictionary
        .encode_tokens(&[(NanlDictionaryClass::Entity, "missing-entity".to_string())])
        .unwrap_err();

    assert!(matches!(err, NanlDictError::UnknownToken(_)));
    assert!(err.to_string().contains("PC_ERR_UNKNOWN_TOKEN"));
}

#[test]
fn nanl_dict_rejects_unstable_ordering() {
    let mut dictionary =
        NanlDictionary::build("PRISM_NATIVE", "cse-hash-order", 1, &required_symbols()).unwrap();
    dictionary.entries.swap(0, 1);

    let err = dictionary.fail_if_invalid().unwrap_err();
    assert!(matches!(err, NanlDictError::Ordering(_)));
    assert!(err.to_string().contains("stable ordering"));
}

#[test]
fn nanl_dict_rejects_missing_source_hash() {
    let err = NanlDictionary::build("PRISM_NATIVE", "", 1, &required_symbols()).unwrap_err();
    assert!(matches!(err, NanlDictError::Format(_)));
    assert!(err.to_string().contains("canonical source hash"));

    let mut dictionary =
        NanlDictionary::build("PRISM_NATIVE", "cse-hash-present", 1, &required_symbols()).unwrap();
    dictionary.header.canonical_source_hash.clear();
    let err = dictionary.fail_if_invalid().unwrap_err();
    assert!(err.to_string().contains("canonical source hash"));
}

#[test]
fn nanl_dict_rejects_bad_reverse_lookup_token() {
    let dictionary =
        NanlDictionary::build("PRISM_NATIVE", "cse-hash-decode", 1, &required_symbols()).unwrap();
    let mut encoded = dictionary
        .encode_tokens(&[(NanlDictionaryClass::Residue, "LYS".to_string())])
        .unwrap();
    encoded[0] = EncodedNanlDictToken {
        symbol_id: encoded[0].symbol_id,
        encoded_value: encoded[0].encoded_value + 1,
        symbol_class: encoded[0].symbol_class,
    };

    let err = dictionary.decode_tokens(&encoded).unwrap_err();
    assert!(err.to_string().contains("encoded token mismatch"));
}

fn required_symbols() -> Vec<NanlDictionarySymbol> {
    REQUIRED_DICTIONARY_CLASSES
        .iter()
        .enumerate()
        .map(|(idx, class)| {
            NanlDictionarySymbol::new(
                *class,
                idx as u64 + 10,
                token_for(*class),
                idx as u64 + 1,
                idx as u64 + 100,
            )
        })
        .collect()
}

fn token_for(class: NanlDictionaryClass) -> &'static str {
    match class {
        NanlDictionaryClass::Entity => "ligand-A",
        NanlDictionaryClass::AtomType => "C.ar",
        NanlDictionaryClass::Residue => "LYS",
        NanlDictionaryClass::EventKind => "EVENT_BIND",
        NanlDictionaryClass::EventMotif => "motif-water-gate",
        NanlDictionaryClass::WaterGate => "gate-open",
        NanlDictionaryClass::Variant => "variant-K42A",
        NanlDictionaryClass::ThermoQuantity => "delta_g",
        NanlDictionaryClass::UncertaintyModel => "calibrated-ensemble",
        NanlDictionaryClass::TeacherLabel => "teacher-agree",
        NanlDictionaryClass::GFlowNetAction => "PROPOSE_CONTACT",
        NanlDictionaryClass::NrfTile => "phase_tile",
        NanlDictionaryClass::DrfChannel => "resonance",
        NanlDictionaryClass::NanlSymbol => "NANL_EVENT_TOKEN",
    }
}
