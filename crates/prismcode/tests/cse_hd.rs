use prismcode::cse_hd::{
    compile_cse_hd_from_json_str, validate_roundtrip_reconstruction, CseError,
};
use prismcode::cse_tables::CseSourceDocument;
use serde_json::{json, Value};

const REFERENCE: &str = include_str!("../../../examples/corpus/cse_hd/reference.pcsrc.json");
const PROTEIN_WATER_VARIANT: &str =
    include_str!("../../../examples/corpus/cse_hd/protein_water_variant.pcsrc.json");

#[test]
fn compiles_canonical_cse_hd_reference_and_roundtrips_hash() {
    let source: CseSourceDocument = serde_json::from_str(REFERENCE).unwrap();
    let compilation = compile_cse_hd_from_json_str(REFERENCE).unwrap();
    assert_eq!(compilation.validation.status, "PASS");
    assert_eq!(compilation.validation.molecule_count, 1);
    assert_eq!(compilation.validation.protein_count, 1);
    assert_eq!(compilation.validation.water_site_count, 1);
    assert_eq!(compilation.validation.event_count, 2);
    assert_eq!(compilation.validation.thermo_count, 1);
    assert_eq!(compilation.validation.nrf_frame_count, 1);
    assert_eq!(compilation.validation.drf_state_count, 1);
    assert_eq!(compilation.canonical_hash.len(), 64);
    assert!(!compilation.tables.entity_rows.is_empty());
    assert!(!compilation.tables.atom_rows.is_empty());
    assert!(!compilation.tables.event_rows.is_empty());

    let roundtrip = validate_roundtrip_reconstruction(&source).unwrap();
    assert_eq!(roundtrip.status, "PASS");
    assert!(roundtrip.canonical_roundtrip_hash_pass);
    assert_eq!(roundtrip.canonical_hash, roundtrip.reconstructed_hash);

    let build_demo = repo_build_demo();
    std::fs::create_dir_all(&build_demo).unwrap();
    std::fs::write(
        build_demo.join("cse_hd_reference_compile_report.json"),
        serde_json::to_vec_pretty(&compilation).unwrap(),
    )
    .unwrap();
    std::fs::write(
        build_demo.join("cse_hd_reference_roundtrip_report.json"),
        serde_json::to_vec_pretty(&roundtrip).unwrap(),
    )
    .unwrap();
}

#[test]
fn compiles_protein_water_variant_fixture() {
    let source: CseSourceDocument = serde_json::from_str(PROTEIN_WATER_VARIANT).unwrap();
    let compilation = compile_cse_hd_from_json_str(PROTEIN_WATER_VARIANT).unwrap();
    assert_eq!(compilation.validation.status, "PASS");
    assert_eq!(compilation.tables.residue_rows.len(), 2);
    assert_eq!(compilation.tables.water_site_rows.len(), 1);
    assert_eq!(compilation.tables.variant_rows.len(), 1);

    let roundtrip = validate_roundtrip_reconstruction(&source).unwrap();
    assert!(roundtrip.canonical_roundtrip_hash_pass);
}

#[test]
fn rejects_untyped_jsonish_domain_objects() {
    let bad = json!({
        "schema_version": "PRISM-CSE/HD ABI v0.1",
        "state_id": "bad_untyped",
        "classification": "SYNTHETIC",
        "synthetic": true,
        "domain_objects": [
            {
                "type": "molecule",
                "payload": {
                    "id": "loose_molecule",
                    "atoms": []
                }
            }
        ],
        "unit_registry": [],
        "provenance": [],
        "timebases": [],
        "molecules": [],
        "proteins": [],
        "water_sites": [],
        "variants": [],
        "events": [],
        "uncertainties": [],
        "thermo": [],
        "teacher_outputs": [],
        "gflownet_traces": [],
        "nanl_symbols": ["ENTITY:BAD"],
        "nrf_frames": [],
        "drf_states": []
    });

    let err = compile_cse_hd_from_json_str(&bad.to_string()).unwrap_err();
    assert_error_contains(err, "PC_ERR_TYPE", "untyped JSON-ish domain object");
}

#[test]
fn rejects_molecule_without_topology_hash() {
    let mut value: Value = serde_json::from_str(REFERENCE).unwrap();
    value["molecules"][0]["topology_hash"] = Value::String(String::new());
    let err = compile_cse_hd_from_json_str(&value.to_string()).unwrap_err();
    assert_error_contains(err, "PC_ERR_TYPE", "molecule.topology_hash");
}

#[test]
fn rejects_protein_without_atom_site_linkage() {
    let mut value: Value = serde_json::from_str(REFERENCE).unwrap();
    value["proteins"][0]["chains"][0]["residues"][0]["atom_site_ids"][0] =
        Value::String("missing_site".to_string());
    let err = compile_cse_hd_from_json_str(&value.to_string()).unwrap_err();
    assert_error_contains(
        err,
        "PC_ERR_TYPE",
        "residue.atom_site_id unresolved reference",
    );
}

#[test]
fn rejects_event_bound_water_without_timebase_or_provenance() {
    let mut value: Value = serde_json::from_str(REFERENCE).unwrap();
    value["water_sites"][0]["time_ref"] = Value::Null;
    value["water_sites"][0]["provenance_ref"] = Value::Null;
    let err = compile_cse_hd_from_json_str(&value.to_string()).unwrap_err();
    assert_error_contains(err, "PC_ERR_TIMEBASE", "event-bound water");
}

#[test]
fn rejects_thermo_without_unit_lineage() {
    let mut value: Value = serde_json::from_str(REFERENCE).unwrap();
    value["thermo"][0]["unit_ref"] = Value::String("missing_unit".to_string());
    let err = compile_cse_hd_from_json_str(&value.to_string()).unwrap_err();
    assert_error_contains(err, "PC_ERR_TYPE", "thermo.unit_ref unresolved reference");
}

#[test]
fn rejects_teacher_without_valid_airlock_state() {
    let mut value: Value = serde_json::from_str(REFERENCE).unwrap();
    value["teacher_outputs"][0]["airlock_state"] = Value::String("TRUSTED_BY_TEXT".to_string());
    let err = compile_cse_hd_from_json_str(&value.to_string()).unwrap_err();
    assert_error_contains(err, "PC_ERR_SCIENCE_ASSERT", "unsupported airlock state");
}

fn assert_error_contains(err: CseError, class: &str, detail: &str) {
    let text = err.to_string();
    assert!(
        text.contains(class),
        "error did not contain {class}: {text}"
    );
    assert!(
        text.contains(detail),
        "error did not contain {detail}: {text}"
    );
}

fn repo_build_demo() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("build/demo")
}
