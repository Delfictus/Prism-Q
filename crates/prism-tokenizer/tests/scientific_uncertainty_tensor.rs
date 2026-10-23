use prism_tokenizer::sidecar::ExactValue;
use prism_tokenizer::{
    detect_family, leverage_tier, route_artifact, unpack_namespace, ArtifactFamily as F, Encoder,
    FieldGeometry, FieldRegistry, TokenStream, GENERATIVE_TIER,
};
use serde_json::json;

const SHA: [u8; 32] = [11u8; 32];
const REQUIRED_UNCERTAINTY_FAMILIES: [(&str, F, &str, u8); 7] = [
    (
        "scientific_uncertainty_tensor",
        F::ScientificUncertaintyTensor,
        "ScientificUncertaintyTensor",
        2,
    ),
    (
        "runtime_numerical_uncertainty_tensor",
        F::RuntimeNumericalUncertaintyTensor,
        "RuntimeNumericalUncertaintyTensor",
        1,
    ),
    (
        "thermal_uncertainty_tensor",
        F::ThermalUncertaintyTensor,
        "ThermalUncertaintyTensor",
        1,
    ),
    (
        "integrator_uncertainty_tensor",
        F::IntegratorUncertaintyTensor,
        "IntegratorUncertaintyTensor",
        1,
    ),
    (
        "observable_uncertainty_tensor",
        F::ObservableUncertaintyTensor,
        "ObservableUncertaintyTensor",
        1,
    ),
    (
        "epoch_uncertainty_tensor",
        F::EpochUncertaintyTensor,
        "EpochUncertaintyTensor",
        1,
    ),
    (
        "signal_to_uncertainty_tensor",
        F::SignalToUncertaintyTensor,
        "SignalToUncertaintyTensor",
        2,
    ),
];

fn assert_join_key(registry: &FieldRegistry, family: F, field: &str) {
    let spec = registry
        .lookup(family, field)
        .unwrap_or_else(|| panic!("{family:?}/{field} must be registered"));
    assert!(
        matches!(&spec.geometry, FieldGeometry::JoinKey),
        "{family:?}/{field} must preserve identity as a JoinKey"
    );
    assert!(
        registry.namespace(family, field).is_some(),
        "{family:?}/{field} must have a routed namespace"
    );
}

fn assert_unit_scalar(registry: &FieldRegistry, family: F, field: &str) {
    let spec = registry
        .lookup(family, field)
        .unwrap_or_else(|| panic!("{family:?}/{field} must be registered"));
    match &spec.geometry {
        FieldGeometry::LinearBounded { lo, hi, .. } => {
            assert_eq!(*lo, 0.0, "{family:?}/{field} lower bound");
            assert_eq!(*hi, 1.0, "{family:?}/{field} upper bound");
        }
        geometry => panic!("{family:?}/{field} must be a unit scalar, got {geometry:?}"),
    }
}

fn assert_sidecar_text(stream: &TokenStream, field: &str, expected: &str) {
    assert!(
        stream.sidecar.iter().any(|entry| {
            entry.field == field
                && matches!(&entry.exact, ExactValue::Text(value) if value == expected)
        }),
        "{field} sidecar text must equal {expected}"
    );
}

fn assert_sidecar_scalar(stream: &TokenStream, field: &str, expected: f64) {
    assert!(
        stream.sidecar.iter().any(|entry| {
            entry.field == field
                && matches!(&entry.exact, ExactValue::Scalar(value) if (*value - expected).abs() < 1.0e-12)
        }),
        "{field} sidecar scalar must equal {expected}"
    );
}

fn mode_for_family(family: F) -> &'static str {
    match family {
        F::RuntimeNumericalUncertaintyTensor => "runtime_numerical",
        F::ThermalUncertaintyTensor => "thermal",
        F::IntegratorUncertaintyTensor => "integrator",
        F::ObservableUncertaintyTensor => "observable",
        F::EpochUncertaintyTensor => "epoch",
        F::SignalToUncertaintyTensor => "signal_to_uncertainty",
        _ => "scientific",
    }
}

#[test]
fn required_uncertainty_tensor_families_are_slot_safe_and_registered() {
    let registry = FieldRegistry::build();
    let expected_slots = [
        (F::ScientificUncertaintyTensor, 55),
        (F::RuntimeNumericalUncertaintyTensor, 56),
        (F::ThermalUncertaintyTensor, 57),
        (F::IntegratorUncertaintyTensor, 58),
        (F::ObservableUncertaintyTensor, 59),
        (F::SignalToUncertaintyTensor, 60),
        (F::EpochUncertaintyTensor, 61),
    ];

    for (family, slot) in expected_slots {
        assert_eq!(
            family.family_slot(),
            slot,
            "{family:?} must keep its append-only family slot"
        );
        assert!(family.family_slot() <= 63, "{family:?} family slot budget");
    }

    for (_, family, _, expected_tier) in REQUIRED_UNCERTAINTY_FAMILIES {
        assert_eq!(leverage_tier(family), expected_tier, "{family:?} tier");
        assert!(
            leverage_tier(family) < GENERATIVE_TIER,
            "{family:?} must not be generative"
        );
        let field_count = registry.iter().filter(|spec| spec.family == family).count();
        assert!(
            field_count <= 64,
            "{family:?} must stay within the 64-field slot budget"
        );
        let measurement_ns = registry
            .namespace(family, "measurement_status")
            .expect("measurement_status namespace");
        assert_eq!(
            unpack_namespace(measurement_ns).1 as usize,
            field_count - 1,
            "{family:?} last uncertainty field slot"
        );

        for field in [
            "row_id",
            "run_id",
            "epoch_id",
            "stream_id",
            "mode",
            "tensor_kind",
            "observable",
            "observable_family",
            "source_family",
            "source_event_id",
            "source_artifact_id",
            "source_signal_id",
            "target_tensor_id",
            "provenance_ref",
            "method",
            "mu",
            "sigma",
            "variance",
            "n_samples",
            "signal_delta",
            "signal_to_uncertainty",
            "consistency_score",
            "measurement_status",
            "source_status",
        ] {
            assert!(
                registry.lookup(family, field).is_some(),
                "{family:?}/{field} must be registered"
            );
        }

        for field in [
            "row_id",
            "run_id",
            "epoch_id",
            "source_event_id",
            "source_artifact_id",
            "source_signal_id",
            "target_tensor_id",
            "provenance_ref",
            "method",
        ] {
            assert_join_key(&registry, family, field);
        }
    }
}

#[test]
fn scientific_uncertainty_tensor_schema_is_registered() {
    let registry = FieldRegistry::build();

    for field in [
        "tensor_id",
        "source_artifact_id",
        "source_signal_id",
        "observable_family",
        "source_family",
        "measurement_window",
        "method",
        "source_refs",
        "sha256",
        "uncertainty_kind",
        "runtime_origin",
        "tokenizer_family",
    ] {
        assert!(
            registry
                .lookup(F::ScientificUncertaintyTensor, field)
                .is_some(),
            "ScientificUncertaintyTensor/{field} must be registered"
        );
    }

    for field in ["tensor_id", "source_artifact_id", "source_signal_id"] {
        assert_join_key(&registry, F::ScientificUncertaintyTensor, field);
    }
    for field in [
        "epistemic_uncertainty",
        "aleatoric_uncertainty",
        "total_uncertainty",
        "confidence",
    ] {
        assert_unit_scalar(&registry, F::ScientificUncertaintyTensor, field);
    }
}

#[test]
fn signal_to_uncertainty_tensor_schema_is_registered() {
    let registry = FieldRegistry::build();

    for field in [
        "mapping_id",
        "source_signal_id",
        "target_tensor_id",
        "observable_family",
        "uncertainty_channel",
        "measurement_window",
        "method",
        "source_refs",
        "sha256",
        "runtime_origin",
        "tokenizer_family",
    ] {
        assert!(
            registry
                .lookup(F::SignalToUncertaintyTensor, field)
                .is_some(),
            "SignalToUncertaintyTensor/{field} must be registered"
        );
    }

    for field in ["mapping_id", "source_signal_id", "target_tensor_id"] {
        assert_join_key(&registry, F::SignalToUncertaintyTensor, field);
    }
    for field in [
        "signal_value",
        "uncertainty_transfer_weight",
        "propagated_uncertainty",
    ] {
        assert_unit_scalar(&registry, F::SignalToUncertaintyTensor, field);
    }
}

#[test]
fn routes_scientific_uncertainty_tensor_records_by_value() {
    let artifact = json!({
        "schema_kind": "scientific_uncertainty_tensor",
        "schema_version": 1,
        "records": [
            {
                "tensor_id": "sut_4lpk_stream0_signal_grid",
                "source_artifact_id": "signal_grid_summary::stream0",
                "source_signal_id": "signal_grid::stream0::occupancy",
                "observable_family": "observable_specific",
                "source_family": "DenseTensorSemanticSummary",
                "measurement_window": "step:2000-35000",
                "method": "measured_sample_standard_deviation",
                "source_refs": ["receipt:stream0"],
                "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "uncertainty_kind": "epistemic_plus_aleatoric",
                "epistemic_uncertainty": 0.18,
                "aleatoric_uncertainty": 0.07,
                "total_uncertainty": 0.25,
                "confidence": 0.75,
                "evidence_count": 3,
                "runtime_origin": "POST_RUN_DERIVED",
                "tokenizer_family": "ScientificUncertaintyTensor"
            },
            {
                "tensor_id": "sut_4lpk_stream0_spike_summary",
                "source_artifact_id": "spike_summary::stream0",
                "source_signal_id": "spike_summary::stream0::rate",
                "observable_family": "runtime_numerical",
                "source_family": "SpikeSummary",
                "measurement_window": "step:2000-35000",
                "method": "measured_sample_standard_deviation",
                "source_refs": ["receipt:stream0"],
                "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "uncertainty_kind": "replicate_dispersion",
                "epistemic_uncertainty": 0.12,
                "aleatoric_uncertainty": 0.05,
                "total_uncertainty": 0.17,
                "confidence": 0.83,
                "evidence_count": 4,
                "runtime_origin": "POST_RUN_DERIVED",
                "tokenizer_family": "ScientificUncertaintyTensor"
            }
        ]
    });

    assert_eq!(
        detect_family(&artifact),
        Some((F::ScientificUncertaintyTensor, "/records"))
    );

    let encoder = Encoder::new();
    let mut stream = TokenStream::new();
    let (family, report) = route_artifact(&encoder, &mut stream, &artifact, true, SHA, true);

    assert_eq!(family, Some(F::ScientificUncertaintyTensor));
    assert_eq!(report.records, 2);
    assert!(
        report.matched >= 20,
        "expected by-value tensor fields to route, got {report:?}"
    );
    assert_eq!(stream.segments.len(), 2);
    assert!(stream
        .segments
        .iter()
        .all(|segment| segment.family == F::ScientificUncertaintyTensor && segment.len > 0));
    assert!(stream
        .sidecar
        .iter()
        .any(|entry| entry.field == "total_uncertainty"));
}

#[test]
fn routes_signal_to_uncertainty_tensor_records_by_value() {
    let artifact = json!({
        "schema_kind": "signal_to_uncertainty_tensor",
        "schema_version": 1,
        "records": [
            {
                "mapping_id": "sig2sut_4lpk_stream0_grid",
                "source_signal_id": "signal_grid::stream0::occupancy",
                "observable_family": "observable_specific",
                "target_tensor_id": "sut_4lpk_stream0_signal_grid",
                "measurement_window": "step:2000-35000",
                "method": "abs_signal_delta_over_sigma",
                "source_refs": ["receipt:stream0"],
                "sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
                "uncertainty_channel": "occupancy_variance",
                "signal_value": 0.64,
                "uncertainty_transfer_weight": 0.80,
                "propagated_uncertainty": 0.20,
                "runtime_origin": "POST_RUN_DERIVED",
                "tokenizer_family": "SignalToUncertaintyTensor"
            }
        ]
    });

    assert_eq!(
        detect_family(&artifact),
        Some((F::SignalToUncertaintyTensor, "/records"))
    );

    let encoder = Encoder::new();
    let mut stream = TokenStream::new();
    let (family, report) = route_artifact(&encoder, &mut stream, &artifact, true, SHA, true);

    assert_eq!(family, Some(F::SignalToUncertaintyTensor));
    assert_eq!(report.records, 1);
    assert!(
        report.matched >= 9,
        "expected by-value signal mapping fields to route, got {report:?}"
    );
    assert_eq!(stream.segments.len(), 1);
    assert_eq!(stream.segments[0].family, F::SignalToUncertaintyTensor);
    assert!(stream
        .sidecar
        .iter()
        .any(|entry| entry.field == "propagated_uncertainty"));
}

#[test]
fn routes_required_uncertainty_tensor_families_with_sidecar_parity() {
    let encoder = Encoder::new();

    for (schema_kind, family, tokenizer_family, _) in REQUIRED_UNCERTAINTY_FAMILIES {
        let row_id = format!("{schema_kind}_row_0001");
        let source_event_id = format!("{schema_kind}_source_event");
        let source_artifact_id = format!("{schema_kind}_source_artifact");
        let source_signal_id = format!("{schema_kind}_source_signal");
        let target_tensor_id = format!("{schema_kind}_target_tensor");
        let provenance_ref = format!("sha256:{schema_kind}");
        let artifact = json!({
            "schema_kind": schema_kind,
            "schema_version": 1,
            "records": [{
                "row_id": row_id.clone(),
                "run_id": "run_4lpk_apo",
                "epoch_id": "epoch_0007",
                "stream_id": 0,
                "mode": mode_for_family(family),
                "tensor_kind": schema_kind,
                "uncertainty_kind": "derived_sigma",
                "observable": "temperature",
                "observable_family": "observable_specific",
                "source_family": "ProtocolState",
                "source_event_id": source_event_id.clone(),
                "source_artifact_id": source_artifact_id.clone(),
                "source_signal_id": source_signal_id.clone(),
                "target_tensor_id": target_tensor_id.clone(),
                "provenance_ref": provenance_ref.clone(),
                "method": "measured_sample_standard_deviation",
                "mu": 298.15,
                "sigma": 0.15,
                "variance": 0.0225,
                "n_samples": 4096,
                "signal_delta": 0.75,
                "signal_to_uncertainty": 5.0,
                "consistency_score": 0.98,
                "measurement_status": "derived",
                "source_status": "present",
                "tokenizer_family": tokenizer_family
            }]
        });

        assert_eq!(detect_family(&artifact), Some((family, "/records")));
        assert_eq!(
            detect_family(&json!({"schema_kind": tokenizer_family})),
            Some((family, "")),
            "{tokenizer_family} CamelCase schema label must route"
        );

        let mut stream = TokenStream::new();
        let (detected, report) = route_artifact(&encoder, &mut stream, &artifact, true, SHA, true);

        assert_eq!(detected, Some(family));
        assert_eq!(report.records, 1);
        assert!(
            report.matched > 0,
            "{family:?} must not fall back to generic JSON"
        );
        assert!(
            report.matched >= 24,
            "{family:?} expected by-value uncertainty fields, got {report:?}"
        );
        assert_eq!(stream.segments.len(), 1);
        assert_eq!(stream.segments[0].family, family);
        assert!(stream.segments[0].len > 0);

        assert_sidecar_text(&stream, "row_id", &row_id);
        assert_sidecar_text(&stream, "source_event_id", &source_event_id);
        assert_sidecar_text(&stream, "source_artifact_id", &source_artifact_id);
        assert_sidecar_text(&stream, "source_signal_id", &source_signal_id);
        assert_sidecar_text(&stream, "target_tensor_id", &target_tensor_id);
        assert_sidecar_text(&stream, "provenance_ref", &provenance_ref);
        assert_sidecar_text(&stream, "tensor_kind", schema_kind);
        assert_sidecar_text(&stream, "observable_family", "observable_specific");
        assert_sidecar_text(&stream, "method", "measured_sample_standard_deviation");
        assert_sidecar_text(&stream, "measurement_status", "derived");
        assert_sidecar_text(&stream, "source_status", "present");
        assert_sidecar_text(&stream, "tokenizer_family", tokenizer_family);
        assert_sidecar_scalar(&stream, "mu", 298.15);
        assert_sidecar_scalar(&stream, "sigma", 0.15);
        assert_sidecar_scalar(&stream, "variance", 0.0225);
        assert_sidecar_scalar(&stream, "n_samples", 4096.0);
        assert_sidecar_scalar(&stream, "signal_to_uncertainty", 5.0);
        assert_sidecar_scalar(&stream, "consistency_score", 0.98);
    }
}
