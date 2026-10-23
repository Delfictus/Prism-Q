//! Positive case: a struct with all non-`ReportingOnly` fields must compile
//! cleanly under `#[derive(MLFeature)]`.

use prism_mlfeature_derive::MLFeature;
use prism_nhs::feature_role::{FeatureRole, MLFeatureRole};

#[derive(MLFeature)]
pub struct CleanFeatures {
    #[role(Localization)]
    pub centroid: [f32; 3],

    #[role(Mechanistic)]
    pub kcc_score: f64,

    #[role(CausalInformation)]
    pub transfer_entropy: f64,

    #[role(Thermodynamic)]
    pub free_energy: f64,

    #[role(StabilityConsensus)]
    pub cross_stream_agreement: f32,

    #[role(QualityControl)]
    pub conservation_residual: f32,
}

fn main() {
    assert_eq!(CleanFeatures::FIELD_ROLES.len(), 6);
    assert!(CleanFeatures::ml_safe());

    // Spot-check the role mapping survived macro expansion.
    let names: Vec<&'static str> = CleanFeatures::FIELD_ROLES
        .iter()
        .map(|(n, _)| *n)
        .collect();
    assert_eq!(
        names,
        vec![
            "centroid",
            "kcc_score",
            "transfer_entropy",
            "free_energy",
            "cross_stream_agreement",
            "conservation_residual",
        ]
    );

    // No field carries `ReportingOnly`.
    assert!(CleanFeatures::FIELD_ROLES
        .iter()
        .all(|(_, r)| !matches!(r, FeatureRole::ReportingOnly)));
}
