//! Positive case (tuple struct): a newtype/tuple struct with all
//! non-`ReportingOnly` fields must compile cleanly under
//! `#[derive(MLFeature)]`. Synthetic field names of the form `field_<N>`
//! must appear in `FIELD_ROLES`.

use prism_mlfeature_derive::MLFeature;
use prism_nhs::feature_role::{FeatureRole, MLFeatureRole};

#[derive(MLFeature)]
pub struct CleanTuple(
    #[role(Localization)] pub [f32; 3],
    #[role(Mechanistic)] pub f64,
    #[role(CausalInformation)] pub f64,
);

/// Newtype wrapper — the canonical shape of `CausalDriverView`,
/// `LiningContactView`, `LocalizedSubclusterView`.
#[derive(MLFeature)]
pub struct CleanNewtype(#[role(Mechanistic)] pub f64);

fn main() {
    assert_eq!(CleanTuple::FIELD_ROLES.len(), 3);
    assert_eq!(CleanTuple::FIELD_ROLES[0].0, "field_0");
    assert_eq!(CleanTuple::FIELD_ROLES[1].0, "field_1");
    assert_eq!(CleanTuple::FIELD_ROLES[2].0, "field_2");
    assert!(matches!(
        CleanTuple::FIELD_ROLES[0].1,
        FeatureRole::Localization
    ));
    assert!(CleanTuple::ml_safe());

    assert_eq!(CleanNewtype::FIELD_ROLES.len(), 1);
    assert_eq!(CleanNewtype::FIELD_ROLES[0].0, "field_0");
    assert!(CleanNewtype::ml_safe());
}
