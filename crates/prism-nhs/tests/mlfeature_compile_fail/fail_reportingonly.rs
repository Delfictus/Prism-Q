//! Negative case: a struct that includes a `#[role(ReportingOnly)]` field
//! MUST fail to compile, and the diagnostic MUST name the offending field
//! (`debug_label` in this fixture).

use prism_mlfeature_derive::MLFeature;

#[derive(MLFeature)]
pub struct PoisonedFeatures {
    #[role(Localization)]
    pub centroid: [f32; 3],

    #[role(ReportingOnly)]
    pub debug_label: &'static str,
}

fn main() {}
