//! Negative case (tuple struct): a tuple struct with a
//! `#[role(ReportingOnly)]` field MUST fail to compile. The diagnostic
//! must use the synthetic name `field_<N>` (here `field_1`) and must
//! include the `(of <TypeName>)` clarifier.

use prism_mlfeature_derive::MLFeature;

#[derive(MLFeature)]
pub struct PoisonedTuple(
    #[role(Localization)] pub [f32; 3],
    #[role(ReportingOnly)] pub &'static str,
);

fn main() {}
