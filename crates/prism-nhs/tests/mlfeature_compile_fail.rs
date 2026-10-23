//! # Pillar 5 firewall — compile-fail tests
//!
//! Asserts that:
//!  - A struct with all non-`ReportingOnly` fields and `#[derive(MLFeature)]`
//!    compiles (positive case).
//!  - A struct with one `#[role(ReportingOnly)]` field fails to compile,
//!    and the diagnostic names the offending field (negative case).
//!
//! The body is gated behind the `trybuild_smoke` feature. Until SL6-B2
//! lands `trybuild` as a dev-dependency and enables this feature in
//! `crates/prism-nhs/Cargo.toml`, this file compiles to an empty test
//! target. After SL6-B2, run with:
//!
//! ```sh
//! cargo test -p prism-nhs --features trybuild_smoke --test mlfeature_compile_fail
//! ```
//!
//! On first run, seed the `.stderr` golden file with:
//! `TRYBUILD=overwrite cargo test --features trybuild_smoke ...`

#[cfg(feature = "trybuild_smoke")]
#[test]
fn mlfeature_compile_fail() {
    let t = trybuild::TestCases::new();
    t.pass("tests/mlfeature_compile_fail/pass_*.rs");
    t.compile_fail("tests/mlfeature_compile_fail/fail_*.rs");
}
