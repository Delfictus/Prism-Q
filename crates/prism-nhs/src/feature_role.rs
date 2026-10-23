//! # Pillar 5 firewall — runtime types
//!
//! Defines the canonical `FeatureRole` enum and the `MLFeatureRole` marker
//! trait that the `#[derive(MLFeature)]` proc-macro (in
//! `prism-mlfeature-derive`) generates an impl for.
//!
//! ## Role taxonomy
//!
//! Per § 5 of the PRISM-4D Entangled Transform Blueprint, every retained
//! data-plane field maps to exactly one of seven roles:
//!
//! | Role                  | Purpose                                                      |
//! |-----------------------|--------------------------------------------------------------|
//! | `Localization`        | Spatial / lining-residue support                             |
//! | `Mechanistic`         | Driver / mechanism information (KCC, transfer-entropy, lag)  |
//! | `CausalInformation`   | Causal-only signals (TE, causal lag, drive direction)        |
//! | `Thermodynamic`       | Free-energy / SDST channels                                  |
//! | `StabilityConsensus`  | Cross-stream / cross-replica stability metrics               |
//! | `QualityControl`      | QC scalars (residuals, audit flags, conservation deltas)     |
//! | `ReportingOnly`       | Forensic / display fields — **forbidden from ML tensors**    |
//!
//! `ReportingOnly` is the firewall: a field with this role cannot enter
//! a struct that derives `MLFeature`. The proc-macro emits a compile
//! error pointing at the offending field. Pillar 5 violations are made
//! impossible at runtime by being made impossible at compile time.

#![allow(dead_code)]

/// Canonical role taxonomy (§ 5 of PRISM-4D Entangled Transform Blueprint).
///
/// `ReportingOnly` is compile-time forbidden from ML training tensors via
/// `#[derive(MLFeature)]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FeatureRole {
    /// Spatial localisation / lining-residue support.
    Localization,
    /// Driver / mechanism information.
    Mechanistic,
    /// Causal-only signals (transfer entropy, causal lag, drive direction).
    CausalInformation,
    /// Free-energy / SDST thermodynamic channels.
    Thermodynamic,
    /// Cross-stream / cross-replica stability consensus metrics.
    StabilityConsensus,
    /// QC scalars (residuals, audit flags, conservation deltas).
    QualityControl,
    /// Forensic / display fields. **Compile-time forbidden from ML
    /// training tensors via the Pillar 5 firewall.**
    ReportingOnly,
}

impl FeatureRole {
    /// Returns `true` if a field with this role may participate in an
    /// ML training tensor. Equivalent to `!matches!(self, ReportingOnly)`.
    pub const fn is_ml_safe(self) -> bool {
        !matches!(self, Self::ReportingOnly)
    }

    /// Human-readable role name (matches the variant identifier).
    pub const fn name(self) -> &'static str {
        match self {
            Self::Localization => "Localization",
            Self::Mechanistic => "Mechanistic",
            Self::CausalInformation => "CausalInformation",
            Self::Thermodynamic => "Thermodynamic",
            Self::StabilityConsensus => "StabilityConsensus",
            Self::QualityControl => "QualityControl",
            Self::ReportingOnly => "ReportingOnly",
        }
    }
}

/// Marker trait emitted by `#[derive(MLFeature)]`.
///
/// A type implements `MLFeatureRole` if and only if every one of its
/// fields carries a non-`ReportingOnly` role. The presence of this impl
/// is therefore proof that the type passes the Pillar 5 firewall.
///
/// `FIELD_ROLES` carries the full per-field role list, observable at
/// compile time and at runtime, for downstream tooling (e.g. tensor
/// builders that need to know which fields to flatten).
///
/// ## Stable-Rust translation note
///
/// The blueprint specifies this trait as
/// `MLFeatureRole<const R: FeatureRole>`. Custom enum types as
/// const-generic parameters require the `adt_const_params` nightly
/// feature. The form below preserves the SEMANTIC — every field
/// carries a compile-time role tag — while remaining stable.
pub trait MLFeatureRole {
    /// Compile-time list of `(field_name, role)` pairs in declaration order.
    const FIELD_ROLES: &'static [(&'static str, FeatureRole)];

    /// Runtime accessor over `FIELD_ROLES`.
    fn field_roles() -> &'static [(&'static str, FeatureRole)] {
        Self::FIELD_ROLES
    }

    /// Always returns `true` for any type that successfully derives
    /// `MLFeature`, since the macro rejects `ReportingOnly` at compile
    /// time. Provided as a runtime sanity check — should never return
    /// false for a derived impl.
    fn ml_safe() -> bool {
        Self::FIELD_ROLES.iter().all(|(_, r)| r.is_ml_safe())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reporting_only_is_not_ml_safe() {
        assert!(!FeatureRole::ReportingOnly.is_ml_safe());
    }

    #[test]
    fn other_roles_are_ml_safe() {
        for r in [
            FeatureRole::Localization,
            FeatureRole::Mechanistic,
            FeatureRole::CausalInformation,
            FeatureRole::Thermodynamic,
            FeatureRole::StabilityConsensus,
            FeatureRole::QualityControl,
        ] {
            assert!(r.is_ml_safe(), "{:?} should be ML-safe", r);
        }
    }

    #[test]
    fn name_matches_variant() {
        assert_eq!(FeatureRole::ReportingOnly.name(), "ReportingOnly");
        assert_eq!(FeatureRole::CausalInformation.name(), "CausalInformation");
    }
}
