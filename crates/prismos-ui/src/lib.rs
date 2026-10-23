//! PrismOS Beta2.0 desktop application library.
//!
//! The crate is split into two layers:
//!
//! * a **pure core** (`adapters`, `model`) that has no GTK dependency and is
//!   fully unit-testable without a display server or the GTK system libraries;
//! * a **GTK4 desktop shell** (`app`, `theme`, `jobs`, `panels`, `viz`) compiled
//!   only when the default `gui` feature is enabled.
//!
//! The desktop shell drives the installed PrismOS backend binaries through the
//! [`adapters`] layer — every backend invocation is an explicit, typed argv
//! vector (no shell-string interpolation) — and renders the T4 evidence surfaces
//! parsed by [`model`].

pub mod adapters;
pub mod model;

#[cfg(feature = "gui")]
pub mod theme;

#[cfg(feature = "gui")]
pub mod jobs;

#[cfg(feature = "gui")]
pub mod app;

#[cfg(feature = "gui")]
pub mod panels;

#[cfg(feature = "gui")]
pub mod viz;

/// Application identifier used for the GTK/desktop integration.
pub const APP_ID: &str = "com.delfictus.PrismOS.Ui";

/// Human-facing application name.
pub const APP_NAME: &str = "PrismOS";
