//! §6 — versioned schema: the `(ArtifactFamily, field) → FieldSpec` registry (the platform-native
//! core) + factored namespace + manifest-gap reservations.

pub mod families;
pub mod registry;
pub mod version;

pub use families::{field_table, UV_LINES};
pub use registry::FieldRegistry;
pub use version::{MANIFEST_GAP_FAMILIES, SCHEMA_VERSION};
