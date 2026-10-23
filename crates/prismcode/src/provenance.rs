//! Provenance schema surface.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EvidenceRef {
    pub instruction_id: String,
    pub evidence_ref: String,
    pub source_artifact_hash: String,
    pub confidence_level: String,
}
