//! PrismCode memory image reader-side schema.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MemoryImageHeader {
    pub isa_version: String,
    pub row_count: u64,
    pub instruction_count: u64,
    pub memory_hash: String,
}
