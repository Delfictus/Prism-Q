use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Deterministic opcode registry entry.
///
/// The entry describes compiler-defined opcode identity and operand contract.
/// It does not assign execution semantics.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct OpcodeSpec {
    pub code: u32,
    pub mnemonic: String,
    pub domain: String,
    pub version: String,
    #[serde(default)]
    pub operand_schema: Value,
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
}

impl OpcodeSpec {
    pub fn normalized(&self) -> Self {
        let mut item = self.clone();
        item.mnemonic = item.mnemonic.trim().to_string();
        item.domain = item.domain.trim().to_string();
        item.version = item.version.trim().to_string();
        item
    }
}
