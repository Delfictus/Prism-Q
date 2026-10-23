use crate::opcode::OpcodeSpec;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Compiler-owned PRISMCODE ISA registry envelope.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct OpcodeRegistry {
    pub isa_version: String,
    pub opcodes: Vec<OpcodeSpec>,
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
}

impl OpcodeRegistry {
    pub fn normalized(&self) -> Self {
        let mut opcodes: Vec<OpcodeSpec> =
            self.opcodes.iter().map(OpcodeSpec::normalized).collect();
        opcodes.sort_by(|left, right| {
            left.code
                .cmp(&right.code)
                .then_with(|| left.mnemonic.cmp(&right.mnemonic))
                .then_with(|| left.domain.cmp(&right.domain))
                .then_with(|| left.version.cmp(&right.version))
        });

        Self {
            isa_version: self.isa_version.trim().to_string(),
            opcodes,
            metadata: self.metadata.clone(),
        }
    }

    pub fn opcode_by_mnemonic(&self, mnemonic: &str) -> Option<&OpcodeSpec> {
        self.opcodes.iter().find(|item| item.mnemonic == mnemonic)
    }
}
