use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Compiler-emitted instruction record.
///
/// Unknown fields are preserved in `extra` so the Python compiler contract can
/// grow without forcing this scaffold to invent execution semantics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Instruction {
    pub opcode: String,
    #[serde(default)]
    pub operands: BTreeMap<String, Value>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl Instruction {
    pub fn normalized_for_hash(&self) -> Self {
        let mut extra = self.extra.clone();
        extra.remove("instruction_hash");
        Self {
            opcode: self.opcode.trim().to_string(),
            operands: self.operands.clone(),
            extra,
        }
    }
}

/// PRISMCODE program envelope consumed by validators and disassemblers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Program {
    pub schema_version: String,
    pub isa_version: String,
    #[serde(default)]
    pub registry_hash: Option<String>,
    pub instructions: Vec<Instruction>,
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
}

impl Program {
    pub fn normalized(&self) -> Self {
        Self {
            schema_version: self.schema_version.trim().to_string(),
            isa_version: self.isa_version.trim().to_string(),
            registry_hash: self.registry_hash.clone(),
            instructions: self
                .instructions
                .iter()
                .map(Instruction::normalized_for_hash)
                .collect(),
            metadata: self.metadata.clone(),
        }
    }
}
