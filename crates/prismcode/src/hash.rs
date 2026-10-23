use crate::instruction::Instruction;
use crate::isa::OpcodeRegistry;
use crate::validator::PrismCodeError;
use serde::Serialize;

pub fn canonical_json<T: Serialize>(value: &T) -> Result<String, PrismCodeError> {
    serde_json::to_string(value).map_err(PrismCodeError::Json)
}

pub fn registry_hash(registry: &OpcodeRegistry) -> Result<String, PrismCodeError> {
    let canonical = canonical_json(&registry.normalized())?;
    Ok(blake3::hash(canonical.as_bytes()).to_hex().to_string())
}

pub fn instruction_hash(instruction: &Instruction) -> Result<String, PrismCodeError> {
    let canonical = canonical_json(&instruction.normalized_for_hash())?;
    Ok(blake3::hash(canonical.as_bytes()).to_hex().to_string())
}
