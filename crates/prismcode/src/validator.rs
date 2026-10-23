use crate::contract::PRISMCODE_PROGRAM_SCHEMA_VERSION;
use crate::hash::{instruction_hash, registry_hash};
use crate::instruction::Program;
use crate::isa::OpcodeRegistry;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, thiserror::Error)]
pub enum PrismCodeError {
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("registry validation failed: {0}")]
    Registry(String),
    #[error("program validation failed: {0}")]
    Program(String),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ValidationReport {
    pub schema_version: String,
    pub isa_version: String,
    pub registry_hash: String,
    pub instruction_count: usize,
    pub instruction_hashes: Vec<String>,
    pub opcode_counts: BTreeMap<String, usize>,
}

pub fn validate_registry(registry: &OpcodeRegistry) -> Result<(), PrismCodeError> {
    if registry.isa_version.trim().is_empty() {
        return Err(PrismCodeError::Registry("isa_version is empty".to_string()));
    }

    let mut codes = BTreeSet::new();
    let mut mnemonics = BTreeSet::new();
    for opcode in &registry.opcodes {
        let normalized = opcode.normalized();
        if normalized.mnemonic.is_empty() {
            return Err(PrismCodeError::Registry(
                "opcode mnemonic is empty".to_string(),
            ));
        }
        if normalized.domain.is_empty() {
            return Err(PrismCodeError::Registry(format!(
                "opcode {} domain is empty",
                normalized.mnemonic
            )));
        }
        if normalized.version.is_empty() {
            return Err(PrismCodeError::Registry(format!(
                "opcode {} version is empty",
                normalized.mnemonic
            )));
        }
        crate::operand_schema::check_schema(&normalized.operand_schema).map_err(|error| {
            PrismCodeError::Registry(format!("opcode {} operand schema: {error}", normalized.mnemonic))
        })?;
        if !codes.insert(normalized.code) {
            return Err(PrismCodeError::Registry(format!(
                "duplicate opcode code {}",
                normalized.code
            )));
        }
        if !mnemonics.insert(normalized.mnemonic.clone()) {
            return Err(PrismCodeError::Registry(format!(
                "duplicate opcode mnemonic {}",
                normalized.mnemonic
            )));
        }
    }

    Ok(())
}

pub fn validate_program(
    program: &Program,
    registry: &OpcodeRegistry,
) -> Result<ValidationReport, PrismCodeError> {
    validate_registry(registry)?;

    let normalized_program = program.normalized();
    if normalized_program.schema_version != PRISMCODE_PROGRAM_SCHEMA_VERSION {
        return Err(PrismCodeError::Program(format!(
            "schema_version must be {}",
            PRISMCODE_PROGRAM_SCHEMA_VERSION
        )));
    }
    if normalized_program.isa_version != registry.normalized().isa_version {
        return Err(PrismCodeError::Program(format!(
            "program isa_version {} does not match registry isa_version {}",
            normalized_program.isa_version, registry.isa_version
        )));
    }

    let actual_registry_hash = registry_hash(registry)?;
    if let Some(expected_hash) = &normalized_program.registry_hash {
        if expected_hash != &actual_registry_hash {
            return Err(PrismCodeError::Program(format!(
                "registry_hash mismatch: expected {}, observed {}",
                expected_hash, actual_registry_hash
            )));
        }
    }

    let normalized_registry = registry.normalized();
    let mut opcode_counts = BTreeMap::new();
    let mut instruction_hashes = Vec::with_capacity(normalized_program.instructions.len());
    for (index, instruction) in normalized_program.instructions.iter().enumerate() {
        let Some(spec) = normalized_registry
            .opcode_by_mnemonic(&instruction.opcode)
        else {
            return Err(PrismCodeError::Program(format!(
                "instruction {} references unknown opcode {}",
                index, instruction.opcode
            )));
        };
        crate::operand_schema::check_value(
            &spec.operand_schema,
            &serde_json::to_value(&instruction.operands)?,
        ).map_err(|error| PrismCodeError::Program(format!("instruction {index} {}: {error}", instruction.opcode)))?;
        instruction_hashes.push(instruction_hash(instruction)?);
        *opcode_counts.entry(instruction.opcode.clone()).or_insert(0) += 1;
    }

    Ok(ValidationReport {
        schema_version: normalized_program.schema_version,
        isa_version: normalized_program.isa_version,
        registry_hash: actual_registry_hash,
        instruction_count: normalized_program.instructions.len(),
        instruction_hashes,
        opcode_counts,
    })
}
