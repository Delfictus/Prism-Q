use crate::hash::instruction_hash;
use crate::instruction::Program;
use crate::isa::OpcodeRegistry;
use crate::validator::PrismCodeError;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DisassemblyLine {
    pub index: usize,
    pub code: Option<u32>,
    pub opcode: String,
    pub domain: Option<String>,
    pub instruction_hash: String,
}

pub fn disassemble_program(
    program: &Program,
    registry: &OpcodeRegistry,
) -> Result<Vec<DisassemblyLine>, PrismCodeError> {
    let normalized_registry = registry.normalized();
    program
        .normalized()
        .instructions
        .iter()
        .enumerate()
        .map(|(index, instruction)| {
            let opcode = normalized_registry.opcode_by_mnemonic(&instruction.opcode);
            Ok(DisassemblyLine {
                index,
                code: opcode.map(|item| item.code),
                opcode: instruction.opcode.clone(),
                domain: opcode.map(|item| item.domain.clone()),
                instruction_hash: instruction_hash(instruction)?,
            })
        })
        .collect()
}

pub fn format_disassembly(lines: &[DisassemblyLine]) -> String {
    lines
        .iter()
        .map(|line| {
            let code = line
                .code
                .map(|value| format!("0x{value:04X}"))
                .unwrap_or_else(|| "unregistered".to_string());
            let domain = line.domain.as_deref().unwrap_or("unknown");
            format!(
                "{:04} {} {} domain={} hash={}",
                line.index, code, line.opcode, domain, line.instruction_hash
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}
