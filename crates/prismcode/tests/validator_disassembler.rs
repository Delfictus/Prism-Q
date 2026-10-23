use prismcode::{
    disassemble_program, format_disassembly, registry_hash, validate_program, Instruction,
    OpcodeRegistry, OpcodeSpec, Program, PRISMCODE_ISA_VERSION, PRISMCODE_PROGRAM_SCHEMA_VERSION,
};
use serde_json::json;
use std::collections::BTreeMap;

fn registry() -> OpcodeRegistry {
    OpcodeRegistry {
        isa_version: PRISMCODE_ISA_VERSION.to_string(),
        opcodes: vec![OpcodeSpec {
            code: 1,
            mnemonic: "STATE_BIND".to_string(),
            domain: "state".to_string(),
            version: "v0.1".to_string(),
            operand_schema: json!({"required":["entity"]}),
            metadata: BTreeMap::new(),
        }],
        metadata: BTreeMap::new(),
    }
}

fn program(opcode: &str, registry: &OpcodeRegistry) -> Program {
    let mut operands = BTreeMap::new();
    operands.insert("entity".to_string(), json!("nanl://row/example"));
    Program {
        schema_version: PRISMCODE_PROGRAM_SCHEMA_VERSION.to_string(),
        isa_version: PRISMCODE_ISA_VERSION.to_string(),
        registry_hash: Some(registry_hash(registry).unwrap()),
        instructions: vec![Instruction {
            opcode: opcode.to_string(),
            operands,
            extra: BTreeMap::new(),
        }],
        metadata: BTreeMap::new(),
    }
}

#[test]
fn prismcode_validator_accepts_registry_bound_program() {
    let registry = registry();
    let report = validate_program(&program("STATE_BIND", &registry), &registry).unwrap();
    assert_eq!(report.instruction_count, 1);
    assert_eq!(report.opcode_counts["STATE_BIND"], 1);
}

#[test]
fn prismcode_validator_rejects_unknown_opcode() {
    let registry = registry();
    assert!(validate_program(&program("UNREGISTERED", &registry), &registry).is_err());
}

#[test]
fn prismcode_disassembler_is_stable() {
    let registry = registry();
    let program = program("STATE_BIND", &registry);
    let lines = disassemble_program(&program, &registry).unwrap();
    let formatted = format_disassembly(&lines);

    assert_eq!(lines[0].code, Some(1));
    assert!(formatted.contains("0000 0x0001 STATE_BIND domain=state hash="));
}
