use prismcode::{instruction_hash, Instruction};
use serde_json::json;
use std::collections::BTreeMap;

fn instruction() -> Instruction {
    let mut operands = BTreeMap::new();
    operands.insert("entity".to_string(), json!("row_0001"));
    operands.insert("score".to_string(), json!(0.75));
    Instruction {
        opcode: "STATE_BIND".to_string(),
        operands,
        extra: BTreeMap::new(),
    }
}

#[test]
fn prismcode_instruction_hash_is_stable() {
    let first = instruction();
    let second = instruction();
    assert_eq!(
        instruction_hash(&first).unwrap(),
        instruction_hash(&second).unwrap()
    );
}

#[test]
fn prismcode_instruction_hash_ignores_existing_instruction_hash_field() {
    let first = instruction();
    let mut second = instruction();
    second
        .extra
        .insert("instruction_hash".to_string(), json!("external-value"));
    assert_eq!(
        instruction_hash(&first).unwrap(),
        instruction_hash(&second).unwrap()
    );
}
