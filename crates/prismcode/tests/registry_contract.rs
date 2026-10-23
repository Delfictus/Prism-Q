use prismcode::{
    registry_hash, validate_registry, OpcodeRegistry, OpcodeSpec, PRISMCODE_ISA_VERSION,
};
use serde_json::json;
use std::collections::BTreeMap;

fn registry_with_reversed_order() -> OpcodeRegistry {
    OpcodeRegistry {
        isa_version: PRISMCODE_ISA_VERSION.to_string(),
        opcodes: vec![
            OpcodeSpec {
                code: 2,
                mnemonic: "TIME_ATTACH".to_string(),
                domain: "time".to_string(),
                version: "v0.1".to_string(),
                operand_schema: json!({"required":["timebase"]}),
                metadata: BTreeMap::new(),
            },
            OpcodeSpec {
                code: 1,
                mnemonic: "STATE_BIND".to_string(),
                domain: "state".to_string(),
                version: "v0.1".to_string(),
                operand_schema: json!({"required":["entity"]}),
                metadata: BTreeMap::new(),
            },
        ],
        metadata: BTreeMap::new(),
    }
}

#[test]
fn prismcode_registry_is_deterministic() {
    let registry = registry_with_reversed_order();
    let mut sorted = registry.clone();
    sorted.opcodes.reverse();

    validate_registry(&registry).unwrap();
    validate_registry(&sorted).unwrap();

    assert_eq!(
        registry_hash(&registry).unwrap(),
        registry_hash(&sorted).unwrap()
    );
}

#[test]
fn prismcode_registry_rejects_duplicate_opcode_code() {
    let mut registry = registry_with_reversed_order();
    registry.opcodes[1].code = registry.opcodes[0].code;

    assert!(validate_registry(&registry).is_err());
}
