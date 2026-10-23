use prismcode::binary_runtime::{self as br, OperandKind, PcInstr32, SectionKind, ValidationLevel};
use prismcode::{validate_program, validate_registry, Instruction, OpcodeRegistry, OpcodeSpec, Program,
    PRISMCODE_ISA_VERSION, PRISMCODE_PROGRAM_SCHEMA_VERSION};
use serde_json::{json, Value};
use std::{collections::BTreeMap, path::Path};

fn validate_operands(schema: Value, operands: Value) -> Result<(), String> {
    let registry = OpcodeRegistry { isa_version: PRISMCODE_ISA_VERSION.into(), metadata: BTreeMap::new(),
        opcodes: vec![OpcodeSpec { code: 1, mnemonic: "TEST.CONTRACT".into(), domain: "test".into(),
            version: "0.1".into(), operand_schema: schema, metadata: BTreeMap::new() }] };
    let program = Program { schema_version: PRISMCODE_PROGRAM_SCHEMA_VERSION.into(),
        isa_version: PRISMCODE_ISA_VERSION.into(), registry_hash: None, metadata: BTreeMap::new(),
        instructions: vec![Instruction { opcode: "TEST.CONTRACT".into(),
            operands: serde_json::from_value(operands).unwrap(), extra: BTreeMap::new() }] };
    validate_program(&program,&registry).map(|_| ()).map_err(|e| e.to_string())
}

#[test]
fn required_operand_contract_is_enforced_and_legacy_positive_survives() {
    let schema = json!({"required":["entity"]});
    assert!(validate_operands(schema.clone(),json!({"entity":"nanl://row/example"})).is_ok());
    assert!(validate_operands(schema,json!({})).unwrap_err().contains("required operand"));
    assert!(validate_operands(Value::Null,json!({"future":17})).is_ok());
}

#[test]
fn nested_types_counts_ranges_and_unknown_properties_are_enforced() {
    let schema = json!({"type":"object","required":["target","shape","gain"],"additionalProperties":false,
        "properties":{"target":{"type":"string","minLength":1,"maxLength":5},
            "shape":{"type":"array","minItems":2,"maxItems":2,"items":{"type":"integer","minimum":1}},
            "gain":{"type":"number","exclusiveMinimum":0,"maximum":2}}});
    let good=json!({"target":"α","shape":[2,3],"gain":1.5});
    assert!(validate_operands(schema.clone(),good.clone()).is_ok());
    for (key,bad) in [("target",json!("")),("target",json!(19)),("shape",json!([2])),
        ("shape",json!([2,0])),("shape",json!([2,1.5])),("gain",json!(0)),("gain",json!(2.1)),("unknown",json!(true))] {
        let mut value=good.clone();value[key]=bad;
        assert!(validate_operands(schema.clone(),value).is_err(),"constraint did not reject {key}");
    }
}

#[test]
fn exact_u64_bounds_do_not_round_through_float() {
    let schema=json!({"properties":{"n":{"type":"integer","minimum":u64::MAX,"maximum":u64::MAX}}});
    assert!(validate_operands(schema.clone(),json!({"n":u64::MAX})).is_ok());
    assert!(validate_operands(schema.clone(),json!({"n":u64::MAX-1})).is_err());
    assert!(validate_operands(schema,json!({"n":-1})).is_err());
}

#[test]
fn mixed_numeric_bounds_and_recursive_equality_preserve_exact_values() {
    let max=json!({"properties":{"n":{"maximum":9007199254740992.0}}});
    assert!(validate_operands(max.clone(),json!({"n":9007199254740992u64})).is_ok());
    assert!(validate_operands(max,json!({"n":9007199254740993u64})).is_err());
    let min=json!({"properties":{"n":{"minimum":-9007199254740992.0}}});
    assert!(validate_operands(min,json!({"n":-9007199254740993i64})).is_err());
    for (schema,value,accepted) in [
        (json!({"exclusiveMaximum":1.0}),json!(1u64),false),
        (json!({"minimum":-1.5}),json!(-1i64),true),
        (json!({"maximum":-1.5}),json!(-1i64),false),
        (json!({"maximum":18446744073709551616.0}),json!(u64::MAX),true),
        (json!({"minimum":18446744073709551616.0}),json!(u64::MAX),false),
        (json!({"maximum":u64::MAX}),json!(18446744073709551616.0),false),
        (json!({"const":{"v":[1.0]}}),json!({"v":[1]}),true),
        (json!({"enum":[1.0,2.0]}),json!(1),true),
        (json!({"uniqueItems":true}),json!([{"v":1},{"v":1.0}]),false),
        (json!({"uniqueItems":true}),json!([9007199254740992.0,9007199254740993u64]),true),
    ] {
        assert_eq!(validate_operands(json!({"properties":{"n":schema}}),json!({"n":value})).is_ok(),accepted);
    }
    assert!(validate_operands(json!({"enum":[1,1.0]}),json!({})).is_err());
}

#[test]
fn compositions_enum_constants_and_array_contracts_are_enforced() {
    let schema=json!({"properties":{"mode":{"enum":["a","b"],"not":{"const":"b"}},
        "x":{"allOf":[{"minimum":0},{"maximum":10}],"oneOf":[{"maximum":3},{"minimum":7}]},
        "v":{"type":"array","prefixItems":[{"const":"id"}],"items":{"type":"integer"},"uniqueItems":true}}});
    assert!(validate_operands(schema.clone(),json!({"mode":"a","x":2,"v":["id",1,2]})).is_ok());
    for bad in [json!({"mode":"b"}),json!({"mode":"c"}),json!({"x":5}),json!({"v":["id",1,1]}),json!({"v":["bad",1]})] {
        assert!(validate_operands(schema.clone(),bad).is_err());
    }
    assert!(validate_operands(json!({"properties":{"v":{"anyOf":[{"type":"null"},{"type":"string"}]}}}),json!({"v":null})).is_ok());
}

#[test]
fn malformed_and_unsupported_schema_constraints_fail_at_registry_admission() {
    for schema in [json!({"required":"x"}),json!({"required":["x","x"]}),json!({"type":"not_a_type"}),
        json!({"minimum":"zero"}),json!({"minItems":-1}),json!({"properties":{"unused":{"pattern":".*"}}}),
        json!({"$ref":"external.json"}),json!({"oneOf":[]})] {
        let registry=OpcodeRegistry {isa_version:PRISMCODE_ISA_VERSION.into(),metadata:BTreeMap::new(),
            opcodes:vec![OpcodeSpec {code:1,mnemonic:"X".into(),domain:"X".into(),version:"1".into(),operand_schema:schema,metadata:BTreeMap::new()}]};
        assert!(validate_registry(&registry).is_err());
    }
}

fn packet() -> br::PCodePacket {
    let tmp=tempfile::tempdir().unwrap();let path=tmp.path().join("reference.pcode");
    // A self-contained compiler input avoids depending on the absent historical
    // examples/reference/reference.json. This is explicitly test-only data.
    let corpus: br::ReferenceCorpus=serde_json::from_value(json!({
        "corpus_id":"OPERAND_BOUNDS_TEST","synthetic":true,"generator_version":"test","seed":0,
        "states":[{"state_id":1,"state_kind":"DRF_TENSOR","dtype":"F64","shape":[1],
            "payload_f64":[1.0],"time_ref":1,"provenance_ref":1,"unit":"DIMENSIONLESS"}],
        "timebase":[{"time_id":1,"unit":"fs","start":0.0,"step":1.0,"count":1,"timebase_hash":"test-time"}],
        "nanl_tokens":[1],
        "events":[{"event_id":1,"event_kind":"PHASE_TILE","time_ref":1,"source_state":1,
            "target_state":1,"value":1.0,"unit":"DIMENSIONLESS"}],
        "thermo":[{"thermo_id":1,"state_ref":1,"value":1.0,"unit":"DIMENSIONLESS","signal_kind":"TEST"}],
        "provenance":[{"provenance_id":1,"label":"test","synthetic":true,"reason":"operand bound unit test"}]
    })).unwrap();
    br::compile_reference_corpus_with_debug(&corpus,&serde_json::to_vec(&corpus).unwrap(),&path).unwrap();
    br::validate_packet(&path,ValidationLevel::L3).expect("unaltered compiler fixture must pass before any mutation");
    br::read_packet(&path).unwrap()
}

fn admitted(mut packet:br::PCodePacket, instr:PcInstr32) -> bool {
    let instruction=packet.sections.iter().find(|s|s.section_type==SectionKind::InstrStream).unwrap();
    let offset=instruction.offset as usize;
    packet.bytes[offset..offset+32].copy_from_slice(&instr.encode());
    let count=u16::from_le_bytes(packet.bytes[24..26].try_into().unwrap()) as usize;
    let directory=u64::from_le_bytes(packet.bytes[28..36].try_into().unwrap()) as usize;
    for i in 0..count {
        let row=directory+i*96;let start=u64::from_le_bytes(packet.bytes[row+8..row+16].try_into().unwrap()) as usize;
        let length=u64::from_le_bytes(packet.bytes[row+16..row+24].try_into().unwrap()) as usize;
        let hash=blake3::hash(&packet.bytes[start..start+length]);packet.bytes[row+64..row+96].copy_from_slice(hash.as_bytes());
    }
    packet.bytes[196..228].fill(0);let hash=blake3::hash(&packet.bytes);packet.bytes[196..228].copy_from_slice(hash.as_bytes());
    let tmp=tempfile::tempdir().unwrap();let path=tmp.path().join("probe.pcode");std::fs::write(&path,packet.bytes).unwrap();
    br::validate_packet(&path,ValidationLevel::L3).is_ok()
}

fn instruction() -> PcInstr32 {
    PcInstr32::new(br::opcode::STATE_HASH,0,OperandKind::None,0,OperandKind::None,0)
}

#[test]
fn all_four_slots_enforce_nonempty_table_reference_bounds() {
    // Each pair has a real, nonempty table in the compiled reference packet.
    for kind in [4u8,5,6,7,8,10,12,16,17] {
        for slot in 0..4 {
            let mut good=instruction();let mut bad=instruction();
            match slot {
                0=>{good.dst_kind=kind;bad.dst_kind=kind;bad.dst=u32::MAX;},
                1=>{good.src0_kind=kind;bad.src0_kind=kind;bad.src0=u32::MAX;},
                2=>{good.src1_kind=kind;bad.src1_kind=kind;bad.src1=u32::MAX;},
                _=>{good.src2_kind=kind;bad.src2_kind=kind;bad.src2_or_len=u32::MAX;},
            }
            assert!(admitted(packet(),good),"valid kind={kind} slot={slot}");
            assert!(!admitted(packet(),bad),"out-of-range kind={kind} slot={slot}");
        }
    }
}

#[test]
fn empty_or_undefined_reference_targets_reject_without_scalar_regression() {
    for kind in [3u8,9,11,13,14,18,19,20] {
        let mut bad=instruction();bad.src2_kind=kind;
        assert!(!admitted(packet(),bad),"unavailable target kind={kind}");
    }
    for kind in [0u8,1,2] {
        let mut scalar=instruction();scalar.src2_kind=kind;scalar.src2_or_len=u32::MAX;
        assert!(admitted(packet(),scalar));
    }
    for value in [1u32,12] {let mut i=instruction();i.src2_kind=15;i.src2_or_len=value;assert!(admitted(packet(),i));}
    for value in [0u32,13,u32::MAX] {let mut i=instruction();i.src2_kind=15;i.src2_or_len=value;assert!(!admitted(packet(),i));}
}

#[test]
fn resealed_directory_counts_cannot_authorize_nonexistent_records() {
    for kind in [SectionKind::InstrStream,SectionKind::StateTable,SectionKind::TimeTable,
        SectionKind::NanlTable,SectionKind::EventTable,SectionKind::TensorTable,
        SectionKind::RewardTable,SectionKind::ProvenanceTable,SectionKind::SealChain,
        SectionKind::OperandTable,SectionKind::GraphTable,SectionKind::ActionTable] {
        let mut p=packet();let index=p.sections.iter().position(|s|s.section_type==kind).unwrap();
        let directory=u64::from_le_bytes(p.bytes[28..36].try_into().unwrap()) as usize;
        p.bytes[directory+index*96+24..directory+index*96+32].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(!admitted(p,instruction()),"accepted forged {kind:?} record count");
    }
    let mut p=packet();let index=p.sections.iter().position(|s|s.section_type==SectionKind::StateTable).unwrap();
    let directory=u64::from_le_bytes(p.bytes[28..36].try_into().unwrap()) as usize;
    p.bytes[directory+index*96+24..directory+index*96+32].copy_from_slice(&(u32::MAX as u64+1).to_le_bytes());
    let mut forged=instruction();forged.src2_kind=OperandKind::StateRef as u8;forged.src2_or_len=u32::MAX;
    assert!(!admitted(p,forged),"forged count must not bypass typed reference bounds");
    let mut p=packet();let index=p.sections.iter().position(|s|s.section_type==SectionKind::GraphTable).unwrap();
    let directory=u64::from_le_bytes(p.bytes[28..36].try_into().unwrap()) as usize;
    p.bytes[directory+index*96+2..directory+index*96+4].copy_from_slice(&(SectionKind::StateTable as u16).to_le_bytes());
    assert!(!admitted(p,instruction()),"duplicate target table must not be ambiguous");
}

#[test]
fn existing_manifest_producer_is_registered_without_renumbering_existing_ops() {
    assert_eq!(br::opcode_id_for_name("MD.SPIKE_MANIFEST_COMMIT"),Some(0x0732));
    assert_eq!(br::opcode_id_for_name("MD.PHOTO_UV_PUMP_SUPPRESSED"),Some(0x0731));
    let rows=br::opcode_rows_for_docs();let ids:std::collections::BTreeSet<_>=rows.iter().map(|r|r.0).collect();
    assert_eq!(rows.len(),ids.len());
    assert_eq!(br::opcode_id_for_name("MD.SENSOR_CHUNK_COMMIT"),None);
}

#[test]
fn canonical_empty_event_packet_keeps_exact_policy_identity() {
    let path=Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../appliance_payload_reference/packets/prismos_hyper_runtime_reference.pcode");
    let parsed=br::read_packet(&path).unwrap();
    assert_eq!(parsed.event_count,0);
    assert!(br::validate_packet(&path,ValidationLevel::L3).is_ok());
    assert!(br::load_validated_md_engine_policy(&path).is_ok());
    let mut unrelated=instruction();unrelated.src0_kind=OperandKind::EventRef as u8;
    assert!(!admitted(parsed,unrelated));
}
