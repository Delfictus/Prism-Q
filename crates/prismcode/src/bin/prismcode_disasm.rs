use prismcode::{disassemble_program, format_disassembly, OpcodeRegistry, Program};
use std::env;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let program_path = args
        .next()
        .ok_or("usage: prismcode-disasm <program.json> <registry.json>")?;
    let registry_path = args
        .next()
        .ok_or("usage: prismcode-disasm <program.json> <registry.json>")?;

    let program: Program = serde_json::from_str(&fs::read_to_string(program_path)?)?;
    let registry: OpcodeRegistry = serde_json::from_str(&fs::read_to_string(registry_path)?)?;
    let lines = disassemble_program(&program, &registry)?;
    println!("{}", format_disassembly(&lines));
    Ok(())
}
