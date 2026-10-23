use prismcode::{registry_hash, validate_registry, OpcodeRegistry};
use std::env;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args()
        .nth(1)
        .ok_or("usage: prismcode-registry <registry.json>")?;
    let text = fs::read_to_string(path)?;
    let registry: OpcodeRegistry = serde_json::from_str(&text)?;
    validate_registry(&registry)?;
    println!("{}", registry_hash(&registry)?);
    Ok(())
}
