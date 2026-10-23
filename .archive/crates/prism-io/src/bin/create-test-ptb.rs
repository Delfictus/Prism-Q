use prism_io::{sovereign_types::Atom, HolographicBinaryFormat, Result};

fn main() -> Result<()> {
    // Create test atoms with coordinates DIFFERENT from template
    let test_atoms = vec![
        Atom {
            coords: [100.123, 200.456, 300.789], // Very different from template (10,20,30)
            element: 7,                          // Nitrogen
            residue_id: 0,
            atom_type: 1,
            charge: -0.5,
            radius: 1.55,
            _reserved: [0; 4],
        },
        Atom {
            coords: [101.234, 201.567, 301.890], // Different from template (11,21,31)
            element: 6,                          // Carbon
            residue_id: 0,
            atom_type: 2,
            charge: 0.0,
            radius: 1.70,
            _reserved: [0; 4],
        },
        Atom {
            coords: [102.345, 202.678, 302.901], // Different from template (12,22,32)
            element: 6,                          // Carbon
            residue_id: 0,
            atom_type: 3,
            charge: 0.1,
            radius: 1.70,
            _reserved: [0; 4],
        },
        Atom {
            coords: [103.456, 203.789, 303.012], // Different from template (13,23,33)
            element: 8,                          // Oxygen
            residue_id: 0,
            atom_type: 4,
            charge: -0.6,
            radius: 1.52,
            _reserved: [0; 4],
        },
    ];

    println!("Creating test PTB file with {} atoms:", test_atoms.len());
    for (i, atom) in test_atoms.iter().enumerate() {
        println!(
            "  Atom {}: element={} at ({:.3}, {:.3}, {:.3})",
            i + 1,
            atom.element,
            atom.coords[0],
            atom.coords[1],
            atom.coords[2]
        );
    }

    // Create holographic binary format
    let export_hash = [42u8; 32]; // Test hash
    let ptb_data = HolographicBinaryFormat::new()
        .with_atoms(test_atoms)
        .with_source_hash(export_hash);

    // Save to test.ptb
    let output_path = "test.ptb";
    ptb_data.write_to_file(output_path)?;

    println!("✅ Created test PTB file: {}", output_path);

    Ok(())
}
