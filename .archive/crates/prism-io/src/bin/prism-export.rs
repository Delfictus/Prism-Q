//! # PRISM Export - PTB to PDB Converter
//!
//! Converts PRISM4D holographic binary (.ptb) files to industry-standard PDB format
//! for compatibility with web and desktop molecular viewers (RCSB, Mol*, PyMOL, etc.)
//!
//! ## Usage
//! ```bash
//! cargo run --bin prism-export -- input.ptb output.pdb
//! ```
//!
//! ## PDB Format Compliance
//! - Follows PDB Format Specification v3.3
//! - Proper column spacing and alignment for maximum compatibility
//! - Sequential residue numbering for continuous chain visualization

use clap::Parser;
use prism_io::{holographic::PtbStructure, sovereign_types::Atom, Result};
use std::{
    fs::File,
    io::{BufRead, BufReader, Write},
    path::Path,
};

#[derive(Parser)]
#[command(
    name = "prism-export",
    about = "Convert PRISM4D PTB files to standard PDB format",
    long_about = "Converts PRISM4D holographic binary (.ptb) files to industry-standard PDB format for compatibility with molecular viewers. Supports template mode for preserving original PDB metadata."
)]
struct Args {
    /// Input PTB file path
    #[arg(value_name = "INPUT.ptb")]
    input: String,

    /// Output PDB file path
    #[arg(value_name = "OUTPUT.pdb")]
    output: String,

    /// Template PDB file to preserve metadata (recommended for PyMOL compatibility)
    #[arg(short, long, value_name = "TEMPLATE.pdb")]
    template: Option<String>,

    /// Chain identifier (default: A) - only used in skeleton mode
    #[arg(short, long, default_value = "A")]
    chain: char,

    /// Set B-factor value for all atoms (default: 0.00) - only used in skeleton mode
    #[arg(short, long, default_value = "0.00")]
    bfactor: f32,
}

/// Element symbol mapping from atomic number
fn element_symbol(atomic_number: u8) -> &'static str {
    match atomic_number {
        1 => "H",
        6 => "C",
        7 => "N",
        8 => "O",
        15 => "P",
        16 => "S",
        _ => "X", // Unknown element
    }
}

/// Standard amino acid mapping (simplified for structural visualization)
fn residue_name_from_id(residue_id: u16) -> &'static str {
    // For molecular dynamics visualization, we use a simplified mapping
    // Real production would require sequence data for accurate residue names
    match residue_id % 20 {
        0 => "ALA",
        1 => "ARG",
        2 => "ASN",
        3 => "ASP",
        4 => "CYS",
        5 => "GLN",
        6 => "GLU",
        7 => "GLY",
        8 => "HIS",
        9 => "ILE",
        10 => "LEU",
        11 => "LYS",
        12 => "MET",
        13 => "PHE",
        14 => "PRO",
        15 => "SER",
        16 => "THR",
        17 => "TRP",
        18 => "TYR",
        19 => "VAL",
        _ => "UNK",
    }
}

/// Atom name based on element and index within residue
fn atom_name_from_element(element: u8, atom_index: usize) -> String {
    let element_str = element_symbol(element);
    match element {
        6 => match atom_index % 4 {
            // Carbon atoms
            0 => "CA".to_string(), // Alpha carbon
            1 => "CB".to_string(), // Beta carbon
            2 => "CG".to_string(), // Gamma carbon
            _ => format!("C{}", atom_index),
        },
        7 => "N".to_string(), // Nitrogen
        8 => "O".to_string(), // Oxygen
        _ => element_str.to_string(),
    }
}

/// Check if a PDB line is an ATOM or HETATM record
fn is_atom_line(line: &str) -> bool {
    line.starts_with("ATOM  ") || line.starts_with("HETATM")
}

/// Process template PDB file and replace coordinates with PTB data using strict column slicing
fn process_template_mode(template_path: &str, atoms: &[Atom], output_path: &str) -> Result<usize> {
    tracing::info!("📋 Processing template PDB: {}", template_path);

    // Open template file
    let template_file =
        File::open(template_path).map_err(|e| prism_io::PrismIoError::IoError(e))?;
    let reader = BufReader::new(template_file);

    // Open output file
    let mut output_file =
        File::create(output_path).map_err(|e| prism_io::PrismIoError::IoError(e))?;

    let mut atom_idx = 0;
    let mut total_atoms_processed = 0;

    // Process template line by line with FORCED coordinate injection and debug verification
    for line_result in reader.lines() {
        let line = line_result.map_err(|e| prism_io::PrismIoError::IoError(e))?;

        // MANDATORY COORDINATE INJECTION WITH DEBUG VERIFICATION
        if line.starts_with("ATOM") || line.starts_with("HETATM") {
            if atom_idx < atoms.len() {
                let atom = &atoms[atom_idx];

                // FORCED COORDINATE INJECTION - EXACT SPECIFICATION
                // Slice the template line
                let prefix = &line[0..30]; // Metadata
                let suffix = if line.len() > 54 { &line[54..] } else { "" };

                // Format new coords from the PTB atom
                let coords = format!(
                    "{:8.3}{:8.3}{:8.3}",
                    atom.coords[0], atom.coords[1], atom.coords[2]
                );

                // Construct the new line
                let new_line = format!("{}{}{}", prefix, coords, suffix);

                // LOUD VERIFICATION: Print debug for first atom only
                if atom_idx == 0 {
                    println!("DEBUG: Original: {}", line);
                    println!("DEBUG: Modified: {}", new_line);
                }

                // WRITE THE MODIFIED LINE (NOT THE ORIGINAL)
                writeln!(output_file, "{}", new_line)
                    .map_err(|e| prism_io::PrismIoError::IoError(e))?;

                atom_idx += 1;
                total_atoms_processed += 1;
            } else {
                // Safety fallback if PTB has fewer atoms than Template
                writeln!(output_file, "{}", line)
                    .map_err(|e| prism_io::PrismIoError::IoError(e))?;
            }
        } else {
            // Keep headers/remarks unchanged
            writeln!(output_file, "{}", line).map_err(|e| prism_io::PrismIoError::IoError(e))?;
        }
    }

    // Safety check: ensure atom counts match
    if atom_idx != atoms.len() {
        return Err(prism_io::PrismIoError::FormatError(format!(
            "Atom count mismatch: Template has {} atoms, PTB has {} atoms",
            atom_idx,
            atoms.len()
        )));
    }

    tracing::info!(
        "✅ Template processing complete: {} atoms updated with strict slicing",
        total_atoms_processed
    );
    Ok(total_atoms_processed)
}

/// Convert PTB atoms to PDB format lines (skeleton mode)
fn atoms_to_pdb_lines(atoms: &[Atom], chain: char, bfactor: f32) -> Vec<String> {
    let mut lines = Vec::new();

    for (atom_index, atom) in atoms.iter().enumerate() {
        let serial = (atom_index + 1).min(99999); // PDB serial number limit
        let atom_name = atom_name_from_element(atom.element, atom_index);
        let residue_name = residue_name_from_id(atom.residue_id);
        let residue_seq = (atom.residue_id as usize + 1).min(9999); // PDB residue limit

        // PDB Format Specification v3.3 - ATOM record
        // Columns 1-6: "ATOM  "
        // Columns 7-11: Serial number (right-justified)
        // Column 12: Space
        // Columns 13-16: Atom name (left-justified, space-padded)
        // Column 17: Alternate location (space)
        // Columns 18-20: Residue name (right-justified)
        // Column 21: Space
        // Column 22: Chain identifier
        // Columns 23-26: Residue sequence number (right-justified)
        // Column 27: Insertion code (space)
        // Columns 28-30: Spaces
        // Columns 31-38: X coordinate (right-justified, 3 decimal places)
        // Columns 39-46: Y coordinate (right-justified, 3 decimal places)
        // Columns 47-54: Z coordinate (right-justified, 3 decimal places)
        // Columns 55-60: Occupancy (right-justified, 2 decimal places)
        // Columns 61-66: Temperature factor (right-justified, 2 decimal places)
        // Columns 67-76: Spaces
        // Columns 77-78: Element symbol (right-justified)

        let line = format!(
            "{:<6}{:>5} {:<4} {:>3} {}{:>4}    {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}          {:>2}",
            "ATOM",                       // Record type
            serial,                       // Serial number
            atom_name,                    // Atom name
            residue_name,                 // Residue name
            chain,                        // Chain identifier
            residue_seq,                  // Residue sequence number
            atom.coords[0],               // X coordinate
            atom.coords[1],               // Y coordinate
            atom.coords[2],               // Z coordinate
            1.00,                         // Occupancy (default: fully occupied)
            bfactor,                      // B-factor (temperature factor)
            element_symbol(atom.element)  // Element symbol
        );

        lines.push(line);
    }

    lines
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Validate input file exists
    if !Path::new(&args.input).exists() {
        eprintln!("❌ Error: Input file '{}' not found", args.input);
        std::process::exit(1);
    }

    // Validate template file if provided
    if let Some(ref template_path) = args.template {
        if !Path::new(template_path).exists() {
            eprintln!("❌ Error: Template file '{}' not found", template_path);
            std::process::exit(1);
        }
    }

    // Load PTB structure
    tracing::info!("📂 Loading PTB structure from {}", args.input);
    let mut structure = PtbStructure::load(&args.input)?;

    // Verify integrity
    structure.verify_integrity()?;

    // Extract atoms
    let atoms = structure.atoms()?;
    tracing::info!("✅ Loaded {} atoms from PTB file", atoms.len());

    if atoms.is_empty() {
        eprintln!("❌ Error: No atoms found in PTB file");
        std::process::exit(1);
    }

    // Choose processing mode based on template argument
    match args.template {
        Some(template_path) => {
            // TEMPLATE MODE: Preserve original PDB metadata
            tracing::info!("🔄 Using Template Mode for PyMOL compatibility");
            tracing::info!("📋 Template: {}", template_path);

            let atoms_processed = process_template_mode(&template_path, atoms, &args.output)?;

            tracing::info!("✅ Template-based PDB export complete");
            tracing::info!("📊 Output: {}", args.output);
            tracing::info!("🎭 Metadata preserved from template for PyMOL cartoon rendering");

            println!(
                "✅ Successfully updated {} atoms using template {}",
                atoms_processed, template_path
            );
            println!("🎭 PyMOL-compatible PDB ready: {}", args.output);
        }
        None => {
            // SKELETON MODE: Generate basic PDB structure
            tracing::info!("🔄 Using Skeleton Mode (basic PDB generation)");
            tracing::info!("💡 Tip: Use --template for PyMOL cartoon compatibility");

            // Convert to PDB format
            let pdb_lines = atoms_to_pdb_lines(atoms, args.chain, args.bfactor);

            // Write PDB file
            tracing::info!("💾 Writing PDB file to {}", args.output);
            let mut file = File::create(&args.output)?;

            // Write PDB header
            writeln!(
                file,
                "HEADER    MOLECULAR DYNAMICS                      01-JAN-26   PRSM"
            )?;
            writeln!(
                file,
                "TITLE     PRISM4D NIPAH VIRUS G GLYCOPROTEIN DYNAMICS"
            )?;
            writeln!(
                file,
                "REMARK   1 GENERATED BY PRISM4D EXPORT TOOL (SKELETON MODE)"
            )?;
            writeln!(file, "REMARK   2 CONVERTED FROM HOLOGRAPHIC BINARY FORMAT")?;
            writeln!(file, "REMARK   3 SOURCE: {}", args.input)?;
            writeln!(
                file,
                "REMARK   4 WARNING: Skeleton mode - use --template for PyMOL compatibility"
            )?;

            // Write atom records
            for line in pdb_lines {
                writeln!(file, "{}", line)?;
            }

            // Write PDB footer
            writeln!(file, "END")?;

            tracing::info!(
                "✅ Skeleton PDB export complete: {} atoms written",
                atoms.len()
            );
            tracing::info!("📊 Output: {}", args.output);
            tracing::info!("⚠️  Note: Basic structure only - PyMOL may not render cartoons");

            println!(
                "✅ Successfully exported {} atoms to {}",
                atoms.len(),
                args.output
            );
            println!("📊 Basic PDB structure ready for molecular visualization");
            println!("💡 For PyMOL cartoon rendering, use: --template original.pdb");
        }
    }

    tracing::info!("🌐 Compatible with: RCSB PDB Viewer, Mol*, PyMOL, ChimeraX");

    Ok(())
}
