//! # PRISM Diff - Atomic Displacement Analyzer
//!
//! Mathematical analyzer that compares two PTB files to identify regions of high atomic movement.
//! Identifies "Cryptic Epitopes" - regions that undergo significant structural changes during dynamics.
//!
//! ## Usage
//! ```bash
//! cargo run --bin prism-diff -- original.ptb relaxed.ptb
//! ```
//!
//! ## Analysis Features
//! - Calculates Euclidean distance for every atom pair
//! - Filters thermal noise (movements < 0.5 Å)
//! - Identifies top 20 most mobile regions
//! - Provides cryptic epitope analysis without PyMOL dependency

use clap::Parser;
use prism_io::{holographic::PtbStructure, sovereign_types::Atom, Result};
use std::path::Path;

#[derive(Parser)]
#[command(
    name = "prism-diff",
    about = "Analyze atomic displacement between two PTB files",
    long_about = "Mathematical analyzer that identifies cryptic epitopes by calculating atomic movement between initial and relaxed protein structures."
)]
struct Args {
    /// Original/initial PTB file
    #[arg(value_name = "ORIGINAL.ptb")]
    original: String,

    /// Relaxed/final PTB file
    #[arg(value_name = "RELAXED.ptb")]
    relaxed: String,

    /// Minimum displacement threshold in Angstroms (default: 0.5)
    #[arg(short, long, default_value = "0.5")]
    threshold: f64,

    /// Number of top movers to display (default: 20)
    #[arg(short = 'n', long, default_value = "20")]
    top_count: usize,

    /// Export detailed analysis to file
    #[arg(short, long)]
    export: Option<String>,
}

#[derive(Debug, Clone)]
struct AtomMovement {
    atom_index: usize,
    residue_id: u16,
    element: u8,
    displacement: f64,
    original_coords: [f32; 3],
    relaxed_coords: [f32; 3],
}

/// Calculate Euclidean distance between two 3D points
fn calculate_displacement(coords1: [f32; 3], coords2: [f32; 3]) -> f64 {
    let dx = (coords2[0] - coords1[0]) as f64;
    let dy = (coords2[1] - coords1[1]) as f64;
    let dz = (coords2[2] - coords1[2]) as f64;

    (dx * dx + dy * dy + dz * dz).sqrt()
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
        _ => "X",
    }
}

/// Analyze atomic movements between two structures
fn analyze_movements(
    original_atoms: &[Atom],
    relaxed_atoms: &[Atom],
    threshold: f64,
) -> Result<Vec<AtomMovement>> {
    if original_atoms.len() != relaxed_atoms.len() {
        return Err(prism_io::PrismIoError::FormatError(format!(
            "Atom count mismatch: original={}, relaxed={}",
            original_atoms.len(),
            relaxed_atoms.len()
        )));
    }

    let mut movements = Vec::new();

    for (i, (orig, relax)) in original_atoms.iter().zip(relaxed_atoms.iter()).enumerate() {
        let displacement = calculate_displacement(orig.coords, relax.coords);

        if displacement >= threshold {
            movements.push(AtomMovement {
                atom_index: i,
                residue_id: orig.residue_id,
                element: orig.element,
                displacement,
                original_coords: orig.coords,
                relaxed_coords: relax.coords,
            });
        }
    }

    // Sort by displacement (highest first)
    movements.sort_by(|a, b| b.displacement.partial_cmp(&a.displacement).unwrap());

    Ok(movements)
}

/// Print cryptic epitope report to stdout
fn print_cryptic_epitope_report(movements: &[AtomMovement], top_count: usize) {
    println!("🧬 CRYPTIC EPITOPE ANALYSIS REPORT");
    println!("=====================================");
    println!();

    if movements.is_empty() {
        println!("📊 No significant atomic movements detected above threshold.");
        println!("🔬 Structure appears stable - no cryptic epitopes identified.");
        return;
    }

    println!("📈 Summary:");
    println!("   Total moving atoms: {}", movements.len());
    println!(
        "   Average displacement: {:.3} Å",
        movements.iter().map(|m| m.displacement).sum::<f64>() / movements.len() as f64
    );
    println!(
        "   Maximum displacement: {:.3} Å",
        movements[0].displacement
    );
    println!();

    println!(
        "🎯 TOP {} MOST MOBILE REGIONS:",
        top_count.min(movements.len())
    );
    println!("=================================");

    for (rank, movement) in movements.iter().take(top_count).enumerate() {
        println!(
            "{:2}. Residue {:>4} ({}) moved {:.3} Å",
            rank + 1,
            movement.residue_id,
            element_symbol(movement.element),
            movement.displacement
        );

        // Show coordinate change for top 5
        if rank < 5 {
            println!(
                "     From: ({:>7.3}, {:>7.3}, {:>7.3})",
                movement.original_coords[0],
                movement.original_coords[1],
                movement.original_coords[2]
            );
            println!(
                "     To:   ({:>7.3}, {:>7.3}, {:>7.3})",
                movement.relaxed_coords[0], movement.relaxed_coords[1], movement.relaxed_coords[2]
            );
        }
        println!();
    }

    // Analyze movement patterns by residue
    let mut residue_movements: std::collections::HashMap<u16, Vec<f64>> =
        std::collections::HashMap::new();
    for movement in movements {
        residue_movements
            .entry(movement.residue_id)
            .or_insert_with(Vec::new)
            .push(movement.displacement);
    }

    println!("🔬 CRYPTIC EPITOPE HOTSPOTS:");
    println!("============================");

    let mut residue_analysis: Vec<_> = residue_movements
        .iter()
        .map(|(residue_id, displacements)| {
            let avg_displacement = displacements.iter().sum::<f64>() / displacements.len() as f64;
            let max_displacement = displacements.iter().fold(0.0f64, |max, &d| max.max(d));
            (
                *residue_id,
                avg_displacement,
                max_displacement,
                displacements.len(),
            )
        })
        .collect();

    residue_analysis.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // Sort by average displacement

    for (i, (residue_id, avg_disp, max_disp, atom_count)) in
        residue_analysis.iter().take(10).enumerate()
    {
        println!(
            "{:2}. Residue {:>4}: {:.3} Å avg ({:.3} Å max) - {} atoms moving",
            i + 1,
            residue_id,
            avg_disp,
            max_disp,
            atom_count
        );
    }

    println!();
    println!("💡 INTERPRETATION:");
    println!("   • High displacement regions may indicate:");
    println!("     - Cryptic allosteric sites");
    println!("     - Flexible loop regions");
    println!("     - Potential drug binding sites");
    println!("     - Conformational change hotspots");
    println!();
    println!("🌐 Export this data to visualization tools for detailed analysis.");
}

/// Export detailed analysis to file
fn export_analysis(movements: &[AtomMovement], output_path: &str) -> Result<()> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(output_path)?;

    writeln!(file, "# PRISM Diff - Atomic Displacement Analysis")?;
    writeln!(file, "# Generated by PRISM4D Analysis Suite")?;
    writeln!(file, "#")?;
    writeln!(file, "# Format: AtomIndex,ResidueID,Element,Displacement(Å),OrigX,OrigY,OrigZ,RelaxX,RelaxY,RelaxZ")?;

    for movement in movements {
        writeln!(
            file,
            "{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            movement.atom_index,
            movement.residue_id,
            movement.element,
            movement.displacement,
            movement.original_coords[0],
            movement.original_coords[1],
            movement.original_coords[2],
            movement.relaxed_coords[0],
            movement.relaxed_coords[1],
            movement.relaxed_coords[2]
        )?;
    }

    tracing::info!("📊 Detailed analysis exported to {}", output_path);
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("🔬 PRISM Diff - Atomic Displacement Analyzer");
    println!("============================================");
    println!();

    // Validate input files exist
    if !Path::new(&args.original).exists() {
        eprintln!("❌ Error: Original file '{}' not found", args.original);
        std::process::exit(1);
    }

    if !Path::new(&args.relaxed).exists() {
        eprintln!("❌ Error: Relaxed file '{}' not found", args.relaxed);
        std::process::exit(1);
    }

    // Load original structure
    tracing::info!("📂 Loading original structure: {}", args.original);
    let mut original_structure = PtbStructure::load(&args.original)?;
    original_structure.verify_integrity()?;
    let original_atoms = original_structure.atoms()?.to_vec();

    // Load relaxed structure
    tracing::info!("📂 Loading relaxed structure: {}", args.relaxed);
    let mut relaxed_structure = PtbStructure::load(&args.relaxed)?;
    relaxed_structure.verify_integrity()?;
    let relaxed_atoms = relaxed_structure.atoms()?.to_vec();

    println!("✅ Loaded structures:");
    println!("   Original: {} atoms", original_atoms.len());
    println!("   Relaxed:  {} atoms", relaxed_atoms.len());
    println!("   Threshold: {:.1} Å", args.threshold);
    println!();

    // Analyze movements
    tracing::info!("🔍 Analyzing atomic displacements...");
    let movements = analyze_movements(&original_atoms, &relaxed_atoms, args.threshold)?;

    // Print analysis report
    print_cryptic_epitope_report(&movements, args.top_count);

    // Export detailed data if requested
    if let Some(export_path) = args.export {
        export_analysis(&movements, &export_path)?;
        println!("📊 Detailed analysis exported to: {}", export_path);
    }

    tracing::info!("✅ Atomic displacement analysis complete");

    Ok(())
}
