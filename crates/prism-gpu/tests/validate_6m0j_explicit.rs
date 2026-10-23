//! PRISM-4D 6M0J RBD Explicit Solvent Validation
//!
//! Validates explicit solvent MD using Phase 7-8 fused kernels.
//! Uses SARS-CoV-2 RBD (6M0J) as test structure.
//!
//! Run with: cargo test -p prism-gpu --test validate_6m0j_explicit --features cuda -- --ignored --nocapture
//!
//! Expected performance: ~6,600 ns/day on RTX 3060

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[cfg(feature = "cuda")]
mod tests {
    use super::*;
    use cudarc::driver::CudaContext;
    use prism_gpu::{AmberMegaFusedHmc, HConstraintCluster};
    use prism_physics::amber_ff14sb::{get_atom_mass, get_lj_param, AmberAtomType, LJParam};
    use prism_physics::{SolvationBox, SolvationConfig, TIP3PWater};

    /// Known escape mutation sites for validation (from publication)
    const ESCAPE_SITES: &[(usize, &str, &str, f32)] = &[
        (496, "G496S", "Omicron", 2.99),
        (375, "S375F", "Omicron", 1.46),
        (477, "S477N", "Omicron", 1.35),
        (346, "R346K", "Omicron BA.1", 0.83),
        (373, "S373P", "Omicron", 0.05),
        (484, "E484K/A", "Beta/Gamma/Omicron", 0.03),
        (478, "T478K", "Delta/Omicron", -0.06),
        (501, "N501Y", "Alpha/Beta/Gamma", -0.37),
        (505, "Y505H", "Omicron", -0.39),
        (417, "K417N", "Beta/Omicron", -0.44),
    ];

    /// Find the 6M0J PDB file
    /// Prioritizes OpenMM-prepared structure for better stability
    fn find_pdb_file() -> PathBuf {
        // Prefer OpenMM-prepared structure (properly protonated, minimized)
        let candidates = [
            "data/structures/6M0J_RBD_prepared.pdb", // OpenMM-prepared (best)
            "../data/structures/6M0J_RBD_prepared.pdb",
            "../../data/structures/6M0J_RBD_prepared.pdb",
            "publication/figures/6M0J_RBD_RMSF_bfactor.pdb", // Fallback
            "../publication/figures/6M0J_RBD_RMSF_bfactor.pdb",
            "../../publication/figures/6M0J_RBD_RMSF_bfactor.pdb",
        ];

        for path in &candidates {
            let p = PathBuf::from(path);
            if p.exists() {
                println!("Using PDB: {:?}", p);
                return p;
            }
        }

        panic!("Could not find 6M0J PDB file. Run: python3 scripts/prepare_structure_openmm.py to generate");
    }

    /// Parse PDB file for explicit solvent simulation
    fn parse_pdb_for_solvation(
        content: &str,
    ) -> (
        Vec<f32>,
        Vec<AmberAtomType>,
        Vec<f32>,
        Vec<(usize, String, String)>,
    ) {
        let mut positions = Vec::new();
        let mut atom_types = Vec::new();
        let mut charges = Vec::new();
        let mut atom_info = Vec::new();

        for line in content.lines() {
            if !line.starts_with("ATOM") {
                continue;
            }

            // Parse coordinates
            let x: f32 = line
                .get(30..38)
                .unwrap_or("0")
                .trim()
                .parse()
                .unwrap_or(0.0);
            let y: f32 = line
                .get(38..46)
                .unwrap_or("0")
                .trim()
                .parse()
                .unwrap_or(0.0);
            let z: f32 = line
                .get(46..54)
                .unwrap_or("0")
                .trim()
                .parse()
                .unwrap_or(0.0);
            positions.extend_from_slice(&[x, y, z]);

            // Parse atom/residue names
            let atom_name = line.get(12..16).unwrap_or("").trim().to_string();
            let res_name = line.get(17..20).unwrap_or("").trim().to_string();
            let res_id: usize = line.get(22..26).unwrap_or("0").trim().parse().unwrap_or(0);

            // Get AMBER atom type
            let atom_type = AmberAtomType::from_pdb(&res_name, &atom_name);
            atom_types.push(atom_type);

            // Simplified charge assignment (in production, use ff14SB parameters)
            let charge = get_simple_charge(&res_name, &atom_name);
            charges.push(charge);

            atom_info.push((res_id, res_name, atom_name));
        }

        (positions, atom_types, charges, atom_info)
    }

    /// Simple charge assignment (simplified for testing)
    fn get_simple_charge(res_name: &str, atom_name: &str) -> f32 {
        match (res_name, atom_name) {
            // Backbone
            (_, "N") => -0.4157,
            (_, "H" | "HN") => 0.2719,
            (_, "CA") => 0.0337,
            (_, "C") => 0.5973,
            (_, "O") => -0.5679,
            // Charged residues
            ("LYS", "NZ") => -0.3854,
            ("ARG", "CZ") => 0.8076,
            ("ASP", "OD1" | "OD2") => -0.8014,
            ("GLU", "OE1" | "OE2") => -0.8195,
            _ => 0.0,
        }
    }

    /// Extract Cα indices from PDB
    fn extract_ca_indices(content: &str) -> Vec<usize> {
        let mut ca_indices = Vec::new();
        let mut atom_idx = 0;

        for line in content.lines() {
            if line.starts_with("ATOM") {
                let atom_name = line.get(12..16).unwrap_or("").trim();
                if atom_name == "CA" {
                    ca_indices.push(atom_idx);
                }
                atom_idx += 1;
            }
        }

        ca_indices
    }

    /// Extract Cα positions from full position array
    fn extract_ca_positions(all_positions: &[f32], ca_indices: &[usize]) -> Vec<[f32; 3]> {
        ca_indices
            .iter()
            .map(|&idx| {
                [
                    all_positions[idx * 3],
                    all_positions[idx * 3 + 1],
                    all_positions[idx * 3 + 2],
                ]
            })
            .collect()
    }

    /// Compute RMSD between two Cα coordinate sets
    fn compute_ca_rmsd(ref_coords: &[[f32; 3]], coords: &[[f32; 3]]) -> f32 {
        if ref_coords.len() != coords.len() || ref_coords.is_empty() {
            return 0.0;
        }

        let n = ref_coords.len() as f32;
        let sum_sq: f32 = ref_coords
            .iter()
            .zip(coords.iter())
            .map(|(r, c)| {
                let dx = c[0] - r[0];
                let dy = c[1] - r[1];
                let dz = c[2] - r[2];
                dx * dx + dy * dy + dz * dz
            })
            .sum();

        (sum_sq / n).sqrt()
    }

    #[test]
    #[ignore] // Run explicitly: cargo test -p prism-gpu --test validate_6m0j_explicit -- --ignored --nocapture
    fn test_6m0j_explicit_solvent_validation() {
        println!("\n");
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║  PRISM-4D EXPLICIT SOLVENT VALIDATION                        ║");
        println!("║  6M0J SARS-CoV-2 RBD - Phase 7-8 Fused Kernels               ║");
        println!("╚══════════════════════════════════════════════════════════════╝");
        println!();

        // ═══════════════════════════════════════════════════════════════
        // STEP 1: Load PDB Structure
        // ═══════════════════════════════════════════════════════════════
        println!("━━━ Step 1: Loading 6M0J RBD Structure ━━━");

        let pdb_path = find_pdb_file();
        println!("  PDB file: {:?}", pdb_path);

        let pdb_content = fs::read_to_string(&pdb_path).expect("Failed to read PDB file");

        let (positions, atom_types, charges, atom_info) = parse_pdb_for_solvation(&pdb_content);
        let n_protein_atoms = atom_types.len();

        // Count residues
        let mut residue_ids: Vec<usize> = atom_info.iter().map(|(id, _, _)| *id).collect();
        residue_ids.dedup();
        let n_residues = residue_ids.len();

        println!("  Protein atoms: {}", n_protein_atoms);
        println!("  Residues: {}", n_residues);

        // Extract Cα positions for RMSD/RMSF
        let ca_indices = extract_ca_indices(&pdb_content);
        println!("  Cα atoms: {}", ca_indices.len());

        // ═══════════════════════════════════════════════════════════════
        // STEP 2: Create Solvation Box
        // ═══════════════════════════════════════════════════════════════
        println!("\n━━━ Step 2: Creating Explicit Solvent Box ━━━");

        let solv_config = SolvationConfig {
            padding: 10.0,
            min_protein_distance: 2.8,
            min_water_distance: 2.5,
            target_density: 0.997,
            max_box_dimension: 80.0,
            salt_concentration: 0.0, // No extra salt - just neutralize
        };

        let mut solvbox =
            SolvationBox::from_protein(&positions, &atom_types, &charges, &solv_config)
                .expect("Failed to create solvation box");

        let n_waters_added = solvbox.add_waters(&solv_config);
        let (n_na, n_cl) = solvbox.neutralize(&solv_config);

        let box_dims = solvbox.box_dimensions;
        let n_waters = solvbox.n_waters();
        let n_total = solvbox.total_atoms;

        println!(
            "  Box dimensions: {:.1} × {:.1} × {:.1} Å",
            box_dims[0], box_dims[1], box_dims[2]
        );
        println!("  Water molecules: {} ({} atoms)", n_waters, n_waters * 3);
        println!("  Ions: {} Na+, {} Cl-", n_na, n_cl);
        println!("  Total atoms: {}", n_total);

        // ═══════════════════════════════════════════════════════════════
        // STEP 3: Initialize GPU MD Engine
        // ═══════════════════════════════════════════════════════════════
        println!("\n━━━ Step 3: Initializing GPU MD Engine ━━━");

        let combined_topology = solvbox.to_topology();
        let combined_positions = solvbox.all_positions_flat();
        let water_oxygen_indices = solvbox.water_oxygen_indices();

        // Build H-constraint clusters for protein X-H bonds
        let h_clusters = build_protein_h_clusters(&atom_info, &atom_types);
        println!("  H-bond constraint clusters: {}", h_clusters.len());

        // Initialize GPU MD engine
        let context = CudaContext::new(0).expect("Failed to initialize CUDA");

        let mut hmc = AmberMegaFusedHmc::new(context.clone(), n_total)
            .expect("Failed to initialize GPU engine");

        // Convert topology to upload format
        // Bonds: (i, j, k_bond, r0)
        let bonds: Vec<(usize, usize, f32, f32)> = combined_topology
            .bonds
            .iter()
            .zip(combined_topology.bond_params.iter())
            .map(|(&(i, j), param)| (i as usize, j as usize, param.k, param.r0))
            .collect();

        // Angles: (i, j, k, k_angle, theta0)
        let angles: Vec<(usize, usize, usize, f32, f32)> = combined_topology
            .angles
            .iter()
            .zip(combined_topology.angle_params.iter())
            .map(|(&(i, j, k), param)| (i as usize, j as usize, k as usize, param.k, param.theta0))
            .collect();

        // Dihedrals: (i, j, k, l, k_dih, phase, n)
        let dihedrals: Vec<(usize, usize, usize, usize, f32, f32, f32)> = combined_topology
            .dihedrals
            .iter()
            .zip(combined_topology.dihedral_params.iter())
            .flat_map(|(&(i, j, k, l), params)| {
                params.iter().map(move |p| {
                    (
                        i as usize, j as usize, k as usize, l as usize, p.k, p.phase, p.n as f32,
                    )
                })
            })
            .collect();

        // Non-bonded params: (sigma, epsilon, charge, mass)
        let nb_params: Vec<(f32, f32, f32, f32)> = (0..n_total)
            .map(|i| {
                let lj = &combined_topology.lj_params[i];
                // Convert rmin_half to sigma: sigma = 2 * rmin_half / 2^(1/6)
                let sigma = lj.rmin_half * 2.0 / 1.122462f32;
                (
                    sigma,
                    lj.epsilon,
                    combined_topology.charges[i],
                    combined_topology.masses[i],
                )
            })
            .collect();

        // Exclusions: one HashSet per atom
        let mut exclusions: Vec<HashSet<usize>> = vec![HashSet::new(); n_total];
        for &(i, j) in &combined_topology.exclusions {
            exclusions[i as usize].insert(j as usize);
            exclusions[j as usize].insert(i as usize);
        }

        // Upload topology and positions
        hmc.upload_topology(
            &combined_positions,
            &bonds,
            &angles,
            &dihedrals,
            &nb_params,
            &exclusions,
        )
        .expect("Failed to upload topology");

        // Enable explicit solvent features
        hmc.enable_explicit_solvent(box_dims)
            .expect("Failed to enable explicit solvent");

        hmc.set_water_molecules(&water_oxygen_indices)
            .expect("Failed to setup SETTLE");

        if !h_clusters.is_empty() {
            hmc.set_h_constraints(&h_clusters)
                .expect("Failed to setup H-bond constraints");
        }

        // Enable mixed precision (Phase 7)
        let mp_config = prism_gpu::MixedPrecisionConfig::max_performance();
        hmc.enable_mixed_precision(mp_config)
            .expect("Failed to enable mixed precision");

        println!("  ✓ Explicit solvent: ENABLED");
        println!("  ✓ PME electrostatics: ENABLED");
        println!(
            "  ✓ SETTLE constraints: {} waters",
            water_oxygen_indices.len()
        );
        println!("  ✓ H-bond constraints: {} clusters", h_clusters.len());
        println!("  ✓ Mixed precision: ENABLED");
        println!("  ✓ Fused kernels: run_fused()");

        // Debug: Check SETTLE constraint satisfaction for first few waters
        if let Some((max_oh_viol, max_hh_viol)) = hmc.check_settle_constraints().unwrap() {
            println!(
                "  SETTLE violations (BEFORE any SETTLE applied): max_OH={:.6}Å, max_HH={:.6}Å",
                max_oh_viol, max_hh_viol
            );
        }

        // Debug: Print first 3 water geometries (INITIAL - before any processing)
        println!("  Initial water geometry (from solvation):");
        {
            let positions = hmc.get_positions().expect("get positions");
            let mut max_oh_err = 0.0f32;
            let mut max_hh_err = 0.0f32;
            for i in 0..n_waters {
                let o_idx = n_protein_atoms + i * 3;
                let h1_idx = o_idx + 1;
                let h2_idx = o_idx + 2;
                let ox = positions[o_idx * 3];
                let oy = positions[o_idx * 3 + 1];
                let oz = positions[o_idx * 3 + 2];
                let h1x = positions[h1_idx * 3];
                let h1y = positions[h1_idx * 3 + 1];
                let h1z = positions[h1_idx * 3 + 2];
                let h2x = positions[h2_idx * 3];
                let h2y = positions[h2_idx * 3 + 1];
                let h2z = positions[h2_idx * 3 + 2];
                let oh1 = ((ox - h1x).powi(2) + (oy - h1y).powi(2) + (oz - h1z).powi(2)).sqrt();
                let oh2 = ((ox - h2x).powi(2) + (oy - h2y).powi(2) + (oz - h2z).powi(2)).sqrt();
                let hh = ((h1x - h2x).powi(2) + (h1y - h2y).powi(2) + (h1z - h2z).powi(2)).sqrt();
                max_oh_err = max_oh_err
                    .max((oh1 - 0.9572).abs())
                    .max((oh2 - 0.9572).abs());
                max_hh_err = max_hh_err.max((hh - 1.5136).abs());
                if i < 3 {
                    println!("    Water {}: O-H1={:.4}Å, O-H2={:.4}Å, H-H={:.4}Å (target: 0.9572, 0.9572, 1.5136)", i, oh1, oh2, hh);
                }
            }
            println!(
                "    Max OH error: {:.6}Å, Max HH error: {:.6}Å across {} waters",
                max_oh_err, max_hh_err, n_waters
            );
        }

        // ═══════════════════════════════════════════════════════════════
        // STEP 3b: Pre-Minimization Distance Check
        // ═══════════════════════════════════════════════════════════════
        println!("\n━━━ Step 3b: Pre-Minimization Distance Check ━━━");
        {
            let positions = hmc.get_positions().expect("Failed to get positions");
            let mut min_dist = f32::MAX;
            let mut min_pair = (0, 0);

            // Check distances between different waters (O atoms only for speed)
            let water_start = n_protein_atoms;
            for i in 0..n_waters {
                let o_i = water_start + i * 3; // O index of water i
                let ox_i = positions[o_i * 3];
                let oy_i = positions[o_i * 3 + 1];
                let oz_i = positions[o_i * 3 + 2];

                for j in (i + 1)..n_waters {
                    let o_j = water_start + j * 3; // O index of water j
                    let ox_j = positions[o_j * 3];
                    let oy_j = positions[o_j * 3 + 1];
                    let oz_j = positions[o_j * 3 + 2];

                    let dist =
                        ((ox_i - ox_j).powi(2) + (oy_i - oy_j).powi(2) + (oz_i - oz_j).powi(2))
                            .sqrt();
                    if dist < min_dist {
                        min_dist = dist;
                        min_pair = (i, j);
                    }
                }
            }
            println!(
                "  Minimum water O-O distance: {:.3} Å (between waters {} and {})",
                min_dist, min_pair.0, min_pair.1
            );

            // Check if any water H atoms are too close to other atoms
            let mut min_h_dist = f32::MAX;
            let mut min_h_pair = (0, 0);
            for i in 0..n_waters.min(100) {
                // Check first 100 waters for speed
                let h1_i = water_start + i * 3 + 1;
                let h2_i = water_start + i * 3 + 2;

                for j in (i + 1)..n_waters.min(100) {
                    let o_j = water_start + j * 3;
                    let h1_j = water_start + j * 3 + 1;
                    let h2_j = water_start + j * 3 + 2;

                    // Check all H-atom distances
                    for &a in &[h1_i, h2_i] {
                        for &b in &[o_j, h1_j, h2_j] {
                            let dist = ((positions[a * 3] - positions[b * 3]).powi(2)
                                + (positions[a * 3 + 1] - positions[b * 3 + 1]).powi(2)
                                + (positions[a * 3 + 2] - positions[b * 3 + 2]).powi(2))
                            .sqrt();
                            if dist < min_h_dist {
                                min_h_dist = dist;
                                min_h_pair = (a, b);
                            }
                        }
                    }
                }
            }
            println!(
                "  Minimum water H-X distance: {:.3} Å (between atoms {} and {})",
                min_h_dist, min_h_pair.0, min_h_pair.1
            );
        }

        // ═══════════════════════════════════════════════════════════════
        // STEP 4: Energy Minimization
        // ═══════════════════════════════════════════════════════════════
        println!("\n━━━ Step 4: Energy Minimization ━━━");

        let min_start = Instant::now();
        // Aggressive minimization with many steps and small step size
        // Goal: reduce average force to <5 kcal/(mol·Å)
        let final_energy = hmc.minimize(10000, 0.002).expect("Minimization failed");
        let min_time = min_start.elapsed();

        println!(
            "  Minimization complete in {:.2} seconds",
            min_time.as_secs_f64()
        );
        println!("  Final energy: {:.2} kcal/mol", final_energy);

        // Force diagnostics after minimization
        println!("\n━━━ Step 4b: Force Diagnostics ━━━");
        let (max_f, avg_f, top_forces) = hmc
            .get_force_diagnostics()
            .expect("Failed to get force diagnostics");
        println!("  Max force: {:.2} kcal/(mol·Å)", max_f);
        println!("  Avg force: {:.2} kcal/(mol·Å)", avg_f);
        println!("  Top 10 high-force atoms:");
        for (idx, force) in &top_forces {
            // Identify atom type
            let atom_type = if *idx < n_protein_atoms {
                // Protein atom
                let (res_id, res_name, atom_name) = &atom_info[*idx];
                format!("Protein {} {} {}", res_id, res_name, atom_name)
            } else {
                let water_start = n_protein_atoms;
                let ion_start = water_start + n_waters * 3;
                if *idx < ion_start {
                    // Water atom
                    let water_idx = (*idx - water_start) / 3;
                    let atom_in_water = (*idx - water_start) % 3;
                    let atom_name = match atom_in_water {
                        0 => "O",
                        1 => "H1",
                        2 => "H2",
                        _ => "?",
                    };
                    format!("Water {} {}", water_idx, atom_name)
                } else {
                    // Ion
                    let ion_idx = *idx - ion_start;
                    if ion_idx < n_na {
                        format!("Na+ #{}", ion_idx)
                    } else {
                        format!("Cl- #{}", ion_idx - n_na)
                    }
                }
            };
            println!(
                "    Atom {:5}: |F|={:7.1} kcal/(mol·Å)  [{}]",
                idx, force, atom_type
            );
        }

        // ═══════════════════════════════════════════════════════════════
        // STEP 4c: Minimum Distance Diagnostic
        // ═══════════════════════════════════════════════════════════════
        println!("\n━━━ Step 4c: Close Contact Analysis ━━━");

        // Get current positions
        let positions = hmc.get_positions().expect("Failed to get positions");

        // Find closest contacts involving the high-force atoms
        for (atom_idx, force) in top_forces.iter().take(3) {
            let px = positions[atom_idx * 3];
            let py = positions[atom_idx * 3 + 1];
            let pz = positions[atom_idx * 3 + 2];

            // Find closest atom to this one
            let mut min_dist = f32::MAX;
            let mut closest_idx = 0;
            for j in 0..n_total {
                if j == *atom_idx {
                    continue;
                }
                let qx = positions[j * 3];
                let qy = positions[j * 3 + 1];
                let qz = positions[j * 3 + 2];
                let dist = ((px - qx).powi(2) + (py - qy).powi(2) + (pz - qz).powi(2)).sqrt();
                if dist < min_dist {
                    min_dist = dist;
                    closest_idx = j;
                }
            }

            // Identify atoms
            let atom_type_a = if *atom_idx < n_protein_atoms {
                format!("Protein {}", atom_idx)
            } else {
                let water_start = n_protein_atoms;
                let ion_start = water_start + n_waters * 3;
                if *atom_idx < ion_start {
                    let water_idx = (*atom_idx - water_start) / 3;
                    let offset = (*atom_idx - water_start) % 3;
                    let name = ["O", "H1", "H2"][offset];
                    format!("Water {} {}", water_idx, name)
                } else {
                    format!("Ion {}", *atom_idx - ion_start)
                }
            };

            let atom_type_b = if closest_idx < n_protein_atoms {
                format!("Protein {}", closest_idx)
            } else {
                let water_start = n_protein_atoms;
                let ion_start = water_start + n_waters * 3;
                if closest_idx < ion_start {
                    let water_idx = (closest_idx - water_start) / 3;
                    let offset = (closest_idx - water_start) % 3;
                    let name = ["O", "H1", "H2"][offset];
                    format!("Water {} {}", water_idx, name)
                } else {
                    format!("Ion {}", closest_idx - ion_start)
                }
            };

            println!(
                "  High-force atom {} ({}) |F|={:.1}:",
                atom_idx, atom_type_a, force
            );
            println!(
                "    Closest: {} ({}) at {:.3} Å",
                closest_idx, atom_type_b, min_dist
            );
        }

        // ═══════════════════════════════════════════════════════════════
        // STEP 5: Equilibration using run_fused() (stable temperatures)
        // ═══════════════════════════════════════════════════════════════
        println!("\n━━━ Step 5: Equilibration (20 ps) ━━━");

        let dt = 1.0_f32; // 1 fs timestep (reduced for stability with high forces)
        let temperature = 310.0_f32; // 37°C
                                     // DISABLE Langevin thermostat for explicit solvent - use velocity rescaling only
                                     // The DOF mismatch between thermostat (3N) and SETTLE constraints (N_water * 3)
                                     // causes energy pumping. Velocity rescaling in run_fused handles temperature.
        let gamma = 0.0_f32; // NVE dynamics with velocity rescaling thermostat

        hmc.initialize_velocities(temperature)
            .expect("Failed to initialize velocities");

        // Debug: Check velocities after initialization
        let velocities_after_init = hmc.get_velocities().expect("Failed to get velocities");
        let vel_sum: f32 = velocities_after_init.iter().map(|v| v.abs()).sum();
        let vel_max: f32 = velocities_after_init
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);
        println!(
            "  DEBUG: After init_velocities - sum(|v|)={:.2}, max(|v|)={:.4}",
            vel_sum, vel_max
        );

        // Debug: Build neighbor lists explicitly and check
        hmc.build_neighbor_lists()
            .expect("Failed to build neighbor lists");
        let max_force = hmc.get_max_force().expect("Failed to get max force");
        println!(
            "  DEBUG: After build_neighbor_lists - max_force={:.4}",
            max_force
        );

        // CRITICAL: Add position restraints on protein heavy atoms during equilibration
        // This prevents the protein from exploding while water equilibrates
        // k = 10 kcal/(mol·Å²) is standard for initial equilibration
        let protein_heavy_atoms: Vec<usize> = (0..n_protein_atoms)
            .filter(|&i| {
                let (_, _, atom_name) = &atom_info[i];
                !atom_name.starts_with('H') // Heavy atoms only
            })
            .collect();

        println!(
            "  Setting position restraints on {} protein heavy atoms (k=100)",
            protein_heavy_atoms.len()
        );
        hmc.set_position_restraints(&protein_heavy_atoms, 100.0) // Strong restraints (100 kcal/(mol·Å²))
            .expect("Failed to set position restraints");

        // Use run_verlet for equilibration - it has proper position restraint support
        // (restraint forces are added BEFORE integration, not after like in run_fused)
        let equil_steps = 50000; // 50 ps with dt=1fs for better thermalization
        let equil_start = Instant::now();

        let result = hmc
            .run_verlet(equil_steps, dt, temperature, gamma)
            .expect("Equilibration failed");

        let equil_time = equil_start.elapsed();
        println!(
            "  Equilibration complete in {:.2} seconds",
            equil_time.as_secs_f64()
        );
        println!("  Final temperature: {:.1} K", result.avg_temperature);

        // ═══════════════════════════════════════════════════════════════
        // STEP 6: Production MD with Trajectory Saving
        // ═══════════════════════════════════════════════════════════════
        println!("\n━━━ Step 6: Production MD (100 ps) ━━━");

        // Disable position restraints for production
        // Now that the system is equilibrated, let the protein move freely
        hmc.disable_position_restraints();
        println!("  Position restraints disabled for production");

        let production_ps = 100.0;
        let save_interval_ps = 1.0;
        let steps_per_chunk = (save_interval_ps * 1000.0 / dt as f64) as usize;
        let n_chunks = (production_ps / save_interval_ps) as usize;

        println!(
            "  Saving every {} ps ({} frames total)",
            save_interval_ps, n_chunks
        );

        let reference_positions = hmc.get_positions().expect("Failed to get positions");
        let reference_ca = extract_ca_positions(&reference_positions, &ca_indices);

        let mut trajectory_ca: Vec<Vec<[f32; 3]>> = Vec::with_capacity(n_chunks);
        trajectory_ca.push(reference_ca.clone());

        let prod_start = Instant::now();

        for chunk in 0..n_chunks {
            let result = hmc
                .run_fused(steps_per_chunk, dt, temperature, gamma, false)
                .expect("Production step failed");

            let positions = hmc.get_positions().expect("Failed to get positions");
            let ca_positions = extract_ca_positions(&positions, &ca_indices);
            trajectory_ca.push(ca_positions);

            // Progress report every 10%
            if (chunk + 1) % (n_chunks / 10).max(1) == 0 {
                let elapsed = prod_start.elapsed().as_secs_f64();
                let progress = (chunk + 1) as f64 / n_chunks as f64 * 100.0;
                let steps_done = (chunk + 1) * steps_per_chunk;
                let steps_per_sec = steps_done as f64 / elapsed;
                let ns_per_day = steps_per_sec * dt as f64 * 86400.0 / 1e6;

                println!(
                    "  [{:5.1}%] {:.0}/{:.0} ps | T={:.1}K | {:.0} ns/day",
                    progress,
                    (chunk + 1) as f64 * save_interval_ps,
                    production_ps,
                    result.avg_temperature,
                    ns_per_day
                );
            }
        }

        let prod_time = prod_start.elapsed();
        let total_steps = n_chunks * steps_per_chunk;
        let steps_per_sec = total_steps as f64 / prod_time.as_secs_f64();
        let ns_per_day = steps_per_sec * dt as f64 * 86400.0 / 1e6;
        let time_per_step_us = prod_time.as_secs_f64() * 1e6 / total_steps as f64;

        println!("\n  Production complete!");
        println!("  Wall time: {:.2} seconds", prod_time.as_secs_f64());
        println!("  Performance: {:.0} ns/day", ns_per_day);
        println!("  Time/step: {:.2} µs", time_per_step_us);

        // ═══════════════════════════════════════════════════════════════
        // STEP 7: RMSD Analysis
        // ═══════════════════════════════════════════════════════════════
        println!("\n━━━ Step 7: RMSD Analysis ━━━");

        // Debug: Check if positions actually changed
        println!("\n  DEBUG: First 3 Cα positions comparison:");
        println!(
            "         Frame 0 (ref)    vs    Frame {}  (last)",
            trajectory_ca.len() - 1
        );
        for i in 0..3.min(reference_ca.len()) {
            let first = &trajectory_ca[0][i];
            let last = &trajectory_ca[trajectory_ca.len() - 1][i];
            let diff = ((last[0] - first[0]).powi(2)
                + (last[1] - first[1]).powi(2)
                + (last[2] - first[2]).powi(2))
            .sqrt();
            println!(
                "    Cα[{}]: [{:.3}, {:.3}, {:.3}]  vs  [{:.3}, {:.3}, {:.3}]  diff={:.4}",
                i, first[0], first[1], first[2], last[0], last[1], last[2], diff
            );
        }

        let rmsd_values: Vec<f32> = trajectory_ca
            .iter()
            .map(|frame| compute_ca_rmsd(&reference_ca, frame))
            .collect();

        let rmsd_mean = rmsd_values.iter().sum::<f32>() / rmsd_values.len() as f32;
        let rmsd_var: f32 = rmsd_values
            .iter()
            .map(|r| (r - rmsd_mean).powi(2))
            .sum::<f32>()
            / rmsd_values.len() as f32;
        let rmsd_std = rmsd_var.sqrt();
        let rmsd_max = rmsd_values.iter().cloned().fold(0.0f32, f32::max);

        println!("  Mean RMSD: {:.3} Å", rmsd_mean);
        println!("  Std RMSD:  {:.3} Å", rmsd_std);
        println!("  Max RMSD:  {:.3} Å", rmsd_max);

        // ═══════════════════════════════════════════════════════════════
        // STEP 8: RMSF Analysis
        // ═══════════════════════════════════════════════════════════════
        println!("\n━━━ Step 8: RMSF Analysis ━━━");

        let n_ca = ca_indices.len();
        let n_frames = trajectory_ca.len() as f32;

        // Calculate mean position for each Cα
        let mut mean_positions: Vec<[f32; 3]> = vec![[0.0; 3]; n_ca];
        for frame in &trajectory_ca {
            for (i, pos) in mean_positions.iter_mut().enumerate() {
                pos[0] += frame[i][0];
                pos[1] += frame[i][1];
                pos[2] += frame[i][2];
            }
        }
        for pos in &mut mean_positions {
            pos[0] /= n_frames;
            pos[1] /= n_frames;
            pos[2] /= n_frames;
        }

        // Calculate RMSF
        let mut rmsf_values: Vec<f32> = vec![0.0; n_ca];
        for frame in &trajectory_ca {
            for (i, mean_pos) in mean_positions.iter().enumerate() {
                let dx = frame[i][0] - mean_pos[0];
                let dy = frame[i][1] - mean_pos[1];
                let dz = frame[i][2] - mean_pos[2];
                rmsf_values[i] += dx * dx + dy * dy + dz * dz;
            }
        }
        for rmsf in &mut rmsf_values {
            *rmsf = (*rmsf / n_frames).sqrt();
        }

        let rmsf_mean = rmsf_values.iter().sum::<f32>() / n_ca as f32;
        let rmsf_var: f32 = rmsf_values
            .iter()
            .map(|r| (r - rmsf_mean).powi(2))
            .sum::<f32>()
            / n_ca as f32;
        let rmsf_std = rmsf_var.sqrt();
        let rmsf_max = rmsf_values.iter().cloned().fold(0.0f32, f32::max);

        println!("  Mean RMSF: {:.3} Å", rmsf_mean);
        println!("  Std RMSF:  {:.3} Å", rmsf_std);
        println!("  Max RMSF:  {:.3} Å", rmsf_max);

        // Calculate z-scores
        let rmsf_zscores: Vec<f32> = rmsf_values
            .iter()
            .map(|r| {
                if rmsf_std > 0.0 {
                    (r - rmsf_mean) / rmsf_std
                } else {
                    0.0
                }
            })
            .collect();

        let high_flex_count = rmsf_zscores.iter().filter(|z| **z > 1.5).count();
        println!("  High-flexibility residues (z>1.5): {}", high_flex_count);

        // ═══════════════════════════════════════════════════════════════
        // STEP 9: Escape Mutation Site Analysis
        // ═══════════════════════════════════════════════════════════════
        println!("\n━━━ Step 9: Escape Mutation Site Analysis ━━━");

        println!();
        println!("  ┌─────────┬───────────┬────────────────────┬────────────┬────────────┐");
        println!("  │ Residue │ Mutation  │ Variant            │ Implicit z │ Explicit z │");
        println!("  ├─────────┼───────────┼────────────────────┼────────────┼────────────┤");

        let mut detected_implicit = 0;
        let mut detected_explicit = 0;

        for &(resid, mutation, variant, implicit_z) in ESCAPE_SITES {
            // Find the Cα index for this residue
            let ca_idx = atom_info
                .iter()
                .enumerate()
                .find(|(i, (rid, _, name))| *rid == resid && name == "CA")
                .map(|(i, _)| ca_indices.iter().position(|&idx| idx == i))
                .flatten();

            let explicit_z = if let Some(idx) = ca_idx {
                rmsf_zscores.get(idx).copied().unwrap_or(0.0)
            } else {
                0.0
            };

            let implicit_detected = implicit_z > 1.0;
            let explicit_detected = explicit_z > 1.0;

            if implicit_detected {
                detected_implicit += 1;
            }
            if explicit_detected {
                detected_explicit += 1;
            }

            println!(
                "  │ {:>7} │ {:>9} │ {:>18} │ {:>10.2} │ {:>10.2} │",
                resid, mutation, variant, implicit_z, explicit_z
            );
        }

        println!("  └─────────┴───────────┴────────────────────┴────────────┴────────────┘");
        println!();

        let total_sites = ESCAPE_SITES.len();
        let implicit_rate = detected_implicit as f64 / total_sites as f64 * 100.0;
        let explicit_rate = detected_explicit as f64 / total_sites as f64 * 100.0;

        println!("  Detection Rate (z > 1.0):");
        println!(
            "    Implicit (baseline):  {}/{} ({:.1}%)",
            detected_implicit, total_sites, implicit_rate
        );
        println!(
            "    Explicit (this run):  {}/{} ({:.1}%)",
            detected_explicit, total_sites, explicit_rate
        );

        // ═══════════════════════════════════════════════════════════════
        // FINAL REPORT
        // ═══════════════════════════════════════════════════════════════
        println!("\n");
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║                    VALIDATION RESULTS                        ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║  SYSTEM                                                      ║");
        println!(
            "║    Protein: 6M0J Chain E ({} atoms, {} residues)         ║",
            n_protein_atoms, n_residues
        );
        println!(
            "║    Waters:  {} TIP3P ({} atoms)                         ║",
            n_waters,
            n_waters * 3
        );
        println!(
            "║    Ions:    {} Na+, {} Cl-                                   ║",
            n_na, n_cl
        );
        println!(
            "║    Total:   {} atoms                                     ║",
            n_total
        );
        println!(
            "║    Box:     {:.1} × {:.1} × {:.1} Å                            ║",
            box_dims[0], box_dims[1], box_dims[2]
        );
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║  PERFORMANCE (Phase 7-8 Fused Kernels)                       ║");
        println!(
            "║    Wall time:     {:6.2} seconds                            ║",
            prod_time.as_secs_f64()
        );
        println!(
            "║    Performance:   {:6.0} ns/day                              ║",
            ns_per_day
        );
        println!(
            "║    Time/step:     {:6.2} µs                                  ║",
            time_per_step_us
        );
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║  STRUCTURE METRICS                                           ║");
        println!(
            "║    RMSD mean:     {:6.3} Å                                   ║",
            rmsd_mean
        );
        println!(
            "║    RMSD std:      {:6.3} Å                                   ║",
            rmsd_std
        );
        println!(
            "║    RMSF mean:     {:6.3} Å                                   ║",
            rmsf_mean
        );
        println!(
            "║    RMSF max:      {:6.3} Å                                   ║",
            rmsf_max
        );
        println!("╠══════════════════════════════════════════════════════════════╣");

        // Overall verdict
        let rmsd_ok = rmsd_mean < 3.0;
        let performance_ok = ns_per_day > 1000.0;

        println!("║  VALIDATION CHECKLIST                                        ║");
        println!(
            "║    [{}] RMSD < 3.0 Å (stable fold)                           ║",
            if rmsd_ok { "✓" } else { "✗" }
        );
        println!(
            "║    [{}] Performance > 1,000 ns/day                           ║",
            if performance_ok { "✓" } else { "✗" }
        );
        println!("╠══════════════════════════════════════════════════════════════╣");

        if rmsd_ok && performance_ok {
            println!("║  VERDICT: ✓ EXPLICIT SOLVENT VALIDATION PASSED              ║");
        } else {
            println!("║  VERDICT: ⚠ PARTIAL PASS - SEE CHECKLIST ABOVE              ║");
        }
        println!("╚══════════════════════════════════════════════════════════════╝");

        // Assert criteria
        assert!(rmsd_mean < 3.0, "RMSD too high: {:.3} Å", rmsd_mean);
        assert!(
            ns_per_day > 500.0,
            "Performance too low: {:.0} ns/day",
            ns_per_day
        );
    }

    /// Build H-bond constraint clusters from protein atoms
    fn build_protein_h_clusters(
        atom_info: &[(usize, String, String)],
        atom_types: &[AmberAtomType],
    ) -> Vec<HConstraintCluster> {
        use std::collections::HashMap;

        // Build map of atom index to info
        let mut heavy_to_hydrogens: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();

        for (i, (_res_id, res_name, atom_name)) in atom_info.iter().enumerate() {
            // Skip if this isn't a hydrogen
            if !atom_name.starts_with('H') {
                continue;
            }

            // Find the heavy atom this H is bonded to
            // Use naming convention: H atoms are named H, HA, HB, HG, etc.
            // Their parent is the atom without the 'H' prefix
            let parent_name = if atom_name.len() > 1 {
                // e.g., HA -> CA, HB2 -> CB, HG1 -> OG1/CG1
                let suffix = &atom_name[1..];
                if suffix.starts_with(|c: char| c.is_alphabetic()) {
                    format!("{}{}", &suffix[0..1], suffix.get(1..).unwrap_or(""))
                } else {
                    continue; // Can't determine parent
                }
            } else {
                "N".to_string() // Backbone H is bonded to N
            };

            // Find the parent atom in the same residue
            let res_id = atom_info[i].0;
            let parent_idx = atom_info
                .iter()
                .position(|(rid, _, name)| *rid == res_id && name == &parent_name);

            if let Some(parent) = parent_idx {
                // Determine bond length based on parent type
                let bond_length = match atom_types.get(parent) {
                    Some(AmberAtomType::N) => 1.01,                     // N-H
                    Some(AmberAtomType::O | AmberAtomType::OH) => 0.96, // O-H
                    Some(AmberAtomType::S | AmberAtomType::SH) => 1.34, // S-H
                    _ => 1.09,                                          // C-H (default)
                };

                heavy_to_hydrogens
                    .entry(parent)
                    .or_default()
                    .push((i, bond_length));
            }
        }

        // Build clusters
        let mut clusters = Vec::new();

        for (heavy_idx, hydrogens) in heavy_to_hydrogens {
            let mass_heavy = get_atom_mass(atom_types[heavy_idx]);
            let mass_h = 1.008;

            let is_nitrogen = matches!(
                atom_types.get(heavy_idx),
                Some(
                    AmberAtomType::N
                        | AmberAtomType::N2
                        | AmberAtomType::N3
                        | AmberAtomType::NA
                        | AmberAtomType::NB
                )
            );

            match hydrogens.len() {
                1 => {
                    let (h, d) = hydrogens[0];
                    clusters.push(HConstraintCluster::single_h(
                        heavy_idx, h, d, mass_heavy, mass_h,
                    ));
                }
                2 => {
                    let (h1, d1) = hydrogens[0];
                    let (h2, d2) = hydrogens[1];
                    clusters.push(HConstraintCluster::two_h(
                        heavy_idx,
                        h1,
                        h2,
                        d1,
                        d2,
                        mass_heavy,
                        mass_h,
                        is_nitrogen,
                    ));
                }
                3 => {
                    let (h1, d1) = hydrogens[0];
                    let (h2, d2) = hydrogens[1];
                    let (h3, d3) = hydrogens[2];
                    clusters.push(HConstraintCluster::three_h(
                        heavy_idx,
                        h1,
                        h2,
                        h3,
                        d1,
                        d2,
                        d3,
                        mass_heavy,
                        mass_h,
                        is_nitrogen,
                    ));
                }
                _ => {} // Skip unusual cases
            }
        }

        clusters
    }

    /// Quick smoke test - just verify the setup works
    #[test]
    fn test_6m0j_setup_only() {
        println!("\n=== 6M0J Setup Test ===\n");

        let pdb_path = find_pdb_file();
        println!("Found PDB: {:?}", pdb_path);

        let content = fs::read_to_string(&pdb_path).expect("Read failed");
        let (positions, atom_types, charges, atom_info) = parse_pdb_for_solvation(&content);

        println!("Parsed {} atoms", atom_types.len());
        assert!(atom_types.len() > 1000, "Expected >1000 atoms in 6M0J");

        let ca_indices = extract_ca_indices(&content);
        println!("Found {} Cα atoms", ca_indices.len());
        assert!(ca_indices.len() > 100, "Expected >100 residues");

        // Test solvation box creation
        let config = SolvationConfig {
            padding: 10.0,
            max_box_dimension: 60.0,
            ..Default::default()
        };

        let mut solvbox = SolvationBox::from_protein(&positions, &atom_types, &charges, &config)
            .expect("Solvation box creation failed");

        let n_waters = solvbox.add_waters(&config);
        println!("Added {} water molecules", n_waters);
        assert!(n_waters > 500, "Expected >500 waters");

        println!("\n✓ Setup test passed");
    }
}
