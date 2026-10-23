//! # RT Target Identification
//! **Stage:** `[STAGE-1-RT-TARGETS]`
//!
//! Identifies spatial probe targets for RT cores to scan for cryptic pockets.
//!
//! ## Overview
//!
//! RT cores use ray tracing to detect:
//! - **Solvation disruption**: Water displacement around protein (earliest signal)
//! - **Geometric voids**: Pocket formation via spatial probes
//! - **Aromatic LIF**: Laser-induced fluorescence from aromatic rings
//!
//! This module identifies three categories of RT targets:
//! 1. **Protein heavy atoms**: Ray trace origins for geometric probes
//! 2. **Water oxygen atoms**: Solvation tracking (explicit mode only)
//! 3. **Aromatic ring centers**: LIF probe points
//!
//! ## Performance
//!
//! Target identification is a one-time operation during Stage 1 (PREP).
//! - Protein atoms: O(n_atoms) - filter by element
//! - Water atoms: O(1) - already identified during solvation
//! - Aromatic centers: O(n_aromatics) - pre-computed ring centroids

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::config::SolventMode;
use crate::input::PrismPrepTopology;

/// 3D vector for positions
pub type Vec3 = [f32; 3];

/// RT probe targets for spatial sensing
///
/// These targets are used by RT cores to:
/// - Launch rays from protein surface atoms
/// - Track water oxygen displacements
/// - Probe aromatic ring centers for LIF signals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RtTargets {
    /// Protein heavy atom indices (no hydrogens)
    ///
    /// Used as ray origins for geometric void detection.
    /// Typical count: 5,000-20,000 atoms for proteins.
    pub protein_atoms: Vec<usize>,

    /// Water oxygen atom indices (explicit mode only)
    ///
    /// Used for solvation disruption detection.
    /// Typical count: 20,000-100,000 waters for explicit mode.
    pub water_atoms: Option<Vec<usize>>,

    /// Aromatic ring center positions
    ///
    /// Used for laser-induced fluorescence (LIF) probing.
    /// Typical count: 10-50 aromatic residues (Phe, Tyr, Trp, His).
    pub aromatic_centers: Vec<Vec3>,

    /// Total target count for logging
    pub total_targets: usize,
}

impl RtTargets {
    /// Create empty RT targets
    pub fn empty() -> Self {
        Self {
            protein_atoms: Vec::new(),
            water_atoms: None,
            aromatic_centers: Vec::new(),
            total_targets: 0,
        }
    }

    /// Compute total target count
    pub fn compute_total(&mut self) {
        let protein_count = self.protein_atoms.len();
        let water_count = self.water_atoms.as_ref().map_or(0, |w| w.len());
        let aromatic_count = self.aromatic_centers.len();
        self.total_targets = protein_count + water_count + aromatic_count;
    }

    /// Get summary statistics
    pub fn summary(&self) -> String {
        let water_info = match &self.water_atoms {
            Some(w) => format!("{} water O atoms", w.len()),
            None => "no waters (implicit)".to_string(),
        };
        format!(
            "RT Targets: {} protein heavy atoms, {}, {} aromatic centers (total: {})",
            self.protein_atoms.len(),
            water_info,
            self.aromatic_centers.len(),
            self.total_targets
        )
    }
}

/// Identify RT probe targets from topology
///
/// # Arguments
/// * `topology` - PRISM-PREP topology with atom information
/// * `solvent_mode` - Solvent mode (determines if water targets included)
///
/// # Returns
/// * `RtTargets` - Identified probe targets
///
/// # Algorithm
///
/// 1. **Protein heavy atoms**: Filter by element (exclude H)
/// 2. **Water oxygens**: Use topology.water_oxygens if explicit mode
/// 3. **Aromatic centers**: Compute centroids of aromatic rings
///
/// # Example
/// ```ignore
/// let targets = identify_rt_targets(&topology, &solvent_mode)?;
/// log::info!("{}", targets.summary());
/// ```
pub fn identify_rt_targets(
    topology: &PrismPrepTopology,
    solvent_mode: &SolventMode,
) -> Result<RtTargets> {
    log::info!("Identifying RT probe targets for {:?} mode", solvent_mode);

    // Step 1: Identify protein heavy atoms (no hydrogens)
    let protein_atoms =
        identify_heavy_atoms(topology).context("Failed to identify protein heavy atoms")?;

    log::info!(
        "Found {} protein heavy atoms (RT ray origins)",
        protein_atoms.len()
    );

    // Step 2: Identify water oxygens (explicit mode only)
    let water_atoms = if solvent_mode.requires_water() {
        let waters = topology.water_oxygens.clone();
        log::info!(
            "Found {} water oxygen atoms (RT solvation probes)",
            waters.len()
        );
        Some(waters)
    } else {
        log::info!("Implicit mode: no water atoms for RT probing");
        None
    };

    // Step 3: Compute aromatic ring centers for LIF
    let aromatic_centers =
        compute_aromatic_centers(topology).context("Failed to compute aromatic centers")?;

    log::info!(
        "Found {} aromatic ring centers (RT LIF probes)",
        aromatic_centers.len()
    );

    // Assemble targets
    let mut targets = RtTargets {
        protein_atoms,
        water_atoms,
        aromatic_centers,
        total_targets: 0,
    };
    targets.compute_total();

    log::info!("{}", targets.summary());

    Ok(targets)
}

/// Identify heavy atoms (exclude hydrogens) from topology
///
/// # Arguments
/// * `topology` - PRISM-PREP topology
///
/// # Returns
/// * `Vec<usize>` - Indices of heavy atoms (element != "H")
///
/// # Notes
/// - Hydrogens are excluded because they don't participate in RT probing
/// - Typical reduction: ~50% atom count (proteins are ~50% hydrogen)
fn identify_heavy_atoms(topology: &PrismPrepTopology) -> Result<Vec<usize>> {
    let mut heavy_atoms = Vec::with_capacity(topology.n_atoms / 2);

    for (idx, element) in topology.elements.iter().enumerate() {
        // Skip hydrogens (case-insensitive)
        if element.eq_ignore_ascii_case("H") {
            continue;
        }

        heavy_atoms.push(idx);
    }

    if heavy_atoms.is_empty() {
        anyhow::bail!("No heavy atoms found in topology (all hydrogens?)");
    }

    Ok(heavy_atoms)
}

/// Compute aromatic ring center positions for LIF probing
///
/// # Arguments
/// * `topology` - PRISM-PREP topology with atom positions
///
/// # Returns
/// * `Vec<Vec3>` - Ring center positions [x, y, z]
///
/// # Algorithm
///
/// For each aromatic residue (PHE, TYR, TRP, HIS):
/// 1. Find aromatic ring atom indices (from topology metadata)
/// 2. Compute centroid = average position of ring atoms
/// 3. Return centroid as LIF probe point
///
/// # Notes
/// - Aromatic residues are pre-identified by PRISM-PREP
/// - Ring centers are used for laser-induced fluorescence (LIF) detection
/// - Typical aromatic content: 10-15% of protein residues
fn compute_aromatic_centers(topology: &PrismPrepTopology) -> Result<Vec<Vec3>> {
    let mut centers = Vec::new();

    // Aromatic residue names to look for (including histidine protonation states)
    const AROMATIC_RESIDUES: &[&str] = &["PHE", "TYR", "TRP", "HIS", "HID", "HIE", "HIP"];

    // Track residues we've already processed
    let mut processed_residues = std::collections::HashSet::new();

    for (atom_idx, residue_id) in topology.residue_ids.iter().enumerate() {
        // Skip if we've already processed this residue
        if processed_residues.contains(residue_id) {
            continue;
        }

        // Check if this residue is aromatic. `residue_names` is ATOM-indexed per the
        // PrismPrepTopology contract (validate() enforces residue_names.len() == n_atoms;
        // see input.rs classify_atoms()), so index by the ATOM index — not residue_id.
        // (Indexing by residue_id mis-read a residue-indexed name and over-counted centers.)
        if atom_idx >= topology.residue_names.len() {
            continue; // defensive: contract guarantees atom_idx < n_atoms == residue_names.len()
        }

        let residue_name = &topology.residue_names[atom_idx];
        if !AROMATIC_RESIDUES.contains(&residue_name.as_str()) {
            continue;
        }

        // Mark as processed
        processed_residues.insert(*residue_id);

        // Collect all heavy atoms in this residue
        let mut ring_atoms = Vec::new();
        for (idx, rid) in topology.residue_ids.iter().enumerate() {
            if *rid == *residue_id {
                // Only include heavy atoms (no hydrogens)
                if !topology.elements[idx].eq_ignore_ascii_case("H") {
                    ring_atoms.push(idx);
                }
            }
        }

        if ring_atoms.is_empty() {
            continue;
        }

        // Compute centroid of ring atoms
        let mut centroid = [0.0f32, 0.0f32, 0.0f32];
        for &atom_idx in &ring_atoms {
            let base = atom_idx * 3;
            if base + 2 < topology.positions.len() {
                centroid[0] += topology.positions[base];
                centroid[1] += topology.positions[base + 1];
                centroid[2] += topology.positions[base + 2];
            }
        }

        let n = ring_atoms.len() as f32;
        centroid[0] /= n;
        centroid[1] /= n;
        centroid[2] /= n;

        centers.push(centroid);

        log::debug!(
            "Aromatic {} residue {} center: [{:.2}, {:.2}, {:.2}] ({} atoms)",
            residue_name,
            residue_id,
            centroid[0],
            centroid[1],
            centroid[2],
            ring_atoms.len()
        );
    }

    Ok(centers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::{
        python_canonical_sha256, AnalysisResidueContract, AnalysisResidueRecord,
        MoleculeIdentityRow, MoleculeIdentityTable,
    };
    use std::collections::BTreeMap;

    fn create_test_topology() -> PrismPrepTopology {
        // Create a minimal topology with some heavy atoms and hydrogens
        let mut topology = PrismPrepTopology {
            aromatic_targets: Vec::new(),
            source_pdb: String::from("test.pdb"),
            n_atoms: 10,
            n_residues: 2,
            total_physical_residue_count: Some(2),
            n_analysis_residues: Some(2),
            n_chains: 1,
            // Positions for 10 atoms (30 floats)
            positions: vec![
                // Residue 0 (ALA): N, CA, C, O, CB, H1, H2, H3, HA, HB
                0.0, 0.0, 0.0, // 0: N
                1.5, 0.0, 0.0, // 1: CA
                2.0, 1.5, 0.0, // 2: C
                3.0, 1.5, 0.0, // 3: O
                1.5, 0.0, 1.5, // 4: CB
                // Residue 1 (PHE): CG, CD1, CD2, CE1, CE2
                5.0, 5.0, 5.0, // 5: CG
                6.0, 5.0, 5.0, // 6: CD1
                5.0, 6.0, 5.0, // 7: CD2
                7.0, 5.5, 5.0, // 8: CE1
                6.0, 6.5, 5.0, // 9: CE2
            ],
            elements: vec![
                "N".into(),
                "C".into(),
                "C".into(),
                "O".into(),
                "C".into(),
                "C".into(),
                "C".into(),
                "C".into(),
                "C".into(),
                "C".into(),
            ],
            atom_names: vec![
                "N".into(),
                "CA".into(),
                "C".into(),
                "O".into(),
                "CB".into(),
                "CG".into(),
                "CD1".into(),
                "CD2".into(),
                "CE1".into(),
                "CE2".into(),
            ],
            // residue_names is ATOM-indexed (one entry per atom) per the PrismPrepTopology
            // contract (validate(): residue_names.len() == n_atoms). Atoms 0-4 = ALA (res 0),
            // atoms 5-9 = PHE (res 1). (Was residue-indexed ["ALA","PHE"], which only worked with
            // the old residue-indexed compute_aromatic_centers bug.)
            residue_names: vec![
                "ALA".into(),
                "ALA".into(),
                "ALA".into(),
                "ALA".into(),
                "ALA".into(),
                "PHE".into(),
                "PHE".into(),
                "PHE".into(),
                "PHE".into(),
                "PHE".into(),
            ],
            residue_ids: vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            atom_physical_residue_ids: Some(vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
            atom_analysis_residue_ids: Some(vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
            analysis_residue_contract: None,
            chain_ids: vec!["A".into(); 10],
            charges: vec![0.0; 10],
            masses: vec![14.0, 12.0, 12.0, 16.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0],
            ca_indices: vec![1],
            bonds: Vec::new(),
            angles: Vec::new(),
            dihedrals: Vec::new(),
            lj_params: Vec::new(),
            exclusions: Vec::new(),
            nonbonded_exceptions: Vec::new(),
            nonbonded_exception_contract: None,
            generated_prmtop_sha256: None,
            h_clusters: Vec::new(),
            hmr_applied: Some(false),
            recommended_timestep_fs: None,
            water_oxygens: vec![],
            residues: Vec::new(),
            dimer_dyad: None,
            lambda_fragment_system: None,
            harmonic_parameter_contract: None,
            schema_kind: None,
            schema_version: None,
            periodic_cell: None,
            water_model_contract: None,
            molecule_identity_table: None,
            topology_provenance: None,
            npt_frozen_cell_contract: None,
        };
        let lineage = serde_json::json!({
            "atom_indices": (0..10).collect::<Vec<_>>(),
            "residue_indices": [0, 1],
            "residue_names": ["ALA", "PHE"],
            "chain_ids": ["A"],
            "kind": "BIOMOLECULE",
            "identity": "A",
        });
        let molecule = MoleculeIdentityRow {
            molecule_id: 0,
            atom_indices: (0..10).collect(),
            residue_indices: vec![0, 1],
            residue_names: vec!["ALA".into(), "PHE".into()],
            chain_ids: vec!["A".into()],
            kind: "BIOMOLECULE".into(),
            identity: "A".into(),
            component_sha256: python_canonical_sha256(&lineage).unwrap(),
        };
        let mut table = MoleculeIdentityTable {
            schema_kind: "prismos_molecule_identity_table".into(),
            schema_version: 1,
            n_molecules: 1,
            n_atoms: 10,
            molecules: vec![molecule],
            table_sha256: String::new(),
        };
        let mut table_value = serde_json::to_value(&table).unwrap();
        table_value.as_object_mut().unwrap().remove("table_sha256");
        table.table_sha256 = python_canonical_sha256(&table_value).unwrap();
        let mut role_counts = BTreeMap::new();
        role_counts.insert("RECEPTOR".into(), 2);
        let analysis_ids = topology.atom_analysis_residue_ids.as_ref().unwrap();
        let mut contract = AnalysisResidueContract {
            schema_kind: "prismos_analysis_residue_mapping".into(),
            schema_version: 1,
            analysis_authority: "KCC_GHOST_PROTEIN_RESIDUE_AXIS".into(),
            excluded_atom_sentinel: -1,
            n_atoms: 10,
            total_physical_residue_count: 2,
            n_analysis_residues: 2,
            physical_residue_roles: vec!["RECEPTOR".into(), "RECEPTOR".into()],
            physical_to_analysis_residue_ids: vec![0, 1],
            analysis_residues: vec![
                AnalysisResidueRecord {
                    analysis_residue_id: 0,
                    physical_residue_id: 0,
                    chain_id: "A".into(),
                    pdb_residue_id: 0,
                    insertion_code: String::new(),
                    residue_name: "ALA".into(),
                    role: "RECEPTOR".into(),
                },
                AnalysisResidueRecord {
                    analysis_residue_id: 1,
                    physical_residue_id: 1,
                    chain_id: "A".into(),
                    pdb_residue_id: 1,
                    insertion_code: String::new(),
                    residue_name: "PHE".into(),
                    role: "RECEPTOR".into(),
                },
            ],
            atom_analysis_residue_ids_sha256: python_canonical_sha256(
                &serde_json::to_value(analysis_ids).unwrap(),
            )
            .unwrap(),
            physical_molecule_table_sha256: table.table_sha256.clone(),
            lambda_fragment_system_sha256: None,
            role_counts,
            excluded_roles: vec![
                "WATER".into(),
                "ION".into(),
                "LAMBDA_FRAGMENT".into(),
                "OTHER".into(),
            ],
            must_recompute_after_fragment_materialization: true,
            contract_sha256: String::new(),
        };
        let mut contract_value = serde_json::to_value(&contract).unwrap();
        contract_value
            .as_object_mut()
            .unwrap()
            .remove("contract_sha256");
        contract.contract_sha256 = python_canonical_sha256(&contract_value).unwrap();
        topology.molecule_identity_table = Some(table);
        topology.analysis_residue_contract = Some(contract);
        topology
    }

    #[test]
    fn test_identify_heavy_atoms() {
        let topo = create_test_topology();
        let heavy = identify_heavy_atoms(&topo).unwrap();

        // All 10 atoms are heavy (no H in our test)
        assert_eq!(heavy.len(), 10, "Should find all heavy atoms");
        assert_eq!(heavy, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_identify_heavy_atoms_with_hydrogens() {
        let mut topo = create_test_topology();
        topo.n_atoms = 12;
        topo.elements.push("H".into());
        topo.elements.push("H".into());
        topo.positions
            .extend_from_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        topo.atom_names.push("H1".into());
        topo.atom_names.push("H2".into());
        topo.residue_ids.push(0);
        topo.residue_ids.push(0);
        topo.chain_ids.push("A".into());
        topo.chain_ids.push("A".into());
        topo.charges.push(0.0);
        topo.charges.push(0.0);
        topo.masses.push(1.0);
        topo.masses.push(1.0);

        let heavy = identify_heavy_atoms(&topo).unwrap();

        // Should exclude the 2 hydrogens
        assert_eq!(heavy.len(), 10, "Should exclude hydrogens");
        assert_eq!(heavy, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_compute_aromatic_centers() {
        let topo = create_test_topology();
        let centers = compute_aromatic_centers(&topo).unwrap();

        // Should find 1 aromatic center (PHE)
        assert_eq!(centers.len(), 1, "Should find 1 aromatic residue");

        // Check centroid is average of PHE atoms (indices 5-9)
        let expected_x = (5.0 + 6.0 + 5.0 + 7.0 + 6.0) / 5.0;
        let expected_y = (5.0 + 5.0 + 6.0 + 5.5 + 6.5) / 5.0;
        let expected_z = (5.0 + 5.0 + 5.0 + 5.0 + 5.0) / 5.0;

        assert!(
            (centers[0][0] - expected_x).abs() < 0.01,
            "X centroid incorrect"
        );
        assert!(
            (centers[0][1] - expected_y).abs() < 0.01,
            "Y centroid incorrect"
        );
        assert!(
            (centers[0][2] - expected_z).abs() < 0.01,
            "Z centroid incorrect"
        );

        println!(
            "Aromatic center: [{:.2}, {:.2}, {:.2}]",
            centers[0][0], centers[0][1], centers[0][2]
        );
    }

    #[test]
    fn test_identify_rt_targets_implicit() {
        let topo = create_test_topology();
        let solvent_mode = SolventMode::Implicit;

        let targets = identify_rt_targets(&topo, &solvent_mode).unwrap();

        // Should have heavy atoms and aromatic centers, but no waters
        assert_eq!(targets.protein_atoms.len(), 10);
        assert!(
            targets.water_atoms.is_none(),
            "Implicit mode should have no waters"
        );
        assert_eq!(targets.aromatic_centers.len(), 1);
        assert_eq!(targets.total_targets, 11); // 10 protein + 1 aromatic

        println!("{}", targets.summary());
    }

    #[test]
    fn test_identify_rt_targets_explicit() {
        let mut topo = create_test_topology();
        // Add some water oxygens
        topo.water_oxygens = vec![100, 101, 102, 103, 104];

        let solvent_mode = SolventMode::Explicit {
            padding_angstroms: 10.0,
        };

        let targets = identify_rt_targets(&topo, &solvent_mode).unwrap();

        // Should have heavy atoms, aromatic centers, AND waters
        assert_eq!(targets.protein_atoms.len(), 10);
        assert!(
            targets.water_atoms.is_some(),
            "Explicit mode should have waters"
        );
        assert_eq!(targets.water_atoms.as_ref().unwrap().len(), 5);
        assert_eq!(targets.aromatic_centers.len(), 1);
        assert_eq!(targets.total_targets, 16); // 10 protein + 5 water + 1 aromatic

        println!("{}", targets.summary());
    }

    #[test]
    fn test_rt_targets_empty() {
        let targets = RtTargets::empty();
        assert_eq!(targets.protein_atoms.len(), 0);
        assert!(targets.water_atoms.is_none());
        assert_eq!(targets.aromatic_centers.len(), 0);
        assert_eq!(targets.total_targets, 0);
    }

    #[test]
    fn test_rt_targets_compute_total() {
        let mut targets = RtTargets {
            protein_atoms: vec![0, 1, 2],
            water_atoms: Some(vec![100, 101]),
            aromatic_centers: vec![[0.0, 0.0, 0.0]],
            total_targets: 0,
        };

        targets.compute_total();
        assert_eq!(targets.total_targets, 6); // 3 + 2 + 1
    }
}
