//! H-Bond Constraint Solver for Protein MD
//!
//! Implements analytic (non-iterative) SHAKE/RATTLE-like constraints for
//! protein X-H bonds. This freezes fast H-bond vibrations (~10 fs period),
//! allowing larger timesteps (2.0 fs vs 0.25 fs) and better temperature control.
//!
//! # Key Insight
//!
//! Proteins have only 5 H-bond cluster topologies:
//!
//! | Type    | Example                    | Count | Algorithm       |
//! |---------|----------------------------|-------|-----------------|
//! | SINGLE_H| C-H, N-H, O-H, S-H         | ~30%  | 1 eq, exact     |
//! | CH2     | Methylene (Lys, Arg)       | ~40%  | 2x2 Cramer      |
//! | CH3     | Methyl (Ala, Val, Leu)     | ~25%  | 3x3 Cramer      |
//! | NH2     | Amide (Asn, Gln)           | <1%   | Same as CH2     |
//! | NH3     | Protonated Lys             | <1%   | Same as CH3     |
//!
//! Each type is solved analytically (no iteration) using Cramer's rule.

use anyhow::{Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// H-bond constraint cluster type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ClusterType {
    SingleH = 1, // C-H, N-H, O-H, S-H
    CH2 = 2,     // Methylene groups
    CH3 = 3,     // Methyl groups
    NH2 = 4,     // Amide groups (Asn, Gln sidechains)
    NH3 = 5,     // Protonated lysine
}

/// H-bond constraint cluster data (matches CUDA struct)
/// 48 bytes total, packed for efficient GPU memory access
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct HConstraintCluster {
    pub central_atom: i32,
    pub hydrogen_atoms: [i32; 3], // -1 for unused slots
    pub bond_lengths: [f32; 3],   // Target distances in Angstroms
    pub inv_mass_central: f32,    // 1/m_heavy
    pub inv_mass_h: f32,          // 1/m_H
    pub n_hydrogens: i32,         // 1, 2, or 3
    pub cluster_type: i32,
}

// Safety: HConstraintCluster is a POD type with no padding issues
unsafe impl cudarc::driver::DeviceRepr for HConstraintCluster {}
unsafe impl cudarc::driver::ValidAsZeroBits for HConstraintCluster {}

impl HConstraintCluster {
    /// Create a new single-H cluster (C-H, N-H, O-H, S-H)
    pub fn single_h(
        central: usize,
        hydrogen: usize,
        bond_length: f32,
        mass_central: f32,
        mass_h: f32,
    ) -> Self {
        Self {
            central_atom: central as i32,
            hydrogen_atoms: [hydrogen as i32, -1, -1],
            bond_lengths: [bond_length, 0.0, 0.0],
            inv_mass_central: 1.0 / mass_central,
            inv_mass_h: 1.0 / mass_h,
            n_hydrogens: 1,
            cluster_type: ClusterType::SingleH as i32,
        }
    }

    /// Create a CH2/NH2 cluster (2 hydrogens)
    pub fn two_h(
        central: usize,
        h1: usize,
        h2: usize,
        d1: f32,
        d2: f32,
        mass_central: f32,
        mass_h: f32,
        is_nitrogen: bool,
    ) -> Self {
        Self {
            central_atom: central as i32,
            hydrogen_atoms: [h1 as i32, h2 as i32, -1],
            bond_lengths: [d1, d2, 0.0],
            inv_mass_central: 1.0 / mass_central,
            inv_mass_h: 1.0 / mass_h,
            n_hydrogens: 2,
            cluster_type: if is_nitrogen {
                ClusterType::NH2 as i32
            } else {
                ClusterType::CH2 as i32
            },
        }
    }

    /// Create a CH3/NH3 cluster (3 hydrogens)
    pub fn three_h(
        central: usize,
        h1: usize,
        h2: usize,
        h3: usize,
        d1: f32,
        d2: f32,
        d3: f32,
        mass_central: f32,
        mass_h: f32,
        is_nitrogen: bool,
    ) -> Self {
        Self {
            central_atom: central as i32,
            hydrogen_atoms: [h1 as i32, h2 as i32, h3 as i32],
            bond_lengths: [d1, d2, d3],
            inv_mass_central: 1.0 / mass_central,
            inv_mass_h: 1.0 / mass_h,
            n_hydrogens: 3,
            cluster_type: if is_nitrogen {
                ClusterType::NH3 as i32
            } else {
                ClusterType::CH3 as i32
            },
        }
    }

    /// Create from JSON cluster data (from OpenMM topology)
    #[cfg(feature = "serde")]
    pub fn from_json(json: &serde_json::Value) -> Option<Self> {
        Some(Self {
            central_atom: json.get("central_atom")?.as_i64()? as i32,
            hydrogen_atoms: [
                json.get("hydrogen_atoms")?.get(0)?.as_i64()? as i32,
                json.get("hydrogen_atoms")?.get(1)?.as_i64().unwrap_or(-1) as i32,
                json.get("hydrogen_atoms")?.get(2)?.as_i64().unwrap_or(-1) as i32,
            ],
            bond_lengths: [
                json.get("bond_lengths")?.get(0)?.as_f64()? as f32,
                json.get("bond_lengths")?.get(1)?.as_f64().unwrap_or(0.0) as f32,
                json.get("bond_lengths")?.get(2)?.as_f64().unwrap_or(0.0) as f32,
            ],
            inv_mass_central: json.get("inv_mass_central")?.as_f64()? as f32,
            inv_mass_h: json.get("inv_mass_h")?.as_f64()? as f32,
            n_hydrogens: json.get("n_hydrogens")?.as_i64()? as i32,
            cluster_type: json.get("type")?.as_i64()? as i32,
        })
    }
}

/// GPU H-bond constraint solver
pub struct HConstraints {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,

    // Kernel function
    kernel_unified: CudaFunction,

    // GPU buffers
    d_clusters: CudaSlice<HConstraintCluster>,

    // Counts
    n_clusters: usize,
    n_single_h: usize,
    n_ch2: usize,
    n_ch3: usize,
}

impl HConstraints {
    /// Create H-bond constraint solver from cluster list
    pub fn new(context: Arc<CudaContext>, clusters: &[HConstraintCluster]) -> Result<Self> {
        if clusters.is_empty() {
            return Err(anyhow::anyhow!("No H-bond clusters provided"));
        }

        // Count by type
        let n_single_h = clusters.iter().filter(|c| c.n_hydrogens == 1).count();
        let n_ch2 = clusters.iter().filter(|c| c.n_hydrogens == 2).count();
        let n_ch3 = clusters.iter().filter(|c| c.n_hydrogens == 3).count();

        log::info!(
            "🔗 HConstraints: {} clusters ({} single, {} CH2/NH2, {} CH3/NH3)",
            clusters.len(),
            n_single_h,
            n_ch2,
            n_ch3
        );

        let stream = context.default_stream();

        // Load H-constraints PTX module - use absolute path for reliability
        let ptx_path = concat!(env!("CARGO_MANIFEST_DIR"), "/target/ptx/h_constraints.ptx");
        let ptx_src = std::fs::read_to_string(ptx_path)
            .with_context(|| format!("Failed to read PTX file: {}", ptx_path))?;
        let ptx = Ptx::from_src(&ptx_src);
        let module = context
            .load_module(ptx)
            .with_context(|| format!("Failed to load H-constraints PTX from {}", ptx_path))?;

        // Load unified kernel
        let kernel_unified = module
            .load_function("constrain_h_bonds_unified")
            .context("Failed to load constrain_h_bonds_unified kernel")?;

        // Upload clusters to GPU
        let mut d_clusters = stream.alloc_zeros::<HConstraintCluster>(clusters.len())?;
        stream.memcpy_htod(clusters, &mut d_clusters)?;

        Ok(Self {
            context,
            stream,
            _module: module,
            kernel_unified,
            d_clusters,
            n_clusters: clusters.len(),
            n_single_h,
            n_ch2,
            n_ch3,
        })
    }

    /// Get total number of constrained bonds (for DOF calculation)
    ///
    /// Each single-H cluster has 1 constraint, CH2/NH2 has 2, CH3/NH3 has 3.
    pub fn n_constraints(&self) -> usize {
        self.n_single_h + 2 * self.n_ch2 + 3 * self.n_ch3
    }

    /// Get number of clusters
    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    /// Apply constraints to positions and velocities
    ///
    /// Call this AFTER the integration step (position update) to enforce
    /// bond length constraints on X-H bonds.
    pub fn apply(
        &self,
        d_positions: &mut CudaSlice<f32>,
        d_velocities: &mut CudaSlice<f32>,
        dt: f32,
    ) -> Result<()> {
        if self.n_clusters == 0 {
            return Ok(());
        }

        let threads = 256;
        let blocks = (self.n_clusters + threads - 1) / threads;

        let config = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_clusters_i32 = self.n_clusters as i32;

        unsafe {
            let mut builder = self.stream.launch_builder(&self.kernel_unified);
            builder.arg(d_positions);
            builder.arg(d_velocities);
            builder.arg(&self.d_clusters);
            builder.arg(&n_clusters_i32);
            builder.arg(&dt);
            builder.launch(config)?;
        }

        self.stream.synchronize()?;

        Ok(())
    }

    /// Get number of single-H constraints
    pub fn n_single_h(&self) -> usize {
        self.n_single_h
    }

    /// Get number of CH2/NH2 constraints
    pub fn n_ch2(&self) -> usize {
        self.n_ch2
    }

    /// Get number of CH3/NH3 constraints
    pub fn n_ch3(&self) -> usize {
        self.n_ch3
    }
}

/// Build H-constraint clusters from topology data
///
/// This automatically detects H-bond clusters from bonds and masses.
pub fn build_h_clusters(
    bonds: &[(usize, usize, f32)], // (i, j, r0)
    masses: &[f32],
    elements: &[&str],
) -> Vec<HConstraintCluster> {
    use std::collections::HashMap;

    // Build adjacency for H-bonds
    let mut h_neighbors: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();

    for &(i, j, r0) in bonds {
        let (heavy, h, bond_len) = if elements.get(i).map(|s| *s) == Some("H")
            && elements.get(j).map(|s| *s) != Some("H")
        {
            (j, i, r0)
        } else if elements.get(j).map(|s| *s) == Some("H")
            && elements.get(i).map(|s| *s) != Some("H")
        {
            (i, j, r0)
        } else {
            continue; // Not an X-H bond
        };

        h_neighbors.entry(heavy).or_default().push((h, bond_len));
    }

    // Build clusters
    let mut clusters = Vec::new();

    for (heavy, hydrogens) in h_neighbors {
        let is_nitrogen = elements.get(heavy).map(|s| *s) == Some("N");
        let mass_central = masses.get(heavy).copied().unwrap_or(12.0);
        let mass_h = if !hydrogens.is_empty() {
            masses.get(hydrogens[0].0).copied().unwrap_or(1.008)
        } else {
            1.008
        };

        let cluster = match hydrogens.len() {
            1 => {
                let (h, d) = hydrogens[0];
                HConstraintCluster::single_h(heavy, h, d, mass_central, mass_h)
            }
            2 => {
                let (h1, d1) = hydrogens[0];
                let (h2, d2) = hydrogens[1];
                HConstraintCluster::two_h(heavy, h1, h2, d1, d2, mass_central, mass_h, is_nitrogen)
            }
            3 => {
                let (h1, d1) = hydrogens[0];
                let (h2, d2) = hydrogens[1];
                let (h3, d3) = hydrogens[2];
                HConstraintCluster::three_h(
                    heavy,
                    h1,
                    h2,
                    h3,
                    d1,
                    d2,
                    d3,
                    mass_central,
                    mass_h,
                    is_nitrogen,
                )
            }
            _ => continue, // Unusual, skip
        };

        clusters.push(cluster);
    }

    log::debug!("Built {} H-constraint clusters", clusters.len());
    clusters
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cluster_creation() {
        let cluster = HConstraintCluster::single_h(10, 11, 1.09, 12.0, 1.008);
        assert_eq!(cluster.central_atom, 10);
        assert_eq!(cluster.hydrogen_atoms[0], 11);
        assert_eq!(cluster.n_hydrogens, 1);
        assert!((cluster.inv_mass_h - 1.0 / 1.008).abs() < 1e-6);
    }

    #[test]
    fn test_cluster_types() {
        assert_eq!(ClusterType::SingleH as i32, 1);
        assert_eq!(ClusterType::CH2 as i32, 2);
        assert_eq!(ClusterType::CH3 as i32, 3);
        assert_eq!(ClusterType::NH2 as i32, 4);
        assert_eq!(ClusterType::NH3 as i32, 5);
    }

    #[test]
    fn test_build_h_clusters() {
        // Simple test: one C-H bond
        let bonds = vec![(0, 1, 1.09)];
        let masses = vec![12.0, 1.008];
        let elements = vec!["C", "H"];

        let clusters = build_h_clusters(&bonds, &masses, &elements);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].central_atom, 0);
        assert_eq!(clusters[0].hydrogen_atoms[0], 1);
        assert_eq!(clusters[0].n_hydrogens, 1);
    }

    #[test]
    fn test_build_ch3_cluster() {
        // Methyl group: C with 3 H
        let bonds = vec![
            (0, 1, 1.09), // C-H1
            (0, 2, 1.09), // C-H2
            (0, 3, 1.09), // C-H3
        ];
        let masses = vec![12.0, 1.008, 1.008, 1.008];
        let elements = vec!["C", "H", "H", "H"];

        let clusters = build_h_clusters(&bonds, &masses, &elements);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].n_hydrogens, 3);
        assert_eq!(clusters[0].cluster_type, ClusterType::CH3 as i32);
    }
}
