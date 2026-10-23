//! AMBER ff14SB Bonded Force Calculator
//!
//! GPU-accelerated calculation of bonded interactions:
//! - Bond stretching (harmonic)
//! - Angle bending (harmonic)
//! - Dihedral torsion (Fourier series)
//! - Improper torsion
//! - 1-4 nonbonded interactions
//!
//! This module provides separate force calculation that can be used
//! independently of the mega-fused kernel for testing/validation.

use anyhow::{Context, Result};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use std::sync::Arc;

/// Bond interaction parameters
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct BondParam {
    pub k: f32,  // Force constant (kcal/mol/Å²)
    pub r0: f32, // Equilibrium distance (Å)
}

/// Angle interaction parameters
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct AngleParam {
    pub k: f32,      // Force constant (kcal/mol/rad²)
    pub theta0: f32, // Equilibrium angle (radians)
}

/// Dihedral interaction parameters
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct DihedralParam {
    pub k: f32,     // Barrier height (kcal/mol)
    pub n: f32,     // Periodicity (1, 2, 3, 4, 6)
    pub phase: f32, // Phase offset (radians)
    pub _pad: f32,  // Padding for alignment
}

/// 1-4 nonbonded interaction parameters
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct NB14Param {
    pub sigma: f32,   // LJ sigma (Å)
    pub epsilon: f32, // LJ epsilon (kcal/mol)
    pub qi: f32,      // Charge of atom i (e)
    pub qj: f32,      // Charge of atom j (e)
}

/// Bond interaction
#[derive(Debug, Clone, Copy)]
pub struct Bond {
    pub atom_i: u32,
    pub atom_j: u32,
    pub params: BondParam,
}

/// Angle interaction
#[derive(Debug, Clone, Copy)]
pub struct Angle {
    pub atom_i: u32,
    pub atom_j: u32,
    pub atom_k: u32,
    pub params: AngleParam,
}

/// Dihedral interaction
#[derive(Debug, Clone, Copy)]
pub struct Dihedral {
    pub atom_i: u32,
    pub atom_j: u32,
    pub atom_k: u32,
    pub atom_l: u32,
    pub params: DihedralParam,
}

/// 1-4 nonbonded pair
#[derive(Debug, Clone, Copy)]
pub struct Pair14 {
    pub atom_i: u32,
    pub atom_j: u32,
    pub params: NB14Param,
}

/// Energy components from force calculation
#[derive(Debug, Clone, Default)]
pub struct EnergyComponents {
    pub bond_energy: f64,
    pub angle_energy: f64,
    pub dihedral_energy: f64,
    pub improper_energy: f64,
    pub nb14_energy: f64,
    pub total: f64,
}

/// Topology builder for setting up force field parameters
#[derive(Debug, Default)]
pub struct TopologyBuilder {
    pub bonds: Vec<Bond>,
    pub angles: Vec<Angle>,
    pub dihedrals: Vec<Dihedral>,
    pub pairs_14: Vec<Pair14>,
}

impl TopologyBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_bond(&mut self, i: usize, j: usize, k: f32, r0: f32) {
        self.bonds.push(Bond {
            atom_i: i as u32,
            atom_j: j as u32,
            params: BondParam { k, r0 },
        });
    }

    pub fn add_angle(&mut self, i: usize, j: usize, k_idx: usize, k: f32, theta0: f32) {
        self.angles.push(Angle {
            atom_i: i as u32,
            atom_j: j as u32,
            atom_k: k_idx as u32,
            params: AngleParam { k, theta0 },
        });
    }

    pub fn add_dihedral(
        &mut self,
        i: usize,
        j: usize,
        k_idx: usize,
        l: usize,
        k: f32,
        periodicity: f32,
        phase: f32,
    ) {
        self.dihedrals.push(Dihedral {
            atom_i: i as u32,
            atom_j: j as u32,
            atom_k: k_idx as u32,
            atom_l: l as u32,
            params: DihedralParam {
                k,
                n: periodicity,
                phase,
                _pad: 0.0,
            },
        });
    }

    pub fn add_pair_14(&mut self, i: usize, j: usize, sigma: f32, epsilon: f32, qi: f32, qj: f32) {
        self.pairs_14.push(Pair14 {
            atom_i: i as u32,
            atom_j: j as u32,
            params: NB14Param {
                sigma,
                epsilon,
                qi,
                qj,
            },
        });
    }
}

/// AMBER bonded force calculator (GPU-accelerated)
pub struct AmberBondedForces {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    n_atoms: usize,

    // GPU buffers
    d_positions: CudaSlice<f32>,
    d_forces: CudaSlice<f32>,

    // Topology data
    bonds: Vec<Bond>,
    angles: Vec<Angle>,
    dihedrals: Vec<Dihedral>,
    pairs_14: Vec<Pair14>,
}

impl AmberBondedForces {
    /// Create a new force calculator
    pub fn new(context: Arc<CudaContext>, n_atoms: usize) -> Result<Self> {
        let stream = context.default_stream();

        // Allocate GPU buffers
        let d_positions = stream
            .alloc_zeros::<f32>(n_atoms * 3)
            .context("Failed to allocate positions buffer")?;
        let d_forces = stream
            .alloc_zeros::<f32>(n_atoms * 3)
            .context("Failed to allocate forces buffer")?;

        Ok(Self {
            context,
            stream,
            n_atoms,
            d_positions,
            d_forces,
            bonds: Vec::new(),
            angles: Vec::new(),
            dihedrals: Vec::new(),
            pairs_14: Vec::new(),
        })
    }

    /// Set bonds for force calculation
    pub fn set_bonds(&mut self, bonds: Vec<Bond>) {
        self.bonds = bonds;
    }

    /// Set angles for force calculation
    pub fn set_angles(&mut self, angles: Vec<Angle>) {
        self.angles = angles;
    }

    /// Set dihedrals for force calculation
    pub fn set_dihedrals(&mut self, dihedrals: Vec<Dihedral>) {
        self.dihedrals = dihedrals;
    }

    /// Set 1-4 pairs for force calculation
    pub fn set_pairs_14(&mut self, pairs: Vec<Pair14>) {
        self.pairs_14 = pairs;
    }

    /// Upload complete topology (bonds, angles, dihedrals, 1-4 pairs)
    pub fn upload_topology(
        &mut self,
        bonds: &[Bond],
        angles: &[Angle],
        dihedrals: &[Dihedral],
        pairs_14: &[Pair14],
    ) -> Result<()> {
        self.bonds = bonds.to_vec();
        self.angles = angles.to_vec();
        self.dihedrals = dihedrals.to_vec();
        self.pairs_14 = pairs_14.to_vec();
        Ok(())
    }

    /// Upload positions to GPU
    pub fn upload_positions(&mut self, positions: &[f32]) -> Result<()> {
        if positions.len() != self.n_atoms * 3 {
            anyhow::bail!(
                "Position array size mismatch: expected {}, got {}",
                self.n_atoms * 3,
                positions.len()
            );
        }
        self.stream.memcpy_htod(positions, &mut self.d_positions)?;
        Ok(())
    }

    /// Compute forces on GPU
    ///
    /// Returns (energy, forces) where forces are in kcal/(mol·Å) for each atom
    pub fn compute(&mut self) -> Result<(f64, Vec<f32>)> {
        // For now, compute forces on CPU and return
        // GPU kernel implementation would go here
        let mut forces = vec![0.0f32; self.n_atoms * 3];

        // Download positions from GPU
        let mut positions = vec![0.0f32; self.n_atoms * 3];
        self.stream.memcpy_dtoh(&self.d_positions, &mut positions)?;

        // Compute bond forces: F = -dE/dr = -2k(r - r0) * (r_vec / |r|)
        for bond in &self.bonds {
            let i = bond.atom_i as usize;
            let j = bond.atom_j as usize;
            let dx = positions[j * 3] - positions[i * 3];
            let dy = positions[j * 3 + 1] - positions[i * 3 + 1];
            let dz = positions[j * 3 + 2] - positions[i * 3 + 2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r < 1e-10 {
                continue;
            }

            let dr = r - bond.params.r0;
            let force_mag = -2.0 * bond.params.k * dr / r;

            let fx = force_mag * dx;
            let fy = force_mag * dy;
            let fz = force_mag * dz;

            forces[i * 3] -= fx;
            forces[i * 3 + 1] -= fy;
            forces[i * 3 + 2] -= fz;
            forces[j * 3] += fx;
            forces[j * 3 + 1] += fy;
            forces[j * 3 + 2] += fz;
        }

        // Compute angle forces
        for angle in &self.angles {
            let (ai, aj, ak) = (
                angle.atom_i as usize,
                angle.atom_j as usize,
                angle.atom_k as usize,
            );
            let (fi, fj, fk) =
                compute_angle_forces(&positions, ai, aj, ak, angle.params.k, angle.params.theta0);

            for c in 0..3 {
                forces[ai * 3 + c] += fi[c];
                forces[aj * 3 + c] += fj[c];
                forces[ak * 3 + c] += fk[c];
            }
        }

        // Compute dihedral forces
        for dihedral in &self.dihedrals {
            let (ai, aj, ak, al) = (
                dihedral.atom_i as usize,
                dihedral.atom_j as usize,
                dihedral.atom_k as usize,
                dihedral.atom_l as usize,
            );
            let (fi, fj, fk, fl) = compute_dihedral_forces(
                &positions,
                ai,
                aj,
                ak,
                al,
                dihedral.params.k,
                dihedral.params.n,
                dihedral.params.phase,
            );

            for c in 0..3 {
                forces[ai * 3 + c] += fi[c];
                forces[aj * 3 + c] += fj[c];
                forces[ak * 3 + c] += fk[c];
                forces[al * 3 + c] += fl[c];
            }
        }

        // Compute total energy (simplified)
        let energy = self.calculate_energies()?.total;

        Ok((energy, forces))
    }

    /// Calculate bonded energies for given positions
    pub fn calculate_energies(&self) -> Result<EnergyComponents> {
        let mut energy = EnergyComponents::default();

        // Download positions from GPU
        let mut positions = vec![0.0f32; self.n_atoms * 3];
        self.stream.memcpy_dtoh(&self.d_positions, &mut positions)?;

        // Bond energies: E = k(r - r0)²
        for bond in &self.bonds {
            let i = bond.atom_i as usize;
            let j = bond.atom_j as usize;
            let dx = positions[j * 3] - positions[i * 3];
            let dy = positions[j * 3 + 1] - positions[i * 3 + 1];
            let dz = positions[j * 3 + 2] - positions[i * 3 + 2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            let dr = r - bond.params.r0;
            energy.bond_energy += (bond.params.k * dr * dr) as f64;
        }

        // Angle energies: E = k(θ - θ0)²
        for angle in &self.angles {
            let (ai, aj, ak) = (
                angle.atom_i as usize,
                angle.atom_j as usize,
                angle.atom_k as usize,
            );
            let theta = compute_angle(&positions, ai, aj, ak);
            let dtheta = theta - angle.params.theta0;
            energy.angle_energy += (angle.params.k * dtheta * dtheta) as f64;
        }

        // Dihedral energies: E = k(1 + cos(n*φ - δ))
        for dihedral in &self.dihedrals {
            let (ai, aj, ak, al) = (
                dihedral.atom_i as usize,
                dihedral.atom_j as usize,
                dihedral.atom_k as usize,
                dihedral.atom_l as usize,
            );
            let phi = compute_dihedral_angle(&positions, ai, aj, ak, al);
            let n = dihedral.params.n;
            energy.dihedral_energy +=
                (dihedral.params.k * (1.0 + (n * phi - dihedral.params.phase).cos())) as f64;
        }

        energy.total =
            energy.bond_energy + energy.angle_energy + energy.dihedral_energy + energy.nb14_energy;
        Ok(energy)
    }
}

/// Compute angle in radians
fn compute_angle(positions: &[f32], i: usize, j: usize, k: usize) -> f32 {
    // Vector from j to i
    let rji = [
        positions[i * 3] - positions[j * 3],
        positions[i * 3 + 1] - positions[j * 3 + 1],
        positions[i * 3 + 2] - positions[j * 3 + 2],
    ];

    // Vector from j to k
    let rjk = [
        positions[k * 3] - positions[j * 3],
        positions[k * 3 + 1] - positions[j * 3 + 1],
        positions[k * 3 + 2] - positions[j * 3 + 2],
    ];

    let dot = rji[0] * rjk[0] + rji[1] * rjk[1] + rji[2] * rjk[2];
    let len_ji = (rji[0] * rji[0] + rji[1] * rji[1] + rji[2] * rji[2]).sqrt();
    let len_jk = (rjk[0] * rjk[0] + rjk[1] * rjk[1] + rjk[2] * rjk[2]).sqrt();
    let cos_theta = (dot / (len_ji * len_jk)).clamp(-1.0, 1.0);
    cos_theta.acos()
}

/// Compute angle forces
fn compute_angle_forces(
    positions: &[f32],
    i: usize,
    j: usize,
    k: usize,
    k_angle: f32,
    theta0: f32,
) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let rji = [
        positions[i * 3] - positions[j * 3],
        positions[i * 3 + 1] - positions[j * 3 + 1],
        positions[i * 3 + 2] - positions[j * 3 + 2],
    ];
    let rjk = [
        positions[k * 3] - positions[j * 3],
        positions[k * 3 + 1] - positions[j * 3 + 1],
        positions[k * 3 + 2] - positions[j * 3 + 2],
    ];

    let rji_len = (rji[0] * rji[0] + rji[1] * rji[1] + rji[2] * rji[2]).sqrt();
    let rjk_len = (rjk[0] * rjk[0] + rjk[1] * rjk[1] + rjk[2] * rjk[2]).sqrt();

    if rji_len < 1e-10 || rjk_len < 1e-10 {
        return ([0.0; 3], [0.0; 3], [0.0; 3]);
    }

    let dot = rji[0] * rjk[0] + rji[1] * rjk[1] + rji[2] * rjk[2];
    let cos_theta = (dot / (rji_len * rjk_len)).clamp(-0.9999, 0.9999);
    let theta = cos_theta.acos();
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt().max(1e-10);

    let dtheta = theta - theta0;
    let prefactor = -2.0 * k_angle * dtheta / sin_theta;

    // Derivatives of cos(theta) with respect to positions
    let inv_rji = 1.0 / rji_len;
    let inv_rjk = 1.0 / rjk_len;

    let mut fi = [0.0f32; 3];
    let mut fk = [0.0f32; 3];

    for c in 0..3 {
        fi[c] = prefactor * (rjk[c] * inv_rji * inv_rjk - cos_theta * rji[c] * inv_rji * inv_rji);
        fk[c] = prefactor * (rji[c] * inv_rji * inv_rjk - cos_theta * rjk[c] * inv_rjk * inv_rjk);
    }

    let fj = [-fi[0] - fk[0], -fi[1] - fk[1], -fi[2] - fk[2]];

    (fi, fj, fk)
}

/// Calculate dihedral angle from positions
fn compute_dihedral_angle(positions: &[f32], i: usize, j: usize, k: usize, l: usize) -> f32 {
    let b1 = [
        positions[j * 3] - positions[i * 3],
        positions[j * 3 + 1] - positions[i * 3 + 1],
        positions[j * 3 + 2] - positions[i * 3 + 2],
    ];
    let b2 = [
        positions[k * 3] - positions[j * 3],
        positions[k * 3 + 1] - positions[j * 3 + 1],
        positions[k * 3 + 2] - positions[j * 3 + 2],
    ];
    let b3 = [
        positions[l * 3] - positions[k * 3],
        positions[l * 3 + 1] - positions[k * 3 + 1],
        positions[l * 3 + 2] - positions[k * 3 + 2],
    ];

    let n1 = cross(b1, b2);
    let n2 = cross(b2, b3);
    let m1 = cross(n1, b2);

    let x = dot(n1, n2);
    let y = dot(m1, n2);
    y.atan2(x)
}

/// Compute dihedral forces
fn compute_dihedral_forces(
    positions: &[f32],
    i: usize,
    j: usize,
    k: usize,
    l: usize,
    k_dih: f32,
    n: f32,
    phase: f32,
) -> ([f32; 3], [f32; 3], [f32; 3], [f32; 3]) {
    let phi = compute_dihedral_angle(positions, i, j, k, l);

    // dE/dphi = k * n * sin(n*phi - phase)
    let dphi = k_dih * n * (n * phi - phase).sin();

    // Numerical derivatives for simplicity (GPU kernel would be analytical)
    let eps = 1e-5f32;
    let mut fi = [0.0f32; 3];
    let mut fj = [0.0f32; 3];
    let mut fk = [0.0f32; 3];
    let mut fl = [0.0f32; 3];

    // Use finite differences
    let mut pos = positions.to_vec();
    for c in 0..3 {
        // Atom i
        pos[i * 3 + c] += eps;
        let phi_p = compute_dihedral_angle(&pos, i, j, k, l);
        pos[i * 3 + c] -= 2.0 * eps;
        let phi_m = compute_dihedral_angle(&pos, i, j, k, l);
        pos[i * 3 + c] += eps;
        fi[c] = -dphi * (phi_p - phi_m) / (2.0 * eps);

        // Atom j
        pos[j * 3 + c] += eps;
        let phi_p = compute_dihedral_angle(&pos, i, j, k, l);
        pos[j * 3 + c] -= 2.0 * eps;
        let phi_m = compute_dihedral_angle(&pos, i, j, k, l);
        pos[j * 3 + c] += eps;
        fj[c] = -dphi * (phi_p - phi_m) / (2.0 * eps);

        // Atom k
        pos[k * 3 + c] += eps;
        let phi_p = compute_dihedral_angle(&pos, i, j, k, l);
        pos[k * 3 + c] -= 2.0 * eps;
        let phi_m = compute_dihedral_angle(&pos, i, j, k, l);
        pos[k * 3 + c] += eps;
        fk[c] = -dphi * (phi_p - phi_m) / (2.0 * eps);

        // Atom l
        pos[l * 3 + c] += eps;
        let phi_p = compute_dihedral_angle(&pos, i, j, k, l);
        pos[l * 3 + c] -= 2.0 * eps;
        let phi_m = compute_dihedral_angle(&pos, i, j, k, l);
        pos[l * 3 + c] += eps;
        fl[c] = -dphi * (phi_p - phi_m) / (2.0 * eps);
    }

    (fi, fj, fk, fl)
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bond_param() {
        let param = BondParam { k: 300.0, r0: 1.5 };
        assert_eq!(param.k, 300.0);
        assert_eq!(param.r0, 1.5);
    }

    #[test]
    fn test_dihedral_param() {
        let param = DihedralParam {
            k: 1.0,
            n: 2.0,
            phase: 0.0,
            _pad: 0.0,
        };
        assert_eq!(param.n, 2.0);
    }
}
