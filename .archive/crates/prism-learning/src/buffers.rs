use anyhow::Result;
use prism_io::sovereign_types::Atom; // Correct Import

#[derive(Debug, Clone)]
pub struct SimulationBuffers {
    pub positions: Vec<f32>,
    pub velocities: Vec<f32>,
    pub anchors: Vec<f32>,
    pub bias_vec: Vec<f32>,
    pub atom_to_res: Vec<u32>,
    pub num_atoms: usize,
    pub global_step: u64,
}

impl SimulationBuffers {
    pub fn from_atoms(atoms: &[Atom]) -> Self {
        let n = atoms.len();
        let mut pos = Vec::with_capacity(n * 4);
        let mut vel = vec![0.0; n * 4];
        let mut anc = Vec::with_capacity(n * 4);
        let mut bias = vec![0.0; n * 4];
        let mut map = Vec::with_capacity(n);

        for atom in atoms {
            // FIX: Use coords array
            pos.extend_from_slice(&[atom.coords[0], atom.coords[1], atom.coords[2], 1.0]);
            anc.extend_from_slice(&[atom.coords[0], atom.coords[1], atom.coords[2], 1.0]);
            map.push(atom.residue_id as u32);
        }

        Self {
            positions: pos,
            velocities: vel,
            anchors: anc,
            bias_vec: bias,
            atom_to_res: map,
            num_atoms: n,
            global_step: 0,
        }
    }
}
