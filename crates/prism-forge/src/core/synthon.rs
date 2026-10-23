use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum MolecularStateError {
    #[error("coordinates must be a flat N x 3 array, got {0} floats")]
    CoordinateArity(usize),
    #[error("charges length {charges} does not match atom count {atoms}")]
    ChargeArity { atoms: usize, charges: usize },
    #[error("attachment atom index {atom_index} is outside atom count {atom_count}")]
    AttachmentAtomOutOfRange {
        atom_index: usize,
        atom_count: usize,
    },
    #[error("leaving group atom index {atom_index} is outside atom count {atom_count}")]
    LeavingGroupOutOfRange {
        atom_index: usize,
        atom_count: usize,
    },
    #[error("dihedral reference atom index {atom_index} is outside atom count {atom_count}")]
    DihedralReferenceOutOfRange {
        atom_index: usize,
        atom_count: usize,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttachmentPoint {
    pub atom_index: usize,
    pub leaving_group_atom_index: Option<usize>,
    pub attachment_vector: [f32; 3],
    pub dihedral_reference_atom_index: Option<usize>,
}

impl AttachmentPoint {
    #[must_use]
    pub const fn new(
        atom_index: usize,
        leaving_group_atom_index: Option<usize>,
        attachment_vector: [f32; 3],
        dihedral_reference_atom_index: Option<usize>,
    ) -> Self {
        Self {
            atom_index,
            leaving_group_atom_index,
            attachment_vector,
            dihedral_reference_atom_index,
        }
    }

    fn validate(&self, atom_count: usize) -> Result<(), MolecularStateError> {
        if self.atom_index >= atom_count {
            return Err(MolecularStateError::AttachmentAtomOutOfRange {
                atom_index: self.atom_index,
                atom_count,
            });
        }
        if let Some(atom_index) = self.leaving_group_atom_index {
            if atom_index >= atom_count {
                return Err(MolecularStateError::LeavingGroupOutOfRange {
                    atom_index,
                    atom_count,
                });
            }
        }
        if let Some(atom_index) = self.dihedral_reference_atom_index {
            if atom_index >= atom_count {
                return Err(MolecularStateError::DihedralReferenceOutOfRange {
                    atom_index,
                    atom_count,
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Synthon3D {
    pub coordinates: Vec<f32>,
    pub charges: Vec<f32>,
    pub attachment_points: Vec<AttachmentPoint>,
    pub bonds: Vec<(usize, usize)>,
}

impl Synthon3D {
    pub fn new(
        coordinates: Vec<f32>,
        charges: Vec<f32>,
        attachment_points: Vec<AttachmentPoint>,
    ) -> Result<Self, MolecularStateError> {
        validate_state(&coordinates, &charges, &attachment_points)?;
        Ok(Self {
            coordinates,
            charges,
            attachment_points,
            bonds: Vec::new(),
        })
    }

    pub fn new_with_bonds(
        coordinates: Vec<f32>,
        charges: Vec<f32>,
        attachment_points: Vec<AttachmentPoint>,
        bonds: Vec<(usize, usize)>,
    ) -> Result<Self, MolecularStateError> {
        validate_state(&coordinates, &charges, &attachment_points)?;
        let atom_count = charges.len();
        Ok(Self {
            coordinates,
            charges,
            attachment_points,
            bonds: normalize_bonds(bonds, atom_count),
        })
    }

    #[must_use]
    pub fn atom_count(&self) -> usize {
        self.charges.len()
    }

    #[must_use]
    pub fn atom_xyz(&self, atom_index: usize) -> [f32; 3] {
        let offset = atom_index * 3;
        [
            self.coordinates[offset],
            self.coordinates[offset + 1],
            self.coordinates[offset + 2],
        ]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScaffoldState3D {
    pub coordinates: Vec<f32>,
    pub charges: Vec<f32>,
    pub attachment_points: Vec<AttachmentPoint>,
    pub bonds: Vec<(usize, usize)>,
}

impl ScaffoldState3D {
    pub fn new(
        coordinates: Vec<f32>,
        charges: Vec<f32>,
        attachment_points: Vec<AttachmentPoint>,
    ) -> Result<Self, MolecularStateError> {
        validate_state(&coordinates, &charges, &attachment_points)?;
        Ok(Self {
            coordinates,
            charges,
            attachment_points,
            bonds: Vec::new(),
        })
    }

    pub fn new_with_bonds(
        coordinates: Vec<f32>,
        charges: Vec<f32>,
        attachment_points: Vec<AttachmentPoint>,
        bonds: Vec<(usize, usize)>,
    ) -> Result<Self, MolecularStateError> {
        validate_state(&coordinates, &charges, &attachment_points)?;
        let atom_count = charges.len();
        let filtered_bonds = normalize_bonds(bonds, atom_count);
        Ok(Self {
            coordinates,
            charges,
            attachment_points,
            bonds: filtered_bonds,
        })
    }

    #[must_use]
    pub fn atom_count(&self) -> usize {
        self.charges.len()
    }

    #[must_use]
    pub fn atom_xyz(&self, atom_index: usize) -> [f32; 3] {
        let offset = atom_index * 3;
        [
            self.coordinates[offset],
            self.coordinates[offset + 1],
            self.coordinates[offset + 2],
        ]
    }
}

fn normalize_bonds(bonds: Vec<(usize, usize)>, atom_count: usize) -> Vec<(usize, usize)> {
    let mut normalized = Vec::new();
    for (lhs, rhs) in bonds {
        if lhs >= atom_count || rhs >= atom_count || lhs == rhs {
            continue;
        }
        let bond = if lhs < rhs { (lhs, rhs) } else { (rhs, lhs) };
        if !normalized.contains(&bond) {
            normalized.push(bond);
        }
    }
    normalized
}

fn validate_state(
    coordinates: &[f32],
    charges: &[f32],
    attachment_points: &[AttachmentPoint],
) -> Result<(), MolecularStateError> {
    if !coordinates.len().is_multiple_of(3) {
        return Err(MolecularStateError::CoordinateArity(coordinates.len()));
    }
    let atom_count = coordinates.len() / 3;
    if charges.len() != atom_count {
        return Err(MolecularStateError::ChargeArity {
            atoms: atom_count,
            charges: charges.len(),
        });
    }
    for attachment_point in attachment_points {
        attachment_point.validate(atom_count)?;
    }
    Ok(())
}
