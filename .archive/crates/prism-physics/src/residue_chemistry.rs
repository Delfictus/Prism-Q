//! Residue Chemistry Module for Chemistry-Aware GNM
//!
//! Provides amino acid-specific properties that affect protein flexibility:
//! - Intrinsic residue flexibility factors (from MD literature)
//! - Amino acid pair stiffness calculations
//! - Residue classification (charged, polar, hydrophobic, aromatic)

/// Standard amino acid 3-letter codes in alphabetical order
pub const AA_ORDER: [&str; 20] = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
    "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
];

/// Intrinsic flexibility factors for each amino acid.
/// Values derived from crystallographic B-factors and MD simulations.
///
/// Lower values = more rigid (stiffer springs in GNM)
/// Higher values = more flexible (weaker springs)
///
/// Key patterns:
/// - GLY (1.40): No sidechain → maximum backbone freedom
/// - PRO (0.55): Ring constrains φ angle → very rigid
/// - TRP (0.75): Large aromatic ring → rigid
/// - LYS/ARG (~1.1-1.15): Long charged sidechains → flexible
pub const RESIDUE_FLEXIBILITY: [(&str, f64); 20] = [
    ("ALA", 0.90), // Small sidechain, relatively rigid
    ("ARG", 1.10), // Long charged sidechain
    ("ASN", 1.05), // Short polar
    ("ASP", 1.00), // Short charged
    ("CYS", 0.85), // Can form disulfides → constrained
    ("GLN", 1.10), // Longer polar
    ("GLU", 1.05), // Longer charged
    ("GLY", 1.40), // No sidechain → maximum flexibility
    ("HIS", 0.90), // Aromatic, can be charged
    ("ILE", 0.85), // Branched β-carbon
    ("LEU", 0.90), // Hydrophobic, moderate
    ("LYS", 1.15), // Long flexible charged sidechain
    ("MET", 1.00), // Long flexible sidechain
    ("PHE", 0.80), // Rigid aromatic ring
    ("PRO", 0.55), // Ring constrains backbone φ → very rigid
    ("SER", 1.05), // Small polar, moderate flexibility
    ("THR", 0.95), // Branched β-carbon constrains
    ("TRP", 0.75), // Largest aromatic → very rigid
    ("TYR", 0.80), // Rigid aromatic + hydroxyl
    ("VAL", 0.85), // Branched β-carbon
];

/// Residue classification for interaction type determination
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidueClass {
    /// Positively charged at physiological pH: ARG, LYS, HIS
    PositivelyCharged,
    /// Negatively charged at physiological pH: ASP, GLU
    NegativelyCharged,
    /// Polar but uncharged: SER, THR, ASN, GLN, CYS
    Polar,
    /// Hydrophobic aliphatic: ALA, VAL, ILE, LEU, MET
    Hydrophobic,
    /// Aromatic: PHE, TYR, TRP
    Aromatic,
    /// Special cases
    Glycine, // Unique flexibility
    Proline, // Unique rigidity
}

impl ResidueClass {
    /// Classify a residue by its 3-letter code
    pub fn from_name(name: &str) -> Self {
        match name.to_uppercase().as_str() {
            "ARG" | "LYS" => ResidueClass::PositivelyCharged,
            "HIS" => ResidueClass::PositivelyCharged, // At physiological pH
            "ASP" | "GLU" => ResidueClass::NegativelyCharged,
            "SER" | "THR" | "ASN" | "GLN" | "CYS" => ResidueClass::Polar,
            "ALA" | "VAL" | "ILE" | "LEU" | "MET" => ResidueClass::Hydrophobic,
            "PHE" | "TYR" | "TRP" => ResidueClass::Aromatic,
            "GLY" => ResidueClass::Glycine,
            "PRO" => ResidueClass::Proline,
            _ => ResidueClass::Hydrophobic, // Default for non-standard
        }
    }

    /// Check if residue is positively charged
    pub fn is_positive(&self) -> bool {
        matches!(self, ResidueClass::PositivelyCharged)
    }

    /// Check if residue is negatively charged
    pub fn is_negative(&self) -> bool {
        matches!(self, ResidueClass::NegativelyCharged)
    }

    /// Check if residue is aromatic
    pub fn is_aromatic(&self) -> bool {
        matches!(self, ResidueClass::Aromatic)
    }
}

/// Get the intrinsic flexibility factor for an amino acid.
/// Returns 1.0 for unknown residues.
pub fn get_flexibility_factor(residue_name: &str) -> f64 {
    let name = residue_name.to_uppercase();
    for (aa, flex) in RESIDUE_FLEXIBILITY.iter() {
        if *aa == name {
            return *flex;
        }
    }
    1.0 // Default for non-standard amino acids
}

/// Calculate the pair stiffness factor for two amino acids in contact.
///
/// Uses a dampened approach to avoid extreme stiffness ranges.
/// The formula: 1.0 + damping * (1.0 - flex[i] * flex[j])
///
/// This creates a more moderate range:
/// - GLY-GLY: 1.0 + 0.3 * (1.0 - 1.96) = 0.71 → weak spring
/// - PRO-PRO: 1.0 + 0.3 * (1.0 - 0.30) = 1.21 → stiff spring
/// - Range: ~1.7x (vs 6.5x in the inverse formula)
pub fn pair_stiffness_factor(res_i: &str, res_j: &str) -> f64 {
    let flex_i = get_flexibility_factor(res_i);
    let flex_j = get_flexibility_factor(res_j);

    // Dampened approach: avoid extreme ranges
    // Product of flexibilities ranges from ~0.30 (PRO-PRO) to ~1.96 (GLY-GLY)
    let product = flex_i * flex_j;
    let damping = 0.3; // Control the strength of the effect

    // Higher flexibility product → lower stiffness (weaker spring)
    // Lower flexibility product → higher stiffness (stiffer spring)
    1.0 + damping * (1.0 - product)
}

/// Calculate raw (undampened) pair stiffness for advanced use.
/// WARNING: This creates a 6.5x range which may be too extreme for some applications.
pub fn pair_stiffness_factor_raw(res_i: &str, res_j: &str) -> f64 {
    let flex_i = get_flexibility_factor(res_i);
    let flex_j = get_flexibility_factor(res_j);
    1.0 / (flex_i * flex_j)
}

/// Enhanced pair stiffness that accounts for interaction type.
/// Uses moderate bonuses to avoid over-correction:
/// - CYS-CYS: potential disulfide bond (+0.3)
/// - Hydrophobic-Hydrophobic: stable core contacts (+0.1)
/// - Opposite charges: salt bridge potential (+0.15)
/// - Same charges: electrostatic repulsion (-0.1)
/// - Aromatic-Aromatic: pi-stacking (+0.1)
///
/// Uses additive (not multiplicative) bonuses for better control.
pub fn enhanced_pair_stiffness(res_i: &str, res_j: &str) -> f64 {
    let base = pair_stiffness_factor(res_i, res_j);

    let name_i = res_i.to_uppercase();
    let name_j = res_j.to_uppercase();

    // Disulfide bond potential (strong bonus)
    if name_i == "CYS" && name_j == "CYS" {
        return base + 0.3;
    }

    let class_i = ResidueClass::from_name(res_i);
    let class_j = ResidueClass::from_name(res_j);

    // Salt bridge (opposite charges) - moderate bonus
    if (class_i.is_positive() && class_j.is_negative())
        || (class_i.is_negative() && class_j.is_positive())
    {
        return base + 0.15;
    }

    // Same-charge repulsion (slight penalty)
    if (class_i.is_positive() && class_j.is_positive())
        || (class_i.is_negative() && class_j.is_negative())
    {
        return (base - 0.1).max(0.5); // Floor at 0.5 to avoid negative springs
    }

    // Aromatic-aromatic (pi-stacking) - slight bonus
    if class_i.is_aromatic() && class_j.is_aromatic() {
        return base + 0.1;
    }

    // Hydrophobic core contacts - slight bonus
    if matches!(class_i, ResidueClass::Hydrophobic) && matches!(class_j, ResidueClass::Hydrophobic)
    {
        return base + 0.1;
    }

    base
}

/// Get index of amino acid in AA_ORDER (for matrix lookups)
pub fn aa_index(name: &str) -> Option<usize> {
    let upper = name.to_uppercase();
    AA_ORDER.iter().position(|&aa| aa == upper)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flexibility_factors() {
        // GLY should be most flexible
        assert!(get_flexibility_factor("GLY") > 1.3);
        // PRO should be most rigid
        assert!(get_flexibility_factor("PRO") < 0.6);
        // Unknown residues return 1.0
        assert_eq!(get_flexibility_factor("XYZ"), 1.0);
    }

    #[test]
    fn test_pair_stiffness() {
        // GLY-GLY should be lower (flexible → weak spring)
        let gly_gly = pair_stiffness_factor("GLY", "GLY");
        // 1.0 + 0.3 * (1.0 - 1.96) = 0.712
        assert!(gly_gly < 0.8);
        assert!(gly_gly > 0.6);

        // PRO-PRO should be higher (rigid → stiff spring)
        let pro_pro = pair_stiffness_factor("PRO", "PRO");
        // 1.0 + 0.3 * (1.0 - 0.3025) = 1.21
        assert!(pro_pro > 1.15);
        assert!(pro_pro < 1.3);

        // Symmetric
        let ab = pair_stiffness_factor("ALA", "VAL");
        let ba = pair_stiffness_factor("VAL", "ALA");
        assert!((ab - ba).abs() < 1e-10);
    }

    #[test]
    fn test_enhanced_stiffness() {
        // CYS-CYS should get disulfide bonus (+0.3)
        let cys_cys = enhanced_pair_stiffness("CYS", "CYS");
        let cys_base = pair_stiffness_factor("CYS", "CYS");
        assert!((cys_cys - (cys_base + 0.3)).abs() < 1e-10);

        // Salt bridge should get bonus (+0.15)
        let salt = enhanced_pair_stiffness("ARG", "ASP");
        let salt_base = pair_stiffness_factor("ARG", "ASP");
        assert!((salt - (salt_base + 0.15)).abs() < 1e-10);

        // Same charges should be penalized (-0.1)
        let same = enhanced_pair_stiffness("ARG", "LYS");
        let same_base = pair_stiffness_factor("ARG", "LYS");
        assert!(same < same_base);
    }

    #[test]
    fn test_residue_classification() {
        assert!(ResidueClass::from_name("ARG").is_positive());
        assert!(ResidueClass::from_name("ASP").is_negative());
        assert!(ResidueClass::from_name("PHE").is_aromatic());
    }
}
