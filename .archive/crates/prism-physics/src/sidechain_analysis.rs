//! Sidechain Analysis for Residue-Type Flexibility
//!
//! Provides residue-specific flexibility factors based on sidechain properties.
//! Different amino acids have intrinsic flexibility differences due to:
//! - Sidechain size and rotational freedom (chi angles)
//! - Backbone constraints (e.g., proline's ring)
//! - Hydrogen bonding capacity
//!
//! # References
//!
//! - Smith et al. (2003) "The relationship between B-factors and protein dynamics"
//! - Radivojac et al. (2004) "Intrinsic disorder and functional proteomics"

use std::collections::HashMap;

/// Residue-type flexibility factors based on empirical observations
///
/// Values derived from:
/// - B-factor analysis across PDB structures
/// - MD simulation RMSF distributions
/// - Rotamer library statistics
///
/// Scale: 1.0 = average flexibility
///        > 1.0 = more flexible
///        < 1.0 = more rigid
pub const RESIDUE_FLEXIBILITY: &[(&str, f64)] = &[
    // Most flexible - small or no sidechain
    ("GLY", 1.40), // No sidechain, maximum backbone freedom
    // Flexible - small sidechains
    ("SER", 1.10), // Small hydroxyl, moderate H-bonding
    ("THR", 1.05), // Beta-branched but small
    ("ASN", 1.08), // Amide group, some H-bonding
    ("ASP", 1.05), // Carboxylate, charged
    // Moderate flexibility
    ("GLU", 1.02), // Longer charged sidechain
    ("GLN", 1.00), // Longer amide
    ("LYS", 1.05), // Long flexible charged chain
    ("ARG", 0.98), // Large but multiple H-bonds constrain
    ("HIS", 0.95), // Aromatic, can be charged
    // Less flexible - bulky or constrained
    ("ALA", 0.90), // Small methyl, relatively rigid
    ("VAL", 0.85), // Beta-branched, restricts backbone
    ("ILE", 0.82), // Beta-branched, bulky
    ("LEU", 0.88), // Bulky but not beta-branched
    ("MET", 0.90), // Long but sulfur provides some flexibility
    // Aromatic - rigid rings
    ("PHE", 0.80), // Large aromatic ring
    ("TYR", 0.78), // Large aromatic + hydroxyl
    ("TRP", 0.72), // Largest aromatic, most rigid
    // Special cases
    ("CYS", 0.85), // Disulfide potential constrains
    ("PRO", 0.60), // Ring structure, very rigid backbone
];

/// Get flexibility factor for a single residue
///
/// # Arguments
/// * `residue_name` - Three-letter residue code (e.g., "ALA", "GLY")
///
/// # Returns
/// Flexibility factor (default 1.0 for unknown residues)
pub fn flexibility_factor(residue_name: &str) -> f64 {
    let name = residue_name.trim().to_uppercase();
    RESIDUE_FLEXIBILITY
        .iter()
        .find(|(r, _)| *r == name)
        .map(|(_, f)| *f)
        .unwrap_or(1.0) // Default for unknown residues
}

/// Sidechain property categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SidechainCategory {
    /// No sidechain (Glycine)
    None,
    /// Small aliphatic (Ala, Val, Leu, Ile)
    SmallAliphatic,
    /// Aromatic (Phe, Tyr, Trp, His)
    Aromatic,
    /// Polar uncharged (Ser, Thr, Asn, Gln)
    PolarUncharged,
    /// Positively charged (Lys, Arg)
    PositiveCharge,
    /// Negatively charged (Asp, Glu)
    NegativeCharge,
    /// Sulfur-containing (Cys, Met)
    Sulfur,
    /// Special (Pro)
    Special,
}

impl SidechainCategory {
    /// Get category for a residue
    pub fn from_residue(residue_name: &str) -> Self {
        match residue_name.trim().to_uppercase().as_str() {
            "GLY" => SidechainCategory::None,
            "ALA" | "VAL" | "LEU" | "ILE" => SidechainCategory::SmallAliphatic,
            "PHE" | "TYR" | "TRP" | "HIS" => SidechainCategory::Aromatic,
            "SER" | "THR" | "ASN" | "GLN" => SidechainCategory::PolarUncharged,
            "LYS" | "ARG" => SidechainCategory::PositiveCharge,
            "ASP" | "GLU" => SidechainCategory::NegativeCharge,
            "CYS" | "MET" => SidechainCategory::Sulfur,
            "PRO" => SidechainCategory::Special,
            _ => SidechainCategory::PolarUncharged, // Default
        }
    }

    /// Get category-based flexibility factor
    pub fn flexibility_factor(&self) -> f64 {
        match self {
            SidechainCategory::None => 1.40,
            SidechainCategory::SmallAliphatic => 0.85,
            SidechainCategory::Aromatic => 0.77,
            SidechainCategory::PolarUncharged => 1.05,
            SidechainCategory::PositiveCharge => 1.00,
            SidechainCategory::NegativeCharge => 1.03,
            SidechainCategory::Sulfur => 0.87,
            SidechainCategory::Special => 0.60,
        }
    }
}

/// Sidechain analyzer for batch processing
pub struct SidechainAnalyzer {
    /// Cache of flexibility factors by residue name
    factor_cache: HashMap<String, f64>,
    /// Use category-based or residue-specific factors
    use_residue_specific: bool,
}

impl Default for SidechainAnalyzer {
    fn default() -> Self {
        Self {
            factor_cache: RESIDUE_FLEXIBILITY
                .iter()
                .map(|(name, factor)| (name.to_string(), *factor))
                .collect(),
            use_residue_specific: true,
        }
    }
}

impl SidechainAnalyzer {
    /// Create analyzer using category-based factors
    pub fn category_based() -> Self {
        Self {
            factor_cache: HashMap::new(),
            use_residue_specific: false,
        }
    }

    /// Get flexibility factor for a residue
    pub fn get_factor(&self, residue_name: &str) -> f64 {
        let name = residue_name.trim().to_uppercase();
        if self.use_residue_specific {
            *self.factor_cache.get(&name).unwrap_or(&1.0)
        } else {
            SidechainCategory::from_residue(&name).flexibility_factor()
        }
    }

    /// Compute flexibility factors for a sequence of residues
    ///
    /// # Arguments
    /// * `residue_names` - Sequence of three-letter residue codes
    ///
    /// # Returns
    /// Vector of flexibility factors (same length as input)
    pub fn compute_factors(&self, residue_names: &[&str]) -> Vec<f64> {
        residue_names
            .iter()
            .map(|name| self.get_factor(name))
            .collect()
    }

    /// Apply flexibility factors to RMSF predictions
    ///
    /// # Arguments
    /// * `rmsf` - Predicted RMSF values
    /// * `residue_names` - Sequence of residue names
    ///
    /// # Returns
    /// Modified RMSF values weighted by sidechain flexibility
    pub fn apply_to_rmsf(&self, rmsf: &[f64], residue_names: &[&str]) -> Vec<f64> {
        if rmsf.len() != residue_names.len() {
            // Return unchanged if lengths don't match
            return rmsf.to_vec();
        }

        rmsf.iter()
            .zip(residue_names.iter())
            .map(|(&r, name)| r * self.get_factor(name))
            .collect()
    }

    /// Compute burial-weighted flexibility
    ///
    /// Buried residues have reduced flexibility regardless of type.
    /// Surface residues express full intrinsic flexibility.
    ///
    /// # Arguments
    /// * `residue_names` - Residue types
    /// * `burial_fractions` - 0.0 = fully exposed, 1.0 = fully buried
    pub fn compute_burial_weighted_factors(
        &self,
        residue_names: &[&str],
        burial_fractions: &[f64],
    ) -> Vec<f64> {
        residue_names
            .iter()
            .zip(burial_fractions.iter())
            .map(|(name, &burial)| {
                let intrinsic = self.get_factor(name);
                // Buried residues are dampened toward 0.7
                // Surface residues keep intrinsic flexibility
                let damping = 1.0 - burial * 0.3; // Max 30% reduction when buried
                intrinsic * damping
            })
            .collect()
    }
}

/// Compute hydropathy index for residues (Kyte-Doolittle scale)
///
/// Positive values = hydrophobic
/// Negative values = hydrophilic
pub fn hydropathy_index(residue_name: &str) -> f64 {
    match residue_name.trim().to_uppercase().as_str() {
        "ILE" => 4.5,
        "VAL" => 4.2,
        "LEU" => 3.8,
        "PHE" => 2.8,
        "CYS" => 2.5,
        "MET" => 1.9,
        "ALA" => 1.8,
        "GLY" => -0.4,
        "THR" => -0.7,
        "SER" => -0.8,
        "TRP" => -0.9,
        "TYR" => -1.3,
        "PRO" => -1.6,
        "HIS" => -3.2,
        "GLU" => -3.5,
        "GLN" => -3.5,
        "ASP" => -3.5,
        "ASN" => -3.5,
        "LYS" => -3.9,
        "ARG" => -4.5,
        _ => 0.0,
    }
}

/// Compute average hydropathy in a window
pub fn hydropathy_profile(residue_names: &[&str], window_size: usize) -> Vec<f64> {
    let n = residue_names.len();
    if n == 0 || window_size == 0 {
        return vec![];
    }

    let half_window = window_size / 2;
    let hydropathy: Vec<f64> = residue_names.iter().map(|r| hydropathy_index(r)).collect();

    (0..n)
        .map(|i| {
            let start = i.saturating_sub(half_window);
            let end = (i + half_window + 1).min(n);
            let sum: f64 = hydropathy[start..end].iter().sum();
            sum / (end - start) as f64
        })
        .collect()
}

/// Sidechain property summary for a protein
#[derive(Debug, Clone)]
pub struct SidechainSummary {
    pub n_residues: usize,
    pub mean_flexibility: f64,
    pub mean_hydropathy: f64,
    pub category_counts: HashMap<SidechainCategory, usize>,
}

impl SidechainSummary {
    /// Compute summary from residue sequence
    pub fn from_sequence(residue_names: &[&str]) -> Self {
        let n = residue_names.len();
        if n == 0 {
            return Self {
                n_residues: 0,
                mean_flexibility: 1.0,
                mean_hydropathy: 0.0,
                category_counts: HashMap::new(),
            };
        }

        let analyzer = SidechainAnalyzer::default();
        let factors = analyzer.compute_factors(residue_names);
        let mean_flexibility = factors.iter().sum::<f64>() / n as f64;

        let mean_hydropathy = residue_names
            .iter()
            .map(|r| hydropathy_index(r))
            .sum::<f64>()
            / n as f64;

        let mut category_counts = HashMap::new();
        for name in residue_names {
            let cat = SidechainCategory::from_residue(name);
            *category_counts.entry(cat).or_insert(0) += 1;
        }

        Self {
            n_residues: n,
            mean_flexibility,
            mean_hydropathy,
            category_counts,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flexibility_factors() {
        // Glycine should be most flexible
        assert!(flexibility_factor("GLY") > flexibility_factor("ALA"));

        // Proline should be most rigid
        assert!(flexibility_factor("PRO") < flexibility_factor("GLY"));
        assert!(flexibility_factor("PRO") < flexibility_factor("ALA"));

        // Aromatics should be rigid
        assert!(flexibility_factor("TRP") < 0.85);
        assert!(flexibility_factor("PHE") < 0.85);
    }

    #[test]
    fn test_sidechain_analyzer() {
        let analyzer = SidechainAnalyzer::default();

        let residues = vec!["ALA", "GLY", "PRO", "TRP", "LYS"];
        let factors = analyzer.compute_factors(&residues);

        assert_eq!(factors.len(), 5);
        // GLY should be most flexible
        assert!(factors[1] > factors[0]); // GLY > ALA
                                          // PRO should be most rigid
        assert!(factors[2] < factors[0]); // PRO < ALA
    }

    #[test]
    fn test_apply_to_rmsf() {
        let analyzer = SidechainAnalyzer::default();

        let rmsf = vec![1.0, 1.0, 1.0, 1.0];
        let residues = vec!["GLY", "ALA", "PRO", "VAL"];

        let adjusted = analyzer.apply_to_rmsf(&rmsf, &residues);

        // GLY adjustment should be highest
        assert!(adjusted[0] > adjusted[1]);
        // PRO adjustment should be lowest
        assert!(adjusted[2] < adjusted[1]);
    }

    #[test]
    fn test_hydropathy() {
        // Isoleucine is most hydrophobic
        assert!(hydropathy_index("ILE") > 4.0);
        // Arginine is most hydrophilic
        assert!(hydropathy_index("ARG") < -4.0);
        // Glycine is neutral
        assert!(hydropathy_index("GLY").abs() < 1.0);
    }

    #[test]
    fn test_hydropathy_profile() {
        let residues = vec!["ALA", "VAL", "ILE", "GLY", "LYS", "ARG"];
        let profile = hydropathy_profile(&residues, 3);

        assert_eq!(profile.len(), 6);

        // First few residues are hydrophobic
        assert!(profile[1] > 0.0);
        // Last few are hydrophilic
        assert!(profile[4] < 0.0);
    }

    #[test]
    fn test_sidechain_summary() {
        let residues = vec!["ALA", "GLY", "PHE", "LYS", "ASP"];
        let summary = SidechainSummary::from_sequence(&residues);

        assert_eq!(summary.n_residues, 5);
        assert!(summary.mean_flexibility > 0.8);
        assert!(summary.mean_flexibility < 1.2);
    }

    #[test]
    fn test_burial_weighted_factors() {
        let analyzer = SidechainAnalyzer::default();

        let residues = vec!["GLY", "GLY"];
        let burial = vec![0.0, 1.0]; // First exposed, second buried

        let factors = analyzer.compute_burial_weighted_factors(&residues, &burial);

        // Both are glycine but buried one should be dampened
        assert!(factors[0] > factors[1]);
    }

    #[test]
    fn test_category_classification() {
        assert_eq!(
            SidechainCategory::from_residue("GLY"),
            SidechainCategory::None
        );
        assert_eq!(
            SidechainCategory::from_residue("PRO"),
            SidechainCategory::Special
        );
        assert_eq!(
            SidechainCategory::from_residue("PHE"),
            SidechainCategory::Aromatic
        );
        assert_eq!(
            SidechainCategory::from_residue("LYS"),
            SidechainCategory::PositiveCharge
        );
    }
}
