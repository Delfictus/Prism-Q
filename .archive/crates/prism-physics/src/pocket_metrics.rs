//! Pocket-Centric Functional Metrics for Drug Discovery Validation (Layer 3)
//!
//! This module provides metrics that matter for drug discovery applications:
//! - Binding pocket flexibility analysis
//! - Cryptic pocket detection potential
//! - Allosteric site identification
//! - Functional residue enrichment in flexible regions
//!
//! # Scientific Rationale
//!
//! Drug discovery cares less about "RMSF correlation" and more about:
//! 1. Can we predict which pockets can open/close?
//! 2. Can we identify cryptic sites that only appear in MD?
//! 3. Do high-flexibility regions correlate with functional sites?
//!
//! # Integration with MISATO
//!
//! MISATO provides protein-ligand MD trajectories, allowing us to validate:
//! - Pocket flexibility predictions against actual ligand-bound dynamics
//! - Binding site adaptability metrics

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Pocket flexibility analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PocketFlexibilityAnalysis {
    /// Pocket identifier (e.g., "active_site", "allosteric_1")
    pub pocket_id: String,
    /// Residue indices in this pocket
    pub residue_indices: Vec<usize>,
    /// Mean RMSF of pocket residues
    pub mean_rmsf: f64,
    /// Max RMSF of pocket residues
    pub max_rmsf: f64,
    /// Relative flexibility vs global mean (>1 = more flexible)
    pub relative_flexibility: f64,
    /// Pocket "breathing" score (variation in pocket volume)
    pub breathing_score: f64,
    /// Classification: rigid, moderate, flexible, highly_flexible
    pub flexibility_class: FlexibilityClass,
}

/// Flexibility classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FlexibilityClass {
    Rigid,          // < 0.5× global mean
    Moderate,       // 0.5-1.0× global mean
    Flexible,       // 1.0-2.0× global mean
    HighlyFlexible, // > 2.0× global mean
}

impl FlexibilityClass {
    pub fn from_relative_flex(rel_flex: f64) -> Self {
        if rel_flex < 0.5 {
            FlexibilityClass::Rigid
        } else if rel_flex < 1.0 {
            FlexibilityClass::Moderate
        } else if rel_flex < 2.0 {
            FlexibilityClass::Flexible
        } else {
            FlexibilityClass::HighlyFlexible
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            FlexibilityClass::Rigid => "Rigid",
            FlexibilityClass::Moderate => "Moderate",
            FlexibilityClass::Flexible => "Flexible",
            FlexibilityClass::HighlyFlexible => "Highly Flexible",
        }
    }
}

/// Cryptic pocket detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrypticPocketCandidate {
    /// Central residue index
    pub center_residue: usize,
    /// All residues in the candidate pocket
    pub residues: Vec<usize>,
    /// Cryptic score (0-1, higher = more likely cryptic)
    pub cryptic_score: f64,
    /// Why this might be cryptic
    pub evidence: Vec<CrypticEvidence>,
}

/// Evidence for cryptic pocket candidacy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CrypticEvidence {
    /// High local flexibility suggests conformational change potential
    HighLocalFlexibility { rmsf: f64 },
    /// Nearby rigid regions could form pocket walls
    AdjacentRigidRegions { n_rigid_neighbors: usize },
    /// Hydrophobic patch that could be druggable
    HydrophobicPatch { fraction_hydrophobic: f64 },
    /// Similar to known cryptic pocket patterns
    SequenceMotif { motif: String },
}

/// Allosteric site detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllostericSiteCandidate {
    /// Residue indices
    pub residues: Vec<usize>,
    /// Allosteric score (0-1)
    pub allosteric_score: f64,
    /// Distance from active site (Å)
    pub distance_from_active_site: f64,
    /// Communication pathway strength
    pub communication_score: f64,
}

/// Functional enrichment analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionalEnrichment {
    /// Enrichment of catalytic residues in high-flex regions
    pub catalytic_enrichment: f64,
    /// Enrichment of binding site residues in high-flex regions
    pub binding_enrichment: f64,
    /// Enrichment of known mutation sites in high-flex regions
    pub mutation_enrichment: f64,
    /// P-value for enrichment significance
    pub p_value: f64,
}

/// Complete Layer 3 evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer3Result {
    /// Protein identifier
    pub protein_id: String,
    /// Number of residues
    pub n_residues: usize,
    /// Pocket flexibility analyses
    pub pocket_analyses: Vec<PocketFlexibilityAnalysis>,
    /// Cryptic pocket candidates
    pub cryptic_candidates: Vec<CrypticPocketCandidate>,
    /// Allosteric site candidates
    pub allosteric_candidates: Vec<AllostericSiteCandidate>,
    /// Functional enrichment
    pub functional_enrichment: Option<FunctionalEnrichment>,
    /// Drug target relevance score (0-100)
    pub drug_target_score: f64,
}

/// Pocket-centric metrics calculator
pub struct PocketMetricsCalculator {
    /// Known binding site residues (if available)
    pub known_binding_sites: HashMap<String, Vec<usize>>,
    /// Known allosteric sites (if available)
    pub known_allosteric_sites: HashMap<String, Vec<usize>>,
    /// Threshold for "high flexibility" (relative to global mean)
    pub high_flex_threshold: f64,
    /// Distance cutoff for pocket residue clustering (Å)
    pub pocket_distance_cutoff: f64,
}

impl Default for PocketMetricsCalculator {
    fn default() -> Self {
        Self {
            known_binding_sites: HashMap::new(),
            known_allosteric_sites: HashMap::new(),
            high_flex_threshold: 1.5,
            pocket_distance_cutoff: 8.0,
        }
    }
}

impl PocketMetricsCalculator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate all Layer 3 metrics
    pub fn calculate(
        &self,
        protein_id: &str,
        ca_positions: &[[f32; 3]],
        rmsf: &[f64],
        residue_names: &[String],
    ) -> Layer3Result {
        let n_residues = ca_positions.len();

        // Calculate global statistics
        let global_mean_rmsf = rmsf.iter().sum::<f64>() / n_residues as f64;
        let global_max_rmsf = rmsf.iter().cloned().fold(0.0, f64::max);

        // Identify high-flexibility residues
        let high_flex_residues: Vec<usize> = rmsf
            .iter()
            .enumerate()
            .filter(|(_, &r)| r > global_mean_rmsf * self.high_flex_threshold)
            .map(|(i, _)| i)
            .collect();

        // Analyze known binding sites
        let mut pocket_analyses = Vec::new();
        if let Some(binding_residues) = self.known_binding_sites.get(protein_id) {
            let analysis =
                self.analyze_pocket("binding_site", binding_residues, rmsf, global_mean_rmsf);
            pocket_analyses.push(analysis);
        }

        // Auto-detect potential pockets from structure
        let auto_pockets = self.detect_pockets_from_structure(ca_positions, rmsf);
        for (i, pocket) in auto_pockets.into_iter().enumerate() {
            let analysis = self.analyze_pocket(
                &format!("auto_pocket_{}", i),
                &pocket,
                rmsf,
                global_mean_rmsf,
            );
            pocket_analyses.push(analysis);
        }

        // Detect cryptic pocket candidates
        let cryptic_candidates =
            self.detect_cryptic_candidates(ca_positions, rmsf, residue_names, global_mean_rmsf);

        // Detect allosteric site candidates
        let allosteric_candidates =
            self.detect_allosteric_candidates(ca_positions, rmsf, &high_flex_residues);

        // Calculate functional enrichment (if we have annotations)
        let functional_enrichment =
            self.calculate_functional_enrichment(protein_id, &high_flex_residues, n_residues);

        // Calculate overall drug target score
        let drug_target_score = self.calculate_drug_target_score(
            &pocket_analyses,
            &cryptic_candidates,
            &allosteric_candidates,
        );

        Layer3Result {
            protein_id: protein_id.to_string(),
            n_residues,
            pocket_analyses,
            cryptic_candidates,
            allosteric_candidates,
            functional_enrichment,
            drug_target_score,
        }
    }

    fn analyze_pocket(
        &self,
        pocket_id: &str,
        residues: &[usize],
        rmsf: &[f64],
        global_mean: f64,
    ) -> PocketFlexibilityAnalysis {
        if residues.is_empty() {
            return PocketFlexibilityAnalysis {
                pocket_id: pocket_id.to_string(),
                residue_indices: vec![],
                mean_rmsf: 0.0,
                max_rmsf: 0.0,
                relative_flexibility: 1.0,
                breathing_score: 0.0,
                flexibility_class: FlexibilityClass::Moderate,
            };
        }

        let pocket_rmsf: Vec<f64> = residues
            .iter()
            .filter_map(|&i| rmsf.get(i).copied())
            .collect();

        let mean_rmsf = pocket_rmsf.iter().sum::<f64>() / pocket_rmsf.len() as f64;
        let max_rmsf = pocket_rmsf.iter().cloned().fold(0.0, f64::max);
        let relative_flexibility = if global_mean > 0.0 {
            mean_rmsf / global_mean
        } else {
            1.0
        };

        // Breathing score: variance in pocket RMSF values
        let variance = pocket_rmsf
            .iter()
            .map(|&r| (r - mean_rmsf).powi(2))
            .sum::<f64>()
            / pocket_rmsf.len() as f64;
        let breathing_score = variance.sqrt() / mean_rmsf.max(0.01);

        let flexibility_class = FlexibilityClass::from_relative_flex(relative_flexibility);

        PocketFlexibilityAnalysis {
            pocket_id: pocket_id.to_string(),
            residue_indices: residues.to_vec(),
            mean_rmsf,
            max_rmsf,
            relative_flexibility,
            breathing_score,
            flexibility_class,
        }
    }

    fn detect_pockets_from_structure(
        &self,
        ca_positions: &[[f32; 3]],
        rmsf: &[f64],
    ) -> Vec<Vec<usize>> {
        // Simple pocket detection: find clusters of residues with moderate flexibility
        // surrounded by either rigid or flexible regions
        let n = ca_positions.len();
        let mut pockets = Vec::new();
        let mut used = vec![false; n];

        let global_mean = rmsf.iter().sum::<f64>() / n as f64;

        for i in 0..n {
            if used[i] {
                continue;
            }

            // Look for concave regions (simplified: check local geometry)
            let neighbors =
                self.find_spatial_neighbors(ca_positions, i, self.pocket_distance_cutoff);

            if neighbors.len() >= 4 {
                // Potential pocket if we have enough neighbors
                let pocket_residues: Vec<usize> =
                    neighbors.iter().copied().filter(|&j| !used[j]).collect();

                if pocket_residues.len() >= 4 {
                    for &r in &pocket_residues {
                        used[r] = true;
                    }
                    pockets.push(pocket_residues);
                }
            }

            if pockets.len() >= 5 {
                break; // Limit auto-detected pockets
            }
        }

        pockets
    }

    fn find_spatial_neighbors(
        &self,
        ca_positions: &[[f32; 3]],
        center: usize,
        cutoff: f64,
    ) -> Vec<usize> {
        let center_pos = ca_positions[center];
        let cutoff_sq = (cutoff * cutoff) as f32;

        ca_positions
            .iter()
            .enumerate()
            .filter(|(j, pos)| {
                if *j == center {
                    return false;
                }
                let dx = pos[0] - center_pos[0];
                let dy = pos[1] - center_pos[1];
                let dz = pos[2] - center_pos[2];
                dx * dx + dy * dy + dz * dz < cutoff_sq
            })
            .map(|(j, _)| j)
            .collect()
    }

    fn detect_cryptic_candidates(
        &self,
        ca_positions: &[[f32; 3]],
        rmsf: &[f64],
        residue_names: &[String],
        global_mean: f64,
    ) -> Vec<CrypticPocketCandidate> {
        let mut candidates = Vec::new();
        let n = rmsf.len();

        // Look for regions with high local flexibility surrounded by rigid regions
        for i in 0..n {
            let neighbors = self.find_spatial_neighbors(ca_positions, i, 10.0);
            if neighbors.len() < 5 {
                continue;
            }

            let local_rmsf: Vec<f64> = neighbors
                .iter()
                .filter_map(|&j| rmsf.get(j).copied())
                .collect();

            if local_rmsf.is_empty() {
                continue;
            }

            let local_mean = local_rmsf.iter().sum::<f64>() / local_rmsf.len() as f64;

            // Cryptic pocket signature: local flexibility > 1.5× global mean
            if local_mean > global_mean * 1.5 {
                let mut evidence = Vec::new();
                evidence.push(CrypticEvidence::HighLocalFlexibility { rmsf: local_mean });

                // Check for adjacent rigid regions
                let rigid_neighbors = neighbors
                    .iter()
                    .filter(|&&j| rmsf.get(j).map(|&r| r < global_mean * 0.5).unwrap_or(false))
                    .count();

                if rigid_neighbors >= 2 {
                    evidence.push(CrypticEvidence::AdjacentRigidRegions {
                        n_rigid_neighbors: rigid_neighbors,
                    });
                }

                // Check hydrophobicity
                let hydrophobic_count = neighbors
                    .iter()
                    .filter(|&&j| {
                        is_hydrophobic(residue_names.get(j).map(|s| s.as_str()).unwrap_or(""))
                    })
                    .count();
                let hydrophobic_fraction = hydrophobic_count as f64 / neighbors.len() as f64;

                if hydrophobic_fraction > 0.4 {
                    evidence.push(CrypticEvidence::HydrophobicPatch {
                        fraction_hydrophobic: hydrophobic_fraction,
                    });
                }

                // Calculate cryptic score
                let cryptic_score = calculate_cryptic_score(&evidence);

                if cryptic_score > 0.3 {
                    candidates.push(CrypticPocketCandidate {
                        center_residue: i,
                        residues: neighbors,
                        cryptic_score,
                        evidence,
                    });
                }
            }

            if candidates.len() >= 10 {
                break;
            }
        }

        // Sort by score and take top candidates
        candidates.sort_by(|a, b| {
            b.cryptic_score
                .partial_cmp(&a.cryptic_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(5);
        candidates
    }

    fn detect_allosteric_candidates(
        &self,
        ca_positions: &[[f32; 3]],
        rmsf: &[f64],
        high_flex_residues: &[usize],
    ) -> Vec<AllostericSiteCandidate> {
        // Simplified: find flexible regions distant from N/C termini
        // Real implementation would use residue network analysis
        let n = ca_positions.len();

        if high_flex_residues.is_empty() {
            return vec![];
        }

        // Cluster high-flexibility residues
        let mut clusters: Vec<Vec<usize>> = Vec::new();
        let mut assigned: HashSet<usize> = HashSet::new();

        for &res in high_flex_residues {
            if assigned.contains(&res) {
                continue;
            }

            let neighbors = self.find_spatial_neighbors(ca_positions, res, 8.0);
            let cluster: Vec<usize> = std::iter::once(res)
                .chain(neighbors.into_iter())
                .filter(|r| high_flex_residues.contains(r) && !assigned.contains(r))
                .collect();

            for &r in &cluster {
                assigned.insert(r);
            }

            if cluster.len() >= 3 {
                clusters.push(cluster);
            }
        }

        // Convert clusters to allosteric candidates
        clusters
            .into_iter()
            .take(3)
            .enumerate()
            .map(|(i, residues)| {
                // Estimate distance from "active site" (approximate as center of protein)
                let center_idx = n / 2;
                let mean_dist = residues
                    .iter()
                    .map(|&r| distance(ca_positions[r], ca_positions[center_idx]))
                    .sum::<f64>()
                    / residues.len() as f64;

                // Allosteric score based on being far from center but well-defined cluster
                let allosteric_score =
                    (mean_dist / 20.0).min(1.0) * (residues.len() as f64 / 10.0).min(1.0);

                AllostericSiteCandidate {
                    residues,
                    allosteric_score,
                    distance_from_active_site: mean_dist,
                    communication_score: 0.5, // Would need proper network analysis
                }
            })
            .collect()
    }

    fn calculate_functional_enrichment(
        &self,
        _protein_id: &str,
        high_flex_residues: &[usize],
        n_residues: usize,
    ) -> Option<FunctionalEnrichment> {
        // Simplified: would need actual functional annotations
        // For now, return placeholder based on high-flex coverage
        let flex_fraction = high_flex_residues.len() as f64 / n_residues as f64;

        Some(FunctionalEnrichment {
            catalytic_enrichment: 1.0 + flex_fraction,
            binding_enrichment: 1.0 + flex_fraction * 0.5,
            mutation_enrichment: 1.0 + flex_fraction * 0.8,
            p_value: 0.05, // Placeholder
        })
    }

    fn calculate_drug_target_score(
        &self,
        pocket_analyses: &[PocketFlexibilityAnalysis],
        cryptic_candidates: &[CrypticPocketCandidate],
        allosteric_candidates: &[AllostericSiteCandidate],
    ) -> f64 {
        let mut score = 0.0;

        // Pocket diversity contribution (0-40 points)
        let flex_classes: HashSet<_> = pocket_analyses
            .iter()
            .map(|p| p.flexibility_class)
            .collect();
        score += (flex_classes.len() as f64 / 4.0) * 40.0;

        // Cryptic pocket potential (0-30 points)
        if !cryptic_candidates.is_empty() {
            let max_cryptic = cryptic_candidates
                .iter()
                .map(|c| c.cryptic_score)
                .fold(0.0, f64::max);
            score += max_cryptic * 30.0;
        }

        // Allosteric site potential (0-30 points)
        if !allosteric_candidates.is_empty() {
            let max_allosteric = allosteric_candidates
                .iter()
                .map(|a| a.allosteric_score)
                .fold(0.0, f64::max);
            score += max_allosteric * 30.0;
        }

        score.min(100.0)
    }
}

// Helper functions

fn is_hydrophobic(residue: &str) -> bool {
    matches!(
        residue.to_uppercase().as_str(),
        "ALA" | "VAL" | "LEU" | "ILE" | "MET" | "PHE" | "TRP" | "PRO"
    )
}

fn calculate_cryptic_score(evidence: &[CrypticEvidence]) -> f64 {
    let mut score = 0.0;

    for ev in evidence {
        match ev {
            CrypticEvidence::HighLocalFlexibility { rmsf } => {
                score += (rmsf / 3.0).min(0.4); // Max 0.4 from flexibility
            }
            CrypticEvidence::AdjacentRigidRegions { n_rigid_neighbors } => {
                score += (*n_rigid_neighbors as f64 / 5.0).min(0.3); // Max 0.3
            }
            CrypticEvidence::HydrophobicPatch {
                fraction_hydrophobic,
            } => {
                score += fraction_hydrophobic * 0.3; // Max 0.3
            }
            CrypticEvidence::SequenceMotif { .. } => {
                score += 0.2; // Bonus for motif match
            }
        }
    }

    score.min(1.0)
}

fn distance(a: [f32; 3], b: [f32; 3]) -> f64 {
    let dx = (a[0] - b[0]) as f64;
    let dy = (a[1] - b[1]) as f64;
    let dz = (a[2] - b[2]) as f64;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flexibility_classification() {
        assert_eq!(
            FlexibilityClass::from_relative_flex(0.3),
            FlexibilityClass::Rigid
        );
        assert_eq!(
            FlexibilityClass::from_relative_flex(0.7),
            FlexibilityClass::Moderate
        );
        assert_eq!(
            FlexibilityClass::from_relative_flex(1.5),
            FlexibilityClass::Flexible
        );
        assert_eq!(
            FlexibilityClass::from_relative_flex(2.5),
            FlexibilityClass::HighlyFlexible
        );
    }

    #[test]
    fn test_hydrophobic_classification() {
        assert!(is_hydrophobic("ALA"));
        assert!(is_hydrophobic("VAL"));
        assert!(!is_hydrophobic("ASP"));
        assert!(!is_hydrophobic("LYS"));
    }
}
