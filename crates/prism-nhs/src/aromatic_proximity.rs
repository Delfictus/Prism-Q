//! Aromatic Proximity Quantification Analysis
//!
//! This module analyzes the spatial relationship between cryptic sites
//! and UV-absorbing chromophores (aromatic residues and disulfide bonds).
//!
//! # Distance Bins
//!
//! Sites are categorized by proximity to chromophores:
//! - **Direct Contact** (< 3Å): van der Waals contact
//! - **Close Proximity** (3-5Å): First hydration shell
//! - **Medium Range** (5-8Å): Second shell / π-stacking range
//! - **Distal** (> 8Å): Beyond direct UV influence
//!
//! # Chromophore Types
//!
//! - **TRP**: Tryptophan (strongest UV absorber, 280nm)
//! - **TYR**: Tyrosine (moderate UV absorber, 274nm)
//! - **PHE**: Phenylalanine (weak UV absorber, 258nm)
//! - **S-S**: Disulfide bond (σ→σ* at 250nm)
//!
//! # Output
//!
//! The analysis produces:
//! - Distance distribution histograms
//! - Chromophore-type breakdown
//! - Spike correlation by proximity
//! - Publication-ready summary statistics

use crate::uv_bias::{
    AromaticTarget, ChromophoreType, DisulfideTarget, SpikeCategory, WavelengthAwareSpike,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// DISTANCE BINS
// =============================================================================

/// Distance bin categories for proximity analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProximityBin {
    /// < 3Å: Direct van der Waals contact
    DirectContact,
    /// 3-5Å: First hydration shell
    CloseProximity,
    /// 5-8Å: Second shell / π-stacking range
    MediumRange,
    /// > 8Å: Beyond direct UV influence
    Distal,
}

impl ProximityBin {
    /// Get bin from distance
    pub fn from_distance(distance: f32) -> Self {
        if distance < 3.0 {
            ProximityBin::DirectContact
        } else if distance < 5.0 {
            ProximityBin::CloseProximity
        } else if distance < 8.0 {
            ProximityBin::MediumRange
        } else {
            ProximityBin::Distal
        }
    }

    /// Get distance range for this bin
    pub fn range(&self) -> (f32, f32) {
        match self {
            ProximityBin::DirectContact => (0.0, 3.0),
            ProximityBin::CloseProximity => (3.0, 5.0),
            ProximityBin::MediumRange => (5.0, 8.0),
            ProximityBin::Distal => (8.0, f32::INFINITY),
        }
    }

    /// Human-readable label
    pub fn label(&self) -> &'static str {
        match self {
            ProximityBin::DirectContact => "Direct Contact (<3Å)",
            ProximityBin::CloseProximity => "Close Proximity (3-5Å)",
            ProximityBin::MediumRange => "Medium Range (5-8Å)",
            ProximityBin::Distal => "Distal (>8Å)",
        }
    }

    /// All bins in order
    pub fn all() -> &'static [ProximityBin] {
        &[
            ProximityBin::DirectContact,
            ProximityBin::CloseProximity,
            ProximityBin::MediumRange,
            ProximityBin::Distal,
        ]
    }
}

// =============================================================================
// CRYPTIC SITE STRUCTURE
// =============================================================================

/// A detected cryptic site for proximity analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrypticSite {
    /// Site identifier
    pub site_id: usize,
    /// Center position (Å)
    pub center: [f32; 3],
    /// Volume (Å³)
    pub volume: f32,
    /// Residues involved
    pub residues: Vec<usize>,
    /// Spike count associated with this site
    pub spike_count: usize,
    /// Weighted spike score
    pub weighted_score: f32,
    /// Is this site classified as cryptic?
    pub is_cryptic: bool,
}

// =============================================================================
// PROXIMITY ANALYSIS RESULTS
// =============================================================================

/// Results of aromatic proximity analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AromaticProximityAnalysis {
    /// Total sites analyzed
    pub total_sites: usize,

    /// Sites per proximity bin
    pub sites_by_bin: HashMap<String, usize>,

    /// Sites per chromophore type
    pub sites_by_chromophore: HashMap<String, usize>,

    /// Cross-tabulation: bin × chromophore type
    pub bin_chromophore_matrix: HashMap<String, HashMap<String, usize>>,

    /// Spike correlation by proximity bin
    pub spike_correlation_by_bin: HashMap<String, f32>,

    /// Average distance to nearest chromophore per site
    pub average_chromophore_distance: f32,

    /// Fraction of sites within direct UV influence (< 8Å)
    pub fraction_uv_influenced: f32,

    /// Per-site detailed analysis
    pub site_analyses: Vec<SiteProximityResult>,

    /// Summary statistics
    pub summary: ProximitySummary,
}

/// Proximity analysis for a single site
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiteProximityResult {
    /// Site ID
    pub site_id: usize,
    /// Nearest chromophore distance (Å)
    pub nearest_distance: f32,
    /// Nearest chromophore type
    pub nearest_type: String,
    /// Proximity bin
    pub bin: String,
    /// All chromophores within 8Å
    pub nearby_chromophores: Vec<NearbyChromophore>,
    /// Spike count
    pub spike_count: usize,
    /// UV-induced spike fraction
    pub uv_spike_fraction: f32,
}

/// A chromophore near a cryptic site
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NearbyChromophore {
    /// Chromophore type
    pub chromophore_type: String,
    /// Residue index
    pub residue_idx: usize,
    /// Distance to site center (Å)
    pub distance: f32,
    /// λmax (nm)
    pub lambda_max: f32,
    /// Extinction coefficient
    pub extinction: f32,
}

/// Summary statistics for proximity analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProximitySummary {
    /// Mean distance to nearest TRP
    pub mean_trp_distance: f32,
    /// Mean distance to nearest TYR
    pub mean_tyr_distance: f32,
    /// Mean distance to nearest PHE
    pub mean_phe_distance: f32,
    /// Mean distance to nearest disulfide
    pub mean_disulfide_distance: Option<f32>,

    /// Percentage of sites with TRP within 5Å
    pub pct_trp_close: f32,
    /// Percentage of sites with TYR within 5Å
    pub pct_tyr_close: f32,
    /// Percentage of sites with any aromatic within 5Å
    pub pct_any_aromatic_close: f32,

    /// Correlation: proximity → spike rate
    pub proximity_spike_correlation: f32,
    /// Best predictor chromophore type
    pub best_predictor: String,
    /// Best predictor correlation
    pub best_predictor_correlation: f32,
}

// =============================================================================
// PROXIMITY ANALYZER
// =============================================================================

/// Aromatic proximity analyzer
pub struct AromaticProximityAnalyzer {
    /// Aromatic targets
    aromatics: Vec<AromaticTarget>,
    /// Disulfide targets
    disulfides: Vec<DisulfideTarget>,
    /// Results
    results: Option<AromaticProximityAnalysis>,
}

impl AromaticProximityAnalyzer {
    /// Create new analyzer with chromophore targets
    pub fn new(aromatics: Vec<AromaticTarget>, disulfides: Vec<DisulfideTarget>) -> Self {
        Self {
            aromatics,
            disulfides,
            results: None,
        }
    }

    /// Analyze proximity of cryptic sites to chromophores
    pub fn analyze(&mut self, sites: &[CrypticSite]) -> AromaticProximityAnalysis {
        let mut sites_by_bin: HashMap<String, usize> = HashMap::new();
        let mut sites_by_chromophore: HashMap<String, usize> = HashMap::new();
        let mut bin_chromophore_matrix: HashMap<String, HashMap<String, usize>> = HashMap::new();
        let mut site_analyses = Vec::with_capacity(sites.len());

        let mut total_distance = 0.0f32;
        let mut uv_influenced_count = 0usize;

        // Initialize bins
        for bin in ProximityBin::all() {
            sites_by_bin.insert(bin.label().to_string(), 0);
            bin_chromophore_matrix.insert(bin.label().to_string(), HashMap::new());
        }

        // Analyze each site
        for site in sites {
            let analysis = self.analyze_site(site);

            // Update bin counts
            *sites_by_bin.entry(analysis.bin.clone()).or_insert(0) += 1;

            // Update chromophore counts
            *sites_by_chromophore
                .entry(analysis.nearest_type.clone())
                .or_insert(0) += 1;

            // Update matrix
            bin_chromophore_matrix
                .entry(analysis.bin.clone())
                .or_default()
                .entry(analysis.nearest_type.clone())
                .and_modify(|c| *c += 1)
                .or_insert(1);

            total_distance += analysis.nearest_distance;

            if analysis.nearest_distance < 8.0 {
                uv_influenced_count += 1;
            }

            site_analyses.push(analysis);
        }

        let n = sites.len() as f32;
        let average_chromophore_distance = if n > 0.0 { total_distance / n } else { 0.0 };
        let fraction_uv_influenced = if n > 0.0 {
            uv_influenced_count as f32 / n
        } else {
            0.0
        };

        // Compute spike correlations
        let spike_correlation_by_bin = self.compute_spike_correlations(&site_analyses);

        // Compute summary
        let summary = self.compute_summary(&site_analyses);

        let analysis = AromaticProximityAnalysis {
            total_sites: sites.len(),
            sites_by_bin,
            sites_by_chromophore,
            bin_chromophore_matrix,
            spike_correlation_by_bin,
            average_chromophore_distance,
            fraction_uv_influenced,
            site_analyses,
            summary,
        };

        self.results = Some(analysis.clone());
        analysis
    }

    /// Analyze a single site
    fn analyze_site(&self, site: &CrypticSite) -> SiteProximityResult {
        let mut nearest_distance = f32::MAX;
        let mut nearest_type = "None".to_string();
        let mut nearby_chromophores = Vec::new();

        // Check aromatic residues
        for aromatic in &self.aromatics {
            let distance = self.compute_distance(&site.center, &aromatic.ring_center);

            if distance < nearest_distance {
                nearest_distance = distance;
                nearest_type = aromatic.residue_type.code().to_string();
            }

            if distance < 8.0 {
                nearby_chromophores.push(NearbyChromophore {
                    chromophore_type: aromatic.residue_type.code().to_string(),
                    residue_idx: aromatic.residue_idx,
                    distance,
                    lambda_max: aromatic.residue_type.lambda_max(),
                    extinction: aromatic.residue_type.epsilon_max(),
                });
            }
        }

        // Check disulfide bonds
        for disulfide in &self.disulfides {
            let distance = self.compute_distance(&site.center, &disulfide.midpoint);

            if distance < nearest_distance {
                nearest_distance = distance;
                nearest_type = "S-S".to_string();
            }

            if distance < 8.0 {
                nearby_chromophores.push(NearbyChromophore {
                    chromophore_type: "S-S".to_string(),
                    residue_idx: disulfide.cys1_residue_idx,
                    distance,
                    lambda_max: ChromophoreType::Disulfide.lambda_max(),
                    extinction: ChromophoreType::Disulfide.epsilon_max(),
                });
            }
        }

        // Sort by distance
        nearby_chromophores.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

        let bin = ProximityBin::from_distance(nearest_distance);

        SiteProximityResult {
            site_id: site.site_id,
            nearest_distance,
            nearest_type,
            bin: bin.label().to_string(),
            nearby_chromophores,
            spike_count: site.spike_count,
            uv_spike_fraction: 0.0, // Updated later if spike data available
        }
    }

    /// Compute distance between two points
    fn compute_distance(&self, p1: &[f32; 3], p2: &[f32; 3]) -> f32 {
        let dx = p1[0] - p2[0];
        let dy = p1[1] - p2[1];
        let dz = p1[2] - p2[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Compute spike rate correlation with proximity
    fn compute_spike_correlations(&self, analyses: &[SiteProximityResult]) -> HashMap<String, f32> {
        let mut correlations = HashMap::new();

        for bin in ProximityBin::all() {
            let bin_sites: Vec<_> = analyses.iter().filter(|a| a.bin == bin.label()).collect();

            if bin_sites.is_empty() {
                correlations.insert(bin.label().to_string(), 0.0);
                continue;
            }

            let avg_spikes: f32 = bin_sites.iter().map(|a| a.spike_count as f32).sum::<f32>()
                / bin_sites.len() as f32;

            correlations.insert(bin.label().to_string(), avg_spikes);
        }

        correlations
    }

    /// Compute summary statistics
    fn compute_summary(&self, analyses: &[SiteProximityResult]) -> ProximitySummary {
        let mut trp_distances = Vec::new();
        let mut tyr_distances = Vec::new();
        let mut phe_distances = Vec::new();
        let mut ss_distances = Vec::new();

        let mut trp_close = 0usize;
        let mut tyr_close = 0usize;
        let mut any_close = 0usize;

        for analysis in analyses {
            // Find distances by type
            let mut min_trp = f32::MAX;
            let mut min_tyr = f32::MAX;
            let mut min_phe = f32::MAX;
            let mut min_ss = f32::MAX;

            for chromophore in &analysis.nearby_chromophores {
                match chromophore.chromophore_type.as_str() {
                    "TRP" => min_trp = min_trp.min(chromophore.distance),
                    "TYR" => min_tyr = min_tyr.min(chromophore.distance),
                    "PHE" => min_phe = min_phe.min(chromophore.distance),
                    "S-S" => min_ss = min_ss.min(chromophore.distance),
                    _ => {}
                }
            }

            // Also check aromatics directly for all-distances
            for aromatic in &self.aromatics {
                let dist = ((analysis.nearest_distance - aromatic.ring_center[0]).powi(2)
                    + (analysis.nearest_distance - aromatic.ring_center[1]).powi(2)
                    + (analysis.nearest_distance - aromatic.ring_center[2]).powi(2))
                .sqrt();

                match aromatic.residue_type {
                    ChromophoreType::Tryptophan => min_trp = min_trp.min(dist),
                    ChromophoreType::Tyrosine => min_tyr = min_tyr.min(dist),
                    ChromophoreType::Phenylalanine => min_phe = min_phe.min(dist),
                    _ => {}
                }
            }

            if min_trp < f32::MAX {
                trp_distances.push(min_trp);
            }
            if min_tyr < f32::MAX {
                tyr_distances.push(min_tyr);
            }
            if min_phe < f32::MAX {
                phe_distances.push(min_phe);
            }
            if min_ss < f32::MAX {
                ss_distances.push(min_ss);
            }

            if min_trp < 5.0 {
                trp_close += 1;
            }
            if min_tyr < 5.0 {
                tyr_close += 1;
            }
            if analysis.nearest_distance < 5.0 {
                any_close += 1;
            }
        }

        let n = analyses.len() as f32;

        let mean_trp_distance = if trp_distances.is_empty() {
            f32::MAX
        } else {
            trp_distances.iter().sum::<f32>() / trp_distances.len() as f32
        };

        let mean_tyr_distance = if tyr_distances.is_empty() {
            f32::MAX
        } else {
            tyr_distances.iter().sum::<f32>() / tyr_distances.len() as f32
        };

        let mean_phe_distance = if phe_distances.is_empty() {
            f32::MAX
        } else {
            phe_distances.iter().sum::<f32>() / phe_distances.len() as f32
        };

        let mean_disulfide_distance = if ss_distances.is_empty() {
            None
        } else {
            Some(ss_distances.iter().sum::<f32>() / ss_distances.len() as f32)
        };

        let pct_trp_close = if n > 0.0 {
            trp_close as f32 / n * 100.0
        } else {
            0.0
        };
        let pct_tyr_close = if n > 0.0 {
            tyr_close as f32 / n * 100.0
        } else {
            0.0
        };
        let pct_any_aromatic_close = if n > 0.0 {
            any_close as f32 / n * 100.0
        } else {
            0.0
        };

        // Simple correlation: negative distance → spike count relationship
        let proximity_spike_correlation = self.compute_proximity_spike_correlation(analyses);

        // Find best predictor
        let (best_predictor, best_correlation) = self.find_best_predictor(analyses);

        ProximitySummary {
            mean_trp_distance,
            mean_tyr_distance,
            mean_phe_distance,
            mean_disulfide_distance,
            pct_trp_close,
            pct_tyr_close,
            pct_any_aromatic_close,
            proximity_spike_correlation,
            best_predictor,
            best_predictor_correlation: best_correlation,
        }
    }

    /// Compute Pearson correlation between proximity and spike rate
    fn compute_proximity_spike_correlation(&self, analyses: &[SiteProximityResult]) -> f32 {
        if analyses.len() < 2 {
            return 0.0;
        }

        let distances: Vec<f32> = analyses.iter().map(|a| a.nearest_distance).collect();
        let spikes: Vec<f32> = analyses.iter().map(|a| a.spike_count as f32).collect();

        pearson_correlation(&distances, &spikes)
    }

    /// Find which chromophore type best predicts spike activity
    fn find_best_predictor(&self, analyses: &[SiteProximityResult]) -> (String, f32) {
        let types = ["TRP", "TYR", "PHE", "S-S"];
        let mut best = ("None".to_string(), 0.0f32);

        for type_name in &types {
            let correlation = self.compute_type_spike_correlation(analyses, type_name);
            if correlation.abs() > best.1.abs() {
                best = (type_name.to_string(), correlation);
            }
        }

        best
    }

    /// Compute correlation between specific chromophore type distance and spikes
    fn compute_type_spike_correlation(
        &self,
        analyses: &[SiteProximityResult],
        chromophore_type: &str,
    ) -> f32 {
        let mut distances = Vec::new();
        let mut spikes = Vec::new();

        for analysis in analyses {
            let type_dist = analysis
                .nearby_chromophores
                .iter()
                .filter(|c| c.chromophore_type == chromophore_type)
                .map(|c| c.distance)
                .min_by(|a, b| a.partial_cmp(b).unwrap());

            if let Some(dist) = type_dist {
                distances.push(dist);
                spikes.push(analysis.spike_count as f32);
            }
        }

        if distances.len() < 2 {
            return 0.0;
        }

        pearson_correlation(&distances, &spikes)
    }

    /// Get analysis results
    pub fn get_results(&self) -> Option<&AromaticProximityAnalysis> {
        self.results.as_ref()
    }

    /// Generate publication-ready report
    pub fn generate_report(&self) -> String {
        let results = match &self.results {
            Some(r) => r,
            None => return "No analysis performed yet.".to_string(),
        };

        let mut report = String::new();

        report.push_str("# Aromatic Proximity Quantification Report\n\n");

        report.push_str("## Overview\n\n");
        report.push_str(&format!(
            "- Total sites analyzed: {}\n",
            results.total_sites
        ));
        report.push_str(&format!(
            "- Average chromophore distance: {:.2} Å\n",
            results.average_chromophore_distance
        ));
        report.push_str(&format!(
            "- Fraction UV-influenced (<8Å): {:.1}%\n\n",
            results.fraction_uv_influenced * 100.0
        ));

        report.push_str("## Sites by Proximity Bin\n\n");
        report.push_str("| Bin | Count | Avg Spikes |\n");
        report.push_str("|-----|-------|------------|\n");
        for bin in ProximityBin::all() {
            let count = results.sites_by_bin.get(bin.label()).unwrap_or(&0);
            let spikes = results
                .spike_correlation_by_bin
                .get(bin.label())
                .unwrap_or(&0.0);
            report.push_str(&format!(
                "| {} | {} | {:.1} |\n",
                bin.label(),
                count,
                spikes
            ));
        }
        report.push('\n');

        report.push_str("## Sites by Chromophore Type\n\n");
        report.push_str("| Type | Count | % of Total |\n");
        report.push_str("|------|-------|------------|\n");
        for (type_name, count) in &results.sites_by_chromophore {
            let pct = *count as f32 / results.total_sites as f32 * 100.0;
            report.push_str(&format!("| {} | {} | {:.1}% |\n", type_name, count, pct));
        }
        report.push('\n');

        report.push_str("## Summary Statistics\n\n");
        let s = &results.summary;
        report.push_str(&format!(
            "- Mean TRP distance: {:.2} Å\n",
            s.mean_trp_distance
        ));
        report.push_str(&format!(
            "- Mean TYR distance: {:.2} Å\n",
            s.mean_tyr_distance
        ));
        report.push_str(&format!(
            "- Mean PHE distance: {:.2} Å\n",
            s.mean_phe_distance
        ));
        if let Some(ss_dist) = s.mean_disulfide_distance {
            report.push_str(&format!("- Mean S-S distance: {:.2} Å\n", ss_dist));
        }
        report.push_str(&format!("- % with TRP < 5Å: {:.1}%\n", s.pct_trp_close));
        report.push_str(&format!("- % with TYR < 5Å: {:.1}%\n", s.pct_tyr_close));
        report.push_str(&format!(
            "- % with any aromatic < 5Å: {:.1}%\n",
            s.pct_any_aromatic_close
        ));
        report.push_str(&format!(
            "- Proximity-spike correlation: {:.3}\n",
            s.proximity_spike_correlation
        ));
        report.push_str(&format!(
            "- Best predictor: {} (r = {:.3})\n",
            s.best_predictor, s.best_predictor_correlation
        ));

        report
    }
}

/// Compute Pearson correlation coefficient
fn pearson_correlation(x: &[f32], y: &[f32]) -> f32 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }

    let n = x.len() as f32;
    let sum_x: f32 = x.iter().sum();
    let sum_y: f32 = y.iter().sum();
    let sum_xx: f32 = x.iter().map(|v| v * v).sum();
    let sum_yy: f32 = y.iter().map(|v| v * v).sum();
    let sum_xy: f32 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();

    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator = ((n * sum_xx - sum_x * sum_x) * (n * sum_yy - sum_y * sum_y)).sqrt();

    if denominator < 1e-10 {
        0.0
    } else {
        numerator / denominator
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proximity_bins() {
        assert_eq!(
            ProximityBin::from_distance(1.5),
            ProximityBin::DirectContact
        );
        assert_eq!(
            ProximityBin::from_distance(4.0),
            ProximityBin::CloseProximity
        );
        assert_eq!(ProximityBin::from_distance(6.5), ProximityBin::MediumRange);
        assert_eq!(ProximityBin::from_distance(10.0), ProximityBin::Distal);
    }

    #[test]
    fn test_pearson_correlation() {
        // Perfect positive correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 0.001, "Expected r=1.0, got {}", r);

        // Perfect negative correlation
        let y_neg = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let r_neg = pearson_correlation(&x, &y_neg);
        assert!(
            (r_neg + 1.0).abs() < 0.001,
            "Expected r=-1.0, got {}",
            r_neg
        );

        // No correlation (constant y)
        let y_const = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let r_const = pearson_correlation(&x, &y_const);
        assert!(r_const.abs() < 0.001, "Expected r=0, got {}", r_const);
    }

    #[test]
    fn test_analyzer_creation() {
        let aromatics = vec![];
        let disulfides = vec![];
        let analyzer = AromaticProximityAnalyzer::new(aromatics, disulfides);
        assert!(analyzer.get_results().is_none());
    }

    #[test]
    fn test_empty_analysis() {
        let mut analyzer = AromaticProximityAnalyzer::new(vec![], vec![]);
        let sites: Vec<CrypticSite> = vec![];
        let results = analyzer.analyze(&sites);

        assert_eq!(results.total_sites, 0);
        assert_eq!(results.fraction_uv_influenced, 0.0);
    }
}
