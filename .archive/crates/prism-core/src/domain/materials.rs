//! Materials Adapter
//!
//! Provides material property prediction and inverse design for
//! materials discovery workflows.

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Materials state for PhaseContext tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialsState {
    /// Predicted band gap (eV)
    pub band_gap: f64,
    /// Formation energy (eV/atom)
    pub formation_energy: f64,
    /// Stability score (0.0-1.0, higher is more stable)
    pub stability: f64,
    /// Material composition string (e.g., "Li2FePO4")
    pub composition: String,
    /// Synthesis confidence (0.0-1.0, higher = more feasible)
    pub synthesis_confidence: f64,
}

/// Target material properties for inverse design
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetProperties {
    /// Target band gap range (min, max) in eV
    pub band_gap_range: (f64, f64),
    /// Maximum formation energy (eV/atom, lower is more stable)
    pub max_formation_energy: f64,
    /// Minimum stability score (0.0-1.0)
    pub min_stability: f64,
    /// Required elements (e.g., ["Li", "Fe", "P", "O"])
    pub required_elements: Vec<String>,
    /// Forbidden elements (e.g., ["Hg", "Pb"])
    pub forbidden_elements: Vec<String>,
}

impl Default for TargetProperties {
    fn default() -> Self {
        Self {
            band_gap_range: (1.0, 3.0), // Semiconductor range
            max_formation_energy: -1.0, // Stable compounds
            min_stability: 0.7,         // Reasonably stable
            required_elements: vec![],
            forbidden_elements: vec!["Hg".to_string(), "Pb".to_string()], // Toxic elements
        }
    }
}

/// Discovered material candidate
#[derive(Debug, Clone)]
pub struct MaterialCandidate {
    /// Material composition
    pub composition: String,
    /// Predicted properties
    pub properties: MaterialProperties,
    /// Design confidence score
    pub confidence: f64,
}

/// Predicted material properties
#[derive(Debug, Clone)]
pub struct MaterialProperties {
    /// Band gap (eV)
    pub band_gap: f64,
    /// Formation energy (eV/atom)
    pub formation_energy: f64,
    /// Stability score (0.0-1.0)
    pub stability: f64,
    /// Elastic modulus (GPa)
    pub elastic_modulus: f64,
    /// Thermal conductivity (W/m·K)
    pub thermal_conductivity: f64,
}

/// Materials Adapter for inverse design and property prediction
pub struct MaterialsAdapter {
    /// Configuration
    config: MaterialsConfig,
}

/// Configuration for materials workflows
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialsConfig {
    /// Number of candidate materials to generate
    pub num_candidates: usize,
    /// Enable GPU acceleration (if available)
    pub use_gpu: bool,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for MaterialsConfig {
    fn default() -> Self {
        Self {
            num_candidates: 10,
            use_gpu: false,
            seed: 42,
        }
    }
}

impl MaterialsAdapter {
    /// Create a new materials adapter
    pub fn new(config: MaterialsConfig) -> Self {
        Self { config }
    }

    /// Discover materials matching target properties
    ///
    /// # Arguments
    /// * `target` - Target material properties and constraints
    ///
    /// # Returns
    /// List of candidate materials ranked by fitness to target
    ///
    /// # Note
    /// This is a placeholder implementation. In production, this would integrate with:
    /// - CGCNN (Crystal Graph Convolutional Neural Networks)
    /// - MEGNet (MatErials Graph Network)
    /// - ALIGNN (Atomistic Line Graph Neural Network)
    /// - Materials Project API for validation
    pub fn discover_material(&self, target: &TargetProperties) -> Result<Vec<MaterialCandidate>> {
        log::info!("MaterialsAdapter: Discovering materials");
        log::info!(
            "  Target band gap: {:.2}-{:.2} eV",
            target.band_gap_range.0,
            target.band_gap_range.1
        );
        log::info!(
            "  Max formation energy: {:.2} eV/atom",
            target.max_formation_energy
        );
        log::info!("  Min stability: {:.2}", target.min_stability);

        // Simulate material discovery
        // In production: use generative models, graph neural networks, or database screening
        let candidates = (0..self.config.num_candidates)
            .map(|i| {
                // Generate pseudo-random compositions
                let composition = self.generate_composition(i, target);

                // Predict properties
                let properties = self.predict_properties(&composition);

                // Compute fitness to target
                let confidence = self.compute_fitness(&properties, target);

                MaterialCandidate {
                    composition,
                    properties,
                    confidence,
                }
            })
            .collect::<Vec<_>>();

        // Sort by confidence (higher is better)
        let mut sorted_candidates = candidates;
        sorted_candidates.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        log::info!(
            "  Generated {} candidate materials",
            sorted_candidates.len()
        );
        if !sorted_candidates.is_empty() {
            let best = &sorted_candidates[0];
            log::info!(
                "  Best candidate: {} (confidence={:.2}, band_gap={:.2} eV)",
                best.composition,
                best.confidence,
                best.properties.band_gap
            );
        }

        Ok(sorted_candidates)
    }

    /// Predict material properties from composition
    ///
    /// # Arguments
    /// * `composition` - Material composition string (e.g., "Li2FePO4")
    ///
    /// # Returns
    /// Predicted material properties
    ///
    /// # Note
    /// This is a placeholder. In production, this would use:
    /// - Trained graph neural networks (CGCNN, MEGNet, ALIGNN)
    /// - DFT calculations (VASP, Quantum ESPRESSO)
    /// - Materials Project API lookups
    pub fn predict_properties(&self, composition: &str) -> MaterialProperties {
        log::debug!("  Predicting properties for: {}", composition);

        // Simulate property prediction
        // In production: run inference through trained GNN or query Materials Project
        let hash = composition
            .bytes()
            .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));

        let band_gap = 1.0 + ((hash % 200) as f64 / 100.0); // 1.0-3.0 eV
        let formation_energy = -2.0 + ((hash % 150) as f64 / 100.0); // -2.0 to -0.5 eV/atom
        let stability = 0.6 + ((hash % 40) as f64 / 100.0); // 0.6-1.0
        let elastic_modulus = 50.0 + ((hash % 200) as f64); // 50-250 GPa
        let thermal_conductivity = 1.0 + ((hash % 50) as f64 / 10.0); // 1-6 W/m·K

        MaterialProperties {
            band_gap,
            formation_energy,
            stability,
            elastic_modulus,
            thermal_conductivity,
        }
    }

    /// Generate a material composition
    ///
    /// # Arguments
    /// * `index` - Candidate index for pseudo-random generation
    /// * `target` - Target properties to guide generation
    ///
    /// # Returns
    /// Material composition string
    fn generate_composition(&self, index: usize, target: &TargetProperties) -> String {
        // Simple pseudo-random composition generation
        // In production: use generative models or combinatorial libraries

        let elements = if !target.required_elements.is_empty() {
            target.required_elements.clone()
        } else {
            vec![
                "Li".to_string(),
                "Na".to_string(),
                "Fe".to_string(),
                "Co".to_string(),
                "Ni".to_string(),
                "Mn".to_string(),
                "P".to_string(),
                "O".to_string(),
                "S".to_string(),
                "Si".to_string(),
            ]
        };

        // Filter out forbidden elements
        let allowed_elements: Vec<String> = elements
            .into_iter()
            .filter(|e| !target.forbidden_elements.contains(e))
            .collect();

        if allowed_elements.is_empty() {
            return "Li2O".to_string(); // Fallback
        }

        // Generate pseudo-random stoichiometry
        let seed = (index as u64).wrapping_mul(self.config.seed);
        let elem1_idx = (seed % allowed_elements.len() as u64) as usize;
        let elem2_idx = ((seed / 10) % allowed_elements.len() as u64) as usize;

        let stoich1 = ((seed / 100) % 4) + 1; // 1-4
        let stoich2 = ((seed / 1000) % 8) + 1; // 1-8

        format!(
            "{}{}{}{}",
            allowed_elements[elem1_idx], stoich1, allowed_elements[elem2_idx], stoich2
        )
    }

    /// Compute fitness of properties to target
    ///
    /// # Arguments
    /// * `properties` - Predicted material properties
    /// * `target` - Target properties
    ///
    /// # Returns
    /// Fitness score (0.0-1.0, higher is better)
    fn compute_fitness(&self, properties: &MaterialProperties, target: &TargetProperties) -> f64 {
        // Band gap fitness
        let band_gap_fit = if properties.band_gap >= target.band_gap_range.0
            && properties.band_gap <= target.band_gap_range.1
        {
            1.0
        } else {
            let distance = if properties.band_gap < target.band_gap_range.0 {
                target.band_gap_range.0 - properties.band_gap
            } else {
                properties.band_gap - target.band_gap_range.1
            };
            (1.0 / (1.0 + distance)).max(0.0)
        };

        // Formation energy fitness (lower is better, so invert)
        let form_energy_fit = if properties.formation_energy <= target.max_formation_energy {
            1.0
        } else {
            (1.0 / (1.0 + (properties.formation_energy - target.max_formation_energy))).max(0.0)
        };

        // Stability fitness
        let stability_fit = if properties.stability >= target.min_stability {
            1.0
        } else {
            (properties.stability / target.min_stability).max(0.0)
        };

        // Weighted average
        (band_gap_fit * 0.4 + form_energy_fit * 0.3 + stability_fit * 0.3).min(1.0)
    }
}
