//! Structural Attention Mechanism
//!
//! Learned attention weights over protein residues based on:
//! - ACE2 interface proximity
//! - Epitope class membership
//! - Escape mutation hotspots
//! - Reservoir state magnitude

use anyhow::Result;

/// ACE2 interface residue positions (from 6M0J structure)
/// These are RBD residues that directly contact the ACE2 receptor
pub const ACE2_INTERFACE_RESIDUES: &[usize] = &[
    417, 446, 449, 453, 455, 456, 475, 476, 484, 486, 487, 489, 490, 493, 494, 496, 498, 500, 501,
    502, 503, 505, 506, 520, 521,
];

/// Epitope class centers (representative residue for each of 10 antibody classes)
/// Based on structural clustering of neutralizing antibody binding sites
pub const EPITOPE_CLASS_CENTERS: &[(usize, &str)] = &[
    (484, "Class 1 - RBM core"),
    (501, "Class 2 - RBM peripheral"),
    (417, "Class 3 - N-terminal"),
    (346, "Class 4 - Outer face"),
    (440, "Class 5 - Loop region"),
    (498, "Class 6 - ACE2 overlap"),
    (373, "Class 7 - Cryptic"),
    (456, "Class 8 - Interface"),
    (384, "Class 9 - S2 proximal"),
    (527, "Class 10 - C-terminal"),
];

/// Known escape mutation hotspots (high DMS escape scores)
pub const ESCAPE_HOTSPOTS: &[usize] = &[
    484, // E484K/Q - Major escape
    501, // N501Y - Enhanced binding + escape
    417, // K417N/T - Escape in Beta/Gamma
    452, // L452R - Delta escape
    478, // T478K - Delta escape
    486, // F486V - Omicron escape
    493, // Q493R - Omicron escape
    498, // Q498R - Omicron escape
    346, // R346K - BA.2.75 escape
    444, // K444T - BQ.1 escape
];

/// Attention configuration
#[derive(Clone, Debug)]
pub struct AttentionConfig {
    /// Weight for interface proximity
    pub interface_weight: f32,
    /// Weight for epitope proximity
    pub epitope_weight: f32,
    /// Weight for escape hotspot
    pub hotspot_weight: f32,
    /// Weight for reservoir state magnitude
    pub reservoir_weight: f32,
    /// Weight for degree centrality
    pub centrality_weight: f32,
    /// Softmax temperature (lower = more focused)
    pub temperature: f32,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            interface_weight: 0.30,
            epitope_weight: 0.20,
            hotspot_weight: 0.20,
            reservoir_weight: 0.15,
            centrality_weight: 0.15,
            temperature: 1.0,
        }
    }
}

/// Check if residue is at ACE2 interface
pub fn is_interface_residue(residue_pos: usize) -> bool {
    ACE2_INTERFACE_RESIDUES.contains(&residue_pos)
}

/// Compute distance to nearest interface residue
pub fn interface_distance(residue_pos: usize) -> f32 {
    ACE2_INTERFACE_RESIDUES
        .iter()
        .map(|&r| (residue_pos as i32 - r as i32).abs() as f32)
        .fold(f32::MAX, f32::min)
}

/// Compute distance to nearest epitope center
pub fn epitope_distance(residue_pos: usize) -> f32 {
    EPITOPE_CLASS_CENTERS
        .iter()
        .map(|&(r, _)| (residue_pos as i32 - r as i32).abs() as f32)
        .fold(f32::MAX, f32::min)
}

/// Check if residue is an escape hotspot
pub fn is_escape_hotspot(residue_pos: usize) -> bool {
    ESCAPE_HOTSPOTS.contains(&residue_pos)
}

/// Compute attention weight for a single residue
pub fn compute_residue_attention(
    residue_pos: usize,
    reservoir_magnitude: f32,
    degree_centrality: f32,
    config: &AttentionConfig,
) -> f32 {
    // Interface proximity score
    let interface_score = if is_interface_residue(residue_pos) {
        2.0
    } else {
        let dist = interface_distance(residue_pos);
        (-dist / 10.0).exp()
    };

    // Epitope proximity score
    let epitope_dist = epitope_distance(residue_pos);
    let epitope_score = (-epitope_dist / 10.0).exp();

    // Escape hotspot score
    let hotspot_score = if is_escape_hotspot(residue_pos) {
        2.0
    } else {
        0.0
    };

    // Combine scores
    let raw_attention = config.interface_weight * interface_score
        + config.epitope_weight * epitope_score
        + config.hotspot_weight * hotspot_score
        + config.reservoir_weight * reservoir_magnitude
        + config.centrality_weight * degree_centrality;

    raw_attention
}

/// Compute attention weights for all residues
pub fn compute_attention_weights(
    residue_positions: &[usize],
    reservoir_magnitudes: &[f32],
    degree_centralities: &[f32],
    config: &AttentionConfig,
) -> Vec<f32> {
    let n = residue_positions.len();
    assert_eq!(n, reservoir_magnitudes.len());
    assert_eq!(n, degree_centralities.len());

    // Compute raw attention scores
    let raw_scores: Vec<f32> = (0..n)
        .map(|i| {
            compute_residue_attention(
                residue_positions[i],
                reservoir_magnitudes[i],
                degree_centralities[i],
                config,
            )
        })
        .collect();

    // Softmax normalization
    let max_score = raw_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_scores: Vec<f32> = raw_scores
        .iter()
        .map(|s| ((s - max_score) / config.temperature).exp())
        .collect();
    let sum_exp: f32 = exp_scores.iter().sum();

    exp_scores.iter().map(|e| e / sum_exp.max(1e-6)).collect()
}

/// Apply attention to features
pub fn apply_attention(
    features: &[f32],  // [N_residues x 136]
    attention: &[f32], // [N_residues]
    n_residues: usize,
) -> Vec<f32> {
    assert_eq!(features.len(), n_residues * 136);
    assert_eq!(attention.len(), n_residues);

    let mut attended = vec![0.0f32; 136];

    for (r, &weight) in attention.iter().enumerate() {
        for f in 0..136 {
            attended[f] += weight * features[r * 136 + f];
        }
    }

    attended
}

/// Get epitope class for a residue
pub fn get_epitope_class(residue_pos: usize) -> Option<usize> {
    let (nearest_class, min_dist) = EPITOPE_CLASS_CENTERS
        .iter()
        .enumerate()
        .map(|(i, &(center, _))| (i, (residue_pos as i32 - center as i32).abs()))
        .min_by_key(|&(_, d)| d)?;

    // Only assign if within 15 residues of center
    if min_dist <= 15 {
        Some(nearest_class)
    } else {
        None
    }
}

/// Compute per-epitope escape scores
pub fn compute_epitope_escape_scores(
    mutations: &[(usize, char, char)], // (position, wt, mut)
    per_site_escape: &[f32],           // [N_sites] escape score per site
    rbd_start: usize,                  // First RBD residue number
) -> [f32; 10] {
    let mut epitope_scores = [0.0f32; 10];
    let mut epitope_counts = [0usize; 10];

    for &(pos, _wt, _mut) in mutations {
        // Get escape score for this position
        let site_idx = pos.saturating_sub(rbd_start);
        if site_idx < per_site_escape.len() {
            let escape = per_site_escape[site_idx];

            // Assign to epitope class
            if let Some(epitope) = get_epitope_class(pos) {
                epitope_scores[epitope] += escape;
                epitope_counts[epitope] += 1;
            }
        }
    }

    // Normalize by count
    for (score, count) in epitope_scores.iter_mut().zip(epitope_counts.iter()) {
        if *count > 0 {
            *score /= *count as f32;
        }
    }

    epitope_scores
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interface_residue() {
        assert!(is_interface_residue(484));
        assert!(is_interface_residue(501));
        assert!(!is_interface_residue(350));
    }

    #[test]
    fn test_interface_distance() {
        let dist = interface_distance(484);
        assert_eq!(dist, 0.0);

        let dist = interface_distance(480);
        assert!(dist < 10.0);
    }

    #[test]
    fn test_escape_hotspot() {
        assert!(is_escape_hotspot(484));
        assert!(is_escape_hotspot(501));
        assert!(!is_escape_hotspot(400));
    }

    #[test]
    fn test_attention_weights() {
        let config = AttentionConfig::default();
        let positions = vec![484, 501, 350];
        let magnitudes = vec![0.5, 0.5, 0.5];
        let centralities = vec![0.3, 0.3, 0.3];

        let weights = compute_attention_weights(&positions, &magnitudes, &centralities, &config);

        assert_eq!(weights.len(), 3);
        // Interface/hotspot residues should have higher attention
        assert!(weights[0] > weights[2]);
        assert!(weights[1] > weights[2]);
        // Sum should be ~1
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_epitope_class() {
        assert_eq!(get_epitope_class(484), Some(0)); // Class 1
        assert_eq!(get_epitope_class(501), Some(1)); // Class 2
        assert_eq!(get_epitope_class(417), Some(2)); // Class 3
    }
}
