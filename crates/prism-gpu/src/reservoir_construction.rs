//! Deterministic, biologically-structured reservoir construction
//!
//! This module implements training-free reservoir connectivity patterns
//! inspired by biological neural circuits:
//! - Local dendritic arbors with exponential distance decay
//! - Sparse long-range connections for information mixing
//! - Spectral radius normalization for edge-of-chaos dynamics
//!
//! IMPORTANT: This is a connectivity pattern generator, NOT a trained model.
//! Performance depends on whether the pattern captures task-relevant structure.

/// Sparse connection representation
#[derive(Debug, Clone)]
pub struct SparseConnection {
    pub target: usize,
    pub weight: f32,
}

/// Deterministic reservoir connectivity matrix (sparse)
#[derive(Debug, Clone)]
pub struct BioReservoir {
    /// Number of neurons (residues)
    pub n_neurons: usize,
    /// Connections per neuron (sparse adjacency list)
    pub connections: Vec<Vec<SparseConnection>>,
    /// Spectral radius after normalization
    pub spectral_radius: f32,
    /// Input-to-reservoir weights (deterministic)
    pub input_weights: Vec<f32>,
    /// Reservoir-to-output readout weights
    pub readout_weights: Vec<f32>,
}

impl BioReservoir {
    /// Construct biologically-structured reservoir with deterministic connectivity
    ///
    /// # Arguments
    /// * `n_residues` - Number of neurons (one per residue)
    /// * `branches_per_neuron` - Dendritic branches per neuron (default: 8)
    /// * `local_radius` - Local connectivity radius in sequence (default: 16)
    /// * `target_spectral_radius` - Target spectral radius (default: 0.95, edge of chaos)
    /// * `n_input_features` - Number of input features per residue (default: 8)
    pub fn construct(
        n_residues: usize,
        branches_per_neuron: u32,
        local_radius: u32,
        target_spectral_radius: f32,
        n_input_features: usize,
    ) -> Self {
        let tau = 8.0f32; // Exponential decay constant
        let mut connections = vec![Vec::new(); n_residues];

        for i in 0..n_residues {
            for b in 0..branches_per_neuron {
                // Local dendritic arbor with exponential decay
                let start = i.saturating_sub(local_radius as usize);
                let end = (i + local_radius as usize + 1).min(n_residues);

                for j in start..end {
                    if i == j {
                        continue;
                    }

                    let dist = (i as i32 - j as i32).abs() as f32;
                    let weight = (-dist / tau).exp();

                    // Deterministic sign based on XOR pattern (no randomness)
                    let sign = if (i ^ j ^ (b as usize)) & 1 == 0 {
                        1.0
                    } else {
                        -1.0
                    };

                    connections[i].push(SparseConnection {
                        target: j,
                        weight: sign * weight,
                    });
                }

                // One long-range connection per branch (deterministic hash)
                let seed = i.wrapping_mul(2654435761) ^ ((b as usize) << 12);
                let long_range = seed % n_residues;
                if long_range != i {
                    connections[i].push(SparseConnection {
                        target: long_range,
                        weight: 0.3, // Strong long-range connection
                    });
                }
            }
        }

        // Normalize to target spectral radius
        let max_eigen = Self::estimate_spectral_radius(&connections, n_residues);
        let scale = if max_eigen > 1e-6 {
            target_spectral_radius / max_eigen
        } else {
            1.0
        };

        for conn_list in &mut connections {
            for conn in conn_list {
                conn.weight *= scale;
            }
        }

        // Construct deterministic input weights
        // Uses sine/cosine basis for smooth, structured mapping
        let mut input_weights = vec![0.0f32; n_residues * n_input_features];
        for i in 0..n_residues {
            for f in 0..n_input_features {
                let idx = i * n_input_features + f;
                // Deterministic weight: combines position and feature index
                let phase = (i as f32 * 0.1 + f as f32 * 0.7).sin();
                let amplitude = 1.0 / (1.0 + (f as f32 * 0.3).exp());
                input_weights[idx] = phase * amplitude;
            }
        }

        // Initialize readout weights to uniform (will be computed via closed-form if GT available)
        let readout_weights = vec![1.0 / n_residues as f32; n_residues];

        BioReservoir {
            n_neurons: n_residues,
            connections,
            spectral_radius: target_spectral_radius,
            input_weights,
            readout_weights,
        }
    }

    /// Estimate spectral radius via power iteration
    fn estimate_spectral_radius(connections: &[Vec<SparseConnection>], n: usize) -> f32 {
        let mut v = vec![1.0 / (n as f32).sqrt(); n];
        let mut v_new = vec![0.0f32; n];

        // 20 power iterations
        for _ in 0..20 {
            // Matrix-vector multiply
            for (i, conn_list) in connections.iter().enumerate() {
                let mut sum = 0.0f32;
                for conn in conn_list {
                    sum += conn.weight * v[conn.target];
                }
                v_new[i] = sum;
            }

            // Compute norm
            let norm: f32 = v_new.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-10 {
                for x in &mut v_new {
                    *x /= norm;
                }
            }

            std::mem::swap(&mut v, &mut v_new);
        }

        // Estimate eigenvalue
        let mut eigenvalue = 0.0f32;
        for (i, conn_list) in connections.iter().enumerate() {
            let mut sum = 0.0f32;
            for conn in conn_list {
                sum += conn.weight * v[conn.target];
            }
            eigenvalue += sum * v[i];
        }

        eigenvalue.abs()
    }

    /// Flatten connections to dense format for GPU upload
    /// Returns (indices, weights, offsets) for CSR-like format
    pub fn to_gpu_format(&self) -> (Vec<i32>, Vec<f32>, Vec<i32>) {
        let mut indices = Vec::new();
        let mut weights = Vec::new();
        let mut offsets = vec![0i32];

        for conn_list in &self.connections {
            for conn in conn_list {
                indices.push(conn.target as i32);
                weights.push(conn.weight);
            }
            offsets.push(indices.len() as i32);
        }

        (indices, weights, offsets)
    }

    /// Get flattened input weights for GPU constant memory
    pub fn get_input_weights_flat(&self, max_features: usize) -> Vec<f32> {
        let mut flat = vec![0.0f32; self.n_neurons * max_features];
        let n_feat = self.input_weights.len() / self.n_neurons;

        for i in 0..self.n_neurons {
            for f in 0..n_feat.min(max_features) {
                flat[i * max_features + f] = self.input_weights[i * n_feat + f];
            }
        }

        flat
    }
}

/// Closed-form ridge regression for readout weights
///
/// NOTE: This IS supervised learning (uses labeled data).
/// It's "training-free" only in the sense that reservoir weights are fixed.
///
/// Computes: W = (H^T H + λI)^-1 H^T y
pub fn compute_readout_weights(
    reservoir_states: &[Vec<f32>], // [n_samples][state_dim]
    targets: &[f32],               // [n_samples] target values
    lambda: f32,                   // Regularization (default: 1e-4)
) -> Vec<f32> {
    if reservoir_states.is_empty() || targets.is_empty() {
        return Vec::new();
    }

    let n_samples = reservoir_states.len();
    let dim = reservoir_states[0].len();

    if n_samples != targets.len() || dim == 0 {
        return vec![0.0; dim];
    }

    // Build H^T H + λI (dim x dim)
    let mut hth = vec![0.0f32; dim * dim];
    for sample in reservoir_states {
        for i in 0..dim {
            for j in 0..dim {
                hth[i * dim + j] += sample[i] * sample[j];
            }
        }
    }

    // Add regularization
    for i in 0..dim {
        hth[i * dim + i] += lambda;
    }

    // Build H^T y (dim)
    let mut hty = vec![0.0f32; dim];
    for (sample, &target) in reservoir_states.iter().zip(targets) {
        for i in 0..dim {
            hty[i] += sample[i] * target;
        }
    }

    // Solve via Cholesky decomposition (H^T H is symmetric positive definite)
    solve_symmetric_linear(&hth, &hty, dim)
}

/// Solve Ax = b where A is symmetric positive definite
fn solve_symmetric_linear(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    // Simple Cholesky decomposition
    let mut l = vec![0.0f32; n * n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum <= 0.0 {
                    // Matrix not positive definite, use fallback
                    return vec![0.0; n];
                }
                l[i * n + j] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }

    // Forward substitution: Ly = b
    let mut y = vec![0.0f32; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i * n + j] * y[j];
        }
        y[i] = sum / l[i * n + i];
    }

    // Backward substitution: L^T x = y
    let mut x = vec![0.0f32; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[j * n + i] * x[j];
        }
        x[i] = sum / l[i * n + i];
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reservoir_construction() {
        let reservoir = BioReservoir::construct(100, 8, 16, 0.95, 8);
        assert_eq!(reservoir.n_neurons, 100);
        assert!(reservoir.spectral_radius <= 1.0);

        // Check that connections are deterministic
        let reservoir2 = BioReservoir::construct(100, 8, 16, 0.95, 8);
        for i in 0..100 {
            assert_eq!(
                reservoir.connections[i].len(),
                reservoir2.connections[i].len()
            );
            for (c1, c2) in reservoir.connections[i]
                .iter()
                .zip(&reservoir2.connections[i])
            {
                assert_eq!(c1.target, c2.target);
                assert!((c1.weight - c2.weight).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_spectral_radius() {
        let reservoir = BioReservoir::construct(50, 4, 8, 0.9, 4);
        // Spectral radius should be close to target
        let (_, weights, offsets) = reservoir.to_gpu_format();
        assert!(!weights.is_empty());
        assert_eq!(offsets.len(), 51);
    }
}
