//=============================================================================
// PRISM-LBS Structural Transfer Entropy Features
// Session 13: Adapt transfer entropy for spatial protein structures
//
// Transfer entropy detects directed information flow → allosteric coupling
// Cryptic sites open via allosteric mechanisms → TE should identify triggers
//=============================================================================

#include <cuda_runtime.h>
#include <cmath>

//=============================================================================
// KERNEL: Compute Structural Transfer Entropy Features
//=============================================================================
// For each residue, compute TE-based allosteric coupling features
// Uses spatial neighborhoods as "time series" proxy
//=============================================================================

extern "C" __global__ void compute_structural_te_features(
    const float* __restrict__ distance_matrix,  // [n_residues * n_residues]
    const float* __restrict__ bfactor,          // [n_residues] flexibility
    const float* __restrict__ conservation,     // [n_residues]
    const float* __restrict__ burial,           // [n_residues]
    const int n_residues,
    float* __restrict__ te_features             // [n_residues * 8] output
) {
    int residue_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (residue_id >= n_residues) return;

    // Get this residue's properties
    float my_bfactor = bfactor[residue_id];
    float my_cons = conservation[residue_id];
    float my_burial = burial[residue_id];

    int base = residue_id * 8;

    // ═════════════════════════════════════════════════════════════════════
    // TE-INSPIRED FEATURES (Simplified for static structures)
    // ═════════════════════════════════════════════════════════════════════

    // We can't compute true TE without time series, but we can compute
    // TE-inspired coupling features using spatial gradients

    // Feature 0: Outgoing coupling (how much this residue influences neighbors)
    // High B-factor + low conservation = broadcasts fluctuations
    float coupling_out = my_bfactor * (1.0f - my_cons);

    // Feature 1: Incoming coupling (how much this residue is influenced)
    // Low B-factor + high conservation = receives signals
    float coupling_in = (1.0f - my_bfactor) * my_cons;

    // Feature 2: Net coupling direction
    float coupling_net = coupling_out - coupling_in;

    // Feature 3: Long-range allosteric potential
    // Sum weighted interactions with distant residues
    float long_range_coupling = 0.0f;
    int long_range_count = 0;

    for (int j = 0; j < n_residues; j++) {
        if (j == residue_id) continue;

        float dist = distance_matrix[residue_id * n_residues + j];

        // Long-range: 12-20Å (allosteric range)
        if (dist >= 12.0f && dist <= 20.0f && dist < 999.0f) {
            float j_bfac = bfactor[j];
            float j_cons = conservation[j];

            // Coupling strength: product of flexibilities × conservation
            float strength = sqrtf(my_bfactor * j_bfac) * sqrtf(my_cons * j_cons);

            long_range_coupling += strength / (dist * dist);  // Inverse square decay
            long_range_count++;
        }
    }

    float long_range_avg = long_range_count > 0 ?
        long_range_coupling / long_range_count : 0.0f;

    // Feature 4: Allosteric hub score
    // High if: flexible + conserved + many long-range contacts
    float hub_score = my_bfactor * my_cons * sqrtf((float)long_range_count / 10.0f);

    // Feature 5: Interface coupling
    // Difference in coupling across burial boundary (surface ↔ core)
    float surface_coupling = 0.0f;
    float core_coupling = 0.0f;
    int surface_count = 0, core_count = 0;

    for (int j = 0; j < n_residues; j++) {
        if (j == residue_id) continue;

        float dist = distance_matrix[residue_id * n_residues + j];
        if (dist < 8.0f && dist < 999.0f) {
            float j_burial = burial[j];
            float j_bfac = bfactor[j];

            if (j_burial < 0.4f) {  // Surface residue
                surface_coupling += j_bfac;
                surface_count++;
            } else if (j_burial > 0.6f) {  // Core residue
                core_coupling += j_bfac;
                core_count++;
            }
        }
    }

    float surface_avg = surface_count > 0 ? surface_coupling / surface_count : 0.0f;
    float core_avg = core_count > 0 ? core_coupling / core_count : 0.0f;
    float interface_coupling = fabsf(surface_avg - core_avg);

    // Feature 6: Frustration energy
    // High-energy contacts that want to break (cryptic site potential)
    float frustration = 0.0f;
    int frustrated_count = 0;

    for (int j = 0; j < n_residues; j++) {
        if (j == residue_id) continue;

        float dist = distance_matrix[residue_id * n_residues + j];
        if (dist < 5.0f && dist < 999.0f) {  // Close contact
            float j_cons = conservation[j];

            // Frustrated if: both conserved but forced close (steric clash risk)
            if (my_cons > 0.7f && j_cons > 0.7f && dist < 4.0f) {
                frustration += 1.0f / (dist * dist);  // Inverse square penalty
                frustrated_count++;
            }
        }
    }

    // Feature 7: Coupling diversity (entropy of coupling distribution)
    // Measure how evenly coupled to all neighbors vs focused on few
    float coupling_variance = 0.0f;
    int total_neighbors = 0;

    for (int j = 0; j < n_residues; j++) {
        if (j == residue_id) continue;

        float dist = distance_matrix[residue_id * n_residues + j];
        if (dist < 12.0f && dist < 999.0f) {
            float j_bfac = bfactor[j];
            coupling_variance += (j_bfac - my_bfactor) * (j_bfac - my_bfactor);
            total_neighbors++;
        }
    }

    float coupling_diversity = total_neighbors > 0 ?
        sqrtf(coupling_variance / total_neighbors) : 0.0f;

    // Store features
    te_features[base + 0] = tanhf(coupling_out);        // Outgoing (normalized)
    te_features[base + 1] = tanhf(coupling_in);         // Incoming
    te_features[base + 2] = tanhf(coupling_net);        // Net direction
    te_features[base + 3] = tanhf(long_range_avg * 10.0f);  // Long-range coupling
    te_features[base + 4] = tanhf(hub_score);           // Hub score
    te_features[base + 5] = tanhf(interface_coupling);  // Interface coupling
    te_features[base + 6] = tanhf(frustration);         // Frustration energy
    te_features[base + 7] = tanhf(coupling_diversity);  // Diversity
}

//=============================================================================
// Simplified TE computation adapted for proteins
// Uses spatial neighborhoods instead of time series
//=============================================================================
