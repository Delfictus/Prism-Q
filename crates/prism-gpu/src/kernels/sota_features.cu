//=============================================================================
// PRISM-LBS SOTA FEATURES KERNEL
// Session 10E: Multi-Pass Architecture - Pass 2
// Computes all 30 SOTA biochemical features using global distance matrix
//=============================================================================

#include <cuda_runtime.h>

//=============================================================================
// CONSTANT MEMORY: Biochemical Parameters
//=============================================================================

// Van der Waals radii (Å) by atom type: C=0, N=1, O=2, S=3, H=4, P=5, other=6
__constant__ float c_vdw_radii[7] = {
    1.70f,  // Carbon
    1.55f,  // Nitrogen
    1.52f,  // Oxygen
    1.80f,  // Sulfur
    1.20f,  // Hydrogen
    1.80f,  // Phosphorus
    1.70f   // Other
};

// Maximum SASA values per residue type (Å²)
// Order: ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL
__constant__ float c_sasa_max[20] = {
    113.0f, 241.0f, 158.0f, 151.0f, 140.0f,  // ALA, ARG, ASN, ASP, CYS
    189.0f, 183.0f,  85.0f, 194.0f, 182.0f,  // GLN, GLU, GLY, HIS, ILE
    180.0f, 211.0f, 204.0f, 218.0f, 143.0f,  // LEU, LYS, MET, PHE, PRO
    122.0f, 146.0f, 259.0f, 229.0f, 160.0f   // SER, THR, TRP, TYR, VAL
};

// Conservation propensity (0-1 scale) - BLOSUM62 derived
__constant__ float c_conservation_propensity[20] = {
    0.45f, 0.70f, 0.55f, 0.65f, 0.80f,  // ALA, ARG, ASN, ASP, CYS(highest)
    0.50f, 0.65f, 0.75f, 0.70f, 0.40f,  // GLN, GLU, GLY, HIS, ILE
    0.35f, 0.55f, 0.45f, 0.50f, 0.70f,  // LEU(lowest), LYS, MET, PHE, PRO
    0.45f, 0.50f, 0.75f, 0.60f, 0.40f   // SER, THR, TRP, TYR, VAL
};

// Hydrophobicity scale (normalized 0-1)
__constant__ float c_hydrophobicity[20] = {
    0.62f, 0.0f,  0.09f, 0.05f, 0.29f,  // ALA, ARG, ASN, ASP, CYS
    0.0f,  0.0f,  0.48f, 0.13f, 0.90f,  // GLN, GLU, GLY, HIS, ILE
    0.90f, 0.0f,  0.64f, 0.88f, 0.68f,  // LEU, LYS, MET, PHE, PRO
    0.05f, 0.13f, 0.84f, 0.41f, 0.76f   // SER, THR, TRP, TYR, VAL
};

//=============================================================================
// KERNEL: Compute SOTA Features (30-dim)
//=============================================================================
// Each thread processes ONE residue, computes all 30 features
// Has global view via distance_matrix
//=============================================================================

extern "C" __global__ void compute_sota_features_kernel(
    const float* __restrict__ distance_matrix,  // [n_residues * n_residues]
    const float* __restrict__ atoms,            // [n_atoms * 3]
    const int* __restrict__ ca_indices,         // [n_residues]
    const float* __restrict__ bfactor,          // [n_residues]
    const float* __restrict__ burial,           // [n_residues] from existing kernel
    const float* __restrict__ conservation,     // [n_residues] from existing kernel
    const int n_residues,
    float* __restrict__ sota_features           // [n_residues * 30] output
) {
    int residue_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (residue_id >= n_residues) return;

    // Get this residue's data
    int ca_idx = ca_indices[residue_id];
    if (ca_idx < 0) {
        // No Cα - zero out features
        for (int f = 0; f < 30; f++) {
            sota_features[residue_id * 30 + f] = 0.0f;
        }
        return;
    }

    float3 my_pos;
    my_pos.x = atoms[ca_idx * 3 + 0];
    my_pos.y = atoms[ca_idx * 3 + 1];
    my_pos.z = atoms[ca_idx * 3 + 2];

    float my_bfactor = bfactor[residue_id];
    float my_burial = burial[residue_id];
    float my_conservation = conservation[residue_id];

    // Infer residue type from conservation (rough proxy until we parse PDB properly)
    int my_type = (my_conservation > 0.7f) ? 4 : (my_conservation > 0.5f) ? 1 : 0;
    my_type = min(max(my_type, 0), 19);

    int base = residue_id * 30;

    // ═════════════════════════════════════════════════════════════════════
    // SASA FEATURES [0-3]: Solvent Accessible Surface Area
    // ═════════════════════════════════════════════════════════════════════
    float total_occlusion = 0.0f;
    int sasa_neighbors = 0;
    const float SASA_CUTOFF = 10.0f;

    for (int j = 0; j < n_residues; j++) {
        if (j == residue_id) continue;

        float dist = distance_matrix[residue_id * n_residues + j];

        if (dist < SASA_CUTOFF && dist < 999.0f) {
            // Gaussian occlusion model
            float sigma = 3.0f;
            float occlusion = expf(-dist * dist / (2.0f * sigma * sigma));
            total_occlusion += occlusion;
            sasa_neighbors++;
        }
    }

    float exposure = expf(-total_occlusion * 0.3f);
    float max_sasa = c_sasa_max[my_type];
    float estimated_sasa = max_sasa * exposure;

    sota_features[base + 0] = estimated_sasa / 259.0f;  // [0] Normalized SASA
    sota_features[base + 1] = estimated_sasa / fmaxf(max_sasa, 1.0f);  // [1] Relative SASA
    sota_features[base + 2] = 1.0f - sota_features[base + 1];  // [2] Burial depth
    sota_features[base + 3] = total_occlusion / fmaxf((float)sasa_neighbors, 1.0f);  // [3] Occlusion gradient

    // ═════════════════════════════════════════════════════════════════════
    // ELECTROSTATICS FEATURES [4-7]: Debye-Hückel Screened Potentials
    // ═════════════════════════════════════════════════════════════════════
    const float ELEC_CUTOFF = 15.0f;
    const float DEBYE_LENGTH = 8.0f;
    const float K_COULOMB = 4.15f;  // 332 / 80 (water dielectric)

    // Estimate charge from conservation + burial
    float my_charge = 0.0f;
    if (my_conservation > 0.65f && exposure > 0.5f) {
        my_charge = 1.0f;  // Likely Arg, Lys, Asp, Glu (charged + surface)
    } else if (my_conservation > 0.75f && exposure < 0.3f) {
        my_charge = -0.3f;  // Likely Cys, Met (conserved + buried)
    }

    float potential = 0.0f;
    float field_x = 0.0f, field_y = 0.0f, field_z = 0.0f;
    float charge_density = fabsf(my_charge);
    float complementarity = 0.0f;
    int charged_neighbors = 0;

    for (int j = 0; j < n_residues; j++) {
        if (j == residue_id) continue;

        float dist = distance_matrix[residue_id * n_residues + j];

        if (dist < ELEC_CUTOFF && dist < 999.0f && dist > 0.5f) {
            // Estimate j's charge
            float j_burial = burial[j];
            float j_cons = conservation[j];
            float j_exposure = 1.0f - j_burial;
            float j_charge = 0.0f;
            if (j_cons > 0.65f && j_exposure > 0.5f) {
                j_charge = 1.0f;
            } else if (j_cons > 0.75f && j_exposure < 0.3f) {
                j_charge = -0.3f;
            }

            // Debye-Hückel screened potential
            float screening = expf(-dist / DEBYE_LENGTH);
            float v = K_COULOMB * j_charge * screening / dist;
            potential += v;

            // Electric field (vector magnitude)
            int ca_j = ca_indices[j];
            if (ca_j >= 0) {
                float dx = atoms[ca_j * 3 + 0] - my_pos.x;
                float dy = atoms[ca_j * 3 + 1] - my_pos.y;
                float dz = atoms[ca_j * 3 + 2] - my_pos.z;
                float field_factor = v * (1.0f/dist + 1.0f/DEBYE_LENGTH) / dist;
                field_x -= field_factor * dx;
                field_y -= field_factor * dy;
                field_z -= field_factor * dz;
            }

            charge_density += fabsf(j_charge) * screening;

            if (fabsf(j_charge) > 0.1f) {
                charged_neighbors++;
                complementarity -= my_charge * j_charge * screening;
            }
        }
    }

    float field_mag = sqrtf(field_x*field_x + field_y*field_y + field_z*field_z);

    sota_features[base + 4] = tanhf(potential / 20.0f);  // [4] Potential
    sota_features[base + 5] = tanhf(field_mag / 5.0f);  // [5] Field magnitude
    sota_features[base + 6] = tanhf(charge_density / 3.0f);  // [6] Charge density
    sota_features[base + 7] = charged_neighbors > 0 ? tanhf(complementarity / charged_neighbors) : 0.0f;  // [7] Complementarity

    // ═════════════════════════════════════════════════════════════════════
    // CONSERVATION FEATURES [8-11]
    // ═════════════════════════════════════════════════════════════════════
    float type_conservation = c_conservation_propensity[my_type];

    // Sequence window (±4 residues)
    float window_sum = 0.0f;
    int window_count = 0;
    for (int off = -4; off <= 4; off++) {
        int j = residue_id + off;
        if (j >= 0 && j < n_residues && j != residue_id) {
            window_sum += conservation[j];
            window_count++;
        }
    }

    float window_avg = window_count > 0 ? window_sum / window_count : my_conservation;
    float relative_cons = my_conservation / fmaxf(window_avg, 0.1f);

    sota_features[base + 8] = type_conservation;  // [8] Type conservation
    sota_features[base + 9] = fminf(relative_cons / 2.0f, 1.0f);  // [9] Relative
    sota_features[base + 10] = my_conservation * exposure;  // [10] Anomaly (conserved + exposed)

    // [11] Functional site indicator (conserved + optimal burial)
    float optimal_burial = 1.0f - fabsf(exposure - 0.5f) * 2.0f;
    sota_features[base + 11] = my_conservation * optimal_burial;

    // ═════════════════════════════════════════════════════════════════════
    // NETWORK CENTRALITY FEATURES [12-15]
    // ═════════════════════════════════════════════════════════════════════
    const float NETWORK_CUTOFF = 8.0f;

    int degree = 0;
    float closeness_sum = 0.0f;
    float neighbor_degree_sum = 0.0f;

    // First pass: my degree and closeness
    for (int j = 0; j < n_residues; j++) {
        if (j == residue_id) continue;

        float dist = distance_matrix[residue_id * n_residues + j];

        if (dist < NETWORK_CUTOFF && dist < 999.0f) {
            degree++;
            closeness_sum += 1.0f / fmaxf(dist, 0.1f);
        }
    }

    // Second pass: eigenvector centrality approximation
    for (int j = 0; j < n_residues; j++) {
        if (j == residue_id) continue;

        float dist_ij = distance_matrix[residue_id * n_residues + j];

        if (dist_ij < NETWORK_CUTOFF && dist_ij < 999.0f) {
            // Count j's degree
            int j_degree = 0;
            for (int k = 0; k < n_residues; k++) {
                if (k == j) continue;
                float dist_jk = distance_matrix[j * n_residues + k];
                if (dist_jk < NETWORK_CUTOFF && dist_jk < 999.0f) {
                    j_degree++;
                }
            }
            neighbor_degree_sum += (float)j_degree;
        }
    }

    float degree_norm = (float)degree / fmaxf((float)(n_residues - 1), 1.0f);

    // Betweenness approximation
    float seq_position = (float)residue_id / fmaxf((float)n_residues, 1.0f);
    float centrality_position = 1.0f - fabsf(seq_position - 0.5f) * 2.0f;
    float betweenness = degree_norm * centrality_position;

    sota_features[base + 12] = degree_norm;  // [12] Degree centrality
    sota_features[base + 13] = betweenness;  // [13] Betweenness approx
    sota_features[base + 14] = tanhf(closeness_sum / 5.0f);  // [14] Closeness
    sota_features[base + 15] = degree > 0 ? tanhf(neighbor_degree_sum / (degree * 10.0f)) : 0.0f;  // [15] Eigenvector approx

    // ═════════════════════════════════════════════════════════════════════
    // SHAPE/CAVITY FEATURES [16-18]
    // ═════════════════════════════════════════════════════════════════════
    const float SHAPE_RADIUS = 10.0f;

    float3 local_centroid = {0.0f, 0.0f, 0.0f};
    int local_count = 0;

    // Compute local centroid
    for (int j = 0; j < n_residues; j++) {
        if (j == residue_id) continue;

        float dist = distance_matrix[residue_id * n_residues + j];

        if (dist < SHAPE_RADIUS && dist < 999.0f) {
            int ca_j = ca_indices[j];
            if (ca_j >= 0) {
                local_centroid.x += atoms[ca_j * 3 + 0];
                local_centroid.y += atoms[ca_j * 3 + 1];
                local_centroid.z += atoms[ca_j * 3 + 2];
                local_count++;
            }
        }
    }

    if (local_count > 0) {
        local_centroid.x /= local_count;
        local_centroid.y /= local_count;
        local_centroid.z /= local_count;
    } else {
        local_centroid = my_pos;
    }

    // Gyration tensor
    float gxx = 0.0f, gyy = 0.0f, gzz = 0.0f;
    float convexity_sum = 0.0f;

    for (int j = 0; j < n_residues; j++) {
        if (j == residue_id) continue;

        float dist = distance_matrix[residue_id * n_residues + j];

        if (dist < SHAPE_RADIUS && dist < 999.0f) {
            int ca_j = ca_indices[j];
            if (ca_j >= 0) {
                float rx = atoms[ca_j * 3 + 0] - local_centroid.x;
                float ry = atoms[ca_j * 3 + 1] - local_centroid.y;
                float rz = atoms[ca_j * 3 + 2] - local_centroid.z;

                gxx += rx * rx;
                gyy += ry * ry;
                gzz += rz * rz;

                // Convexity check
                float to_cent_x = local_centroid.x - my_pos.x;
                float to_cent_y = local_centroid.y - my_pos.y;
                float to_cent_z = local_centroid.z - my_pos.z;
                float to_j_x = atoms[ca_j * 3 + 0] - my_pos.x;
                float to_j_y = atoms[ca_j * 3 + 1] - my_pos.y;
                float to_j_z = atoms[ca_j * 3 + 2] - my_pos.z;
                float dot = to_cent_x * to_j_x + to_cent_y * to_j_y + to_cent_z * to_j_z;
                convexity_sum += (dot > 0) ? 1.0f : -1.0f;
            }
        }
    }

    if (local_count > 0) {
        gxx /= local_count;
        gyy /= local_count;
        gzz /= local_count;
    }

    float trace = gxx + gyy + gzz;
    float avg_eigen = trace / 3.0f + 1e-6f;
    float min_diag = fminf(fminf(gxx, gyy), gzz);
    float sphericity = min_diag / avg_eigen;

    sota_features[base + 16] = fminf(1.0f, sphericity);  // [16] Sphericity
    sota_features[base + 17] = local_count > 0 ? convexity_sum / local_count : 0.0f;  // [17] Convexity
    sota_features[base + 18] = tanhf(sqrtf(gxx + gyy + gzz) / 5.0f);  // [18] Pocket depth

    // ═════════════════════════════════════════════════════════════════════
    // NMA MOBILITY FEATURES [19-21]
    // ═════════════════════════════════════════════════════════════════════
    float bfactor_norm = (my_bfactor - 10.0f) / 70.0f;
    bfactor_norm = fmaxf(0.0f, fminf(1.0f, bfactor_norm));

    sota_features[base + 19] = bfactor_norm;  // [19] B-factor mobility
    sota_features[base + 20] = bfactor_norm * exposure;  // [20] Flexible surface
    sota_features[base + 21] = bfactor_norm * (1.0f - degree_norm);  // [21] Conformational freedom

    // ═════════════════════════════════════════════════════════════════════
    // CONTACT ANALYSIS FEATURES [22-25]
    // ═════════════════════════════════════════════════════════════════════
    const float CONTACT_CUTOFF = 6.0f;

    float hydrophobic_contacts = 0.0f;
    float polar_contacts = 0.0f;
    float aromatic_contacts = 0.0f;

    for (int j = 0; j < n_residues; j++) {
        if (j == residue_id) continue;

        float dist = distance_matrix[residue_id * n_residues + j];

        if (dist < CONTACT_CUTOFF && dist < 999.0f) {
            float weight = 1.0f - dist / CONTACT_CUTOFF;

            // Use burial + conservation as type proxies
            float j_burial = burial[j];
            float j_cons = conservation[j];
            float j_exposure = 1.0f - j_burial;

            // Hydrophobic: buried (Leu, Ile, Val, Ala, Met, Phe, Trp)
            if (j_burial > 0.6f) {
                hydrophobic_contacts += weight;
            }

            // Polar: surface + conserved (Arg, Lys, Asp, Glu, His)
            if (j_exposure > 0.6f && j_cons > 0.6f) {
                polar_contacts += weight;
            }

            // Aromatic: moderate burial + conserved (Phe, Trp, Tyr, His)
            if (j_burial > 0.3f && j_burial < 0.7f && j_cons > 0.55f) {
                aromatic_contacts += weight;
            }
        }
    }

    sota_features[base + 22] = tanhf(hydrophobic_contacts / 3.0f);  // [22] Hydrophobic contacts
    sota_features[base + 23] = tanhf(polar_contacts / 3.0f);  // [23] Polar contacts
    sota_features[base + 24] = tanhf(aromatic_contacts / 2.0f);  // [24] Aromatic contacts

    // [25] Contact diversity (entropy)
    float total_contacts = hydrophobic_contacts + polar_contacts + aromatic_contacts + 0.01f;
    float h_frac = hydrophobic_contacts / total_contacts;
    float p_frac = polar_contacts / total_contacts;
    float a_frac = aromatic_contacts / total_contacts;
    float diversity = 0.0f;
    if (h_frac > 0.01f) diversity -= h_frac * logf(h_frac + 1e-10f);
    if (p_frac > 0.01f) diversity -= p_frac * logf(p_frac + 1e-10f);
    if (a_frac > 0.01f) diversity -= a_frac * logf(a_frac + 1e-10f);

    sota_features[base + 25] = diversity / 1.1f;

    // ═════════════════════════════════════════════════════════════════════
    // DRUGGABILITY FEATURES [26-29]
    // ═════════════════════════════════════════════════════════════════════
    float my_hydrophobicity = c_hydrophobicity[my_type];
    float burial_score = sota_features[base + 2];
    float concavity = (1.0f - sota_features[base + 17]) / 2.0f;  // Map convexity to concavity

    // [26] Druggability score
    float druggability = 0.35f * my_hydrophobicity +
                         0.25f * burial_score +
                         0.20f * concavity +
                         0.20f * (1.0f - fabsf(sota_features[base + 4]));  // Neutral electrostatics

    // [27] Binding site likelihood
    float binding = 0.30f * my_conservation +
                    0.35f * druggability +
                    0.20f * sota_features[base + 18] +  // Depth
                    0.15f * sota_features[base + 22];  // Hydrophobic contacts

    // [28] Ligand efficiency predictor
    float optimal_exposure = 1.0f - fabsf(exposure - 0.4f) * 2.0f;  // Peak at 40%
    float efficiency = optimal_exposure * sota_features[base + 25] *  // Contact diversity
                       (1.0f - fabsf(my_hydrophobicity - 0.5f));  // Balanced hydrophobicity

    // [29] Fragment hotspot
    float hotspot = 0.25f * sota_features[base + 15] +  // Eigenvector centrality
                    0.25f * sota_features[base + 13] +  // Betweenness
                    0.25f * concavity +
                    0.25f * sota_features[base + 10];  // Conservation anomaly

    sota_features[base + 26] = druggability;
    sota_features[base + 27] = binding;
    sota_features[base + 28] = fmaxf(0.0f, efficiency);
    sota_features[base + 29] = hotspot;
}
