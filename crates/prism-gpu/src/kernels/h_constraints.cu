/**
 * H-Bond Constraint CUDA Kernels for PRISM-4D
 *
 * Implements ANALYTIC (non-iterative) constraint solvers for protein X-H bonds.
 * This allows freezing fast H-bond vibrations (~10 fs period) enabling larger
 * timesteps (2.0 fs vs 0.25 fs) and better temperature control.
 *
 * Key Insight: Proteins have only 5 H-bond cluster topologies that can be
 * solved analytically without iteration:
 *
 *   Type     | Example              | Count | Algorithm
 *   ---------|----------------------|-------|------------------
 *   SINGLE_H | C-H, N-H, O-H, S-H   | ~30%  | 1 equation, exact
 *   CH2      | Methylene (Lys, etc) | ~40%  | 2x2 system, Cramer
 *   CH3      | Methyl (Ala, Val)    | ~25%  | 3x3 system, Cramer
 *   NH2      | Amide (Asn, Gln)     | <1%   | Same as CH2
 *   NH3      | Protonated Lys       | <1%   | Same as CH3
 *
 * These kernels complement SETTLE (for water) to constrain ALL fast vibrations.
 *
 * Algorithm: SHAKE-like position correction but solved analytically
 *   1. Compute bond vectors after unconstrained move
 *   2. Solve for Lagrange multipliers λ (exact, no iteration)
 *   3. Apply position corrections: Δr = λ * inv_mass * unit_vector
 *   4. Apply velocity corrections (RATTLE): remove velocity along constraint
 */

extern "C" {

/**
 * Constraint cluster data structure
 * Packed for efficient GPU memory access (48 bytes total)
 */
struct HConstraintCluster {
    int central_atom;           // Heavy atom index
    int hydrogen_atoms[3];      // Up to 3 H indices (-1 if unused)
    float bond_lengths[3];      // Target X-H distances (Angstroms)
    float inv_mass_central;     // 1/m_heavy (1/Dalton)
    float inv_mass_h;           // 1/m_H (same for all H in cluster)
    int n_hydrogens;            // 1, 2, or 3
    int cluster_type;           // 1=SINGLE_H, 2=CH2, 3=CH3, 4=NH2, 5=NH3
};

// ============================================================================
// SINGLE_H CONSTRAINT (1 equation, exact solution)
// ============================================================================
// Covers: C-H (backbone Cα), N-H (backbone amide), O-H, S-H (Cys)
// This is the simplest case with a closed-form solution.

__global__ void constrain_single_h(
    float* __restrict__ pos,           // Positions [n_atoms * 3]
    float* __restrict__ vel,           // Velocities [n_atoms * 3]
    const HConstraintCluster* __restrict__ clusters,
    const int n_clusters,
    const float dt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_clusters) return;

    HConstraintCluster c = clusters[idx];

    // Only process SINGLE_H clusters
    if (c.n_hydrogens != 1) return;

    int A = c.central_atom;
    int H = c.hydrogen_atoms[0];
    float d0 = c.bond_lengths[0];
    float inv_m_A = c.inv_mass_central;
    float inv_m_H = c.inv_mass_h;
    float inv_m_sum = inv_m_A + inv_m_H;

    // Load positions
    float3 rA = make_float3(pos[A*3], pos[A*3+1], pos[A*3+2]);
    float3 rH = make_float3(pos[H*3], pos[H*3+1], pos[H*3+2]);

    // Bond vector after unconstrained move
    float3 rAH = make_float3(rH.x - rA.x, rH.y - rA.y, rH.z - rA.z);
    float d2 = rAH.x*rAH.x + rAH.y*rAH.y + rAH.z*rAH.z;
    float d = sqrtf(d2);

    // Exact solution for single constraint:
    // We want |rA' - rH'|² = d0² where rA' = rA + λ*inv_m_A*u, rH' = rH - λ*inv_m_H*u
    // This gives: (d - λ*inv_m_sum)² = d0²
    // Solving: λ = (d - d0) / inv_m_sum
    float lambda = (d - d0) / inv_m_sum;

    // Unit vector along bond
    float inv_d = 1.0f / (d + 1e-10f);  // Avoid division by zero
    float3 u = make_float3(rAH.x * inv_d, rAH.y * inv_d, rAH.z * inv_d);

    // Position corrections
    float dA = lambda * inv_m_A;
    float dH = lambda * inv_m_H;

    pos[A*3+0] += u.x * dA;
    pos[A*3+1] += u.y * dA;
    pos[A*3+2] += u.z * dA;

    pos[H*3+0] -= u.x * dH;
    pos[H*3+1] -= u.y * dH;
    pos[H*3+2] -= u.z * dH;

    // RATTLE velocity correction: remove velocity component along constraint
    // This ensures velocities don't violate the constraint
    float3 vA = make_float3(vel[A*3], vel[A*3+1], vel[A*3+2]);
    float3 vH = make_float3(vel[H*3], vel[H*3+1], vel[H*3+2]);
    float3 vAH = make_float3(vH.x - vA.x, vH.y - vA.y, vH.z - vA.z);

    // Project relative velocity onto bond direction
    float v_along = vAH.x * u.x + vAH.y * u.y + vAH.z * u.z;

    // Distribute correction by mass ratio
    float v_corr_A = v_along * inv_m_A / inv_m_sum;
    float v_corr_H = v_along * inv_m_H / inv_m_sum;

    vel[A*3+0] += u.x * v_corr_A;
    vel[A*3+1] += u.y * v_corr_A;
    vel[A*3+2] += u.z * v_corr_A;

    vel[H*3+0] -= u.x * v_corr_H;
    vel[H*3+1] -= u.y * v_corr_H;
    vel[H*3+2] -= u.z * v_corr_H;
}

// ============================================================================
// CH2/NH2 CONSTRAINT (2 coupled equations, 2x2 Cramer's rule)
// ============================================================================
// Covers: Methylene groups (Lys CE, Arg CG, etc.), Amide groups (Asn ND2, Gln NE2)
// Two constraints sharing a central atom form a coupled system.

__global__ void constrain_ch2(
    float* __restrict__ pos,
    float* __restrict__ vel,
    const HConstraintCluster* __restrict__ clusters,
    const int n_clusters,
    const float dt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_clusters) return;

    HConstraintCluster c = clusters[idx];

    // Only process 2-H clusters (CH2 or NH2)
    if (c.n_hydrogens != 2) return;

    int C = c.central_atom;
    int H1 = c.hydrogen_atoms[0];
    int H2 = c.hydrogen_atoms[1];
    float d1 = c.bond_lengths[0];
    float d2_target = c.bond_lengths[1];
    float inv_m_C = c.inv_mass_central;
    float inv_m_H = c.inv_mass_h;

    // Load positions
    float3 rC = make_float3(pos[C*3], pos[C*3+1], pos[C*3+2]);
    float3 rH1 = make_float3(pos[H1*3], pos[H1*3+1], pos[H1*3+2]);
    float3 rH2 = make_float3(pos[H2*3], pos[H2*3+1], pos[H2*3+2]);

    // Bond vectors
    float3 r1 = make_float3(rH1.x - rC.x, rH1.y - rC.y, rH1.z - rC.z);
    float3 r2 = make_float3(rH2.x - rC.x, rH2.y - rC.y, rH2.z - rC.z);

    float len1 = sqrtf(r1.x*r1.x + r1.y*r1.y + r1.z*r1.z);
    float len2 = sqrtf(r2.x*r2.x + r2.y*r2.y + r2.z*r2.z);

    // Unit vectors
    float inv_len1 = 1.0f / (len1 + 1e-10f);
    float inv_len2 = 1.0f / (len2 + 1e-10f);
    float3 u1 = make_float3(r1.x * inv_len1, r1.y * inv_len1, r1.z * inv_len1);
    float3 u2 = make_float3(r2.x * inv_len2, r2.y * inv_len2, r2.z * inv_len2);

    // Coupling coefficient: how much the two constraints interact
    float u1_dot_u2 = u1.x*u2.x + u1.y*u2.y + u1.z*u2.z;

    // Build 2x2 system: A * λ = b
    // A = [a11, a12]   b = [b1]
    //     [a21, a22]       [b2]
    //
    // a_ii = inv_m_C + inv_m_H (self-term)
    // a_ij = inv_m_C * (u_i · u_j) (coupling through central atom)
    float a11 = inv_m_C + inv_m_H;
    float a22 = inv_m_C + inv_m_H;
    float a12 = inv_m_C * u1_dot_u2;
    float a21 = a12;  // Symmetric

    float b1 = len1 - d1;
    float b2 = len2 - d2_target;

    // Solve 2x2 system with Cramer's rule
    float det = a11 * a22 - a12 * a21;
    float inv_det = 1.0f / (det + 1e-10f);

    float lambda1 = (a22 * b1 - a12 * b2) * inv_det;
    float lambda2 = (a11 * b2 - a21 * b1) * inv_det;

    // Position corrections for central atom (receives both corrections)
    float3 dC = make_float3(
        inv_m_C * (lambda1 * u1.x + lambda2 * u2.x),
        inv_m_C * (lambda1 * u1.y + lambda2 * u2.y),
        inv_m_C * (lambda1 * u1.z + lambda2 * u2.z)
    );

    // Position corrections for hydrogens
    float3 dH1 = make_float3(
        -inv_m_H * lambda1 * u1.x,
        -inv_m_H * lambda1 * u1.y,
        -inv_m_H * lambda1 * u1.z
    );
    float3 dH2 = make_float3(
        -inv_m_H * lambda2 * u2.x,
        -inv_m_H * lambda2 * u2.y,
        -inv_m_H * lambda2 * u2.z
    );

    // Apply position corrections
    pos[C*3+0] += dC.x;   pos[C*3+1] += dC.y;   pos[C*3+2] += dC.z;
    pos[H1*3+0] += dH1.x; pos[H1*3+1] += dH1.y; pos[H1*3+2] += dH1.z;
    pos[H2*3+0] += dH2.x; pos[H2*3+1] += dH2.y; pos[H2*3+2] += dH2.z;

    // RATTLE velocity corrections (same pattern)
    float3 vC = make_float3(vel[C*3], vel[C*3+1], vel[C*3+2]);
    float3 vH1 = make_float3(vel[H1*3], vel[H1*3+1], vel[H1*3+2]);
    float3 vH2 = make_float3(vel[H2*3], vel[H2*3+1], vel[H2*3+2]);

    // Relative velocities
    float3 v1 = make_float3(vH1.x - vC.x, vH1.y - vC.y, vH1.z - vC.z);
    float3 v2 = make_float3(vH2.x - vC.x, vH2.y - vC.y, vH2.z - vC.z);

    float v1_along = v1.x * u1.x + v1.y * u1.y + v1.z * u1.z;
    float v2_along = v2.x * u2.x + v2.y * u2.y + v2.z * u2.z;

    // Solve velocity constraint (same matrix A)
    float mu1 = (a22 * v1_along - a12 * v2_along) * inv_det;
    float mu2 = (a11 * v2_along - a21 * v1_along) * inv_det;

    vel[C*3+0] += inv_m_C * (mu1 * u1.x + mu2 * u2.x);
    vel[C*3+1] += inv_m_C * (mu1 * u1.y + mu2 * u2.y);
    vel[C*3+2] += inv_m_C * (mu1 * u1.z + mu2 * u2.z);

    vel[H1*3+0] -= inv_m_H * mu1 * u1.x;
    vel[H1*3+1] -= inv_m_H * mu1 * u1.y;
    vel[H1*3+2] -= inv_m_H * mu1 * u1.z;

    vel[H2*3+0] -= inv_m_H * mu2 * u2.x;
    vel[H2*3+1] -= inv_m_H * mu2 * u2.y;
    vel[H2*3+2] -= inv_m_H * mu2 * u2.z;
}

// ============================================================================
// CH3/NH3 CONSTRAINT (3 coupled equations, 3x3 Cramer's rule)
// ============================================================================
// Covers: Methyl groups (Ala CB, Val CG1/CG2, Leu CD1/CD2, Ile CG2/CD1, Met CE)
//         Protonated lysine (NZ with 3 H)

__global__ void constrain_ch3(
    float* __restrict__ pos,
    float* __restrict__ vel,
    const HConstraintCluster* __restrict__ clusters,
    const int n_clusters,
    const float dt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_clusters) return;

    HConstraintCluster c = clusters[idx];

    // Only process 3-H clusters (CH3 or NH3)
    if (c.n_hydrogens != 3) return;

    int C = c.central_atom;
    int H1 = c.hydrogen_atoms[0];
    int H2 = c.hydrogen_atoms[1];
    int H3 = c.hydrogen_atoms[2];
    float d1 = c.bond_lengths[0];
    float d2_target = c.bond_lengths[1];
    float d3 = c.bond_lengths[2];
    float inv_m_C = c.inv_mass_central;
    float inv_m_H = c.inv_mass_h;

    // Load positions
    float3 rC = make_float3(pos[C*3], pos[C*3+1], pos[C*3+2]);
    float3 rH1 = make_float3(pos[H1*3], pos[H1*3+1], pos[H1*3+2]);
    float3 rH2 = make_float3(pos[H2*3], pos[H2*3+1], pos[H2*3+2]);
    float3 rH3 = make_float3(pos[H3*3], pos[H3*3+1], pos[H3*3+2]);

    // Bond vectors
    float3 r1 = make_float3(rH1.x - rC.x, rH1.y - rC.y, rH1.z - rC.z);
    float3 r2 = make_float3(rH2.x - rC.x, rH2.y - rC.y, rH2.z - rC.z);
    float3 r3 = make_float3(rH3.x - rC.x, rH3.y - rC.y, rH3.z - rC.z);

    float len1 = sqrtf(r1.x*r1.x + r1.y*r1.y + r1.z*r1.z);
    float len2 = sqrtf(r2.x*r2.x + r2.y*r2.y + r2.z*r2.z);
    float len3 = sqrtf(r3.x*r3.x + r3.y*r3.y + r3.z*r3.z);

    // Unit vectors
    float inv_len1 = 1.0f / (len1 + 1e-10f);
    float inv_len2 = 1.0f / (len2 + 1e-10f);
    float inv_len3 = 1.0f / (len3 + 1e-10f);

    float3 u1 = make_float3(r1.x * inv_len1, r1.y * inv_len1, r1.z * inv_len1);
    float3 u2 = make_float3(r2.x * inv_len2, r2.y * inv_len2, r2.z * inv_len2);
    float3 u3 = make_float3(r3.x * inv_len3, r3.y * inv_len3, r3.z * inv_len3);

    // Dot products for coupling
    float u12 = u1.x*u2.x + u1.y*u2.y + u1.z*u2.z;
    float u13 = u1.x*u3.x + u1.y*u3.y + u1.z*u3.z;
    float u23 = u2.x*u3.x + u2.y*u3.y + u2.z*u3.z;

    // Build 3x3 system
    float diag = inv_m_C + inv_m_H;
    float a11 = diag, a22 = diag, a33 = diag;
    float a12 = inv_m_C * u12, a21 = a12;
    float a13 = inv_m_C * u13, a31 = a13;
    float a23 = inv_m_C * u23, a32 = a23;

    float b1 = len1 - d1;
    float b2 = len2 - d2_target;
    float b3 = len3 - d3;

    // 3x3 determinant: det = a(ei-fh) - b(di-fg) + c(dh-eg)
    // where [a,b,c; d,e,f; g,h,i] is our matrix
    float det = a11 * (a22*a33 - a23*a32)
              - a12 * (a21*a33 - a23*a31)
              + a13 * (a21*a32 - a22*a31);

    float inv_det = 1.0f / (det + 1e-10f);

    // Cramer's rule for λ1, λ2, λ3
    // λ1 = det(B1) / det where B1 replaces column 1 with b
    float det1 = b1 * (a22*a33 - a23*a32)
               - a12 * (b2*a33 - a23*b3)
               + a13 * (b2*a32 - a22*b3);

    float det2 = a11 * (b2*a33 - a23*b3)
               - b1 * (a21*a33 - a23*a31)
               + a13 * (a21*b3 - b2*a31);

    float det3 = a11 * (a22*b3 - b2*a32)
               - a12 * (a21*b3 - b2*a31)
               + b1 * (a21*a32 - a22*a31);

    float lambda1 = det1 * inv_det;
    float lambda2 = det2 * inv_det;
    float lambda3 = det3 * inv_det;

    // Position corrections
    float3 dC = make_float3(
        inv_m_C * (lambda1*u1.x + lambda2*u2.x + lambda3*u3.x),
        inv_m_C * (lambda1*u1.y + lambda2*u2.y + lambda3*u3.y),
        inv_m_C * (lambda1*u1.z + lambda2*u2.z + lambda3*u3.z)
    );

    pos[C*3+0] += dC.x;
    pos[C*3+1] += dC.y;
    pos[C*3+2] += dC.z;

    pos[H1*3+0] -= inv_m_H * lambda1 * u1.x;
    pos[H1*3+1] -= inv_m_H * lambda1 * u1.y;
    pos[H1*3+2] -= inv_m_H * lambda1 * u1.z;

    pos[H2*3+0] -= inv_m_H * lambda2 * u2.x;
    pos[H2*3+1] -= inv_m_H * lambda2 * u2.y;
    pos[H2*3+2] -= inv_m_H * lambda2 * u2.z;

    pos[H3*3+0] -= inv_m_H * lambda3 * u3.x;
    pos[H3*3+1] -= inv_m_H * lambda3 * u3.y;
    pos[H3*3+2] -= inv_m_H * lambda3 * u3.z;

    // RATTLE velocity corrections
    float3 vC = make_float3(vel[C*3], vel[C*3+1], vel[C*3+2]);
    float3 vH1 = make_float3(vel[H1*3], vel[H1*3+1], vel[H1*3+2]);
    float3 vH2 = make_float3(vel[H2*3], vel[H2*3+1], vel[H2*3+2]);
    float3 vH3 = make_float3(vel[H3*3], vel[H3*3+1], vel[H3*3+2]);

    float3 v1 = make_float3(vH1.x - vC.x, vH1.y - vC.y, vH1.z - vC.z);
    float3 v2 = make_float3(vH2.x - vC.x, vH2.y - vC.y, vH2.z - vC.z);
    float3 v3 = make_float3(vH3.x - vC.x, vH3.y - vC.y, vH3.z - vC.z);

    float v1_along = v1.x*u1.x + v1.y*u1.y + v1.z*u1.z;
    float v2_along = v2.x*u2.x + v2.y*u2.y + v2.z*u2.z;
    float v3_along = v3.x*u3.x + v3.y*u3.y + v3.z*u3.z;

    // Solve velocity constraint with same matrix
    float vdet1 = v1_along * (a22*a33 - a23*a32)
                - a12 * (v2_along*a33 - a23*v3_along)
                + a13 * (v2_along*a32 - a22*v3_along);

    float vdet2 = a11 * (v2_along*a33 - a23*v3_along)
                - v1_along * (a21*a33 - a23*a31)
                + a13 * (a21*v3_along - v2_along*a31);

    float vdet3 = a11 * (a22*v3_along - v2_along*a32)
                - a12 * (a21*v3_along - v2_along*a31)
                + v1_along * (a21*a32 - a22*a31);

    float mu1 = vdet1 * inv_det;
    float mu2 = vdet2 * inv_det;
    float mu3 = vdet3 * inv_det;

    vel[C*3+0] += inv_m_C * (mu1*u1.x + mu2*u2.x + mu3*u3.x);
    vel[C*3+1] += inv_m_C * (mu1*u1.y + mu2*u2.y + mu3*u3.y);
    vel[C*3+2] += inv_m_C * (mu1*u1.z + mu2*u2.z + mu3*u3.z);

    vel[H1*3+0] -= inv_m_H * mu1 * u1.x;
    vel[H1*3+1] -= inv_m_H * mu1 * u1.y;
    vel[H1*3+2] -= inv_m_H * mu1 * u1.z;

    vel[H2*3+0] -= inv_m_H * mu2 * u2.x;
    vel[H2*3+1] -= inv_m_H * mu2 * u2.y;
    vel[H2*3+2] -= inv_m_H * mu2 * u2.z;

    vel[H3*3+0] -= inv_m_H * mu3 * u3.x;
    vel[H3*3+1] -= inv_m_H * mu3 * u3.y;
    vel[H3*3+2] -= inv_m_H * mu3 * u3.z;
}

// ============================================================================
// UNIFIED DISPATCH KERNEL
// ============================================================================
// Dispatches to appropriate constraint kernel based on cluster type.
// This allows a single kernel launch to handle all H-bond constraints.

__global__ void constrain_h_bonds_unified(
    float* __restrict__ pos,
    float* __restrict__ vel,
    const HConstraintCluster* __restrict__ clusters,
    const int n_clusters,
    const float dt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_clusters) return;

    HConstraintCluster c = clusters[idx];

    // Dispatch based on number of hydrogens
    // (Type 1,4 = single; Type 2,4 = double; Type 3,5 = triple)
    switch (c.n_hydrogens) {
        case 1: {
            // SINGLE_H constraint (same logic as constrain_single_h)
            int A = c.central_atom;
            int H = c.hydrogen_atoms[0];
            float d0 = c.bond_lengths[0];
            float inv_m_A = c.inv_mass_central;
            float inv_m_H = c.inv_mass_h;
            float inv_m_sum = inv_m_A + inv_m_H;

            float3 rA = make_float3(pos[A*3], pos[A*3+1], pos[A*3+2]);
            float3 rH = make_float3(pos[H*3], pos[H*3+1], pos[H*3+2]);
            float3 rAH = make_float3(rH.x - rA.x, rH.y - rA.y, rH.z - rA.z);
            float d = sqrtf(rAH.x*rAH.x + rAH.y*rAH.y + rAH.z*rAH.z);

            float lambda = (d - d0) / inv_m_sum;
            float inv_d = 1.0f / (d + 1e-10f);
            float3 u = make_float3(rAH.x * inv_d, rAH.y * inv_d, rAH.z * inv_d);

            pos[A*3+0] += u.x * lambda * inv_m_A;
            pos[A*3+1] += u.y * lambda * inv_m_A;
            pos[A*3+2] += u.z * lambda * inv_m_A;
            pos[H*3+0] -= u.x * lambda * inv_m_H;
            pos[H*3+1] -= u.y * lambda * inv_m_H;
            pos[H*3+2] -= u.z * lambda * inv_m_H;

            // Velocity correction
            float3 vA = make_float3(vel[A*3], vel[A*3+1], vel[A*3+2]);
            float3 vH = make_float3(vel[H*3], vel[H*3+1], vel[H*3+2]);
            float3 vAH = make_float3(vH.x - vA.x, vH.y - vA.y, vH.z - vA.z);
            float v_along = vAH.x*u.x + vAH.y*u.y + vAH.z*u.z;

            vel[A*3+0] += u.x * v_along * inv_m_A / inv_m_sum;
            vel[A*3+1] += u.y * v_along * inv_m_A / inv_m_sum;
            vel[A*3+2] += u.z * v_along * inv_m_A / inv_m_sum;
            vel[H*3+0] -= u.x * v_along * inv_m_H / inv_m_sum;
            vel[H*3+1] -= u.y * v_along * inv_m_H / inv_m_sum;
            vel[H*3+2] -= u.z * v_along * inv_m_H / inv_m_sum;
            break;
        }

        case 2: {
            // CH2/NH2 constraint (inline for performance)
            int C = c.central_atom;
            int H1 = c.hydrogen_atoms[0];
            int H2 = c.hydrogen_atoms[1];
            float d1 = c.bond_lengths[0];
            float d2_t = c.bond_lengths[1];
            float inv_m_C = c.inv_mass_central;
            float inv_m_H = c.inv_mass_h;

            float3 rC = make_float3(pos[C*3], pos[C*3+1], pos[C*3+2]);
            float3 rH1 = make_float3(pos[H1*3], pos[H1*3+1], pos[H1*3+2]);
            float3 rH2 = make_float3(pos[H2*3], pos[H2*3+1], pos[H2*3+2]);

            float3 r1 = make_float3(rH1.x - rC.x, rH1.y - rC.y, rH1.z - rC.z);
            float3 r2 = make_float3(rH2.x - rC.x, rH2.y - rC.y, rH2.z - rC.z);

            float len1 = sqrtf(r1.x*r1.x + r1.y*r1.y + r1.z*r1.z);
            float len2 = sqrtf(r2.x*r2.x + r2.y*r2.y + r2.z*r2.z);

            float3 u1 = make_float3(r1.x/(len1+1e-10f), r1.y/(len1+1e-10f), r1.z/(len1+1e-10f));
            float3 u2 = make_float3(r2.x/(len2+1e-10f), r2.y/(len2+1e-10f), r2.z/(len2+1e-10f));

            float u12 = u1.x*u2.x + u1.y*u2.y + u1.z*u2.z;
            float diag = inv_m_C + inv_m_H;
            float off = inv_m_C * u12;
            float det = diag*diag - off*off;
            float inv_det = 1.0f / (det + 1e-10f);

            float b1 = len1 - d1;
            float b2 = len2 - d2_t;
            float l1 = (diag * b1 - off * b2) * inv_det;
            float l2 = (diag * b2 - off * b1) * inv_det;

            pos[C*3+0] += inv_m_C * (l1*u1.x + l2*u2.x);
            pos[C*3+1] += inv_m_C * (l1*u1.y + l2*u2.y);
            pos[C*3+2] += inv_m_C * (l1*u1.z + l2*u2.z);
            pos[H1*3+0] -= inv_m_H * l1 * u1.x;
            pos[H1*3+1] -= inv_m_H * l1 * u1.y;
            pos[H1*3+2] -= inv_m_H * l1 * u1.z;
            pos[H2*3+0] -= inv_m_H * l2 * u2.x;
            pos[H2*3+1] -= inv_m_H * l2 * u2.y;
            pos[H2*3+2] -= inv_m_H * l2 * u2.z;

            // Velocity correction
            float3 vC = make_float3(vel[C*3], vel[C*3+1], vel[C*3+2]);
            float3 vH1 = make_float3(vel[H1*3], vel[H1*3+1], vel[H1*3+2]);
            float3 vH2 = make_float3(vel[H2*3], vel[H2*3+1], vel[H2*3+2]);
            float v1 = (vH1.x-vC.x)*u1.x + (vH1.y-vC.y)*u1.y + (vH1.z-vC.z)*u1.z;
            float v2 = (vH2.x-vC.x)*u2.x + (vH2.y-vC.y)*u2.y + (vH2.z-vC.z)*u2.z;
            float m1 = (diag * v1 - off * v2) * inv_det;
            float m2 = (diag * v2 - off * v1) * inv_det;

            vel[C*3+0] += inv_m_C * (m1*u1.x + m2*u2.x);
            vel[C*3+1] += inv_m_C * (m1*u1.y + m2*u2.y);
            vel[C*3+2] += inv_m_C * (m1*u1.z + m2*u2.z);
            vel[H1*3+0] -= inv_m_H * m1 * u1.x;
            vel[H1*3+1] -= inv_m_H * m1 * u1.y;
            vel[H1*3+2] -= inv_m_H * m1 * u1.z;
            vel[H2*3+0] -= inv_m_H * m2 * u2.x;
            vel[H2*3+1] -= inv_m_H * m2 * u2.y;
            vel[H2*3+2] -= inv_m_H * m2 * u2.z;
            break;
        }

        case 3: {
            // CH3/NH3 constraint - full 3x3 solve inlined
            int C = c.central_atom;
            int H1 = c.hydrogen_atoms[0];
            int H2 = c.hydrogen_atoms[1];
            int H3 = c.hydrogen_atoms[2];
            float d1 = c.bond_lengths[0];
            float d2_t = c.bond_lengths[1];
            float d3 = c.bond_lengths[2];
            float inv_m_C = c.inv_mass_central;
            float inv_m_H = c.inv_mass_h;

            float3 rC = make_float3(pos[C*3], pos[C*3+1], pos[C*3+2]);
            float3 rH1 = make_float3(pos[H1*3], pos[H1*3+1], pos[H1*3+2]);
            float3 rH2 = make_float3(pos[H2*3], pos[H2*3+1], pos[H2*3+2]);
            float3 rH3 = make_float3(pos[H3*3], pos[H3*3+1], pos[H3*3+2]);

            float3 r1 = make_float3(rH1.x-rC.x, rH1.y-rC.y, rH1.z-rC.z);
            float3 r2 = make_float3(rH2.x-rC.x, rH2.y-rC.y, rH2.z-rC.z);
            float3 r3 = make_float3(rH3.x-rC.x, rH3.y-rC.y, rH3.z-rC.z);

            float len1 = sqrtf(r1.x*r1.x + r1.y*r1.y + r1.z*r1.z) + 1e-10f;
            float len2 = sqrtf(r2.x*r2.x + r2.y*r2.y + r2.z*r2.z) + 1e-10f;
            float len3 = sqrtf(r3.x*r3.x + r3.y*r3.y + r3.z*r3.z) + 1e-10f;

            float3 u1 = make_float3(r1.x/len1, r1.y/len1, r1.z/len1);
            float3 u2 = make_float3(r2.x/len2, r2.y/len2, r2.z/len2);
            float3 u3 = make_float3(r3.x/len3, r3.y/len3, r3.z/len3);

            float u12 = u1.x*u2.x + u1.y*u2.y + u1.z*u2.z;
            float u13 = u1.x*u3.x + u1.y*u3.y + u1.z*u3.z;
            float u23 = u2.x*u3.x + u2.y*u3.y + u2.z*u3.z;

            float diag = inv_m_C + inv_m_H;
            float a12 = inv_m_C * u12;
            float a13 = inv_m_C * u13;
            float a23 = inv_m_C * u23;

            float det = diag * (diag*diag - a23*a23) - a12 * (a12*diag - a23*a13) + a13 * (a12*a23 - diag*a13);
            float inv_det = 1.0f / (det + 1e-10f);

            float b1 = len1 - d1;
            float b2 = len2 - d2_t;
            float b3 = len3 - d3;

            float det1 = b1*(diag*diag - a23*a23) - a12*(b2*diag - a23*b3) + a13*(b2*a23 - diag*b3);
            float det2 = diag*(b2*diag - a23*b3) - b1*(a12*diag - a23*a13) + a13*(a12*b3 - b2*a13);
            float det3 = diag*(diag*b3 - b2*a23) - a12*(a12*b3 - b2*a13) + b1*(a12*a23 - diag*a13);

            float l1 = det1 * inv_det;
            float l2 = det2 * inv_det;
            float l3 = det3 * inv_det;

            pos[C*3+0] += inv_m_C * (l1*u1.x + l2*u2.x + l3*u3.x);
            pos[C*3+1] += inv_m_C * (l1*u1.y + l2*u2.y + l3*u3.y);
            pos[C*3+2] += inv_m_C * (l1*u1.z + l2*u2.z + l3*u3.z);

            pos[H1*3+0] -= inv_m_H * l1 * u1.x;
            pos[H1*3+1] -= inv_m_H * l1 * u1.y;
            pos[H1*3+2] -= inv_m_H * l1 * u1.z;
            pos[H2*3+0] -= inv_m_H * l2 * u2.x;
            pos[H2*3+1] -= inv_m_H * l2 * u2.y;
            pos[H2*3+2] -= inv_m_H * l2 * u2.z;
            pos[H3*3+0] -= inv_m_H * l3 * u3.x;
            pos[H3*3+1] -= inv_m_H * l3 * u3.y;
            pos[H3*3+2] -= inv_m_H * l3 * u3.z;

            // Velocity correction (same matrix)
            float3 vC = make_float3(vel[C*3], vel[C*3+1], vel[C*3+2]);
            float v1a = (vel[H1*3]-vC.x)*u1.x + (vel[H1*3+1]-vC.y)*u1.y + (vel[H1*3+2]-vC.z)*u1.z;
            float v2a = (vel[H2*3]-vC.x)*u2.x + (vel[H2*3+1]-vC.y)*u2.y + (vel[H2*3+2]-vC.z)*u2.z;
            float v3a = (vel[H3*3]-vC.x)*u3.x + (vel[H3*3+1]-vC.y)*u3.y + (vel[H3*3+2]-vC.z)*u3.z;

            float vd1 = v1a*(diag*diag-a23*a23) - a12*(v2a*diag-a23*v3a) + a13*(v2a*a23-diag*v3a);
            float vd2 = diag*(v2a*diag-a23*v3a) - v1a*(a12*diag-a23*a13) + a13*(a12*v3a-v2a*a13);
            float vd3 = diag*(diag*v3a-v2a*a23) - a12*(a12*v3a-v2a*a13) + v1a*(a12*a23-diag*a13);

            float m1 = vd1 * inv_det;
            float m2 = vd2 * inv_det;
            float m3 = vd3 * inv_det;

            vel[C*3+0] += inv_m_C * (m1*u1.x + m2*u2.x + m3*u3.x);
            vel[C*3+1] += inv_m_C * (m1*u1.y + m2*u2.y + m3*u3.y);
            vel[C*3+2] += inv_m_C * (m1*u1.z + m2*u2.z + m3*u3.z);

            vel[H1*3+0] -= inv_m_H * m1 * u1.x;
            vel[H1*3+1] -= inv_m_H * m1 * u1.y;
            vel[H1*3+2] -= inv_m_H * m1 * u1.z;
            vel[H2*3+0] -= inv_m_H * m2 * u2.x;
            vel[H2*3+1] -= inv_m_H * m2 * u2.y;
            vel[H2*3+2] -= inv_m_H * m2 * u2.z;
            vel[H3*3+0] -= inv_m_H * m3 * u3.x;
            vel[H3*3+1] -= inv_m_H * m3 * u3.y;
            vel[H3*3+2] -= inv_m_H * m3 * u3.z;
            break;
        }
    }
}

} // extern "C"
