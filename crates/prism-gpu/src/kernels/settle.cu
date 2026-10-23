/**
 * SETTLE - Analytical Constraint Solver for Rigid Water
 *
 * Implements the SETTLE algorithm for maintaining rigid TIP3P water geometry
 * during molecular dynamics. This is an analytical solution that satisfies
 * all three distance constraints (OH1, OH2, HH) exactly in a single pass.
 *
 * Reference: Miyamoto & Kollman (1992) J. Comput. Chem. 13:952-962
 *
 * Algorithm overview (Miyamoto-Kollman SETTLE):
 * 1. Preserve the unconstrained mass-weighted center-of-mass displacement.
 * 2. Build the old plane normal and a rotation-sensitive molecular frame.
 * 3. Solve the phi, psi, and theta rotations analytically.
 * 4. Transform the exact rigid geometry back to the laboratory frame.
 *
 * Canonical frame:
 * - Origin at COM of water molecule
 * - Z-axis along bisector (from H midpoint to O)
 * - X-axis along H-H direction
 * - Y-axis perpendicular to molecular plane
 *
 * Constrained positions in canonical frame:
 * - O: (0, 0, -ra)
 * - H1: (-rc, 0, rb)
 * - H2: (+rc, 0, rb)
 *
 * where:
 * - ra = distance from O to COM
 * - rb = distance from H to COM along bisector
 * - rc = half of H-H distance
 */

extern "C" {

/**
 * Apply SETTLE constraints to water molecules
 *
 * Each thread handles one water molecule.
 * Modifies new_positions in-place to satisfy constraints.
 *
 * @param new_pos New positions after integration [n_atoms * 3]
 * @param old_pos Old positions before integration [n_atoms * 3]
 * @param water_idx Water indices [n_waters * 3] - (O, H1, H2) for each water
 * @param n_waters Number of water molecules
 * @param mO Mass of oxygen
 * @param mH Mass of hydrogen
 * @param ra Distance from O to COM (negative z direction)
 * @param rb Distance from H to COM along bisector (positive z direction)
 * @param rc Half of HH distance (x direction)
 * @param rOH2 Target OH distance squared
 * @param rHH2 Target HH distance squared
 */
__global__ void settle_constraints(
    float* __restrict__ new_pos,
    const float* __restrict__ old_pos,
    const int* __restrict__ water_idx,
    int n_waters,
    float mO, float mH,
    float ra, float rb, float rc,
    float rOH2, float rHH2,
    int* __restrict__ failure_count
) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= n_waters) return;

    // Get atom indices for this water
    int idxO = water_idx[w * 3];
    int idxH1 = water_idx[w * 3 + 1];
    int idxH2 = water_idx[w * 3 + 2];

    // Load old positions (reference configuration for frame definition)
    float3 oldO = make_float3(
        old_pos[idxO * 3], old_pos[idxO * 3 + 1], old_pos[idxO * 3 + 2]
    );
    float3 oldH1 = make_float3(
        old_pos[idxH1 * 3], old_pos[idxH1 * 3 + 1], old_pos[idxH1 * 3 + 2]
    );
    float3 oldH2 = make_float3(
        old_pos[idxH2 * 3], old_pos[idxH2 * 3 + 1], old_pos[idxH2 * 3 + 2]
    );

    // Load new positions (unconstrained)
    float3 newO = make_float3(
        new_pos[idxO * 3], new_pos[idxO * 3 + 1], new_pos[idxO * 3 + 2]
    );
    float3 newH1 = make_float3(
        new_pos[idxH1 * 3], new_pos[idxH1 * 3 + 1], new_pos[idxH1 * 3 + 2]
    );
    float3 newH2 = make_float3(
        new_pos[idxH2 * 3], new_pos[idxH2 * 3 + 1], new_pos[idxH2 * 3 + 2]
    );

    float3 b0 = make_float3(oldH1.x-oldO.x, oldH1.y-oldO.y, oldH1.z-oldO.z);
    float3 c0 = make_float3(oldH2.x-oldO.x, oldH2.y-oldO.y, oldH2.z-oldO.z);
    float3 dpO = make_float3(newO.x-oldO.x, newO.y-oldO.y, newO.z-oldO.z);
    float3 dpH1 = make_float3(newH1.x-oldH1.x, newH1.y-oldH1.y, newH1.z-oldH1.z);
    float3 dpH2 = make_float3(newH2.x-oldH2.x, newH2.y-oldH2.y, newH2.z-oldH2.z);
    float invmT = 1.0f/(mO+2.0f*mH);

    // Unconstrained new COM relative to old O.  Constraint impulses preserve it.
    float3 com;
    com.x=(mO*dpO.x+mH*(b0.x+dpH1.x)+mH*(c0.x+dpH2.x))*invmT;
    com.y=(mO*dpO.y+mH*(b0.y+dpH1.y)+mH*(c0.y+dpH2.y))*invmT;
    com.z=(mO*dpO.z+mH*(b0.z+dpH1.z)+mH*(c0.z+dpH2.z))*invmT;
    float3 a1=make_float3(dpO.x-com.x, dpO.y-com.y, dpO.z-com.z);
    float3 b1=make_float3(b0.x+dpH1.x-com.x, b0.y+dpH1.y-com.y, b0.z+dpH1.z-com.z);
    float3 c1=make_float3(c0.x+dpH2.x-com.x, c0.y+dpH2.y-com.y, c0.z+dpH2.z-com.z);

    // ez is the old plane normal.  ex depends on the unconstrained oxygen,
    // which carries the incremental rotation discarded by the old shortcut.
    float3 ez=make_float3(b0.y*c0.z-b0.z*c0.y, b0.z*c0.x-b0.x*c0.z, b0.x*c0.y-b0.y*c0.x);
    float3 ex=make_float3(a1.y*ez.z-a1.z*ez.y, a1.z*ez.x-a1.x*ez.z, a1.x*ez.y-a1.y*ez.x);
    float3 ey=make_float3(ez.y*ex.z-ez.z*ex.y, ez.z*ex.x-ez.x*ex.z, ez.x*ex.y-ez.y*ex.x);
    float ex2=ex.x*ex.x+ex.y*ex.y+ex.z*ex.z;
    float ey2=ey.x*ey.x+ey.y*ey.y+ey.z*ey.z;
    float ez2=ez.x*ez.x+ez.y*ez.y+ez.z*ez.z;
    if (ex2<1.0e-16f || ey2<1.0e-16f || ez2<1.0e-16f) {
        atomicAdd(failure_count, 1);
        return;
    }
    float ix=rsqrtf(ex2), iy=rsqrtf(ey2), iz=rsqrtf(ez2);
    ex.x*=ix; ex.y*=ix; ex.z*=ix;
    ey.x*=iy; ey.y*=iy; ey.z*=iy;
    ez.x*=iz; ez.y*=iz; ez.z*=iz;

#define DOT_FRAME(axis,v) (axis.x*v.x+axis.y*v.y+axis.z*v.z)
    float xb0d=DOT_FRAME(ex,b0), yb0d=DOT_FRAME(ey,b0);
    float xc0d=DOT_FRAME(ex,c0), yc0d=DOT_FRAME(ey,c0);
    float za1d=DOT_FRAME(ez,a1);
    float xb1d=DOT_FRAME(ex,b1), yb1d=DOT_FRAME(ey,b1), zb1d=DOT_FRAME(ez,b1);
    float xc1d=DOT_FRAME(ex,c1), yc1d=DOT_FRAME(ey,c1), zc1d=DOT_FRAME(ez,c1);

    float sinphi=fminf(1.0f,fmaxf(-1.0f,za1d/ra));
    float cosphi=sqrtf(fmaxf(0.0f,1.0f-sinphi*sinphi));
    if (cosphi<1.0e-7f) {
        atomicAdd(failure_count, 1);
        return;
    }
    float sinpsi=fminf(1.0f,fmaxf(-1.0f,(zb1d-zc1d)/(2.0f*rc*cosphi)));
    float cospsi=sqrtf(fmaxf(0.0f,1.0f-sinpsi*sinpsi));
    float ya2d=ra*cosphi;
    float xb2d=-rc*cospsi;
    float yb2d=-rb*cosphi-rc*sinpsi*sinphi;
    float yc2d=-rb*cosphi+rc*sinpsi*sinphi;
    float xb2d2=xb2d*xb2d;
    float hh2=4.0f*xb2d2+(yb2d-yc2d)*(yb2d-yc2d)+(zb1d-zc1d)*(zb1d-zc1d);
    xb2d-=0.5f*(2.0f*xb2d+sqrtf(fmaxf(0.0f,4.0f*xb2d2-hh2+rHH2)));

    float alpha=xb2d*(xb0d-xc0d)+yb0d*yb2d+yc0d*yc2d;
    float beta=xb2d*(yc0d-yb0d)+xb0d*yb2d+xc0d*yc2d;
    float gamma=xb0d*yb1d-xb1d*yb0d+xc0d*yc1d-xc1d*yc0d;
    float ab2=alpha*alpha+beta*beta;
    if (ab2<1.0e-16f) {
        atomicAdd(failure_count, 1);
        return;
    }
    float sintheta=(alpha*gamma-beta*sqrtf(fmaxf(0.0f,ab2-gamma*gamma)))/ab2;
    sintheta=fminf(1.0f,fmaxf(-1.0f,sintheta));
    float costheta=sqrtf(fmaxf(0.0f,1.0f-sintheta*sintheta));
    float3 a3d=make_float3(-ya2d*sintheta,ya2d*costheta,za1d);
    float3 b3d=make_float3(xb2d*costheta-yb2d*sintheta,xb2d*sintheta+yb2d*costheta,zb1d);
    float3 c3d=make_float3(-xb2d*costheta-yc2d*sintheta,-xb2d*sintheta+yc2d*costheta,zc1d);

#define BACK_X(v) (ex.x*v.x+ey.x*v.y+ez.x*v.z)
#define BACK_Y(v) (ex.y*v.x+ey.y*v.y+ez.y*v.z)
#define BACK_Z(v) (ex.z*v.x+ey.z*v.y+ez.z*v.z)
    new_pos[idxO*3]=oldO.x+com.x+BACK_X(a3d);
    new_pos[idxO*3+1]=oldO.y+com.y+BACK_Y(a3d);
    new_pos[idxO*3+2]=oldO.z+com.z+BACK_Z(a3d);
    new_pos[idxH1*3]=oldO.x+com.x+BACK_X(b3d);
    new_pos[idxH1*3+1]=oldO.y+com.y+BACK_Y(b3d);
    new_pos[idxH1*3+2]=oldO.z+com.z+BACK_Z(b3d);
    new_pos[idxH2*3]=oldO.x+com.x+BACK_X(c3d);
    new_pos[idxH2*3+1]=oldO.y+com.y+BACK_Y(c3d);
    new_pos[idxH2*3+2]=oldO.z+com.z+BACK_Z(c3d);
#undef BACK_X
#undef BACK_Y
#undef BACK_Z
#undef DOT_FRAME
}

/**
 * RATTLE-style SETTLE Velocity Correction
 *
 * After position constraints, velocities may have components along bond
 * directions that would violate constraints in the next step. This kernel
 * uses the RATTLE algorithm to project out ONLY the constraint-violating
 * velocity components, preserving rotational kinetic energy.
 *
 * For water with constraints C1(O-H1), C2(O-H2), C3(H1-H2):
 * Velocity constraint: r_ij · (v_i - v_j) = 0 for each bond
 *
 * This is solved via 3x3 linear system for Lagrange multipliers.
 *
 * After this correction:
 * - Relative velocities along bond directions are zero
 * - COM velocity is preserved (momentum conserved)
 * - Rotational motion perpendicular to bonds is preserved
 *
 * @param velocities Velocity array [n_atoms * 3]
 * @param positions  Position array [n_atoms * 3] (for computing bond vectors)
 * @param water_idx Water indices [n_waters * 3] - (O, H1, H2) for each water
 * @param n_waters Number of water molecules
 * @param mO Mass of oxygen
 * @param mH Mass of hydrogen
 */
__global__ void settle_velocity_correction(
    float* __restrict__ velocities,
    const float* __restrict__ positions,
    const int* __restrict__ water_idx,
    int n_waters,
    float mO, float mH,
    int* __restrict__ failure_count
) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= n_waters) return;

    // Get atom indices for this water
    int idxO = water_idx[w * 3];
    int idxH1 = water_idx[w * 3 + 1];
    int idxH2 = water_idx[w * 3 + 2];

    // Load positions (SETTLED, constrained)
    float3 pO = make_float3(
        positions[idxO * 3], positions[idxO * 3 + 1], positions[idxO * 3 + 2]
    );
    float3 pH1 = make_float3(
        positions[idxH1 * 3], positions[idxH1 * 3 + 1], positions[idxH1 * 3 + 2]
    );
    float3 pH2 = make_float3(
        positions[idxH2 * 3], positions[idxH2 * 3 + 1], positions[idxH2 * 3 + 2]
    );

    // Load velocities
    float3 vO = make_float3(
        velocities[idxO * 3], velocities[idxO * 3 + 1], velocities[idxO * 3 + 2]
    );
    float3 vH1 = make_float3(
        velocities[idxH1 * 3], velocities[idxH1 * 3 + 1], velocities[idxH1 * 3 + 2]
    );
    float3 vH2 = make_float3(
        velocities[idxH2 * 3], velocities[idxH2 * 3 + 1], velocities[idxH2 * 3 + 2]
    );

    // Compute bond vectors (from constrained positions)
    float3 r_OH1 = make_float3(pH1.x - pO.x, pH1.y - pO.y, pH1.z - pO.z);
    float3 r_OH2 = make_float3(pH2.x - pO.x, pH2.y - pO.y, pH2.z - pO.z);
    float3 r_H1H2 = make_float3(pH2.x - pH1.x, pH2.y - pH1.y, pH2.z - pH1.z);

    // Compute relative velocities
    float3 v_OH1 = make_float3(vH1.x - vO.x, vH1.y - vO.y, vH1.z - vO.z);
    float3 v_OH2 = make_float3(vH2.x - vO.x, vH2.y - vO.y, vH2.z - vO.z);
    float3 v_H1H2 = make_float3(vH2.x - vH1.x, vH2.y - vH1.y, vH2.z - vH1.z);

    // Velocity constraint violations: sigma_k = r_k · v_k
    // These should be zero after correction
    float sigma1 = r_OH1.x * v_OH1.x + r_OH1.y * v_OH1.y + r_OH1.z * v_OH1.z;
    float sigma2 = r_OH2.x * v_OH2.x + r_OH2.y * v_OH2.y + r_OH2.z * v_OH2.z;
    float sigma3 = r_H1H2.x * v_H1H2.x + r_H1H2.y * v_H1H2.y + r_H1H2.z * v_H1H2.z;

    // Build the 3x3 matrix A for the RATTLE linear system: A * lambda = sigma
    // where velocity correction is: v_i -= sum_k (lambda_k * grad_i C_k / m_i)
    //
    // For constraint C1 (O-H1): grad_O = -r_OH1, grad_H1 = +r_OH1
    // For constraint C2 (O-H2): grad_O = -r_OH2, grad_H2 = +r_OH2
    // For constraint C3 (H1-H2): grad_H1 = -r_H1H2, grad_H2 = +r_H1H2
    //
    // Matrix elements A_ij = sum_atoms (grad_i C_k · grad_j C_k / m_atom)

    float invMO = 1.0f / mO;
    float invMH = 1.0f / mH;

    // Self-interactions (diagonal)
    float r_OH1_sq = r_OH1.x * r_OH1.x + r_OH1.y * r_OH1.y + r_OH1.z * r_OH1.z;
    float r_OH2_sq = r_OH2.x * r_OH2.x + r_OH2.y * r_OH2.y + r_OH2.z * r_OH2.z;
    float r_H1H2_sq = r_H1H2.x * r_H1H2.x + r_H1H2.y * r_H1H2.y + r_H1H2.z * r_H1H2.z;

    float A11 = r_OH1_sq * (invMO + invMH);  // C1 with C1: O and H1 contribute
    float A22 = r_OH2_sq * (invMO + invMH);  // C2 with C2: O and H2 contribute
    float A33 = r_H1H2_sq * (invMH + invMH); // C3 with C3: H1 and H2 contribute

    // Cross-interactions (off-diagonal)
    float r_OH1_OH2 = r_OH1.x * r_OH2.x + r_OH1.y * r_OH2.y + r_OH1.z * r_OH2.z;
    float r_OH1_H1H2 = r_OH1.x * r_H1H2.x + r_OH1.y * r_H1H2.y + r_OH1.z * r_H1H2.z;
    float r_OH2_H1H2 = r_OH2.x * r_H1H2.x + r_OH2.y * r_H1H2.y + r_OH2.z * r_H1H2.z;

    // A12 = C1 with C2: O contributes (both have O), signs: (-r_OH1)·(-r_OH2) = +
    float A12 = r_OH1_OH2 * invMO;
    // A13 = C1 with C3: H1 contributes (C1 has H1, C3 has H1), signs: (+r_OH1)·(-r_H1H2) = -
    float A13 = -r_OH1_H1H2 * invMH;
    // A23 = C2 with C3: H2 contributes (C2 has H2, C3 has H2), signs: (+r_OH2)·(+r_H1H2) = +
    float A23 = r_OH2_H1H2 * invMH;

    // Matrix is symmetric
    float A21 = A12;
    float A31 = A13;
    float A32 = A23;

    // Solve 3x3 system A * lambda = sigma using Cramer's rule
    // det(A) = A11*(A22*A33 - A23*A32) - A12*(A21*A33 - A23*A31) + A13*(A21*A32 - A22*A31)
    float det = A11 * (A22 * A33 - A23 * A32)
              - A12 * (A21 * A33 - A23 * A31)
              + A13 * (A21 * A32 - A22 * A31);

    // Protect against singular matrix (shouldn't happen for valid water geometry)
    if (fabsf(det) < 1e-12f) {
        atomicAdd(failure_count, 1);
        return;
    }

    float invDet = 1.0f / det;

    // Compute lambda using Cramer's rule
    float lambda1 = invDet * (sigma1 * (A22 * A33 - A23 * A32)
                            - sigma2 * (A12 * A33 - A13 * A32)
                            + sigma3 * (A12 * A23 - A13 * A22));

    float lambda2 = invDet * (A11 * (sigma2 * A33 - sigma3 * A32)
                            - sigma1 * (A21 * A33 - A23 * A31)
                            + A13 * (A21 * sigma3 - sigma2 * A31));

    float lambda3 = invDet * (A11 * (A22 * sigma3 - A23 * sigma2)
                            - A12 * (A21 * sigma3 - sigma2 * A31)
                            + sigma1 * (A21 * A32 - A22 * A31));

    // Apply velocity corrections
    // v_O -= lambda1 * (-r_OH1) / mO + lambda2 * (-r_OH2) / mO
    //      = (lambda1 * r_OH1 + lambda2 * r_OH2) / mO
    vO.x += (lambda1 * r_OH1.x + lambda2 * r_OH2.x) * invMO;
    vO.y += (lambda1 * r_OH1.y + lambda2 * r_OH2.y) * invMO;
    vO.z += (lambda1 * r_OH1.z + lambda2 * r_OH2.z) * invMO;

    // v_H1 -= lambda1 * (+r_OH1) / mH + lambda3 * (-r_H1H2) / mH
    //       = (-lambda1 * r_OH1 + lambda3 * r_H1H2) / mH
    vH1.x += (-lambda1 * r_OH1.x + lambda3 * r_H1H2.x) * invMH;
    vH1.y += (-lambda1 * r_OH1.y + lambda3 * r_H1H2.y) * invMH;
    vH1.z += (-lambda1 * r_OH1.z + lambda3 * r_H1H2.z) * invMH;

    // v_H2 -= lambda2 * (+r_OH2) / mH + lambda3 * (+r_H1H2) / mH
    //       = (-lambda2 * r_OH2 - lambda3 * r_H1H2) / mH
    vH2.x += (-lambda2 * r_OH2.x - lambda3 * r_H1H2.x) * invMH;
    vH2.y += (-lambda2 * r_OH2.y - lambda3 * r_H1H2.y) * invMH;
    vH2.z += (-lambda2 * r_OH2.z - lambda3 * r_H1H2.z) * invMH;

    // Store corrected velocities
    velocities[idxO * 3]     = vO.x;
    velocities[idxO * 3 + 1] = vO.y;
    velocities[idxO * 3 + 2] = vO.z;

    velocities[idxH1 * 3]     = vH1.x;
    velocities[idxH1 * 3 + 1] = vH1.y;
    velocities[idxH1 * 3 + 2] = vH1.z;

    velocities[idxH2 * 3]     = vH2.x;
    velocities[idxH2 * 3 + 1] = vH2.y;
    velocities[idxH2 * 3 + 2] = vH2.z;
}

}  // extern "C"
