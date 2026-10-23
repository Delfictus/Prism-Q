/**
 * PME (Particle Mesh Ewald) CUDA Kernels
 *
 * Implements long-range electrostatics via Ewald summation with FFT.
 * Uses OpenMM-compatible 5th-order cardinal B-spline charge spreading and
 * force interpolation.  The reciprocal influence function uses the exact
 * discrete B-spline moduli for this same assignment operator.
 *
 * Algorithm:
 * 1. Spread atom charges to grid using B-splines
 * 2. FFT grid to reciprocal space
 * 3. Apply Green's function (energy/potential)
 * 4. Inverse FFT back to real space
 * 5. Interpolate forces from grid to atoms
 */

#include "nhs_amber_coordinate_geometry.cuh"

extern "C" {

// Constants
#define PME_ORDER 5
#define ONE_4PI_EPS0 332.0636f  // kcal*Å/(mol*e²)

// Box dimensions are passed as kernel parameters, not device globals
// This allows PME to be compiled independently from amber_mega_fused.cu

/**
 * Compute fifth-order cardinal B-spline values and their derivatives.
 *
 * This is the Darden/OpenMM recurrence used by OpenMM's CPU PME plugin.  The
 * derivative is formed from the order-four coefficients before the final
 * recurrence, then the values are advanced to order five.  Keeping both in a
 * single helper prevents spread, force interpolation, and charge-derivative
 * gathering from silently using different assignment operators.
 */
__device__ __forceinline__ void bspline_5(float u, float* w, float* dw) {
    w[PME_ORDER-1] = 0.0f;
    w[1] = u;
    w[0] = 1.0f-u;

    // Advance the linear coefficients to order four.
    for (int order = 3; order < PME_ORDER; ++order) {
        float div = 1.0f/(float)(order-1);
        w[order-1] = div*u*w[order-2];
        for (int j = 1; j < order-1; ++j) {
            w[order-j-1] = div*((u+(float)j)*w[order-j-2]
                              +((float)(order-j)-u)*w[order-j-1]);
        }
        w[0] = div*(1.0f-u)*w[0];
    }

    // Derivatives of the order-five coefficients are adjacent differences of
    // the order-four coefficients.
    dw[0] = -w[0];
    for (int j = 1; j < PME_ORDER; ++j) {
        dw[j] = w[j-1]-w[j];
    }

    // Final order-five recurrence.
    float div = 1.0f/(float)(PME_ORDER-1);
    w[PME_ORDER-1] = div*u*w[PME_ORDER-2];
    for (int j = 1; j < PME_ORDER-1; ++j) {
        w[PME_ORDER-j-1] = div*((u+(float)j)*w[PME_ORDER-j-2]
                              +((float)(PME_ORDER-j)-u)*w[PME_ORDER-j-1]);
    }
    w[0] = div*(1.0f-u)*w[0];
}

// Same order-five recurrence with uncollapsed compensated coordinates.
__device__ __forceinline__ void bspline_5_f64(double u, double* w, double* dw) {
    w[PME_ORDER-1] = 0.0f;
    w[1] = u;
    w[0] = 1.0f-u;

    // Advance the linear coefficients to order four.
    for (int order = 3; order < PME_ORDER; ++order) {
        double div = 1.0f/(double)(order-1);
        w[order-1] = div*u*w[order-2];
        for (int j = 1; j < order-1; ++j) {
            w[order-j-1] = div*((u+(double)j)*w[order-j-2]
                              +((double)(order-j)-u)*w[order-j-1]);
        }
        w[0] = div*(1.0f-u)*w[0];
    }

    // Derivatives of the order-five coefficients are adjacent differences of
    // the order-four coefficients.
    dw[0] = -w[0];
    for (int j = 1; j < PME_ORDER; ++j) {
        dw[j] = w[j-1]-w[j];
    }

    // Final order-five recurrence.
    double div = 1.0f/(double)(PME_ORDER-1);
    w[PME_ORDER-1] = div*u*w[PME_ORDER-2];
    for (int j = 1; j < PME_ORDER-1; ++j) {
        w[PME_ORDER-j-1] = div*((u+(double)j)*w[PME_ORDER-j-2]
                              +((double)(PME_ORDER-j)-u)*w[PME_ORDER-j-1]);
    }
    w[0] = div*(1.0f-u)*w[0];
}

/**
 * Exact discrete squared modulus of the order-five cardinal B-spline used by
 * spread/gather.  A continuous sinc^10 approximation is not the PME
 * deconvolution implemented by OpenMM and is especially wrong at Nyquist.
 */
__device__ __forceinline__ float bspline_modulus_5_raw(int frequency, int n) {
    // At u=0, the order-five coefficients are
    // [1, 11, 11, 1, 0]/24.  The one-cell forward shift used by OpenMM only
    // changes phase, so it does not change the squared modulus.
    float theta = 2.0f*M_PI*(float)frequency/(float)n;
    float sc = 0.0f;
    float ss = 0.0f;
    const float coeff[PME_ORDER] = {
        1.0f/24.0f, 11.0f/24.0f, 11.0f/24.0f, 1.0f/24.0f, 0.0f
    };
    for (int j = 0; j < PME_ORDER; ++j) {
        // OpenMM stores data[j] at grid sample j+1.
        float arg = theta*(float)(j+1);
        sc += coeff[j]*cosf(arg);
        ss += coeff[j]*sinf(arg);
    }
    return sc*sc+ss*ss;
}

__device__ __forceinline__ float bspline_modulus_5(int frequency, int n) {
    float modulus = bspline_modulus_5_raw(frequency, n);
    if (modulus < 1.0e-7f) {
        int previous = (frequency-1+n)%n;
        int next = (frequency+1)%n;
        modulus = 0.5f*(bspline_modulus_5_raw(previous, n)
                       +bspline_modulus_5_raw(next, n));
    }
    return modulus;
}

/**
 * Spread atom charges to PME grid using B-splines
 *
 * Each atom contributes charge to PME_ORDER^3 = 125 grid points
 * Uses atomicAdd for thread-safe accumulation
 */
__global__ void pme_spread_charges(
    const float* __restrict__ positions,  // [n_atoms * 3]
    const float* __restrict__ charges,    // [n_atoms]
    float* __restrict__ grid,             // [nx * ny * nz] - zeroed
    int n_atoms,
    int nx, int ny, int nz,
    float box_inv_x, float box_inv_y, float box_inv_z  // 1/L for each dimension
) {
    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= n_atoms) return;

    float q = charges[atom];
    if (fabsf(q) < 1e-10f) return;  // Skip uncharged atoms

    // Get atom position
    float px = positions[atom * 3];
    float py = positions[atom * 3 + 1];
    float pz = positions[atom * 3 + 2];

    // Convert to fractional coordinates [0, 1)
    float fx = px * box_inv_x;
    float fy = py * box_inv_y;
    float fz = pz * box_inv_z;

    // Wrap to [0, 1)
    fx -= floorf(fx);
    fy -= floorf(fy);
    fz -= floorf(fz);

    // Scale to grid units [0, nx), [0, ny), [0, nz)
    float gx = fx * (float)nx;
    float gy = fy * (float)ny;
    float gz = fz * (float)nz;

    // Grid indices of nearest grid point below atom
    int i0 = (int)gx;
    int j0 = (int)gy;
    int k0 = (int)gz;

    // Fractional distance from nearest grid point
    float ux = gx - (float)i0;
    float uy = gy - (float)j0;
    float uz = gz - (float)k0;

    // Compute B-spline weights
    float wx[PME_ORDER], wy[PME_ORDER], wz[PME_ORDER];
    float dw_unused_x[PME_ORDER], dw_unused_y[PME_ORDER], dw_unused_z[PME_ORDER];
    bspline_5(ux, wx, dw_unused_x);
    bspline_5(uy, wy, dw_unused_y);
    bspline_5(uz, wz, dw_unused_z);

    // Spread charge to PME_ORDER^3 grid points
    for (int di = 0; di < PME_ORDER; di++) {
        int i = (i0 + di) % nx;  // OpenMM forward support, wrapped by PBC
        for (int dj = 0; dj < PME_ORDER; dj++) {
            int j = (j0 + dj) % ny;
            for (int dk = 0; dk < PME_ORDER; dk++) {
                int k = (k0 + dk) % nz;

                float weight = wx[di] * wy[dj] * wz[dk];
                int grid_idx = i * ny * nz + j * nz + k;

                atomicAdd(&grid[grid_idx], q * weight);
            }
        }
    }
}

// Explicit descriptor-bound native precision route; legacy export above is unchanged.
__global__ void pme_spread_charges_compensated_v1(
    const float* __restrict__ positions,  // [n_atoms * 3]
    const float* __restrict__ charges,    // [n_atoms]
    float* __restrict__ grid,             // [nx * ny * nz] - zeroed
    int n_atoms,
    int nx, int ny, int nz,
    float box_inv_x, float box_inv_y, float box_inv_z  // 1/L for each dimension
,
    const BiprismCoordinateOwnerFfi* coordinate_owner,uint32_t* precision_fault
) {
    if(!bp_native_coordinate_owner_valid(coordinate_owner,positions,n_atoms,
        BP_COORD_CURRENT_INITIALIZED,precision_fault))return;
    if(nx<=0||ny<=0||nz<=0||!isfinite(box_inv_x)||!isfinite(box_inv_y)||!isfinite(box_inv_z)||
       box_inv_x<=0.0f||box_inv_y<=0.0f||box_inv_z<=0.0f) {
        bp_native_coordinate_fault(precision_fault);return;
    }

    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= n_atoms) return;
    const auto point=BpNativeGeometry<true>::load(reinterpret_cast<const float3*>(positions),
        atom,coordinate_owner,precision_fault);
    if(!isfinite(point.x)||!isfinite(point.y)||!isfinite(point.z))return;


    double q = charges[atom];
    if (fabs(q) < 1e-10f) return;  // Skip uncharged atoms

    // Get atom position
    double px = point.x;
    double py = point.y;
    double pz = point.z;

    // Convert to fractional coordinates [0, 1)
    double fx = px * box_inv_x;
    double fy = py * box_inv_y;
    double fz = pz * box_inv_z;

    // Wrap to [0, 1)
    fx -= floor(fx);
    fy -= floor(fy);
    fz -= floor(fz);

    // Scale to grid units [0, nx), [0, ny), [0, nz)
    double gx = fx * (double)nx;
    double gy = fy * (double)ny;
    double gz = fz * (double)nz;

    if(!isfinite(gx)||!isfinite(gy)||!isfinite(gz)||gx<0.0||gy<0.0||gz<0.0||
       gx>=(double)nx||gy>=(double)ny||gz>=(double)nz) {
        bp_native_coordinate_fault(precision_fault);return;
    }
    // Grid indices of nearest grid point below atom
    int i0 = (int)gx;
    int j0 = (int)gy;
    int k0 = (int)gz;

    // Fractional distance from nearest grid point
    double ux = gx - (double)i0;
    double uy = gy - (double)j0;
    double uz = gz - (double)k0;

    // Compute B-spline weights
    double wx[PME_ORDER], wy[PME_ORDER], wz[PME_ORDER];
    double dw_unused_x[PME_ORDER], dw_unused_y[PME_ORDER], dw_unused_z[PME_ORDER];
    bspline_5_f64(ux, wx, dw_unused_x);
    bspline_5_f64(uy, wy, dw_unused_y);
    bspline_5_f64(uz, wz, dw_unused_z);

    // Spread charge to PME_ORDER^3 grid points
    for (int di = 0; di < PME_ORDER; di++) {
        int i = (i0 + di) % nx;  // OpenMM forward support, wrapped by PBC
        for (int dj = 0; dj < PME_ORDER; dj++) {
            int j = (j0 + dj) % ny;
            for (int dk = 0; dk < PME_ORDER; dk++) {
                int k = (k0 + dk) % nz;

                double weight = wx[di] * wy[dj] * wz[dk];
                int grid_idx = i * ny * nz + j * nz + k;

                atomicAdd(&grid[grid_idx], (float)(q * weight));
            }
        }
    }
}

/**
 * Apply reciprocal space convolution (Green's function)
 *
 * After forward FFT, this kernel multiplies each k-space component
 * by the reciprocal-space influence function:
 *
 * G(k) = (4π/(V k²)) * exp(-k²/(4β²)) / |B(k)|²
 *
 * where β is the Ewald splitting parameter and B(k) is the B-spline
 * structure factor correction.
 *
 * NOTE: cuFFT outputs interleaved complex data: [re0, im0, re1, im1, ...]
 * So we index with idx*2 for real and idx*2+1 for imaginary.
 */
__global__ void pme_reciprocal_convolution(
    float* __restrict__ complex_grid,  // Interleaved complex: [re0, im0, re1, im1, ...]
    double* __restrict__ energy,       // Output: reciprocal energy
    float beta,                        // Ewald splitting parameter
    int nx, int ny, int nz,
    float box_inv_x, float box_inv_y, float box_inv_z
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * ny * (nz / 2 + 1);  // R2C output is hermitian
    if (idx >= total) return;

    // Convert linear index to 3D
    int nz_complex = nz / 2 + 1;
    int k = idx % nz_complex;
    int j = (idx / nz_complex) % ny;
    int i = idx / (nz_complex * ny);

    // Interleaved indexing for complex data
    int re_idx = idx * 2;
    int im_idx = idx * 2 + 1;

    // Skip DC component (k=0)
    if (i == 0 && j == 0 && k == 0) {
        complex_grid[re_idx] = 0.0f;
        complex_grid[im_idx] = 0.0f;
        return;
    }

    // Compute wave vector (handle negative frequencies)
    int mi = (i <= nx / 2) ? i : i - nx;
    int mj = (j <= ny / 2) ? j : j - ny;
    int mk = k;  // k only goes to nz/2 due to R2C

    // k² in reciprocal space
    float kx = 2.0f * M_PI * (float)mi * box_inv_x;
    float ky = 2.0f * M_PI * (float)mj * box_inv_y;
    float kz = 2.0f * M_PI * (float)mk * box_inv_z;
    float k2 = kx * kx + ky * ky + kz * kz;

    if (k2 < 1e-10f) {
        complex_grid[re_idx] = 0.0f;
        complex_grid[im_idx] = 0.0f;
        return;
    }

    // Ewald screening: exp(-k²/(4β²))
    float four_beta2 = 4.0f * beta * beta;
    float ewald_factor = expf(-k2 / four_beta2);

    // Exact discrete cardinal B-spline moduli.  Each returned value is already
    // |B_axis(k)|^2, so the three-dimensional inverse correction is one over
    // their product.  Applying another square would over-deconvolve the mesh.
    float bx_modulus = bspline_modulus_5(i, nx);
    float by_modulus = bspline_modulus_5(j, ny);
    float bz_modulus = bspline_modulus_5(k, nz);
    float b_corr = 1.0f/(bx_modulus*by_modulus*bz_modulus);

    // Green's function: (4π/(V*k²)) * exp(-k²/(4β²)) / B(k)²
    // The 1/V factor is CRITICAL - without it, energy scales with box size!
    float inv_volume = box_inv_x * box_inv_y * box_inv_z;  // 1/(Lx*Ly*Lz)
    float green = 4.0f * M_PI * ewald_factor * b_corr * inv_volume / k2;

    // Load complex values from interleaved array
    float re = complex_grid[re_idx];
    float im = complex_grid[im_idx];

    // Accumulate energy: E = (1/2) * sum_k |ρ(k)|² * G(k)
    // Factor of 2 for k != 0 (hermitian symmetry)
    double rho2 = (double)re * (double)re + (double)im * (double)im;
    double e_contrib = 0.5 * rho2 * (double)green;
    bool is_nyquist_plane = ((nz & 1) == 0) && (k == nz / 2);
    if (k > 0 && !is_nyquist_plane) {
        e_contrib *= 2.0;  // Account for hermitian conjugate
    }
    atomicAdd(energy, e_contrib * (double)ONE_4PI_EPS0);

    // Multiply grid by Green's function and store back
    complex_grid[re_idx] = re * green;
    complex_grid[im_idx] = im * green;
}

/**
 * Interpolate forces from PME grid to atoms
 *
 * The force is the negative gradient of the potential:
 * F_i = -q_i * ∇φ(r_i)
 *
 * We compute ∇φ by differentiating the B-spline interpolation.
 */
__global__ void pme_interpolate_forces(
    const float* __restrict__ positions,
    const float* __restrict__ charges,
    const float* __restrict__ potential_grid,  // After inverse FFT
    float* __restrict__ forces,
    int n_atoms,
    int nx, int ny, int nz,
    float box_inv_x, float box_inv_y, float box_inv_z
) {
    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= n_atoms) return;

    float q = charges[atom];
    if (fabsf(q) < 1e-10f) return;

    // Get atom position
    float px = positions[atom * 3];
    float py = positions[atom * 3 + 1];
    float pz = positions[atom * 3 + 2];

    // Convert to fractional coordinates
    float fx = px * box_inv_x;
    float fy = py * box_inv_y;
    float fz = pz * box_inv_z;

    // Wrap to [0, 1)
    fx -= floorf(fx);
    fy -= floorf(fy);
    fz -= floorf(fz);

    // Scale to grid units
    float gx = fx * (float)nx;
    float gy = fy * (float)ny;
    float gz = fz * (float)nz;

    int i0 = (int)gx;
    int j0 = (int)gy;
    int k0 = (int)gz;

    float ux = gx - (float)i0;
    float uy = gy - (float)j0;
    float uz = gz - (float)k0;

    // B-spline weights and derivatives
    float wx[PME_ORDER], wy[PME_ORDER], wz[PME_ORDER];
    float dwx[PME_ORDER], dwy[PME_ORDER], dwz[PME_ORDER];
    bspline_5(ux, wx, dwx);
    bspline_5(uy, wy, dwy);
    bspline_5(uz, wz, dwz);

    // Interpolate gradient
    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

    for (int di = 0; di < PME_ORDER; di++) {
        int i = (i0 + di) % nx;
        for (int dj = 0; dj < PME_ORDER; dj++) {
            int j = (j0 + dj) % ny;
            for (int dk = 0; dk < PME_ORDER; dk++) {
                int k = (k0 + dk) % nz;

                int grid_idx = i * ny * nz + j * nz + k;
                float phi = potential_grid[grid_idx];

                // Gradient via chain rule: dφ/dr = dφ/du * du/dr
                // du/dr = n/L (grid points per box length)
                Fx += dwx[di] * wy[dj] * wz[dk] * phi;
                Fy += wx[di] * dwy[dj] * wz[dk] * phi;
                Fz += wx[di] * wy[dj] * dwz[dk] * phi;
            }
        }
    }

    // Scale by grid spacing and charge
    // F = -q * ∇φ * (n/L)
    float scale_x = -q * ONE_4PI_EPS0 * (float)nx * box_inv_x;
    float scale_y = -q * ONE_4PI_EPS0 * (float)ny * box_inv_y;
    float scale_z = -q * ONE_4PI_EPS0 * (float)nz * box_inv_z;

    // Add to forces (atomic for thread safety)
    atomicAdd(&forces[atom * 3 + 0], Fx * scale_x);
    atomicAdd(&forces[atom * 3 + 1], Fy * scale_y);
    atomicAdd(&forces[atom * 3 + 2], Fz * scale_z);
}

// Explicit descriptor-bound native precision route; legacy export above is unchanged.
__global__ void pme_interpolate_forces_compensated_v1(
    const float* __restrict__ positions,
    const float* __restrict__ charges,
    const float* __restrict__ potential_grid,  // After inverse FFT
    float* __restrict__ forces,
    int n_atoms,
    int nx, int ny, int nz,
    float box_inv_x, float box_inv_y, float box_inv_z
,
    const BiprismCoordinateOwnerFfi* coordinate_owner,uint32_t* precision_fault
) {
    if(!bp_native_coordinate_owner_valid(coordinate_owner,positions,n_atoms,
        BP_COORD_CURRENT_INITIALIZED,precision_fault))return;
    if(nx<=0||ny<=0||nz<=0||!isfinite(box_inv_x)||!isfinite(box_inv_y)||!isfinite(box_inv_z)||
       box_inv_x<=0.0f||box_inv_y<=0.0f||box_inv_z<=0.0f) {
        bp_native_coordinate_fault(precision_fault);return;
    }

    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= n_atoms) return;
    const auto point=BpNativeGeometry<true>::load(reinterpret_cast<const float3*>(positions),
        atom,coordinate_owner,precision_fault);
    if(!isfinite(point.x)||!isfinite(point.y)||!isfinite(point.z))return;


    double q = charges[atom];
    if (fabs(q) < 1e-10f) return;

    // Get atom position
    double px = point.x;
    double py = point.y;
    double pz = point.z;

    // Convert to fractional coordinates
    double fx = px * box_inv_x;
    double fy = py * box_inv_y;
    double fz = pz * box_inv_z;

    // Wrap to [0, 1)
    fx -= floor(fx);
    fy -= floor(fy);
    fz -= floor(fz);

    // Scale to grid units
    double gx = fx * (double)nx;
    double gy = fy * (double)ny;
    double gz = fz * (double)nz;

    if(!isfinite(gx)||!isfinite(gy)||!isfinite(gz)||gx<0.0||gy<0.0||gz<0.0||
       gx>=(double)nx||gy>=(double)ny||gz>=(double)nz) {
        bp_native_coordinate_fault(precision_fault);return;
    }
    int i0 = (int)gx;
    int j0 = (int)gy;
    int k0 = (int)gz;

    double ux = gx - (double)i0;
    double uy = gy - (double)j0;
    double uz = gz - (double)k0;

    // B-spline weights and derivatives
    double wx[PME_ORDER], wy[PME_ORDER], wz[PME_ORDER];
    double dwx[PME_ORDER], dwy[PME_ORDER], dwz[PME_ORDER];
    bspline_5_f64(ux, wx, dwx);
    bspline_5_f64(uy, wy, dwy);
    bspline_5_f64(uz, wz, dwz);

    // Interpolate gradient
    double Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

    for (int di = 0; di < PME_ORDER; di++) {
        int i = (i0 + di) % nx;
        for (int dj = 0; dj < PME_ORDER; dj++) {
            int j = (j0 + dj) % ny;
            for (int dk = 0; dk < PME_ORDER; dk++) {
                int k = (k0 + dk) % nz;

                int grid_idx = i * ny * nz + j * nz + k;
                double phi = potential_grid[grid_idx];

                // Gradient via chain rule: dφ/dr = dφ/du * du/dr
                // du/dr = n/L (grid points per box length)
                Fx += dwx[di] * wy[dj] * wz[dk] * phi;
                Fy += wx[di] * dwy[dj] * wz[dk] * phi;
                Fz += wx[di] * wy[dj] * dwz[dk] * phi;
            }
        }
    }

    // Scale by grid spacing and charge
    // F = -q * ∇φ * (n/L)
    double scale_x = -q * ONE_4PI_EPS0 * (double)nx * box_inv_x;
    double scale_y = -q * ONE_4PI_EPS0 * (double)ny * box_inv_y;
    double scale_z = -q * ONE_4PI_EPS0 * (double)nz * box_inv_z;

    // Add to forces (atomic for thread safety)
    atomicAdd(&forces[atom * 3 + 0], (float)(Fx * scale_x));
    atomicAdd(&forces[atom * 3 + 1], (float)(Fy * scale_y));
    atomicAdd(&forces[atom * 3 + 2], (float)(Fz * scale_z));
}

/**
 * Accumulate analytic PME corrections and compact charge telemetry.
 *
 * components[1] = -k_e*β/sqrt(pi) * sum(q_i^2)
 * components[2] = sum(q_i)
 * A subsequent one-thread kernel writes
 * components[3] = -k_e*pi*Q^2/(2*β^2*V).
 *
 * The final entry is the uniform-neutralizing-plasma correction associated
 * with a non-neutral fully periodic cell.  It is position independent for a
 * fixed box, so it contributes no force.
 */
__global__ void pme_accumulate_corrections(
    const float* __restrict__ charges,
    double* __restrict__ components,
    int n_atoms,
    float beta
) {
    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= n_atoms) return;

    float q = charges[atom];
    double qd = (double)q;
    double e_self = -(double)beta / sqrt((double)M_PI) * qd * qd
                  * (double)ONE_4PI_EPS0;

    atomicAdd(&components[1], e_self);
    atomicAdd(&components[2], (double)q);
}

/**
 * Accumulate reciprocal dU/dlambda for a dynamic charge path.
 *
 * The mesh energy is quadratic in the assigned charges, hence its charge
 * derivative is sum_i dq_i/dlambda * phi_mesh(r_i).  phi_mesh is gathered
 * with the same cardinal B-spline used by the force path, which makes this
 * derivative consistent with the implemented mesh Hamiltonian.
 */
__global__ void pme_accumulate_charge_derivative(
    const float* __restrict__ positions,
    const float* __restrict__ charges,
    const float* __restrict__ dcharges_dlambda,
    const float* __restrict__ potential_grid,
    double* __restrict__ components,
    double* __restrict__ atom_energy_components,
    double* __restrict__ atom_dudlambda_components,
    int n_atoms,
    int nx, int ny, int nz,
    float box_inv_x, float box_inv_y, float box_inv_z,
    float inv_volume,
    float beta
) {
    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= n_atoms) return;
    float dq = dcharges_dlambda[atom];
    float q = charges[atom];

    float fx = positions[atom*3]*box_inv_x;
    float fy = positions[atom*3+1]*box_inv_y;
    float fz = positions[atom*3+2]*box_inv_z;
    fx -= floorf(fx); fy -= floorf(fy); fz -= floorf(fz);
    float gx=fx*(float)nx, gy=fy*(float)ny, gz=fz*(float)nz;
    int i0=(int)gx, j0=(int)gy, k0=(int)gz;
    float wx[PME_ORDER], wy[PME_ORDER], wz[PME_ORDER];
    float dw_unused_x[PME_ORDER], dw_unused_y[PME_ORDER], dw_unused_z[PME_ORDER];
    bspline_5(gx-(float)i0, wx, dw_unused_x);
    bspline_5(gy-(float)j0, wy, dw_unused_y);
    bspline_5(gz-(float)k0, wz, dw_unused_z);

    float phi=0.0f;
    for (int di=0; di<PME_ORDER; ++di) {
        int i=(i0+di)%nx;
        for (int dj=0; dj<PME_ORDER; ++dj) {
            int j=(j0+dj)%ny;
            for (int dk=0; dk<PME_ORDER; ++dk) {
                int k=(k0+dk)%nz;
                phi += wx[di]*wy[dj]*wz[dk]*potential_grid[i*ny*nz+j*nz+k];
            }
        }
    }

    const double qd = (double)q;
    const double dqd = (double)dq;
    const double beta_d = (double)beta;
    const double coulomb = (double)ONE_4PI_EPS0;
    const double mesh_energy = 0.5*coulomb*qd*(double)phi;
    const double self_energy = -coulomb*beta_d/sqrt((double)M_PI)*qd*qd;
    const double total_charge = components[2];
    const double background_prefactor = -coulomb*(double)M_PI*(double)inv_volume
        /(2.0*beta_d*beta_d);
    // This atom decomposition is exactly summable:
    //   sum_i A*Q*q_i = A*Q^2.
    const double background_energy = background_prefactor*total_charge*qd;
    const double mesh_dudl = coulomb*dqd*(double)phi;
    const double self_dudl = -2.0*coulomb*beta_d/sqrt((double)M_PI)*qd*dqd;
    // d(A*Q^2)/dl = 2*A*Q*dQ/dl.  Keeping one contribution per atom
    // lets the caller reduce exact partial derivatives by fragment slot.
    const double background_dudl = 2.0*background_prefactor*total_charge*dqd;
    atom_energy_components[atom] = mesh_energy + self_energy + background_energy;
    atom_dudlambda_components[atom] = mesh_dudl + self_dudl + background_dudl;

    atomicAdd(&components[4], mesh_dudl);
    atomicAdd(&components[5], self_dudl);
    atomicAdd(&components[6], (double)dq);
}

// Explicit descriptor-bound native precision route; legacy export above is unchanged.
__global__ void pme_accumulate_charge_derivative_compensated_v1(
    const float* __restrict__ positions,
    const float* __restrict__ charges,
    const float* __restrict__ dcharges_dlambda,
    const float* __restrict__ potential_grid,
    double* __restrict__ components,
    double* __restrict__ atom_energy_components,
    double* __restrict__ atom_dudlambda_components,
    int n_atoms,
    int nx, int ny, int nz,
    float box_inv_x, float box_inv_y, float box_inv_z,
    float inv_volume,
    float beta
,
    const BiprismCoordinateOwnerFfi* coordinate_owner,uint32_t* precision_fault
) {
    if(!bp_native_coordinate_owner_valid(coordinate_owner,positions,n_atoms,
        BP_COORD_CURRENT_INITIALIZED,precision_fault))return;
    if(nx<=0||ny<=0||nz<=0||!isfinite(box_inv_x)||!isfinite(box_inv_y)||!isfinite(box_inv_z)||
       box_inv_x<=0.0f||box_inv_y<=0.0f||box_inv_z<=0.0f) {
        bp_native_coordinate_fault(precision_fault);return;
    }

    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= n_atoms) return;
    const auto point=BpNativeGeometry<true>::load(reinterpret_cast<const float3*>(positions),
        atom,coordinate_owner,precision_fault);
    if(!isfinite(point.x)||!isfinite(point.y)||!isfinite(point.z))return;

    double dq = dcharges_dlambda[atom];
    double q = charges[atom];

    double fx = point.x*box_inv_x;
    double fy = point.y*box_inv_y;
    double fz = point.z*box_inv_z;
    fx -= floor(fx); fy -= floor(fy); fz -= floor(fz);
    double gx=fx*(double)nx, gy=fy*(double)ny, gz=fz*(double)nz;
    if(!isfinite(gx)||!isfinite(gy)||!isfinite(gz)||gx<0.0||gy<0.0||gz<0.0||
       gx>=(double)nx||gy>=(double)ny||gz>=(double)nz) {
        bp_native_coordinate_fault(precision_fault);return;
    }
    int i0=(int)gx, j0=(int)gy, k0=(int)gz;
    double wx[PME_ORDER], wy[PME_ORDER], wz[PME_ORDER];
    double dw_unused_x[PME_ORDER], dw_unused_y[PME_ORDER], dw_unused_z[PME_ORDER];
    bspline_5_f64(gx-(double)i0, wx, dw_unused_x);
    bspline_5_f64(gy-(double)j0, wy, dw_unused_y);
    bspline_5_f64(gz-(double)k0, wz, dw_unused_z);

    double phi=0.0f;
    for (int di=0; di<PME_ORDER; ++di) {
        int i=(i0+di)%nx;
        for (int dj=0; dj<PME_ORDER; ++dj) {
            int j=(j0+dj)%ny;
            for (int dk=0; dk<PME_ORDER; ++dk) {
                int k=(k0+dk)%nz;
                phi += wx[di]*wy[dj]*wz[dk]*potential_grid[i*ny*nz+j*nz+k];
            }
        }
    }

    const double qd = (double)q;
    const double dqd = (double)dq;
    const double beta_d = (double)beta;
    const double coulomb = (double)ONE_4PI_EPS0;
    const double mesh_energy = 0.5*coulomb*qd*(double)phi;
    const double self_energy = -coulomb*beta_d/sqrt((double)M_PI)*qd*qd;
    const double total_charge = components[2];
    const double background_prefactor = -coulomb*(double)M_PI*(double)inv_volume
        /(2.0*beta_d*beta_d);
    // This atom decomposition is exactly summable:
    //   sum_i A*Q*q_i = A*Q^2.
    const double background_energy = background_prefactor*total_charge*qd;
    const double mesh_dudl = coulomb*dqd*(double)phi;
    const double self_dudl = -2.0*coulomb*beta_d/sqrt((double)M_PI)*qd*dqd;
    // d(A*Q^2)/dl = 2*A*Q*dQ/dl.  Keeping one contribution per atom
    // lets the caller reduce exact partial derivatives by fragment slot.
    const double background_dudl = 2.0*background_prefactor*total_charge*dqd;
    atom_energy_components[atom] = mesh_energy + self_energy + background_energy;
    atom_dudlambda_components[atom] = mesh_dudl + self_dudl + background_dudl;

    atomicAdd(&components[4], mesh_dudl);
    atomicAdd(&components[5], self_dudl);
    atomicAdd(&components[6], (double)dq);
}

/**
 * Compose atom-resolved PME outputs into the fused Hamiltonian accumulators.
 * The force kernel has already written direct-space/bonded components.  This
 * launch is ordered on the same stream and therefore performs an exact-once
 * addition before integration or boundary reduction.
 */
__global__ void pme_merge_lambda_atom_components(
    const double* __restrict__ pme_atom_energy,
    const double* __restrict__ pme_atom_dudlambda,
    const int* __restrict__ lambda_atom_slot,
    double* __restrict__ total_atom_energy,
    double* __restrict__ lambda_atom_energy,
    double* __restrict__ lambda_atom_dudlambda,
    int n_atoms
) {
    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= n_atoms) return;
    const double energy = pme_atom_energy[atom];
    total_atom_energy[atom] += energy;
    if (lambda_atom_slot[atom] >= 0) {
        lambda_atom_energy[atom] += energy;
        lambda_atom_dudlambda[atom] += pme_atom_dudlambda[atom];
    }
}

/**
 * Select one isolated slot charge plane.  This is the exact reciprocal-space
 * companion to the direct-space same-slot rule: atoms outside target_slot are
 * zero, so the PME result is the slot's complete mesh/images+self+background
 * operator without any environment or cross-slot term.
 * source_plane: 0=endpoint A, 1=endpoint B, 2=current q(lambda).
 */
__global__ void pme_select_isolated_lambda_slot_charges(
    const int* __restrict__ lambda_atom_slot,
    const float* __restrict__ endpoint_a_charges,
    const float* __restrict__ endpoint_b_charges,
    const float* __restrict__ current_charges,
    const float* __restrict__ current_charge_derivatives,
    int target_slot,
    int source_plane,
    float* __restrict__ isolated_charges,
    float* __restrict__ isolated_charge_derivatives,
    int n_atoms
) {
    const int atom = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (atom >= n_atoms) return;
    if (lambda_atom_slot[atom] != target_slot) {
        isolated_charges[atom] = 0.0f;
        isolated_charge_derivatives[atom] = 0.0f;
        return;
    }
    if (source_plane == 0) {
        isolated_charges[atom] = endpoint_a_charges[atom];
        isolated_charge_derivatives[atom] = 0.0f;
    } else if (source_plane == 1) {
        isolated_charges[atom] = endpoint_b_charges[atom];
        isolated_charge_derivatives[atom] = 0.0f;
    } else if (source_plane == 2) {
        isolated_charges[atom] = current_charges[atom];
        isolated_charge_derivatives[atom] = current_charge_derivatives[atom];
    } else {
        asm volatile("trap;");
    }
}

/**
 * Add one basis plane of
 *   U_internal_desired - U_PME(q_current_slot)
 * to the fused Hamiltonian.  endpoint_mask bit 0=A occupied, bit 1=B
 * occupied. A one-sided occupancy path copies the occupied intramolecular
 * operator to the empty endpoint; a two-sided identity path linearly
 * interpolates endpoint operators. The current isolated plane is always
 * subtracted, including its exact analytic charge derivative.
 */
__global__ void pme_accumulate_lambda_internal_correction(
    const double* __restrict__ pme_atom_energy,
    const double* __restrict__ pme_atom_dudlambda,
    const float* __restrict__ isolated_forces,
    const int* __restrict__ lambda_atom_slot,
    const float* __restrict__ lambda_slot_values,
    const int* __restrict__ endpoint_occupancy_masks,
    int target_slot,
    int source_plane,
    float* __restrict__ total_forces,
    double* __restrict__ total_atom_energy,
    double* __restrict__ lambda_atom_energy,
    double* __restrict__ lambda_atom_dudlambda,
    int n_atoms
) {
    const int atom = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (atom >= n_atoms) return;
    const float lambda = lambda_slot_values[target_slot];
    const int mask = endpoint_occupancy_masks[target_slot];
    if (!isfinite(lambda) || lambda < 0.0f || lambda > 1.0f
        || mask < 1 || mask > 3 || source_plane < 0 || source_plane > 2
        || (mask == 1 && source_plane != 0)
        || (mask == 2 && source_plane != 1))
        asm volatile("trap;");

    double energy_scale = 0.0;
    double energy_to_dudlambda_scale = 0.0;
    double pme_dudlambda_scale = 0.0;
    if (source_plane == 0) {
        if (mask == 1) {
            // q_current=(1-lambda)qA. Replace its quadratic isolated PME
            // operator with the persistent occupied-A internal operator using
            // one endpoint basis, without a redundant current-plane FFT.
            energy_scale = 2.0*(double)lambda
                - (double)lambda*(double)lambda;
            energy_to_dudlambda_scale = 2.0 - 2.0*(double)lambda;
        }
        else if (mask == 3) {
            energy_scale = 1.0 - (double)lambda;
            energy_to_dudlambda_scale = -1.0;
        }
    } else if (source_plane == 1) {
        if (mask == 2) {
            // q_current=lambda*qB. The exact correction is
            // (1-lambda^2) U_B with derivative -2 lambda U_B.
            energy_scale = 1.0 - (double)lambda*(double)lambda;
            energy_to_dudlambda_scale = -2.0*(double)lambda;
        }
        else if (mask == 3) {
            energy_scale = (double)lambda;
            energy_to_dudlambda_scale = 1.0;
        }
    } else {
        energy_scale = -1.0;
        pme_dudlambda_scale = -1.0;
    }

    total_forces[atom*3] += (float)energy_scale * isolated_forces[atom*3];
    total_forces[atom*3+1] += (float)energy_scale * isolated_forces[atom*3+1];
    total_forces[atom*3+2] += (float)energy_scale * isolated_forces[atom*3+2];
    const double energy = pme_atom_energy[atom];
    total_atom_energy[atom] += energy_scale * energy;
    if (lambda_atom_slot[atom] == target_slot) {
        lambda_atom_energy[atom] += energy_scale * energy;
        lambda_atom_dudlambda[atom] +=
            pme_dudlambda_scale * pme_atom_dudlambda[atom]
            + energy_to_dudlambda_scale * energy;
    }
}

/** Complete the uniform-neutralizing-plasma correction after Q reduction. */
__global__ void pme_finalize_net_charge_correction(
    double* __restrict__ components,
    float beta,
    float inv_volume
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    double q = components[2];
    double beta_d = (double)beta;
    double prefactor = -(double)ONE_4PI_EPS0 * (double)M_PI * (double)inv_volume;
    components[3] = prefactor * q * q / (2.0 * beta_d * beta_d);
    double dq = components[6];
    components[7] = prefactor * q * dq / (beta_d * beta_d);
}

/**
 * Zero the PME grid
 */
__global__ void pme_zero_grid(
    float* __restrict__ grid,
    int total_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        grid[idx] = 0.0f;
    }
}

}  // extern "C"
