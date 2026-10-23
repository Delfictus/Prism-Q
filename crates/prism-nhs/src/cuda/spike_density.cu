// SNDC Stage 2: Spike Density Grid CUDA Kernels
//
// 3D non-maximum suppression for deterministic spike-density peak finding.
//
// Density splatting is performed in ordered f64 on the Rust side before the
// canonical f32 grid is uploaded. This file intentionally keeps only the GPU
// NMS kernel, which has no cross-thread floating-point accumulation.

#include <stdint.h>

extern "C" {

// 3D non-maximum suppression to find density peaks
// A voxel is a peak if it's the maximum in its 3×3×3 neighborhood
// These peaks seed the hierarchical clustering in Stage 3
__global__ void find_density_peaks(
    const float* __restrict__ density_grid,
    uint32_t* __restrict__ peak_mask,
    int Dx, int Dy, int Dz,
    float min_density
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= Dx || iy >= Dy || iz >= Dz) return;

    float val = density_grid[ix * Dy * Dz + iy * Dz + iz];
    if (val < min_density) { peak_mask[ix * Dy * Dz + iy * Dz + iz] = 0; return; }

    bool is_max = true;
    for (int dx = -1; dx <= 1 && is_max; dx++)
        for (int dy = -1; dy <= 1 && is_max; dy++)
            for (int dz = -1; dz <= 1 && is_max; dz++) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                int nx = ix+dx, ny = iy+dy, nz = iz+dz;
                if (nx >= 0 && nx < Dx && ny >= 0 && ny < Dy && nz >= 0 && nz < Dz)
                    if (density_grid[nx * Dy * Dz + ny * Dz + nz] > val)
                        is_max = false;
            }
    peak_mask[ix * Dy * Dz + iy * Dz + iz] = is_max ? 1 : 0;
}

} // extern "C"
