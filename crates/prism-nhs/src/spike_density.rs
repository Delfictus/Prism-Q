//! GPU Spike Density Grid — SNDC Stage 2
//!
//! Converts sparse [`GpuSpikeEvent`] positions into a continuous 3D density field
//! using deterministic Gaussian splatting with intensity² weighting.
//!
//! Density accumulation is ordered on the host in f64, then uploaded as the
//! canonical f32 grid for GPU 3×3×3 non-maximum suppression.
//!
//! The density peaks ARE the binding hotspots — analogous to FTMap consensus sites
//! but derived from neuromorphic thermodynamic trapping events instead of probe docking.

use anyhow::{Context, Result};
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;

use crate::fused_engine::GpuSpikeEvent;

/// PTX compiled from cuda/spike_density.cu at build time
const SPIKE_DENSITY_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/spike_density.ptx"));

/// Default minimum density threshold for peak detection.
/// A single spike with intensity 1.0 produces density 1.0 at its center,
/// so this is the lowest meaningful signal.
const DEFAULT_MIN_DENSITY: f32 = 1.0;

/// 3D density grid computed entirely on GPU.
///
/// Grid layout: x-major row-major, `density[ix * Dy * Dz + iy * Dz + iz]`.
///
/// After construction, `d_density` holds the continuous density field and
/// `d_peaks` holds a binary mask of local maxima above the density threshold.
pub struct SpikeDensityGrid {
    /// Grid dimensions [Dx, Dy, Dz]
    pub dims: [u32; 3],
    /// Grid origin (min corner, Å)
    pub origin: [f32; 3],
    /// Voxel spacing (Å)
    pub spacing: f32,
    /// GPU buffer: density[x][y][z] = Σ intensity² × K(r)
    /// where K(r) is Gaussian kernel with width σ
    pub d_density: CudaSlice<f32>,
    /// GPU buffer: peak mask after non-maximum suppression (1 = peak, 0 = not)
    pub d_peaks: CudaSlice<u32>,

    // -- Internal state --
    sigma: f32,
    stream: Arc<CudaStream>,
    /// Keep CUDA module alive (functions reference it)
    _module: Arc<CudaModule>,
    /// Cached host-side density for sample_at() (lazy download)
    h_density: Option<Vec<f32>>,
}

impl SpikeDensityGrid {
    /// Build a spike density grid from spike events on GPU.
    ///
    /// Computes the bounding box from protein positions, adds a 3σ margin,
    /// splatters spikes with Gaussian weighting, and finds density peaks.
    ///
    /// # Arguments
    /// * `spikes` — spike events from simulation (may be empty)
    /// * `protein_positions` — flattened `[x0,y0,z0, x1,y1,z1, ...]` atom coords for bounding box
    /// * `spacing` — voxel edge length in Å (typically 1.0)
    /// * `sigma` — Gaussian kernel σ in Å (typically 2.0)
    /// * `context` — shared CUDA context
    pub fn from_spikes(
        spikes: &[GpuSpikeEvent],
        protein_positions: &[f32],
        spacing: f32,
        sigma: f32,
        context: Arc<CudaContext>,
    ) -> Result<Self> {
        let stream = context.default_stream();

        // ── Load CUDA module from embedded PTX ─────────────────────────
        let ptx = Ptx::from_src(SPIKE_DENSITY_PTX);
        let module = context
            .load_module(ptx)
            .context("Failed to load spike_density PTX module")?;
        let fn_find_peaks: CudaFunction = module
            .load_function("find_density_peaks")
            .context("Failed to load find_density_peaks kernel")?;

        // ── Compute bounding box from protein positions ────────────────
        let n_atoms = protein_positions.len() / 3;
        anyhow::ensure!(n_atoms > 0, "No protein positions provided");

        let mut bbox_min = [f32::MAX; 3];
        let mut bbox_max = [f32::MIN; 3];
        for i in 0..n_atoms {
            for d in 0..3 {
                let v = protein_positions[i * 3 + d];
                if v < bbox_min[d] {
                    bbox_min[d] = v;
                }
                if v > bbox_max[d] {
                    bbox_max[d] = v;
                }
            }
        }

        // Add margin: 3σ + spacing ensures Gaussian splatting doesn't clip at edges
        let margin = 3.0 * sigma + spacing;
        let origin = [
            bbox_min[0] - margin,
            bbox_min[1] - margin,
            bbox_min[2] - margin,
        ];
        let dims = [
            ((bbox_max[0] + margin - origin[0]) / spacing).ceil() as u32 + 1,
            ((bbox_max[1] + margin - origin[1]) / spacing).ceil() as u32 + 1,
            ((bbox_max[2] + margin - origin[2]) / spacing).ceil() as u32 + 1,
        ];
        let grid_size = (dims[0] as usize) * (dims[1] as usize) * (dims[2] as usize);

        log::info!(
            "  SpikeDensityGrid: {}x{}x{} = {} voxels (spacing={:.1}Å, σ={:.1}Å, {} spikes)",
            dims[0],
            dims[1],
            dims[2],
            grid_size,
            spacing,
            sigma,
            spikes.len()
        );

        // ── Deterministic density accumulation ─────────────────────────
        //
        // The previous GPU scatter used unordered f32 accumulation into voxels,
        // making peak labels order-dependent under high spike density. Keep
        // the same Gaussian support and x-major layout, but accumulate in a
        // single fixed spike/voxel order with f64 precision before uploading
        // the canonical f32 grid for GPU NMS.
        let mut density_f64 = vec![0.0f64; grid_size];
        if !spikes.is_empty() {
            let inv_2sigma2 = 1.0f64 / (2.0f64 * sigma as f64 * sigma as f64);
            let radius = (3.0f32 * sigma / spacing).ceil() as i32;
            let dx_n = dims[0] as i32;
            let dy_n = dims[1] as i32;
            let dz_n = dims[2] as i32;
            let origin64 = [origin[0] as f64, origin[1] as f64, origin[2] as f64];
            let spacing64 = spacing as f64;

            for spike in spikes {
                let px = spike.position[0] as f64;
                let py = spike.position[1] as f64;
                let pz = spike.position[2] as f64;
                let w = spike.intensity as f64 * spike.intensity as f64;

                let cx = ((px - origin64[0]) / spacing64) as i32;
                let cy = ((py - origin64[1]) / spacing64) as i32;
                let cz = ((pz - origin64[2]) / spacing64) as i32;

                for ox in -radius..=radius {
                    let ix = cx + ox;
                    if ix < 0 || ix >= dx_n {
                        continue;
                    }
                    let fx = (ix as f64 * spacing64 + origin64[0]) - px;
                    for oy in -radius..=radius {
                        let iy = cy + oy;
                        if iy < 0 || iy >= dy_n {
                            continue;
                        }
                        let fy = (iy as f64 * spacing64 + origin64[1]) - py;
                        for oz in -radius..=radius {
                            let iz = cz + oz;
                            if iz < 0 || iz >= dz_n {
                                continue;
                            }
                            let fz = (iz as f64 * spacing64 + origin64[2]) - pz;
                            let r2 = fx * fx + fy * fy + fz * fz;
                            let val = w * (-r2 * inv_2sigma2).exp();
                            let idx = (ix as usize) * (dims[1] as usize) * (dims[2] as usize)
                                + (iy as usize) * (dims[2] as usize)
                                + iz as usize;
                            density_f64[idx] += val;
                        }
                    }
                }
            }
        }
        let density_host: Vec<f32> = density_f64.into_iter().map(|v| v as f32).collect();
        let d_density: CudaSlice<f32> = stream.clone_htod(&density_host)?;

        // ── Find density peaks via 3D NMS ──────────────────────────────
        let d_peaks: CudaSlice<u32> = stream.clone_htod(&vec![0u32; grid_size])?;

        let min_density = DEFAULT_MIN_DENSITY;
        let dx = dims[0] as i32;
        let dy = dims[1] as i32;
        let dz = dims[2] as i32;

        let block_dim = 8u32;
        let grid_x = (dims[0] + block_dim - 1) / block_dim;
        let grid_y = (dims[1] + block_dim - 1) / block_dim;
        let grid_z = (dims[2] + block_dim - 1) / block_dim;

        unsafe {
            stream
                .launch_builder(&fn_find_peaks)
                .arg(&d_density)
                .arg(&d_peaks)
                .arg(&dx)
                .arg(&dy)
                .arg(&dz)
                .arg(&min_density)
                .launch(LaunchConfig {
                    grid_dim: (grid_x, grid_y, grid_z),
                    block_dim: (block_dim, block_dim, block_dim),
                    shared_mem_bytes: 0,
                })
                .context("Failed to launch find_density_peaks")?;
        }

        stream.synchronize()?;

        Ok(Self {
            dims,
            origin,
            spacing,
            sigma,
            d_density,
            d_peaks,
            stream,
            _module: module,
            h_density: Some(density_host),
        })
    }

    /// Count density peaks (local maxima above the minimum density threshold).
    pub fn count_peaks(&self) -> Result<usize> {
        let grid_size = self.grid_size();
        let mut peaks_host = vec![0u32; grid_size];
        self.stream.memcpy_dtoh(&self.d_peaks, &mut peaks_host)?;
        Ok(peaks_host.iter().filter(|&&v| v != 0).count())
    }

    /// Sample density at an arbitrary coordinate using trilinear interpolation.
    ///
    /// The density grid is downloaded from GPU on first call and cached
    /// for subsequent calls.
    pub fn sample_at(&mut self, point: [f32; 3]) -> Result<f32> {
        if self.h_density.is_none() {
            let grid_size = self.grid_size();
            let mut density = vec![0.0f32; grid_size];
            self.stream.memcpy_dtoh(&self.d_density, &mut density)?;
            self.h_density = Some(density);
        }
        let density = self.h_density.as_ref().unwrap();
        Ok(trilinear_sample(
            density,
            self.dims,
            self.origin,
            self.spacing,
            point,
        ))
    }

    /// Total number of voxels in the grid.
    #[inline]
    pub fn grid_size(&self) -> usize {
        (self.dims[0] as usize) * (self.dims[1] as usize) * (self.dims[2] as usize)
    }

    /// Gaussian sigma used during construction.
    #[inline]
    pub fn sigma(&self) -> f32 {
        self.sigma
    }

    /// Get peak positions in world coordinates (Å).
    ///
    /// Downloads the peak mask from GPU and converts non-zero voxel indices
    /// back to Cartesian coordinates.
    pub fn get_peak_positions(&self) -> Result<Vec<[f32; 3]>> {
        let grid_size = self.grid_size();
        let mut peaks_host = vec![0u32; grid_size];
        self.stream.memcpy_dtoh(&self.d_peaks, &mut peaks_host)?;

        let dy = self.dims[1] as usize;
        let dz = self.dims[2] as usize;

        let mut positions = Vec::new();
        for (flat_idx, &v) in peaks_host.iter().enumerate() {
            if v != 0 {
                let ix = flat_idx / (dy * dz);
                let iy = (flat_idx % (dy * dz)) / dz;
                let iz = flat_idx % dz;
                positions.push([
                    self.origin[0] + ix as f32 * self.spacing,
                    self.origin[1] + iy as f32 * self.spacing,
                    self.origin[2] + iz as f32 * self.spacing,
                ]);
            }
        }
        Ok(positions)
    }
}

// ============================================================================
// Trilinear interpolation (CPU helper)
// ============================================================================

/// Trilinear interpolation on a 3D grid (x-major row-major layout).
///
/// Index layout: `density[ix * Dy * Dz + iy * Dz + iz]`
///
/// Points outside the grid are clamped to the nearest boundary voxel.
fn trilinear_sample(
    density: &[f32],
    dims: [u32; 3],
    origin: [f32; 3],
    spacing: f32,
    point: [f32; 3],
) -> f32 {
    let dy = dims[1] as usize;
    let dz = dims[2] as usize;

    // Convert to continuous grid coordinates
    let gx = (point[0] - origin[0]) / spacing;
    let gy = (point[1] - origin[1]) / spacing;
    let gz = (point[2] - origin[2]) / spacing;

    // Clamp to valid range [0, dim-1]
    let gx = gx.max(0.0).min((dims[0] - 1) as f32);
    let gy = gy.max(0.0).min((dims[1] - 1) as f32);
    let gz = gz.max(0.0).min((dims[2] - 1) as f32);

    // Integer corners
    let ix0 = gx.floor() as usize;
    let iy0 = gy.floor() as usize;
    let iz0 = gz.floor() as usize;
    let ix1 = (ix0 + 1).min(dims[0] as usize - 1);
    let iy1 = (iy0 + 1).min(dims[1] as usize - 1);
    let iz1 = (iz0 + 1).min(dims[2] as usize - 1);

    // Fractional offsets
    let xd = gx - ix0 as f32;
    let yd = gy - iy0 as f32;
    let zd = gz - iz0 as f32;

    let idx = |x: usize, y: usize, z: usize| -> usize { x * dy * dz + y * dz + z };

    // Sample 8 corners
    let c000 = density[idx(ix0, iy0, iz0)];
    let c001 = density[idx(ix0, iy0, iz1)];
    let c010 = density[idx(ix0, iy1, iz0)];
    let c011 = density[idx(ix0, iy1, iz1)];
    let c100 = density[idx(ix1, iy0, iz0)];
    let c101 = density[idx(ix1, iy0, iz1)];
    let c110 = density[idx(ix1, iy1, iz0)];
    let c111 = density[idx(ix1, iy1, iz1)];

    // Interpolate along z, then y, then x
    let c00 = c000 * (1.0 - zd) + c001 * zd;
    let c01 = c010 * (1.0 - zd) + c011 * zd;
    let c10 = c100 * (1.0 - zd) + c101 * zd;
    let c11 = c110 * (1.0 - zd) + c111 * zd;

    let c0 = c00 * (1.0 - yd) + c01 * yd;
    let c1 = c10 * (1.0 - yd) + c11 * yd;

    c0 * (1.0 - xd) + c1 * xd
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trilinear_at_center() {
        // 3×3×3 grid, spacing=1Å, origin=(0,0,0)
        // Only center voxel (1,1,1) has density 100.0
        let dims = [3u32, 3, 3];
        let origin = [0.0f32; 3];
        let spacing = 1.0;

        let mut density = vec![0.0f32; 27];
        density[1 * 3 * 3 + 1 * 3 + 1] = 100.0;

        // Exact center → 100.0
        let val = trilinear_sample(&density, dims, origin, spacing, [1.0, 1.0, 1.0]);
        assert!((val - 100.0).abs() < 0.01, "center: {}", val);

        // Halfway to (2,1,1) → 50.0 (linear interpolation)
        let val = trilinear_sample(&density, dims, origin, spacing, [1.5, 1.0, 1.0]);
        assert!((val - 50.0).abs() < 0.01, "half-x: {}", val);

        // Corner (0,0,0) → 0.0
        let val = trilinear_sample(&density, dims, origin, spacing, [0.0, 0.0, 0.0]);
        assert!(val.abs() < 0.01, "corner: {}", val);

        // Body center of the 2×2×2 sub-cube around (1,1,1) → 100/8 = 12.5
        let val = trilinear_sample(&density, dims, origin, spacing, [0.5, 0.5, 0.5]);
        assert!((val - 12.5).abs() < 0.01, "sub-cube center: {}", val);
    }

    #[test]
    fn test_trilinear_uniform() {
        // Uniform density 42.0 → any interior point samples to 42.0
        let dims = [5u32, 5, 5];
        let density = vec![42.0f32; 125];
        let origin = [0.0f32; 3];
        let spacing = 1.0;

        let val = trilinear_sample(&density, dims, origin, spacing, [2.3, 1.7, 3.1]);
        assert!((val - 42.0).abs() < 0.01, "uniform: {}", val);
    }

    #[test]
    fn test_trilinear_clamping() {
        // Point outside grid should clamp to boundary
        let dims = [3u32, 3, 3];
        let mut density = vec![0.0f32; 27];
        density[0] = 99.0; // (0,0,0) corner
        let origin = [0.0f32; 3];
        let spacing = 1.0;

        // Way outside: should clamp to (0,0,0) → 99.0
        let val = trilinear_sample(&density, dims, origin, spacing, [-10.0, -10.0, -10.0]);
        assert!((val - 99.0).abs() < 0.01, "clamped: {}", val);
    }

    #[test]
    fn test_grid_dimension_math() {
        // Verify the bounding-box → grid-dimension formula
        let spacing = 1.0f32;
        let sigma = 2.0f32;
        let margin = 3.0 * sigma + spacing; // 7.0

        // Protein bbox: [5, 5, 5] to [25, 20, 15]
        let bbox_min = [5.0f32, 5.0, 5.0];
        let bbox_max = [25.0f32, 20.0, 15.0];

        let origin = [
            bbox_min[0] - margin,
            bbox_min[1] - margin,
            bbox_min[2] - margin,
        ];
        let dims: [u32; 3] = [
            ((bbox_max[0] + margin - origin[0]) / spacing).ceil() as u32 + 1,
            ((bbox_max[1] + margin - origin[1]) / spacing).ceil() as u32 + 1,
            ((bbox_max[2] + margin - origin[2]) / spacing).ceil() as u32 + 1,
        ];

        // X: (25+7) - (5-7) = 34 → ceil(34/1) + 1 = 35
        assert_eq!(dims[0], 35);
        // Y: (20+7) - (5-7) = 29 → ceil(29/1) + 1 = 30
        assert_eq!(dims[1], 30);
        // Z: (15+7) - (5-7) = 24 → ceil(24/1) + 1 = 25
        assert_eq!(dims[2], 25);
    }

    /// Full GPU integration test: synthetic spikes → density grid → peak detection.
    ///
    /// Requires a CUDA-capable GPU. Run with:
    ///   cargo test -p prism-nhs --features gpu -- --ignored test_synthetic_peaks_gpu
    #[test]
    #[ignore]
    fn test_synthetic_peaks_gpu() {
        use crate::fused_engine::GpuSpikeEvent;

        let context = CudaContext::new(0).expect("No CUDA device available");

        // Protein bounding box: 30×30×30 Å cube at (10..40, 10..40, 10..40)
        let mut protein_positions = Vec::new();
        for x in (10..=40).step_by(5) {
            for y in (10..=40).step_by(5) {
                for z in (10..=40).step_by(5) {
                    protein_positions.push(x as f32);
                    protein_positions.push(y as f32);
                    protein_positions.push(z as f32);
                }
            }
        }

        // Create 3 spike clusters at known locations with known intensities
        let cluster_a = [15.0f32, 15.0, 15.0]; // cluster A
        let cluster_b = [35.0f32, 35.0, 35.0]; // cluster B — far from A
        let cluster_c = [25.0f32, 15.0, 35.0]; // cluster C — equidistant

        let mut spikes = Vec::new();
        for (center, intensity) in [
            (cluster_a, 10.0f32),
            (cluster_b, 8.0f32),
            (cluster_c, 12.0f32),
        ] {
            // 25 spikes per cluster, tightly grouped (±0.3Å jitter)
            for i in 0..25 {
                let jitter = (i as f32 - 12.0) * 0.025;
                let mut spike = GpuSpikeEvent::default();
                spike.position = [
                    center[0] + jitter * ((i % 3) as f32 - 1.0),
                    center[1] + jitter * (((i + 1) % 3) as f32 - 1.0),
                    center[2] + jitter * (((i + 2) % 3) as f32 - 1.0),
                ];
                spike.intensity = intensity;
                spikes.push(spike);
            }
        }

        // Build density grid
        let mut grid = SpikeDensityGrid::from_spikes(
            &spikes,
            &protein_positions,
            1.0, // 1Å spacing
            2.0, // 2Å sigma
            context,
        )
        .expect("Failed to build density grid");

        // ── Verify grid dimensions are reasonable ──────────────────────
        assert!(grid.dims[0] > 30, "grid too small in X: {}", grid.dims[0]);
        assert!(grid.dims[1] > 30, "grid too small in Y: {}", grid.dims[1]);
        assert!(grid.dims[2] > 30, "grid too small in Z: {}", grid.dims[2]);

        // ── Verify peaks detected ──────────────────────────────────────
        let n_peaks = grid.count_peaks().expect("count_peaks failed");
        assert!(
            n_peaks >= 3,
            "Expected ≥3 peaks for 3 clusters, got {}",
            n_peaks
        );

        // ── Verify high density at cluster centers ─────────────────────
        // 25 spikes × intensity² at center: cluster C (intensity=12) → 25 × 144 = 3600
        for (label, center) in [("A", cluster_a), ("B", cluster_b), ("C", cluster_c)] {
            let d = grid.sample_at(center).expect("sample_at failed");
            assert!(
                d > 50.0,
                "Cluster {} at {:?}: expected high density, got {:.1}",
                label,
                center,
                d
            );
        }

        // ── Verify near-zero density far from clusters ─────────────────
        let empty_point = [50.0f32, 50.0, 50.0]; // well outside protein
        let d = grid.sample_at(empty_point).expect("sample_at failed");
        assert!(d < 1.0, "Expected ~0 at empty point, got {:.1}", d);

        // ── Verify peak positions are near cluster centers ─────────────
        let peak_positions = grid
            .get_peak_positions()
            .expect("get_peak_positions failed");
        for (label, center) in [("A", cluster_a), ("B", cluster_b), ("C", cluster_c)] {
            let nearest = peak_positions
                .iter()
                .map(|p| {
                    let dx = p[0] - center[0];
                    let dy = p[1] - center[1];
                    let dz = p[2] - center[2];
                    (dx * dx + dy * dy + dz * dz).sqrt()
                })
                .fold(f32::MAX, f32::min);
            assert!(
                nearest < 3.0,
                "Cluster {} at {:?}: nearest peak {:.1}Å away (expected <3Å)",
                label,
                center,
                nearest
            );
        }

        println!(
            "GPU test passed: {} peaks found for 3 clusters ({} grid voxels)",
            n_peaks,
            grid.grid_size()
        );
    }
}
