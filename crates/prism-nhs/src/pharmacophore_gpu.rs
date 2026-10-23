//! GPU-accelerated pharmacophore hotspot map extraction.
//! Replaces Python pharmacophore_extract.py — ~1000x faster.

use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "gpu")]
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
#[cfg(feature = "gpu")]
use cudarc::nvrtc::Ptx;

/// Spike data in SoA layout for GPU upload.
pub struct SpikeData {
    pub pos_x: Vec<f32>,
    pub pos_y: Vec<f32>,
    pub pos_z: Vec<f32>,
    pub intensities: Vec<f32>,
    pub types: Vec<i32>,
    pub frame_indices: Vec<i32>,
}

impl SpikeData {
    pub fn len(&self) -> usize {
        self.pos_x.len()
    }
    pub fn is_empty(&self) -> bool {
        self.pos_x.is_empty()
    }

    /// Build from in-memory spike events (no JSON roundtrip).
    #[cfg(feature = "gpu")]
    pub fn from_gpu_spikes(
        spikes: &[super::fused_engine::GpuSpikeEvent],
        centroid: [f32; 3],
        radius: f32,
    ) -> Self {
        let r2 = radius * radius;
        let mut data = SpikeData {
            pos_x: Vec::with_capacity(spikes.len()),
            pos_y: Vec::with_capacity(spikes.len()),
            pos_z: Vec::with_capacity(spikes.len()),
            intensities: Vec::with_capacity(spikes.len()),
            types: Vec::with_capacity(spikes.len()),
            frame_indices: Vec::with_capacity(spikes.len()),
        };
        for s in spikes {
            let dx = s.position[0] - centroid[0];
            let dy = s.position[1] - centroid[1];
            let dz = s.position[2] - centroid[2];
            if dx * dx + dy * dy + dz * dz <= r2 {
                data.pos_x.push(s.position[0]);
                data.pos_y.push(s.position[1]);
                data.pos_z.push(s.position[2]);
                data.intensities.push(s.intensity);
                data.types.push(s.aromatic_type);
                data.frame_indices.push(s.timestep / 1000);
            }
        }
        data
    }

    /// Build from spike events JSON (standalone mode).
    pub fn from_json(path: &Path) -> anyhow::Result<(Self, [f32; 3], i32)> {
        log::info!("Loading spike events from {}...", path.display());
        let t = Instant::now();
        let data: serde_json::Value =
            serde_json::from_reader(std::io::BufReader::new(std::fs::File::open(path)?))?;
        log::info!("  JSON parsed in {:.1}s", t.elapsed().as_secs_f64());

        let ca = data["centroid"].as_array().unwrap();
        let centroid = [
            ca[0].as_f64().unwrap() as f32,
            ca[1].as_f64().unwrap() as f32,
            ca[2].as_f64().unwrap() as f32,
        ];
        let site_id = data["site_id"].as_i64().unwrap() as i32;
        let spikes = data["spikes"].as_array().unwrap();
        let n = spikes.len();

        let type_to_code = |t: &str| -> i32 {
            match t {
                "TRP" => 0,
                "TYR" => 1,
                "PHE" => 2,
                "SS" => 3,
                "BNZ" => 4,
                "CATION" => 5,
                "ANION" => 6,
                _ => 7,
            }
        };

        let mut sd = SpikeData {
            pos_x: Vec::with_capacity(n),
            pos_y: Vec::with_capacity(n),
            pos_z: Vec::with_capacity(n),
            intensities: Vec::with_capacity(n),
            types: Vec::with_capacity(n),
            frame_indices: Vec::with_capacity(n),
        };
        for s in spikes {
            sd.pos_x.push(s["x"].as_f64().unwrap() as f32);
            sd.pos_y.push(s["y"].as_f64().unwrap() as f32);
            sd.pos_z.push(s["z"].as_f64().unwrap() as f32);
            sd.intensities.push(s["intensity"].as_f64().unwrap() as f32);
            sd.types
                .push(type_to_code(s["type"].as_str().unwrap_or("UNK")));
            sd.frame_indices
                .push(s["frame_index"].as_i64().unwrap_or(0) as i32);
        }
        log::info!("  {} spikes loaded in {:.1}s", n, t.elapsed().as_secs_f64());
        Ok((sd, centroid, site_id))
    }
}

/// 3D density grid.
pub struct DensityGrid {
    pub data: Vec<f32>,
    pub origin: [f32; 3],
    pub dims: [usize; 3],
    pub spacing: f32,
    pub max_val: f32,
}

/// GPU pharmacophore engine.
#[cfg(feature = "gpu")]
pub struct PharmacophoreGpu {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    splat_kernel: CudaFunction,
    splat_typed_kernel: CudaFunction,
}

#[cfg(feature = "gpu")]
impl PharmacophoreGpu {
    pub fn new(context: Arc<CudaContext>, stream: Arc<CudaStream>) -> anyhow::Result<Self> {
        // Load PTX — same path resolution as fused_engine
        let ptx_paths = vec![
            std::path::PathBuf::from("target/ptx/pharmacophore_splat.ptx"),
            std::path::PathBuf::from("../../target/ptx/pharmacophore_splat.ptx"),
        ];

        let mut module: Option<Arc<CudaModule>> = None;
        if let Ok(env_dir) = std::env::var("PRISM4D_PTX_DIR") {
            let p = std::path::PathBuf::from(&env_dir).join("pharmacophore_splat.ptx");
            if p.exists() {
                module = Some(context.load_module(Ptx::from_file(&p.display().to_string()))?);
            }
        }
        if module.is_none() {
            for p in &ptx_paths {
                if p.exists() {
                    if let Ok(m) = context.load_module(Ptx::from_file(&p.display().to_string())) {
                        module = Some(m);
                        break;
                    }
                }
            }
        }
        let module = module.ok_or_else(|| anyhow::anyhow!("pharmacophore_splat.ptx not found"))?;

        Ok(Self {
            splat_kernel: module.load_function("gaussian_splat")?,
            splat_typed_kernel: module.load_function("gaussian_splat_typed")?,
            context,
            stream,
        })
    }

    /// Build combined density grid on GPU.
    pub fn build_density_grid(
        &self,
        spikes: &SpikeData,
        spacing: f32,
        sigma: f32,
    ) -> anyhow::Result<DensityGrid> {
        self.splat_impl(spikes, spacing, sigma, None)
    }

    /// Build per-type density grid on GPU.
    pub fn build_density_grid_typed(
        &self,
        spikes: &SpikeData,
        spacing: f32,
        sigma: f32,
        type_code: i32,
    ) -> anyhow::Result<DensityGrid> {
        self.splat_impl(spikes, spacing, sigma, Some(type_code))
    }

    fn splat_impl(
        &self,
        spikes: &SpikeData,
        spacing: f32,
        sigma: f32,
        type_filter: Option<i32>,
    ) -> anyhow::Result<DensityGrid> {
        let n = spikes.len();
        if n == 0 {
            return Ok(DensityGrid {
                data: vec![0.0],
                origin: [0.0; 3],
                dims: [1, 1, 1],
                spacing,
                max_val: 0.0,
            });
        }

        let pad = 4.0f32;
        let min_x = spikes.pos_x.iter().cloned().fold(f32::INFINITY, f32::min) - pad;
        let min_y = spikes.pos_y.iter().cloned().fold(f32::INFINITY, f32::min) - pad;
        let min_z = spikes.pos_z.iter().cloned().fold(f32::INFINITY, f32::min) - pad;
        let max_x = spikes
            .pos_x
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
            + pad;
        let max_y = spikes
            .pos_y
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
            + pad;
        let max_z = spikes
            .pos_z
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
            + pad;

        let nx = ((max_x - min_x) / spacing).ceil() as usize + 1;
        let ny = ((max_y - min_y) / spacing).ceil() as usize + 1;
        let nz = ((max_z - min_z) / spacing).ceil() as usize + 1;
        let n_voxels = nx * ny * nz;
        let inv_2sigma2 = 1.0f32 / (2.0 * sigma * sigma);
        let cutoff = (3.0 * sigma / spacing).ceil() as i32 + 1;

        // Upload to GPU
        let mut d_px: CudaSlice<f32> = self.stream.alloc_zeros(n)?;
        let mut d_py: CudaSlice<f32> = self.stream.alloc_zeros(n)?;
        let mut d_pz: CudaSlice<f32> = self.stream.alloc_zeros(n)?;
        let mut d_int: CudaSlice<f32> = self.stream.alloc_zeros(n)?;
        self.stream.memcpy_htod(&spikes.pos_x, &mut d_px)?;
        self.stream.memcpy_htod(&spikes.pos_y, &mut d_py)?;
        self.stream.memcpy_htod(&spikes.pos_z, &mut d_pz)?;
        self.stream.memcpy_htod(&spikes.intensities, &mut d_int)?;

        let d_grid: CudaSlice<f32> = self.stream.alloc_zeros(n_voxels)?;

        let block = 256u32;
        let grid_dim = ((n as u32 + block - 1) / block, 1, 1);
        let cfg = LaunchConfig {
            grid_dim,
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };

        let ni = n as i32;
        let nxi = nx as i32;
        let nyi = ny as i32;
        let nzi = nz as i32;

        if let Some(tc) = type_filter {
            let mut d_types: CudaSlice<i32> = self.stream.alloc_zeros(n)?;
            self.stream.memcpy_htod(&spikes.types, &mut d_types)?;
            unsafe {
                let mut b = self.stream.launch_builder(&self.splat_typed_kernel);
                b.arg(&d_px);
                b.arg(&d_py);
                b.arg(&d_pz);
                b.arg(&d_int);
                b.arg(&d_types);
                b.arg(&tc);
                b.arg(&ni);
                b.arg(&d_grid);
                b.arg(&nxi);
                b.arg(&nyi);
                b.arg(&nzi);
                b.arg(&min_x);
                b.arg(&min_y);
                b.arg(&min_z);
                b.arg(&spacing);
                b.arg(&inv_2sigma2);
                b.arg(&cutoff);
                b.launch(cfg)?;
            }
        } else {
            unsafe {
                let mut b = self.stream.launch_builder(&self.splat_kernel);
                b.arg(&d_px);
                b.arg(&d_py);
                b.arg(&d_pz);
                b.arg(&d_int);
                b.arg(&ni);
                b.arg(&d_grid);
                b.arg(&nxi);
                b.arg(&nyi);
                b.arg(&nzi);
                b.arg(&min_x);
                b.arg(&min_y);
                b.arg(&min_z);
                b.arg(&spacing);
                b.arg(&inv_2sigma2);
                b.arg(&cutoff);
                b.launch(cfg)?;
            }
        }

        let mut grid_data = vec![0.0f32; n_voxels];
        self.stream.memcpy_dtoh(&d_grid, &mut grid_data)?;
        self.stream.synchronize()?;

        let max_val = grid_data.iter().cloned().fold(0.0f32, f32::max);
        Ok(DensityGrid {
            data: grid_data,
            origin: [min_x, min_y, min_z],
            dims: [nx, ny, nz],
            spacing,
            max_val,
        })
    }
}

// ── DX writer ──────────────────────────────────────────────────────

pub fn write_dx(grid: &DensityGrid, filepath: &Path, comment: &str) -> anyhow::Result<()> {
    let [nx, ny, nz] = grid.dims;
    let f = std::fs::File::create(filepath)?;
    let mut w = BufWriter::with_capacity(1 << 20, f);
    writeln!(w, "# {}", comment)?;
    writeln!(
        w,
        "object 1 class gridpositions counts {} {} {}",
        nx, ny, nz
    )?;
    writeln!(
        w,
        "origin {:.6} {:.6} {:.6}",
        grid.origin[0], grid.origin[1], grid.origin[2]
    )?;
    writeln!(w, "delta {:.6} 0.000000 0.000000", grid.spacing)?;
    writeln!(w, "delta 0.000000 {:.6} 0.000000", grid.spacing)?;
    writeln!(w, "delta 0.000000 0.000000 {:.6}", grid.spacing)?;
    writeln!(
        w,
        "object 2 class gridconnections counts {} {} {}",
        nx, ny, nz
    )?;
    writeln!(
        w,
        "object 3 class array type double rank 0 items {} data follows",
        nx * ny * nz
    )?;
    let mut count = 0;
    for val in &grid.data {
        write!(w, "{:.6e}", val)?;
        count += 1;
        if count % 3 == 0 {
            writeln!(w)?;
        } else {
            write!(w, " ")?;
        }
    }
    if count % 3 != 0 {
        writeln!(w)?;
    }
    writeln!(w, "attribute \"dep\" string \"positions\"")?;
    writeln!(
        w,
        "object \"regular positions regular connections\" class field"
    )?;
    writeln!(w, "component \"positions\" value 1")?;
    writeln!(w, "component \"connections\" value 2")?;
    writeln!(w, "component \"data\" value 3")?;
    Ok(())
}

// ── Frame finder ───────────────────────────────────────────────────

pub fn find_open_frame(spikes: &SpikeData, centroid: [f32; 3]) -> (i32, Vec<(i32, f64)>) {
    let mut scores: HashMap<i32, f64> = HashMap::new();
    let [cx, cy, cz] = centroid;
    for i in 0..spikes.len() {
        let dx = spikes.pos_x[i] - cx;
        let dy = spikes.pos_y[i] - cy;
        let dz = spikes.pos_z[i] - cz;
        if dx * dx + dy * dy + dz * dz < 36.0 {
            *scores.entry(spikes.frame_indices[i]).or_default() += spikes.intensities[i] as f64;
        }
    }
    let mut sorted: Vec<_> = scores.into_iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let best = sorted.first().map(|x| x.0).unwrap_or(0);
    (best, sorted)
}

// ── PyMOL script ───────────────────────────────────────────────────

pub fn write_pymol_script(
    output_dir: &Path,
    site_id: i32,
    centroid: [f32; 3],
    receptor_pdb: &str,
    combined_max: f32,
    type_grids: &[(String, f32, usize)],
) -> anyhow::Result<PathBuf> {
    let p = output_dir.join(format!("site{}_visualize.pml", site_id));
    let mut w = BufWriter::new(std::fs::File::create(&p)?);
    writeln!(w, "# PRISM-4D Site {} Pharmacophore (GPU)", site_id)?;
    writeln!(w, "load {}, receptor", receptor_pdb)?;
    writeln!(
        w,
        "color gray80, receptor\nshow cartoon, receptor\nhide lines, receptor\n"
    )?;
    writeln!(w, "load site{}_combined.dx, combined_hotspot", site_id)?;
    writeln!(
        w,
        "isosurface combined_surf, combined_hotspot, {:.1}",
        combined_max * 0.3
    )?;
    writeln!(
        w,
        "color red, combined_surf\nset transparency, 0.4, combined_surf\n"
    )?;
    for (name, mx, count) in type_grids {
        let dx = format!("site{}_{}", site_id, name.to_lowercase());
        let c = match name.as_str() {
            "BNZ" => "orange",
            "TYR" => "cyan",
            "TRP" => "magenta",
            "PHE" => "yellow",
            "ANION" => "red",
            "CATION" => "blue",
            _ => "white",
        };
        writeln!(w, "# {} ({} spikes)\nload {}.dx, {}", name, count, dx, dx)?;
        writeln!(w, "isosurface {}_surf, {}, {:.1}", dx, dx, mx * 0.25)?;
        writeln!(
            w,
            "color {}, {}_surf\nset transparency, 0.5, {}_surf\n",
            c, dx, dx
        )?;
    }
    writeln!(
        w,
        "pseudoatom centroid, pos=[{:.3}, {:.3}, {:.3}]",
        centroid[0], centroid[1], centroid[2]
    )?;
    writeln!(
        w,
        "show spheres, centroid\ncolor red, centroid\nset sphere_scale, 0.5, centroid"
    )?;
    writeln!(
        w,
        "select pocket, receptor within 8.0 of centroid\nshow sticks, pocket"
    )?;
    writeln!(
        w,
        "color palegreen, pocket and elem C\ncenter centroid\nzoom centroid, 15"
    )?;
    writeln!(w, "bg_color white")?;
    Ok(p)
}

// ── Full pipeline ──────────────────────────────────────────────────

const TYPE_INFO: &[(&str, i32)] = &[
    ("TRP", 0),
    ("TYR", 1),
    ("PHE", 2),
    ("SS", 3),
    ("BNZ", 4),
    ("CATION", 5),
    ("ANION", 6),
    ("UNK", 7),
];

#[cfg(feature = "gpu")]
pub fn extract_pharmacophore_gpu(
    engine: &PharmacophoreGpu,
    spikes: &SpikeData,
    centroid: [f32; 3],
    site_id: i32,
    output_dir: &Path,
    receptor_pdb: &str,
) -> anyhow::Result<()> {
    let t0 = Instant::now();
    std::fs::create_dir_all(output_dir)?;
    log::info!(
        "GPU Pharmacophore: site {} — {} spikes",
        site_id,
        spikes.len()
    );

    // Type counts
    let mut type_counts: HashMap<i32, usize> = HashMap::new();
    for &t in &spikes.types {
        *type_counts.entry(t).or_default() += 1;
    }

    // Combined grid
    let t = Instant::now();
    let combined = engine.build_density_grid(spikes, 1.0, 1.5)?;
    let cp = output_dir.join(format!("site{}_combined.dx", site_id));
    write_dx(&combined, &cp, "PRISM-4D combined hotspot (GPU)")?;
    log::info!(
        "  Combined: {}x{}x{} max={:.1} [{:.0}ms]",
        combined.dims[0],
        combined.dims[1],
        combined.dims[2],
        combined.max_val,
        t.elapsed().as_secs_f64() * 1000.0
    );

    // Per-type grids
    let mut tg = Vec::new();
    for &(name, code) in TYPE_INFO {
        let count = type_counts.get(&code).copied().unwrap_or(0);
        if count < 10 {
            continue;
        }
        let t = Instant::now();
        let grid = engine.build_density_grid_typed(spikes, 1.0, 1.5, code)?;
        let dp = output_dir.join(format!("site{}_{}.dx", site_id, name.to_lowercase()));
        write_dx(&grid, &dp, &format!("PRISM-4D {} hotspot (GPU)", name))?;
        tg.push((name.to_string(), grid.max_val, count));
        log::info!(
            "  {}: {} spikes max={:.1} [{:.0}ms]",
            name,
            count,
            grid.max_val,
            t.elapsed().as_secs_f64() * 1000.0
        );
    }

    // Open frame
    let (best, top) = find_open_frame(spikes, centroid);
    log::info!(
        "  Open frame: {} (top5: {:?})",
        best,
        &top[..top.len().min(5)]
    );

    // PyMOL
    write_pymol_script(
        output_dir,
        site_id,
        centroid,
        receptor_pdb,
        combined.max_val,
        &tg,
    )?;

    log::info!("  Total: {:.0}ms", t0.elapsed().as_secs_f64() * 1000.0);
    Ok(())
}
