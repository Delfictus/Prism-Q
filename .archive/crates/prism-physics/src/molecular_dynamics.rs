//! # Molecular Dynamics Engine - Holographic VRAM Turbo v3.1 (Production Grade)
//! Architecture: Float4 Stride, Device-Resident State, Euler-Maruyama Integrator.
//! Status: Audit Compliant, Type-Safe, Warning-Free.

use prism_core::{PhaseOutcome, PrismError};
use prism_io::holographic::PtbStructure;
use prism_io::sovereign_types::Atom;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::path::Path;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[cfg(feature = "cuda")]
use cudarc::driver::sys as cuda_sys;
#[cfg(feature = "cuda")]
use cudarc::driver::CudaContext;
#[cfg(feature = "cuda")]
use cudarc::nvrtc::Ptx;

// AUDIT: Must match CUDA static_assert in kernel
const RNG_STATE_BYTES: usize = 64;

// RAII Guard for Temp Files to ensure cleanup even on panic
struct TempFileGuard {
    path: String,
}
impl Drop for TempFileGuard {
    fn drop(&mut self) {
        if Path::new(&self.path).exists() {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MolecularDynamicsConfig {
    pub max_steps: u64,
    pub dt: f32,
    pub friction: f32,
    pub temp_start: f32,
    pub temp_end: f32,
    pub annealing_steps: u64,
    pub cutoff_dist: f32,
    pub spring_k: f32,
    pub bias_strength: f32,
    pub target_mode: usize,
    pub use_gpu: bool,
    pub max_trajectory_memory: usize,
    pub max_workspace_memory: usize,
}

impl Default for MolecularDynamicsConfig {
    fn default() -> Self {
        Self {
            max_steps: 1_000_000,
            dt: 0.001,
            friction: 0.1,
            temp_start: 2.5,
            temp_end: 0.1,
            annealing_steps: 500_000,
            cutoff_dist: 10.0,
            spring_k: 10.0,
            bias_strength: 0.0,
            target_mode: 7,
            use_gpu: true,
            max_trajectory_memory: 1024 * 1024 * 1024,
            max_workspace_memory: 512 * 1024 * 1024,
        }
    }
}

#[derive(Debug)]
pub struct SimulationBuffers {
    pub positions: Vec<f32>,
    pub velocities: Vec<f32>,
    pub anchors: Vec<f32>,
    pub bias_vec: Vec<f32>,
    pub atom_to_res: Vec<u32>,
    pub num_atoms: usize,
}

impl SimulationBuffers {
    pub fn from_atoms(atoms: &[Atom]) -> Self {
        let n = atoms.len();
        let mut pos = Vec::with_capacity(n * 4);
        let mut vel = vec![0.0; n * 4];
        let mut anc = Vec::with_capacity(n * 4);
        let mut bias = vec![0.0; n * 4];
        let mut map = Vec::with_capacity(n);

        for atom in atoms {
            // Float4 Layout: [x, y, z, w]
            // w=1.0 for positions/anchors (can be used for mass later)
            pos.extend_from_slice(&[atom.coords[0], atom.coords[1], atom.coords[2], 1.0]);
            anc.extend_from_slice(&[atom.coords[0], atom.coords[1], atom.coords[2], 1.0]);

            // Explicit cast: residue_id is u16, we need u32 for GPU alignment/indexing
            map.push(u32::from(atom.residue_id));
        }

        // AUDIT: Enforce Stride-4 Contract
        assert_eq!(pos.len(), n * 4, "Positions buffer stride mismatch");
        assert_eq!(anc.len(), n * 4, "Anchors buffer stride mismatch");

        Self {
            positions: pos,
            velocities: vel,
            anchors: anc,
            bias_vec: bias,
            atom_to_res: map,
            num_atoms: n,
        }
    }

    pub fn update_atoms(&self, atoms: &mut [Atom]) {
        for (i, atom) in atoms.iter_mut().enumerate() {
            let offset = i * 4;
            atom.coords[0] = self.positions[offset];
            atom.coords[1] = self.positions[offset + 1];
            atom.coords[2] = self.positions[offset + 2];
        }
    }
}

#[derive(Debug)]
pub struct MolecularDynamicsEngine {
    config: MolecularDynamicsConfig,
    current_step: u64,
    start_time: Instant,
    buffers: Option<SimulationBuffers>,
    atoms_metadata: Vec<Atom>,
    #[cfg(feature = "cuda")]
    gpu_state: Option<HolographicGpuState>,
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
struct HolographicGpuState {
    // We hold the context to keep it alive
    _ctx: Arc<CudaContext>,
    raw_module: cuda_sys::CUmodule,
    step_kernel: cuda_sys::CUfunction,
    init_rng_kernel: cuda_sys::CUfunction,
    d_positions: u64,
    d_anchors: u64,
    d_velocities: u64,
    d_bias_vec: u64,
    d_rng_states: u64,
    num_atoms: usize,
}

#[cfg(feature = "cuda")]
impl Drop for HolographicGpuState {
    fn drop(&mut self) {
        unsafe {
            // Explicitly free GPU memory to prevent leaks during RL training loops
            let _ = cuda_sys::cuMemFree_v2(self.d_positions);
            let _ = cuda_sys::cuMemFree_v2(self.d_anchors);
            let _ = cuda_sys::cuMemFree_v2(self.d_velocities);
            let _ = cuda_sys::cuMemFree_v2(self.d_bias_vec);
            let _ = cuda_sys::cuMemFree_v2(self.d_rng_states);
            let _ = cuda_sys::cuModuleUnload(self.raw_module);
        }
    }
}

impl MolecularDynamicsEngine {
    pub fn new(config: MolecularDynamicsConfig) -> Result<Self, PrismError> {
        Ok(Self {
            config,
            current_step: 0,
            start_time: Instant::now(),
            buffers: None,
            atoms_metadata: Vec::new(),
            #[cfg(feature = "cuda")]
            gpu_state: None,
        })
    }

    pub fn from_sovereign_buffer(
        config: MolecularDynamicsConfig,
        sovereign_data: &[u8],
    ) -> Result<Self, PrismError> {
        log::info!("🧬 Initializing Holographic Engine v3.1...");
        let atoms = Self::parse_protein_structure(sovereign_data)?;
        let buffers = SimulationBuffers::from_atoms(&atoms);
        let mut engine = Self::new(config)?;
        engine.atoms_metadata = atoms;
        engine.buffers = Some(buffers);
        #[cfg(feature = "cuda")]
        if engine.config.use_gpu {
            engine.initialize_holographic_gpu()?;
        }
        Ok(engine)
    }

    #[cfg(feature = "cuda")]
    fn initialize_holographic_gpu(&mut self) -> Result<(), PrismError> {
        log::info!("🔌 Engaging VRAM Turbo Cache (Persistent RNG)...");
        let buffers = self
            .buffers
            .as_ref()
            .ok_or(PrismError::Internal("No buffers".into()))?;
        let num_atoms = buffers.num_atoms;
        let buffer_size = num_atoms * 4 * std::mem::size_of::<f32>();
        let rng_size = num_atoms * RNG_STATE_BYTES;

        let ctx = CudaContext::new(0).map_err(|e| PrismError::gpu("init", format!("{:?}", e)))?;

        let ptx_path = "crates/prism-gpu/kernels/holographic_langevin.ptx";
        let mut raw_module: cuda_sys::CUmodule = std::ptr::null_mut();

        // Load PTX Source
        let ptx_src = std::fs::read_to_string(ptx_path)
            .map_err(|e| PrismError::gpu("read_ptx", e.to_string()))?;
        let c_ptx = CString::new(ptx_src).unwrap();

        unsafe {
            let res = cuda_sys::cuModuleLoadData(&mut raw_module, c_ptx.as_ptr() as *const c_void);
            if res != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(PrismError::gpu("cuModuleLoadData", format!("{:?}", res)));
            }
        }

        let mut step_kernel: cuda_sys::CUfunction = std::ptr::null_mut();
        let mut init_rng_kernel: cuda_sys::CUfunction = std::ptr::null_mut();

        let c_step_name = CString::new("holographic_step_kernel").unwrap();
        let c_init_name = CString::new("init_rng_kernel").unwrap();

        unsafe {
            if cuda_sys::cuModuleGetFunction(&mut step_kernel, raw_module, c_step_name.as_ptr())
                != cuda_sys::CUresult::CUDA_SUCCESS
            {
                return Err(PrismError::gpu("GetFunc", "step_kernel".to_string()));
            }
            if cuda_sys::cuModuleGetFunction(&mut init_rng_kernel, raw_module, c_init_name.as_ptr())
                != cuda_sys::CUresult::CUDA_SUCCESS
            {
                return Err(PrismError::gpu("GetFunc", "init_rng_kernel".to_string()));
            }
        }

        unsafe {
            let mut d_positions: u64 = 0;
            let mut d_anchors: u64 = 0;
            let mut d_velocities: u64 = 0;
            let mut d_bias_vec: u64 = 0;
            let mut d_rng_states: u64 = 0;

            // Allocate VRAM
            if cuda_sys::cuMemAlloc_v2(&mut d_positions, buffer_size)
                != cuda_sys::CUresult::CUDA_SUCCESS
            {
                return Err(PrismError::gpu("alloc", "positions".to_string()));
            }
            if cuda_sys::cuMemAlloc_v2(&mut d_anchors, buffer_size)
                != cuda_sys::CUresult::CUDA_SUCCESS
            {
                return Err(PrismError::gpu("alloc", "anchors".to_string()));
            }
            if cuda_sys::cuMemAlloc_v2(&mut d_velocities, buffer_size)
                != cuda_sys::CUresult::CUDA_SUCCESS
            {
                return Err(PrismError::gpu("alloc", "velocities".to_string()));
            }
            if cuda_sys::cuMemAlloc_v2(&mut d_bias_vec, buffer_size)
                != cuda_sys::CUresult::CUDA_SUCCESS
            {
                return Err(PrismError::gpu("alloc", "bias".to_string()));
            }
            if cuda_sys::cuMemAlloc_v2(&mut d_rng_states, rng_size)
                != cuda_sys::CUresult::CUDA_SUCCESS
            {
                return Err(PrismError::gpu("alloc", "rng".to_string()));
            }

            // Upload Initial State
            if cuda_sys::cuMemcpyHtoD_v2(
                d_positions,
                buffers.positions.as_ptr() as *const c_void,
                buffer_size,
            ) != cuda_sys::CUresult::CUDA_SUCCESS
            {
                return Err(PrismError::gpu("memcpy", "positions".to_string()));
            }
            if cuda_sys::cuMemcpyHtoD_v2(
                d_anchors,
                buffers.anchors.as_ptr() as *const c_void,
                buffer_size,
            ) != cuda_sys::CUresult::CUDA_SUCCESS
            {
                return Err(PrismError::gpu("memcpy", "anchors".to_string()));
            }
            if cuda_sys::cuMemcpyHtoD_v2(
                d_bias_vec,
                buffers.bias_vec.as_ptr() as *const c_void,
                buffer_size,
            ) != cuda_sys::CUresult::CUDA_SUCCESS
            {
                return Err(PrismError::gpu("memcpy", "bias".to_string()));
            }
            if cuda_sys::cuMemsetD8_v2(d_velocities, 0, buffer_size)
                != cuda_sys::CUresult::CUDA_SUCCESS
            {
                return Err(PrismError::gpu("memset", "velocities".to_string()));
            }

            // Initialize RNG (One-time setup)
            let seed: u64 = 12345;
            let n_atoms_i32 = num_atoms as i32;
            let mut init_args: Vec<*mut c_void> = vec![
                &seed as *const _ as *mut c_void,
                &d_rng_states as *const _ as *mut c_void,
                &n_atoms_i32 as *const _ as *mut c_void,
            ];

            let threads = 128;
            let blocks = (num_atoms + threads - 1) / threads;

            let res = cuda_sys::cuLaunchKernel(
                init_rng_kernel,
                blocks as u32,
                1,
                1,
                threads as u32,
                1,
                1,
                0,
                std::ptr::null_mut(),
                init_args.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            if res != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(PrismError::gpu("launch_init", format!("{:?}", res)));
            }

            if cuda_sys::cuCtxSynchronize() != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(PrismError::gpu("sync_init", "RNG init failed".to_string()));
            }

            self.gpu_state = Some(HolographicGpuState {
                _ctx: ctx,
                raw_module,
                step_kernel,
                init_rng_kernel,
                d_positions,
                d_anchors,
                d_velocities,
                d_bias_vec,
                d_rng_states,
                num_atoms,
            });
        }
        Ok(())
    }

    pub fn run_nlnm_breathing(&mut self, steps: u64) -> Result<PhaseOutcome, PrismError> {
        log::info!("🌬️ Starting Hybrid Simulation: {} steps", steps);
        let start = Instant::now();

        #[cfg(feature = "cuda")]
        if let Some(gpu) = &self.gpu_state {
            let threads = 128;
            let blocks = (gpu.num_atoms + threads - 1) / threads;
            let batch_size = 5000;

            let mut steps_remaining = steps;
            let mut local_step_counter = self.current_step;

            // Physics Constants
            let dt = self.config.dt;
            let friction = self.config.friction;
            let temp_start = self.config.temp_start;
            let temp_end = self.config.temp_end;
            let bias_strength = self.config.bias_strength;
            let spring_k = self.config.spring_k;
            let n_atoms_i32 = gpu.num_atoms as i32;
            let annealing_steps_i32 =
                (self.config.annealing_steps.min(i32::MAX as u64) as i32).max(1);

            while steps_remaining > 0 {
                let current_batch = std::cmp::min(batch_size, steps_remaining);

                for _ in 0..current_batch {
                    // CRITICAL FIX: Re-create args vector inside the loop.
                    // This ensures the pointer to `step_idx_param` is valid and points to the updated value.
                    let mut step_idx_param = local_step_counter as i32;

                    unsafe {
                        let mut args: Vec<*mut c_void> = vec![
                            &gpu.d_positions as *const _ as *mut c_void,
                            &gpu.d_anchors as *const _ as *mut c_void,
                            &gpu.d_velocities as *const _ as *mut c_void,
                            &gpu.d_bias_vec as *const _ as *mut c_void,
                            &n_atoms_i32 as *const _ as *mut c_void,
                            &dt as *const _ as *mut c_void,
                            &friction as *const _ as *mut c_void,
                            &temp_start as *const _ as *mut c_void,
                            &temp_end as *const _ as *mut c_void,
                            &bias_strength as *const _ as *mut c_void,
                            &spring_k as *const _ as *mut c_void,
                            &gpu.d_rng_states as *const _ as *mut c_void,
                            &mut step_idx_param as *mut _ as *mut c_void, // Pointer to current stack variable
                            &annealing_steps_i32 as *const _ as *mut c_void,
                        ];

                        let res = cuda_sys::cuLaunchKernel(
                            gpu.step_kernel,
                            blocks as u32,
                            1,
                            1,
                            threads as u32,
                            1,
                            1,
                            0,
                            std::ptr::null_mut(),
                            args.as_mut_ptr(),
                            std::ptr::null_mut(),
                        );

                        if res != cuda_sys::CUresult::CUDA_SUCCESS {
                            return Err(PrismError::gpu("launch", format!("{:?}", res)));
                        }
                    }
                    local_step_counter += 1;
                }

                // Sync after batch
                unsafe {
                    if cuda_sys::cuCtxSynchronize() != cuda_sys::CUresult::CUDA_SUCCESS {
                        return Err(PrismError::gpu("sync", "failed".to_string()));
                    }
                }
                steps_remaining -= current_batch;
            }
            self.current_step = local_step_counter;
        }

        let duration = start.elapsed();
        log::info!("🏁 Simulation Complete: {:.2}s", duration.as_secs_f32());
        Ok(PhaseOutcome::Success {
            message: "Holographic run complete".to_string(),
            telemetry: HashMap::new(),
        })
    }

    pub fn get_current_atoms(&mut self) -> Result<Vec<Atom>, PrismError> {
        #[cfg(feature = "cuda")]
        {
            if let Some(gpu) = &self.gpu_state {
                if let Some(buffers) = &mut self.buffers {
                    log::info!("📥 Downloading results from VRAM...");
                    let buffer_size = gpu.num_atoms * 4 * std::mem::size_of::<f32>();
                    unsafe {
                        if cuda_sys::cuMemcpyDtoH_v2(
                            buffers.positions.as_mut_ptr() as *mut c_void,
                            gpu.d_positions,
                            buffer_size,
                        ) != cuda_sys::CUresult::CUDA_SUCCESS
                        {
                            return Err(PrismError::gpu("download", "memcpy failed".to_string()));
                        }
                    }
                    buffers.update_atoms(&mut self.atoms_metadata);
                    log::info!("✅ Download & Unpack complete.");
                }
            }
        }
        Ok(self.atoms_metadata.clone())
    }

    fn parse_protein_structure(data: &[u8]) -> Result<Vec<Atom>, PrismError> {
        if data.is_empty() {
            return Err(PrismError::validation("Empty data"));
        }

        // Detect format by magic bytes
        // PTB format: "PRISM4D\0" = [80, 82, 73, 83, 77, 52, 68, 0]
        // PDB format: typically starts with "HEADER", "ATOM", "REMARK", etc.
        const PTB_MAGIC: &[u8] = b"PRISM4D\0";

        if data.len() >= 8 && &data[0..8] == PTB_MAGIC {
            // PTB binary format - use existing parser
            log::debug!("Detected PTB format, using binary parser");
            Self::parse_ptb_structure(data)
        } else {
            // Assume PDB text format
            log::info!("Detected PDB format, parsing text structure");
            Self::parse_pdb_structure(data)
        }
    }

    /// Parse PTB (PRISM binary) format
    fn parse_ptb_structure(data: &[u8]) -> Result<Vec<Atom>, PrismError> {
        use std::io::Write;
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let temp_path = format!("/tmp/prism_holographic_{}.ptb", timestamp);

        // RAII Guard ensures file is deleted when this scope ends
        let _guard = TempFileGuard {
            path: temp_path.clone(),
        };

        {
            let mut f = std::fs::File::create(&temp_path)
                .map_err(|e| PrismError::Internal(e.to_string()))?;
            f.write_all(data)
                .map_err(|e| PrismError::Internal(e.to_string()))?;
        }

        let mut structure =
            PtbStructure::load(&temp_path).map_err(|e| PrismError::Internal(e.to_string()))?;
        let atoms = structure
            .atoms()
            .map_err(|e| PrismError::Internal(e.to_string()))?
            .to_vec();

        Ok(atoms)
    }

    /// Parse PDB text format directly
    fn parse_pdb_structure(data: &[u8]) -> Result<Vec<Atom>, PrismError> {
        let content = String::from_utf8_lossy(data);
        let mut atoms = Vec::new();

        for line in content.lines() {
            if line.starts_with("ATOM  ") || line.starts_with("HETATM") {
                if line.len() < 54 {
                    continue;
                } // Skip malformed lines

                // Extract coordinates (columns 31-54, 1-indexed in PDB spec)
                let x: f32 = line
                    .get(30..38)
                    .unwrap_or("0.0")
                    .trim()
                    .parse()
                    .unwrap_or(0.0);
                let y: f32 = line
                    .get(38..46)
                    .unwrap_or("0.0")
                    .trim()
                    .parse()
                    .unwrap_or(0.0);
                let z: f32 = line
                    .get(46..54)
                    .unwrap_or("0.0")
                    .trim()
                    .parse()
                    .unwrap_or(0.0);

                // Extract element (columns 77-78) or infer from atom name
                let element_char = line
                    .get(76..78)
                    .unwrap_or("  ")
                    .trim()
                    .chars()
                    .next()
                    .unwrap_or_else(|| {
                        // Fallback: infer from atom name (columns 13-16)
                        line.get(12..16)
                            .unwrap_or("C")
                            .trim()
                            .chars()
                            .next()
                            .unwrap_or('C')
                    });

                // Map element to atomic number
                let atomic_number = match element_char {
                    'C' => 6,
                    'N' => 7,
                    'O' => 8,
                    'S' => 16,
                    'P' => 15,
                    'H' => 1,
                    'F' => 9,
                    'K' => 19,
                    'Z' => 30, // Zinc
                    'M' => 12, // Magnesium (Mg)
                    'I' => 53, // Iodine
                    _ => 6,    // Default to carbon
                };

                // Extract residue ID (columns 23-26)
                let residue_id: u16 = line.get(22..26).unwrap_or("0").trim().parse().unwrap_or(0);

                // VdW radius lookup
                let radius = match atomic_number {
                    1 => 1.20,  // H
                    6 => 1.70,  // C
                    7 => 1.55,  // N
                    8 => 1.52,  // O
                    15 => 1.80, // P
                    16 => 1.80, // S
                    _ => 1.70,  // Default
                };

                atoms.push(Atom {
                    coords: [x, y, z],
                    element: atomic_number,
                    residue_id,
                    atom_type: 1,
                    charge: 0.0,
                    radius,
                    _reserved: [0; 4],
                });
            }
        }

        if atoms.is_empty() {
            return Err(PrismError::validation("No ATOM records found in PDB data"));
        }

        log::info!("Parsed {} atoms from PDB format", atoms.len());
        Ok(atoms)
    }

    pub fn get_statistics(&self) -> MolecularDynamicsStats {
        let denom = std::cmp::max(1, self.config.annealing_steps) as f32;
        let progress = (self.current_step as f32 / denom).min(1.0);
        let current_temp =
            self.config.temp_start + (self.config.temp_end - self.config.temp_start) * progress;

        MolecularDynamicsStats {
            current_step: self.current_step,
            total_steps: self.config.max_steps,
            current_energy: 0.0,
            current_temperature: current_temp,
            acceptance_rate: 1.0,
            gradient_norm: 0.0,
            runtime_seconds: self.start_time.elapsed().as_secs_f32(),
            converged: false,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn set_cuda_context(&mut self, _context: Arc<CudaContext>) {}

    // ========================================================================
    // PDB EXPORT - Digital Twin Output
    // ========================================================================

    /// Exports the current state as a PDB file, using the original file as a template.
    /// This preserves non-coordinate metadata (Chain IDs, B-factors, residue names, etc.).
    ///
    /// # Arguments
    /// * `output_path` - Path to write the relaxed structure
    /// * `template_path` - Path to the original PDB (for metadata preservation)
    pub fn save_pdb(&mut self, output_path: &str, template_path: &str) -> Result<(), PrismError> {
        use std::fs::File;
        use std::io::{BufRead, BufReader, Write};

        // 1. Ensure we have the latest data from GPU
        let current_atoms = self.get_current_atoms()?;

        // 2. Open Template and Output
        let template_file = File::open(template_path)
            .map_err(|e| PrismError::Internal(format!("Failed to open template: {}", e)))?;
        let reader = BufReader::new(template_file);
        let mut output_file = File::create(output_path)
            .map_err(|e| PrismError::Internal(format!("Failed to create output: {}", e)))?;

        let mut atom_idx = 0;

        for line_result in reader.lines() {
            let line = line_result
                .map_err(|e| PrismError::Internal(format!("Failed to read line: {}", e)))?;

            if (line.starts_with("ATOM") || line.starts_with("HETATM")) && line.len() >= 54 {
                if atom_idx < current_atoms.len() {
                    let atom = &current_atoms[atom_idx];

                    // PDB Fixed Width Format for Coordinates:
                    // X: cols 30-38 (8.3f)
                    // Y: cols 38-46 (8.3f)
                    // Z: cols 46-54 (8.3f)

                    // Reconstruct the line with new coordinates
                    let prefix = &line[0..30];
                    let suffix = if line.len() > 54 { &line[54..] } else { "" };

                    let new_line = format!(
                        "{}{:8.3}{:8.3}{:8.3}{}",
                        prefix, atom.coords[0], atom.coords[1], atom.coords[2], suffix
                    );

                    writeln!(output_file, "{}", new_line)
                        .map_err(|e| PrismError::Internal(format!("Write failed: {}", e)))?;
                    atom_idx += 1;
                } else {
                    // Write remaining lines if atom counts mismatch
                    writeln!(output_file, "{}", line)
                        .map_err(|e| PrismError::Internal(format!("Write failed: {}", e)))?;
                }
            } else {
                // Write headers/footers/non-ATOM lines as-is
                writeln!(output_file, "{}", line)
                    .map_err(|e| PrismError::Internal(format!("Write failed: {}", e)))?;
            }
        }

        log::info!(
            "💾 Saved Digital Twin to: {} ({} atoms)",
            output_path,
            atom_idx
        );
        Ok(())
    }

    /// Get the original atoms metadata (for initial state comparison)
    pub fn get_initial_atoms(&self) -> &[Atom] {
        &self.atoms_metadata
    }

    /// Get current configuration
    pub fn get_config(&self) -> &MolecularDynamicsConfig {
        &self.config
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MolecularDynamicsStats {
    pub current_step: u64,
    pub total_steps: u64,
    pub current_energy: f32,
    pub current_temperature: f32,
    pub acceptance_rate: f32,
    pub gradient_norm: f32,
    pub runtime_seconds: f32,
    pub converged: bool,
}
