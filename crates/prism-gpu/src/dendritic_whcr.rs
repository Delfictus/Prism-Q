//! Dendritic Reservoir for WHCR (DR-WHCR) GPU Module
//!
//! Live neuromorphic co-processor that evolves WITH the repair process,
//! providing real-time adaptive guidance for conflict resolution.
//!
//! # Architecture
//! - 4-compartment dendritic processing per vertex
//! - Multi-timescale dynamics (fast → slow → long-term memory)
//! - Pattern detection (oscillation, stubborn conflicts, cascades)
//! - Priority modulation output for WHCR

use anyhow::{Context, Result};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, CudaFunction, LaunchConfig, PushKernelArg, DeviceSlice};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// C-compatible structs (must match CUDA exactly!)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DendriticCompartment {
    pub activation: f32,
    pub calcium: f32,
    pub threshold: f32,
    pub refractory: f32,
}

unsafe impl cudarc::driver::ValidAsZeroBits for DendriticCompartment {}
unsafe impl cudarc::driver::DeviceRepr for DendriticCompartment {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VertexDendriticState {
    pub compartments: [DendriticCompartment; 4],
    pub soma_potential: f32,
    pub spike_history: f32,
    pub conflict_memory: f32,
    pub last_repair_iteration: i32,
}

unsafe impl cudarc::driver::ValidAsZeroBits for VertexDendriticState {}
unsafe impl cudarc::driver::DeviceRepr for VertexDendriticState {}

impl Default for VertexDendriticState {
    fn default() -> Self {
        Self {
            compartments: [DendriticCompartment {
                activation: 0.0,
                calcium: 0.0,
                threshold: 0.5,
                refractory: 0.0,
            }; 4],
            soma_potential: 0.0,
            spike_history: 0.0,
            conflict_memory: 0.0,
            last_repair_iteration: -100,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ReservoirConnection {
    pub source_vertex: i32,
    pub target_vertex: i32,
    pub source_compartment: i32,
    pub target_compartment: i32,
    pub weight: f32,
    pub delay: f32,
}

unsafe impl cudarc::driver::ValidAsZeroBits for ReservoirConnection {}
unsafe impl cudarc::driver::DeviceRepr for ReservoirConnection {}

/// Number of dendritic compartments per vertex
pub const NUM_COMPARTMENTS: usize = 4;

/// Number of output signals
pub const NUM_OUTPUTS: usize = 4;

/// GPU-accelerated Dendritic Reservoir
pub struct DendriticReservoirGpu {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,

    // Loaded kernels (init_reservoir only used once during creation)
    process_dendritic_input: CudaFunction,
    process_recurrent: CudaFunction,
    compute_soma: CudaFunction,
    compute_outputs: CudaFunction,
    modulate_priorities: CudaFunction,

    // Device memory
    d_vertex_states: CudaSlice<VertexDendriticState>,
    d_connections: CudaSlice<ReservoirConnection>,
    d_connection_row_ptr: CudaSlice<i32>,
    d_input_weights: CudaSlice<f32>,
    d_output_weights: CudaSlice<f32>,
    d_outputs: CudaSlice<f32>, // [num_vertices * 4 outputs]
    d_conflict_history: CudaSlice<f32>,

    // Config
    num_vertices: usize,
    num_connections: usize,
    history_length: usize,
    history_index: usize,
    reservoir_influence: f32,
}

impl DendriticReservoirGpu {
    /// Create new dendritic reservoir GPU instance
    pub fn new(
        context: Arc<CudaContext>,
        num_vertices: usize,
        graph_row_ptr: &[i32],
        graph_col_idx: &[i32],
        initial_conflicts: &[f32],
        reservoir_influence: f32,
    ) -> Result<Self> {
        let stream = context.default_stream();
        log::info!(
            "Initializing Dendritic Reservoir GPU for {} vertices",
            num_vertices
        );

        // Load the dendritic_whcr PTX module using cudarc 0.18.1 API
        let ptx_paths = vec![
            "target/ptx/dendritic_whcr.ptx",
        ];

        let mut ptx = None;
        for ptx_path in &ptx_paths {
            if std::path::Path::new(ptx_path).exists() {
                ptx = Some(Ptx::from_file(ptx_path));
                break;
            }
        }

        let ptx = ptx.context("Failed to load dendritic_whcr PTX module")?;

        // Mangled names from original code
        let func_names = [
            "_Z18init_vertex_statesP20VertexDendriticStatePKfiy",
            "_Z18init_input_weightsPfiy",
            "_Z23process_dendritic_inputP20VertexDendriticStatePKf18WHCRIterationStatei",
            "_Z29process_recurrent_connectionsP20VertexDendriticStatePKiPK19ReservoirConnectionPfii",
            "_Z24compute_soma_integrationP20VertexDendriticStatei",
            "_Z25compute_reservoir_outputsPK20VertexDendriticStatePKfS3_iiPfii",
            "_Z24modulate_whcr_prioritiesPKfS0_Pfif",
        ];

        let module = context.load_module(ptx)
            .context("Failed to load dendritic_whcr PTX module")?;

        let get_kernel = |name| context.get_func("dendritic_whcr", name)
            .context("Failed to load kernel");

        // Get kernel functions using mangled names
        let init_vertex_states = get_kernel("_Z18init_vertex_statesP20VertexDendriticStatePKfiy")?;
        let init_input_weights = get_kernel("_Z18init_input_weightsPfiy")?;
        let process_dendritic_input = get_kernel("_Z23process_dendritic_inputP20VertexDendriticStatePKf18WHCRIterationStatei")?;
        let process_recurrent = get_kernel("_Z29process_recurrent_connectionsP20VertexDendriticStatePKiPK19ReservoirConnectionPfii")?;
        let compute_soma = get_kernel("_Z24compute_soma_integrationP20VertexDendriticStatei")?;
        let compute_outputs = get_kernel("_Z25compute_reservoir_outputsPK20VertexDendriticStatePKfS3_iiPfii")?;
        let modulate_priorities = get_kernel("_Z24modulate_whcr_prioritiesPKfS0_Pfif")?;

        // Estimate number of connections (sparsity ~ 0.1)
        let avg_degree = graph_col_idx.len() / num_vertices;
        let num_connections = (num_vertices * avg_degree / 10).max(num_vertices);

        // Allocate device memory
        let d_vertex_states = stream.alloc_zeros::<VertexDendriticState>(num_vertices)?;
        let d_connections = stream.alloc_zeros::<ReservoirConnection>(num_connections)?;
        let d_connection_row_ptr = stream.clone_htod(graph_row_ptr)?;
        let d_input_weights = stream.alloc_zeros::<f32>(num_vertices * NUM_COMPARTMENTS)?;
        let d_output_weights = stream.alloc_zeros::<f32>(num_vertices * NUM_OUTPUTS)?;
        let d_outputs = stream.alloc_zeros::<f32>(num_vertices * NUM_OUTPUTS)?;

        let history_length = 50;
        let d_conflict_history = stream.alloc_zeros::<f32>(history_length * num_vertices)?;

        // Initialize reservoir on GPU
        let d_initial_conflicts = stream.clone_htod(&initial_conflicts)?;

        let cfg = LaunchConfig::for_num_elems(num_vertices as u32);
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Initialize vertex states
        unsafe {
            &stream.launch_builder(&init_vertex_states)
                .arg(&d_vertex_states)
                .arg(&d_initial_conflicts)
                .arg(&(num_vertices as i32))
                .arg(&seed)
                .launch(cfg)?;
        }

        // Initialize input weights
        unsafe {
            &stream.launch_builder(&init_input_weights)
                .arg(&d_input_weights)
                .arg(&(num_vertices as i32))
                .arg(&seed)
                .launch(cfg)?;
        }

        stream.synchronize()?;

        log::info!(
            "Dendritic Reservoir GPU initialized with {} connections",
            num_connections
        );

        Ok(Self {
            context,
            stream,
            process_dendritic_input,
            process_recurrent,
            compute_soma,
            compute_outputs,
            modulate_priorities,
            d_vertex_states,
            d_connections,
            d_connection_row_ptr,
            d_input_weights,
            d_output_weights,
            d_outputs,
            d_conflict_history,
            num_vertices,
            num_connections,
            history_length,
            history_index: 0,
            reservoir_influence,
        })
    }

    /// Process one WHCR iteration through the reservoir
    pub fn step(
        &mut self,
        conflict_counts: &[f32],
        conflict_deltas: &[f32],
        colors: &[i32],
        moves_applied: &[i32],
        _wavelet_details: Option<&[f32]>,
        iteration: i32,
    ) -> Result<()> {
        // Upload WHCR state to GPU
        let d_conflicts = self.stream.clone_htod(&conflict_counts)?;
        let d_deltas = self.stream.clone_htod(&conflict_deltas)?;
        let _d_colors = self.stream.clone_htod(&colors)?;
        let d_moves = self.stream.clone_htod(&moves_applied)?;

        let cfg = LaunchConfig::for_num_elems(self.num_vertices as u32);

        // 1. Process dendritic input
        unsafe {
            &self.stream.launch_builder(&self.process_dendritic_input)
                .arg(&self.d_vertex_states)
                .arg(&self.d_input_weights)
                .arg(&d_conflicts)
                .arg(&d_deltas)
                .arg(&d_moves)
                .arg(&iteration)
                .arg(&(self.num_vertices as i32))
                .launch(cfg)?;
        }

        // 2. Process recurrent connections
        unsafe {
            &self.stream.launch_builder(&self.process_recurrent)
                .arg(&self.d_vertex_states)
                .arg(&self.d_connection_row_ptr)
                .arg(&self.d_connections)
                .arg(&(self.num_vertices as i32))
                .arg(&iteration)
                .launch(cfg)?;
        }

        // 3. Compute soma integration
        unsafe {
            &self.stream.launch_builder(&self.compute_soma)
                .arg(&self.d_vertex_states)
                .arg(&(self.num_vertices as i32))
                .launch(cfg)?;
        }

        // 4. Compute outputs
        unsafe {
            &self.stream.launch_builder(&self.compute_outputs)
                .arg(&self.d_vertex_states)
                .arg(&self.d_output_weights)
                .arg(&self.d_conflict_history)
                .arg(&(self.history_length as i32))
                .arg(&(self.history_index as i32))
                .arg(&self.d_outputs)
                .arg(&(self.num_vertices as i32))
                .arg(&iteration)
                .launch(cfg)?;
        }

        self.stream.synchronize()?;
        self.history_index = (self.history_index + 1) % self.history_length;

        Ok(())
    }

    /// Get reservoir-modulated priorities for WHCR
    pub fn get_modulated_priorities(&self, wavelet_priorities: &[f32]) -> Result<Vec<f32>> {
        let d_wavelet = self.stream.clone_htod(&wavelet_priorities)?;
        let mut d_final = self.stream.alloc_zeros::<f32>(self.num_vertices)?;

        let cfg = LaunchConfig::for_num_elems(self.num_vertices as u32);
        unsafe {
            &self.stream.launch_builder(&self.modulate_priorities)
                .arg(&self.d_outputs)
                .arg(&d_wavelet)
                .arg(&mut d_final)
                .arg(&(self.num_vertices as i32))
                .arg(&self.reservoir_influence)
                .launch(cfg)?;
        }

        let result = self.stream.clone_dtoh(&d_final)?;
        Ok(result)
    }
}
