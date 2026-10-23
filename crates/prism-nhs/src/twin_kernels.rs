//! PRISM-TWIN GPU kernel bindings
//!
//! Loads ring_buffer.ptx and tensor_ccf.ptx at runtime.
//! Provides safe wrappers for ring buffer threshold coupling
//! and Tensor Core cross-correlation computation.

pub mod endpoint_capture;

use anyhow::{Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
};
use std::path::Path;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// PTX file discovery
// ─────────────────────────────────────────────────────────────────────────────

/// Find a TWIN PTX file by name, searching standard paths.
pub fn find_twin_ptx(name: &str) -> Result<String> {
    // Check PRISM4D_PTX_DIR / PRISM_PTX_DIR first
    for env_key in &["PRISM4D_PTX_DIR", "PRISM_PTX_DIR"] {
        if let Ok(dir) = std::env::var(env_key) {
            let p = std::path::PathBuf::from(&dir).join(name);
            if p.exists() {
                return Ok(p.display().to_string());
            }
        }
    }
    let candidates = [
        format!("target/ptx/{}", name),
        format!("../../target/ptx/{}", name),
        format!(
            "crates/prism-gpu/src/kernels/{}",
            name.replace(".ptx", ".ptx")
        ),
    ];
    // Also check OUT_DIR from build.rs
    if let Ok(out_dir) = std::env::var("OUT_DIR") {
        let path = format!("{}/ptx/{}", out_dir, name);
        if Path::new(&path).exists() {
            return Ok(path);
        }
    }
    for c in &candidates {
        if Path::new(c).exists() {
            return Ok(c.clone());
        }
    }
    // Search in build output directories (release first, then debug — covers both
    // build profiles when running from a fresh clone where target/ptx/ doesn't exist)
    for profile in &["release", "debug"] {
        let build_glob = format!("target/{}/build/prism-gpu-*/out/ptx/{}", profile, name);
        if let Ok(entries) = glob::glob(&build_glob) {
            for entry in entries.flatten() {
                return Ok(entry.to_string_lossy().to_string());
            }
        }
    }
    anyhow::bail!("TWIN PTX '{}' not found in any search path", name)
}

// ─────────────────────────────────────────────────────────────────────────────
// Stream Completion Signaling
// ─────────────────────────────────────────────────────────────────────────────

/// Device-side stream completion flags for persistent coupling kernel.
///
/// Each physics stream appends `signal_stream_done(flag, step)` after its
/// LADD kernels. The persistent coupling kernel spin-waits on both flags.
/// Step-numbered flags prevent ABA race conditions.
pub struct TwinSignal {
    pub flag_a: CudaSlice<u32>, // [1] — stream A completion flag
    pub flag_b: CudaSlice<u32>, // [1] — stream B completion flag
    signal_fn: CudaFunction,
    clear_fn: CudaFunction,
}

impl TwinSignal {
    pub fn new(stream: &Arc<CudaStream>, module: &Arc<CudaModule>) -> Result<Self> {
        let flag_a = stream.alloc_zeros::<u32>(1)?;
        let flag_b = stream.alloc_zeros::<u32>(1)?;
        let signal_fn = module
            .load_function("signal_stream_done")
            .context("signal_stream_done not found in PTX")?;
        let clear_fn = module
            .load_function("clear_signals")
            .context("clear_signals not found in PTX")?;
        Ok(Self {
            flag_a,
            flag_b,
            signal_fn,
            clear_fn,
        })
    }

    /// Signal that stream A has completed step `step_number`.
    /// Must be launched on stream A's CUDA stream.
    pub fn signal_a(&mut self, stream: &Arc<CudaStream>, step_number: u32) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.signal_fn)
                .arg(&mut self.flag_a)
                .arg(&step_number)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Signal that stream B has completed step `step_number`.
    /// Must be launched on stream B's CUDA stream.
    pub fn signal_b(&mut self, stream: &Arc<CudaStream>, step_number: u32) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.signal_fn)
                .arg(&mut self.flag_b)
                .arg(&step_number)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Clear both flags (called from host or coupling kernel after exchange).
    pub fn clear(&mut self, stream: &Arc<CudaStream>) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.clear_fn)
                .arg(&mut self.flag_a)
                .arg(&mut self.flag_b)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Read flag values from GPU (for diagnostics).
    pub fn read_flags(&self, stream: &Arc<CudaStream>) -> Result<(u32, u32)> {
        let mut a = [0u32; 1];
        let mut b = [0u32; 1];
        stream.memcpy_dtoh(&self.flag_a, &mut a)?;
        stream.memcpy_dtoh(&self.flag_b, &mut b)?;
        Ok((a[0], b[0]))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Ring Buffer
// ─────────────────────────────────────────────────────────────────────────────

/// RingSpikeEvent matches the CUDA struct in ring_buffer.cu (48 bytes).
/// This is a SUBSET of the sealed 96-byte GpuSpikeEvent packet — only the fields needed
/// for threshold adaptation are included.
#[repr(C, packed)]
#[derive(Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct RingSpikeEvent {
    pub timestep: i32,
    pub voxel_idx: i32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub intensity: f32,
    pub vibrational_energy: f32,
    pub water_density: f32,
    pub n_nearby_excited: i32,
    pub spike_source: i32,
    pub wavelength_nm: f32,
    pub primary_residue_id: i32, // lossless residue attribution (was: pad)
}

/// Device receipt for one physical recipient of a shared source-group ring.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct RingRecipientReceipt {
    pub cursor: u32,
    pub last_head: u32,
    pub last_completed_step: u32,
    pub last_available: u32,
    pub total_inspected: u64,
    pub total_overwritten: u64,
    pub read_windows: u64,
}


/// Source-bound publication cursor. Allocate once per physical producer before
/// capture; keep source allocations and this owner alive through graph teardown.
/// It does not archive events or execute MD. The engine integrates the archive
/// callback at its existing quiescent full-record boundary.
pub struct TwinSpikePublication {
    cursor: CudaSlice<i32>,
    continuation_pending: CudaSlice<u32>,
    failure: CudaSlice<u32>,
    source_ptr: u64,
    count_ptr: u64,
    source_capacity: u32,
    destination_ptr: Option<u64>,
}
impl TwinSpikePublication {
    pub fn new(stream: &Arc<CudaStream>, events: &CudaSlice<u8>, count: &CudaSlice<i32>) -> Result<Self> {
        let bytes = std::mem::size_of::<crate::fused_engine::GpuSpikeEvent>();
        anyhow::ensure!(bytes == 96 && events.len() % bytes == 0 && count.len() == 1, "lossless source packet/count ABI");
        let source_capacity = u32::try_from(events.len() / bytes)?;
        anyhow::ensure!(source_capacity > 0 && source_capacity <= i32::MAX as u32, "lossless source capacity");
        let source_ptr = events.device_ptr(stream).0;
        let count_ptr = count.device_ptr(stream).0;
        Ok(Self { cursor: stream.alloc_zeros::<i32>(1)?, continuation_pending: stream.clone_htod(&[1u32])?, failure: stream.alloc_zeros::<u32>(1)?, source_ptr, count_ptr, source_capacity, destination_ptr: None })
    }
    fn check_source(&self, stream: &Arc<CudaStream>, events: &CudaSlice<u8>, count: &CudaSlice<i32>) -> Result<()> {
        anyhow::ensure!(count.len() == 1 && events.len() == self.source_capacity as usize * 96,
            "publication source allocation shape changed");
        anyhow::ensure!(events.device_ptr(stream).0 == self.source_ptr && count.device_ptr(stream).0 == self.count_ptr,
            "publication cursor belongs to another physical source");
        Ok(())
    }
    /// Quiescent closeout only; no per-body host readback.
    pub fn published_count(&self, stream: &Arc<CudaStream>) -> Result<i32> {
        let mut cursor = [0i32]; let mut failure = [0u32];
        stream.memcpy_dtoh(&self.cursor, &mut cursor)?;
        stream.memcpy_dtoh(&self.failure, &mut failure)?;
        stream.synchronize()?;
        anyhow::ensure!(failure[0] == 0, "lossless publication device failure bits={}", failure[0]);
        Ok(cursor[0])
    }
    /// Check the complete host archive count against this exact resident source
    /// before its engine-owned reset. Publication cannot authorize raw-data loss.
    pub fn validate_archived_population(
        &self, stream: &Arc<CudaStream>, events: &CudaSlice<u8>, count: &CudaSlice<i32>, archived: usize,
    ) -> Result<()> {
        self.check_source(stream, events, count)?;
        let published = self.published_count(stream)?;
        let mut resident = [0i32]; stream.memcpy_dtoh(count, &mut resident)?; stream.synchronize()?;
        anyhow::ensure!(resident[0] >= 0 && resident[0] as usize == archived && published == resident[0],
            "lossless archive/publication mismatch: archived={archived} resident={} published={published}",resident[0]);
        Ok(())
    }

    /// After the independently checked engine reset, open the next source epoch.
    pub fn acknowledge_source_reset(&mut self, stream: &Arc<CudaStream>, events: &CudaSlice<u8>, count: &CudaSlice<i32>) -> Result<()> {
        self.check_source(stream, events, count)?;
        let mut resident=[0i32]; stream.memcpy_dtoh(count,&mut resident)?; stream.synchronize()?;
        anyhow::ensure!(resident[0] == 0, "source reset not complete");
        stream.memset_zeros(&mut self.cursor)?; stream.synchronize()?;
        Ok(())
    }

    /// At an externally quiescent boundary, require complete publication before
    /// invoking the engine's existing full-event archive/reset. A failed callback
    /// never clears this cursor. Verify source reset before starting a new epoch.
    /// The callback must durably archive full events; publication is not archival.
    pub fn archive_and_reset(
        &mut self, stream: &Arc<CudaStream>, events: &CudaSlice<u8>, count: &CudaSlice<i32>,
        archive_source: impl FnOnce() -> Result<()>,
    ) -> Result<()> {
        self.check_source(stream, events, count)?;
        let published = self.published_count(stream)?;
        let mut resident = [0i32]; stream.memcpy_dtoh(count, &mut resident)?; stream.synchronize()?;
        anyhow::ensure!(resident[0] >= 0 && resident[0] as u32 <= self.source_capacity && published == resident[0],
            "cannot archive/reset unpublished source: published={published} resident={}", resident[0]);
        archive_source()?;
        stream.memcpy_dtoh(count, &mut resident)?; stream.synchronize()?;
        anyhow::ensure!(resident[0] == 0, "archive callback did not reset source count");
        stream.memset_zeros(&mut self.cursor)?;
        stream.synchronize()?;
        Ok(())
    }
}

struct LosslessBroadcast {
    delta: CudaFunction,
    delta_continue: CudaFunction,
    epoch_reset: CudaFunction,
    retire: CudaFunction,
    recipient_pointers: CudaSlice<u64>,
    participating: Vec<usize>,
    failure: CudaSlice<u32>,
    window_kernel: CudaFunction,
    windows: Vec<CudaSlice<u8>>,
}

/// GPU-side ring buffer for spike exchange between twin observation groups.
pub struct TwinRingBuffer {
    // Public for cross-crate access (prism-cuda-ext device compactor)
    pub buffer: CudaSlice<u8>,
    pub head: CudaSlice<u32>,
    pub tail: CudaSlice<u32>,
    pub overflow: CudaSlice<u32>,
    pub capacity: u32,
    /// CPU-side staging for spike compaction (GpuSpikeEvent → RingSpikeEvent)
    staging: Vec<RingSpikeEvent>,
    /// GPU staging buffer for compacted spikes
    d_staging: CudaSlice<u8>,
    // Kernel functions
    push_batch_fn: CudaFunction,
    compact_push_fn: Option<CudaFunction>, // Gate 3: device-side compact+push (zero CPU memcpy)
    read_adapt_fn: CudaFunction,
    recipient_read_fn: Option<CudaFunction>,
    recipient_receipts: Vec<CudaSlice<u8>>,
    lossless: Option<LosslessBroadcast>,
    recovery_fn: CudaFunction,
    reset_fn: CudaFunction,
}

impl TwinRingBuffer {
    pub fn new(
        context: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        capacity: u32,
    ) -> Result<Self> {
        let spike_size = std::mem::size_of::<RingSpikeEvent>();
        assert_eq!(spike_size, 48, "RingSpikeEvent must be exactly 48 bytes");

        let buffer = stream.alloc_zeros::<u8>((capacity as usize) * spike_size)?;
        let head = stream.alloc_zeros::<u32>(1)?;
        let tail = stream.alloc_zeros::<u32>(1)?;
        let overflow = stream.alloc_zeros::<u32>(1)?;

        // Staging buffer: max 4096 spikes per push (can grow if needed)
        let staging_capacity = 4096;
        let d_staging = stream.alloc_zeros::<u8>(staging_capacity * spike_size)?;

        let push_batch_fn = module
            .load_function("ring_buffer_push_batch")
            .context("ring_buffer_push_batch not found in PTX")?;
        let compact_push_fn = module.load_function("compact_and_push").ok();
        if compact_push_fn.is_some() {
            log::info!("  Ring buffer compact_and_push (device-side): loaded");
        }
        let read_adapt_fn = module
            .load_function("ring_buffer_read_and_adapt")
            .context("ring_buffer_read_and_adapt not found in PTX")?;
        let recovery_fn = module
            .load_function("ring_buffer_threshold_recovery")
            .context("ring_buffer_threshold_recovery not found in PTX")?;
        let reset_fn = module
            .load_function("ring_buffer_reset")
            .context("ring_buffer_reset not found in PTX")?;

        Ok(Self {
            buffer,
            head,
            tail,
            overflow,
            capacity,
            staging: Vec::with_capacity(staging_capacity),
            d_staging,
            push_batch_fn,
            compact_push_fn,
            read_adapt_fn,
            recipient_read_fn: None,
            recipient_receipts: Vec::new(),
            lossless: None,
            recovery_fn,
            reset_fn,
        })
    }

    /// Allocate independent cursors before capture. Recipient IDs are physical stream IDs.
    /// Producers must finish before readers; every reader must finish before overwrite.
    pub fn new_broadcast(
        context: &Arc<CudaContext>, stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>, capacity: u32, recipients: usize,
    ) -> Result<Self> {
        anyhow::ensure!(capacity > 0 && recipients > 0, "broadcast ring requires capacity and recipients");
        assert_eq!(std::mem::size_of::<RingRecipientReceipt>(), 40);
        let mut ring = Self::new(context, stream, module, capacity)?;
        ring.recipient_read_fn = Some(module.load_function("ring_buffer_read_and_adapt_recipient")
            .context("recipient-aware ring PTX is required for broadcast")?);
        for _ in 0..recipients {
            ring.recipient_receipts.push(stream.alloc_zeros::<u8>(40)?);
        }
        Ok(ring)
    }


    /// Explicit lossless policy. Registered slots keep physical IDs, but only the
    /// listed recipients participate in safe-tail retirement. Build before capture.
    pub fn new_lossless_broadcast(
        context: &Arc<CudaContext>, stream: &Arc<CudaStream>, module: &Arc<CudaModule>,
        capacity: u32, recipients: usize, participating: &[usize],
    ) -> Result<Self> {
        anyhow::ensure!(capacity.is_power_of_two() && capacity < (1u32 << 31), "lossless capacity must be a power of two below 2^31 for wrap-safe indexing");
        anyhow::ensure!(!participating.is_empty() && participating.len() <= u32::MAX as usize,
            "lossless broadcast needs participating recipients");
        let mut ids = participating.to_vec(); ids.sort_unstable();
        anyhow::ensure!(ids.iter().all(|&i| i < recipients) && ids.windows(2).all(|w| w[0] != w[1]),
            "duplicate or unregistered participating recipient");
        let mut ring = Self::new_broadcast(context, stream, module, capacity, recipients)?;
        let pointers: Vec<u64> = ids.iter().map(|&id| ring.recipient_receipts[id].device_ptr(stream).0).collect();
        let mut recipient_pointers = stream.alloc_zeros::<u64>(pointers.len())?;
        stream.memcpy_htod(&pointers, &mut recipient_pointers)?;
        ring.lossless = Some(LosslessBroadcast {
            delta: module.load_function("compact_and_push_delta_lossless")?,
            delta_continue: module.load_function("compact_and_push_delta_lossless_continue")?,
            epoch_reset: module.load_function("ring_buffer_verify_archived_epoch_and_reset")?,
            retire: module.load_function("ring_buffer_retire_participating_recipients")?,
            recipient_pointers, participating: ids, failure: stream.alloc_zeros::<u32>(1)?,
            window_kernel: module.load_function("ring_buffer_adapt_window")?,
            windows: (0..recipients).map(|_| stream.alloc_zeros::<u8>(20)).collect::<std::result::Result<_, _>>()?,
        });
        Ok(ring)
    }

    /// Capturable, one bounded batch. Deferred suffix stays in the source and its
    /// cursor remains unchanged until a later batch. No full-record source reset.
    pub fn push_device_delta(
        &mut self, stream: &Arc<CudaStream>, events: &CudaSlice<u8>, count: &CudaSlice<i32>,
        publication: &mut TwinSpikePublication,
    ) -> Result<()> {
        publication.check_source(stream, events, count)?;
        let state = self.lossless.as_ref().context("delta publication requires explicit lossless broadcast")?;
        let destination = self.buffer.device_ptr(stream).0;
        if let Some(bound) = publication.destination_ptr {
            anyhow::ensure!(bound == destination, "publication cursor belongs to another destination ring");
        } else {
            publication.destination_ptr = Some(destination);
        }
        unsafe {
            stream.launch_builder(&state.delta)
                .arg(events).arg(count).arg(&publication.source_capacity).arg(&mut publication.cursor)
                .arg(&mut self.buffer).arg(&mut self.head).arg(&self.tail).arg(&mut self.overflow)
                .arg(&self.capacity).arg(&mut publication.failure)
                .launch(LaunchConfig { grid_dim: (1,1,1), block_dim: (256,1,1), shared_mem_bytes: 0 })?;
        }
        Ok(())
    }

    /// Same publication law with a separately versioned device continuation output.
    /// The source allocation and count remain immutable throughout endpoint exchange.
    pub(crate) fn push_device_delta_captured(
        &mut self, stream: &Arc<CudaStream>, events: &CudaSlice<u8>, count: &CudaSlice<i32>,
        publication: &mut TwinSpikePublication,
    ) -> Result<()> {
        publication.check_source(stream, events, count)?;
        let state = self.lossless.as_ref().context("captured publication requires lossless ring")?;
        let destination = self.buffer.device_ptr(stream).0;
        anyhow::ensure!(publication.destination_ptr.is_none_or(|bound| bound == destination),
            "publication cursor belongs to another destination ring");
        publication.destination_ptr = Some(destination);
        unsafe {
            stream.launch_builder(&state.delta_continue)
                .arg(events).arg(count).arg(&publication.source_capacity).arg(&mut publication.cursor)
                .arg(&mut self.buffer).arg(&mut self.head).arg(&self.tail).arg(&mut self.overflow)
                .arg(&self.capacity).arg(&mut publication.failure).arg(&mut publication.continuation_pending)
                .launch(LaunchConfig {grid_dim:(1,1,1),block_dim:(256,1,1),shared_mem_bytes:0})?;
        }
        Ok(())
    }

    /// Capturable retirement node, scheduled after every participating reader has
    /// completed and before the next producer wave. Does not advance physical time.
    pub fn retire_participating_recipients(&mut self, stream: &Arc<CudaStream>) -> Result<()> {
        let state = self.lossless.as_mut().context("retirement requires lossless broadcast")?;
        let n = state.participating.len() as u32;
        unsafe {
            stream.launch_builder(&state.retire).arg(&state.recipient_pointers).arg(&n)
                .arg(&self.head).arg(&mut self.tail).arg(&self.capacity).arg(&mut state.failure)
                .launch(LaunchConfig { grid_dim: (1,1,1), block_dim: (1,1,1), shared_mem_bytes: 0 })?;
        }
        Ok(())
    }

    /// Required closeout validation, independent of the pressure flag. Not a
    /// claim that full records were archived or that physical TWIN16 executed.
    pub fn validate_lossless_retirement(&self, stream: &Arc<CudaStream>) -> Result<()> {
        let state = self.lossless.as_ref().context("not a lossless broadcast")?;
        let mut failure = [0u32]; stream.memcpy_dtoh(&state.failure, &mut failure)?; stream.synchronize()?;
        anyhow::ensure!(failure[0] == 0, "lossless retirement device failure bits={}", failure[0]);
        Ok(())
    }

    /// Begin/accumulate/finalize one source-group endpoint. The existing
    /// adaptation law executes across batches, publishing steering only once.
    /// operation: 0 begin; 1 accumulate; 2 finalize. No MD may run between them.
    #[allow(clippy::too_many_arguments)]
    pub fn adapt_lossless_window(
        &mut self, stream: &Arc<CudaStream>, recipient: usize, operation: u32,
        osc_thresholds: &mut CudaSlice<f32>, base_thresholds: &CudaSlice<f32>,
        grid_dims: (i32,i32,i32), grid_origin: (f32,f32,f32), voxel_size: f32,
        sensitivity_boost: f32, max_reduction: f32, decay_constant: f32,
        protocol: &mut CudaSlice<u8>,
    ) -> Result<()> {
        anyhow::ensure!(operation <= 2, "unknown lossless adaptation operation");
        let state = self.lossless.as_mut().context("window requires explicit lossless policy")?;
        anyhow::ensure!(state.participating.binary_search(&recipient).is_ok(), "unregistered window recipient");
        anyhow::ensure!(protocol.len() >= std::mem::size_of::<crate::protocol_state::ProtocolState>(), "truncated window protocol");
        let receipt = &mut self.recipient_receipts[recipient];
        let window = &mut state.windows[recipient];
        unsafe {
            stream.launch_builder(&state.window_kernel)
                .arg(&self.buffer).arg(&self.head).arg(receipt).arg(&self.capacity)
                .arg(osc_thresholds).arg(base_thresholds)
                .arg(&grid_dims.0).arg(&grid_dims.1).arg(&grid_dims.2)
                .arg(&grid_origin.0).arg(&grid_origin.1).arg(&grid_origin.2)
                .arg(&voxel_size).arg(&sensitivity_boost).arg(&max_reduction)
                .arg(&decay_constant).arg(protocol).arg(window).arg(&operation)
                .launch(LaunchConfig { grid_dim:(1,1,1),block_dim:(1,1,1),shared_mem_bytes:0 })?;
        }
        Ok(())
    }

    pub fn validate_lossless_window(&self, stream: &Arc<CudaStream>, recipient: usize) -> Result<()> {
        let state = self.lossless.as_ref().context("not a lossless window")?;
        let window = state.windows.get(recipient).context("unregistered window")?;
        let mut bytes = [0u8;20]; stream.memcpy_dtoh(window,&mut bytes)?; stream.synchronize()?;
        let active = u32::from_le_bytes(bytes[12..16].try_into()?);
        let failure = u32::from_le_bytes(bytes[16..20].try_into()?);
        anyhow::ensure!(active == 0 && failure == 0, "lossless window unfinished/failed: active={active} failure={failure}");
        Ok(())
    }

    /// Adapt from the recipient's unconsumed window using its live physical device clock.
    /// All device allocations remain owned by this ring for captured graph replay.
    #[allow(clippy::too_many_arguments)]
    pub fn read_and_adapt_for_recipient(
        &mut self, stream: &Arc<CudaStream>, recipient: usize,
        osc_thresholds: &mut CudaSlice<f32>, base_thresholds: &CudaSlice<f32>,
        grid_dims: (i32, i32, i32), grid_origin: (f32, f32, f32), voxel_size: f32,
        sensitivity_boost: f32, max_reduction: f32, decay_constant: f32,
        d_protocol_state: &mut CudaSlice<u8>,
    ) -> Result<()> {
        if let Some(state) = &self.lossless {
            anyhow::ensure!(state.participating.binary_search(&recipient).is_ok(), "recipient does not participate in this source group");
        }
        let kernel = self.recipient_read_fn.as_ref().context("ring was not created for broadcast")?;
        let receipt = self.recipient_receipts.get_mut(recipient).context("unregistered physical ring recipient")?;
        anyhow::ensure!(d_protocol_state.len() >= std::mem::size_of::<crate::protocol_state::ProtocolState>(), "recipient protocol state is truncated");
        unsafe {
            stream.launch_builder(kernel)
                .arg(&self.buffer).arg(&self.head).arg(receipt).arg(&self.capacity)
                .arg(osc_thresholds).arg(base_thresholds)
                .arg(&grid_dims.0).arg(&grid_dims.1).arg(&grid_dims.2)
                .arg(&grid_origin.0).arg(&grid_origin.1).arg(&grid_origin.2)
                .arg(&voxel_size).arg(&sensitivity_boost).arg(&max_reduction)
                .arg(&decay_constant).arg(d_protocol_state)
                .launch(LaunchConfig { grid_dim: (1,1,1), block_dim: (1,1,1), shared_mem_bytes: 0 })?;
        }
        Ok(())
    }

    /// Closeout telemetry; not a per-window host copy or a claim of retained raw events.
    pub fn recipient_receipt(&self, stream: &Arc<CudaStream>, recipient: usize) -> Result<RingRecipientReceipt> {
        let device = self.recipient_receipts.get(recipient).context("unregistered physical ring recipient")?;
        let mut bytes = [0u8; 40];
        stream.memcpy_dtoh(device, &mut bytes)?;
        stream.synchronize()?;
        Ok(unsafe { std::ptr::read_unaligned(bytes.as_ptr().cast::<RingRecipientReceipt>()) })
    }

    /// Compact GpuSpikeEvents from the engine into RingSpikeEvents and push to ring buffer.
    ///
    /// This handles the 96B→48B struct conversion on CPU, then uploads the
    /// compacted spikes and calls the ring buffer push kernel.
    pub fn push_compacted(
        &mut self,
        stream: &Arc<CudaStream>,
        gpu_spikes: &[crate::fused_engine::GpuSpikeEvent],
    ) -> Result<()> {
        anyhow::ensure!(self.lossless.is_none(), "cannot mix recent-window publication with lossless broadcast");
        if gpu_spikes.is_empty() {
            return Ok(());
        }

        // Compact on CPU: extract the 10 fields RingSpikeEvent needs.
        // Only push the most recent spikes up to ring capacity to avoid
        // overwhelming the ring buffer. The ring is for RECENT evidence —
        // if we push 1M spikes into a 8192 ring, 99.9% overflow and the
        // adapter only sees the last 8192 anyway. Better to truncate here.
        let max_to_push = self.capacity as usize;
        let spikes_to_push = if gpu_spikes.len() > max_to_push {
            // Take the MOST RECENT spikes (tail of the vector, which are
            // the latest timesteps since accumulated_spikes is append-only)
            &gpu_spikes[gpu_spikes.len() - max_to_push..]
        } else {
            gpu_spikes
        };

        self.staging.clear();
        for s in spikes_to_push {
            self.staging.push(RingSpikeEvent {
                timestep: s.timestep,
                voxel_idx: s.voxel_idx,
                x: s.position[0],
                y: s.position[1],
                z: s.position[2],
                intensity: s.intensity,
                vibrational_energy: s.vibrational_energy,
                water_density: s.water_density,
                n_nearby_excited: s.n_nearby_excited,
                spike_source: s.spike_source,
                wavelength_nm: s.wavelength_nm,
                primary_residue_id: {
                    let residue = if s.n_residues > 0 { s.nearby_residues[0] } else { -1 };
                    ((residue as u32 & 0xFFFF) | ((s.phase_bits & 0x3FF) << 16)) as i32
                },
            });
        }

        let n_spikes = self.staging.len();

        // Reallocate GPU staging if needed
        let needed_bytes = n_spikes * 48;
        if needed_bytes > self.d_staging.len() {
            self.d_staging = stream.alloc_zeros::<u8>(needed_bytes)?;
        }

        // Upload compacted spikes to GPU staging buffer
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(self.staging.as_ptr() as *const u8, n_spikes * 48)
        };
        stream.memcpy_htod(bytes, &mut self.d_staging)?;

        // Launch push_batch kernel on the staging buffer
        let n_blocks = ((n_spikes as u32) + 255) / 256;
        let cfg = LaunchConfig {
            grid_dim: (n_blocks.max(1), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let prev_count = 0u32;
        let curr_count = n_spikes as u32;
        unsafe {
            stream
                .launch_builder(&self.push_batch_fn)
                .arg(&mut self.buffer)
                .arg(&mut self.head)
                .arg(&mut self.tail)
                .arg(&mut self.overflow)
                .arg(&self.capacity)
                .arg(&self.d_staging)
                .arg(&prev_count)
                .arg(&curr_count)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Check if device-side compact_and_push is available.
    pub fn has_device_push(&self) -> bool {
        self.compact_push_fn.is_some()
    }

    /// GPU-direct spike compaction and push — zero CPU memcpy (Gate 3).
    ///
    /// Reads the sealed 96-byte GpuSpikeEvent packet directly from device memory,
    /// extracts 12 fields into RingSpikeEvent (48 bytes), and pushes
    /// to ring buffer. All on GPU, never touches CPU or PCIe bus.
    ///
    /// This replaces push_compacted() for the autonomous graph path.
    /// Now accepts ProtocolState for bit-packed phase angle computation.
    pub fn push_device(
        &mut self,
        stream: &Arc<CudaStream>,
        d_spike_events: &CudaSlice<u8>, // engine's d_spike_events buffer
        d_spike_count: &CudaSlice<i32>, // engine's d_spike_count
    ) -> Result<()> {
        self.push_device_with_phase(stream, d_spike_events, d_spike_count, None)
    }

    /// Push with bit-packed phase angle from ProtocolState.
    pub fn push_device_with_phase(
        &mut self,
        stream: &Arc<CudaStream>,
        d_spike_events: &CudaSlice<u8>,
        d_spike_count: &CudaSlice<i32>,
        d_protocol_state: Option<&CudaSlice<u8>>,
    ) -> Result<()> {
        anyhow::ensure!(self.lossless.is_none(), "cannot mix recent-window publication with lossless broadcast");
        anyhow::ensure!(d_spike_count.len() == 1, "ring source count must contain one i32");
        if let Some(ps) = d_protocol_state {
            anyhow::ensure!(ps.len() >= std::mem::size_of::<crate::protocol_state::ProtocolState>(), "ring producer protocol state is truncated");
        }
        let compact_fn = match &self.compact_push_fn {
            Some(f) => f,
            None => anyhow::bail!("compact_and_push kernel not loaded — cannot use device path"),
        };

        // compact_and_push emits the bounded recent-evidence window that fits
        // the ring. Launch exactly enough threads for that window; the kernel
        // validates the device-reported count against this source allocation
        // before deriving the tail offset.
        let record_bytes = std::mem::size_of::<crate::fused_engine::GpuSpikeEvent>();
        anyhow::ensure!(d_spike_events.len() % record_bytes == 0, "ring source packet alignment");
        let source_capacity = u32::try_from(d_spike_events.len() / record_bytes)
            .context("ring source capacity exceeds u32")?;
        let n_blocks = self.capacity.div_ceil(256).max(1);
        let cfg = LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let null_ptr = 0u64;
        if let Some(ps) = d_protocol_state {
            unsafe {
                stream
                    .launch_builder(compact_fn)
                    .arg(d_spike_events)
                    .arg(d_spike_count)
                    .arg(&mut self.buffer)
                    .arg(&mut self.head)
                    .arg(&mut self.overflow)
                    .arg(&self.capacity)
                    .arg(&source_capacity)
                    .arg(ps) // ProtocolState for phase angle
                    .launch(cfg)?;
            }
        } else {
            unsafe {
                stream
                    .launch_builder(compact_fn)
                    .arg(d_spike_events)
                    .arg(d_spike_count)
                    .arg(&mut self.buffer)
                    .arg(&mut self.head)
                    .arg(&mut self.overflow)
                    .arg(&self.capacity)
                    .arg(&source_capacity)
                    .arg(&null_ptr) // nullptr — no phase angle
                    .launch(cfg)?;
            }
        }

        Ok(())
    }

    /// Read spikes from ring buffer and modify target engine's thresholds.
    /// If `d_protocol_state` is provided, writes ASC steering fields on hotspot detection.
    pub fn read_and_adapt(
        &mut self,
        stream: &Arc<CudaStream>,
        osc_thresholds: &mut CudaSlice<f32>,
        base_thresholds: &CudaSlice<f32>,
        grid_dims: (i32, i32, i32),
        grid_origin: (f32, f32, f32),
        voxel_size: f32,
        sensitivity_boost: f32,
        max_reduction: f32,
        current_step: u32,
        decay_constant: f32,
    ) -> Result<()> {
        self.read_and_adapt_with_steering(
            stream,
            osc_thresholds,
            base_thresholds,
            grid_dims,
            grid_origin,
            voxel_size,
            sensitivity_boost,
            max_reduction,
            current_step,
            decay_constant,
            None,
        )
    }

    /// Read spikes from ring buffer with optional ASC steering.
    pub fn read_and_adapt_with_steering(
        &mut self,
        stream: &Arc<CudaStream>,
        osc_thresholds: &mut CudaSlice<f32>,
        base_thresholds: &CudaSlice<f32>,
        grid_dims: (i32, i32, i32),
        grid_origin: (f32, f32, f32),
        voxel_size: f32,
        sensitivity_boost: f32,
        max_reduction: f32,
        current_step: u32,
        decay_constant: f32,
        d_protocol_state: Option<&mut CudaSlice<u8>>,
    ) -> Result<()> {
        anyhow::ensure!(self.lossless.is_none(), "lossless broadcast requires recipient-specific reads");
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        let null_ptr = 0u64; // nullptr for ASC when no protocol state
        if let Some(ps) = d_protocol_state {
            unsafe {
                stream
                    .launch_builder(&self.read_adapt_fn)
                    .arg(&self.buffer)
                    .arg(&mut self.head)
                    .arg(&mut self.tail)
                    .arg(&self.capacity)
                    .arg(osc_thresholds)
                    .arg(base_thresholds)
                    .arg(&grid_dims.0)
                    .arg(&grid_dims.1)
                    .arg(&grid_dims.2)
                    .arg(&grid_origin.0)
                    .arg(&grid_origin.1)
                    .arg(&grid_origin.2)
                    .arg(&voxel_size)
                    .arg(&sensitivity_boost)
                    .arg(&max_reduction)
                    .arg(&current_step)
                    .arg(&decay_constant)
                    .arg(ps) // ASC: ProtocolState*
                    .launch(cfg)?;
            }
        } else {
            unsafe {
                stream
                    .launch_builder(&self.read_adapt_fn)
                    .arg(&self.buffer)
                    .arg(&mut self.head)
                    .arg(&mut self.tail)
                    .arg(&self.capacity)
                    .arg(osc_thresholds)
                    .arg(base_thresholds)
                    .arg(&grid_dims.0)
                    .arg(&grid_dims.1)
                    .arg(&grid_dims.2)
                    .arg(&grid_origin.0)
                    .arg(&grid_origin.1)
                    .arg(&grid_origin.2)
                    .arg(&voxel_size)
                    .arg(&sensitivity_boost)
                    .arg(&max_reduction)
                    .arg(&current_step)
                    .arg(&decay_constant)
                    .arg(&null_ptr) // ASC: nullptr
                    .launch(cfg)?;
            }
        }
        Ok(())
    }

    /// Gradually restore thresholds toward baseline.
    pub fn threshold_recovery(
        &self,
        stream: &Arc<CudaStream>,
        osc_thresholds: &mut CudaSlice<f32>,
        base_thresholds: &CudaSlice<f32>,
        n_voxels_total: u32,
        recovery_rate: f32,
    ) -> Result<()> {
        let n_blocks = (n_voxels_total + 255) / 256;
        let cfg = LaunchConfig {
            grid_dim: (n_blocks.max(1), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.recovery_fn)
                .arg(osc_thresholds)
                .arg(base_thresholds)
                .arg(&n_voxels_total)
                .arg(&recovery_rate)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Reset ring buffer state.
    pub fn reset(&mut self, stream: &Arc<CudaStream>) -> Result<()> {
        anyhow::ensure!(self.lossless.is_none(), "lossless counters are monotonic; use per-source archive/reset, never legacy ring reset");
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.reset_fn)
                .arg(&mut self.head)
                .arg(&mut self.tail)
                .arg(&mut self.overflow)
                .launch(cfg)?;
        }
        for receipt in &mut self.recipient_receipts {
            stream.memset_zeros(receipt)?;
        }
        Ok(())
    }

    /// Read overflow counter from GPU.
    pub fn overflow_count(&self, stream: &Arc<CudaStream>) -> Result<u32> {
        let mut count = [0u32; 1];
        stream.memcpy_dtoh(&self.overflow, &mut count)?;
        Ok(count[0])
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tensor Core CCF
// ─────────────────────────────────────────────────────────────────────────────

/// GPU-side Tensor Core cross-correlation compute context.
pub struct TwinCcfCompute {
    spike_matrix_a: CudaSlice<u16>, // FP16 as u16 (half::f16 → u16 bitcast)
    spike_matrix_b: CudaSlice<u16>,
    ccf_output: CudaSlice<f32>,
    norm_a: CudaSlice<f32>,
    norm_b: CudaSlice<f32>,
    n_res: i32,
    n_res_padded: i32,
    n_bins_padded: i32,
    ccf_compute_fn: CudaFunction,
    norms_fn: CudaFunction,
}

impl TwinCcfCompute {
    /// Allocate CCF buffers. Pads dimensions to multiples of 16 for WMMA.
    pub fn new(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        n_res: i32,
        n_bins: i32,
    ) -> Result<Self> {
        let n_res_padded = ((n_res + 15) / 16) * 16;
        let n_bins_padded = ((n_bins + 15) / 16) * 16;

        let mat_size = (n_res_padded * n_bins_padded) as usize;
        let spike_matrix_a = stream.alloc_zeros::<u16>(mat_size)?;
        let spike_matrix_b = stream.alloc_zeros::<u16>(mat_size)?;
        let ccf_output = stream.alloc_zeros::<f32>((n_res * n_res) as usize)?;
        let norm_a = stream.alloc_zeros::<f32>(n_res as usize)?;
        let norm_b = stream.alloc_zeros::<f32>(n_res as usize)?;

        let ccf_compute_fn = module
            .load_function("tensor_ccf_compute")
            .context("tensor_ccf_compute not found in PTX")?;
        let norms_fn = module
            .load_function("compute_spike_norms")
            .context("compute_spike_norms not found in PTX")?;

        Ok(Self {
            spike_matrix_a,
            spike_matrix_b,
            ccf_output,
            norm_a,
            norm_b,
            n_res,
            n_res_padded,
            n_bins_padded,
            ccf_compute_fn,
            norms_fn,
        })
    }

    /// Upload mean-centered spike matrices (CPU → GPU).
    /// Input is f32, converted to FP16 (u16 bitcast) before upload.
    pub fn upload_matrices(
        &mut self,
        stream: &Arc<CudaStream>,
        matrix_a_f32: &[f32],
        matrix_b_f32: &[f32],
    ) -> Result<()> {
        // Convert f32 → f16 → u16 bitcast
        let to_u16 = |vals: &[f32]| -> Vec<u16> {
            vals.iter()
                .map(|&v| half::f16::from_f32(v).to_bits())
                .collect()
        };
        let a_u16 = to_u16(matrix_a_f32);
        let b_u16 = to_u16(matrix_b_f32);

        stream.memcpy_htod(&a_u16, &mut self.spike_matrix_a)?;
        stream.memcpy_htod(&b_u16, &mut self.spike_matrix_b)?;
        Ok(())
    }

    /// Compute norms, then CCF = A × B^T / (norm_a × norm_b).
    pub fn compute(&mut self, stream: &Arc<CudaStream>) -> Result<()> {
        // Step 1: per-row norms
        let norm_blocks = ((self.n_res as u32) + 255) / 256;
        let norm_cfg = LaunchConfig {
            grid_dim: (norm_blocks.max(1), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.norms_fn)
                .arg(&self.spike_matrix_a)
                .arg(&mut self.norm_a)
                .arg(&self.n_res)
                .arg(&self.n_bins_padded)
                .launch(norm_cfg)?;
            stream
                .launch_builder(&self.norms_fn)
                .arg(&self.spike_matrix_b)
                .arg(&mut self.norm_b)
                .arg(&self.n_res)
                .arg(&self.n_bins_padded)
                .launch(norm_cfg)?;
        }

        // Step 2: WMMA CCF
        // Grid: tiles over n_res_padded, Block: (32, 4) = 4 warps
        let tile_x = ((self.n_res_padded as u32) + 16 * 4 - 1) / (16 * 4);
        let tile_y = ((self.n_res_padded as u32) + 15) / 16;
        let ccf_cfg = LaunchConfig {
            grid_dim: (tile_x.max(1), tile_y.max(1), 1),
            block_dim: (32, 4, 1),
            shared_mem_bytes: 4 * 16 * 16 * 4, // 4 warps × 16×16 × sizeof(float)
        };
        unsafe {
            stream
                .launch_builder(&self.ccf_compute_fn)
                .arg(&self.spike_matrix_a)
                .arg(&self.spike_matrix_b)
                .arg(&mut self.ccf_output)
                .arg(&self.norm_a)
                .arg(&self.norm_b)
                .arg(&self.n_res)
                .arg(&self.n_res_padded)
                .arg(&self.n_bins_padded)
                .launch(ccf_cfg)?;
        }
        Ok(())
    }

    /// Download CCF matrix to CPU.
    pub fn download_ccf(&self, stream: &Arc<CudaStream>) -> Result<Vec<f32>> {
        let size = (self.n_res * self.n_res) as usize;
        let mut host = vec![0.0f32; size];
        stream.memcpy_dtoh(&self.ccf_output, &mut host)?;
        Ok(host)
    }

    /// Get matrix dimensions for external use.
    pub fn dimensions(&self) -> (i32, i32, i32) {
        (self.n_res, self.n_res_padded, self.n_bins_padded)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CCF matrix feature extraction (CPU-side)
// ─────────────────────────────────────────────────────────────────────────────

/// Per-residue features extracted from the CCF matrix.
#[derive(Debug, Clone, Default)]
pub struct CcfResidueFeatures {
    pub ccf_peak_lag: i32,
    pub ccf_peak_value: f32,
    pub ccf_width: f32,
    pub ccf_asymmetry: f32,
    pub ccf_reproducibility: f32,
}

/// Sanitize a float: replace NaN/Inf with 0.0.
/// GPU kernels can produce NaN from 0/0 or Inf from overflow.
fn sanitize(v: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

/// Extract per-residue CCF features from n_res × n_res CCF matrix.
/// Handles NaN values from GPU computation (norm division by zero when
/// a residue has zero spikes → zero norm → NaN in CCF output).
pub fn extract_ccf_features(ccf_matrix: &[f32], n_res: usize) -> Vec<CcfResidueFeatures> {
    if ccf_matrix.len() != n_res * n_res {
        log::warn!(
            "CCF matrix size {} != expected {}×{}, returning empty features",
            ccf_matrix.len(),
            n_res,
            n_res
        );
        return vec![CcfResidueFeatures::default(); n_res];
    }

    (0..n_res)
        .map(|r| {
            let row = &ccf_matrix[r * n_res..(r + 1) * n_res];

            // Filter out NaN/Inf values before any computation
            let clean_row: Vec<f32> = row.iter().map(|&v| sanitize(v)).collect();

            // Peak value and column (max off-diagonal, excluding self-correlation)
            let (peak_col, peak_val) = clean_row
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != r)
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(j, &v)| (j as i32, v))
                .unwrap_or((0, 0.0));

            // Width: count of columns with CCF > peak_val * 0.5 (FWHM proxy)
            let half_max = peak_val * 0.5;
            let width = clean_row.iter().filter(|&&v| v > half_max).count() as f32;

            // Asymmetry: (sum upper triangle - sum lower triangle) / total
            // Positive asymmetry = more correlation with higher-index residues
            let upper: f32 = if r + 1 < n_res {
                clean_row[r + 1..].iter().sum()
            } else {
                0.0
            };
            let lower: f32 = clean_row[..r].iter().sum();
            let denom = (upper.abs() + lower.abs()).max(1e-8);
            let asymmetry = (upper - lower) / denom;

            // Reproducibility: fraction of residues with CCF > 0.1
            // High reproducibility = this residue correlates with many others
            let reproducibility =
                clean_row.iter().filter(|&&v| v > 0.1).count() as f32 / n_res.max(1) as f32;

            CcfResidueFeatures {
                ccf_peak_lag: peak_col - r as i32,
                ccf_peak_value: sanitize(peak_val),
                ccf_width: width,
                ccf_asymmetry: sanitize(asymmetry),
                ccf_reproducibility: sanitize(reproducibility),
            }
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Spike matrix builder (CPU-side prep for CCF)
// ─────────────────────────────────────────────────────────────────────────────

/// Build time-binned spike matrices for CCF computation.
/// Returns (matrix_a, matrix_b) as f32 vectors, mean-centered, padded.
pub fn build_ccf_matrices(
    spikes_a: &[crate::fused_engine::GpuSpikeEvent],
    spikes_b: &[crate::fused_engine::GpuSpikeEvent],
    n_residues: usize,
    total_steps: i32,
    bin_size: i32,
) -> (Vec<f32>, Vec<f32>, i32, i32) {
    let n_bins = (total_steps / bin_size.max(1)) as usize;
    let n_res_padded = ((n_residues + 15) / 16) * 16;
    let n_bins_padded = ((n_bins + 15) / 16) * 16;

    let mut mat_a = vec![0.0f32; n_res_padded * n_bins_padded];
    let mut mat_b = vec![0.0f32; n_res_padded * n_bins_padded];

    // Accumulate spike intensities into bins.
    // LINK #1: the `resid >= n_residues` filter drops residue-entries outside the
    // reference frame [0, n_residues). n_residues is sourced from the PRIMARY
    // topology (= reference A) at coupled_md.rs, so in a two-system contrast this
    // is exactly the holo/topology_b ligand residues (appended at ordinals
    // >= n_residues). The drop is COUNTED and surfaced (logged) below rather than
    // silently discarded. Numeric accumulation is unchanged from before.
    let accumulate = |mat: &mut Vec<f32>, spikes: &[crate::fused_engine::GpuSpikeEvent]| -> u64 {
        let mut dropped_ligand: u64 = 0;
        for spike in spikes {
            let bin = (spike.timestep / bin_size.max(1)) as usize;
            if bin >= n_bins {
                continue;
            }
            let n = spike.n_residues.min(8) as usize;
            for r in 0..n {
                let resid = spike.nearby_residues[r];
                if resid < 0 {
                    continue;
                }
                if resid >= n_residues as i32 {
                    dropped_ligand += 1; // ligand / out-of-frame residue (Path-2 drop)
                    continue;
                }
                mat[resid as usize * n_bins_padded + bin] += spike.intensity;
            }
        }
        dropped_ligand
    };
    let dropped_a = accumulate(&mut mat_a, spikes_a);
    let dropped_b = accumulate(&mut mat_b, spikes_b);
    if dropped_a > 0 || dropped_b > 0 {
        log::info!(
            "  LINK #1 Path-2 (build_ccf_matrices): dropped {} A + {} B ligand/out-of-frame residue-entr(ies) (resid >= n_residues={})",
            dropped_a,
            dropped_b,
            n_residues
        );
    }

    // Mean-center each row (critical for WMMA — raw counts saturate FP16).
    // Also clamp to FP16 representable range [-65504, 65504] to prevent
    // NaN propagation in the WMMA kernel.
    let fp16_max = 65504.0f32;
    for r in 0..n_residues {
        let start = r * n_bins_padded;

        let sum_a: f32 = mat_a[start..start + n_bins].iter().sum();
        let mean_a = sum_a / n_bins.max(1) as f32;
        for v in mat_a[start..start + n_bins].iter_mut() {
            *v = (*v - mean_a).clamp(-fp16_max, fp16_max);
        }

        let sum_b: f32 = mat_b[start..start + n_bins].iter().sum();
        let mean_b = sum_b / n_bins.max(1) as f32;
        for v in mat_b[start..start + n_bins].iter_mut() {
            *v = (*v - mean_b).clamp(-fp16_max, fp16_max);
        }
    }

    (mat_a, mat_b, n_res_padded as i32, n_bins_padded as i32)
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU Transfer Entropy
// ─────────────────────────────────────────────────────────────────────────────

/// GPU-side binned Transfer Entropy computation.
///
/// Uses the same time-binned spike matrices as the CCF kernel.
/// Produces [n_res × n_res] TE matrices (A→B and B→A) plus
/// per-residue mutual information and causal flow direction.
pub struct TwinTeCompute {
    te_a_to_b: CudaSlice<f32>,   // [n_res × n_res]
    te_b_to_a: CudaSlice<f32>,   // [n_res × n_res]
    mean_a: CudaSlice<f32>,      // [n_res]
    mean_b: CudaSlice<f32>,      // [n_res]
    mutual_info: CudaSlice<f32>, // [n_res]
    causal_flow: CudaSlice<f32>, // [n_res]
    n_res: i32,
    n_bins: i32,
    n_res_padded: i32,
    n_bins_padded: i32,
    mean_fn: CudaFunction,
    te_fn: CudaFunction,
    mi_fn: CudaFunction,
}

impl TwinTeCompute {
    pub fn new(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        n_res: i32,
        n_bins: i32,
        n_res_padded: i32,
        n_bins_padded: i32,
    ) -> Result<Self> {
        let te_a_to_b = stream.alloc_zeros::<f32>((n_res * n_res) as usize)?;
        let te_b_to_a = stream.alloc_zeros::<f32>((n_res * n_res) as usize)?;
        let mean_a = stream.alloc_zeros::<f32>(n_res as usize)?;
        let mean_b = stream.alloc_zeros::<f32>(n_res as usize)?;
        let mutual_info = stream.alloc_zeros::<f32>(n_res as usize)?;
        let causal_flow = stream.alloc_zeros::<f32>(n_res as usize)?;

        let mean_fn = module
            .load_function("twin_compute_row_means")
            .context("twin_compute_row_means not found in PTX")?;
        let te_fn = module
            .load_function("twin_binned_te")
            .context("twin_binned_te not found in PTX")?;
        let mi_fn = module
            .load_function("twin_compute_mutual_info")
            .context("twin_compute_mutual_info not found in PTX")?;

        Ok(Self {
            te_a_to_b,
            te_b_to_a,
            mean_a,
            mean_b,
            mutual_info,
            causal_flow,
            n_res,
            n_bins,
            n_res_padded,
            n_bins_padded,
            mean_fn,
            te_fn,
            mi_fn,
        })
    }

    /// Compute means, TE matrices, MI, and causal flow.
    /// Spike matrices must already be uploaded to GPU (same as CCF input).
    pub fn compute(
        &mut self,
        stream: &Arc<CudaStream>,
        spike_matrix_a: &CudaSlice<u16>, // FP16 as u16, same as CCF
        spike_matrix_b: &CudaSlice<u16>,
    ) -> Result<()> {
        // Step 1: compute per-row means for binarization
        let mean_blocks = ((self.n_res as u32) + 255) / 256;
        let mean_cfg = LaunchConfig {
            grid_dim: (mean_blocks.max(1), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.mean_fn)
                .arg(spike_matrix_a)
                .arg(&mut self.mean_a)
                .arg(&self.n_res)
                .arg(&self.n_bins)
                .arg(&self.n_bins_padded)
                .launch(mean_cfg)?;
            stream
                .launch_builder(&self.mean_fn)
                .arg(spike_matrix_b)
                .arg(&mut self.mean_b)
                .arg(&self.n_res)
                .arg(&self.n_bins)
                .arg(&self.n_bins_padded)
                .launch(mean_cfg)?;
        }

        // Step 2: compute TE matrices
        let tile = 16u32;
        let grid_x = ((self.n_res as u32) + tile - 1) / tile;
        let grid_y = ((self.n_res as u32) + tile - 1) / tile;
        let te_cfg = LaunchConfig {
            grid_dim: (grid_x.max(1), grid_y.max(1), 1),
            block_dim: (tile, tile, 1),
            shared_mem_bytes: 0,
        };
        let lag = 1i32;
        unsafe {
            stream
                .launch_builder(&self.te_fn)
                .arg(spike_matrix_a)
                .arg(spike_matrix_b)
                .arg(&mut self.te_a_to_b)
                .arg(&mut self.te_b_to_a)
                .arg(&self.mean_a)
                .arg(&self.mean_b)
                .arg(&self.n_res)
                .arg(&self.n_res_padded)
                .arg(&self.n_bins)
                .arg(&self.n_bins_padded)
                .arg(&lag)
                .launch(te_cfg)?;
        }

        // Step 3: compute per-residue MI and causal flow from TE matrices
        let mi_blocks = ((self.n_res as u32) + 255) / 256;
        let mi_cfg = LaunchConfig {
            grid_dim: (mi_blocks.max(1), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.mi_fn)
                .arg(&self.te_a_to_b)
                .arg(&self.te_b_to_a)
                .arg(&mut self.mutual_info)
                .arg(&mut self.causal_flow)
                .arg(&self.n_res)
                .launch(mi_cfg)?;
        }

        Ok(())
    }

    /// Download per-residue MI and causal flow to CPU.
    pub fn download_per_residue(&self, stream: &Arc<CudaStream>) -> Result<(Vec<f32>, Vec<f32>)> {
        let mut mi = vec![0.0f32; self.n_res as usize];
        let mut cf = vec![0.0f32; self.n_res as usize];
        stream.memcpy_dtoh(&self.mutual_info, &mut mi)?;
        stream.memcpy_dtoh(&self.causal_flow, &mut cf)?;
        Ok((mi, cf))
    }

    /// Download per-residue TE(A→B) sum (total outgoing TE from each residue).
    pub fn download_te_per_residue(&self, stream: &Arc<CudaStream>) -> Result<Vec<f32>> {
        let n = self.n_res as usize;
        let mut te_ab = vec![0.0f32; n * n];
        stream.memcpy_dtoh(&self.te_a_to_b, &mut te_ab)?;
        // Sum across columns for each row: total outgoing TE from residue i
        let per_res: Vec<f32> = (0..n)
            .map(|i| {
                te_ab[i * n..(i + 1) * n]
                    .iter()
                    .filter(|&&v| v > 0.001)
                    .sum()
            })
            .collect();
        Ok(per_res)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Persistent Cooperative Coupling Kernel
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the persistent coupling kernel launch.
pub struct TwinCouplingConfig {
    pub sensitivity_boost: f32,
    pub max_reduction_fraction: f32,
    pub decay_constant: f32,
    pub recovery_rate: f32,
    pub recovery_interval: u32,
    pub total_steps: u32,
}

/// GPU state for the persistent cooperative coupling kernel.
///
/// This kernel runs for the ENTIRE simulation on a dedicated CUDA stream.
/// It handles ring buffer exchange + threshold adaptation without returning
/// to host. Physics kernels signal completion via TwinSignal atomic flags.
pub struct TwinCouplingPersistent {
    kernel_fn: CudaFunction,
}

impl TwinCouplingPersistent {
    /// Load the persistent coupling kernel from PTX.
    pub fn new(module: &Arc<CudaModule>) -> Result<Self> {
        let kernel_fn = module
            .load_function("twin_coupling_persistent")
            .context("twin_coupling_persistent not found in PTX")?;
        Ok(Self { kernel_fn })
    }

    /// Launch the persistent coupling kernel. This call returns immediately —
    /// the kernel runs asynchronously for `config.total_steps` steps.
    ///
    /// The kernel spin-waits on signal flags, so the physics streams must
    /// call `TwinSignal::signal_a/b()` after each step's LADD kernels.
    ///
    /// # Safety
    /// All buffer references must remain valid for the duration of the
    /// simulation (the kernel doesn't return until total_steps complete).
    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        stream: &Arc<CudaStream>,
        // Signal flags
        signal: &mut TwinSignal,
        // Ring buffers
        ring_a: &mut TwinRingBuffer,
        ring_b: &mut TwinRingBuffer,
        ring_capacity: u32,
        // Spike buffers (from physics engines)
        spike_buf_a: &CudaSlice<u8>,
        spike_count_a: &CudaSlice<i32>,
        spike_buf_b: &CudaSlice<u8>,
        spike_count_b: &CudaSlice<i32>,
        // Threshold buffers
        thresh_a: &mut CudaSlice<f32>,
        base_a: &CudaSlice<f32>,
        thresh_b: &mut CudaSlice<f32>,
        base_b: &CudaSlice<f32>,
        // Grid geometry
        grid_dims: (i32, i32, i32),
        grid_origin: (f32, f32, f32),
        voxel_size: f32,
        // Config
        config: &TwinCouplingConfig,
    ) -> Result<()> {
        // 2 blocks × 256 threads — minimal SM footprint.
        // The persistent kernel only needs 2 blocks (one per stream direction).
        // Using 168 blocks would occupy ALL SMs and starve the physics kernels.
        // With 2 blocks: grid.sync() across 2 blocks is near-instant, and
        // the remaining 82 SMs are free for the fused physics kernel.
        let launch_cfg = LaunchConfig {
            grid_dim: (2, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&self.kernel_fn)
                // Signal flags (params 0-1)
                .arg(&mut signal.flag_a)
                .arg(&mut signal.flag_b)
                // Ring A: buffer, head, tail, overflow (params 2-5)
                .arg(&mut ring_a.buffer)
                .arg(&mut ring_a.head)
                .arg(&mut ring_a.tail)
                .arg(&mut ring_a.overflow)
                // Ring B: buffer, head, tail, overflow (params 6-9)
                .arg(&mut ring_b.buffer)
                .arg(&mut ring_b.head)
                .arg(&mut ring_b.tail)
                .arg(&mut ring_b.overflow)
                // Ring capacity (param 10)
                .arg(&ring_capacity)
                // Spike buffers (params 11-14)
                .arg(spike_buf_a)
                .arg(spike_count_a)
                .arg(spike_buf_b)
                .arg(spike_count_b)
                // Threshold buffers (params 15-18)
                .arg(thresh_a)
                .arg(base_a)
                .arg(thresh_b)
                .arg(base_b)
                // Grid geometry (params 19-25)
                .arg(&grid_dims.0)
                .arg(&grid_dims.1)
                .arg(&grid_dims.2)
                .arg(&grid_origin.0)
                .arg(&grid_origin.1)
                .arg(&grid_origin.2)
                .arg(&voxel_size)
                // Coupling parameters (params 26-30)
                .arg(&config.sensitivity_boost)
                .arg(&config.max_reduction_fraction)
                .arg(&config.decay_constant)
                .arg(&config.recovery_rate)
                .arg(&config.recovery_interval)
                // Total steps (param 31)
                .arg(&config.total_steps)
                .launch_cooperative(launch_cfg)?;
        }

        log::info!("  Persistent coupling kernel launched: 2 blocks × 256 threads (minimal SM footprint), {} steps",
            config.total_steps);
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_spike_event_size() {
        assert_eq!(
            std::mem::size_of::<RingSpikeEvent>(),
            48,
            "RingSpikeEvent must be exactly 48 bytes to match CUDA struct"
        );
    }

    #[test]
    fn test_ccf_feature_extraction_identity() {
        // 3×3 CCF with known structure
        let ccf = vec![1.0, 0.2, 0.1, 0.2, 1.0, 0.3, 0.1, 0.3, 1.0];
        let features = extract_ccf_features(&ccf, 3);
        assert_eq!(features.len(), 3);
        // Residue 2's highest off-diagonal is column 1 (0.3)
        assert_eq!(features[2].ccf_peak_lag, 1 - 2); // relative: -1
        assert!((features[2].ccf_peak_value - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_ccf_feature_extraction_zeros() {
        let ccf = vec![0.0; 9]; // 3×3 all zeros
        let features = extract_ccf_features(&ccf, 3);
        assert_eq!(features.len(), 3);
        for f in &features {
            assert_eq!(f.ccf_peak_value, 0.0);
        }
    }

    #[test]
    fn test_build_ccf_matrices_padding() {
        // 5 residues → padded to 16
        // 100 steps / bin_size=10 = 10 bins → padded to 16
        let (mat_a, _mat_b, n_res_p, n_bins_p) = build_ccf_matrices(&[], &[], 5, 100, 10);
        assert_eq!(n_res_p, 16);
        assert_eq!(n_bins_p, 16);
        assert_eq!(mat_a.len(), 16 * 16);
    }

    #[test]
    fn test_build_ccf_matrices_mean_centering() {
        use crate::fused_engine::GpuSpikeEvent;
        // Create spikes at residue 0, various bins
        let spikes: Vec<GpuSpikeEvent> = (0..10)
            .map(|t| {
                let mut s = GpuSpikeEvent::default();
                s.timestep = t * 10;
                s.intensity = 1.0;
                s.nearby_residues[0] = 0;
                s.n_residues = 1;
                s
            })
            .collect();

        let (mat_a, _, n_res_p, n_bins_p) = build_ccf_matrices(&spikes, &[], 3, 100, 10);

        // After mean-centering, row 0 should sum to ~0
        let row_sum: f32 = mat_a[0..n_bins_p as usize].iter().sum();
        assert!(
            row_sum.abs() < 0.01,
            "Mean-centered row should sum to ~0, got {}",
            row_sum
        );
    }
}


/// Whole coupling owner state at the externally established all-producer fence.
/// Pointer tables are allocation bindings and are rebuilt from verified owners.
#[derive(serde::Serialize,serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TwinOwnerContinuationV1 {
    schema_version:u32, owner_kind:String, shape:Vec<u64>,
    #[serde(with="crate::checkpoint_codec::byte_map")]
    planes:std::collections::BTreeMap<String,Vec<u8>>, #[serde(with="crate::checkpoint_codec::bytes")] host:Vec<u8>,
}
fn capture_twin_planes(stream:&Arc<CudaStream>,kind:&str,shape:Vec<u64>,
    regions:&[(String,u64,usize)],host:Vec<u8>)->Result<TwinOwnerContinuationV1> {
    stream.synchronize()?;let mut planes=std::collections::BTreeMap::new();
    for (name,pointer,len) in regions {anyhow::ensure!(!planes.contains_key(name),"duplicate coupling plane");
        planes.insert(name.clone(),unsafe{crate::scientific_checkpoint::device::read_raw(stream,*pointer,*len)?});}
    Ok(TwinOwnerContinuationV1{schema_version:1,owner_kind:kind.into(),shape,planes,host})
}
fn validate_twin_planes(saved:&TwinOwnerContinuationV1,kind:&str,shape:&[u64],regions:&[(String,u64,usize)])->Result<()> {
    anyhow::ensure!(saved.schema_version==1 && saved.owner_kind==kind && saved.shape==shape,"coupling owner identity/shape changed");
    anyhow::ensure!(saved.planes.len()==regions.len(),"coupling plane population changed");
    for (name,pointer,len) in regions {anyhow::ensure!(*pointer!=0 || *len==0,"unbound coupling plane {name}");
        anyhow::ensure!(saved.planes.get(name).is_some_and(|bytes|bytes.len()==*len),"coupling plane size changed: {name}");}Ok(())
}
fn restore_twin_planes(stream:&Arc<CudaStream>,saved:&TwinOwnerContinuationV1,regions:&[(String,u64,usize)])->Result<()> {
    for (name,pointer,_) in regions {unsafe {crate::scientific_checkpoint::device::write_raw(stream,*pointer,&saved.planes[name])?;}}Ok(())
}
impl TwinSpikePublication {
    fn continuation_regions(&self,stream:&Arc<CudaStream>)->Vec<(String,u64,usize)> {
        vec![("cursor".into(),self.cursor.device_ptr(stream).0,4),
            ("continuation_pending".into(),self.continuation_pending.device_ptr(stream).0,4),
            ("failure".into(),self.failure.device_ptr(stream).0,4)]
    }
    pub fn continuation_allocation_plan(&self,stream:&Arc<CudaStream>)->Result<Vec<(String,u64)>> {
        self.continuation_regions(stream).into_iter().map(|(name,_,bytes)|Ok((name,u64::try_from(bytes)?))).collect()
    }
    pub fn capture_continuation(&self,stream:&Arc<CudaStream>)->Result<TwinOwnerContinuationV1> {
        capture_twin_planes(stream,"publication",vec![self.source_capacity as u64,self.destination_ptr.is_some() as u64],
            &self.continuation_regions(stream),Vec::new())
    }
    pub fn restore_continuation(&self,stream:&Arc<CudaStream>,saved:&TwinOwnerContinuationV1)->Result<()> {
        let regions=self.continuation_regions(stream);
        validate_twin_planes(saved,"publication",&[self.source_capacity as u64,self.destination_ptr.is_some() as u64],&regions)?;
        anyhow::ensure!(saved.host.is_empty(),"unexpected publication host state");
        let cursor=i32::from_le_bytes(saved.planes["cursor"].as_slice().try_into()?);
        anyhow::ensure!(cursor>=0 && cursor as u32<=self.source_capacity,"invalid publication cursor");
        // A deferred source is valid checkpoint state. Do not acknowledge/reset it.
        restore_twin_planes(stream,saved,&regions)
    }
}
impl TwinRingBuffer {
    fn continuation_regions(&self,stream:&Arc<CudaStream>)->Vec<(String,u64,usize)> {
        let mut out=vec![("buffer".into(),self.buffer.device_ptr(stream).0,self.buffer.len()),
            ("head".into(),self.head.device_ptr(stream).0,4),("tail".into(),self.tail.device_ptr(stream).0,4),
            ("overflow".into(),self.overflow.device_ptr(stream).0,4),
            ("staging".into(),self.d_staging.device_ptr(stream).0,self.d_staging.len())];
        for (i,receipt) in self.recipient_receipts.iter().enumerate(){out.push((format!("recipient_{i}"),receipt.device_ptr(stream).0,receipt.len()));}
        if let Some(lossless)=&self.lossless {
            out.push(("lossless_failure".into(),lossless.failure.device_ptr(stream).0,4));
            for (i,window) in lossless.windows.iter().enumerate(){out.push((format!("window_{i}"),window.device_ptr(stream).0,window.len()));}
        }out
    }
    fn continuation_shape(&self)->Vec<u64>{vec![self.capacity as u64,self.recipient_receipts.len() as u64,
        self.lossless.is_some() as u64,self.lossless.as_ref().map_or(0,|x|x.windows.len()) as u64]}
    pub fn continuation_allocation_plan(&self,stream:&Arc<CudaStream>)->Result<Vec<(String,u64)>> {
        let mut rows=self.continuation_regions(stream).into_iter().map(|(name,_,bytes)|Ok((name,u64::try_from(bytes)?)))
            .collect::<Result<Vec<_>>>()?;
        rows.push(("host_staging_and_participants_wire".into(),crate::checkpoint_codec::encoded_size(&(&self.staging,self.lossless.as_ref().map(|v|&v.participating)),u64::MAX)?));Ok(rows)
    }
    pub fn capture_continuation(&self,stream:&Arc<CudaStream>)->Result<TwinOwnerContinuationV1> {
        let host=crate::checkpoint_codec::to_bytes(&(&self.staging,self.lossless.as_ref().map(|v|&v.participating)))?;
        capture_twin_planes(stream,"ring",self.continuation_shape(),&self.continuation_regions(stream),host)
    }
    /// Invoke before recapturing graphs. Growing the CPU staging plane is safe
    /// only at that construction boundary; graph-visible pointers never change.
    pub fn restore_continuation_before_capture(&mut self,stream:&Arc<CudaStream>,saved:&TwinOwnerContinuationV1)->Result<()> {
        let (staging,participating):(Vec<RingSpikeEvent>,Option<Vec<usize>>)=crate::checkpoint_codec::from_bytes(&saved.host)?;
        anyhow::ensure!(self.lossless.as_ref().map(|v|&v.participating)==participating.as_ref(),"ring recipient identity changed");
        let staging_bytes=saved.planes.get("staging").context("missing ring staging")?.len();
        anyhow::ensure!(staging_bytes%std::mem::size_of::<RingSpikeEvent>()==0 && staging_bytes>=staging.len()*48,
            "ring staging extent does not cover retained events");
        if staging_bytes!=self.d_staging.len(){self.d_staging=stream.alloc_zeros::<u8>(staging_bytes)?;}
        let regions=self.continuation_regions(stream);validate_twin_planes(saved,"ring",&self.continuation_shape(),&regions)?;
        // recipient_pointers was freshly built by new_lossless_broadcast from
        // this process's receipt allocations. No previous CUDA pointer is read.
        restore_twin_planes(stream,saved,&regions)?;self.staging=staging;Ok(())
    }
}

impl TwinSignal {
    fn continuation_regions(&self,stream:&Arc<CudaStream>)->Vec<(String,u64,usize)> {vec![("flag_a".into(),self.flag_a.device_ptr(stream).0,self.flag_a.len()*4),("flag_b".into(),self.flag_b.device_ptr(stream).0,self.flag_b.len()*4)]}
    pub fn capture_continuation(&self,stream:&Arc<CudaStream>)->Result<TwinOwnerContinuationV1> {capture_twin_planes(stream,"TwinSignal",vec![],&self.continuation_regions(stream),Vec::new())}
    pub fn restore_continuation(&self,stream:&Arc<CudaStream>,saved:&TwinOwnerContinuationV1)->Result<()> {let regions=self.continuation_regions(stream);
        validate_twin_planes(saved,"TwinSignal",&vec![],&regions)?;anyhow::ensure!(saved.host.is_empty(),"unexpected coupling host state");restore_twin_planes(stream,saved,&regions)}
}

impl TwinCcfCompute {
    fn continuation_regions(&self,stream:&Arc<CudaStream>)->Vec<(String,u64,usize)> {vec![("spike_matrix_a".into(),self.spike_matrix_a.device_ptr(stream).0,self.spike_matrix_a.len()*2),("spike_matrix_b".into(),self.spike_matrix_b.device_ptr(stream).0,self.spike_matrix_b.len()*2),("ccf_output".into(),self.ccf_output.device_ptr(stream).0,self.ccf_output.len()*4),("norm_a".into(),self.norm_a.device_ptr(stream).0,self.norm_a.len()*4),("norm_b".into(),self.norm_b.device_ptr(stream).0,self.norm_b.len()*4)]}
    pub fn capture_continuation(&self,stream:&Arc<CudaStream>)->Result<TwinOwnerContinuationV1> {capture_twin_planes(stream,"TwinCcfCompute",vec![self.n_res as u64,self.n_res_padded as u64,self.n_bins_padded as u64],&self.continuation_regions(stream),Vec::new())}
    pub fn restore_continuation(&self,stream:&Arc<CudaStream>,saved:&TwinOwnerContinuationV1)->Result<()> {let regions=self.continuation_regions(stream);
        validate_twin_planes(saved,"TwinCcfCompute",&vec![self.n_res as u64,self.n_res_padded as u64,self.n_bins_padded as u64],&regions)?;anyhow::ensure!(saved.host.is_empty(),"unexpected coupling host state");restore_twin_planes(stream,saved,&regions)}
}

impl TwinTeCompute {
    fn continuation_regions(&self,stream:&Arc<CudaStream>)->Vec<(String,u64,usize)> {vec![("te_a_to_b".into(),self.te_a_to_b.device_ptr(stream).0,self.te_a_to_b.len()*4),("te_b_to_a".into(),self.te_b_to_a.device_ptr(stream).0,self.te_b_to_a.len()*4),("mean_a".into(),self.mean_a.device_ptr(stream).0,self.mean_a.len()*4),("mean_b".into(),self.mean_b.device_ptr(stream).0,self.mean_b.len()*4),("mutual_info".into(),self.mutual_info.device_ptr(stream).0,self.mutual_info.len()*4),("causal_flow".into(),self.causal_flow.device_ptr(stream).0,self.causal_flow.len()*4)]}
    pub fn capture_continuation(&self,stream:&Arc<CudaStream>)->Result<TwinOwnerContinuationV1> {capture_twin_planes(stream,"TwinTeCompute",vec![self.n_res as u64,self.n_bins as u64,self.n_res_padded as u64,self.n_bins_padded as u64],&self.continuation_regions(stream),Vec::new())}
    pub fn restore_continuation(&self,stream:&Arc<CudaStream>,saved:&TwinOwnerContinuationV1)->Result<()> {let regions=self.continuation_regions(stream);
        validate_twin_planes(saved,"TwinTeCompute",&vec![self.n_res as u64,self.n_bins as u64,self.n_res_padded as u64,self.n_bins_padded as u64],&regions)?;anyhow::ensure!(saved.host.is_empty(),"unexpected coupling host state");restore_twin_planes(stream,saved,&regions)}
}
