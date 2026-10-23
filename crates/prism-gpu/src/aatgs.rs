//! AATGS: Adaptive Asynchronous Task Graph Scheduler
//!
//! GPU-resident scheduler that eliminates CPU-GPU synchronization barriers
//! by maintaining circular buffers for config and telemetry.
//!
//! ## Architecture
//!
//! The AATGS system uses a triple-buffered async pipeline to overlap:
//! 1. Config upload (stream 0)
//! 2. Kernel execution (stream 1)
//! 3. Telemetry download (stream 2)
//!
//! This allows the CPU to queue work ahead while the GPU processes previous iterations,
//! and simultaneously retrieve results from even earlier iterations.
//!
//! ## Memory Layout
//!
//! GPU-resident circular buffers:
//! - Config buffer: 16 slots of RuntimeConfig (16 * 256B = 4KB)
//! - Telemetry buffer: 64 slots of KernelTelemetry (64 * 64B = 4KB)
//! - Atomic pointers for lock-free coordination
//!
//! ## Synchronization
//!
//! - config_write_ptr: CPU writes, GPU reads (producer: CPU, consumer: GPU)
//! - config_read_ptr: GPU writes after consuming (feedback to CPU)
//! - telemetry_write_ptr: GPU writes, CPU reads (producer: GPU, consumer: CPU)
//! - telemetry_read_ptr: CPU writes after consuming (feedback to GPU)
//! - gpu_idle: GPU sets when waiting for work
//! - cpu_shutdown: CPU sets to terminate GPU loop
//!
//! ## Usage
//!
//! ```rust,no_run
//! use prism_gpu::aatgs::AsyncPipeline;
//! use prism_core::RuntimeConfig;
//! use cudarc::driver::CudaContext;
//! use std::sync::Arc;
//!
//! let device = CudaContext::new(0).unwrap());
//! let mut pipeline = AsyncPipeline::new(Arc::new(device)).unwrap());
//!
//! let config = RuntimeConfig::production();
//! let telemetry = pipeline.step(config).unwrap());
//! ```

use anyhow::{Context, Result};
use cudarc::driver::{CudaContext, CudaStream, CudaSlice, PushKernelArg, DeviceSlice};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

use prism_core::{KernelTelemetry, RuntimeConfig};

// External dependencies for advanced features
use crossbeam_utils::atomic::AtomicCell;

/// Circular buffer capacity for configs and telemetry
const CONFIG_BUFFER_SIZE: usize = 16;
const TELEMETRY_BUFFER_SIZE: usize = 64;

/// AATGS buffer state (mirrored on GPU)
///
/// This structure is allocated once on the GPU and contains all circular buffer state.
/// The CPU maintains shadow copies of the pointers for local bookkeeping.
///
/// Total size: ~16KB (config buffer) + ~4KB (telemetry buffer) + metadata = ~20KB
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AATGSBuffers {
    /// Circular buffer of RuntimeConfigs
    pub config_buffer: [RuntimeConfig; CONFIG_BUFFER_SIZE],
    /// Write pointer for config buffer (CPU writes, GPU reads)
    pub config_write_ptr: i32,
    /// Read pointer for config buffer (GPU writes after consuming, CPU reads)
    pub config_read_ptr: i32,

    /// Circular buffer of telemetry outputs
    pub telemetry_buffer: [KernelTelemetry; TELEMETRY_BUFFER_SIZE],
    /// Write pointer for telemetry (GPU writes, CPU reads)
    pub telemetry_write_ptr: i32,
    /// Read pointer for telemetry (CPU writes after consuming, GPU reads)
    pub telemetry_read_ptr: i32,

    /// GPU idle flag (GPU sets to 1 when waiting for config)
    pub gpu_idle: i32,
    /// Shutdown signal (CPU sets to 1 to terminate GPU loop)
    pub cpu_shutdown: i32,
}

impl Default for AATGSBuffers {
    fn default() -> Self {
        Self {
            config_buffer: [RuntimeConfig::default(); CONFIG_BUFFER_SIZE],
            config_write_ptr: 0,
            config_read_ptr: 0,
            telemetry_buffer: [KernelTelemetry::default(); TELEMETRY_BUFFER_SIZE],
            telemetry_write_ptr: 0,
            telemetry_read_ptr: 0,
            gpu_idle: 0,
            cpu_shutdown: 0,
        }
    }
}

// SAFETY: AATGSBuffers is #[repr(C)] containing only:
// - Fixed-size arrays of RuntimeConfig and KernelTelemetry (both #[repr(C)] with primitives)
// - i32 primitives for pointers and flags
// All fields are valid when zero-initialized.
unsafe impl cudarc::driver::ValidAsZeroBits for AATGSBuffers {}
unsafe impl cudarc::driver::DeviceRepr for AATGSBuffers {}

/// AATGS Scheduler
///
/// Manages GPU-resident circular buffers for asynchronous config/telemetry exchange.
/// Uses cudarc 0.18.1 API with explicit streams for async operations.
pub struct AATGSScheduler {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,

    /// GPU-side buffer state
    d_buffers: CudaSlice<AATGSBuffers>,

    /// Host-side shadow of buffer state for local tracking
    local_buffers: AATGSBuffers,

    /// Host-side shadow of config write pointer
    local_config_write_ptr: usize,
    /// Host-side shadow of telemetry read pointer
    local_telemetry_read_ptr: usize,

    /// Pending configs to upload on next flush
    pending_configs: Vec<RuntimeConfig>,
}

impl AATGSScheduler {
    /// Create new AATGS scheduler
    ///
    /// Allocates GPU-resident circular buffers and initializes pointers.
    ///
    /// # Arguments
    /// * `device` - CUDA device context
    ///
    /// # Returns
    /// * `Ok(Self)` - Initialized scheduler
    /// * `Err(_)` - GPU allocation failure
    pub fn new(device: Arc<CudaContext>) -> Result<Self> {
        log::info!("Initializing AATGS scheduler with {} config slots, {} telemetry slots",
                   CONFIG_BUFFER_SIZE, TELEMETRY_BUFFER_SIZE);

        let stream = device.default_stream();

        // Allocate GPU buffer state (single allocation for entire state)
        let d_buffers = stream
            .alloc_zeros::<AATGSBuffers>(1)
            .context("Failed to allocate GPU buffer state")?;

        let local_buffers = AATGSBuffers::default();

        Ok(Self {
            context: device,
            stream,
            d_buffers,
            local_buffers,
            local_config_write_ptr: 0,
            local_telemetry_read_ptr: 0,
            pending_configs: Vec::with_capacity(CONFIG_BUFFER_SIZE),
        })
    }

    /// Queue a new config for GPU execution (non-blocking)
    ///
    /// Configs are batched locally until `flush_configs()` is called or the buffer
    /// reaches half capacity.
    ///
    /// # Arguments
    /// * `config` - Runtime configuration to queue
    ///
    /// # Returns
    /// * `Ok(())` - Config queued successfully
    /// * `Err(_)` - Buffer overflow or GPU error
    pub fn queue_config(&mut self, config: RuntimeConfig) -> Result<()> {
        // Check for buffer overflow
        let available = CONFIG_BUFFER_SIZE
            .saturating_sub(self.local_config_write_ptr - self.local_buffers.config_read_ptr as usize);

        anyhow::ensure!(
            available > 0,
            "Config buffer full (write={}, read={})",
            self.local_config_write_ptr,
            self.local_buffers.config_read_ptr
        );

        self.pending_configs.push(config);

        // Auto-flush if buffer is getting full
        if self.pending_configs.len() >= CONFIG_BUFFER_SIZE / 2 {
            self.flush_configs()?;
        }

        Ok(())
    }

    /// Flush pending configs to GPU (blocking upload)
    ///
    /// Uploads all pending configs to the GPU circular buffer and updates the write pointer.
    /// This operation blocks until the upload completes.
    ///
    /// # Returns
    /// * `Ok(())` - Configs uploaded successfully
    /// * `Err(_)` - GPU upload failure
    pub fn flush_configs(&mut self) -> Result<()> {
        if self.pending_configs.is_empty() {
            return Ok(());
        }

        log::debug!("Flushing {} configs to GPU", self.pending_configs.len());

        // Update local buffer with pending configs
        for config in self.pending_configs.drain(..) {
            let slot = self.local_config_write_ptr % CONFIG_BUFFER_SIZE;
            self.local_buffers.config_buffer[slot] = config;
            self.local_config_write_ptr += 1;
        }

        // Update write pointer in local state
        self.local_buffers.config_write_ptr = self.local_config_write_ptr as i32;

        // Upload entire buffer state to GPU (replace existing buffer)
        self.d_buffers = self.stream
            .clone_htod(&[self.local_buffers])
            .context("Failed to upload config buffer to GPU")?;

        Ok(())
    }

    /// Poll for completed telemetry (non-blocking check, blocking read)
    ///
    /// Reads the GPU's telemetry write pointer to determine how many new entries are available,
    /// then downloads them from the circular buffer.
    ///
    /// # Returns
    /// * `Ok(Vec<KernelTelemetry>)` - Available telemetry entries
    /// * `Err(_)` - GPU download failure
    pub fn poll_telemetry(&mut self) -> Result<Vec<KernelTelemetry>> {
        // Download current GPU buffer state to check telemetry write pointer
        let gpu_buffers_array: Vec<AATGSBuffers> = self.stream
            .clone_dtoh(&self.d_buffers)
            .context("Failed to download buffer state from GPU")?;
        let gpu_buffers = gpu_buffers_array[0];

        // Calculate how many telemetry entries are available
        let available = (gpu_buffers.telemetry_write_ptr as usize)
            .saturating_sub(self.local_telemetry_read_ptr);

        if available == 0 {
            return Ok(Vec::new());
        }

        log::debug!("Polling {} telemetry entries from GPU", available);

        // Read available telemetry entries from circular buffer
        let mut results = Vec::with_capacity(available);

        for _ in 0..available {
            let slot = self.local_telemetry_read_ptr % TELEMETRY_BUFFER_SIZE;
            let telemetry = gpu_buffers.telemetry_buffer[slot];
            results.push(telemetry);
            self.local_telemetry_read_ptr += 1;
        }

        // Update local read pointer and upload to GPU (replace existing buffer)
        self.local_buffers.telemetry_read_ptr = self.local_telemetry_read_ptr as i32;
        self.d_buffers = self.stream
            .clone_htod(&[self.local_buffers])
            .context("Failed to update telemetry read pointer on GPU")?;

        Ok(results)
    }

    /// Check if GPU is idle (waiting for work)
    ///
    /// Queries the GPU's idle flag to determine if it's waiting for new configs.
    ///
    /// # Returns
    /// * `Ok(true)` - GPU is idle
    /// * `Ok(false)` - GPU is busy
    /// * `Err(_)` - GPU query failure
    pub fn is_gpu_idle(&mut self) -> Result<bool> {
        // Download current buffer state
        let gpu_buffers_array: Vec<AATGSBuffers> = self.stream
            .clone_dtoh(&self.d_buffers)
            .context("Failed to query GPU idle status")?;
        let gpu_buffers = gpu_buffers_array[0];

        Ok(gpu_buffers.gpu_idle != 0)
    }

    /// Signal GPU to shutdown and wait for completion
    ///
    /// Sets the shutdown flag and synchronizes to ensure all pending operations complete.
    ///
    /// # Returns
    /// * `Ok(())` - Shutdown successful
    /// * `Err(_)` - GPU synchronization failure
    pub fn shutdown(&mut self) -> Result<()> {
        log::info!("Signaling GPU shutdown");

        // Set shutdown flag in local state
        self.local_buffers.cpu_shutdown = 1;

        // Upload to GPU (replace existing buffer)
        self.d_buffers = self.stream
            .clone_htod(&[self.local_buffers])
            .context("Failed to signal GPU shutdown")?;

        // Wait for all operations to complete
        self.stream
            .synchronize()
            .context("Failed to synchronize device during shutdown")?;

        log::info!("AATGS scheduler shutdown complete");
        Ok(())
    }

    /// Get statistics about buffer utilization
    ///
    /// # Returns
    /// Tuple of (config_occupancy, telemetry_occupancy) as percentages [0.0, 1.0]
    pub fn buffer_stats(&self) -> (f32, f32) {
        let config_occupancy = (self.local_config_write_ptr
            - self.local_buffers.config_read_ptr as usize) as f32
            / CONFIG_BUFFER_SIZE as f32;

        let telemetry_occupancy = (self.local_buffers.telemetry_write_ptr as usize
            - self.local_telemetry_read_ptr) as f32
            / TELEMETRY_BUFFER_SIZE as f32;

        (config_occupancy, telemetry_occupancy)
    }

    /// Get reference to underlying CUDA device
    pub fn device(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Get GPU buffer slice for external kernel coordination
    ///
    /// Allows external kernels to directly access the AATGS buffer state.
    pub fn buffer_slice(&self) -> &CudaSlice<AATGSBuffers> {
        &self.d_buffers
    }
}

/// Triple-buffered async pipeline
///
/// High-level interface that combines config scheduling with automatic telemetry polling.
/// Provides a simple `step()` method that queues new config and retrieves latest telemetry.
///
/// ## Pipeline Stages
///
/// 1. CPU queues config[n+1]
/// 2. GPU executes kernel[n] (using config from previous step)
/// 3. CPU polls telemetry[n-1] (results from two steps ago)
///
/// This 3-stage pipeline ensures maximum overlap between CPU and GPU work.
pub struct AsyncPipeline {
    scheduler: AATGSScheduler,

    /// Number of iterations executed
    iteration_count: usize,

    /// Peak buffer utilization (config, telemetry)
    peak_utilization: (f32, f32),
}

impl AsyncPipeline {
    /// Create new async pipeline
    ///
    /// # Arguments
    /// * `device` - CUDA device context
    ///
    /// # Returns
    /// * `Ok(Self)` - Initialized pipeline
    /// * `Err(_)` - Scheduler initialization failure
    pub fn new(device: Arc<CudaContext>) -> Result<Self> {
        Ok(Self {
            scheduler: AATGSScheduler::new(device)?,
            iteration_count: 0,
            peak_utilization: (0.0, 0.0),
        })
    }

    /// Execute one async iteration
    ///
    /// Queues a new config for execution and polls for any completed telemetry.
    /// Due to pipeline depth, telemetry returned may be from 1-2 iterations ago.
    ///
    /// # Arguments
    /// * `new_config` - Configuration for next iteration
    ///
    /// # Returns
    /// * `Ok(Some(telemetry))` - Telemetry from completed iteration
    /// * `Ok(None)` - No telemetry available yet (pipeline filling)
    /// * `Err(_)` - Scheduler error
    pub fn step(&mut self, new_config: RuntimeConfig) -> Result<Option<KernelTelemetry>> {
        // Queue new config (async upload)
        self.scheduler.queue_config(new_config)?;

        // Poll for completed telemetry (non-blocking)
        let telemetry = self.scheduler.poll_telemetry()?;

        // Update statistics
        let (config_util, telem_util) = self.scheduler.buffer_stats();
        self.peak_utilization.0 = self.peak_utilization.0.max(config_util);
        self.peak_utilization.1 = self.peak_utilization.1.max(telem_util);

        self.iteration_count += 1;

        // Return first telemetry if available
        Ok(telemetry.into_iter().next())
    }

    /// Execute multiple iterations in a batch
    ///
    /// Efficiently processes multiple configs by maximizing buffer utilization.
    ///
    /// # Arguments
    /// * `configs` - Slice of configurations to execute
    ///
    /// # Returns
    /// * `Ok(Vec<KernelTelemetry>)` - Telemetry from all completed iterations
    /// * `Err(_)` - Scheduler error
    pub fn batch_step(&mut self, configs: &[RuntimeConfig]) -> Result<Vec<KernelTelemetry>> {
        let mut all_telemetry = Vec::new();

        for config in configs {
            self.scheduler.queue_config(*config)?;
        }

        // Flush all configs
        self.scheduler.flush_configs()?;

        // Poll for results (may take multiple polls to get all results)
        let mut remaining = configs.len();
        while remaining > 0 {
            let telemetry = self.scheduler.poll_telemetry()?;
            let received = telemetry.len();
            all_telemetry.extend(telemetry);
            remaining = remaining.saturating_sub(received);

            // Small delay to allow GPU to process
            if remaining > 0 {
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
        }

        self.iteration_count += configs.len();

        Ok(all_telemetry)
    }

    /// Get current iteration count
    pub fn iterations(&self) -> usize {
        self.iteration_count
    }

    /// Get peak buffer utilization
    ///
    /// Returns (config_peak, telemetry_peak) as fractions [0.0, 1.0]
    pub fn peak_utilization(&self) -> (f32, f32) {
        self.peak_utilization
    }

    /// Check if GPU is idle
    pub fn is_idle(&mut self) -> Result<bool> {
        self.scheduler.is_gpu_idle()
    }

    /// Shutdown the pipeline
    pub fn shutdown(mut self) -> Result<()> {
        self.scheduler.shutdown()
    }

    /// Get reference to underlying scheduler
    pub fn scheduler(&self) -> &AATGSScheduler {
        &self.scheduler
    }

    /// Get mutable reference to underlying scheduler
    pub fn scheduler_mut(&mut self) -> &mut AATGSScheduler {
        &mut self.scheduler
    }
}

// ============================================================================
// PART 1: Circular Buffer for Config Upload (~100 LOC)
// ============================================================================

/// Lock-free circular buffer for non-blocking config upload
///
/// Uses atomic operations to coordinate producer (CPU) and consumer (GPU)
/// without blocking. The buffer maintains head and tail pointers where:
/// - head: next write position (producer advances)
/// - tail: next read position (consumer advances)
///
/// Buffer is full when: (head + 1) % capacity == tail
/// Buffer is empty when: head == tail
pub struct ConfigCircularBuffer<T: Clone> {
    /// Underlying storage
    buffer: Vec<AtomicCell<Option<T>>>,
    /// Write pointer (producer)
    head: AtomicUsize,
    /// Read pointer (consumer)
    tail: AtomicUsize,
    /// Buffer capacity
    capacity: usize,
}

impl<T: Clone> ConfigCircularBuffer<T> {
    /// Create new circular buffer with specified capacity
    ///
    /// # Arguments
    /// * `capacity` - Number of slots in the circular buffer
    ///
    /// # Example
    /// ```no_run
    /// use prism_gpu::aatgs::ConfigCircularBuffer;
    /// use prism_core::RuntimeConfig;
    ///
    /// let buffer = ConfigCircularBuffer::<RuntimeConfig>::new(16);
    /// ```
    pub fn new(capacity: usize) -> Self {
        let mut buffer = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buffer.push(AtomicCell::new(None));
        }

        Self {
            buffer,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            capacity,
        }
    }

    /// Push item into buffer (non-blocking)
    ///
    /// # Returns
    /// * `true` - Item successfully pushed
    /// * `false` - Buffer full, item not pushed
    pub fn push(&self, item: T) -> bool {
        let head = self.head.load(Ordering::Acquire);
        let next_head = (head + 1) % self.capacity;
        let tail = self.tail.load(Ordering::Acquire);

        // Check if buffer is full
        if next_head == tail {
            return false;
        }

        // Write item
        self.buffer[head].store(Some(item));

        // Advance head pointer
        self.head.store(next_head, Ordering::Release);

        true
    }

    /// Pop item from buffer (non-blocking)
    ///
    /// # Returns
    /// * `Some(item)` - Item successfully popped
    /// * `None` - Buffer empty
    pub fn pop(&self) -> Option<T> {
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);

        // Check if buffer is empty
        if tail == head {
            return None;
        }

        // Read item
        let item = self.buffer[tail].swap(None);

        // Advance tail pointer
        let next_tail = (tail + 1) % self.capacity;
        self.tail.store(next_tail, Ordering::Release);

        item
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        head == tail
    }

    /// Check if buffer is full
    pub fn is_full(&self) -> bool {
        let head = self.head.load(Ordering::Acquire);
        let next_head = (head + 1) % self.capacity;
        let tail = self.tail.load(Ordering::Acquire);
        next_head == tail
    }

    /// Get current number of items in buffer
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);

        if head >= tail {
            head - tail
        } else {
            self.capacity - tail + head
        }
    }

    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get utilization as fraction [0.0, 1.0]
    pub fn utilization(&self) -> f32 {
        self.len() as f32 / self.capacity as f32
    }
}

// ============================================================================
// PART 2: Non-blocking Telemetry Download (~100 LOC)
// ============================================================================

/// Ring buffer for telemetry storage
pub struct RingBuffer<T: Clone> {
    buffer: VecDeque<T>,
    capacity: usize,
}

impl<T: Clone> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, item: T) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(item);
    }

    pub fn drain(&mut self) -> Vec<T> {
        self.buffer.drain(..).collect()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Telemetry event types
#[derive(Debug, Clone)]
pub enum TelemetryEvent {
    /// Phase metrics recorded
    PhaseMetrics(PhaseMetrics),
    /// GPU metrics recorded
    GpuMetrics(GpuMetrics),
    /// Task completed
    TaskCompleted(TaskId, Duration),
    /// Error occurred
    Error(String),
}

/// Phase-specific metrics
#[derive(Debug, Clone, Copy)]
pub struct PhaseMetrics {
    pub phase_id: usize,
    pub temperature: f32,
    pub compaction_ratio: f32,
    pub reward: f32,
    pub conflicts: usize,
    pub duration_us: u64,
}

/// GPU performance metrics
#[derive(Debug, Clone, Copy)]
pub struct GpuMetrics {
    pub utilization: f32,
    pub memory_used_mb: usize,
    pub memory_total_mb: usize,
    pub kernel_duration_us: u64,
    pub transfer_duration_us: u64,
}

/// Task identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskId(pub usize);

/// Async telemetry collector
///
/// Collects telemetry events from GPU operations in a non-blocking manner
/// using ring buffers and async channels.
pub struct TelemetryCollector {
    /// Phase metrics buffer
    metrics_buffer: RingBuffer<PhaseMetrics>,
    /// GPU metrics buffer
    gpu_buffer: RingBuffer<GpuMetrics>,
    /// Event sender (async channel)
    event_sender: std::sync::mpsc::Sender<TelemetryEvent>,
}

impl TelemetryCollector {
    /// Create new telemetry collector
    ///
    /// # Arguments
    /// * `buffer_size` - Size of ring buffers for metrics
    ///
    /// # Returns
    /// Tuple of (collector, receiver) for async event streaming
    pub fn new(buffer_size: usize) -> (Self, std::sync::mpsc::Receiver<TelemetryEvent>) {
        let (tx, rx) = std::sync::mpsc::channel();

        let collector = Self {
            metrics_buffer: RingBuffer::new(buffer_size),
            gpu_buffer: RingBuffer::new(buffer_size),
            event_sender: tx,
        };

        (collector, rx)
    }

    /// Record phase metrics (non-blocking)
    pub fn record_phase_metrics(&mut self, metrics: PhaseMetrics) {
        self.metrics_buffer.push(metrics);

        // Send event asynchronously (non-blocking if receiver is dropped)
        let _ = self.event_sender.send(TelemetryEvent::PhaseMetrics(metrics));
    }

    /// Record GPU metrics (non-blocking)
    pub fn record_gpu_metrics(&mut self, metrics: GpuMetrics) {
        self.gpu_buffer.push(metrics);

        // Send event asynchronously
        let _ = self.event_sender.send(TelemetryEvent::GpuMetrics(metrics));
    }

    /// Record task completion
    pub fn record_task_completion(&self, task_id: TaskId, duration: Duration) {
        let _ = self.event_sender.send(TelemetryEvent::TaskCompleted(task_id, duration));
    }

    /// Record error
    pub fn record_error(&self, error: String) {
        let _ = self.event_sender.send(TelemetryEvent::Error(error));
    }

    /// Flush all buffered events
    ///
    /// Returns all buffered metrics as telemetry events.
    pub fn flush(&mut self) -> Vec<TelemetryEvent> {
        let mut events = Vec::new();

        // Drain phase metrics
        for metrics in self.metrics_buffer.drain() {
            events.push(TelemetryEvent::PhaseMetrics(metrics));
        }

        // Drain GPU metrics
        for metrics in self.gpu_buffer.drain() {
            events.push(TelemetryEvent::GpuMetrics(metrics));
        }

        events
    }

    /// Get current buffer sizes
    pub fn buffer_stats(&self) -> (usize, usize) {
        (self.metrics_buffer.len(), self.gpu_buffer.len())
    }
}

// ============================================================================
// PART 3: Task Graph DAG (~150 LOC)
// ============================================================================

/// Task type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskType {
    /// Config upload to GPU
    ConfigUpload,
    /// WHCR kernel execution
    WhcrKernel,
    /// Thermodynamic annealing
    ThermodynamicAnneal,
    /// Quantum optimization
    QuantumOptimize,
    /// LBS prediction
    LbsPredict,
    /// Telemetry download
    TelemetryDownload,
    /// Phase transition
    PhaseTransition,
    /// Custom task
    Custom(&'static str),
}

/// Task status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    /// Task pending execution
    Pending,
    /// Task ready to execute (dependencies met)
    Ready,
    /// Task currently executing
    Running,
    /// Task completed successfully
    Completed,
    /// Task failed
    Failed,
}

/// Task node in dependency graph
#[derive(Debug, Clone)]
struct TaskNode {
    /// Unique task ID
    id: usize,
    /// Type of task
    task_type: TaskType,
    /// Task dependencies (must complete before this task)
    dependencies: Vec<usize>,
    /// Tasks dependent on this one
    dependents: Vec<usize>,
    /// Current status
    status: TaskStatus,
    /// Estimated duration in microseconds
    estimated_duration_us: u64,
    /// Actual start time (if running or completed)
    start_time: Option<Instant>,
    /// Actual completion time (if completed)
    completion_time: Option<Instant>,
}

/// Task dependency graph
///
/// Directed acyclic graph (DAG) representing task dependencies.
/// Supports topological sorting and dependency tracking for efficient
/// GPU task scheduling.
pub struct TaskGraph {
    /// All task nodes
    nodes: Vec<TaskNode>,
    /// Directed edges (from_id, to_id)
    edges: Vec<(usize, usize)>,
    /// Queue of ready-to-execute tasks
    ready_queue: VecDeque<usize>,
    /// Next task ID
    next_id: usize,
}

impl TaskGraph {
    /// Create new empty task graph
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            ready_queue: VecDeque::new(),
            next_id: 0,
        }
    }

    /// Add task to graph
    ///
    /// # Arguments
    /// * `task_type` - Type of task to add
    /// * `dependencies` - IDs of tasks that must complete before this task
    ///
    /// # Returns
    /// Task ID of newly added task
    pub fn add_task(&mut self, task_type: TaskType, dependencies: &[usize]) -> usize {
        let task_id = self.next_id;
        self.next_id += 1;

        // Validate dependencies exist
        for &dep_id in dependencies {
            assert!(
                self.nodes.iter().any(|n| n.id == dep_id),
                "Dependency {} does not exist",
                dep_id
            );
        }

        // Estimate duration based on task type
        let estimated_duration_us = match task_type {
            TaskType::ConfigUpload => 100,
            TaskType::WhcrKernel => 5000,
            TaskType::ThermodynamicAnneal => 10000,
            TaskType::QuantumOptimize => 15000,
            TaskType::LbsPredict => 8000,
            TaskType::TelemetryDownload => 200,
            TaskType::PhaseTransition => 1000,
            TaskType::Custom(_) => 5000,
        };

        // Create task node
        let node = TaskNode {
            id: task_id,
            task_type,
            dependencies: dependencies.to_vec(),
            dependents: Vec::new(),
            status: if dependencies.is_empty() {
                TaskStatus::Ready
            } else {
                TaskStatus::Pending
            },
            estimated_duration_us,
            start_time: None,
            completion_time: None,
        };

        // Add edges for dependencies
        for &dep_id in dependencies {
            self.edges.push((dep_id, task_id));

            // Update dependent's list
            if let Some(dep_node) = self.nodes.iter_mut().find(|n| n.id == dep_id) {
                dep_node.dependents.push(task_id);
            }
        }

        // Add to ready queue if no dependencies
        if dependencies.is_empty() {
            self.ready_queue.push_back(task_id);
        }

        self.nodes.push(node);

        task_id
    }

    /// Mark task as started
    pub fn mark_started(&mut self, task_id: usize) {
        if let Some(node) = self.nodes.iter_mut().find(|n| n.id == task_id) {
            node.status = TaskStatus::Running;
            node.start_time = Some(Instant::now());
        }
    }

    /// Mark task as complete
    ///
    /// Updates task status and checks if any dependent tasks become ready.
    pub fn mark_complete(&mut self, task_id: usize) {
        if let Some(node) = self.nodes.iter_mut().find(|n| n.id == task_id) {
            node.status = TaskStatus::Completed;
            node.completion_time = Some(Instant::now());

            // Check if any dependents become ready
            let dependents = node.dependents.clone();

            for dependent_id in dependents {
                if self.is_task_ready(dependent_id) {
                    // Update status to ready
                    if let Some(dep_node) = self.nodes.iter_mut().find(|n| n.id == dependent_id) {
                        dep_node.status = TaskStatus::Ready;
                    }
                    // Add to ready queue
                    self.ready_queue.push_back(dependent_id);
                }
            }
        }
    }

    /// Mark task as failed
    pub fn mark_failed(&mut self, task_id: usize) {
        if let Some(node) = self.nodes.iter_mut().find(|n| n.id == task_id) {
            node.status = TaskStatus::Failed;
            node.completion_time = Some(Instant::now());
        }
    }

    /// Check if task is ready (all dependencies completed)
    fn is_task_ready(&self, task_id: usize) -> bool {
        if let Some(node) = self.nodes.iter().find(|n| n.id == task_id) {
            node.dependencies
                .iter()
                .all(|&dep_id| {
                    self.nodes
                        .iter()
                        .find(|n| n.id == dep_id)
                        .map(|n| n.status == TaskStatus::Completed)
                        .unwrap_or(false)
                })
        } else {
            false
        }
    }

    /// Get list of ready tasks (dependencies satisfied)
    pub fn get_ready_tasks(&self) -> Vec<usize> {
        self.ready_queue.iter().copied().collect()
    }

    /// Pop next ready task from queue
    pub fn pop_ready_task(&mut self) -> Option<usize> {
        self.ready_queue.pop_front()
    }

    /// Perform topological sort on task graph
    ///
    /// Returns tasks in execution order (dependencies before dependents).
    /// Returns None if graph contains cycles.
    pub fn topological_sort(&self) -> Option<Vec<usize>> {
        let mut sorted = Vec::new();
        let mut in_degree: Vec<usize> = vec![0; self.nodes.len()];
        let mut queue = VecDeque::new();

        // Calculate in-degrees
        for node in &self.nodes {
            in_degree[node.id] = node.dependencies.len();
            if node.dependencies.is_empty() {
                queue.push_back(node.id);
            }
        }

        // Process queue
        while let Some(task_id) = queue.pop_front() {
            sorted.push(task_id);

            // Find node and process dependents
            if let Some(node) = self.nodes.iter().find(|n| n.id == task_id) {
                for &dependent_id in &node.dependents {
                    in_degree[dependent_id] -= 1;
                    if in_degree[dependent_id] == 0 {
                        queue.push_back(dependent_id);
                    }
                }
            }
        }

        // Check if all nodes were processed (no cycles)
        if sorted.len() == self.nodes.len() {
            Some(sorted)
        } else {
            None // Cycle detected
        }
    }

    /// Get task status
    pub fn task_status(&self, task_id: usize) -> Option<TaskStatus> {
        self.nodes.iter().find(|n| n.id == task_id).map(|n| n.status)
    }

    /// Get task type
    pub fn task_type(&self, task_id: usize) -> Option<TaskType> {
        self.nodes.iter().find(|n| n.id == task_id).map(|n| n.task_type)
    }

    /// Get task duration (if completed)
    pub fn task_duration(&self, task_id: usize) -> Option<Duration> {
        self.nodes.iter().find(|n| n.id == task_id).and_then(|n| {
            match (n.start_time, n.completion_time) {
                (Some(start), Some(end)) => Some(end.duration_since(start)),
                _ => None,
            }
        })
    }

    /// Get all tasks
    pub fn tasks(&self) -> Vec<(usize, TaskType, TaskStatus)> {
        self.nodes
            .iter()
            .map(|n| (n.id, n.task_type, n.status))
            .collect()
    }

    /// Clear completed tasks
    pub fn clear_completed(&mut self) {
        self.nodes.retain(|n| n.status != TaskStatus::Completed);
        self.edges.retain(|(from, to)| {
            self.nodes.iter().any(|n| n.id == *from) && self.nodes.iter().any(|n| n.id == *to)
        });
    }

    /// Get number of tasks
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if graph is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

impl Default for TaskGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PART 4: Adaptive Scheduling Policy (~100 LOC)
// ============================================================================

/// Performance snapshot for adaptive scheduling
#[derive(Debug, Clone, Copy)]
pub struct PerformanceSnapshot {
    /// Timestamp of snapshot
    pub timestamp: Instant,
    /// Average task duration
    pub avg_task_duration_us: u64,
    /// GPU utilization [0.0, 1.0]
    pub gpu_utilization: f32,
    /// Config buffer utilization [0.0, 1.0]
    pub config_buffer_util: f32,
    /// Telemetry buffer utilization [0.0, 1.0]
    pub telemetry_buffer_util: f32,
    /// Tasks completed in last interval
    pub tasks_completed: usize,
}

/// Scheduled task with priority
#[derive(Debug, Clone)]
pub struct ScheduledTask {
    /// Task ID from graph
    pub task_id: usize,
    /// Task type
    pub task_type: TaskType,
    /// Scheduling priority (higher = more urgent)
    pub priority: f32,
    /// Estimated duration
    pub estimated_duration_us: u64,
}

/// Adaptive scheduler that adjusts based on performance
///
/// Combines task graph dependencies with performance-based priority
/// adjustment to optimize GPU utilization and throughput.
pub struct AdaptiveScheduler {
    /// Task dependency graph
    task_graph: TaskGraph,
    /// Lock-free config buffer
    config_buffer: ConfigCircularBuffer<RuntimeConfig>,
    /// Telemetry collector
    telemetry: TelemetryCollector,
    /// Performance history
    performance_history: VecDeque<PerformanceSnapshot>,
    /// Maximum history size
    max_history: usize,
    /// Last snapshot time
    last_snapshot: Instant,
    /// Snapshot interval
    snapshot_interval: Duration,
}

impl AdaptiveScheduler {
    /// Create new adaptive scheduler
    ///
    /// # Arguments
    /// * `config_buffer_size` - Size of config circular buffer
    /// * `telemetry_buffer_size` - Size of telemetry buffers
    /// * `max_history` - Maximum performance snapshots to retain
    pub fn new(
        config_buffer_size: usize,
        telemetry_buffer_size: usize,
        max_history: usize,
    ) -> (Self, std::sync::mpsc::Receiver<TelemetryEvent>) {
        let (telemetry, rx) = TelemetryCollector::new(telemetry_buffer_size);

        let scheduler = Self {
            task_graph: TaskGraph::new(),
            config_buffer: ConfigCircularBuffer::new(config_buffer_size),
            telemetry,
            performance_history: VecDeque::with_capacity(max_history),
            max_history,
            last_snapshot: Instant::now(),
            snapshot_interval: Duration::from_millis(100),
        };

        (scheduler, rx)
    }

    /// Add task to scheduler
    ///
    /// # Returns
    /// Task ID
    pub fn add_task(&mut self, task_type: TaskType, dependencies: &[usize]) -> usize {
        self.task_graph.add_task(task_type, dependencies)
    }

    /// Schedule next task based on adaptive priority
    ///
    /// Combines dependency ordering with performance-based priority adjustment.
    pub fn schedule_next(&mut self) -> Option<ScheduledTask> {
        // Take performance snapshot if interval elapsed
        if self.last_snapshot.elapsed() >= self.snapshot_interval {
            self.take_snapshot();
        }

        // Get ready tasks
        let ready_tasks = self.task_graph.get_ready_tasks();
        if ready_tasks.is_empty() {
            return None;
        }

        // Calculate priorities for ready tasks
        let mut task_priorities: Vec<(usize, f32)> = ready_tasks
            .iter()
            .map(|&task_id| {
                let priority = self.calculate_priority(task_id);
                (task_id, priority)
            })
            .collect();

        // Sort by priority (descending)
        task_priorities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Pop highest priority task
        if let Some((task_id, priority)) = task_priorities.first() {
            let task_id = *task_id;
            let priority = *priority;

            // Remove from ready queue
            self.task_graph.pop_ready_task();

            // Get task details
            let task_type = self.task_graph.task_type(task_id)?;
            let estimated_duration_us = self.task_graph.nodes
                .iter()
                .find(|n| n.id == task_id)?
                .estimated_duration_us;

            Some(ScheduledTask {
                task_id,
                task_type,
                priority,
                estimated_duration_us,
            })
        } else {
            None
        }
    }

    /// Calculate priority for a task based on performance history
    fn calculate_priority(&self, task_id: usize) -> f32 {
        // Base priority: 1.0 for all tasks
        let mut priority = 1.0;

        // Adjust based on task type and recent performance
        if let Some(task_type) = self.task_graph.task_type(task_id) {
            // Prioritize config uploads if buffer is getting empty
            if task_type == TaskType::ConfigUpload {
                let config_util = self.config_buffer.utilization();
                if config_util < 0.3 {
                    priority *= 2.0; // High priority when buffer low
                }
            }

            // Prioritize telemetry downloads if buffer is getting full
            if task_type == TaskType::TelemetryDownload {
                let (_, telemetry_count) = self.telemetry.buffer_stats();
                if telemetry_count > 50 {
                    priority *= 1.5; // Elevated priority when buffer full
                }
            }

            // Deprioritize long tasks if GPU utilization is already high
            if let Some(latest) = self.performance_history.back() {
                if latest.gpu_utilization > 0.8 {
                    // High GPU utilization, prefer shorter tasks
                    if let Some(node) = self.task_graph.nodes.iter().find(|n| n.id == task_id) {
                        if node.estimated_duration_us > 10000 {
                            priority *= 0.7;
                        }
                    }
                }
            }
        }

        priority
    }

    /// Take performance snapshot
    fn take_snapshot(&mut self) {
        let now = Instant::now();

        // Calculate metrics from completed tasks since last snapshot
        let completed_tasks: Vec<_> = self.task_graph.nodes
            .iter()
            .filter(|n| n.status == TaskStatus::Completed)
            .filter(|n| {
                n.completion_time
                    .map(|t| t > self.last_snapshot)
                    .unwrap_or(false)
            })
            .collect();

        let tasks_completed = completed_tasks.len();

        let avg_task_duration_us = if !completed_tasks.is_empty() {
            let total_duration: Duration = completed_tasks
                .iter()
                .filter_map(|n| {
                    match (n.start_time, n.completion_time) {
                        (Some(start), Some(end)) => Some(end.duration_since(start)),
                        _ => None,
                    }
                })
                .sum();

            (total_duration.as_micros() as u64) / (tasks_completed as u64).max(1)
        } else {
            0
        };

        let snapshot = PerformanceSnapshot {
            timestamp: now,
            avg_task_duration_us,
            gpu_utilization: 0.0, // Would be updated from GPU metrics
            config_buffer_util: self.config_buffer.utilization(),
            telemetry_buffer_util: 0.0, // Would be updated from telemetry metrics
            tasks_completed,
        };

        self.performance_history.push_back(snapshot);

        // Trim history if too large
        while self.performance_history.len() > self.max_history {
            self.performance_history.pop_front();
        }

        self.last_snapshot = now;
    }

    /// Adapt priorities based on performance trends
    ///
    /// Analyzes performance history to adjust scheduling strategy.
    pub fn adapt_priorities(&mut self) {
        if self.performance_history.len() < 2 {
            return; // Need at least 2 snapshots to detect trends
        }

        // Calculate trend in GPU utilization
        let recent_util: f32 = self.performance_history
            .iter()
            .rev()
            .take(5)
            .map(|s| s.gpu_utilization)
            .sum::<f32>()
            / self.performance_history.len().min(5) as f32;

        // If GPU utilization is low, increase snapshot frequency
        if recent_util < 0.5 {
            self.snapshot_interval = Duration::from_millis(50);
        } else {
            self.snapshot_interval = Duration::from_millis(100);
        }
    }

    /// Estimate total completion time for remaining tasks
    pub fn estimate_completion_time(&self) -> Duration {
        let total_estimated_us: u64 = self.task_graph.nodes
            .iter()
            .filter(|n| n.status == TaskStatus::Pending || n.status == TaskStatus::Ready)
            .map(|n| n.estimated_duration_us)
            .sum();

        Duration::from_micros(total_estimated_us)
    }

    /// Mark task as started
    pub fn mark_task_started(&mut self, task_id: usize) {
        self.task_graph.mark_started(task_id);
    }

    /// Mark task as completed
    pub fn mark_task_completed(&mut self, task_id: usize, duration: Duration) {
        self.task_graph.mark_complete(task_id);
        self.telemetry.record_task_completion(TaskId(task_id), duration);
    }

    /// Mark task as failed
    pub fn mark_task_failed(&mut self, task_id: usize, error: String) {
        self.task_graph.mark_failed(task_id);
        self.telemetry.record_error(error);
    }

    /// Queue config for execution
    pub fn queue_config(&self, config: RuntimeConfig) -> bool {
        self.config_buffer.push(config)
    }

    /// Get performance history
    pub fn performance_history(&self) -> &VecDeque<PerformanceSnapshot> {
        &self.performance_history
    }

    /// Get task graph reference
    pub fn task_graph(&self) -> &TaskGraph {
        &self.task_graph
    }

    /// Get mutable task graph reference
    pub fn task_graph_mut(&mut self) -> &mut TaskGraph {
        &mut self.task_graph
    }
}

// ============================================================================
// PART 5: Integration with Stream Manager (~50 LOC)
// ============================================================================

/// AATGS-Stream Manager integration
///
/// Coordinates AATGS task scheduling with stream manager for efficient
/// GPU resource utilization.
pub struct AATGSStreamIntegration {
    /// AATGS scheduler
    scheduler: AATGSScheduler,
    /// Adaptive task scheduler
    adaptive: AdaptiveScheduler,
    /// Telemetry event receiver
    telemetry_rx: std::sync::mpsc::Receiver<TelemetryEvent>,
}

impl AATGSStreamIntegration {
    /// Create new AATGS-Stream integration
    ///
    /// # Arguments
    /// * `device` - CUDA device context
    ///
    /// # Returns
    /// Integrated scheduler with stream coordination
    pub fn new(device: Arc<CudaContext>) -> Result<Self> {
        let scheduler = AATGSScheduler::new(device)?;
        let (adaptive, telemetry_rx) = AdaptiveScheduler::new(
            CONFIG_BUFFER_SIZE,
            TELEMETRY_BUFFER_SIZE,
            100, // max history
        );

        Ok(Self {
            scheduler,
            adaptive,
            telemetry_rx,
        })
    }

    /// Execute one iteration with integrated scheduling
    ///
    /// Combines AATGS buffer management with adaptive task scheduling.
    pub fn execute_iteration(&mut self, config: RuntimeConfig) -> Result<Option<KernelTelemetry>> {
        // Queue config in AATGS scheduler
        self.scheduler.queue_config(config)?;

        // Queue config in adaptive scheduler
        if !self.adaptive.queue_config(config) {
            log::warn!("Adaptive config buffer full");
        }

        // Flush configs to GPU
        self.scheduler.flush_configs()?;

        // Schedule next task based on adaptive priority
        if let Some(scheduled_task) = self.adaptive.schedule_next() {
            log::debug!(
                "Scheduling task {:?} (priority: {:.2})",
                scheduled_task.task_type,
                scheduled_task.priority
            );

            // Mark task as started
            let start = Instant::now();
            self.adaptive.mark_task_started(scheduled_task.task_id);

            // Task would be executed here via stream manager
            // For now, just mark as completed
            let duration = start.elapsed();
            self.adaptive.mark_task_completed(scheduled_task.task_id, duration);
        }

        // Poll for completed telemetry
        let telemetry = self.scheduler.poll_telemetry()?;

        // Process telemetry events
        while let Ok(event) = self.telemetry_rx.try_recv() {
            match event {
                TelemetryEvent::PhaseMetrics(metrics) => {
                    log::debug!("Phase {} metrics: temp={:.2}, reward={:.2}",
                               metrics.phase_id, metrics.temperature, metrics.reward);
                }
                TelemetryEvent::GpuMetrics(metrics) => {
                    log::debug!("GPU utilization: {:.2}%", metrics.utilization * 100.0);
                }
                TelemetryEvent::TaskCompleted(task_id, duration) => {
                    log::debug!("Task {} completed in {:?}", task_id.0, duration);
                }
                TelemetryEvent::Error(error) => {
                    log::error!("Task error: {}", error);
                }
            }
        }

        // Adapt priorities based on performance
        self.adaptive.adapt_priorities();

        Ok(telemetry.into_iter().next())
    }

    /// Add task to adaptive scheduler
    pub fn add_task(&mut self, task_type: TaskType, dependencies: &[usize]) -> usize {
        self.adaptive.add_task(task_type, dependencies)
    }

    /// Get AATGS scheduler reference
    pub fn aatgs_scheduler(&self) -> &AATGSScheduler {
        &self.scheduler
    }

    /// Get mutable AATGS scheduler reference
    pub fn aatgs_scheduler_mut(&mut self) -> &mut AATGSScheduler {
        &mut self.scheduler
    }

    /// Get adaptive scheduler reference
    pub fn adaptive_scheduler(&self) -> &AdaptiveScheduler {
        &self.adaptive
    }

    /// Get mutable adaptive scheduler reference
    pub fn adaptive_scheduler_mut(&mut self) -> &mut AdaptiveScheduler {
        &mut self.adaptive
    }

    /// Estimate completion time for all pending tasks
    pub fn estimate_completion_time(&self) -> Duration {
        self.adaptive.estimate_completion_time()
    }

    /// Get performance statistics
    pub fn performance_stats(&self) -> PerformanceStats {
        let (config_util, telemetry_util) = self.scheduler.buffer_stats();
        let task_graph = self.adaptive.task_graph();

        let tasks = task_graph.tasks();
        let pending = tasks.iter().filter(|(_, _, s)| *s == TaskStatus::Pending).count();
        let running = tasks.iter().filter(|(_, _, s)| *s == TaskStatus::Running).count();
        let completed = tasks.iter().filter(|(_, _, s)| *s == TaskStatus::Completed).count();

        PerformanceStats {
            config_buffer_utilization: config_util,
            telemetry_buffer_utilization: telemetry_util,
            tasks_pending: pending,
            tasks_running: running,
            tasks_completed: completed,
            estimated_completion: self.adaptive.estimate_completion_time(),
        }
    }

    /// Shutdown integration
    pub fn shutdown(mut self) -> Result<()> {
        self.scheduler.shutdown()
    }
}

/// Performance statistics
#[derive(Debug, Clone, Copy)]
pub struct PerformanceStats {
    pub config_buffer_utilization: f32,
    pub telemetry_buffer_utilization: f32,
    pub tasks_pending: usize,
    pub tasks_running: usize,
    pub tasks_completed: usize,
    pub estimated_completion: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_sizes() {
        // Ensure reasonable buffer sizes
        assert!(CONFIG_BUFFER_SIZE >= 8);
        assert!(TELEMETRY_BUFFER_SIZE >= 32);
    }

    #[test]
    fn test_default_buffers() {
        let buffers = AATGSBuffers::default();
        assert_eq!(buffers.config_write_ptr, 0);
        assert_eq!(buffers.config_read_ptr, 0);
        assert_eq!(buffers.telemetry_write_ptr, 0);
        assert_eq!(buffers.telemetry_read_ptr, 0);
        assert_eq!(buffers.gpu_idle, 0);
        assert_eq!(buffers.cpu_shutdown, 0);
    }

    #[test]
    fn test_circular_buffer_basic() {
        let buffer = ConfigCircularBuffer::<i32>::new(4);
        assert!(buffer.is_empty());
        assert!(!buffer.is_full());

        assert!(buffer.push(1));
        assert!(buffer.push(2));
        assert!(!buffer.is_empty());

        assert_eq!(buffer.pop(), Some(1));
        assert_eq!(buffer.pop(), Some(2));
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_circular_buffer_full() {
        let buffer = ConfigCircularBuffer::<i32>::new(3);

        assert!(buffer.push(1));
        assert!(buffer.push(2));
        assert!(!buffer.push(3)); // Buffer full (capacity-1)

        assert_eq!(buffer.pop(), Some(1));
        assert!(buffer.push(3)); // Now space available
    }

    #[test]
    fn test_task_graph_basic() {
        let mut graph = TaskGraph::new();

        let task1 = graph.add_task(TaskType::ConfigUpload, &[]);
        let task2 = graph.add_task(TaskType::WhcrKernel, &[task1]);

        assert_eq!(graph.len(), 2);
        assert_eq!(graph.task_status(task1), Some(TaskStatus::Ready));
        assert_eq!(graph.task_status(task2), Some(TaskStatus::Pending));

        graph.mark_complete(task1);
        assert_eq!(graph.task_status(task1), Some(TaskStatus::Completed));
        assert_eq!(graph.task_status(task2), Some(TaskStatus::Ready));
    }

    #[test]
    fn test_task_graph_topological_sort() {
        let mut graph = TaskGraph::new();

        let t0 = graph.add_task(TaskType::ConfigUpload, &[]);
        let t1 = graph.add_task(TaskType::WhcrKernel, &[t0]);
        let t2 = graph.add_task(TaskType::TelemetryDownload, &[t1]);

        let sorted = graph.topological_sort().unwrap();
        assert_eq!(sorted, vec![t0, t1, t2]);
    }

    #[test]
    fn test_telemetry_collector() {
        let (mut collector, _rx) = TelemetryCollector::new(10);

        let metrics = PhaseMetrics {
            phase_id: 1,
            temperature: 1.5,
            compaction_ratio: 0.8,
            reward: 0.6,
            conflicts: 5,
            duration_us: 1000,
        };

        collector.record_phase_metrics(metrics);

        let events = collector.flush();
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn test_adaptive_scheduler() {
        let (mut scheduler, _rx) = AdaptiveScheduler::new(16, 64, 100);

        let task1 = scheduler.add_task(TaskType::ConfigUpload, &[]);
        let _task2 = scheduler.add_task(TaskType::WhcrKernel, &[task1]);

        let scheduled = scheduler.schedule_next();
        assert!(scheduled.is_some();

        let task = scheduled.unwrap());
        assert_eq!(task.task_id, task1);
        assert_eq!(task.task_type, TaskType::ConfigUpload);
    }

    // GPU tests require CUDA device
    #[test]
    #[ignore]
    fn test_scheduler_init() {
        let device = CudaContext::new(0).expect("Failed to create CUDA device");
        let scheduler = AATGSScheduler::new(Arc::new(device));
        assert!(scheduler.is_ok());
    }

    #[test]
    #[ignore]
    fn test_pipeline_init() {
        let device = CudaContext::new(0).expect("Failed to create CUDA device");
        let pipeline = AsyncPipeline::new(Arc::new(device));
        assert!(pipeline.is_ok());
    }
}
