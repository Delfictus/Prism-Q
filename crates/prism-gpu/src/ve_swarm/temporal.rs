//! Temporal Convolution Stack
//!
//! Multi-scale 1D convolutions over variant frequency time series.

use anyhow::{Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceSlice, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// Temporal embedding dimension
pub const TEMPORAL_DIM: usize = 64;

/// Maximum weeks in time series
pub const MAX_WEEKS: usize = 52;

/// Configuration for temporal convolution
#[derive(Clone, Debug)]
pub struct TemporalConfig {
    /// Dilation rates for multi-scale convolution
    pub dilation_rates: Vec<usize>,
    /// Hidden dimension for conv layers
    pub hidden_dim: usize,
    /// Output embedding dimension
    pub output_dim: usize,
    /// Whether to apply velocity inversion correction
    pub velocity_correction: bool,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            dilation_rates: vec![1, 2, 4, 8],
            hidden_dim: 64,
            output_dim: TEMPORAL_DIM,
            velocity_correction: true,
        }
    }
}

/// Temporal trajectory features (computed on CPU for analysis)
#[derive(Clone, Debug, Default)]
pub struct TrajectoryFeatures {
    /// Total frequency change over period
    pub total_change: f32,
    /// Mean velocity
    pub mean_velocity: f32,
    /// Velocity trend (accelerating vs decelerating)
    pub velocity_trend: f32,
    /// Peak frequency achieved
    pub peak_frequency: f32,
    /// Time to peak (normalized 0-1)
    pub time_to_peak: f32,
    /// Post-peak decline
    pub post_peak_decline: f32,
    /// Velocity variance (stability)
    pub velocity_variance: f32,
    /// Sign changes in velocity (oscillation)
    pub oscillation: f32,
    /// Recent velocity (last 4 weeks)
    pub recent_velocity: f32,
    /// Current frequency
    pub current_frequency: f32,
    /// Growth rate (log scale)
    pub growth_rate: f32,
    /// S-curve detection (max curvature)
    pub s_curve_score: f32,
    /// Corrected momentum (velocity with inversion fix)
    pub corrected_momentum: f32,
}

/// Temporal convolution processor
pub struct TemporalConv {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    config: TemporalConfig,

    // Kernel functions
    fn_preprocess: CudaFunction,
    fn_conv: CudaFunction,
    fn_velocity_correction: CudaFunction,
    fn_batch: CudaFunction,
}

impl TemporalConv {
    /// Create new temporal convolution processor
    pub fn new(ctx: Arc<CudaContext>, ptx_path: &str, config: TemporalConfig) -> Result<Self> {
        let stream = ctx.default_stream();
        let module = ctx
            .load_module(Ptx::from_file(ptx_path))
            .context("Failed to load temporal conv PTX")?;

        let fn_preprocess = module.load_function("ve_swarm_preprocess_temporal")?;
        let fn_conv = module.load_function("ve_swarm_temporal_conv")?;
        let fn_velocity_correction = module.load_function("ve_swarm_velocity_correction")?;
        let fn_batch = module.load_function("ve_swarm_batch_temporal")?;

        Ok(Self {
            ctx,
            stream,
            config,
            fn_preprocess,
            fn_conv,
            fn_velocity_correction,
            fn_batch,
        })
    }

    /// Compute temporal embedding from frequency time series
    pub fn compute(
        &self,
        freq_series: &[f32],
        feature_series: Option<&[f32]>, // [N_weeks x 136]
    ) -> Result<(CudaSlice<f32>, f32)> {
        let n_weeks = freq_series.len();
        anyhow::ensure!(
            n_weeks <= MAX_WEEKS,
            "Time series too long: {} > {}",
            n_weeks,
            MAX_WEEKS
        );

        // Upload frequency series
        let d_freq: CudaSlice<f32> = self.stream.clone_htod(freq_series)?;

        // Feature series (use zeros if not provided)
        let features = feature_series
            .map(|f| f.to_vec())
            .unwrap_or_else(|| vec![0.0f32; n_weeks * 136]);
        let d_features: CudaSlice<f32> = self.stream.clone_htod(&features[..])?;

        // Allocate outputs
        let mut d_embedding: CudaSlice<f32> = self.stream.alloc_zeros(self.config.output_dim)?;
        let mut d_corrected: CudaSlice<f32> = self.stream.alloc_zeros(1)?;

        // Run batch kernel (simplified)
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 8192,
        };

        unsafe {
            self.stream
                .launch_builder(&self.fn_batch)
                .arg(&d_freq)
                .arg(&d_features)
                .arg(&mut d_embedding)
                .arg(&mut d_corrected)
                .arg(&1i32)
                .arg(&(n_weeks as i32))
                .launch(cfg)
        }?;

        self.ctx.synchronize()?;

        let corrected = self.stream.clone_dtoh(&d_corrected)?;

        Ok((d_embedding, corrected[0]))
    }

    /// Compute trajectory features on CPU (for analysis)
    pub fn compute_trajectory_features(&self, freq_series: &[f32]) -> TrajectoryFeatures {
        let n = freq_series.len();
        if n == 0 {
            return TrajectoryFeatures::default();
        }

        // Compute velocity
        let velocity: Vec<f32> = (1..n)
            .map(|i| freq_series[i] - freq_series[i - 1])
            .collect();

        // Total change
        let total_change = freq_series[n - 1] - freq_series[0];

        // Mean velocity
        let mean_velocity = velocity.iter().sum::<f32>() / velocity.len().max(1) as f32;

        // Velocity trend
        let half = velocity.len() / 2;
        let early_vel: f32 = velocity[..half.max(1)].iter().sum();
        let late_vel: f32 = velocity[half..].iter().sum();
        let velocity_trend = late_vel - early_vel;

        // Peak frequency and time to peak
        let (peak_week, peak_frequency) = freq_series
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap_or((0, &0.0));
        let time_to_peak = peak_week as f32 / n.max(1) as f32;

        // Post-peak decline
        let post_peak_decline = if peak_week < n - 1 {
            *peak_frequency - freq_series[n - 1]
        } else {
            0.0
        };

        // Velocity variance
        let vel_var: f32 = velocity
            .iter()
            .map(|v| (v - mean_velocity).powi(2))
            .sum::<f32>()
            / velocity.len().max(1) as f32;
        let velocity_variance = vel_var.sqrt();

        // Oscillation (sign changes)
        let sign_changes = velocity
            .windows(2)
            .filter(|w| (w[0] > 0.0) != (w[1] > 0.0))
            .count();
        let oscillation = sign_changes as f32 / velocity.len().max(1) as f32;

        // Recent velocity
        let recent_vel: f32 = velocity.iter().rev().take(4).sum::<f32>() / 4.0;

        // Current frequency
        let current_frequency = freq_series[n - 1];

        // Growth rate
        let start = freq_series[0].max(1e-6);
        let end = freq_series[n - 1].max(1e-6);
        let growth_rate = (end / start).ln();

        // S-curve detection (max curvature)
        let s_curve_score = if velocity.len() >= 3 {
            velocity
                .windows(3)
                .map(|w| {
                    let acc = (w[2] - w[0]) / 2.0;
                    let curv = acc.abs() / (1.0 + w[1].powi(2)).powf(1.5);
                    curv
                })
                .fold(0.0f32, |a, b| a.max(b))
        } else {
            0.0
        };

        // Corrected momentum
        let corrected_momentum = if self.config.velocity_correction {
            self.correct_velocity(recent_vel, current_frequency)
        } else {
            recent_vel
        };

        TrajectoryFeatures {
            total_change,
            mean_velocity,
            velocity_trend,
            peak_frequency: *peak_frequency,
            time_to_peak,
            post_peak_decline,
            velocity_variance,
            oscillation,
            recent_velocity: recent_vel,
            current_frequency,
            growth_rate,
            s_curve_score,
            corrected_momentum,
        }
    }

    /// Apply velocity inversion correction
    fn correct_velocity(&self, velocity: f32, frequency: f32) -> f32 {
        if frequency > 0.5 {
            // High frequency: variant is saturating, invert velocity
            -velocity * 2.0
        } else if frequency > 0.2 && velocity > 0.05 {
            // Near peak: dampen positive velocity
            velocity * 0.3
        } else if frequency < 0.1 && velocity > 0.0 {
            // True growth phase: amplify
            velocity * 1.5
        } else {
            velocity
        }
    }
}

/// Detect if trajectory shows RISE pattern
pub fn detect_rise_pattern(features: &TrajectoryFeatures) -> (bool, f32) {
    let mut score = 0.0f32;
    let mut signals = 0;

    // Corrected momentum positive (most important)
    if features.corrected_momentum > 0.01 {
        score += 0.3;
        signals += 1;
    }

    // Low current frequency (room to grow)
    if features.current_frequency < 0.3 {
        score += 0.2;
        signals += 1;
    }

    // Positive velocity trend (accelerating)
    if features.velocity_trend > 0.0 {
        score += 0.15;
        signals += 1;
    }

    // S-curve signature (inflection point)
    if features.s_curve_score > 0.1 && features.time_to_peak < 0.5 {
        score += 0.15;
        signals += 1;
    }

    // Early growth phase
    if features.time_to_peak > 0.6 {
        score += 0.1;
        signals += 1;
    }

    // Positive overall growth
    if features.total_change > 0.05 {
        score += 0.1;
        signals += 1;
    }

    let is_rise = score > 0.4 || (signals >= 3 && score > 0.3);
    let confidence = (signals as f32 / 6.0) * (score / 1.0);

    (is_rise, confidence)
}

/// Detect if trajectory shows FALL pattern
pub fn detect_fall_pattern(features: &TrajectoryFeatures) -> (bool, f32) {
    let mut score = 0.0f32;
    let mut signals = 0;

    // Corrected momentum negative
    if features.corrected_momentum < -0.01 {
        score += 0.3;
        signals += 1;
    }

    // High current frequency (saturated)
    if features.current_frequency > 0.5 {
        score += 0.2;
        signals += 1;
    }

    // Already past peak
    if features.time_to_peak < 0.3 {
        score += 0.15;
        signals += 1;
    }

    // Post-peak decline ongoing
    if features.post_peak_decline > 0.05 {
        score += 0.15;
        signals += 1;
    }

    // Negative velocity trend (decelerating)
    if features.velocity_trend < 0.0 {
        score += 0.1;
        signals += 1;
    }

    // Negative recent velocity
    if features.recent_velocity < 0.0 {
        score += 0.1;
        signals += 1;
    }

    let is_fall = score > 0.4 || (signals >= 3 && score > 0.3);
    let confidence = (signals as f32 / 6.0) * (score / 1.0);

    (is_fall, confidence)
}
