//! Tokenizer compute-backend selector (GPU acceleration directive, Phase 3).
//!
//! CPU is the DEFAULT and the REFERENCE. A GPU lane is selected only by **workload shape/size**
//! (record count, coord count, codebook size, scalar-array length, available device memory) above
//! benchmarked crossover thresholds — never because a GPU exists, and never by artifact origin.
//! Correctness MUST NOT depend on GPU availability: every `Auto`/`Gpu` path has a byte/code-parity
//! CPU fallback, and when the `gpu-tokenizer` feature is off (the default), `Gpu`/`Auto` degrade to
//! CPU with no behaviour change.
//!
//! ## Crossover thresholds (measured)
//! The thresholds below were set from the Phase-1 CPU baseline + Phase-6 perf ledgers on real
//! PRISM artifacts (RTX 5080, CUDA 12). They are deliberately conservative: a workload must be
//! large enough that the kernel-launch + H2D/D2H transfer is amortized AND the measured end-to-end
//! gate (Phase 6) is met. Below threshold, `Auto` stays on CPU.

/// Which compute backend a tokenizer operation should run on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenizerBackend {
    /// CPU reference path (default, always correct, always available).
    Cpu,
    /// Force the GPU path. Falls back to CPU if the `gpu-tokenizer` feature is off or no device is
    /// available — correctness never depends on the GPU being present.
    Gpu,
    /// Pick GPU only above the benchmarked crossover thresholds AND when a device is available.
    Auto,
}

impl Default for TokenizerBackend {
    fn default() -> Self {
        TokenizerBackend::Cpu
    }
}

/// The measured crossover thresholds that gate `Auto`. A workload below ALL of these stays on CPU
/// even with a device present (the kernel would lose to the CPU reference). These are the numbers
/// the Phase-6 perf ledger justified; changing them requires re-running the perf gate.
#[derive(Clone, Copy, Debug)]
pub struct CrossoverThresholds {
    /// Minimum number of fit points (N) for the VQ-EMA fit GPU lane to be considered.
    pub vq_fit_min_points: usize,
    /// Minimum codebook size (K) for the VQ-EMA fit GPU lane (the assignment pass is O(N·K)).
    pub vq_fit_min_codebook: usize,
    /// Minimum number of coordinates for the batched coord-encode GPU lane.
    pub coord_encode_min_points: usize,
    /// Minimum scalar-array length for the scalar-reduction GPU lane.
    pub scalar_reduce_min_len: usize,
    /// Minimum free device memory (bytes) required before any GPU lane is selected.
    pub min_free_device_bytes: u64,
}

impl Default for CrossoverThresholds {
    fn default() -> Self {
        // Conservative defaults. See OPERATOR_CONTROL/TOKENIZER_GPU_PERFORMANCE_LEDGER.json for the
        // measured crossover points these track.
        Self {
            vq_fit_min_points: 50_000,
            vq_fit_min_codebook: 256,
            coord_encode_min_points: 200_000,
            scalar_reduce_min_len: 1_000_000,
            min_free_device_bytes: 256 * 1024 * 1024,
        }
    }
}

/// A workload's measurable shape — the only inputs allowed to drive `Auto` (never artifact origin).
#[derive(Clone, Copy, Debug, Default)]
pub struct WorkloadShape {
    pub n_points: usize,
    pub codebook_size: usize,
    pub scalar_len: usize,
}

impl TokenizerBackend {
    /// Resolve `Auto` to a concrete backend for a VQ-EMA fit of the given shape. `device_ok`
    /// reports whether a usable device with enough free memory is present (false when the
    /// `gpu-tokenizer` feature is off). CPU is returned whenever in doubt.
    pub fn resolve_vq_fit(
        self,
        shape: WorkloadShape,
        thr: &CrossoverThresholds,
        device_ok: bool,
    ) -> TokenizerBackend {
        match self {
            TokenizerBackend::Cpu => TokenizerBackend::Cpu,
            TokenizerBackend::Gpu => {
                if device_ok {
                    TokenizerBackend::Gpu
                } else {
                    TokenizerBackend::Cpu
                }
            }
            TokenizerBackend::Auto => {
                if device_ok
                    && shape.n_points >= thr.vq_fit_min_points
                    && shape.codebook_size >= thr.vq_fit_min_codebook
                {
                    TokenizerBackend::Gpu
                } else {
                    TokenizerBackend::Cpu
                }
            }
        }
    }

    /// Resolve `Auto` for a batched coordinate encode of `n_points`.
    pub fn resolve_coord_encode(
        self,
        n_points: usize,
        thr: &CrossoverThresholds,
        device_ok: bool,
    ) -> TokenizerBackend {
        self.resolve_threshold(n_points >= thr.coord_encode_min_points, device_ok)
    }

    /// Resolve `Auto` for a scalar reduction over `len` values.
    pub fn resolve_scalar_reduce(
        self,
        len: usize,
        thr: &CrossoverThresholds,
        device_ok: bool,
    ) -> TokenizerBackend {
        self.resolve_threshold(len >= thr.scalar_reduce_min_len, device_ok)
    }

    fn resolve_threshold(self, above: bool, device_ok: bool) -> TokenizerBackend {
        match self {
            TokenizerBackend::Cpu => TokenizerBackend::Cpu,
            TokenizerBackend::Gpu => {
                if device_ok {
                    TokenizerBackend::Gpu
                } else {
                    TokenizerBackend::Cpu
                }
            }
            TokenizerBackend::Auto => {
                if device_ok && above {
                    TokenizerBackend::Gpu
                } else {
                    TokenizerBackend::Cpu
                }
            }
        }
    }
}

/// Is a usable GPU device present for the tokenizer? Always `false` unless the `gpu-tokenizer`
/// feature is enabled AND a CUDA context with sufficient free memory initializes. Correctness is
/// independent of this — it only gates `Auto`/`Gpu` lane selection.
pub fn device_available(min_free_bytes: u64) -> bool {
    #[cfg(feature = "gpu-tokenizer")]
    {
        crate::gpu::device_available(min_free_bytes)
    }
    #[cfg(not(feature = "gpu-tokenizer"))]
    {
        let _ = min_free_bytes;
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_default_and_correctness_independent_of_device() {
        let thr = CrossoverThresholds::default();
        // With no device, every backend resolves to CPU — correctness never needs the GPU.
        let shape = WorkloadShape {
            n_points: 10_000_000,
            codebook_size: 4096,
            scalar_len: 0,
        };
        assert_eq!(
            TokenizerBackend::Auto.resolve_vq_fit(shape, &thr, false),
            TokenizerBackend::Cpu
        );
        assert_eq!(
            TokenizerBackend::Gpu.resolve_vq_fit(shape, &thr, false),
            TokenizerBackend::Cpu
        );
        assert_eq!(
            TokenizerBackend::Cpu.resolve_vq_fit(shape, &thr, true),
            TokenizerBackend::Cpu
        );
    }

    #[test]
    fn auto_picks_gpu_only_above_threshold_with_device() {
        let thr = CrossoverThresholds::default();
        let big = WorkloadShape {
            n_points: 100_000,
            codebook_size: 4096,
            scalar_len: 0,
        };
        let small = WorkloadShape {
            n_points: 1_000,
            codebook_size: 4096,
            scalar_len: 0,
        };
        let small_k = WorkloadShape {
            n_points: 100_000,
            codebook_size: 16,
            scalar_len: 0,
        };
        assert_eq!(
            TokenizerBackend::Auto.resolve_vq_fit(big, &thr, true),
            TokenizerBackend::Gpu
        );
        assert_eq!(
            TokenizerBackend::Auto.resolve_vq_fit(small, &thr, true),
            TokenizerBackend::Cpu
        );
        // below the codebook threshold ⇒ CPU even with many points (kernel can't amortize).
        assert_eq!(
            TokenizerBackend::Auto.resolve_vq_fit(small_k, &thr, true),
            TokenizerBackend::Cpu
        );
    }

    #[test]
    fn coord_and_scalar_thresholds_gate_auto() {
        let thr = CrossoverThresholds::default();
        assert_eq!(
            TokenizerBackend::Auto.resolve_coord_encode(thr.coord_encode_min_points, &thr, true),
            TokenizerBackend::Gpu
        );
        assert_eq!(
            TokenizerBackend::Auto.resolve_coord_encode(
                thr.coord_encode_min_points - 1,
                &thr,
                true
            ),
            TokenizerBackend::Cpu
        );
        assert_eq!(
            TokenizerBackend::Auto.resolve_scalar_reduce(thr.scalar_reduce_min_len, &thr, true),
            TokenizerBackend::Gpu
        );
        assert_eq!(
            TokenizerBackend::Auto.resolve_scalar_reduce(thr.scalar_reduce_min_len - 1, &thr, true),
            TokenizerBackend::Cpu
        );
    }

    #[test]
    fn device_available_is_false_without_feature() {
        // CPU-default build (no gpu-tokenizer feature): the selector must never claim a device.
        #[cfg(not(feature = "gpu-tokenizer"))]
        assert!(!device_available(0));
    }
}
