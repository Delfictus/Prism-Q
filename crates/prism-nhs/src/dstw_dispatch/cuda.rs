//! GPU dispatch path for the DSTW variant projection.
//!
//! This module is the Rust FFI binding to the CUDA kernel at
//! `crates/prism-gpu/src/kernels/dstw_propagation.cu`. It provides:
//!
//!   * the shape contract (matrix layouts, channel count, output strides)
//!     so CPU and GPU paths share invariants;
//!   * a CPU-equivalent reference (`propagate_cpu_reference`) used by
//!     the matrix-shape unit tests.  Identical results to what the CUDA
//!     kernel would produce given the same inputs;
//!   * the production `CudaPropagationKernel` loader and dispatcher, which
//!     require a real PTX module, perform HtoD transfers, launch the kernel,
//!     perform DtoH readback, and reject non-finite output. There is no CPU
//!     fallback in this type.
//!
//! Layout invariants (matched by both CPU reference and CUDA kernel):
//!
//!   K           : 3 channels × n_residues × n_residues, row-major,
//!                 channel-major outermost.  Index:
//!                     K[(c * n_res + i) * n_res + j]
//!   residue_id  : n_variants i32, 0-indexed mutation site
//!   perturbation: n_variants f32, α_q·Δq + α_v·ΔV per variant
//!   delta_p     : 3 × n_variants, row-major, channel-major outermost.
//!                 Index: delta_p[c * n_variants + b]
//!
//! No CUDA dependency in this file (the actual cudarc surface lives in
//! crates/prism-gpu, gated behind the `gpu` feature).

#![allow(clippy::too_many_arguments)]

/// Number of thermodynamic channels emitted per variant.
pub const N_CHANNELS: usize = 3;

/// Channel indices in the K and delta_p arrays.
pub const CH_ACTIVE: usize = 0;
pub const CH_LOCK: usize = 1;
pub const CH_ENSEMBLE: usize = 2;

/// Shape descriptor passed to the kernel + CPU reference.
#[derive(Debug, Clone, Copy)]
pub struct PropagationShape {
    pub n_residues: usize,
    pub n_variants: usize,
}

impl PropagationShape {
    pub fn k_len(&self) -> usize {
        N_CHANNELS * self.n_residues * self.n_residues
    }
    pub fn delta_p_len(&self) -> usize {
        N_CHANNELS * self.n_variants
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.n_residues == 0 {
            return Err("n_residues must be positive".into());
        }
        if self.n_variants == 0 {
            return Err("n_variants must be positive".into());
        }
        if self.n_residues > 1024 {
            return Err(format!(
                "n_residues={} exceeds DSTW_MAX_RESIDUES_PER_BLOCK=1024; \
                 split the receptor or run the kernel iteratively",
                self.n_residues
            ));
        }
        Ok(())
    }
}

/// CPU reference for the propagation kernel.  Bit-equivalent to what
/// the CUDA kernel WOULD compute given the same f32 inputs (modulo
/// floating-point summation order, which differs between the warp-shuffle
/// reduction and a left-to-right scalar sum — tests assert
/// approximate equality with a per-channel tolerance, not exact bits).
///
/// Layout:
///   * K          slice of length 3 * n_residues * n_residues
///   * residue_id slice of length n_variants
///   * pert       slice of length n_variants
///   * delta_p    output slice of length 3 * n_variants
pub fn propagate_cpu_reference(
    shape: PropagationShape,
    k: &[f32],
    residue_id: &[i32],
    pert: &[f32],
    delta_p: &mut [f32],
) -> Result<(), String> {
    shape.validate()?;
    if k.len() != shape.k_len() {
        return Err(format!(
            "K length {} != expected {}",
            k.len(),
            shape.k_len()
        ));
    }
    if residue_id.len() != shape.n_variants {
        return Err(format!(
            "residue_id length {} != n_variants {}",
            residue_id.len(),
            shape.n_variants
        ));
    }
    if pert.len() != shape.n_variants {
        return Err(format!(
            "pert length {} != n_variants {}",
            pert.len(),
            shape.n_variants
        ));
    }
    if delta_p.len() != shape.delta_p_len() {
        return Err(format!(
            "delta_p length {} != expected {}",
            delta_p.len(),
            shape.delta_p_len()
        ));
    }
    let n = shape.n_residues;
    let nv = shape.n_variants;
    for c in 0..N_CHANNELS {
        let channel_offset = c * n * n;
        for b in 0..nv {
            let i = residue_id[b];
            if i < 0 || (i as usize) >= n {
                delta_p[c * nv + b] = 0.0;
                continue;
            }
            let row_offset = channel_offset + (i as usize) * n;
            let mut s: f32 = 0.0;
            for j in 0..n {
                s += k[row_offset + j];
            }
            delta_p[c * nv + b] = pert[b] * s;
        }
    }
    Ok(())
}

/// Row sums of K per channel for every variant's residue.
///
/// Used as an intermediate by the epistemic sigma kernel; exposed here
/// so the CPU path can reuse the same arithmetic and round-trip the
/// uncertainty budget without the GPU.
pub fn row_sums_cpu_reference(
    shape: PropagationShape,
    k: &[f32],
    residue_id: &[i32],
    row_sums: &mut [f32],
) -> Result<(), String> {
    shape.validate()?;
    if k.len() != shape.k_len() {
        return Err(format!(
            "K length mismatch: {} != {}",
            k.len(),
            shape.k_len()
        ));
    }
    if residue_id.len() != shape.n_variants {
        return Err(format!(
            "residue_id length mismatch: {} != {}",
            residue_id.len(),
            shape.n_variants
        ));
    }
    if row_sums.len() != N_CHANNELS * shape.n_variants {
        return Err(format!(
            "row_sums length {} != expected {}",
            row_sums.len(),
            N_CHANNELS * shape.n_variants
        ));
    }
    let n = shape.n_residues;
    let nv = shape.n_variants;
    for c in 0..N_CHANNELS {
        let channel_offset = c * n * n;
        for b in 0..nv {
            let i = residue_id[b];
            if i < 0 || (i as usize) >= n {
                row_sums[c * nv + b] = 0.0;
                continue;
            }
            let row_offset = channel_offset + (i as usize) * n;
            let mut s: f32 = 0.0;
            for j in 0..n {
                s += k[row_offset + j];
            }
            row_sums[c * nv + b] = s;
        }
    }
    Ok(())
}

/// Dispatch backend identifier.  Operator-authorised 2026-05-20 to
/// release the GPU gate; `CpuFallback` was removed in commit-FIRE.  Any
/// caller that lands on a host without a GPU (or builds without the
/// `gpu` feature) gets a hard error from `CudaPropagationKernel::new()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cuda,
}

// ---------------------------------------------------------------------------
// GPU dispatch path (cudarc 0.18.2, sm_120).
//
// Loads `target/ptx/dstw_propagation.ptx` (compiled by
// crates/prism-gpu/build.rs from src/kernels/dstw_propagation.cu),
// uploads K, residue_id, perturbation to the GPU, launches the
// `dstw_propagation_kernel`, copies the result back.
//
// Gated behind the prism-nhs `gpu` feature so the CPU-only test target
// (lib.rs without --features gpu) still builds.  The dispatcher binary
// declares `required-features = ["gpu"]` so production callers always
// have the GPU path.
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
pub use gpu_impl::CudaPropagationKernel;

#[cfg(not(feature = "gpu"))]
pub struct CudaPropagationKernel {
    _phantom: (),
}

#[cfg(not(feature = "gpu"))]
impl CudaPropagationKernel {
    pub fn new() -> Result<Self, String> {
        Err(
            "CudaPropagationKernel requires the prism-nhs `gpu` feature.  \
             Build with `--features gpu` (the default) or call \
             `propagate_cpu_reference` directly for CPU-only invocation."
                .to_string(),
        )
    }
}

#[cfg(feature = "gpu")]
mod gpu_impl {
    use super::{propagate_cpu_reference, Backend, PropagationShape, N_CHANNELS};
    use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::Ptx;
    use sha2::{Digest, Sha256};
    use std::{fs::File, io::Read, path::Path, sync::Arc};


    pub struct CudaPropagationKernel {
        stream: Arc<CudaStream>,
        propagation_fn: CudaFunction,
        block_dim_x: u32,
        ptx_path: String,
        ptx_sha256: String,
        ptx_bytes: u64,
    }

    impl std::fmt::Debug for CudaPropagationKernel {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("CudaPropagationKernel")
                .field("block_dim_x", &self.block_dim_x)
                .field("ptx_path", &self.ptx_path)
                .field("ptx_sha256", &self.ptx_sha256)
                .field("ptx_bytes", &self.ptx_bytes)
                .finish_non_exhaustive()
        }
    }

    /// `$DSTW_PROPAGATION_PTX` (explicit file pin; must exist) or
    /// dstw_propagation.ptx from the single PTX authority (P0.5). The former
    /// cwd-relative target/ptx candidates are gone.
    fn find_ptx_path() -> Result<String, String> {
        if let Ok(env_path) = std::env::var("DSTW_PROPAGATION_PTX") {
            return if std::path::Path::new(&env_path).is_file() {
                Ok(env_path)
            } else {
                Err(format!("DSTW_PROPAGATION_PTX={env_path} is not a file"))
            };
        }
        prism_gpu::ptx_authority::ptx_file("dstw_propagation.ptx")
            .map(|path| path.display().to_string())
            .map_err(|e| format!("dstw_propagation.ptx: {e}"))
    }

    impl CudaPropagationKernel {
        /// Open the CUDA context, load the PTX, resolve the kernel
        /// function.  Returns an informative error if anything fails;
        /// callers are expected to halt the dispatch on Err.
        pub fn new() -> Result<Self, String> {
            let ptx_path = find_ptx_path()?;
            Self::from_ptx(ptx_path)
        }

        /// Load an explicitly selected PTX module. Production callers use this
        /// constructor so the receipt binds the exact module bytes rather than
        /// relying on a current-working-directory search.
        pub fn from_ptx(path: impl AsRef<Path>) -> Result<Self, String> {
            let path = path.as_ref();
            let canonical_path = std::fs::canonicalize(path)
                .map_err(|e| format!("canonicalize {} failed: {e}", path.display()))?;
            let metadata = canonical_path
                .metadata()
                .map_err(|e| format!("stat {} failed: {e}", canonical_path.display()))?;
            if !metadata.is_file() || metadata.len() == 0 {
                return Err(format!(
                    "DSTW PTX is not a non-empty regular file: {}",
                    canonical_path.display()
                ));
            }
            let ptx_sha256 = sha256_file(&canonical_path)?;
            let ptx_path = canonical_path.display().to_string();
            let context = CudaContext::new(0).map_err(|e| {
                format!("CudaContext::new(0) failed: {e}.  Is a CUDA device available?")
            })?;
            let stream = context.default_stream();
            let ptx = Ptx::from_file(&ptx_path);
            let module = context
                .load_module(ptx)
                .map_err(|e| format!("load_module({ptx_path}) failed: {e}"))?;
            let propagation_fn = module
                .load_function("dstw_propagation_kernel")
                .map_err(|e| {
                    format!("module.load_function(dstw_propagation_kernel) failed: {e}")
                })?;
            Ok(Self {
                stream,
                propagation_fn,
                block_dim_x: 128,
                ptx_path,
                ptx_sha256,
                ptx_bytes: metadata.len(),
            })
        }

        pub fn ptx_path(&self) -> &str {
            &self.ptx_path
        }

        pub fn ptx_sha256(&self) -> &str {
            &self.ptx_sha256
        }

        pub fn ptx_bytes(&self) -> u64 {
            self.ptx_bytes
        }

        pub fn select_backend(&self) -> Backend {
            Backend::Cuda
        }

        /// Launch the propagation kernel on the GPU and copy the result
        /// back into `delta_p`.  Returns the dispatch backend used.
        pub fn dispatch(
            &self,
            shape: PropagationShape,
            k: &[f32],
            residue_id: &[i32],
            pert: &[f32],
            delta_p: &mut [f32],
        ) -> Result<Backend, String> {
            shape.validate()?;
            if k.len() != shape.k_len() {
                return Err(format!("K length {} != {}", k.len(), shape.k_len()));
            }
            if residue_id.len() != shape.n_variants {
                return Err(format!(
                    "residue_id length {} != n_variants {}",
                    residue_id.len(),
                    shape.n_variants
                ));
            }
            if pert.len() != shape.n_variants {
                return Err(format!(
                    "pert length {} != n_variants {}",
                    pert.len(),
                    shape.n_variants
                ));
            }
            if delta_p.len() != shape.delta_p_len() {
                return Err(format!(
                    "delta_p length {} != {}",
                    delta_p.len(),
                    shape.delta_p_len()
                ));
            }

            // Host → device uploads.
            let mut d_k = self
                .stream
                .alloc_zeros::<f32>(k.len())
                .map_err(|e| format!("alloc_zeros K failed: {e}"))?;
            self.stream
                .memcpy_htod(k, &mut d_k)
                .map_err(|e| format!("memcpy_htod K failed: {e}"))?;

            let mut d_residue_id = self
                .stream
                .alloc_zeros::<i32>(residue_id.len())
                .map_err(|e| format!("alloc_zeros residue_id failed: {e}"))?;
            self.stream
                .memcpy_htod(residue_id, &mut d_residue_id)
                .map_err(|e| format!("memcpy_htod residue_id failed: {e}"))?;

            let mut d_pert = self
                .stream
                .alloc_zeros::<f32>(pert.len())
                .map_err(|e| format!("alloc_zeros pert failed: {e}"))?;
            self.stream
                .memcpy_htod(pert, &mut d_pert)
                .map_err(|e| format!("memcpy_htod pert failed: {e}"))?;

            let mut d_delta_p = self
                .stream
                .alloc_zeros::<f32>(delta_p.len())
                .map_err(|e| format!("alloc_zeros delta_p failed: {e}"))?;

            let n_variants_i32 = shape.n_variants as i32;
            let n_residues_i32 = shape.n_residues as i32;
            let n_channels_u32 = N_CHANNELS as u32;
            let n_variants_u32 = shape.n_variants as u32;

            let cfg = LaunchConfig {
                grid_dim: (n_variants_u32, n_channels_u32, 1),
                block_dim: (self.block_dim_x, 1, 1),
                shared_mem_bytes: 0,
            };

            // SAFETY: launch_builder requires unsafe because it dispatches
            // to a raw GPU kernel; the surrounding allocations have been
            // dimensioned against `shape.validate()` and the kernel itself
            // bounds-checks `residue_id[b]` before any K-row read.
            unsafe {
                self.stream
                    .launch_builder(&self.propagation_fn)
                    .arg(&d_k)
                    .arg(&d_residue_id)
                    .arg(&d_pert)
                    .arg(&n_variants_i32)
                    .arg(&n_residues_i32)
                    .arg(&mut d_delta_p)
                    .launch(cfg)
                    .map_err(|e| format!("launch dstw_propagation_kernel failed: {e}"))?;
            }
            self.stream
                .memcpy_dtoh(&d_delta_p, delta_p)
                .map_err(|e| format!("memcpy_dtoh delta_p failed: {e}"))?;
            self.stream
                .synchronize()
                .map_err(|e| format!("stream.synchronize failed: {e}"))?;

            // Sanity check: every emitted value finite.  The kernel
            // promises no NaN escape, so any non-finite output here is
            // an integrity failure of the upstream K / pert.  We surface
            // it loud rather than let DSTW catch it later.
            for (idx, v) in delta_p.iter().enumerate() {
                if !v.is_finite() {
                    return Err(format!(
                        "non-finite delta_p[{}]={}; integrity failure in WT tensors",
                        idx, v
                    ));
                }
            }

            Ok(Backend::Cuda)
        }

        /// Reference-equivalent CPU dispatch, used by tests that want
        /// to compare GPU vs CPU outputs at fixed inputs.
        pub fn dispatch_cpu_reference(
            &self,
            shape: PropagationShape,
            k: &[f32],
            residue_id: &[i32],
            pert: &[f32],
            delta_p: &mut [f32],
        ) -> Result<(), String> {
            propagate_cpu_reference(shape, k, residue_id, pert, delta_p)
        }
    }

    fn sha256_file(path: &Path) -> Result<String, String> {
        let mut file = File::open(path)
            .map_err(|e| format!("open {} for SHA-256 failed: {e}", path.display()))?;
        let mut hasher = Sha256::new();
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let read = file
                .read(&mut buffer)
                .map_err(|e| format!("read {} for SHA-256 failed: {e}", path.display()))?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        Ok(format!("{:x}", hasher.finalize()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthesise a K with a known row sum per channel.  Channel c, row
    /// i has every entry equal to `1.0 + 0.01 * (c * n + i)`, so
    /// row_sum(c, i) = n * (1.0 + 0.01 * (c * n + i)).  Lets us assert
    /// against an analytic expectation.
    fn synth_k(n: usize) -> Vec<f32> {
        let mut k = vec![0.0f32; N_CHANNELS * n * n];
        for c in 0..N_CHANNELS {
            for i in 0..n {
                let entry = 1.0 + 0.01 * ((c * n + i) as f32);
                for j in 0..n {
                    k[(c * n + i) * n + j] = entry;
                }
            }
        }
        k
    }

    fn expected_row_sum(n: usize, channel: usize, i: usize) -> f32 {
        let entry = 1.0 + 0.01 * ((channel * n + i) as f32);
        n as f32 * entry
    }

    #[test]
    fn shape_sweep_residue_counts() {
        for &n in &[32usize, 64, 128, 256, 352, 512] {
            let shape = PropagationShape {
                n_residues: n,
                n_variants: 4,
            };
            shape.validate().expect("valid shape");
            let k = synth_k(n);
            let residue_id = vec![0, 1, (n / 2) as i32, (n - 1) as i32];
            let pert = vec![1.0, -1.0, 2.0, 0.5];
            let mut delta_p = vec![0.0f32; shape.delta_p_len()];
            propagate_cpu_reference(shape, &k, &residue_id, &pert, &mut delta_p).unwrap();
            for c in 0..N_CHANNELS {
                for (b, &i) in residue_id.iter().enumerate() {
                    let expected = pert[b] * expected_row_sum(n, c, i as usize);
                    let got = delta_p[c * shape.n_variants + b];
                    assert!(
                        (got - expected).abs() / expected.abs().max(1.0) < 1e-4,
                        "n_residues={n} c={c} b={b} expected={expected} got={got}"
                    );
                }
            }
        }
    }

    #[test]
    fn shape_sweep_variant_counts() {
        for &nv in &[1usize, 5, 25, 100] {
            let n = 128usize;
            let shape = PropagationShape {
                n_residues: n,
                n_variants: nv,
            };
            shape.validate().expect("valid shape");
            let k = synth_k(n);
            let residue_id: Vec<i32> = (0..nv).map(|b| (b % n) as i32).collect();
            let pert: Vec<f32> = (0..nv).map(|b| (b as f32) * 0.01).collect();
            let mut delta_p = vec![0.0f32; shape.delta_p_len()];
            propagate_cpu_reference(shape, &k, &residue_id, &pert, &mut delta_p).unwrap();
            for c in 0..N_CHANNELS {
                for b in 0..nv {
                    let i = residue_id[b] as usize;
                    let expected = pert[b] * expected_row_sum(n, c, i);
                    let got = delta_p[c * nv + b];
                    assert!(
                        (got - expected).abs() / expected.abs().max(1.0) < 1e-4,
                        "nv={nv} c={c} b={b}"
                    );
                }
            }
        }
    }

    #[test]
    fn out_of_range_residue_yields_zero() {
        let n = 64usize;
        let shape = PropagationShape {
            n_residues: n,
            n_variants: 3,
        };
        let k = synth_k(n);
        let residue_id = vec![0, -1, 999];
        let pert = vec![1.0, 1.0, 1.0];
        let mut delta_p = vec![0.0f32; shape.delta_p_len()];
        propagate_cpu_reference(shape, &k, &residue_id, &pert, &mut delta_p).unwrap();
        for c in 0..N_CHANNELS {
            // variant 0 (valid) -> non-zero
            assert!(delta_p[c * 3 + 0].abs() > 0.0);
            // variant 1 (-1) -> zero
            assert_eq!(delta_p[c * 3 + 1], 0.0);
            // variant 2 (999) -> zero
            assert_eq!(delta_p[c * 3 + 2], 0.0);
        }
    }

    #[test]
    fn row_sums_reference_matches_propagation() {
        let n = 96usize;
        let nv = 7usize;
        let shape = PropagationShape {
            n_residues: n,
            n_variants: nv,
        };
        let k = synth_k(n);
        let residue_id: Vec<i32> = (0..nv).map(|b| ((b * 13) % n) as i32).collect();
        let pert = vec![1.0; nv];
        let mut deltas = vec![0.0f32; shape.delta_p_len()];
        let mut rows = vec![0.0f32; N_CHANNELS * nv];
        propagate_cpu_reference(shape, &k, &residue_id, &pert, &mut deltas).unwrap();
        row_sums_cpu_reference(shape, &k, &residue_id, &mut rows).unwrap();
        // delta_p = pert * row_sum, with pert=1 they must be equal.
        for c in 0..N_CHANNELS {
            for b in 0..nv {
                assert!(
                    (deltas[c * nv + b] - rows[c * nv + b]).abs() < 1e-4,
                    "delta_p != row_sum at (c={c}, b={b})"
                );
            }
        }
    }

    #[test]
    fn shape_validation_rejects_invalid_inputs() {
        let shape = PropagationShape {
            n_residues: 0,
            n_variants: 10,
        };
        assert!(shape.validate().is_err());
        let shape = PropagationShape {
            n_residues: 10,
            n_variants: 0,
        };
        assert!(shape.validate().is_err());
        let shape = PropagationShape {
            n_residues: 2048,
            n_variants: 10,
        };
        assert!(shape
            .validate()
            .unwrap_err()
            .contains("MAX_RESIDUES_PER_BLOCK"));
    }

    /// GPU dispatch test — actually launches the kernel on the host's
    /// CUDA device when the `gpu` feature is enabled.  Compared against
    /// the analytic row-sum expectation.  Skipped automatically when
    /// the GPU is unavailable (CudaPropagationKernel::new returns Err).
    #[cfg(feature = "gpu")]
    #[ignore = "GPU_RUNTIME_REQUIRED: needs real GPU + built PTX; run on the RTX workstation via `cargo test -- --ignored`"]
    #[test]
    fn cuda_dispatch_matches_analytic_row_sum() {
        let kernel = match CudaPropagationKernel::new() {
            Ok(k) => k,
            Err(e) => {
                eprintln!("[skip] CUDA unavailable in test environment: {e}");
                return;
            }
        };
        let n = 64;
        let nv = 4;
        let shape = PropagationShape {
            n_residues: n,
            n_variants: nv,
        };
        let k = synth_k(n);
        let residue_id = vec![0, 7, (n / 2) as i32, (n - 1) as i32];
        let pert = vec![1.0, -1.0, 2.0, 0.5];
        let mut delta_p = vec![0.0f32; shape.delta_p_len()];
        let backend = kernel
            .dispatch(shape, &k, &residue_id, &pert, &mut delta_p)
            .unwrap();
        assert_eq!(backend, Backend::Cuda);
        for c in 0..N_CHANNELS {
            for (b, &i) in residue_id.iter().enumerate() {
                let expected = pert[b] * expected_row_sum(n, c, i as usize);
                let got = delta_p[c * nv + b];
                // Warp-shuffle reduction reorders sums vs left-to-right;
                // tolerate a relative drift of 1e-4.
                let rel = (got - expected).abs() / expected.abs().max(1.0);
                assert!(rel < 1e-4, "c={c} b={b} expected={expected} got={got}");
            }
        }
    }

    #[cfg(all(not(feature = "gpu"), not(feature = "cpu-analysis")))]
    #[test]
    fn cuda_kernel_requires_gpu_feature_when_disabled() {
        let err = CudaPropagationKernel::new().unwrap_err();
        assert!(err.contains("gpu"));
    }
}
