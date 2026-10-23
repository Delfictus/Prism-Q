//! GPU acceleration lanes for the tokenizer (feature `gpu-tokenizer`, default OFF).
//!
//! These are OPTIONAL accelerators for the workloads Phase 1 proved hot — they NEVER change a
//! result. Every kernel here has an exact CPU parity contract documented at its definition and a
//! CPU reference that the parity bin (`tok_gpu_parity`) checks bit/code-for-bit. When the device
//! or feature is absent, the public API still works (it routes back to the CPU reference), so
//! correctness is independent of GPU availability.
//!
//! Boundary (EXECUTION POLICY): kernels live in `.cu` source compiled at runtime via cudarc NVRTC
//! (same pattern as `prism-gpu`); Rust owns orchestration / buffers / error propagation. No graph
//! capture, no engine execution, no live-engine dependency.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaModule, LaunchConfig, PushKernelArg};

/// The VQ assignment kernel: for each 3-D point, find the index of the nearest of `k` codebook
/// entries by squared Euclidean distance, ties broken by LOWEST index (exact CPU parity). Also
/// emits the squared distance to that nearest code, so the host can do the EMA accumulation and
/// the worst-point reinit deterministically.
const VQ_ASSIGN_KERNEL: &str = r#"
extern "C" __global__ void vq_assign(
    const double* __restrict__ pts,   // n * 3
    const double* __restrict__ cb,    // k * 3
    int n, int k,
    int* __restrict__ out_idx,        // n
    double* __restrict__ out_d2)      // n
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double px = pts[3*i + 0];
    double py = pts[3*i + 1];
    double pz = pts[3*i + 2];
    int best = 0;
    double bd = 1.0e300;
    for (int j = 0; j < k; ++j) {
        double dx = px - cb[3*j + 0];
        double dy = py - cb[3*j + 1];
        double dz = pz - cb[3*j + 2];
        // Match the CPU's `(0..3).map(|i| diff[i].powi(2)).sum()` exactly: a left fold from 0.0,
        // separate squares, no FMA (see fmad=false at compile). `powi(2)` == `x*x`.
        double d = 0.0;
        d = d + dx*dx;
        d = d + dy*dy;
        d = d + dz*dz;
        // strict '<' so the FIRST (lowest-index) minimum wins on ties — matches the CPU loop.
        if (d < bd) { bd = d; best = j; }
    }
    out_idx[i] = best;
    out_d2[i]  = bd;
}
"#;

/// A live GPU context + the compiled VQ-assign kernel. Created lazily; if anything fails the
/// caller falls back to CPU (this struct is only built when a device is confirmed usable).
pub struct GpuVqContext {
    ctx: Arc<CudaContext>,
    stream: Arc<cudarc::driver::CudaStream>,
    _module: Arc<CudaModule>,
    assign: cudarc::driver::CudaFunction,
}

impl GpuVqContext {
    /// Initialize device 0 and compile the assignment kernel. None on any failure (no panic).
    pub fn try_new() -> Option<Self> {
        let ctx = CudaContext::new(0).ok()?;
        let stream = ctx.default_stream();
        // Compile for sm_90 (forward-JITs onto sm_120 Blackwell at load); fall back to a generic
        // virtual arch if the toolkit rejects it.
        // PARITY-CRITICAL: disable FMA contraction (`fmad=false`) so `dx*dx+dy*dy+dz*dz` is the
        // SAME IEEE-754 op sequence as the CPU reference (separate multiplies, then adds). With FMA
        // on, the squared distance differs by 1 ULP and the bit-for-bit d2 parity gate fails (the
        // argmin is unaffected, but the directive's exact-distortion gate demands bitwise equality).
        let opts = |arch: &'static str| cudarc::nvrtc::CompileOptions {
            arch: Some(arch),
            fmad: Some(false),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(VQ_ASSIGN_KERNEL, opts("sm_90"))
            .or_else(|_| cudarc::nvrtc::compile_ptx_with_opts(VQ_ASSIGN_KERNEL, opts("compute_70")))
            .ok()?;
        let module = ctx.load_module(ptx).ok()?;
        let assign = module.load_function("vq_assign").ok()?;
        Some(Self {
            ctx,
            stream,
            _module: module,
            assign,
        })
    }

    /// Free device memory in bytes (current free / total), or None if the query fails.
    pub fn free_mem(&self) -> Option<(u64, u64)> {
        // cudarc exposes memory info on the context's primary device.
        self.ctx.bind_to_thread().ok()?;
        let (free, total) = cudarc::driver::result::mem_get_info().ok()?;
        Some((free as u64, total as u64))
    }

    /// GPU assignment pass: returns `(nearest_index, squared_distance)` per point, EXACTLY matching
    /// [`vq_assign_cpu`] (same argmin, same lowest-index tie-break, same f64 arithmetic order). On
    /// any device error, returns None so the caller uses the CPU reference.
    pub fn vq_assign(
        &self,
        pts: &[[f64; 3]],
        codebook: &[[f64; 3]],
    ) -> Option<(Vec<i32>, Vec<f64>)> {
        let n = pts.len();
        let k = codebook.len();
        if n == 0 || k == 0 {
            return Some((vec![0; n], vec![0.0; n]));
        }
        let mut flat_pts = Vec::with_capacity(n * 3);
        for p in pts {
            flat_pts.extend_from_slice(p);
        }
        let mut flat_cb = Vec::with_capacity(k * 3);
        for c in codebook {
            flat_cb.extend_from_slice(c);
        }
        let d_pts = self.stream.clone_htod(&flat_pts).ok()?;
        let d_cb = self.stream.clone_htod(&flat_cb).ok()?;
        let mut d_idx = self.stream.alloc_zeros::<i32>(n).ok()?;
        let mut d_d2 = self.stream.alloc_zeros::<f64>(n).ok()?;

        let threads = 256u32;
        let blocks = ((n as u32) + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i = n as i32;
        let k_i = k as i32;
        let mut builder = self.stream.launch_builder(&self.assign);
        builder.arg(&d_pts);
        builder.arg(&d_cb);
        builder.arg(&n_i);
        builder.arg(&k_i);
        builder.arg(&mut d_idx);
        builder.arg(&mut d_d2);
        unsafe { builder.launch(cfg).ok()? };

        let mut idx = vec![0i32; n];
        let mut d2 = vec![0.0f64; n];
        self.stream.memcpy_dtoh(&d_idx, &mut idx).ok()?;
        self.stream.memcpy_dtoh(&d_d2, &mut d2).ok()?;
        self.stream.synchronize().ok()?;
        Some((idx, d2))
    }
}

/// CPU reference for the VQ assignment pass. This is the byte-for-byte truth the GPU kernel must
/// match: for each point, nearest codebook index by squared distance, lowest index on ties.
pub fn vq_assign_cpu(pts: &[[f64; 3]], codebook: &[[f64; 3]]) -> (Vec<i32>, Vec<f64>) {
    let mut idx = Vec::with_capacity(pts.len());
    let mut d2 = Vec::with_capacity(pts.len());
    for &p in pts {
        let mut best = 0usize;
        let mut bd = f64::INFINITY;
        for (j, &c) in codebook.iter().enumerate() {
            // identical op sequence to the kernel and to `Vq3dEma::d2`: left fold from 0.0 of the
            // three separate squares (`.powi(2)` == `x*x`), so all three paths agree bit-for-bit.
            let d = (0..3).map(|i| (p[i] - c[i]).powi(2)).sum::<f64>();
            if d < bd {
                bd = d;
                best = j;
            }
        }
        idx.push(best as i32);
        d2.push(bd);
    }
    (idx, d2)
}

/// Is a usable device present with at least `min_free_bytes` free? Builds (and discards) a probe
/// context. False on any failure — correctness never depends on this.
pub fn device_available(min_free_bytes: u64) -> bool {
    match GpuVqContext::try_new() {
        Some(g) => match g.free_mem() {
            Some((free, _total)) => free >= min_free_bytes,
            None => false,
        },
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::synth_coords;

    #[test]
    fn cpu_assign_lowest_index_tie_break() {
        // two identical codes: the lowest index must win on the tie.
        let cb = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]];
        let pts = [[0.0, 0.0, 0.0]];
        let (idx, d2) = vq_assign_cpu(&pts, &cb);
        assert_eq!(idx[0], 0, "lowest index wins on exact tie");
        assert_eq!(d2[0], 0.0);
    }

    #[test]
    fn gpu_assign_matches_cpu_exactly_when_available() {
        let Some(g) = GpuVqContext::try_new() else {
            eprintln!("skip: no GPU device for vq_assign parity");
            return;
        };
        let pts = synth_coords(5000, 4, 11);
        // a deterministic codebook (strided subsample, like the real fit init).
        let k = 256usize;
        let cb: Vec<[f64; 3]> = (0..k).map(|j| pts[(j * pts.len()) / k]).collect();
        let (ci, cd) = vq_assign_cpu(&pts, &cb);
        let (gi, gd) = g.vq_assign(&pts, &cb).expect("gpu assign");
        assert_eq!(ci, gi, "GPU nearest-index must equal CPU exactly");
        // squared distances must match bit-for-bit (same op order, same f64).
        for (a, b) in cd.iter().zip(gd.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "GPU d2 must equal CPU d2 bit-for-bit"
            );
        }
    }
}
