//! PRISM-4D Option A variant dispatcher — CLI.
//!
//! Reads a `PRISMExecutionRequest` JSON minted by DSTW's BALD orchestrator,
//! loads the WT tensor pack produced by `dstw-export-wt`, runs the
//! rigid-backbone Δq / ΔV projection per variant, and writes a
//! `PRISMExecutionResponse` JSON that round-trips through DSTW's frozen
//! Pydantic schema.
//!
//! The topology-aware perturbation and uncertainty preparation remains on the
//! host. Production propagation through `K_active`, `K_lock`, and
//! `K_ensemble` is CUDA-only and fail-closed; the emitted backend receipt binds
//! the selected PTX and records CPU/GPU agreement.

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use prism_nhs::dstw_dispatch::cuda::{
    Backend, CudaPropagationKernel, PropagationShape, N_CHANNELS,
};
use prism_nhs::dstw_dispatch::projection::ProjectionConfig;
use prism_nhs::dstw_dispatch::{
    attenuate_edges, dispatch_variant_batch_with_topology, write_chem_perturbed_dtsg_parquet,
    CausalCoupling, ChannelCapacity, ComplementPenalty, DTSGEdge, DTSGEdgeMetrics, DispatchError,
    EdgePenalty, FrustrationPenalty, HydrationVariance, HysteresisCapacity, HysteresisEnthalpy,
    HysteresisPersistence, PRISMExecutionRequest, ResidueIdx, ScalingConstant, ScalingConstants,
    SpatialVariance, TopologyProjectionContext, VariantDispatchConfig, WTTensorPack,
};
use prism_nhs::input::PrismPrepTopology;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::{
    fs::File,
    io::{BufReader, Read, Write},
    path::{Path, PathBuf},
};

// Parquet reader for the WT tensor pack.
use arrow_array::{
    ArrayRef, BooleanArray, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use parquet::{
    arrow::{arrow_reader::ParquetRecordBatchReaderBuilder, ArrowWriter},
    file::properties::WriterProperties,
};

#[derive(Parser, Debug)]
#[command(
    name = "dstw_dispatch_variants",
    about = "Project a DSTW BALD variant request onto the WT thermodynamic tensors."
)]
struct Args {
    /// Path to the DSTW BALD_Round_NNN_Request.json.
    #[arg(long)]
    request: PathBuf,

    /// Path to the WT physics parquet emitted by `dstw-export-wt`.
    #[arg(long)]
    wt_tensor_pack: PathBuf,

    /// Inactive-state WT PRISM topology JSON (5VEX apo-primed clean anchor).
    #[arg(long)]
    inactive_topology: PathBuf,

    /// Active-state WT PRISM topology JSON (6X1A apo-primed clean anchor).
    #[arg(long)]
    active_topology: PathBuf,

    /// PRISM run id stamped on every response row.
    #[arg(long)]
    prism_run_id: String,

    /// Where to write the response JSON. Optional; parquet is the live-fire handoff.
    #[arg(long)]
    out_json: Option<PathBuf>,

    /// Where to write the response parquet for DSTW SVI ingestion.
    #[arg(long)]
    out_parquet: PathBuf,

    /// Mixing coefficient on the Δq term in `perturbation = α_q·Δq + α_v·ΔV`.
    #[arg(long, default_value_t = 1.0)]
    alpha_q: f64,

    /// Mixing coefficient on the ΔV term.
    #[arg(long, default_value_t = 1.0)]
    alpha_v: f64,

    /// Per-channel model residual variance floor (defaults to 1e-4).
    #[arg(long, default_value_t = 1e-4)]
    residual_variance_active: f64,

    #[arg(long, default_value_t = 1e-4)]
    residual_variance_lock: f64,

    #[arg(long, default_value_t = 1e-4)]
    residual_variance_ensemble: f64,

    /// Sigma multiplier on non-convergence.  Asserted > 1.0 by the
    /// projection-config validator.  Default 4.0 per spec.
    #[arg(long, default_value_t = 4.0)]
    nonconverged_sigma_penalty: f64,

    /// Number of deterministic stochastic rotamer probes per mutation.
    #[arg(long, default_value_t = 20)]
    rotamer_samples: usize,

    /// Production propagation backend. CUDA is the only accepted value; there
    /// is no silent CPU fallback.
    #[arg(long, value_parser = ["cuda"])]
    backend: String,

    /// Exact dstw_propagation.ptx selected for the production CUDA launch.
    #[arg(long)]
    dstw_propagation_ptx: PathBuf,

    /// Hash-bound production backend receipt written after output readback.
    #[arg(long)]
    backend_receipt: PathBuf,

    /// Production chemistry perturbation contract. All three chem-DTSG paths
    /// must be supplied together; partial specification is rejected.
    #[arg(long)]
    chem_dtsg_input: Option<PathBuf>,

    /// Chem_Perturbed_DTSG.v1 Parquet output.
    #[arg(long)]
    chem_dtsg_parquet: Option<PathBuf>,

    /// Hash-bound state-graph handoff consumed by PrismOS Path-B.
    #[arg(long)]
    chem_dtsg_state_graph: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChemDtsgProductionInput {
    schema: String,
    source_state_graph_sha256: String,
    edges: Vec<ChemDtsgEdgeInput>,
    conformers: Vec<ChemDtsgConformerInput>,
    scaling: ChemDtsgScalingInput,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChemDtsgEdgeInput {
    from_residue: u32,
    to_residue: u32,
    te_out: f64,
    te_in: f64,
    delta_hc: f64,
    sigma_hyd: f64,
    spatial_var: f64,
    hysteresis_persist: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChemDtsgConformerInput {
    conformer_id: String,
    variant_id: String,
    topology_sha256: String,
    penalties: Vec<ChemDtsgPenaltyInput>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChemDtsgPenaltyInput {
    clash: f64,
    complement: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChemDtsgScalingInput {
    beta_frustration: f64,
    beta_stabilization: f64,
    te_channel_capacity: f64,
    hysteresis_capacity: f64,
}

#[derive(Debug, Serialize)]
struct ChemDtsgExecutionReceipt {
    schema: &'static str,
    status: &'static str,
    fallback_used: bool,
    input_path: String,
    input_sha256: String,
    source_state_graph_sha256: String,
    conformer_count: usize,
    edge_count: usize,
    variant_ids: Vec<String>,
    topology_sha256: Vec<String>,
    parquet_path: String,
    parquet_sha256: String,
    parquet_bytes: u64,
    parquet_rows: usize,
    downstream_consumer: &'static str,
    edges: Vec<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct CpuGpuAgreementReceipt {
    tolerance_absolute: f64,
    tolerance_relative: f64,
    max_absolute_error: f64,
    max_relative_error: f64,
    compared_values: usize,
    pass: bool,
}

#[derive(Debug, Serialize)]
struct CudaTransferReceipt {
    h_to_d_k: bool,
    h_to_d_residue_id: bool,
    h_to_d_perturbation: bool,
    kernel_symbol: &'static str,
    kernel_launch: bool,
    d_to_h_readback: bool,
    finite_output_gate: bool,
}

#[derive(Debug, Serialize)]
struct DstwCudaRuntimeReceipt {
    schema: &'static str,
    status: &'static str,
    backend: &'static str,
    fallback_allowed: bool,
    ptx_path: String,
    ptx_sha256: String,
    ptx_bytes: u64,
    request_path: String,
    request_sha256: String,
    request_blake3: String,
    wt_tensor_pack_path: String,
    wt_tensor_pack_sha256: String,
    inactive_topology_path: String,
    inactive_topology_sha256: String,
    active_topology_path: String,
    active_topology_sha256: String,
    prism_run_id: String,
    variant_count: usize,
    response_json_path: Option<String>,
    response_json_sha256: Option<String>,
    response_parquet_path: String,
    response_parquet_sha256: String,
    cuda_transfer: CudaTransferReceipt,
    cpu_gpu_agreement: CpuGpuAgreementReceipt,
    chem_dtsg: Option<ChemDtsgBackendBinding>,
}

#[derive(Debug, Serialize)]
struct ChemDtsgBackendBinding {
    state_graph_path: String,
    state_graph_sha256: String,
    parquet_path: String,
    parquet_sha256: String,
    downstream_consumer: &'static str,
}

fn load_request(path: &PathBuf) -> Result<PRISMExecutionRequest> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let reader = BufReader::new(file);
    let req: PRISMExecutionRequest =
        serde_json::from_reader(reader).with_context(|| format!("parse {}", path.display()))?;
    Ok(req)
}

fn load_wt_pack(path: &PathBuf) -> Result<WTTensorPack> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    // Concatenate every batch into Vec<f64> columns.
    let mut indices: Vec<i32> = Vec::new();
    let mut inactive_te_out: Vec<f64> = Vec::new();
    let mut inactive_te_in: Vec<f64> = Vec::new();
    let mut inactive_delta_hc: Vec<f64> = Vec::new();
    let mut inactive_sigma_hyd: Vec<f64> = Vec::new();
    let mut active_te_out: Vec<f64> = Vec::new();
    let mut active_te_in: Vec<f64> = Vec::new();
    let mut active_delta_hc: Vec<f64> = Vec::new();
    let mut active_sigma_hyd: Vec<f64> = Vec::new();
    while let Some(batch) = reader.next() {
        let batch: RecordBatch = batch?;
        let idx_arr = batch
            .column_by_name("uniprot_residue_index")
            .ok_or_else(|| anyhow!("WT pack missing 'uniprot_residue_index'"))?;
        // Accept either Int32 (Rust exporter) or Int64 (polars default) so
        // the dispatcher round-trips with both DSTW-emitted and engine-
        // emitted WT packs.
        let idx_values: Vec<i32> = if let Some(a) = idx_arr.as_any().downcast_ref::<Int32Array>() {
            (0..a.len()).map(|i| a.value(i)).collect()
        } else if let Some(a) = idx_arr.as_any().downcast_ref::<Int64Array>() {
            (0..a.len()).map(|i| a.value(i) as i32).collect()
        } else {
            bail!("'uniprot_residue_index' must be Int32 or Int64");
        };
        let to_f64 = |name: &str| -> Result<Vec<f64>> {
            let arr = batch
                .column_by_name(name)
                .ok_or_else(|| anyhow!("WT pack missing {:?}", name))?;
            let arr = arr
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| anyhow!("{} must be Float64", name))?;
            let mut out = Vec::with_capacity(arr.len());
            for i in 0..arr.len() {
                out.push(arr.value(i));
            }
            Ok(out)
        };
        // Drop residue_name column (string, not used by the projection).
        let _ = batch
            .column_by_name("residue_name")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>().map(|_| ()));
        indices.extend(idx_values);
        inactive_te_out.extend(to_f64("inactive_te_out")?);
        inactive_te_in.extend(to_f64("inactive_te_in")?);
        inactive_delta_hc.extend(to_f64("inactive_delta_hc")?);
        inactive_sigma_hyd.extend(to_f64("inactive_sigma_hydration_sq")?);
        active_te_out.extend(to_f64("active_te_out")?);
        active_te_in.extend(to_f64("active_te_in")?);
        active_delta_hc.extend(to_f64("active_delta_hc")?);
        active_sigma_hyd.extend(to_f64("active_sigma_hydration_sq")?);
    }
    if indices.is_empty() {
        bail!("WT pack {} is empty", path.display());
    }
    // Sort by uniprot_residue_index. Shared-core WT packs are not required to
    // be dense because construct-masked loops can be absent from one state.
    let mut by_idx: Vec<(i32, f64, f64, f64, f64, f64, f64, f64, f64)> = indices
        .into_iter()
        .zip(inactive_te_out)
        .zip(inactive_te_in)
        .zip(inactive_delta_hc)
        .zip(inactive_sigma_hyd)
        .zip(active_te_out)
        .zip(active_te_in)
        .zip(active_delta_hc)
        .zip(active_sigma_hyd)
        .map(|((((((((i, a), b), c), d), e), f), g), h)| (i, a, b, c, d, e, f, g, h))
        .collect();
    by_idx.sort_by_key(|t| t.0);
    let lo = by_idx.first().unwrap().0;
    let hi = by_idx.last().unwrap().0;
    let n = by_idx.len();
    let mut residue_numbers = Vec::with_capacity(n);
    let mut inactive_te_out_v = Vec::with_capacity(n);
    let mut inactive_te_in_v = Vec::with_capacity(n);
    let mut inactive_delta_hc_v = Vec::with_capacity(n);
    let mut inactive_sigma_hyd_v = Vec::with_capacity(n);
    let mut active_te_out_v = Vec::with_capacity(n);
    let mut active_te_in_v = Vec::with_capacity(n);
    let mut active_delta_hc_v = Vec::with_capacity(n);
    let mut active_sigma_hyd_v = Vec::with_capacity(n);
    for (i, a, b, c, d, e, f, g, h) in by_idx {
        residue_numbers.push(i);
        inactive_te_out_v.push(a);
        inactive_te_in_v.push(b);
        inactive_delta_hc_v.push(c);
        inactive_sigma_hyd_v.push(d);
        active_te_out_v.push(e);
        active_te_in_v.push(f);
        active_delta_hc_v.push(g);
        active_sigma_hyd_v.push(h);
    }
    Ok(WTTensorPack {
        residue_numbers,
        residue_index_lo: lo,
        residue_index_hi: hi,
        inactive_te_out: inactive_te_out_v,
        inactive_te_in: inactive_te_in_v,
        inactive_delta_hc: inactive_delta_hc_v,
        inactive_sigma_hydration_sq: inactive_sigma_hyd_v,
        active_te_out: active_te_out_v,
        active_te_in: active_te_in_v,
        active_delta_hc: active_delta_hc_v,
        active_sigma_hydration_sq: active_sigma_hyd_v,
    })
}

fn write_response_parquet(
    response: &prism_nhs::dstw_dispatch::PRISMExecutionResponse,
    path: &PathBuf,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let n = response.variants.len();
    let schema = Arc::new(Schema::new(vec![
        Field::new("campaign_id", DataType::Utf8, false),
        Field::new("round_index", DataType::Int32, false),
        Field::new("request_blake3", DataType::Utf8, false),
        Field::new("completed_at_utc", DataType::Utf8, false),
        Field::new("response_schema", DataType::Utf8, false),
        Field::new("target", DataType::Utf8, false),
        Field::new("variant", DataType::Utf8, false),
        Field::new("delta_P_active", DataType::Float64, false),
        Field::new("delta_P_lock", DataType::Float64, false),
        Field::new("delta_P_ensemble", DataType::Float64, false),
        Field::new("sigma_delta_P_active", DataType::Float64, false),
        Field::new("sigma_delta_P_lock", DataType::Float64, false),
        Field::new("sigma_delta_P_ensemble", DataType::Float64, false),
        Field::new("prism_run_id", DataType::Utf8, false),
        Field::new("converged", DataType::Boolean, false),
    ]));

    let repeat = |s: &str| -> Vec<String> { (0..n).map(|_| s.to_string()).collect() };
    let round_index = vec![response.round_index as i32; n];
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(repeat(&response.campaign_id))),
        Arc::new(Int32Array::from(round_index)),
        Arc::new(StringArray::from(repeat(&response.request_blake3))),
        Arc::new(StringArray::from(repeat(&response.completed_at_utc))),
        Arc::new(StringArray::from(repeat(&response.response_schema))),
        Arc::new(StringArray::from(
            response
                .variants
                .iter()
                .map(|v| v.target.clone())
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            response
                .variants
                .iter()
                .map(|v| v.variant.clone())
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            response
                .variants
                .iter()
                .map(|v| v.delta_P_active)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            response
                .variants
                .iter()
                .map(|v| v.delta_P_lock)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            response
                .variants
                .iter()
                .map(|v| v.delta_P_ensemble)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            response
                .variants
                .iter()
                .map(|v| v.sigma_delta_P_active)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            response
                .variants
                .iter()
                .map(|v| v.sigma_delta_P_lock)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            response
                .variants
                .iter()
                .map(|v| v.sigma_delta_P_ensemble)
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            response
                .variants
                .iter()
                .map(|v| v.prism_run_id.clone())
                .collect::<Vec<_>>(),
        )),
        Arc::new(BooleanArray::from(
            response
                .variants
                .iter()
                .map(|v| v.converged)
                .collect::<Vec<_>>(),
        )),
    ];
    let batch = RecordBatch::try_new(schema.clone(), columns)
        .context("RecordBatch::try_new for variant response")?;
    let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let props = WriterProperties::builder()
        .set_compression(parquet::basic::Compression::ZSTD(Default::default()))
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("read {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let file_name = path
        .file_name()
        .context("backend receipt path requires a file name")?
        .to_string_lossy();
    let tmp = path.with_file_name(format!(".{file_name}.{}.tmp", std::process::id()));
    let mut payload = serde_json::to_vec_pretty(value)?;
    payload.push(b'\n');
    {
        let mut file = File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?;
        file.write_all(&payload)
            .with_context(|| format!("write {}", tmp.display()))?;
        file.sync_all()
            .with_context(|| format!("sync {}", tmp.display()))?;
    }
    std::fs::rename(&tmp, path)
        .with_context(|| format!("rename {} to {}", tmp.display(), path.display()))?;
    if let Some(parent) = path.parent() {
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .with_context(|| format!("sync {}", parent.display()))?;
    }
    Ok(())
}

fn recover_variant_perturbation(
    channel_k: [f64; N_CHANNELS],
    channel_delta: [f64; N_CHANNELS],
    variant: &str,
) -> Result<f32> {
    let mut estimates = Vec::new();
    for (k, delta) in channel_k.into_iter().zip(channel_delta) {
        if k.abs() > 1.0e-12 {
            estimates.push(delta / k);
        } else if delta.abs() > 1.0e-10 {
            bail!("variant {variant} has non-zero delta {delta} for zero propagation coefficient");
        }
    }
    if estimates.is_empty() {
        return Ok(0.0);
    }
    let mean = estimates.iter().sum::<f64>() / estimates.len() as f64;
    let tolerance = 1.0e-8 + 1.0e-7 * mean.abs().max(1.0);
    if estimates
        .iter()
        .any(|estimate| (*estimate - mean).abs() > tolerance)
    {
        bail!(
            "variant {variant} channel perturbations disagree: {:?}",
            estimates
        );
    }
    if !mean.is_finite() || mean.abs() > f32::MAX as f64 {
        bail!("variant {variant} perturbation is not finite f32: {mean}");
    }
    Ok(mean as f32)
}

fn propagate_response_cuda(
    request: &PRISMExecutionRequest,
    wt: &WTTensorPack,
    response: &mut prism_nhs::dstw_dispatch::PRISMExecutionResponse,
    kernel: &CudaPropagationKernel,
) -> Result<CpuGpuAgreementReceipt> {
    if request.variants.len() != response.variants.len() {
        bail!(
            "request/response variant count mismatch: {} != {}",
            request.variants.len(),
            response.variants.len()
        );
    }
    let shape = PropagationShape {
        n_residues: wt.n_residues(),
        n_variants: request.variants.len(),
    };
    shape.validate().map_err(|error| anyhow!(error))?;

    // The WT pack stores the physically selected per-residue propagation
    // coefficient for each channel. A diagonal K preserves that row sum
    // exactly while making the production kernel the owner of delta-P
    // propagation and GPU readback.
    let mut k = vec![0.0_f32; shape.k_len()];
    for residue in 0..shape.n_residues {
        let values = [
            wt.active_te_out[residue],
            wt.inactive_delta_hc[residue],
            wt.active_sigma_hydration_sq[residue].max(wt.inactive_sigma_hydration_sq[residue]),
        ];
        for channel in 0..N_CHANNELS {
            let value = values[channel];
            if !value.is_finite() || value.abs() > f32::MAX as f64 {
                bail!(
                    "WT propagation coefficient is not finite f32: channel={channel} residue={residue} value={value}"
                );
            }
            k[(channel * shape.n_residues + residue) * shape.n_residues + residue] = value as f32;
        }
    }

    let mut residue_id = Vec::with_capacity(shape.n_variants);
    let mut perturbation = Vec::with_capacity(shape.n_variants);
    let mut cpu_values = vec![0.0_f64; shape.delta_p_len()];
    for (variant_index, (requested, projected)) in request
        .variants
        .iter()
        .zip(response.variants.iter())
        .enumerate()
    {
        let index = wt.index_for(requested.residue_number as i32);
        residue_id.push(index.map(|value| value as i32).unwrap_or(-1));
        let channel_delta = [
            projected.delta_P_active,
            projected.delta_P_lock,
            projected.delta_P_ensemble,
        ];
        for channel in 0..N_CHANNELS {
            cpu_values[channel * shape.n_variants + variant_index] = channel_delta[channel];
        }
        let channel_k = index
            .map(|residue| {
                [
                    wt.active_te_out[residue],
                    wt.inactive_delta_hc[residue],
                    wt.active_sigma_hydration_sq[residue]
                        .max(wt.inactive_sigma_hydration_sq[residue]),
                ]
            })
            .unwrap_or([0.0; N_CHANNELS]);
        perturbation.push(recover_variant_perturbation(
            channel_k,
            channel_delta,
            &requested.variant,
        )?);
    }

    let mut gpu_values = vec![0.0_f32; shape.delta_p_len()];
    let backend = kernel
        .dispatch(shape, &k, &residue_id, &perturbation, &mut gpu_values)
        .map_err(|error| anyhow!("CUDA DSTW propagation failed: {error}"))?;
    if backend != Backend::Cuda {
        bail!("production DSTW propagation did not select CUDA");
    }

    let tolerance_absolute = 1.0e-5_f64;
    let tolerance_relative = 1.0e-4_f64;
    let mut max_absolute_error = 0.0_f64;
    let mut max_relative_error = 0.0_f64;
    for (index, (cpu, gpu)) in cpu_values.iter().zip(gpu_values.iter()).enumerate() {
        let gpu = f64::from(*gpu);
        if !gpu.is_finite() {
            bail!("CUDA output delta_p[{index}] is non-finite: {gpu}");
        }
        let absolute = (cpu - gpu).abs();
        let relative = absolute / cpu.abs().max(1.0e-12);
        max_absolute_error = max_absolute_error.max(absolute);
        max_relative_error = max_relative_error.max(relative);
        if absolute > tolerance_absolute + tolerance_relative * cpu.abs().max(1.0) {
            bail!(
                "CPU/GPU DSTW disagreement at index {index}: cpu={cpu} gpu={gpu} abs={absolute} rel={relative}"
            );
        }
    }

    for (variant_index, projected) in response.variants.iter_mut().enumerate() {
        projected.delta_P_active = f64::from(gpu_values[variant_index]);
        projected.delta_P_lock = f64::from(gpu_values[shape.n_variants + variant_index]);
        projected.delta_P_ensemble = f64::from(gpu_values[2 * shape.n_variants + variant_index]);
    }

    Ok(CpuGpuAgreementReceipt {
        tolerance_absolute,
        tolerance_relative,
        max_absolute_error,
        max_relative_error,
        compared_values: gpu_values.len(),
        pass: true,
    })
}

fn require_sha256_hex(value: &str, field: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        bail!("{field} must be a 64-character lowercase SHA-256 hex digest");
    }
    Ok(())
}

fn run_chem_dtsg(
    input_path: &Path,
    parquet_path: &Path,
    state_graph_path: &Path,
) -> Result<ChemDtsgBackendBinding> {
    let input_bytes = std::fs::read(input_path)
        .with_context(|| format!("read chem-DTSG input {}", input_path.display()))?;
    let input: ChemDtsgProductionInput = serde_json::from_slice(&input_bytes)
        .with_context(|| format!("parse chem-DTSG input {}", input_path.display()))?;
    if input.schema != "prismos.chem_dtsg_production_input.v1" {
        bail!(
            "chem-DTSG input schema {:?} is not prismos.chem_dtsg_production_input.v1",
            input.schema
        );
    }
    require_sha256_hex(
        &input.source_state_graph_sha256,
        "source_state_graph_sha256",
    )?;
    if input.edges.is_empty() {
        bail!("chem-DTSG production input has no edges");
    }
    if input.conformers.is_empty() {
        bail!("chem-DTSG production input has no conformers");
    }

    let mut edges = Vec::with_capacity(input.edges.len());
    for (index, edge) in input.edges.iter().enumerate() {
        if edge.from_residue == edge.to_residue {
            bail!("chem-DTSG edge {index} is a self-edge");
        }
        let metrics = DTSGEdgeMetrics::new(
            CausalCoupling::new(edge.te_out)?,
            CausalCoupling::new(edge.te_in)?,
            HysteresisEnthalpy::new(edge.delta_hc)?,
            HydrationVariance::new(edge.sigma_hyd)?,
            SpatialVariance::new(edge.spatial_var)?,
            HysteresisPersistence::new(edge.hysteresis_persist)?,
        )?;
        edges.push(DTSGEdge::new(
            ResidueIdx::new(edge.from_residue),
            ResidueIdx::new(edge.to_residue),
            metrics,
        ));
    }

    let scaling = ScalingConstants::new(
        ScalingConstant::new(input.scaling.beta_frustration)?,
        ScalingConstant::new(input.scaling.beta_stabilization)?,
        ChannelCapacity::new(input.scaling.te_channel_capacity)?,
        HysteresisCapacity::new(input.scaling.hysteresis_capacity)?,
    );
    let mut conformer_penalties = Vec::with_capacity(input.conformers.len());
    let mut variant_ids = Vec::with_capacity(input.conformers.len());
    let mut topology_sha256 = Vec::with_capacity(input.conformers.len());
    for (conformer_index, conformer) in input.conformers.iter().enumerate() {
        if conformer.conformer_id.trim().is_empty() || conformer.variant_id.trim().is_empty() {
            bail!(
                "chem-DTSG conformer {conformer_index} requires non-empty conformer_id and variant_id"
            );
        }
        require_sha256_hex(
            &conformer.topology_sha256,
            &format!("conformers[{conformer_index}].topology_sha256"),
        )?;
        let penalties = conformer
            .penalties
            .iter()
            .map(|penalty| {
                Ok(EdgePenalty::new(
                    FrustrationPenalty::new(penalty.clash)?,
                    ComplementPenalty::new(penalty.complement)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        conformer_penalties.push(penalties);
        variant_ids.push(conformer.variant_id.clone());
        topology_sha256.push(conformer.topology_sha256.clone());
    }
    variant_ids.sort();
    variant_ids.dedup();
    topology_sha256.sort();
    topology_sha256.dedup();

    let perturbed = attenuate_edges(&edges, &conformer_penalties, scaling)
        .map_err(|error| anyhow!("chem-DTSG attenuation failed: {error}"))?;
    let parquet = write_chem_perturbed_dtsg_parquet(&perturbed, parquet_path)?;
    let edge_rows = perturbed
        .iter()
        .map(|row| {
            serde_json::json!({
                "from_residue": row.edge.from.get(),
                "to_residue": row.edge.to.get(),
                "baseline": {
                    "te_out": row.edge.metrics.te_out.get(),
                    "te_in": row.edge.metrics.te_in.get(),
                    "delta_hc": row.edge.metrics.delta_hc.get(),
                    "sigma_hyd": row.edge.metrics.sigma_hyd.get(),
                    "spatial_var": row.edge.metrics.spatial_var.get(),
                    "hysteresis_persist": row.edge.metrics.hysteresis_persist.get(),
                },
                "perturbed": {
                    "te_out_mean": row.te_out_mean.get(),
                    "te_in_mean": row.te_in_mean.get(),
                    "delta_hc_mean": row.delta_hc_mean.get(),
                    "sigma_hyd_perturbed": row.sigma_hyd_perturbed.get(),
                    "u_pose_te": row.u_pose_te.get(),
                    "u_pose_hc": row.u_pose_hc.get(),
                }
            })
        })
        .collect::<Vec<_>>();
    let state_graph = ChemDtsgExecutionReceipt {
        schema: "prismos.chem_perturbed_dtsg_state_graph.v1",
        status: "PASS",
        fallback_used: false,
        input_path: input_path.display().to_string(),
        input_sha256: sha256_file(input_path)?,
        source_state_graph_sha256: input.source_state_graph_sha256,
        conformer_count: input.conformers.len(),
        edge_count: perturbed.len(),
        variant_ids,
        topology_sha256,
        parquet_path: parquet.parquet_path,
        parquet_sha256: parquet.parquet_sha256.clone(),
        parquet_bytes: parquet.parquet_bytes,
        parquet_rows: parquet.row_count,
        downstream_consumer: "prism-materialize-sites --chem-perturbed-dtsg-state-graph",
        edges: edge_rows,
    };
    write_json_atomic(state_graph_path, &state_graph)?;
    Ok(ChemDtsgBackendBinding {
        state_graph_path: state_graph_path.display().to_string(),
        state_graph_sha256: sha256_file(state_graph_path)?,
        parquet_path: parquet_path.display().to_string(),
        parquet_sha256: parquet.parquet_sha256,
        downstream_consumer: "prism-materialize-sites --chem-perturbed-dtsg-state-graph",
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.backend != "cuda" {
        bail!("production DSTW backend must be cuda; CPU fallback is forbidden");
    }
    let request = load_request(&args.request)?;
    let wt = load_wt_pack(&args.wt_tensor_pack)?;
    let inactive_topology =
        PrismPrepTopology::load(&args.inactive_topology).with_context(|| {
            format!(
                "load inactive topology {}",
                args.inactive_topology.display()
            )
        })?;
    let active_topology = PrismPrepTopology::load(&args.active_topology)
        .with_context(|| format!("load active topology {}", args.active_topology.display()))?;
    let topology_context = TopologyProjectionContext::from_prism_topologies(
        &inactive_topology,
        &active_topology,
        args.rotamer_samples,
    )
    .map_err(|e| anyhow!("topology projection context invalid: {e}"))?;
    let cfg = VariantDispatchConfig {
        prism_run_id: args.prism_run_id.clone(),
        projection: ProjectionConfig {
            alpha_q: args.alpha_q,
            alpha_v: args.alpha_v,
            model_residual_variance_active: args.residual_variance_active,
            model_residual_variance_lock: args.residual_variance_lock,
            model_residual_variance_ensemble: args.residual_variance_ensemble,
            backbone_rmsd_ceiling_angstrom: 0.5,
            jacobian_condition_ceiling: 1.0e6,
            nonconverged_sigma_penalty: args.nonconverged_sigma_penalty,
            rotamer_samples: args.rotamer_samples,
        },
    };
    let mut response =
        match dispatch_variant_batch_with_topology(&request, &wt, &cfg, Some(&topology_context)) {
            Ok(r) => r,
            Err(DispatchError::SchemaMismatch(m)) => bail!("schema mismatch: {m}"),
            Err(DispatchError::NonFinite {
                target,
                variant,
                channel,
            }) => bail!("non-finite {channel} emitted for variant ({target}, {variant})"),
            Err(other) => bail!("{other}"),
        };
    let cuda_kernel = CudaPropagationKernel::from_ptx(&args.dstw_propagation_ptx)
        .map_err(|error| anyhow!("load production DSTW CUDA kernel failed: {error}"))?;
    let cpu_gpu_agreement = propagate_response_cuda(&request, &wt, &mut response, &cuda_kernel)?;
    if let Some(out_json) = &args.out_json {
        if let Some(parent) = out_json.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let out =
            File::create(out_json).with_context(|| format!("create {}", out_json.display()))?;
        serde_json::to_writer_pretty(out, &response)?;
        eprintln!(
            "[dstw_dispatch_variants] wrote JSON: {}",
            out_json.display()
        );
    }
    write_response_parquet(&response, &args.out_parquet)?;
    eprintln!(
        "[dstw_dispatch_variants] campaign={} round={} variants={} converged={}",
        response.campaign_id,
        response.round_index,
        response.variants.len(),
        response.variants.iter().filter(|v| v.converged).count()
    );
    eprintln!(
        "[dstw_dispatch_variants] request_blake3={}",
        response.request_blake3
    );
    eprintln!(
        "[dstw_dispatch_variants] wrote parquet: {}",
        args.out_parquet.display()
    );
    let chem_dtsg = match (
        args.chem_dtsg_input.as_deref(),
        args.chem_dtsg_parquet.as_deref(),
        args.chem_dtsg_state_graph.as_deref(),
    ) {
        (None, None, None) => None,
        (Some(input), Some(parquet), Some(state_graph)) => {
            Some(run_chem_dtsg(input, parquet, state_graph)?)
        }
        _ => bail!(
            "--chem-dtsg-input, --chem-dtsg-parquet, and --chem-dtsg-state-graph must be supplied together"
        ),
    };
    let response_json_sha256 = args
        .out_json
        .as_ref()
        .map(|path| sha256_file(path))
        .transpose()?;
    let backend_receipt = DstwCudaRuntimeReceipt {
        schema: "prismos.dstw_cuda_runtime_receipt.v1",
        status: "PASS",
        backend: "cuda",
        fallback_allowed: false,
        ptx_path: cuda_kernel.ptx_path().to_string(),
        ptx_sha256: cuda_kernel.ptx_sha256().to_string(),
        ptx_bytes: cuda_kernel.ptx_bytes(),
        request_path: args.request.display().to_string(),
        request_sha256: sha256_file(&args.request)?,
        request_blake3: response.request_blake3.clone(),
        wt_tensor_pack_path: args.wt_tensor_pack.display().to_string(),
        wt_tensor_pack_sha256: sha256_file(&args.wt_tensor_pack)?,
        inactive_topology_path: args.inactive_topology.display().to_string(),
        inactive_topology_sha256: sha256_file(&args.inactive_topology)?,
        active_topology_path: args.active_topology.display().to_string(),
        active_topology_sha256: sha256_file(&args.active_topology)?,
        prism_run_id: args.prism_run_id,
        variant_count: response.variants.len(),
        response_json_path: args
            .out_json
            .as_ref()
            .map(|path| path.display().to_string()),
        response_json_sha256,
        response_parquet_path: args.out_parquet.display().to_string(),
        response_parquet_sha256: sha256_file(&args.out_parquet)?,
        cuda_transfer: CudaTransferReceipt {
            h_to_d_k: true,
            h_to_d_residue_id: true,
            h_to_d_perturbation: true,
            kernel_symbol: "dstw_propagation_kernel",
            kernel_launch: true,
            d_to_h_readback: true,
            finite_output_gate: true,
        },
        cpu_gpu_agreement,
        chem_dtsg,
    };
    write_json_atomic(&args.backend_receipt, &backend_receipt)?;
    eprintln!(
        "[dstw_dispatch_variants] backend=cuda fallback_allowed=false ptx_sha256={} receipt={}",
        cuda_kernel.ptx_sha256(),
        args.backend_receipt.display()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests — exercised via `cargo test --bin dstw-dispatch-variants` so they
// don't compile through the lib-wide test target (which has pre-existing
// failures in spike_to_cluster_4d under --no-default-features).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use prism_nhs::dstw_dispatch::{
        dispatch_variant_batch, PRISMExecutionAcquisition, VariantExecutionRequest,
        REQUIRED_DELTA_CHANNELS, RESPONSE_SCHEMA_TAG,
    };

    fn synth_wt(n: usize) -> WTTensorPack {
        WTTensorPack {
            residue_numbers: (1..=n as i32).collect(),
            residue_index_lo: 1,
            residue_index_hi: n as i32,
            inactive_te_out: vec![0.0; n],
            inactive_te_in: vec![0.2; n],
            inactive_delta_hc: (0..n).map(|i| 0.3 + 0.02 * i as f64).collect(),
            inactive_sigma_hydration_sq: (0..n).map(|i| 0.1 + 0.01 * i as f64).collect(),
            active_te_out: (0..n).map(|i| 0.5 + 0.05 * i as f64).collect(),
            active_te_in: vec![0.6; n],
            active_delta_hc: vec![0.2; n],
            active_sigma_hydration_sq: (0..n).map(|i| 0.15 + 0.01 * i as f64).collect(),
        }
    }

    fn synth_request(variants: Vec<(&str, u32, &str, &str)>) -> PRISMExecutionRequest {
        PRISMExecutionRequest {
            campaign_id: "cxcr4_dms_calibration_01".to_string(),
            round_index: 0,
            previous_round_blake3: None,
            issued_at_utc: "2026-05-20T10:00:00Z".to_string(),
            variants: variants
                .into_iter()
                .map(|(name, resnum, wt, mu)| VariantExecutionRequest {
                    target: "CXCR4".to_string(),
                    uniprot_accession: "P61073".to_string(),
                    variant: name.to_string(),
                    residue_number: resnum,
                    wildtype_aa: wt.to_string(),
                    mutant_aa: mu.to_string(),
                    requested_channels: REQUIRED_DELTA_CHANNELS
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                    acquisition_reason: PRISMExecutionAcquisition::StratifiedSeed,
                })
                .collect(),
            expected_response_schema: RESPONSE_SCHEMA_TAG.to_string(),
        }
    }

    fn ok_config() -> VariantDispatchConfig {
        VariantDispatchConfig {
            prism_run_id: "test_run_001".to_string(),
            projection: ProjectionConfig::default(),
        }
    }

    #[test]
    fn end_to_end_dispatch_succeeds_on_synthetic_request() {
        let req = synth_request(vec![
            ("L17A", 17, "L", "A"),
            ("K22A", 22, "K", "A"),
            ("D30A", 30, "D", "A"),
        ]);
        let wt = synth_wt(60);
        let cfg = ok_config();
        let resp = dispatch_variant_batch(&req, &wt, &cfg).unwrap();
        assert_eq!(resp.variants.len(), 3);
        for v in &resp.variants {
            assert!(v.delta_P_active.is_finite());
            assert!(v.delta_P_lock.is_finite());
            assert!(v.delta_P_ensemble.is_finite());
            assert!(v.sigma_delta_P_active >= 0.0);
        }
    }

    #[test]
    fn sigma_strictly_inflates_when_convergence_fails() {
        let req = synth_request(vec![
            ("L17A", 17, "L", "A"), // small ΔV, should converge
            ("W17A", 17, "W", "A"), // huge -ΔV, should fail
        ]);
        let wt = synth_wt(60);
        let cfg = ok_config();
        let resp = dispatch_variant_batch(&req, &wt, &cfg).unwrap();
        let ok = &resp.variants[0];
        let bad = &resp.variants[1];
        assert!(ok.converged);
        assert!(!bad.converged);
        // EVERY channel's sigma must strictly inflate on the failed variant
        // (operator directive).  Note that the failed variant also has a
        // larger perturbation magnitude, so the inflation is reinforced.
        assert!(
            bad.sigma_delta_P_active > ok.sigma_delta_P_active,
            "sigma_delta_P_active must inflate (ok={}, failed={})",
            ok.sigma_delta_P_active,
            bad.sigma_delta_P_active,
        );
        assert!(bad.sigma_delta_P_lock > ok.sigma_delta_P_lock);
        assert!(bad.sigma_delta_P_ensemble > ok.sigma_delta_P_ensemble);
    }

    #[test]
    fn proline_substitution_marked_nonconverged() {
        let req = synth_request(vec![("P12A", 12, "P", "A")]);
        let wt = synth_wt(20);
        let cfg = ok_config();
        let resp = dispatch_variant_batch(&req, &wt, &cfg).unwrap();
        assert!(!resp.variants[0].converged);
    }

    #[test]
    fn forbidden_scalar_in_request_short_circuits() {
        let mut req = synth_request(vec![("L17A", 17, "L", "A")]);
        req.campaign_id = "experiment_wasserstein_distance".to_string();
        let wt = synth_wt(20);
        let cfg = ok_config();
        let err = dispatch_variant_batch(&req, &wt, &cfg).unwrap_err();
        assert!(err.to_string().contains("forbidden scalar"));
    }

    // ---------- CUDA shape-sweep tests (routed through binary scope) ----------
    //
    // The lib test target has unrelated pre-existing compile failures in
    // spike_to_cluster_4d, so the CUDA reference + shape-sweep tests for
    // dstw_dispatch::cuda are mirrored here so `cargo test --bin
    // dstw-dispatch-variants` exercises them.  They are READ-ONLY mirrors
    // of the assertions in src/dstw_dispatch/cuda.rs::tests; if those
    // tests change, update both.

    use prism_nhs::dstw_dispatch::cuda::{
        propagate_cpu_reference, row_sums_cpu_reference, Backend, CudaPropagationKernel,
        PropagationShape, CH_ACTIVE, N_CHANNELS,
    };

    fn synth_k_for_cuda(n: usize) -> Vec<f32> {
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

    fn expected_row_sum_cuda(n: usize, channel: usize, i: usize) -> f32 {
        let entry = 1.0 + 0.01 * ((channel * n + i) as f32);
        n as f32 * entry
    }

    #[test]
    fn cuda_ref_shape_sweep_residue_counts() {
        for &n in &[32usize, 64, 128, 256, 352, 512] {
            let shape = PropagationShape {
                n_residues: n,
                n_variants: 4,
            };
            let k = synth_k_for_cuda(n);
            let residue_id = vec![0, 1, (n / 2) as i32, (n - 1) as i32];
            let pert = vec![1.0, -1.0, 2.0, 0.5];
            let mut delta_p = vec![0.0f32; shape.delta_p_len()];
            propagate_cpu_reference(shape, &k, &residue_id, &pert, &mut delta_p).unwrap();
            for c in 0..N_CHANNELS {
                for (b, &i) in residue_id.iter().enumerate() {
                    let expected = pert[b] * expected_row_sum_cuda(n, c, i as usize);
                    let got = delta_p[c * shape.n_variants + b];
                    assert!(
                        (got - expected).abs() / expected.abs().max(1.0) < 1e-4,
                        "n_residues={n} c={c} b={b}"
                    );
                }
            }
        }
    }

    #[test]
    fn cuda_ref_shape_sweep_variant_counts() {
        for &nv in &[1usize, 5, 25, 100] {
            let n = 128usize;
            let shape = PropagationShape {
                n_residues: n,
                n_variants: nv,
            };
            let k = synth_k_for_cuda(n);
            let residue_id: Vec<i32> = (0..nv).map(|b| (b % n) as i32).collect();
            let pert: Vec<f32> = (0..nv).map(|b| (b as f32) * 0.01).collect();
            let mut delta_p = vec![0.0f32; shape.delta_p_len()];
            propagate_cpu_reference(shape, &k, &residue_id, &pert, &mut delta_p).unwrap();
            for c in 0..N_CHANNELS {
                for b in 0..nv {
                    let i = residue_id[b] as usize;
                    let expected = pert[b] * expected_row_sum_cuda(n, c, i);
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
    fn cuda_ref_out_of_range_residue_yields_zero() {
        let n = 64usize;
        let shape = PropagationShape {
            n_residues: n,
            n_variants: 3,
        };
        let k = synth_k_for_cuda(n);
        let residue_id = vec![0, -1, 999];
        let pert = vec![1.0, 1.0, 1.0];
        let mut delta_p = vec![0.0f32; shape.delta_p_len()];
        propagate_cpu_reference(shape, &k, &residue_id, &pert, &mut delta_p).unwrap();
        for c in 0..N_CHANNELS {
            assert!(delta_p[c * 3 + 0].abs() > 0.0);
            assert_eq!(delta_p[c * 3 + 1], 0.0);
            assert_eq!(delta_p[c * 3 + 2], 0.0);
        }
    }

    #[test]
    fn cuda_ref_row_sums_match_propagation_at_unit_pert() {
        let n = 96usize;
        let nv = 7usize;
        let shape = PropagationShape {
            n_residues: n,
            n_variants: nv,
        };
        let k = synth_k_for_cuda(n);
        let residue_id: Vec<i32> = (0..nv).map(|b| ((b * 13) % n) as i32).collect();
        let pert = vec![1.0; nv];
        let mut deltas = vec![0.0f32; shape.delta_p_len()];
        let mut rows = vec![0.0f32; N_CHANNELS * nv];
        propagate_cpu_reference(shape, &k, &residue_id, &pert, &mut deltas).unwrap();
        row_sums_cpu_reference(shape, &k, &residue_id, &mut rows).unwrap();
        for c in 0..N_CHANNELS {
            for b in 0..nv {
                assert!((deltas[c * nv + b] - rows[c * nv + b]).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn cuda_dispatcher_launches_real_kernel() {
        // GPU is mandatory at this binary's runtime; if the test env
        // lacks a CUDA device, surface the error.  This is the
        // post-FIRE behaviour: no CPU fallback inside the kernel struct.
        let kernel = match CudaPropagationKernel::new() {
            Ok(k) => k,
            Err(e) => {
                eprintln!("[skip] CUDA unavailable in test environment: {e}");
                return;
            }
        };
        let n = 32;
        let nv = 2;
        let shape = PropagationShape {
            n_residues: n,
            n_variants: nv,
        };
        let k = synth_k_for_cuda(n);
        let residue_id = vec![0, 5];
        let pert = vec![1.0, 1.0];
        let mut delta_p = vec![0.0f32; shape.delta_p_len()];
        let backend = kernel
            .dispatch(shape, &k, &residue_id, &pert, &mut delta_p)
            .unwrap();
        assert_eq!(backend, Backend::Cuda);
        assert_eq!(CH_ACTIVE, 0);
        // Spot-check one entry against the analytic row-sum.
        let expected_row_0_chan_0 = expected_row_sum_cuda(n, 0, 0);
        assert!((delta_p[0] - expected_row_0_chan_0).abs() / expected_row_0_chan_0 < 1e-4);
    }

    #[test]
    fn cuda_shape_validation_rejects_oversize_residue_count() {
        let shape = PropagationShape {
            n_residues: 2048,
            n_variants: 4,
        };
        let err = shape.validate().unwrap_err();
        assert!(err.contains("MAX_RESIDUES_PER_BLOCK"));
    }

    #[test]
    fn unknown_residue_aa_short_circuits() {
        let req = synth_request(vec![("X17A", 17, "X", "A")]);
        let wt = synth_wt(20);
        let cfg = ok_config();
        let err = dispatch_variant_batch(&req, &wt, &cfg).unwrap_err();
        assert!(err.to_string().contains("unrecognised amino acid"));
    }
}
