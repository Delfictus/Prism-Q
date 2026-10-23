//! Direct eight-leaf native biprism execution through nhs_rt_full.
//! Input identity and required surfaces are resolved before device creation.
//! Physical engines, governed modules, queues and graphs retain one owner;
//! accepted chunks feed the actual native Arrow, clustering and SDST consumers.
//! A rejected physical body remains a diagnostic event and cannot be replayed.
use crate::{
    biprism_collective_runtime::{
        BiprismCollectiveConfig, BiprismCollectiveFunctions, BiprismCollectiveRuntime,
        BiprismCollectiveStatus,
    },
    biprism_lif_runtime::NativeObservationFunctions,
    biprism_mesh::{select_native_population, validate_mesh_readback},
    biprism_spike::{
        decode_native_lif_observation_queue, decode_spike_queue_for_arrow_and_pathb_in_frame,
        BiprismDecodeLimits, BiprismEventFrame, DecodedVoronoiSpikeQueue,
    },
    biprism_spike_arrow::decode_voronoi_spike_export_in_frame,
    biprism_v10_runtime::{
        BiprismIdentity, BiprismV10DeviceState, BiprismV10Functions, BorrowedPhysicalLeaf,
        ConstraintGroup, DeviceRegion, GovernedModuleIdentity, LeafMassAndSelection,
        V10AcceptanceLimits, V10Step,
    },
    fused_engine::{
        BiprismConstraintPolicy, BiprismPhysicalLeafAdapter, CryoUvProtocol, GpuSpikeEvent,
        NhsAmberFusedEngine, SignalPreservationData,
    },
    gpu_cluster_backend::GpuSpatialHashBackend,
    input::{HarmonicEnergyConvention, NonbondedExecutionPolicy, PrismPrepTopology},
    persistent_engine::ClusteredBindingSite,
    rt_clustering::RtClusteringResult,
    sdst_bridge::{SdstBridge, SdstGridGeometry},
    spike_arrow_writer::{
        chunk_idx_for_timestep, classify_background, compute_background_percentiles,
        CcnsPhaseCoding, SpikeClassification,
    },
};
use anyhow::{anyhow, bail, ensure, Context, Result};
use cudarc::{
    driver::{CudaContext, CudaStream},
    nvrtc::Ptx,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

const PS_TO_ATOMIC_TIME: f64 = 41341.37333517015;
const INV_DALTON_TO_INV_ELECTRON: f64 = 0.0005485799090441;
const LEAVES: usize = 8;
const CORE: usize = 192;
const VERTICES: usize = 256;
const NEIGHBOR_STRIDE: usize = 1024;

#[path = "biprism_harmonic_binding.rs"]
mod harmonic_binding;

#[derive(clap::Args, Clone)]
pub struct BiprismRunArgs {
    /// Hash-pinned physical inputs, eight protocols and full requested run budget.
    #[arg(long)]
    pub config: PathBuf,
    /// Resolve inputs, masses and governed membership without CUDA initialization.
    #[arg(long, default_value_t = false)]
    pub validate_only: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PinnedFile {
    pub path: PathBuf,
    pub sha256: String,
}
impl PinnedFile {
    fn read(&self) -> Result<Vec<u8>> {
        ensure!(
            self.path.is_absolute(),
            "pinned path must be absolute: {}",
            self.path.display()
        );
        ensure!(
            self.sha256.len() == 64 && self.sha256.bytes().all(|x| x.is_ascii_hexdigit()),
            "invalid SHA256 pin"
        );
        let meta = fs::symlink_metadata(&self.path)?;
        ensure!(
            meta.is_file() && !meta.file_type().is_symlink(),
            "pinned input must be a regular file"
        );
        let bytes = fs::read(&self.path)?;
        ensure!(
            hash(&bytes).eq_ignore_ascii_case(&self.sha256),
            "SHA256 mismatch: {}",
            self.path.display()
        );
        Ok(bytes)
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrustClass {
    Signed,
    SourceOwnedDevHashBound,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GovernedModules {
    pub trust_manifest: PinnedFile,
    pub trust_class: TrustClass,
    pub ptx_directory: PathBuf,
    pub biprism_sha256: String,
    pub fused_sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicalMassOrigin {
    pub authority: PinnedFile,
    pub mapping: PinnedFile,
    pub true_inverse_masses_au: PinnedFile,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NmaConfiguration {
    pub modes: PinnedFile,
    pub amplification: f32,
    pub scan_fraction: f32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum AscConfiguration {
    Disabled,
    SelectedSupport { policy: PinnedFile, alpha: f32 },
    ExplicitLegacyWholeSystem { alpha: f32 },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GovernorConfiguration {
    FixedStepMeasurement,
    CollectiveSelectedMeasurement,
    ObserverClock,
    ArmedAdaptive,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LeafConfiguration {
    pub seed: u64,
    pub stream_id: u8,
    pub center_atom: u32,
    pub protocol: CryoUvProtocol,
    pub uv_enabled: bool,
    pub nma: Option<NmaConfiguration>,
    pub asc: AscConfiguration,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericalContract {
    pub dt_ps: f32,
    pub kappa_foliated: f32,
    pub constraint_policy: BiprismConstraintPolicy,
    pub gonzalez_exact_energy: bool,
    pub lif_decay: f32,
    pub lif_threshold: f32,
    pub lif_kick: f32,
    pub picard_residual: f32,
    pub dec_residual: f32,
    pub absolute_energy_drift: f32,
    pub unitarity_defect: f32,
    pub dec_iteration_cap: u32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BiprismRunConfig {
    pub schema: String,
    pub run_id: String,
    pub output_root: PathBuf,
    pub device_ordinal: usize,
    pub topology: PinnedFile,
    /// Optional exact-row provider contract import; the target file stays unchanged.
    #[serde(default)]
    pub harmonic_source_binding: Option<PinnedFile>,
    pub masses: PhysicalMassOrigin,
    pub unit_bridge_source: PinnedFile,
    pub modules: GovernedModules,
    pub leaves: [LeafConfiguration; 8],
    pub numerical: NumericalContract,
    pub physical_time_generation:u64,
    pub director_adaptive_dt:bool,
    pub hold_terminal_controls:bool,
    pub governor: GovernorConfiguration,
    #[serde(default)]
    pub adaptive_dt_bounds:Option<crate::biprism_shared_step::AdaptiveDtBounds>,
    #[serde(default)]
    pub max_seed_peel_passes:Option<u32>,
    pub required_surfaces: Vec<String>,
    pub grid_dimension: u32,
    pub grid_spacing_angstrom: f32,
    pub target_tetrahedra: u32,
    pub queue_capacity: u32,
    /// Logical observational CAS quota; scientific restart state is a separate contract.
    pub chronicle_max_bytes: u64,
    pub chunk_steps: u32,
    pub full_step_budget: u32,
    pub clustering_epsilon_angstrom: f32,
    pub source_percentile: u32,
    pub wavefront_max_dt_steps: Option<u32>,
}

pub struct BiprismRunConsumers {
    pub write_signal_grid:
        fn(&Path, &SignalPreservationData, u32, &str, &str) -> Result<(u64, u64)>,
    pub build_sites: fn(&[GpuSpikeEvent], &RtClusteringResult) -> Result<Vec<ClusteredBindingSite>>,
}
struct ResolvedInputs {
    topology: PrismPrepTopology,
    harmonic_source_binding: Option<Value>,
    true_inverse_masses: Vec<f32>,
    constraints: Vec<ConstraintGroup>,
    elements: Vec<u8>,
    atom_map_sha256: String,
    manifest: Value,
    biprism_ptx: Vec<u8>,
    fused_ptx: Vec<u8>,
    nma_text: [Option<Vec<u8>>; 8],
    asc_policies: [Option<crate::asc_support::AscSelectedSupportPolicy>; 8],
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn serr<T>(r: std::result::Result<T, String>) -> Result<T> {
    r.map_err(|e| anyhow!(e))
}
fn retain(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("retain {}", path.display()))?;
    f.write_all(bytes)?;
    f.sync_all()?;
    if let Some(p) = path.parent() {
        File::open(p)?.sync_all()?;
    }
    Ok(())
}
fn retain_json(path: &Path, value: &Value) -> Result<()> {
    retain(path, &serde_json::to_vec_pretty(value)?)
}

impl BiprismRunConfig {
    fn validate_shape(&self) -> Result<()> {
        ensure!(
            self.schema == "prism.biprism.native_run.v1",
            "unsupported run configuration schema"
        );
        ensure!(
            !self.run_id.is_empty()
                && self.run_id.len() <= 128
                && self
                    .run_id
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-'),
            "unsafe run identifier"
        );
        ensure!(
            self.output_root.is_absolute(),
            "output root must be absolute"
        );
        ensure!(
            self.chunk_steps == 500
                && self.full_step_budget >= self.chunk_steps
                && self.full_step_budget <= i32::MAX as u32,
            "full production budget must be explicit and chunk_steps must be500"
        );
        ensure!(
            (1..=1024).contains(&self.target_tetrahedra)
                && self.queue_capacity > 0
                && self.queue_capacity <= i32::MAX as u32,
            "invalid tile/queue capacity"
        );
        ensure!(
            (1..=128).contains(&self.grid_dimension)
                && self.grid_spacing_angstrom.is_finite()
                && self.grid_spacing_angstrom > 0.0,
            "invalid native grid"
        );
        ensure!(
            self.clustering_epsilon_angstrom.is_finite()
                && self.clustering_epsilon_angstrom > 0.0
                && self.source_percentile <= 100,
            "invalid native analysis contract"
        );
        let known_clock_bytes = u64::from(self.full_step_budget) * 8 * 24;
        ensure!(
            self.chronicle_max_bytes >= known_clock_bytes + 64 * 1024 * 1024,
            "Chronicle quota cannot hold complete native clocks and metadata reserve"
        );
        let n = &self.numerical;
        ensure!(
            n.dt_ps.is_finite()
                && n.dt_ps > 0.0
                && n.kappa_foliated.is_finite()
                && n.dec_iteration_cap == 192,
            "invalid physical timestep/kappa or DEC iteration cap"
        );
        ensure!(
            [
                n.picard_residual,
                n.dec_residual,
                n.absolute_energy_drift,
                n.unitarity_defect
            ]
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0),
            "invalid acceptance threshold"
        );
        ensure!(
            n.picard_residual <= 1e-7 && n.dec_residual <= 1e-6
                && n.absolute_energy_drift <= 1e-6 && n.unitarity_defect <= 1e-7,
            "acceptance cannot relax the declared residual, energy or unitarity gates"
        );
        ensure!(
            n.lif_decay.is_finite()
                && (0.0..=1.0).contains(&n.lif_decay)
                && n.lif_threshold.is_finite()
                && n.lif_threshold > 0.0
                && n.lif_kick.is_finite()
                && n.lif_kick >= 0.0,
            "invalid LIF configuration"
        );
        ensure!(matches!(self.governor,GovernorConfiguration::CollectiveSelectedMeasurement|GovernorConfiguration::ArmedAdaptive)&&self.physical_time_generation!=0&&self.hold_terminal_controls,"this coupled profile requires a selected common-time owner, nonzero generation and terminal continuation");
        match self.governor {
            GovernorConfiguration::ArmedAdaptive=>{
                let b=self.adaptive_dt_bounds.context("armed_adaptive requires explicit adaptive_dt_bounds")?;
                serr(b.validate())?;
                ensure!(self.director_adaptive_dt,"armed_adaptive must preserve live Director requests");
                ensure!(n.dt_ps>=b.dt_min_ps&&n.dt_ps<=b.dt_max_ps,"initial request outside explicit adaptive bounds");
            },
            _=>ensure!(self.adaptive_dt_bounds.is_none(),"adaptive bounds require the explicit armed_adaptive common selector"),
        }
        if let Some(cap)=self.max_seed_peel_passes{ensure!(cap>0,"seed peel acceptance bound must be positive");}
        let mut stream_ids = BTreeSet::new();
        for (i, l) in self.leaves.iter().enumerate() {
            ensure!(
                stream_ids.insert(l.stream_id),
                "leaf stream identities must be unique"
            );
            let p = &l.protocol;
            ensure!(p.current_step==0,"this factory starts a new physical epoch; restart requires complete state continuity");
            ensure!(
                [p.start_temp, p.end_temp]
                    .iter()
                    .all(|t| t.is_finite() && *t > 0.0),
                "leaf{i} invalid temperature"
            );
            ensure!(p.start_temp >= 50.0,
                "leaf{i} native replica initialization floors temperature below50K; explicit alternate initializer required");
            ensure!(
                [
                    p.cold_hold_steps,
                    p.ramp_steps,
                    p.warm_hold_steps,
                    p.ramp_down_steps,
                    p.cold_return_steps
                ]
                .iter()
                .all(|x| *x >= 0),
                "leaf{i} negative protocol duration"
            );
            let extent = [
                p.cold_hold_steps,
                p.ramp_steps,
                p.warm_hold_steps,
                p.ramp_down_steps,
                p.cold_return_steps,
            ]
            .iter()
            .map(|v| *v as u64)
            .sum::<u64>();
            ensure!(
                extent>0&&extent<=i32::MAX as u64,
                "leaf{i} nominal protocol extent is invalid for terminal baseline capture"
            );
            ensure!(
                p.uv_burst_energy.is_finite()
                    && p.uv_burst_energy >= 0.0
                    && p.uv_burst_interval > 0
                    && p.uv_burst_duration >= 0
                    && p.uv_burst_duration <= p.uv_burst_interval
                    && p.wavelength_dwell_steps > 0
                    && !p.scan_wavelengths.is_empty()
                    && p.scan_wavelengths.iter().all(|v| v.is_finite() && *v > 0.0),
                "leaf{i} invalid UV protocol"
            );
            ensure!(
                p.stepped_holds
                    .iter()
                    .all(|(t, s)| t.is_finite() && *t > 0.0 && *s > 0),
                "leaf{i} invalid stepped hold"
            );
            if let Some(n) = &l.nma {
                ensure!(
                    n.amplification.is_finite() && (0.05..=0.95).contains(&n.scan_fraction),
                    "NMA controls must not trigger silent scan-fraction clamping"
                );
            }
            match l.asc {
                AscConfiguration::Disabled => {}
                AscConfiguration::SelectedSupport { alpha, .. }
                | AscConfiguration::ExplicitLegacyWholeSystem { alpha } => {
                    ensure!(alpha.is_finite(), "invalid ASC gain")
                }
            }
        }
        let supported: BTreeSet<&str> = [
            "amber",
            "protocol",
            "uv",
            "constraints",
            "work",
            "v10",
            "voronoi_lif",
            "native_observation",
            "arrow_v5",
            "full_ledger",
            "chronicle",
            "four_grids",
            "pathb",
            "clustering",
            "nma",
            "asc",
            "governor",
        ]
        .into_iter()
        .collect();
        let required: BTreeSet<&str> = self.required_surfaces.iter().map(String::as_str).collect();
        ensure!(
            required.len() == self.required_surfaces.len()
                && required.iter().all(|s| supported.contains(s)),
            "unknown or duplicate required surface"
        );
        for fixed in [
            "amber",
            "protocol",
            "constraints",
            "work",
            "v10",
            "voronoi_lif",
            "native_observation",
            "arrow_v5",
            "full_ledger",
            "chronicle",
            "four_grids",
            "pathb",
            "clustering",
        ] {
            ensure!(
                required.contains(fixed),
                "required native execution surface missing: {fixed}"
            );
        }
        if required.contains("uv") {
            ensure!(
                self.leaves.iter().all(|l| l.uv_enabled),
                "requested UV must be enabled on all8 declared leaves"
            );
        }
        if required.contains("nma") {
            ensure!(
                self.leaves.iter().all(|l| l.nma.is_some()),
                "requested NMA has no mode source for one or more leaves"
            );
        }
        if required.contains("asc") {
            ensure!(
                self.leaves
                    .iter()
                    .all(|l| !matches!(l.asc, AscConfiguration::Disabled)),
                "requested ASC has an unbound leaf"
            );
        }
        if required.contains("governor") {
            ensure!(
                matches!(self.governor, GovernorConfiguration::CollectiveSelectedMeasurement|GovernorConfiguration::ArmedAdaptive),
                "requested governor is not the declared physical clock owner"
            );
        }
        Ok(())
    }
}

/// Topology-side prerequisites of the exact constructor used by this command.
/// Run these before CUDA creation in both validate-only and physical execution.
/// Lowering is read-only; construction still performs its own admission.
fn validate_native_constructor_inputs(topology: &PrismPrepTopology) -> Result<()> {
    serr(topology.validate())?;
    let historical = serr(topology.historical_may_literal_execution_enabled())?;
    serr(topology.lower_harmonic_parameters(HarmonicEnergyConvention::AmberNoHalfEnergy))
        .context("biprism native harmonic construction admission")?;
    ensure!(
        !historical,
        "biprism exact production constructor cannot use a historical May literal/model admission"
    );
    serr(topology.validate_for_nonbonded_execution(NonbondedExecutionPolicy::RequireExactV2))
        .context("biprism native nonbonded construction admission")?;
    serr(topology.validate_lambda_fragment_preparation())
        .context("biprism native lambda-fragment preparation admission")?;
    ensure!(
        topology.lambda_fragment_system.is_none(),
        "biprism constructor has no lambda runtime binding; lambda-fragment topology requires an explicit stream/phase/output binding"
    );
    Ok(())
}

fn resolve_inputs(c: &BiprismRunConfig) -> Result<ResolvedInputs> {
    c.validate_shape()?;
    let raw = c.topology.read()?;
    let target: Value = serde_json::from_slice(&raw)?;
    let mut topology: PrismPrepTopology = serde_json::from_slice(&raw)?;
    let harmonic_source_binding = c.harmonic_source_binding.as_ref().map(|pin| {
        harmonic_binding::bind(&mut topology, &target, &c.topology, &pin.read()?)
    }).transpose()?;
    validate_native_constructor_inputs(&topology)?;
    ensure!(
        topology.n_atoms >= 256 && topology.n_atoms < 0x0100_0000,
        "atom population outside packed ABI"
    );
    ensure!(
        c.leaves
            .iter()
            .all(|l| l.center_atom < topology.n_atoms as u32),
        "center atom outside topology"
    );
    let authority: Value = serde_json::from_slice(&c.masses.authority.read()?)?;
    let mapping: Value = serde_json::from_slice(&c.masses.mapping.read()?)?;
    ensure!(
        mapping["schema"] == "prism.physical_mass_mapping.v1"
            && mapping["target_topology"]["sha256"] == c.topology.sha256
            && mapping["mass_authority"]["sha256"] == c.masses.authority.sha256
            && mapping["mass_field"] == "forcefield_masses",
        "physical mass mapping does not bind the selected sources"
    );
    for field in [
        "n_atoms",
        "elements",
        "atom_names",
        "residue_ids",
        "chain_ids",
        "residue_names",
        "residue_to_atom_indices",
        "pdb_atom_serial_to_topo_index",
    ] {
        ensure!(
            target.get(field) == authority.get(field),
            "mass authority atom identity differs at {field}"
        );
    }
    let atoms = mapping["atoms"]
        .as_array()
        .context("mass mapping rows absent")?;
    let masses = authority["forcefield_masses"]
        .as_array()
        .context("pre-HMR mass authority lacks forcefield_masses")?;
    ensure!(
        masses.len() == topology.n_atoms && atoms.len() == topology.n_atoms,
        "mass row count mismatch"
    );
    let inverse_bytes = c.masses.true_inverse_masses_au.read()?;
    ensure!(
        inverse_bytes.len() == topology.n_atoms * 4,
        "physical inverse mass byte population mismatch"
    );
    let mut true_inverse_masses = Vec::with_capacity(topology.n_atoms);
    for i in 0..topology.n_atoms {
        let m = masses[i].as_f64().context("invalid physical mass")?;
        let row = &atoms[i];
        let actual = f32::from_le_bytes(inverse_bytes[4 * i..4 * i + 4].try_into().unwrap());
        ensure!(
            m.is_finite()
                && m > 0.0
                && row["atom_index"].as_u64() == Some(i as u64)
                && row["element"] == topology.elements[i]
                && row["atom_name"] == topology.atom_names[i]
                && row["residue_id"].as_u64() == Some(topology.residue_ids[i] as u64)
                && row["physical_mass_dalton"].as_f64() == Some(m)
                && row["serialized_hmr_mass_dalton"] == target["masses"][i],
            "mass mapping row{i} identity/value mismatch"
        );
        let expected = ((1.0 / m) * INV_DALTON_TO_INV_ELECTRON) as f32;
        ensure!(
            actual.is_finite() && actual > 0.0 && actual.to_bits() == expected.to_bits(),
            "physical AU inverse mass mismatch at atom{i}"
        );
        true_inverse_masses.push(actual);
    }
    let mut constraints = Vec::new();
    for h in &topology.h_clusters {
        ensure!(
            (1..=3).contains(&h.n_hydrogens) && h.hydrogen_atoms.len() >= h.n_hydrogens,
            "invalid topology constraint population"
        );
        let mut ids = vec![u32::try_from(h.central_atom)?];
        for &a in &h.hydrogen_atoms[..h.n_hydrogens] {
            ids.push(u32::try_from(a)?);
        }
        constraints.push(ConstraintGroup { atoms: ids });
    }
    // Bond and constraint identities are also part of the pre-HMR join.
    fn bonds(v: &Value) -> Result<BTreeSet<(u64, u64)>> {
        v["bonds"]
            .as_array()
            .context("bonds absent")?
            .iter()
            .map(|b| {
                let a = b
                    .get("i")
                    .or_else(|| b.get("atom1"))
                    .and_then(Value::as_u64)
                    .context("bond endpoint absent")?;
                let z = b
                    .get("j")
                    .or_else(|| b.get("atom2"))
                    .and_then(Value::as_u64)
                    .context("bond endpoint absent")?;
                Ok((a.min(z), a.max(z)))
            })
            .collect()
    }
    ensure!(
        bonds(&target)? == bonds(&authority)?,
        "pre-HMR bond population differs"
    );
    for field in ["central_atom", "hydrogen_atoms", "n_hydrogens"] {
        let keys = |v: &Value| -> Result<Vec<Value>> {
            Ok(v["h_clusters"]
                .as_array()
                .context("constraints absent")?
                .iter()
                .map(|h| h[field].clone())
                .collect())
        };
        ensure!(
            keys(&target)? == keys(&authority)?,
            "pre-HMR constraint identity differs at {field}"
        );
    }
    let elements = topology
        .elements
        .iter()
        .map(|s| match s.as_str() {
            "H" => Ok(0),
            "C" => Ok(1),
            "N" => Ok(2),
            "O" => Ok(3),
            "P" => Ok(4),
            "S" => Ok(5),
            "F" | "Cl" => Ok(6),
            _ => bail!("element {s} has no explicitly selected xTB class"),
        })
        .collect::<Result<Vec<_>>>()?;
    let atom_map_sha256 = hash(&serde_json::to_vec(
        &json!({"elements":topology.elements,"names":topology.atom_names,"residue_ids":topology.residue_ids,"chain_ids":topology.chain_ids,"constraints":constraints.iter().map(|g|g.atoms.clone()).collect::<Vec<_>>()}),
    )?);
    c.unit_bridge_source.read()?;
    let manifest: Value = serde_json::from_slice(&c.modules.trust_manifest.read()?)?;
    match c.modules.trust_class {
        TrustClass::Signed => ensure!(
            prismcode::cuda_manifest::manifest_is_signed(&manifest)
                && prismcode::cuda_manifest::trust_manifest_sidecar_signature_verified(
                    &c.modules.trust_manifest.path
                ),
            "signed trust manifest authentication failed"
        ),
        TrustClass::SourceOwnedDevHashBound => ensure!(
            prismcode::cuda_manifest::manifest_is_dev_hash_bound(&manifest),
            "manifest is not an explicitly selected source-owned development hash binding"
        ),
    }
    prismcode::cuda_manifest::validate_existing_governed_ptx_evidence(&manifest)
        .map_err(|e| anyhow!(e.to_string()))?;
    let canonical_ptx = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/ptx")
        .canonicalize()?;
    ensure!(
        c.modules.ptx_directory.canonicalize()? == canonical_ptx,
        "module directory differs from the source-owned governed build output"
    );
    let module = |name: &str, expected: &str| -> Result<Vec<u8>> {
        ensure!(
            manifest["governed_ptx_membership"]["per_kernel_sha256"][name].as_str()
                == Some(expected),
            "governed member pin differs: {name}"
        );
        PinnedFile {
            path: canonical_ptx.join(name),
            sha256: expected.to_string(),
        }
        .read()
    };
    let biprism_ptx = module("nhs_biprism_native.ptx", &c.modules.biprism_sha256)?;
    let fused_ptx = module("nhs_amber_fused.ptx", &c.modules.fused_sha256)?;
    let nma_text = c
        .leaves
        .iter()
        .map(|l| l.nma.as_ref().map(|n| n.modes.read()).transpose())
        .collect::<Result<Vec<_>>>()?
        .try_into()
        .map_err(|_| anyhow!("NMA leaf count"))?;
    let asc_policies = c
        .leaves
        .iter()
        .map(|l| match &l.asc {
            AscConfiguration::SelectedSupport { policy, .. } => {
                let p = crate::asc_support::AscSelectedSupportPolicy::from_bytes(
                    &policy.read()?,
                    &policy.sha256,
                )?;
                p.validate_and_expand(&topology, &c.topology.sha256)?;
                Ok(Some(p))
            }
            _ => Ok(None),
        })
        .collect::<Result<Vec<_>>>()?
        .try_into()
        .map_err(|_| anyhow!("ASC leaf count"))?;
    Ok(ResolvedInputs {
        topology,
        harmonic_source_binding,
        true_inverse_masses,
        constraints,
        elements,
        atom_map_sha256,
        manifest,
        biprism_ptx,
        fused_ptx,
        nma_text,
        asc_policies,
    })
}

fn prepare_runtime(
    c: &BiprismRunConfig,
    r: &ResolvedInputs,
) -> Result<(
    BiprismCollectiveRuntime,
    [Arc<Mutex<NhsAmberFusedEngine>>; 8],
    Vec<Option<Arc<crate::governor_device::GovernorDevice>>>,
    Arc<CudaContext>,
)> {
    // Explicitly selected governed directory is the native loader's strict path.
    // A pre-existing QNANL packet requires a separate pinned packet configuration;
    // it cannot be ignored or silently replaced by this source-owned image route.
    ensure!(std::env::var_os("PRISMOS_QNANL_FORCE_KERNEL_PACKET").is_none(),"configured QNANL force packet is a different loading authority; select its hash-bound route explicitly before this command");
    if let Some(old) = std::env::var_os("PRISM4D_PTX_DIR") {
        ensure!(
            PathBuf::from(old).canonicalize()? == c.modules.ptx_directory.canonicalize()?,
            "PRISM4D_PTX_DIR conflicts with the declared governed image directory"
        );
    }
    std::env::set_var("PRISM4D_PTX_DIR", &c.modules.ptx_directory);
    let context = CudaContext::new(c.device_ordinal)?;
    let stream = context.new_stream()?;
    let module = context.load_module(Ptx::from_src(String::from_utf8(r.biprism_ptx.clone())?))?;
    let fused = context.load_module(Ptx::from_src(String::from_utf8(r.fused_ptx.clone())?))?;
    let identity = GovernedModuleIdentity {
        member: "nhs_biprism_native.ptx".into(),
        image_sha256: c.modules.biprism_sha256.clone(),
    };
    let functions = unsafe {
        serr(BiprismV10Functions::from_governed_module(
            module.clone(),
            context.clone(),
            identity.clone(),
        ))?
    };
    let collective_functions = unsafe {
        serr(BiprismCollectiveFunctions::from_governed_module(
            module,
            context.clone(),
            identity,
        ))?
    };
    let observation_functions = unsafe {
        serr(NativeObservationFunctions::from_governed_module(
            fused,
            context.clone(),
        ))?
    };
    let mut owners = Vec::new();
    let mut adapters = Vec::new();
    let mut selected = Vec::new();
    let mut governors = Vec::new();
    let mut selections = Vec::new();
    let mut ownership = Vec::new();
    for leaf in 0..LEAVES {
        let l = &c.leaves[leaf];
        let mut engine = NhsAmberFusedEngine::new_on_stream(
            context.clone(),
            stream.clone(),
            &r.topology,
            c.grid_dimension as usize,
            c.grid_spacing_angstrom,
        )?;
        engine.set_fused_inner_steps(1);
        engine.set_dt(c.numerical.dt_ps as f64);
        engine.set_adaptive_dt(c.director_adaptive_dt);
        engine.set_spike_stream_id(l.stream_id.into());
        engine.set_cryo_uv_protocol(l.protocol.clone())?;
        engine.set_uv_enabled(l.uv_enabled);
        // Initialization reads the selected protocol's start temperature and
        // native serialized HMR masses. The flexible AU core later uses the
        // separately verified physical masses; no velocity rescaling is hidden.
        engine.reset_for_replica(l.seed)?;
        let initial_positions = engine.get_positions()?;
        let initial_velocities = engine.get_velocities()?;
        ensure!(initial_positions.iter().chain(&initial_velocities).all(|x| x.is_finite()),
            "nonfinite native initialization leaf{leaf}");
        retain(&c.output_root.join(format!("leaf{leaf:02}.initial.positions.angstrom.f32le")),
            &initial_positions.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())?;
        retain(&c.output_root.join(format!("leaf{leaf:02}.initial.velocities.angstrom_per_ps.f32le")),
            &initial_velocities.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())?;
        retain_json(&c.output_root.join(format!("leaf{leaf:02}.initialization.json")),
            &json!({"source":"NhsAmberFusedEngine::reset_for_replica","replica_seed":l.seed,
                "start_temperature_kelvin":l.protocol.start_temp,"initialization_mass_authority":"serialized_native_HMR_masses",
                "velocity_rule":"native seeded Maxwell Gaussian, engine Angstrom/ps units, native HMR center-of-mass removal",
                "coordinate_rule":"native seeded thermal position jitter using serialized HMR masses",
                "flexible_core_propagation_mass_authority":"separately verified physical masses after AU conversion",
                "physical_mass_Maxwell_initialization_claimed":false,"velocity_rescaling_applied":false}))?;
        // The actual CA intervention is retained and has complete work chronology.
        engine.bind_ca_intervention_journal(c.full_step_budget, c.chunk_steps)?;
        engine.bind_conservative_constraint_journal(c.full_step_budget, c.chunk_steps)?;
        if let Some(n) = &l.nma {
            ensure!(
                hash(r.nma_text[leaf].as_ref().unwrap()) == n.modes.sha256,
                "retained NMA bytes differ"
            );
            n.modes.read()?;
            engine.load_nma_modes(n.modes.path.to_str().context("NMA path is not UTF8")?)?;
            n.modes.read()?;
            engine.set_nma_amplification(n.amplification)?;
            engine.set_nma_scan_fraction(n.scan_fraction)?;
            engine.bind_nma_work_journal(c.full_step_budget, c.chunk_steps)?;
        }
        match &l.asc {
            AscConfiguration::Disabled => {}
            AscConfiguration::SelectedSupport { alpha, .. } => {
                engine.set_asc_device(crate::asc_device::AscDevice::new_selected_support(
                    stream.clone(),
                    l.stream_id.into(),
                    &r.topology,
                    &c.topology.sha256,
                    r.asc_policies[leaf].as_ref().unwrap(),
                    *alpha,
                )?)?
            }
            AscConfiguration::ExplicitLegacyWholeSystem { alpha } => {
                engine.set_asc_device(crate::asc_device::AscDevice::new_legacy_whole_system(
                    stream.clone(),
                    l.stream_id.into(),
                    r.topology.n_atoms,
                    &c.topology.sha256,
                    *alpha,
                )?)?
            }
        }
        engine.rebuild_neighbor_lists()?;
        let (neighbors, counts) = engine.download_native_neighbor_population()?;
        ensure!(
            neighbors.len() == r.topology.n_atoms * NEIGHBOR_STRIDE
                && counts.len() == r.topology.n_atoms,
            "native neighbor allocation shape changed"
        );
        let selection = serr(select_native_population(
            &neighbors,
            &counts,
            r.topology.n_atoms as u32,
            &r.constraints,
            l.center_atom,
        ))?;
        selections.push(json!({"leaf_id":leaf,"stream_id":l.stream_id,"center":l.center_atom,"selected_atom_ids":selection.selected_atom_ids.to_vec(),"native_neighbor_stride":NEIGHBOR_STRIDE,"counts_sha256":hash(&counts.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>()),"neighbors_sha256":hash(&neighbors.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>()),"witness":format!("{:?}",selection.witness),"reachable_components":selection.reachable_components}));
        let owner = Arc::new(Mutex::new(engine));
        let mut adapter = BiprismPhysicalLeafAdapter::bind(
            owner.clone(),
            &selection.selected_atom_ids[..CORE],
            leaf as u32,
            c.numerical.constraint_policy,
        )?;
        let p = adapter.pointers()?;
        let governor = match c.governor {
            GovernorConfiguration::CollectiveSelectedMeasurement|GovernorConfiguration::ArmedAdaptive => {
                let mut policy = crate::gearbox::governor_default_policy()
                    .map_err(|e| anyhow!(e.to_string()))?;
                policy.armed = 0;
                policy.chunk_span_steps = 1;
                // ArmedAdaptive arms the shared bounded selector, not independent leaf
                // transitions. Native measurement keeps all energy/constraint guards
                // with dt_status2/gear0 and never overwrites the chosen common dt.
                policy.gear_dt_fs = [c.numerical.dt_ps * 1000.0; 4];
                let g = crate::governor_device::GovernorDevice::new(
                    stream.clone(),
                    &policy,
                    l.stream_id.into(),
                    adapter.ownership_plan().retained_constraint_count(),
                )
                .map_err(|rc| anyhow!("retained physical governor create rc={rc}"))?;
                let bindings = crate::governor_device::PhysicalGuardBindings {
                    abi: 1,
                    byte_size: 0,
                    source: 0,
                    n_atoms: r.topology.n_atoms as u32,
                    generation: 0,
                    governor: 0,
                    forces: p.forces,
                    potential: p.potential_energy,
                    velocities: p.velocities,
                    masses: p.masses,
                    dt: p.protocol_dt,
                    step_work: p.external_work_window,
                };
                unsafe {
                    g.bind_physical_guard(bindings)
                        .map_err(|rc| anyhow!("physical guard binding rc={rc}"))?;
                    g.bind_physical_time(p.protocol_current_step, c.full_step_budget as u64)
                        .map_err(|rc| anyhow!("physical clock binding rc={rc}"))?;
                }
                adapter.bind_retained_governor(g.clone())?;
                Some(g)
            }
            GovernorConfiguration::ObserverClock => {
                owner
                    .lock()
                    .map_err(|_| anyhow!("native engine lock poisoned"))?
                    .bind_observer_physical_clock(c.full_step_budget as u64)?;
                None
            }
            GovernorConfiguration::FixedStepMeasurement => {
                bail!("unselected physical governor profile was not rejected")
            }
        };
        ownership.push(json!({"ownership":adapter.ownership_plan(),"actual_configured_execution":adapter.feature_execution_map()?}));
        selected.push(selection);
        owners.push(owner);
        adapters.push(adapter);
        governors.push(governor);
    }
    // Verify source-owned images again after all file-based native constructors.
    c.modules.trust_manifest.read()?;
    prismcode::cuda_manifest::validate_existing_governed_ptx_evidence(&r.manifest)
        .map_err(|e| anyhow!(e.to_string()))?;
    retain_json(
        &c.output_root.join("mesh_selection.json"),
        &json!({"selection":selections,"motion_ownership":ownership}),
    )?;
    let adapters: [BiprismPhysicalLeafAdapter; 8] = adapters
        .try_into()
        .map_err(|_| anyhow!("adapter leaf extent"))?;
    let owners: [Arc<Mutex<NhsAmberFusedEngine>>; 8] = owners
        .try_into()
        .map_err(|_| anyhow!("native engine extent"))?;
    let physical_identity = BiprismIdentity {
        topology_sha256: c.topology.sha256.clone(),
        atom_map_sha256: r.atom_map_sha256.clone(),
        unit_bridge_sha256: c.unit_bridge_source.sha256.clone(),
        physical_mass_sha256: c.masses.mapping.sha256.clone(),
    };
    let mut borrowed = Vec::new();
    let mut mass_selection = Vec::new();
    for leaf in 0..8 {
        let p = adapters[leaf].pointers()?;
        let bytes = r.topology.n_atoms * 3 * 4;
        let owner: Arc<dyn std::any::Any> = owners[leaf].clone();
        borrowed.push(unsafe {
            serr(BorrowedPhysicalLeaf::new(
                owner,
                stream.clone(),
                r.topology.n_atoms as u32,
                physical_identity.clone(),
                c.leaves[leaf].stream_id,
                DeviceRegion {
                    address: p.positions,
                    bytes,
                },
                DeviceRegion {
                    address: p.velocities,
                    bytes,
                },
                DeviceRegion {
                    address: p.forces,
                    bytes,
                },
                p.coordinate_owner,
            ))?
        });
        let hmr_inverse = r
            .topology
            .masses
            .iter()
            .map(|&m| ((1.0 / f64::from(m)) * INV_DALTON_TO_INV_ELECTRON) as f32)
            .collect();
        mass_selection.push(LeafMassAndSelection {
            inverse_masses_au: hmr_inverse,
            true_inverse_masses_au: r.true_inverse_masses.clone(),
            core_shake_mask: selected[leaf].core_mask.clone(),
            selected_atom_ids: selected[leaf].selected_atom_ids,
        });
    }
    let geometry = SdstGridGeometry {
        dimension: c.grid_dimension,
        spacing_a: c.grid_spacing_angstrom,
        origin_a: adapters[0].pointers()?.grid_origin,
    };
    let residues = match &r.topology.atom_analysis_residue_ids {
        Some(ids) => ids.clone(),
        None => r
            .topology
            .residue_ids
            .iter()
            .map(|&v| i32::try_from(v))
            .collect::<std::result::Result<Vec<_>, _>>()?,
    };
    let state = serr(BiprismV10DeviceState::allocate(
        stream,
        functions,
        borrowed
            .try_into()
            .map_err(|_| anyhow!("borrowed extent"))?,
        mass_selection
            .try_into()
            .map_err(|_| anyhow!("mass extent"))?,
        &r.topology.charges,
        &r.elements,
        &residues,
        &r.constraints,
        c.queue_capacity,
        format!("{}:native_biprism_queue", c.run_id),
    ))?;
    let dt_au = (f64::from(c.numerical.dt_ps) * PS_TO_ATOMIC_TIME) as f32;
    let mode = if c.numerical.constraint_policy == BiprismConstraintPolicy::FlexibleCore {
        1
    } else {
        0
    } | if c.numerical.gonzalez_exact_energy {
        2
    } else {
        0
    };
    let config = BiprismCollectiveConfig {
        step: V10Step {
            dt_outer_au: dt_au,
            kappa_foliated: c.numerical.kappa_foliated,
            integration_mode: mode,
        },
        dt_feedback_au: dt_au,
        work_window_capacity: c.chunk_steps,
        physical_time_generation:c.physical_time_generation,
        adaptive_dt_bounds:c.adaptive_dt_bounds,
        lif_decay: c.numerical.lif_decay,
        lif_threshold: c.numerical.lif_threshold,
        lif_kick: c.numerical.lif_kick,
        geometry,
        acceptance: V10AcceptanceLimits {
            picard_residual: c.numerical.picard_residual,
            dec_residual: c.numerical.dec_residual,
            absolute_energy_drift: c.numerical.absolute_energy_drift,
            unitarity_defect: c.numerical.unitarity_defect,
        },
    };
    let runtime = unsafe {
        serr(BiprismCollectiveRuntime::new(
            adapters,
            state,
            collective_functions,
            observation_functions,
            config,
            0,
        ))?
    };
    Ok((runtime, owners, governors, context))
}

fn status_json(s: &BiprismCollectiveStatus) -> Value {
    json!({"control":format!("{:?}",s.control),"last_completed_step":s.control.last_completed_step,"successful_bodies":s.control.successful_bodies,"sticky_fault":s.control.sticky_fault,"next_scheduled_step":s.next_scheduled_step,"remaining_steps":s.remaining_steps,"raw_queue_count":s.raw_queue_count,"convergence":format!("{:?}",s.convergence),"unit_bridge":format!("{:?}",s.unit_bridge),"carryover":format!("{:?}",s.carryover),"native_observation":format!("{:?}",s.observation),"asc_mixed_work":format!("{:?}",s.asc),"lif_actual_work":format!("{:?}",s.lif_work),"phase2_operator_work":s.phase2_work,"mixed_energy":s.mixed_energy,"shared_time":s.shared_time,"electronic_exchange":format!("{:?}",s.electronic_exchange)})
}
fn limits(c: &BiprismRunConfig, r: &ResolvedInputs) -> BiprismDecodeLimits {
    BiprismDecodeLimits {
        num_atoms: r.topology.n_atoms as u32,
        num_residues: r
            .topology
            .n_analysis_residues
            .unwrap_or(r.topology.n_residues) as u32,
        total_voxels: c.grid_dimension.pow(3),
        expected_step: None,
    }
}

fn classify(
    c: &BiprismRunConfig,
    decoded: &DecodedVoronoiSpikeQueue,
    backend: &mut GpuSpatialHashBackend,
    consumers: &BiprismRunConsumers,
) -> Result<Vec<SpikeClassification>> {
    let mut classes: Vec<Option<SpikeClassification>> = vec![None; decoded.native_events.len()];
    for leaf in 0..8 {
        let indices = decoded
            .ledger
            .iter()
            .enumerate()
            .filter(|(_, r)| r.event.leaf_id as usize == leaf)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        if indices.is_empty() {
            continue;
        }
        let events = indices
            .iter()
            .map(|&i| decoded.native_events[i])
            .collect::<Vec<_>>();
        let positions = events.iter().flat_map(|e| e.position).collect::<Vec<_>>();
        let cluster = backend.cluster(&positions, c.clustering_epsilon_angstrom)?;
        ensure!(
            cluster.cluster_ids.len() == events.len(),
            "native clustering changed row population"
        );
        let sites = (consumers.build_sites)(&events, &cluster)?;
        let mut intensity = events.iter().map(|e| e.intensity).collect::<Vec<_>>();
        intensity.sort_by(f32::total_cmp);
        let mut local = Vec::new();
        let p = &c.leaves[leaf].protocol;
        for (i, event) in events.iter().enumerate() {
            let site_id = cluster.cluster_ids[i];
            let nearest = sites
                .iter()
                .map(|s| {
                    let center = s.geometric_voxel_mass_centroid();
                    let d = (0..3)
                        .map(|a| (event.position[a] - center[a]).powi(2))
                        .sum::<f32>()
                        .sqrt();
                    (s.cluster_id, d)
                })
                .min_by(|a, b| a.1.total_cmp(&b.1));
            let rank = intensity.partition_point(|x| *x <= event.intensity);
            let intensity_percentile = ((rank as u64 * 100) / (events.len() as u64)).min(100) as u8;
            local.push(SpikeClassification {
                stream_id: c.leaves[leaf].stream_id,
                group_id: 0,
                chunk_idx: chunk_idx_for_timestep(
                    event.timestep,
                    c.chunk_steps as i32,
                    c.full_step_budget.div_ceil(c.chunk_steps) as i32,
                ),
                ccns_phase: CcnsPhaseCoding::resolve().encode(
                    event.timestep,
                    p.cold_hold_steps,
                    p.ramp_steps,
                    p.warm_hold_steps,
                    p.ramp_down_steps,
                ),
                site_id,
                nearest_site_id: nearest.map_or(-1, |x| x.0),
                nearest_site_dist: nearest.map_or(f32::MAX, |x| x.1),
                background_class: 0,
                burial_score: f32::from(event.n_residues as u8).min(8.0) / 8.0,
                intensity_percentile,
            });
        }
        let (p10, p50) = compute_background_percentiles(&local);
        for (i, mut class) in local.into_iter().enumerate() {
            class.background_class = classify_background(
                class.site_id,
                class.nearest_site_dist,
                class.intensity_percentile,
                class.burial_score,
                p10,
                p50,
            );
            classes[indices[i]] = Some(class);
        }
    }
    classes
        .into_iter()
        .map(|c| c.context("classification omitted a raw event leaf"))
        .collect()
}

fn sites_json(sites: &[ClusteredBindingSite]) -> Value {
    json!(sites.iter().map(|s|json!({"cluster_id":s.cluster_id,"spike_count":s.spike_count,"spike_indices":s.spike_indices,"geometric_voxel_mass_centroid_angstrom":s.geometric_voxel_mass_centroid(),"avg_intensity":s.avg_intensity,"estimated_volume_angstrom3":s.estimated_volume,"bounding_box_angstrom":s.bounding_box,"quality_score":s.quality_score,"druggability":format!("{:?}",s.druggability),"classification":format!("{:?}",s.classification),"lining_residues":format!("{:?}",s.lining_residues)})).collect::<Vec<_>>())
}

fn retain_work(
    c: &BiprismRunConfig,
    owners: &[Arc<Mutex<NhsAmberFusedEngine>>; 8],
    governors: &[Option<Arc<crate::governor_device::GovernorDevice>>],
    begin: u32,
    end: u32,
    dir: &Path,
) -> Result<()> {
    for leaf in 0..8 {
        let mut e = owners[leaf]
            .lock()
            .map_err(|_| anyhow!("native engine lock poisoned"))?;
        let prefix = format!("leaf{leaf:02}");
        let (ca_index, ca_values) = e.read_ca_intervention_atom_window(begin as u64, end as u64)?;
        retain(
            &dir.join(format!("{prefix}.ca_work.f64le")),
            &ca_values
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        retain_json(
            &dir.join(format!("{prefix}.ca_work.index.json")),
            &json!(ca_index),
        )?;
        let (constraint_index, constraint_rows) =
            e.read_conservative_constraint_window(begin as u64, end as u64)?;
        retain(
            &dir.join(format!("{prefix}.conservative_constraint.f64le")),
            &constraint_rows.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
        )?;
        retain_json(
            &dir.join(format!("{prefix}.conservative_constraint.index.json")),
            &json!({"source_manifest":e.conservative_constraint_manifest(),
                "records":constraint_index,
                "propagation_scope":"actual retained native constraint phases under the bound bath/core ownership plan"}),
        )?;
        e.commit_conservative_constraint_window(begin as u64, end as u64)?;
        if c.leaves[leaf].nma.is_some() {
            let (index, values) = e.read_nma_work_window(begin as u64, end as u64)?;
            retain(
                &dir.join(format!("{prefix}.nma_work.f64le")),
                &values
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            retain_json(
                &dir.join(format!("{prefix}.nma_work.index.json")),
                &json!(index),
            )?;
            e.commit_nma_work_window(begin as u64, end as u64)?;
        }
        if let Some(g) = &governors[leaf] {
            let time = g
                .physical_time_records(begin as u64, end as u64)
                .map_err(|rc| anyhow!("physical time readback leaf{leaf} rc={rc}"))?;
            let energy = g
                .physical_energy_balance_records(begin as u64, end as u64)
                .map_err(|rc| anyhow!("physical energy readback leaf{leaf} rc={rc}"))?;
            retain_json(
                &dir.join(format!("{prefix}.physical_time.json")),
                &serde_json::to_value(&time)?,
            )?;
            retain_json(
                &dir.join(format!("{prefix}.energy_balance.json")),
                &serde_json::to_value(&energy)?,
            )?;
            let (governor, constraints) = g
                .snapshot()
                .map_err(|rc| anyhow!("governor readback rc={rc}"))?;
            retain_json(
                &dir.join(format!("{prefix}.governor.json")),
                &json!({"governor":format!("{governor:?}"),"constraints":format!("{constraints:?}")}),
            )?;
        } else {
            let time = e.observer_physical_time_records(begin as u64, end as u64)?;
            retain_json(
                &dir.join(format!("{prefix}.physical_time.json")),
                &serde_json::to_value(&time)?,
            )?;
        }
        if let Some(a) = e.asc_device() {
            let mut receipt=a.receipt().map_err(|rc| anyhow!("ASC receipt rc={rc}"))?;
            receipt["measurement_scope"]=json!("Native force compositions on selected atoms; actual two pre-constraint half-kicks on selected bath atoms only. Core held-source impulse/work is the separate per-body ASC journal. Core ForceNp1 is composed but unconsumed by a native half-kick.");
            receipt["core_work_journal"]=json!({"index":"asc_mixed_work.index.json","ledgers":"asc_mixed_work.192b","core_rows":"asc_mixed_core.208b"});
            retain_json(&dir.join(format!("{prefix}.asc.json")),&receipt)?;
        }
        e.commit_ca_intervention_atom_window(begin as u64, end as u64)?;
    }
    Ok(())
}

fn postprocess(
    c: &BiprismRunConfig,
    r: &ResolvedInputs,
    geometry: SdstGridGeometry,
    chunks: &[PathBuf],
    backend: &mut GpuSpatialHashBackend,
    consumers: &BiprismRunConsumers,
) -> Result<Value> {
    let mut outcomes = Vec::new();
    let limits = limits(c, r);
    for leaf in 0..8 {
        let path = c.output_root.join(format!("leaf{leaf:02}"));
        fs::create_dir(&path)?;
        let mut raw = Vec::new();
        let mut observation_raw = Vec::new();
        for chunk in chunks {
            let input = fs::read(chunk.join("events.v10s96"))?;
            let obs = fs::read(chunk.join("native_observations.64b"))?;
            ensure!(
                input.len() % 96 == 0 && obs.len() / 64 == input.len() / 96 && obs.len() % 64 == 0,
                "retained raw row population mismatch"
            );
            for (event, observation) in input.chunks_exact(96).zip(obs.chunks_exact(64)) {
                let id = u16::from_le_bytes(event[64..66].try_into().unwrap());
                if usize::from(id) == leaf {
                    raw.try_reserve(96)?;
                    observation_raw.try_reserve(64)?;
                    raw.extend_from_slice(event);
                    observation_raw.extend_from_slice(observation);
                }
            }
        }
        let count = u32::try_from(raw.len() / 96)?;
        let observations = serr(decode_native_lif_observation_queue(
            &observation_raw,
            count,
            count,
            limits,
        ))?;
        let decoded = serr(decode_spike_queue_for_arrow_and_pathb_in_frame(
            &raw,
            count,
            count,
            limits,
            &observations,
            geometry,
            BiprismEventFrame::AtomicBohr,
        ))?;
        let classes = classify(c, &decoded, backend, consumers)?;
        let mut export = serr(decode_voronoi_spike_export_in_frame(
            &raw,
            count,
            count,
            limits,
            &observations,
            geometry,
            &classes,
            c.leaves[leaf].seed,
            &c.leaves.each_ref().map(|l| l.stream_id),
            BiprismEventFrame::AtomicBohr,
        ))?;
        serr(export.bind_replica_seeds(c.leaves.each_ref().map(|l| l.seed)))?;
        serr(export.bind_queue_export(&format!("{}:leaf{leaf}:full_population", c.run_id), 0))?;
        let (native, ledger) = serr(export.to_paired_ipc())?;
        retain(&path.join("spike_events.arrow"), &native)?;
        retain(&path.join("voronoi_ledger.arrow"), &ledger)?;
        let all = export.decoded.native_events;
        let (selected, selection) =
            crate::sdst_bridge::select_native_percentile_spikes(all, c.source_percentile);
        retain_json(
            &path.join("source_population_selection.json"),
            &json!({"unfiltered_rows":count,"selected_rows":selected.len(),"source_percentile":c.source_percentile,"selection":selection}),
        )?;
        let positions = selected.iter().flat_map(|e| e.position).collect::<Vec<_>>();
        let cluster = backend.cluster(&positions, c.clustering_epsilon_angstrom)?;
        ensure!(cluster.cluster_ids.len() == selected.len(), "full-run clustering changed selected row population");
        let sites = (consumers.build_sites)(&selected, &cluster)?;
        retain_json(
            &path.join("binding_sites.json"),
            &json!({"source":"actual_native_gpu_spatial_hash_and_existing_site_builder","surface":"source6_voronoi_lif","sites":sites_json(&sites),"cluster_ids":cluster.cluster_ids,"cluster_count":cluster.num_clusters,"total_neighbors":cluster.total_neighbors,"gpu_time_ms":cluster.gpu_time_ms}),
        )?;
        let analysis = if selected.is_empty() {
            json!({"status":"NO_EMITTED_POPULATION","native_ingest_rows":0,"sites":[]})
        } else {
            let mut bridge = SdstBridge::new_for_spikes_with_recorded_grid(
                &r.topology,
                &c.leaves[leaf].protocol,
                &selected,
                c.wavefront_max_dt_steps,
                geometry,
            )?;
            bridge.preserve_full_tide_output();
            let ingested = bridge.ingest_all_spikes(&selected)?;
            validate_pathb_population(selected.len(), ingested, None)?;
            let analyzed = bridge.analyze(&sites)?;
            let state_path = path.join("pathb.native");
            let state = bridge.save_native_state(&state_path)?;
            retain_complete_pathb_state(&state_path, &state, selected.len())?;
            json!({"status":"NATIVE_ANALYSIS_EXECUTED","native_ingest_rows":ingested,"analysis":analyzed,"native_state":state})
        };
        retain_json(&path.join("pathb_analysis.json"), &analysis)?;
        outcomes.push(json!({"leaf_id":leaf,"stream_id":c.leaves[leaf].stream_id,"replica_seed":c.leaves[leaf].seed,"unfiltered_emission_groups":count,"selected_emission_groups":selected.len(),"site_count":sites.len(),"pathb_status":analysis["status"]}));
    }
    Ok(json!(outcomes))
}

fn artifact_receipts(root: &Path) -> Result<Vec<Value>> {
    let mut result = Vec::new();
    fn visit(root: &Path, current: &Path, out: &mut Vec<Value>) -> Result<()> {
        let mut entries = fs::read_dir(current)?.collect::<std::io::Result<Vec<_>>>()?;
        entries.sort_by_key(|e| e.path());
        for entry in entries {
            let path = entry.path();
            let kind = entry.file_type()?;
            ensure!(!kind.is_symlink(), "consumer output contains a symlink");
            if kind.is_dir() {
                visit(root, &path, out)?;
            } else if kind.is_file() {
                let bytes = fs::read(&path)?;
                out.push(json!({"path":path.strip_prefix(root)?,"bytes":bytes.len(),"sha256":hash(&bytes)}));
            }
        }
        Ok(())
    }
    visit(root, root, &mut result)?;
    Ok(result)
}

/// Bind the native source clocks to the independently accepted shared journal
/// before either becomes a durable event-time sidecar.
fn validate_native_shared_clock_rows(
    native: &[Vec<[u64; 3]>; 8], shared: &[[u64; 3]], begin: u32, end: u32,
) -> Result<()> {
    ensure!(end > begin && shared.len() == (end - begin) as usize,
        "shared accepted clock population differs from the completed window");
    for (i, row) in shared.iter().enumerate() {
        ensure!(row[0] == u64::from(begin) + i as u64 + 1
            && f64::from_bits(row[1]).is_finite() && f64::from_bits(row[1]) > 0.0
            && f64::from_bits(row[2]).is_finite() && f64::from_bits(row[2]) > 0.0,
            "invalid accepted shared clock row {i}");
    }
    for (leaf, rows) in native.iter().enumerate() {
        ensure!(rows.len() == shared.len(), "native/shared clock coverage differs for leaf{leaf}");
        for (i, (actual, accepted)) in rows.iter().zip(shared).enumerate() {
            ensure!(actual == accepted, "native/shared clock identity differs at leaf{leaf} row{i}");
        }
    }
    Ok(())
}

fn validate_pathb_population(expected: usize, ingested: u32, saved: Option<u64>) -> Result<()> {
    let expected = u32::try_from(expected).context("PathB selected population exceeds its native count ABI")?;
    ensure!(ingested == expected, "PathB ingested {ingested} of {expected} selected source events");
    if let Some(saved) = saved {
        ensure!(saved == u64::from(expected), "PathB serialized population differs from selected source events");
    }
    Ok(())
}

fn retain_complete_pathb_state(path: &Path, state: &Value, expected: usize) -> Result<()> {
    let count = state["event_count"].as_u64().context("PathB state lacks its actual event count")?;
    validate_pathb_population(expected, u32::try_from(count)?, Some(count))?;
    let offset = state["event_count_offset"].as_u64().context("PathB state lacks count offset")?;
    let config_size = state["config_size_bytes"].as_u64().context("PathB state lacks configuration extent")?;
    let buffer = state["event_buffer_offset"].as_u64().context("PathB state lacks event offset")?;
    let stride = state["event_stride_bytes"].as_u64().context("PathB state lacks event extent")?;
    ensure!(cfg!(target_endian = "little") && state["endianness"] == "little"
        && offset == config_size.checked_add(8).context("PathB offset overflow")?
        && buffer == offset.checked_add(4).context("PathB offset overflow")? && stride > 0,
        "PathB serialized layout differs from its native contract");
    let minimum = buffer.checked_add(count.checked_mul(stride).context("PathB event extent overflow")?)
        .context("PathB serialized extent overflow")?;
    let metadata = fs::symlink_metadata(path)?;
    ensure!(metadata.is_file() && !metadata.file_type().is_symlink() && metadata.len() >= minimum,
        "PathB native state is absent, truncated or not a regular file");
    let mut file = File::open(path)?;
    let mut header = [0u8; 8]; file.read_exact(&mut header)?;
    ensure!(u32::from_le_bytes(header[..4].try_into().unwrap()) == 0x54534453
        && u32::from_le_bytes(header[4..].try_into().unwrap()) == 1, "PathB native state header differs");
    file.seek(SeekFrom::Start(offset))?;
    let mut bytes = [0u8; 4]; file.read_exact(&mut bytes)?;
    validate_pathb_population(expected, u32::from_le_bytes(bytes), Some(count))?;
    file.sync_all()?;
    File::open(path.parent().context("PathB state lacks a parent directory")?)?.sync_all()?;
    Ok(())
}

fn validate_completed_output_extent(requested: u32, completed: u32, chunk_steps: u32,
    retained_chunks: usize, graph_launches: u64, grid_step: u32, analysis_leaves: usize) -> Result<()> {
    ensure!(requested > 0 && chunk_steps > 0 && completed == requested && grid_step == requested,
        "terminal output does not cover the requested accepted-step budget");
    let expected_chunks = u64::from(requested).div_ceil(u64::from(chunk_steps));
    ensure!(retained_chunks as u64 == expected_chunks && graph_launches == expected_chunks
        && analysis_leaves == LEAVES, "terminal output is missing an accepted chunk or physical leaf");
    Ok(())
}

fn require_complete_terminal_outputs(result: Result<Value>, failures: &[String]) -> Result<Value> {
    if failures.is_empty() { return result; }
    let detail = format!("required terminal output readback failed: {}", failures.join("; "));
    match result { Ok(_) => Err(anyhow!(detail)), Err(error) => Err(error.context(detail)) }
}

fn retain_terminal_failure(root: &Path, filename: &str, label: &str, error: impl std::fmt::Display,
    failures: &mut Vec<String>) -> Result<()> {
    let error = error.to_string();
    failures.push(format!("{label}: {error}"));
    retain_json(&root.join(filename), &json!({"error":error,"accepted":false}))
}
fn native_clock_words(
    owners: &[Arc<Mutex<NhsAmberFusedEngine>>; 8],
    governors: &[Option<Arc<crate::governor_device::GovernorDevice>>],
    begin: u32,
    end: u32,
) -> Result<[Vec<[u64; 3]>; 8]> {
    let mut result = std::array::from_fn(|_| Vec::new());
    for leaf in 0..8 {
        result[leaf] = if let Some(g) = &governors[leaf] {
            g.physical_time_records(u64::from(begin), u64::from(end))
                .map_err(|rc| anyhow!("native clock leaf{leaf} rc={rc}"))?
                .iter()
                .map(|r| {
                    [
                        r.step,
                        r.applied_dt_ps.to_bits(),
                        r.cumulative_simulated_time_ps.to_bits(),
                    ]
                })
                .collect()
        } else {
            owners[leaf]
                .lock()
                .map_err(|_| anyhow!("native clock owner poisoned"))?
                .observer_physical_time_records(u64::from(begin), u64::from(end))?
                .iter()
                .map(|r| {
                    [
                        r.step,
                        r.applied_dt_ps.to_bits(),
                        r.cumulative_simulated_time_ps.to_bits(),
                    ]
                })
                .collect()
        };
    }
    Ok(result)
}
fn retain_coordinate_snapshot(root:&Path,prefix:&str,image:&crate::biprism_coordinates::BiprismCoordinateSnapshot)->Result<()>{
    serr(image.validate())?;
    for (name,plane) in image.planes(){retain(&root.join(format!("{prefix}.{name}.f32le")),&plane.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?;}
    retain_json(&root.join(format!("{prefix}.index.json")),&json!({"schema":"biprism.coordinate_owner.v1","identity":image.identity,
        "planes":["current_high","current_low","step_start_high","step_start_low","mesh_reference_high","mesh_reference_low"],
        "physical_value":"FP64(high)+FP64(low)","scope":"actual terminal coordinate planes; other physics/RNG/clock state remains required for complete restart"}))
}
fn retain_spinor_snapshot(root:&Path,prefix:&str,image:&[crate::biprism_v10_runtime::BiprismSpinorOwnerFfi])->Result<()>{
    ensure!(image.len()==8,"spinor owner population");
    let bytes=unsafe{core::slice::from_raw_parts(image.as_ptr().cast::<u8>(),core::mem::size_of_val(image))};
    retain(&root.join(format!("{prefix}.spinor.owners.bps1")),bytes)
}
fn retain_potential_snapshot(root:&Path,prefix:&str,image:&crate::biprism_v10_runtime::BiprismPotentialReadback)->Result<()>{
    retain(&root.join(format!("{prefix}.owners.128b")),serr(image.owner_bytes())?)?;
    retain(&root.join(format!("{prefix}.accepted.f64le")),serr(image.accepted_bytes())?)?;
    retain(&root.join(format!("{prefix}.prepared.f64le")),serr(image.prepared_bytes())?)?;
    retain(&root.join(format!("{prefix}.working.f64le")),serr(image.working_bytes())?)?;
    retain_json(&root.join(format!("{prefix}.index.json")),&image.index())
}

fn open_chronicle(c: &BiprismRunConfig) -> Result<observation_chronicle::BiprismChronicle> {
    use crate::scientific_checkpoint::CheckpointIdentity;
    use std::os::unix::ffi::OsStrExt;
    let name = std::ffi::CString::new(c.output_root.as_os_str().as_bytes())?;
    let mut stat = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    ensure!(
        unsafe { libc::statvfs(name.as_ptr(), stat.as_mut_ptr()) } == 0,
        "cannot inspect Chronicle filesystem capacity"
    );
    let stat = unsafe { stat.assume_init() };
    let free = (stat.f_bavail as u64)
        .checked_mul(stat.f_frsize as u64)
        .context("filesystem capacity overflow")?;
    let required = c
        .chronicle_max_bytes
        .checked_add(1024 * 1024 * 1024)
        .context("Chronicle resource admission overflow")?;
    ensure!(
        free >= required,
        "insufficient filesystem capacity for declared Chronicle quota and1GiB reserve"
    );
    let engine_hash = hash(&fs::read(std::env::current_exe()?)?);
    let request = hash(&serde_json::to_vec(c)?);
    let identity=CheckpointIdentity{run_id:c.run_id.clone(),topology_sha256:c.topology.sha256.clone(),engine_sha256:engine_hash.clone(),
        kernel_set_sha256:c.modules.trust_manifest.sha256.clone(),protocol_sha256:hash(&serde_json::to_vec(&c.leaves)?),execution_request_sha256:request.clone(),
        state_abi_sha256:hash(b"biprism.observation.v2:raw96_nob2_64_queue_order24_native_clock24_drain64;not_restart"),
        provider:"native_biprism_collective_source6".into(),stream_count:8};
    retain_json(
        &c.output_root.join("chronicle_admission.json"),
        &json!({"logical_quota_bytes":c.chronicle_max_bytes,"observed_free_bytes":free,
        "reserve_bytes":1024*1024*1024u64,"atomic_store":"existing scientific_checkpoint::CheckpointStore","identity":identity,
        "quota_exhaustion":"refuse source reset and stop; preserve committed prefix","scientific_restart":false}),
    )?;
    observation_chronicle::BiprismChronicle::open(
        &c.output_root.join("chronicle"),
        c.chronicle_max_bytes,
        identity,
        c.leaves.each_ref().map(|l| l.seed),
        c.leaves.each_ref().map(|l| l.stream_id),
        &request,
        &engine_hash,
    )
}

fn execute(
    c: &BiprismRunConfig,
    r: &ResolvedInputs,
    consumers: &BiprismRunConsumers,
) -> Result<Value> {
    let (mut runtime, owners, governors, context) = prepare_runtime(c, r)?;
    let geometry = runtime.config().geometry;
    // Native ASC decision producers are distinct from the v10 diagnostic ring.
    // The same actual analysis-residue map feeds native events and GASP anchors.
    let analysis_residues=match &r.topology.atom_analysis_residue_ids {Some(x)=>x.clone(),None=>r.topology.residue_ids.iter().map(|&x|i32::try_from(x)).collect::<std::result::Result<Vec<_>,_>>()?};
    let mut decision_owners=Vec::with_capacity(8);
    for (leaf,engine) in owners.iter().enumerate(){
        let asc=engine.lock().map_err(|_|anyhow!("native engine owner poisoned"))?.asc_device().cloned();
        if let Some(asc)=asc {
            let (anchors,fallback)=serr(crate::captured_pipeline::biprism_adjudication::adjudication_anchor_lut(&analysis_residues,&r.topology.ca_indices,runtime.native_pointers()[leaf].n_residues))?;
            let retained:Arc<dyn std::any::Any>=engine.clone();
            decision_owners.push(Some(unsafe{serr(crate::captured_pipeline::biprism_adjudication::BiprismAscDecisionOwner::new(runtime.stream(),asc,retained,runtime.native_pointers()[leaf].clone(),anchors,fallback,runtime.completed_step()))?}));
        }else{decision_owners.push(None);}
    }
    let mut chronicle = open_chronicle(c)?;
    let pre = || -> Result<()> {
        for (leaf, owner) in owners.iter().enumerate() {
            let e = owner.lock().map_err(|_| anyhow!("native owner poisoned"))?;
            if c.leaves[leaf].nma.is_some() {
                retain_json(
                    &c.output_root
                        .join(format!("leaf{leaf:02}.nma_population.json")),
                    &e.nma_work_manifest()?,
                )?;
            }
        }
        Ok(())
    };
    pre()?;
    let mut chunks = Vec::new();
    let mut backend = GpuSpatialHashBackend::new(context, runtime.stream())?;
    let result = (|| -> Result<Value> {
        unsafe {
            serr(runtime.prepare(c.target_tetrahedra,|readback|{
            if let Some(cap)=c.max_seed_peel_passes {
                if let Some(status)=readback.statuses().iter().find(|s|s.peel_passes>cap){
                    return Err(format!("leaf{} actual seed peel passes{} exceed configured{}",status.leaf_id,status.peel_passes,cap));
                }
            }
            let admitted=validate_mesh_readback(readback,r.topology.n_atoms as u32,&r.constraints)?;
            retain_json(&c.output_root.join("initial_electronic_population.json"),&json!({"producer":"actual Bowyer-Watson seed initializer","all_leaves_all256_vertices":[1.0,0.0,0.0,0.0],"population_z":1.0,"validated_exact_fp32_bytes":true,"configurable_spinor_initializer":false})).map_err(|e|e.to_string())?;
            retain(&c.output_root.join("prepared_tiles.bytes"),readback.tile_bytes()).map_err(|e|e.to_string())?;
            retain_json(&c.output_root.join("mesh_admission.json"),&json!({"admission":format!("{admitted:?}"),"bw_status":format!("{:?}",readback.statuses()),"actual_gpu_geometry":true})).map_err(|e|e.to_string())?;
            Ok(())
        }))?;
            let initial=runtime.initial_operator().ok_or_else(||anyhow!("actual initial operator is unavailable"))?;
            retain(&c.output_root.join("initial_operator_before_tiles.bytes"),&initial.before_tiles)?;
            retain(&c.output_root.join("initial_operator_after_tiles.bytes"),&initial.after_tiles)?;
            retain(&c.output_root.join("initial_operator.240b"),serr(initial.ledger_bytes())?)?;
            retain_json(&c.output_root.join("initial_operator.index.json"),&initial.index())?;
            retain_potential_snapshot(&c.output_root,"initial_potential",&serr(runtime.potential_snapshot())?)?;
            retain_spinor_snapshot(&c.output_root,"initial",&serr(runtime.spinor_snapshot())?)?;
            let maps=runtime.state().peer_edge_maps().ok_or_else(||anyhow!("accepted topology has no complete peer edge map"))?;
            let map_bytes:Vec<u8>=maps.iter().flatten().flat_map(|x|x.to_le_bytes()).collect();
            retain(&c.output_root.join("peer_edge_maps.u16"),&map_bytes)?;
            retain_json(&c.output_root.join("peer_edge_maps.index.json"),&json!({"leaves":8,"entries_per_leaf":1536,"next_leaf":"(leaf+1)%8","key":"ordered physical atom endpoints","total_active_population":true,"inactive_sentinel":65535,"rebuild_required":"every preparation/reseed/topology change","cold_topology_immutable_in_captured_body":true,"preparation_generation":runtime.state().peer_edge_map_generation(),"physical_atom_ids":runtime.state().selected_atom_ids().iter().map(|r|r.to_vec()).collect::<Vec<_>>(),"ordered_pair_source":"prepared_tiles.bytes"}))?;
            serr(runtime.capture_once())?;
        }
        let limit = limits(c, r);
        while runtime.completed_step() < c.full_step_budget {
            let begin = runtime.completed_step();
            let steps = (c.full_step_budget - begin).min(c.chunk_steps);
            let dir = c.output_root.join(format!("chunk_{begin:08}"));
            fs::create_dir(&dir)?;
            let outcome = unsafe { serr(runtime.run_chunk(steps))? };
            retain_json(
                &dir.join("execution.json"),
                &json!({"requested_steps":steps,"completed_before":outcome.completed_before,"accepted":outcome.accepted,"refusal":outcome.refusal,"graph_launches":outcome.graph_launches,"status":status_json(&outcome.status)}),
            )?;
            let exchange=serr(runtime.electronic_exchange())?;
            retain_potential_snapshot(&dir,"potential",&serr(runtime.potential_snapshot())?)?;
            retain_spinor_snapshot(&dir,"physical",&serr(runtime.spinor_snapshot())?)?;
            retain(&dir.join("electronic_exchange.17792b"),&exchange.raw_bytes)?;
            retain_json(&dir.join("electronic_exchange.index.json"),&json!({"leaf_count":8,"record_bytes":17792,"footer_offset":17664,"scope":"derived final-substep scratch; not a whole-window work journal or restart","footers":format!("{:?}",exchange.footers()),"accepted_outer_window":outcome.accepted}))?;
            let asc_work=serr(runtime.asc_work_window())?;
            retain(&dir.join("asc_mixed_work.192b"),serr(asc_work.ledger_bytes())?)?;
            retain(&dir.join("asc_mixed_core.208b"),serr(asc_work.row_bytes())?)?;
            retain_json(&dir.join("asc_mixed_work.index.json"),&asc_work.index())?;
            let lif_work=serr(runtime.lif_work_window())?;
            retain(&dir.join("lif_actual_work.288b"),serr(lif_work.ledger_bytes())?)?;
            retain(&dir.join("lif_actual_rows.464b"),serr(lif_work.row_bytes())?)?;
            retain_json(&dir.join("lif_actual_work.index.json"),&lif_work.index())?;
            let mixed=serr(runtime.mixed_energy_window())?;
            retain(&dir.join("mixed_energy.256b"),serr(mixed.global_bytes())?)?;
            retain(&dir.join("mixed_energy_leaves.256b"),serr(mixed.leaf_bytes())?)?;
            retain(&dir.join("mixed_energy_stages.256b"),serr(mixed.sample_bytes())?)?;
            retain_json(&dir.join("mixed_energy.index.json"),&mixed.index())?;
            let phase2_work=serr(runtime.phase2_work_window())?;
            retain(&dir.join("phase2_operator_work.240b"),serr(phase2_work.ledger_bytes())?)?;
            retain(&dir.join("phase2_operator_vertices.256b"),serr(phase2_work.vertex_bytes())?)?;
            retain(&dir.join("phase2_operator_edges.64b"),serr(phase2_work.edge_bytes())?)?;
            retain_json(&dir.join("phase2_operator_work.index.json"),&phase2_work.index())?;
            let shared_time=serr(runtime.shared_time_window())?;
            retain(&dir.join("shared_physical_time.896b"),serr(shared_time.bytes())?)?;
            retain_json(&dir.join("shared_physical_time.index.json"),&shared_time.index())?;
            let (count, raw, native) = unsafe { serr(runtime.raw_observation_inputs())? };
            let retained_count = count.min(c.queue_capacity) as usize;
            retain(&dir.join("events.v10s96"), &raw[..retained_count * 96])?;
            retain(
                &dir.join("native_observations.64b"),
                &native.raw_records[..retained_count * 64],
            )?;
            retain_json(
                &dir.join("queue_state.json"),
                &json!({"count":count,"capacity":c.queue_capacity,"native_state":format!("{:?}",native.state),"accepted_body":outcome.accepted}),
            )?;
            ensure!(
                outcome.accepted,
                "collective physical chunk refused: {:?}",
                outcome.refusal
            );
            ensure!(
                count <= c.queue_capacity && native.count == count,
                "native queue overflow or row mismatch"
            );
            let clock = native_clock_words(&owners, &governors, begin, runtime.completed_step())?;
            let shared_clock = serr(runtime.shared_time_window())?.records.iter().map(|row| [
                u64::from(row.step.scheduled_step), f64::from(row.step.native_dt_ps).to_bits(),
                row.step.cumulative_accepted_time_ps.to_bits(),
            ]).collect::<Vec<_>>();
            validate_native_shared_clock_rows(&clock, &shared_clock, begin, runtime.completed_step())?;
            let durable_source = chronicle.append(
                begin,
                runtime.completed_step(),
                &raw[..count as usize * 96],
                &native.raw_records[..count as usize * 64],
                &clock,
                c.leaves.each_ref().map(|l| l.stream_id),
            )?;
            retain_json(&dir.join("chronicle_source_commit.json"), &durable_source)?;
            let observations = serr(decode_native_lif_observation_queue(
                &native.raw_records,
                count,
                c.queue_capacity,
                limit,
            ))?;
            let decoded = serr(decode_spike_queue_for_arrow_and_pathb_in_frame(
                &raw,
                count,
                c.queue_capacity,
                limit,
                &observations,
                geometry,
                BiprismEventFrame::AtomicBohr,
            ))?;
            // Every accepted row is retained first and then enters its own
            // leaf's existing native SO3/GASP/NodeC producer. No host score is
            // substituted for the pending decision; no physical clock advances.
            for (leaf,owner) in decision_owners.iter_mut().enumerate(){
                if let Some(owner)=owner {
                    let events=decoded.ledger.iter().enumerate().filter(|(_,row)|row.event.leaf_id as usize==leaf)
                        .map(|(i,_)|Ok((u32::try_from(i)?,decoded.native_events[i])))
                        .collect::<Result<Vec<_>>>()?;
                    if events.is_empty(){retain_json(&dir.join(format!("leaf{leaf:02}.asc_decision_empty.json")),&serr(owner.acknowledge_empty(begin,runtime.completed_step()))?)?;continue;}
                    let seeds=serr(crate::captured_pipeline::biprism_adjudication::prepare_adjudication_seeds(&events,begin,runtime.completed_step()))?;
                    let native_seeds=crate::rich_spike::NativeV2SeedPopulation::prepare(&seeds.rich);
                    retain(&dir.join(format!("leaf{leaf:02}.asc_native_seeds.bytes")),native_seeds.bytes(&seeds.rich))?;
                    retain_json(&dir.join(format!("leaf{leaf:02}.asc_decision_input.json")),&owner.seed_manifest(&seeds,begin,runtime.completed_step()))?;
                    let attempted=unsafe{owner.run_boundary(&seeds,begin,runtime.completed_step(),outcome.status.control.successful_bodies)};
                    match attempted {
                        Ok(decision)=>{
                            retain(&dir.join(format!("leaf{leaf:02}.asc_native_adjudicator.256b")),&decision.adjudicator)?;
                            retain(&dir.join(format!("leaf{leaf:02}.asc_relaxed_tiles.1280b")),&decision.relaxed_tiles)?;
                            retain(&dir.join(format!("leaf{leaf:02}.asc_perturbed_tiles.1280b")),&decision.perturbed_tiles)?;
                            retain(&dir.join(format!("leaf{leaf:02}.asc_dynamic_t7.32b")),&decision.calibration)?;
                            retain_json(&dir.join(format!("leaf{leaf:02}.asc_decision_execution.json")),&decision.index)?;
                            ensure!(decision.accepted,"native ASC decision refused: {:?}",decision.error);
                        }
                        Err(error)=>{retain_json(&dir.join(format!("leaf{leaf:02}.asc_decision_refusal.json")),&json!({"error":error,"accepted":false}))?;bail!("native ASC decision preparation refused: {error}");}
                    }
                }
            }
            let classification = classify(c, &decoded, &mut backend, consumers)?;
            let (mut export, receipt) =
                unsafe { serr(runtime.export_captured(&classification, 0))? };
            serr(export.bind_replica_seeds(c.leaves.each_ref().map(|l| l.seed)))?;
            let (arrow, ledger) = serr(export.to_paired_ipc())?;
            retain(&dir.join("spike_events.arrow"), &arrow)?;
            retain(&dir.join("voronoi_ledger.arrow"), &ledger)?;
            retain_json(
                &dir.join("classification_scope.json"),
                &json!({"scope":"this accepted physical chunk, separately clustered per native leaf","whole_run_classification":"leafNN/spike_events.arrow after complete run","seed_binding":"raw96.leaf_id to actual physical replica seed","group_id":0,"group_id_scope":"no multi-differential assignment declared"}),
            )?;
            retain_work(
                c,
                &owners,
                &governors,
                begin,
                runtime.completed_step(),
                &dir,
            )?;
            let durable_output=chronicle.acknowledge("biprism_paired_arrow",json!({"run_id":c.run_id,"completed_step":runtime.completed_step(),
                "queue_emission_groups":count,"artifacts":artifact_receipts(&dir)?,"source_commit":durable_source,
                "retained_export_ack_order":"source CAS, actual paired outputs/work fsync, consumer CAS, then live count reset"}))?;
            retain_json(&dir.join("chronicle_consumer_commit.json"), &durable_output)?;
            unsafe {
                serr(runtime.acknowledge_export(&receipt))?;
            }
            chunks.push(dir);
        }
        let grids = serr(runtime.aggregate_signal_grids())?;
        let output = c.output_root.join("eight_leaf_signal_grids.bin");
        let voxel_count=grids.data.voxel_hit_grid.len();
        let (serialized_bytes, checksum) = (consumers.write_signal_grid)(
            &output,
            &grids.data,
            u32::MAX,
            &c.run_id,
            "eight_leaf_aggregate",
        )?;
        File::open(&output)?.sync_all()?;
        retain_json(
            &c.output_root.join("signal_grid_scope.json"),
            &json!({"aggregation":grids.aggregation,"through_completed_step":grids.through_completed_step,"stream_id_field":u32::MAX,"stream_id_semantics":"aggregate sentinel, never a physical leaf","voxel_count":voxel_count,"serialized_bytes":serialized_bytes,"payload_fnv1a64":checksum,"coupled_grid_semantics":"LIF coupled multiplicity, not UV causality","native_serializer":"PRSGD001 four signed-i32 planes"}),
        )?;
        let (heuristic_water, heuristic_exclusion) = serr(runtime.lif_heuristic_grids())?;
        retain(
            &c.output_root.join("lif_heuristic_water.f32le"),
            &heuristic_water
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        retain(
            &c.output_root.join("lif_heuristic_exclusion.f32le"),
            &heuristic_exclusion
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        retain_json(
            &c.output_root.join("lif_heuristic_grid_scope.json"),
            &json!({"water_rows":heuristic_water.len(),"exclusion_rows":heuristic_exclusion.len(),
            "scope":"actual optional LIF heuristic output accumulators; separate from native atomic density, native UV state and signed four-grid counts"}),
        )?;
        for (leaf, owner) in owners.iter().enumerate() {
            let e = owner.lock().map_err(|_| anyhow!("engine owner poisoned"))?;
            let (positions, position_lows) = e.get_biprism_coordinate_split()?;
            let precise_positions = crate::biprism_coordinates::represented_positions(&positions,&position_lows)
                .map_err(anyhow::Error::msg)?;
            let velocities = e.get_velocities()?;
            ensure!(
                positions.iter().chain(&velocities).all(|v| v.is_finite()),
                "terminal physical state is nonfinite"
            );
            retain(
                &c.output_root
                    .join(format!("leaf{leaf:02}.positions.angstrom.f32le")),
                &positions
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            retain(
                &c.output_root
                    .join(format!("leaf{leaf:02}.velocities.angstrom_per_ps.f32le")),
                &velocities
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            retain(&c.output_root.join(format!("leaf{leaf:02}.positions.angstrom.low.f32le")),
                &position_lows.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?;
            retain(&c.output_root.join(format!("leaf{leaf:02}.positions.angstrom.f64le")),
                &precise_positions.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?;
            retain_json(&c.output_root.join(format!("leaf{leaf:02}.coordinate_authority.json")),
                &json!({"schema":"biprism.coordinates.split.v1","unit":"angstrom","atoms":positions.len()/3,
                "representation":"physical coordinate = FP64(high) + FP64(low); f32 file is high projection",
                "identity":crate::biprism_coordinates::BiprismCoordinateIdentity::from(e.biprism_coordinate_descriptor()?)}))?;
        }
        let analysis = postprocess(c, r, geometry, &chunks, &mut backend, consumers)?;
        validate_completed_output_extent(c.full_step_budget, runtime.completed_step(), c.chunk_steps,
            chunks.len(), runtime.graph_launch_count(), grids.through_completed_step,
            analysis.as_array().context("full-run analysis is not a physical leaf population")?.len())?;
        let mut artifacts = Vec::new();
        for leaf in 0..8 {
            artifacts.push(json!({"leaf":leaf,"replica_seed":c.leaves[leaf].seed,
            "artifacts":artifact_receipts(&c.output_root.join(format!("leaf{leaf:02}")))?}));
        }
        let acknowledged=chronicle.acknowledge("biprism_native_pathb",json!({"completed_step":runtime.completed_step(),"analysis":analysis,"artifacts":artifacts,
            "scope":"actual full retained source population classified/selected, native GPU clustering and source6 PathB outputs"}))?;
        retain_json(
            &c.output_root.join("chronicle_analysis_commit.json"),
            &acknowledged,
        )?;
        Ok(
            json!({"status":"IMPLEMENTATION_VERIFIED","accepted_full_run":true,"requested_physical_steps_per_leaf":c.full_step_budget,"completed_physical_steps_per_leaf":runtime.completed_step(),"total_leaf_physical_steps":u64::from(runtime.completed_step())*8,"graph_launches":runtime.graph_launch_count(),"retained_chunks":chunks.len(),"analysis":analysis,"scope":"actual configured native computational execution; no wet-lab or physical calibration inference","restart_checkpoint":false}),
        )
    })();
    let mut terminal_failures = Vec::new();
    // Complete coordinate-owner planes are retained on success or refusal.
    // Their continuation use still requires the complete physical-state owner.
    for (leaf,engine) in owners.iter().enumerate(){
        let image = engine.lock().map_err(|_|anyhow!("engine coordinate owner poisoned"))
            .and_then(|engine| engine.get_biprism_coordinate_snapshot());
        match image {
            Ok(image)=>retain_coordinate_snapshot(&c.output_root,&format!("leaf{leaf:02}.native_coordinates_terminal"),&image)?,
            Err(error)=>retain_terminal_failure(&c.output_root,&format!("leaf{leaf:02}.native_coordinates_terminal.error.json"),
                &format!("leaf{leaf:02} native coordinate state"),error,&mut terminal_failures)?,
        }
    }
    match runtime.coordinate_snapshots(){
        Ok(images)=>{
            if images.len()!=LEAVES {retain_terminal_failure(&c.output_root,"atomic_coordinates_terminal.error.json",
                "atomic coordinate population",format!("expected eight leaves, read {}",images.len()),&mut terminal_failures)?;}
            for (leaf,image) in images.iter().enumerate(){retain_coordinate_snapshot(&c.output_root,&format!("leaf{leaf:02}.atomic_coordinates_terminal"),image)?;}
        },
        Err(error)=>retain_terminal_failure(&c.output_root,"atomic_coordinates_terminal.error.json","atomic coordinate state",error,&mut terminal_failures)?,
    }
    // Unfiltered terminal diagnostics are retained even after an acceptance error.
    match runtime.potential_snapshot(){
        Ok(image)=>{retain_potential_snapshot(&c.output_root,"potential_terminal",&image)?;
            retain_spinor_snapshot(&c.output_root,"terminal",&serr(runtime.spinor_snapshot())?)?;},
        Err(error)=>retain_terminal_failure(&c.output_root,"potential_terminal.error.json","FP64 potential state",error,&mut terminal_failures)?,
    }
    let diagnostic = match runtime.read_status() {
        Ok(s) => status_json(&s),
        Err(error) => { terminal_failures.push(format!("terminal device status: {error}")); json!({"readback_error":error}) }
    };
    if let Some(initial)=runtime.initial_operator(){
        retain(&c.output_root.join("initial_operator_terminal_before.bytes"),&initial.before_tiles)?;
        retain(&c.output_root.join("initial_operator_terminal_after.bytes"),&initial.after_tiles)?;
        match initial.ledger_bytes() {
            Ok(bytes)=>retain(&c.output_root.join("initial_operator_terminal.240b"),bytes)?,
            Err(error)=>retain_terminal_failure(&c.output_root,"initial_operator_terminal.error.json","initial operator ledger",error,&mut terminal_failures)?,
        }
        retain_json(&c.output_root.join("initial_operator_terminal.index.json"),&initial.index())?;
    } else {
        retain_terminal_failure(&c.output_root,"initial_operator_terminal.error.json","initial operator",
            "no completed cold operator readback",&mut terminal_failures)?;
    }
    match runtime.mixed_energy_current_attempt(){Ok(x)=>{
        retain(&c.output_root.join("mixed_energy_attempt_terminal.256b"),serr(x.pending_bytes())?)?;
        retain(&c.output_root.join("mixed_energy_leaves_terminal.256b"),serr(x.leaf_bytes())?)?;
        retain(&c.output_root.join("mixed_energy_stages_terminal.256b"),serr(x.sample_bytes())?)?;
        retain(&c.output_root.join("mixed_energy_vertices_terminal.256b"),serr(x.vertex_bytes())?)?;
        retain(&c.output_root.join("mixed_energy_edges_terminal.64b"),serr(x.edge_bytes())?)?;
        retain_json(&c.output_root.join("mixed_energy_attempt_terminal.index.json"),&x.index())?;
    },Err(error)=>retain_terminal_failure(&c.output_root,"mixed_energy_attempt_terminal.error.json","mixed energy attempt",error,&mut terminal_failures)?}
    match runtime.phase2_work_current_attempt(){Ok(x)=>{
        retain(&c.output_root.join("phase2_work_attempt_terminal.240b"),serr(x.ledger_bytes())?)?;
        retain(&c.output_root.join("phase2_vertices_attempt_terminal.256b"),serr(x.vertex_bytes())?)?;
        retain(&c.output_root.join("phase2_edges_attempt_terminal.64b"),serr(x.edge_bytes())?)?;
        retain_json(&c.output_root.join("phase2_work_attempt_terminal.index.json"),&x.index())?;
    },Err(error)=>retain_terminal_failure(&c.output_root,"phase2_work_attempt_terminal.error.json","Phase2 work attempt",error,&mut terminal_failures)?}
    match runtime.lif_work_current_attempt(){Ok(x)=>{
        retain(&c.output_root.join("lif_work_attempt_terminal.288b"),serr(x.ledger_bytes())?)?;
        retain(&c.output_root.join("lif_rows_attempt_terminal.464b"),serr(x.row_bytes())?)?;
        retain_json(&c.output_root.join("lif_work_attempt_terminal.index.json"),&x.index())?;
    },Err(error)=>retain_terminal_failure(&c.output_root,"lif_work_attempt_terminal.error.json","LIF work attempt",error,&mut terminal_failures)?}
    match runtime.shared_time_current_attempt(){Ok(x)=>retain_json(&c.output_root.join("shared_time_attempt_terminal.json"),&serde_json::to_value(x)?)?,
        Err(error)=>retain_terminal_failure(&c.output_root,"shared_time_attempt_terminal.error.json","shared time attempt",error,&mut terminal_failures)?}
    // A failed body can return before its journal copy. Retain the actual live
    // rows independently; never reinterpret stale slots as accepted production.
    match runtime.asc_current_attempt() {
        Ok(attempt)=>{
            retain(&c.output_root.join("asc_attempt_terminal.192b"),serr(attempt.ledger_bytes())?)?;
            retain(&c.output_root.join("asc_core_attempt_terminal.208b"),serr(attempt.row_bytes())?)?;
            retain_json(&c.output_root.join("asc_attempt_terminal.index.json"),&attempt.index())?;
        }
        Err(error)=>retain_terminal_failure(&c.output_root,"asc_attempt_terminal.readback_error.json","ASC attempt",error,&mut terminal_failures)?,
    }
    match runtime.electronic_exchange(){
        Ok(exchange)=>{retain(&c.output_root.join("electronic_exchange_terminal.17792b"),&exchange.raw_bytes)?;
            retain_json(&c.output_root.join("electronic_exchange_terminal.index.json"),&json!({"leaf_count":8,"record_bytes":17792,"phase_scope":"4 accepted final substep only with matching outer convergence;5 fault;1/2/3 temporary","footers":format!("{:?}",exchange.footers())}))?;}
        Err(error)=>retain_terminal_failure(&c.output_root,"electronic_exchange_terminal.readback_error.json","electronic exchange",error,&mut terminal_failures)?,
    }
    retain_json(
        &c.output_root.join("chronicle_terminal.json"),
        &chronicle.summary()?,
    )?;
    retain_json(
        &c.output_root.join("terminal_device_state.json"),
        &diagnostic,
    )?;
    retain_json(&c.output_root.join("terminal_output_completeness.json"), &json!({
        "required_terminal_readbacks_complete":terminal_failures.is_empty(),"failures":terminal_failures,
        "scope":"actual terminal coordinate, operator, work, clock and exchange readbacks; complete scientific restart is not asserted",
    }))?;
    require_complete_terminal_outputs(result, &terminal_failures)
}

pub fn run_biprism(args: &BiprismRunArgs, consumers: BiprismRunConsumers) -> Result<()> {
    let raw = fs::read(&args.config).with_context(|| format!("read {}", args.config.display()))?;
    let config: BiprismRunConfig = serde_json::from_slice(&raw)?;
    let inputs = resolve_inputs(&config)?;
    if args.validate_only {
        println!(
            "{}",
            serde_json::to_string(
                &json!({"status":"INPUTS_VERIFIED","configuration_sha256":hash(&raw),"topology_sha256":config.topology.sha256,"harmonic_source_binding":inputs.harmonic_source_binding,"atom_count":inputs.topology.n_atoms,"physical_steps_per_leaf":config.full_step_budget,"device_initialized":false,"required_surfaces":config.required_surfaces})
            )?
        );
        return Ok(());
    }
    fs::create_dir(&config.output_root)
        .context("new native run requires an unused output directory")?;
    retain(&config.output_root.join("run_configuration.json"), &raw)?;
    if let Some(binding) = inputs.harmonic_source_binding.as_ref() {
        retain_json(&config.output_root.join("harmonic_source_binding.execution.json"), binding)?;
    }
    retain_json(
        &config.output_root.join("execution_contract.json"),
        &json!({"schema":"prism.biprism.native_execution.v1","run_id":config.run_id,"entrypoint":"nhs_rt_full biprism-run","direct_nonprivileged_owner":"eight isolated NhsAmberFusedEngine allocations, one captured collective WHILE","original_baseline_run_counts_reused":false,"required_surfaces":config.required_surfaces,"trust_class":config.modules.trust_class,"governed_membership":inputs.manifest["governed_ptx_membership"],"physical_mass_mapping":config.masses.mapping,"atomic_unit_bridge":config.unit_bridge_source,"dt_ps":config.numerical.dt_ps,"dt_au":(f64::from(config.numerical.dt_ps)*PS_TO_ATOMIC_TIME) as f32,"dt_fields_scope":"initial configured request, not the accepted physical timestep sequence","physical_time_policy":"common minimum of eight live Director requests, capped by explicit maximum when armed_adaptive; any below-minimum request refuses; held terminal baselines preserve live ASC modifiers","governor_configuration":config.governor,"adaptive_dt_bounds":config.adaptive_dt_bounds,"native_governor_scope":"selected-step energy/constraint measurement; no independent leaf transition, rescale or clock writer","authoritative_time_journal":"chunk_*/shared_physical_time.896b and shared_physical_time.index.json","physical_time_generation":config.physical_time_generation,"leaf_seeds":config.leaves.each_ref().map(|l|l.seed),"source_channel":6,"consumer_clock":"native same-emission protocol step","raw_ledger":"96 bytes retained; native observation is separately matched NOB2 64 bytes","full_ledger_and_native_arrow_required":true,"state_commit_scope":"failed native body is terminal and is not graph-replayed; whole-body rollback not asserted"}),
    )?;
    let result = execute(&config, &inputs, &consumers);
    let terminal = match &result {
        Ok(value) => value.clone(),
        Err(error) => {
            json!({"status":"IMPLEMENTATION_PARTIAL","error":format!("{error:#}"),"requested_steps_per_leaf":config.full_step_budget,"accepted_full_run":false})
        }
    };
    retain_json(&config.output_root.join("execution_result.json"), &terminal)?;
    let mut inventory = Vec::new();
    fn visit(root: &Path, path: &Path, out: &mut Vec<Value>) -> Result<()> {
        let mut entries = fs::read_dir(path)?.collect::<std::io::Result<Vec<_>>>()?;
        entries.sort_by_key(|e| e.path());
        for e in entries {
            let p = e.path();
            if e.file_type()?.is_dir() {
                visit(root, &p, out)?;
            } else if e.file_type()?.is_file() {
                let b = fs::read(&p)?;
                out.push(json!({"path":p.strip_prefix(root)? ,"bytes":b.len(),"sha256":hash(&b)}));
            }
        }
        Ok(())
    }
    visit(&config.output_root, &config.output_root, &mut inventory)?;
    retain_json(
        &config.output_root.join("artifact_inventory.json"),
        &json!({"run_id":config.run_id,"files":inventory,"configuration_sha256":hash(&raw)}),
    )?;
    println!("{}", serde_json::to_string(&terminal)?);
    result.map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn output_clock_join_rejects_single_leaf_or_uniform_wrong_native_time() {
        let dt0=f64::from(0.0005f32); let dt1=f64::from(0.002f32);
        let shared=vec![[1,dt0.to_bits(),dt0.to_bits()],[2,dt1.to_bits(),(dt0+dt1).to_bits()]];
        let native:[Vec<[u64;3]>;8]=std::array::from_fn(|_|shared.clone());
        assert!(validate_native_shared_clock_rows(&native,&shared,0,2).is_ok());
        for leaf in 0..8 { for field in 0..3 {
            let mut changed=native.clone(); changed[leaf][1][field]^=1;
            assert!(validate_native_shared_clock_rows(&changed,&shared,0,2).is_err());
        }}
        let mut wrong=native.clone();for leaf in &mut wrong {leaf[1][2]^=1;}
        assert!(validate_native_shared_clock_rows(&wrong,&shared,0,2).is_err());
        let mut partial=native;partial[7].pop();
        assert!(validate_native_shared_clock_rows(&partial,&shared,0,2).is_err());
        assert!(validate_native_shared_clock_rows(&partial,&[],2,2).is_err());
    }
    #[test]
    fn output_completion_requires_all_eighty_chunks_and_eight_leaves() {
        assert!(validate_completed_output_extent(40000,40000,500,80,80,40000,8).is_ok());
        for args in [(40000,39999,500,80,80,40000,8),(40000,40000,500,79,80,40000,8),
            (40000,40000,500,80,79,40000,8),(40000,40000,500,80,80,39999,8),
            (40000,40000,500,80,80,40000,7),(0,0,500,0,0,0,8),(40000,40000,0,80,80,40000,8)] {
            assert!(validate_completed_output_extent(args.0,args.1,args.2,args.3,args.4,args.5,args.6).is_err());
        }
        assert!(validate_completed_output_extent(1501,1501,500,4,4,1501,8).is_ok());
    }
    #[test]
    fn output_terminal_error_cannot_retain_success_or_hide_prior_failure() {
        assert!(require_complete_terminal_outputs(Ok(json!({"accepted_full_run":true})),&[]).is_ok());
        let missing=vec!["leaf7 low coordinates unavailable".to_string()];
        assert!(require_complete_terminal_outputs(Ok(json!({"accepted_full_run":true})),&missing).is_err());
        let error=require_complete_terminal_outputs(Err(anyhow!("earlier mixed energy refusal")),&missing).unwrap_err();
        let text=format!("{error:#}");assert!(text.contains("earlier mixed energy refusal")&&text.contains("leaf7 low coordinates unavailable"));
    }
    #[test]
    fn output_pathb_population_must_equal_selected_source() {
        assert!(validate_pathb_population(2,2,Some(2)).is_ok());
        assert!(validate_pathb_population(2,1,None).is_err());
        assert!(validate_pathb_population(2,2,Some(1)).is_err());
        assert!(validate_pathb_population(2,2,Some(3)).is_err());
        assert!(validate_pathb_population(u32::MAX as usize+1,0,None).is_err());
    }
    #[test]
    fn output_pathb_serialized_count_and_extent_are_checked_before_acknowledgement() {
        let root=std::env::temp_dir().join(format!("biprism_output_completeness_{}_{}",std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        fs::create_dir(&root).unwrap();let path=root.join("pathb.native");
        let state=json!({"event_count":2,"event_count_offset":16,"config_size_bytes":8,
            "event_buffer_offset":20,"event_stride_bytes":36,"endianness":"little"});
        // Deliberate file-format control, not an SDST execution or physics result.
        let mut bytes=vec![0u8;92];bytes[..4].copy_from_slice(&0x54534453u32.to_le_bytes());
        bytes[4..8].copy_from_slice(&1u32.to_le_bytes());bytes[16..20].copy_from_slice(&2u32.to_le_bytes());
        fs::write(&path,&bytes).unwrap();assert!(retain_complete_pathb_state(&path,&state,2).is_ok());
        bytes[16..20].copy_from_slice(&1u32.to_le_bytes());fs::write(&path,&bytes).unwrap();
        assert!(retain_complete_pathb_state(&path,&state,2).is_err());
        bytes[16..20].copy_from_slice(&2u32.to_le_bytes());fs::write(&path,&bytes[..91]).unwrap();
        assert!(retain_complete_pathb_state(&path,&state,2).is_err());
        bytes[0]=0;fs::write(&path,&bytes).unwrap();assert!(retain_complete_pathb_state(&path,&state,2).is_err());
        let mut failures=Vec::new();retain_terminal_failure(&root,"terminal.error.json","clock","unavailable",&mut failures).unwrap();
        assert_eq!(failures,vec!["clock: unavailable"]);
        let retained:Value=serde_json::from_slice(&fs::read(root.join("terminal.error.json")).unwrap()).unwrap();
        assert_eq!(retained["accepted"],false);assert_eq!(retained["error"],"unavailable");
        fs::remove_dir_all(root).unwrap();
    }
    fn configuration() -> BiprismRunConfig {
        let pin = PinnedFile {
            path: PathBuf::from("/test/input"),
            sha256: "a".repeat(64),
        };
        let protocol = CryoUvProtocol {
            start_temp: 77.0,
            end_temp: 300.0,
            cold_hold_steps: 500,
            ramp_steps: 500,
            warm_hold_steps: 500,
            current_step: 0,
            uv_burst_energy: 1.0,
            uv_burst_interval: 100,
            uv_burst_duration: 1,
            scan_wavelengths: vec![280.0],
            wavelength_dwell_steps: 100,
            ramp_down_steps: 0,
            cold_return_steps: 0,
            stepped_holds: vec![],
        };
        BiprismRunConfig {
            schema: "prism.biprism.native_run.v1".into(),
            run_id: "actual_shape_control".into(),
            output_root: PathBuf::from("/test/output"),
            device_ordinal: 0,
            topology: pin.clone(),
            harmonic_source_binding: None,
            masses: PhysicalMassOrigin {
                authority: pin.clone(),
                mapping: pin.clone(),
                true_inverse_masses_au: pin.clone(),
            },
            unit_bridge_source: pin.clone(),
            modules: GovernedModules {
                trust_manifest: pin,
                trust_class: TrustClass::SourceOwnedDevHashBound,
                ptx_directory: PathBuf::from("/test/ptx"),
                biprism_sha256: "b".repeat(64),
                fused_sha256: "c".repeat(64),
            },
            leaves: std::array::from_fn(|i| LeafConfiguration {
                seed: i as u64,
                stream_id: i as u8,
                center_atom: 0,
                protocol: protocol.clone(),
                uv_enabled: true,
                nma: None,
                asc: AscConfiguration::Disabled,
            }),
            numerical: NumericalContract {
                dt_ps: 0.0005,
                kappa_foliated: 0.0,
                constraint_policy: BiprismConstraintPolicy::FlexibleCore,
                gonzalez_exact_energy: false,
                lif_decay: 0.95,
                lif_threshold: 1.0,
                lif_kick: 0.1,
                picard_residual: 1e-7,
                dec_residual: 1e-6,
                absolute_energy_drift: 1e-6,
                unitarity_defect: 1e-7,
                dec_iteration_cap: 192,
            },
            governor: GovernorConfiguration::CollectiveSelectedMeasurement,
            adaptive_dt_bounds:None,max_seed_peel_passes:None,
            physical_time_generation:1,director_adaptive_dt:true,hold_terminal_controls:true,
            required_surfaces: [
                "amber",
                "protocol",
                "uv",
                "constraints",
                "work",
                "v10",
                "voronoi_lif",
                "native_observation",
                "arrow_v5",
                "full_ledger",
                "chronicle",
                "four_grids",
                "pathb",
                "clustering",
                "governor",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
            grid_dimension: 32,
            grid_spacing_angstrom: 1.0,
            target_tetrahedra: 960,
            queue_capacity: 800000,
            chronicle_max_bytes: 128 * 1024 * 1024,
            chunk_steps: 500,
            full_step_budget: 1500,
            clustering_epsilon_angstrom: 5.0,
            source_percentile: 0,
            wavefront_max_dt_steps: None,
        }
    }
    #[test]
    fn full_run_configuration_preserves_terminal_budget_and_requires_explicit_adaptive_bounds() {
        let mut c = configuration();
        assert!(c.validate_shape().is_ok());
        c.governor = GovernorConfiguration::ArmedAdaptive;
        assert!(c.validate_shape().is_err());
        c.adaptive_dt_bounds=Some(crate::biprism_shared_step::AdaptiveDtBounds{dt_min_ps:0.0005,dt_max_ps:0.002});
        assert!(c.validate_shape().is_ok());
        c.adaptive_dt_bounds=None;
        c.governor = GovernorConfiguration::FixedStepMeasurement;
        assert!(c.validate_shape().is_err());
        c.governor = GovernorConfiguration::CollectiveSelectedMeasurement;
        c.chunk_steps = 1;
        assert!(c.validate_shape().is_err());
        c.chunk_steps = 500;
        c.full_step_budget = 1501;
        assert!(c.validate_shape().is_ok());
        c.hold_terminal_controls=false;
        assert!(c.validate_shape().is_err());
    }
    #[test]
    fn bounded_adaptive_configuration_is_explicit_and_serialized() {
        let mut c=configuration();c.governor=GovernorConfiguration::ArmedAdaptive;
        c.adaptive_dt_bounds=Some(crate::biprism_shared_step::AdaptiveDtBounds{dt_min_ps:0.0005,dt_max_ps:0.002});
        c.numerical.dt_ps=0.002;c.max_seed_peel_passes=Some(256);
        assert!(c.validate_shape().is_ok());
        let value=serde_json::to_value(&c).unwrap();
        assert_eq!(value["governor"],"armed_adaptive");
        let round:BiprismRunConfig=serde_json::from_value(value.clone()).unwrap();assert!(round.validate_shape().is_ok());
        let mut unknown=value;unknown["radius"]=serde_json::json!(8.5);
        assert!(serde_json::from_value::<BiprismRunConfig>(unknown).is_err());
        c.director_adaptive_dt=false;assert!(c.validate_shape().is_err());c.director_adaptive_dt=true;
        c.numerical.dt_ps=0.004;assert!(c.validate_shape().is_err());c.numerical.dt_ps=0.002;
        c.adaptive_dt_bounds.as_mut().unwrap().dt_min_ps=f32::NAN;assert!(c.validate_shape().is_err());
    }
    #[test]
    fn required_surface_and_leaf_identity_cannot_be_silently_dropped() {
        let mut c = configuration();
        c.required_surfaces.push("nma".into());
        assert!(c.validate_shape().is_err());
        c.required_surfaces.pop();
        c.required_surfaces.push("asc".into());
        assert!(c.validate_shape().is_err());
        c.required_surfaces.pop();
        c.leaves[7].stream_id = 0;
        assert!(c.validate_shape().is_err());
        c.leaves[7].stream_id = 7;
        c.leaves[4].uv_enabled = false;
        assert!(c.validate_shape().is_err());
    }
    #[test]
    fn selected_initial_temperature_must_not_trigger_native_floor() {
        let mut c = configuration();
        c.leaves[0].protocol.start_temp = 49.0;
        assert!(c.validate_shape().unwrap_err().to_string().contains("below50K"));
    }

    #[test]
    fn solver_cap_expansion_does_not_relax_residual_admission() {
        let mut c = configuration();
        c.numerical.dec_iteration_cap = 193;
        assert!(c.validate_shape().is_err());
        c.numerical.dec_iteration_cap = 192;
        c.numerical.dec_residual = 1.01e-6;
        assert!(c.validate_shape().is_err());
        c.numerical.dec_residual = 1e-6;
        c.numerical.absolute_energy_drift = 1.01e-6;
        assert!(c.validate_shape().is_err());
        c.numerical.absolute_energy_drift = 1e-6;
        c.numerical.unitarity_defect = 1.01e-7;
        assert!(c.validate_shape().is_err());
        c.numerical.unitarity_defect = 1e-7;
        c.numerical.dt_ps = f32::NAN;
        assert!(c.validate_shape().is_err());
    }
}

/// Observation-only adapter over the existing durable Chronicle CAS. Scientific
/// restart remains a separate owner-registry contract; no HEAD is manufactured.
mod observation_chronicle {
    use crate::scientific_checkpoint::{chronicle::*, CheckpointIdentity, CheckpointStore, Head};
    use anyhow::{ensure, Context, Result};
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::{collections::BTreeSet, path::Path};
    const RAW: u32 = 0;
    const NATIVE: u32 = 1;
    const ORDINAL: u32 = 2;
    const CLOCK: u32 = 3;
    const DRAIN: u32 = 4;
    const FIELDS: usize = 5;
    const MAX_SLAB: usize = 4 * 1024 * 1024;
    const PAIRED: &str = "biprism_paired_arrow";
    const ANALYSIS: &str = "biprism_native_pathb";
    fn component(
        id: u32,
        name: &str,
        offset: u32,
        width: u32,
        encoding: &str,
        role: &str,
        unit: &str,
    ) -> ComponentDescriptor {
        ComponentDescriptor {
            id,
            name: name.into(),
            byte_offset: offset,
            byte_width: width,
            encoding: encoding.into(),
            quantity_role: role.into(),
            unit: unit.into(),
        }
    }
    fn field(
        id: u32,
        name: &str,
        width: u32,
        components: Vec<ComponentDescriptor>,
        clock: RecordClockBinding,
    ) -> DatumDescriptor {
        DatumDescriptor {
            id,
            owner: "biprism_native_observation".into(),
            name: name.into(),
            class: DatumClass::Observation,
            quantity_role: "source_owned_native_bits".into(),
            unit: "source_contract".into(),
            encoding: format!(
                "opaque.sha256:{:x}",
                Sha256::digest(
                    serde_json::to_vec(&(name, width, &components)).expect("serializable layout")
                )
            ),
            element_bytes: width,
            elements_per_record: 1,
            shape: vec![1],
            subject: "exact_per_leaf_subsequence_of_collective_native_emission".into(),
            index_order:
                "per_leaf_native_subsequence; original mixed-queue ordinal retained in field2"
                    .into(),
            record_clock: clock,
            original_ordinal: OriginalOrdinalBinding::NativeEmission,
            components,
        }
    }
    fn sidecar(step: u32) -> RecordClockBinding {
        RecordClockBinding::NativeStepSidecar {
            record_step_component: step,
            sidecar_field: CLOCK,
            sidecar_step_component: 0,
            sidecar_physical_time_component: 2,
        }
    }
    pub(super) struct BiprismChronicle {
        store: CheckpointStore,
        identity: CheckpointIdentity,
        dictionary: ChronicleDictionary,
        head: Option<Head>,
        ordinal: [[u64; FIELDS]; 8],
        last_step: u32,
        last_time: [u64; 8],
        queue_ordinal: u64,
    }
    impl BiprismChronicle {
        pub(super) fn open(
            root: &Path,
            quota: u64,
            identity: CheckpointIdentity,
            seeds: [u64; 8],
            stream_ids: [u8; 8],
            source_hash: &str,
            consumer_hash: &str,
        ) -> Result<Self> {
            ensure!(
                identity.stream_count == 8,
                "collective Chronicle requires exactly eight source leaves"
            );
            let dictionary = ChronicleDictionary {
                fields: vec![
                    field(
                        RAW,
                        "v10_raw96",
                        96,
                        vec![
                            component(0, "protocol_step", 12, 4, "u32.le", "protocol_step", "step"),
                            component(1, "leaf_id", 64, 2, "u16.le", "physical_leaf", "index"),
                            component(2, "local_vertex", 66, 2, "u16.le", "tile_vertex", "index"),
                        ],
                        sidecar(0),
                    ),
                    field(
                        NATIVE,
                        "native_observation64",
                        64,
                        vec![
                            component(0, "protocol_step", 0, 4, "u32.le", "protocol_step", "step"),
                            component(1, "leaf_id", 4, 2, "u16.le", "physical_leaf", "index"),
                            component(2, "local_vertex", 6, 2, "u16.le", "tile_vertex", "index"),
                        ],
                        sidecar(0),
                    ),
                    field(
                        ORDINAL,
                        "collective_queue_order",
                        24,
                        vec![
                            component(
                                0,
                                "collective_ordinal",
                                0,
                                8,
                                "u64.le",
                                "source_record_ordinal",
                                "index",
                            ),
                            component(
                                1,
                                "leaf_ordinal",
                                8,
                                8,
                                "u64.le",
                                "leaf_record_ordinal",
                                "index",
                            ),
                            component(2, "protocol_step", 16, 4, "u32.le", "protocol_step", "step"),
                            component(3, "leaf_id", 20, 4, "u32.le", "physical_leaf", "index"),
                        ],
                        sidecar(2),
                    ),
                    field(
                        CLOCK,
                        "native_physical_clock24",
                        24,
                        vec![
                            component(0, "protocol_step", 0, 8, "u64.le", "protocol_step", "step"),
                            component(1, "applied_dt", 8, 8, "f64.le", "native_applied_dt", "ps"),
                            component(
                                2,
                                "physical_time",
                                16,
                                8,
                                "f64.le",
                                "native_accumulated_time",
                                "ps",
                            ),
                        ],
                        RecordClockBinding::Embedded {
                            step_component: 0,
                            physical_time_component: 2,
                        },
                    ),
                    field(
                        DRAIN,
                        "collective_drain64",
                        64,
                        vec![
                            component(
                                0,
                                "first_step",
                                0,
                                4,
                                "u32.le",
                                "completed_interval_start",
                                "step",
                            ),
                            component(
                                1,
                                "last_step",
                                4,
                                4,
                                "u32.le",
                                "completed_interval_end",
                                "step",
                            ),
                            component(
                                2,
                                "physical_time",
                                8,
                                8,
                                "f64.le",
                                "native_endpoint_time",
                                "ps",
                            ),
                            component(
                                3,
                                "leaf_records",
                                16,
                                8,
                                "u64.le",
                                "emission_group_count",
                                "count",
                            ),
                            component(
                                4,
                                "queue_records",
                                24,
                                8,
                                "u64.le",
                                "collective_emission_group_count",
                                "count",
                            ),
                            component(
                                5,
                                "queue_first",
                                32,
                                8,
                                "u64.le",
                                "source_record_ordinal",
                                "index",
                            ),
                            component(
                                6,
                                "queue_next",
                                40,
                                8,
                                "u64.le",
                                "source_record_ordinal",
                                "index",
                            ),
                            component(7, "leaf_id", 48, 4, "u32.le", "physical_leaf", "index"),
                            component(8, "stream_id", 52, 4, "u32.le", "native_stream", "index"),
                            component(
                                9,
                                "replica_seed",
                                56,
                                8,
                                "u64.le",
                                "native_replica_seed",
                                "seed",
                            ),
                        ],
                        RecordClockBinding::Embedded {
                            step_component: 1,
                            physical_time_component: 2,
                        },
                    ),
                ],
                contexts: (0..8)
                    .map(|leaf| DatumContext {
                        id: leaf as u32,
                        stream_id: leaf as u32,
                        replica_epoch: 1,
                        replica_seed: seeds[leaf],
                        numerical_phase: "accepted_collective_body_observation".into(),
                        thermal_phase: format!(
                            "native_stream_{}_protocol_step_authoritative",
                            stream_ids[leaf]
                        ),
                        clock_domain: "native_applied_dt_accumulation_ps".into(),
                        source_lineage_sha256: source_hash.into(),
                    })
                    .collect(),
                consumers: [PAIRED, ANALYSIS]
                    .into_iter()
                    .map(|name| ConsumerContract {
                        name: name.into(),
                        implementation_sha256: consumer_hash.into(),
                        required_fields: (0..FIELDS as u32).collect::<BTreeSet<_>>(),
                    })
                    .collect(),
            };
            dictionary.validate(&identity)?;
            let store = CheckpointStore::create(root, quota)?;
            ensure!(
                store.chronicle_head()?.is_none(),
                "existing collective Chronicle needs explicit qualified continuation"
            );
            Ok(Self {
                store,
                identity,
                dictionary,
                head: None,
                ordinal: [[0; FIELDS]; 8],
                last_step: 0,
                last_time: [0; 8],
                queue_ordinal: 0,
            })
        }
        /// Clocks contain original three u64 words: native step, f64 applied dt,
        /// and f64 accumulated time. No time is reconstructed from nominal dt.
        pub(super) fn append(
            &mut self,
            begin: u32,
            end: u32,
            raw: &[u8],
            native: &[u8],
            clocks: &[Vec<[u64; 3]>; 8],
            stream_ids: [u8; 8],
        ) -> Result<Value> {
            ensure!(
                begin == self.last_step && end > begin,
                "collective Chronicle step gap/replay"
            );
            ensure!(
                raw.len() % 96 == 0
                    && native.len() % 64 == 0
                    && raw.len() / 96 == native.len() / 64,
                "Chronicle source row mismatch"
            );
            let count = raw.len() / 96;
            let next_queue = self
                .queue_ordinal
                .checked_add(count as u64)
                .context("Chronicle queue ordinal overflow")?;
            let mut payloads: Vec<[Vec<u8>; FIELDS]> = (0..8)
                .map(|_| std::array::from_fn(|_| Vec::new()))
                .collect();
            let mut times = [0u64; 8];
            let mut row_steps: [Vec<u32>; 8] = std::array::from_fn(|_| Vec::new());
            for leaf in 0..8 {
                ensure!(
                    clocks[leaf].len() == (end - begin) as usize,
                    "Chronicle physical clock coverage differs"
                );
                let mut previous = f64::from_bits(self.last_time[leaf]);
                for (i, words) in clocks[leaf].iter().enumerate() {
                    let dt = f64::from_bits(words[1]);
                    let time = f64::from_bits(words[2]);
                    ensure!(
                        words[0] == u64::from(begin) + i as u64 + 1
                            && dt.is_finite()
                            && dt > 0.0
                            && time.is_finite()
                            && time > previous
                            && previous + dt == time,
                        "Chronicle native physical clock gap or altered accumulation"
                    );
                    previous = time;
                    for word in words {
                        payloads[leaf][CLOCK as usize].extend_from_slice(&word.to_le_bytes());
                    }
                }
                times[leaf] = previous.to_bits();
            }
            for (i, (event, observation)) in raw
                .chunks_exact(96)
                .zip(native.chunks_exact(64))
                .enumerate()
            {
                let leaf = u16::from_le_bytes(event[64..66].try_into().unwrap()) as usize;
                let vertex = u16::from_le_bytes(event[66..68].try_into().unwrap());
                let step = u32::from_le_bytes(event[12..16].try_into().unwrap());
                ensure!(
                    leaf < 8 && step > begin && step <= end,
                    "Chronicle event belongs outside accepted interval"
                );
                ensure!(
                    u32::from_le_bytes(observation[0..4].try_into().unwrap()) == step
                        && u16::from_le_bytes(observation[4..6].try_into().unwrap()) as usize
                            == leaf
                        && u16::from_le_bytes(observation[6..8].try_into().unwrap()) == vertex,
                    "Chronicle same-ordinal companion identity differs"
                );
                ensure!(
                    row_steps[leaf].last().is_none_or(|last| step >= *last),
                    "Chronicle native queue chronology regressed within leaf"
                );
                let leaf_ordinal = self.ordinal[leaf][RAW as usize] + row_steps[leaf].len() as u64;
                row_steps[leaf].push(step);
                payloads[leaf][RAW as usize].extend_from_slice(event);
                payloads[leaf][NATIVE as usize].extend_from_slice(observation);
                let map = &mut payloads[leaf][ORDINAL as usize];
                map.extend_from_slice(&(self.queue_ordinal + i as u64).to_le_bytes());
                map.extend_from_slice(&leaf_ordinal.to_le_bytes());
                map.extend_from_slice(&step.to_le_bytes());
                map.extend_from_slice(&(leaf as u32).to_le_bytes());
            }
            for leaf in 0..8 {
                let drain = &mut payloads[leaf][DRAIN as usize];
                drain.extend_from_slice(&begin.to_le_bytes());
                drain.extend_from_slice(&end.to_le_bytes());
                drain.extend_from_slice(&times[leaf].to_le_bytes());
                drain.extend_from_slice(&(row_steps[leaf].len() as u64).to_le_bytes());
                drain.extend_from_slice(&(count as u64).to_le_bytes());
                drain.extend_from_slice(&self.queue_ordinal.to_le_bytes());
                drain.extend_from_slice(&next_queue.to_le_bytes());
                drain.extend_from_slice(&(leaf as u32).to_le_bytes());
                drain.extend_from_slice(&u32::from(stream_ids[leaf]).to_le_bytes());
                drain.extend_from_slice(&self.dictionary.contexts[leaf].replica_seed.to_le_bytes());
            }
            let widths = [96, 64, 24, 24, 64];
            let old = self.ordinal;
            let mut next = old;
            let head = self.store.commit_chronicle_streaming(
                self.head.as_ref(),
                &self.identity,
                &self.dictionary,
                MAX_SLAB as u64,
                &[],
                None,
                |append| {
                    for leaf in 0..8 {
                        for field in 0..FIELDS {
                            let width = widths[field];
                            let rows_per_slab = MAX_SLAB / width;
                            for (slab, bytes) in payloads[leaf][field]
                                .chunks(rows_per_slab * width)
                                .enumerate()
                            {
                                let first_row = slab * rows_per_slab;
                                let rows = bytes.len() / width;
                                let (first_step, last_step) = if field <= ORDINAL as usize {
                                    (
                                        row_steps[leaf][first_row],
                                        row_steps[leaf][first_row + rows - 1],
                                    )
                                } else if field == CLOCK as usize {
                                    (
                                        begin + first_row as u32 + 1,
                                        begin + (first_row + rows) as u32,
                                    )
                                } else {
                                    (end, end)
                                };
                                let time_at =
                                    |step: u32| clocks[leaf][(step - begin - 1) as usize][2];
                                let first = old[leaf][field] + first_row as u64;
                                let end_ordinal = first + rows as u64;
                                append(&LeafInput {
                                    producer: ProducerKey {
                                        field_id: field as u32,
                                        stream_id: leaf as u32,
                                        replica_epoch: 1,
                                    },
                                    context_id: leaf as u32,
                                    lineage: None,
                                    coverage: CoverageRange {
                                        first_ordinal: first,
                                        next_ordinal: end_ordinal,
                                        first_step: u64::from(first_step),
                                        last_step: u64::from(last_step),
                                        first_physical_time_ps_bits: time_at(first_step),
                                        last_physical_time_ps_bits: time_at(last_step),
                                    },
                                    bytes: bytes.to_vec(),
                                })?;
                                next[leaf][field] = end_ordinal;
                            }
                        }
                    }
                    Ok(())
                },
            )?;
            // No retry after ambiguous publication failure: these cursors and
            // the live queue remain unchanged unless the actual CAS returns Ok.
            self.head = Some(head.clone());
            self.ordinal = next;
            self.last_step = end;
            self.last_time = times;
            self.queue_ordinal = next_queue;
            Ok(
                json!({"schema":"prism.biprism.observation_chronicle.v1","head":head,"completed_step":end,"collective_original_next_ordinal":next_queue,
                "leaf_ordinals":self.ordinal,"raw96_and_nob2_64_unchanged":true,"native_clocks_not_reconstructed":true,"scientific_restart_anchor":false}),
            )
        }
        pub(super) fn acknowledge(&mut self, consumer: &str, receipt: Value) -> Result<Value> {
            ensure!(
                [PAIRED, ANALYSIS].contains(&consumer),
                "unknown collective Chronicle consumer"
            );
            let head = self
                .head
                .as_ref()
                .context("consumer has no committed input Chronicle")?
                .clone();
            let index = self.store.validate_chronicle_index(&head, &self.identity)?;
            let receipt_bytes = serde_json::to_vec(&receipt)?;
            let acks = index
                .foliage()
                .producers
                .iter()
                .filter(|p| {
                    !index.foliage().consumers.iter().any(|a| {
                        a.consumer == consumer
                            && a.producer == p.producer
                            && a.next_ordinal >= p.next_ordinal
                    })
                })
                .map(|p| ConsumerAck {
                    consumer: consumer.into(),
                    producer: p.producer.clone(),
                    next_ordinal: p.next_ordinal,
                    input_foliage: head.clone(),
                    receipt_bytes: receipt_bytes.clone(),
                    output_payloads: Vec::new(),
                })
                .collect::<Vec<_>>();
            let next = self.store.commit_chronicle(
                Some(&head),
                &self.identity,
                &self.dictionary,
                &[],
                &acks,
                None,
            )?;
            self.head = Some(next.clone());
            Ok(
                json!({"consumer":consumer,"input_head":head,"acknowledged_head":next,"receipt":receipt}),
            )
        }
        pub(super) fn summary(&self) -> Result<Value> {
            let index = self
                .head
                .as_ref()
                .map(|h| self.store.validate_chronicle_index(h, &self.identity))
                .transpose()?;
            Ok(
                json!({"head":self.head,"identity":self.identity,"producer_ordinals":self.ordinal,"completed_step":self.last_step,
                "original_collective_next_ordinal":self.queue_ordinal,"committed_payload_leaves":index.as_ref().map(|i|i.foliage().leaves.len()),
                "scope":"existing Chronicle CAS observation record; no complete-owner scientific restart capsule"}),
            )
        }
    }
    #[cfg(test)]
    mod tests {
        use super::*;
        fn identity() -> CheckpointIdentity {
            CheckpointIdentity {
                run_id: "biprism_observation_test".into(),
                topology_sha256: "1".repeat(64),
                engine_sha256: "2".repeat(64),
                kernel_set_sha256: "3".repeat(64),
                protocol_sha256: "4".repeat(64),
                execution_request_sha256: "5".repeat(64),
                state_abi_sha256: "6".repeat(64),
                provider: "native_test_input".into(),
                stream_count: 8,
            }
        }
        #[test]
        fn collective_chronicle_preserves_raw_bits_mixed_order_and_actual_clocks() {
            let root = std::env::temp_dir().join(format!(
                "biprism_chronicle_{}_{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            let mut writer = BiprismChronicle::open(
                &root,
                16 * 1024 * 1024,
                identity(),
                [11, 22, 33, 44, 55, 66, 77, 88],
                [7, 6, 5, 4, 3, 2, 1, 0],
                &"7".repeat(64),
                &"8".repeat(64),
            )
            .unwrap();
            let clocks = std::array::from_fn(|_| vec![[1, 0.125f64.to_bits(), 0.125f64.to_bits()]]);
            let mut raw = vec![0xa5; 192];
            let mut obs = vec![0x5a; 128];
            for (i, leaf) in [7u16, 0u16].into_iter().enumerate() {
                raw[i * 96 + 12..i * 96 + 16].copy_from_slice(&1u32.to_le_bytes());
                raw[i * 96 + 64..i * 96 + 66].copy_from_slice(&leaf.to_le_bytes());
                raw[i * 96 + 66..i * 96 + 68].copy_from_slice(&9u16.to_le_bytes());
                obs[i * 64..i * 64 + 4].copy_from_slice(&1u32.to_le_bytes());
                obs[i * 64 + 4..i * 64 + 6].copy_from_slice(&leaf.to_le_bytes());
                obs[i * 64 + 6..i * 64 + 8].copy_from_slice(&9u16.to_le_bytes());
            }
            writer
                .append(0, 1, &raw, &obs, &clocks, [7, 6, 5, 4, 3, 2, 1, 0])
                .unwrap();
            let index = writer
                .store
                .validate_chronicle_index(writer.head.as_ref().unwrap(), &writer.identity)
                .unwrap();
            for (leaf, ordinal) in [(7, 0), (0, 1)] {
                let idx = index
                    .foliage()
                    .leaves
                    .iter()
                    .position(|l| l.producer.field_id == RAW && l.producer.stream_id == leaf)
                    .unwrap();
                assert_eq!(
                    writer
                        .store
                        .read_indexed_chronicle_payload(&index, idx)
                        .unwrap(),
                    raw[ordinal * 96..ordinal * 96 + 96]
                );
                let map = index
                    .foliage()
                    .leaves
                    .iter()
                    .position(|l| l.producer.field_id == ORDINAL && l.producer.stream_id == leaf)
                    .unwrap();
                assert_eq!(
                    u64::from_le_bytes(
                        writer
                            .store
                            .read_indexed_chronicle_payload(&index, map)
                            .unwrap()[..8]
                            .try_into()
                            .unwrap()
                    ),
                    ordinal as u64
                );
            }
            assert!(writer
                .append(0, 1, &raw, &obs, &clocks, [7, 6, 5, 4, 3, 2, 1, 0])
                .is_err());
            writer
                .acknowledge(PAIRED, json!({"actual_test_output":"first_drain_retained"}))
                .unwrap();
            let mut bad = std::array::from_fn(|_| vec![[2, 0.125f64.to_bits(), 0.25f64.to_bits()]]);
            bad[3][0][2] = 0.251f64.to_bits();
            assert!(writer
                .append(1, 2, &[], &[], &bad, [7, 6, 5, 4, 3, 2, 1, 0])
                .is_err());
            let next = std::array::from_fn(|_| vec![[2, 0.125f64.to_bits(), 0.25f64.to_bits()]]);
            writer
                .append(1, 2, &[], &[], &next, [7, 6, 5, 4, 3, 2, 1, 0])
                .unwrap();
            writer
                .acknowledge(PAIRED, json!({"actual_test_output":"retained"}))
                .unwrap();
            assert_eq!(writer.last_step, 2);
            assert_eq!(writer.queue_ordinal, 2);
            let index = writer
                .store
                .validate_chronicle_index(writer.head.as_ref().unwrap(), &writer.identity)
                .unwrap();
            for ack in &index.foliage().consumers {
                let expected = if ack.producer.field_id <= ORDINAL {
                    1
                } else {
                    2
                };
                assert_eq!(
                    ack.next_ordinal, expected,
                    "zero-emission drain advances only clock and drain producers"
                );
            }
            assert!(writer.store.head().unwrap().is_none());
            std::fs::remove_dir_all(root).unwrap();
        }

        fn new_test_writer(root: &Path) -> BiprismChronicle {
            BiprismChronicle::open(
                root,
                16 * 1024 * 1024,
                identity(),
                [11, 22, 33, 44, 55, 66, 77, 88],
                [7, 6, 5, 4, 3, 2, 1, 0],
                &"7".repeat(64),
                &"8".repeat(64),
            )
            .unwrap()
        }

        fn empty_clocks() -> [Vec<[u64; 3]>; 8] {
            std::array::from_fn(|_| vec![[1, 0.125f64.to_bits(), 0.125f64.to_bits()]])
        }

        // A real child exits without Rust destructors. The pre-publication
        // case uses the production streaming callback after one immutable
        // object is written; it does not replace the actual CAS implementation.
        #[test]
        fn collective_chronicle_crash_child() {
            let Ok(mode) = std::env::var("BIPRISM_CHRONICLE_TEST_CRASH") else {
                return;
            };
            let root =
                std::path::PathBuf::from(std::env::var_os("BIPRISM_CHRONICLE_TEST_ROOT").unwrap());
            let mut writer = new_test_writer(&root);
            if mode == "staged_before_source_head" {
                let words = empty_clocks()[0][0];
                let input = LeafInput {
                    producer: ProducerKey {
                        field_id: CLOCK,
                        stream_id: 0,
                        replica_epoch: 1,
                    },
                    context_id: 0,
                    lineage: None,
                    coverage: CoverageRange {
                        first_ordinal: 0,
                        next_ordinal: 1,
                        first_step: 1,
                        last_step: 1,
                        first_physical_time_ps_bits: words[2],
                        last_physical_time_ps_bits: words[2],
                    },
                    bytes: words.into_iter().flat_map(u64::to_le_bytes).collect(),
                };
                writer
                    .store
                    .commit_chronicle_streaming(
                        None,
                        &writer.identity,
                        &writer.dictionary,
                        MAX_SLAB as u64,
                        &[],
                        None,
                        |append| {
                            append(&input)?;
                            std::process::exit(91);
                        },
                    )
                    .unwrap();
                unreachable!();
            }
            writer
                .append(0, 1, &[], &[], &empty_clocks(), [7, 6, 5, 4, 3, 2, 1, 0])
                .unwrap();
            if mode == "source_head_before_consumer_ack" {
                std::process::exit(91)
            }
            assert_eq!(mode, "consumer_head_before_live_queue_ack");
            writer
                .acknowledge(PAIRED, json!({"actual_test_output":"retained_before_ack"}))
                .unwrap();
            std::process::exit(91);
        }

        #[test]
        fn collective_chronicle_process_crash_exposes_only_committed_prefix() {
            for (mode, generation, acks) in [
                ("staged_before_source_head", 0, 0),
                ("source_head_before_consumer_ack", 1, 0),
                ("consumer_head_before_live_queue_ack", 2, 16),
            ] {
                let root = std::env::temp_dir().join(format!(
                    "biprism_chronicle_crash_{}_{}_{}",
                    std::process::id(),
                    mode,
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos()
                ));
                let child = std::process::Command::new(std::env::current_exe().unwrap())
                    .args(["--exact","biprism_run::observation_chronicle::tests::collective_chronicle_crash_child","--nocapture"])
                    .env("BIPRISM_CHRONICLE_TEST_CRASH",mode)
                    .env("BIPRISM_CHRONICLE_TEST_ROOT",&root)
                    .output().unwrap();
                // Standalone exact-source validation has no outer biprism_run
                // module. Retry only when the exact filter selected no test.
                let child = if child.status.success()
                    && String::from_utf8_lossy(&child.stdout).contains("running 0 tests")
                {
                    std::process::Command::new(std::env::current_exe().unwrap())
                        .args([
                            "--exact",
                            "observation_chronicle::tests::collective_chronicle_crash_child",
                            "--nocapture",
                        ])
                        .env("BIPRISM_CHRONICLE_TEST_CRASH", mode)
                        .env("BIPRISM_CHRONICLE_TEST_ROOT", &root)
                        .output()
                        .unwrap()
                } else {
                    child
                };
                assert_eq!(
                    child.status.code(),
                    Some(91),
                    "{}: {} {}",
                    mode,
                    String::from_utf8_lossy(&child.stdout),
                    String::from_utf8_lossy(&child.stderr)
                );
                let store = CheckpointStore::create(&root, 16 * 1024 * 1024).unwrap();
                let head = store.chronicle_head().unwrap();
                assert_eq!(head.as_ref().map_or(0, |h| h.generation), generation);
                if let Some(head) = head {
                    let index = store.validate_chronicle_index(&head, &identity()).unwrap();
                    assert_eq!(index.foliage().leaves.len(), 16);
                    assert_eq!(index.foliage().consumers.len(), acks);
                    assert!(index.foliage().restart_anchor.is_none());
                    for n in 0..index.foliage().leaves.len() {
                        let bytes = store.read_indexed_chronicle_payload(&index, n).unwrap();
                        assert_eq!(
                            bytes.len(),
                            if index.foliage().leaves[n].producer.field_id == CLOCK {
                                24
                            } else {
                                64
                            }
                        );
                    }
                    assert!(
                        BiprismChronicle::open(
                            &root,
                            16 * 1024 * 1024,
                            identity(),
                            [11, 22, 33, 44, 55, 66, 77, 88],
                            [7, 6, 5, 4, 3, 2, 1, 0],
                            &"7".repeat(64),
                            &"8".repeat(64)
                        )
                        .is_err(),
                        "observation prefix must not silently resume physical owners"
                    );
                } else {
                    assert!(
                        std::fs::read_dir(root.join("objects"))
                            .unwrap()
                            .next()
                            .is_some(),
                        "prepublication injection must have staged a real object"
                    );
                }
                assert!(store.head().unwrap().is_none());
                println!("actual_checkpoint_store_crash_boundary={mode} recovered_generation={generation} consumer_watermarks={acks}");
                std::fs::remove_dir_all(root).unwrap();
            }
        }

        #[test]
        fn collective_chronicle_stale_writer_does_not_advance_ordinals() {
            let root = std::env::temp_dir().join(format!(
                "biprism_chronicle_stale_{}_{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            let mut first = new_test_writer(&root);
            let mut stale = new_test_writer(&root);
            first
                .append(0, 1, &[], &[], &empty_clocks(), [7, 6, 5, 4, 3, 2, 1, 0])
                .unwrap();
            let head = first.head.clone();
            assert!(stale
                .append(0, 1, &[], &[], &empty_clocks(), [7, 6, 5, 4, 3, 2, 1, 0])
                .unwrap_err()
                .to_string()
                .contains("stale Chronicle HEAD"));
            assert_eq!(stale.last_step, 0);
            assert_eq!(stale.ordinal, [[0; FIELDS]; 8]);
            assert_eq!(stale.store.chronicle_head().unwrap(), head);
            std::fs::remove_dir_all(root).unwrap();
        }
    }
}
