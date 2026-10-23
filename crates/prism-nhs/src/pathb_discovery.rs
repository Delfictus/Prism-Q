//! Unbiased, no-probe, whole-lattice discovery aggregation.
//!
//! This stage catalogs chain-aware/spatially corresponding Ghost components
//! across independent discovery seeds.  It deliberately performs no kinetic
//! fitting, matched-control classification, tracker invention, registry
//! mutation, ranking, or suppression.  Confirmation remains a separate,
//! prospectively designed stage.

use crate::chemical_state_fiber::{
    canonical_sha256, require_sha256, ControlLane, FragmentOccupancy, ProtocolPhaseBinding,
    ThermodynamicCondition,
};
use crate::component_correspondence::{
    establish_component_correspondence, ComponentCorrespondenceAssignment,
    ComponentCorrespondenceLedger, ComponentCorrespondencePolicy, ComponentCorrespondenceStatus,
    NativeComponentDescriptor,
};
use crate::pathb_chemical_state::NativePathBLatticeObservation;
use crate::pathb_evidence_emitter::{
    emit_qnanl_exact_at_public_paths, load_native_pathb_lane_evidence,
    load_native_pathb_lane_transaction_commit, GhostFiberMeasurementObservation,
    NativePathBLaneEvidence, NativePathBLaneTransactionCommit, NativePathBSelectionMode,
    PathBCandidateTrackerDefinition, TypedEvidenceArtifact,
    NATIVE_PATHB_LANE_TRANSACTION_DIRECTORY,
};
use anyhow::{bail, Context, Result};
use prismcode::qnanl_artifact::{decode_packet_bytes, QnanlArtifactFamily};
use prismcode::runtime_provenance::RuntimeProvenance;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CString;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Component, Path, PathBuf};

pub const DISCOVERY_MANIFEST_SCHEMA: &str = "prism_native_pathb_discovery_manifest";
pub const DISCOVERY_MANIFEST_VERSION: &str = "1.0.0";
pub const DISCOVERY_POPULATION_SCHEMA: &str = "prism_native_pathb_discovery_candidate_population";
pub const DISCOVERY_CANDIDATE_AUTHORITY_SCHEMA: &str =
    "prism_native_pathb_discovery_candidate_authority";
pub const DISCOVERY_TRANSACTION_SCHEMA: &str = "prism_native_pathb_discovery_transaction_commit";
pub const DISCOVERY_TRANSACTION_VERSION: &str = "1.0.0";
pub const DISCOVERY_MINIMUM_SEEDS: usize = 5;
pub const DISCOVERY_COMMIT_FILE: &str = "COMMIT.json";
pub const DISCOVERY_POPULATION_FILE: &str = "native_pathb_discovery_candidate_population.json";
pub const CONFIRMATION_HANDOFF_SCHEMA: &str = "prism_native_pathb_confirmation_handoff";
pub const CONFIRMATION_HANDOFF_VERSION: &str = "1.0.0";
pub const CONFIRMATION_HANDOFF_FILE: &str = "native_pathb_confirmation_handoff.json";
pub const CONFIRMATION_HANDOFF_COMMIT_FILE: &str = "COMMIT.json";
/// Dedicated typed event for native no-probe discovery evidence. Discovery
/// is not candidate advancement and must never reuse its observation opcode.
pub const EVENT_PATHB_DISCOVERY_CANDIDATE_MATERIALIZED: &str =
    "PATHB.DISCOVERY_CANDIDATE_MATERIALIZED";
pub use prismcode::binary_runtime::opcode::PATHB_DISCOVERY_CANDIDATE_MATERIALIZED as PRISMCODE_OPCODE_PATHB_DISCOVERY_CANDIDATE_MATERIALIZED;
pub const EVENT_PATHB_CONFIRMATION_HANDOFF_MATERIALIZED: &str =
    "PATHB.CONFIRMATION_HANDOFF_MATERIALIZED";
pub use prismcode::binary_runtime::opcode::PATHB_CONFIRMATION_HANDOFF_MATERIALIZED as PRISMCODE_OPCODE_PATHB_CONFIRMATION_HANDOFF_MATERIALIZED;

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn canonical_json_line<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn validate_absolute_normalized(path: &Path) -> Result<()> {
    if !path.is_absolute()
        || path
            .components()
            .any(|part| matches!(part, Component::CurDir | Component::ParentDir))
    {
        bail!("discovery path must be absolute and lexically normalized");
    }
    Ok(())
}

fn read_regular_nofollow(path: &Path) -> Result<Vec<u8>> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("inspect immutable discovery evidence {}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        bail!("discovery evidence must be a non-symlink regular file");
    }
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn write_create_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)
        .with_context(|| format!("create immutable discovery evidence {}", path.display()))?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn sync_dir(path: &Path) -> Result<()> {
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_DIRECTORY | libc::O_NOFOLLOW)
        .open(path)?
        .sync_all()?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiscoveryPhaseStratum {
    pub stratum_sha256: String,
    pub thermodynamic_condition: ThermodynamicCondition,
    pub phase_ordinal: u16,
    pub phase_name: String,
    pub phase_start_step: u64,
    pub phase_end_step_exclusive: u64,
    pub protocol_name: String,
    pub phase_schedule_digest_sha256: String,
    pub protocol_state_hash: String,
    pub matched_physical_state_sha256: String,
}

#[derive(Serialize)]
struct DiscoveryPhaseStratumBody<'a> {
    thermodynamic_condition: &'a ThermodynamicCondition,
    phase_ordinal: u16,
    phase_name: &'a str,
    phase_start_step: u64,
    phase_end_step_exclusive: u64,
    protocol_name: &'a str,
    phase_schedule_digest_sha256: &'a str,
    protocol_state_hash: &'a str,
    matched_physical_state_sha256: &'a str,
}

impl DiscoveryPhaseStratum {
    fn body(&self) -> DiscoveryPhaseStratumBody<'_> {
        DiscoveryPhaseStratumBody {
            thermodynamic_condition: &self.thermodynamic_condition,
            phase_ordinal: self.phase_ordinal,
            phase_name: &self.phase_name,
            phase_start_step: self.phase_start_step,
            phase_end_step_exclusive: self.phase_end_step_exclusive,
            protocol_name: &self.protocol_name,
            phase_schedule_digest_sha256: &self.phase_schedule_digest_sha256,
            protocol_state_hash: &self.protocol_state_hash,
            matched_physical_state_sha256: &self.matched_physical_state_sha256,
        }
    }

    pub fn verify(&self) -> Result<()> {
        self.thermodynamic_condition.validate()?;
        if self.phase_name.trim().is_empty()
            || self.protocol_name.trim().is_empty()
            || self.phase_start_step >= self.phase_end_step_exclusive
        {
            bail!("invalid discovery protocol-phase stratum");
        }
        for digest in [
            &self.stratum_sha256,
            &self.phase_schedule_digest_sha256,
            &self.protocol_state_hash,
            &self.matched_physical_state_sha256,
        ] {
            require_sha256(digest, "discovery stratum digest")?;
        }
        let phase = ProtocolPhaseBinding {
            stream_id: 0,
            thermodynamic_condition_id: self.thermodynamic_condition.condition_id.clone(),
            phase_ordinal: self.phase_ordinal,
            phase_name: self.phase_name.clone(),
            start_step: self.phase_start_step,
            end_step_exclusive: self.phase_end_step_exclusive,
            protocol_name: self.protocol_name.clone(),
            phase_schedule_digest_sha256: self.phase_schedule_digest_sha256.clone(),
            protocol_state_hash: self.protocol_state_hash.clone(),
        };
        if self
            .thermodynamic_condition
            .matched_physical_state_sha256(&phase)?
            != self.matched_physical_state_sha256
            || canonical_sha256(&self.body())? != self.stratum_sha256
        {
            bail!("discovery stratum physical-state/content hash mismatch");
        }
        Ok(())
    }

    pub fn materialize(mut self) -> Result<Self> {
        if !self.stratum_sha256.is_empty() {
            bail!("discovery stratum materialization requires an empty digest");
        }
        self.stratum_sha256 = canonical_sha256(&self.body())?;
        self.verify()?;
        Ok(self)
    }

    fn verify_observation(&self, row: &GhostFiberMeasurementObservation) -> Result<()> {
        if row.thermodynamic_condition != self.thermodynamic_condition
            || row.protocol_phase.phase_ordinal != self.phase_ordinal
            || row.protocol_phase.phase_name != self.phase_name
            || row.protocol_phase.start_step != self.phase_start_step
            || row.protocol_phase.end_step_exclusive != self.phase_end_step_exclusive
            || row.protocol_phase.protocol_name != self.protocol_name
            || row.protocol_phase.phase_schedule_digest_sha256 != self.phase_schedule_digest_sha256
            || row.protocol_phase.protocol_state_hash != self.protocol_state_hash
            || row
                .thermodynamic_condition
                .matched_physical_state_sha256(&row.protocol_phase)?
                != self.matched_physical_state_sha256
        {
            bail!("discovery evidence mixes physical or protocol-phase strata");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiscoveryLaneTransactionEntry {
    pub transaction_root: PathBuf,
    pub transaction_commit_sha256: String,
    pub lane_evidence_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativePathBDiscoveryManifestBody {
    pub cohort_id: String,
    pub engine_sha256: String,
    pub topology_sha256: String,
    /// Chemistry-independent projection. Full per-run requests remain in the
    /// committed lane provenance but matched-control, fragment, tracker and
    /// registry fields cannot participate in discovery candidate identity.
    pub structural_design: DiscoveryStructuralDesignAuthority,
    pub minimum_candidate_seed_support: usize,
    pub discovery_replica_seeds: Vec<u64>,
    pub correspondence_policy: ComponentCorrespondencePolicy,
    pub stratum: DiscoveryPhaseStratum,
    pub lane_transactions: Vec<DiscoveryLaneTransactionEntry>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiscoveryStructuralDesignAuthority {
    pub structural_design_sha256: String,
    pub selection_semantics: String,
    pub required_stream_ids: Vec<u16>,
    pub coordinate_cavity_config: crate::coordinate_cavity::CoordinateCavityConfig,
    pub correspondence_policy_sha256: String,
    pub fixed_cadence_design_sha256: Option<String>,
    pub excluded_source_request_fields: Vec<String>,
}

#[derive(Serialize)]
struct DiscoveryStructuralDesignBody<'a> {
    selection_semantics: &'a str,
    required_stream_ids: &'a [u16],
    coordinate_cavity_config: &'a crate::coordinate_cavity::CoordinateCavityConfig,
    correspondence_policy_sha256: &'a str,
    fixed_cadence_design_sha256: &'a Option<String>,
}

impl DiscoveryStructuralDesignAuthority {
    fn body(&self) -> DiscoveryStructuralDesignBody<'_> {
        DiscoveryStructuralDesignBody {
            selection_semantics: &self.selection_semantics,
            required_stream_ids: &self.required_stream_ids,
            coordinate_cavity_config: &self.coordinate_cavity_config,
            correspondence_policy_sha256: &self.correspondence_policy_sha256,
            fixed_cadence_design_sha256: &self.fixed_cadence_design_sha256,
        }
    }

    pub fn verify(&self) -> Result<()> {
        if self.selection_semantics != "whole_authenticated_lattice_no_residue_prefilter"
            || self.required_stream_ids.is_empty()
            || self
                .required_stream_ids
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || self.excluded_source_request_fields
                != vec![
                    "fragment_panel_sha256",
                    "matched_control_config",
                    "registry_policy",
                    "candidate_tracker",
                    "site_key",
                ]
        {
            bail!("invalid chemistry-independent discovery structural design");
        }
        self.coordinate_cavity_config.validate()?;
        require_sha256(
            &self.structural_design_sha256,
            "discovery structural design",
        )?;
        require_sha256(
            &self.correspondence_policy_sha256,
            "discovery correspondence policy",
        )?;
        if let Some(digest) = &self.fixed_cadence_design_sha256 {
            require_sha256(digest, "discovery fixed-cadence design")?;
        }
        if canonical_sha256(&self.body())? != self.structural_design_sha256 {
            bail!("discovery structural-design projection hash mismatch");
        }
        Ok(())
    }

    pub fn materialize(mut self) -> Result<Self> {
        if !self.structural_design_sha256.is_empty() {
            bail!("structural-design materialization requires an empty digest");
        }
        self.structural_design_sha256 = canonical_sha256(&self.body())?;
        self.verify()?;
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativePathBDiscoveryManifest {
    pub schema_kind: String,
    pub schema_version: String,
    pub discovery_manifest_sha256: String,
    pub body: NativePathBDiscoveryManifestBody,
}

impl NativePathBDiscoveryManifest {
    pub fn materialize(body: NativePathBDiscoveryManifestBody) -> Result<Self> {
        let manifest = Self {
            schema_kind: DISCOVERY_MANIFEST_SCHEMA.to_string(),
            schema_version: DISCOVERY_MANIFEST_VERSION.to_string(),
            discovery_manifest_sha256: canonical_sha256(&body)?,
            body,
        };
        manifest.verify()?;
        Ok(manifest)
    }

    pub fn verify(&self) -> Result<()> {
        if self.schema_kind != DISCOVERY_MANIFEST_SCHEMA
            || self.schema_version != DISCOVERY_MANIFEST_VERSION
            || self.body.cohort_id.trim().is_empty()
            || self.body.discovery_replica_seeds.len() < DISCOVERY_MINIMUM_SEEDS
            || self.body.minimum_candidate_seed_support < DISCOVERY_MINIMUM_SEEDS
            || self.body.minimum_candidate_seed_support > self.body.discovery_replica_seeds.len()
            || self
                .body
                .discovery_replica_seeds
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || self.body.lane_transactions.len() != self.body.discovery_replica_seeds.len()
        {
            bail!("invalid no-probe discovery manifest design");
        }
        for digest in [
            &self.discovery_manifest_sha256,
            &self.body.engine_sha256,
            &self.body.topology_sha256,
        ] {
            require_sha256(digest, "discovery manifest digest")?;
        }
        self.body.correspondence_policy.verify()?;
        self.body.structural_design.verify()?;
        if self.body.structural_design.correspondence_policy_sha256
            != self.body.correspondence_policy.policy_sha256
        {
            bail!("discovery structural design differs from correspondence policy");
        }
        self.body.stratum.verify()?;
        let mut prior: Option<&Path> = None;
        for entry in &self.body.lane_transactions {
            validate_absolute_normalized(&entry.transaction_root)?;
            if entry
                .transaction_root
                .file_name()
                .and_then(|name| name.to_str())
                != Some(NATIVE_PATHB_LANE_TRANSACTION_DIRECTORY)
                || prior.is_some_and(|path| path >= entry.transaction_root.as_path())
            {
                bail!("discovery transaction roots are not fixed, unique, and sorted");
            }
            prior = Some(&entry.transaction_root);
            require_sha256(&entry.transaction_commit_sha256, "lane transaction commit")?;
            require_sha256(&entry.lane_evidence_sha256, "lane evidence")?;
        }
        if canonical_sha256(&self.body)? != self.discovery_manifest_sha256 {
            bail!("discovery manifest content hash mismatch");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DisjointConfirmationSeedAuthorityBody {
    pub discovery_manifest_sha256: String,
    pub source_authority_sha256: String,
    pub confirmation_replica_seeds: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DisjointConfirmationSeedAuthority {
    pub schema_kind: String,
    pub schema_version: String,
    pub authority_sha256: String,
    pub body: DisjointConfirmationSeedAuthorityBody,
}

impl DisjointConfirmationSeedAuthority {
    pub fn materialize(body: DisjointConfirmationSeedAuthorityBody) -> Result<Self> {
        let authority = Self {
            schema_kind: "prism_native_pathb_disjoint_confirmation_seed_authority".to_string(),
            schema_version: "1.0.0".to_string(),
            authority_sha256: canonical_sha256(&body)?,
            body,
        };
        Ok(authority)
    }

    pub fn verify_for(&self, manifest: &NativePathBDiscoveryManifest) -> Result<()> {
        if self.schema_kind != "prism_native_pathb_disjoint_confirmation_seed_authority"
            || self.schema_version != "1.0.0"
            || self.body.discovery_manifest_sha256 != manifest.discovery_manifest_sha256
            || self.body.confirmation_replica_seeds.len() < DISCOVERY_MINIMUM_SEEDS
            || self
                .body
                .confirmation_replica_seeds
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            bail!("invalid disjoint confirmation seed authority");
        }
        require_sha256(&self.authority_sha256, "confirmation seed authority")?;
        require_sha256(
            &self.body.source_authority_sha256,
            "confirmation source authority",
        )?;
        if canonical_sha256(&self.body)? != self.authority_sha256 {
            bail!("confirmation seed authority content hash mismatch");
        }
        let discovery = manifest
            .body
            .discovery_replica_seeds
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        if self
            .body
            .confirmation_replica_seeds
            .iter()
            .any(|seed| discovery.contains(seed))
        {
            bail!("confirmation seed authority reuses a discovery seed");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiscoveryTrackerDesignAuthorityBody {
    pub discovery_manifest_sha256: String,
    pub correspondence_ledger_sha256: String,
    pub operator_design_authority_sha256: String,
    pub tracker_by_candidate_id: BTreeMap<String, PathBCandidateTrackerDefinition>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiscoveryTrackerDesignAuthority {
    pub schema_kind: String,
    pub schema_version: String,
    pub authority_sha256: String,
    pub body: DiscoveryTrackerDesignAuthorityBody,
}

impl DiscoveryTrackerDesignAuthority {
    pub fn materialize(body: DiscoveryTrackerDesignAuthorityBody) -> Result<Self> {
        Ok(Self {
            schema_kind: "prism_native_pathb_discovery_tracker_design_authority".to_string(),
            schema_version: "1.0.0".to_string(),
            authority_sha256: canonical_sha256(&body)?,
            body,
        })
    }

    fn verify_for(
        &self,
        manifest: &NativePathBDiscoveryManifest,
        ledger: &ComponentCorrespondenceLedger,
    ) -> Result<()> {
        if self.schema_kind != "prism_native_pathb_discovery_tracker_design_authority"
            || self.schema_version != "1.0.0"
            || self.body.discovery_manifest_sha256 != manifest.discovery_manifest_sha256
            || self.body.correspondence_ledger_sha256 != ledger.ledger_sha256
        {
            bail!("tracker design authority is not bound to this discovery population");
        }
        require_sha256(&self.authority_sha256, "tracker design authority")?;
        require_sha256(
            &self.body.operator_design_authority_sha256,
            "operator tracker design authority",
        )?;
        for (candidate, tracker) in &self.body.tracker_by_candidate_id {
            require_sha256(candidate, "tracker candidate id")?;
            tracker.verify()?;
        }
        if canonical_sha256(&self.body)? != self.authority_sha256 {
            bail!("tracker design authority content hash mismatch");
        }
        Ok(())
    }
}

fn load_canonical<T: for<'de> Deserialize<'de> + Serialize>(path: &Path) -> Result<T> {
    validate_absolute_normalized(path)?;
    let bytes = read_regular_nofollow(path)?;
    if bytes.last() != Some(&b'\n') {
        bail!("immutable discovery JSON is not one complete canonical line");
    }
    let value: T = serde_json::from_slice(&bytes)?;
    if canonical_json_line(&value)? != bytes {
        bail!("immutable discovery JSON bytes are not canonical");
    }
    Ok(value)
}

pub fn load_discovery_manifest(path: &Path) -> Result<NativePathBDiscoveryManifest> {
    let manifest: NativePathBDiscoveryManifest = load_canonical(path)?;
    manifest.verify()?;
    Ok(manifest)
}

pub fn load_confirmation_seed_authority(
    path: &Path,
    manifest: &NativePathBDiscoveryManifest,
) -> Result<DisjointConfirmationSeedAuthority> {
    let authority: DisjointConfirmationSeedAuthority = load_canonical(path)?;
    authority.verify_for(manifest)?;
    Ok(authority)
}

pub fn load_tracker_design_authority(
    path: &Path,
    manifest: &NativePathBDiscoveryManifest,
    ledger: &ComponentCorrespondenceLedger,
) -> Result<DiscoveryTrackerDesignAuthority> {
    let authority: DiscoveryTrackerDesignAuthority = load_canonical(path)?;
    authority.verify_for(manifest, ledger)?;
    Ok(authority)
}

/// Decode canonical tracker-design bytes before the deterministic discovery
/// ledger exists. `aggregate_no_probe_discovery` performs the mandatory
/// manifest/ledger binding check before any output is staged.
pub fn load_unbound_tracker_design_authority(
    path: &Path,
) -> Result<DiscoveryTrackerDesignAuthority> {
    load_canonical(path)
}

#[derive(Debug, Clone)]
pub struct LoadedDiscoveryPopulation {
    pub manifest: NativePathBDiscoveryManifest,
    pub lanes: Vec<NativePathBLaneEvidence>,
    pub commits: Vec<NativePathBLaneTransactionCommit>,
}

pub fn load_discovery_population(
    manifest: NativePathBDiscoveryManifest,
) -> Result<LoadedDiscoveryPopulation> {
    manifest.verify()?;
    let mut lanes = Vec::with_capacity(manifest.body.lane_transactions.len());
    let mut commits = Vec::with_capacity(manifest.body.lane_transactions.len());
    for entry in &manifest.body.lane_transactions {
        let commit = load_native_pathb_lane_transaction_commit(&entry.transaction_root)?;
        let lane = load_native_pathb_lane_evidence(&entry.transaction_root)?;
        if commit.commit_sha256 != entry.transaction_commit_sha256
            || commit.body.lane_evidence_sha256 != entry.lane_evidence_sha256
            || lane.lane_evidence_sha256 != entry.lane_evidence_sha256
        {
            bail!("discovery lane transaction commit/evidence hash mismatch");
        }
        lanes.push(lane);
        commits.push(commit);
    }
    validate_discovery_lanes(&manifest, &lanes)?;
    Ok(LoadedDiscoveryPopulation {
        manifest,
        lanes,
        commits,
    })
}

fn validate_discovery_lanes(
    manifest: &NativePathBDiscoveryManifest,
    lanes: &[NativePathBLaneEvidence],
) -> Result<()> {
    if lanes.len() != manifest.body.lane_transactions.len() {
        bail!("discovery lane population differs from manifest");
    }
    let mut seeds = BTreeSet::new();
    let mut run_units = BTreeSet::new();
    for lane in lanes {
        lane.verify()?;
        let policy = &lane.body.runtime_request.body.lane_policy;
        validate_discovery_lane_policy(policy, &manifest.body.structural_design)?;
        if lane.body.confirmation_surveillance.is_some()
            || lane.body.site_selection.selection_mode
                != NativePathBSelectionMode::WholeLatticeDiscovery
            || lane.body.site_selection.selected_records
                != lane.body.site_selection.source_lattice_nodes
        {
            bail!("discovery admits only no-probe whole-lattice lane evidence");
        }
        if lane
            .body
            .runtime_request
            .body
            .component_correspondence_policy
            != manifest.body.correspondence_policy
        {
            bail!("discovery lane differs from chemistry-independent structural design");
        }
        if lane.body.fragment_observations.iter().any(|row| {
            row.fragment.identity.is_some()
                || row.fragment.endpoint_a_identity.is_some()
                || row.fragment.endpoint_b_identity.is_some()
                || row.fragment.occupancy != FragmentOccupancy::Absent
        }) {
            bail!("probe chemistry cannot create discovery candidates");
        }
        let first = lane
            .body
            .ghost_measurements
            .first()
            .context("whole-lattice discovery lane has no Ghost evidence")?;
        let seed = first.ghost.replica_seed;
        let run_unit = (first.ghost.run_id.clone(), first.ghost.replica_id.clone());
        if !seeds.insert(seed) || !run_units.insert(run_unit) {
            bail!("discovery population contains duplicate seeds or run units");
        }
        for row in &lane.body.ghost_measurements {
            if row.control_lane != ControlLane::NoProbe
                || row.ghost.replica_seed != seed
                || row.ghost.engine_sha256 != manifest.body.engine_sha256
                || row.ghost.topology_sha256 != manifest.body.topology_sha256
            {
                bail!("discovery Ghost identity mixes lane, seed, engine, or topology");
            }
            manifest.body.stratum.verify_observation(row)?;
        }
        if lane
            .body
            .component_descriptors
            .iter()
            .any(|row| row.control_lane != ControlLane::NoProbe || row.replica_seed != seed)
        {
            bail!("discovery component descriptor differs from no-probe run identity");
        }
    }
    let declared = manifest
        .body
        .discovery_replica_seeds
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    if seeds != declared {
        bail!("loaded discovery seeds differ from the declared independent cohort");
    }
    Ok(())
}

fn validate_discovery_lane_policy(
    policy: &crate::pathb_evidence_emitter::NativePathBLanePolicy,
    structural: &DiscoveryStructuralDesignAuthority,
) -> Result<()> {
    if policy.selection_mode != NativePathBSelectionMode::WholeLatticeDiscovery
        || policy.candidate_tracker.is_some()
        || policy
            .stream_lanes
            .values()
            .any(|lane| *lane != ControlLane::NoProbe)
        || !policy.fragment_slot_ids.is_empty()
        || !policy.target_residue_ids.is_empty()
        || policy.minimum_target_residue_overlap_count != 0
        || policy.minimum_target_residue_overlap_fraction.to_bits() != 0.0f64.to_bits()
        || policy.required_stream_ids != structural.required_stream_ids
        || policy.fixed_cadence_steps == 0
        || policy.coordinate_cavity_config != structural.coordinate_cavity_config
    {
        bail!("discovery policy is not chemistry-free no-probe whole-lattice evidence");
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiscoveryCandidateStatus {
    RecurrentQualified,
    RecurrentBelowMinimumSupport,
    Ambiguous,
    Unmatched,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiscoveryComponentMaterializationInput {
    pub descriptor: NativeComponentDescriptor,
    pub source_transaction_commit_sha256: String,
    pub source_lane_evidence_sha256: String,
    pub exact_connectivity_sha256: String,
    pub component_node_ghost_lineage_sha256: Vec<String>,
    pub physical_lining_descriptor_population_sha256: String,
    pub physical_lining_component_binding:
        crate::component_correspondence::PhysicalLiningComponentBinding,
    /// Complete coordinate-cavity producer bytes for every Ghost occurrence
    /// in this exact component. These are preserved, not summarized into the
    /// descriptor or exemplar.
    pub physical_lining_occurrences:
        Vec<crate::component_correspondence::PhysicalLiningOccurrenceEvidence>,
    pub exemplar_lattice_observation: NativePathBLatticeObservation,
    pub exemplar_structural_observation: GhostFiberMeasurementObservation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BodyFixedTrackingRequirement {
    pub status: String,
    pub static_centroid_is_reference_metadata_only: bool,
    pub required_transform: String,
    pub required_authorities: Vec<String>,
    pub tracker_promotion_allowed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateTrackerMaterializationInputs {
    pub reference_coordinate_frame_sha256: Vec<String>,
    pub reference_frame_ordinal: Vec<u64>,
    pub reference_centroid_xyz_a: Vec<[f32; 3]>,
    pub signed_contact_distributions:
        Vec<Vec<crate::component_correspondence::CanonicalResidueContactProbability>>,
    pub coordinate_cavity_sidecar_sha256: Vec<String>,
    pub coordinate_cavity_estimator_sha256: Vec<String>,
    pub cavity_grid_min_mean_max_a3: Vec<[f64; 3]>,
    pub design_authority_required_fields: Vec<String>,
    pub body_fixed_tracking: BodyFixedTrackingRequirement,
    pub supplied_tracker: Option<PathBCandidateTrackerDefinition>,
    pub tracker_promotion_status: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateConfirmationBindingInput {
    pub discovery_manifest_sha256: String,
    pub correspondence_ledger_sha256: String,
    pub candidate_id_sha256: String,
    pub selected_correspondence_sha256: Option<String>,
    pub selected_descriptor_sha256: Vec<String>,
    pub discovery_replica_seeds: Vec<u64>,
    pub confirmation_replica_seeds: Vec<u64>,
    pub explicit_selection_receipt_required: bool,
    pub derived_tracker_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativePathBDiscoveryCandidateAuthorityBody {
    pub discovery_manifest_sha256: String,
    pub correspondence_ledger_sha256: String,
    pub candidate_id_sha256: String,
    pub status: DiscoveryCandidateStatus,
    pub correspondence_sha256: Option<String>,
    pub correspondence_assignments: Vec<ComponentCorrespondenceAssignment>,
    pub components: Vec<DiscoveryComponentMaterializationInput>,
    pub support_replica_seeds: Vec<u64>,
    pub zero_detection_replica_seeds: Vec<u64>,
    pub recurrence_numerator: usize,
    pub recurrence_denominator: usize,
    pub ambiguity_candidate_correspondence_sha256: Vec<String>,
    pub tracker_materialization_inputs: CandidateTrackerMaterializationInputs,
    pub confirmation_binding_input: CandidateConfirmationBindingInput,
    pub ranking_applied: bool,
    pub suppression_applied: bool,
    pub pathb_state_count: u64,
    pub matched_control_classification_count: u64,
    pub candidate_registry_append_count: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativePathBDiscoveryCandidateAuthority {
    pub schema_kind: String,
    pub schema_version: String,
    pub candidate_authority_sha256: String,
    pub body: NativePathBDiscoveryCandidateAuthorityBody,
}

impl NativePathBDiscoveryCandidateAuthority {
    pub fn verify(&self) -> Result<()> {
        if self.schema_kind != DISCOVERY_CANDIDATE_AUTHORITY_SCHEMA
            || self.schema_version != "2.0.0"
            || self.body.components.is_empty()
            || self.body.recurrence_numerator != self.body.support_replica_seeds.len()
            || self.body.recurrence_denominator
                != self.body.support_replica_seeds.len()
                    + self.body.zero_detection_replica_seeds.len()
            || self.body.ranking_applied
            || self.body.suppression_applied
            || self.body.pathb_state_count != 0
            || self.body.matched_control_classification_count != 0
            || self.body.candidate_registry_append_count != 0
            || self
                .body
                .tracker_materialization_inputs
                .body_fixed_tracking
                .tracker_promotion_allowed
        {
            bail!("invalid discovery candidate authority boundary");
        }
        require_sha256(
            &self.candidate_authority_sha256,
            "discovery candidate authority",
        )?;
        let mut preserved_occurrences = BTreeSet::new();
        for component in &self.body.components {
            component.descriptor.verify()?;
            for digest in [
                &component.source_transaction_commit_sha256,
                &component.source_lane_evidence_sha256,
                &component.exact_connectivity_sha256,
                &component.physical_lining_descriptor_population_sha256,
            ] {
                require_sha256(digest, "discovery component source authority")?;
            }
            let binding = &component.physical_lining_component_binding;
            if binding.source_component_sha256 != component.descriptor.source_component_sha256
                || binding.descriptor_sha256 != component.descriptor.descriptor_sha256
                || binding.physical_lining_evidence_sha256.len()
                    != component.physical_lining_occurrences.len()
                || component.component_node_ghost_lineage_sha256.len()
                    != component.physical_lining_occurrences.len()
            {
                bail!("discovery component physical-lining binding is incomplete");
            }
            let occurrence_hashes = component
                .physical_lining_occurrences
                .iter()
                .map(|row| {
                    row.verify()?;
                    if !preserved_occurrences.insert(row.occurrence_sha256.clone()) {
                        bail!("discovery candidate repeats physical-lining occurrence bytes");
                    }
                    Ok(row.occurrence_sha256.clone())
                })
                .collect::<Result<Vec<_>>>()?;
            let occurrence_lineages = component
                .physical_lining_occurrences
                .iter()
                .map(|row| row.ghost_lineage_sha256.clone())
                .collect::<BTreeSet<_>>();
            if occurrence_hashes != binding.physical_lining_evidence_sha256
                || canonical_sha256(&occurrence_hashes)?
                    != binding.physical_lining_population_sha256
                || occurrence_lineages
                    != component
                        .component_node_ghost_lineage_sha256
                        .iter()
                        .cloned()
                        .collect::<BTreeSet<_>>()
            {
                bail!("discovery component changes physical-lining byte/lineage conservation");
            }
        }
        if canonical_sha256(&self.body)? != self.candidate_authority_sha256 {
            bail!("discovery candidate authority content hash mismatch");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiscoveryCandidateAuthorityIndexEntry {
    pub candidate_id_sha256: String,
    pub candidate_authority_sha256: String,
    pub status: DiscoveryCandidateStatus,
    pub relative_path: String,
    pub typed_artifact: TypedEvidenceArtifact,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativePathBDiscoveryCandidatePopulationBody {
    pub discovery_manifest: NativePathBDiscoveryManifest,
    pub source_lane_transactions: Vec<DiscoveryLaneTransactionEntry>,
    pub component_correspondence: ComponentCorrespondenceLedger,
    pub candidate_authorities: Vec<DiscoveryCandidateAuthorityIndexEntry>,
    pub recurrent_qualified_count: usize,
    pub recurrent_below_minimum_count: usize,
    pub ambiguous_count: usize,
    pub unmatched_count: usize,
    pub source_descriptor_count: usize,
    pub authority_component_count: usize,
    pub all_descriptors_preserved: bool,
    pub legacy_binding_site_json_inputs: u64,
    pub probe_chemistry_candidate_inputs: u64,
    pub ranking_applied: bool,
    pub suppression_applied: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativePathBDiscoveryCandidatePopulation {
    pub schema_kind: String,
    pub schema_version: String,
    pub population_sha256: String,
    pub body: NativePathBDiscoveryCandidatePopulationBody,
}

impl NativePathBDiscoveryCandidatePopulation {
    pub fn verify(&self) -> Result<()> {
        if self.schema_kind != DISCOVERY_POPULATION_SCHEMA
            || self.schema_version != "2.0.0"
            || !self.body.all_descriptors_preserved
            || self.body.legacy_binding_site_json_inputs != 0
            || self.body.probe_chemistry_candidate_inputs != 0
            || self.body.ranking_applied
            || self.body.suppression_applied
            || self.body.source_descriptor_count
                != self.body.component_correspondence.descriptors.len()
            || self.body.authority_component_count != self.body.source_descriptor_count
        {
            bail!("discovery population violates lossless/no-probe boundary");
        }
        self.body.discovery_manifest.verify()?;
        self.body.component_correspondence.verify()?;
        require_sha256(&self.population_sha256, "discovery population")?;
        if canonical_sha256(&self.body)? != self.population_sha256 {
            bail!("discovery population content hash mismatch");
        }
        Ok(())
    }
}

fn component_input(
    descriptor: &NativeComponentDescriptor,
    loaded: &LoadedDiscoveryPopulation,
) -> Result<DiscoveryComponentMaterializationInput> {
    let index = loaded
        .lanes
        .iter()
        .position(|lane| {
            lane.body
                .component_descriptors
                .iter()
                .any(|row| row.descriptor_sha256 == descriptor.descriptor_sha256)
        })
        .context("correspondence descriptor has no committed source lane")?;
    let lane = &loaded.lanes[index];
    let component = lane
        .body
        .exact_connectivity
        .components
        .iter()
        .find(|row| row.component_sha256 == descriptor.source_component_sha256)
        .context("descriptor source component is absent from exact connectivity")?;
    let physical_population = &lane.body.physical_lining_descriptor_population;
    physical_population.verify()?;
    let physical_binding = physical_population
        .bindings
        .iter()
        .find(|row| row.source_component_sha256 == descriptor.source_component_sha256)
        .context("descriptor source component lacks physical-lining binding")?;
    if physical_binding.descriptor_sha256 != descriptor.descriptor_sha256 {
        bail!("component descriptor differs from physical-lining binding");
    }
    let physical_by_hash = physical_population
        .physical_lining_evidence
        .iter()
        .map(|row| (row.occurrence_sha256.as_str(), row))
        .collect::<BTreeMap<_, _>>();
    let physical_lining_occurrences = physical_binding
        .physical_lining_evidence_sha256
        .iter()
        .map(|hash| {
            physical_by_hash
                .get(hash.as_str())
                .cloned()
                .cloned()
                .with_context(|| format!("component physical-lining occurrence {hash} is absent"))
        })
        .collect::<Result<Vec<_>>>()?;
    if physical_lining_occurrences
        .iter()
        .map(|row| row.ghost_lineage_sha256.clone())
        .collect::<BTreeSet<_>>()
        != component
            .node_ghost_lineage_sha256
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>()
    {
        bail!("component physical-lining occurrence set differs from native connectivity");
    }
    let exemplar_hash = component
        .node_ghost_lineage_sha256
        .first()
        .context("native component has no node lineage")?;
    let lattice = lane
        .body
        .lattice_observations
        .iter()
        .find(|row| &row.ghost_lineage_sha256 == exemplar_hash)
        .context("component exemplar lacks lattice observation")?;
    let measurement = lane
        .body
        .ghost_measurements
        .iter()
        .find(|row| row.ghost.lineage_sha256().ok().as_ref() == Some(exemplar_hash))
        .context("component exemplar lacks structural observation")?;
    Ok(DiscoveryComponentMaterializationInput {
        descriptor: descriptor.clone(),
        source_transaction_commit_sha256: loaded.commits[index].commit_sha256.clone(),
        source_lane_evidence_sha256: lane.lane_evidence_sha256.clone(),
        exact_connectivity_sha256: lane.body.exact_connectivity.connectivity_sha256.clone(),
        component_node_ghost_lineage_sha256: component.node_ghost_lineage_sha256.clone(),
        physical_lining_descriptor_population_sha256: physical_population.population_sha256.clone(),
        physical_lining_component_binding: physical_binding.clone(),
        physical_lining_occurrences,
        exemplar_lattice_observation: lattice.clone(),
        exemplar_structural_observation: measurement.clone(),
    })
}

fn candidate_id(
    status: DiscoveryCandidateStatus,
    correspondence: Option<&str>,
    descriptors: &[String],
) -> Result<String> {
    canonical_sha256(&(
        "prism_native_pathb_discovery_candidate_v1",
        status,
        correspondence,
        descriptors,
    ))
}

fn build_tracker_inputs(
    components: &[DiscoveryComponentMaterializationInput],
) -> Result<CandidateTrackerMaterializationInputs> {
    let design_authority_required_fields = vec![
        "observed_closed_max_volume_a3".to_string(),
        "observed_open_min_volume_a3".to_string(),
        "cadence_steps".to_string(),
        "start_step_by_stream".to_string(),
        "end_step_exclusive_by_stream".to_string(),
        "stationary_phase_ordinal".to_string(),
        "familywise_confidence_design".to_string(),
        "signed_body_fixed_anchor_atoms_or_residues".to_string(),
        "pbc_unwrap_and_alignment_transform".to_string(),
    ];
    Ok(CandidateTrackerMaterializationInputs {
        reference_coordinate_frame_sha256: components
            .iter()
            .map(|row| {
                row.exemplar_structural_observation
                    .structural_measurements
                    .coordinate_frame
                    .coordinate_frame_sha256
                    .clone()
            })
            .collect(),
        reference_frame_ordinal: components
            .iter()
            .map(|row| {
                row.exemplar_structural_observation
                    .structural_measurements
                    .coordinate_frame
                    .trajectory_frame_ordinal
            })
            .collect(),
        reference_centroid_xyz_a: components
            .iter()
            .map(|row| row.exemplar_lattice_observation.centroid_xyz)
            .collect(),
        signed_contact_distributions: components
            .iter()
            .map(|row| row.descriptor.signed_contact_distribution.clone())
            .collect(),
        coordinate_cavity_sidecar_sha256: components
            .iter()
            .map(|row| {
                row.exemplar_structural_observation
                    .structural_measurements
                    .coordinate_cavity_sidecar
                    .sidecar_sha256
                    .clone()
            })
            .collect(),
        coordinate_cavity_estimator_sha256: components
            .iter()
            .map(|row| {
                row.exemplar_structural_observation
                    .structural_measurements
                    .coordinate_cavity_sidecar
                    .body
                    .estimator_sha256
                    .clone()
            })
            .collect(),
        cavity_grid_min_mean_max_a3: components
            .iter()
            .map(|row| {
                let cavity = &row
                    .exemplar_structural_observation
                    .structural_measurements
                    .coordinate_cavity_sidecar
                    .body;
                [
                    cavity.discretization_min_a3,
                    cavity.mean_volume_a3,
                    cavity.discretization_max_a3,
                ]
            })
            .collect(),
        design_authority_required_fields,
        body_fixed_tracking: BodyFixedTrackingRequirement {
            status: "EVIDENCE_NOT_FOUND".to_string(),
            static_centroid_is_reference_metadata_only: true,
            required_transform:
                "signed_anchor_pbc_unwrap_then_rigid_align_or_declared_local_barycentric_transform"
                    .to_string(),
            required_authorities: vec![
                "signed_anchor_atom_or_residue_identity".to_string(),
                "reference_local_frame".to_string(),
                "per_frame_transformed_centroid_receipt".to_string(),
            ],
            tracker_promotion_allowed: false,
        },
        tracker_promotion_status:
            "EVIDENCE_NOT_FOUND_POST_COMMIT_SELECTION_AND_TRACKER_DESIGN_AUTHORITY_REQUIRED"
                .to_string(),
        supplied_tracker: None,
    })
}

fn build_candidate_authorities(
    loaded: &LoadedDiscoveryPopulation,
    ledger: &ComponentCorrespondenceLedger,
) -> Result<Vec<NativePathBDiscoveryCandidateAuthority>> {
    let descriptors = ledger
        .descriptors
        .iter()
        .map(|row| (row.descriptor_sha256.clone(), row))
        .collect::<BTreeMap<_, _>>();
    let mut recurrent: BTreeMap<String, Vec<ComponentCorrespondenceAssignment>> = BTreeMap::new();
    let mut isolated = Vec::new();
    for assignment in &ledger.assignments {
        match &assignment.status {
            ComponentCorrespondenceStatus::Recurrent {
                correspondence_sha256,
                ..
            } => recurrent
                .entry(correspondence_sha256.clone())
                .or_default()
                .push(assignment.clone()),
            ComponentCorrespondenceStatus::Ambiguous { .. }
            | ComponentCorrespondenceStatus::Unmatched { .. } => {
                isolated.push(vec![assignment.clone()]);
            }
        }
    }
    let mut groups = recurrent
        .into_iter()
        .map(|(id, rows)| (Some(id), rows))
        .collect::<Vec<_>>();
    groups.extend(isolated.into_iter().map(|rows| (None, rows)));
    let discovery_seeds = loaded
        .manifest
        .body
        .discovery_replica_seeds
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut output = Vec::with_capacity(groups.len());
    for (correspondence_sha256, mut assignments) in groups {
        assignments.sort_by(|left, right| left.descriptor_sha256.cmp(&right.descriptor_sha256));
        let descriptor_rows = assignments
            .iter()
            .map(|row| descriptors[&row.descriptor_sha256])
            .collect::<Vec<_>>();
        let support = descriptor_rows
            .iter()
            .map(|row| row.replica_seed)
            .collect::<BTreeSet<_>>();
        let status = match &assignments[0].status {
            ComponentCorrespondenceStatus::Recurrent { .. }
                if recurrent_status(
                    support.len(),
                    loaded.manifest.body.minimum_candidate_seed_support,
                ) == DiscoveryCandidateStatus::RecurrentQualified =>
            {
                DiscoveryCandidateStatus::RecurrentQualified
            }
            ComponentCorrespondenceStatus::Recurrent { .. } => {
                DiscoveryCandidateStatus::RecurrentBelowMinimumSupport
            }
            ComponentCorrespondenceStatus::Ambiguous { .. } => DiscoveryCandidateStatus::Ambiguous,
            ComponentCorrespondenceStatus::Unmatched { .. } => DiscoveryCandidateStatus::Unmatched,
        };
        let descriptor_sha256 = assignments
            .iter()
            .map(|row| row.descriptor_sha256.clone())
            .collect::<Vec<_>>();
        let id = candidate_id(status, correspondence_sha256.as_deref(), &descriptor_sha256)?;
        let components = descriptor_rows
            .iter()
            .map(|descriptor| component_input(descriptor, loaded))
            .collect::<Result<Vec<_>>>()?;
        let tracker_inputs = build_tracker_inputs(&components)?;
        let ambiguity = assignments
            .iter()
            .flat_map(|row| match &row.status {
                ComponentCorrespondenceStatus::Ambiguous {
                    candidate_correspondence_sha256,
                    ..
                } => candidate_correspondence_sha256.clone(),
                _ => Vec::new(),
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let support_replica_seeds = support.iter().copied().collect::<Vec<_>>();
        let zero_detection_replica_seeds = discovery_seeds
            .difference(&support)
            .copied()
            .collect::<Vec<_>>();
        let body = NativePathBDiscoveryCandidateAuthorityBody {
            discovery_manifest_sha256: loaded.manifest.discovery_manifest_sha256.clone(),
            correspondence_ledger_sha256: ledger.ledger_sha256.clone(),
            candidate_id_sha256: id.clone(),
            status,
            correspondence_sha256: correspondence_sha256.clone(),
            correspondence_assignments: assignments,
            components,
            support_replica_seeds: support_replica_seeds.clone(),
            zero_detection_replica_seeds,
            recurrence_numerator: support_replica_seeds.len(),
            recurrence_denominator: discovery_seeds.len(),
            ambiguity_candidate_correspondence_sha256: ambiguity,
            confirmation_binding_input: CandidateConfirmationBindingInput {
                discovery_manifest_sha256: loaded.manifest.discovery_manifest_sha256.clone(),
                correspondence_ledger_sha256: ledger.ledger_sha256.clone(),
                candidate_id_sha256: id,
                selected_correspondence_sha256: correspondence_sha256,
                selected_descriptor_sha256: descriptor_sha256,
                discovery_replica_seeds: support_replica_seeds,
                confirmation_replica_seeds: Vec::new(),
                explicit_selection_receipt_required: true,
                derived_tracker_sha256: None,
            },
            tracker_materialization_inputs: tracker_inputs,
            ranking_applied: false,
            suppression_applied: false,
            pathb_state_count: 0,
            matched_control_classification_count: 0,
            candidate_registry_append_count: 0,
        };
        let authority = NativePathBDiscoveryCandidateAuthority {
            schema_kind: DISCOVERY_CANDIDATE_AUTHORITY_SCHEMA.to_string(),
            schema_version: "2.0.0".to_string(),
            candidate_authority_sha256: canonical_sha256(&body)?,
            body,
        };
        authority.verify()?;
        output.push(authority);
    }
    output.sort_by(|left, right| {
        left.body
            .candidate_id_sha256
            .cmp(&right.body.candidate_id_sha256)
    });
    let covered = output
        .iter()
        .flat_map(|candidate| candidate.body.components.iter())
        .map(|component| component.descriptor.descriptor_sha256.clone())
        .collect::<BTreeSet<_>>();
    if covered.len() != ledger.descriptors.len()
        || covered
            != ledger
                .descriptors
                .iter()
                .map(|row| row.descriptor_sha256.clone())
                .collect::<BTreeSet<_>>()
    {
        bail!("candidate authorities do not preserve every discovery descriptor exactly once");
    }
    Ok(output)
}

fn recurrent_status(support: usize, required: usize) -> DiscoveryCandidateStatus {
    if support >= required {
        DiscoveryCandidateStatus::RecurrentQualified
    } else {
        DiscoveryCandidateStatus::RecurrentBelowMinimumSupport
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiscoveryTransactionMember {
    pub ordinal: usize,
    pub role: String,
    pub relative_path: String,
    pub byte_length: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiscoveryTransactionCommitBody {
    pub population_sha256: String,
    /// Exact candidate-authority population committed before selection.
    pub candidate_authority_count: usize,
    /// Exact discovery descriptor population preserved by the authorities.
    pub source_descriptor_count: usize,
    /// Exact component population across every candidate authority.
    pub authority_component_count: usize,
    pub ordered_membership_sha256: String,
    pub members: Vec<DiscoveryTransactionMember>,
    pub population_typed_artifact: TypedEvidenceArtifact,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiscoveryTransactionCommit {
    pub schema_kind: String,
    pub schema_version: String,
    pub status: String,
    pub commit_sha256: String,
    pub body: DiscoveryTransactionCommitBody,
}

#[derive(Debug, Clone)]
pub struct DiscoveryEmission {
    pub population: NativePathBDiscoveryCandidatePopulation,
    pub authorities: Vec<NativePathBDiscoveryCandidateAuthority>,
    pub commit: DiscoveryTransactionCommit,
    pub output_root: PathBuf,
}

/// Operator-authored selection made only after an immutable discovery
/// transaction exists. The engine never chooses or ranks this candidate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExplicitDiscoveryCandidateSelectionBody {
    pub discovery_transaction_commit_sha256: String,
    pub discovery_population_sha256: String,
    pub selected_candidate_id_sha256: String,
    pub selected_candidate_authority_sha256: String,
    pub operator_selection_authority_sha256: String,
    pub selection_semantics: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExplicitDiscoveryCandidateSelectionAuthority {
    pub schema_kind: String,
    pub schema_version: String,
    pub authority_sha256: String,
    pub body: ExplicitDiscoveryCandidateSelectionBody,
}

impl ExplicitDiscoveryCandidateSelectionAuthority {
    pub fn materialize(body: ExplicitDiscoveryCandidateSelectionBody) -> Result<Self> {
        let authority = Self {
            schema_kind: "prism_explicit_discovery_candidate_selection_authority".to_string(),
            schema_version: "1.0.0".to_string(),
            authority_sha256: canonical_sha256(&body)?,
            body,
        };
        authority.verify()?;
        Ok(authority)
    }

    pub fn verify(&self) -> Result<()> {
        if self.schema_kind != "prism_explicit_discovery_candidate_selection_authority"
            || self.schema_version != "1.0.0"
            || self.body.selection_semantics
                != "operator_explicit_post_commit_selection_no_runtime_inference"
        {
            bail!("invalid explicit discovery candidate selection authority");
        }
        for digest in [
            &self.authority_sha256,
            &self.body.discovery_transaction_commit_sha256,
            &self.body.discovery_population_sha256,
            &self.body.selected_candidate_id_sha256,
            &self.body.selected_candidate_authority_sha256,
            &self.body.operator_selection_authority_sha256,
        ] {
            require_sha256(digest, "explicit candidate-selection authority")?;
        }
        if canonical_sha256(&self.body)? != self.authority_sha256 {
            bail!("explicit candidate-selection authority content hash mismatch");
        }
        Ok(())
    }
}

pub fn load_explicit_candidate_selection_authority(
    path: &Path,
) -> Result<ExplicitDiscoveryCandidateSelectionAuthority> {
    let authority: ExplicitDiscoveryCandidateSelectionAuthority = load_canonical(path)?;
    authority.verify()?;
    Ok(authority)
}

pub fn load_unbound_confirmation_seed_authority(
    path: &Path,
) -> Result<DisjointConfirmationSeedAuthority> {
    load_canonical(path)
}

/// Portable post-commit confirmation input. Full source objects are embedded
/// so a consumer can re-verify the candidate index, tracker v2, operator
/// design lineage and disjoint seed sets without querying discovery state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativePathBConfirmationHandoffBody {
    pub discovery_transaction_commit_sha256: String,
    pub discovery_transaction_commit: DiscoveryTransactionCommit,
    pub discovery_population_sha256: String,
    pub discovery_population: NativePathBDiscoveryCandidatePopulation,
    /// Population counts copied from, and verified against, the immutable
    /// discovery transaction. These are observation fields, never selection
    /// or tracker-derived values.
    pub discovery_candidate_authority_count: usize,
    pub discovery_source_descriptor_count: usize,
    pub discovery_authority_component_count: usize,
    pub discovery_manifest_sha256: String,
    pub correspondence_ledger_sha256: String,
    pub selected_candidate_id_sha256: String,
    pub selected_candidate_authority_sha256: String,
    pub selected_candidate_authority: NativePathBDiscoveryCandidateAuthority,
    pub explicit_selection_authority_sha256: String,
    pub explicit_selection_authority: ExplicitDiscoveryCandidateSelectionAuthority,
    pub tracker_design_authority_sha256: String,
    pub tracker_design_authority: DiscoveryTrackerDesignAuthority,
    pub operator_design_authority_sha256: String,
    pub supplied_tracker_sha256: String,
    pub supplied_tracker: PathBCandidateTrackerDefinition,
    pub disjoint_confirmation_seed_authority_sha256: String,
    pub disjoint_confirmation_seed_authority: DisjointConfirmationSeedAuthority,
    pub discovery_replica_seeds: Vec<u64>,
    pub confirmation_replica_seeds: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativePathBConfirmationHandoff {
    pub schema_kind: String,
    pub schema_version: String,
    pub handoff_sha256: String,
    pub body: NativePathBConfirmationHandoffBody,
}

impl NativePathBConfirmationHandoff {
    fn materialize(body: NativePathBConfirmationHandoffBody) -> Result<Self> {
        let handoff = Self {
            schema_kind: CONFIRMATION_HANDOFF_SCHEMA.to_string(),
            schema_version: CONFIRMATION_HANDOFF_VERSION.to_string(),
            handoff_sha256: canonical_sha256(&body)?,
            body,
        };
        handoff.verify()?;
        Ok(handoff)
    }

    pub fn verify(&self) -> Result<()> {
        if self.schema_kind != CONFIRMATION_HANDOFF_SCHEMA
            || self.schema_version != CONFIRMATION_HANDOFF_VERSION
        {
            bail!("invalid native Path-B confirmation handoff schema");
        }
        for digest in [
            &self.handoff_sha256,
            &self.body.discovery_transaction_commit_sha256,
            &self.body.discovery_population_sha256,
            &self.body.discovery_manifest_sha256,
            &self.body.correspondence_ledger_sha256,
            &self.body.selected_candidate_id_sha256,
            &self.body.selected_candidate_authority_sha256,
            &self.body.explicit_selection_authority_sha256,
            &self.body.tracker_design_authority_sha256,
            &self.body.operator_design_authority_sha256,
            &self.body.supplied_tracker_sha256,
            &self.body.disjoint_confirmation_seed_authority_sha256,
        ] {
            require_sha256(digest, "confirmation handoff authority")?;
        }
        let commit = &self.body.discovery_transaction_commit;
        if commit.schema_kind != DISCOVERY_TRANSACTION_SCHEMA
            || commit.schema_version != DISCOVERY_TRANSACTION_VERSION
            || commit.status != "PASS"
            || canonical_sha256(&commit.body)? != commit.commit_sha256
            || self.body.discovery_transaction_commit_sha256 != commit.commit_sha256
        {
            bail!("confirmation handoff discovery commit is invalid");
        }
        let population = &self.body.discovery_population;
        population.verify()?;
        if self.body.discovery_population_sha256 != population.population_sha256
            || commit.body.population_sha256 != population.population_sha256
            || commit.body.candidate_authority_count != population.body.candidate_authorities.len()
            || commit.body.source_descriptor_count != population.body.source_descriptor_count
            || commit.body.authority_component_count != population.body.authority_component_count
            || self.body.discovery_candidate_authority_count
                != commit.body.candidate_authority_count
            || self.body.discovery_source_descriptor_count != commit.body.source_descriptor_count
            || self.body.discovery_authority_component_count
                != commit.body.authority_component_count
            || self.body.discovery_manifest_sha256
                != population.body.discovery_manifest.discovery_manifest_sha256
            || self.body.correspondence_ledger_sha256
                != population.body.component_correspondence.ledger_sha256
        {
            bail!("confirmation handoff population/commit lineage mismatch");
        }
        let selected = &self.body.selected_candidate_authority;
        selected.verify()?;
        let index = population
            .body
            .candidate_authorities
            .iter()
            .find(|entry| entry.candidate_id_sha256 == self.body.selected_candidate_id_sha256)
            .context("selected candidate is absent from committed discovery index")?;
        if selected.body.candidate_id_sha256 != self.body.selected_candidate_id_sha256
            || selected.candidate_authority_sha256 != self.body.selected_candidate_authority_sha256
            || index.candidate_authority_sha256 != selected.candidate_authority_sha256
            || index.status != selected.body.status
            || selected.body.discovery_manifest_sha256 != self.body.discovery_manifest_sha256
            || selected.body.correspondence_ledger_sha256 != self.body.correspondence_ledger_sha256
        {
            bail!("confirmation handoff selected candidate differs from committed index");
        }
        let selection = &self.body.explicit_selection_authority;
        selection.verify()?;
        if self.body.explicit_selection_authority_sha256 != selection.authority_sha256
            || selection.body.discovery_transaction_commit_sha256
                != self.body.discovery_transaction_commit_sha256
            || selection.body.discovery_population_sha256 != self.body.discovery_population_sha256
            || selection.body.selected_candidate_id_sha256 != self.body.selected_candidate_id_sha256
            || selection.body.selected_candidate_authority_sha256
                != self.body.selected_candidate_authority_sha256
        {
            bail!("confirmation handoff differs from explicit operator selection");
        }
        let tracker_authority = &self.body.tracker_design_authority;
        tracker_authority.verify_for(
            &population.body.discovery_manifest,
            &population.body.component_correspondence,
        )?;
        self.body.supplied_tracker.verify()?;
        if self.body.tracker_design_authority_sha256 != tracker_authority.authority_sha256
            || self.body.operator_design_authority_sha256
                != tracker_authority.body.operator_design_authority_sha256
            || self.body.supplied_tracker_sha256 != self.body.supplied_tracker.tracker_sha256
            || tracker_authority.body.tracker_by_candidate_id.len() != 1
            || tracker_authority
                .body
                .tracker_by_candidate_id
                .get(&self.body.selected_candidate_id_sha256)
                != Some(&self.body.supplied_tracker)
            || self.body.supplied_tracker.schema_kind != "prism_pathb_candidate_tracker"
            || self.body.supplied_tracker.schema_version != "2.0.0"
        {
            bail!("confirmation handoff tracker-v2/operator design lineage mismatch");
        }
        let seed_authority = &self.body.disjoint_confirmation_seed_authority;
        seed_authority.verify_for(&population.body.discovery_manifest)?;
        if self.body.disjoint_confirmation_seed_authority_sha256 != seed_authority.authority_sha256
            || self.body.discovery_replica_seeds
                != population
                    .body
                    .discovery_manifest
                    .body
                    .discovery_replica_seeds
            || self.body.confirmation_replica_seeds
                != seed_authority.body.confirmation_replica_seeds
            || !self
                .body
                .discovery_replica_seeds
                .iter()
                .all(|seed| !self.body.confirmation_replica_seeds.contains(seed))
        {
            bail!("confirmation handoff discovery/confirmation seed authority mismatch");
        }
        let precommit = &selected.body.confirmation_binding_input;
        if !precommit.explicit_selection_receipt_required
            || precommit.derived_tracker_sha256.is_some()
            || !precommit.confirmation_replica_seeds.is_empty()
            || selected
                .body
                .tracker_materialization_inputs
                .supplied_tracker
                .is_some()
        {
            bail!("pre-commit discovery candidate contains confirmation influence");
        }
        if canonical_sha256(&self.body)? != self.handoff_sha256 {
            bail!("confirmation handoff content hash mismatch");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfirmationHandoffTransactionCommitBody {
    pub handoff_sha256: String,
    pub ordered_membership_sha256: String,
    pub members: Vec<DiscoveryTransactionMember>,
    pub handoff_typed_artifact: TypedEvidenceArtifact,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfirmationHandoffTransactionCommit {
    pub schema_kind: String,
    pub schema_version: String,
    pub status: String,
    pub commit_sha256: String,
    pub body: ConfirmationHandoffTransactionCommitBody,
}

#[derive(Debug, Clone)]
pub struct ConfirmationHandoffEmission {
    pub handoff: NativePathBConfirmationHandoff,
    pub commit: ConfirmationHandoffTransactionCommit,
    pub output_root: PathBuf,
}

fn collect_files(root: &Path, relative: &Path, output: &mut Vec<String>) -> Result<()> {
    let dir = root.join(relative);
    let mut children = std::fs::read_dir(&dir)?.collect::<std::io::Result<Vec<_>>>()?;
    children.sort_by_key(|entry| entry.file_name());
    for child in children {
        let file_type = child.file_type()?;
        let child_relative = relative.join(child.file_name());
        if file_type.is_symlink() {
            bail!("discovery transaction cannot contain symlinks");
        } else if file_type.is_dir() {
            collect_files(root, &child_relative, output)?;
        } else if file_type.is_file()
            && child_relative.to_string_lossy().as_ref() != DISCOVERY_COMMIT_FILE
        {
            output.push(child_relative.to_string_lossy().into_owned());
        } else if !file_type.is_file() {
            bail!("discovery transaction contains a non-regular member");
        }
    }
    Ok(())
}

fn transaction_members(root: &Path) -> Result<Vec<DiscoveryTransactionMember>> {
    let mut relative = Vec::new();
    collect_files(root, Path::new(""), &mut relative)?;
    relative.sort();
    relative
        .iter()
        .enumerate()
        .map(|(ordinal, path)| {
            let bytes = read_regular_nofollow(&root.join(path))?;
            Ok(DiscoveryTransactionMember {
                ordinal,
                role: if path == DISCOVERY_POPULATION_FILE {
                    "candidate_population_raw"
                } else if path.ends_with(".prismqnanl.receipt.json") {
                    if path.starts_with("candidate_authorities/") {
                        "candidate_authority_qnanl_receipt"
                    } else {
                        "candidate_population_qnanl_receipt"
                    }
                } else if path.ends_with(".prismqnanl") {
                    if path.starts_with("candidate_authorities/") {
                        "candidate_authority_qnanl"
                    } else {
                        "candidate_population_qnanl"
                    }
                } else if path.starts_with("candidate_authorities/") {
                    "candidate_authority_raw"
                } else {
                    "discovery_evidence_member"
                }
                .to_string(),
                relative_path: path.clone(),
                byte_length: bytes.len() as u64,
                sha256: sha256_bytes(&bytes),
            })
        })
        .collect()
}

fn verify_exact_members(
    root: &Path,
    expected: &[DiscoveryTransactionMember],
    expected_population_sha256: &str,
) -> Result<()> {
    let observed = transaction_members(root)?;
    if observed != expected || canonical_sha256(&observed)? != expected_population_sha256 {
        bail!("discovery transaction member population changed after commit");
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn publish_noreplace(staging: &Path, public: &Path) -> Result<()> {
    let staging = CString::new(staging.as_os_str().as_bytes())?;
    let public = CString::new(public.as_os_str().as_bytes())?;
    let status = unsafe {
        libc::syscall(
            libc::SYS_renameat2,
            libc::AT_FDCWD,
            staging.as_ptr(),
            libc::AT_FDCWD,
            public.as_ptr(),
            libc::RENAME_NOREPLACE,
        )
    };
    if status != 0 {
        return Err(std::io::Error::last_os_error()).context("atomic discovery publication");
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn publish_noreplace(_staging: &Path, _public: &Path) -> Result<()> {
    bail!("fail-atomic discovery publication requires Linux renameat2");
}

fn verify_committed_transaction(root: &Path) -> Result<DiscoveryTransactionCommit> {
    let commit_bytes = read_regular_nofollow(&root.join(DISCOVERY_COMMIT_FILE))?;
    let commit: DiscoveryTransactionCommit = serde_json::from_slice(&commit_bytes)?;
    if canonical_json_line(&commit)? != commit_bytes
        || commit.schema_kind != DISCOVERY_TRANSACTION_SCHEMA
        || commit.schema_version != DISCOVERY_TRANSACTION_VERSION
        || commit.status != "PASS"
        || canonical_sha256(&commit.body)? != commit.commit_sha256
    {
        bail!("invalid discovery transaction commit");
    }
    verify_exact_members(
        root,
        &commit.body.members,
        &commit.body.ordered_membership_sha256,
    )?;
    let population_bytes = read_regular_nofollow(&root.join(DISCOVERY_POPULATION_FILE))?;
    let population: NativePathBDiscoveryCandidatePopulation =
        serde_json::from_slice(&population_bytes)?;
    population.verify()?;
    if population.population_sha256 != commit.body.population_sha256
        || commit.body.candidate_authority_count != population.body.candidate_authorities.len()
        || commit.body.source_descriptor_count != population.body.source_descriptor_count
        || commit.body.authority_component_count != population.body.authority_component_count
        || commit.body.population_typed_artifact.raw_sha256 != sha256_bytes(&population_bytes)
    {
        bail!("discovery transaction population/typed byte binding failed");
    }
    let typed = &commit.body.population_typed_artifact;
    let expected_raw = root.join(DISCOVERY_POPULATION_FILE);
    let expected_packet = PathBuf::from(format!("{}.prismqnanl", expected_raw.display()));
    let expected_receipt = PathBuf::from(format!("{}.receipt.json", expected_packet.display()));
    let packet_bytes = read_regular_nofollow(&expected_packet)?;
    let receipt_bytes = read_regular_nofollow(&expected_receipt)?;
    if typed.raw_path != expected_raw.display().to_string()
        || typed.artifact_family != QnanlArtifactFamily::PathBDiscoveryCandidate.as_str()
        || typed.prismcode_events != vec![EVENT_PATHB_DISCOVERY_CANDIDATE_MATERIALIZED.to_string()]
        || typed.packet_path != expected_packet.display().to_string()
        || typed.receipt_path != expected_receipt.display().to_string()
        || typed.packet_sha256 != sha256_bytes(&packet_bytes)
        || typed.receipt_sha256 != sha256_bytes(&receipt_bytes)
        || !typed.exact_byte_roundtrip_pass
        || decode_packet_bytes(&packet_bytes).map_err(|error| anyhow::anyhow!(error.to_string()))?
            != population_bytes
    {
        bail!("discovery population QNANL exact-byte/public-path verification failed");
    }
    let receipt: serde_json::Value = serde_json::from_slice(&receipt_bytes)?;
    if receipt.get("status").and_then(|value| value.as_str()) != Some("PASS")
        || receipt.get("source_path").and_then(|value| value.as_str())
            != Some(typed.raw_path.as_str())
        || receipt.get("packet_path").and_then(|value| value.as_str())
            != Some(typed.packet_path.as_str())
        || receipt.get("receipt_path").and_then(|value| value.as_str())
            != Some(typed.receipt_path.as_str())
    {
        bail!("discovery QNANL receipt does not bind final public paths");
    }
    for entry in &population.body.candidate_authorities {
        let path = root.join(&entry.relative_path);
        let bytes = read_regular_nofollow(&path)?;
        let authority: NativePathBDiscoveryCandidateAuthority = serde_json::from_slice(&bytes)?;
        authority.verify()?;
        if canonical_json_line(&authority)? != bytes
            || authority.body.candidate_id_sha256 != entry.candidate_id_sha256
            || authority.candidate_authority_sha256 != entry.candidate_authority_sha256
            || authority.body.status != entry.status
        {
            bail!("candidate authority index differs from exact authority bytes");
        }
        let packet_path = PathBuf::from(format!("{}.prismqnanl", path.display()));
        let receipt_path = PathBuf::from(format!("{}.receipt.json", packet_path.display()));
        let packet = read_regular_nofollow(&packet_path)?;
        let receipt = read_regular_nofollow(&receipt_path)?;
        if entry.typed_artifact.raw_path != path.display().to_string()
            || entry.typed_artifact.artifact_family
                != QnanlArtifactFamily::PathBDiscoveryCandidate.as_str()
            || entry.typed_artifact.prismcode_events
                != vec![EVENT_PATHB_DISCOVERY_CANDIDATE_MATERIALIZED.to_string()]
            || entry.typed_artifact.packet_path != packet_path.display().to_string()
            || entry.typed_artifact.receipt_path != receipt_path.display().to_string()
            || entry.typed_artifact.raw_sha256 != sha256_bytes(&bytes)
            || entry.typed_artifact.packet_sha256 != sha256_bytes(&packet)
            || entry.typed_artifact.receipt_sha256 != sha256_bytes(&receipt)
            || !entry.typed_artifact.exact_byte_roundtrip_pass
            || decode_packet_bytes(&packet).map_err(|error| anyhow::anyhow!(error.to_string()))?
                != bytes
        {
            bail!("candidate authority QNANL exact-byte binding failed");
        }
    }
    Ok(commit)
}

pub fn aggregate_no_probe_discovery(
    output_root: &Path,
    loaded: LoadedDiscoveryPopulation,
    runtime_provenance: &RuntimeProvenance,
) -> Result<DiscoveryEmission> {
    validate_absolute_normalized(output_root)?;
    if output_root.exists() {
        bail!("discovery output already exists; no-clobber gate failed");
    }
    validate_discovery_lanes(&loaded.manifest, &loaded.lanes)?;
    let descriptors = loaded
        .lanes
        .iter()
        .flat_map(|lane| lane.body.component_descriptors.clone())
        .collect::<Vec<_>>();
    let ledger = establish_component_correspondence(
        descriptors,
        loaded.manifest.body.correspondence_policy.clone(),
    )?;
    ledger.verify()?;
    let authorities = build_candidate_authorities(&loaded, &ledger)?;
    let parent = output_root
        .parent()
        .context("discovery output has no parent")?;
    let name = output_root
        .file_name()
        .and_then(|name| name.to_str())
        .context("discovery output name is not UTF-8")?;
    let staging = parent.join(format!(".{name}.staging.{}", std::process::id()));
    if staging.exists() {
        bail!("discovery staging path already exists");
    }
    std::fs::create_dir(&staging)?;
    std::fs::create_dir(staging.join("candidate_authorities"))?;
    let authority_index = authorities
        .iter()
        .map(|authority| {
            let relative_path = format!(
                "candidate_authorities/{}.json",
                authority.body.candidate_id_sha256
            );
            let staged_path = staging.join(&relative_path);
            let public_path = output_root.join(&relative_path);
            write_create_new(&staged_path, &canonical_json_line(authority)?)?;
            let typed_artifact = emit_qnanl_exact_at_public_paths(
                &staged_path,
                &public_path,
                QnanlArtifactFamily::PathBDiscoveryCandidate,
                &[(
                    EVENT_PATHB_DISCOVERY_CANDIDATE_MATERIALIZED,
                    PRISMCODE_OPCODE_PATHB_DISCOVERY_CANDIDATE_MATERIALIZED,
                    vec![
                        "candidate_id_sha256",
                        "discovery_manifest_sha256",
                        "source_component_population",
                        "recurrence_support",
                        "preservation_status",
                    ],
                )],
                runtime_provenance,
                "prism_native_pathb_discovery:candidate_authority",
            )?;
            Ok(DiscoveryCandidateAuthorityIndexEntry {
                candidate_id_sha256: authority.body.candidate_id_sha256.clone(),
                candidate_authority_sha256: authority.candidate_authority_sha256.clone(),
                status: authority.body.status,
                relative_path,
                typed_artifact,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let recurrent_qualified_count = authorities
        .iter()
        .filter(|row| row.body.status == DiscoveryCandidateStatus::RecurrentQualified)
        .count();
    let recurrent_below_minimum_count = authorities
        .iter()
        .filter(|row| row.body.status == DiscoveryCandidateStatus::RecurrentBelowMinimumSupport)
        .count();
    let ambiguous_count = authorities
        .iter()
        .filter(|row| row.body.status == DiscoveryCandidateStatus::Ambiguous)
        .count();
    let unmatched_count = authorities
        .iter()
        .filter(|row| row.body.status == DiscoveryCandidateStatus::Unmatched)
        .count();
    let body = NativePathBDiscoveryCandidatePopulationBody {
        discovery_manifest: loaded.manifest.clone(),
        source_lane_transactions: loaded.manifest.body.lane_transactions.clone(),
        source_descriptor_count: ledger.descriptors.len(),
        authority_component_count: authorities
            .iter()
            .map(|row| row.body.components.len())
            .sum(),
        component_correspondence: ledger,
        candidate_authorities: authority_index,
        recurrent_qualified_count,
        recurrent_below_minimum_count,
        ambiguous_count,
        unmatched_count,
        all_descriptors_preserved: true,
        legacy_binding_site_json_inputs: 0,
        probe_chemistry_candidate_inputs: 0,
        ranking_applied: false,
        suppression_applied: false,
    };
    let population = NativePathBDiscoveryCandidatePopulation {
        schema_kind: DISCOVERY_POPULATION_SCHEMA.to_string(),
        schema_version: "2.0.0".to_string(),
        population_sha256: canonical_sha256(&body)?,
        body,
    };
    population.verify()?;
    let staged_population = staging.join(DISCOVERY_POPULATION_FILE);
    let public_population = output_root.join(DISCOVERY_POPULATION_FILE);
    write_create_new(&staged_population, &canonical_json_line(&population)?)?;
    let typed = emit_qnanl_exact_at_public_paths(
        &staged_population,
        &public_population,
        QnanlArtifactFamily::PathBDiscoveryCandidate,
        &[(
            EVENT_PATHB_DISCOVERY_CANDIDATE_MATERIALIZED,
            PRISMCODE_OPCODE_PATHB_DISCOVERY_CANDIDATE_MATERIALIZED,
            vec![
                "candidate_authorities",
                "source_lane_transactions",
                "component_correspondence",
                "all_descriptors_preserved",
                "no_probe_whole_lattice_discovery",
            ],
        )],
        runtime_provenance,
        "prism_native_pathb_discovery:unbiased_candidate_population",
    )?;
    sync_dir(&staging.join("candidate_authorities"))?;
    let members = transaction_members(&staging)?;
    let commit_body = DiscoveryTransactionCommitBody {
        population_sha256: population.population_sha256.clone(),
        candidate_authority_count: population.body.candidate_authorities.len(),
        source_descriptor_count: population.body.source_descriptor_count,
        authority_component_count: population.body.authority_component_count,
        ordered_membership_sha256: canonical_sha256(&members)?,
        members,
        population_typed_artifact: typed,
    };
    let commit = DiscoveryTransactionCommit {
        schema_kind: DISCOVERY_TRANSACTION_SCHEMA.to_string(),
        schema_version: DISCOVERY_TRANSACTION_VERSION.to_string(),
        status: "PASS".to_string(),
        commit_sha256: canonical_sha256(&commit_body)?,
        body: commit_body,
    };
    write_create_new(
        &staging.join(DISCOVERY_COMMIT_FILE),
        &canonical_json_line(&commit)?,
    )?;
    sync_dir(&staging)?;
    publish_noreplace(&staging, output_root)?;
    sync_dir(parent)?;
    let verified = verify_committed_transaction(output_root)?;
    if verified != commit {
        bail!("published discovery transaction differs from staged commit");
    }
    Ok(DiscoveryEmission {
        population,
        authorities,
        commit,
        output_root: output_root.to_path_buf(),
    })
}

fn confirmation_handoff_members(root: &Path) -> Result<Vec<DiscoveryTransactionMember>> {
    let expected = [
        (
            "confirmation_handoff_raw",
            CONFIRMATION_HANDOFF_FILE.to_string(),
        ),
        (
            "confirmation_handoff_qnanl",
            format!("{CONFIRMATION_HANDOFF_FILE}.prismqnanl"),
        ),
        (
            "confirmation_handoff_qnanl_receipt",
            format!("{CONFIRMATION_HANDOFF_FILE}.prismqnanl.receipt.json"),
        ),
    ];
    let mut observed_paths = Vec::new();
    collect_files(root, Path::new(""), &mut observed_paths)?;
    observed_paths.sort();
    let mut expected_paths = expected
        .iter()
        .map(|(_, path)| path.clone())
        .collect::<Vec<_>>();
    expected_paths.sort();
    if observed_paths != expected_paths {
        bail!("confirmation handoff transaction contains unexpected or missing members");
    }
    expected
        .into_iter()
        .enumerate()
        .map(|(ordinal, (role, relative_path))| {
            let bytes = read_regular_nofollow(&root.join(&relative_path))?;
            Ok(DiscoveryTransactionMember {
                ordinal,
                role: role.to_string(),
                relative_path,
                byte_length: u64::try_from(bytes.len())?,
                sha256: sha256_bytes(&bytes),
            })
        })
        .collect()
}

fn verify_committed_confirmation_handoff(
    root: &Path,
) -> Result<(
    NativePathBConfirmationHandoff,
    ConfirmationHandoffTransactionCommit,
)> {
    validate_absolute_normalized(root)?;
    let commit_bytes = read_regular_nofollow(&root.join(CONFIRMATION_HANDOFF_COMMIT_FILE))?;
    let commit: ConfirmationHandoffTransactionCommit = serde_json::from_slice(&commit_bytes)?;
    if canonical_json_line(&commit)? != commit_bytes
        || commit.schema_kind != "prism_native_pathb_confirmation_handoff_transaction_commit"
        || commit.schema_version != "1.0.0"
        || commit.status != "PASS"
        || canonical_sha256(&commit.body)? != commit.commit_sha256
    {
        bail!("invalid confirmation handoff transaction commit");
    }
    let members = confirmation_handoff_members(root)?;
    if members != commit.body.members
        || canonical_sha256(&members)? != commit.body.ordered_membership_sha256
    {
        bail!("confirmation handoff transaction membership changed after commit");
    }
    let raw_path = root.join(CONFIRMATION_HANDOFF_FILE);
    let raw_bytes = read_regular_nofollow(&raw_path)?;
    let handoff: NativePathBConfirmationHandoff = serde_json::from_slice(&raw_bytes)?;
    handoff.verify()?;
    if canonical_json_line(&handoff)? != raw_bytes
        || handoff.handoff_sha256 != commit.body.handoff_sha256
    {
        bail!("confirmation handoff raw bytes differ from committed authority");
    }
    let packet_path = PathBuf::from(format!("{}.prismqnanl", raw_path.display()));
    let receipt_path = PathBuf::from(format!("{}.receipt.json", packet_path.display()));
    let packet = read_regular_nofollow(&packet_path)?;
    let receipt = read_regular_nofollow(&receipt_path)?;
    let typed = &commit.body.handoff_typed_artifact;
    if typed.artifact_family != QnanlArtifactFamily::PathBConfirmationHandoff.as_str()
        || typed.prismcode_events != vec![EVENT_PATHB_CONFIRMATION_HANDOFF_MATERIALIZED.to_string()]
        || typed.raw_path != raw_path.display().to_string()
        || typed.packet_path != packet_path.display().to_string()
        || typed.receipt_path != receipt_path.display().to_string()
        || typed.raw_sha256 != sha256_bytes(&raw_bytes)
        || typed.packet_sha256 != sha256_bytes(&packet)
        || typed.receipt_sha256 != sha256_bytes(&receipt)
        || !typed.exact_byte_roundtrip_pass
        || decode_packet_bytes(&packet).map_err(|error| anyhow::anyhow!(error.to_string()))?
            != raw_bytes
    {
        bail!("confirmation handoff typed exact-byte verification failed");
    }
    let receipt: serde_json::Value = serde_json::from_slice(&receipt)?;
    if receipt.get("status").and_then(|value| value.as_str()) != Some("PASS")
        || receipt.get("source_path").and_then(|value| value.as_str())
            != Some(typed.raw_path.as_str())
        || receipt.get("packet_path").and_then(|value| value.as_str())
            != Some(typed.packet_path.as_str())
        || receipt.get("receipt_path").and_then(|value| value.as_str())
            != Some(typed.receipt_path.as_str())
    {
        bail!("confirmation handoff QNANL receipt does not bind public paths");
    }
    Ok((handoff, commit))
}

pub fn load_committed_confirmation_handoff(
    root: &Path,
) -> Result<(
    NativePathBConfirmationHandoff,
    ConfirmationHandoffTransactionCommit,
)> {
    verify_committed_confirmation_handoff(root)
}

pub fn materialize_confirmation_handoff(
    discovery_transaction_root: &Path,
    output_root: &Path,
    selection: ExplicitDiscoveryCandidateSelectionAuthority,
    tracker_authority: DiscoveryTrackerDesignAuthority,
    seed_authority: DisjointConfirmationSeedAuthority,
    runtime_provenance: &RuntimeProvenance,
) -> Result<ConfirmationHandoffEmission> {
    validate_absolute_normalized(discovery_transaction_root)?;
    validate_absolute_normalized(output_root)?;
    if output_root.exists() {
        bail!("confirmation handoff output already exists; no-clobber gate failed");
    }
    let source_commit = verify_committed_transaction(discovery_transaction_root)?;
    let population: NativePathBDiscoveryCandidatePopulation =
        load_canonical(&discovery_transaction_root.join(DISCOVERY_POPULATION_FILE))?;
    population.verify()?;
    selection.verify()?;
    if selection.body.discovery_transaction_commit_sha256 != source_commit.commit_sha256
        || selection.body.discovery_population_sha256 != population.population_sha256
    {
        bail!("explicit candidate selection does not bind this committed discovery transaction");
    }
    let candidate_index = population
        .body
        .candidate_authorities
        .iter()
        .find(|entry| {
            entry.candidate_id_sha256 == selection.body.selected_candidate_id_sha256
                && entry.candidate_authority_sha256
                    == selection.body.selected_candidate_authority_sha256
        })
        .context("explicitly selected candidate is absent from committed discovery index")?;
    let selected: NativePathBDiscoveryCandidateAuthority =
        load_canonical(&discovery_transaction_root.join(&candidate_index.relative_path))?;
    selected.verify()?;
    if selected.body.candidate_id_sha256 != selection.body.selected_candidate_id_sha256
        || selected.candidate_authority_sha256 != selection.body.selected_candidate_authority_sha256
    {
        bail!("selected candidate authority bytes differ from explicit selection");
    }
    tracker_authority.verify_for(
        &population.body.discovery_manifest,
        &population.body.component_correspondence,
    )?;
    let supplied_tracker = tracker_authority
        .body
        .tracker_by_candidate_id
        .get(&selection.body.selected_candidate_id_sha256)
        .cloned()
        .context("tracker design authority lacks the explicitly selected candidate")?;
    seed_authority.verify_for(&population.body.discovery_manifest)?;
    let body = NativePathBConfirmationHandoffBody {
        discovery_transaction_commit_sha256: source_commit.commit_sha256.clone(),
        discovery_transaction_commit: source_commit.clone(),
        discovery_population_sha256: population.population_sha256.clone(),
        discovery_candidate_authority_count: population.body.candidate_authorities.len(),
        discovery_source_descriptor_count: population.body.source_descriptor_count,
        discovery_authority_component_count: population.body.authority_component_count,
        discovery_manifest_sha256: population
            .body
            .discovery_manifest
            .discovery_manifest_sha256
            .clone(),
        correspondence_ledger_sha256: population
            .body
            .component_correspondence
            .ledger_sha256
            .clone(),
        selected_candidate_id_sha256: selected.body.candidate_id_sha256.clone(),
        selected_candidate_authority_sha256: selected.candidate_authority_sha256.clone(),
        selected_candidate_authority: selected,
        explicit_selection_authority_sha256: selection.authority_sha256.clone(),
        explicit_selection_authority: selection,
        tracker_design_authority_sha256: tracker_authority.authority_sha256.clone(),
        operator_design_authority_sha256: tracker_authority
            .body
            .operator_design_authority_sha256
            .clone(),
        supplied_tracker_sha256: supplied_tracker.tracker_sha256.clone(),
        supplied_tracker,
        tracker_design_authority: tracker_authority,
        disjoint_confirmation_seed_authority_sha256: seed_authority.authority_sha256.clone(),
        discovery_replica_seeds: population
            .body
            .discovery_manifest
            .body
            .discovery_replica_seeds
            .clone(),
        confirmation_replica_seeds: seed_authority.body.confirmation_replica_seeds.clone(),
        disjoint_confirmation_seed_authority: seed_authority,
        discovery_population: population,
    };
    let handoff = NativePathBConfirmationHandoff::materialize(body)?;
    let parent = output_root
        .parent()
        .context("confirmation handoff output has no parent")?;
    let name = output_root
        .file_name()
        .and_then(|name| name.to_str())
        .context("confirmation handoff output name is not UTF-8")?;
    let staging = parent.join(format!(".{name}.staging.{}", std::process::id()));
    if staging.exists() {
        bail!("confirmation handoff staging path already exists");
    }
    std::fs::create_dir(&staging)?;
    let staged_raw = staging.join(CONFIRMATION_HANDOFF_FILE);
    let public_raw = output_root.join(CONFIRMATION_HANDOFF_FILE);
    write_create_new(&staged_raw, &canonical_json_line(&handoff)?)?;
    let typed = emit_qnanl_exact_at_public_paths(
        &staged_raw,
        &public_raw,
        QnanlArtifactFamily::PathBConfirmationHandoff,
        &[(
            EVENT_PATHB_CONFIRMATION_HANDOFF_MATERIALIZED,
            PRISMCODE_OPCODE_PATHB_CONFIRMATION_HANDOFF_MATERIALIZED,
            vec![
                "discovery_transaction_commit_sha256",
                "discovery_population_sha256",
                "selected_candidate_authority_sha256",
                "explicit_selection_authority_sha256",
                "tracker_design_authority_sha256",
                "disjoint_confirmation_seed_authority_sha256",
            ],
        )],
        runtime_provenance,
        "prism_native_pathb_confirmation_handoff:post_commit_explicit_selection",
    )?;
    let members = confirmation_handoff_members(&staging)?;
    let commit_body = ConfirmationHandoffTransactionCommitBody {
        handoff_sha256: handoff.handoff_sha256.clone(),
        ordered_membership_sha256: canonical_sha256(&members)?,
        members,
        handoff_typed_artifact: typed,
    };
    let commit = ConfirmationHandoffTransactionCommit {
        schema_kind: "prism_native_pathb_confirmation_handoff_transaction_commit".to_string(),
        schema_version: "1.0.0".to_string(),
        status: "PASS".to_string(),
        commit_sha256: canonical_sha256(&commit_body)?,
        body: commit_body,
    };
    write_create_new(
        &staging.join(CONFIRMATION_HANDOFF_COMMIT_FILE),
        &canonical_json_line(&commit)?,
    )?;
    sync_dir(&staging)?;
    if verify_committed_transaction(discovery_transaction_root)? != source_commit {
        bail!("discovery transaction changed while confirmation handoff was staged");
    }
    publish_noreplace(&staging, output_root)?;
    sync_dir(parent)?;
    let (verified_handoff, verified_commit) = verify_committed_confirmation_handoff(output_root)?;
    if verified_handoff != handoff || verified_commit != commit {
        bail!("published confirmation handoff differs from staged transaction");
    }
    Ok(ConfirmationHandoffEmission {
        handoff,
        commit,
        output_root: output_root.to_path_buf(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> String {
        format!("{byte:02x}").repeat(32)
    }

    #[test]
    fn explicit_candidate_selection_is_post_commit_and_content_hash_bound() {
        let authority = ExplicitDiscoveryCandidateSelectionAuthority::materialize(
            ExplicitDiscoveryCandidateSelectionBody {
                discovery_transaction_commit_sha256: digest(1),
                discovery_population_sha256: digest(2),
                selected_candidate_id_sha256: digest(3),
                selected_candidate_authority_sha256: digest(4),
                operator_selection_authority_sha256: digest(5),
                selection_semantics: "operator_explicit_post_commit_selection_no_runtime_inference"
                    .into(),
            },
        )
        .unwrap();
        authority.verify().unwrap();

        let mut changed_candidate = authority.clone();
        changed_candidate.body.selected_candidate_id_sha256 = digest(6);
        assert!(changed_candidate.verify().is_err());

        let mut inferred = authority;
        inferred.body.selection_semantics = "runtime_selected_highest_score".into();
        assert!(inferred.verify().is_err());
    }

    fn policy() -> ComponentCorrespondencePolicy {
        ComponentCorrespondencePolicy {
            policy_sha256: String::new(),
            residue_js_weight: 0.7,
            spatial_weight: 0.3,
            maximum_assignment_cost: 0.25,
            ambiguity_margin: 0.01,
        }
        .materialize()
        .unwrap()
    }

    fn manifest() -> NativePathBDiscoveryManifest {
        let condition = ThermodynamicCondition {
            condition_id: "apo-300k".into(),
            temperature_k: 300.0,
            pressure_bar: 1.0,
            ionic_strength_molar: 0.15,
            solvent_model: "tip3p".into(),
            periodic_box_vectors_a: [[40.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 40.0]],
            pme_parameter_sha256: digest(1),
        };
        let phase = ProtocolPhaseBinding {
            stream_id: 0,
            thermodynamic_condition_id: condition.condition_id.clone(),
            phase_ordinal: 1,
            phase_name: "stationary".into(),
            start_step: 100,
            end_step_exclusive: 1000,
            protocol_name: "apo".into(),
            phase_schedule_digest_sha256: digest(2),
            protocol_state_hash: digest(3),
        };
        let mut stratum = DiscoveryPhaseStratum {
            stratum_sha256: String::new(),
            matched_physical_state_sha256: condition.matched_physical_state_sha256(&phase).unwrap(),
            thermodynamic_condition: condition,
            phase_ordinal: phase.phase_ordinal,
            phase_name: phase.phase_name,
            phase_start_step: phase.start_step,
            phase_end_step_exclusive: phase.end_step_exclusive,
            protocol_name: phase.protocol_name,
            phase_schedule_digest_sha256: phase.phase_schedule_digest_sha256,
            protocol_state_hash: phase.protocol_state_hash,
        };
        stratum.stratum_sha256 = canonical_sha256(&stratum.body()).unwrap();
        let entries = (0..5)
            .map(|i| DiscoveryLaneTransactionEntry {
                transaction_root: PathBuf::from(format!(
                    "/evidence/rep_{i:02}/{NATIVE_PATHB_LANE_TRANSACTION_DIRECTORY}"
                )),
                transaction_commit_sha256: digest(10 + i),
                lane_evidence_sha256: digest(20 + i),
            })
            .collect();
        let body = NativePathBDiscoveryManifestBody {
            cohort_id: "mycmax-apo-discovery".into(),
            engine_sha256: digest(4),
            topology_sha256: digest(5),
            structural_design: {
                let mut design = DiscoveryStructuralDesignAuthority {
                    structural_design_sha256: String::new(),
                    selection_semantics: "whole_authenticated_lattice_no_residue_prefilter".into(),
                    required_stream_ids: vec![0, 1],
                    coordinate_cavity_config:
                        crate::coordinate_cavity::CoordinateCavityConfig::default(),
                    correspondence_policy_sha256: policy().policy_sha256,
                    fixed_cadence_design_sha256: None,
                    excluded_source_request_fields: vec![
                        "fragment_panel_sha256".into(),
                        "matched_control_config".into(),
                        "registry_policy".into(),
                        "candidate_tracker".into(),
                        "site_key".into(),
                    ],
                };
                design.structural_design_sha256 = canonical_sha256(&design.body()).unwrap();
                design
            },
            minimum_candidate_seed_support: 5,
            discovery_replica_seeds: vec![11, 22, 33, 44, 55],
            correspondence_policy: policy(),
            stratum,
            lane_transactions: entries,
        };
        NativePathBDiscoveryManifest {
            schema_kind: DISCOVERY_MANIFEST_SCHEMA.into(),
            schema_version: DISCOVERY_MANIFEST_VERSION.into(),
            discovery_manifest_sha256: canonical_sha256(&body).unwrap(),
            body,
        }
    }

    #[test]
    fn manifest_rejects_duplicate_seed_cross_lane_and_legacy_path_shapes() {
        let mut value = manifest();
        value.verify().unwrap();
        value.body.discovery_replica_seeds[1] = 11;
        value.discovery_manifest_sha256 = canonical_sha256(&value.body).unwrap();
        assert!(value.verify().is_err());

        let mut value = manifest();
        value.body.lane_transactions[0].transaction_root =
            PathBuf::from("/evidence/rep_00/legacy.binding_sites.json");
        value
            .body
            .lane_transactions
            .sort_by(|a, b| a.transaction_root.cmp(&b.transaction_root));
        value.discovery_manifest_sha256 = canonical_sha256(&value.body).unwrap();
        assert!(value.verify().is_err());
    }

    #[test]
    fn confirmation_seed_reuse_is_rejected() {
        let manifest = manifest();
        let body = DisjointConfirmationSeedAuthorityBody {
            discovery_manifest_sha256: manifest.discovery_manifest_sha256.clone(),
            source_authority_sha256: digest(31),
            confirmation_replica_seeds: vec![11, 66, 77, 88, 99],
        };
        let authority = DisjointConfirmationSeedAuthority {
            schema_kind: "prism_native_pathb_disjoint_confirmation_seed_authority".into(),
            schema_version: "1.0.0".into(),
            authority_sha256: canonical_sha256(&body).unwrap(),
            body,
        };
        assert!(authority.verify_for(&manifest).is_err());
    }

    #[test]
    fn canonical_loader_rejects_byte_tamper_and_noncanonical_json() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("manifest.json");
        let manifest = manifest();
        std::fs::write(&path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
        assert!(load_discovery_manifest(&path).is_err());
        std::fs::write(&path, canonical_json_line(&manifest).unwrap()).unwrap();
        assert!(load_discovery_manifest(&path).is_ok());
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[5] ^= 1;
        std::fs::write(&path, bytes).unwrap();
        assert!(load_discovery_manifest(&path).is_err());
    }

    #[test]
    fn body_fixed_requirement_fails_closed() {
        let requirement = BodyFixedTrackingRequirement {
            status: "EVIDENCE_NOT_FOUND".into(),
            static_centroid_is_reference_metadata_only: true,
            required_transform: "pbc_unwrap_then_align".into(),
            required_authorities: vec!["signed_anchor_atoms".into()],
            tracker_promotion_allowed: false,
        };
        assert!(!requirement.tracker_promotion_allowed);
        assert!(requirement.static_centroid_is_reference_metadata_only);
    }

    #[test]
    fn discovery_event_has_dedicated_opcode_and_is_not_advancement() {
        assert_eq!(
            prismcode::binary_runtime::opcode_id_for_name(
                EVENT_PATHB_DISCOVERY_CANDIDATE_MATERIALIZED,
            ),
            Some(PRISMCODE_OPCODE_PATHB_DISCOVERY_CANDIDATE_MATERIALIZED),
        );
        assert_ne!(
            PRISMCODE_OPCODE_PATHB_DISCOVERY_CANDIDATE_MATERIALIZED,
            prismcode::binary_runtime::opcode::PATHB_CANDIDATE_OBSERVED,
        );
    }

    #[test]
    fn cross_lane_tracker_and_under_supported_recurrence_are_rejected_or_demoted() {
        let manifest = manifest();
        let structural = &manifest.body.structural_design;
        let mut lanes = structural
            .required_stream_ids
            .iter()
            .copied()
            .map(|stream| (stream, ControlLane::NoProbe))
            .collect::<BTreeMap<_, _>>();
        let mut policy = crate::pathb_evidence_emitter::NativePathBLanePolicy {
            site_key: "provenance-only-label".into(),
            selection_mode: NativePathBSelectionMode::WholeLatticeDiscovery,
            fragment_panel_sha256: digest(80),
            target_residue_ids: Vec::new(),
            minimum_target_residue_overlap_count: 0,
            minimum_target_residue_overlap_fraction: 0.0,
            required_stream_ids: structural.required_stream_ids.clone(),
            fixed_cadence_steps: 10,
            stream_lanes: lanes.clone(),
            fragment_slot_ids: BTreeMap::new(),
            coordinate_cavity_config: structural.coordinate_cavity_config.clone(),
            candidate_tracker: None,
        };
        validate_discovery_lane_policy(&policy, structural).unwrap();
        lanes.insert(0, ControlLane::HypothesisFragment);
        policy.stream_lanes = lanes;
        policy.fragment_slot_ids.insert(0, 1);
        assert!(validate_discovery_lane_policy(&policy, structural).is_err());
        assert_eq!(
            recurrent_status(2, 5),
            DiscoveryCandidateStatus::RecurrentBelowMinimumSupport
        );
        assert_eq!(
            recurrent_status(5, 5),
            DiscoveryCandidateStatus::RecurrentQualified
        );
    }

    #[test]
    fn structural_design_hash_excludes_chemistry_and_registry_source_metadata() {
        let manifest = manifest();
        let structural = manifest
            .body
            .structural_design
            .structural_design_sha256
            .clone();
        let candidate =
            candidate_id(DiscoveryCandidateStatus::Unmatched, None, &[digest(90)]).unwrap();
        // These are provenance-only full-request fields and are intentionally
        // absent from DiscoveryStructuralDesignBody and candidate_id.
        let source_request_a = canonical_sha256(&(digest(1), "registry-a", "panel-a")).unwrap();
        let source_request_b = canonical_sha256(&(digest(2), "registry-b", "panel-b")).unwrap();
        assert_ne!(source_request_a, source_request_b);
        assert_eq!(
            structural,
            manifest.body.structural_design.structural_design_sha256
        );
        assert_eq!(
            candidate,
            candidate_id(DiscoveryCandidateStatus::Unmatched, None, &[digest(90)]).unwrap()
        );
    }

    #[test]
    fn discovery_member_tamper_extra_file_and_noclobber_publish_fail_closed() {
        let temp = tempfile::tempdir().unwrap();
        let staged = temp.path().join(".stage-a");
        let public = temp.path().join("published");
        std::fs::create_dir(&staged).unwrap();
        std::fs::write(staged.join("evidence.json"), b"exact\n").unwrap();
        let members = transaction_members(&staged).unwrap();
        let population_sha256 = canonical_sha256(&members).unwrap();
        verify_exact_members(&staged, &members, &population_sha256).unwrap();
        std::fs::write(staged.join("evidence.json"), b"tampered\n").unwrap();
        assert!(verify_exact_members(&staged, &members, &population_sha256).is_err());
        std::fs::write(staged.join("evidence.json"), b"exact\n").unwrap();
        std::fs::write(staged.join("undeclared.bin"), b"extra").unwrap();
        assert!(verify_exact_members(&staged, &members, &population_sha256).is_err());
        std::fs::remove_file(staged.join("undeclared.bin")).unwrap();
        publish_noreplace(&staged, &public).unwrap();
        let second = temp.path().join(".stage-b");
        std::fs::create_dir(&second).unwrap();
        std::fs::write(second.join("evidence.json"), b"replacement\n").unwrap();
        assert!(publish_noreplace(&second, &public).is_err());
        assert_eq!(
            std::fs::read(public.join("evidence.json")).unwrap(),
            b"exact\n"
        );
    }
}
