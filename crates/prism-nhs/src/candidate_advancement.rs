//! Immutable, complete-population candidate advancement registry.
//!
//! Every candidate is preserved as a content-addressed observation. Pareto
//! membership is deterministic from declared objective directions; it never
//! deletes or rewrites a candidate. Changes in the active champion frontier
//! are represented as append-only supersession edges in a SHA-256 hash chain.

use crate::chemical_state_fiber::{canonical_sha256, require_sha256};
use crate::pathb_chemical_state::{MatchedControlResult, PocketResponseClass};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::fd::{AsRawFd, RawFd};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

/// Candidate registry schema identifier.
pub const CANDIDATE_REGISTRY_SCHEMA: &str = "prism_candidate_advancement_registry";
/// Candidate registry schema version.
pub const CANDIDATE_REGISTRY_VERSION: &str = "1.0.0";
/// PrismCode opcode for `PATHB.CANDIDATE_OBSERVED`.
pub use prismcode::binary_runtime::opcode::PATHB_CANDIDATE_OBSERVED as PRISMCODE_OPCODE_CANDIDATE_OBSERVED;
/// PrismCode opcode for `PATHB.PARETO_CHAMPION_PROMOTED`.
pub use prismcode::binary_runtime::opcode::PATHB_PARETO_CHAMPION_PROMOTED as PRISMCODE_OPCODE_PARETO_CHAMPION_PROMOTED;
/// PrismCode opcode for `PATHB.PARETO_CHAMPION_SUPERSEDED`.
pub use prismcode::binary_runtime::opcode::PATHB_PARETO_CHAMPION_SUPERSEDED as PRISMCODE_OPCODE_PARETO_CHAMPION_SUPERSEDED;
/// Typed event semantic for complete-population observation.
pub const EVENT_CANDIDATE_OBSERVED: &str = "PATHB.CANDIDATE_OBSERVED";
/// Typed event semantic for Pareto promotion.
pub const EVENT_PARETO_CHAMPION_PROMOTED: &str = "PATHB.PARETO_CHAMPION_PROMOTED";
/// Typed event semantic for a directed supersession edge.
pub const EVENT_PARETO_CHAMPION_SUPERSEDED: &str = "PATHB.PARETO_CHAMPION_SUPERSEDED";
/// Canonical QNANL artifact-family code for the advancement registry.
pub const QNANL_ARTIFACT_CODE_CANDIDATE_ADVANCEMENT_REGISTRY: u16 =
    prismcode::qnanl_artifact::QnanlArtifactFamily::CandidateAdvancementRegistry as u16;
/// Canonical QNANL artifact family for the advancement registry.
pub const QNANL_FAMILY_CANDIDATE_ADVANCEMENT_REGISTRY: &str =
    prismcode::qnanl_artifact::QnanlArtifactFamily::CandidateAdvancementRegistry.as_str();

const GENESIS_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";

/// Optimization direction for one Pareto objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveDirection {
    /// Larger values are preferred.
    Maximize,
    /// Smaller values are preferred.
    Minimize,
}

/// Declared objective and optimization direction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObjectiveDefinition {
    /// Stable objective name.
    pub objective_id: String,
    /// Optimization direction.
    pub direction: ObjectiveDirection,
}

/// Hashable candidate body retaining its complete source population.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateBody {
    /// Stable operator/engine candidate identity.
    pub candidate_id: String,
    /// Stable chain/spatial correspondence identity. This is either a
    /// recurrent component-correspondence SHA or the predeclared signed
    /// confirmation tracker SHA; raw Ghost site IDs are forbidden here.
    pub component_correspondence_sha256: String,
    /// Prior observation hash clarified by this observation. The stable
    /// candidate/site identity may recur only through this explicit edge.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revises_candidate_sha256: Option<String>,
    /// Target/site key shared with chemical-state fibers.
    pub site_key: String,
    /// Matched-control mechanistic classification.
    pub control_classification: PocketResponseClass,
    /// Content hash of the matched-control result.
    pub matched_control_result_sha256: String,
    /// Every fiber supporting the candidate, sorted and unique.
    pub source_fiber_sha256: Vec<String>,
    /// Common independent seed count.
    pub independent_seed_count: usize,
    /// Minimum ESS over the three matched lanes.
    pub minimum_lane_effective_sample_size: f64,
    /// Simultaneous confidence interval width for the primary selective effect.
    pub primary_effect_confidence_width: f64,
    /// Complete objective vector.
    pub objectives: BTreeMap<String, f64>,
    /// Additional evidence artifact hashes.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub evidence_artifact_sha256: BTreeMap<String, String>,
}

/// Content-addressed candidate record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateRecord {
    /// SHA-256 of the canonical candidate body.
    pub candidate_sha256: String,
    /// Complete candidate body.
    pub body: CandidateBody,
}

impl CandidateRecord {
    /// Materialize a deterministic candidate record.
    pub fn materialize(mut body: CandidateBody) -> Result<Self> {
        body.source_fiber_sha256.sort();
        if body
            .source_fiber_sha256
            .windows(2)
            .any(|pair| pair[0] == pair[1])
        {
            bail!("candidate source fiber hashes must be unique");
        }
        validate_candidate_body(&body)?;
        let candidate_sha256 = canonical_sha256(&body)?;
        Ok(Self {
            candidate_sha256,
            body,
        })
    }

    /// Construct a candidate from a verified matched-control result.
    pub fn from_matched_control(
        candidate_id: String,
        component_correspondence_sha256: String,
        result: &MatchedControlResult,
        objectives: BTreeMap<String, f64>,
        evidence_artifact_sha256: BTreeMap<String, String>,
    ) -> Result<Self> {
        Self::from_matched_control_revision(
            candidate_id,
            component_correspondence_sha256,
            None,
            result,
            objectives,
            evidence_artifact_sha256,
        )
    }

    /// Construct a clarified candidate observation with an explicit prior edge.
    pub fn from_matched_control_revision(
        candidate_id: String,
        component_correspondence_sha256: String,
        revises_candidate_sha256: Option<String>,
        result: &MatchedControlResult,
        objectives: BTreeMap<String, f64>,
        evidence_artifact_sha256: BTreeMap<String, String>,
    ) -> Result<Self> {
        result.verify()?;
        Self::materialize(CandidateBody {
            candidate_id,
            component_correspondence_sha256,
            revises_candidate_sha256,
            site_key: result.site_key.clone(),
            control_classification: result.classification,
            matched_control_result_sha256: result.result_sha256.clone(),
            source_fiber_sha256: result.source_fiber_sha256.clone(),
            independent_seed_count: result.common_seeds.len(),
            minimum_lane_effective_sample_size: result
                .no_probe
                .effective_sample_size
                .min(result.hypothesis.effective_sample_size)
                .min(result.orthogonal.effective_sample_size),
            primary_effect_confidence_width: result.hypothesis_minus_orthogonal.upper
                - result.hypothesis_minus_orthogonal.lower,
            objectives,
            evidence_artifact_sha256,
        })
    }

    /// Verify content identity and source evidence domains.
    pub fn verify(&self) -> Result<()> {
        validate_candidate_body(&self.body)?;
        require_sha256(&self.candidate_sha256, "candidate_sha256")?;
        if canonical_sha256(&self.body)? != self.candidate_sha256 {
            bail!("candidate content hash mismatch");
        }
        Ok(())
    }
}

fn validate_candidate_body(body: &CandidateBody) -> Result<()> {
    if body.candidate_id.trim().is_empty() || body.site_key.trim().is_empty() {
        bail!("candidate and site identities must be non-empty");
    }
    require_sha256(
        &body.component_correspondence_sha256,
        "component_correspondence_sha256",
    )?;
    require_sha256(
        &body.matched_control_result_sha256,
        "matched_control_result_sha256",
    )?;
    if let Some(revises) = &body.revises_candidate_sha256 {
        require_sha256(revises, "revises_candidate_sha256")?;
    }
    if body.source_fiber_sha256.is_empty() {
        bail!("candidate must retain at least one source fiber");
    }
    for hash in &body.source_fiber_sha256 {
        require_sha256(hash, "source_fiber_sha256")?;
    }
    if body.independent_seed_count < 5 {
        bail!("candidate advancement requires at least five independent seeds");
    }
    if !body.minimum_lane_effective_sample_size.is_finite()
        || body.minimum_lane_effective_sample_size <= 0.0
        || !body.primary_effect_confidence_width.is_finite()
        || body.primary_effect_confidence_width < 0.0
    {
        bail!("candidate uncertainty fields are invalid");
    }
    if body.objectives.is_empty()
        || body
            .objectives
            .iter()
            .any(|(name, value)| name.trim().is_empty() || !value.is_finite() || *value == -0.0)
    {
        bail!("candidate objective vector is empty, non-finite, or non-normalized");
    }
    for (name, hash) in &body.evidence_artifact_sha256 {
        if name.trim().is_empty() {
            bail!("candidate evidence artifact name is empty");
        }
        require_sha256(hash, "evidence_artifact_sha256")?;
    }
    Ok(())
}

/// One immutable supersession edge.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SupersessionEdge {
    /// Prior active champion content hash.
    pub superseded_candidate_sha256: String,
    /// New champion content hash.
    pub successor_candidate_sha256: String,
}

/// Immutable clarification/re-estimation edge for one stable candidate identity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CandidateRevisionEdge {
    /// Prior observation content hash.
    pub prior_candidate_sha256: String,
    /// Clarified observation content hash.
    pub revised_candidate_sha256: String,
}

/// One append transaction in the registry hash chain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "record_type", rename_all = "snake_case")]
pub enum RegistryAction {
    /// Declare registry identity and immutable objective schema.
    Initialize {
        /// Content address of the objective declaration.
        registry_id_sha256: String,
        /// Sorted objective definitions.
        objectives: Vec<ObjectiveDefinition>,
    },
    /// Observe and atomically adjudicate one candidate.
    AdvanceCandidate {
        /// Complete candidate record; always preserved.
        candidate: CandidateRecord,
        /// Whether the candidate joined the active Pareto frontier.
        promoted: bool,
        /// Active champions dominated by this candidate.
        supersedes: Vec<SupersessionEdge>,
    },
}

/// Hashable record body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegistryRecordBody {
    /// Monotonic record sequence.
    pub sequence: u64,
    /// SHA-256 of the preceding record or the all-zero genesis hash.
    pub previous_record_sha256: String,
    /// Atomic action.
    pub action: RegistryAction,
}

/// Content-addressed append-only registry record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegistryRecord {
    /// SHA-256 of `body`.
    pub record_sha256: String,
    /// Complete record body.
    pub body: RegistryRecordBody,
}

impl RegistryRecord {
    fn materialize(body: RegistryRecordBody) -> Result<Self> {
        require_sha256(&body.previous_record_sha256, "previous_record_sha256")?;
        Ok(Self {
            record_sha256: canonical_sha256(&body)?,
            body,
        })
    }

    fn verify(&self) -> Result<()> {
        require_sha256(&self.record_sha256, "record_sha256")?;
        require_sha256(&self.body.previous_record_sha256, "previous_record_sha256")?;
        if canonical_sha256(&self.body)? != self.record_sha256 {
            bail!("registry record content hash mismatch");
        }
        Ok(())
    }
}

/// In-memory verified state reconstructed from the append-only log.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParetoRegistry {
    /// Schema identifier.
    pub schema_kind: String,
    /// Schema version.
    pub schema_version: String,
    /// Immutable ordered records.
    pub records: Vec<RegistryRecord>,
    /// Complete observed population by content hash.
    pub population: BTreeMap<String, CandidateRecord>,
    /// Current non-dominated candidate hashes.
    pub active_champions: BTreeSet<String>,
    /// All historical supersession edges.
    pub supersession_edges: BTreeSet<SupersessionEdge>,
    /// All explicit same-identity clarification edges.
    pub revision_edges: BTreeSet<CandidateRevisionEdge>,
}

/// Result of one candidate append transaction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdvancementOutcome {
    /// Candidate content hash, always preserved.
    pub candidate_sha256: String,
    /// Whether it joined the active frontier.
    pub promoted: bool,
    /// Directed supersession edges emitted atomically.
    pub supersession_edges: Vec<SupersessionEdge>,
    /// Complete population size after the transaction.
    pub complete_population_count: usize,
    /// Active Pareto frontier size after the transaction.
    pub active_champion_count: usize,
}

impl ParetoRegistry {
    /// Initialize a registry with an immutable objective declaration.
    pub fn new(mut objectives: Vec<ObjectiveDefinition>) -> Result<Self> {
        validate_and_sort_objectives(&mut objectives)?;
        let registry_id_sha256 = canonical_sha256(&objectives)?;
        let first = RegistryRecord::materialize(RegistryRecordBody {
            sequence: 0,
            previous_record_sha256: GENESIS_HASH.to_string(),
            action: RegistryAction::Initialize {
                registry_id_sha256,
                objectives,
            },
        })?;
        let registry = Self {
            schema_kind: CANDIDATE_REGISTRY_SCHEMA.to_string(),
            schema_version: CANDIDATE_REGISTRY_VERSION.to_string(),
            records: vec![first],
            population: BTreeMap::new(),
            active_champions: BTreeSet::new(),
            supersession_edges: BTreeSet::new(),
            revision_edges: BTreeSet::new(),
        };
        registry.verify()?;
        Ok(registry)
    }

    /// Reconstruct a registry exclusively from immutable records.
    pub fn from_records(records: Vec<RegistryRecord>) -> Result<Self> {
        let mut registry = Self {
            schema_kind: CANDIDATE_REGISTRY_SCHEMA.to_string(),
            schema_version: CANDIDATE_REGISTRY_VERSION.to_string(),
            records,
            population: BTreeMap::new(),
            active_champions: BTreeSet::new(),
            supersession_edges: BTreeSet::new(),
            revision_edges: BTreeSet::new(),
        };
        let (population, champions, edges, revision_edges) = replay_records(&registry.records)?;
        registry.population = population;
        registry.active_champions = champions;
        registry.supersession_edges = edges;
        registry.revision_edges = revision_edges;
        registry.verify()?;
        Ok(registry)
    }

    /// Immutable objective definitions.
    pub fn objectives(&self) -> &[ObjectiveDefinition] {
        match &self.records[0].body.action {
            RegistryAction::Initialize { objectives, .. } => objectives,
            RegistryAction::AdvanceCandidate { .. } => {
                unreachable!("verified registry starts with initialize")
            }
        }
    }

    /// Preserve and Pareto-adjudicate a candidate in one logical transaction.
    pub fn advance(&mut self, candidate: CandidateRecord) -> Result<AdvancementOutcome> {
        self.verify()?;
        candidate.verify()?;
        validate_candidate_objectives(&candidate, self.objectives())?;
        if self.population.contains_key(&candidate.candidate_sha256) {
            bail!("candidate content already exists");
        }
        validate_revision_edge(&candidate, &self.population, &self.revision_edges)?;
        let dominated_by_active = self
            .active_champions
            .iter()
            .any(|hash| dominates(&self.population[hash], &candidate, self.objectives()));
        let promoted = !dominated_by_active;
        let mut supersedes = if promoted {
            self.active_champions
                .iter()
                .filter(|hash| dominates(&candidate, &self.population[*hash], self.objectives()))
                .map(|hash| SupersessionEdge {
                    superseded_candidate_sha256: hash.clone(),
                    successor_candidate_sha256: candidate.candidate_sha256.clone(),
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        supersedes.sort();
        let record = RegistryRecord::materialize(RegistryRecordBody {
            sequence: self.records.len() as u64,
            previous_record_sha256: self.records.last().unwrap().record_sha256.clone(),
            action: RegistryAction::AdvanceCandidate {
                candidate: candidate.clone(),
                promoted,
                supersedes: supersedes.clone(),
            },
        })?;
        self.records.push(record);
        self.population
            .insert(candidate.candidate_sha256.clone(), candidate.clone());
        if let Some(prior) = &candidate.body.revises_candidate_sha256 {
            self.revision_edges.insert(CandidateRevisionEdge {
                prior_candidate_sha256: prior.clone(),
                revised_candidate_sha256: candidate.candidate_sha256.clone(),
            });
        }
        if promoted {
            for edge in &supersedes {
                self.active_champions
                    .remove(&edge.superseded_candidate_sha256);
                self.supersession_edges.insert(edge.clone());
            }
            self.active_champions
                .insert(candidate.candidate_sha256.clone());
        }
        self.verify()?;
        Ok(AdvancementOutcome {
            candidate_sha256: candidate.candidate_sha256,
            promoted,
            supersession_edges: supersedes,
            complete_population_count: self.population.len(),
            active_champion_count: self.active_champions.len(),
        })
    }

    /// Recover the immutable outcome for an exactly identical prior append.
    ///
    /// This is intentionally narrower than candidate identity: only the exact
    /// content-addressed candidate is retry-idempotent. A scientifically
    /// clarified observation must remain a new record with an explicit
    /// `revises_candidate_sha256` edge.
    fn exact_retry_outcome(
        &self,
        candidate: &CandidateRecord,
    ) -> Result<Option<AdvancementOutcome>> {
        self.verify()?;
        candidate.verify()?;
        validate_candidate_objectives(candidate, self.objectives())?;
        let Some(existing) = self.population.get(&candidate.candidate_sha256) else {
            return Ok(None);
        };
        if existing != candidate {
            bail!("candidate SHA-256 collision with non-identical content");
        }
        let mut matching_actions = self.records.iter().filter_map(|record| {
            let RegistryAction::AdvanceCandidate {
                candidate: recorded,
                promoted,
                supersedes,
            } = &record.body.action
            else {
                return None;
            };
            (recorded.candidate_sha256 == candidate.candidate_sha256)
                .then_some((*promoted, supersedes.clone()))
        });
        let (promoted, supersession_edges) = matching_actions
            .next()
            .context("population candidate lacks its immutable advancement record")?;
        if matching_actions.next().is_some() {
            bail!("candidate appears in more than one immutable advancement record");
        }
        Ok(Some(AdvancementOutcome {
            candidate_sha256: candidate.candidate_sha256.clone(),
            promoted,
            supersession_edges,
            complete_population_count: self.population.len(),
            active_champion_count: self.active_champions.len(),
        }))
    }

    /// Verify hash chain, exact replay, complete population, and Pareto frontier.
    pub fn verify(&self) -> Result<()> {
        if self.schema_kind != CANDIDATE_REGISTRY_SCHEMA
            || self.schema_version != CANDIDATE_REGISTRY_VERSION
        {
            bail!("unsupported candidate registry schema");
        }
        let (population, champions, edges, revision_edges) = replay_records(&self.records)?;
        if population != self.population
            || champions != self.active_champions
            || edges != self.supersession_edges
            || revision_edges != self.revision_edges
        {
            bail!("candidate registry materialized state disagrees with immutable replay");
        }
        let expected_champions = pareto_frontier(&population, self.objectives());
        if expected_champions != self.active_champions {
            bail!("active champion set is not the exact deterministic Pareto frontier");
        }
        Ok(())
    }
}

fn validate_and_sort_objectives(objectives: &mut Vec<ObjectiveDefinition>) -> Result<()> {
    objectives.sort_by(|a, b| a.objective_id.cmp(&b.objective_id));
    if objectives.is_empty()
        || objectives
            .iter()
            .any(|objective| objective.objective_id.trim().is_empty())
        || objectives
            .windows(2)
            .any(|pair| pair[0].objective_id == pair[1].objective_id)
    {
        bail!("objective declarations must be non-empty and uniquely named");
    }
    Ok(())
}

fn validate_candidate_objectives(
    candidate: &CandidateRecord,
    definitions: &[ObjectiveDefinition],
) -> Result<()> {
    let expected = definitions
        .iter()
        .map(|definition| definition.objective_id.as_str())
        .collect::<BTreeSet<_>>();
    let actual = candidate
        .body
        .objectives
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if actual != expected {
        bail!("candidate objective vector does not exactly match registry declaration");
    }
    Ok(())
}

fn dominates(
    left: &CandidateRecord,
    right: &CandidateRecord,
    definitions: &[ObjectiveDefinition],
) -> bool {
    let mut strictly_better = false;
    for definition in definitions {
        let a = left.body.objectives[&definition.objective_id];
        let b = right.body.objectives[&definition.objective_id];
        match definition.direction {
            ObjectiveDirection::Maximize => {
                if a < b {
                    return false;
                }
                strictly_better |= a > b;
            }
            ObjectiveDirection::Minimize => {
                if a > b {
                    return false;
                }
                strictly_better |= a < b;
            }
        }
    }
    strictly_better
}

fn pareto_frontier(
    population: &BTreeMap<String, CandidateRecord>,
    definitions: &[ObjectiveDefinition],
) -> BTreeSet<String> {
    population
        .iter()
        .filter(|(hash, candidate)| {
            !population.iter().any(|(other_hash, other)| {
                other_hash != *hash && dominates(other, candidate, definitions)
            })
        })
        .map(|(hash, _)| hash.clone())
        .collect()
}

type ReplayState = (
    BTreeMap<String, CandidateRecord>,
    BTreeSet<String>,
    BTreeSet<SupersessionEdge>,
    BTreeSet<CandidateRevisionEdge>,
);

fn replay_records(records: &[RegistryRecord]) -> Result<ReplayState> {
    if records.is_empty() {
        bail!("candidate registry has no initialization record");
    }
    let mut previous = GENESIS_HASH.to_string();
    let mut definitions: Option<Vec<ObjectiveDefinition>> = None;
    let mut population: BTreeMap<String, CandidateRecord> = BTreeMap::new();
    let mut champions: BTreeSet<String> = BTreeSet::new();
    let mut edges = BTreeSet::new();
    let mut revision_edges = BTreeSet::new();
    for (index, record) in records.iter().enumerate() {
        record.verify()?;
        if record.body.sequence != index as u64 || record.body.previous_record_sha256 != previous {
            bail!("candidate registry sequence or previous-hash chain is broken");
        }
        match &record.body.action {
            RegistryAction::Initialize {
                registry_id_sha256,
                objectives,
            } => {
                if index != 0 || definitions.is_some() {
                    bail!("registry initialize action must occur exactly once at sequence zero");
                }
                let mut normalized = objectives.clone();
                validate_and_sort_objectives(&mut normalized)?;
                if &normalized != objectives || canonical_sha256(objectives)? != *registry_id_sha256
                {
                    bail!("registry objective declaration is not canonical");
                }
                definitions = Some(objectives.clone());
            }
            RegistryAction::AdvanceCandidate {
                candidate,
                promoted,
                supersedes,
            } => {
                let defs = definitions
                    .as_deref()
                    .context("candidate precedes registry initialization")?;
                candidate.verify()?;
                validate_candidate_objectives(candidate, defs)?;
                if population.contains_key(&candidate.candidate_sha256) {
                    bail!("duplicate candidate in immutable registry");
                }
                validate_revision_edge(candidate, &population, &revision_edges)?;
                let expected_promoted = !champions
                    .iter()
                    .any(|hash| dominates(&population[hash], candidate, defs));
                let mut expected_edges = if expected_promoted {
                    champions
                        .iter()
                        .filter(|hash| dominates(candidate, &population[*hash], defs))
                        .map(|hash| SupersessionEdge {
                            superseded_candidate_sha256: hash.clone(),
                            successor_candidate_sha256: candidate.candidate_sha256.clone(),
                        })
                        .collect::<Vec<_>>()
                } else {
                    Vec::new()
                };
                expected_edges.sort();
                if *promoted != expected_promoted || *supersedes != expected_edges {
                    bail!("stored candidate adjudication is not the deterministic Pareto result");
                }
                population.insert(candidate.candidate_sha256.clone(), candidate.clone());
                if let Some(prior) = &candidate.body.revises_candidate_sha256 {
                    revision_edges.insert(CandidateRevisionEdge {
                        prior_candidate_sha256: prior.clone(),
                        revised_candidate_sha256: candidate.candidate_sha256.clone(),
                    });
                }
                if *promoted {
                    for edge in supersedes {
                        if !champions.remove(&edge.superseded_candidate_sha256) {
                            bail!("supersession edge does not originate at an active champion");
                        }
                        edges.insert(edge.clone());
                    }
                    champions.insert(candidate.candidate_sha256.clone());
                }
            }
        }
        previous = record.record_sha256.clone();
    }
    if definitions.is_none() {
        bail!("registry initialization missing");
    }
    Ok((population, champions, edges, revision_edges))
}

fn validate_revision_edge(
    candidate: &CandidateRecord,
    population: &BTreeMap<String, CandidateRecord>,
    existing_edges: &BTreeSet<CandidateRevisionEdge>,
) -> Result<()> {
    let same_identity = population
        .values()
        .filter(|prior| prior.body.candidate_id == candidate.body.candidate_id)
        .collect::<Vec<_>>();
    match (
        &candidate.body.revises_candidate_sha256,
        same_identity.is_empty(),
    ) {
        (None, true) => Ok(()),
        (None, false) => {
            bail!("repeated stable candidate identity requires an explicit revision edge")
        }
        (Some(_), true) => bail!("revision target cannot precede the stable candidate identity"),
        (Some(prior_hash), false) => {
            let prior = population
                .get(prior_hash)
                .context("candidate revision target is absent from prior population")?;
            if prior.body.candidate_id != candidate.body.candidate_id {
                bail!("candidate revision target has a different stable identity");
            }
            if existing_edges
                .iter()
                .any(|edge| edge.prior_candidate_sha256 == *prior_hash)
            {
                bail!("candidate revision chain cannot branch from an already revised observation");
            }
            // Targets are necessarily prior immutable records, so a cycle cannot
            // be introduced. Verify no existing path already reaches the target.
            let mut cursor = Some(prior_hash.as_str());
            let mut visited = BTreeSet::new();
            while let Some(hash) = cursor {
                if !visited.insert(hash.to_string()) {
                    bail!("cycle detected in candidate revision lineage");
                }
                cursor = population
                    .get(hash)
                    .and_then(|record| record.body.revises_candidate_sha256.as_deref());
            }
            Ok(())
        }
    }
}

/// Load and verify an append-only NDJSON registry.
pub fn load_append_only_registry(path: &Path) -> Result<ParetoRegistry> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("inspect registry {}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        bail!("candidate registry path must be a regular non-symlink file");
    }
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)
        .with_context(|| format!("open registry {}", path.display()))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .with_context(|| format!("read registry {}", path.display()))?;
    parse_registry_bytes(&bytes)
}

fn parse_registry_bytes(bytes: &[u8]) -> Result<ParetoRegistry> {
    if bytes.is_empty() || bytes.last() != Some(&b'\n') {
        bail!("candidate registry lacks a complete newline-terminated durable record");
    }
    let text = std::str::from_utf8(bytes).context("candidate registry is not UTF-8 NDJSON")?;
    let mut records = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            bail!("blank line in append-only registry at {}", line_index + 1);
        }
        records.push(
            serde_json::from_str::<RegistryRecord>(line)
                .with_context(|| format!("parse registry line {}", line_index + 1))?,
        );
    }
    ParetoRegistry::from_records(records)
}

/// Durably append one candidate transaction under an exclusive file lock.
///
/// Existing bytes are never truncated or rewritten. On every append the full
/// hash chain is re-read and verified before new records are admitted.
pub fn commit_candidate_append_only(
    path: &Path,
    objectives: Vec<ObjectiveDefinition>,
    candidate: CandidateRecord,
) -> Result<AdvancementOutcome> {
    if let Ok(metadata) = std::fs::symlink_metadata(path) {
        if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
            bail!("candidate registry path must be a regular non-symlink file");
        }
    }
    let mut file = OpenOptions::new()
        .read(true)
        .append(true)
        .create(true)
        .mode(0o600)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_DSYNC)
        .open(path)
        .with_context(|| format!("open append-only registry {}", path.display()))?;
    let _lock = ExclusiveFlock::lock(&file)?;
    file.seek(SeekFrom::Start(0))?;
    let mut existing = Vec::new();
    file.read_to_end(&mut existing)?;
    let (mut registry, first_new_record) = if existing.is_empty() {
        (ParetoRegistry::new(objectives)?, 0usize)
    } else {
        let registry = parse_registry_bytes(&existing)?;
        let mut normalized = objectives;
        validate_and_sort_objectives(&mut normalized)?;
        if registry.objectives() != normalized {
            bail!("requested objective declaration differs from durable registry");
        }
        let first_new_record = registry.records.len();
        (registry, first_new_record)
    };
    // A process may lose its acknowledgement after the O_DSYNC append and
    // retry the same content-addressed transaction. Recover the immutable
    // original decision without emitting a second record or changing bytes.
    if let Some(outcome) = registry.exact_retry_outcome(&candidate)? {
        drop(_lock);
        return Ok(outcome);
    }
    let outcome = registry.advance(candidate)?;
    let mut appended = Vec::new();
    for record in &registry.records[first_new_record..] {
        serde_json::to_writer(&mut appended, record)?;
        appended.push(b'\n');
    }
    file.write_all(&appended)?;
    file.sync_data()?;
    sync_parent_directory(path)?;
    drop(_lock);
    let reloaded = load_append_only_registry(path)?;
    if reloaded != registry {
        bail!("durable candidate registry does not reconstruct after append");
    }
    Ok(outcome)
}

fn sync_parent_directory(path: &Path) -> Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let directory = File::open(parent)
        .with_context(|| format!("open registry parent directory {}", parent.display()))?;
    directory
        .sync_all()
        .with_context(|| format!("fsync registry parent directory {}", parent.display()))
}

struct ExclusiveFlock {
    fd: RawFd,
}

impl ExclusiveFlock {
    fn lock(file: &File) -> Result<Self> {
        let fd = file.as_raw_fd();
        let rc = unsafe { libc::flock(fd, libc::LOCK_EX) };
        if rc != 0 {
            return Err(std::io::Error::last_os_error()).context("lock candidate registry");
        }
        Ok(Self { fd })
    }
}

impl Drop for ExclusiveFlock {
    fn drop(&mut self) {
        unsafe {
            libc::flock(self.fd, libc::LOCK_UN);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> String {
        format!("{byte:02x}").repeat(32)
    }

    fn definitions() -> Vec<ObjectiveDefinition> {
        vec![
            ObjectiveDefinition {
                objective_id: "response_specificity".to_string(),
                direction: ObjectiveDirection::Maximize,
            },
            ObjectiveDefinition {
                objective_id: "uncertainty_width".to_string(),
                direction: ObjectiveDirection::Minimize,
            },
        ]
    }

    fn candidate(id: &str, specificity: f64, uncertainty: f64) -> CandidateRecord {
        CandidateRecord::materialize(CandidateBody {
            candidate_id: id.to_string(),
            component_correspondence_sha256: digest(7),
            revises_candidate_sha256: None,
            site_key: format!("site-{id}"),
            control_classification: PocketResponseClass::HypothesisFragmentStabilized,
            matched_control_result_sha256: digest(1),
            source_fiber_sha256: vec![digest(id.as_bytes()[0])],
            independent_seed_count: 5,
            minimum_lane_effective_sample_size: 100.0,
            primary_effect_confidence_width: uncertainty,
            objectives: BTreeMap::from([
                ("response_specificity".to_string(), specificity),
                ("uncertainty_width".to_string(), uncertainty),
            ]),
            evidence_artifact_sha256: BTreeMap::from([("pathb".to_string(), digest(2))]),
        })
        .unwrap()
    }

    #[test]
    fn complete_population_and_append_only_supersession_are_preserved() {
        let mut registry = ParetoRegistry::new(definitions()).unwrap();
        let first = candidate("a", 0.5, 0.2);
        let first_hash = first.candidate_sha256.clone();
        assert!(registry.advance(first).unwrap().promoted);
        let inferior = candidate("b", 0.4, 0.3);
        let inferior_hash = inferior.candidate_sha256.clone();
        assert!(!registry.advance(inferior).unwrap().promoted);
        let superior = candidate("c", 0.8, 0.1);
        let superior_hash = superior.candidate_sha256.clone();
        let outcome = registry.advance(superior).unwrap();
        assert!(outcome.promoted);
        assert_eq!(registry.population.len(), 3);
        assert!(registry.population.contains_key(&inferior_hash));
        assert_eq!(
            registry.active_champions,
            BTreeSet::from([superior_hash.clone()])
        );
        assert!(registry.supersession_edges.contains(&SupersessionEdge {
            superseded_candidate_sha256: first_hash,
            successor_candidate_sha256: superior_hash,
        }));
        registry.verify().unwrap();
    }

    #[test]
    fn pareto_frontier_is_deterministic_under_input_permutation() {
        let candidates = vec![
            candidate("a", 0.8, 0.3),
            candidate("b", 0.6, 0.1),
            candidate("c", 0.4, 0.4),
        ];
        let expected = candidates[0..2]
            .iter()
            .map(|candidate| candidate.candidate_sha256.clone())
            .collect::<BTreeSet<_>>();
        let mut forward = ParetoRegistry::new(definitions()).unwrap();
        for candidate in candidates.clone() {
            forward.advance(candidate).unwrap();
        }
        let mut reverse = ParetoRegistry::new(definitions()).unwrap();
        for candidate in candidates.into_iter().rev() {
            reverse.advance(candidate).unwrap();
        }
        assert_eq!(forward.active_champions, expected);
        assert_eq!(reverse.active_champions, expected);
    }

    #[test]
    fn hash_chain_tampering_fails_closed() {
        let mut registry = ParetoRegistry::new(definitions()).unwrap();
        registry.advance(candidate("a", 0.5, 0.2)).unwrap();
        registry.records[1].body.previous_record_sha256 = digest(99);
        assert!(registry.verify().is_err());
    }

    #[test]
    fn durable_append_never_rewrites_existing_prefix() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("champions.ndjson");
        commit_candidate_append_only(&path, definitions(), candidate("a", 0.5, 0.2)).unwrap();
        let prefix = std::fs::read(&path).unwrap();
        commit_candidate_append_only(&path, definitions(), candidate("b", 0.8, 0.1)).unwrap();
        let after = std::fs::read(&path).unwrap();
        assert_eq!(&after[..prefix.len()], prefix.as_slice());
        let registry = load_append_only_registry(&path).unwrap();
        assert_eq!(registry.population.len(), 2);
        registry.verify().unwrap();
    }

    #[test]
    fn durable_exact_retry_is_byte_idempotent() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("champions.ndjson");
        let observation = candidate("a", 0.5, 0.2);
        let first =
            commit_candidate_append_only(&path, definitions(), observation.clone()).unwrap();
        let committed = std::fs::read(&path).unwrap();
        let retry = commit_candidate_append_only(&path, definitions(), observation).unwrap();
        let after_retry = std::fs::read(&path).unwrap();
        assert_eq!(after_retry, committed);
        assert_eq!(retry, first);
        let registry = load_append_only_registry(&path).unwrap();
        assert_eq!(registry.population.len(), 1);
        assert_eq!(registry.records.len(), 2);
        registry.verify().unwrap();
    }

    #[test]
    fn clarified_observation_preserves_prior_and_revision_lineage() {
        let mut registry = ParetoRegistry::new(definitions()).unwrap();
        let prior = candidate("same-site", 0.5, 0.2);
        let prior_hash = prior.candidate_sha256.clone();
        registry.advance(prior).unwrap();
        let mut revised_body = candidate("new-materialization", 0.8, 0.1).body;
        revised_body.candidate_id = "same-site".to_string();
        revised_body.revises_candidate_sha256 = Some(prior_hash.clone());
        let revised = CandidateRecord::materialize(revised_body).unwrap();
        let revised_hash = revised.candidate_sha256.clone();
        registry.advance(revised).unwrap();
        assert_eq!(registry.population.len(), 2);
        assert!(registry.population.contains_key(&prior_hash));
        assert!(registry.population.contains_key(&revised_hash));
        assert!(registry.revision_edges.contains(&CandidateRevisionEdge {
            prior_candidate_sha256: prior_hash,
            revised_candidate_sha256: revised_hash,
        }));
        registry.verify().unwrap();
    }
}
