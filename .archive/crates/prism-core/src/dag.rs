//! Transform/evidence DAG schema for run provenance.
//!
//! This module is intentionally data-only: it defines the serialized
//! provenance envelope and local consistency checks without touching runtime
//! execution, graph topology, CUDA, or artifact materialization.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::path::PathBuf;

/// Current transform/evidence DAG schema version.
pub const DAG_SCHEMA_VERSION: u32 = 1;

/// Transform/evidence DAG manifest for a single run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DagRunManifest {
    /// Version of the DAG envelope schema.
    pub schema_version: u32,
    /// Canonical run identifier minted by the caller.
    pub run_id: String,
    /// Target or structure name for the run.
    pub target: String,
    /// UTC creation timestamp supplied by the caller.
    pub created_utc: String,
    /// Nodes in the provenance graph.
    pub nodes: Vec<DagNode>,
    /// Directed relationships between nodes.
    pub edges: Vec<DagEdge>,
    /// Artifacts referenced by artifact nodes.
    pub artifacts: Vec<DagArtifactRef>,
    /// Invariants asserted or recorded for this manifest.
    pub invariants: Vec<DagInvariant>,
    /// Reserved bag for non-load-bearing fields.
    pub metadata: Value,
}

impl DagRunManifest {
    /// Creates an empty DAG manifest with schema version 1.
    pub fn new(
        run_id: impl Into<String>,
        target: impl Into<String>,
        created_utc: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: DAG_SCHEMA_VERSION,
            run_id: run_id.into(),
            target: target.into(),
            created_utc: created_utc.into(),
            nodes: Vec::new(),
            edges: Vec::new(),
            artifacts: Vec::new(),
            invariants: Vec::new(),
            metadata: serde_json::json!({}),
        }
    }

    /// Validates that every edge endpoint references an existing node id.
    pub fn validate_no_orphan_edges(&self) -> Result<(), String> {
        let node_ids: HashSet<&str> = self.nodes.iter().map(|node| node.id.as_str()).collect();

        for edge in &self.edges {
            if !node_ids.contains(edge.from.as_str()) {
                return Err(format!(
                    "edge from '{}' to '{}' references missing source node '{}'",
                    edge.from, edge.to, edge.from
                ));
            }

            if !node_ids.contains(edge.to.as_str()) {
                return Err(format!(
                    "edge from '{}' to '{}' references missing target node '{}'",
                    edge.from, edge.to, edge.to
                ));
            }
        }

        Ok(())
    }
}

/// Node in a transform/evidence DAG manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DagNode {
    /// Stable node identifier.
    pub id: String,
    /// Semantic node kind.
    pub kind: DagNodeKind,
    /// Human-readable node label.
    pub label: String,
    /// Optional stream id for stream-scoped nodes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_id: Option<u32>,
    /// Optional chunk id for chunk-scoped nodes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_id: Option<u64>,
    /// Optional protocol phase label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
    /// Optional reference to `DagArtifactRef::id`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_ref: Option<String>,
    /// Optional hexadecimal content hash.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hash: Option<String>,
    /// Reserved bag for node-specific metadata.
    pub metadata: Value,
}

impl DagNode {
    /// Creates a node with empty metadata and no optional scopes.
    pub fn new(id: impl Into<String>, kind: DagNodeKind, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            kind,
            label: label.into(),
            stream_id: None,
            chunk_id: None,
            phase: None,
            artifact_ref: None,
            hash: None,
            metadata: serde_json::json!({}),
        }
    }
}

/// Builds a deterministic node id from a node kind and stable caller-supplied components.
///
/// Components are normalized to lowercase ASCII id segments so equivalent
/// labels with whitespace or path separators do not produce unstable ids.
pub fn stable_dag_node_id<I, S>(kind: DagNodeKind, components: I) -> String
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut id = String::from(kind.stable_id_prefix());

    for component in components {
        id.push_str("::");
        id.push_str(&normalize_node_id_component(component.as_ref()));
    }

    id
}

fn normalize_node_id_component(component: &str) -> String {
    let mut normalized = String::new();
    let mut last_was_separator = false;

    for ch in component.trim().chars() {
        if ch.is_ascii_alphanumeric() {
            normalized.push(ch.to_ascii_lowercase());
            last_was_separator = false;
        } else if matches!(ch, '-' | '_' | '.') {
            normalized.push(ch);
            last_was_separator = false;
        } else if !last_was_separator {
            normalized.push('-');
            last_was_separator = true;
        }
    }

    let trimmed = normalized.trim_matches('-');
    if trimmed.is_empty() {
        "_".to_string()
    } else {
        trimmed.to_string()
    }
}

/// Directed edge in a transform/evidence DAG manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DagEdge {
    /// Source node id.
    pub from: String,
    /// Target node id.
    pub to: String,
    /// Semantic edge kind.
    pub kind: DagEdgeKind,
    /// Optional transform id, mirroring transform provenance vocabulary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transform: Option<String>,
    /// Reserved bag for edge-specific metadata.
    pub metadata: Value,
}

impl DagEdge {
    /// Creates an edge with empty metadata and no transform id.
    pub fn new(from: impl Into<String>, to: impl Into<String>, kind: DagEdgeKind) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            kind,
            transform: None,
            metadata: serde_json::json!({}),
        }
    }
}

/// Serialized artifact reference used by artifact nodes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DagArtifactRef {
    /// Stable artifact reference id.
    pub id: String,
    /// Filesystem path recorded by the producer.
    pub path: PathBuf,
    /// Artifact kind, usually the basename or logical artifact class.
    pub kind: String,
    /// Version of the referenced artifact schema when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema_version: Option<u32>,
    /// Artifact size in bytes.
    pub size_bytes: u64,
    /// Hexadecimal artifact hash or fingerprint.
    pub hash: String,
}

/// Invariant recorded in a transform/evidence DAG manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DagInvariant {
    /// Stable invariant id.
    pub id: String,
    /// Human-readable invariant description.
    pub description: String,
    /// Invariant status, such as "asserted", "unverified", or "violated".
    pub status: String,
    /// Node ids supporting the invariant status.
    pub evidence_node_ids: Vec<String>,
}

/// Semantic node kinds for the transform/evidence DAG.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub enum DagNodeKind {
    Run,
    InputTopology,
    RunConfig,
    Stream,
    Chunk,
    ProtocolPhase,
    GraphLaunch,
    G26Decision,
    F1Decision,
    F2RingStatus,
    F2WriteCommit,
    WhileIteration,
    ASCSnapshot,
    KCCSnapshot,
    SpikeBatch,
    VoxelSupportSet,
    ClusterEvent,
    SiteCandidate,
    RankedSite,
    StaticEquivalentPocket,
    DossierArtifact,
    ValidationMetric,
    BinaryArtifact,
    JsonArtifact,
}

impl DagNodeKind {
    /// Stable lowercase prefix used by generated node ids.
    pub fn stable_id_prefix(self) -> &'static str {
        match self {
            DagNodeKind::Run => "run",
            DagNodeKind::InputTopology => "input-topology",
            DagNodeKind::RunConfig => "run-config",
            DagNodeKind::Stream => "stream",
            DagNodeKind::Chunk => "chunk",
            DagNodeKind::ProtocolPhase => "protocol-phase",
            DagNodeKind::GraphLaunch => "graph-launch",
            DagNodeKind::G26Decision => "g26-decision",
            DagNodeKind::F1Decision => "f1-decision",
            DagNodeKind::F2RingStatus => "f2-ring-status",
            DagNodeKind::F2WriteCommit => "f2-write-commit",
            DagNodeKind::WhileIteration => "while-iteration",
            DagNodeKind::ASCSnapshot => "asc-snapshot",
            DagNodeKind::KCCSnapshot => "kcc-snapshot",
            DagNodeKind::SpikeBatch => "spike-batch",
            DagNodeKind::VoxelSupportSet => "voxel-support-set",
            DagNodeKind::ClusterEvent => "cluster-event",
            DagNodeKind::SiteCandidate => "site-candidate",
            DagNodeKind::RankedSite => "ranked-site",
            DagNodeKind::StaticEquivalentPocket => "static-equivalent-pocket",
            DagNodeKind::DossierArtifact => "dossier-artifact",
            DagNodeKind::ValidationMetric => "validation-metric",
            DagNodeKind::BinaryArtifact => "binary-artifact",
            DagNodeKind::JsonArtifact => "json-artifact",
        }
    }
}

/// Semantic edge kinds for the transform/evidence DAG.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub enum DagEdgeKind {
    Contains,
    Produces,
    DerivedFrom,
    Supports,
    TransformsTo,
    Validates,
    Rejects,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_manifest() -> DagRunManifest {
        let mut manifest = DagRunManifest::new("run-001", "4lpk", "2026-05-04T12:00:00Z");

        manifest.nodes.push(DagNode::new(
            "run::run-001",
            DagNodeKind::Run,
            "Run run-001",
        ));
        manifest.nodes.push(DagNode::new(
            "config::run-001",
            DagNodeKind::RunConfig,
            "Run configuration",
        ));
        let mut artifact_node = DagNode::new(
            "artifact::binding-sites",
            DagNodeKind::JsonArtifact,
            "binding_sites.json",
        );
        artifact_node.artifact_ref = Some("artifact-ref::binding-sites".to_string());
        manifest.nodes.push(artifact_node);

        manifest.edges.push(DagEdge::new(
            "run::run-001",
            "config::run-001",
            DagEdgeKind::Contains,
        ));
        manifest.edges.push(DagEdge::new(
            "run::run-001",
            "artifact::binding-sites",
            DagEdgeKind::Produces,
        ));

        manifest.artifacts.push(DagArtifactRef {
            id: "artifact-ref::binding-sites".to_string(),
            path: PathBuf::from("binding_sites.json"),
            kind: "binding_sites.json".to_string(),
            schema_version: Some(2),
            size_bytes: 128,
            hash: "0123456789abcdef".to_string(),
        });

        manifest.invariants.push(DagInvariant {
            id: "inv::no-orphan-edges".to_string(),
            description: "Every edge endpoint references a manifest node".to_string(),
            status: "asserted".to_string(),
            evidence_node_ids: vec!["run::run-001".to_string()],
        });

        manifest.metadata = serde_json::json!({ "replica_seed": 42_u64 });
        manifest
    }

    #[test]
    fn dag_manifest_serializes_round_trip() {
        let manifest = sample_manifest();

        let json = serde_json::to_string_pretty(&manifest).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["schema_version"], DAG_SCHEMA_VERSION);
        assert_eq!(value["nodes"][0]["kind"], "Run");
        assert_eq!(value["edges"][1]["kind"], "Produces");
        assert_eq!(
            value["nodes"][2]["artifact_ref"],
            "artifact-ref::binding-sites"
        );
        assert_eq!(value["artifacts"][0]["path"], "binding_sites.json");
        assert_eq!(value["artifacts"][0]["schema_version"], 2);

        let deserialized: DagRunManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, manifest);
    }

    #[test]
    fn dag_artifact_path_serializes_as_explicit_path_string() {
        let artifact = DagArtifactRef {
            id: "artifact-ref::f2-write-commit".to_string(),
            path: PathBuf::from("artifacts/f2/write_commit.json"),
            kind: "f2-write-commit".to_string(),
            schema_version: None,
            size_bytes: 256,
            hash: "abcdef0123456789".to_string(),
        };

        let json = serde_json::to_string(&artifact).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["path"], "artifacts/f2/write_commit.json");
        assert!(value.get("schema_version").is_none());

        let deserialized: DagArtifactRef = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, artifact);
    }

    #[test]
    fn dag_optional_node_fields_are_omitted_when_none() {
        let node = DagNode::new("run::run-001", DagNodeKind::Run, "Run run-001");

        let json = serde_json::to_string(&node).unwrap();
        assert!(!json.contains("stream_id"));
        assert!(!json.contains("chunk_id"));
        assert!(!json.contains("phase"));
        assert!(!json.contains("artifact_ref"));
        assert!(!json.contains("hash"));
    }

    #[test]
    fn dag_new_f2_node_kinds_serialize_round_trip() {
        let ring_status = DagNode::new(
            "f2-ring-status::run-001::stream-7",
            DagNodeKind::F2RingStatus,
            "F2 ring status",
        );
        let write_commit = DagNode::new(
            "f2-write-commit::run-001::chunk-9",
            DagNodeKind::F2WriteCommit,
            "F2 write commit",
        );

        let json = serde_json::to_string(&vec![ring_status.clone(), write_commit.clone()]).unwrap();
        assert!(json.contains("\"kind\":\"F2RingStatus\""));
        assert!(json.contains("\"kind\":\"F2WriteCommit\""));

        let deserialized: Vec<DagNode> = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, vec![ring_status, write_commit]);
    }

    #[test]
    fn dag_stable_node_id_helper_is_deterministic() {
        let first = stable_dag_node_id(
            DagNodeKind::F2WriteCommit,
            ["Run 001", "Stream/7", "Chunk:0009"],
        );
        let second = stable_dag_node_id(
            DagNodeKind::F2WriteCommit,
            ["Run 001", "Stream/7", "Chunk:0009"],
        );

        assert_eq!(first, second);
        assert_eq!(first, "f2-write-commit::run-001::stream-7::chunk-0009");
    }

    #[test]
    fn dag_stable_node_id_helper_handles_empty_components() {
        let node_id = stable_dag_node_id(DagNodeKind::F2RingStatus, ["  ", "Ring Status"]);

        assert_eq!(node_id, "f2-ring-status::_::ring-status");
    }

    #[test]
    fn dag_stable_node_id_helper_normalizes_equivalent_components() {
        let slash_separated = stable_dag_node_id(
            DagNodeKind::F2WriteCommit,
            ["Run 001", "Stream/7", "Chunk:0009"],
        );
        let space_separated = stable_dag_node_id(
            DagNodeKind::F2WriteCommit,
            ["run-001", "stream 7", "chunk 0009"],
        );

        assert_eq!(slash_separated, space_separated);
    }

    #[test]
    fn dag_stable_node_id_prefixes_are_schema_stable() {
        let expected = [
            (DagNodeKind::Run, "run"),
            (DagNodeKind::InputTopology, "input-topology"),
            (DagNodeKind::RunConfig, "run-config"),
            (DagNodeKind::Stream, "stream"),
            (DagNodeKind::Chunk, "chunk"),
            (DagNodeKind::ProtocolPhase, "protocol-phase"),
            (DagNodeKind::GraphLaunch, "graph-launch"),
            (DagNodeKind::G26Decision, "g26-decision"),
            (DagNodeKind::F1Decision, "f1-decision"),
            (DagNodeKind::F2RingStatus, "f2-ring-status"),
            (DagNodeKind::F2WriteCommit, "f2-write-commit"),
            (DagNodeKind::WhileIteration, "while-iteration"),
            (DagNodeKind::ASCSnapshot, "asc-snapshot"),
            (DagNodeKind::KCCSnapshot, "kcc-snapshot"),
            (DagNodeKind::SpikeBatch, "spike-batch"),
            (DagNodeKind::VoxelSupportSet, "voxel-support-set"),
            (DagNodeKind::ClusterEvent, "cluster-event"),
            (DagNodeKind::SiteCandidate, "site-candidate"),
            (DagNodeKind::RankedSite, "ranked-site"),
            (
                DagNodeKind::StaticEquivalentPocket,
                "static-equivalent-pocket",
            ),
            (DagNodeKind::DossierArtifact, "dossier-artifact"),
            (DagNodeKind::ValidationMetric, "validation-metric"),
            (DagNodeKind::BinaryArtifact, "binary-artifact"),
            (DagNodeKind::JsonArtifact, "json-artifact"),
        ];

        for (kind, prefix) in expected {
            assert_eq!(kind.stable_id_prefix(), prefix);
            assert_eq!(
                stable_dag_node_id(kind, ["Node 7"]),
                format!("{prefix}::node-7")
            );
        }
    }

    #[test]
    fn dag_no_orphan_edge_validation_accepts_known_endpoints() {
        let manifest = sample_manifest();

        assert!(manifest.validate_no_orphan_edges().is_ok());
    }

    #[test]
    fn dag_no_orphan_edge_validation_rejects_missing_source() {
        let mut manifest = sample_manifest();
        manifest.edges.push(DagEdge::new(
            "missing::node",
            "run::run-001",
            DagEdgeKind::Supports,
        ));

        let error = manifest.validate_no_orphan_edges().unwrap_err();
        assert!(error.contains("missing source node 'missing::node'"));
    }

    #[test]
    fn dag_no_orphan_edge_validation_rejects_missing_target() {
        let mut manifest = sample_manifest();
        manifest.edges.push(DagEdge::new(
            "run::run-001",
            "missing::node",
            DagEdgeKind::Supports,
        ));

        let error = manifest.validate_no_orphan_edges().unwrap_err();
        assert!(error.contains("missing target node 'missing::node'"));
    }
}
