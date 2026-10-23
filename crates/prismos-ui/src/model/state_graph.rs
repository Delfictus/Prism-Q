//! Parser + in-memory graph for `PRISM_NATIVE_STATE_GRAPH.jsonl`.
//!
//! The file is JSON-Lines. Each line is one record. There are two record
//! shapes, distinguished by `record_type`:
//!
//! * `state_graph_node` — a scientific state vertex, a transition (edge), or a
//!   non-graph annotation (next-operator recommendation, dense-tensor summary,
//!   validation state). Vertices carry `vertex_id`; transitions carry
//!   `source_vertex_id` + `target_vertex_id` + `transition_id`.
//! * `orphan_check` — a trailing summary record carrying the legal
//!   `state_object_class` enumeration and the orphan verdict.
//!
//! The headline Beta2.0 requirement is to **distinguish native runtime
//! emissions from post-run / external-derived artifacts**. That distinction is
//! carried by [`RuntimeOrigin`] (+ the boolean `runtime_native`), surfaced here
//! so the viewer can colour/badge each node accordingly.

use serde::Deserialize;

/// Provenance of an emitted artifact. The two `RUNTIME_*` values are the only
/// "PrismOS-native" origins; everything else is derived after the run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeOrigin {
    /// Emitted inline by the runtime itself.
    RuntimeInline,
    /// Emitted by a subprocess the PrismOS runtime orchestrated.
    RuntimeSubprocessOrchestrated,
    /// Derived after the run from runtime outputs.
    PostRunDerived,
    /// Derived from an external (non-PrismOS) tool/source.
    ExternalDerived,
    /// A blocked state — the evidence could not be produced.
    Blocked,
    /// Any value not in the known set.
    Other(String),
}

impl RuntimeOrigin {
    pub fn parse(value: &str) -> Self {
        match value {
            "RUNTIME_INLINE" => RuntimeOrigin::RuntimeInline,
            "RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS" => {
                RuntimeOrigin::RuntimeSubprocessOrchestrated
            }
            "POST_RUN_DERIVED" => RuntimeOrigin::PostRunDerived,
            "EXTERNAL_DERIVED" => RuntimeOrigin::ExternalDerived,
            "BLOCKED" => RuntimeOrigin::Blocked,
            other => RuntimeOrigin::Other(other.to_string()),
        }
    }

    /// Whether this origin is a PrismOS-native runtime emission. This is the
    /// load-bearing distinction the brief asks the UI to make explicit.
    pub fn is_native(&self) -> bool {
        matches!(
            self,
            RuntimeOrigin::RuntimeInline | RuntimeOrigin::RuntimeSubprocessOrchestrated
        )
    }

    /// A short human label.
    pub fn label(&self) -> String {
        match self {
            RuntimeOrigin::RuntimeInline => "RUNTIME_INLINE".into(),
            RuntimeOrigin::RuntimeSubprocessOrchestrated => {
                "RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS".into()
            }
            RuntimeOrigin::PostRunDerived => "POST_RUN_DERIVED".into(),
            RuntimeOrigin::ExternalDerived => "EXTERNAL_DERIVED".into(),
            RuntimeOrigin::Blocked => "BLOCKED".into(),
            RuntimeOrigin::Other(value) => value.clone(),
        }
    }

    /// CSS class name used by the GTK theme to colour this origin.
    pub fn css_class(&self) -> &'static str {
        match self {
            RuntimeOrigin::RuntimeInline | RuntimeOrigin::RuntimeSubprocessOrchestrated => {
                "origin-native"
            }
            RuntimeOrigin::PostRunDerived => "origin-derived",
            RuntimeOrigin::ExternalDerived => "origin-external",
            RuntimeOrigin::Blocked => "origin-blocked",
            RuntimeOrigin::Other(_) => "origin-unknown",
        }
    }
}

/// A raw record as it appears on a single JSONL line. Every field is optional
/// so both record shapes deserialize through the same struct.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct RawRecord {
    pub record_type: Option<String>,
    pub artifact_kind: Option<String>,
    pub artifact_sha256: Option<String>,
    pub decode_proof_pass: Option<bool>,
    pub emission_unix_ns: Option<String>,
    pub is_orphan: Option<bool>,
    pub json_path: Option<String>,
    pub operator_name: Option<String>,
    pub qnanl_packet_path: Option<String>,
    pub run_pid: Option<i64>,
    pub runtime_native: Option<bool>,
    pub runtime_origin: Option<String>,
    pub schema_kind: Option<String>,
    pub state_object_class: Option<String>,
    pub source_vertex_id: Option<String>,
    pub target_vertex_id: Option<String>,
    pub tokenizer_family: Option<String>,
    pub transition_id: Option<String>,
    pub vertex_id: Option<String>,
    // orphan_check fields
    pub total_artifacts: Option<i64>,
    pub orphan_count: Option<i64>,
    pub verdict: Option<String>,
    pub valid_state_object_classes: Option<Vec<String>>,
}

/// A scientific state vertex (a graph node).
#[derive(Debug, Clone)]
pub struct Vertex {
    pub id: String,
    pub state_object_class: String,
    pub artifact_kind: String,
    pub origin: RuntimeOrigin,
    pub runtime_native: bool,
    pub decode_proof_pass: bool,
    pub json_path: Option<String>,
    pub qnanl_packet_path: Option<String>,
    pub tokenizer_family: Option<String>,
    pub artifact_sha256: Option<String>,
}

/// A transition (a graph edge) between two vertices.
#[derive(Debug, Clone)]
pub struct Edge {
    pub id: String,
    pub source: String,
    pub target: String,
    pub operator_name: Option<String>,
    pub origin: RuntimeOrigin,
    pub runtime_native: bool,
    pub decode_proof_pass: bool,
}

/// A non-graph annotation record (recommendation, dense-tensor summary, …).
#[derive(Debug, Clone)]
pub struct Annotation {
    pub artifact_kind: String,
    pub state_object_class: String,
    pub origin: RuntimeOrigin,
    pub runtime_native: bool,
    pub json_path: Option<String>,
}

/// The trailing orphan-check verdict.
#[derive(Debug, Clone, Default)]
pub struct OrphanCheck {
    pub total_artifacts: i64,
    pub orphan_count: i64,
    pub verdict: String,
    pub valid_state_object_classes: Vec<String>,
}

/// The whole parsed state graph.
#[derive(Debug, Clone, Default)]
pub struct StateGraph {
    pub vertices: Vec<Vertex>,
    pub edges: Vec<Edge>,
    pub annotations: Vec<Annotation>,
    pub orphan_check: Option<OrphanCheck>,
    /// Lines that failed to parse, for honest diagnostics.
    pub parse_errors: Vec<String>,
}

impl StateGraph {
    /// Parse a JSONL document. Blank lines are ignored; malformed lines are
    /// collected into `parse_errors` rather than aborting the whole graph.
    pub fn parse_jsonl(text: &str) -> Self {
        let mut graph = StateGraph::default();
        for (idx, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let record: RawRecord = match serde_json::from_str(line) {
                Ok(record) => record,
                Err(err) => {
                    graph
                        .parse_errors
                        .push(format!("line {}: {}", idx + 1, err));
                    continue;
                }
            };
            graph.ingest(record);
        }
        graph
    }

    fn ingest(&mut self, record: RawRecord) {
        match record.record_type.as_deref() {
            Some("orphan_check") => {
                self.orphan_check = Some(OrphanCheck {
                    total_artifacts: record.total_artifacts.unwrap_or_default(),
                    orphan_count: record.orphan_count.unwrap_or_default(),
                    verdict: record.verdict.unwrap_or_default(),
                    valid_state_object_classes: record
                        .valid_state_object_classes
                        .unwrap_or_default(),
                });
            }
            // Default: a graph node (state_graph_node) or anything node-shaped.
            _ => self.ingest_node(record),
        }
    }

    fn ingest_node(&mut self, record: RawRecord) {
        let origin = record
            .runtime_origin
            .as_deref()
            .map(RuntimeOrigin::parse)
            .unwrap_or_else(|| RuntimeOrigin::Other("UNSPECIFIED".into()));
        let runtime_native = record.runtime_native.unwrap_or(false);
        let decode_proof_pass = record.decode_proof_pass.unwrap_or(false);

        // A transition carries source + target (+ a transition id).
        if record.source_vertex_id.is_some() && record.target_vertex_id.is_some() {
            self.edges.push(Edge {
                id: record
                    .transition_id
                    .clone()
                    .unwrap_or_else(|| "txn_unknown".into()),
                source: record.source_vertex_id.clone().unwrap(),
                target: record.target_vertex_id.clone().unwrap(),
                operator_name: record.operator_name.clone(),
                origin,
                runtime_native,
                decode_proof_pass,
            });
            return;
        }

        // A vertex carries a vertex id.
        if let Some(id) = record.vertex_id.clone() {
            self.vertices.push(Vertex {
                id,
                state_object_class: record
                    .state_object_class
                    .clone()
                    .unwrap_or_else(|| "Unknown".into()),
                artifact_kind: record.artifact_kind.clone().unwrap_or_default(),
                origin,
                runtime_native,
                decode_proof_pass,
                json_path: record.json_path.clone(),
                qnanl_packet_path: record.qnanl_packet_path.clone(),
                tokenizer_family: record.tokenizer_family.clone(),
                artifact_sha256: record.artifact_sha256.clone(),
            });
            return;
        }

        // Otherwise a non-graph annotation row.
        self.annotations.push(Annotation {
            artifact_kind: record.artifact_kind.clone().unwrap_or_default(),
            state_object_class: record.state_object_class.clone().unwrap_or_default(),
            origin,
            runtime_native,
            json_path: record.json_path.clone(),
        });
    }

    /// Count of vertices whose origin is a PrismOS-native runtime emission.
    pub fn native_vertex_count(&self) -> usize {
        self.vertices
            .iter()
            .filter(|v| v.origin.is_native())
            .count()
    }

    /// Whether every parsed node's decode proof passed.
    pub fn all_decode_proofs_pass(&self) -> bool {
        self.vertices.iter().all(|v| v.decode_proof_pass)
            && self.edges.iter().all(|e| e.decode_proof_pass)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = r#"
{"record_type":"state_graph_node","artifact_kind":"pre_md_state_vertex","decode_proof_pass":true,"runtime_native":false,"runtime_origin":"POST_RUN_DERIVED","state_object_class":"ScientificStateVertex","vertex_id":"vtx_a","json_path":"/x/a.json","tokenizer_family":"DstwStateVertex"}
{"record_type":"state_graph_node","artifact_kind":"post_md_state_vertex","decode_proof_pass":true,"runtime_native":false,"runtime_origin":"EXTERNAL_DERIVED","state_object_class":"ScientificStateVertex","vertex_id":"vtx_b"}
{"record_type":"state_graph_node","artifact_kind":"md_state_transition","decode_proof_pass":true,"runtime_native":false,"runtime_origin":"POST_RUN_DERIVED","operator_name":"MD.INTEGRATE_VELOCITY_VERLET","state_object_class":"Transition","source_vertex_id":"vtx_a","target_vertex_id":"vtx_b","transition_id":"txn_1"}
{"record_type":"state_graph_node","artifact_kind":"blocked_gdp_bound_state_vertex","decode_proof_pass":true,"runtime_native":false,"runtime_origin":"BLOCKED","state_object_class":"BlockedState","vertex_id":"vtx_c"}
{"record_type":"state_graph_node","artifact_kind":"next_operator_recommendation","decode_proof_pass":true,"runtime_native":false,"runtime_origin":"POST_RUN_DERIVED","state_object_class":"NextOperatorRecommendation"}
{"record_type":"orphan_check","total_artifacts":10,"orphan_count":0,"verdict":"ALL_ARTIFACTS_PRISM_NATIVE","valid_state_object_classes":["ScientificStateVertex","Transition","BlockedState"]}
"#;

    #[test]
    fn parses_vertices_edges_annotations_and_orphan_check() {
        let graph = StateGraph::parse_jsonl(FIXTURE);
        assert_eq!(graph.vertices.len(), 3, "vtx_a, vtx_b, vtx_c");
        assert_eq!(graph.edges.len(), 1);
        assert_eq!(graph.annotations.len(), 1);
        assert!(graph.parse_errors.is_empty());

        let edge = &graph.edges[0];
        assert_eq!(edge.source, "vtx_a");
        assert_eq!(edge.target, "vtx_b");
        assert_eq!(edge.id, "txn_1");
        assert_eq!(
            edge.operator_name.as_deref(),
            Some("MD.INTEGRATE_VELOCITY_VERLET")
        );

        let orphan = graph.orphan_check.unwrap();
        assert_eq!(orphan.total_artifacts, 10);
        assert_eq!(orphan.verdict, "ALL_ARTIFACTS_PRISM_NATIVE");
        assert_eq!(orphan.valid_state_object_classes.len(), 3);
    }

    #[test]
    fn native_detection_matches_runtime_origin_semantics() {
        let graph = StateGraph::parse_jsonl(FIXTURE);
        // The fixture has no RUNTIME_* origins, so nothing is native.
        assert_eq!(graph.native_vertex_count(), 0);

        assert!(RuntimeOrigin::parse("RUNTIME_INLINE").is_native());
        assert!(RuntimeOrigin::parse("RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS").is_native());
        assert!(!RuntimeOrigin::parse("POST_RUN_DERIVED").is_native());
        assert!(!RuntimeOrigin::parse("EXTERNAL_DERIVED").is_native());
        assert!(!RuntimeOrigin::parse("BLOCKED").is_native());
    }

    #[test]
    fn css_classes_separate_native_from_derived() {
        assert_eq!(RuntimeOrigin::RuntimeInline.css_class(), "origin-native");
        assert_eq!(RuntimeOrigin::PostRunDerived.css_class(), "origin-derived");
        assert_eq!(
            RuntimeOrigin::ExternalDerived.css_class(),
            "origin-external"
        );
        assert_eq!(RuntimeOrigin::Blocked.css_class(), "origin-blocked");
    }

    #[test]
    fn malformed_lines_are_collected_not_fatal() {
        let graph = StateGraph::parse_jsonl(
            "{not json}\n{\"record_type\":\"state_graph_node\",\"vertex_id\":\"vtx_z\"}",
        );
        assert_eq!(graph.vertices.len(), 1);
        assert_eq!(graph.parse_errors.len(), 1);
    }
}
