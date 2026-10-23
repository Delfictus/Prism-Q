//! Typed operand ledger for the PrismOS native MD evidence pipeline (DSTW-T4 Step 2).
//!
//! Replaces the opcode-NAME/COUNT-only records (`md_native_runtime.rs:72`
//! `opcodes: Vec<String>`; `:102` `opcode_counts: BTreeMap<String,u64>`) with a
//! REAL typed [`OperandRecord`] carrying the 22-field operand schema the
//! DSTW-T4 plan (§1.C, Step 2) requires.
//!
//! HONESTY BOUNDARIES (do not violate — these mirror the plan's FORBIDDEN
//! GUARDRAILS):
//!  * `runtime_origin` is MANDATORY on every record. Only [`RuntimeOrigin::RuntimeInline`]
//!    and [`RuntimeOrigin::RuntimeSubprocessOrchestratedByPrismos`] are
//!    `PRISMOS_NATIVE` (see [`RuntimeOrigin::is_prismos_native`]). A
//!    from-existing-artifacts exerciser MUST be labeled
//!    [`RuntimeOrigin::PostRunManual`]; a packet that was only sealed after the
//!    run is [`RuntimeOrigin::PostRunSealOnly`].
//!  * Per-dispatch operand VALUES are not available from the engine today
//!    (`MdNativeKernelDispatch.opcodes` is `Vec<String>`). Records derived from
//!    the engine subprocess are therefore COUNT-level: per-operand columns stay
//!    `None` and `blocker_reason` names the gap. They are NEVER null-padded into
//!    a fake "typed" record.
//!  * Records that prismos itself drives (QNANL seals it performs inline, the
//!    State/modality flags it feeds the subprocess) get FULL typed operands, and
//!    every hash field is VALUE-verified: [`OperandRecord::typed_from_artifacts`]
//!    OPENS the file and computes the sha256, never stores a caller-supplied
//!    string blindly.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::Path;

pub const OPERAND_LEDGER_SCHEMA_KIND: &str = "prismcode_typed_operand_ledger";
pub const OPERAND_LEDGER_SCHEMA_VERSION: u32 = 1;

/// Honest provenance discriminator for every emitted operand record.
///
/// Only the first two variants qualify as `PRISMOS_NATIVE`. The string forms are
/// the canonical labels required by the directive.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimeOrigin {
    /// Produced inline, in-process, by the prismos runtime itself (e.g. a QNANL
    /// seal prismos performs during `md-run`). PRISMOS_NATIVE.
    RuntimeInline,
    /// Produced by an engine subprocess that prismos launched and orchestrated
    /// (`nhs_rt_full` via `prism-validate-and-run.sh`). PRISMOS_NATIVE.
    RuntimeSubprocessOrchestratedByPrismos,
    /// Produced by a separate, after-the-fact exerciser reading existing
    /// artifacts. NOT native — must be labeled honestly.
    PostRunManual,
    /// A `.prismqnanl` packet that was only sealed after the run finished; bytes
    /// round-trip but the runtime did not emit it in-loop. NOT native.
    PostRunSealOnly,
    /// Derived from an external / upstream artifact not produced by this prismos
    /// invocation (e.g. a prior-run dossier folded in). NOT native.
    ExternalDerived,
    /// The operation itself is blocked (precondition failure / engine-owner gap);
    /// the record exists only to name the blocker, no bytes were emitted by the
    /// runtime for it. NOT native.
    Blocked,
    /// DSTW-T4 Step h5 — a POST_RUN_DERIVED semantic reduction over an existing
    /// engine artifact (dense-tensor semantic summary). The bytes summarized DO
    /// exist (real source_hash), but the reduction is computed after the run from
    /// those artifacts; the inline-at-teardown native reduction is engine-owner.
    /// NOT native (distinct from `PostRunManual`: a deterministic derived summary,
    /// not a hand-run exerciser). Appended last (serialized by label, not ordinal).
    PostRunDerived,
}

impl RuntimeOrigin {
    pub fn as_str(self) -> &'static str {
        match self {
            RuntimeOrigin::RuntimeInline => "RUNTIME_INLINE",
            RuntimeOrigin::RuntimeSubprocessOrchestratedByPrismos => {
                "RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS"
            }
            RuntimeOrigin::PostRunManual => "POST_RUN_MANUAL",
            RuntimeOrigin::PostRunSealOnly => "POST_RUN_SEAL_ONLY",
            RuntimeOrigin::ExternalDerived => "EXTERNAL_DERIVED",
            RuntimeOrigin::Blocked => "BLOCKED",
            RuntimeOrigin::PostRunDerived => "POST_RUN_DERIVED",
        }
    }

    /// Parse a canonical label back to the enum (infallible callers use
    /// `.unwrap_or(PostRunSealOnly)` — the conservative honest default).
    pub fn from_label(label: &str) -> Option<RuntimeOrigin> {
        match label.trim() {
            "RUNTIME_INLINE" => Some(RuntimeOrigin::RuntimeInline),
            "RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS" => {
                Some(RuntimeOrigin::RuntimeSubprocessOrchestratedByPrismos)
            }
            "POST_RUN_MANUAL" => Some(RuntimeOrigin::PostRunManual),
            "POST_RUN_SEAL_ONLY" => Some(RuntimeOrigin::PostRunSealOnly),
            "EXTERNAL_DERIVED" => Some(RuntimeOrigin::ExternalDerived),
            "BLOCKED" => Some(RuntimeOrigin::Blocked),
            "POST_RUN_DERIVED" => Some(RuntimeOrigin::PostRunDerived),
            _ => None,
        }
    }

    /// True only for the two PRISMOS_NATIVE origins. `ExternalDerived` and
    /// `Blocked` are explicitly NOT native.
    pub fn is_prismos_native(self) -> bool {
        matches!(
            self,
            RuntimeOrigin::RuntimeInline | RuntimeOrigin::RuntimeSubprocessOrchestratedByPrismos
        )
    }
}

/// Semantic operator class for an operand row (DSTW-T4 Step h3).
///
/// Classifies WHICH PRISM subsystem the opcode belongs to. Derived from the
/// opcode name by [`OperatorClass::classify`] — every row carries one, never a
/// bare/absent label.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperatorClass {
    /// Molecular-dynamics stepping / force / kernel dispatch (engine subprocess).
    Md,
    /// Kinetic coherence / KCC layer.
    Kcc,
    /// Bayesian online change-point (BOCPD) layer.
    Bocpd,
    /// QNANL Primary Artifact Compression seal (SECTION_ZSTD body + NANL-Q4 seal).
    Qnanl,
    /// DSTW scientific-state graph (vertices / transitions / classification).
    Dstw,
    /// Path B post-MD evidence (binding sites / feasible tensors).
    PathB,
    /// PrismCode tokenizer state-graph rows.
    Tokenizer,
    /// Provenance / lineage record.
    Lineage,
    /// Validation / decode-proof / gate record.
    Validation,
}

impl OperatorClass {
    pub fn as_str(self) -> &'static str {
        match self {
            OperatorClass::Md => "MD",
            OperatorClass::Kcc => "KCC",
            OperatorClass::Bocpd => "BOCPD",
            OperatorClass::Qnanl => "QNANL",
            OperatorClass::Dstw => "DSTW",
            OperatorClass::PathB => "PATH_B",
            OperatorClass::Tokenizer => "TOKENIZER",
            OperatorClass::Lineage => "LINEAGE",
            OperatorClass::Validation => "VALIDATION",
        }
    }

    /// Classify an opcode/kernel name into its subsystem. Substring checks are
    /// ordered most-specific first so `MD.QNANL_SEAL` resolves to `Qnanl` (the
    /// seal subsystem) and `STATE.*` resolves to `Dstw`. Engine MD opcodes
    /// (`MD.*`, `KERNEL::*`) and the conservative fallback resolve to `Md`,
    /// matching the directive's rule that engine count-level dispatch rows carry
    /// `operator_class=MD`.
    pub fn classify(opcode_name: &str) -> OperatorClass {
        let up = opcode_name.to_ascii_uppercase();
        if up.contains("QNANL") || up.contains("NANL") {
            OperatorClass::Qnanl
        } else if up.contains("KCC") {
            OperatorClass::Kcc
        } else if up.contains("BOCPD") {
            OperatorClass::Bocpd
        } else if up.contains("DSTW") || up.starts_with("STATE.") {
            OperatorClass::Dstw
        } else if up.contains("PATHB") || up.contains("PATH_B") || up.contains("PATH.B") {
            OperatorClass::PathB
        } else if up.contains("TOKEN") || up.starts_with("TOK.") {
            OperatorClass::Tokenizer
        } else if up.contains("LINEAGE") {
            OperatorClass::Lineage
        } else if up.contains("VALID") || up.contains("DECODE_PROOF") {
            OperatorClass::Validation
        } else {
            // MD.*, KERNEL::*, and any unclassified engine dispatch.
            OperatorClass::Md
        }
    }
}

/// What an operand row's value actually targets (DSTW-T4 Step h3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperandTarget {
    /// A numeric tensor (coordinates / forces / signal grid / kinetic field).
    Tensor,
    /// A scientific-state vertex.
    State,
    /// A state-graph transition / edge.
    Transition,
    /// A sealed/serialized artifact (e.g. a `.prismqnanl` packet, json dossier).
    Artifact,
    /// A receipt / summary record.
    Receipt,
}

impl OperandTarget {
    pub fn as_str(self) -> &'static str {
        match self {
            OperandTarget::Tensor => "TENSOR",
            OperandTarget::State => "STATE",
            OperandTarget::Transition => "TRANSITION",
            OperandTarget::Artifact => "ARTIFACT",
            OperandTarget::Receipt => "RECEIPT",
        }
    }

    /// Default operand target inferred from the opcode name. Explicit overrides
    /// (DSTW state/transition rows) use [`OperandRecord::with_operand_target`].
    pub fn classify(opcode_name: &str) -> OperandTarget {
        let up = opcode_name.to_ascii_uppercase();
        if up.contains("VERTEX") {
            OperandTarget::State
        } else if up.contains("TRANSITION")
            || up.contains("OPERATOR_APPLY")
            || up.contains("CLASSIFY")
            || up.contains("NEXT_OPERATOR")
        {
            OperandTarget::Transition
        } else if up.contains("RECEIPT") {
            OperandTarget::Receipt
        } else if up.contains("SEAL") || up.contains("QNANL") {
            OperandTarget::Artifact
        } else {
            // engine MD opcode / kernel dispatch operates on tensors.
            OperandTarget::Tensor
        }
    }
}

/// VALUE-verified operand proof (DSTW-T4 Step h3): hash + shape + count + unit +
/// source_ref. This is NEVER null-padded — when a sub-proof is genuinely absent
/// (engine count-level, non-tensor artifact) the field is `None` and
/// `absent_reason` records WHY. At minimum `count` (a real multiplicity from the
/// receipt) and a `source_ref` are present for every row.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValueProof {
    /// sha256 of the operand artifact, computed by OPENING the file (never a
    /// caller-supplied string). `None` for engine count-level rows.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hash: Option<String>,
    /// Tensor shape when the operand is a tensor; `None` for scalar/artifact rows.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shape: Option<Vec<u64>>,
    /// Real multiplicity / element count from the source receipt. Always present.
    pub count: u64,
    /// Physical/byte unit when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    /// Where the proof was read from (receipt / artifact path). Always set when
    /// known; the no-null-padding guardrail records a reason otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_ref: Option<String>,
    /// WHY hash/shape/unit are absent (set IFF `hash` is `None`). No silent nulls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub absent_reason: Option<String>,
}

impl ValueProof {
    /// Count-level proof: real `count` + `source_ref`, hash/shape/unit absent
    /// with an explicit reason. Used for engine per-dispatch COUNT-only rows.
    pub fn count_only(
        count: u64,
        source_ref: Option<String>,
        absent_reason: impl Into<String>,
    ) -> Self {
        ValueProof {
            hash: None,
            shape: None,
            count,
            unit: None,
            source_ref,
            absent_reason: Some(absent_reason.into()),
        }
    }

    /// VALUE-verified proof: a real sha256 hash is present (computed upstream by
    /// opening the file). `absent_reason` stays `None` because the proof is real.
    pub fn verified(hash: String, count: u64, source_ref: Option<String>) -> Self {
        ValueProof {
            hash: Some(hash),
            shape: None,
            count,
            unit: None,
            source_ref,
            absent_reason: None,
        }
    }

    /// True when a real sha256 backs this proof (not count-level).
    pub fn is_value_verified(&self) -> bool {
        self.hash.is_some()
    }
}

/// A single typed operand record — one row of the operand ledger.
///
/// The 22 operand fields the directive enumerates are all present. For
/// COUNT-level records (engine per-operand data absent) the per-operand columns
/// are `None` and `blocker_reason` is set; for prismos-driven records they are
/// VALUE-filled.
///
/// DSTW-T4 Step h3 adds the SEMANTIC OPERATOR CLASS + OPERAND PROOF columns:
/// `operator_class`, `operand_target`, `value_proof`, `parent_state_vertex_id`,
/// and `child_state_vertex_id_or_emitted_tensor`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OperandRecord {
    // ---- identity ----
    pub opcode_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub opcode_id: Option<u16>,
    pub operand_schema_version: u32,
    /// COUNT-level multiplicity (number of dispatches of this opcode). For a
    /// fully typed single-event record this is 1.
    pub count: u64,
    pub runtime_origin: RuntimeOrigin,
    /// Convenience mirror of `runtime_origin.is_prismos_native()` for consumers
    /// that only read the flattened ledger.
    pub runtime_native: bool,

    // ---- SEMANTIC OPERATOR CLASS + OPERAND PROOF (DSTW-T4 Step h3) ----
    /// Which PRISM subsystem this opcode belongs to. Always present.
    pub operator_class: OperatorClass,
    /// What the operand value targets (tensor / state / transition / ...).
    pub operand_target: OperandTarget,
    /// VALUE-verified proof (hash + shape + count + unit + source_ref). Never
    /// null-padded — absence carries an explicit reason.
    pub value_proof: ValueProof,
    /// Upstream state-graph vertex this operand was produced from. `None` when no
    /// bound state vertex applies (set by the emit path via `with_state_lineage`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_state_vertex_id: Option<String>,
    /// Downstream state vertex id OR the emitted-tensor/artifact ref this operand
    /// produced.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub child_state_vertex_id_or_emitted_tensor: Option<String>,

    // ---- input / output operands (VALUE-verified hashes) ----
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_artifact_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_artifact_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_sha256: Option<String>,

    // ---- tensor semantics ----
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_shape: Option<Vec<u64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub units: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coordinate_frame: Option<String>,

    // ---- run context ----
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_id: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ligand_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detection_mode: Option<String>,

    // ---- QNANL / PrismCode provenance ----
    #[serde(skip_serializing_if = "Option::is_none")]
    pub qnanl_packet_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub qnanl_packet_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pcode_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_trace_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pcout_path: Option<String>,

    // ---- tool-coverage status ----
    pub disassembly_status: String,
    pub validation_status: String,
    pub replay_status: String,

    // ---- blocker (set on COUNT-level / blocked records) ----
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocker_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_receipt: Option<String>,

    // ---- PROCESS-PROVEN provenance (DSTW-T4 Step h1) ----
    /// Backs `runtime_origin` with a captured, value-real provenance block
    /// (binary sha256, run pid, emission time, source stage, upstream parent
    /// hashes). A bare `runtime_origin` label is INVALID without this. Attached by
    /// the prismos `md-run` emit path; `None` only on records built outside a
    /// captured run (the builder always carries WHY via `runtime_origin`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_provenance: Option<crate::runtime_provenance::ArtifactProvenance>,
}

const COUNT_LEVEL_BLOCKER: &str =
    "engine emits opcode-COUNT only (md_native_runtime.rs:72 opcodes: Vec<String>); \
     per-dispatch typed operands are BLOCKED-ENGINE";

/// DSTW-T4 Step h3 directive wording for the engine count-level operand-proof gap.
const COUNT_LEVEL_VALUE_PROOF_ABSENT: &str =
    "engine count-level, per-dispatch operand typing is engine-owner";

impl OperandRecord {
    /// COUNT-level record (the honest default for engine-subprocess opcodes).
    ///
    /// `runtime_origin` MUST be supplied explicitly. Per-operand columns are left
    /// `None` and `blocker_reason` records the engine gap — never null-padded.
    pub fn count_level(
        opcode_name: impl Into<String>,
        opcode_id: Option<u16>,
        count: u64,
        runtime_origin: RuntimeOrigin,
        source_receipt: Option<String>,
    ) -> Self {
        let opcode_name = opcode_name.into();
        let operator_class = OperatorClass::classify(&opcode_name);
        let operand_target = OperandTarget::classify(&opcode_name);
        let value_proof = ValueProof::count_only(
            count,
            source_receipt.clone(),
            COUNT_LEVEL_VALUE_PROOF_ABSENT,
        );
        OperandRecord {
            opcode_name,
            opcode_id,
            operand_schema_version: OPERAND_LEDGER_SCHEMA_VERSION,
            count,
            runtime_origin,
            runtime_native: runtime_origin.is_prismos_native(),
            operator_class,
            operand_target,
            value_proof,
            parent_state_vertex_id: None,
            child_state_vertex_id_or_emitted_tensor: None,
            input_artifact_path: None,
            input_sha256: None,
            output_artifact_path: None,
            output_sha256: None,
            tensor_shape: None,
            dtype: None,
            units: None,
            coordinate_frame: None,
            stream_id: None,
            target_id: None,
            ligand_mode: None,
            detection_mode: None,
            qnanl_packet_path: None,
            qnanl_packet_sha256: None,
            pcode_path: None,
            execution_trace_path: None,
            pcout_path: None,
            disassembly_status: "REGISTRY_COVERED".to_string(),
            validation_status: "REGISTRY_COVERED".to_string(),
            replay_status: "NOT_APPLICABLE".to_string(),
            blocker_reason: Some(COUNT_LEVEL_BLOCKER.to_string()),
            source_receipt,
            runtime_provenance: None,
        }
    }

    /// Fully typed record for an opcode prismos itself drives. Every hash is
    /// VALUE-verified: each `Some(path)` is OPENED and sha256'd here. A path that
    /// does not exist yields an `Err` rather than a fabricated/empty hash.
    #[allow(clippy::too_many_arguments)]
    pub fn typed_from_artifacts(
        opcode_name: impl Into<String>,
        opcode_id: Option<u16>,
        runtime_origin: RuntimeOrigin,
        input_artifact_path: Option<&Path>,
        output_artifact_path: Option<&Path>,
        qnanl_packet_path: Option<&Path>,
        source_receipt: Option<String>,
    ) -> std::io::Result<Self> {
        let input_sha256 = match input_artifact_path {
            Some(path) => Some(sha256_file_hex(path)?),
            None => None,
        };
        let output_sha256 = match output_artifact_path {
            Some(path) => Some(sha256_file_hex(path)?),
            None => None,
        };
        let qnanl_packet_sha256 = match qnanl_packet_path {
            Some(path) => Some(sha256_file_hex(path)?),
            None => None,
        };
        let opcode_name = opcode_name.into();
        let operator_class = OperatorClass::classify(&opcode_name);
        let operand_target = OperandTarget::classify(&opcode_name);
        // VALUE proof: a REAL sha256 (output preferred, else packet, else input).
        // typed_from_artifacts only reaches here after at least one file opened, so
        // a hash is always available — this is never a null/placeholder proof.
        let proof_hash = output_sha256
            .clone()
            .or_else(|| qnanl_packet_sha256.clone())
            .or_else(|| input_sha256.clone());
        let proof_source = output_artifact_path
            .or(qnanl_packet_path)
            .or(input_artifact_path)
            .map(path_string);
        let value_proof = match proof_hash {
            Some(hash) => ValueProof::verified(hash, 1, proof_source.clone()),
            None => ValueProof::count_only(
                1,
                proof_source.clone(),
                "typed record has no openable artifact (no input/output/packet path supplied)",
            ),
        };
        // child = the artifact this opcode emitted (output, else sealed packet).
        let child = output_artifact_path.or(qnanl_packet_path).map(path_string);
        Ok(OperandRecord {
            opcode_name,
            opcode_id,
            operand_schema_version: OPERAND_LEDGER_SCHEMA_VERSION,
            count: 1,
            runtime_origin,
            runtime_native: runtime_origin.is_prismos_native(),
            operator_class,
            operand_target,
            value_proof,
            parent_state_vertex_id: None,
            child_state_vertex_id_or_emitted_tensor: child,
            input_artifact_path: input_artifact_path.map(path_string),
            input_sha256,
            output_artifact_path: output_artifact_path.map(path_string),
            output_sha256,
            tensor_shape: None,
            dtype: None,
            units: None,
            coordinate_frame: None,
            stream_id: None,
            target_id: None,
            ligand_mode: None,
            detection_mode: None,
            qnanl_packet_path: qnanl_packet_path.map(path_string),
            qnanl_packet_sha256,
            pcode_path: None,
            execution_trace_path: None,
            pcout_path: None,
            disassembly_status: "REGISTRY_COVERED".to_string(),
            validation_status: "TYPED_VALUE_VERIFIED".to_string(),
            replay_status: "NOT_APPLICABLE".to_string(),
            blocker_reason: None,
            source_receipt,
            runtime_provenance: None,
        })
    }

    // ---- typed-field setters (builder style; only for prismos-driven records) ----

    pub fn with_tensor(mut self, shape: Vec<u64>, dtype: impl Into<String>) -> Self {
        self.tensor_shape = Some(shape.clone());
        self.dtype = Some(dtype.into());
        // mirror the shape into the value proof (Step h3).
        self.value_proof.shape = Some(shape);
        self.operand_target = OperandTarget::Tensor;
        self
    }

    pub fn with_units(mut self, units: impl Into<String>) -> Self {
        let units = units.into();
        self.units = Some(units.clone());
        self.value_proof.unit = Some(units);
        self
    }

    /// Override the operand target (DSTW-T4 Step h3) — used by the DSTW state-graph
    /// emitter to mark vertex rows as `State` and edge rows as `Transition`.
    pub fn with_operand_target(mut self, target: OperandTarget) -> Self {
        self.operand_target = target;
        self
    }

    /// Set the state-graph lineage refs (DSTW-T4 Step h3). `parent` is the
    /// upstream vertex id; `child` is the downstream vertex id or emitted tensor.
    /// A `None` child preserves the artifact ref the constructor already derived.
    pub fn with_state_lineage(mut self, parent: Option<String>, child: Option<String>) -> Self {
        self.parent_state_vertex_id = parent;
        if let Some(child) = child {
            self.child_state_vertex_id_or_emitted_tensor = Some(child);
        }
        self
    }

    pub fn with_coordinate_frame(mut self, frame: impl Into<String>) -> Self {
        self.coordinate_frame = Some(frame.into());
        self
    }

    pub fn with_run_context(
        mut self,
        stream_id: Option<u32>,
        target_id: Option<String>,
        ligand_mode: Option<String>,
        detection_mode: Option<String>,
    ) -> Self {
        self.stream_id = stream_id;
        self.target_id = target_id;
        self.ligand_mode = ligand_mode;
        self.detection_mode = detection_mode;
        self
    }

    pub fn with_blocker(mut self, reason: impl Into<String>) -> Self {
        self.blocker_reason = Some(reason.into());
        self
    }

    /// Attach PROCESS-PROVEN per-artifact provenance (DSTW-T4 Step h1) so the
    /// record's `runtime_origin` label is backed by the captured run context.
    pub fn with_runtime_provenance(
        mut self,
        provenance: crate::runtime_provenance::ArtifactProvenance,
    ) -> Self {
        self.runtime_provenance = Some(provenance);
        self
    }

    fn tsv_cell(value: &Option<String>) -> String {
        value.clone().unwrap_or_default()
    }
}

/// The full ledger: an ordered list of typed records + the TSV writer.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OperandLedger {
    pub schema_kind: String,
    pub schema_version: u32,
    pub records: Vec<OperandRecord>,
}

/// TSV column order (kept stable; appended-to only).
pub const TSV_COLUMNS: &[&str] = &[
    "opcode_name",
    "opcode_id",
    "operand_schema_version",
    "count",
    "runtime_origin",
    "runtime_native",
    "input_artifact_path",
    "input_sha256",
    "output_artifact_path",
    "output_sha256",
    "tensor_shape",
    "dtype",
    "units",
    "coordinate_frame",
    "stream_id",
    "target_id",
    "ligand_mode",
    "detection_mode",
    "qnanl_packet_path",
    "qnanl_packet_sha256",
    "pcode_path",
    "execution_trace_path",
    "pcout_path",
    "disassembly_status",
    "validation_status",
    "replay_status",
    "blocker_reason",
    "source_receipt",
    // ---- DSTW-T4 Step h3 (appended) ----
    "operator_class",
    "operand_target",
    "value_proof_hash",
    "value_proof_shape",
    "value_proof_count",
    "value_proof_unit",
    "value_proof_source_ref",
    "value_proof_absent_reason",
    "parent_state_vertex_id",
    "child_state_vertex_id_or_emitted_tensor",
];

impl OperandLedger {
    pub fn new() -> Self {
        OperandLedger {
            schema_kind: OPERAND_LEDGER_SCHEMA_KIND.to_string(),
            schema_version: OPERAND_LEDGER_SCHEMA_VERSION,
            records: Vec::new(),
        }
    }

    pub fn push(&mut self, record: OperandRecord) {
        self.records.push(record);
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Render the ledger as a TSV body (header comment + column header + rows).
    pub fn to_tsv(&self) -> String {
        let mut lines: Vec<String> = Vec::new();
        lines.push(format!(
            "# {} v{} — typed operand ledger (COUNT-level rows: per-operand BLOCKED-ENGINE; \
             typed rows: VALUE-verified sha256). runtime_origin is mandatory.",
            OPERAND_LEDGER_SCHEMA_KIND, OPERAND_LEDGER_SCHEMA_VERSION
        ));
        lines.push(TSV_COLUMNS.join("\t"));
        for r in &self.records {
            let shape = r
                .tensor_shape
                .as_ref()
                .map(|s| {
                    s.iter()
                        .map(|v| v.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                })
                .unwrap_or_default();
            let cells = vec![
                r.opcode_name.clone(),
                r.opcode_id
                    .map(|id| format!("0x{id:04X}"))
                    .unwrap_or_default(),
                r.operand_schema_version.to_string(),
                r.count.to_string(),
                r.runtime_origin.as_str().to_string(),
                r.runtime_native.to_string(),
                OperandRecord::tsv_cell(&r.input_artifact_path),
                OperandRecord::tsv_cell(&r.input_sha256),
                OperandRecord::tsv_cell(&r.output_artifact_path),
                OperandRecord::tsv_cell(&r.output_sha256),
                shape,
                OperandRecord::tsv_cell(&r.dtype),
                OperandRecord::tsv_cell(&r.units),
                OperandRecord::tsv_cell(&r.coordinate_frame),
                r.stream_id.map(|s| s.to_string()).unwrap_or_default(),
                OperandRecord::tsv_cell(&r.target_id),
                OperandRecord::tsv_cell(&r.ligand_mode),
                OperandRecord::tsv_cell(&r.detection_mode),
                OperandRecord::tsv_cell(&r.qnanl_packet_path),
                OperandRecord::tsv_cell(&r.qnanl_packet_sha256),
                OperandRecord::tsv_cell(&r.pcode_path),
                OperandRecord::tsv_cell(&r.execution_trace_path),
                OperandRecord::tsv_cell(&r.pcout_path),
                r.disassembly_status.clone(),
                r.validation_status.clone(),
                r.replay_status.clone(),
                OperandRecord::tsv_cell(&r.blocker_reason),
                OperandRecord::tsv_cell(&r.source_receipt),
                // ---- DSTW-T4 Step h3 (appended) ----
                r.operator_class.as_str().to_string(),
                r.operand_target.as_str().to_string(),
                OperandRecord::tsv_cell(&r.value_proof.hash),
                r.value_proof
                    .shape
                    .as_ref()
                    .map(|s| {
                        s.iter()
                            .map(|v| v.to_string())
                            .collect::<Vec<_>>()
                            .join(",")
                    })
                    .unwrap_or_default(),
                r.value_proof.count.to_string(),
                OperandRecord::tsv_cell(&r.value_proof.unit),
                OperandRecord::tsv_cell(&r.value_proof.source_ref),
                OperandRecord::tsv_cell(&r.value_proof.absent_reason),
                OperandRecord::tsv_cell(&r.parent_state_vertex_id),
                OperandRecord::tsv_cell(&r.child_state_vertex_id_or_emitted_tensor),
            ];
            lines.push(cells.join("\t"));
        }
        let mut body = lines.join("\n");
        body.push('\n');
        body
    }

    /// Write the ledger TSV to `path`, creating parent dirs.
    pub fn write_tsv(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, self.to_tsv())
    }

    /// Render `OPCODE_OPERAND_COVERAGE_LEDGER.tsv` (DSTW-T4 Step h3): one row per
    /// `(operator_class, operand_target)` bucket aggregating coverage —
    /// total rows, VALUE-verified rows (real sha256 proof), count-level rows
    /// (engine per-dispatch operand typing is engine-owner), native vs blocked
    /// rows. Derived purely FROM the ledger records (no recomputation/guessing).
    pub fn to_coverage_tsv(&self) -> String {
        use std::collections::BTreeMap;
        // key: (operator_class, operand_target) -> [total, value_verified,
        //       count_level, native, blocked]
        let mut buckets: BTreeMap<(&'static str, &'static str), [u64; 5]> = BTreeMap::new();
        for r in &self.records {
            let key = (r.operator_class.as_str(), r.operand_target.as_str());
            let e = buckets.entry(key).or_insert([0; 5]);
            e[0] += 1;
            if r.value_proof.is_value_verified() {
                e[1] += 1;
            } else {
                e[2] += 1;
            }
            if r.runtime_native {
                e[3] += 1;
            }
            if r.runtime_origin == RuntimeOrigin::Blocked || r.blocker_reason.is_some() {
                e[4] += 1;
            }
        }
        let mut lines: Vec<String> = Vec::new();
        lines.push(format!(
            "# {} v{} — OPCODE_OPERAND_COVERAGE_LEDGER (DSTW-T4 Step h3). Coverage \
             aggregated per (operator_class, operand_target). value_verified = real \
             sha256 operand proof; count_level = engine per-dispatch operand typing \
             is engine-owner (BLOCKED-ENGINE), NOT null-padded.",
            OPERAND_LEDGER_SCHEMA_KIND, OPERAND_LEDGER_SCHEMA_VERSION
        ));
        lines.push(
            [
                "operator_class",
                "operand_target",
                "total_rows",
                "value_verified_rows",
                "count_level_rows",
                "native_rows",
                "blocked_rows",
                "value_coverage_pct",
            ]
            .join("\t"),
        );
        for ((class, target), c) in &buckets {
            let pct = if c[0] == 0 {
                0.0
            } else {
                (c[1] as f64) * 100.0 / (c[0] as f64)
            };
            lines.push(
                [
                    class.to_string(),
                    target.to_string(),
                    c[0].to_string(),
                    c[1].to_string(),
                    c[2].to_string(),
                    c[3].to_string(),
                    c[4].to_string(),
                    format!("{pct:.1}"),
                ]
                .join("\t"),
            );
        }
        let mut body = lines.join("\n");
        body.push('\n');
        body
    }

    /// Write the coverage ledger TSV (`OPCODE_OPERAND_COVERAGE_LEDGER.tsv`).
    pub fn write_coverage_tsv(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, self.to_coverage_tsv())
    }
}

fn path_string(path: &Path) -> String {
    path.display().to_string()
}

/// Streaming sha256 of a file, lowercase hex. VALUE proof for operand hashes.
pub fn sha256_file_hex(path: &Path) -> std::io::Result<String> {
    let mut hasher = Sha256::new();
    let mut file = std::fs::File::open(path)?;
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn runtime_origin_native_classification_is_honest() {
        assert!(RuntimeOrigin::RuntimeInline.is_prismos_native());
        assert!(RuntimeOrigin::RuntimeSubprocessOrchestratedByPrismos.is_prismos_native());
        assert!(!RuntimeOrigin::PostRunManual.is_prismos_native());
        assert!(!RuntimeOrigin::PostRunSealOnly.is_prismos_native());
        assert_eq!(RuntimeOrigin::PostRunManual.as_str(), "POST_RUN_MANUAL");
        assert_eq!(
            RuntimeOrigin::RuntimeSubprocessOrchestratedByPrismos.as_str(),
            "RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS"
        );
    }

    #[test]
    fn count_level_record_is_not_null_padded_into_typed() {
        let r = OperandRecord::count_level(
            "MD.SPIKE_EMIT",
            Some(0x0714),
            42,
            RuntimeOrigin::RuntimeSubprocessOrchestratedByPrismos,
            Some("s0_prismcode_md_native_receipt.json".to_string()),
        );
        assert_eq!(r.count, 42);
        assert!(r.runtime_native);
        // per-operand columns MUST be absent, and the gap MUST be named.
        assert!(r.input_sha256.is_none());
        assert!(r.output_sha256.is_none());
        assert!(r.tensor_shape.is_none());
        assert!(r.blocker_reason.is_some());
    }

    #[test]
    fn typed_from_artifacts_value_verifies_sha256_against_real_file() {
        // sha256("") is the canonical empty-string digest; use a known payload.
        let dir = std::env::temp_dir().join(format!(
            "prismcode_operand_ledger_test_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("input.bin");
        let mut f = std::fs::File::create(&input).unwrap();
        f.write_all(b"abc").unwrap();
        drop(f);
        // sha256("abc") known answer.
        const SHA256_ABC: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

        let rec = OperandRecord::typed_from_artifacts(
            "MD.QNANL_SEAL",
            Some(0x071B),
            RuntimeOrigin::RuntimeInline,
            Some(&input),
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(rec.input_sha256.as_deref(), Some(SHA256_ABC));
        assert!(rec.runtime_native);
        assert!(rec.blocker_reason.is_none());
        assert_eq!(rec.validation_status, "TYPED_VALUE_VERIFIED");

        // missing file must ERROR, never fabricate a hash.
        let bad = OperandRecord::typed_from_artifacts(
            "MD.QNANL_SEAL",
            Some(0x071B),
            RuntimeOrigin::RuntimeInline,
            Some(&dir.join("does_not_exist.bin")),
            None,
            None,
            None,
        );
        assert!(bad.is_err());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn tsv_has_stable_columns_and_one_row_per_record() {
        let mut ledger = OperandLedger::new();
        ledger.push(OperandRecord::count_level(
            "MD.FORCE_NONBONDED",
            Some(0x0705),
            7,
            RuntimeOrigin::RuntimeSubprocessOrchestratedByPrismos,
            None,
        ));
        ledger.push(
            OperandRecord::count_level(
                "STATE.CLASSIFY_BLOCKED",
                Some(0x010B),
                1,
                RuntimeOrigin::RuntimeInline,
                None,
            )
            .with_blocker("NO_GDP_PARAMETERIZED_TOPOLOGY"),
        );
        let tsv = ledger.to_tsv();
        let lines: Vec<&str> = tsv.trim_end().split('\n').collect();
        // comment + header + 2 rows
        assert_eq!(lines.len(), 4);
        let header: Vec<&str> = lines[1].split('\t').collect();
        assert_eq!(header.len(), TSV_COLUMNS.len());
        let row0: Vec<&str> = lines[2].split('\t').collect();
        assert_eq!(row0.len(), TSV_COLUMNS.len());
        assert_eq!(row0[0], "MD.FORCE_NONBONDED");
        assert_eq!(row0[1], "0x0705");
        assert_eq!(row0[4], "RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS");
    }

    #[test]
    fn step_h3_operator_class_and_target_classification_is_grounded() {
        // QNANL substring wins over the MD. prefix (seal is the QNANL subsystem).
        assert_eq!(
            OperatorClass::classify("MD.QNANL_SEAL"),
            OperatorClass::Qnanl
        );
        assert_eq!(
            OperatorClass::classify("MD.FORCE_NONBONDED"),
            OperatorClass::Md
        );
        assert_eq!(
            OperatorClass::classify("KERNEL::integrate"),
            OperatorClass::Md
        );
        assert_eq!(
            OperatorClass::classify("STATE.VERTEX_CREATE"),
            OperatorClass::Dstw
        );
        assert_eq!(
            OperatorClass::classify("PATHB.BIND_SITE"),
            OperatorClass::PathB
        );
        assert_eq!(
            OperandTarget::classify("STATE.VERTEX_CREATE"),
            OperandTarget::State
        );
        assert_eq!(
            OperandTarget::classify("STATE.EMIT_TRANSITION"),
            OperandTarget::Transition
        );
        assert_eq!(
            OperandTarget::classify("MD.QNANL_SEAL"),
            OperandTarget::Artifact
        );
        assert_eq!(
            OperandTarget::classify("MD.FORCE_NONBONDED"),
            OperandTarget::Tensor
        );
    }

    #[test]
    fn step_h3_count_level_value_proof_is_not_null_padded() {
        let r = OperandRecord::count_level(
            "MD.SPIKE_EMIT",
            Some(0x0714),
            42,
            RuntimeOrigin::RuntimeSubprocessOrchestratedByPrismos,
            Some("s0_md_native_receipt.json".to_string()),
        );
        // operator_class is the directive-mandated MD for engine count-level rows.
        assert_eq!(r.operator_class, OperatorClass::Md);
        // value_proof carries the REAL count + source, hash absent WITH a reason.
        assert_eq!(r.value_proof.count, 42);
        assert!(r.value_proof.hash.is_none());
        assert!(!r.value_proof.is_value_verified());
        assert_eq!(
            r.value_proof.absent_reason.as_deref(),
            Some("engine count-level, per-dispatch operand typing is engine-owner")
        );
        assert!(r.value_proof.source_ref.is_some());
    }

    #[test]
    fn step_h3_typed_record_value_proof_is_value_verified() {
        let dir =
            std::env::temp_dir().join(format!("prismcode_operand_h3_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("out.bin");
        let mut f = std::fs::File::create(&out).unwrap();
        f.write_all(b"abc").unwrap();
        drop(f);
        const SHA256_ABC: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

        let rec = OperandRecord::typed_from_artifacts(
            "STATE.VERTEX_CREATE",
            Some(0x0101),
            RuntimeOrigin::RuntimeInline,
            None,
            Some(&out),
            None,
            None,
        )
        .unwrap()
        .with_operand_target(OperandTarget::State)
        .with_state_lineage(Some("v_pre_md".to_string()), None);

        assert!(rec.value_proof.is_value_verified());
        assert_eq!(rec.value_proof.hash.as_deref(), Some(SHA256_ABC));
        assert!(rec.value_proof.absent_reason.is_none());
        assert_eq!(rec.operand_target, OperandTarget::State);
        assert_eq!(rec.parent_state_vertex_id.as_deref(), Some("v_pre_md"));
        // child defaults to the emitted artifact when not overridden.
        assert!(rec
            .child_state_vertex_id_or_emitted_tensor
            .as_deref()
            .unwrap()
            .ends_with("out.bin"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn step_h3_coverage_ledger_aggregates_by_class_and_target() {
        let mut ledger = OperandLedger::new();
        ledger.push(OperandRecord::count_level(
            "MD.FORCE_NONBONDED",
            Some(0x0705),
            7,
            RuntimeOrigin::RuntimeSubprocessOrchestratedByPrismos,
            None,
        ));
        ledger.push(OperandRecord::count_level(
            "MD.SPIKE_EMIT",
            Some(0x0714),
            3,
            RuntimeOrigin::RuntimeSubprocessOrchestratedByPrismos,
            None,
        ));
        let cov = ledger.to_coverage_tsv();
        let lines: Vec<&str> = cov.trim_end().split('\n').collect();
        // comment + header + 1 aggregated bucket (MD/TENSOR).
        assert_eq!(lines.len(), 3);
        let header: Vec<&str> = lines[1].split('\t').collect();
        assert_eq!(header[0], "operator_class");
        assert_eq!(header[2], "total_rows");
        let row: Vec<&str> = lines[2].split('\t').collect();
        assert_eq!(row[0], "MD");
        assert_eq!(row[1], "TENSOR");
        assert_eq!(row[2], "2"); // total_rows
        assert_eq!(row[3], "0"); // value_verified_rows
        assert_eq!(row[4], "2"); // count_level_rows
    }
}
