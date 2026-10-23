//! Engine-surface COMPOSITION and pre-CUDA capability admission.
//!
//! Two facts about the engine used to be entangled in one closed `match`:
//! WHICH capability layers an operator named, and WHETHER the resulting field
//! set is a runnable configuration. Because `--engine-surface` was a closed
//! enumeration with one arm per value, naming two capabilities was impossible —
//! not because the engine could not run them together (the arms' field sets are
//! disjoint apart from `multi_differential`, which every writer of it sets
//! `true`), but purely as an artefact of the parser. This module separates the
//! two facts:
//!
//! * a GRAMMAR — a `+`-separated, order-independent composition of registry
//!   tokens, canonicalised into registry order. A SINGLE token normalises to
//!   ITSELF byte-for-byte, so every historical single-token argv reproduces
//!   exactly and `default_value = "none"` still parses.
//! * an ADMISSION — rules evaluated against the RESOLVED FIELD SET, never
//!   against the surface NAME, so a composed surface is policed by exactly the
//!   rules its expanded fields select. Composition widens the vocabulary; it
//!   can never widen the authority.
//!
//! # Where the numbers come from
//!
//! No admission threshold in this module is a source literal at the comparison
//! site. The stream-topology operands (`stream_floor`, `bridge_group_count`)
//! are resolved by [`crate::physical_governor::resolve_stream_topology`], which
//! layers documented defaults under a versioned policy file, the process
//! environment and the command line, and records a
//! [`crate::physical_governor::ValueOrigin`] per field. Resolving with nothing
//! set reproduces the historical eight-stream / four-group admission decision
//! exactly.
//!
//! # The one rule that is structural, not tunable
//!
//! `multi_stream % bridge_group_count == 0` is NOT a policy preference. The
//! multi-differential emitter computes `epg_val = (n_streams / 4).max(1)` and
//! then indexes per-group state with `group_idx = i / epg_val` against vectors
//! FIXED at width `bridge_group_count`. At `n_streams = 10`, `epg_val = 2` and
//! stream 9 yields `group_idx = 4`, which is out of bounds. Relaxing the old
//! `multi_stream != 8` equality to a bare floor therefore does not widen the
//! engine, it crashes it after the device lease is paid. The congruence is kept
//! as a hard pre-CUDA refusal; what is policy-resolved is the DIVISOR, not
//! whether the division must be exact. Deliberately there is no "observe" level
//! for it: a downgradable structural rule is an out-of-bounds index with extra
//! provenance.
//!
//! # Capacity is a second, later stage
//!
//! Admissibility says a stream count is structurally legal. It does not say the
//! device can carry it. Capacity is measured against real free VRAM after the
//! CUDA context exists ([`admit_stream_capacity`]); the driver-headroom reserve
//! it measures against is itself policy-resolved with an origin rather than
//! clamped by a literal at the comparison site.

use std::fmt;

use crate::physical_governor::{
    resolve_stream_topology, StreamTopologyOverrides, StreamTopologyResolution, ValueOrigin,
    STREAM_TOPOLOGY_RULE_ID,
};

// ═══════════════════════════════════════════════════════════════════════════
// Engine-surface composition grammar
// ═══════════════════════════════════════════════════════════════════════════

/// Separator between capability tokens in a composed `--engine-surface` value.
pub const ENGINE_SURFACE_SEPARATOR: char = '+';

/// The token that selects NO capability layer. May not be composed.
pub const ENGINE_SURFACE_NONE_TOKEN: &str = "none";

/// One capability layer a `--engine-surface` value may name.
///
/// Declaration order IS registry order, and registry order IS the canonical
/// spelling order and the field-application order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SurfaceLayer {
    /// Eight-or-more-stream resident captured-graph twin with GPU-direct
    /// chunk-boundary coupling.
    CapturedGraphTwin,
    /// Bare multi-differential interferometric twin.
    InterferometricTwin,
    /// The full 2x2x2x2 sixteen-cell factorial.
    Twin16,
    /// Ghost-phase-lattice-4d clustering on MAR-v2 telemetry.
    GhostTiles,
    /// Governed G26 gearbox.
    GovernedGearbox,
    /// Path-A bounded MD-only evidence handoff.
    MdEvidence,
    /// TSH stage-II emitter. Nested inside the MD-only evidence block, so it
    /// IMPLIES [`SurfaceLayer::MdEvidence`].
    TshStageIi,
}

/// Registry row for one [`SurfaceLayer`].
#[derive(Debug, Clone, Copy)]
pub struct SurfaceLayerSpec {
    /// The layer this row describes.
    pub layer: SurfaceLayer,
    /// The accepted `--engine-surface` token.
    pub token: &'static str,
    /// One-line operator-facing summary.
    pub summary: &'static str,
    /// Layers this one structurally requires.
    pub implies: &'static [SurfaceLayer],
}

/// Every capability layer, in registry order.
pub const SURFACE_LAYER_REGISTRY: &[SurfaceLayerSpec] = &[
    SurfaceLayerSpec {
        layer: SurfaceLayer::CapturedGraphTwin,
        token: CAPTURED_GRAPH_TWIN_TOKEN,
        summary: "resident per-stream captured CUDA physics graphs plus GPU-direct \
                  chunk-boundary ring coupling",
        implies: &[],
    },
    SurfaceLayerSpec {
        layer: SurfaceLayer::InterferometricTwin,
        token: "interferometric-twin",
        summary: "multi-differential interferometric twin",
        implies: &[],
    },
    SurfaceLayerSpec {
        layer: SurfaceLayer::Twin16,
        token: "twin16",
        summary: "multi-differential at the sixteen-cell factorial stream floor",
        implies: &[],
    },
    SurfaceLayerSpec {
        layer: SurfaceLayer::GhostTiles,
        token: "ghost-tiles",
        summary: "ghost-phase-lattice-4d clustering on MAR-v2 telemetry with \
                  monolithic discovery",
        implies: &[],
    },
    SurfaceLayerSpec {
        layer: SurfaceLayer::GovernedGearbox,
        token: "governed-gearbox",
        summary: "governed G26 gearbox with pre-CUDA policy resolution",
        implies: &[],
    },
    SurfaceLayerSpec {
        layer: SurfaceLayer::MdEvidence,
        token: "md-evidence",
        summary: "Path-A bounded MD-only evidence handoff",
        implies: &[],
    },
    SurfaceLayerSpec {
        layer: SurfaceLayer::TshStageIi,
        token: "tsh-stage-ii",
        summary: "TSH stage-II emitter (implies md-evidence: the emitter is nested \
                  inside the MD-only evidence block)",
        implies: &[SurfaceLayer::MdEvidence],
    },
];

/// The captured-graph twin token, named once so receipts and membership tests
/// cannot drift apart.
pub const CAPTURED_GRAPH_TWIN_TOKEN: &str = "captured-graph-twin";

/// Cell count of the TWIN16 factorial: `2 * 2 * 2 * 2`.
///
/// This is a DESIGN CARDINALITY, not a tunable threshold. It is how many cells
/// a full four-factor two-level factorial has, so the `twin16` layer raises
/// `--multi-stream` to it (never lowers a larger operator-set value). It is
/// deliberately NOT exposed as a policy knob: an operator-settable "TWIN16
/// stream floor" invites values such as 10, which clear a floor but leave a
/// partial bridge group and index four-element per-group state out of bounds.
/// The governed rows that DO decide admissibility are `stream_floor` and
/// `bridge_group_count`, resolved by
/// [`crate::physical_governor::resolve_stream_topology`], and they still police
/// this value like any other.
pub const TWIN16_FACTORIAL_CELL_COUNT: usize = 2 * 2 * 2 * 2;

impl SurfaceLayer {
    /// This layer's registry row.
    #[must_use]
    pub fn spec(self) -> &'static SurfaceLayerSpec {
        &SURFACE_LAYER_REGISTRY[self.registry_index()]
    }

    /// This layer's accepted token.
    #[must_use]
    pub fn token(self) -> &'static str {
        self.spec().token
    }

    /// This layer's operator-facing summary.
    #[must_use]
    pub fn summary(self) -> &'static str {
        self.spec().summary
    }

    /// Layers this one structurally requires.
    #[must_use]
    pub fn implies(self) -> &'static [SurfaceLayer] {
        self.spec().implies
    }

    /// Index of this layer in [`SURFACE_LAYER_REGISTRY`].
    #[must_use]
    pub fn registry_index(self) -> usize {
        match self {
            Self::CapturedGraphTwin => 0,
            Self::InterferometricTwin => 1,
            Self::Twin16 => 2,
            Self::GhostTiles => 3,
            Self::GovernedGearbox => 4,
            Self::MdEvidence => 5,
            Self::TshStageIi => 6,
        }
    }

    /// Resolve a token to its layer.
    #[must_use]
    pub fn from_token(token: &str) -> Option<Self> {
        SURFACE_LAYER_REGISTRY
            .iter()
            .find(|spec| spec.token == token)
            .map(|spec| spec.layer)
    }

    /// Every accepted layer token, in registry order.
    #[must_use]
    pub fn all_tokens() -> Vec<&'static str> {
        SURFACE_LAYER_REGISTRY.iter().map(|s| s.token).collect()
    }
}

impl fmt::Display for SurfaceLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.token())
    }
}

/// The full accepted vocabulary, quoted, for help text and rejection messages.
///
/// Replacing clap's closed `possible_values` list costs its generated
/// `[possible values: ...]` line, so the accepted set is re-supplied here and
/// quoted in full inside every rejection this module returns.
#[must_use]
pub fn engine_surface_accepted_summary() -> String {
    let mut tokens = vec![ENGINE_SURFACE_NONE_TOKEN.to_string()];
    tokens.extend(SurfaceLayer::all_tokens().into_iter().map(str::to_string));
    format!(
        "accepted tokens: {} (compose with '{}', order-independent; '{}' selects no layer and may not be composed)",
        tokens.join(", "),
        ENGINE_SURFACE_SEPARATOR,
        ENGINE_SURFACE_NONE_TOKEN
    )
}

/// A parsed, canonicalised `--engine-surface` value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineSurfaceComposition {
    canonical: String,
    requested: Vec<SurfaceLayer>,
}

impl EngineSurfaceComposition {
    /// The canonical spelling: requested tokens in registry order joined by
    /// [`ENGINE_SURFACE_SEPARATOR`], or [`ENGINE_SURFACE_NONE_TOKEN`].
    ///
    /// A single-token input canonicalises to ITSELF byte-for-byte.
    #[must_use]
    pub fn canonical(&self) -> &str {
        &self.canonical
    }

    /// The layers the operator named, in registry order, without implications.
    #[must_use]
    pub fn requested_layers(&self) -> &[SurfaceLayer] {
        &self.requested
    }

    /// Layers pulled in by a requested layer's `implies` set and not named
    /// directly.
    #[must_use]
    pub fn implied_layers(&self) -> Vec<SurfaceLayer> {
        let mut out: Vec<SurfaceLayer> = Vec::new();
        for layer in &self.requested {
            for implied in layer.implies() {
                if !self.requested.contains(implied) && !out.contains(implied) {
                    out.push(*implied);
                }
            }
        }
        out.sort_by_key(|l| l.registry_index());
        out
    }

    /// Requested plus implied layers, deduplicated, in registry order. This is
    /// the set the field fold applies.
    #[must_use]
    pub fn effective_layers(&self) -> Vec<SurfaceLayer> {
        let mut out = self.requested.clone();
        for implied in self.implied_layers() {
            if !out.contains(&implied) {
                out.push(implied);
            }
        }
        out.sort_by_key(|l| l.registry_index());
        out
    }

    /// Whether the composition declares `layer`, directly or by implication.
    #[must_use]
    pub fn contains(&self, layer: SurfaceLayer) -> bool {
        self.effective_layers().contains(&layer)
    }

    /// Whether this is the identity composition (`none`).
    #[must_use]
    pub fn selects_no_layer(&self) -> bool {
        self.requested.is_empty()
    }

    /// Receipt fragment naming exactly what was requested and what that pulled
    /// in, so a run's surface can be audited without re-parsing the argv.
    #[must_use]
    pub fn receipt_json(&self) -> serde_json::Value {
        serde_json::json!({
            "canonical": self.canonical,
            "requested_layers": self
                .requested
                .iter()
                .map(|l| l.token())
                .collect::<Vec<_>>(),
            "implied_layers": self
                .implied_layers()
                .iter()
                .map(|l| l.token())
                .collect::<Vec<_>>(),
            "effective_layers": self
                .effective_layers()
                .iter()
                .map(|l| l.token())
                .collect::<Vec<_>>(),
        })
    }
}

impl fmt::Display for EngineSurfaceComposition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.canonical)
    }
}

/// Parse a `+`-separated `--engine-surface` composition.
///
/// # Errors
///
/// A human-readable rejection quoting the full accepted set, for: an empty
/// value, an empty token between separators, an unknown token, a repeated
/// token, or `none` composed with anything else. Every rejection is loud —
/// a silently-ignored capability token is a run that did not do what the
/// operator asked and said nothing.
pub fn parse_engine_surface_composition(raw: &str) -> Result<EngineSurfaceComposition, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(format!(
            "empty engine-surface value; {}",
            engine_surface_accepted_summary()
        ));
    }

    let mut requested: Vec<SurfaceLayer> = Vec::new();
    let mut saw_none = false;
    let mut token_count = 0usize;

    for token in trimmed.split(ENGINE_SURFACE_SEPARATOR) {
        let token = token.trim();
        token_count += 1;
        if token.is_empty() {
            return Err(format!(
                "empty engine-surface token in '{}'; {}",
                trimmed,
                engine_surface_accepted_summary()
            ));
        }
        if token == ENGINE_SURFACE_NONE_TOKEN {
            saw_none = true;
            continue;
        }
        let Some(layer) = SurfaceLayer::from_token(token) else {
            return Err(format!(
                "unknown engine-surface token '{}'; {}",
                token,
                engine_surface_accepted_summary()
            ));
        };
        if requested.contains(&layer) {
            return Err(format!(
                "engine-surface token '{}' repeats in '{}'; {}",
                token,
                trimmed,
                engine_surface_accepted_summary()
            ));
        }
        requested.push(layer);
    }

    if saw_none && token_count > 1 {
        return Err(format!(
            "engine-surface '{}' composes '{}' with another token; {}",
            trimmed,
            ENGINE_SURFACE_NONE_TOKEN,
            engine_surface_accepted_summary()
        ));
    }

    requested.sort_by_key(|l| l.registry_index());
    let canonical = if requested.is_empty() {
        ENGINE_SURFACE_NONE_TOKEN.to_string()
    } else {
        requested
            .iter()
            .map(|l| l.token())
            .collect::<Vec<_>>()
            .join(&ENGINE_SURFACE_SEPARATOR.to_string())
    };

    Ok(EngineSurfaceComposition {
        canonical,
        requested,
    })
}

/// clap `value_parser` body for `--engine-surface`: parse, then hand back the
/// canonical spelling so the rest of the binary reads an already-normalised
/// value.
///
/// # Errors
///
/// As [`parse_engine_surface_composition`].
pub fn parse_engine_surface_argument(raw: &str) -> Result<String, String> {
    parse_engine_surface_composition(raw).map(|c| c.canonical().to_string())
}

/// Whether an already-normalised `--engine-surface` value declares `layer`.
///
/// MEMBERSHIP, NOT IDENTITY. A composed value that still names
/// `captured-graph-twin` owes exactly the evidence a bare one does; a raw
/// `String` equality emits no receipt at all for a composed run, which is a
/// silent evidence hole and strictly worse than a loud failure.
///
/// Token equality, never substring containment, so `captured-graph-twin-legacy`
/// does not match `captured-graph-twin`. Implications are honoured, so a value
/// naming `tsh-stage-ii` also declares `md-evidence`.
#[must_use]
pub fn engine_surface_declares_layer(raw: &str, layer: SurfaceLayer) -> bool {
    match parse_engine_surface_composition(raw) {
        Ok(composition) => composition.contains(layer),
        // An unparseable value cannot have reached a running engine: the clap
        // value parser refuses it at argv time. Fall back to exact token
        // membership rather than claiming a layer that was never resolved.
        Err(_) => raw
            .split(ENGINE_SURFACE_SEPARATOR)
            .map(str::trim)
            .any(|token| token == layer.token()),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Stream-topology admission (single implementation of the rule)
// ═══════════════════════════════════════════════════════════════════════════

/// Resolve the captured-graph stream topology from every governed layer.
///
/// Thin, single-call wrapper over
/// [`crate::physical_governor::resolve_stream_topology`] so that exactly ONE
/// implementation of the `CAPTURED_GRAPH_TWIN_STREAM_TOPOLOGY` rule exists in
/// the tree, and every consumer (pre-CUDA dispatch, post-run acceptance,
/// capacity confirmation) reads the same resolved rows with the same origins.
///
/// # Errors
///
/// The resolution error text, when a policy file or environment override is
/// unusable or the resolved rows cannot describe a legal stream set. Refusing
/// here is refusing at RESOLUTION time, where the operator can still see and
/// repair the cause.
pub fn resolve_captured_graph_topology() -> Result<StreamTopologyResolution, String> {
    resolve_stream_topology(None, None, StreamTopologyOverrides::default())
        .map_err(|e| e.to_string())
}

// ═══════════════════════════════════════════════════════════════════════════
// Engine dispatch admission
// ═══════════════════════════════════════════════════════════════════════════

/// [`EngineDispatchVerdict::reason`]: the resolved field set is runnable.
pub const REASON_ADMITTED: &str = "ADMITTED";

/// [`EngineDispatchVerdict::reason`]: both twin lanes were demanded at once.
pub const REASON_MULTI_DIFFERENTIAL_AND_COUPLED_TWIN_CONFLICT: &str =
    "MULTI_DIFFERENTIAL_AND_COUPLED_TWIN_CONFLICT";

/// [`EngineDispatchVerdict::reason`]: grouped coupled graph capture has no
/// implementation.
pub const REASON_GROUPED_COUPLED_GRAPH_CAPTURE_NOT_IMPLEMENTED: &str =
    "GROUPED_COUPLED_GRAPH_CAPTURE_NOT_IMPLEMENTED";

/// [`EngineDispatchVerdict::reason`]: wall-bound coupled twin has no
/// implementation.
pub const REASON_COUPLED_TWIN_WALL_BOUND_NOT_IMPLEMENTED: &str =
    "COUPLED_TWIN_WALL_BOUND_NOT_IMPLEMENTED";

/// The resolved field set an admission decision is taken against.
///
/// Fields, never the surface NAME: this is what makes composition unable to
/// widen authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineDispatchRequest {
    /// `--multi-differential`.
    pub multi_differential: bool,
    /// `--coupled-twin`.
    pub coupled_twin: bool,
    /// `--graph-coupling`.
    pub graph_coupling: bool,
    /// `--multi-stream`.
    pub multi_stream: usize,
    /// Whether `--path-a-max-wall-seconds` was set.
    pub wall_bound_requested: bool,
}

impl EngineDispatchRequest {
    /// The lane this field set selects. Precedence is unchanged from the
    /// original adjudicator.
    #[must_use]
    pub fn effective_dispatch(&self) -> &'static str {
        if self.multi_differential && self.graph_coupling && !self.coupled_twin {
            "captured_graph_twin_multi_stream"
        } else if self.multi_differential {
            "multi_differential"
        } else if self.coupled_twin {
            "coupled_twin"
        } else if self.multi_stream > 1 {
            "multi_stream"
        } else {
            "single_structure"
        }
    }
}

/// One pre-CUDA dispatch decision with every operand it was taken on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineDispatchVerdict {
    /// `"PASS"` or `"FAIL"`.
    pub status: &'static str,
    /// The selected lane.
    pub effective_dispatch: &'static str,
    /// Named outcome code. For a topology refusal this is
    /// [`crate::physical_governor::STREAM_TOPOLOGY_RULE_ID`], with the specific
    /// sub-reason carried in `topology`.
    pub reason: &'static str,
    /// The stream-topology verdict, when the captured-graph lane was selected
    /// and the field set reached the topology rule.
    pub topology: Option<crate::physical_governor::StreamTopologyVerdict>,
}

impl EngineDispatchVerdict {
    /// The historical `(status, effective_dispatch, reason)` triple, so callers
    /// and pinned tests that only need the verdict keep working unchanged.
    #[must_use]
    pub fn as_tuple(&self) -> (&'static str, &'static str, &'static str) {
        (self.status, self.effective_dispatch, self.reason)
    }

    /// Whether this verdict refuses the run.
    #[must_use]
    pub fn refuses(&self) -> bool {
        self.status != "PASS"
    }

    /// The arithmetic behind a topology refusal, for the failure message.
    #[must_use]
    pub fn detail(&self) -> Option<&str> {
        self.topology.as_ref().map(|v| v.detail.as_str())
    }

    /// Receipt fragment: the rule that decided, and the numbers it decided on.
    #[must_use]
    pub fn receipt_json(&self) -> serde_json::Value {
        match &self.topology {
            Some(v) => serde_json::json!({
                "rule_id": v.rule_id,
                "admitted": v.admitted,
                "reason": v.reason,
                "observed_streams": v.observed_streams,
                "stream_floor": v.stream_floor,
                "bridge_group_count": v.bridge_group_count,
                "streams_per_bridge_group": v.streams_per_bridge_group,
                "structural_congruence_required": true,
                "structural_congruence_downgradable": false,
                "detail": v.detail,
            }),
            None => serde_json::json!({
                "rule_id": serde_json::Value::Null,
                "admitted": !self.refuses(),
                "reason": self.reason,
                "structural_congruence_required": false,
                "structural_congruence_downgradable": false,
            }),
        }
    }
}

/// Adjudicate a resolved field set against the resolved stream topology,
/// BEFORE any CUDA context exists.
///
/// The captured-graph arm no longer tests `multi_stream != 8`. It asks the
/// resolved topology whether the stream count clears the floor AND divides
/// evenly into the bridge groups, so a structurally legal larger stream set
/// (16 for the TWIN16 factorial) is admitted without a source edit, while a set
/// that cannot be split across the bridge groups the emitter builds (10) is
/// still refused before a lease is paid.
#[must_use]
pub fn adjudicate_engine_dispatch(
    request: &EngineDispatchRequest,
    topology: &StreamTopologyResolution,
) -> EngineDispatchVerdict {
    let effective_dispatch = request.effective_dispatch();

    if request.multi_differential && request.coupled_twin {
        return EngineDispatchVerdict {
            status: "FAIL",
            effective_dispatch,
            reason: REASON_MULTI_DIFFERENTIAL_AND_COUPLED_TWIN_CONFLICT,
            topology: None,
        };
    }
    if request.multi_differential && request.graph_coupling {
        let verdict = topology.admits(u32::try_from(request.multi_stream).unwrap_or(u32::MAX));
        if !verdict.admitted {
            return EngineDispatchVerdict {
                status: "FAIL",
                effective_dispatch,
                reason: STREAM_TOPOLOGY_RULE_ID,
                topology: Some(verdict),
            };
        }
        return EngineDispatchVerdict {
            status: "PASS",
            effective_dispatch,
            reason: REASON_ADMITTED,
            topology: Some(verdict),
        };
    }
    if request.coupled_twin && request.graph_coupling && request.multi_stream > 1 {
        return EngineDispatchVerdict {
            status: "FAIL",
            effective_dispatch,
            reason: REASON_GROUPED_COUPLED_GRAPH_CAPTURE_NOT_IMPLEMENTED,
            topology: None,
        };
    }
    if request.coupled_twin && !request.multi_differential && request.wall_bound_requested {
        return EngineDispatchVerdict {
            status: "FAIL",
            effective_dispatch,
            reason: REASON_COUPLED_TWIN_WALL_BOUND_NOT_IMPLEMENTED,
            topology: None,
        };
    }
    EngineDispatchVerdict {
        status: "PASS",
        effective_dispatch,
        reason: REASON_ADMITTED,
        topology: None,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Stage 2 — device capacity admission
// ═══════════════════════════════════════════════════════════════════════════

/// Stable id of the capacity admission rule.
pub const STREAM_CAPACITY_RULE_ID: &str = "STREAM_CAPACITY_POST_CONTEXT";

/// Environment override for the driver-headroom reserve, in bytes.
pub const ENV_DEVICE_HEADROOM_RESERVE_BYTES: &str = "PRISM_DEVICE_HEADROOM_RESERVE_BYTES";

/// Documented default: the reserve is a twentieth of total device memory.
pub const DEFAULT_DEVICE_HEADROOM_DIVISOR: u64 = 20;

/// Documented default: never reserve less than this.
pub const DEFAULT_DEVICE_HEADROOM_MIN_BYTES: u64 = 512 * 1024 * 1024;

/// Documented default: never reserve more than this.
pub const DEFAULT_DEVICE_HEADROOM_MAX_BYTES: u64 = 1536 * 1024 * 1024;

/// The driver-headroom reserve a capacity budget is computed against, with the
/// authority that set it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceHeadroomReserve {
    /// Bytes withheld from the capacity budget for the driver and allocator.
    pub bytes: u64,
    /// Where that number came from.
    pub origin: ValueOrigin,
}

/// Resolve the driver-headroom reserve for a device with `total_bytes` of
/// memory.
///
/// The documented default reproduces the historical rule
/// `(total_bytes / 20).clamp(512 MiB, 1536 MiB)` EXACTLY, and yields 0 when the
/// device total is unknown, so an unchanged operator invocation plans the same
/// grid and admits the same stream count bit-for-bit. What changed is that the
/// number is now published with an origin instead of being a clamp literal at
/// the comparison site.
///
/// # Errors
///
/// The rejection text when [`ENV_DEVICE_HEADROOM_RESERVE_BYTES`] is present but
/// is not a decimal `u64`. A malformed override is an error, never a silent
/// fall-through: an operator must not be told the run was planned against a
/// reserve they did not choose.
pub fn resolve_device_headroom_reserve(total_bytes: u64) -> Result<DeviceHeadroomReserve, String> {
    if let Some(raw) = std::env::var_os(ENV_DEVICE_HEADROOM_RESERVE_BYTES) {
        let text = raw.to_string_lossy().trim().to_string();
        if !text.is_empty() {
            let bytes = text.parse::<u64>().map_err(|e| {
                format!(
                    "{ENV_DEVICE_HEADROOM_RESERVE_BYTES}={text} is unusable: {e}; expected a decimal byte count"
                )
            })?;
            return Ok(DeviceHeadroomReserve {
                bytes,
                origin: ValueOrigin::Environment,
            });
        }
    }
    let bytes = if total_bytes > 0 {
        (total_bytes / DEFAULT_DEVICE_HEADROOM_DIVISOR).clamp(
            DEFAULT_DEVICE_HEADROOM_MIN_BYTES,
            DEFAULT_DEVICE_HEADROOM_MAX_BYTES,
        )
    } else {
        0
    };
    Ok(DeviceHeadroomReserve {
        bytes,
        origin: ValueOrigin::Default,
    })
}

/// One device-capacity decision, taken against MEASURED free VRAM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamCapacityVerdict {
    /// [`STREAM_CAPACITY_RULE_ID`].
    pub rule_id: &'static str,
    /// Whether the admitted stream count fits the measured budget.
    pub admitted: bool,
    /// Stream count being confirmed.
    pub n_streams: usize,
    /// Free device bytes as measured after the context exists.
    pub free_bytes: u64,
    /// Total device bytes.
    pub total_bytes: u64,
    /// Reserve withheld for the driver and allocator.
    pub headroom_reserve_bytes: u64,
    /// Where the reserve came from.
    pub headroom_reserve_origin: &'static str,
    /// Budget the requirement was compared against.
    pub budget_bytes: u64,
    /// Bytes the planner estimates the stream set needs.
    pub required_bytes: u64,
    /// Observed-vs-required arithmetic.
    pub detail: String,
}

impl StreamCapacityVerdict {
    /// Receipt object for the capacity stage.
    #[must_use]
    pub fn receipt_json(&self) -> serde_json::Value {
        serde_json::json!({
            "rule_id": self.rule_id,
            "stage": "stage_2_post_context_measured_capacity",
            "admitted": self.admitted,
            "n_streams": self.n_streams,
            "free_bytes": self.free_bytes,
            "total_bytes": self.total_bytes,
            "headroom_reserve_bytes": self.headroom_reserve_bytes,
            "headroom_reserve_origin": self.headroom_reserve_origin,
            "budget_bytes": self.budget_bytes,
            "required_bytes": self.required_bytes,
            "detail": self.detail,
        })
    }
}

/// Confirm that an ADMISSIBLE stream count is also CARRIABLE by the device.
///
/// Stage 1 ([`adjudicate_engine_dispatch`]) decides structural admissibility
/// from policy before `cuInit`; it cannot see VRAM. Relaxing the old
/// `multi_stream != 8` equality without this second stage would convert a clean
/// pre-CUDA refusal into an out-of-memory failure after the lease is paid,
/// which is why the two must land together.
#[must_use]
pub fn admit_stream_capacity(
    n_streams: usize,
    free_bytes: u64,
    total_bytes: u64,
    budget_bytes: u64,
    required_bytes: u64,
    reserve: &DeviceHeadroomReserve,
) -> StreamCapacityVerdict {
    let admitted = required_bytes <= budget_bytes;
    let detail = if admitted {
        format!(
            "{n_streams} streams require {required_bytes} B within the {budget_bytes} B budget \
             (free {free_bytes} B of {total_bytes} B, driver headroom reserve {} B origin={})",
            reserve.bytes,
            reserve.origin.as_str()
        )
    } else {
        format!(
            "{n_streams} streams require {required_bytes} B, exceeding the {budget_bytes} B budget \
             (free {free_bytes} B of {total_bytes} B, driver headroom reserve {} B origin={})",
            reserve.bytes,
            reserve.origin.as_str()
        )
    };
    StreamCapacityVerdict {
        rule_id: STREAM_CAPACITY_RULE_ID,
        admitted,
        n_streams,
        free_bytes,
        total_bytes,
        headroom_reserve_bytes: reserve.bytes,
        headroom_reserve_origin: reserve.origin.as_str(),
        budget_bytes,
        required_bytes,
        detail,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physical_governor::{resolve_stream_topology_layered, StreamTopologyOverrides};

    fn documented_topology() -> StreamTopologyResolution {
        resolve_stream_topology_layered(
            None,
            None,
            StreamTopologyOverrides::default(),
            StreamTopologyOverrides::default(),
        )
        .expect("documented defaults must resolve")
    }

    #[test]
    fn a_single_token_normalizes_to_itself_byte_for_byte() {
        for token in [
            "none",
            "captured-graph-twin",
            "interferometric-twin",
            "twin16",
            "ghost-tiles",
            "governed-gearbox",
            "md-evidence",
            "tsh-stage-ii",
        ] {
            assert_eq!(
                parse_engine_surface_argument(token).expect("token must parse"),
                token,
                "single token must be its own canonical spelling"
            );
        }
    }

    #[test]
    fn composition_is_order_independent_and_canonicalized() {
        let a = parse_engine_surface_argument("twin16+ghost-tiles").expect("parse");
        let b = parse_engine_surface_argument("ghost-tiles+twin16").expect("parse");
        assert_eq!(a, b);
        assert_eq!(a, "twin16+ghost-tiles");
    }

    #[test]
    fn the_closed_enumeration_no_longer_excludes_capabilities_from_each_other() {
        let c = parse_engine_surface_composition(
            "captured-graph-twin+twin16+ghost-tiles+governed-gearbox+tsh-stage-ii",
        )
        .expect("composition must parse");
        assert!(c.contains(SurfaceLayer::CapturedGraphTwin));
        assert!(c.contains(SurfaceLayer::Twin16));
        assert!(c.contains(SurfaceLayer::GhostTiles));
        assert!(c.contains(SurfaceLayer::GovernedGearbox));
        // tsh-stage-ii implies md-evidence: without it the emitter's enclosing
        // block is never entered and the run produces no acceptance evidence.
        assert!(c.contains(SurfaceLayer::MdEvidence));
        assert_eq!(c.implied_layers(), vec![SurfaceLayer::MdEvidence]);
    }

    #[test]
    fn malformed_compositions_are_refused_loudly_not_ignored() {
        assert!(parse_engine_surface_composition("").is_err());
        assert!(parse_engine_surface_composition("twin16+").is_err());
        assert!(parse_engine_surface_composition("twin16+twin16").is_err());
        assert!(parse_engine_surface_composition("none+twin16").is_err());
        let unknown = parse_engine_surface_composition("captured-graph-twin-legacy")
            .expect_err("unknown token must be refused");
        assert!(unknown.contains("captured-graph-twin"));
    }

    #[test]
    fn layer_membership_is_token_equality_not_substring_containment() {
        assert!(engine_surface_declares_layer(
            "captured-graph-twin",
            SurfaceLayer::CapturedGraphTwin
        ));
        assert!(engine_surface_declares_layer(
            "twin16+captured-graph-twin+ghost-tiles",
            SurfaceLayer::CapturedGraphTwin
        ));
        assert!(!engine_surface_declares_layer(
            "captured-graph-twin-legacy",
            SurfaceLayer::CapturedGraphTwin
        ));
        assert!(!engine_surface_declares_layer(
            "none",
            SurfaceLayer::CapturedGraphTwin
        ));
        assert!(engine_surface_declares_layer(
            "tsh-stage-ii",
            SurfaceLayer::MdEvidence
        ));
    }

    #[test]
    fn the_documented_defaults_reproduce_todays_admitted_run_exactly() {
        let topology = documented_topology();
        assert_eq!(topology.policy.stream_floor, 8);
        assert_eq!(topology.policy.bridge_group_count, 4);
        assert_eq!(
            topology.origin_of("stream_floor"),
            Some(ValueOrigin::Default)
        );
        assert_eq!(
            topology.origin_of("bridge_group_count"),
            Some(ValueOrigin::Default)
        );
        let request = EngineDispatchRequest {
            multi_differential: true,
            coupled_twin: false,
            graph_coupling: true,
            multi_stream: 8,
            wall_bound_requested: true,
        };
        assert_eq!(
            adjudicate_engine_dispatch(&request, &topology).as_tuple(),
            ("PASS", "captured_graph_twin_multi_stream", REASON_ADMITTED)
        );
    }

    #[test]
    fn sixteen_streams_are_admitted_and_ten_are_still_refused() {
        let topology = documented_topology();
        let at = |n: usize| {
            adjudicate_engine_dispatch(
                &EngineDispatchRequest {
                    multi_differential: true,
                    coupled_twin: false,
                    graph_coupling: true,
                    multi_stream: n,
                    wall_bound_requested: true,
                },
                &topology,
            )
        };
        // The capability this work exists to unblock.
        assert_eq!(at(16).as_tuple().0, "PASS");
        assert_eq!(at(32).as_tuple().0, "PASS");
        assert_eq!(at(12).as_tuple().0, "PASS");
        // Below the floor.
        assert_eq!(
            at(4).as_tuple(),
            (
                "FAIL",
                "captured_graph_twin_multi_stream",
                STREAM_TOPOLOGY_RULE_ID
            )
        );
        // THE STRUCTURAL SAFETY CASE: 10 >= 8 clears a bare floor, but
        // epg_val = 2 and stream 9 yields group_idx 4 against four-element
        // per-group state. A bare floor would admit this and crash after the
        // lease is paid.
        assert_eq!(
            at(10).as_tuple(),
            (
                "FAIL",
                "captured_graph_twin_multi_stream",
                STREAM_TOPOLOGY_RULE_ID
            )
        );
        assert_eq!(at(7).as_tuple().0, "FAIL");
    }

    #[test]
    fn the_non_topology_refusals_are_unchanged() {
        let topology = documented_topology();
        let at = |md: bool, ct: bool, gc: bool, n: usize, wall: bool| {
            adjudicate_engine_dispatch(
                &EngineDispatchRequest {
                    multi_differential: md,
                    coupled_twin: ct,
                    graph_coupling: gc,
                    multi_stream: n,
                    wall_bound_requested: wall,
                },
                &topology,
            )
            .as_tuple()
        };
        assert_eq!(
            at(true, true, true, 8, true),
            (
                "FAIL",
                "multi_differential",
                REASON_MULTI_DIFFERENTIAL_AND_COUPLED_TWIN_CONFLICT
            )
        );
        assert_eq!(
            at(false, true, true, 8, false),
            (
                "FAIL",
                "coupled_twin",
                REASON_GROUPED_COUPLED_GRAPH_CAPTURE_NOT_IMPLEMENTED
            )
        );
        assert_eq!(
            at(false, true, false, 1, true),
            (
                "FAIL",
                "coupled_twin",
                REASON_COUPLED_TWIN_WALL_BOUND_NOT_IMPLEMENTED
            )
        );
        assert_eq!(
            at(true, false, false, 8, true),
            ("PASS", "multi_differential", REASON_ADMITTED)
        );
    }

    #[test]
    fn the_headroom_reserve_default_reproduces_the_historical_clamp() {
        // The exact arithmetic the planner used as a source literal.
        for total in [
            0u64,
            1,
            8 * 1024 * 1024 * 1024,
            32 * 1024 * 1024 * 1024,
            u64::MAX,
        ] {
            let expected = if total > 0 {
                (total / 20).clamp(512 * 1024 * 1024, 1536 * 1024 * 1024)
            } else {
                0
            };
            let resolved =
                resolve_device_headroom_reserve(total).expect("default resolution cannot fail");
            assert_eq!(resolved.bytes, expected, "total={total}");
            assert_eq!(resolved.origin, ValueOrigin::Default);
        }
    }

    #[test]
    fn capacity_is_a_second_stage_that_measures_rather_than_asserts() {
        let reserve = DeviceHeadroomReserve {
            bytes: 1024,
            origin: ValueOrigin::Default,
        };
        let fits = admit_stream_capacity(16, 4_000, 8_000, 3_000, 2_500, &reserve);
        assert!(fits.admitted);
        assert_eq!(fits.rule_id, STREAM_CAPACITY_RULE_ID);
        let does_not = admit_stream_capacity(16, 4_000, 8_000, 3_000, 3_001, &reserve);
        assert!(!does_not.admitted);
        assert!(does_not.detail.contains("exceeding"));
        assert_eq!(
            does_not.receipt_json()["stage"],
            "stage_2_post_context_measured_capacity"
        );
    }
}
