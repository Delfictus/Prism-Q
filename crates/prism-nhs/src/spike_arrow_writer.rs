//! Per-spike Apache Arrow IPC writer (Stage 1B-1).
//!
//! Replaces the per-site `*.spike_events.json` dump (15 GB across 5 sites,
//! partial schema, JSON serialization on the critical path) with a single
//! per-target `*.spike_events.arrow` file (~3-4 GB total uncompressed,
//! columnar, 30 columns of full per-spike tagging including the stratified
//! background spike bucket).
//!
//! ## Why Apache Arrow IPC and not Parquet?
//!
//! Parquet is a *rest* format, not a *streaming* format. The Apache Arrow
//! IPC streaming format has the property that **the in-memory layout is the
//! wire format** — there is no serialize step, so the cost of "writing"
//! a batch is just the cost of constructing the columnar arrays and a
//! single `cuMemcpyDtoHAsync` (when used in a streaming pipeline). For the
//! initial Stage 1B-1 implementation we write at end-of-pipeline rather
//! than streaming per-chunk, but the format choice keeps the door open
//! for that upgrade in Stage 1B-2 / Stage 4.
//!
//! Arrow IPC is also the *lingua franca* for downstream analytics: Polars,
//! DuckDB, cuDF, Pandas, R, Julia, MATLAB can all read it without
//! conversion. Compared to JSON, an Arrow IPC file is:
//!
//!   • ~5× smaller on disk (binary, columnar, no key repetition)
//!   • ~100× faster to read (mmap, no parser)
//!   • Strongly typed (no string ↔ int conversion bugs)
//!   • Schema-evolving (downstream can add columns without breaking readers)
//!
//! Reference: https://arrow.apache.org/docs/format/Columnar.html#serialization-and-interprocess-communication-ipc
//!
//! ## What's in the schema (30 columns)
//!
//! Each row is one spike, with three classes of fields. Counts: 6
//! provenance + 18 physical state + 6 classification = 30 columns total.
//!
//! ### Provenance (6 columns)
//!   - `spike_id`         u64  — dense row ordinal **within this file only**.
//!                                NOT globally unique across a multi-seed
//!                                merge; join on `(replica_seed, spike_id)`.
//!   - `replica_seed`     u64  — `--replica-seed` value (reproducibility)
//!   - `stream_id`        u8   — stream index 0..n_streams-1
//!   - `group_id`         u8   — TWIN group: 0=TS, 1=EQ, 2=UV, 3=HY.
//!                                DERIVED from `stream_id`; forced to 0 for
//!                                non-`--multi-differential` runs, which is
//!                                indistinguishable from a measured TS
//!                                assignment. See `group_id_for_stream`.
//!   - `chunk_idx`        u16  — DERIVED, `min(timestep / chunk_size,
//!                                max_chunks-1)` with a caller-supplied
//!                                `chunk_size`. This is a re-derivation from
//!                                the protocol-local clock, NOT a record of
//!                                the chunk that emitted the spike.
//!   - `voxel_idx`        i32  — unique voxel ID in the engine grid
//!                                (Opp #4 — was previously dropped on dump)
//!
//! ### Physical state (18 columns from GpuSpikeEvent)
//!   - `timestep`         i32  — PROTOCOL-LOCAL step counter. **Not** the
//!                                global MD step and **not** a frame index.
//!                                See "Clock domain" below before joining.
//!   - `x`, `y`, `z`      f32 × 3 — voxel center position
//!   - `intensity`        f32  — spike intensity
//!   - `spike_source`     i32  — raw numeric source (Opp #1, was string).
//!                                Encodes the physics channel, NOT the
//!                                producing kernel.
//!   - `mechanism_tag`    utf8 — DERIVED from `spike_source` + `aromatic_type`;
//!                                stable mechanism class for downstream ML;
//!                                UNK + LIF maps to LIF_THERMAL_SHAPE;
//!                                source6 maps to VORONOI_LIF independently.
//!   - `aromatic_type`    i32  — raw numeric type (Opp #3, was string)
//!   - `aromatic_residue_id` i32
//!   - `phase_bits`       u32  — 10-bit CCNS phase (Opp #2, was dropped).
//!                                Re-encoded from the protocol step counter,
//!                                so it is a lossy view of the same clock as
//!                                `timestep`, not an independent witness.
//!   - `n_residues`       u8   — count for nearby_residues (saturated into u8)
//!   - `nearby_residues`  FixedSizeList<i32, 8> — Opp #2, was dropped.
//!                                Unused slots padded with the -1 sentinel.
//!   - `n_nearby_excited` u8   — saturated into u8 by this writer
//!   - `vibrational_energy` f32
//!   - `water_density`    f32
//!   - `wd_change`        f32  — Opp #5 (was computed but never exported)
//!   - `wavelength_nm`    f32
//!   - `ccns_phase`       u8   — DERIVED from `timestep` + the protocol
//!                                cold_hold/ramp/warm_hold/ramp_down step
//!                                counts. Schema v5 domain: 0=cold_hold,
//!                                1=ramp, 2=warm_hold, 3=ramp_down,
//!                                4=cold_return. Schemas v1..v4 emitted only
//!                                0..2 and ALIASED ramp_down + cold_return
//!                                onto 2, collapsing 10/30 of the default
//!                                5:10:5:6:4 hysteresis protocol — the
//!                                release-and-recovery half of the kinetic
//!                                signature — into the warm steady state.
//!                                The coding actually in force is stamped per
//!                                file under
//!                                `prism.spike_arrow.ccns_phase_coding`
//!                                ([`SCHEMA_META_CCNS_PHASE_CODING`]); read it
//!                                before assuming a domain. A protocol with
//!                                `ramp_down_steps <= 0` declares no return
//!                                stages and still emits only 0..2.
//!
//! ### Classification (6 columns, host-side derived)
//!   - `site_id`          i32  — consensus site cluster_id, -1 for background
//!   - `nearest_site_id`  i32  — closest consensus site (== site_id when assigned)
//!   - `nearest_site_dist` f32 — distance in Å
//!   - `background_class` u8   — 0=primary_site, 1=bulk_thermal,
//!                                2=surface_noise, 3=near_miss,
//!                                4=relabel_candidate
//!   - `burial_score`     f32  — DERIVED neighbour-count proxy in [0,1]
//!                                (`n_residues / 8`, capped by the 8-slot
//!                                warp matrix). NOT an atom-density-within-a-
//!                                sphere measurement.
//!   - `intensity_percentile` u8 — rank within (channel) for this run [0,100]
//!
//! ## Clock domain — `timestep` is NOT a frame index (schema v2, v4)
//!
//! `timestep` is a **protocol-local** counter. It must not be joined to
//! coordinate/trajectory frames, and no divisor of it may be presented as a
//! frame index. Three distinct clocks run in this engine and this schema
//! carries no witness identifying which one produced a given row:
//!
//!   * **Clock A (spikes)** — `ProtocolState.current_step`, advanced by the
//!     Protocol Director. The Director sits inside the captured graph body,
//!     so it advances `--captured-graph-body-unroll` times per graph replay.
//!   * **Clock B (frames)** — the device frame counter, advanced by exactly
//!     `1` per graph replay. With body-unroll 8, clock A ticks 8× per B tick.
//!   * **Clock C (LADD)** — the host `FusedEngine::timestep`, which is never
//!     advanced under the captured-graph path because `step()`/`step_batch()`
//!     are bypassed. LADD-channel rows therefore carry a **frozen** clock in
//!     the same file as running-clock rows, and `spike_source` cannot be used
//!     to demultiplex this because it names the physics channel rather than
//!     the producing kernel.
//!
//! Consequences for consumers:
//!
//!   * Do not divide `timestep` by anything to obtain a frame index.
//!   * Do not join spike rows to trajectory frames, coordinate dumps, or any
//!     other artifact keyed on the global MD step.
//!   * Do not compute a duty cycle / open-frequency from `timestep` buckets;
//!     the denominator is a mixture of three clocks.
//!   * A real join goes through the per-chunk governor sidecar
//!     (`coupled_chronology`, the `.ccr` stream), which stamps
//!     `global_md_step` / `cumulative_simulated_time_ps` per DECISION and is
//!     keyed by `(stream_id, chunk)`. Join on `chunk_idx`, never on
//!     `timestep`. Schema v3 briefly tried to carry the witness in the spike
//!     record itself; that widened the record past the sealed QNANL packet
//!     stride and, in exchange, resolved only to graph-REPLAY granularity —
//!     coarser than the sidecar. It is withdrawn and is not coming back. The
//!     absence is recorded here rather than papered over with a placeholder.
//!
//! ## Schema versioning
//!
//! [`SPIKE_ARROW_SCHEMA_VERSION`] is stamped into the Arrow **schema-level
//! metadata** under [`SCHEMA_META_VERSION`], alongside the clock-domain note
//! and the changelog. Because Arrow schema equality includes both the field
//! list and the metadata map, a v1 file and a v2 file cannot be silently
//! concatenated — [`crate::multi_seed_merger`] rejects the mix with
//! `SchemaMismatch`, and every known name-based reader fails loudly on the
//! removed column rather than misparsing a shifted position.
//!
//! ### v1 → v2 (column count 31 → 30)
//!
//! **REMOVED `frame_index` (UInt16).** It was synthesised as
//! `clamp(timestep / 1000, 0, u16::MAX)`: an undeclared magic divisor applied
//! to a protocol-local clock, exported under a name that asserts precisely the
//! measurement that was never taken. It was the most likely column for an
//! external analyst to trust, and a post-hoc join encoded as data is worse
//! than an absent field because it silently rescales downstream analysis.
//!
//! Removal is lossless: the retained `timestep` column reproduces the old
//! value exactly (`clamp(timestep / 1000, 0, 65535)`) for anyone who needs
//! bit-compatibility with a v1 file — but see "Clock domain" above, because
//! that quantity is **not** a frame index and must not be used as one.
//!
//! ## Background spike stratification (no magic constants)
//!
//! Background spikes (those that fell outside any consensus site's radius)
//! are NOT discarded. They are preserved with `site_id = -1` and
//! sub-classified by `background_class` so the training pipeline can sample
//! them deliberately:
//!
//!   0 = primary_site         (site_id != -1, in a consensus site)
//!   1 = bulk_thermal         (deep solvent / surface, low burial, far from sites)
//!   2 = surface_noise        (surface, moderate burial, mid-distance from sites)
//!   3 = near_miss            (just outside a site's radius — closest 10% of bg)
//!   4 = relabel_candidate    (high burial AND high intensity AND close to a site)
//!
//! The thresholds for these classes are **per-run percentiles** computed
//! from the actual distribution of background spikes — no fixed numeric
//! cutoffs. See `classify_background_class` below.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, FixedSizeListBuilder, Float32Builder, Int32Builder, RecordBatch, StringBuilder,
    UInt16Builder, UInt32Builder, UInt64Builder, UInt8Builder,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::{FileWriter, StreamWriter};

use crate::fused_engine::GpuSpikeEvent;

/// Number of nearby residues stored per spike (matches GpuSpikeEvent).
pub const NEARBY_RESIDUES_LEN: i32 = 8;

/// Number of columns in the current canonical spike schema.
pub const SPIKE_ARROW_COLUMN_COUNT: usize = 30;

/// Version of the per-spike Arrow schema emitted by [`build_spike_schema`].
///
/// * v1 — 31 columns, included the synthesised `frame_index` column.
/// * v2 — 30 columns, `frame_index` removed (see the module doc changelog),
///   schema-level metadata added (version, clock domain, derived-column
///   inventory, changelog).
/// * v3 — 36 columns; the spike record had been widened to 128 B to carry
///   `global_md_step`, `cumulative_simulated_time_ps`, `record_stream_id`,
///   `record_chunk_id`, `clock_domain` and `spike_record_schema_version`.
/// * v4 — 30 columns; those six are REMOVED again. Widening the record broke
///   the sealed QNANL force-kernel packet, and the in-record global step only
///   ever resolved to graph-REPLAY granularity. The frame bridge is the
///   per-step governor sidecar (`coupled_chronology`, the `.ccr` stream),
///   which is strictly finer.
///
/// * v5 — 30 columns; the field list is byte-identical to v4 and the 96 B
///   device record is untouched (so the QNANL force-kernel seal is untouched).
///   What changes is the declared VALUE DOMAIN of the existing `ccns_phase`
///   column: `ramp_down` (3) and `cold_return` (4) stop being aliased onto
///   `warm_hold` (2). The coding in force for a given file is selected by
///   policy ([`CcnsPhaseCoding`], resolved from [`CCNS_PHASE_CODING_ENV`]) and
///   stamped under [`SCHEMA_META_CCNS_PHASE_CODING`]. A reader that assumed
///   `ccns_phase <= 2` MUST consult that key rather than the column type.
///
/// Bump this whenever a field is added, removed, retyped, or reordered, or
/// whenever the declared value domain of an existing column changes.
pub const SPIKE_ARROW_SCHEMA_VERSION: u32 = 5;

/// Metadata key carrying [`SPIKE_ARROW_SCHEMA_VERSION`] as a decimal string.
pub const SCHEMA_META_VERSION: &str = "prism.spike_arrow.schema_version";
/// Metadata key carrying the clock-domain warning ([`CLOCK_DOMAIN_NOTE`]).
pub const SCHEMA_META_CLOCK_DOMAIN: &str = "prism.spike_arrow.clock_domain";
/// Metadata key carrying the derived-column inventory ([`DERIVED_COLUMNS_NOTE`]).
pub const SCHEMA_META_DERIVED_COLUMNS: &str = "prism.spike_arrow.derived_columns";
/// Metadata key carrying the schema changelog ([`SCHEMA_CHANGELOG`]).
pub const SCHEMA_META_CHANGELOG: &str = "prism.spike_arrow.changelog";
/// Metadata key carrying the resolved `ccns_phase` coding descriptor
/// ([`CcnsPhaseCoding::descriptor`]).
///
/// The `ccns_phase` column is a `UInt8` under every schema version, so its
/// Arrow type cannot express which coding produced it. This key does, and it
/// travels inside the file: a consumer reads the descriptor rather than
/// inferring a domain from the schema version alone.
pub const SCHEMA_META_CCNS_PHASE_CODING: &str = "prism.spike_arrow.ccns_phase_coding";

/// Schema-level warning that the `timestep` column is a protocol-local clock.
///
/// Stamped into every emitted file so the warning travels with the data
/// instead of living only in this source tree. See the module-level "Clock
/// domain" section for the full derivation.
pub const CLOCK_DOMAIN_NOTE: &str = "\
`timestep` is a PROTOCOL-LOCAL step counter. It is NOT the global MD step and NOT a frame index. \
It MUST NOT be joined to coordinate frames, trajectory frames, or any artifact keyed on the global \
MD clock, and no divisor of it may be reported as a frame index. Three clocks run concurrently and \
this schema carries no witness identifying which one produced a given row: (A) the protocol step \
counter advanced by the Protocol Director, which sits inside the captured graph body and therefore \
advances once per unrolled body iteration (captured-graph-body-unroll times per replay); (B) the \
device frame counter, advanced exactly once per graph replay; (C) the host engine timestep, which \
is never advanced under the captured-graph path. LADD-channel rows carry the frozen clock C in the \
same file as rows carrying a running clock, and `spike_source` cannot demultiplex this because it \
names the physics channel rather than the producing kernel. `phase_bits` is re-encoded from the \
same protocol step counter as `timestep`, so it is a lossy view of clock A rather than an \
independent temporal witness. SCHEMA v4: the per-spike record carries NO global-MD-step witness \
and is not going to -- widening the 96 B record invalidates the sealed QNANL force-kernel packet, \
and the field only ever resolved to graph-REPLAY granularity. Join to the coordinate-frame axis \
through the per-step governor sidecar (`coupled_chronology`, the `.ccr` stream), which records \
global_md_step and cumulative_simulated_time_ps per DECISION and is keyed by chunk. Elapsed \
physics MUST come from that sidecar once the gearbox varies dt between 0.5, 2.0 and 4.0 fs, \
because a step index stops being proportional to simulated time.";

/// Schema-level inventory of columns that are host-derived rather than
/// measured, so no consumer mistakes a re-derivation for an observation.
pub const DERIVED_COLUMNS_NOTE: &str = "\
Host-derived (NOT device measurements): \
`chunk_idx` = min(timestep / chunk_size, max_chunks-1) with a caller-supplied chunk_size; a \
re-derivation from the protocol-local clock, not a record of the chunk that emitted the spike. \
`ccns_phase` = re-derivation from `timestep` plus the protocol \
cold_hold/ramp/warm_hold/ramp_down step counts. Schema v5 emits 0=cold_hold / 1=ramp / \
2=warm_hold / 3=ramp_down / 4=cold_return; the coding actually in force is stamped under \
prism.spike_arrow.ccns_phase_coding and MUST be read before assuming a domain. Schemas v1..v4 \
emitted only 0..2 and aliased ramp_down + cold_return onto 2. Code 4 also absorbs post-protocol \
overrun, matching the engine-side convention. A protocol declaring ramp_down_steps<=0 has no \
return stages and emits only 0..2, byte-identical to the legacy coding. Under \
--multi-differential the stage boundaries used are the RUN-LEVEL protocol's, not the per-stream \
condition cell's -- a property codes 0..2 already had before v5, not one introduced by the \
widening. \
`group_id` = stream_id / engines_per_group, forced to 0 when the run is not multi-differential, \
which is indistinguishable from a measured TS-group assignment. \
`burial_score` = clamp(n_residues / 8, 0, 1); a distinct-neighbour-count proxy bounded by the \
8-slot warp matrix, NOT an atom-density-within-a-sphere measurement. \
`mechanism_tag` = function of `spike_source` and `aromatic_type`. \
`intensity_percentile` = per-channel rank within THIS run only; not comparable across runs. \
`spike_id` = dense row ordinal within THIS file only; it collides across files after a multi-seed \
merge, so join on (replica_seed, spike_id). \
`nearby_residues` pads unused slots with the -1 sentinel; `n_residues` gives the valid prefix \
length. `n_residues` and `n_nearby_excited` are saturated into u8 by this writer.";

/// Schema-level changelog. Old readers see the version bump and the reason.
pub const SCHEMA_CHANGELOG: &str = "\
v1 -> v2: REMOVED the `frame_index` UInt16 column (column count 31 -> 30). It was synthesised as \
clamp(timestep / 1000, 0, 65535) -- an undeclared magic divisor applied to a protocol-local clock, \
exported under a name asserting a frame-clock measurement that was never taken. Removal is \
lossless: the retained `timestep` column reproduces the old value exactly, but per \
prism.spike_arrow.clock_domain that quantity is not a frame index and must not be used as one. \
v2 also adds schema-level metadata (version, clock domain, derived-column inventory, changelog). \
Readers that require `frame_index` by name fail loudly on the missing column; positional readers \
must consult prism.spike_arrow.schema_version before indexing. \
v2 -> v3: ADDED six clock-provenance columns (column count 30 -> 36): `global_md_step` (UInt64, \
nullable), `cumulative_simulated_time_ps` (Float64, nullable), `record_stream_id` (UInt32, \
nullable), `record_chunk_id` (UInt32, nullable), `clock_domain` (UInt8) and \
`spike_record_schema_version` (UInt32). The underlying device record widened from 96 B to 128 B to \
carry them. This is the first schema in which a spike row can be joined to a coordinate frame: use \
`global_md_step`. NULL means the record declared the field unmeasured -- distinct from 0, which is \
a legitimate step / stream / chunk. Positional readers built for v2 MUST re-read \
prism.spike_arrow.schema_version; Arrow schema equality includes the metadata map, so a v2 and a \
v3 file cannot be silently concatenated. \
v3 -> v4: REMOVED those same six clock-provenance columns (column count 36 -> 30). The 128 B \
record they came from invalidated the SEALED QNANL force-kernel packet (installed PTX \
78dc626587a498b2fc789b733af2accc9493289c36cb9d1f7525fba49b0bb1fc), and the in-record global step \
resolved only to graph-REPLAY granularity -- coarser than the per-step governor sidecar that \
supersedes it. Frame joins now go through the `.ccr` chronology sidecar emitted alongside each \
spike packet, keyed by chunk. A v3 file is NOT a v4 file: positional readers MUST re-read \
prism.spike_arrow.schema_version. \
v4 -> v5: NO column added, removed, retyped or reordered (count stays 30) and the 96 B device \
record is unchanged, so the QNANL force-kernel seal is unaffected. What changes is the declared \
VALUE DOMAIN of the existing `ccns_phase` UInt8 column: it widens from 3 codes to 5. The producer \
had exactly three branches against only two boundaries (cold_hold, ramp), so ramp_down and \
cold_return -- 10/30 of the default 5:10:5:6:4 hysteresis protocol, i.e. the entire release and \
recovery half of the kinetic signature -- were emitted as code 2 alongside the warm steady state. \
Any residence-time, reversibility or hysteresis-asymmetry measurement built on the v1..v4 column \
was reading a constant across changing physics. v5 emits 3=ramp_down and 4=cold_return against \
the protocol's own cumulative stage boundaries. The coding is POLICY-selected \
(PRISM_CCNS_PHASE_CODING, documented default 5stage) and the resolved descriptor is stamped under \
prism.spike_arrow.ccns_phase_coding, so the domain travels with the file; the legacy 3-code \
encoder stays reachable for forensic A/B against archived runs. A v4 file is NOT a v5 file: Arrow \
schema equality includes the metadata map, so the added key already blocks a silent mixed-domain \
concatenation in multi_seed_merger.";

/// Build the schema-level metadata map stamped onto every emitted file.
fn spike_schema_metadata() -> HashMap<String, String> {
    let mut meta = HashMap::with_capacity(5);
    meta.insert(
        SCHEMA_META_VERSION.to_string(),
        SPIKE_ARROW_SCHEMA_VERSION.to_string(),
    );
    meta.insert(
        SCHEMA_META_CLOCK_DOMAIN.to_string(),
        CLOCK_DOMAIN_NOTE.to_string(),
    );
    meta.insert(
        SCHEMA_META_DERIVED_COLUMNS.to_string(),
        DERIVED_COLUMNS_NOTE.to_string(),
    );
    meta.insert(
        SCHEMA_META_CHANGELOG.to_string(),
        SCHEMA_CHANGELOG.to_string(),
    );
    // The `ccns_phase` column is UInt8 under every version, so its Arrow type
    // cannot express which coding produced it. Stamp the RESOLVED policy, not
    // a literal: `CcnsPhaseCoding::resolve()` is memoized for the process, so
    // the descriptor written here and the codes written into the column by the
    // producer are provably the same resolution.
    meta.insert(
        SCHEMA_META_CCNS_PHASE_CODING.to_string(),
        CcnsPhaseCoding::resolve().descriptor().to_string(),
    );
    meta
}

/// Build the canonical Arrow schema for the per-spike output.
///
/// [`SPIKE_ARROW_COLUMN_COUNT`] columns total — see the module-level
/// documentation for the field list and rationale. Field order is **stable**
/// and used by downstream readers; do not reorder, add, or remove a field
/// without bumping [`SPIKE_ARROW_SCHEMA_VERSION`] and extending
/// [`SCHEMA_CHANGELOG`].
///
/// The returned schema carries the version, clock-domain warning, derived
/// column inventory, and changelog as schema-level metadata so the caveats
/// travel with the file. Arrow schema equality includes this metadata, so a
/// version mismatch is a hard, loud failure at merge time rather than a
/// silent positional misparse.
pub fn build_spike_schema() -> Arc<Schema> {
    // Note: the inner field of `nearby_residues` MUST be `nullable: true`
    // even though the builder always pads unused slots with -1 — this is
    // because `arrow::array::FixedSizeListBuilder` defaults its inner
    // field to nullable, and `RecordBatch::try_new` does a strict type
    // equality check including the nullable flag. Caught by
    // `record_batch_builds_from_synthetic_spikes` test.
    let nearby_field = Arc::new(Field::new("item", DataType::Int32, true));
    let fields = vec![
        // Provenance
        Field::new("spike_id", DataType::UInt64, false),
        Field::new("replica_seed", DataType::UInt64, false),
        Field::new("stream_id", DataType::UInt8, false),
        Field::new("group_id", DataType::UInt8, false),
        Field::new("chunk_idx", DataType::UInt16, false),
        Field::new("voxel_idx", DataType::Int32, false),
        // Physical state
        //
        // NOTE: `timestep` is a protocol-local clock. There is deliberately
        // NO `frame_index` column here — see SCHEMA_CHANGELOG and
        // CLOCK_DOMAIN_NOTE. Do not reintroduce one without a per-spike
        // global-MD-step witness to back it.
        Field::new("timestep", DataType::Int32, false),
        Field::new("x", DataType::Float32, false),
        Field::new("y", DataType::Float32, false),
        Field::new("z", DataType::Float32, false),
        Field::new("intensity", DataType::Float32, false),
        Field::new("spike_source", DataType::Int32, false),
        Field::new("mechanism_tag", DataType::Utf8, false),
        Field::new("aromatic_type", DataType::Int32, false),
        Field::new("aromatic_residue_id", DataType::Int32, false),
        Field::new("phase_bits", DataType::UInt32, false),
        Field::new("n_residues", DataType::UInt8, false),
        Field::new(
            "nearby_residues",
            DataType::FixedSizeList(nearby_field, NEARBY_RESIDUES_LEN),
            false,
        ),
        Field::new("n_nearby_excited", DataType::UInt8, false),
        Field::new("vibrational_energy", DataType::Float32, false),
        Field::new("water_density", DataType::Float32, false),
        Field::new("wd_change", DataType::Float32, false),
        Field::new("wavelength_nm", DataType::Float32, false),
        Field::new("ccns_phase", DataType::UInt8, false),
        // NO clock-provenance columns here.
        //
        // Schema v3 briefly carried global_md_step / cumulative_simulated_time_ps /
        // record_stream_id / record_chunk_id / clock_domain /
        // spike_record_schema_version, sourced from fields added to
        // GpuSpikeEvent. That layout change grew the record 96 -> 128 bytes and
        // broke the QNANL-sealed force-kernel packet, whose PTX is pinned by
        // digest. The spike record is reverted to 96 bytes and these columns go
        // with it.
        //
        // The clock bridge is NOT abandoned — it moves to a per-chunk governor
        // sidecar, which resolves finer than an in-record stamp could anyway:
        // the spike is emitted inside a body iteration while the global counter
        // advances once per replay, so an in-record global step only ever
        // resolved to replay granularity. Frames are at most one per chunk, so
        // chunk-resolution provenance is frame-resolution provenance.
        // Classification
        Field::new("site_id", DataType::Int32, false),
        Field::new("nearest_site_id", DataType::Int32, false),
        Field::new("nearest_site_dist", DataType::Float32, false),
        Field::new("background_class", DataType::UInt8, false),
        Field::new("burial_score", DataType::Float32, false),
        Field::new("intensity_percentile", DataType::UInt8, false),
    ];
    debug_assert_eq!(
        fields.len(),
        SPIKE_ARROW_COLUMN_COUNT,
        "SPIKE_ARROW_COLUMN_COUNT is out of sync with the field list; \
         bump SPIKE_ARROW_SCHEMA_VERSION and update SCHEMA_CHANGELOG"
    );
    Arc::new(Schema::new_with_metadata(fields, spike_schema_metadata()))
}

/// Per-spike classification metadata produced by the host before writing.
///
/// One `SpikeClassification` corresponds to one `GpuSpikeEvent` in the
/// input slice (parallel arrays). The writer expects them in matching
/// order.
#[derive(Debug, Clone, Copy)]
pub struct SpikeClassification {
    /// Stream index that emitted this spike (0..n_streams-1).
    pub stream_id: u8,
    /// TWIN group DERIVED from `stream_id` (0..3) — see
    /// [`group_id_for_stream`]. Forced to 0 for non-multi-differential runs,
    /// where it carries no measured group assignment.
    pub group_id: u8,
    /// Chunk index DERIVED host-side by dividing the protocol-local
    /// `timestep` by a caller-supplied `chunk_size` (see
    /// [`chunk_idx_for_timestep`]). This is a re-derivation, **not** a record
    /// of the chunk that actually emitted the spike, and it inherits every
    /// caveat in [`CLOCK_DOMAIN_NOTE`]. BOCPD-derived chunk boundaries land
    /// in Stage 1B-2 and will make this a real lookup.
    pub chunk_idx: u16,
    /// CCNS protocol phase DERIVED from `timestep` against the protocol's own
    /// cumulative stage boundaries. Schema v5 domain: 0=cold_hold, 1=ramp,
    /// 2=warm_hold, 3=ramp_down, 4=cold_return — see [`ccns_phase_code`].
    ///
    /// The producer must fill this through [`CcnsPhaseCoding::encode`] so the
    /// value agrees with the coding descriptor stamped under
    /// [`SCHEMA_META_CCNS_PHASE_CODING`]. Codes 3 and 4 appear only when the
    /// protocol actually declared those stages (`ramp_down_steps > 0`);
    /// otherwise the column stays 3-valued and is byte-identical to schemas
    /// v1..v4, which could never emit a cooling code at all.
    pub ccns_phase: u8,
    /// Consensus site cluster_id this spike was assigned to, or -1 for
    /// background (no site within radius).
    pub site_id: i32,
    /// Closest consensus site cluster_id (== `site_id` when assigned;
    /// for background spikes, this is the nearest site even though the
    /// spike is outside its radius).
    pub nearest_site_id: i32,
    /// Distance in Å to `nearest_site_id`.
    pub nearest_site_dist: f32,
    /// Background sub-classification (see module doc).
    pub background_class: u8,
    /// Burial proxy in [0, 1] DERIVED as `clamp(n_residues / 8, 0, 1)` — the
    /// count of distinct nearby residues the warp matrix tracked for this
    /// voxel, normalized by its 8-slot capacity. It is **not** an atom count
    /// within a sphere and **not** an atom-density measurement; a value of
    /// 1.0 means "the 8 tracked slots were full", which is a saturation
    /// artifact rather than an observation of full burial.
    pub burial_score: f32,
    /// Intensity rank within this spike's source channel [0, 100].
    pub intensity_percentile: u8,
}

/// Background sub-classes for stratified training data.
pub mod background_class {
    /// Spike fell inside a consensus site (`site_id != -1`).
    pub const PRIMARY_SITE: u8 = 0;
    /// Background spike with below-median per-channel intensity rank.
    pub const BULK_THERMAL: u8 = 1;
    /// Background spike with at-or-above-median per-channel intensity rank.
    pub const SURFACE_NOISE: u8 = 2;
    /// Background spike inside the closest per-run distance decile — the
    /// spatial frontier just outside a site radius (hard negatives).
    pub const NEAR_MISS: u8 = 3;
    /// Background spike that is both high-intensity and fully neighbour-
    /// saturated — the audit channel for potentially missed pockets.
    pub const RELABEL_CANDIDATE: u8 = 4;
}

/// Map a spike to its stable mechanism class string.
///
/// DERIVED from `spike_source` plus `aromatic_type`; it is not an independent
/// observation. `spike_source` names the physics channel rather than the
/// kernel that produced the record, so this tag must not be read as kernel
/// provenance (see [`CLOCK_DOMAIN_NOTE`], which turns on the same fact).
#[inline]
pub fn mechanism_tag_for_spike(spike: &GpuSpikeEvent) -> &'static str {
    match spike.spike_source {
        1 => "UV_AROMATIC_PERTURBATION",
        3 => "EFP_ELECTROSTATIC_FIELD",
        4 => "LADD_ATOM_DEPARTURE",
        5 => "COFIRE_COHERENCE",
        6 => "VORONOI_LIF",
        _ if spike.aromatic_type < 0 => "LIF_THERMAL_SHAPE",
        _ => "LIF_LOCAL_INTENSITY",
    }
}

/// Compute the TWIN group_id from a stream index given the multi-differential
/// engines-per-group layout.
///
/// In `--multi-differential` mode the streams are organized as 4 groups
/// (TS, EQ, UV, HY) with `n_streams / 4` engines per group. The mapping
/// is `group = stream_idx / engines_per_group`. For non-multi-diff runs
/// (where there is only one logical group), all streams map to group 0.
///
/// AUDIT NOTE: the non-multi-diff return of 0 is **not** a measured TWIN
/// group. It collides with the measured "group 0 == TS" encoding, so a
/// consumer cannot distinguish "this stream was assigned to TS" from "this
/// run had no group axis". Callers that need that distinction must carry the
/// multi-differential flag alongside the data; this function cannot express
/// it without widening the `group_id` column, which would be a schema change.
#[inline]
pub fn group_id_for_stream(stream_idx: usize, n_streams: usize, multi_diff: bool) -> u8 {
    if !multi_diff || n_streams < 4 {
        return 0;
    }
    let epg = (n_streams / 4).max(1);
    ((stream_idx / epg).min(3)) as u8
}

/// Build the chunk-boundary lookup table for host-side derivation of
/// `chunk_idx` per spike.
///
/// In Stage 1B-1 the chunk size is fixed (the `chunk_size = 500` constant
/// in the autonomous chunk loop). Each spike's `chunk_idx` is computed as
/// `min(timestep / chunk_size, max_chunk_idx)`. When BOCPD lands in Stage
/// 1B-2 the chunk boundaries become non-uniform; this function will be
/// replaced by a lookup against the BOCPD-emitted boundary table.
///
/// AUDIT NOTE: until that lookup exists this is a *re-derivation* from the
/// protocol-local clock, not a record of the emitting chunk, and it inherits
/// every caveat in [`CLOCK_DOMAIN_NOTE`]. `chunk_size` is supplied by the
/// caller rather than hardcoded here, but the caller's value is still a
/// literal; the resulting `chunk_idx` is only interpretable against that
/// same literal. Do not treat it as a wall-clock or trajectory ordering.
#[inline]
pub fn chunk_idx_for_timestep(timestep: i32, chunk_size: i32, max_chunks: i32) -> u16 {
    if chunk_size <= 0 || timestep < 0 {
        return 0;
    }
    let idx = (timestep / chunk_size).min(max_chunks - 1).max(0);
    idx as u16
}

/// Numeric codes carried by the `ccns_phase` Arrow column.
///
/// These are the five CCNS protocol stages in execution order. Codes 0..=2 are
/// byte-compatible with the legacy 3-stage encoder ([`ccns_phase_for_step`]);
/// codes 3 and 4 are new in schema v5. Before v5 they were aliased onto
/// `WARM_HOLD`, collapsing `ramp_down` + `cold_return` — 10/30 of the default
/// 5:10:5:6:4 hysteresis ratio — into the warm steady state.
pub mod ccns_phase_code {
    /// Hold at the cryogenic start temperature
    /// (`CryoUvProtocol::cold_hold_steps`).
    pub const COLD_HOLD: u8 = 0;
    /// Temperature ramp toward `end_temp` (`CryoUvProtocol::ramp_steps`).
    pub const RAMP: u8 = 1;
    /// Hold at the warm end temperature (`CryoUvProtocol::warm_hold_steps`).
    pub const WARM_HOLD: u8 = 2;
    /// Hysteresis ramp back down toward `start_temp`
    /// (`CryoUvProtocol::ramp_down_steps`). Aliased onto `WARM_HOLD` by every
    /// schema before v5 — this is the pocket-closing / ligand-egress half of
    /// the kinetic signature.
    pub const RAMP_DOWN: u8 = 3;
    /// Cold hold after the ramp-down (`CryoUvProtocol::cold_return_steps`),
    /// absorbing any post-protocol overrun. Aliased onto `WARM_HOLD` before v5.
    pub const COLD_RETURN: u8 = 4;
    /// Highest code the 5-stage encoder can emit.
    pub const MAX_CODE_5STAGE: u8 = COLD_RETURN;
    /// Highest code the legacy 3-stage encoder can emit.
    pub const MAX_CODE_3STAGE: u8 = WARM_HOLD;
}

/// Map a CCNS protocol step + phase boundaries to a phase code (LEGACY,
/// 3-value).
///
/// 0 = cold_hold, 1 = ramp, 2 = warm_hold. Matches the convention in the
/// standard step() path's phase computation at fused_engine.rs:5889 and the
/// Stage 1A.5 KCC drive at nhs_rt_full.rs:3460+.
///
/// AUDIT NOTE (retained as history; SUPERSEDED by
/// [`ccns_phase_for_step_5stage`]): this function has three branches and
/// therefore can never return a "cooling" code. The `ccns_phase` column was
/// consequently a 3-value column, not a 4-value one, and `timestep` here is
/// the protocol-local clock ([`CLOCK_DOMAIN_NOTE`]) — the phase boundaries are
/// counted in protocol steps, which are not global MD steps.
///
/// It is DEPRECATED, not removed: it remains the only way to reproduce the
/// exact `ccns_phase` values in an archived schema v1..v4 file, so a forensic
/// A/B against those runs stays reachable. It is still selectable at runtime
/// through [`CcnsPhaseCoding::ThreeStage`].
#[deprecated(
    since = "0.1.0",
    note = "3-code encoder: aliases ramp_down and cold_return onto warm_hold (code 2), \
            collapsing 10/30 of the default hysteresis protocol. Use \
            `CcnsPhaseCoding::resolve().encode(..)`, or `ccns_phase_for_step_5stage` for \
            an explicit coding. Retained and reachable for forensic A/B against archived \
            schema v1..v4 runs."
)]
#[inline]
pub fn ccns_phase_for_step(timestep: i32, cold_hold: i32, ramp: i32) -> u8 {
    if timestep < cold_hold {
        0
    } else if timestep < cold_hold + ramp {
        1
    } else {
        2
    }
}

/// Map a CCNS protocol step onto the full 5-stage phase code, against the
/// protocol's own cumulative stage boundaries.
///
/// Returns one of [`ccns_phase_code::COLD_HOLD`] (0), [`ccns_phase_code::RAMP`]
/// (1), [`ccns_phase_code::WARM_HOLD`] (2), [`ccns_phase_code::RAMP_DOWN`] (3)
/// or [`ccns_phase_code::COLD_RETURN`] (4). The boundaries are prefix sums of
/// the caller-supplied stage lengths — the same construction used by the
/// engine-side `coupled_md::ccns_phase` and by `nhs_rt_full::protocol_phase_at`,
/// so all three agree code-for-stage.
///
/// # The existing 3-value contract is preserved, not broken
///
/// A protocol that declares `ramp_down <= 0` has no hysteresis return: there is
/// no ramp-down stage, and therefore no cold-return stage for the tail to
/// belong to. In that case every step at or past the end of `warm_hold` stays
/// [`ccns_phase_code::WARM_HOLD`], which makes this function **bit-identical**
/// to [`ccns_phase_for_step`] for every non-hysteresis protocol. Codes 3 and 4
/// appear only when the protocol actually ran those stages, so the widening can
/// never invent a stage that was not executed. This matters because the TWIN16
/// `return_extended` axis raises `ramp_down_steps` and `cold_return_steps`
/// through `.max(1)`, so a TWIN16 cell always engages the 5-code path while a
/// plain non-hysteresis run is left exactly as it was.
///
/// `cold_return` is deliberately NOT a parameter: the final branch absorbs both
/// the declared cold-return stage and any post-protocol overrun, exactly as
/// `coupled_md::ccns_phase` does. Splitting overrun into a sixth code would
/// require a boundary the protocol does not define for steps past its own end,
/// which would be a fabricated stage.
///
/// # Clock domain
///
/// `timestep` is the protocol-local clock ([`CLOCK_DOMAIN_NOTE`]): the stage
/// lengths are counted in protocol steps, which are not global MD steps and not
/// frame indices. This function re-derives a phase; it does not observe one.
///
/// # Totality
///
/// Negative inputs clamp to 0 and every prefix sum saturates, so no input can
/// panic or wrap and every `i32` maps to exactly one code.
#[inline]
pub fn ccns_phase_for_step_5stage(
    timestep: i32,
    cold_hold: i32,
    ramp: i32,
    warm_hold: i32,
    ramp_down: i32,
) -> u8 {
    let t = timestep.max(0);
    let b1 = cold_hold.max(0);
    let b2 = b1.saturating_add(ramp.max(0));
    let b3 = b2.saturating_add(warm_hold.max(0));
    if t < b1 {
        ccns_phase_code::COLD_HOLD
    } else if t < b2 {
        ccns_phase_code::RAMP
    } else if t < b3 || ramp_down <= 0 {
        // `ramp_down <= 0` is the no-return-stage case: the protocol never ran
        // a ramp-down, so the tail is warm-hold overrun and NOT a cold return.
        // This is the branch that keeps the legacy 3-value contract exact.
        ccns_phase_code::WARM_HOLD
    } else if t < b3.saturating_add(ramp_down) {
        ccns_phase_code::RAMP_DOWN
    } else {
        ccns_phase_code::COLD_RETURN
    }
}

/// Environment key selecting which CCNS coding the `ccns_phase` column carries.
///
/// See [`CcnsPhaseCoding::resolve`] for the accepted tokens and
/// [`CCNS_PHASE_CODING_DEFAULT`] for the documented default.
pub const CCNS_PHASE_CODING_ENV: &str = "PRISM_CCNS_PHASE_CODING";

/// Documented default coding when [`CCNS_PHASE_CODING_ENV`] is unset, empty or
/// unrecognised: the full 5-stage coding.
pub const CCNS_PHASE_CODING_DEFAULT: CcnsPhaseCoding = CcnsPhaseCoding::FiveStage;

/// Which coding the `ccns_phase` column carries for this run.
///
/// This is a POLICY selection, not a source literal. It is resolved once per
/// process from [`CCNS_PHASE_CODING_ENV`] with the documented default
/// [`CCNS_PHASE_CODING_DEFAULT`], and the resolved descriptor is stamped into
/// every emitted file under [`SCHEMA_META_CCNS_PHASE_CODING`], so the column's
/// value domain travels with the data instead of living in a call site.
///
/// Resolution cannot fail and cannot refuse a run: an unrecognised token warns
/// and falls back to the default. Adding a further coding is an additive
/// variant here plus one arm in [`CcnsPhaseCoding::encode`] — no consumer of
/// this module hardcodes the domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CcnsPhaseCoding {
    /// Legacy 3-value coding: 0=cold_hold, 1=ramp, 2=warm_hold, with
    /// `ramp_down` and `cold_return` aliased onto `warm_hold`. This is the
    /// schema v1..v4 domain, retained and reachable so a forensic A/B against
    /// an archived run is still possible.
    ThreeStage,
    /// 5-value coding: 0=cold_hold, 1=ramp, 2=warm_hold, 3=ramp_down,
    /// 4=cold_return. Degrades to exactly the 3-value coding for any protocol
    /// declaring `ramp_down_steps <= 0`, so it never reports a stage the
    /// protocol did not run.
    FiveStage,
}

/// Process-wide memoized resolution. Memoizing is not only an optimisation for
/// the per-spike loop (tens of millions of calls per run): it is what makes the
/// descriptor stamped by [`spike_schema_metadata`] and the codes written into
/// the column provably the same resolution.
static RESOLVED_CCNS_PHASE_CODING: std::sync::OnceLock<CcnsPhaseCoding> =
    std::sync::OnceLock::new();

impl CcnsPhaseCoding {
    /// Parse a policy token.
    ///
    /// Case-insensitive, surrounding whitespace ignored. Accepted:
    /// `5stage` / `5` / `five` / `five_stage` / `5_stage` for
    /// [`CcnsPhaseCoding::FiveStage`]; `3stage` / `3` / `three` /
    /// `three_stage` / `3_stage` / `legacy` for [`CcnsPhaseCoding::ThreeStage`].
    /// Returns `None` for anything else so the caller decides the fallback.
    pub fn parse_token(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "5stage" | "5" | "five" | "five_stage" | "5_stage" => Some(Self::FiveStage),
            "3stage" | "3" | "three" | "three_stage" | "3_stage" | "legacy" => {
                Some(Self::ThreeStage)
            }
            _ => None,
        }
    }

    /// Resolve the coding for this process, memoized after the first call.
    ///
    /// Reads [`CCNS_PHASE_CODING_ENV`]; an unset variable yields
    /// [`CCNS_PHASE_CODING_DEFAULT`] silently, an unrecognised value yields the
    /// same default with a loud `warn`. This never fails and never blocks a run
    /// — it selects an encoding, it is not an admission gate.
    pub fn resolve() -> Self {
        *RESOLVED_CCNS_PHASE_CODING.get_or_init(|| match std::env::var(CCNS_PHASE_CODING_ENV) {
            Ok(raw) => match Self::parse_token(&raw) {
                Some(coding) => {
                    log::info!(
                        "[CCNS] {}={:?} -> ccns_phase coding {}",
                        CCNS_PHASE_CODING_ENV,
                        raw,
                        coding.as_str()
                    );
                    coding
                }
                None => {
                    log::warn!(
                        "[CCNS] {}={:?} unrecognised (accepted: 5stage|5|five|five_stage, \
                         3stage|3|three|three_stage|legacy) -- using default {}",
                        CCNS_PHASE_CODING_ENV,
                        raw,
                        CCNS_PHASE_CODING_DEFAULT.as_str()
                    );
                    CCNS_PHASE_CODING_DEFAULT
                }
            },
            Err(_) => CCNS_PHASE_CODING_DEFAULT,
        })
    }

    /// Canonical policy token for this coding (round-trips through
    /// [`CcnsPhaseCoding::parse_token`]).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ThreeStage => "3stage",
            Self::FiveStage => "5stage",
        }
    }

    /// Highest code this coding can emit.
    pub fn max_code(self) -> u8 {
        match self {
            Self::ThreeStage => ccns_phase_code::MAX_CODE_3STAGE,
            Self::FiveStage => ccns_phase_code::MAX_CODE_5STAGE,
        }
    }

    /// Self-describing domain string stamped into the emitted file under
    /// [`SCHEMA_META_CCNS_PHASE_CODING`]. Begins with
    /// [`CcnsPhaseCoding::as_str`].
    pub fn descriptor(self) -> &'static str {
        match self {
            Self::ThreeStage => {
                "3stage: 0=cold_hold 1=ramp 2=warm_hold. ramp_down and cold_return are \
                 ALIASED onto 2 -- this is the legacy schema v1..v4 domain, retained for \
                 forensic A/B against archived runs only. Do not read a residence time, \
                 reversibility or hysteresis asymmetry off this column."
            }
            Self::FiveStage => {
                "5stage: 0=cold_hold 1=ramp 2=warm_hold 3=ramp_down 4=cold_return. A \
                 protocol declaring ramp_down_steps<=0 has no return stages and emits only \
                 0..2, byte-identical to 3stage. Code 4 also absorbs post-protocol overrun. \
                 Boundaries are protocol-local step counts, not global MD steps."
            }
        }
    }

    /// Encode one protocol-local step under this coding.
    ///
    /// `warm_hold` and `ramp_down` are ignored by
    /// [`CcnsPhaseCoding::ThreeStage`], which cannot express those stages; they
    /// are still taken so a producer has one call shape regardless of the
    /// resolved policy and switching codings needs no call-site change.
    #[allow(deprecated)]
    #[inline]
    pub fn encode(
        self,
        timestep: i32,
        cold_hold: i32,
        ramp: i32,
        warm_hold: i32,
        ramp_down: i32,
    ) -> u8 {
        match self {
            Self::ThreeStage => ccns_phase_for_step(timestep, cold_hold, ramp),
            Self::FiveStage => {
                ccns_phase_for_step_5stage(timestep, cold_hold, ramp, warm_hold, ramp_down)
            }
        }
    }
}

/// Classify a background spike using **per-run percentile thresholds**
/// (no magic constants — every cutoff is derived from the run's actual
/// distribution) PLUS the per-channel intensity rank (which is itself a
/// percentile so requires no additional run-level statistics).
///
/// Inputs:
/// - `site_id` of this spike (-1 if background)
/// - `nearest_site_dist` of this spike in Å
/// - `intensity_percentile` of this spike (0-100, rank within source channel)
/// - `burial_score` of this spike (0-1, linear normalization of n_residues / 8)
/// - `bg_dist_p10` — 10th percentile of nearest_site_dist across all background
/// - `bg_dist_p50` — median nearest_site_dist across all background (currently
///                   computed but reserved for future refinements; not used in
///                   the v5 decision tree)
///
/// Decision tree (in order):
///   1. site_id != -1                                                    → PRIMARY_SITE (0)
///   2. nearest_site_dist < bg_dist_p10                                  → NEAR_MISS (3)
///   3. intensity_percentile > 75 AND burial_score >= 1.0                → RELABEL_CANDIDATE (4)
///   4. intensity_percentile >= 50                                       → SURFACE_NOISE (2)
///   5. otherwise (low intensity rank within channel)                    → BULK_THERMAL (1)
///
/// ## Why intensity_percentile primary, distance+burial secondary
///
/// The classifier went through three iterations on the 4LPK validation:
///
/// **v3 (percentile burial cutoffs)** — `burial > bg_burial_p50` for
/// surface_noise. The `n_residues` distribution for background spikes is
/// heavily skewed to one end on 4LPK (and similar mid-size proteins),
/// putting `bg_burial_p50 == bg_burial_p75 == 1.000`. Strict inequality
/// `burial > 1.0` matched NO spikes. surface_noise + relabel_candidate
/// were both empty.
///
/// **v4 (binary burial)** — `burial < 1.0` for surface_noise. I assumed
/// the saturation meant "75% of background is buried" but the actual
/// distribution on 4LPK was the inverse: ~90% of background is surface
/// (`burial < 1.0`), ~10% is internal (`burial == 1.0`). v4 collapsed
/// almost everything into surface_noise (12M of 13.3M = 90.4%). Bulk
/// thermal got 420 spikes, relabel_candidate got 122 — neither bucket
/// was usable for training.
///
/// **v5 (this version) — intensity_percentile primary**:
///
/// `intensity_percentile` is the only feature in the classifier that is
/// guaranteed to be well-distributed by construction (it's the per-spike
/// rank within the source channel, so it always spans 0-100 evenly with
/// no possibility of saturation). Using it as the surface_noise vs
/// bulk_thermal divider gives a roughly even split of the residual
/// background population, regardless of the underlying burial distribution.
///
/// distance is still used for near_miss (the spatial frontier — hard
/// negatives), and burial is still used as a tiebreaker for
/// relabel_candidate (the high-intensity-AND-internal class — the
/// audit channel for missed pockets), but neither distance nor burial
/// drives the bulk surface/thermal split anymore. That avoids the
/// bimodal-distribution failure mode that broke v3 and v4.
///
/// Predicted v5 distribution on 4LPK background (13.3M spikes):
///   - NEAR_MISS:         ~10% = 1.3M (closest distance decile, by definition)
///   - RELABEL_CANDIDATE: ~2-3% (high intensity AND buried)
///   - SURFACE_NOISE:     ~40-45% (above-median intensity)
///   - BULK_THERMAL:      ~40-45% (below-median intensity)
///
/// All four background classes have meaningful population, suitable for
/// stratified training data sampling by difficulty.
///
/// `burial_score` is still emitted as an Arrow column for downstream
/// fine-grained queries — the classification field is a coarse 5-way
/// stratification, not a complete description. Users with different
/// distributions on different proteins can re-classify offline using
/// the raw columns.
#[inline]
pub fn classify_background(
    site_id: i32,
    nearest_site_dist: f32,
    intensity_percentile: u8,
    burial_score: f32,
    bg_dist_p10: f32,
    _bg_dist_p50: f32,
) -> u8 {
    if site_id != -1 {
        return background_class::PRIMARY_SITE;
    }
    if nearest_site_dist < bg_dist_p10 {
        return background_class::NEAR_MISS;
    }
    if intensity_percentile > 75 && burial_score >= 1.0 {
        return background_class::RELABEL_CANDIDATE;
    }
    if intensity_percentile >= 50 {
        return background_class::SURFACE_NOISE;
    }
    background_class::BULK_THERMAL
}

/// Compute the per-run percentiles needed for `classify_background`.
///
/// Walks the background spike list once, collecting `nearest_site_dist`,
/// sorts, returns `(dist_p10, dist_p50)`. Returns zeros if there are no
/// background spikes.
///
/// Burial percentiles are no longer needed — see the v3 smoke validation
/// note in `classify_background` for why burial is used as a binary
/// feature instead of a percentile-ranked one.
pub fn compute_background_percentiles(
    background_classifications: &[SpikeClassification],
) -> (f32, f32) {
    if background_classifications.is_empty() {
        return (0.0, 0.0);
    }
    let mut dists: Vec<f32> = background_classifications
        .iter()
        .filter(|c| c.site_id == -1)
        .map(|c| c.nearest_site_dist)
        .collect();
    if dists.is_empty() {
        return (0.0, 0.0);
    }
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = dists.len();
    let p10_idx = ((n as f32) * 0.10) as usize;
    let p50_idx = n / 2;
    (
        dists.get(p10_idx.min(n - 1)).copied().unwrap_or(0.0),
        dists.get(p50_idx.min(n - 1)).copied().unwrap_or(0.0),
    )
}

/// Build a `RecordBatch` from parallel slices of spikes and classifications.
///
/// Both slices must have the same length. Returns an error if any column
/// builder fails. The output batch contains all
/// [`SPIKE_ARROW_COLUMN_COUNT`] columns of the canonical schema, carrying
/// the schema-level metadata from [`build_spike_schema`].
///
/// `spike_id` is assigned as the dense row ordinal of this call. It is unique
/// within the resulting file only — concatenating two files (see
/// [`crate::multi_seed_merger`]) produces duplicate `spike_id` values, so the
/// joint key across a merged corpus is `(replica_seed, spike_id)`.
pub fn build_spike_record_batch(
    spikes: &[GpuSpikeEvent],
    classifications: &[SpikeClassification],
    replica_seed: u64,
) -> Result<RecordBatch> {
    build_spike_record_batch_at_offset(
        spikes,
        classifications,
        replica_seed,
        0,
        build_spike_schema(),
    )
}

// One authoritative row conversion for monolithic and bounded exports. Freeze
// schema metadata once per export rather than resolving environment policy
// independently at every batch boundary.
fn build_spike_record_batch_at_offset(
    spikes: &[GpuSpikeEvent],
    classifications: &[SpikeClassification],
    replica_seed: u64,
    first_spike_id: u64,
    schema: Arc<Schema>,
) -> Result<RecordBatch> {
    if spikes.len() != classifications.len() {
        anyhow::bail!(
            "spike/classification length mismatch: {} vs {}",
            spikes.len(),
            classifications.len()
        );
    }
    let n = spikes.len();
    let row_count = u64::try_from(n).context("spike row count exceeds u64")?;
    if row_count > 0 {
        first_spike_id
            .checked_add(row_count - 1)
            .context("spike_id range exceeds u64")?;
    }

    // Allocate column builders with capacity hints.
    let mut spike_id = UInt64Builder::with_capacity(n);
    let mut replica_seed_b = UInt64Builder::with_capacity(n);
    let mut stream_id_b = UInt8Builder::with_capacity(n);
    let mut group_id_b = UInt8Builder::with_capacity(n);
    let mut chunk_idx_b = UInt16Builder::with_capacity(n);
    let mut voxel_idx_b = Int32Builder::with_capacity(n);
    let mut timestep_b = Int32Builder::with_capacity(n);
    let mut x_b = Float32Builder::with_capacity(n);
    let mut y_b = Float32Builder::with_capacity(n);
    let mut z_b = Float32Builder::with_capacity(n);
    let mut intensity_b = Float32Builder::with_capacity(n);
    let mut spike_source_b = Int32Builder::with_capacity(n);
    let mut mechanism_tag_b = StringBuilder::with_capacity(n, n * 24);
    let mut aromatic_type_b = Int32Builder::with_capacity(n);
    let mut aromatic_resid_b = Int32Builder::with_capacity(n);
    let mut phase_bits_b = UInt32Builder::with_capacity(n);
    let mut n_residues_b = UInt8Builder::with_capacity(n);
    let mut nearby_b = FixedSizeListBuilder::with_capacity(
        Int32Builder::with_capacity(n * NEARBY_RESIDUES_LEN as usize),
        NEARBY_RESIDUES_LEN,
        n,
    );
    let mut n_excited_b = UInt8Builder::with_capacity(n);
    let mut vib_energy_b = Float32Builder::with_capacity(n);
    let mut water_density_b = Float32Builder::with_capacity(n);
    let mut wd_change_b = Float32Builder::with_capacity(n);
    let mut wavelength_b = Float32Builder::with_capacity(n);
    let mut ccns_phase_b = UInt8Builder::with_capacity(n);
    // No clock-provenance builders: schema v3's six clock columns were
    // withdrawn with the 96 B record revert. See SCHEMA_CHANGELOG.
    let mut site_id_b = Int32Builder::with_capacity(n);
    let mut nearest_site_id_b = Int32Builder::with_capacity(n);
    let mut nearest_site_dist_b = Float32Builder::with_capacity(n);
    let mut background_class_b = UInt8Builder::with_capacity(n);
    let mut burial_score_b = Float32Builder::with_capacity(n);
    let mut intensity_pct_b = UInt8Builder::with_capacity(n);

    for (i, (spike, cls)) in spikes.iter().zip(classifications.iter()).enumerate() {
        // File-local row ordinal, continuous across bounded batches. Range
        // checked before allocating builders; not unique across seed files.
        spike_id.append_value(first_spike_id + i as u64);
        replica_seed_b.append_value(replica_seed);
        stream_id_b.append_value(cls.stream_id);
        group_id_b.append_value(cls.group_id);
        chunk_idx_b.append_value(cls.chunk_idx);
        voxel_idx_b.append_value(spike.voxel_idx);
        // `timestep` is exported raw and unscaled. Schema v1 also emitted a
        // `frame_index` column here as `clamp(timestep / 1000, 0, u16::MAX)`;
        // that was a fabricated frame clock and has been removed (see
        // SCHEMA_CHANGELOG). Never re-derive a frame index from this value.
        timestep_b.append_value(spike.timestep);
        x_b.append_value(spike.position[0]);
        y_b.append_value(spike.position[1]);
        z_b.append_value(spike.position[2]);
        intensity_b.append_value(spike.intensity);
        spike_source_b.append_value(spike.spike_source);
        mechanism_tag_b.append_value(mechanism_tag_for_spike(spike));
        aromatic_type_b.append_value(spike.aromatic_type);
        aromatic_resid_b.append_value(spike.aromatic_residue_id);
        phase_bits_b.append_value(spike.phase_bits);
        n_residues_b.append_value(spike.n_residues.clamp(0, NEARBY_RESIDUES_LEN) as u8);

        // Fixed-size list of 8 nearby residues. Always exactly 8 entries; pad
        // unused slots with -1 so the schema is rectangular.
        let n_valid = spike.n_residues.clamp(0, NEARBY_RESIDUES_LEN) as usize;
        for j in 0..NEARBY_RESIDUES_LEN as usize {
            if j < n_valid {
                nearby_b.values().append_value(spike.nearby_residues[j]);
            } else {
                nearby_b.values().append_value(-1);
            }
        }
        nearby_b.append(true);

        n_excited_b.append_value(spike.n_nearby_excited.clamp(0, u8::MAX as i32) as u8);
        vib_energy_b.append_value(spike.vibrational_energy);
        water_density_b.append_value(spike.water_density);
        wd_change_b.append_value(spike.wd_change);
        wavelength_b.append_value(spike.wavelength_nm);
        ccns_phase_b.append_value(cls.ccns_phase);
        // No clock provenance is written per spike. The 96 B record carries no
        // global-MD-step / simulated-time / stream / chunk witness, and nothing
        // host-side may be substituted here: `cls.stream_id` and
        // `cls.chunk_idx` are host re-derivations already exported under their
        // own names, and re-exporting them as record provenance would assert a
        // device measurement that was never taken. Frame joins go through the
        // per-chunk governor sidecar (`coupled_chronology`, the `.ccr` stream).
        site_id_b.append_value(cls.site_id);
        nearest_site_id_b.append_value(cls.nearest_site_id);
        nearest_site_dist_b.append_value(cls.nearest_site_dist);
        background_class_b.append_value(cls.background_class);
        burial_score_b.append_value(cls.burial_score);
        intensity_pct_b.append_value(cls.intensity_percentile);
    }

    let columns: Vec<ArrayRef> = vec![
        Arc::new(spike_id.finish()),
        Arc::new(replica_seed_b.finish()),
        Arc::new(stream_id_b.finish()),
        Arc::new(group_id_b.finish()),
        Arc::new(chunk_idx_b.finish()),
        Arc::new(voxel_idx_b.finish()),
        Arc::new(timestep_b.finish()),
        Arc::new(x_b.finish()),
        Arc::new(y_b.finish()),
        Arc::new(z_b.finish()),
        Arc::new(intensity_b.finish()),
        Arc::new(spike_source_b.finish()),
        Arc::new(mechanism_tag_b.finish()),
        Arc::new(aromatic_type_b.finish()),
        Arc::new(aromatic_resid_b.finish()),
        Arc::new(phase_bits_b.finish()),
        Arc::new(n_residues_b.finish()),
        Arc::new(nearby_b.finish()),
        Arc::new(n_excited_b.finish()),
        Arc::new(vib_energy_b.finish()),
        Arc::new(water_density_b.finish()),
        Arc::new(wd_change_b.finish()),
        Arc::new(wavelength_b.finish()),
        Arc::new(ccns_phase_b.finish()),
        Arc::new(site_id_b.finish()),
        Arc::new(nearest_site_id_b.finish()),
        Arc::new(nearest_site_dist_b.finish()),
        Arc::new(background_class_b.finish()),
        Arc::new(burial_score_b.finish()),
        Arc::new(intensity_pct_b.finish()),
    ];

    // Diagnostic: log column lengths and types so RecordBatch::try_new failures
    // can be diagnosed without running the full pipeline twice.
    if log::log_enabled!(log::Level::Debug) || cfg!(debug_assertions) {
        for (i, (field, col)) in schema.fields().iter().zip(columns.iter()).enumerate() {
            log::debug!(
                "  Arrow col {}: {} expected={:?} got={:?} len={}",
                i,
                field.name(),
                field.data_type(),
                col.data_type(),
                col.len()
            );
        }
    }
    RecordBatch::try_new(schema.clone(), columns).map_err(|e| {
        // Re-walk on failure to give actionable diagnostics in the log.
        log::warn!(
            "  RecordBatch::try_new failed with {} expected rows: {}",
            n,
            e
        );
        anyhow::anyhow!("RecordBatch::try_new failed: {}", e)
    })
}

/// Write a `RecordBatch` to disk as an Arrow IPC file (`.arrow` extension).
///
/// Uses `arrow::ipc::writer::FileWriter` which produces a self-describing
/// file with the schema header followed by one or more record batches.
/// For Stage 1B-1 we write a single batch containing all spikes from the
/// run; Stage 1B-2 will switch to streaming per-chunk batches.
pub fn write_spike_arrow_file(path: &Path, batch: &RecordBatch) -> Result<()> {
    use std::fs::File;
    let file = File::create(path)
        .with_context(|| format!("Failed to create Arrow file: {}", path.display()))?;
    let mut writer =
        FileWriter::try_new(file, &batch.schema()).context("FileWriter::try_new failed")?;
    writer.write(batch).context("FileWriter::write failed")?;
    writer.finish().context("FileWriter::finish failed")?;
    Ok(())
}

/// Default maximum rows held in Arrow column builders at one time. This bounds
/// the *export* working set; caller-owned spikes/classifications remain live.
pub const SPIKE_ARROW_DEFAULT_BATCH_ROWS: usize = 65_536;

/// Successful container close; file API additionally syncs file and parent.
/// This is not a restart/checkpoint or scientific-acceptance gate.
/// The run identifier is receipt-only: canonical schema metadata is unchanged.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct SpikeArrowWriteReceipt {
    pub run_id: String,
    pub row_count: usize,
    pub batch_count: usize,
    pub batch_rows_limit: usize,
    pub max_batch_rows: usize,
    pub first_spike_id: Option<u64>,
    pub last_spike_id: Option<u64>,
    pub column_count: usize,
    pub format: &'static str,
    pub durability: &'static str,
}

fn validate_bounded_export(
    spikes: &[GpuSpikeEvent],
    classifications: &[SpikeClassification],
    batch_rows: usize,
) -> Result<()> {
    anyhow::ensure!(batch_rows > 0, "Arrow batch_rows must be positive");
    anyhow::ensure!(
        spikes.len() == classifications.len(),
        "spike/classification length mismatch: {} vs {}",
        spikes.len(),
        classifications.len()
    );
    u64::try_from(spikes.len()).context("spike row count exceeds u64")?;
    Ok(())
}

fn write_spike_batches(
    spikes: &[GpuSpikeEvent],
    classifications: &[SpikeClassification],
    replica_seed: u64,
    run_id: &str,
    batch_rows: usize,
    schema: Arc<Schema>,
    format: &'static str,
    mut write_batch: impl FnMut(&RecordBatch) -> Result<()>,
) -> Result<SpikeArrowWriteReceipt> {
    validate_bounded_export(spikes, classifications, batch_rows)?;
    let mut batch_count = 0;
    let mut max_batch_rows = 0;
    for start in (0..spikes.len()).step_by(batch_rows) {
        let end = start.saturating_add(batch_rows).min(spikes.len());
        let batch = build_spike_record_batch_at_offset(
            &spikes[start..end],
            &classifications[start..end],
            replica_seed,
            u64::try_from(start).context("spike offset exceeds u64")?,
            schema.clone(),
        )?;
        write_batch(&batch).with_context(|| {
            format!("Arrow batch {} rows {}..{} failed", batch_count, start, end)
        })?;
        batch_count += 1;
        max_batch_rows = max_batch_rows.max(end - start);
        // No batch collection: the columns are released before the next build.
    }
    Ok(SpikeArrowWriteReceipt {
        run_id: run_id.to_owned(),
        row_count: spikes.len(),
        batch_count,
        batch_rows_limit: batch_rows,
        max_batch_rows,
        first_spike_id: if spikes.is_empty() { None } else { Some(0) },
        last_spike_id: if spikes.is_empty() {
            None
        } else {
            Some(spikes.len() as u64 - 1)
        },
        column_count: schema.fields().len(),
        format,
        durability: "caller_managed_sink",
    })
}

/// Write bounded record batches in the existing Arrow IPC *file* container.
/// Existing FileReader consumers can iterate every batch without a migration.
/// Preserves current schema, metadata, row order and file-local spike IDs.
/// Preflight failures do not create/truncate the destination. I/O failures may
/// leave an incomplete file and are returned, never promoted to a receipt.

/// Explicit new-profile authority. Existing default schema/metadata stay exact.
pub const SCHEMA_META_CCNS_STREAM_SCHEDULE:&str="prism.spike_arrow.ccns_stream_schedule";

fn explicit_stream_phase_schema(
    spikes:&[GpuSpikeEvent], classifications:&[SpikeClassification],
    authority:&serde_json::Value,
)->Result<Arc<Schema>> {
    anyhow::ensure!(CcnsPhaseCoding::resolve()==CcnsPhaseCoding::FiveStage
        && authority["schema"]=="prism.native.executed_stream_phase_authority.v1"
        && authority["protocol_profile_id"]=="differential_initial_cold2000_fixed2_v1"
        && authority["step_domain"]=="recorded protocol-local timestep"
        && authority["continuation_rule"]=="Director keeps cold_return after nominal total; no cycle reset"
        && authority["retained_event_count"].as_u64()==Some(spikes.len() as u64),
        "explicit stream-phase authority identity/coding/population differs");
    let streams=authority["streams"].as_array().context("stream-phase authority streams absent")?;
    anyhow::ensure!(streams.len()==8,"stream-phase authority requires eight owners");
    let mut next=0usize;
    for (sid,row) in streams.iter().enumerate() {
        let first=usize::try_from(row["first_event_ordinal"].as_u64().context("stream first ordinal absent")?)?;
        let end=usize::try_from(row["next_event_ordinal"].as_u64().context("stream next ordinal absent")?)?;
        let boundaries=row["phase_end_steps"].as_array().context("stream phase boundaries absent")?;
        anyhow::ensure!(row["stream_id"].as_u64()==Some(sid as u64)
            && row["group_id"].as_u64()==Some((sid/2) as u64)
            && row["shared_horizon_steps"].as_i64()==Some(36000)
            && row["post_nominal_phase_code"].as_u64()==Some(4)
            && first==next && end>=first && end<=spikes.len() && boundaries.len()==5,
            "stream-phase authority owner/range/continuation mismatch");
        let mut b=[0i32;5];
        for i in 0..5 {b[i]=i32::try_from(boundaries[i].as_i64().context("noninteger stream boundary")?)?;}
        anyhow::ensure!(b[0]>0 && b.windows(2).all(|w|w[0]<w[1]) && b[4]<=36000,
            "stream-phase authority must retain every five-phase interval");
        for index in first..end {
            let step=spikes[index].timestep;
            let expected=CcnsPhaseCoding::FiveStage.encode(step,b[0],b[1]-b[0],b[2]-b[1],b[3]-b[2]);
            anyhow::ensure!(step>=0 && step<=36000
                && classifications[index].stream_id as usize==sid
                && classifications[index].group_id as usize==sid/2
                && classifications[index].ccns_phase==expected,
                "actual Arrow datum owner/ccns phase differs from executed stream schedule at ordinal {index}");
        }
        next=end;
    }
    anyhow::ensure!(next==spikes.len(),"stream-phase authority omits retained event tail");
    let base=build_spike_schema();let mut metadata=base.metadata().clone();
    let legacy="Under --multi-differential the stage boundaries used are the RUN-LEVEL protocol's, not the per-stream condition cell's -- a property codes 0..2 already had before v5, not one introduced by the widening. ";
    let current="Under this explicit stream-phase authority the recorded timestep is classified with its owning stream's executed cell boundaries. The stream ordinal intervals and complete five-phase schedules travel under prism.spike_arrow.ccns_stream_schedule. Cold return continues after nominal total without resetting the cycle. This remains a host-derived protocol-local classification, not a measured per-event device phase or kernel/subphase identity. ";
    anyhow::ensure!(DERIVED_COLUMNS_NOTE.contains(legacy),"legacy derived-column note changed unexpectedly");
    metadata.insert(SCHEMA_META_DERIVED_COLUMNS.to_string(),DERIVED_COLUMNS_NOTE.replace(legacy,current));
    metadata.insert(SCHEMA_META_CCNS_STREAM_SCHEDULE.to_string(),serde_json::to_string(authority)?);
    Ok(Arc::new(Schema::new_with_metadata(base.fields().clone(),metadata)))
}

pub fn write_spike_arrow_file_batched_with_stream_phase_authority(
    path: &Path,
    spikes: &[GpuSpikeEvent],
    classifications: &[SpikeClassification],
    replica_seed: u64,
    run_id: &str,
    batch_rows: usize,
    authority:&serde_json::Value,
) -> Result<SpikeArrowWriteReceipt> {
    validate_bounded_export(spikes, classifications, batch_rows)?;
    let schema = explicit_stream_phase_schema(spikes,classifications,authority)?;
    let file = std::fs::File::create(path)
        .with_context(|| format!("Failed to create Arrow file: {}", path.display()))?;
    let mut writer = FileWriter::try_new(file, &schema).context("FileWriter::try_new failed")?;
    let mut receipt = write_spike_batches(
        spikes,
        classifications,
        replica_seed,
        run_id,
        batch_rows,
        schema,
        "arrow_ipc_file",
        |batch| writer.write(batch).context("FileWriter::write failed"),
    )?;
    writer.finish().context("FileWriter::finish failed")?;
    writer
        .get_ref()
        .sync_all()
        .context("Arrow file sync_all failed")?;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::File::open(parent)
        .with_context(|| format!("Failed to open Arrow parent: {}", parent.display()))?
        .sync_all()
        .context("Arrow parent directory sync_all failed")?;
    receipt.durability = "file_and_parent_fsynced";
    Ok(receipt)
}

pub fn write_spike_arrow_file_batched(
    path: &Path,
    spikes: &[GpuSpikeEvent],
    classifications: &[SpikeClassification],
    replica_seed: u64,
    run_id: &str,
    batch_rows: usize,
) -> Result<SpikeArrowWriteReceipt> {
    validate_bounded_export(spikes, classifications, batch_rows)?;
    let schema = build_spike_schema();
    let file = std::fs::File::create(path)
        .with_context(|| format!("Failed to create Arrow file: {}", path.display()))?;
    let mut writer = FileWriter::try_new(file, &schema).context("FileWriter::try_new failed")?;
    let mut receipt = write_spike_batches(
        spikes,
        classifications,
        replica_seed,
        run_id,
        batch_rows,
        schema,
        "arrow_ipc_file",
        |batch| writer.write(batch).context("FileWriter::write failed"),
    )?;
    writer.finish().context("FileWriter::finish failed")?;
    writer
        .get_ref()
        .sync_all()
        .context("Arrow file sync_all failed")?;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::File::open(parent)
        .with_context(|| format!("Failed to open Arrow parent: {}", parent.display()))?
        .sync_all()
        .context("Arrow parent directory sync_all failed")?;
    receipt.durability = "file_and_parent_fsynced";
    Ok(receipt)
}

/// Explicit IPC *stream* alternative for a StreamReader consumer. This does
/// not silently change the `.arrow` file API's established container format.
/// The output sink is owned by the caller; no per-run schema metadata is added.
pub fn write_spike_arrow_stream_batched<W: std::io::Write>(
    output: W,
    spikes: &[GpuSpikeEvent],
    classifications: &[SpikeClassification],
    replica_seed: u64,
    run_id: &str,
    batch_rows: usize,
) -> Result<SpikeArrowWriteReceipt> {
    validate_bounded_export(spikes, classifications, batch_rows)?;
    let schema = build_spike_schema();
    let mut writer =
        StreamWriter::try_new(output, &schema).context("StreamWriter::try_new failed")?;
    let receipt = write_spike_batches(
        spikes,
        classifications,
        replica_seed,
        run_id,
        batch_rows,
        schema,
        "arrow_ipc_stream",
        |batch| writer.write(batch).context("StreamWriter::write failed"),
    )?;
    writer.finish().context("StreamWriter::finish failed")?;
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Int32Array};

    #[test]
    fn voronoi_source_has_distinct_mechanism_and_legacy_tags_are_unchanged() {
        for aromatic_type in [-1, 0] {
            let event = GpuSpikeEvent { spike_source: 6, aromatic_type, ..GpuSpikeEvent::default() };
            assert_eq!(mechanism_tag_for_spike(&event), "VORONOI_LIF");
        }
        for (source, expected) in [(1,"UV_AROMATIC_PERTURBATION"),(3,"EFP_ELECTROSTATIC_FIELD"),
                                    (4,"LADD_ATOM_DEPARTURE"),(5,"COFIRE_COHERENCE")] {
            let event = GpuSpikeEvent { spike_source: source, aromatic_type: -1, ..GpuSpikeEvent::default() };
            assert_eq!(mechanism_tag_for_spike(&event), expected);
        }
        for (aromatic_type, expected) in [(-1,"LIF_THERMAL_SHAPE"),(0,"LIF_LOCAL_INTENSITY")] {
            let event = GpuSpikeEvent { spike_source: 2, aromatic_type, ..GpuSpikeEvent::default() };
            assert_eq!(mechanism_tag_for_spike(&event), expected);
        }
    }

    #[test]
    fn schema_has_expected_columns() {
        // 6 provenance + 18 physical state + 6 classification = 30 columns
        // total (see module-level documentation for the exact field list).
        // Schema v3's 6 clock-provenance columns were withdrawn in v4.
        let s = build_spike_schema();
        assert_eq!(s.fields().len(), SPIKE_ARROW_COLUMN_COUNT);
        assert_eq!(s.fields().len(), 30);
        assert!(s.field_with_name("mechanism_tag").is_ok());
        assert!(s.field_with_name("timestep").is_ok());
    }

    /// Schema v4 withdrew the six per-spike clock-provenance columns v3 had
    /// added. The 96 B device record carries no such witness, and nothing
    /// host-side may be re-exported under those names: doing so would assert a
    /// device measurement that was never taken. The frame bridge is the
    /// per-chunk governor sidecar (`coupled_chronology`, the `.ccr` stream).
    #[test]
    fn schema_carries_no_per_spike_clock_provenance() {
        let s = build_spike_schema();
        for name in [
            "global_md_step",
            "cumulative_simulated_time_ps",
            "record_stream_id",
            "record_chunk_id",
            "clock_domain",
            "spike_record_schema_version",
        ] {
            assert!(
                s.field_with_name(name).is_err(),
                "`{name}` was withdrawn with the 96 B record revert and must not \
                 be reintroduced without a real per-spike witness behind it; \
                 see SCHEMA_CHANGELOG"
            );
        }
    }

    /// Schema v2 removed the synthesised `frame_index` column. It must not be
    /// reintroduced under that name (or any name that reads as a frame clock)
    /// without a real per-spike global-MD-step witness behind it.
    #[test]
    fn schema_carries_no_fabricated_frame_index() {
        let s = build_spike_schema();
        assert!(
            s.field_with_name("frame_index").is_err(),
            "`frame_index` was a fabricated column (timestep / 1000) and must \
             stay removed; see SCHEMA_CHANGELOG"
        );
        for f in s.fields() {
            let name = f.name().to_ascii_lowercase();
            assert!(
                !(name.contains("frame") && !name.contains("not_a_frame")),
                "field `{}` names a frame clock, but no per-spike global-MD-step \
                 witness exists in this schema (see CLOCK_DOMAIN_NOTE)",
                f.name()
            );
        }
    }

    /// The clock-domain caveat must travel inside every emitted file, not
    /// just in this source tree, and the version stamp must match the
    /// constant so a merge of mixed-version files fails loudly.
    #[test]
    fn schema_metadata_records_version_and_clock_domain() {
        let s = build_spike_schema();
        let meta = s.metadata();
        let expected_version = SPIKE_ARROW_SCHEMA_VERSION.to_string();
        assert_eq!(
            meta.get(SCHEMA_META_VERSION).map(String::as_str),
            Some(expected_version.as_str())
        );
        let clock = meta
            .get(SCHEMA_META_CLOCK_DOMAIN)
            .expect("clock-domain note must be stamped into the schema");
        assert!(clock.contains("PROTOCOL-LOCAL"));
        assert!(clock.contains("NOT a frame index"));
        assert!(meta.contains_key(SCHEMA_META_DERIVED_COLUMNS));
        assert!(meta
            .get(SCHEMA_META_CHANGELOG)
            .is_some_and(|c| c.contains("frame_index")));
    }

    /// A v1 reader that pinned the old 31-column schema must not be able to
    /// concatenate against a v2 batch: Arrow schema equality covers both the
    /// field list and the metadata map, which is what `multi_seed_merger`
    /// relies on to refuse a silent mixed-version merge.
    #[test]
    fn v2_schema_is_not_equal_to_the_v1_field_list() {
        let v2 = build_spike_schema();
        // Reconstruct the v1 field list from the CURRENT one: add
        // `frame_index` after `timestep` and drop the metadata map. v4 restored
        // the v2 field list exactly, so no clock columns need filtering out.
        let mut v1_fields: Vec<Field> = Vec::with_capacity(31);
        for f in v2.fields() {
            v1_fields.push(f.as_ref().clone());
            if f.name() == "timestep" {
                v1_fields.push(Field::new("frame_index", DataType::UInt16, false));
            }
        }
        let v1 = Schema::new(v1_fields);
        assert_eq!(v1.fields().len(), 31);
        assert_ne!(&v1, v2.as_ref());
    }

    #[test]
    fn group_id_for_stream_handles_multi_diff() {
        // 8 streams in multi-diff: 2 engines per group, groups 0..3
        assert_eq!(group_id_for_stream(0, 8, true), 0);
        assert_eq!(group_id_for_stream(1, 8, true), 0);
        assert_eq!(group_id_for_stream(2, 8, true), 1);
        assert_eq!(group_id_for_stream(3, 8, true), 1);
        assert_eq!(group_id_for_stream(7, 8, true), 3);
        // Without multi-diff, all streams are group 0
        assert_eq!(group_id_for_stream(5, 8, false), 0);
        // Edge case: fewer than 4 streams
        assert_eq!(group_id_for_stream(2, 3, true), 0);
    }

    #[test]
    fn classify_background_decision_tree() {
        // v5 args: (site_id, nearest_dist, intensity_pct, burial,
        //          bg_dist_p10, bg_dist_p50)

        // Primary site (assigned)
        assert_eq!(
            classify_background(2, 0.0, 50, 0.5, 5.0, 12.0),
            background_class::PRIMARY_SITE
        );
        // Near miss (closer than p10 = 5.0)
        assert_eq!(
            classify_background(-1, 4.0, 50, 1.0, 5.0, 12.0),
            background_class::NEAR_MISS
        );
        // Relabel candidate (high intensity AND fully buried)
        assert_eq!(
            classify_background(-1, 15.0, 90, 1.0, 5.0, 12.0),
            background_class::RELABEL_CANDIDATE
        );
        // Surface noise (above-median intensity_percentile, not buried-and-high)
        assert_eq!(
            classify_background(-1, 15.0, 60, 0.5, 5.0, 12.0),
            background_class::SURFACE_NOISE
        );
        // Surface noise (above-median intensity, fully buried but
        // intensity_pct <= 75 so doesn't qualify for relabel)
        assert_eq!(
            classify_background(-1, 15.0, 60, 1.0, 5.0, 12.0),
            background_class::SURFACE_NOISE
        );
        // Bulk thermal (low intensity_percentile, anything else)
        assert_eq!(
            classify_background(-1, 15.0, 30, 1.0, 5.0, 12.0),
            background_class::BULK_THERMAL
        );
        assert_eq!(
            classify_background(-1, 15.0, 10, 0.3, 5.0, 12.0),
            background_class::BULK_THERMAL
        );
    }

    /// The legacy 3-code encoder stays reachable and stays exactly as it was.
    /// `allow(deprecated)` is deliberate: this test IS the reachability proof.
    #[test]
    #[allow(deprecated)]
    fn ccns_phase_boundaries() {
        let cold = 14000;
        let ramp = 6000;
        assert_eq!(ccns_phase_for_step(0, cold, ramp), 0); // cold
        assert_eq!(ccns_phase_for_step(13999, cold, ramp), 0); // cold
        assert_eq!(ccns_phase_for_step(14000, cold, ramp), 1); // ramp
        assert_eq!(ccns_phase_for_step(19999, cold, ramp), 1); // ramp
        assert_eq!(ccns_phase_for_step(20000, cold, ramp), 2); // warm
    }

    /// Every one of the five protocol stages must land on its own code. This is
    /// the assertion that the 33% temporal aliasing is actually gone.
    #[test]
    fn ccns_phase_5stage_resolves_every_protocol_stage() {
        // 5:10:5:6:4 scaled to a 3000-step budget -> 500:1000:500:600:400.
        let (cold, ramp, warm, down) = (500, 1000, 500, 600);
        for (t, want) in [
            (0, ccns_phase_code::COLD_HOLD),
            (499, ccns_phase_code::COLD_HOLD),
            (500, ccns_phase_code::RAMP),
            (1499, ccns_phase_code::RAMP),
            (1500, ccns_phase_code::WARM_HOLD),
            (1999, ccns_phase_code::WARM_HOLD),
            (2000, ccns_phase_code::RAMP_DOWN),
            (2599, ccns_phase_code::RAMP_DOWN),
            (2600, ccns_phase_code::COLD_RETURN),
            (2999, ccns_phase_code::COLD_RETURN),
            // Post-protocol overrun folds into cold_return, matching the
            // engine-side `coupled_md::ccns_phase` convention.
            (9_999_999, ccns_phase_code::COLD_RETURN),
        ] {
            assert_eq!(
                ccns_phase_for_step_5stage(t, cold, ramp, warm, down),
                want,
                "protocol-local step {t} landed in the wrong CCNS stage"
            );
        }
    }

    /// A protocol with no ramp-down declares no return stages, so the widened
    /// encoder must reproduce the legacy 3-value coding EXACTLY. This is the
    /// contract that lets the default be 5stage without reinterpreting a single
    /// non-hysteresis run.
    #[test]
    #[allow(deprecated)]
    fn ccns_phase_5stage_is_bit_identical_without_a_return_stage() {
        let (cold, ramp, warm) = (14000, 6000, 5000);
        for t in [
            -5, 0, 1, 13_999, 14_000, 19_999, 20_000, 24_999, 25_000, 1_000_000,
        ] {
            assert_eq!(
                ccns_phase_for_step_5stage(t, cold, ramp, warm, 0),
                ccns_phase_for_step(t, cold, ramp),
                "ramp_down<=0 must reproduce the legacy 3-value coding exactly at step {t}"
            );
        }
    }

    /// The encoder is total: no input panics, wraps, or falls through.
    #[test]
    fn ccns_phase_5stage_is_total_over_pathological_boundaries() {
        assert_eq!(
            ccns_phase_for_step_5stage(i32::MAX, i32::MAX, i32::MAX, i32::MAX, i32::MAX),
            ccns_phase_code::COLD_RETURN
        );
        assert_eq!(
            ccns_phase_for_step_5stage(-1, -1, -1, -1, -1),
            ccns_phase_code::WARM_HOLD
        );
        assert_eq!(
            ccns_phase_for_step_5stage(0, 0, 0, 0, 0),
            ccns_phase_code::WARM_HOLD
        );
    }

    /// The coding is a resolved policy with a documented default, not a source
    /// literal, and BOTH codings stay reachable with their declared domains.
    #[test]
    fn ccns_phase_coding_policy_is_resolved_not_literal() {
        assert_eq!(
            CcnsPhaseCoding::parse_token("5stage"),
            Some(CcnsPhaseCoding::FiveStage)
        );
        assert_eq!(
            CcnsPhaseCoding::parse_token("  5 "),
            Some(CcnsPhaseCoding::FiveStage)
        );
        assert_eq!(
            CcnsPhaseCoding::parse_token("THREE"),
            Some(CcnsPhaseCoding::ThreeStage)
        );
        assert_eq!(
            CcnsPhaseCoding::parse_token("legacy"),
            Some(CcnsPhaseCoding::ThreeStage)
        );
        assert_eq!(CcnsPhaseCoding::parse_token("nonsense"), None);
        assert_eq!(CCNS_PHASE_CODING_DEFAULT, CcnsPhaseCoding::FiveStage);
        assert_eq!(
            CcnsPhaseCoding::FiveStage.max_code(),
            ccns_phase_code::COLD_RETURN
        );
        assert_eq!(
            CcnsPhaseCoding::ThreeStage.max_code(),
            ccns_phase_code::WARM_HOLD
        );
        assert!(CcnsPhaseCoding::FiveStage
            .descriptor()
            .starts_with(CcnsPhaseCoding::FiveStage.as_str()));
        assert!(CcnsPhaseCoding::ThreeStage
            .descriptor()
            .starts_with(CcnsPhaseCoding::ThreeStage.as_str()));

        // Every code the 5-stage policy admits must be reachable from `encode`
        // on the documented 5:10:5:6:4 hysteresis ratio.
        let (cold, ramp, warm, down) = (5, 10, 5, 6);
        let mut seen = std::collections::BTreeSet::new();
        for t in 0..30 {
            seen.insert(CcnsPhaseCoding::FiveStage.encode(t, cold, ramp, warm, down));
        }
        assert_eq!(seen.into_iter().collect::<Vec<u8>>(), vec![0, 1, 2, 3, 4]);

        // The legacy coding stays reachable and stays 3-valued.
        let mut legacy = std::collections::BTreeSet::new();
        for t in 0..30 {
            legacy.insert(CcnsPhaseCoding::ThreeStage.encode(t, cold, ramp, warm, down));
        }
        assert_eq!(legacy.into_iter().collect::<Vec<u8>>(), vec![0, 1, 2]);
    }

    /// The `ccns_phase` column is UInt8 under every version, so the file must
    /// carry its own domain declaration or a downstream reader cannot know
    /// whether 2 means "warm_hold" or "warm_hold + ramp_down + cold_return".
    #[test]
    fn schema_metadata_declares_the_ccns_phase_coding() {
        let s = build_spike_schema();
        let coding = s
            .metadata()
            .get(SCHEMA_META_CCNS_PHASE_CODING)
            .expect("the resolved ccns_phase coding must be stamped into the schema");
        assert!(
            coding.starts_with(CcnsPhaseCoding::resolve().as_str()),
            "stamped descriptor {coding:?} must begin with the resolved policy token"
        );
        assert!(coding.contains("cold_hold"));
        assert!(s.field_with_name("ccns_phase").is_ok());
    }

    #[test]
    fn chunk_idx_clamps_to_max() {
        assert_eq!(chunk_idx_for_timestep(0, 500, 110), 0);
        assert_eq!(chunk_idx_for_timestep(499, 500, 110), 0);
        assert_eq!(chunk_idx_for_timestep(500, 500, 110), 1);
        assert_eq!(chunk_idx_for_timestep(54999, 500, 110), 109); // would be 109
        assert_eq!(chunk_idx_for_timestep(99999, 500, 110), 109); // clamped
    }

    /// End-to-end RecordBatch build test with 3 synthetic spikes.
    /// Catches schema/column-length/type mismatches that the full smoke
    /// test would otherwise only find after 14 minutes of MD.
    #[test]
    fn record_batch_builds_from_synthetic_spikes() {
        let mk = |i: usize| GpuSpikeEvent {
            timestep: (i * 100) as i32,
            voxel_idx: i as i32,
            position: [i as f32, i as f32 * 2.0, i as f32 * 3.0],
            intensity: 50.0 + i as f32,
            nearby_residues: [i as i32, i as i32 + 1, i as i32 + 2, -1, -1, -1, -1, -1],
            n_residues: 3,
            spike_source: 1,
            wavelength_nm: 280.0,
            aromatic_type: 1,
            aromatic_residue_id: i as i32,
            water_density: 0.013,
            vibrational_energy: 0.001,
            n_nearby_excited: 2,
            wd_change: 0.0001,
            phase_bits: 100 + i as u32,
        };
        let spikes: Vec<GpuSpikeEvent> = (0..3).map(mk).collect();
        let cls: Vec<SpikeClassification> = (0..3)
            .map(|i| SpikeClassification {
                stream_id: (i % 8) as u8,
                group_id: ((i % 8) / 2) as u8,
                chunk_idx: i as u16,
                ccns_phase: 1,
                site_id: if i == 0 { 2 } else { -1 },
                nearest_site_id: 2,
                nearest_site_dist: i as f32,
                background_class: 0,
                burial_score: 0.5,
                intensity_percentile: (i * 30) as u8,
            })
            .collect();
        let batch = build_spike_record_batch(&spikes, &cls, 42)
            .expect("RecordBatch build should succeed for synthetic spikes");
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), SPIKE_ARROW_COLUMN_COUNT);
        // All columns should have length 3
        for (i, col) in batch.columns().iter().enumerate() {
            assert_eq!(col.len(), 3, "column {} length mismatch", i);
        }
        // The batch must carry the schema metadata, not just the field list —
        // otherwise the clock-domain caveat never reaches the written file.
        let batch_schema = batch.schema();
        let expected_version = SPIKE_ARROW_SCHEMA_VERSION.to_string();
        assert_eq!(
            batch_schema.metadata().get(SCHEMA_META_VERSION),
            Some(&expected_version)
        );
        assert!(batch_schema.field_with_name("frame_index").is_err());
        // `timestep` is exported raw and unscaled.
        let ts = batch
            .column_by_name("timestep")
            .expect("timestep column")
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("timestep is Int32");
        assert_eq!(ts.value(0), 0);
        assert_eq!(ts.value(1), 100);
        assert_eq!(ts.value(2), 200);
    }

    /// The clock-domain caveat must survive the Arrow IPC round trip: a
    /// downstream analyst reading the `.arrow` file must see the warning
    /// without access to this source tree.
    #[test]
    fn written_arrow_file_round_trips_schema_metadata() {
        use arrow::ipc::reader::FileReader;
        use std::fs::File;

        let spike = GpuSpikeEvent {
            timestep: 12_345,
            voxel_idx: 7,
            position: [1.0, 2.0, 3.0],
            intensity: 42.0,
            nearby_residues: [1, 2, -1, -1, -1, -1, -1, -1],
            n_residues: 2,
            spike_source: 4,
            wavelength_nm: 280.0,
            aromatic_type: -1,
            aromatic_residue_id: 11,
            water_density: 0.01,
            vibrational_energy: 0.002,
            n_nearby_excited: 1,
            wd_change: 0.0,
            phase_bits: 3,
        };
        let cls = SpikeClassification {
            stream_id: 0,
            group_id: 0,
            chunk_idx: 24,
            ccns_phase: 0,
            site_id: -1,
            nearest_site_id: 0,
            nearest_site_dist: 9.0,
            background_class: background_class::BULK_THERMAL,
            burial_score: 0.25,
            intensity_percentile: 10,
        };
        let batch = build_spike_record_batch(&[spike], &[cls], 7).expect("batch builds");

        let dir =
            std::env::temp_dir().join(format!("spike_arrow_meta_roundtrip_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let path = dir.join("spike_events.arrow");
        write_spike_arrow_file(&path, &batch).expect("arrow write");

        let reader = FileReader::try_new(File::open(&path).expect("reopen"), None)
            .expect("valid Arrow IPC file");
        let schema = reader.schema();
        assert_eq!(schema.fields().len(), SPIKE_ARROW_COLUMN_COUNT);
        assert!(
            schema.field_with_name("frame_index").is_err(),
            "the written file must not carry the fabricated frame_index column"
        );
        let expected_version = SPIKE_ARROW_SCHEMA_VERSION.to_string();
        assert_eq!(
            schema.metadata().get(SCHEMA_META_VERSION),
            Some(&expected_version),
            "schema version must survive the IPC round trip"
        );
        assert!(schema
            .metadata()
            .get(SCHEMA_META_CLOCK_DOMAIN)
            .is_some_and(|n| n.contains("MUST NOT be joined to coordinate frames")));
        assert!(schema
            .metadata()
            .get(SCHEMA_META_DERIVED_COLUMNS)
            .is_some_and(|n| n.contains("burial_score")));

        let _ = std::fs::remove_dir_all(&dir);
    }
    // Exact logical values across actual Arrow IPC containers. Floating-point
    // fields compare their bits, including signed zero and NaN payloads.
    fn bounded_fixture(n: usize) -> (Vec<GpuSpikeEvent>, Vec<SpikeClassification>) {
        let values = [
            0.0,
            -0.0,
            f32::from_bits(0x7fc0_1234),
            f32::from_bits(0xffc0_4321),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(1),
            f32::MAX,
        ];
        let mut spikes = Vec::with_capacity(n);
        let mut classes = Vec::with_capacity(n);
        for i in 0..n {
            let f = |j: usize| values[(i + j) % values.len()];
            spikes.push(GpuSpikeEvent {
                timestep: i as i32 * 499 - 1,
                voxel_idx: i as i32 * 17 - 3,
                position: [f(0), f(1), f(2)],
                intensity: f(3),
                nearby_residues: [i as i32, 2, -3, 4, 5, 6, 7, 8],
                n_residues: [-2, 0, 3, 8, 12][i % 5],
                spike_source: (i % 7) as i32,
                wavelength_nm: f(4),
                aromatic_type: (i % 6) as i32 - 1,
                aromatic_residue_id: i as i32 - 7,
                water_density: f(5),
                vibrational_energy: f(6),
                n_nearby_excited: [-1, 0, 2, 254, 999][i % 5],
                wd_change: f(7),
                phase_bits: 1u32.rotate_left(i as u32),
            });
            classes.push(SpikeClassification {
                stream_id: (i % 16) as u8,
                group_id: (i % 4) as u8,
                chunk_idx: (i * 331) as u16,
                ccns_phase: (i % 5) as u8,
                site_id: i as i32 - 2,
                nearest_site_id: 29 - i as i32,
                nearest_site_dist: f(1),
                background_class: (i % 5) as u8,
                burial_score: f(3),
                intensity_percentile: (i * 13 % 101) as u8,
            });
        }
        (spikes, classes)
    }

    fn assert_batch_rows_bit_exact(expected: &RecordBatch, actual: &RecordBatch, start: usize) {
        use arrow::array::{
            FixedSizeListArray, Float32Array, StringArray, UInt16Array, UInt32Array, UInt64Array,
            UInt8Array,
        };
        assert_eq!(actual.schema(), expected.schema(), "fields and metadata");
        assert_eq!(actual.num_columns(), expected.num_columns());
        assert!(start + actual.num_rows() <= expected.num_rows());
        for col in 0..expected.num_columns() {
            let a = expected.column(col);
            let b = actual.column(col);
            assert_eq!(a.data_type(), b.data_type());
            for row in 0..actual.num_rows() {
                assert_eq!(a.is_null(start + row), b.is_null(row));
                macro_rules! scalar {
                    ($ty:ty) => {{
                        let aa = a.as_any().downcast_ref::<$ty>().unwrap();
                        let bb = b.as_any().downcast_ref::<$ty>().unwrap();
                        assert_eq!(
                            aa.value(start + row),
                            bb.value(row),
                            "column {} row {}",
                            col,
                            start + row
                        );
                    }};
                }
                match a.data_type() {
                    DataType::UInt8 => scalar!(UInt8Array),
                    DataType::UInt16 => scalar!(UInt16Array),
                    DataType::UInt32 => scalar!(UInt32Array),
                    DataType::UInt64 => scalar!(UInt64Array),
                    DataType::Int32 => scalar!(Int32Array),
                    DataType::Utf8 => scalar!(StringArray),
                    DataType::Float32 => {
                        let aa = a.as_any().downcast_ref::<Float32Array>().unwrap();
                        let bb = b.as_any().downcast_ref::<Float32Array>().unwrap();
                        assert_eq!(
                            aa.value(start + row).to_bits(),
                            bb.value(row).to_bits(),
                            "float bits column {} row {}",
                            col,
                            start + row
                        );
                    }
                    DataType::FixedSizeList(_, _) => {
                        let aa = a
                            .as_any()
                            .downcast_ref::<FixedSizeListArray>()
                            .unwrap()
                            .value(start + row);
                        let bb = b
                            .as_any()
                            .downcast_ref::<FixedSizeListArray>()
                            .unwrap()
                            .value(row);
                        assert_eq!(
                            aa.as_any().downcast_ref::<Int32Array>().unwrap().values(),
                            bb.as_any().downcast_ref::<Int32Array>().unwrap().values()
                        );
                    }
                    t => panic!("Uncovered canonical column type: {:?}", t),
                }
            }
        }
    }

    #[test]
    fn bounded_file_multibatch_matches_monolithic_every_value_and_id() {
        use arrow::array::UInt64Array;
        use arrow::ipc::reader::FileReader;
        use std::fs::File;
        let (spikes, classes) = bounded_fixture(29);
        let expected = build_spike_record_batch(&spikes, &classes, u64::MAX).unwrap();
        let dir = std::env::temp_dir().join(format!("bounded_arrow_file_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        for batch_rows in [1, 3, 7, 29, 64] {
            let old_path = dir.join("old.arrow");
            write_spike_arrow_file(&old_path, &expected).unwrap();
            let legacy = FileReader::try_new(File::open(&old_path).unwrap(), None)
                .unwrap()
                .next()
                .unwrap()
                .unwrap();
            let new_path = dir.join(format!("new_{batch_rows}.arrow"));
            let receipt = write_spike_arrow_file_batched(
                &new_path,
                &spikes,
                &classes,
                u64::MAX,
                "run-bit-parity",
                batch_rows,
            )
            .unwrap();
            let reader = FileReader::try_new(File::open(&new_path).unwrap(), None).unwrap();
            assert_eq!(reader.schema(), legacy.schema());
            let mut start = 0;
            let mut batches = 0;
            let mut ids = Vec::new();
            for batch in reader {
                let batch = batch.unwrap();
                assert!(batch.num_rows() <= batch_rows);
                assert_batch_rows_bit_exact(&legacy, &batch, start);
                ids.extend_from_slice(
                    batch
                        .column_by_name("spike_id")
                        .unwrap()
                        .as_any()
                        .downcast_ref::<UInt64Array>()
                        .unwrap()
                        .values(),
                );
                start += batch.num_rows();
                batches += 1;
            }
            assert_eq!(
                ids,
                (0u64..29).collect::<Vec<_>>(),
                "no duplicate/omitted/reset IDs"
            );
            assert_eq!(start, spikes.len());
            assert_eq!(receipt.row_count, 29);
            assert_eq!(receipt.batch_count, batches);
            assert_eq!(batches, 29usize.div_ceil(batch_rows));
            assert_eq!(receipt.max_batch_rows, 29.min(batch_rows));
            assert_eq!(receipt.first_spike_id, Some(0));
            assert_eq!(receipt.last_spike_id, Some(28));
            assert_eq!(receipt.column_count, expected.num_columns());
            assert_eq!(receipt.durability, "file_and_parent_fsynced");
            let json = serde_json::to_value(&receipt).unwrap();
            assert_eq!(json["run_id"], "run-bit-parity");
            assert_eq!(json["row_count"], 29);
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn bounded_stream_multibatch_roundtrip_preserves_metadata_and_float_bits() {
        use arrow::ipc::reader::StreamReader;
        let (spikes, classes) = bounded_fixture(29);
        let expected = build_spike_record_batch(&spikes, &classes, 42).unwrap();
        let mut output = Vec::new();
        let receipt = write_spike_arrow_stream_batched(
            &mut output,
            &spikes,
            &classes,
            42,
            "stream-parity",
            4,
        )
        .unwrap();
        let reader = StreamReader::try_new(std::io::Cursor::new(output), None).unwrap();
        assert_eq!(reader.schema(), expected.schema());
        let mut start = 0;
        let mut batches = 0;
        for batch in reader {
            let batch = batch.unwrap();
            assert_batch_rows_bit_exact(&expected, &batch, start);
            start += batch.num_rows();
            batches += 1;
        }
        assert_eq!(start, 29);
        assert_eq!(batches, 8);
        assert_eq!(receipt.batch_count, 8);
        assert_eq!(receipt.format, "arrow_ipc_stream");
        assert_eq!(receipt.durability, "caller_managed_sink");
    }

    #[test]
    fn bounded_preflight_rejects_invalid_inputs_without_truncation() {
        let (spikes, classes) = bounded_fixture(3);
        let path =
            std::env::temp_dir().join(format!("bounded_arrow_preflight_{}", uuid::Uuid::new_v4()));
        std::fs::write(&path, b"retain-existing-output").unwrap();
        assert!(write_spike_arrow_file_batched(&path, &spikes, &classes, 1, "bad", 0).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"retain-existing-output");
        assert!(
            write_spike_arrow_file_batched(&path, &spikes, &classes[..2], 1, "bad", 2).is_err()
        );
        assert_eq!(std::fs::read(&path).unwrap(), b"retain-existing-output");
        let mut sink = Vec::new();
        assert!(
            write_spike_arrow_stream_batched(&mut sink, &spikes, &classes, 1, "bad", 0).is_err()
        );
        assert!(sink.is_empty());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn bounded_empty_containers_retain_schema_without_inventing_ids() {
        use arrow::ipc::reader::{FileReader, StreamReader};
        let path =
            std::env::temp_dir().join(format!("bounded_arrow_empty_{}", uuid::Uuid::new_v4()));
        let receipt = write_spike_arrow_file_batched(&path, &[], &[], 1, "empty", 8).unwrap();
        let mut file = FileReader::try_new(std::fs::File::open(&path).unwrap(), None).unwrap();
        assert_eq!(file.schema(), build_spike_schema());
        assert!(file.next().is_none());
        assert_eq!(receipt.row_count, 0);
        assert_eq!(receipt.batch_count, 0);
        assert_eq!(receipt.max_batch_rows, 0);
        assert_eq!(receipt.first_spike_id, None);
        assert_eq!(receipt.last_spike_id, None);
        let mut output = Vec::new();
        write_spike_arrow_stream_batched(&mut output, &[], &[], 1, "empty", 8).unwrap();
        let mut stream = StreamReader::try_new(std::io::Cursor::new(output), None).unwrap();
        assert_eq!(stream.schema(), build_spike_schema());
        assert!(stream.next().is_none());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn bounded_nonzero_offset_and_overflow_are_explicit() {
        let (spikes, classes) = bounded_fixture(3);
        let expected = build_spike_record_batch_at_offset(
            &spikes,
            &classes,
            1,
            u64::MAX - 2,
            build_spike_schema(),
        )
        .unwrap();
        let ids = expected
            .column_by_name("spike_id")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .unwrap();
        assert_eq!(
            ids.values().as_ref(),
            &[u64::MAX - 2, u64::MAX - 1, u64::MAX]
        );
        assert!(build_spike_record_batch_at_offset(
            &spikes,
            &classes,
            1,
            u64::MAX - 1,
            build_spike_schema()
        )
        .is_err());
    }

    #[test]
    fn bounded_sink_error_cannot_return_successful_receipt() {
        struct FailsAfterHeader {
            remaining: usize,
        }
        impl std::io::Write for FailsAfterHeader {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                if self.remaining == 0 {
                    return Err(std::io::Error::other("injected sink failure"));
                }
                let n = self.remaining.min(bytes.len());
                self.remaining -= n;
                Ok(n)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let (spikes, classes) = bounded_fixture(29);
        let mut header = Vec::new();
        {
            let _writer = StreamWriter::try_new(&mut header, &build_spike_schema()).unwrap();
        }
        // Size the fault from the actual Arrow schema write so it cannot
        // accidentally fail in the header instead of exercising a batch write.
        let mut sink = FailsAfterHeader {
            remaining: header.len() + 32,
        };
        let result =
            write_spike_arrow_stream_batched(&mut sink, &spikes, &classes, 1, "must-fail", 4);
        let error = result.unwrap_err();
        assert!(error.to_string().contains("Arrow batch 0"));
        assert_eq!(sink.remaining, 0);
    }
}
