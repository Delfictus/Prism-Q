//! Lossless observational Chronicle records in the existing CheckpointStore.
//! Payload bytes are never decoded or normalized. Dictionaries are per run;
//! leaves address records/elements/components without repeating scalar metadata.
//! CHRONICLE_HEAD is observational: scientific HEAD remains restart authority.
//! Leaf insertion order is storage order. Only each producer's native ordinal
//! asserts order; concurrent producers acquire no invented causal ordering.
//! The byte cap bounds one logical foliage snapshot, not cumulative CAS disk use.
//! Named acknowledgment receipts are attestations, not proof a consumer ran.
use super::*;

pub const CHRONICLE_SCHEMA: u32 = 1;
const MAX_LEAVES: usize = 100_000;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatumClass {
    Observation,
    RestartState,
    Derived,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComponentDescriptor {
    pub id: u32,
    pub name: String,
    pub byte_offset: u32,
    pub byte_width: u32,
    pub encoding: String,
    pub quantity_role: String,
    pub unit: String,
}
/// Bind exact per-record coordinates without duplicating context on each scalar.
/// These declarations must be populated by the native producer; endpoint clocks
/// alone never assert that an event's exact physical timestamp is available.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RecordClockBinding {
    Embedded {
        step_component: u32,
        physical_time_component: u32,
    },
    NativeStepSidecar {
        record_step_component: u32,
        sidecar_field: u32,
        sidecar_step_component: u32,
        sidecar_physical_time_component: u32,
    },
    Unavailable {
        reason: String,
    },
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum OriginalOrdinalBinding {
    NativeEmission,
    SelectionMap {
        map_field: u32,
        original_ordinal_component: u32,
        source_field: u32,
    },
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatumDescriptor {
    pub id: u32,
    pub owner: String,
    pub name: String,
    pub class: DatumClass,
    pub quantity_role: String,
    pub unit: String,
    pub encoding: String,
    pub element_bytes: u32,
    pub elements_per_record: u64,
    pub shape: Vec<u64>,
    /// Source-owned subject and flattening order; not inferred from row counts.
    pub subject: String,
    pub index_order: String,
    pub record_clock: RecordClockBinding,
    pub original_ordinal: OriginalOrdinalBinding,
    pub components: Vec<ComponentDescriptor>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatumContext {
    pub id: u32,
    pub stream_id: u32,
    pub replica_epoch: u32,
    pub replica_seed: u64,
    pub numerical_phase: String,
    pub thermal_phase: String,
    pub clock_domain: String,
    pub source_lineage_sha256: String,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConsumerContract {
    pub name: String,
    pub implementation_sha256: String,
    pub required_fields: BTreeSet<u32>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChronicleDictionary {
    pub fields: Vec<DatumDescriptor>,
    pub contexts: Vec<DatumContext>,
    pub consumers: Vec<ConsumerContract>,
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProducerKey {
    pub field_id: u32,
    pub stream_id: u32,
    pub replica_epoch: u32,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoverageRange {
    pub first_ordinal: u64,
    pub next_ordinal: u64,
    pub first_step: u64,
    pub last_step: u64,
    /// Native f64 clock bits, independently attached by the source owner.
    /// Endpoint clocks do not assert per-record clock resolution.
    pub first_physical_time_ps_bits: u64,
    pub last_physical_time_ps_bits: u64,
}
/// Exact source identity is separate from content identity: identical raw bytes
/// may belong to different streams, ordinal ranges, clocks or record subjects.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PayloadReference {
    pub producer: ProducerKey,
    pub context_id: u32,
    pub coverage: CoverageRange,
    pub object_sha256: String,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LeafLineage {
    pub input_foliage: Head,
    pub parent_payloads: Vec<PayloadReference>,
    pub transform_sha256: String,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PayloadLeaf {
    pub producer: ProducerKey,
    pub context_id: u32,
    pub lineage: Option<LeafLineage>,
    pub coverage: CoverageRange,
    pub byte_count: u64,
    pub object_sha256: String,
}
impl PayloadLeaf {
    pub fn reference(&self) -> PayloadReference {
        PayloadReference {
            producer: self.producer.clone(),
            context_id: self.context_id,
            coverage: self.coverage.clone(),
            object_sha256: self.object_sha256.clone(),
        }
    }
}
impl PayloadReference {
    fn matches(&self, leaf: &PayloadLeaf) -> bool {
        *self == leaf.reference()
    }
}
pub struct LeafInput {
    pub producer: ProducerKey,
    pub context_id: u32,
    pub lineage: Option<LeafLineage>,
    pub coverage: CoverageRange,
    pub bytes: Vec<u8>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProducerWatermark {
    pub producer: ProducerKey,
    pub next_ordinal: u64,
    pub last_step: u64,
    pub physical_time_ps_bits: u64,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConsumerWatermark {
    pub consumer: String,
    pub producer: ProducerKey,
    pub next_ordinal: u64,
    pub input_foliage: Head,
    pub receipt_sha256: String,
    pub receipt_bytes: u64,
    pub output_payloads: Vec<PayloadReference>,
}
pub struct ConsumerAck {
    pub consumer: String,
    pub producer: ProducerKey,
    pub next_ordinal: u64,
    /// The exact already-committed foliage the consumer read.
    pub input_foliage: Head,
    pub receipt_bytes: Vec<u8>,
    pub output_payloads: Vec<PayloadReference>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RestartAnchorReference {
    pub checkpoint_head: Head,
    pub closure_sha256: String,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Foliage {
    pub schema_version: u32,
    pub identity: CheckpointIdentity,
    pub generation: u64,
    pub previous_foliage: Option<Head>,
    pub dictionary: ChronicleDictionary,
    /// Ordered leaves. Each leaf covers a contiguous producer ordinal range.
    pub leaves: Vec<PayloadLeaf>,
    pub producers: Vec<ProducerWatermark>,
    pub consumers: Vec<ConsumerWatermark>,
    /// A separately validated full checkpoint, never inferred from observations.
    pub restart_anchor: Option<RestartAnchorReference>,
}
/// An in-process capability for one hash-validated immutable foliage. Private
/// fields prevent callers from injecting an unvalidated Foliage or leaf address.
/// This proves an immutable historical population, not freshness or restart closure.
/// The handle is intentionally not Serialize/Deserialize and exposes no mutable view.
pub struct ValidatedChronicleIndex {
    store_root: PathBuf,
    head: Head,
    foliage: Foliage,
    admitted_total_bytes: u64,
}
impl ValidatedChronicleIndex {
    pub fn head(&self) -> &Head { &self.head }
    pub fn foliage(&self) -> &Foliage { &self.foliage }
}
pub struct RestartAnchorRequest<'a> {
    pub head: &'a Head,
    pub closure: &'a ClosureContract,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DatumAddress {
    pub leaf_index: usize,
    pub record_ordinal: u64,
    pub element_index: u64,
    pub component_id: Option<u32>,
}

fn nonempty(value: &str) -> Result<()> {
    ensure!(
        !value.is_empty() && value.len() <= 4096,
        "empty or excessive Chronicle descriptor"
    );
    Ok(())
}
fn typed_width(encoding: &str, width: u32) -> Result<()> {
    let expected = match encoding {
        "u8" | "i8" => Some(1),
        "u16.le" | "i16.le" => Some(2),
        "u32.le" | "i32.le" | "f32.le" => Some(4),
        "u64.le" | "i64.le" | "f64.le" => Some(8),
        _ => None,
    };
    if let Some(n) = expected {
        ensure!(width == n, "typed encoding/width mismatch");
    } else {
        let digest = encoding
            .strip_prefix("opaque.sha256:")
            .context("unregistered typed encoding; use a hash-bound owner layout")?;
        validate_digest(digest)?;
    }
    Ok(())
}
fn physical(bits: u64) -> Result<f64> {
    let t = f64::from_bits(bits);
    ensure!(t.is_finite() && t >= 0.0, "invalid Chronicle native clock");
    Ok(t)
}
impl ChronicleDictionary {
    pub fn validate(&self, identity: &CheckpointIdentity) -> Result<()> {
        identity.validate()?;
        ensure!(
            !self.fields.is_empty() && self.fields.len() <= MAX_FIELDS,
            "invalid Chronicle dictionary size"
        );
        ensure!(
            !self.contexts.is_empty() && self.contexts.len() <= MAX_FIELDS,
            "invalid Chronicle context population"
        );
        let mut fields = BTreeSet::new();
        for f in &self.fields {
            ensure!(fields.insert(f.id), "duplicate Chronicle field ID");
            validate_label(&f.owner)?;
            validate_label(&f.name)?;
            for s in [&f.quantity_role, &f.unit, &f.encoding] {
                nonempty(s)?;
            }
            ensure!(
                f.element_bytes > 0 && f.elements_per_record > 0,
                "zero Chronicle record layout"
            );
            typed_width(&f.encoding, f.element_bytes)?;
            nonempty(&f.subject)?;
            nonempty(&f.index_order)?;
            ensure!(
                !f.shape.is_empty() && f.shape.len() <= 32,
                "missing/excessive record shape"
            );
            let shape = f.shape.iter().try_fold(1u64, |n, &d| {
                ensure!(d > 0, "zero shape dimension");
                n.checked_mul(d).context("shape overflow")
            })?;
            ensure!(
                shape == f.elements_per_record,
                "shape/element population mismatch"
            );
            u64::from(f.element_bytes)
                .checked_mul(f.elements_per_record)
                .context("Chronicle record width overflow")?;
            let mut ids = BTreeSet::new();
            let mut occupied = Vec::new();
            for c in &f.components {
                ensure!(ids.insert(c.id), "duplicate component ID");
                validate_label(&c.name)?;
                for s in [&c.encoding, &c.quantity_role, &c.unit] {
                    nonempty(s)?;
                }
                typed_width(&c.encoding, c.byte_width)?;
                let end = c
                    .byte_offset
                    .checked_add(c.byte_width)
                    .context("component offset overflow")?;
                ensure!(
                    c.byte_width > 0 && end <= f.element_bytes,
                    "component outside element"
                );
                ensure!(
                    occupied
                        .iter()
                        .all(|&(a, b)| end <= a || c.byte_offset >= b),
                    "overlapping components"
                );
                occupied.push((c.byte_offset, end));
            }
        }
        // Dictionaries are immutable for this first slice. Producers predeclare
        // every thermal/numerical context and its clock/selection-map fields.
        for f in &self.fields {
            let component = |field: u32, id: u32| -> Result<&ComponentDescriptor> {
                self.field(field)?
                    .components
                    .iter()
                    .find(|c| c.id == id)
                    .context("coordinate component missing")
            };
            let step_ok = |c: &ComponentDescriptor| -> Result<()> {
                ensure!(
                    matches!(c.encoding.as_str(), "u64.le" | "u32.le" | "i32.le"),
                    "step component must be an explicit integer"
                );
                Ok(())
            };
            let time_ok = |c: &ComponentDescriptor| -> Result<()> {
                ensure!(
                    c.encoding == "f64.le" && c.unit == "ps",
                    "physical clock component must contain native f64 ps"
                );
                Ok(())
            };
            match &f.record_clock {
                RecordClockBinding::Embedded {
                    step_component,
                    physical_time_component,
                } => {
                    step_ok(component(f.id, *step_component)?)?;
                    time_ok(component(f.id, *physical_time_component)?)?;
                }
                RecordClockBinding::NativeStepSidecar {
                    record_step_component,
                    sidecar_field,
                    sidecar_step_component,
                    sidecar_physical_time_component,
                } => {
                    step_ok(component(f.id, *record_step_component)?)?;
                    step_ok(component(*sidecar_field, *sidecar_step_component)?)?;
                    time_ok(component(*sidecar_field, *sidecar_physical_time_component)?)?;
                    ensure!(
                        *sidecar_field != f.id,
                        "clock sidecar aliases the source field"
                    );
                }
                RecordClockBinding::Unavailable { reason } => nonempty(reason)?,
            }
            if let OriginalOrdinalBinding::SelectionMap {
                map_field,
                original_ordinal_component,
                source_field,
            } = &f.original_ordinal
            {
                self.field(*source_field)?;
                let c = component(*map_field, *original_ordinal_component)?;
                ensure!(
                    c.encoding == "u64.le" && c.quantity_role == "source_record_ordinal",
                    "selection map must preserve original native ordinal"
                );
                ensure!(
                    *map_field != f.id && *source_field != f.id,
                    "selection lineage self-reference"
                );
            }
        }
        let mut contexts = BTreeSet::new();
        let mut replicas = BTreeMap::new();
        for c in &self.contexts {
            ensure!(
                contexts.insert(c.id) && c.stream_id < identity.stream_count,
                "duplicate or invalid context"
            );
            for s in [&c.numerical_phase, &c.thermal_phase, &c.clock_domain] {
                nonempty(s)?;
            }
            validate_digest(&c.source_lineage_sha256)?;
            if let Some(seed) = replicas.insert((c.stream_id, c.replica_epoch), c.replica_seed) {
                ensure!(
                    seed == c.replica_seed,
                    "replica seed changed across contexts"
                );
            }
        }
        let mut consumers = BTreeSet::new();
        for c in &self.consumers {
            validate_label(&c.name)?;
            validate_digest(&c.implementation_sha256)?;
            ensure!(
                consumers.insert(&c.name) && !c.required_fields.is_empty(),
                "duplicate or empty consumer contract"
            );
            ensure!(
                c.required_fields.is_subset(&fields),
                "consumer requires unknown datum field"
            );
        }
        Ok(())
    }
    fn field(&self, id: u32) -> Result<&DatumDescriptor> {
        self.fields
            .iter()
            .find(|f| f.id == id)
            .context("unknown Chronicle field")
    }
    fn context(&self, id: u32) -> Result<&DatumContext> {
        self.contexts
            .iter()
            .find(|c| c.id == id)
            .context("unknown Chronicle context")
    }
}
impl Foliage {
    fn validate(&self, limit: u64) -> Result<()> {
        ensure!(
            self.schema_version == CHRONICLE_SCHEMA && self.generation > 0,
            "invalid Chronicle schema/generation"
        );
        self.dictionary.validate(&self.identity)?;
        if let Some(previous) = &self.previous_foliage {
            validate_digest(&previous.manifest_sha256)?;
            ensure!(
                previous.generation.checked_add(1) == Some(self.generation),
                "foliage chain generation mismatch"
            );
        } else {
            ensure!(self.generation == 1, "foliage ancestry omitted");
        }
        if let Some(a) = &self.restart_anchor {
            validate_digest(&a.checkpoint_head.manifest_sha256)?;
            validate_digest(&a.closure_sha256)?;
            ensure!(
                a.checkpoint_head.generation > 0,
                "invalid restart reference"
            );
        }
        ensure!(
            self.leaves.len() <= MAX_LEAVES,
            "Chronicle leaf population exceeds admission"
        );
        let mut wm: BTreeMap<ProducerKey, ProducerWatermark> = BTreeMap::new();
        let mut total = 0u64;
        for (leaf_index, leaf) in self.leaves.iter().enumerate() {
            let f = self.dictionary.field(leaf.producer.field_id)?;
            let c = self.dictionary.context(leaf.context_id)?;
            ensure!(
                leaf.producer.stream_id == c.stream_id
                    && leaf.producer.replica_epoch == c.replica_epoch,
                "leaf context/producer mismatch"
            );
            validate_digest(&leaf.object_sha256)?;
            if let Some(lineage) = &leaf.lineage {
                validate_digest(&lineage.input_foliage.manifest_sha256)?;
                validate_digest(&lineage.transform_sha256)?;
                ensure!(
                    lineage.input_foliage.generation > 0
                        && lineage.input_foliage.generation < self.generation,
                    "derived input was not committed before transform"
                );
                ensure!(
                    !lineage.parent_payloads.is_empty(),
                    "derived parents omitted"
                );
                let mut parents = BTreeSet::new();
                for parent in &lineage.parent_payloads {
                    validate_digest(&parent.object_sha256)?;
                    ensure!(
                        parents.insert(serde_json::to_vec(parent)?),
                        "duplicate lineage parent"
                    );
                    ensure!(
                        self.leaves[..leaf_index].iter().any(|l| parent.matches(l)),
                        "lineage parent must precede derived leaf"
                    );
                }
            }
            if f.class == DatumClass::Derived {
                ensure!(
                    leaf.lineage.is_some(),
                    "derived datum lacks input/parent/transform lineage"
                );
            }
            let r = &leaf.coverage;
            ensure!(
                r.next_ordinal > r.first_ordinal && r.first_step <= r.last_step,
                "empty or reversed leaf coverage"
            );
            let first = physical(r.first_physical_time_ps_bits)?;
            let last = physical(r.last_physical_time_ps_bits)?;
            ensure!(last >= first, "leaf physical clock reversed");
            let width = u64::from(f.element_bytes)
                .checked_mul(f.elements_per_record)
                .context("record width overflow")?;
            let size = (r.next_ordinal - r.first_ordinal)
                .checked_mul(width)
                .context("leaf size overflow")?;
            ensure!(
                size == leaf.byte_count,
                "leaf payload does not match typed record population"
            );
            total = total
                .checked_add(size)
                .context("Chronicle payload budget overflow")?;
            ensure!(total <= limit, "Chronicle payload exceeds byte budget");
            if let Some(p) = wm.get(&leaf.producer) {
                ensure!(
                    r.first_ordinal == p.next_ordinal,
                    "producer omission or duplicate"
                );
                if f.class != DatumClass::Derived {
                    ensure!(
                        r.first_step >= p.last_step && first >= physical(p.physical_time_ps_bits)?,
                        "raw producer chronology regression"
                    );
                }
                // A derived selection/permutation has its own ordered row index;
                // original mixed-channel ordinals and exact clocks stay in its map.
            } else {
                ensure!(r.first_ordinal == 0, "producer initial coverage omitted");
            }
            wm.insert(
                leaf.producer.clone(),
                ProducerWatermark {
                    producer: leaf.producer.clone(),
                    next_ordinal: r.next_ordinal,
                    last_step: r.last_step,
                    physical_time_ps_bits: r.last_physical_time_ps_bits,
                },
            );
        }
        ensure!(
            self.producers == wm.into_values().collect::<Vec<_>>(),
            "producer watermarks disagree with ordered foliage"
        );
        let mut seen = BTreeSet::new();
        for a in &self.consumers {
            ensure!(
                seen.insert((a.consumer.clone(), a.producer.clone())),
                "duplicate consumer watermark"
            );
            let contract = self
                .dictionary
                .consumers
                .iter()
                .find(|c| c.name == a.consumer)
                .context("unregistered consumer")?;
            ensure!(
                contract.required_fields.contains(&a.producer.field_id),
                "consumer acknowledged unbound field"
            );
            let p = self
                .producers
                .iter()
                .find(|p| p.producer == a.producer)
                .context("consumer acknowledged unknown producer")?;
            ensure!(
                a.next_ordinal > 0 && a.next_ordinal <= p.next_ordinal,
                "consumer acknowledged unproduced records"
            );
            validate_digest(&a.input_foliage.manifest_sha256)?;
            validate_digest(&a.receipt_sha256)?;
            ensure!(
                a.receipt_bytes > 0 && a.receipt_bytes <= MAX_MANIFEST_BYTES,
                "invalid acknowledgment receipt size"
            );
            total = total
                .checked_add(a.receipt_bytes)
                .context("Chronicle receipt budget overflow")?;
            ensure!(
                total <= limit,
                "Chronicle snapshot payload/receipt budget exceeded"
            );
            ensure!(
                a.input_foliage.generation < self.generation,
                "consumer acknowledged uncommitted foliage"
            );
            for digest in &a.output_payloads {
                validate_digest(&digest.object_sha256)?;
                ensure!(
                    self.leaves.iter().any(|l| digest.matches(l)),
                    "acknowledgment output leaf omitted"
                );
            }
        }
        Ok(())
    }
    /// Consumer closure only: this never authorizes deletion, compaction or restart-state release.
    pub fn required_consumers_acknowledged(&self, leaf_index: usize) -> Result<bool> {
        let leaf = self.leaves.get(leaf_index).context("unknown leaf index")?;
        let contracts: Vec<_> = self
            .dictionary
            .consumers
            .iter()
            .filter(|c| c.required_fields.contains(&leaf.producer.field_id))
            .collect();
        ensure!(
            !contracts.is_empty(),
            "no declared consumer closure; cannot infer retention eligibility"
        );
        Ok(contracts.iter().all(|c| {
            self.consumers.iter().any(|a| {
                a.consumer == c.name
                    && a.producer == leaf.producer
                    && a.next_ordinal >= leaf.coverage.next_ordinal
            })
        }))
    }
}

impl CheckpointStore {
    pub fn chronicle_head(&self) -> Result<Option<Head>> {
        self.read_head_named("CHRONICLE_HEAD")
    }
    pub(super) fn read_foliage(&self, head: &Head) -> Result<Foliage> {
        validate_digest(&head.manifest_sha256)?;
        let bytes = read_bounded(
            &self.root.join("manifests").join(&head.manifest_sha256),
            MAX_MANIFEST_BYTES,
        )?;
        ensure!(
            hash(&bytes) == head.manifest_sha256,
            "Chronicle foliage hash mismatch"
        );
        let f: Foliage = serde_json::from_slice(&bytes)?;
        ensure!(
            f.generation == head.generation,
            "Chronicle generation mismatch"
        );
        f.validate(self.max_total_bytes)?;
        Ok(f)
    }
    fn verify_foliage_objects(&self, f: &Foliage) -> Result<()> {
        for leaf in &f.leaves {
            let b = read_bounded(
                &self.root.join("objects").join(&leaf.object_sha256),
                leaf.byte_count,
            )?;
            ensure!(
                b.len() as u64 == leaf.byte_count && hash(&b) == leaf.object_sha256,
                "corrupt or truncated Chronicle payload"
            );
        }
        if let Some(a) = &f.restart_anchor {
            let manifest = self.read_manifest(&a.checkpoint_head)?;
            ensure!(
                manifest.identity == f.identity
                    && manifest.closure_contract_sha256 == a.closure_sha256,
                "restart reference identity/closure mismatch"
            );
            // Full closure/object admission uses the existing restore decoder,
            // at anchor commit and on explicit scientific restore, not this view.
        }
        for ack in &f.consumers {
            let b = read_bounded(
                &self.root.join("objects").join(&ack.receipt_sha256),
                ack.receipt_bytes,
            )?;
            ensure!(
                b.len() as u64 == ack.receipt_bytes && hash(&b) == ack.receipt_sha256,
                "corrupt consumer acknowledgment receipt"
            );
        }
        Ok(())
    }
    /// Hash-verified metadata only; payload integrity is checked by the explicit
    /// full inspection or individual leaf read. Appending never rereads all raw history.
    pub fn inspect_chronicle_index(
        &self,
        expected: &Head,
        identity: &CheckpointIdentity,
    ) -> Result<Foliage> {
        let _lock = self.lock()?;
        ensure!(
            self.chronicle_head()?.as_ref() == Some(expected),
            "stale Chronicle index inspection"
        );
        let f = self.read_foliage(expected)?;
        ensure!(
            f.identity == *identity,
            "Chronicle execution identity mismatch"
        );
        Ok(f)
    }
    /// Validate the current durable index once. Later payload reads through this
    /// capability may read its immutable prefix after CHRONICLE_HEAD advances.
    /// They make no statement about the new head or a scientific restart anchor.
    pub fn validate_chronicle_index(
        &self,
        expected: &Head,
        identity: &CheckpointIdentity,
    ) -> Result<ValidatedChronicleIndex> {
        let foliage = self.inspect_chronicle_index(expected, identity)?;
        self.validated_chronicle_index_from_foliage(expected,foliage)
    }
    /// Called only through the exact PausedGeneration all-owner authority.
    /// This is a metadata read, never a mutation or a general lock bypass.
    pub(crate) fn validate_chronicle_index_with_paused_writer(&self,receiving:&CheckpointStore,
        expected:&Head,identity:&CheckpointIdentity)->Result<ValidatedChronicleIndex> {
        use std::os::unix::fs::MetadataExt;
        ensure!(receiving.root==self.root,"paused Chronicle read root differs");
        let _writer=self.scientific_staging_guard()?;
        let source_root=fs::symlink_metadata(&self.root)?;let target_root=fs::symlink_metadata(&receiving.root)?;
        ensure!(source_root.is_dir() && !source_root.file_type().is_symlink()
            && target_root.is_dir() && !target_root.file_type().is_symlink()
            && source_root.dev()==target_root.dev() && source_root.ino()==target_root.ino(),
            "paused Chronicle read directory identity differs");
        let held=match &_writer {
            super::ScientificStagingGuard::Shared(lease)=>lease._writer.0.metadata()?,
            super::ScientificStagingGuard::Exclusive(lock)=>lock.0.metadata()?,
        };
        let named=fs::symlink_metadata(receiving.root.join("LOCK"))?;
        ensure!(held.is_file() && named.is_file() && !named.file_type().is_symlink()
            && held.dev()==named.dev() && held.ino()==named.ino(),"paused Chronicle held writer inode differs");
        ensure!(receiving.chronicle_head()?.as_ref()==Some(expected),"stale paused Chronicle index inspection");
        let foliage=receiving.read_foliage(expected)?;
        ensure!(foliage.identity==*identity,"paused Chronicle execution identity mismatch");
        receiving.validated_chronicle_index_from_foliage(expected,foliage)
    }
    fn validated_chronicle_index_from_foliage(&self,expected:&Head,foliage:Foliage)->Result<ValidatedChronicleIndex> {
        // Match Foliage::validate's cumulative logical payload + acknowledgment
        // receipt admission; preserve that bound when a same-root store handle
        // is reopened with a different budget. This is not cumulative CAS usage.
        let admitted_total_bytes = foliage.leaves.iter().map(|l| l.byte_count)
            .chain(foliage.consumers.iter().map(|a| a.receipt_bytes))
            .try_fold(0u64, |n, bytes| n.checked_add(bytes)
                .context("Chronicle index admission byte overflow"))?;
        ensure!(admitted_total_bytes <= self.max_total_bytes,
            "Chronicle index exceeds store admission budget");
        Ok(ValidatedChronicleIndex {
            store_root: self.root.clone(),
            head: expected.clone(),
            foliage,
            admitted_total_bytes,
        })
    }
    /// Read an exact member of a previously validated immutable index without
    /// reparsing the cumulative foliage. Every requested raw object is still
    /// byte-count checked and SHA-256 verified on every read. A handle from a
    /// different store is refused even if its identity and payload bytes match.
    pub fn read_indexed_chronicle_payload(
        &self,
        index: &ValidatedChronicleIndex,
        leaf_index: usize,
    ) -> Result<Vec<u8>> {
        ensure!(index.store_root == self.root, "Chronicle index belongs to another store");
        ensure!(index.admitted_total_bytes <= self.max_total_bytes,
            "Chronicle index exceeds receiving store admission budget");
        let leaf = index.foliage.leaves.get(leaf_index).context("unknown Chronicle index leaf")?;
        let bytes = read_bounded(
            &self.root.join("objects").join(&leaf.object_sha256), leaf.byte_count,
        )?;
        ensure!(bytes.len() as u64 == leaf.byte_count && hash(&bytes) == leaf.object_sha256,
            "corrupt indexed Chronicle payload");
        Ok(bytes)
    }
    pub fn read_chronicle_payload(
        &self,
        expected: &Head,
        identity: &CheckpointIdentity,
        leaf_index: usize,
    ) -> Result<Vec<u8>> {
        let _lock = self.lock()?;
        ensure!(
            self.chronicle_head()?.as_ref() == Some(expected),
            "stale Chronicle payload request"
        );
        let f = self.read_foliage(expected)?;
        ensure!(
            f.identity == *identity,
            "Chronicle execution identity mismatch"
        );
        let leaf = f.leaves.get(leaf_index).context("unknown Chronicle leaf")?;
        let b = read_bounded(
            &self.root.join("objects").join(&leaf.object_sha256),
            leaf.byte_count,
        )?;
        ensure!(
            b.len() as u64 == leaf.byte_count && hash(&b) == leaf.object_sha256,
            "corrupt Chronicle payload"
        );
        Ok(b)
    }
    pub fn inspect_chronicle(
        &self,
        expected: &Head,
        identity: &CheckpointIdentity,
    ) -> Result<Foliage> {
        let _lock = self.lock()?;
        ensure!(
            self.chronicle_head()?.as_ref() == Some(expected),
            "stale Chronicle inspection"
        );
        let f = self.read_foliage(expected)?;
        ensure!(
            f.identity == *identity,
            "Chronicle execution identity mismatch"
        );
        self.verify_foliage_objects(&f)?;
        Ok(f)
    }
    pub fn commit_chronicle(
        &self,
        expected: Option<&Head>,
        identity: &CheckpointIdentity,
        dictionary: &ChronicleDictionary,
        leaves: &[LeafInput],
        acks: &[ConsumerAck],
        anchor: Option<RestartAnchorRequest<'_>>,
    ) -> Result<Head> {
        self.commit_chronicle_with_hook(
            expected,
            identity,
            dictionary,
            leaves,
            acks,
            anchor,
            |_| Ok(()),
        )
    }
    pub(crate) fn commit_chronicle_with_hook(
        &self,
        expected: Option<&Head>,
        identity: &CheckpointIdentity,
        dictionary: &ChronicleDictionary,
        leaves: &[LeafInput],
        acks: &[ConsumerAck],
        anchor: Option<RestartAnchorRequest<'_>>,
        hook: impl FnMut(&str) -> Result<()>,
    ) -> Result<Head> {
        self.commit_chronicle_streaming_with_hook(
            expected, identity, dictionary, self.max_total_bytes, acks, anchor,
            |append| { for input in leaves { append(input)?; } Ok(()) }, hook,
        )
    }
    /// Write bounded leaf objects incrementally under the existing store LOCK,
    /// then publish one cumulative foliage and one CHRONICLE_HEAD transition.
    /// `produce` owns each slab; append borrows it only for the call and never
    /// retains its bytes. The supplied slab may be dropped/reused after append.
    /// Before HEAD publication, callback/admission/write failure leaves the
    /// previous HEAD readable; staged unreachable CAS objects are not a commit.
    /// Publication errors after the atomic HEAD rename may return Err with a
    /// complete new HEAD visible. Callers must reread and reconcile HEAD before
    /// retrying or advancing source counters; Err never implies an unchanged HEAD.
    /// No callback may recursively acquire this store lock. Producer counters or
    /// source-buffer resets must advance only after this method returns success.
    pub fn commit_chronicle_streaming(
        &self,
        expected: Option<&Head>,
        identity: &CheckpointIdentity,
        dictionary: &ChronicleDictionary,
        max_leaf_bytes: u64,
        acks: &[ConsumerAck],
        anchor: Option<RestartAnchorRequest<'_>>,
        produce: impl FnOnce(&mut dyn FnMut(&LeafInput) -> Result<()>) -> Result<()>,
    ) -> Result<Head> {
        self.commit_chronicle_streaming_with_hook(
            expected, identity, dictionary, max_leaf_bytes, acks, anchor, produce, |_| Ok(()),
        )
    }
    pub(crate) fn commit_chronicle_streaming_with_hook(
        &self,
        expected: Option<&Head>,
        identity: &CheckpointIdentity,
        dictionary: &ChronicleDictionary,
        max_leaf_bytes: u64,
        acks: &[ConsumerAck],
        anchor: Option<RestartAnchorRequest<'_>>,
        produce: impl FnOnce(&mut dyn FnMut(&LeafInput) -> Result<()>) -> Result<()>,
        mut hook: impl FnMut(&str) -> Result<()>,
    ) -> Result<Head> {
        dictionary.validate(identity)?;
        ensure!(max_leaf_bytes > 0 && max_leaf_bytes <= self.max_total_bytes,
            "invalid Chronicle streaming leaf byte limit");
        let lock = self.lock()?;
        let current = self.chronicle_head()?;
        ensure!(
            current.as_ref() == expected,
            "stale Chronicle HEAD; revalidate producer/consumer state"
        );
        let prior = current.as_ref().map(|h| self.read_foliage(h)).transpose()?;
        if let Some(p) = &prior {
            ensure!(
                p.identity == *identity && p.dictionary == *dictionary,
                "Chronicle identity/dictionary changed"
            );
        }
        let generation = current.as_ref().map_or(Ok(1), |h| {
            h.generation
                .checked_add(1)
                .context("Chronicle generation overflow")
        })?;
        let mut f = Foliage {
            schema_version: CHRONICLE_SCHEMA,
            identity: identity.clone(),
            generation,
            previous_foliage: current.clone(),
            dictionary: dictionary.clone(),
            leaves: prior.as_ref().map(|p| p.leaves.clone()).unwrap_or_default(),
            producers: Vec::new(),
            consumers: prior
                .as_ref()
                .map(|p| p.consumers.clone())
                .unwrap_or_default(),
            restart_anchor: prior.as_ref().and_then(|p| p.restart_anchor.clone()),
        };
        let prior_leaf_count = f.leaves.len();
        let mut admitted_bytes = f.leaves.iter().try_fold(0u64, |n, l|
            n.checked_add(l.byte_count).context("Chronicle cumulative byte overflow"))?;
        let mut first_append_error: Option<String> = None;
        let mut append = |input: &LeafInput| -> Result<()> {
            // A producer cannot recover a failed append by ignoring its result
            // or retrying; only a new transaction may retry after reconciliation.
            ensure!(first_append_error.is_none(), "Chronicle append already failed");
            let result: Result<()> = (|| {
            ensure!(!input.bytes.is_empty() && input.bytes.len() as u64 <= max_leaf_bytes,
                "Chronicle leaf exceeds streaming slab limit");
            ensure!(f.leaves.len() < MAX_LEAVES, "too many Chronicle leaves");
            admitted_bytes = admitted_bytes.checked_add(input.bytes.len() as u64)
                .context("Chronicle cumulative byte overflow")?;
            ensure!(admitted_bytes <= self.max_total_bytes, "Chronicle payload budget exceeded");
            if let Some(lineage) = &input.lineage {
                ensure!(
                    current.as_ref() == Some(&lineage.input_foliage),
                    "derived input foliage is not the committed input head"
                );
                let prior = prior
                    .as_ref()
                    .context("derived transform has no committed input")?;
                for parent in &lineage.parent_payloads {
                    ensure!(
                        prior.leaves.iter().any(|l| parent.matches(l)),
                        "derived parent not present in committed input"
                    );
                }
            }
            let descriptor = PayloadLeaf {
                producer: input.producer.clone(),
                context_id: input.context_id,
                lineage: input.lineage.clone(),
                coverage: input.coverage.clone(),
                byte_count: input.bytes.len() as u64,
                object_sha256: hash(&input.bytes),
            };
            // Immutable objects can be staged before final population validation.
            // Only the final validated foliage/HEAD exposes them to consumers.
            self.write_immutable("objects", &hash(&input.bytes), &input.bytes)?;
            hook("leaf_object_staged")?;
            f.leaves.push(descriptor);
            Ok(())
            })();
            if let Err(error) = &result {
                first_append_error = Some(format!("{error:#}"));
            }
            result
        };
        let producer_result = produce(&mut append);
        drop(append);
        // Check the latched error even when the callback swallowed the result,
        // retried, or returned its own error. No failed-source metadata publishes.
        ensure!(first_append_error.is_none(), "Chronicle streaming append failed: {}",
            first_append_error.as_deref().unwrap_or("unknown append error"));
        producer_result?;
        ensure!(f.leaves.len() > prior_leaf_count || !acks.is_empty() || anchor.is_some(),
            "empty Chronicle transition");
        let mut wm = BTreeMap::new();
        for leaf in &f.leaves {
            wm.insert(
                leaf.producer.clone(),
                ProducerWatermark {
                    producer: leaf.producer.clone(),
                    next_ordinal: leaf.coverage.next_ordinal,
                    last_step: leaf.coverage.last_step,
                    physical_time_ps_bits: leaf.coverage.last_physical_time_ps_bits,
                },
            );
        }
        f.producers = wm.into_values().collect();
        let mut seen = BTreeSet::new();
        for ack in acks {
            ensure!(
                seen.insert((ack.consumer.clone(), ack.producer.clone())),
                "duplicate acknowledgment transition"
            );
            ensure!(
                current.as_ref() == Some(&ack.input_foliage),
                "consumer input is not current committed foliage"
            );
            let p = prior
                .as_ref()
                .and_then(|p| p.producers.iter().find(|p| p.producer == ack.producer))
                .context("consumer input producer was not committed")?;
            ensure!(
                ack.next_ordinal > 0 && ack.next_ordinal <= p.next_ordinal,
                "consumer input is not committed"
            );
            ensure!(
                !ack.receipt_bytes.is_empty()
                    && ack.receipt_bytes.len() as u64 <= MAX_MANIFEST_BYTES,
                "missing/excessive consumer receipt"
            );
            for d in &ack.output_payloads {
                ensure!(
                    f.leaves.iter().any(|l| d.matches(l)),
                    "consumer output has no committed payload leaf"
                );
            }
            if let Some(old) = f
                .consumers
                .iter_mut()
                .find(|a| a.consumer == ack.consumer && a.producer == ack.producer)
            {
                ensure!(
                    ack.next_ordinal > old.next_ordinal,
                    "consumer watermark did not advance"
                );
                *old = ConsumerWatermark {
                    consumer: ack.consumer.clone(),
                    producer: ack.producer.clone(),
                    next_ordinal: ack.next_ordinal,
                    input_foliage: ack.input_foliage.clone(),
                    receipt_sha256: hash(&ack.receipt_bytes),
                    receipt_bytes: ack.receipt_bytes.len() as u64,
                    output_payloads: ack.output_payloads.clone(),
                };
            } else {
                f.consumers.push(ConsumerWatermark {
                    consumer: ack.consumer.clone(),
                    producer: ack.producer.clone(),
                    next_ordinal: ack.next_ordinal,
                    input_foliage: ack.input_foliage.clone(),
                    receipt_sha256: hash(&ack.receipt_bytes),
                    receipt_bytes: ack.receipt_bytes.len() as u64,
                    output_payloads: ack.output_payloads.clone(),
                });
            }
        }
        if let Some(a) = anchor {
            self.restore_under_writer_lock(&lock, a.head, identity, a.closure)?;
            f.restart_anchor = Some(RestartAnchorReference {
                checkpoint_head: a.head.clone(),
                closure_sha256: a.closure.digest()?,
            });
        }
        f.validate(self.max_total_bytes)?;
        let bytes = serde_json::to_vec(&f)?;
        ensure!(
            bytes.len() as u64 <= MAX_MANIFEST_BYTES,
            "Chronicle foliage metadata exceeds limit"
        );
        // All final population/metadata admission precedes durable visibility.
        // Incrementally staged objects remain unreachable if admission fails.
        for ack in acks {
            self.write_immutable("objects", &hash(&ack.receipt_bytes), &ack.receipt_bytes)?;
        }
        sync_dir(&self.root.join("objects"))?;
        hook("objects_synced")?;
        let digest = hash(&bytes);
        self.write_immutable("manifests", &digest, &bytes)?;
        sync_dir(&self.root.join("manifests"))?;
        hook("manifest_synced")?;
        let head = Head {
            generation,
            manifest_sha256: digest,
        };
        self.publish_head_named("CHRONICLE_HEAD", &head, &mut hook)?;
        Ok(head)
    }
    /// Observational leaves alone cannot authorize scientific restart.
    pub fn restore_chronicle_anchor(
        &self,
        expected: &Head,
        identity: &CheckpointIdentity,
        closure: &ClosureContract,
    ) -> Result<RestoredCheckpoint> {
        let lock = self.lock()?;
        ensure!(
            self.chronicle_head()?.as_ref() == Some(expected),
            "stale Chronicle restart request"
        );
        let f = self.read_foliage(expected)?;
        ensure!(
            f.identity == *identity,
            "Chronicle restart identity mismatch"
        );
        self.verify_foliage_objects(&f)?;
        let anchor = f
            .restart_anchor
            .context("observational leaves are not a complete scientific restart anchor")?;
        ensure!(
            anchor.closure_sha256 == closure.digest()?,
            "Chronicle restart closure mismatch"
        );
        self.restore_under_writer_lock(&lock, &anchor.checkpoint_head, identity, closure)
    }
    /// Reads/hash-checks one complete leaf before slicing the requested datum.
    /// Leaf size is admitted by the store byte cap; this is not a constant-memory
    /// indexed-file reader. Native producers must choose bounded leaf sizes.
    pub fn read_chronicle_datum(
        &self,
        expected: &Head,
        identity: &CheckpointIdentity,
        address: &DatumAddress,
    ) -> Result<Vec<u8>> {
        let _lock = self.lock()?;
        ensure!(
            self.chronicle_head()?.as_ref() == Some(expected),
            "stale Chronicle datum request"
        );
        let f = self.read_foliage(expected)?;
        ensure!(f.identity == *identity, "Chronicle datum identity mismatch");
        let l = f
            .leaves
            .get(address.leaf_index)
            .context("unknown Chronicle leaf")?;
        let d = f.dictionary.field(l.producer.field_id)?;
        ensure!(
            address.record_ordinal >= l.coverage.first_ordinal
                && address.record_ordinal < l.coverage.next_ordinal,
            "record outside leaf"
        );
        ensure!(
            address.element_index < d.elements_per_record,
            "element outside record"
        );
        let element = (address.record_ordinal - l.coverage.first_ordinal)
            .checked_mul(d.elements_per_record)
            .and_then(|x| x.checked_add(address.element_index))
            .context("datum address overflow")?;
        let base = element
            .checked_mul(u64::from(d.element_bytes))
            .context("datum byte offset overflow")?;
        let (offset, width) = if let Some(id) = address.component_id {
            let c = d
                .components
                .iter()
                .find(|c| c.id == id)
                .context("unknown component")?;
            (
                base.checked_add(u64::from(c.byte_offset))
                    .context("component address overflow")?,
                u64::from(c.byte_width),
            )
        } else {
            (base, u64::from(d.element_bytes))
        };
        let bytes = read_bounded(
            &self.root.join("objects").join(&l.object_sha256),
            l.byte_count,
        )?;
        ensure!(
            bytes.len() as u64 == l.byte_count && hash(&bytes) == l.object_sha256,
            "Chronicle leaf corrupted"
        );
        let end = offset.checked_add(width).context("datum end overflow")?;
        ensure!(end <= l.byte_count, "datum outside payload");
        Ok(bytes[usize::try_from(offset)?..usize::try_from(end)?].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::bail; // CPU-only tests must not depend on the parent's gpu import.
    fn identity() -> CheckpointIdentity {
        CheckpointIdentity {
            run_id: "chronicle-native-test".into(),
            topology_sha256: "1".repeat(64),
            engine_sha256: "2".repeat(64),
            kernel_set_sha256: "3".repeat(64),
            protocol_sha256: "4".repeat(64),
            execution_request_sha256: "5".repeat(64),
            state_abi_sha256: "6".repeat(64),
            provider: "native-store-test".into(),
            stream_count: 1,
        }
    }
    fn dictionary() -> ChronicleDictionary {
        ChronicleDictionary {
            fields: vec![DatumDescriptor {
                id: 1,
                owner: "stream_0/observer".into(),
                name: "raw_f64".into(),
                class: DatumClass::Observation,
                quantity_role: "observer_value".into(),
                unit: "1".into(),
                encoding: "f64.le".into(),
                element_bytes: 8,
                elements_per_record: 1,
                shape: vec![1],
                subject: "source-native-record".into(),
                index_order: "original-per-stream-emission-ordinal".into(),
                record_clock: RecordClockBinding::Unavailable {
                    reason: "fixture tests exact bytes, not per-record time".into(),
                },
                original_ordinal: OriginalOrdinalBinding::NativeEmission,
                components: vec![ComponentDescriptor {
                    id: 0,
                    name: "value".into(),
                    byte_offset: 0,
                    byte_width: 8,
                    encoding: "f64.le".into(),
                    quantity_role: "observer_value".into(),
                    unit: "1".into(),
                }],
            }],
            contexts: vec![DatumContext {
                id: 1,
                stream_id: 0,
                replica_epoch: 0,
                replica_seed: 42,
                numerical_phase: "completed_step".into(),
                thermal_phase: "cold_hold".into(),
                clock_domain: "native-physical-ps".into(),
                source_lineage_sha256: "7".repeat(64),
            }],
            consumers: vec![ConsumerContract {
                name: "path_b".into(),
                implementation_sha256: "8".repeat(64),
                required_fields: BTreeSet::from([1]),
            }],
        }
    }
    fn leaf(first: u64, words: &[u64]) -> LeafInput {
        LeafInput {
            producer: ProducerKey {
                field_id: 1,
                stream_id: 0,
                replica_epoch: 0,
            },
            context_id: 1,
            lineage: None,
            coverage: CoverageRange {
                first_ordinal: first,
                next_ordinal: first + words.len() as u64,
                first_step: first,
                last_step: first + words.len() as u64 - 1,
                first_physical_time_ps_bits: (first as f64).to_bits(),
                last_physical_time_ps_bits: ((first + words.len() as u64 - 1) as f64).to_bits(),
            },
            bytes: words.iter().flat_map(|x| x.to_le_bytes()).collect(),
        }
    }
    #[test]
    fn streaming_slabs_publish_one_generation_and_exact_byte_population() {
        let dir = tempfile::tempdir().unwrap();
        let store = CheckpointStore::create(&dir.path().join("store"), 4096).unwrap();
        let words = [0x7ff8_1234_5678_9abc, 0x8000_0000_0000_0000, 3];
        let h = store.commit_chronicle_streaming(None, &identity(), &dictionary(), 8,
            &[], None, |append| {
                for (i, word) in words.iter().enumerate() {
                    let slab = leaf(i as u64, &[*word]);
                    append(&slab)?;
                    // Source slab is dropped each iteration; no whole-chunk Vec.
                }
                Ok(())
            }).unwrap();
        assert_eq!(h.generation, 1);
        assert!(store.head().unwrap().is_none());
        let index = store.validate_chronicle_index(&h, &identity()).unwrap();
        assert_eq!(index.foliage().leaves.len(), 3);
        assert_eq!(index.foliage().producers[0].next_ordinal, 3);
        for (i, word) in words.iter().enumerate() {
            assert_eq!(store.read_indexed_chronicle_payload(&index, i).unwrap(), word.to_le_bytes());
        }
        assert!(store.read_indexed_chronicle_payload(&index, 3).is_err());
    }
    #[test]
    fn streaming_callback_failure_keeps_prior_head_and_watermark() {
        let dir = tempfile::tempdir().unwrap();
        let store = CheckpointStore::create(&dir.path().join("store"), 4096).unwrap();
        let h = store.commit_chronicle(None, &identity(), &dictionary(), &[leaf(0, &[1])], &[], None).unwrap();
        let result = store.commit_chronicle_streaming(Some(&h), &identity(), &dictionary(), 8,
            &[], None, |append| { append(&leaf(1, &[2]))?; bail!("producer read failed") });
        assert!(result.is_err());
        assert_eq!(store.chronicle_head().unwrap(), Some(h.clone()));
        let f = store.inspect_chronicle(&h, &identity()).unwrap();
        assert_eq!(f.producers[0].next_ordinal, 1);
        assert_eq!(f.leaves.len(), 1);
    }
    #[test]
    fn streaming_late_gap_and_budget_refuse_prior_head_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let store = CheckpointStore::create(&dir.path().join("store"), 16).unwrap();
        let h = store.commit_chronicle(None, &identity(), &dictionary(), &[leaf(0, &[1])], &[], None).unwrap();
        assert!(store.commit_chronicle_streaming(Some(&h), &identity(), &dictionary(), 8,
            &[], None, |append| { append(&leaf(2, &[3])) }).is_err());
        assert_eq!(store.chronicle_head().unwrap(), Some(h.clone()));
        assert!(store.commit_chronicle_streaming(Some(&h), &identity(), &dictionary(), 8,
            &[], None, |append| { append(&leaf(1, &[2]))?; append(&leaf(2, &[3])) }).is_err());
        assert_eq!(store.chronicle_head().unwrap(), Some(h));
    }
    #[test]
    fn streaming_phase_failures_preserve_readable_old_or_complete_new_head() {
        for phase in ["leaf_object_staged", "objects_synced", "manifest_synced", "head_published", "head_directory_synced"] {
            let dir = tempfile::tempdir().unwrap();
            let store = CheckpointStore::create(&dir.path().join("store"), 4096).unwrap();
            let h = store.commit_chronicle(None, &identity(), &dictionary(), &[leaf(0, &[1])], &[], None).unwrap();
            let result = store.commit_chronicle_streaming_with_hook(Some(&h), &identity(), &dictionary(), 8,
                &[], None, |append| append(&leaf(1, &[2])), |at| {
                    if at == phase { bail!("injected failure at {phase}"); } Ok(())
                });
            assert!(result.is_err(), "phase must be reached: {phase}");
            let visible = store.chronicle_head().unwrap().unwrap();
            let f = store.inspect_chronicle(&visible, &identity()).unwrap();
            assert!(f.leaves.len() == 1 || f.leaves.len() == 2);
            if f.leaves.len() == 1 { assert_eq!(visible, h); }
            else { assert_eq!(f.producers[0].next_ordinal, 2); }
            assert!(store.head().unwrap().is_none());
        }
    }
    #[test]
    fn validated_index_prefix_survives_append_without_granting_stale_commit() {
        let dir = tempfile::tempdir().unwrap();
        let store = CheckpointStore::create(&dir.path().join("store"), 4096).unwrap();
        let h = store.commit_chronicle(None, &identity(), &dictionary(), &[leaf(0, &[1])], &[], None).unwrap();
        let index = store.validate_chronicle_index(&h, &identity()).unwrap();
        let next = store.commit_chronicle(Some(&h), &identity(), &dictionary(), &[leaf(1, &[2])], &[], None).unwrap();
        assert_eq!(store.read_indexed_chronicle_payload(&index, 0).unwrap(), 1u64.to_le_bytes());
        assert_eq!(index.head(), &h);
        assert!(store.read_indexed_chronicle_payload(&index, 1).is_err());
        assert!(store.validate_chronicle_index(&h, &identity()).is_err());
        assert!(store.commit_chronicle(Some(&h), &identity(), &dictionary(), &[leaf(1, &[2])], &[], None).is_err());
        assert_eq!(store.chronicle_head().unwrap(), Some(next));
    }
    #[test]
    fn validated_index_rejects_wrong_store_identity_forged_head_and_corrupt_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let store = CheckpointStore::create(&dir.path().join("a"), 4096).unwrap();
        let other = CheckpointStore::create(&dir.path().join("b"), 4096).unwrap();
        let h = store.commit_chronicle(None, &identity(), &dictionary(), &[leaf(0, &[1])], &[], None).unwrap();
        let mut wrong_identity = identity(); wrong_identity.run_id = "another-run".into();
        assert!(store.validate_chronicle_index(&h, &wrong_identity).is_err());
        let mut forged = h.clone(); forged.manifest_sha256 = "f".repeat(64);
        assert!(store.validate_chronicle_index(&forged, &identity()).is_err());
        let index = store.validate_chronicle_index(&h, &identity()).unwrap();
        assert!(other.read_indexed_chronicle_payload(&index, 0).is_err());
        let object = store.root.join("objects").join(&index.foliage().leaves[0].object_sha256);
        std::fs::write(object, 2u64.to_le_bytes()).unwrap();
        assert!(store.read_indexed_chronicle_payload(&index, 0).is_err());
    }
    #[test]
    fn ignored_object_write_failure_cannot_publish_even_after_retry() {
        let dir = tempfile::tempdir().unwrap();
        let store = CheckpointStore::create(&dir.path().join("store"), 4096).unwrap();
        let h = store.commit_chronicle(None, &identity(), &dictionary(), &[leaf(0, &[1])], &[], None).unwrap();
        let next = leaf(2, &[3]);
        let digest = hash(&next.bytes);
        // This exact existing-store staging path must refuse a directory.
        let obstruction = store.root.join("objects").join(format!(".{digest}.next"));
        std::fs::create_dir(&obstruction).unwrap();
        let result = store.commit_chronicle_streaming(Some(&h), &identity(), &dictionary(), 8,
            &[], None, |append| {
                append(&leaf(1, &[2]))?;
                assert!(append(&next).is_err());
                std::fs::remove_dir(&obstruction)?;
                // Even a later retry after removing the cause cannot unpoison.
                assert!(append(&next).is_err());
                Ok(())
            });
        assert!(result.is_err());
        assert!(!store.root.join("objects").join(digest).exists());
        assert_eq!(store.chronicle_head().unwrap(), Some(h.clone()));
        assert_eq!(store.inspect_chronicle(&h, &identity()).unwrap().leaves.len(), 1);
    }
    #[test]
    fn ignored_staged_object_hook_failure_cannot_publish() {
        let dir = tempfile::tempdir().unwrap();
        let store = CheckpointStore::create(&dir.path().join("store"), 4096).unwrap();
        let h = store.commit_chronicle(None, &identity(), &dictionary(), &[leaf(0, &[1])], &[], None).unwrap();
        let mut calls = 0;
        let result = store.commit_chronicle_streaming_with_hook(Some(&h), &identity(), &dictionary(), 8,
            &[], None, |append| {
                append(&leaf(1, &[2]))?;
                let _ = append(&leaf(2, &[3]));
                let _ = append(&leaf(2, &[3]));
                Ok(())
            }, |phase| {
                if phase == "leaf_object_staged" {
                    calls += 1;
                    if calls == 2 { bail!("injected staged object failure"); }
                }
                Ok(())
            });
        assert!(result.is_err());
        assert_eq!(calls, 2);
        assert_eq!(store.chronicle_head().unwrap(), Some(h.clone()));
        assert_eq!(store.inspect_chronicle(&h, &identity()).unwrap().producers[0].next_ordinal, 1);
    }
    #[test]
    fn immutable_index_rechecks_cumulative_budget_of_receiving_store() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("store");
        let store = CheckpointStore::create(&root, 4096).unwrap();
        let h = store.commit_chronicle(None, &identity(), &dictionary(), &[leaf(0, &[1]), leaf(1, &[2])], &[], None).unwrap();
        let index = store.validate_chronicle_index(&h, &identity()).unwrap();
        let stricter = CheckpointStore::create(&root, 8).unwrap();
        assert!(stricter.read_indexed_chronicle_payload(&index, 0).is_err());
        assert!(stricter.validate_chronicle_index(&h, &identity()).is_err());
        let equal = CheckpointStore::create(&root, 16).unwrap();
        assert_eq!(equal.read_indexed_chronicle_payload(&index, 0).unwrap(), 1u64.to_le_bytes());
        assert_eq!(store.read_indexed_chronicle_payload(&index, 1).unwrap(), 2u64.to_le_bytes());
    }
    #[test]
    fn immutable_index_budget_includes_acknowledgment_receipt_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("store");
        let store = CheckpointStore::create(&root, 4096).unwrap();
        let h = store.commit_chronicle(None, &identity(), &dictionary(), &[leaf(0, &[1])], &[], None).unwrap();
        let receipt = b"consumer input population receipt".to_vec();
        let total = 8 + receipt.len() as u64;
        let ack = ConsumerAck { consumer: "path_b".into(), producer: ProducerKey {
            field_id: 1, stream_id: 0, replica_epoch: 0 }, next_ordinal: 1,
            input_foliage: h.clone(), receipt_bytes: receipt, output_payloads: vec![] };
        let next = store.commit_chronicle(Some(&h), &identity(), &dictionary(), &[], &[ack], None).unwrap();
        let index = store.validate_chronicle_index(&next, &identity()).unwrap();
        let payload_only = CheckpointStore::create(&root, 8).unwrap();
        assert!(payload_only.read_indexed_chronicle_payload(&index, 0).is_err());
        let exact = CheckpointStore::create(&root, total).unwrap();
        assert_eq!(exact.read_indexed_chronicle_payload(&index, 0).unwrap(), 1u64.to_le_bytes());
    }
    #[test]
    fn streaming_slab_limit_and_empty_transition_refused() {
        let dir = tempfile::tempdir().unwrap();
        let store = CheckpointStore::create(&dir.path().join("store"), 4096).unwrap();
        assert!(store.commit_chronicle_streaming(None, &identity(), &dictionary(), 7,
            &[], None, |append| append(&leaf(0, &[1]))).is_err());
        assert!(store.commit_chronicle_streaming(None, &identity(), &dictionary(), 8,
            &[], None, |_| Ok(())).is_err());
        assert!(store.chronicle_head().unwrap().is_none());
    }
    #[test]
    fn raw_bits_and_component_address_survive_observational_commit() {
        let dir = tempfile::tempdir().unwrap();
        let store = CheckpointStore::create(&dir.path().join("store"), 4096).unwrap();
        let words = [0x7ff8_1234_5678_9abcu64, 0x8000_0000_0000_0000];
        let input = leaf(0, &words);
        let expected = input.bytes.clone();
        let h = store
            .commit_chronicle(None, &identity(), &dictionary(), &[input], &[], None)
            .unwrap();
        assert!(store.head().unwrap().is_none());
        assert_eq!(
            store.read_chronicle_payload(&h, &identity(), 0).unwrap(),
            expected
        );
        let address = DatumAddress {
            leaf_index: 0,
            record_ordinal: 1,
            element_index: 0,
            component_id: Some(0),
        };
        assert_eq!(
            store
                .read_chronicle_datum(&h, &identity(), &address)
                .unwrap(),
            words[1].to_le_bytes()
        );
        let f = store.inspect_chronicle(&h, &identity()).unwrap();
        assert!(!f.required_consumers_acknowledged(0).unwrap());
        assert!(store
            .restore_chronicle_anchor(
                &h,
                &identity(),
                &ClosureContract {
                    schema_version: SCHEMA,
                    fields: vec![],
                    required_owners: BTreeSet::new(),
                    unresolved_owners: BTreeSet::new()
                }
            )
            .is_err());
    }
    #[test]
    fn producer_gap_overlap_and_context_identity_refused_without_head_change() {
        let dir = tempfile::tempdir().unwrap();
        let store = CheckpointStore::create(&dir.path().join("store"), 4096).unwrap();
        let id = identity();
        let d = dictionary();
        let h = store
            .commit_chronicle(None, &id, &d, &[leaf(0, &[1, 2])], &[], None)
            .unwrap();
        for start in [0, 1, 3] {
            assert!(store
                .commit_chronicle(Some(&h), &id, &d, &[leaf(start, &[3])], &[], None)
                .is_err());
            assert_eq!(store.chronicle_head().unwrap(), Some(h.clone()));
        }
        let mut bad = leaf(2, &[3]);
        bad.context_id = 99;
        assert!(store
            .commit_chronicle(Some(&h), &id, &d, &[bad], &[], None)
            .is_err());
        let mut changed = d.clone();
        changed.contexts[0].thermal_phase = "warm_hold".into();
        assert!(store
            .commit_chronicle(Some(&h), &id, &changed, &[leaf(2, &[3])], &[], None)
            .is_err());
        assert!(store
            .commit_chronicle(None, &id, &d, &[leaf(2, &[3])], &[], None)
            .is_err());
    }
    #[test]
    fn named_consumer_advances_only_over_previously_committed_population() {
        let dir = tempfile::tempdir().unwrap();
        let store = CheckpointStore::create(&dir.path().join("store"), 4096).unwrap();
        let id = identity();
        let d = dictionary();
        let h = store
            .commit_chronicle(None, &id, &d, &[leaf(0, &[1, 2])], &[], None)
            .unwrap();
        let ack = |end| ConsumerAck {
            consumer: "path_b".into(),
            producer: ProducerKey {
                field_id: 1,
                stream_id: 0,
                replica_epoch: 0,
            },
            next_ordinal: end,
            input_foliage: h.clone(),
            receipt_bytes: b"native-consumer-input-digest-receipt".to_vec(),
            output_payloads: vec![],
        };
        assert!(store
            .commit_chronicle(Some(&h), &id, &d, &[], &[ack(3)], None)
            .is_err());
        let next = store
            .commit_chronicle(Some(&h), &id, &d, &[], &[ack(2)], None)
            .unwrap();
        assert!(store
            .inspect_chronicle(&next, &id)
            .unwrap()
            .required_consumers_acknowledged(0)
            .unwrap());
        assert!(store
            .commit_chronicle(Some(&next), &id, &d, &[], &[ack(2)], None)
            .is_err());
        assert!(store.inspect_chronicle(&h, &id).is_err());
    }
    #[test]
    fn typed_shape_width_and_payload_population_negatives() {
        let id = identity();
        let mut d = dictionary();
        d.fields[0].shape = vec![2];
        assert!(d.validate(&id).is_err());
        let mut d = dictionary();
        d.fields[0].element_bytes = 4;
        assert!(d.validate(&id).is_err());
        let mut d = dictionary();
        d.fields[0].components[0].byte_offset = 4;
        assert!(d.validate(&id).is_err());
        let dir = tempfile::tempdir().unwrap();
        let store = CheckpointStore::create(&dir.path().join("store"), 4096).unwrap();
        let mut l = leaf(0, &[1]);
        l.bytes.pop();
        assert!(store
            .commit_chronicle(None, &id, &dictionary(), &[l], &[], None)
            .is_err());
        assert!(store.chronicle_head().unwrap().is_none());
    }
    #[test]
    fn budget_and_injected_publication_failures_preserve_previous_readable_head() {
        let dir = tempfile::tempdir().unwrap();
        let store = CheckpointStore::create(&dir.path().join("store"), 16).unwrap();
        let id = identity();
        let d = dictionary();
        let h = store
            .commit_chronicle(None, &id, &d, &[leaf(0, &[1])], &[], None)
            .unwrap();
        assert!(store
            .commit_chronicle(Some(&h), &id, &d, &[leaf(1, &[2, 3])], &[], None)
            .is_err());
        assert_eq!(store.chronicle_head().unwrap(), Some(h.clone()));
        for point in ["objects_synced", "manifest_synced", "head_staged_synced"] {
            let result = store.commit_chronicle_with_hook(
                Some(&h),
                &id,
                &d,
                &[leaf(1, &[2])],
                &[],
                None,
                |phase| {
                    ensure!(phase != point, "injected write/fsync failure");
                    Ok(())
                },
            );
            assert!(result.is_err());
            assert_eq!(store.chronicle_head().unwrap(), Some(h.clone()));
            assert!(store.inspect_chronicle(&h, &id).is_ok());
        }
    }
    #[test]
    fn raw_corruption_is_detected_by_payload_read_and_full_inspection() {
        let dir = tempfile::tempdir().unwrap();
        let store = CheckpointStore::create(&dir.path().join("store"), 4096).unwrap();
        let id = identity();
        let input = leaf(0, &[1]);
        let digest = hash(&input.bytes);
        let h = store
            .commit_chronicle(None, &id, &dictionary(), &[input], &[], None)
            .unwrap();
        fs::write(store.root.join("objects").join(digest), [0u8; 8]).unwrap();
        assert!(store.inspect_chronicle_index(&h, &id).is_ok()); // index is explicitly metadata-only
        assert!(store.read_chronicle_payload(&h, &id, 0).is_err());
        assert!(store.inspect_chronicle(&h, &id).is_err());
    }
    #[test]
    fn complete_restart_anchor_uses_existing_full_closure_decoder() {
        let dir = tempfile::tempdir().unwrap();
        let store = CheckpointStore::create(&dir.path().join("store"), 4096).unwrap();
        let id = identity();
        let fc = FieldContract {
            owner: "stream_0/engine".into(),
            name: "state".into(),
            encoding: "u64.le.v1".into(),
            expected_bytes: Some(8),
            may_be_absent: false,
        };
        let closure = ClosureContract {
            schema_version: SCHEMA,
            fields: vec![fc.clone()],
            required_owners: BTreeSet::from([fc.owner.clone()]),
            unresolved_owners: BTreeSet::new(),
        };
        let boundaries = vec![StreamBoundary {
            stream_id: 0,
            replica_epoch: 0,
            replica_seed: 42,
            completed_step: 2,
            physical_time_ps_bits: 2f64.to_bits(),
            next_integrator_phase: "begin_step".into(),
            coupling_epoch: 2,
            coupling_parity: 0,
            committed_output_sequence: 0,
        }];
        let state = CapturedField {
            contract: fc,
            bytes: Some(17u64.to_le_bytes().to_vec()),
        };
        let anchor = store
            .commit(None, &id, &closure, &boundaries, &[state])
            .unwrap();
        let h = store
            .commit_chronicle(
                None,
                &id,
                &dictionary(),
                &[leaf(0, &[1, 2])],
                &[],
                Some(RestartAnchorRequest {
                    head: &anchor,
                    closure: &closure,
                }),
            )
            .unwrap();
        assert_eq!(store.head().unwrap(), Some(anchor.clone()));
        let restored = store.restore_chronicle_anchor(&h, &id, &closure).unwrap();
        assert_eq!(restored.head, anchor);
        let mut incomplete = closure.clone();
        incomplete.unresolved_owners.insert("rng".into());
        assert!(store
            .restore_chronicle_anchor(&h, &id, &incomplete)
            .is_err());
    }
    #[test]
    fn derived_lineage_is_per_leaf_and_permuted_clocks_do_not_reorder_raw_identity() {
        let dir = tempfile::tempdir().unwrap();
        let store = CheckpointStore::create(&dir.path().join("store"), 4096).unwrap();
        let id = identity();
        let mut d = dictionary();
        let mut derived = d.fields[0].clone();
        derived.id = 2;
        derived.name = "derived_view".into();
        derived.class = DatumClass::Derived;
        derived.index_order = "explicit-transform-row-order".into();
        d.fields.push(derived);
        let raw = leaf(0, &[1, 2]);
        let h = store
            .commit_chronicle(None, &id, &d, &[raw], &[], None)
            .unwrap();
        let raw_ref = store.inspect_chronicle_index(&h, &id).unwrap().leaves[0].reference();
        let output = |start, step, input: Head| {
            let mut l = leaf(start, &[3]);
            l.producer.field_id = 2;
            l.coverage.first_step = step;
            l.coverage.last_step = step;
            l.coverage.first_physical_time_ps_bits = (step as f64).to_bits();
            l.coverage.last_physical_time_ps_bits = (step as f64).to_bits();
            l.lineage = Some(LeafLineage {
                input_foliage: input,
                parent_payloads: vec![raw_ref.clone()],
                transform_sha256: "9".repeat(64),
            });
            l
        };
        let first = store
            .commit_chronicle(Some(&h), &id, &d, &[output(0, 1, h.clone())], &[], None)
            .unwrap();
        let next = store
            .commit_chronicle(
                Some(&first),
                &id,
                &d,
                &[output(1, 0, first.clone())],
                &[],
                None,
            )
            .unwrap();
        let f = store.inspect_chronicle(&next, &id).unwrap();
        assert_eq!(f.leaves[2].coverage.first_step, 0);
        assert_eq!(f.leaves[0].coverage.first_ordinal, 0);
        let mut missing = output(2, 1, next.clone());
        missing.lineage = None;
        assert!(store
            .commit_chronicle(Some(&next), &id, &d, &[missing], &[], None)
            .is_err());
        let mut unknown = output(2, 1, next.clone());
        unknown.lineage.as_mut().unwrap().parent_payloads[0].object_sha256 = "a".repeat(64);
        assert!(store
            .commit_chronicle(Some(&next), &id, &d, &[unknown], &[], None)
            .is_err());
    }
    #[test]
    fn identical_content_cannot_alias_distinct_causal_or_output_leaf_addresses() {
        let dir = tempfile::tempdir().unwrap();
        let store = CheckpointStore::create(&dir.path().join("store"), 8192).unwrap();
        let mut id = identity();
        id.stream_count = 2;
        let mut d = dictionary();
        let mut other = d.contexts[0].clone();
        other.id = 2;
        other.stream_id = 1;
        other.replica_seed = 99;
        d.contexts.push(other);
        let mut derived = d.fields[0].clone();
        derived.id = 2;
        derived.name = "derived".into();
        derived.class = DatumClass::Derived;
        d.fields.push(derived);
        let a = leaf(0, &[1]);
        let mut b = leaf(0, &[1]);
        b.producer.stream_id = 1;
        b.context_id = 2;
        b.coverage.first_step = 9;
        b.coverage.last_step = 9;
        b.coverage.first_physical_time_ps_bits = 9f64.to_bits();
        b.coverage.last_physical_time_ps_bits = 9f64.to_bits();
        let h = store
            .commit_chronicle(None, &id, &d, &[a, b], &[], None)
            .unwrap();
        let f = store.inspect_chronicle_index(&h, &id).unwrap();
        let first = f.leaves[0].reference();
        let second = f.leaves[1].reference();
        assert_eq!(first.object_sha256, second.object_sha256);
        assert_ne!(first, second);
        let mut view = leaf(0, &[2]);
        view.producer.field_id = 2;
        view.lineage = Some(LeafLineage {
            input_foliage: h.clone(),
            parent_payloads: vec![first.clone()],
            transform_sha256: "a".repeat(64),
        });
        let ack = ConsumerAck {
            consumer: "path_b".into(),
            producer: first.producer.clone(),
            next_ordinal: 1,
            input_foliage: h.clone(),
            receipt_bytes: b"source+output-address-receipt".to_vec(),
            output_payloads: vec![second.clone()],
        };
        let next = store
            .commit_chronicle(Some(&h), &id, &d, &[view], &[ack], None)
            .unwrap();
        let stored = store.inspect_chronicle(&next, &id).unwrap();
        assert_eq!(
            stored.leaves[2].lineage.as_ref().unwrap().parent_payloads[0],
            first
        );
        assert_eq!(stored.consumers[0].output_payloads[0], second);
        let mut forged = second.clone();
        forged.producer = first.producer.clone();
        let mut bad = leaf(1, &[3]);
        bad.producer.field_id = 2;
        bad.lineage = Some(LeafLineage {
            input_foliage: next.clone(),
            parent_payloads: vec![forged],
            transform_sha256: "a".repeat(64),
        });
        assert!(store
            .commit_chronicle(Some(&next), &id, &d, &[bad], &[], None)
            .is_err());
        assert_eq!(store.chronicle_head().unwrap(), Some(next));
    }

    // These tests exercise the real CHRONICLE_HEAD namespace and the existing
    // store helpers. They qualify process termination and injected hook errors,
    // not physical MD continuation, power loss, or a genuinely full filesystem.
    const CHRONICLE_PROCESS_GUARD: &[u8] = b"PRISM_CHRONICLE_PROCESS_TEST_V1\n";
    const CHRONICLE_PHASES: [&str; 5] = [
        "objects_synced",
        "manifest_synced",
        "head_staged_synced",
        "head_published",
        "head_directory_synced",
    ];

    fn chronicle_process_fixture() -> (tempfile::TempDir, CheckpointStore, Head) {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("CHRONICLE_TEST_GUARD"),
            CHRONICLE_PROCESS_GUARD,
        )
        .unwrap();
        let store = CheckpointStore::create(&dir.path().join("chronicle-store"), 4096).unwrap();
        let head = store
            .commit_chronicle(
                None,
                &identity(),
                &dictionary(),
                &[leaf(0, &[1])],
                &[],
                None,
            )
            .unwrap();
        (dir, store, head)
    }

    fn assert_chronicle_fixture_prefix(store: &CheckpointStore, head: &Head, records: u64) {
        assert!(
            store.head().unwrap().is_none(),
            "restart HEAD must remain separate"
        );
        let f = store.inspect_chronicle(head, &identity()).unwrap();
        assert_eq!(f.generation, records);
        assert_eq!(f.leaves.len() as u64, records);
        assert_eq!(f.producers.len(), 1);
        assert_eq!(f.producers[0].next_ordinal, records);
        for i in 0..records {
            assert_eq!(f.leaves[i as usize].coverage.first_ordinal, i);
            assert_eq!(f.leaves[i as usize].coverage.next_ordinal, i + 1);
            assert_eq!(
                store
                    .read_chronicle_payload(head, &identity(), i as usize)
                    .unwrap(),
                (i + 1).to_le_bytes()
            );
        }
    }

    // Own only the subprocess started by this test. Panic/timeout paths also
    // retire it; no process group, external service or native GPU PID is used.
    struct ChronicleChildGuard(Option<std::process::Child>);
    impl ChronicleChildGuard {
        fn spawn(command: &mut std::process::Command) -> Self {
            Self(Some(command.spawn().unwrap()))
        }
        fn id(&self) -> u32 {
            self.0.as_ref().unwrap().id()
        }
        fn wait(mut self) -> std::process::ExitStatus {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            loop {
                if let Some(status) = self.0.as_mut().unwrap().try_wait().unwrap() {
                    self.0.take();
                    return status;
                }
                assert!(
                    std::time::Instant::now() < deadline,
                    "Chronicle test child timed out"
                );
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }
    }
    impl Drop for ChronicleChildGuard {
        fn drop(&mut self) {
            if let Some(mut child) = self.0.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }

    fn chronicle_child_command(
        root: &Path,
        expected: &Head,
        action: &str,
        phase: &str,
    ) -> std::process::Command {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .arg("--exact")
            .arg("scientific_checkpoint::chronicle::tests::chronicle_process_child")
            .arg("--ignored")
            .arg("--nocapture")
            .env("PRISM_CHRONICLE_TEST_ROOT", root)
            .env(
                "PRISM_CHRONICLE_TEST_EXPECTED",
                serde_json::to_string(expected).unwrap(),
            )
            .env("PRISM_CHRONICLE_TEST_ACTION", action)
            .env("PRISM_CHRONICLE_TEST_PHASE", phase);
        command
    }

    #[test]
    #[ignore = "guarded child entry; invoked only by Chronicle namespace process tests"]
    fn chronicle_process_child() {
        let root = PathBuf::from(
            std::env::var_os("PRISM_CHRONICLE_TEST_ROOT").expect("missing test root"),
        );
        assert!(root.is_absolute() && root.is_dir());
        assert_eq!(
            root.file_name().and_then(|n| n.to_str()),
            Some("chronicle-store")
        );
        let parent = root.parent().unwrap();
        assert_eq!(
            fs::read(parent.join("CHRONICLE_TEST_GUARD")).unwrap(),
            CHRONICLE_PROCESS_GUARD
        );
        let expected: Head =
            serde_json::from_str(&std::env::var("PRISM_CHRONICLE_TEST_EXPECTED").unwrap()).unwrap();
        let action = std::env::var("PRISM_CHRONICLE_TEST_ACTION").unwrap();
        let phase = std::env::var("PRISM_CHRONICLE_TEST_PHASE").unwrap();
        assert!(matches!(action.as_str(), "kill" | "race" | "verify"));
        let store = CheckpointStore::create(&root, 4096).unwrap();
        if action == "verify" {
            let records: u64 = phase.parse().unwrap();
            assert!((1..=3).contains(&records));
            assert_eq!(store.chronicle_head().unwrap(), Some(expected.clone()));
            assert_chronicle_fixture_prefix(&store, &expected, records);
            return;
        }
        if action == "race" {
            assert!(phase.is_empty());
            fs::write(
                parent.join(format!("CHRONICLE_READY_{}", std::process::id())),
                b"ready",
            )
            .unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while !parent.join("CHRONICLE_RACE_GO").is_file() {
                assert!(std::time::Instant::now() < deadline, "race gate timed out");
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        } else {
            assert!(CHRONICLE_PHASES.contains(&phase.as_str()));
        }
        let result = store.commit_chronicle_with_hook(
            Some(&expected),
            &identity(),
            &dictionary(),
            &[leaf(1, &[2])],
            &[],
            None,
            |point| {
                if action == "kill" && point == phase {
                    // SIGKILL bypasses Rust drops and normal exception cleanup.
                    unsafe {
                        libc::raise(libc::SIGKILL);
                    }
                    anyhow::bail!("SIGKILL unexpectedly returned");
                }
                Ok(())
            },
        );
        match result {
            Ok(_) => assert_eq!(action, "race", "kill hook was not reached"),
            Err(error) => {
                let text = format!("{error:#}");
                eprintln!("{text}");
                // Only expected CAS/lock contention is a race-loser result.
                if action == "race"
                    && (text.contains("checkpoint writer is already active")
                        || text.contains("stale Chronicle HEAD"))
                {
                    std::process::exit(74);
                }
                std::process::exit(75);
            }
        }
    }

    fn verify_chronicle_in_fresh_process(root: &Path, head: &Head, records: u64) {
        let mut command = chronicle_child_command(root, head, "verify", &records.to_string());
        assert!(ChronicleChildGuard::spawn(&mut command).wait().success());
    }

    #[test]
    fn chronicle_sigkill_at_all_publication_phases_preserves_complete_prefix_and_resume() {
        use std::os::unix::process::ExitStatusExt;
        for point in CHRONICLE_PHASES {
            let (_dir, store, first) = chronicle_process_fixture();
            let mut command = chronicle_child_command(&store.root, &first, "kill", point);
            let status = ChronicleChildGuard::spawn(&mut command).wait();
            assert_eq!(status.signal(), Some(libc::SIGKILL), "{point}");
            let reopened = CheckpointStore::create(&store.root, 4096).unwrap();
            let current = reopened.chronicle_head().unwrap().unwrap();
            let count = if ["head_published", "head_directory_synced"].contains(&point) {
                2
            } else {
                1
            };
            assert_eq!(current.generation, count, "{point}");
            assert_chronicle_fixture_prefix(&reopened, &current, count);
            verify_chronicle_in_fresh_process(&store.root, &current, count);
            let resumed = reopened
                .commit_chronicle(
                    Some(&current),
                    &identity(),
                    &dictionary(),
                    &[leaf(count, &[count + 1])],
                    &[],
                    None,
                )
                .unwrap();
            assert_chronicle_fixture_prefix(&reopened, &resumed, count + 1);
            verify_chronicle_in_fresh_process(&store.root, &resumed, count + 1);
        }
    }

    #[test]
    fn chronicle_independent_writer_processes_publish_exactly_one_next_generation() {
        let (dir, store, first) = chronicle_process_fixture();
        let mut a_command = chronicle_child_command(&store.root, &first, "race", "");
        let a = ChronicleChildGuard::spawn(&mut a_command);
        let mut b_command = chronicle_child_command(&store.root, &first, "race", "");
        let b = ChronicleChildGuard::spawn(&mut b_command);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while ![a.id(), b.id()]
            .iter()
            .all(|pid| dir.path().join(format!("CHRONICLE_READY_{pid}")).is_file())
        {
            assert!(
                std::time::Instant::now() < deadline,
                "race participants did not reach gate"
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        fs::write(dir.path().join("CHRONICLE_RACE_GO"), b"go").unwrap();
        let codes = [a.wait().code(), b.wait().code()];
        assert!(
            codes == [Some(0), Some(74)] || codes == [Some(74), Some(0)],
            "{codes:?}"
        );
        let reopened = CheckpointStore::create(&store.root, 4096).unwrap();
        let current = reopened.chronicle_head().unwrap().unwrap();
        assert_eq!(current.generation, 2);
        assert_chronicle_fixture_prefix(&reopened, &current, 2);
        verify_chronicle_in_fresh_process(&store.root, &current, 2);
    }

    #[test]
    fn chronicle_injected_enospc_hook_errors_reconcile_pre_and_post_rename_heads() {
        for point in CHRONICLE_PHASES {
            let (_dir, store, first) = chronicle_process_fixture();
            let mut reached = false;
            let result = store.commit_chronicle_with_hook(
                Some(&first),
                &identity(),
                &dictionary(),
                &[leaf(1, &[2])],
                &[],
                None,
                |phase| {
                    if phase == point {
                        reached = true;
                        // Deliberate hook injection, not real filesystem exhaustion.
                        return Err(std::io::Error::from_raw_os_error(libc::ENOSPC).into());
                    }
                    Ok(())
                },
            );
            assert!(reached, "{point}");
            let error = result.unwrap_err();
            assert_eq!(
                error
                    .downcast_ref::<std::io::Error>()
                    .unwrap()
                    .raw_os_error(),
                Some(libc::ENOSPC)
            );
            let reopened = CheckpointStore::create(&store.root, 4096).unwrap();
            let current = reopened.chronicle_head().unwrap().unwrap();
            let count = if ["head_published", "head_directory_synced"].contains(&point) {
                2
            } else {
                1
            };
            assert_eq!(current.generation, count, "{point}");
            assert_chronicle_fixture_prefix(&reopened, &current, count);
            verify_chronicle_in_fresh_process(&store.root, &current, count);
            let resumed = reopened
                .commit_chronicle(
                    Some(&current),
                    &identity(),
                    &dictionary(),
                    &[leaf(count, &[count + 1])],
                    &[],
                    None,
                )
                .unwrap();
            assert_chronicle_fixture_prefix(&reopened, &resumed, count + 1);
        }
    }

}

impl CheckpointStore {
    /// Fork only an explicit verified historical observation prefix. The source
    /// remains intact and cannot be silently rewound by continuation.
    pub fn fork_chronicle_history(&self,destination:&std::path::Path,expected_current:&Head,
        anchor:&Head,identity:&CheckpointIdentity)->Result<Head> {
        ensure!(destination!=self.root && !destination.starts_with(&self.root) && !self.root.starts_with(destination),"Chronicle branch namespaces overlap");
        let target=CheckpointStore::create(destination,self.max_total_bytes)?;
        let _source_lock=self.lock()?;let _target_lock=target.branch_writer_lock()?;
        ensure!(self.chronicle_head()?.as_ref()==Some(expected_current),"Chronicle source advanced");
        if let Some(existing)=target.chronicle_head()? {
            ensure!(&existing==anchor,"Chronicle branch destination contains another generation");
            let expected=serde_json::json!({"schema_version":1,"kind":"explicit_historical_observation_branch","source_root":self.root,"source_current_head":expected_current,"selected_anchor":anchor,"identity":identity,"source_head_unchanged":true});
            let actual:serde_json::Value=serde_json::from_slice(&super::read_bounded(&target.root.join("CHRONICLE_BRANCH_ORIGIN.json"),65536)?)?;
            ensure!(actual==expected,"existing Chronicle branch origin differs from exact ingress contract");
            target.verify_foliage_objects(&target.read_foliage(anchor)?)?;return Ok(anchor.clone());
        }
        if let Some(head)=target.head()? {ensure!(target.read_manifest(&head)?.identity==*identity,"scientific/observational branch identities differ");}
        let mut cursor=Some(expected_current.clone());let mut found=false;let mut count=0usize;
        while let Some(head)=cursor {
            count+=1;ensure!(count<=100000,"Chronicle branch ancestry exceeds admitted bound");
            let f=self.read_foliage(&head)?;ensure!(f.identity==*identity,"Chronicle ancestor identity differs");
            if head==*anchor {found=true;break;}
            if let Some(previous)=&f.previous_foliage {ensure!(previous.generation<head.generation,"Chronicle ancestry does not decrease");}
            cursor=f.previous_foliage;
        }
        ensure!(found,"Chronicle requested anchor is not a verified current ancestor");
        let mut cursor=Some(anchor.clone());let mut objects=std::collections::BTreeSet::new();let mut count=0usize;
        while let Some(head)=cursor {
            count+=1;ensure!(count<=100000,"Chronicle retained ancestry exceeds admitted bound");
            let f=self.read_foliage(&head)?;ensure!(f.identity==*identity,"Chronicle retained ancestor identity differs");
            for (digest,bytes) in f.leaves.iter().map(|l|(&l.object_sha256,l.byte_count)).chain(f.consumers.iter().map(|a|(&a.receipt_sha256,a.receipt_bytes))) {
                ensure!(objects.len()<262144,"Chronicle branch object metadata exceeds bounded admission");
                if objects.insert(digest.clone()) {self.verify_object_streaming(digest,bytes)?;self.link_immutable_to(&target,"objects",digest)?;}
            }
            self.link_immutable_to(&target,"manifests",&head.manifest_sha256)?;
            if let Some(previous)=&f.previous_foliage {ensure!(previous.generation<head.generation,"Chronicle retained ancestry does not decrease");}
            cursor=f.previous_foliage;
        }
        sync_dir(&target.root.join("objects"))?;sync_dir(&target.root.join("manifests"))?;
        let origin=serde_json::json!({"schema_version":1,"kind":"explicit_historical_observation_branch","source_root":self.root,"source_current_head":expected_current,"selected_anchor":anchor,"identity":identity,"source_head_unchanged":true});
        super::restart_transaction::publish_branch_metadata(&target.root,std::path::Path::new("CHRONICLE_BRANCH_ORIGIN.json"),&serde_json::to_vec(&origin)?)?;
        target.publish_head_named("CHRONICLE_HEAD",anchor,&mut |_|Ok(()))?;Ok(anchor.clone())
    }
}

impl CheckpointStore {
    pub fn chronicle_branch_allocation_bound(&self,anchor:&Head,identity:&CheckpointIdentity)->Result<u64> {
        let mut cursor=Some(anchor.clone());let mut count=0usize;let mut bound=4*1024*1024u64;let mut objects=std::collections::BTreeSet::new();
        while let Some(head)=cursor {count+=1;ensure!(count<=100000,"Chronicle allocation ancestry exceeds bounded admission");
            let foliage=self.read_foliage(&head)?;ensure!(foliage.identity==*identity,"Chronicle allocation identity changed");
            bound=bound.checked_add(super::restart_transaction::branch_copy_member_bound(0)?).context("Chronicle manifest link bound overflow")?;
            for digest in foliage.leaves.iter().map(|leaf|&leaf.object_sha256).chain(foliage.consumers.iter().map(|ack|&ack.receipt_sha256)) {
                if objects.insert(digest.clone()) {bound=bound.checked_add(super::restart_transaction::branch_copy_member_bound(0)?).context("Chronicle object link bound overflow")?;}
                ensure!(objects.len()<=262144,"Chronicle allocation object metadata exceeds bounded admission");
            }
            if let Some(previous)=&foliage.previous_foliage {ensure!(previous.generation<head.generation,"Chronicle allocation ancestry does not decrease");}cursor=foliage.previous_foliage;
        }Ok(bound)
    }
}
