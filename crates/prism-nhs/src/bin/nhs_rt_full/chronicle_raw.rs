//! Native 96-byte event preservation at the existing, completed readout boundary.
//! Never changes MD control, selector thresholds, or the downstream execution route.
//! Original order means the native queue's atomic reservation order, not causal order.
use anyhow::{ensure, Context, Result};
use cudarc::driver::DevicePtr;
use std::os::unix::fs::OpenOptionsExt;
use prism_nhs::fused_engine::ObserverPhysicalTimeRecord as PhysicalTimeRecord;
use prism_nhs::fused_engine::{GpuSpikeEvent, NhsAmberFusedEngine, PRISM_SPIKE_RECORD_BYTES};
use prism_nhs::scientific_checkpoint::chronicle::*;
use prism_nhs::scientific_checkpoint::{CheckpointIdentity, CheckpointStore, Head};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, path::Path, sync::Mutex};

#[derive(Debug)]
pub struct PlannedDurableStop;
impl std::fmt::Display for PlannedDurableStop {
    fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result {
        f.write_str("PLANNED_DURABLE_STOP: all workers committed observation frontiers; campaign intentionally interrupted before PathB")
    }
}
impl std::error::Error for PlannedDurableStop {}

const REPLICA_EPOCH: u32 = 1; // Fresh constructor0, then the one actual reset_for_replica.
const RAW: u32 = 0;
const CLOCK: u32 = 1;
const SELECTION: u32 = 2;
const DRAIN: u32 = 3;
const QUEUE: u32 = 4;
const QUEUE_BYTES:usize=80;
const DRAIN_BYTES: usize = 112;
const SOURCE_DECLARED_QUEUE_CAPACITY: u64 = 2_000_000;
const PINNED_STORE_MANIFEST_LIMIT: u64 = 16 * 1024 * 1024;
const MAX_SLAB_BYTES: usize = 4 * 1024 * 1024;
const RAW_RECORDS_PER_SLAB: usize = MAX_SLAB_BYTES / PRISM_SPIKE_RECORD_BYTES;
const MAP_RECORDS_PER_SLAB: usize = MAX_SLAB_BYTES / 24;
const PHASES: [&str; 6] = [
    "cold_hold",
    "ramp_up",
    "warm_hold",
    "ramp_down",
    "cold_return",
    "post_protocol",
];
const CONSUMER: &str = "native_pathb_channel_selection";
// The same raw slot has different meanings for each source. It is never globally
// labelled as energy, charged-atom count, or water density. Inactive residue
// slots are preserved bits and acquire no membership merely by being stored.
const SOURCE_ROLES: &str = r#"{"layout":"GpuSpikeEvent96.le.v1","discriminator":{"offset":60,"encoding":"i32.le"},"sources":{"1":{"name":"UV","vibrational_energy":"aromatic_reservoir_kcal_per_mol","n_nearby_excited":"excited_target_count"},"2":{"name":"LIF","vibrational_energy":"aromatic_reservoir_kcal_per_mol","n_nearby_excited":"excited_target_count"},"3":{"name":"EFP","vibrational_energy":"abs_phi_minus_phi_previous_not_energy","n_nearby_excited":"nearby_charged_atom_count","aromatic_type":"polar_sign_codes_5_6"},"4":{"name":"LADD","intensity":"depletion","water_density":"current_atomic_density","vibrational_energy":"displacement_coherence","n_nearby_excited":"adjacent_depleted_voxel_count","wd_change":"departed_atom_count_as_f32","aromatic_residue_id":"maximum_motion_residue"},"5":{"name":"COFIRE","intensity":"depletion","water_density":"current_atomic_density","vibrational_energy":"displacement_coherence","n_nearby_excited":"adjacent_depleted_voxel_count","wd_change":"departed_atom_count_as_f32","aromatic_residue_id":"maximum_motion_residue"}},"unknown_source":"raw_bits_retained_semantic_role_unavailable","nearby_residues":"only_declared_active_slots_up_to_8_have_membership","order":"per_stream_native_queue_reservation_not_cross_stream_causality"}"#;
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
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
fn raw_components() -> Vec<ComponentDescriptor> {
    let mut c = vec![
        component(
            0,
            "protocol_step",
            0,
            4,
            "i32.le",
            "protocol_local_step",
            "step",
        ),
        component(
            1,
            "voxel_idx",
            4,
            4,
            "i32.le",
            "native_voxel_index",
            "index",
        ),
    ];
    for k in 0..3 {
        c.push(component(
            2 + k,
            &format!("position_{k}"),
            8 + 4 * k,
            4,
            "f32.le",
            "event_position",
            "angstrom",
        ));
    }
    c.push(component(
        5,
        "intensity",
        20,
        4,
        "f32.le",
        "source_discriminated_intensity",
        "source_contract",
    ));
    for k in 0..8 {
        c.push(component(
            6 + k,
            &format!("nearby_residue_{k}"),
            24 + 4 * k,
            4,
            "i32.le",
            "conditionally_active_residue_slot",
            "index",
        ));
    }
    for (id, name, off, enc, role, unit) in [
        (
            14,
            "n_residues",
            56,
            "i32.le",
            "active_residue_slot_count",
            "count",
        ),
        (
            15,
            "spike_source",
            60,
            "i32.le",
            "source_discriminator",
            "code",
        ),
        (
            16,
            "wavelength_nm",
            64,
            "f32.le",
            "source_discriminated_wavelength",
            "source_contract",
        ),
        (
            17,
            "aromatic_type",
            68,
            "i32.le",
            "source_discriminated_type_or_polar_sign",
            "source_contract",
        ),
        (
            18,
            "aromatic_residue_id",
            72,
            "i32.le",
            "source_discriminated_residue_identity",
            "source_contract",
        ),
        (
            19,
            "water_density",
            76,
            "f32.le",
            "source_discriminated_density",
            "source_contract",
        ),
        (
            20,
            "vibrational_energy",
            80,
            "f32.le",
            "source_discriminated_reservoir_or_phi_change_or_coherence",
            "source_contract",
        ),
        (
            21,
            "n_nearby_excited",
            84,
            "i32.le",
            "source_discriminated_population_count",
            "source_contract",
        ),
        (
            22,
            "wd_change",
            88,
            "f32.le",
            "source_discriminated_density_change_or_departed_count",
            "source_contract",
        ),
        (
            23,
            "phase_bits",
            92,
            "u32.le",
            "native_ccns_phase_bits",
            "code",
        ),
    ] {
        c.push(component(id, name, off, 4, enc, role, unit));
    }
    c
}
fn dictionary(identity: &CheckpointIdentity, seeds: &[u64]) -> ChronicleDictionary {
    let source = digest(include_bytes!("chronicle_raw.rs"));
    let fields = vec![
        DatumDescriptor {
            id: RAW,
            owner: "native_fused_spike_queue".into(),
            name: "mixed_channel_wire_events".into(),
            class: DatumClass::Observation,
            quantity_role: "source_discriminated_native_event".into(),
            unit: "source_contract".into(),
            encoding: format!("opaque.sha256:{}", digest(SOURCE_ROLES.as_bytes())),
            element_bytes: 96,
            elements_per_record: 1,
            shape: vec![1],
            subject: SOURCE_ROLES.into(),
            index_order: "native_queue_reservation_order_across_all_channels".into(),
            record_clock: RecordClockBinding::NativeStepSidecar {
                record_step_component: 0,
                sidecar_field: CLOCK,
                sidecar_step_component: 0,
                sidecar_physical_time_component: 2,
            },
            original_ordinal: OriginalOrdinalBinding::NativeEmission,
            components: raw_components(),
        },
        DatumDescriptor {
            id: CLOCK,
            owner: "native_physical_clock".into(),
            name: "complete_physical_step_clock".into(),
            class: DatumClass::Observation,
            quantity_role: "actual_executed_dt_and_cumulative_native_clock".into(),
            unit: "mixed_step_ps".into(),
            encoding: format!(
                "opaque.sha256:{}",
                digest(b"step:u64.le@0,applied_dt:f64.le@8,cumulative_time:f64.le@16;24B.v1")
            ),
            element_bytes: 24,
            elements_per_record: 1,
            shape: vec![1],
            subject: "every_executed_physical_step_including_zero_event_steps".into(),
            index_order: "native_physical_step_order; ordinal0_is_completed_step1".into(),
            record_clock: RecordClockBinding::Embedded {
                step_component: 0,
                physical_time_component: 2,
            },
            original_ordinal: OriginalOrdinalBinding::NativeEmission,
            components: vec![
                component(0, "step", 0, 8, "u64.le", "protocol_local_step", "step"),
                component(
                    1,
                    "applied_dt_ps",
                    8,
                    8,
                    "f64.le",
                    "actual_integrator_dt",
                    "ps",
                ),
                component(
                    2,
                    "physical_time_ps",
                    16,
                    8,
                    "f64.le",
                    "native_cumulative_physical_time",
                    "ps",
                ),
            ],
        },
        DatumDescriptor {
            id: SELECTION,
            owner: "native_pathb_channel_selection".into(),
            name: "consumer_order_to_original_event_ordinal".into(),
            class: DatumClass::Derived,
            quantity_role: "lossless_selection_and_permutation_map".into(),
            unit: "mixed_ordinal_step_ps".into(),
            encoding: format!(
                "opaque.sha256:{}",
                digest(b"raw_ordinal:u64.le@0,step:u64.le@8,time:f64.le@16;24B.v1")
            ),
            element_bytes: 24,
            elements_per_record: 1,
            shape: vec![1],
            subject: "selected_PathB_input_event_in_consumer_order".into(),
            index_order: "exact_existing_channel_selector_output_order".into(),
            record_clock: RecordClockBinding::Embedded {
                step_component: 1,
                physical_time_component: 2,
            },
            original_ordinal: OriginalOrdinalBinding::NativeEmission,
            components: vec![
                component(
                    0,
                    "original_ordinal",
                    0,
                    8,
                    "u64.le",
                    "source_record_ordinal",
                    "index",
                ),
                component(1, "step", 8, 8, "u64.le", "protocol_local_step", "step"),
                component(
                    2,
                    "physical_time_ps",
                    16,
                    8,
                    "f64.le",
                    "native_cumulative_physical_time",
                    "ps",
                ),
            ],
        },
        DatumDescriptor{id:DRAIN,owner:"native_fused_spike_readout".into(),name:"completed_raw_drain_accounting".into(),class:DatumClass::Observation,
            quantity_role:"independent_attempted_capacity_written_and_frontier_accounting".into(),unit:"mixed_count_step_ps".into(),
            encoding:format!("opaque.sha256:{}",digest(b"end_step:u64@0,attempted:i64@8,capacity:u64@16,captured:u64@24,raw_first:u64@32,raw_next:u64@40,clock_first:u64@48,clock_next:u64@56,clock_first_step:u64@64,clock_last_step:u64@72,first_time:f64@80,last_time:f64@88,raw_bytes:u64@96,status:u64@104;112B.le.v1")),
            element_bytes:DRAIN_BYTES as u32,elements_per_record:1,shape:vec![1],subject:"one_verified_readout_attempt_before_each_existing_source_reset_including_final_and_zero_event_drains".into(),index_order:"per_stream_completed_readout_order; not_global_causality".into(),
            record_clock:RecordClockBinding::Embedded{step_component:0,physical_time_component:11},original_ordinal:OriginalOrdinalBinding::NativeEmission,
            components:vec![component(0,"end_step",0,8,"u64.le","native_completed_step","step"),component(1,"attempted",8,8,"i64.le","actual_device_attempted_event_count","count"),component(2,"capacity",16,8,"u64.le","actual_allocated_event_capacity","count"),component(3,"captured",24,8,"u64.le","successfully_captured_event_count","count"),component(4,"raw_first",32,8,"u64.le","raw_ordinal_frontier_before_readout","index"),component(5,"raw_next",40,8,"u64.le","raw_ordinal_frontier_after_readout","index"),component(6,"clock_first",48,8,"u64.le","clock_ordinal_frontier_before_readout","index"),component(7,"clock_next",56,8,"u64.le","clock_ordinal_frontier_after_readout","index"),component(8,"clock_first_step",64,8,"u64.le","first_new_native_clock_step_or_endpoint","step"),component(9,"clock_last_step",72,8,"u64.le","last_native_clock_step","step"),component(10,"first_physical_time_ps",80,8,"f64.le","first_new_native_time_or_endpoint","ps"),component(11,"last_physical_time_ps",88,8,"f64.le","native_completed_physical_time","ps"),component(12,"source_bytes",96,8,"u64.le","actual_captured_original_wire_bytes","byte"),component(13,"status",104,8,"u64.le","zero_means_count_and_capacity_verified_complete_readout","code")]},

        DatumDescriptor{id:QUEUE,owner:"native_fused_spike_readout".into(),name:"native_queue_prefix_accounting".into(),class:DatumClass::Observation,
            quantity_role:"append_only_queue_epoch_prefix_and_existing_reset_admission".into(),unit:"mixed_count_step_ps".into(),
            encoding:format!("opaque.sha256:{}",digest(b"end_step:u64,attempted:i64,capacity:u64,prior_prefix:u64,next_prefix:u64,queue_raw_first:u64,raw_next:u64,queue_epoch:u64,existing_reset_expected:u64,time:f64;80B.le.v1")),
            element_bytes:80,elements_per_record:1,shape:vec![1],subject:"one_verified_readout; partial_capture_preserves_source_queue".into(),
            index_order:"per_stream_native_queue_epoch_and_readout_order".into(),record_clock:RecordClockBinding::Embedded{step_component:0,physical_time_component:9},original_ordinal:OriginalOrdinalBinding::NativeEmission,
            components:vec![component(0,"end_step",0,8,"u64.le","native_completed_step","step"),component(1,"attempted",8,8,"i64.le","device_attempted_events","count"),
                component(2,"capacity",16,8,"u64.le","allocated_queue_capacity","count"),component(3,"prior_prefix",24,8,"u64.le","previously_durable_unreset_queue_prefix","count"),
                component(4,"next_prefix",32,8,"u64.le","durable_unreset_queue_prefix","count"),component(5,"queue_raw_first",40,8,"u64.le","original_queue_first_raw_ordinal","index"),
                component(6,"raw_next",48,8,"u64.le","durable_raw_next_ordinal","index"),component(7,"queue_epoch",56,8,"u64.le","existing_device_count_reset_epoch","index"),
                component(8,"existing_reset_expected",64,8,"u64.le","one_requires_existing_owner_reset_ack_before_next_propagation","boolean"),component(9,"time",72,8,"f64.le","native_cumulative_physical_time","ps")]},

    ];
    let mut contexts = Vec::new();
    for (stream, &seed) in seeds.iter().enumerate() {
        for (phase, name) in PHASES.iter().enumerate() {
            contexts.push(DatumContext {
                id: (stream * 7 + phase) as u32,
                stream_id: stream as u32,
                replica_epoch: REPLICA_EPOCH,
                replica_seed: seed,
                numerical_phase: "completed_native_window_readout; per_event_kernel_and_numerical_subphase_unresolved".into(),
                thermal_phase: format!("configured_schedule_classification:{name}"),
                clock_domain: "owning_ProtocolState.current_step_to_exact_native_physical_time"
                    .into(),
                source_lineage_sha256: source.clone(),
            });
        }
        contexts.push(DatumContext {
            id: (stream * 7 + 6) as u32,
            stream_id: stream as u32,
            replica_epoch: REPLICA_EPOCH,
            replica_seed: seed,
            numerical_phase: "post_md_native_channel_selection".into(),
            thermal_phase: "configured_schedule_classification_retained_by_original_record_lineage"
                .into(),
            clock_domain: "original_event_step_clock_retained_through_selection".into(),
            source_lineage_sha256: source.clone(),
        });
    }
    let _ = identity;
    ChronicleDictionary {
        fields,
        contexts,
        consumers: vec![
            ConsumerContract {
                name: CONSUMER.into(),
                implementation_sha256: source,
                required_fields: BTreeSet::from([RAW,CLOCK,DRAIN,QUEUE]),
            },
            ConsumerContract {
                name: "native_pathb_site_materialization".into(),
                implementation_sha256: identity.engine_sha256.clone(),
                required_fields: BTreeSet::from([SELECTION]),
            },
            ConsumerContract {
                name: "native_arrow_export".into(),
                implementation_sha256: identity.engine_sha256.clone(),
                required_fields: BTreeSet::from([SELECTION]),
            },
        ],
    }
}
/// Compile-time ABI plus the selected native module's real entry declarations
/// are checked before CUDA creation. Actual symbol binding and zero-step state
/// checks still execute against the real owning module before any MD launch.
pub fn check_observation_continuation(
    scientific_checkpoint_requested: bool,
    resume_requested: bool,
) -> Result<()> {
    ensure!(!scientific_checkpoint_requested&&!resume_requested,"native observation Chronicle host/store cursor closure is not in the scientific capsule; refusing checkpoint/resume combination before CUDA");
    Ok(())
}
pub fn preflight_selected_module(path: &Path) -> Result<()> {
    ensure!(
        prism_nhs::fused_engine::OBSERVER_PHYSICAL_CLOCK_ABI == 1,
        "unknown native observer physical clock ABI"
    );
    let text = std::fs::read_to_string(path).context(
        "native observer requires an inspectable selected PTX module before GPU creation",
    )?;
    let entries = ptx_entries(&text);
    for name in [
        "nhs_observer_physical_clock_begin",
        "nhs_observer_physical_clock_commit",
    ] {
        ensure!(
            entries.contains(name),
            "selected native module lacks exact observer clock entry {name}; refusing GPU launch"
        );
    }
    Ok(())
}
fn ptx_entries(text: &str) -> BTreeSet<String> {
    let input = text.as_bytes();
    let mut clean = Vec::with_capacity(input.len());
    let mut i = 0;
    while i < input.len() {
        if input[i..].starts_with(b"//") {
            while i < input.len() && input[i] != b'\n' {
                i += 1;
            }
            clean.push(b' ');
        } else if input[i..].starts_with(b"/*") {
            i += 2;
            while i < input.len() && !input[i..].starts_with(b"*/") {
                i += 1;
            }
            i = (i + 2).min(input.len());
            clean.push(b' ');
        } else if input[i] == b'"' {
            i += 1;
            while i < input.len() {
                if input[i] == b'\\' {
                    i = (i + 2).min(input.len());
                } else if input[i] == b'"' {
                    i += 1;
                    break;
                } else {
                    i += 1;
                }
            }
            clean.push(b' ');
        } else {
            clean.push(input[i]);
            i += 1;
        }
    }
    let text =
        String::from_utf8(clean).expect("removing complete PTX comments/strings preserves UTF8");
    let mut tokens = text.split_whitespace();
    let mut entries = BTreeSet::new();
    while let Some(token) = tokens.next() {
        if token == ".entry" {
            if let Some(name) = tokens.next() {
                entries.insert(name.split('(').next().unwrap_or("").to_string());
            }
        }
    }
    entries
}
// This is a quota-based admission, not a claim that every physically possible
// native event population fits the selected quota. All capacity/size assumptions
// are exposed; the actual native queue allocation is verified before graph capture.
/// Explicit authority chosen before any graph capture. This enum adds no
/// independent clock state: existing observer or guard continuation owns all
/// counters/rows. The compiled recipe and recorded admission bind the choice.
#[derive(Clone,Copy,Debug,PartialEq,Eq,serde::Serialize)]
#[serde(rename_all="snake_case")]
pub enum RawClockAuthorityV1 { StandaloneObserver, UnarmedPhysicalGuard }
pub fn verify_clock_prelaunch(owner:&NhsAmberFusedEngine,authority:RawClockAuthorityV1,stream_id:usize,capacity:u64)
    ->Result<serde_json::Value> {
    ensure!(owner.device_protocol_current_step()?==0,"raw clock authority must be admitted at zero physical steps");
    verify_source_layout_prelaunch(owner)?;
    let native=match authority {
        RawClockAuthorityV1::StandaloneObserver=>{
            ensure!(owner.observer_physical_clock_bound() && owner.governor_device().is_none(),"standalone raw clock has another physical owner");
            ensure!(owner.observer_physical_time_records(0,0)?.is_empty(),"standalone zero-step clock is not empty");
            serde_json::json!({"capacity_steps":capacity,"completed_step":0,"initial_time_ps_bits":0})
        },
        RawClockAuthorityV1::UnarmedPhysicalGuard=>{
            let state=owner.chronicle_unarmed_guard_clock_state(capacity)?;
            ensure!(state.source_stream_id==u32::try_from(stream_id)? && state.completed_step==0,"raw guard-clock stream/initial frontier differs");
            serde_json::to_value(state)?
        },
    };
    Ok(serde_json::json!({"schema":"prism.raw_native_clock_authority.v1","authority":authority,"stream_id":stream_id,
        "native":native,"single_authoritative_journal":true,"standalone_observer_and_guard_exclusive":true,
        "record_bytes":24,"time_unit":"ps","step_authority":"ProtocolState.current_step",
        "dt_authority":"original native applied-f32 dt widened to f64 before original native clock accumulation",
        "bit_transport":"source step/applied_dt/cumulative_time unchanged; no nominal dt-times-step reconstruction",
        "continuation_owner":match authority{RawClockAuthorityV1::StandaloneObserver=>"existing fused observer state/rows/capacity",RawClockAuthorityV1::UnarmedPhysicalGuard=>"existing GovernorContinuationV1 full private guard/time/energy history and native writer cursors"}}))
}
pub fn read_physical_clock(owner:&NhsAmberFusedEngine,authority:RawClockAuthorityV1,start:u64,end:u64,capacity:u64,
    paired_time_rows:&[prism_nhs::governor_device::PhysicalTimeRecord])->Result<Vec<PhysicalTimeRecord>> {
    let rows=match authority {
        RawClockAuthorityV1::StandaloneObserver=>owner.observer_physical_time_records(start,end)?,
        RawClockAuthorityV1::UnarmedPhysicalGuard=>owner.chronicle_unarmed_guard_time_records(start,end,capacity)?,
    };
    if authority==RawClockAuthorityV1::UnarmedPhysicalGuard {
        ensure!(paired_time_rows.len() as u64==end,"raw clock and original time/energy writer frontier differ");
        let paired=paired_time_rows.get(usize::try_from(start)?..usize::try_from(end)?).context("raw clock writer prefix missing")?;
        ensure!(rows.len()==paired.len() && rows.iter().zip(paired).all(|(raw,native)|raw.step==native.step
            && raw.applied_dt_ps.to_bits()==native.applied_dt_ps.to_bits()
            && raw.cumulative_simulated_time_ps.to_bits()==native.cumulative_simulated_time_ps.to_bits()),
            "raw Chronicle and original guard time/energy records are not bit-identical");
    }Ok(rows)
}

/// May's standalone observer reuses the exact owner's synchronized endpoint.
/// Guard-owned clocks retain their existing independent physical authority.
pub fn read_verified_autonomous_physical_clock(
    endpoint:&prism_nhs::persistent_engine::VerifiedAutonomousChunkEndpoint<'_>,
    authority:RawClockAuthorityV1,start:u64,end:u64,
)->Result<Vec<PhysicalTimeRecord>> {
    ensure!(authority==RawClockAuthorityV1::StandaloneObserver,
        "autonomous May endpoint requires its admitted standalone physical observer");
    endpoint.observer_physical_time_records(start,end)
}

pub fn verify_source_layout_prelaunch(owner: &NhsAmberFusedEngine) -> Result<()> {
    ensure!(
        owner.spike_buffer_gpu().len() as u64 == SOURCE_DECLARED_QUEUE_CAPACITY * 96,
        "actual native event allocation differs from the source-pinned prelaunch model"
    );
    Ok(())
}
fn resource_model(
    streams: usize,
    horizon: u64,
    chunk: u64,
    quota: u64,
) -> Result<serde_json::Value> {
    ensure!(
        streams > 0 && horizon > 0 && chunk > 0 && quota > 0,
        "empty Chronicle resource admission"
    );
    let mul = |a: u64, b: u64| {
        a.checked_mul(b)
            .context("Chronicle resource product overflow")
    };
    let add = |a: u64, b: u64| a.checked_add(b).context("Chronicle resource sum overflow");
    let ceiling = |n: u64, d: u64| {
        n.checked_add(d - 1)
            .context("Chronicle resource ceiling overflow")
            .map(|v| v / d)
    };
    let drains_per_stream = add(ceiling(horizon, chunk)?, 1)?; // includes the actual final readout
    let drains = mul(streams as u64, drains_per_stream)?;
    let queue_readouts=add(drains,streams as u64)?; // at most one explicitly requested partial anchor per stream/run
    let publications = add(add(queue_readouts, mul(streams as u64, 2)?)?, 1)?; // source, selected maps, selection acks, Arrow ack
    let clock_rows = mul(streams as u64, horizon)?;
    let clock_bytes = mul(clock_rows, 24)?;
    let drain_bytes = add(mul(drains,DRAIN_BYTES as u64)?,mul(queue_readouts,QUEUE_BYTES as u64)?)?;
    let minimum = add(add(clock_bytes, drain_bytes)?, PINNED_STORE_MANIFEST_LIMIT)?; // receipt reserve
    ensure!(
        quota >= minimum,
        "Chronicle quota cannot contain known complete clock/drain populations and receipt reserve"
    );
    let kernel_capacity_events = mul(drains, SOURCE_DECLARED_QUEUE_CAPACITY)?;
    let kernel_capacity_payload = add(
        add(mul(kernel_capacity_events, 120)?, clock_bytes)?,
        drain_bytes,
    )?; // raw96+max selected24
    let raw_leaves = mul(
        drains,
        add(
            ceiling(SOURCE_DECLARED_QUEUE_CAPACITY, RAW_RECORDS_PER_SLAB as u64)?,
            5,
        )?,
    )?;
    let clock_leaves = mul(drains, add(ceiling(chunk, MAX_SLAB_BYTES as u64 / 24)?, 5)?)?;
    let selected_leaves = mul(
        streams as u64,
        ceiling(
            mul(drains_per_stream, SOURCE_DECLARED_QUEUE_CAPACITY)?,
            MAP_RECORDS_PER_SLAB as u64,
        )?,
    )?;
    // Once per chunk, rather than once per raw slab, under the pinned16MiB
    // cumulative manifest cap. This bounds retained version bytes, not wall time.
    let manifest_rewrite_upper = mul(publications, PINNED_STORE_MANIFEST_LIMIT)?;
    let orphan_upper = mul(
        streams as u64,
        add(
            mul(SOURCE_DECLARED_QUEUE_CAPACITY, 96)?,
            MAX_SLAB_BYTES as u64,
        )?,
    )?;
    let disk_required = add(
        add(add(quota, manifest_rewrite_upper)?, orphan_upper)?,
        1024 * 1024 * 1024,
    )?;
    Ok(
        serde_json::json!({"schema":"native_md_chronicle_resource_admission.v1","admission_kind":"explicit_logical_quota_plus_disk_and_bounded_work; not_event_density_fit_guarantee",
        "streams":streams,"physical_horizon_per_stream":horizon,"physical_chunk_steps":chunk,"maximum_completed_source_drains":drains,"maximum_queue_readouts_including_one_partial_per_stream":queue_readouts,"source_queue_prefix_witness_bytes":QUEUE_BYTES,
        "maximum_foliage_publications":publications,"source_publications_per_native_chunk":1,"selection_publications_per_stream":1,
        "source_pinned_queue_capacity_records":SOURCE_DECLARED_QUEUE_CAPACITY,"observed_queue_capacity_pre_graph_check":"required_equal_to_source_bound",
        "clock_device_bytes":add(clock_bytes,mul(streams as u64,64)?)?,"clock_host_bytes":clock_bytes,
        "raw_temporary_bytes_upper":mul((RAW_RECORDS_PER_SLAB*96)as u64,2)?,"streaming_leaf_bytes_limit":MAX_SLAB_BYTES,
        "configured_logical_population_quota_bytes":quota,"known_clock_drain_and_receipt_reserve_bytes":minimum,
        "kernel_capacity_worst_case_events":kernel_capacity_events,"kernel_capacity_worst_case_payload_bytes":kernel_capacity_payload,
        "worst_case_payload_exceeds_quota":kernel_capacity_payload>quota,
        "theoretical_leaf_counts":{"raw":raw_leaves,"clock":clock_leaves,"drain":drains,"selection":selected_leaves},
        "metadata_manifest_limit_bytes":PINNED_STORE_MANIFEST_LIMIT,"cumulative_metadata_rewrite_bytes_upper":manifest_rewrite_upper,
        "metadata_admission":"actual source-owned store rejects excessive leaf/parent/consumer metadata before publication; kernel-capacity population is not assumed to fit this ceiling",
        "disk_required_available_bytes":disk_required,"includes_disk_reserve_bytes":1024*1024*1024u64,"failure_orphan_bytes_upper":orphan_upper,
        "quota_exhaustion":"refuse publication/source reset; retain committed prefix; stop run without accepting complete population",
        "ambiguous_commit_error":"after atomic HEAD rename an error may expose the complete new head; do not retry or reset source; inspect durable head",
        "performance_claim":"metadata/object work bounds only; actual MD wall time and I/O overhead require measured native execution"}),
    )
}
pub fn admit_resources_with_clock_authority(root:&Path,streams:usize,horizon:u64,chunk:u64,quota:u64,
    authority:RawClockAuthorityV1)->Result<serde_json::Value> {
    let mut plan=admit_resources(root,streams,horizon,chunk,quota)?;
    plan["clock_authority"]=serde_json::to_value(authority)?;
    plan["clock_device_bytes_scope"]=serde_json::json!("additional allocations owned exclusively by raw Chronicle");
    if authority==RawClockAuthorityV1::UnarmedPhysicalGuard {
        let shared=(streams as u64).checked_mul(horizon).and_then(|n|n.checked_mul(184)).context("shared native guard journal byte count overflow")?;
        plan["clock_device_bytes"]=serde_json::json!(0);
        plan["shared_guard_time_energy_device_bytes"]=serde_json::json!(shared);
        plan["shared_guard_allocation_scope"]=serde_json::json!("existing GovernorDevice bind_physical_time owns one24B time plus160B energy record per step; counted once by native construction admission, not a second raw clock allocation");
    }Ok(plan)
}
pub fn admit_resources(
    root: &Path,
    streams: usize,
    horizon: u64,
    chunk: u64,
    quota: u64,
) -> Result<serde_json::Value> {
    use std::os::unix::ffi::OsStrExt;
    ensure!(root.is_absolute(), "Chronicle output root must be absolute");
    let parent = root.parent().context("Chronicle root has no parent")?;
    let path = std::ffi::CString::new(parent.as_os_str().as_bytes())?;
    let mut stats = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    let rc = unsafe { libc::statvfs(path.as_ptr(), stats.as_mut_ptr()) };
    ensure!(
        rc == 0,
        "cannot inspect Chronicle filesystem admission: {}",
        std::io::Error::last_os_error()
    );
    let stats = unsafe { stats.assume_init() };
    let available = (stats.f_bavail as u64)
        .checked_mul(stats.f_frsize as u64)
        .context("Chronicle filesystem byte count overflow")?;
    let mut plan = resource_model(streams, horizon, chunk, quota)?;
    let required = plan["disk_required_available_bytes"]
        .as_u64()
        .context("invalid Chronicle disk resource plan")?;
    ensure!(available>=required,"Chronicle filesystem free capacity {available}B is below source/metadata/orphan quota admission {required}B");
    plan["observed_filesystem_available_bytes"] = serde_json::json!(available);
    plan["filesystem_parent"] = serde_json::json!(parent);
    plan["admission_passed"] = serde_json::json!(true);
    Ok(plan)
}
/// Host producer/consumer frontiers at the existing all-worker pause. Payload
/// objects remain in the original Chronicle CAS; they are referenced, not copied.
#[derive(Clone,Debug,serde::Serialize,serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawChronicleContinuationV2 {
    pub schema_version:u32,
    pub identity:CheckpointIdentity,
    pub dictionary:ChronicleDictionary,
    pub head:Option<Head>,
    pub ordinals:Vec<u64>,
    pub clock_ordinals:Vec<u64>,
    pub drain_ordinals:Vec<u64>,
    pub queue_ordinals:Vec<u64>,pub queue_prefixes:Vec<QueuePrefixContinuationV1>,
    pub seeds:Vec<u64>,
    pub boundaries:Vec<RawProducerBoundaryV1>,
    pub common_next_chunk:i32,
    pub checkpoint_ordinal:u64,
}
#[derive(Clone,Debug,serde::Serialize,serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawProducerBoundaryV1 {
    pub stream_id:u32,pub completed_step:u64,pub physical_time_ps_bits:u64,
}
#[derive(Clone,Debug,Default,serde::Serialize,serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueuePrefixContinuationV1 {pub epoch:u64,pub captured_records:u64,pub awaiting_existing_reset:bool}
struct State {
    store: CheckpointStore,
    identity: CheckpointIdentity,
    dictionary: ChronicleDictionary,
    head: Option<Head>,
    ordinals: Vec<u64>,
    clock_ordinals: Vec<u64>,
    drain_ordinals: Vec<u64>,
    queue_ordinals:Vec<u64>,queue_prefixes:Vec<QueuePrefixContinuationV1>,
    native_archive_witnesses:Vec<Option<prism_nhs::scientific_checkpoint::native_spike_archive::NativeSpikeArchiveWitness>>,
    seeds: Vec<u64>,
}
pub struct RawChronicle {
    state: Mutex<State>,
}
impl RawChronicle {
    pub fn open(
        root: &Path,
        max_bytes: u64,
        identity: CheckpointIdentity,
        seeds: Vec<u64>,
    ) -> Result<Self> {
        ensure!(
            seeds.len() == identity.stream_count as usize,
            "Chronicle seed population mismatch"
        );
        let store = CheckpointStore::create(root, max_bytes)?;
        ensure!(store.chronicle_head()?.is_none(),"existing observation Chronicle requires qualified explicit continuation; refusing implicit append");
        let dictionary = dictionary(&identity, &seeds);
        dictionary.validate(&identity)?;
        Ok(Self {
            state: Mutex::new(State {
                store,
                identity,
                dictionary,
                head: None,
                ordinals: vec![0; seeds.len()],
                clock_ordinals: vec![0; seeds.len()],
                drain_ordinals: vec![0; seeds.len()],
                queue_ordinals:vec![0;seeds.len()],queue_prefixes:vec![QueuePrefixContinuationV1::default();seeds.len()],native_archive_witnesses:vec![None;seeds.len()],
                seeds,
            }),
        })
    }
    pub fn checkpoint_continuation(
        &self,proof:&prism_nhs::checkpoint_epoch::PausedGeneration<'_>,common_next_chunk:i32,
    )->Result<RawChronicleContinuationV2> {
        let state=self.state.lock().map_err(|_|anyhow::anyhow!("poisoned raw Chronicle"))?;
        ensure!(proof.identity()==&state.identity,"Chronicle checkpoint proof belongs to another execution");
        ensure!(state.store.chronicle_head()?==state.head,"Chronicle source cursor has stale durable HEAD");
        ensure!(common_next_chunk>=0,"negative common native next tick");
        let index=state.head.as_ref().map(|head|proof.validate_chronicle_index(&state.store,head,&state.identity)).transpose()?;
        let mut boundaries=Vec::with_capacity(state.seeds.len());
        for stream in 0..state.seeds.len() {
            let marker=index.as_ref().and_then(|index|index.foliage().producers.iter().find(|p|
                p.producer.field_id==CLOCK && p.producer.stream_id==stream as u32 && p.producer.replica_epoch==REPLICA_EPOCH));
            let (completed_step,physical_time_ps_bits)=marker.map_or((0,0f64.to_bits()),|p|(p.last_step,p.physical_time_ps_bits));
            ensure!(state.clock_ordinals[stream]==completed_step,"raw clock row/physical-step population differs");
            boundaries.push(RawProducerBoundaryV1{stream_id:stream as u32,completed_step,physical_time_ps_bits});
        }
        let saved=RawChronicleContinuationV2{schema_version:2,identity:state.identity.clone(),dictionary:state.dictionary.clone(),
            head:state.head.clone(),ordinals:state.ordinals.clone(),clock_ordinals:state.clock_ordinals.clone(),
            drain_ordinals:state.drain_ordinals.clone(),queue_ordinals:state.queue_ordinals.clone(),queue_prefixes:state.queue_prefixes.clone(),seeds:state.seeds.clone(),boundaries,common_next_chunk,checkpoint_ordinal:proof.ordinal()};
        Self::verify_saved_cursors_with_index(&state.store,&saved,false,index.as_ref())?;
        Ok(saved)
    }
    fn verify_saved_cursors(store:&CheckpointStore,saved:&RawChronicleContinuationV2,verify_payloads:bool)->Result<()> {
        let index=saved.head.as_ref().map(|head|store.validate_chronicle_index(head,&saved.identity)).transpose()?;
        Self::verify_saved_cursors_with_index(store,saved,verify_payloads,index.as_ref())
    }
    fn verify_saved_cursors_with_index(store:&CheckpointStore,saved:&RawChronicleContinuationV2,verify_payloads:bool,
        index:Option<&prism_nhs::scientific_checkpoint::chronicle::ValidatedChronicleIndex>)->Result<()> {
        ensure!(saved.schema_version==2 && saved.common_next_chunk>=0 && saved.boundaries.len()==saved.seeds.len() && saved.seeds.len()==saved.identity.stream_count as usize
            && saved.ordinals.len()==saved.seeds.len() && saved.clock_ordinals.len()==saved.seeds.len()
            && saved.drain_ordinals.len()==saved.seeds.len() && saved.queue_ordinals.len()==saved.seeds.len() && saved.queue_prefixes.len()==saved.seeds.len(),"Chronicle checkpoint cursor population differs");
        ensure!(saved.dictionary==dictionary(&saved.identity,&saved.seeds),"Chronicle implementation/consumer dictionary changed");
        ensure!(store.chronicle_head()?==saved.head,"observation HEAD differs from exact scientific anchor; use explicit branch first");
        let Some(head)=&saved.head else {
            ensure!(index.is_none(),"missing Chronicle HEAD has an unexpected validated index");
            ensure!(saved.boundaries.iter().all(|b|b.completed_step==0 && b.physical_time_ps_bits==0f64.to_bits()) && saved.ordinals.iter().chain(&saved.clock_ordinals).chain(&saved.drain_ordinals).chain(&saved.queue_ordinals).all(|v|*v==0)
                && saved.queue_prefixes.iter().all(|q|q.epoch==0 && q.captured_records==0 && !q.awaiting_existing_reset),
                "missing Chronicle HEAD at nonzero scientific boundary");return Ok(());
        };
        let index=index.context("saved Chronicle cursor validation lacks the actual validated index")?;
        ensure!(index.head()==head && index.foliage().identity==saved.identity,"saved Chronicle index identity/head differs");
        ensure!(index.foliage().dictionary==saved.dictionary,"saved Chronicle dictionary differs from actual durable foliage");
        for stream in 0..saved.seeds.len() {
            let boundary=&saved.boundaries[stream];
            ensure!(boundary.stream_id==stream as u32 && boundary.completed_step==saved.clock_ordinals[stream],"raw per-owner boundary identity differs");
            for (field_id,next) in [(RAW,saved.ordinals[stream]),(CLOCK,saved.clock_ordinals[stream]),(DRAIN,saved.drain_ordinals[stream]),(QUEUE,saved.queue_ordinals[stream])] {
                let producer=ProducerKey{field_id,stream_id:u32::try_from(stream)?,replica_epoch:REPLICA_EPOCH};
                let marker=index.foliage().producers.iter().find(|p|p.producer==producer);
                ensure!(marker.map_or(0,|v|v.next_ordinal)==next,"Chronicle persisted cursor differs from scientific checkpoint");
                if field_id==CLOCK || field_id==QUEUE {
                    ensure!(marker.is_some_and(|p|p.last_step==boundary.completed_step && p.physical_time_ps_bits==boundary.physical_time_ps_bits),"clock/drain lacks exact per-owner scientific checkpoint frontier");
                }
            }
        }
        if verify_payloads {for i in 0..index.foliage().leaves.len(){drop(store.read_indexed_chronicle_payload(&index,i)?);}}
        Ok(())
    }
    /// Reopen only the exact cursor/head population committed by a full native
    /// capsule. A historical source requires explicit immutable branch creation.
    pub fn open_verified_checkpoint(root:&Path,max_bytes:u64,identity:CheckpointIdentity,seeds:Vec<u64>,saved:&RawChronicleContinuationV2)->Result<Self> {
        ensure!(identity==saved.identity && seeds==saved.seeds,"Chronicle replica/provider/execution identity changed on restore");
        let store=CheckpointStore::create(root,max_bytes)?;
        Self::verify_saved_cursors(&store,saved,true)?;
        Ok(Self{state:Mutex::new(State{store,identity,dictionary:saved.dictionary.clone(),head:saved.head.clone(),
            ordinals:saved.ordinals.clone(),clock_ordinals:saved.clock_ordinals.clone(),drain_ordinals:saved.drain_ordinals.clone(),queue_ordinals:saved.queue_ordinals.clone(),queue_prefixes:saved.queue_prefixes.clone(),native_archive_witnesses:vec![None;seeds.len()],seeds})})
    }
    pub fn stage_native_archive(&self,epoch:&prism_nhs::checkpoint_epoch::CheckpointEpoch,engine:&NhsAmberFusedEngine,stream_id:usize,
        plan:&prism_nhs::fused_engine::ScientificFieldPlan)->Result<prism_nhs::scientific_checkpoint::restart_transaction::StagedField> {
        let (identity,head,producer,seed,first,next,prior)={
            let state=self.state.lock().map_err(|_|anyhow::anyhow!("poisoned raw Chronicle"))?;
            ensure!(stream_id<state.seeds.len() && engine.replica_epoch()==REPLICA_EPOCH && engine.replica_seed()==Some(state.seeds[stream_id]),"native archive replica/owner differs");
            let next=state.ordinals[stream_id].checked_sub(state.queue_prefixes[stream_id].captured_records).context("native archive raw/live queue boundary differs")?;
            let first=next.checked_sub(engine.get_accumulated_spikes().len() as u64).context("native host archive exceeds original raw population")?;
            ensure!(plan.present_bytes==Some((next-first)*96),"native archive declared vector extent differs");
            (state.identity.clone(),state.head.clone(),ProducerKey{field_id:RAW,stream_id:stream_id as u32,replica_epoch:REPLICA_EPOCH},state.seeds[stream_id],first,next,state.native_archive_witnesses[stream_id].clone())
        };
        let (instance,mutation_epoch)=engine.native_spike_archive_identity();
        // Do not hold raw-state mutex while acquiring the existing epoch CAS
        // projection mutex: shared owner capture takes those in the other order.
        let (staged,witness)=epoch.stage_native_spike_archive(&plan.contract,head.as_ref(),producer,seed,instance,mutation_epoch,first,next,prior.as_ref(),|offset,bytes|{
            ensure!(bytes.len()%96==0,"native archive comparison has partial records");
            let first=usize::try_from(offset)?;let rows=engine.get_accumulated_spikes().get(first..first+bytes.len()/96).context("native archive source shortened")?;
            for (row,original) in rows.iter().zip(bytes.chunks_exact(96)) {ensure!(row.to_le_bytes().as_slice()==original,"native host datum bits/order differ from original committed raw event");}Ok(())
        })?;
        let mut state=self.state.lock().map_err(|_|anyhow::anyhow!("poisoned raw Chronicle"))?;
        ensure!(state.identity==identity && state.head==head,"raw Chronicle changed during paused native archive projection");
        state.native_archive_witnesses[stream_id]=Some(witness);Ok(staged)
    }
    pub fn run_id(&self) -> Result<String> {
        Ok(self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("poisoned raw Chronicle"))?
            .identity
            .run_id
            .clone())
    }
    /// Caller is between completed graph windows and has launched no further
    /// producer on this stream. Other streams own different event allocations.
    /// Readback completion and CAS publication both precede the existing reset.
    pub fn capture_source(
        &self,
        engine: &NhsAmberFusedEngine,
        stream_id: usize,
        clock: &[PhysicalTimeRecord],
        phase: impl Fn(i32) -> &'static str,
    ) -> Result<usize> {
        self.capture_source_impl(engine,stream_id,clock,phase,true,None)
    }
    pub fn capture_source_preserving_queue(&self,engine:&NhsAmberFusedEngine,stream_id:usize,
        clock:&[PhysicalTimeRecord],phase:impl Fn(i32)->&'static str)->Result<usize> {
        self.capture_source_impl(engine,stream_id,clock,phase,false,None)
    }
    /// Consume the persistent owner's actual synchronized endpoint while its
    /// immutable borrow excludes another physical dispatch or mutable reset.
    /// Queue count, original96-byte records and full physical-time population
    /// still pass the existing strict native readback and durable CAS path.
    pub fn capture_verified_autonomous_source(
        &self, endpoint:&prism_nhs::persistent_engine::VerifiedAutonomousChunkEndpoint<'_>,
        stream_id:usize, clock:&[PhysicalTimeRecord], phase:impl Fn(i32)->&'static str,
    )->Result<usize> {
        self.capture_source_impl(endpoint.native_owner(),stream_id,clock,phase,true,
            Some(endpoint.completed_step()))
    }
    fn capture_source_impl(&self,engine:&NhsAmberFusedEngine,stream_id:usize,clock:&[PhysicalTimeRecord],
        phase:impl Fn(i32)->&'static str,reset_expected:bool,verified_endpoint:Option<u32>)->Result<usize> {
        let stream = engine.stream();
        // Explicit stream synchronization is part of the source lifetime proof.
        let (count_ptr, count_guard) = engine.spike_count_gpu().device_ptr(stream);
        let count_bytes = read_owned_raw(stream, count_ptr, 4)?;
        drop(count_guard);
        let signed = i32::from_le_bytes(count_bytes.as_slice().try_into()?);
        ensure!(
            engine.spike_buffer_gpu().len() % 96 == 0,
            "native event allocation violates96-byte layout"
        );
        let capacity = engine.spike_buffer_gpu().len() / 96;
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("poisoned raw Chronicle"))?;
        ensure!(
            stream_id < state.ordinals.len()
                && engine.replica_epoch() == REPLICA_EPOCH
                && engine.replica_seed() == Some(state.seeds[stream_id]),
            "raw producer replica/stream identity differs"
        );
        let device_step = u64::from(match verified_endpoint {
            Some(step)=>step,
            None=>engine.device_protocol_current_step()?,
        });
        ensure!(
            clock.last().map(|r| r.step) == Some(device_step),
            "raw drain endpoint lacks the actual complete native physical clock"
        );
        let (base, guard) = engine.spike_buffer_gpu().device_ptr(stream);
        let result = publish_capture_range(
            &mut state,
            stream_id,
            signed,
            capacity,
            clock,
            &phase,
            reset_expected,
            &mut |first, n| {
                read_owned_raw(
                    stream,
                    base.checked_add((first * 96) as u64)
                        .context("spike pointer overflow")?,
                    n * 96,
                )
            },
        );
        drop(guard);
        result
    }
    /// Join restored native queue bytes to the exact committed raw suffix before
    /// capturing or replaying any graph. No reset, decode, or phase inference.
    pub fn verify_restored_queue_prefix(&self,engine:&NhsAmberFusedEngine,stream_id:usize)->Result<()> {
        let state=self.state.lock().map_err(|_|anyhow::anyhow!("poisoned raw Chronicle"))?;
        let queue=state.queue_prefixes.get(stream_id).context("restored queue source absent")?;
        ensure!(!queue.awaiting_existing_reset,"scientific restore cannot launch before an outstanding original queue reset");
        let stream=engine.stream();stream.synchronize()?;
        let (count_ptr,count_guard)=engine.spike_count_gpu().device_ptr(stream);
        let count_bytes=read_owned_raw(stream,count_ptr,4)?;drop(count_guard);
        let count=i32::from_le_bytes(count_bytes.as_slice().try_into()?);
        ensure!(count>=0 && count as u64==queue.captured_records,"restored GPU count differs from committed queue prefix");
        if count==0 {return Ok(());}
        ensure!(PRISM_SPIKE_RECORD_BYTES==96 && engine.spike_buffer_gpu().len()%96==0,"restored raw source ABI changed");
        let next=state.ordinals[stream_id];let first=next.checked_sub(queue.captured_records).context("raw queue prefix exceeds committed population")?;
        let index=state.store.validate_chronicle_index(state.head.as_ref().context("queue prefix lacks durable raw HEAD")?,&state.identity)?;
        let (base,guard)=engine.spike_buffer_gpu().device_ptr(stream);
        let mut frontier=first;
        for (leaf_index,leaf) in index.foliage().leaves.iter().enumerate().filter(|(_,leaf)|
            leaf.producer.field_id==RAW && leaf.producer.stream_id==stream_id as u32
            && leaf.producer.replica_epoch==REPLICA_EPOCH && leaf.coverage.next_ordinal>first && leaf.coverage.first_ordinal<next) {
            let start=leaf.coverage.first_ordinal.max(first);let end=leaf.coverage.next_ordinal.min(next);
            ensure!(start==frontier && end>start,"restored raw queue prefix contains an ordinal gap");
            let bytes=state.store.read_indexed_chronicle_payload(&index,leaf_index)?;
            let source_start=usize::try_from(start-leaf.coverage.first_ordinal)?.checked_mul(96).context("raw suffix byte overflow")?;
            let byte_count=usize::try_from(end-start)?.checked_mul(96).context("raw suffix byte overflow")?;
            ensure!(byte_count<=MAX_SLAB_BYTES,"raw suffix exceeds bounded read slab");
            let device_offset=(start-first).checked_mul(96).context("queue suffix pointer overflow")?;
            let actual=read_owned_raw(stream,base.checked_add(device_offset).context("queue suffix pointer overflow")?,byte_count)?;
            ensure!(bytes.get(source_start..source_start+byte_count)==Some(actual.as_slice()),"restored live queue bytes differ from committed original raw suffix");
            frontier=end;
        }
        drop(guard);ensure!(frontier==next,"restored raw queue prefix coverage incomplete");Ok(())
    }
    pub fn acknowledge_existing_queue_reset(&self,engine:&NhsAmberFusedEngine,stream_id:usize)->Result<()> {
        let stream=engine.stream();let (pointer,guard)=engine.spike_count_gpu().device_ptr(stream);
        let bytes=read_owned_raw(stream,pointer,4)?;drop(guard);
        let count=i32::from_le_bytes(bytes.as_slice().try_into()?);
        let mut state=self.state.lock().map_err(|_|anyhow::anyhow!("poisoned raw Chronicle"))?;
        ensure!(stream_id<state.queue_prefixes.len(),"unknown queue reset source");
        let queue=&mut state.queue_prefixes[stream_id];
        if !queue.awaiting_existing_reset {
            ensure!(queue.captured_records==0 && count==0,"unadmitted queue reset acknowledgement");return Ok(());
        }
        ensure!(count==0,"existing source owner did not complete its count reset");
        queue.epoch=queue.epoch.checked_add(1).context("queue reset epoch overflow")?;
        queue.captured_records=0;queue.awaiting_existing_reset=false;Ok(())
    }
    pub fn acknowledge_arrow(
        &self,
        path: &Path,
        writer_receipt: &[u8],
        row_count: usize,
    ) -> Result<()> {
        use std::io::Read;
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("poisoned raw Chronicle"))?;
        let head = state
            .head
            .clone()
            .context("Arrow output has no committed input foliage")?;
        let foliage = state
            .store
            .inspect_chronicle_index(&head, &state.identity)?;
        let inputs = foliage
            .producers
            .iter()
            .filter(|p| p.producer.field_id == SELECTION)
            .collect::<Vec<_>>();
        let expected = inputs.iter().try_fold(0u64, |total, p| {
            total
                .checked_add(p.next_ordinal)
                .context("selected row total overflow")
        })?;
        ensure!(
            expected == row_count as u64,
            "native Arrow output and selected input cardinalities differ"
        );
        let receipt: serde_json::Value = serde_json::from_slice(writer_receipt)?;
        ensure!(
            receipt["run_id"].as_str() == Some(state.identity.run_id.as_str())
                && receipt["row_count"].as_u64() == Some(expected),
            "native Arrow writer receipt identity differs"
        );
        let mut output = std::fs::File::open(path)?;
        ensure!(
            output.metadata()?.is_file(),
            "native Arrow output is not a regular file"
        );
        output.sync_all()?;
        let mut hash = Sha256::new();
        let mut block = vec![0u8; 1024 * 1024];
        let mut bytes = 0u64;
        loop {
            let n = output.read(&mut block)?;
            if n == 0 {
                break;
            }
            hash.update(&block[..n]);
            bytes += n as u64;
        }
        ensure!(
            bytes > 0 && bytes == output.metadata()?.len(),
            "native Arrow output changed or is empty"
        );
        std::fs::File::open(path.parent().context("Arrow output parent absent")?)?.sync_all()?;
        let artifact = serde_json::json!({"relative_consumer":"native_arrow_export","output_path":path,"output_sha256":format!("{:x}",hash.finalize()),"output_bytes":bytes,"writer_receipt":receipt,"input_foliage":head,"selected_population":inputs,"claim":"actual selected native view exported; no raw event reread by Arrow asserted"});
        let acks = inputs
            .iter()
            .map(|p| {
                Ok(ConsumerAck {
                    consumer: "native_arrow_export".into(),
                    producer: p.producer.clone(),
                    next_ordinal: p.next_ordinal,
                    input_foliage: head.clone(),
                    receipt_bytes: serde_json::to_vec(&artifact)?,
                    output_payloads: Vec::new(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if !acks.is_empty() {
            state.head = Some(state.store.commit_chronicle(
                Some(&head),
                &state.identity,
                &state.dictionary,
                &[],
                &acks,
                None,
            )?);
        }
        Ok(())
    }
    /// Acknowledge only the completed native site consumer, after canonical
    /// prediction/publication and optional site exports are durable and Arrow
    /// has independently acknowledged this same complete selected population.
    /// Site neighborhoods may overlap or exclude background; their row sum is
    /// deliberately not substituted for the complete SELECTION input count.
    pub fn acknowledge_site_materialization(
        &self,
        prediction_path: &Path,
        publication_path: &Path,
        selected_counts: &[u64],
        site_json_outputs: &[serde_json::Value],
    ) -> Result<()> {
        use std::io::Read;
        use std::os::unix::fs::MetadataExt;
        fn durable_small_output(path:&Path)->Result<(Vec<u8>,serde_json::Value)> {
            let file=std::fs::OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW).open(path)?;
            let before=file.metadata()?;
            ensure!(before.is_file(),
                "site consumer inventory is not a regular file");
            file.sync_all()?;
            let mut bytes=Vec::with_capacity(usize::try_from(before.len())?);
            (&file).read_to_end(&mut bytes)?;
            let after=file.metadata()?;
            ensure!(before.dev()==after.dev() && before.ino()==after.ino()
                && before.len()==after.len() && before.mtime()==after.mtime()
                && before.mtime_nsec()==after.mtime_nsec() && before.ctime()==after.ctime()
                && before.ctime_nsec()==after.ctime_nsec() && bytes.len() as u64==after.len(),
                "site consumer inventory changed during durable read");
            std::fs::File::open(path.parent().context("site output parent absent")?)?.sync_all()?;
            let pin=serde_json::json!({"path":path,"sha256":digest(&bytes),"bytes":bytes.len()});
            Ok((bytes,pin))
        }
        let mut state=self.state.lock().map_err(|_|anyhow::anyhow!("poisoned raw Chronicle"))?;
        ensure!(selected_counts.len()==state.ordinals.len(),"site selected stream population count differs");
        let head=state.head.clone().context("site consumer has no committed selection foliage")?;
        let foliage=state.store.inspect_chronicle_index(&head,&state.identity)?;
        let inputs=foliage.producers.iter().filter(|p|p.producer.field_id==SELECTION).collect::<Vec<_>>();
        let mut observed=vec![0u64;selected_counts.len()];
        let mut seen=BTreeSet::new();
        for producer in &inputs {
            let sid=usize::try_from(producer.producer.stream_id)?;
            ensure!(sid<observed.len() && producer.producer.replica_epoch==REPLICA_EPOCH
                && seen.insert(sid),"site consumer selection source identity differs");
            observed[sid]=producer.next_ordinal;
            ensure!(foliage.consumers.iter().any(|ack|ack.consumer=="native_arrow_export"
                && ack.producer==producer.producer && ack.next_ordinal==producer.next_ordinal),
                "site consumer lacks completed Arrow acknowledgement of selected population");
        }
        ensure!(observed.as_slice()==selected_counts,"site materializer input and committed selected populations differ");
        if inputs.is_empty() { return Ok(()); }
        let (prediction_bytes,prediction_pin)=durable_small_output(prediction_path)?;
        let prediction:serde_json::Value=serde_json::from_slice(&prediction_bytes)?;
        let (publication_bytes,publication_pin)=durable_small_output(publication_path)?;
        let publication:serde_json::Value=serde_json::from_slice(&publication_bytes)?;
        let sites=prediction["sites"].as_array().context("site consumer final native inventory absent")?;
        ensure!(prediction["run_id"].as_str()==Some(state.identity.run_id.as_str())
            && prediction["n_streams"].as_u64()==Some(selected_counts.len() as u64)
            && prediction["binding_sites"].as_u64()==Some(sites.len() as u64)
            && publication["schema"]=="prism.canonical_binding_site_publication.v1"
            && publication["source_json_sha256"]==prediction_pin["sha256"]
            && publication["sites"]==prediction["sites"]
            && publication["site_count"].as_u64()==Some(sites.len() as u64),
            "site consumer publication does not bind actual final native prediction");
        let parent=publication_path.parent().context("site publication parent absent")?;
        ensure!(prediction_path.parent()==Some(parent),"site publication and prediction roots differ");
        let files=publication["files"].as_array().context("site publication visual inventory absent")?;
        ensure!(files.len()==4,"site publication visual inventory incomplete");
        let mut names=BTreeSet::new();
        let mut visuals=Vec::new();
        for expected in files {
            let name=expected["file"].as_str().context("site publication filename absent")?;
            ensure!(Path::new(name).file_name().and_then(|s|s.to_str())==Some(name)
                && names.insert(name),"site publication path is not one distinct local output");
            let (_,actual)=durable_small_output(&parent.join(name))?;
            ensure!(actual["sha256"]==expected["sha256"] && actual["bytes"]==expected["bytes"],
                "site publication visual bytes differ");
            visuals.push(actual);
        }
        let mut site_ids=BTreeSet::new();
        for output in site_json_outputs {
            ensure!(site_ids.insert(output["site_id"].as_i64().context("site export identity absent")?)
                && output["n_spikes"].as_u64().is_some(),"site export identities/counts differ");
            let path=Path::new(output["path"].as_str().context("site export path absent")?);
            ensure!(path.parent()==Some(parent),"site export root differs from native prediction");
            let file=std::fs::OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW).open(path)?;
            let metadata=file.metadata()?;
            ensure!(metadata.is_file() && Some(metadata.len())==output["bytes"].as_u64()
                && Some(metadata.dev())==output["device"].as_u64()
                && Some(metadata.ino())==output["inode"].as_u64()
                && Some(metadata.mtime())==output["mtime"].as_i64()
                && Some(metadata.mtime_nsec())==output["mtime_nsec"].as_i64()
                && Some(metadata.ctime())==output["ctime"].as_i64()
                && Some(metadata.ctime_nsec())==output["ctime_nsec"].as_i64()
                && output["sha256"].as_str().is_some_and(|s|s.len()==64),
                "completed site export changed since its actual streaming digest and durable publication");
            file.sync_all()?;
        }
        std::fs::File::open(parent)?.sync_all()?;
        let receipt=serde_json::to_vec(&serde_json::json!({
            "schema":"native_pathb_site_materialization.v1","input_foliage":head,
            "selected_population":inputs,"selected_counts_by_stream":selected_counts,
            "prediction":prediction_pin,"canonical_publication":publication_pin,
            "visual_outputs":visuals,"site_json_outputs":site_json_outputs,
            "execution_scope":"actual completed native clustering/ranking/site publication; complete selected inputs including background; optional geometric site exports can overlap and do not partition selection; Arrow acknowledged separately",
            "scientific_rank_or_population_modified":false}))?;
        let acks=inputs.iter().map(|p|ConsumerAck {consumer:"native_pathb_site_materialization".into(),
            producer:p.producer.clone(),next_ordinal:p.next_ordinal,input_foliage:head.clone(),
            receipt_bytes:receipt.clone(),output_payloads:Vec::new()}).collect::<Vec<_>>();
        if !acks.is_empty() {
            state.head=Some(state.store.commit_chronicle(Some(&head),&state.identity,
                &state.dictionary,&[],&acks,None)?);
        }
        Ok(())
    }
    pub fn flow_summary(&self) -> Result<serde_json::Value> {
        let state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("poisoned raw Chronicle"))?;
        let Some(head) = state.head.as_ref() else {
            return Ok(
                serde_json::json!({"schema":"native_md_chronicle_flow.v1","identity":state.identity,"foliage_head":null,"raw_events":0,"selected_events":0,"consumer_status":"NO_EVENT_POPULATION_COMMITTED","restart":"NOT_ESTABLISHED_BY_OBSERVATION_LEAVES"}),
            );
        };
        let f = state.store.inspect_chronicle_index(head, &state.identity)?;
        let mut pending = Vec::new();
        for consumer in &f.dictionary.consumers {
            for producer in f
                .producers
                .iter()
                .filter(|p| consumer.required_fields.contains(&p.producer.field_id))
            {
                let acknowledged = f
                    .consumers
                    .iter()
                    .find(|a| a.consumer == consumer.name && a.producer == producer.producer)
                    .map_or(0, |a| a.next_ordinal);
                if acknowledged < producer.next_ordinal {
                    pending.push(serde_json::json!({"consumer":consumer.name,"producer":producer.producer,"first_pending_ordinal":acknowledged,"next_pending_ordinal":producer.next_ordinal}));
                }
            }
        }
        Ok(
            serde_json::json!({"schema":"native_md_chronicle_flow.v1","identity":f.identity,"foliage_head":head,"source_dictionary":f.dictionary,"producer_populations":f.producers,"actual_consumer_watermarks":f.consumers,"pending_consumer_ranges":pending,"native_queue_prefixes":state.queue_prefixes,"queue_readout_ordinals":state.queue_ordinals,
            "transitive_lineage":{"authority":"immutable foliage leaves plus exact typed parent references; 24-byte selection maps identify each selected original datum","source_population":"all original mixed-channel per-stream raw ordinals","unselected_raw":"retained durably; absence from a selection map is not deletion or a claim downstream consumed it","pending_downstream":"follow derived leaf lineage from each source payload and the explicit pending consumer ranges","causal_order":"native per-stream queue reservation order only","drain_provenance":"independent typed attempted/capacity/written/frontier records include zero-event resets; source completeness requires these records"},
            "ontology_coverage":{"thermal_context":"configured_schedule_classification; actual device phase witness remains raw phase_bits","per_event_kernel_id":"UNRESOLVED_MULTIPLE_ELIGIBLE_EMITTERS_CAN_SHARE_SPIKE_SOURCE","per_event_numerical_subphase":"UNRESOLVED; readout boundary is known","source_lineage_digest":"helper implementation identity, not an observed per-event kernel identity"},"deletion_authorized":false,"restart":"NOT_ESTABLISHED_BY_OBSERVATION_LEAVES","receipt_scope":"source preservation and actual recorded consumers through this call; not whole runtime acceptance"}),
        )
    }
    pub fn record_selection(
        &self,
        stream_id: usize,
        raw: &[GpuSpikeEvent],
        selected: &[usize],
        transform: &[u8],
    ) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("poisoned raw Chronicle"))?;
        ensure!(
            stream_id < state.ordinals.len() && state.ordinals[stream_id] == raw.len() as u64,
            "selector input differs from fully committed raw event population"
        );
        if raw.is_empty() {
            ensure!(selected.is_empty(), "selection exists without raw source");
        }
        let source_head = state
            .head
            .clone()
            .context("raw population has no durable foliage")?;
        let source_index = state
            .store
            .validate_chronicle_index(&source_head, &state.identity)?;
        let source = source_index.foliage();
        let raw_leaves = source
            .leaves
            .iter()
            .enumerate()
            .filter(|(_, l)| l.producer.field_id == RAW && l.producer.stream_id == stream_id as u32)
            .collect::<Vec<_>>();
        let clock_leaves = source
            .leaves
            .iter()
            .enumerate()
            .filter(|(_, l)| {
                l.producer.field_id == CLOCK && l.producer.stream_id == stream_id as u32
            })
            .collect::<Vec<_>>();
        let mut clock_cache: Option<(usize, Vec<u8>)> = None;
        // Source decoder/encoder identity is checked against every original
        // committed byte without cloning the full accumulated event vector.
        for (index, leaf) in
            source.leaves.iter().enumerate().filter(|(_, l)| {
                l.producer.field_id == RAW && l.producer.stream_id == stream_id as u32
            })
        {
            let bytes = state
                .store
                .read_indexed_chronicle_payload(&source_index, index)?;
            let begin = usize::try_from(leaf.coverage.first_ordinal)?;
            for (offset, record) in bytes.chunks_exact(96).enumerate() {
                ensure!(
                    raw.get(begin + offset)
                        .context("raw source index absent")?
                        .to_le_bytes()
                        .as_slice()
                        == record,
                    "native raw-to-decoded event bytes changed before selection"
                );
            }
        }
        let mut previous_clock = 0.0f64;
        let mut observed_clock_rows = 0u64;
        for (leaf_index, leaf) in &clock_leaves {
            let bytes = state
                .store
                .read_indexed_chronicle_payload(&source_index, *leaf_index)?;
            for record in bytes.chunks_exact(24) {
                let step = u64::from_le_bytes(record[..8].try_into()?);
                let dt = f64::from_bits(u64::from_le_bytes(record[8..16].try_into()?));
                let time = f64::from_bits(u64::from_le_bytes(record[16..24].try_into()?));
                ensure!(
                    step == observed_clock_rows + 1
                        && dt.is_finite()
                        && dt > 0.0
                        && time.is_finite()
                        && time > previous_clock
                        && previous_clock + dt == time,
                    "committed native clock population has a gap or changed actual dt/time bits"
                );
                previous_clock = time;
                observed_clock_rows += 1;
            }
            ensure!(
                observed_clock_rows == leaf.coverage.next_ordinal,
                "committed clock leaf population differs"
            );
        }
        ensure!(
            observed_clock_rows == state.clock_ordinals[stream_id],
            "native physical clock producer/consumer counts differ"
        );
        let drains = verify_drain_population(
            &state.store,
            &source_index,
            stream_id,
            raw.len() as u64,
            observed_clock_rows,
        )?;
        ensure!(
            drains == state.drain_ordinals[stream_id],
            "native drain producer/consumer population differs"
        );
        let queue=&state.queue_prefixes[stream_id];
        ensure!(!queue.awaiting_existing_reset && queue.captured_records==0,
            "PathB selection cannot precede the existing source owner's verified reset acknowledgement");
        let transform_sha256 = digest(transform);
        let mut outputs = Vec::new();
        if !selected.is_empty() {
            let selection_head = state.store.commit_chronicle_streaming(
                state.head.as_ref(),
                &state.identity,
                &state.dictionary,
                MAX_SLAB_BYTES as u64,
                &[],
                None,
                |append| {
                    for (chunk, indices) in selected.chunks(MAP_RECORDS_PER_SLAB).enumerate() {
                        let mut bytes = Vec::with_capacity(indices.len() * 24);
                        let mut parents = Vec::new();
                        // Exact immutable foliage indices distinguish identical bytes
                        // from different sources. Clone each typed reference once,
                        // retaining original first-seen lineage order.
                        let mut parent_indices = BTreeSet::new();
                        let mut first_step = u64::MAX;
                        let mut last_step = 0;
                        let mut min_time = f64::INFINITY;
                        let mut max_time = 0.0f64;
                        for &ordinal in indices {
                            let event = raw
                                .get(ordinal)
                                .context("selected native index out of bounds")?;
                            let step = u64::try_from(event.timestep)?;
                            let raw_index = raw_leaves.partition_point(|(_, l)| {
                                l.coverage.next_ordinal <= ordinal as u64
                            });
                            let (raw_leaf_index, leaf) = raw_leaves
                                .get(raw_index)
                                .filter(|(_, l)| l.coverage.first_ordinal <= ordinal as u64)
                                .context("selected event lacks original durable leaf")?;
                            if parent_indices.insert(*raw_leaf_index) {
                                parents.push(leaf.reference());
                            }
                            let clock_index =
                                clock_leaves.partition_point(|(_, l)| l.coverage.last_step < step);
                            let (leaf_index, clock_leaf) = clock_leaves
                                .get(clock_index)
                                .filter(|(_, l)| l.coverage.first_step <= step)
                                .context("selected event lacks exact clock leaf")?;
                            if parent_indices.insert(*leaf_index) {
                                parents.push(clock_leaf.reference());
                            }
                            if clock_cache.as_ref().map(|(index, _)| index) != Some(leaf_index) {
                                clock_cache = Some((
                                    *leaf_index,
                                    state.store.read_indexed_chronicle_payload(
                                        &source_index,
                                        *leaf_index,
                                    )?,
                                ));
                            }
                            let offset = usize::try_from(step - clock_leaf.coverage.first_step)?
                                .checked_mul(24)
                                .context("clock sidecar offset overflow")?;
                            let clock_record =
                                &clock_cache.as_ref().context("clock cache absent")?.1
                                    [offset..offset + 24];
                            ensure!(
                                u64::from_le_bytes(clock_record[..8].try_into()?) == step,
                                "selector event/committed exact clock step differs"
                            );
                            let time = f64::from_bits(u64::from_le_bytes(
                                clock_record[16..24].try_into()?,
                            ));
                            ensure!(
                                time.is_finite() && time > 0.0,
                                "corrupt native exact event clock"
                            );
                            bytes.extend_from_slice(&(ordinal as u64).to_le_bytes());
                            bytes.extend_from_slice(&step.to_le_bytes());
                            bytes.extend_from_slice(&time.to_bits().to_le_bytes());
                            first_step = first_step.min(step);
                            last_step = last_step.max(step);
                            min_time = min_time.min(time);
                            max_time = max_time.max(time);
                        }

                        let first_ordinal = (chunk * MAP_RECORDS_PER_SLAB) as u64;
                        let leaf = LeafInput {
                            producer: ProducerKey {
                                field_id: SELECTION,
                                stream_id: stream_id as u32,
                                replica_epoch: REPLICA_EPOCH,
                            },
                            context_id: (stream_id * 7 + 6) as u32,
                            coverage: CoverageRange {
                                first_ordinal,
                                next_ordinal: first_ordinal + indices.len() as u64,
                                first_step,
                                last_step,
                                first_physical_time_ps_bits: min_time.to_bits(),
                                last_physical_time_ps_bits: max_time.to_bits(),
                            },
                            bytes,
                            lineage: Some(LeafLineage {
                                input_foliage: source_head.clone(),
                                parent_payloads: parents,
                                transform_sha256: transform_sha256.clone(),
                            }),
                        };
                        outputs.push(PayloadReference {
                            producer: leaf.producer.clone(),
                            context_id: leaf.context_id,
                            coverage: leaf.coverage.clone(),
                            object_sha256: digest(&leaf.bytes),
                        });
                        append(&leaf)?;
                    }
                    Ok(())
                },
            )?;
            state.head = Some(selection_head);
        }
        let input_head = state.head.clone().context("consumer input HEAD absent")?;
        let acknowledgment_index = state
            .store
            .validate_chronicle_index(&input_head, &state.identity)?;
        ensure!(
            acknowledgment_index.foliage().leaves.len() >= source.leaves.len()
                && acknowledgment_index.foliage().leaves[..source.leaves.len()]
                    == source.leaves[..],
            "consumer acknowledgment snapshot changed its actually read source prefix"
        );
        let receipt = serde_json::to_vec(
            &serde_json::json!({"schema":"native_pathb_channel_selection.v1","stream_id":stream_id,"original_raw_events":raw.len(),"selected_events":selected.len(),"input_source_foliage":source_head,"source_prefix_revalidated_in_acknowledgment_foliage":input_head,"selection_output_payloads":outputs,"transform_sha256":transform_sha256,"transform_bytes":transform,"execution_scope":"actual_native_selector_returned_these_original_indices; downstream_clustering_and_ranking_not_acknowledged"}),
        )?;
        let acks = [
            (RAW, raw.len() as u64),
            (CLOCK, observed_clock_rows),
            (DRAIN, drains),
            (QUEUE,state.queue_ordinals[stream_id]),
        ]
        .into_iter()
        .filter(|(_, count)| *count > 0)
        .map(|(field_id, next_ordinal)| ConsumerAck {
            consumer: CONSUMER.into(),
            producer: ProducerKey {
                field_id,
                stream_id: stream_id as u32,
                replica_epoch: REPLICA_EPOCH,
            },
            next_ordinal,
            input_foliage: input_head.clone(),
            receipt_bytes: receipt.clone(),
            output_payloads: outputs.clone(),
        })
        .collect::<Vec<_>>();
        state.head = Some(state.store.commit_chronicle(
            Some(&input_head),
            &state.identity,
            &state.dictionary,
            &[],
            &acks,
            None,
        )?);
        Ok(())
    }
}
// Each readout publishes exactly one foliage. Leaf bytes are borrowed only
// by append; no vector of a whole window or whole raw run is retained here.
fn publish_capture(
    state: &mut State,
    stream_id: usize,
    signed_count: i32,
    capacity: usize,
    clock: &[PhysicalTimeRecord],
    phase: &impl Fn(i32) -> &'static str,
    read_slab: &mut dyn FnMut(usize, usize) -> Result<Vec<u8>>,
) -> Result<usize> {
    publish_capture_range(state,stream_id,signed_count,capacity,clock,phase,true,read_slab)
}
fn publish_capture_range(state:&mut State,stream_id:usize,signed_count:i32,capacity:usize,
    clock:&[PhysicalTimeRecord],phase:&impl Fn(i32)->&'static str,reset_expected:bool,
    read_slab:&mut dyn FnMut(usize,usize)->Result<Vec<u8>>)->Result<usize> {
    ensure!(stream_id < state.ordinals.len(), "unknown source stream");
    ensure!(
        signed_count >= 0,
        "negative actual device attempted spike count"
    );
    let count = signed_count as usize;
    ensure!(
        count <= capacity,
        "actual attempted count {} exceeds native capacity {}; refusing truncated raw coverage",count,capacity
    );
    let queue=state.queue_prefixes[stream_id].clone();
    ensure!(!queue.awaiting_existing_reset,"prior completed readout has no verified existing owner reset acknowledgement");
    let prefix=usize::try_from(queue.captured_records)?;
    ensure!(prefix<=count,"native queue regressed before existing reset acknowledgement");
    let first_raw = state.ordinals[stream_id];
    let queue_raw_first=first_raw.checked_sub(queue.captured_records).context("raw queue prefix ordinal underflow")?;
    let queue_ordinal=state.queue_ordinals[stream_id];
    let first_clock = state.clock_ordinals[stream_id];
    let drain_ordinal = state.drain_ordinals[stream_id];
    let end = clock
        .last()
        .context("completed drain lacks actual native clock endpoint")?;
    ensure!(
        first_clock <= clock.len() as u64,
        "native clock population regressed"
    );
    let mut next_raw = first_raw;
    let head = state.store.commit_chronicle_streaming(
        state.head.as_ref(),
        &state.identity,
        &state.dictionary,
        MAX_SLAB_BYTES as u64,
        &[],
        None,
        |append| {
            append_clock_rows(first_clock, stream_id, clock, phase, append)?;
            for first in (prefix..count).step_by(RAW_RECORDS_PER_SLAB) {
                let n = (count - first).min(RAW_RECORDS_PER_SLAB);
                let raw = read_slab(first, n)?;
                ensure!(raw.len() == n * 96, "short native original-wire slab");
                let mut start = 0;
                while start < n {
                    let step = raw_step(&raw[start * 96..])?;
                    let name = phase(i32::try_from(step)?);
                    let context = PHASES
                        .iter()
                        .position(|p| *p == name)
                        .context("unknown configured schedule classification")?;
                    let mut stop = start + 1;
                    while stop < n && phase(i32::try_from(raw_step(&raw[stop * 96..])?)?) == name {
                        stop += 1;
                    }
                    let mut previous = None;
                    for record in raw[start * 96..stop * 96].chunks_exact(96) {
                        let step = raw_step(record)?;
                        let time = exact_clock(clock, step)?;
                        if let Some((old_step, old_time)) = previous {
                            ensure!(
                                step >= old_step && time >= old_time,
                                "native queue chronology regressed"
                            );
                        }
                        previous = Some((step, time));
                    }
                    let first_step = raw_step(&raw[start * 96..])?;
                    let last_step = raw_step(&raw[(stop - 1) * 96..])?;
                    let next = next_raw
                        .checked_add((stop - start) as u64)
                        .context("raw ordinal overflow")?;
                    let leaf = LeafInput {
                        producer: ProducerKey {
                            field_id: RAW,
                            stream_id: stream_id as u32,
                            replica_epoch: REPLICA_EPOCH,
                        },
                        context_id: (stream_id * 7 + context) as u32,
                        lineage: None,
                        coverage: CoverageRange {
                            first_ordinal: next_raw,
                            next_ordinal: next,
                            first_step,
                            last_step,
                            first_physical_time_ps_bits: exact_clock(clock, first_step)?.to_bits(),
                            last_physical_time_ps_bits: exact_clock(clock, last_step)?.to_bits(),
                        },
                        bytes: raw[start * 96..stop * 96].to_vec(),
                    };
                    append(&leaf)?;
                    next_raw = next;
                    start = stop;
                }
            }
            let first_new = if first_clock < clock.len() as u64 {
                &clock[usize::try_from(first_clock)?]
            } else {
                end
            };
            let phase_step=i32::try_from(end.step).context("queue native step width")?;
            let context=PHASES.iter().position(|p|*p==phase(phase_step)).context("queue schedule classification missing")?;
            let mut witness=Vec::with_capacity(QUEUE_BYTES);
            for word in [end.step,signed_count as i64 as u64,capacity as u64,prefix as u64,count as u64,
                queue_raw_first,next_raw,queue.epoch,reset_expected as u64,end.cumulative_simulated_time_ps.to_bits()] {witness.extend_from_slice(&word.to_le_bytes());}
            append(&LeafInput{producer:ProducerKey{field_id:QUEUE,stream_id:stream_id as u32,replica_epoch:REPLICA_EPOCH},
                context_id:(stream_id*7+context) as u32,lineage:None,
                coverage:CoverageRange{first_ordinal:queue_ordinal,next_ordinal:queue_ordinal.checked_add(1).context("queue witness ordinal overflow")?,
                    first_step:end.step,last_step:end.step,first_physical_time_ps_bits:end.cumulative_simulated_time_ps.to_bits(),last_physical_time_ps_bits:end.cumulative_simulated_time_ps.to_bits()},bytes:witness})?;
            if reset_expected {
            let mut bytes = Vec::with_capacity(DRAIN_BYTES);
            for word in [
                end.step,
                signed_count as i64 as u64,
                capacity as u64,
                (count-prefix) as u64,
                first_raw,
                next_raw,
                first_clock,
                clock.len() as u64,
                first_new.step,
                end.step,
                first_new.cumulative_simulated_time_ps.to_bits(),
                end.cumulative_simulated_time_ps.to_bits(),
                ((count-prefix) as u64)
                    .checked_mul(96)
                    .context("raw byte count overflow")?,
                0,
            ] {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
            let context = PHASES
                .iter()
                .position(|p| {
                    *p == phase(
                        i32::try_from(end.step).expect("native event clock ABI already bounded"),
                    )
                })
                .context("unknown drain schedule classification")?;
            let leaf = LeafInput {
                producer: ProducerKey {
                    field_id: DRAIN,
                    stream_id: stream_id as u32,
                    replica_epoch: REPLICA_EPOCH,
                },
                context_id: (stream_id * 7 + context) as u32,
                lineage: None,
                coverage: CoverageRange {
                    first_ordinal: drain_ordinal,
                    next_ordinal: drain_ordinal
                        .checked_add(1)
                        .context("drain ordinal overflow")?,
                    first_step: end.step,
                    last_step: end.step,
                    first_physical_time_ps_bits: end.cumulative_simulated_time_ps.to_bits(),
                    last_physical_time_ps_bits: end.cumulative_simulated_time_ps.to_bits(),
                },
                bytes,
            };
            append(&leaf)?;
            }
            Ok(())
        },
    ).context("native Chronicle transaction failed; source was not reset; reconcile durable HEAD before any retry because publication may have completed")?;
    // No producer cursor advances before the full readout's durable HEAD.
    state.head = Some(head);
    state.ordinals[stream_id] = next_raw;
    state.clock_ordinals[stream_id] = clock.len() as u64;
    if reset_expected {state.drain_ordinals[stream_id] = drain_ordinal + 1;}
    state.queue_ordinals[stream_id]=queue_ordinal+1;
    state.queue_prefixes[stream_id]=QueuePrefixContinuationV1{epoch:queue.epoch,captured_records:count as u64,awaiting_existing_reset:reset_expected};
    Ok(count)
}
fn append_clock_rows(
    first_ordinal: u64,
    stream_id: usize,
    clock: &[PhysicalTimeRecord],
    phase: &impl Fn(i32) -> &'static str,
    append: &mut dyn FnMut(&LeafInput) -> Result<()>,
) -> Result<()> {
    let mut first = usize::try_from(first_ordinal)?;
    ensure!(
        first <= clock.len(),
        "native exact physical clock population regressed"
    );
    while first < clock.len() {
        let name = phase(i32::try_from(clock[first].step)?);
        let context = PHASES
            .iter()
            .position(|p| *p == name)
            .context("unknown configured schedule classification")?;
        let limit = (first + MAX_SLAB_BYTES / 24).min(clock.len());
        let mut end = first + 1;
        while end < limit && phase(i32::try_from(clock[end].step)?) == name {
            end += 1;
        }
        let mut bytes = Vec::with_capacity((end - first) * 24);
        for (offset, row) in clock[first..end].iter().enumerate() {
            ensure!(
                row.step == (first + offset) as u64 + 1,
                "native clock row/physical ordinal mismatch"
            );
            let previous = if first + offset == 0 {
                0.0
            } else {
                clock[first + offset - 1].cumulative_simulated_time_ps
            };
            ensure!(
                row.applied_dt_ps.is_finite()
                    && row.applied_dt_ps > 0.0
                    && row.cumulative_simulated_time_ps.is_finite()
                    && previous + row.applied_dt_ps == row.cumulative_simulated_time_ps,
                "native exact dt/time population invalid"
            );
            bytes.extend_from_slice(&row.step.to_le_bytes());
            bytes.extend_from_slice(&row.applied_dt_ps.to_bits().to_le_bytes());
            bytes.extend_from_slice(&row.cumulative_simulated_time_ps.to_bits().to_le_bytes());
        }
        let leaf = LeafInput {
            producer: ProducerKey {
                field_id: CLOCK,
                stream_id: stream_id as u32,
                replica_epoch: REPLICA_EPOCH,
            },
            context_id: (stream_id * 7 + context) as u32,
            lineage: None,
            coverage: CoverageRange {
                first_ordinal: first as u64,
                next_ordinal: end as u64,
                first_step: clock[first].step,
                last_step: clock[end - 1].step,
                first_physical_time_ps_bits: clock[first].cumulative_simulated_time_ps.to_bits(),
                last_physical_time_ps_bits: clock[end - 1].cumulative_simulated_time_ps.to_bits(),
            },
            bytes,
        };
        append(&leaf)?;
        first = end;
    }
    Ok(())
}
fn verify_drain_population(store:&CheckpointStore,index:&ValidatedChronicleIndex,stream_id:usize,
    raw_count:u64,clock_count:u64)->Result<u64> {
    let mut raw=0u64;let mut clock=0u64;let mut epoch=0u64;let mut prefix=0u64;
    let mut queue_ord=0u64;let mut previous_reset=false;
    let mut complete=Vec::<([u64;10],u64,u64)>::new();
    for (leaf_index,leaf) in index.foliage().leaves.iter().enumerate().filter(|(_,l)|l.producer.field_id==QUEUE && l.producer.stream_id==stream_id as u32) {
        let bytes=store.read_indexed_chronicle_payload(index,leaf_index)?;
        ensure!(bytes.len()==QUEUE_BYTES,"queue prefix witness width changed");
        let mut w=[0u64;10];for (word,bytes) in w.iter_mut().zip(bytes.chunks_exact(8)){*word=u64::from_le_bytes(bytes.try_into()?);}
        if queue_ord>0 && previous_reset {
            ensure!(w[7]==epoch.checked_add(1).context("queue epoch overflow")? && w[3]==0,
                "native queue epoch advanced without an existing reset-bound readout");prefix=0;
        }else{ensure!(w[7]==epoch,"native queue epoch regressed/jumped");}
        ensure!(leaf.coverage.first_ordinal==queue_ord && leaf.coverage.next_ordinal==queue_ord+1
            && (w[1] as i64)>=0 && w[1]<=w[2] && w[3]==prefix && w[4]>=prefix && w[4]==w[1]
            && w[5].checked_add(prefix)==Some(raw) && w[5].checked_add(w[4])==Some(w[6])
            && w[0]>=clock && w[8]<=1 && w[0]==leaf.coverage.last_step
            && w[9]==leaf.coverage.last_physical_time_ps_bits && f64::from_bits(w[9]).is_finite(),
            "native queue prefix/attempted/capacity/ordinal/clock accounting differs");
        if w[8]==1 {complete.push((w,raw,clock));}
        raw=w[6];clock=w[0];epoch=w[7];prefix=w[4];previous_reset=w[8]==1;queue_ord+=1;
    }
    ensure!(raw==raw_count && clock==clock_count && previous_reset,
        "raw/clock population lacks complete append-only queue and final reset-bound coverage");
    let mut drains=0usize;
    for (leaf_index,leaf) in index.foliage().leaves.iter().enumerate().filter(|(_,l)|l.producer.field_id==DRAIN && l.producer.stream_id==stream_id as u32) {
        let bytes=store.read_indexed_chronicle_payload(index,leaf_index)?;
        ensure!(bytes.len()==DRAIN_BYTES,"native drain accounting width changed");
        let mut w=[0u64;14];for (word,bytes) in w.iter_mut().zip(bytes.chunks_exact(8)){*word=u64::from_le_bytes(bytes.try_into()?);}
        let (queue,raw_before,clock_before)=complete.get(drains).context("drain lacks matching reset-bound queue readout")?;
        let captured=queue[4]-queue[3];
        ensure!(leaf.coverage.first_ordinal==drains as u64 && leaf.coverage.next_ordinal==drains as u64+1
            && w[0]==queue[0] && w[1]==queue[1] && w[2]==queue[2] && w[3]==captured && w[4]==*raw_before && w[5]==queue[6]
            && w[6]==*clock_before && w[7]==queue[0] && w[9]==w[7] && w[11]==queue[9]
            && w[12]==captured.checked_mul(96).context("native drain wire extent overflow")? && w[13]==0,
            "drain and independent original queue prefix/readout coverage disagree");
        ensure!(w[8]==if w[7]>*clock_before{*clock_before+1}else{w[9]},"native drain first clock step differs");
        let first=f64::from_bits(w[10]);let last=f64::from_bits(w[11]);
        ensure!(first.is_finite() && last.is_finite() && first>0.0 && last>=first
            && last.to_bits()==leaf.coverage.last_physical_time_ps_bits,"native drain physical clock interval differs");
        drains+=1;
    }
    ensure!(drains==complete.len(),"reset-bound queue readout lacks original native drain provenance");Ok(drains as u64)
}
// Match the authoritative existing force-spike readout driver contract. A
// cudarc bind can drain a deferred error from an already dropped sibling graph;
// checked direct driver operations here never suppress any actual transfer or
// stream error. Source allocation guards remain held by capture_source.
fn read_owned_raw(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    pointer: u64,
    length: usize,
) -> Result<Vec<u8>> {
    use cudarc::driver::sys;
    ensure!(
        pointer != 0 || length == 0,
        "null owned raw source allocation"
    );
    let context = unsafe { sys::cuCtxSetCurrent(stream.context().cu_ctx()) };
    ensure!(
        context == sys::CUresult::CUDA_SUCCESS,
        "native raw Chronicle context failed: {context:?}"
    );
    let fence = unsafe { sys::cuStreamSynchronize(stream.cu_stream()) };
    ensure!(
        fence == sys::CUresult::CUDA_SUCCESS,
        "native raw Chronicle source stream fence failed: {fence:?}"
    );
    let mut bytes = vec![0u8; length];
    if length > 0 {
        let copied = unsafe { sys::cuMemcpyDtoH_v2(bytes.as_mut_ptr().cast(), pointer, length) };
        ensure!(
            copied == sys::CUresult::CUDA_SUCCESS,
            "native raw Chronicle source DtoH failed: {copied:?}"
        );
    }
    Ok(bytes)
}
fn raw_step(record: &[u8]) -> Result<u64> {
    ensure!(record.len() >= 96, "truncated native record");
    Ok(u64::try_from(i32::from_le_bytes(record[..4].try_into()?))?)
}
fn exact_clock(clock: &[PhysicalTimeRecord], step: u64) -> Result<f64> {
    let index = usize::try_from(
        step.checked_sub(1)
            .context("event step0 has no admitted completed physical clock")?,
    )?;
    let row = clock
        .get(index)
        .context("native event step has no exact physical clock; refusing fabricated timestamp")?;
    ensure!(
        row.step == step
            && row.cumulative_simulated_time_ps.is_finite()
            && row.cumulative_simulated_time_ps > 0.0,
        "native event/clock identity invalid"
    );
    Ok(row.cumulative_simulated_time_ps)
}
// Same output order, float arithmetic, tiny-sample bypass and sanity-band policy
// as the existing live Otsu branch. Index partitioning avoids cloning raw events.
pub fn otsu_indices(
    raw: &[GpuSpikeEvent],
    percentile: u32,
    otsu: impl Fn(&[f32]) -> f32,
    low: f64,
    high: f64,
) -> Vec<usize> {
    if raw.len() <= 1000 {
        return (0..raw.len()).collect();
    }
    let mut channels: [Vec<usize>; 5] = std::array::from_fn(|_| Vec::new());
    for (index, event) in raw.iter().enumerate() {
        channels[match event.spike_source {
            1 => 0,
            3 => 2,
            4 | 5 => 3,
            6 => 4,
            _ => 1,
        }]
        .push(index);
    }
    let pct_f = (percentile.min(99) as f32) / 100.0;
    let mut result = Vec::new();
    for (channel, indices) in channels.into_iter().enumerate() {
        if channel == 3 || indices.len() < 100 {
            result.extend(indices);
            continue;
        }
        let label = ["UV", "LIF", "EFP", "LADD", "VORONOI_LIF"][channel];
        let n_pre = indices.len();
        let ints = indices
            .iter()
            .map(|&i| raw[i].intensity)
            .collect::<Vec<_>>();
        let i_min = ints.iter().copied().fold(f32::INFINITY, f32::min);
        let i_max = ints.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let percentile_thresh = {
            let mut sorted = ints.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            sorted
                .get((sorted.len() as f32 * pct_f) as usize)
                .copied()
                .unwrap_or(0.0)
        };
        let otsu_t = otsu(&ints);
        let kept_frac = ints.iter().filter(|&&v| v >= otsu_t).count() as f64 / n_pre as f64;
        log::info!(
            "  [Otsu] {}: n={} threshold={:.4} kept_frac={:.4} (min={:.4} max={:.4})",
            label,
            n_pre,
            otsu_t,
            kept_frac,
            i_min,
            i_max
        );
        let threshold = if kept_frac < low || kept_frac > high {
            log::info!("  [Otsu] {}: kept_frac {:.4} outside sanity band [{:.2},{:.2}] → falling back to percentile (thresh={:.4})",label,kept_frac,low,high,percentile_thresh);
            percentile_thresh
        } else {
            otsu_t
        };
        drop(ints);
        let before = result.len();
        result.extend(
            indices
                .into_iter()
                .filter(|&i| raw[i].intensity >= threshold),
        );
        log::info!(
            "  [Otsu] {}: kept {}/{} ({:.1}%) at final threshold {:.4}",
            label,
            result.len() - before,
            n_pre,
            100.0 * (result.len() - before) as f32 / n_pre as f32,
            threshold
        );
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    fn identity() -> CheckpointIdentity {
        CheckpointIdentity {
            run_id: "raw-observer-cpu-contract-test".into(),
            topology_sha256: "1".repeat(64),
            engine_sha256: "2".repeat(64),
            kernel_set_sha256: "3".repeat(64),
            protocol_sha256: "4".repeat(64),
            execution_request_sha256: "5".repeat(64),
            state_abi_sha256: "6".repeat(64),
            provider: "explicit_CPU_contract_fixture_not_native_execution".into(),
            stream_count: 2,
        }
    }
    #[test]
    fn module_admission_uses_entries_not_comments_or_debug_strings() {
        let entries=ptx_entries("// .entry fake_one()\n/* .entry fake_two() */\n.file 1 \".entry fake_three()\"\n.visible .entry nhs_observer_physical_clock_begin( .param .u64 p ) {}\n.visible .entry nhs_observer_physical_clock_commit () {}\n");
        assert_eq!(
            entries,
            BTreeSet::from([
                "nhs_observer_physical_clock_begin".into(),
                "nhs_observer_physical_clock_commit".into()
            ])
        );
    }
    #[test]
    fn observation_route_refuses_every_scientific_checkpoint_or_resume_combination() {
        assert!(check_observation_continuation(false, false).is_ok());
        for (checkpoint, resume) in [(true, false), (false, true), (true, true)] {
            assert!(check_observation_continuation(checkpoint, resume).is_err());
        }
    }
    #[test]
    fn resource_admission_separates_quota_from_kernel_capacity_and_bounds_publication_work() {
        let model = resource_model(8, 40000, 500, 64 * 1024 * 1024 * 1024).unwrap();
        assert_eq!(model["maximum_completed_source_drains"].as_u64(), Some(648));
        assert_eq!(model["maximum_foliage_publications"].as_u64(), Some(665));
        assert_eq!(
            model["source_publications_per_native_chunk"].as_u64(),
            Some(1)
        );
        assert_eq!(
            model["worst_case_payload_exceeds_quota"].as_bool(),
            Some(true)
        );
        assert!(resource_model(8, 40000, 500, 1024).is_err());
        assert!(resource_model(8, 40000, 0, 64 * 1024 * 1024 * 1024).is_err());
    }
    fn fixture_clock(n: usize) -> Vec<PhysicalTimeRecord> {
        (0..n)
            .map(|i| PhysicalTimeRecord {
                step: i as u64 + 1,
                applied_dt_ps: 0.125,
                cumulative_simulated_time_ps: (i + 1) as f64 * 0.125,
            })
            .collect()
    }
    #[test]
    fn complete_clock_and_zero_drain_are_independent_complete_populations() {
        let dir = tempfile::tempdir().unwrap();
        let chronicle = RawChronicle::open(
            &dir.path().join("store"),
            1024 * 1024,
            identity(),
            vec![42, 12387],
        )
        .unwrap();
        let clock = fixture_clock(3);
        let events = [
            GpuSpikeEvent {
                timestep: 1,
                spike_source: 1,
                ..GpuSpikeEvent::default()
            },
            GpuSpikeEvent {
                timestep: 3,
                spike_source: 3,
                ..GpuSpikeEvent::default()
            },
        ];
        {
            let mut state = chronicle.state.lock().unwrap();
            publish_capture(
                &mut state,
                0,
                2,
                2,
                &clock,
                &|_| "cold_hold",
                &mut |first, n| {
                    Ok(events[first..first + n]
                        .iter()
                        .flat_map(GpuSpikeEvent::to_le_bytes)
                        .collect())
                },
            )
            .unwrap();
            assert_eq!(state.head.as_ref().unwrap().generation, 1);
            assert_eq!(state.ordinals[0], 2);
            assert_eq!(state.clock_ordinals[0], 3);
            assert_eq!(state.drain_ordinals[0], 1);
            publish_capture(
                &mut state,
                0,
                0,
                2,
                &clock,
                &|_| "cold_hold",
                &mut |_, _| panic!("zero-event readout must not request raw bytes"),
            )
            .unwrap();
            assert_eq!(state.head.as_ref().unwrap().generation, 2);
            assert_eq!(state.ordinals[0], 2);
            assert_eq!(state.drain_ordinals[0], 2);
        }
        chronicle
            .record_selection(0, &events, &[1, 0], b"permuted_exact_selection_fixture")
            .unwrap();
        let state = chronicle.state.lock().unwrap();
        let head = state.head.as_ref().unwrap();
        let f = state
            .store
            .inspect_chronicle_index(head, &state.identity)
            .unwrap();
        let clock_leaf = f
            .leaves
            .iter()
            .position(|l| l.producer.field_id == CLOCK)
            .unwrap();
        let clock_bytes = state
            .store
            .read_chronicle_payload(head, &state.identity, clock_leaf)
            .unwrap();
        assert_eq!(clock_bytes.len(), 72);
        assert_eq!(&clock_bytes[24..32], &2u64.to_le_bytes());
        let zero = f
            .leaves
            .iter()
            .enumerate()
            .filter(|(_, l)| l.producer.field_id == DRAIN)
            .nth(1)
            .unwrap();
        let bytes = state
            .store
            .read_chronicle_payload(head, &state.identity, zero.0)
            .unwrap();
        assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 0);
        assert_eq!(u64::from_le_bytes(bytes[24..32].try_into().unwrap()), 0);
        assert_eq!(&bytes[32..40], &bytes[40..48]);
        let map = f
            .leaves
            .iter()
            .position(|l| l.producer.field_id == SELECTION)
            .unwrap();
        let bytes = state
            .store
            .read_chronicle_payload(head, &state.identity, map)
            .unwrap();
        assert_eq!(u64::from_le_bytes(bytes[..8].try_into().unwrap()), 1);
        assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 3);
        assert_eq!(
            u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            0.375f64.to_bits()
        );
        assert_eq!(u64::from_le_bytes(bytes[24..32].try_into().unwrap()), 0);
        assert_eq!(
            f.consumers
                .iter()
                .find(|a| a.producer.field_id == CLOCK)
                .unwrap()
                .next_ordinal,
            3
        );
        assert_eq!(
            f.consumers
                .iter()
                .find(|a| a.producer.field_id == RAW)
                .unwrap()
                .next_ordinal,
            2
        );
        assert_eq!(
            f.consumers
                .iter()
                .find(|a| a.producer.field_id == DRAIN)
                .unwrap()
                .next_ordinal,
            2
        );
        assert!(f.consumers.iter().all(|a| a.consumer == CONSUMER));
    }
    #[test]
    fn zero_event_source_has_durable_attempted_capacity_written_frontiers() {
        let dir = tempfile::tempdir().unwrap();
        let chronicle = RawChronicle::open(
            &dir.path().join("store"),
            1024 * 1024,
            identity(),
            vec![42, 12387],
        )
        .unwrap();
        let clock = fixture_clock(2);
        {
            let mut state = chronicle.state.lock().unwrap();
            publish_capture(
                &mut state,
                0,
                0,
                1234,
                &clock,
                &|_| "cold_hold",
                &mut |_, _| panic!("zero source has no raw payload"),
            )
            .unwrap();
        }
        chronicle
            .record_selection(0, &[], &[], b"verified_empty_source")
            .unwrap();
        let state = chronicle.state.lock().unwrap();
        let head = state.head.as_ref().unwrap();
        let index = state
            .store
            .validate_chronicle_index(head, &state.identity)
            .unwrap();
        assert!(index
            .foliage()
            .leaves
            .iter()
            .all(|l| l.producer.field_id != RAW));
        assert_eq!(
            verify_drain_population(&state.store, &index, 0, 0, 2).unwrap(),
            1
        );
        let drain = index
            .foliage()
            .leaves
            .iter()
            .position(|l| l.producer.field_id == DRAIN)
            .unwrap();
        let bytes = state
            .store
            .read_indexed_chronicle_payload(&index, drain)
            .unwrap();
        assert_eq!(u64::from_le_bytes(bytes[16..24].try_into().unwrap()), 1234);
    }
    #[test]
    fn large_multi_slab_selection_crosses_later_clock_leaf_and_preserves_stale_head_refusal() {
        let dir = tempfile::tempdir().unwrap();
        let chronicle = RawChronicle::open(
            &dir.path().join("store"),
            64 * 1024 * 1024,
            identity(),
            vec![42, 12387],
        )
        .unwrap();
        let n = MAP_RECORDS_PER_SLAB + 17;
        let clock = fixture_clock(n);
        let events = (0..n)
            .map(|i| GpuSpikeEvent {
                timestep: i as i32 + 1,
                spike_source: 1,
                ..GpuSpikeEvent::default()
            })
            .collect::<Vec<_>>();
        let (source_head, index) = {
            let mut state = chronicle.state.lock().unwrap();
            publish_capture(
                &mut state,
                0,
                n as i32,
                n,
                &clock,
                &|_| "cold_hold",
                &mut |first, count| {
                    Ok(events[first..first + count]
                        .iter()
                        .flat_map(GpuSpikeEvent::to_le_bytes)
                        .collect())
                },
            )
            .unwrap();
            assert_eq!(state.head.as_ref().unwrap().generation, 1);
            let h = state.head.clone().unwrap();
            let index = state
                .store
                .validate_chronicle_index(&h, &state.identity)
                .unwrap();
            (h, index)
        };
        let selected = (0..n).collect::<Vec<_>>();
        chronicle
            .record_selection(0, &events, &selected, b"large_multislice_original_order")
            .unwrap();
        let state = chronicle.state.lock().unwrap();
        let head = state.head.as_ref().unwrap();
        assert_eq!(head.generation, 3); // one raw, one selection, one acknowledgment publication
        let f = state
            .store
            .inspect_chronicle_index(head, &state.identity)
            .unwrap();
        let maps = f
            .leaves
            .iter()
            .filter(|l| l.producer.field_id == SELECTION)
            .collect::<Vec<_>>();
        assert_eq!(maps.len(), 2);
        assert_eq!(maps[1].coverage.first_ordinal, MAP_RECORDS_PER_SLAB as u64);
        let original_raw = index
            .foliage()
            .leaves
            .iter()
            .filter(|leaf| leaf.producer.field_id == RAW)
            .collect::<Vec<_>>();
        let original_clocks = index
            .foliage()
            .leaves
            .iter()
            .filter(|leaf| leaf.producer.field_id == CLOCK)
            .collect::<Vec<_>>();
        // Each source is retained exactly once in first-seen order, including
        // the raw leaf spanning the output-map boundary.
        let mut first_parents = vec![original_raw[0].reference(), original_clocks[0].reference()];
        first_parents.extend(original_raw.iter().skip(1).map(|leaf| leaf.reference()));
        assert_eq!(
            maps[0].lineage.as_ref().unwrap().parent_payloads,
            first_parents
        );
        assert_eq!(
            maps[1].lineage.as_ref().unwrap().parent_payloads,
            vec![
                original_raw.last().unwrap().reference(),
                original_clocks[1].reference()
            ]
        );
        let later_clock = index
            .foliage()
            .leaves
            .iter()
            .enumerate()
            .filter(|(_, l)| l.producer.field_id == CLOCK)
            .nth(1)
            .unwrap()
            .0;
        assert!(state
            .store
            .read_chronicle_payload(&source_head, &state.identity, later_clock)
            .is_err());
        let later = state
            .store
            .read_indexed_chronicle_payload(&index, later_clock)
            .unwrap();
        assert_eq!(
            u64::from_le_bytes(later[..8].try_into().unwrap()),
            MAP_RECORDS_PER_SLAB as u64 + 1
        );
    }
    #[test]
    fn failed_partial_slab_and_overflow_leave_head_and_all_source_cursors_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let chronicle = RawChronicle::open(
            &dir.path().join("store"),
            32 * 1024 * 1024,
            identity(),
            vec![42, 12387],
        )
        .unwrap();
        let clock = fixture_clock(1);
        let mut state = chronicle.state.lock().unwrap();
        publish_capture(
            &mut state,
            0,
            0,
            RAW_RECORDS_PER_SLAB + 1,
            &clock,
            &|_| "cold_hold",
            &mut |_, _| unreachable!(),
        )
        .unwrap();
        let before = state.head.clone();
        let mut calls = 0;
        let result = publish_capture(
            &mut state,
            0,
            (RAW_RECORDS_PER_SLAB + 1) as i32,
            RAW_RECORDS_PER_SLAB + 1,
            &clock,
            &|_| "cold_hold",
            &mut |_, n| {
                calls += 1;
                if calls == 2 {
                    anyhow::bail!("injected source transfer failure");
                }
                Ok((0..n)
                    .flat_map(|_| {
                        GpuSpikeEvent {
                            timestep: 1,
                            ..GpuSpikeEvent::default()
                        }
                        .to_le_bytes()
                    })
                    .collect())
            },
        );
        assert!(result.is_err());
        assert_eq!(state.head, before);
        assert_eq!(state.store.chronicle_head().unwrap(), before);
        assert_eq!(state.ordinals[0], 0);
        assert_eq!(state.clock_ordinals[0], 1);
        assert_eq!(state.drain_ordinals[0], 1);
        assert!(publish_capture(
            &mut state,
            0,
            2,
            1,
            &clock,
            &|_| "cold_hold",
            &mut |_, _| unreachable!()
        )
        .is_err());
        assert_eq!(state.head, before);
    }
    #[test]
    fn actual_dictionary_has_full_wire_coverage_and_native_clock_binding() {
        let d = dictionary(&identity(), &[42, 12387]);
        d.validate(&identity()).unwrap();
        let f = &d.fields[0];
        let mut covered = vec![false; 96];
        for c in &f.components {
            for byte in c.byte_offset..c.byte_offset + c.byte_width {
                assert!(!covered[byte as usize]);
                covered[byte as usize] = true;
            }
        }
        assert!(covered.into_iter().all(|v| v));
        assert_eq!(d.contexts.len(), 14);
        assert!(f.subject.contains("abs_phi_minus_phi_previous_not_energy"));
        assert!(f.subject.contains("only_declared_active_slots"));
    }
    #[test]
    fn every_original_wire_bit_survives_native_decoder_and_encoder() {
        let mut bytes = [0u8; 96];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = ((i * 37 + 129) % 256) as u8;
        }
        bytes[..4].copy_from_slice(&1i32.to_le_bytes());
        // Signed zero, quiet and signaling NaN payloads remain bits here. No
        // arithmetic or JSON normalization is applied to event components.
        for (offset, bits) in [
            (8, 0x80000000u32),
            (12, 0x7fcabcde),
            (16, 0x7f812345),
            (20, 0xff800000),
            (80, 1),
        ] {
            bytes[offset..offset + 4].copy_from_slice(&bits.to_le_bytes());
        }
        assert_eq!(
            GpuSpikeEvent::from_le_bytes(&bytes).unwrap().to_le_bytes(),
            bytes
        );
        assert!(GpuSpikeEvent::from_le_bytes_with_stride(&bytes, 64).is_err());
    }
    #[test]
    fn nonuniform_clock_join_is_exact_and_missing_or_zero_step_refuses() {
        let rows = [
            PhysicalTimeRecord {
                step: 1,
                applied_dt_ps: 0.001,
                cumulative_simulated_time_ps: 0.001,
            },
            PhysicalTimeRecord {
                step: 2,
                applied_dt_ps: 0.0025,
                cumulative_simulated_time_ps: 0.0035,
            },
        ];
        assert_eq!(
            exact_clock(&rows, 2).unwrap().to_bits(),
            rows[1].cumulative_simulated_time_ps.to_bits()
        );
        assert!(exact_clock(&rows, 0).is_err());
        assert!(exact_clock(&rows, 3).is_err());
        let bad = [rows[0], PhysicalTimeRecord { step: 9, ..rows[1] }];
        assert!(exact_clock(&bad, 2).is_err());
    }
    // Independent value-vector form of the existing live branch, retained only
    // as an equivalence oracle for the new index-only observational path.
    fn prior_otsu(raw: Vec<GpuSpikeEvent>, percentile: u32) -> Vec<GpuSpikeEvent> {
        if raw.len() <= 1000 {
            return raw;
        }
        let mut channels: [Vec<GpuSpikeEvent>; 4] = std::array::from_fn(|_| Vec::new());
        for event in raw {
            channels[match event.spike_source {
                1 => 0,
                3 => 2,
                4 | 5 => 3,
                _ => 1,
            }]
            .push(event);
        }
        let pct_f = percentile.min(99) as f32 / 100.0;
        let filter = |events: Vec<GpuSpikeEvent>| {
            if events.len() < 100 {
                return events;
            }
            let ints = events
                .iter()
                .map(|event| event.intensity)
                .collect::<Vec<_>>();
            let mut sorted = ints.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let percentile_t = sorted
                .get((sorted.len() as f32 * pct_f) as usize)
                .copied()
                .unwrap_or(0.0);
            let otsu_t = crate::otsu_intensity_threshold(&ints);
            let fraction =
                ints.iter().filter(|&&v| v >= otsu_t).count() as f64 / events.len() as f64;
            let threshold = if fraction < crate::OTSU_SANITY_BAND_LOW
                || fraction > crate::OTSU_SANITY_BAND_HIGH
            {
                percentile_t
            } else {
                otsu_t
            };
            events
                .into_iter()
                .filter(|event| event.intensity >= threshold)
                .collect::<Vec<_>>()
        };
        let [uv, lif, efp, ladd] = channels;
        let mut output = filter(uv);
        output.extend(filter(lif));
        output.extend(filter(efp));
        output.extend(ladd);
        output
    }
    #[test]
    fn voronoi_otsu_uses_a_distinct_population() {
        let events: Vec<_> = (0..1200).map(|i| GpuSpikeEvent {
            timestep: i,
            spike_source: if i < 600 { 2 } else { 6 },
            intensity: if i < 600 { i as f32 } else { (i - 600 + 10000) as f32 },
            ..GpuSpikeEvent::default()
        }).collect();
        let midpoint = |values: &[f32]| {
            0.5 * (values.iter().copied().fold(f32::INFINITY, f32::min)
                + values.iter().copied().fold(f32::NEG_INFINITY, f32::max))
        };
        let actual = otsu_indices(&events, 50, midpoint, 0.1, 0.9);
        assert_eq!(actual, (300..600).chain(900..1200).collect::<Vec<_>>());
    }

    #[test]
    fn index_otsu_preserves_prior_live_values_channel_order_and_original_ordinals() {
        for length in [0, 1, 99, 100, 1000, 1001, 2017] {
            for percentile in [0, 70, 90, 99, 100] {
                for constant in [false, true] {
                    let events = (0..length)
                        .map(|i| GpuSpikeEvent {
                            timestep: i as i32 + 1,
                            spike_source: [1, 2, 3, 4, 5, 0][i % 6],
                            intensity: if constant {
                                1.0
                            } else {
                                ((i * 37) % 101) as f32 / 7.0
                            },
                            ..GpuSpikeEvent::default()
                        })
                        .collect::<Vec<_>>();
                    let expected = prior_otsu(events.clone(), percentile);
                    let indices = otsu_indices(
                        &events,
                        percentile,
                        crate::otsu_intensity_threshold,
                        crate::OTSU_SANITY_BAND_LOW,
                        crate::OTSU_SANITY_BAND_HIGH,
                    );
                    let actual = indices
                        .iter()
                        .map(|&index| events[index].to_le_bytes())
                        .collect::<Vec<_>>();
                    assert_eq!(
                        actual,
                        expected
                            .iter()
                            .map(GpuSpikeEvent::to_le_bytes)
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(
                        indices,
                        expected
                            .iter()
                            .map(|event| (event.timestep - 1) as usize)
                            .collect::<Vec<_>>()
                    );
                }
            }
        }
    }
}

// Planned-stop coordination reuses the native SIGINT watchdog and its existing
// cancel-on-worker-exit barrier. Optional full-capsule mode uses only the
// existing CheckpointStore/epoch and carries its committed proof; this control
// owner never implements a second scientific checkpoint manager.
pub struct DurableStop {
    root: std::path::PathBuf,
    token: String,
    run_id: String,
    pid: u32,
    streams: usize,
    stop_requested: std::sync::atomic::AtomicBool,
    scientific_capture:bool,
    scientific_stop_proof:Mutex<Option<serde_json::Value>>,
    pub barrier: prism_nhs::rescue_controller::CoupledWindowBarrier,
    acknowledgments: Mutex<Vec<Option<serde_json::Value>>>,
}
fn stop_write(path: &Path, value: &serde_json::Value) -> Result<()> {
    super::write_json_pretty_atomic(path, value)?;
    std::fs::File::open(path)?.sync_all()?;
    std::fs::File::open(path.parent().context("stop receipt parent absent")?)?.sync_all()?;
    Ok(())
}
impl DurableStop {
    pub fn from_env(
        chronicle: Option<&std::sync::Arc<RawChronicle>>,
        streams: usize,
    ) -> Result<Option<std::sync::Arc<Self>>> {
        Self::from_env_with_scientific_capture(chronicle,streams,false)
    }
    pub fn from_env_with_scientific_capture(
        chronicle:Option<&std::sync::Arc<RawChronicle>>,streams:usize,scientific_capture:bool,
    )->Result<Option<std::sync::Arc<Self>>> {
        let root = std::env::var_os("PRISM_DURABLE_STOP_ROOT");
        let token = std::env::var("PRISM_DURABLE_STOP_TOKEN").ok();
        if root.is_none() && token.is_none() {
            return Ok(None);
        }
        let root = std::path::PathBuf::from(root.context("durable stop root missing")?);
        let token = token.context("durable stop token missing")?;
        ensure!(
            root.is_absolute()
                && root.is_dir()
                && !std::fs::symlink_metadata(&root)?.file_type().is_symlink()
                && !token.is_empty()
                && token.len() <= 128,
            "invalid private durable stop control binding"
        );
        let run_id = chronicle
            .context("durable planned stop requires original raw/clock Chronicle")?
            .run_id()?;
        ensure!(
            !root.join("ready.json").try_exists()? && !root.join("request.json").try_exists()?,
            "durable stop control must be fresh"
        );
        let value = Self {
            root,
            token,
            run_id,
            pid: std::process::id(),
            streams,
            stop_requested: std::sync::atomic::AtomicBool::new(false),
            scientific_capture,scientific_stop_proof:Mutex::new(None),
            barrier: prism_nhs::rescue_controller::CoupledWindowBarrier::new(streams),
            acknowledgments: Mutex::new(vec![None; streams]),
        };
        Ok(Some(std::sync::Arc::new(value)))
    }
    pub fn publish_ready(&self) -> Result<()> {
        stop_write(
            &self.root.join("ready.json"),
            &serde_json::json!({"schema":"prism.native_durable_stop.v1","run_token":self.token,"run_id":self.run_id,"native_pid":self.pid,"streams":self.streams,"mechanism":"existing_SIGINT_watchdog_and_final_Chronicle_capture","normal_full_PathB_preserved":true,
                "scientific_capture_enabled":self.scientific_capture,
                "planned_stop_request_protocol":if self.scientific_capture{"authenticated_request_file_then_common_boundary_scientific_capsule_before_cancel"}else{"existing_SIGINT_raw_frontier"}}),
        )?;
        Ok(())
    }
    pub fn record_scientific_stop_proof(&self,proof:serde_json::Value)->Result<()> {
        ensure!(self.scientific_capture && proof["identity"]["run_id"]==self.run_id && proof["identity"]["stream_count"].as_u64()==Some(self.streams as u64) && proof["schema"]=="prism.native_scientific_stop_capsule.v1"
            && proof["status"]=="SCIENTIFIC_STOP_CAPSULE_COMMITTED","foreign scientific stop proof");
        let mut slot=self.scientific_stop_proof.lock().map_err(|_|anyhow::anyhow!("scientific stop proof poisoned"))?;
        if let Some(previous)=slot.as_ref(){ensure!(*previous==proof,"conflicting scientific stop proof");return Ok(());}
        stop_write(&self.root.join("scientific_complete.json"),&serde_json::json!({"schema":"prism.native_durable_scientific_stop.v1",
            "run_token":self.token,"run_id":self.run_id,"native_pid":self.pid,"streams":self.streams,"capsule":proof}))?;
        *slot=Some(proof);Ok(())
    }
    /// Native deadline expiry uses the same authenticated control owner and
    /// common scientific stop route; first published cause is never replaced.
    pub fn request_native_budget_stop(&self,stream:usize)->Result<()> {
        ensure!(self.scientific_capture && stream<self.streams,"native budget stop requires scientific participant");
        if self.requested()?{return Ok(());}
        let temporary=self.root.join(format!(".native_budget_request_{}_{}.json",self.pid,stream));
        stop_write(&temporary,&serde_json::json!({"schema":"prism.native_durable_stop.v1","run_token":self.token,
            "run_id":self.run_id,"native_pid":self.pid,"reason":"cumulative_native_MD900_deadline","requested_by":"native_cumulative_budget_owner"}))?;
        match std::fs::hard_link(&temporary,self.root.join("request.json")) {
            Ok(())=>{},Err(error) if error.kind()==std::io::ErrorKind::AlreadyExists=>{},Err(error)=>return Err(error.into()),
        }
        std::fs::File::open(&self.root)?.sync_all()?;
        std::fs::remove_file(temporary)?;std::fs::File::open(&self.root)?.sync_all()?;
        ensure!(self.requested()?,"native budget request not published");Ok(())
    }
    /// Selected startup-body stop uses the SAME authenticated first-cause
    /// publication as the deadline owner, only after the all8 capsule commit.
    pub fn request_startup_boundary_stop(&self,stream:usize,physical:u32)->Result<()> {
        ensure!(self.scientific_capture && stream<self.streams && (1..6).contains(&physical),
            "startup stop requires selected physical1..5 scientific participant");
        if self.requested()?{return Ok(());}
        let temporary=self.root.join(format!(".native_startup_request_{}_{}.json",self.pid,stream));
        stop_write(&temporary,&serde_json::json!({"schema":"prism.native_durable_stop.v1","run_token":self.token,
            "run_id":self.run_id,"native_pid":self.pid,"reason":format!("fullcaps_planned_startup_physical_step:{physical}"),
            "requested_by":"native_selected_startup_owner","startup_physical_step":physical}))?;
        match std::fs::hard_link(&temporary,self.root.join("request.json")) {
            Ok(())=>{},Err(error) if error.kind()==std::io::ErrorKind::AlreadyExists=>{},Err(error)=>return Err(error.into()),
        }
        std::fs::File::open(&self.root)?.sync_all()?;std::fs::remove_file(temporary)?;std::fs::File::open(&self.root)?.sync_all()?;
        ensure!(self.requested()?,"native selected startup request not published");Ok(())
    }
    /// Read the original committed stop authority for terminal validation.
    /// This is NOT an ACK, raw final-close proof or reservation handoff.
    pub fn verified_scientific_stop_boundary(&self,stream:usize,identity:&CheckpointIdentity,
        head:&Head,common_next_tick:i32)->Result<prism_nhs::scientific_checkpoint::StreamBoundary> {
        ensure!(self.scientific_capture&&self.requested()?&&stream<self.streams,
            "partial terminal alignment lacks authenticated scientific stop");
        let slot=self.scientific_stop_proof.lock().map_err(|_|anyhow::anyhow!("scientific stop proof poisoned"))?;
        let proof=slot.as_ref().context("partial terminal alignment lacks original committed proof")?;
        ensure!(proof["identity"]==serde_json::to_value(identity)?&&proof["scientific_head"]==serde_json::to_value(head)?
            && proof["common_next_tick"].as_i64()==Some(i64::from(common_next_tick)),
            "partial terminal alignment differs from original scientific identity/head/tick");
        let boundaries:Vec<prism_nhs::scientific_checkpoint::StreamBoundary>=serde_json::from_value(proof["boundaries"].clone())?;
        let ids=boundaries.iter().map(|b|b.stream_id).collect::<BTreeSet<_>>();
        ensure!(boundaries.len()==self.streams&&ids==(0..identity.stream_count).collect::<BTreeSet<_>>(),
            "partial terminal alignment lacks exact all-worker boundary population");
        boundaries.into_iter().find(|b|b.stream_id==stream as u32).context("partial terminal stream absent")
    }
    /// Called by the native parent only AFTER every scoped worker returned.
    /// All ACKs and the exact scientific proof were produced by this owner.
    pub fn completed_scientific_handoff(&self,identity:&CheckpointIdentity,head:&Head)->Result<serde_json::Value> {
        ensure!(self.scientific_capture && self.requested()?,"scientific handoff lacks authenticated planned stop");
        let proof=self.scientific_stop_proof.lock().map_err(|_|anyhow::anyhow!("scientific stop proof poisoned"))?.clone().context("scientific stop proof absent")?;
        ensure!(proof["identity"]==serde_json::to_value(identity)? && proof["scientific_head"]==serde_json::to_value(head)?,"scientific handoff owner/head mismatch");
        let rows=self.acknowledgments.lock().map_err(|_|anyhow::anyhow!("stop ACKs poisoned"))?;
        ensure!(rows.len()==self.streams && rows.iter().all(Option::is_some),"scientific handoff lacks all worker ACKs");
        let path=self.root.join("complete.json");
        use std::io::Read;
        let file=std::fs::OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(&path)?;
        ensure!(file.metadata()?.is_file() && file.metadata()?.len()<=1048576,"scientific handoff receipt exceeds bound");
        let mut bytes=Vec::new();file.take(1048577).read_to_end(&mut bytes)?;ensure!(bytes.len()<=1048576,"scientific handoff grew");
        let complete:serde_json::Value=serde_json::from_slice(&bytes)?;
        ensure!(complete["status"]=="ALL_WORKERS_DURABLE" && complete["native_pid"].as_u64()==Some(self.pid as u64)
            && complete["run_token"]==self.token && complete["run_id"]==self.run_id && complete["streams"].as_u64()==Some(self.streams as u64)
            && complete["scientific_capsule"]==proof && complete["scientific_capsule_matches_all_final_physical_frontiers"]==true
            && complete["acknowledgments"]==serde_json::to_value(&*rows)?,"scientific handoff is not the original complete all-worker ACK proof");
        Ok(serde_json::json!({"boundary":"AUTHENTICATED_PLANNED_SCIENTIFIC_STOP","all_worker_threads_joined":true,
            "checkpoint_epoch_quiescent":true,"stop_complete_path":path,"stop_complete_sha256":digest(&bytes),
            "scientific_capsule":proof,"final_foliage_head":complete["final_foliage_head"]}))
    }
    pub fn requested(&self) -> Result<bool> {
        let path = self.root.join("request.json");
        if !path.try_exists()? {
            return Ok(false);
        }
        use std::io::Read;
        use std::os::unix::fs::OpenOptionsExt;
        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(path)?;
        ensure!(
            file.metadata()?.is_file() && file.metadata()?.len() <= 16384,
            "invalid stop request file"
        );
        let mut bytes = Vec::new();
        file.take(16385).read_to_end(&mut bytes)?;
        ensure!(bytes.len() <= 16384, "stop request exceeds bound");
        let v: serde_json::Value = serde_json::from_slice(&bytes)?;
        ensure!(
            v["schema"] == "prism.native_durable_stop.v1"
                && v["run_token"] == self.token
                && v["run_id"] == self.run_id
                && v["native_pid"].as_u64() == Some(self.pid as u64),
            "durable stop request identity differs"
        );
        Ok(true)
    }
    pub fn acknowledge(
        &self,
        stream: usize,
        frontier: serde_json::Value,
        chronicle: &RawChronicle,
    ) -> Result<bool> {
        ensure!(stream < self.streams, "invalid durable boundary stream");
        let ack = serde_json::json!({"schema":"prism.native_durable_stop.v1","run_token":self.token,"run_id":self.run_id,"native_pid":self.pid,"stream_id":stream,"status":"COMMITTED_OBSERVATION_FRONTIER","frontier":frontier,"next_physical_launch_admitted":false,"restart_capability":false});
        stop_write(&self.root.join(format!("stream_{stream}.json")), &ack)?;
        {
            let mut rows = self
                .acknowledgments
                .lock()
                .map_err(|_| anyhow::anyhow!("stop acknowledgments poisoned"))?;
            ensure!(rows[stream].is_none(), "duplicate stop acknowledgment");
            rows[stream] = Some(ack);
        }
        self.barrier.wait(|| false).map_err(anyhow::Error::msg)?;
        if stream == 0 {
            self.stop_requested
                .store(self.requested()?, std::sync::atomic::Ordering::Release);
            let rows = self
                .acknowledgments
                .lock()
                .map_err(|_| anyhow::anyhow!("stop acknowledgments poisoned"))?;
            ensure!(
                rows.iter().all(Option::is_some),
                "incomplete stopped worker population"
            );
            let final_head = chronicle
                .state
                .lock()
                .map_err(|_| anyhow::anyhow!("raw Chronicle poisoned"))?
                .head
                .clone();
            let scientific_proof=self.scientific_stop_proof.lock().map_err(|_|anyhow::anyhow!("scientific stop proof poisoned"))?.clone();
            let scientific_matches_current=scientific_proof.as_ref().is_some_and(|proof|{
                proof["boundaries"].as_array().is_some_and(|boundaries|boundaries.len()==self.streams
                    && rows.iter().all(|row|row.as_ref().is_some_and(|row|{
                        let sid=row["stream_id"].as_u64();
                        boundaries.iter().any(|boundary|boundary["stream_id"].as_u64()==sid
                            && boundary["completed_step"]==row["frontier"]["native_completed_step"]
                            && boundary["replica_epoch"]==row["frontier"]["replica_epoch"]
                            && boundary["replica_seed"]==row["frontier"]["replica_seed"])
                    })))
            });
            stop_write(
                &self.root.join("complete.json"),
                &serde_json::json!({"schema":"prism.native_durable_stop.v1","run_token":self.token,"run_id":self.run_id,"native_pid":self.pid,"status":"ALL_WORKERS_DURABLE","streams":self.streams,"acknowledgments":*rows,"final_foliage_head":final_head,"scope":"completed native clock/raw/drain frontier; no fullcampaign or scientificrestart claim","scientific_capture_enabled":self.scientific_capture,
                    "scientific_capsule":scientific_proof,"scientific_capsule_matches_all_final_physical_frontiers":scientific_matches_current,
                    "scientific_raw_cursor_scope":"capsule binds its exact original RawHEAD; final zero-drain RawHEAD may be a later descendant; historical restore branches from exact capsule cursors"}),
            )?;
        }
        self.barrier.wait(|| false).map_err(anyhow::Error::msg)?;
        Ok(self
            .stop_requested
            .load(std::sync::atomic::Ordering::Acquire))
    }
}
impl RawChronicle {
    pub fn verify_zero_stop_endpoint(
        &self,
        engine: &NhsAmberFusedEngine,
        stream_id: usize,
    ) -> Result<()> {
        ensure!(
            engine.device_protocol_current_step()? == 0,
            "zero stop has physical steps"
        );
        let stream = engine.stream();
        let (ptr, guard) = engine.spike_count_gpu().device_ptr(stream);
        let bytes = read_owned_raw(stream, ptr, 4)?;
        drop(guard);
        ensure!(
            i32::from_le_bytes(bytes.as_slice().try_into()?) == 0,
            "zero-step source has events without an admissible physical clock; no reset permitted"
        );
        let state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("raw Chronicle poisoned"))?;
        ensure!(
            stream_id < state.ordinals.len()
                && state.ordinals[stream_id] == 0
                && state.clock_ordinals[stream_id] == 0
                && state.drain_ordinals[stream_id] == 0,
            "zero stop producer frontier differs"
        );
        Ok(())
    }
    pub fn durable_stop_frontier(
        &self,
        engine: &NhsAmberFusedEngine,
        stream_id: usize,
    ) -> Result<serde_json::Value> {
        let step = u64::from(engine.device_protocol_current_step()?);
        let state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("raw Chronicle poisoned"))?;
        ensure!(
            stream_id < state.ordinals.len()
                && engine.replica_epoch() == REPLICA_EPOCH
                && engine.replica_seed() == Some(state.seeds[stream_id])
                && state.clock_ordinals[stream_id] == step,
            "stop endpoint differs from fully committed exact-clock frontier"
        );
        ensure!(
            step == 0 || (state.head.is_some() && state.drain_ordinals[stream_id] > 0),
            "stop endpoint has no durable drain authority"
        );
        Ok(
            serde_json::json!({"stream_id":stream_id,"replica_epoch":REPLICA_EPOCH,"replica_seed":state.seeds[stream_id],"native_completed_step":step,"clock_next_ordinal":state.clock_ordinals[stream_id],"raw_next_ordinal":state.ordinals[stream_id],"drain_next_ordinal":state.drain_ordinals[stream_id],"foliage_head":state.head,"zero_step_verified_empty":step==0,"completed_but_uncommitted_clock_steps":0,"raw_reset_after_commit_only":true}),
        )
    }
}
