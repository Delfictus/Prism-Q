//! Actual mixed endpoint energy and declared-work admission. Native AMBER
//! quantities remain separate and unchanged. The operator's explicit1e-6Ha
//! acceptance applies to both the body and accumulated mixed residual.
use crate::biprism_v10_runtime::BiprismDeviceViews;
use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DeviceRepr, LaunchConfig,
    PushKernelArg,
};
use serde::Serialize;
use std::sync::Arc;
pub const MIXED_MAGIC: u32 = 0x424d4536;
pub const MIXED_TOL: f64 = 1e-6;
pub const MIXED_STAGES: usize = 7;
pub const KCAL_TO_HARTREE: f64 = 0.0008432975634897884 * 1.8897261259077824;
#[derive(Clone, Copy, Debug)]
#[repr(u32)]
pub(crate) enum MixedStage {
    Start = 0,
    PreLif = 1,
    PostLif = 2,
    PostNonham = 3,
    PostCa = 4,
    End = 5,
    HaloCounterfactual = 6,
}
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(crate) struct MixedBinding {
    pub native_hmr_masses: u64,
    pub native_potential: u64,
    pub native_atom_work: u64,
    pub native_cumulative_work: u64,
    pub atoms: u32,
    pub source: u32,
    pub reserved: u64,
}
unsafe impl DeviceRepr for MixedBinding {}
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct MixedVertex {
    pub packed_atom: u32,
    pub leaf: u32,
    pub local_vertex: u32,
    pub stage: u32,
    pub native_x: [f32; 3],
    pub native_v: [f32; 3],
    pub x: [f32; 3],
    pub v: [f32; 3],
    pub reference: [f32; 3],
    pub potential_alignment:u32,
    pub phi: f64,
    pub psi: [f32; 4],
    pub omega: f32,
    pub amber_charge: f32,
    pub true_inverse_mass: f32,
    pub propagation_inverse_mass: f32,
    pub main_mass: f32,
    pub native_hmr_mass: f32,
    pub metric: [f32; 6],
    pub charge: f64,
    pub rho00: f64,
    pub rho11: f64,
    pub population: f64,
    pub kinetic_metric: f64,
    pub kinetic_native_core: f64,
    pub native_x_lo: [f32; 3],
    pub x_lo: [f32; 3],
    pub reference_lo: [f32; 3],
    pub potential_entry_alignment:u32,
    pub phi_entry:f64,
    pub coordinate_reserved: [u32; 2],
    pub metric_f64: [f64; 6],
    pub psi_precise:[f64;4],
}
unsafe impl DeviceRepr for MixedVertex {}
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct MixedEdge {
    pub leaf: u32,
    pub edge: u32,
    pub u: u32,
    pub v: u32,
    pub base: f32,
    pub reserved: f32,
    pub live: f64,
    pub e0: f64,
    pub e1: f64,
    pub v01: f64,
    pub overlap: f64,
}
unsafe impl DeviceRepr for MixedEdge {}
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct MixedSample {
    pub magic: u32,
    pub leaf: u32,
    pub stage: u32,
    pub state: u32,
    pub scheduled_step: u32,
    pub committed_substeps: u32,
    pub mode: u32,
    pub sticky_fault: u32,
    pub generation: u64,
    pub body_ordinal: u64,
    pub preparation_generation: u64,
    pub dec_iterations: u32,
    pub source: u32,
    pub values: [f64; 24],
}
unsafe impl DeviceRepr for MixedSample {}
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct MixedLeafLedger {
    pub magic: u32,
    pub leaf: u32,
    pub state: u32,
    pub sticky_fault: u32,
    pub scheduled_step: u32,
    pub committed_substeps: u32,
    pub mode: u32,
    pub source: u32,
    pub generation: u64,
    pub body_ordinal: u64,
    pub preparation_generation: u64,
    pub reserved: u64,
    pub values: [f64; 24],
}
unsafe impl DeviceRepr for MixedLeafLedger {}
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct MixedGlobalLedger {
    pub magic: u32,
    pub state: u32,
    pub sticky_fault: u32,
    pub scheduled_step: u32,
    pub committed_substeps: u32,
    pub stage_mask: u32,
    pub reserved: [u32; 2],
    pub generation: u64,
    pub body_ordinal: u64,
    pub preparation_generation: u64,
    pub accepted_bodies: u64,
    pub values: [f64; 24],
}
unsafe impl DeviceRepr for MixedGlobalLedger {}
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct MixedEnergyState {
    pub magic: u32,
    pub state: u32,
    pub sticky_fault: u32,
    pub last_accepted_step: u32,
    pub generation: u64,
    pub accepted_bodies: u64,
    pub preparation_generation: u64,
    pub reserved: [u64; 3],
    pub totals: [f64; 16],
    pub previous_native_work: [f64; 8],
    pub previous_leaf_energy: [f64; 8],
    pub previous_leaf_kinetic: [f64; 8],
    pub reserved_tail: [f64; 16],
}
unsafe impl DeviceRepr for MixedEnergyState {}
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct MixedWindow {
    pub magic: u32,
    pub capacity: u32,
    pub first_body: u32,
    pub requested_bodies: u32,
    pub first_step: u32,
    pub reserved: u32,
    pub generation: u64,
    pub preparation_generation: u64,
    pub reserved_tail: u64,
}
unsafe impl DeviceRepr for MixedWindow {}
const _: () = {
    assert!(core::mem::size_of::<MixedBinding>() == 48);
    assert!(core::mem::size_of::<MixedVertex>() == 336);
    assert!(core::mem::offset_of!(MixedVertex,metric_f64) == 256);
    assert!(core::mem::offset_of!(MixedVertex,phi)==80);
    assert!(core::mem::offset_of!(MixedVertex, native_x_lo) == 200);
    assert!(core::mem::offset_of!(MixedVertex,potential_entry_alignment)==236);
    assert!(core::mem::offset_of!(MixedVertex,phi_entry)==240);
    assert!(core::mem::offset_of!(MixedVertex, coordinate_reserved) == 248);
    assert!(core::mem::offset_of!(MixedVertex, charge) == 152);
    assert!(core::mem::offset_of!(MixedEdge,live)==24&&core::mem::offset_of!(MixedEdge,e0)==32);
    assert!(core::mem::size_of::<MixedEdge>() == 64);
    assert!(core::mem::size_of::<MixedSample>() == 256);
    assert!(core::mem::size_of::<MixedLeafLedger>() == 256);
    assert!(core::mem::size_of::<MixedGlobalLedger>() == 256);
    assert!(core::mem::size_of::<MixedEnergyState>() == 512);
    assert!(core::mem::size_of::<MixedWindow>() == 48);
};
pub const MIXED_SAMPLE_NAMES: [&str; 24] = [
    "native_potential_kcal",
    "native_kinetic_all_kcal",
    "native_kinetic_core_kcal",
    "metric_propagation_kinetic_core_Ha",
    "onsite_Ha",
    "edge_electronic_Ha",
    "DEC_legendre_Ha",
    "ring_anchor_Ha",
    "owned_directed_pair_Ha",
    "model_potential_Ha",
    "native_to_mixed_correction_Ha",
    "mixed_energy_Ha",
    "native_cumulative_work_kcal",
    "current_native_atom_work_kcal",
    "true_DEC_residual",
    "core_spinor_norm_sum",
    "true_mass_metric_kinetic_core_Ha",
    "native_kinetic_bath_kcal",
    "true_lab_kinetic_core_Ha",
    "private_DEC_entry_refinement_delta_Ha",
    "reserved20",
    "reserved21",
    "reserved22",
    "reserved23",
];
pub const MIXED_LEAF_NAMES: [&str; 24] = [
    "native_declared_work_Ha",
    "phase2_parameter_work_Ha",
    "LIF_correction_work_Ha",
    "native_postUV_nonham_correction_work_Ha",
    "native_CA_correction_work_Ha",
    "total_declared_work_Ha",
    "energy_delta_Ha",
    "local_ungated_residual_Ha",
    "halo_geometry_counterfactual_model_delta_Ha",
    "initial_parameter_prefix_Ha",
    "native_LIF_work_identity_kcal",
    "previous_energy_Ha",
    "current_energy_Ha",
    "previous_mixed_kinetic_Ha",
    "current_mixed_kinetic_Ha",
    "body_start_energy_Ha",
    "uncredited_start_to_preLIF_correction_delta_Ha",
    "uncredited_postCA_to_end_correction_delta_Ha",
    "reserved18",
    "reserved19",
    "reserved20",
    "reserved21",
    "reserved22",
    "reserved23",
];
pub const MIXED_GLOBAL_NAMES: [&str; 24] = [
    "initial_energy_Ha",
    "previous_energy_Ha",
    "current_energy_Ha",
    "declared_work_Ha",
    "body_residual_Ha",
    "cumulative_work_Ha",
    "expected_energy_Ha",
    "run_residual_Ha",
    "previous_mixed_kinetic_Ha",
    "current_mixed_kinetic_Ha",
    "initial_parameter_prefix_Ha",
    "max_abs_body_residual_Ha",
    "max_abs_run_residual_Ha",
    "absolute_tolerance_Ha",
    "cumulative_initial_parameter_prefix_Ha",
    "reserved15",
    "reserved16",
    "reserved17",
    "reserved18",
    "reserved19",
    "reserved20",
    "reserved21",
    "reserved22",
    "reserved23",
];
pub struct MixedReadback {
    pub window: MixedWindow,
    pub before: MixedEnergyState,
    pub after: MixedEnergyState,
    pub globals: Vec<MixedGlobalLedger>,
    pub leaves: Vec<MixedLeafLedger>,
    pub samples: Vec<MixedSample>,
}
pub struct MixedCurrentAttempt {
    pub state: MixedEnergyState,
    pub pending: MixedGlobalLedger,
    pub leaves: Vec<MixedLeafLedger>,
    pub samples: Vec<MixedSample>,
    pub vertices: Vec<MixedVertex>,
    pub edges: Vec<MixedEdge>,
}
fn bytes<T>(x: &[T]) -> Result<&[u8], String> {
    if !cfg!(target_endian = "little") {
        return Err("mixed energy requires little endian".into());
    }
    Ok(unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), core::mem::size_of_val(x)) })
}
impl MixedReadback {
    pub fn global_bytes(&self) -> Result<&[u8], String> {
        bytes(&self.globals)
    }
    pub fn leaf_bytes(&self) -> Result<&[u8], String> {
        bytes(&self.leaves)
    }
    pub fn sample_bytes(&self) -> Result<&[u8], String> {
        bytes(&self.samples)
    }
    pub fn index(&self) -> serde_json::Value {
        serde_json::json!({"schema":"biprism.mixed_energy.v3","window":self.window,"before":self.before,"after":self.after,
        "global_bytes":256,"leaf_bytes":256,"stage_sample_bytes":256,"sample_order":"body,stage,leaf","leaf_order":"body,leaf",
        "sample_names":MIXED_SAMPLE_NAMES,"leaf_names":MIXED_LEAF_NAMES,"global_names":MIXED_GLOBAL_NAMES,
        "stages":["start_after_actual_ForceN_before_Phase2","before_LIF","after_LIF_native_export","after_native_UV_clamp_RATTLE_nonham","after_native_CA_intervention","completed_native_endpoint","diagnostic_endpoint_core_with_start_halo_geometry"],
        "hamiltonian":"native_U + native_K_bath + metric_K_core(selected propagation mass) + electronic_DEC_edge_onsite + directed_ring",
        "acceptance":"global eight-leaf absolute body and accumulated run work-corrected residual each <=1e-6 Hartree; local leaves are not separately energy-conserved",
        "initial_reference":"first actual ForceN/pre-motion Hamiltonian; separately retained pre-baseline parameter work excluded exactly once; earlier unmeasured energy not claimed",
        "declared_work":"native work unchanged plus Phase2 parameter refresh and matching LIF/postUV-nonham/CA model-correction differences",
        "not_work":"ordinary conservative constraints, bridge representation changes, lagged halo geometry, unexplained residual",
        "primitive_scope":"all current-body seven-stage336B BME6 vertices and64B edges retained on terminal readback; bounded per-step journal stores actual measured stage energies",
        "potential":"full binary64 entry and privately solved phi; sample.values[19] is numerical refinement diagnostic, never declared work",
        "native_governor_unchanged":true,"whole_state_rollback":false,"scientific_restart":false})
    }
}
impl MixedCurrentAttempt {
    pub fn pending_bytes(&self) -> Result<&[u8], String> {
        bytes(std::slice::from_ref(&self.pending))
    }
    pub fn leaf_bytes(&self) -> Result<&[u8], String> {
        bytes(&self.leaves)
    }
    pub fn sample_bytes(&self) -> Result<&[u8], String> {
        bytes(&self.samples)
    }
    pub fn vertex_bytes(&self) -> Result<&[u8], String> {
        bytes(&self.vertices)
    }
    pub fn edge_bytes(&self) -> Result<&[u8], String> {
        bytes(&self.edges)
    }
    pub fn index(&self) -> serde_json::Value {
        serde_json::json!({"schema":"biprism.mixed_energy.current_attempt.v3","state":self.state,"pending":self.pending,
        "vertex_bytes":336,"edge_bytes":64,"sample_bytes":256,"order":"stage,leaf,local_vertex or1536 edge slots","scope":"may be incomplete/stale on refusal; every stage identity must match before interpreting it as produced"})
    }
}
fn close(a: f64, b: f64) -> bool {
    a.is_finite()
        && b.is_finite()
        && (a - b).abs() <= 64. * f64::EPSILON * a.abs().max(b.abs()).max(1.)
}
pub fn validate_mixed_readback(r: &MixedReadback) -> Result<(), String> {
    let w = r.window;
    let n = w.requested_bodies as usize;
    if w.magic != MIXED_MAGIC
        || w.capacity == 0
        || w.capacity > 500
        || n == 0
        || n > w.capacity as usize
        || w.generation == 0
        || w.preparation_generation == 0
        || w.reserved != 0
        || w.reserved_tail != 0
        || r.globals.len() != n
        || r.leaves.len() != 8 * n
        || r.samples.len() != 56 * n
        || r.before.accepted_bodies != u64::from(w.first_body)
    {
        return Err("mixed window population/identity".into());
    }
    let mut initial = r.before.totals[0];
    let mut prev = r.before.totals[1];
    let mut cumulative = r.before.totals[3];
    let mut previous_native = r.before.previous_native_work;
    let mut previous_leaf = r.before.previous_leaf_energy;
    let mut previous_kinetic = r.before.previous_leaf_kinetic;
    let mut prefix_total = r.before.totals[6];
    let mut peak_body = r.before.totals[7];
    let mut peak_run = r.before.totals[8];
    let mut initial_kinetic = r.before.totals[4];
    if r.before.reserved != [0; 3]
        || r.before.reserved_tail.iter().any(|v| *v != 0.)
        || r.before.totals.iter().any(|v| !v.is_finite())
        || r.before.totals[9..].iter().any(|v| *v != 0.)
        || previous_native
            .iter()
            .chain(previous_leaf.iter())
            .chain(previous_kinetic.iter())
            .any(|v| !v.is_finite())
        || peak_body < 0.
        || peak_body > MIXED_TOL
        || peak_run < 0.
        || peak_run > MIXED_TOL
    {
        return Err("mixed prior reference fields".into());
    }
    if w.first_body == 0
        && (r.before.magic != 0
            || r.before.state != 0
            || r.before.sticky_fault != 0
            || r.before.last_accepted_step != 0
            || r.before.generation != 0
            || r.before.preparation_generation != 0
            || r.before.totals.iter().any(|v| *v != 0.)
            || previous_native
                .iter()
                .chain(previous_leaf.iter())
                .chain(previous_kinetic.iter())
                .any(|v| *v != 0.))
    {
        return Err("mixed cold reference is not zero initialized".into());
    }
    if w.first_body > 0
        && (r.before.magic != MIXED_MAGIC
            || r.before.state != 3
            || r.before.sticky_fault != 0
            || r.before.generation != w.generation
            || r.before.preparation_generation != w.preparation_generation
            || r.before.last_accepted_step.checked_add(1) != Some(w.first_step)
            || !close(r.before.totals[2], initial + cumulative)
            || !close(r.before.totals[1], previous_leaf.iter().sum())
            || !close(r.before.totals[5], previous_kinetic.iter().sum())
            || initial_kinetic < 0.)
    {
        return Err("mixed prior accepted reference".into());
    }
    for body in 0..n {
        let ordinal = u64::from(w.first_body) + body as u64;
        let step = w
            .first_step
            .checked_add(body as u32)
            .ok_or("mixed step overflow")?;
        let committed = u32::try_from(
            (ordinal + 1)
                .checked_mul(4)
                .ok_or("mixed substep overflow")?,
        )
        .map_err(|_| "mixed substep overflow")?;
        let g = r.globals[body];
        let mut total = 0.;
        let mut work = 0.;
        let mut prev_sum = 0.;
        let mut start_sum = 0.;
        let mut prior_k = 0.;
        let mut current_k = 0.;
        let mut prefix_sum = 0.;
        if g.magic != MIXED_MAGIC
            || g.state != 3
            || g.sticky_fault != 0
            || g.scheduled_step != step
            || g.committed_substeps != committed
            || g.stage_mask != 127
            || g.reserved != [0; 2]
            || g.generation != w.generation
            || g.preparation_generation != w.preparation_generation
            || g.body_ordinal != ordinal
            || g.accepted_bodies != ordinal + 1
            || g.values.iter().any(|x| !x.is_finite())
            || g.values[13] != MIXED_TOL
            || g.values[4].abs() > MIXED_TOL
            || g.values[7].abs() > MIXED_TOL
            || g.values[15..].iter().any(|v| *v != 0.)
        {
            return Err(format!("mixed global admission body{ordinal}"));
        }
        for leaf in 0..8 {
            let l = r.leaves[body * 8 + leaf];
            let s = |stage: usize| r.samples[body * 56 + stage * 8 + leaf];
            let a = s(0);
            let z = s(5);
            if l.magic != MIXED_MAGIC
                || l.leaf != leaf as u32
                || l.state != 3
                || l.sticky_fault != 0
                || l.scheduled_step != step
                || l.committed_substeps != committed
                || l.generation != w.generation
                || l.preparation_generation != w.preparation_generation
                || l.body_ordinal != ordinal
                || l.reserved != 0
                || l.values.iter().any(|v| !v.is_finite())
                || l.values[18..].iter().any(|v| *v != 0.)
            {
                return Err(format!("mixed leaf identity body{ordinal} leaf{leaf}"));
            }
            for stage in 0..7 {
                let x = s(stage);
                let v = x.values;
                if x.magic != MIXED_MAGIC
                    || x.leaf != leaf as u32
                    || x.stage != stage as u32
                    || x.state != 1
                    || x.sticky_fault != 0
                    || x.scheduled_step != step
                    || x.committed_substeps != if stage == 0 { committed - 4 } else { committed }
                    || x.generation != w.generation
                    || x.preparation_generation != w.preparation_generation
                    || x.body_ordinal != ordinal
                    || x.source != l.source
                    || x.mode != l.mode
                    || x.dec_iterations > 192
                    || v.iter().any(|v| !v.is_finite())
                    || v[14] < 0.
                    || v[14] > 1e-6
                    || v[20..].iter().any(|v| *v != 0.)
                {
                    return Err(format!(
                        "mixed stage identity body{ordinal} leaf{leaf} stage{stage}"
                    ));
                }
                let model = v[4] + v[5] + v[6] + v[7] + v[8];
                let correction = v[3] - v[2] * KCAL_TO_HARTREE + model;
                if !close(v[9], model)
                    || !close(v[10], correction)
                    || !close(v[11], (v[0] + v[1]) * KCAL_TO_HARTREE + correction)
                    || !close(v[17], v[1] - v[2])
                {
                    return Err(format!(
                        "mixed stage reconstruction body{ordinal} leaf{leaf} stage{stage}"
                    ));
                }
            }
            let prefix = if ordinal == 0 {
                a.values[13] * KCAL_TO_HARTREE
            } else {
                0.
            };
            let native = if ordinal == 0 {
                a.values[12]
            } else {
                previous_native[leaf]
            };
            if ordinal > 0 && a.values[12] != native {
                return Err("native work reference was reset".into());
            }
            let prior = if ordinal == 0 {
                a.values[11]
            } else {
                previous_leaf[leaf]
            };
            let pk = if ordinal == 0 {
                a.values[3] + a.values[17] * KCAL_TO_HARTREE
            } else {
                previous_kinetic[leaf]
            };
            let ck = z.values[3] + z.values[17] * KCAL_TO_HARTREE;
            let native_work = (z.values[12] - native) * KCAL_TO_HARTREE - prefix;
            let terms = [
                native_work,
                l.values[1],
                s(2).values[10] - s(1).values[10],
                s(3).values[10] - s(2).values[10],
                s(4).values[10] - s(3).values[10],
            ];
            for (k, x) in terms.iter().enumerate() {
                if !close(l.values[k], *x) {
                    return Err(format!("mixed work term{k} body{ordinal} leaf{leaf}"));
                }
            }
            let declared = terms.iter().sum::<f64>();
            for (k, x) in [
                (5, declared),
                (6, z.values[11] - prior),
                (7, (z.values[11] - prior) - declared),
                (8, z.values[10] - s(6).values[10]),
                (9, prefix),
                (11, prior),
                (12, z.values[11]),
                (13, pk),
                (14, ck),
                (15, a.values[11]),
                (16, s(1).values[10] - a.values[10] - l.values[1]),
                (17, z.values[10] - s(4).values[10]),
            ] {
                if !close(l.values[k], x) {
                    return Err(format!("mixed leaf scalar{k} body{ordinal} leaf{leaf}"));
                }
            }
            if pk < 0. || ck < 0. {
                return Err("mixed kinetic reference is invalid".into());
            }
            previous_native[leaf] = z.values[12];
            previous_leaf[leaf] = z.values[11];
            previous_kinetic[leaf] = ck;
            start_sum += a.values[11];
            prev_sum += prior;
            total += z.values[11];
            work += declared;
            prior_k += pk;
            current_k += ck;
            prefix_sum += prefix;
        }
        if ordinal == 0 {
            initial = start_sum;
            prev = prev_sum;
            cumulative = 0.;
            initial_kinetic = prior_k;
        }
        let residual = (total - prev_sum) - work;
        cumulative += work;
        let run = (total - initial) - cumulative;
        prefix_total += prefix_sum;
        peak_body = peak_body.max(residual.abs());
        peak_run = peak_run.max(run.abs());
        if residual.abs() > MIXED_TOL || run.abs() > MIXED_TOL {
            return Err(format!(
                "reconstructed mixed energy fails body{ordinal}:body={residual} run={run}"
            ));
        }
        for (k, x) in [
            (0, initial),
            (1, prev_sum),
            (2, total),
            (3, work),
            (4, residual),
            (5, cumulative),
            (6, initial + cumulative),
            (7, run),
            (8, prior_k),
            (9, current_k),
            (10, prefix_sum),
            (11, peak_body),
            (12, peak_run),
            (14, prefix_total),
        ] {
            if !close(g.values[k], x) {
                return Err(format!("mixed global scalar{k} body{ordinal}"));
            }
        }
        if !close(prev, prev_sum) {
            return Err("mixed prior reference changed without accepted work".into());
        }
        prev = total;
    }
    let last = r.globals.last().unwrap();
    let after = r.after;
    if after.magic != MIXED_MAGIC
        || after.state != 3
        || after.sticky_fault != 0
        || after.generation != w.generation
        || after.preparation_generation != w.preparation_generation
        || after.accepted_bodies != u64::from(w.first_body) + n as u64
        || after.last_accepted_step != last.scheduled_step
        || after.reserved != [0; 3]
        || after.reserved_tail.iter().any(|v| *v != 0.)
        || after.totals.iter().any(|v| !v.is_finite())
        || after.totals[9..].iter().any(|v| *v != 0.)
        || !close(after.totals[0], initial)
        || !close(after.totals[1], prev)
        || !close(after.totals[2], initial + cumulative)
        || !close(after.totals[3], cumulative)
        || !close(after.totals[4], initial_kinetic)
        || !close(after.totals[5], last.values[9])
        || !close(after.totals[6], prefix_total)
        || !close(after.totals[7], peak_body)
        || !close(after.totals[8], peak_run)
        || after.previous_native_work != previous_native
        || after.previous_leaf_energy != previous_leaf
        || after.previous_leaf_kinetic != previous_kinetic
    {
        return Err("mixed persistent accepted reference mismatch".into());
    }
    Ok(())
}
pub(crate) struct MixedGuardPointers {
    pub state: u64,
    pub pending: u64,
    pub leaves: u64,
    pub samples: u64,
    pub window: u64,
    pub globals_journal: u64,
    pub leaves_journal: u64,
    pub samples_journal: u64,
}
pub(crate) struct BiprismMixedEnergyOwner {
    stream: Arc<CudaStream>,
    bindings: CudaSlice<MixedBinding>,
    samples: CudaSlice<MixedSample>,
    vertices: CudaSlice<MixedVertex>,
    edges: CudaSlice<MixedEdge>,
    leaves: CudaSlice<MixedLeafLedger>,
    pending: CudaSlice<MixedGlobalLedger>,
    state: CudaSlice<MixedEnergyState>,
    window: CudaSlice<MixedWindow>,
    globals_journal: CudaSlice<MixedGlobalLedger>,
    leaves_journal: CudaSlice<MixedLeafLedger>,
    samples_journal: CudaSlice<MixedSample>,
    sample: CudaFunction,
    evaluate: CudaFunction,
    capacity: u32,
    generation: u64,
    mode: u32,
    kappa: f32,
    current: Option<(MixedWindow, MixedEnergyState)>,
}
pub(crate) struct MixedEndpointBinding<'a> {
    owner: &'a BiprismMixedEnergyOwner,
    views: BiprismDeviceViews,
    shared: u64,
    leaf: u32,
}
impl MixedEndpointBinding<'_> {
    pub(crate) unsafe fn after_native_nonham(&self) -> Result<(), String> {
        self.owner
            .sample_one(self.views, self.shared, self.leaf, MixedStage::PostNonham)
    }
    pub(crate) unsafe fn after_native_ca(&self) -> Result<(), String> {
        self.owner
            .sample_one(self.views, self.shared, self.leaf, MixedStage::PostCa)
    }
}
fn upload<T: DeviceRepr>(s: &Arc<CudaStream>, x: &[T]) -> Result<CudaSlice<T>, String> {
    let r = s.clone_htod(x);
    let drain = s.synchronize();
    match (r, drain) {
        (Ok(v), Ok(())) => Ok(v),
        (Err(e), Ok(())) => Err(e.to_string()),
        (Ok(_), Err(e)) => Err(e.to_string()),
        (Err(a), Err(b)) => Err(format!("copy:{a};completion:{b}")),
    }
}
fn download<T: DeviceRepr + Default + Clone>(
    s: &Arc<CudaStream>,
    x: &CudaSlice<T>,
) -> Result<Vec<T>, String> {
    let mut out = vec![T::default(); x.len()];
    let c = s.memcpy_dtoh(x, &mut out);
    let d = s.synchronize();
    match (c, d) {
        (Ok(()), Ok(())) => Ok(out),
        (Err(e), Ok(())) => Err(e.to_string()),
        (Ok(()), Err(e)) => Err(e.to_string()),
        (Err(a), Err(b)) => Err(format!("copy:{a};completion:{b}")),
    }
}
impl BiprismMixedEnergyOwner {
    pub(crate) fn new(
        stream: Arc<CudaStream>,
        module: &Arc<CudaModule>,
        bindings: [MixedBinding; 8],
        capacity: u32,
        generation: u64,
        mode: u32,
        kappa: f32,
    ) -> Result<Self, String> {
        if !(1..=500).contains(&capacity)
            || generation == 0
            || !kappa.is_finite()
            || bindings.iter().enumerate().any(|(i, b)| {
                b.atoms < 256
                    || b.native_hmr_masses == 0
                    || b.native_potential == 0
                    || b.native_atom_work == 0
                    || b.native_cumulative_work == 0
                    || b.reserved != 0
                    || bindings[..i].iter().any(|q| q.source == b.source)
            })
        {
            return Err("mixed owner binding/generation/capacity".into());
        }
        Ok(Self {
            sample: module
                .load_function("nhs_biprism_mixed_sample_v1")
                .map_err(|e| e.to_string())?,
            evaluate: module
                .load_function("nhs_biprism_mixed_evaluate_v1")
                .map_err(|e| e.to_string())?,
            bindings: upload(&stream, &bindings)?,
            samples: upload(&stream, &vec![MixedSample::default(); 56])?,
            vertices: upload(&stream, &vec![MixedVertex::default(); 56 * 256])?,
            edges: upload(&stream, &vec![MixedEdge::default(); 56 * 1536])?,
            leaves: upload(&stream, &[MixedLeafLedger::default(); 8])?,
            pending: upload(&stream, &[MixedGlobalLedger::default()])?,
            state: upload(&stream, &[MixedEnergyState::default()])?,
            window: upload(&stream, &[MixedWindow::default()])?,
            globals_journal: upload(
                &stream,
                &vec![MixedGlobalLedger::default(); capacity as usize],
            )?,
            leaves_journal: upload(
                &stream,
                &vec![MixedLeafLedger::default(); capacity as usize * 8],
            )?,
            samples_journal: upload(
                &stream,
                &vec![MixedSample::default(); capacity as usize * 56],
            )?,
            stream,
            capacity,
            generation,
            mode,
            kappa,
            current: None,
        })
    }
    pub(crate) fn begin_window(
        &mut self,
        body: u64,
        step: u32,
        count: u32,
        prep: u64,
    ) -> Result<(), String> {
        if self.current.is_some()
            || count == 0
            || count > self.capacity
            || body >= 0x3fffffff
            || body + u64::from(count) >= 0x3fffffff
            || prep == 0
            || step.checked_add(count).is_none()
        {
            return Err("mixed window chronology/capacity".into());
        }
        let before = self.read_state()?;
        if before.accepted_bodies != body || before.sticky_fault != 0 {
            return Err("mixed reference does not match next window".into());
        }
        let w = MixedWindow {
            magic: MIXED_MAGIC,
            capacity: self.capacity,
            first_body: body as u32,
            requested_bodies: count,
            first_step: step,
            generation: self.generation,
            preparation_generation: prep,
            ..Default::default()
        };
        let rows = [w];
        let c = self.stream.memcpy_htod(&rows, &mut self.window);
        let d = self.stream.synchronize();
        match (c, d) {
            (Ok(()), Ok(())) => {}
            (Err(a), Ok(())) => return Err(a.to_string()),
            (Ok(()), Err(b)) => return Err(b.to_string()),
            (Err(a), Err(b)) => return Err(format!("copy:{a};completion:{b}")),
        }
        self.current = Some((w, before));
        Ok(())
    }
    unsafe fn sample_one(
        &self,
        v: BiprismDeviceViews,
        shared: u64,
        leaf: u32,
        stage: MixedStage,
    ) -> Result<(), String> {
        let p = self.guard_pointers();
        let b = self.bindings.device_ptr(&self.stream).0;
        let vertices = self.vertices.device_ptr(&self.stream).0;
        let edges = self.edges.device_ptr(&self.stream).0;
        self.stream
            .launch_builder(&self.sample)
            .arg(&b)
            .arg(&v.tiles)
            .arg(&v.native_stream_table)
            .arg(&v.stream_table)
            .arg(&v.charges)
            .arg(&shared)
            .arg(&v.convergence)
            .arg(&p.window)
            .arg(&leaf)
            .arg(&(stage as u32))
            .arg(&self.mode)
            .arg(&self.kappa)
            .arg(&p.samples)
            .arg(&vertices)
            .arg(&edges)
            .launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|e| e.to_string())?;
        Ok(())
    }
    pub(crate) unsafe fn sample_all(
        &self,
        v: BiprismDeviceViews,
        shared: u64,
        stage: MixedStage,
    ) -> Result<(), String> {
        for leaf in 0..8 {
            self.sample_one(v, shared, leaf, stage)?;
        }
        Ok(())
    }
    pub(crate) fn endpoint(
        &self,
        v: BiprismDeviceViews,
        shared: u64,
        leaf: u32,
    ) -> MixedEndpointBinding<'_> {
        MixedEndpointBinding {
            owner: self,
            views: v,
            shared,
            leaf,
        }
    }
    pub(crate) unsafe fn evaluate(
        &self,
        v: BiprismDeviceViews,
        shared: u64,
        phase2: u64,
        lif: u64,
    ) -> Result<(), String> {
        let p = self.guard_pointers();
        let vertices = self.vertices.device_ptr(&self.stream).0;
        self.stream
            .launch_builder(&self.evaluate)
            .arg(&p.samples)
            .arg(&vertices)
            .arg(&phase2)
            .arg(&lif)
            .arg(&shared)
            .arg(&p.window)
            .arg(&p.state)
            .arg(&p.leaves)
            .arg(&p.pending)
            .arg(&v.convergence)
            .launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|e| e.to_string())?;
        Ok(())
    }
    pub(crate) fn guard_pointers(&self) -> MixedGuardPointers {
        let s = &self.stream;
        MixedGuardPointers {
            state: self.state.device_ptr(s).0,
            pending: self.pending.device_ptr(s).0,
            leaves: self.leaves.device_ptr(s).0,
            samples: self.samples.device_ptr(s).0,
            window: self.window.device_ptr(s).0,
            globals_journal: self.globals_journal.device_ptr(s).0,
            leaves_journal: self.leaves_journal.device_ptr(s).0,
            samples_journal: self.samples_journal.device_ptr(s).0,
        }
    }
    pub(crate) fn read_state(&self) -> Result<MixedEnergyState, String> {
        Ok(download(&self.stream, &self.state)?[0])
    }
    pub(crate) fn read_current(&self) -> Result<MixedCurrentAttempt, String> {
        Ok(MixedCurrentAttempt {
            state: self.read_state()?,
            pending: download(&self.stream, &self.pending)?[0],
            leaves: download(&self.stream, &self.leaves)?,
            samples: download(&self.stream, &self.samples)?,
            vertices: download(&self.stream, &self.vertices)?,
            edges: download(&self.stream, &self.edges)?,
        })
    }
    pub(crate) fn read_window(&self) -> Result<MixedReadback, String> {
        let (window, before) = self.current.ok_or("no mixed window")?;
        let n = window.requested_bodies as usize;
        let mut globals = download(&self.stream, &self.globals_journal)?;
        globals.truncate(n);
        let mut leaves = download(&self.stream, &self.leaves_journal)?;
        leaves.truncate(n * 8);
        let mut samples = download(&self.stream, &self.samples_journal)?;
        samples.truncate(n * 56);
        Ok(MixedReadback {
            window,
            before,
            after: self.read_state()?,
            globals,
            leaves,
            samples,
        })
    }
    pub(crate) fn acknowledge(&mut self) {
        self.current = None;
    }
}
