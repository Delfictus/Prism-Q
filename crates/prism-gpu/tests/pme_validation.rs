//! Focused reciprocal-PME and rigid-water validation.
//!
//! GPU tests are ignored by default so CPU-only builders can compile the
//! crate.  Execute the science gate with:
//! `cargo test -p prism-gpu --features cuda --test pme_validation -- --ignored --nocapture`

use cudarc::driver::{sys, CudaContext};
use prism_gpu::pme::{lambda_internal_pme_correction_coefficients, PmeSplitContract, PME};
use prism_gpu::settle::{
    RigidWaterAdmission, RigidWaterParameters, Settle, TIP3P_HH_DISTANCE, TIP3P_MASS_H,
    TIP3P_MASS_O, TIP3P_OH_DISTANCE,
};

const KE: f64 = 332.0636;

#[test]
fn persistent_slot_operator_coefficients_close_insertion_and_identity_measures() {
    let us = 13.75f64;
    for lambda in [0.0f64, 0.13, 0.5, 0.91, 1.0] {
        // The uncorrected isolated slot PME is quadratic in q=lambda*q_s.
        let current = lambda * lambda * us;
        let current_dudl = 2.0 * lambda * us;
        let mut corrected = current;
        let mut corrected_dudl = current_dudl;
        for (source, energy, dudl) in [(1, us, 0.0)] {
            let (energy_scale, energy_to_dudl, pme_dudl_scale) =
                lambda_internal_pme_correction_coefficients(lambda, 2, source).unwrap();
            corrected += energy_scale * energy;
            corrected_dudl += energy_to_dudl * energy + pme_dudl_scale * dudl;
        }
        assert!((corrected - us).abs() <= 1.0e-12);
        assert!(corrected_dudl.abs() <= 1.0e-12);

        let ua = -2.25;
        let ub = 8.5;
        let qmix_operator = 1.7 + 3.1 * lambda + 2.4 * lambda * lambda;
        let qmix_dudl = 3.1 + 4.8 * lambda;
        let mut identity = qmix_operator;
        let mut identity_dudl = qmix_dudl;
        for (source, energy, dudl) in [(0, ua, 0.0), (1, ub, 0.0), (2, qmix_operator, qmix_dudl)] {
            let (energy_scale, energy_to_dudl, pme_dudl_scale) =
                lambda_internal_pme_correction_coefficients(lambda, 3, source).unwrap();
            identity += energy_scale * energy;
            identity_dudl += energy_to_dudl * energy + pme_dudl_scale * dudl;
        }
        assert!((identity - ((1.0 - lambda) * ua + lambda * ub)).abs() <= 1.0e-12);
        assert!((identity_dudl - (ub - ua)).abs() <= 1.0e-12);
    }
}

fn erfc_reference(x: f64) -> f64 {
    // Numerical Recipes approximation; sufficient for an independent
    // real-space composition oracle at the single-precision GPU threshold.
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let tau = t
        * (-z * z - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    if x >= 0.0 {
        tau
    } else {
        2.0 - tau
    }
}

fn add_real_space_ewald(
    positions: &[[f64; 3]],
    charges: &[f64],
    box_dims: [f64; 3],
    alpha: f64,
    cutoff: f64,
    energy: &mut f64,
    forces: &mut [f64],
) {
    for i in 0..positions.len() {
        for j in i + 1..positions.len() {
            let mut displacement = [0.0; 3];
            for axis in 0..3 {
                displacement[axis] = positions[i][axis] - positions[j][axis];
                displacement[axis] -=
                    box_dims[axis] * (displacement[axis] / box_dims[axis]).round();
            }
            let r2 = displacement.iter().map(|value| value * value).sum::<f64>();
            let r = r2.sqrt();
            if r >= cutoff {
                continue;
            }
            let charge_product = charges[i] * charges[j];
            let ar = alpha * r;
            let erfc = erfc_reference(ar);
            *energy += KE * charge_product * erfc / r;
            let coefficient = KE
                * charge_product
                * (erfc / (r2 * r)
                    + 2.0 * alpha / std::f64::consts::PI.sqrt() * (-ar * ar).exp() / r2);
            for axis in 0..3 {
                let force = coefficient * displacement[axis];
                forces[i * 3 + axis] += force;
                forces[j * 3 + axis] -= force;
            }
        }
    }
}

fn tip3pfb_admission(n_waters: usize) -> RigidWaterAdmission {
    let oh = 1.011_810_824_94_f64;
    let angle = 1.887_546_402_88_f64;
    RigidWaterAdmission {
        model_id: "TIP3P-FB".into(),
        // Synthetic focused-test contract. Production binds the exact
        // PRISM-PREP water_model_contract.contract_sha256 at construction.
        contract_sha256: "1f".repeat(32),
        n_waters,
        atom_order: "OHH".into(),
        oh_distances_angstrom: [oh, oh],
        hh_distance_angstrom: 2.0 * oh * (0.5 * angle).sin(),
        hoh_angle_radians: angle,
        masses_dalton: [15.999_43, 1.007_947, 1.007_947],
    }
}

fn direct_reciprocal_ewald(
    positions: &[[f64; 3]],
    charges: &[f64],
    dcharges: &[f64],
    box_dims: [f64; 3],
    beta: f64,
    kmax: i32,
) -> (f64, f64) {
    let volume = box_dims.iter().product::<f64>();
    let mut energy = 0.0;
    let mut dudl = 0.0;
    for mx in -kmax..=kmax {
        for my in -kmax..=kmax {
            for mz in -kmax..=kmax {
                if mx == 0 && my == 0 && mz == 0 {
                    continue;
                }
                let k = [
                    2.0 * std::f64::consts::PI * mx as f64 / box_dims[0],
                    2.0 * std::f64::consts::PI * my as f64 / box_dims[1],
                    2.0 * std::f64::consts::PI * mz as f64 / box_dims[2],
                ];
                let k2 = k.iter().map(|x| x * x).sum::<f64>();
                let influence =
                    4.0 * std::f64::consts::PI / (volume * k2) * (-k2 / (4.0 * beta * beta)).exp();
                let mut sre = 0.0;
                let mut sim = 0.0;
                let mut dsre = 0.0;
                let mut dsim = 0.0;
                for ((r, q), dq) in positions.iter().zip(charges).zip(dcharges) {
                    let phase = k[0] * r[0] + k[1] * r[1] + k[2] * r[2];
                    sre += q * phase.cos();
                    sim -= q * phase.sin();
                    dsre += dq * phase.cos();
                    dsim -= dq * phase.sin();
                }
                energy += 0.5 * KE * influence * (sre * sre + sim * sim);
                dudl += KE * influence * (sre * dsre + sim * dsim);
            }
        }
    }
    let q = charges.iter().sum::<f64>();
    let dq = dcharges.iter().sum::<f64>();
    energy -= KE * beta / std::f64::consts::PI.sqrt() * charges.iter().map(|x| x * x).sum::<f64>();
    dudl -= 2.0 * KE * beta / std::f64::consts::PI.sqrt()
        * charges
            .iter()
            .zip(dcharges)
            .map(|(q, dq)| q * dq)
            .sum::<f64>();
    energy -= KE * std::f64::consts::PI * q * q / (2.0 * beta * beta * volume);
    dudl -= KE * std::f64::consts::PI * q * dq / (beta * beta * volume);
    (energy, dudl)
}

fn run_gpu_pme(
    positions: &[[f64; 3]],
    charges: &[f64],
    dcharges: &[f64],
    box_dims: [f64; 3],
) -> (prism_gpu::pme::PmeReciprocalResult, Vec<f32>) {
    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.new_stream().expect("non-default engine stream");
    let mut pme = PME::new_on_stream(
        context,
        stream.clone(),
        positions.len(),
        box_dims.map(|x| x as f32),
        PmeSplitContract::external_real_space_and_exclusions(),
    )
    .expect("PME construction");
    let pos: Vec<f32> = positions
        .iter()
        .flat_map(|r| r.iter().map(|x| *x as f32))
        .collect();
    let q: Vec<f32> = charges.iter().map(|x| *x as f32).collect();
    let dq: Vec<f32> = dcharges.iter().map(|x| *x as f32).collect();
    let d_pos = stream.clone_htod(&pos).unwrap();
    let d_q = stream.clone_htod(&q).unwrap();
    let d_dq = stream.clone_htod(&dq).unwrap();
    let mut d_forces = stream.alloc_zeros::<f32>(pos.len()).unwrap();
    pme.enqueue_reciprocal_with_charge_derivatives(&d_pos, &d_q, &d_dq, &mut d_forces)
        .expect("zero-DtoH enqueue");
    let result = pme.readback_reciprocal_result().unwrap();
    let mut forces = vec![0.0; pos.len()];
    stream.memcpy_dtoh(&d_forces, &mut forces).unwrap();
    let receipt = pme.capability_receipt();
    assert_eq!(receipt.graph_hot_path_dtoh_count, 0);
    assert!(receipt.cufft_stream_bound && receipt.cufft_plan_handles_nonzero);
    (result, forces)
}

#[test]
#[ignore = "requires CUDA GPU"]
fn pme_matches_independent_direct_ewald_neutral_and_charged() {
    let box_dims = [24.0, 21.0, 27.0];
    let positions = vec![[3.1, 4.7, 5.3], [11.2, 9.4, 7.8], [17.3, 15.1, 19.7]];
    for (charges, dcharges) in [
        (vec![0.7, -0.2, -0.5], vec![0.12, -0.04, -0.08]),
        (vec![0.7, -0.2, 0.1], vec![0.12, -0.04, 0.03]),
    ] {
        let (gpu, _) = run_gpu_pme(&positions, &charges, &dcharges, box_dims);
        let beta = prism_gpu::pme::compute_ewald_beta(12.0, 1e-5) as f64;
        let (reference_energy, reference_dudl) =
            direct_reciprocal_ewald(&positions, &charges, &dcharges, box_dims, beta, 14);
        let energy_err = (gpu.reciprocal_energy - reference_energy).abs();
        let dudl_err = (gpu.reciprocal_dudlambda.unwrap() - reference_dudl).abs();
        eprintln!(
            "PME reference: gpu_E={:.9} ref_E={:.9} |dE|={:.3e}; gpu_dUdl={:.9} ref_dUdl={:.9} |d|={:.3e}",
            gpu.reciprocal_energy,
            reference_energy,
            energy_err,
            gpu.reciprocal_dudlambda.unwrap(),
            reference_dudl,
            dudl_err
        );
        assert!(energy_err < 2.0e-2, "reciprocal energy error {energy_err}");
        assert!(dudl_err < 8.0e-3, "reciprocal dU/dlambda error {dudl_err}");
    }
}

#[test]
#[ignore = "requires CUDA GPU and the appliance OpenMM reference runtime"]
fn order5_composed_pme_matches_local_openmm_same_parameters() {
    use std::process::Command;

    let box_dims = [32.0_f64, 30.0, 34.0];
    let positions = vec![[3.1, 4.7, 5.3], [9.2, 8.4, 7.8], [14.3, 12.1, 11.7]];
    let charges = vec![0.7, -0.2, -0.5];
    let cutoff = 10.0_f64;
    let tolerance = 1.0e-5_f64;
    let alpha = prism_gpu::pme::compute_ewald_beta(cutoff as f32, tolerance as f32) as f64;
    let grid = [72, 72, 75];

    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.new_stream().expect("engine stream");
    let mut pme = PME::new_on_stream_with_exact_parameters(
        context,
        stream.clone(),
        positions.len(),
        box_dims.map(|value| value as f32),
        grid,
        5,
        alpha as f32,
        cutoff as f32,
        tolerance as f32,
        PmeSplitContract::external_real_space_and_exclusions(),
    )
    .expect("exact order-five PME construction");
    let flat_positions: Vec<f32> = positions
        .iter()
        .flat_map(|position| position.iter().map(|value| *value as f32))
        .collect();
    let q: Vec<f32> = charges.iter().map(|value| *value as f32).collect();
    let d_positions = stream.clone_htod(&flat_positions).unwrap();
    let d_charges = stream.clone_htod(&q).unwrap();
    let mut d_forces = stream.alloc_zeros::<f32>(flat_positions.len()).unwrap();
    let reciprocal = pme
        .compute_reciprocal(&d_positions, &d_charges, &mut d_forces)
        .unwrap();
    let mut reciprocal_forces = vec![0.0_f32; flat_positions.len()];
    stream
        .memcpy_dtoh(&d_forces, &mut reciprocal_forces)
        .unwrap();
    let mut prism_energy = reciprocal.reciprocal_energy;
    let mut prism_forces: Vec<f64> = reciprocal_forces
        .iter()
        .map(|value| *value as f64)
        .collect();
    add_real_space_ewald(
        &positions,
        &charges,
        box_dims,
        alpha,
        cutoff,
        &mut prism_energy,
        &mut prism_forces,
    );

    let openmm_python = std::env::var("PRISM_OPENMM_PYTHON").expect(
        "PRISM_OPENMM_PYTHON must name the appliance OpenMM reference Python interpreter",
    );
    let script = format!(
        r#"
import openmm
from openmm import unit
system = openmm.System()
for _ in range(3): system.addParticle(12.0)
system.setDefaultPeriodicBoxVectors(openmm.Vec3({lx},0,0), openmm.Vec3(0,{ly},0), openmm.Vec3(0,0,{lz}))
force = openmm.NonbondedForce()
force.setNonbondedMethod(openmm.NonbondedForce.PME)
force.setCutoffDistance({cutoff})
force.setEwaldErrorTolerance({tol})
force.setPMEParameters({alpha}, {nx}, {ny}, {nz})
force.setUseDispersionCorrection(False)
for q in [{q0}, {q1}, {q2}]: force.addParticle(q, 0.3, 0.0)
system.addForce(force)
integrator = openmm.VerletIntegrator(0.001)
context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName('Reference'))
context.setPositions([openmm.Vec3({x0},{y0},{z0}), openmm.Vec3({x1},{y1},{z1}), openmm.Vec3({x2},{y2},{z2})])
state = context.getState(getEnergy=True, getForces=True)
energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
forces = state.getForces(asNumpy=True).value_in_unit(unit.kilocalories_per_mole/unit.angstrom)
print(' '.join([repr(float(energy))]+[repr(float(v)) for row in forces for v in row]))
"#,
        lx = box_dims[0] * 0.1,
        ly = box_dims[1] * 0.1,
        lz = box_dims[2] * 0.1,
        cutoff = cutoff * 0.1,
        tol = tolerance,
        alpha = alpha * 10.0,
        nx = grid[0],
        ny = grid[1],
        nz = grid[2],
        q0 = charges[0],
        q1 = charges[1],
        q2 = charges[2],
        x0 = positions[0][0] * 0.1,
        y0 = positions[0][1] * 0.1,
        z0 = positions[0][2] * 0.1,
        x1 = positions[1][0] * 0.1,
        y1 = positions[1][1] * 0.1,
        z1 = positions[1][2] * 0.1,
        x2 = positions[2][0] * 0.1,
        y2 = positions[2][1] * 0.1,
        z2 = positions[2][2] * 0.1,
    );
    let output = Command::new(&openmm_python)
        .arg("-c")
        .arg(script)
        .output()
        .expect("launch local OpenMM reference");
    assert!(
        output.status.success(),
        "OpenMM reference failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let values: Vec<f64> = String::from_utf8(output.stdout)
        .unwrap()
        .split_whitespace()
        .map(|value| value.parse().unwrap())
        .collect();
    assert_eq!(values.len(), 10);
    let openmm_energy = values[0];
    let openmm_forces = &values[1..];
    let energy_error = (prism_energy - openmm_energy).abs();
    let max_force_error = prism_forces
        .iter()
        .zip(openmm_forces)
        .map(|(prism, openmm)| (prism - openmm).abs())
        .fold(0.0_f64, f64::max);
    eprintln!(
        "order5 composed parity: prism_E={prism_energy:.9} openmm_E={openmm_energy:.9} |dE|={energy_error:.3e} max|dF|={max_force_error:.3e}"
    );
    assert!(
        energy_error <= 2.0e-3,
        "OpenMM energy parity error {energy_error}"
    );
    assert!(
        max_force_error <= 2.0e-3,
        "OpenMM force-component parity error {max_force_error}"
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn pme_runtime_receipt_proves_fd_translation_and_atom_order_bounds() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let mut pme = PME::new_on_stream(
        context,
        stream,
        4,
        [24.0, 21.0, 27.0],
        PmeSplitContract::external_real_space_and_exclusions(),
    )
    .unwrap();
    let receipt = pme.validate_runtime_capability().unwrap();
    let evidence = receipt.validation.expect("validation evidence");
    assert_eq!(evidence.status, "PASSED");
    assert!(evidence.force_energy_rel_error <= 5.0e-3);
    assert!(evidence.translation_energy_abs_error <= 3.0e-3);
    assert!(evidence.charge_dudlambda_rel_error <= 5.0e-3);
    assert!(evidence.atom_order_energy_abs_error <= 2.0e-5);
    assert!(evidence.atom_order_dudlambda_abs_error <= 2.0e-5);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn pme_atom_resolved_lambda_components_close_scalar_energy_and_dudl() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let positions = vec![
        3.7f32, 5.2, 7.1, 11.3, 8.4, 4.9, 17.2, 14.1, 19.3, 21.4, 18.2, 9.7,
    ];
    let charges = vec![0.41f32, -0.73, 0.19, 0.13];
    let dcharges = vec![-0.17f32, 0.09, 0.05, 0.03];
    let d_positions = stream.clone_htod(&positions).unwrap();
    let d_charges = stream.clone_htod(&charges).unwrap();
    let d_dcharges = stream.clone_htod(&dcharges).unwrap();
    let d_slots = stream.clone_htod(&[0i32, 0, 0, 0]).unwrap();
    let mut d_forces = stream.alloc_zeros::<f32>(positions.len()).unwrap();
    let mut total_atom_energy = stream.alloc_zeros::<f64>(charges.len()).unwrap();
    let mut lambda_atom_energy = stream.alloc_zeros::<f64>(charges.len()).unwrap();
    let mut lambda_atom_dudl = stream.alloc_zeros::<f64>(charges.len()).unwrap();
    let mut pme = PME::new_on_stream(
        context,
        stream.clone(),
        charges.len(),
        [28.0, 29.0, 31.0],
        PmeSplitContract::external_real_space_and_exclusions(),
    )
    .unwrap();
    pme.enqueue_reciprocal_with_charge_derivatives(
        &d_positions,
        &d_charges,
        &d_dcharges,
        &mut d_forces,
    )
    .unwrap();
    pme.enqueue_merge_lambda_atom_components(
        &d_slots,
        &mut total_atom_energy,
        &mut lambda_atom_energy,
        &mut lambda_atom_dudl,
    )
    .unwrap();
    let scalar = pme.readback_reciprocal_result().unwrap();
    let mut total = vec![0.0f64; charges.len()];
    let mut lambda_energy = vec![0.0f64; charges.len()];
    let mut lambda_dudl = vec![0.0f64; charges.len()];
    stream.memcpy_dtoh(&total_atom_energy, &mut total).unwrap();
    stream
        .memcpy_dtoh(&lambda_atom_energy, &mut lambda_energy)
        .unwrap();
    stream
        .memcpy_dtoh(&lambda_atom_dudl, &mut lambda_dudl)
        .unwrap();
    let total_sum = total.iter().sum::<f64>();
    let lambda_energy_sum = lambda_energy.iter().sum::<f64>();
    let lambda_dudl_sum = lambda_dudl.iter().sum::<f64>();
    assert!((total_sum - scalar.reciprocal_energy).abs() <= 2.0e-8);
    assert!((lambda_energy_sum - scalar.reciprocal_energy).abs() <= 2.0e-8);
    assert!((lambda_dudl_sum - scalar.reciprocal_dudlambda.unwrap()).abs() <= 2.0e-8);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn pme_persistent_slot_internal_operator_is_lambda_invariant_including_background() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let positions = vec![3.7f32, 5.2, 7.1, 11.3, 8.4, 4.9];
    // Deliberately net charged: omitting self or neutralizing-background
    // correction makes this test fail even if the reciprocal mesh is restored.
    let endpoint_b = vec![0.63f32, -0.17];
    let endpoint_a = vec![0.0f32; 2];
    let slots = vec![0i32, 0];
    let endpoint_masks = vec![2i32]; // EMPTY A -> OCCUPIED B
    let d_positions = stream.clone_htod(&positions).unwrap();
    let d_endpoint_a = stream.clone_htod(&endpoint_a).unwrap();
    let d_endpoint_b = stream.clone_htod(&endpoint_b).unwrap();
    let d_slots = stream.clone_htod(&slots).unwrap();
    let d_masks = stream.clone_htod(&endpoint_masks).unwrap();
    let d_zero_dq = stream.clone_htod(&vec![0.0f32; 2]).unwrap();
    let mut reference_forces = stream.alloc_zeros::<f32>(6).unwrap();
    let mut pme = PME::new_on_stream(
        context,
        stream.clone(),
        2,
        [28.0, 29.0, 31.0],
        PmeSplitContract::external_real_space_and_exclusions(),
    )
    .unwrap();
    let reference = pme
        .compute_reciprocal_with_charge_derivatives(
            &d_positions,
            &d_endpoint_b,
            &d_zero_dq,
            &mut reference_forces,
        )
        .unwrap();
    let mut reference_total_energy = stream.alloc_zeros::<f64>(2).unwrap();
    let mut reference_lambda_energy = stream.alloc_zeros::<f64>(2).unwrap();
    let mut reference_lambda_dudl = stream.alloc_zeros::<f64>(2).unwrap();
    pme.enqueue_merge_lambda_atom_components(
        &d_slots,
        &mut reference_total_energy,
        &mut reference_lambda_energy,
        &mut reference_lambda_dudl,
    )
    .unwrap();
    let mut reference_total_host = vec![0.0f64; 2];
    stream
        .memcpy_dtoh(&reference_total_energy, &mut reference_total_host)
        .unwrap();
    let reference_atom_energy = reference_total_host.iter().sum::<f64>();
    assert!((reference_atom_energy - reference.reciprocal_energy).abs() <= 1.0e-5);
    let mut reference_force_host = vec![0.0f32; 6];
    stream
        .memcpy_dtoh(&reference_forces, &mut reference_force_host)
        .unwrap();

    for lambda in [0.0f32, 0.37, 1.0] {
        let current = endpoint_b.iter().map(|q| lambda * q).collect::<Vec<_>>();
        let d_current = stream.clone_htod(&current).unwrap();
        let d_current_dq = stream.clone_htod(&endpoint_b).unwrap();
        let d_lambda = stream.clone_htod(&[lambda]).unwrap();
        let mut forces = stream.alloc_zeros::<f32>(6).unwrap();
        let mut total_energy = stream.alloc_zeros::<f64>(2).unwrap();
        let mut lambda_energy = stream.alloc_zeros::<f64>(2).unwrap();
        let mut lambda_dudl = stream.alloc_zeros::<f64>(2).unwrap();
        pme.enqueue_reciprocal_with_charge_derivatives(
            &d_positions,
            &d_current,
            &d_current_dq,
            &mut forces,
        )
        .unwrap();
        pme.enqueue_merge_lambda_atom_components(
            &d_slots,
            &mut total_energy,
            &mut lambda_energy,
            &mut lambda_dudl,
        )
        .unwrap();
        let mut isolated_q = stream.alloc_zeros::<f32>(2).unwrap();
        let mut isolated_dq = stream.alloc_zeros::<f32>(2).unwrap();
        let mut isolated_forces = stream.alloc_zeros::<f32>(6).unwrap();
        for source_plane in [1i32] {
            pme.enqueue_select_isolated_lambda_slot_charges(
                &d_slots,
                &d_endpoint_a,
                &d_endpoint_b,
                &d_current,
                &d_current_dq,
                0,
                source_plane,
                &mut isolated_q,
                &mut isolated_dq,
            )
            .unwrap();
            stream.memset_zeros(&mut isolated_forces).unwrap();
            pme.enqueue_reciprocal_with_charge_derivatives(
                &d_positions,
                &isolated_q,
                &isolated_dq,
                &mut isolated_forces,
            )
            .unwrap();
            pme.enqueue_accumulate_lambda_internal_correction(
                &isolated_forces,
                &d_slots,
                &d_lambda,
                &d_masks,
                0,
                source_plane,
                &mut forces,
                &mut total_energy,
                &mut lambda_energy,
                &mut lambda_dudl,
            )
            .unwrap();
        }
        let mut total = vec![0.0f64; 2];
        let mut dudl = vec![0.0f64; 2];
        let mut force_host = vec![0.0f32; 6];
        stream.memcpy_dtoh(&total_energy, &mut total).unwrap();
        stream.memcpy_dtoh(&lambda_dudl, &mut dudl).unwrap();
        stream.memcpy_dtoh(&forces, &mut force_host).unwrap();
        let total_sum = total.iter().sum::<f64>();
        let dudl_sum = dudl.iter().sum::<f64>();
        assert!(
            (total_sum - reference_atom_energy).abs() <= 2.0e-7,
            "lambda={lambda} corrected={total_sum} reference={} delta={}",
            reference_atom_energy,
            total_sum - reference_atom_energy,
        );
        assert!(dudl_sum.abs() <= 2.0e-7, "lambda={lambda} dudl={dudl_sum}");
        for (actual, expected) in force_host.iter().zip(&reference_force_host) {
            assert!((actual - expected).abs() <= 3.0e-5);
        }
    }
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}
fn rotate(v: [f64; 3], axis: [f64; 3], angle: f64) -> [f64; 3] {
    let axis = scale(axis, 1.0 / norm(axis));
    add(
        add(scale(v, angle.cos()), scale(cross(axis, v), angle.sin())),
        scale(axis, dot(axis, v) * (1.0 - angle.cos())),
    )
}
fn com(p: &[[f64; 3]; 3]) -> [f64; 3] {
    let mt = TIP3P_MASS_O as f64 + 2.0 * TIP3P_MASS_H as f64;
    scale(
        add(
            scale(p[0], TIP3P_MASS_O as f64),
            scale(add(p[1], p[2]), TIP3P_MASS_H as f64),
        ),
        1.0 / mt,
    )
}
fn momentum(v: &[[f64; 3]; 3]) -> [f64; 3] {
    add(
        scale(v[0], TIP3P_MASS_O as f64),
        scale(add(v[1], v[2]), TIP3P_MASS_H as f64),
    )
}
fn angular_momentum(p: &[[f64; 3]; 3], v: &[[f64; 3]; 3]) -> [f64; 3] {
    let c = com(p);
    let mut l = [0.0; 3];
    for i in 0..3 {
        let m = if i == 0 {
            TIP3P_MASS_O as f64
        } else {
            TIP3P_MASS_H as f64
        };
        l = add(l, scale(cross(sub(p[i], c), v[i]), m));
    }
    l
}

#[test]
#[ignore = "requires CUDA GPU"]
fn settle_randomized_water_box_preserves_geometry_com_orientation_and_momenta() {
    const N: usize = 64;
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let oxygens: Vec<usize> = (0..N).map(|w| 3 * w).collect();
    let mut settle = Settle::new_on_stream(context, stream.clone(), &oxygens, 3 * N).unwrap();
    let half = 104.52_f64.to_radians() / 2.0;
    let hx = TIP3P_OH_DISTANCE as f64 * half.sin();
    let hy = TIP3P_OH_DISTANCE as f64 * half.cos();
    let base = [[0.0, 0.0, 0.0], [hx, hy, 0.0], [-hx, hy, 0.0]];
    let base_com = com(&base);
    let mut old = Vec::with_capacity(9 * N);
    let mut unconstrained = Vec::with_capacity(9 * N);
    let mut unconstrained_coms = Vec::with_capacity(N);
    let mut target_normals = Vec::with_capacity(N);
    for w in 0..N {
        let axis = [0.3 + 0.01 * w as f64, -0.4, 0.8];
        let old_angle = 0.017 * w as f64;
        let step_angle = -0.035 + 0.001 * w as f64;
        let origin = [
            3.0 * (w % 4) as f64,
            3.0 * ((w / 4) % 4) as f64,
            3.0 * (w / 16) as f64,
        ];
        let old_p = base.map(|r| add(origin, rotate(sub(r, base_com), axis, old_angle)));
        let old_c = com(&old_p);
        let step_axis = [0.2, 0.7 - 0.003 * w as f64, -0.5];
        let translation = [0.013, -0.009, 0.006];
        let mut trial = old_p.map(|r| {
            add(
                add(old_c, rotate(sub(r, old_c), step_axis, step_angle)),
                translation,
            )
        });
        target_normals.push(cross(sub(trial[1], trial[0]), sub(trial[2], trial[0])));
        for atom in 0..3 {
            trial[atom][0] += (atom as f64 - 1.0) * 3.0e-4;
            trial[atom][1] += ((w + atom) % 3) as f64 * 2.0e-4;
            trial[atom][2] -= atom as f64 * 1.0e-4;
        }
        unconstrained_coms.push(com(&trial));
        old.extend(old_p.iter().flat_map(|r| r.map(|x| x as f32)));
        unconstrained.extend(trial.iter().flat_map(|r| r.map(|x| x as f32)));
    }
    let d_old = stream.clone_htod(&old).unwrap();
    settle.save_positions(&d_old).unwrap();
    let mut d_pos = stream.clone_htod(&unconstrained).unwrap();
    settle.reset_failure_latch().unwrap();
    settle.enqueue_apply(&mut d_pos).unwrap();
    assert_eq!(settle.readback_failure_latch().unwrap(), 0);
    let mut constrained = vec![0.0f32; unconstrained.len()];
    stream.memcpy_dtoh(&d_pos, &mut constrained).unwrap();
    let mut max_oh_error = 0.0_f64;
    let mut max_hh_error = 0.0_f64;
    let mut max_com_error = 0.0_f64;
    for w in 0..N {
        let p: [[f64; 3]; 3] = std::array::from_fn(|a| {
            std::array::from_fn(|d| constrained[(3 * w + a) * 3 + d] as f64)
        });
        let oh1_error = (norm(sub(p[1], p[0])) - TIP3P_OH_DISTANCE as f64).abs();
        let oh2_error = (norm(sub(p[2], p[0])) - TIP3P_OH_DISTANCE as f64).abs();
        let hh_error = (norm(sub(p[2], p[1])) - TIP3P_HH_DISTANCE as f64).abs();
        let com_error = norm(sub(com(&p), unconstrained_coms[w]));
        max_oh_error = max_oh_error.max(oh1_error).max(oh2_error);
        max_hh_error = max_hh_error.max(hh_error);
        max_com_error = max_com_error.max(com_error);
        assert!(oh1_error < 3.0e-5);
        assert!(oh2_error < 3.0e-5);
        assert!(hh_error < 3.0e-5);
        assert!(com_error < 5.0e-6);
        let normal = cross(sub(p[1], p[0]), sub(p[2], p[0]));
        assert!(
            dot(normal, target_normals[w]) > 0.0,
            "reflection at water {w}"
        );
    }

    // Randomized deterministic velocities: RATTLE must preserve linear and
    // angular momentum while driving all three velocity residuals to zero.
    let before_v: Vec<f32> = (0..3 * N)
        .flat_map(|i| {
            let x = i as f64;
            [
                (0.17 * x).sin() as f32,
                (0.11 * x + 0.3).cos() as f32,
                (0.07 * x - 0.2).sin() as f32,
            ]
        })
        .collect();
    let mut d_v = stream.clone_htod(&before_v).unwrap();
    settle.reset_failure_latch().unwrap();
    settle
        .enqueue_velocity_correction(&mut d_v, &d_pos)
        .unwrap();
    assert_eq!(settle.readback_failure_latch().unwrap(), 0);
    let mut after_v = vec![0.0f32; before_v.len()];
    stream.memcpy_dtoh(&d_v, &mut after_v).unwrap();
    let mut max_momentum_error = 0.0_f64;
    let mut max_angular_momentum_error = 0.0_f64;
    let mut max_velocity_residual = 0.0_f64;
    for w in 0..N {
        let p: [[f64; 3]; 3] = std::array::from_fn(|a| {
            std::array::from_fn(|d| constrained[(3 * w + a) * 3 + d] as f64)
        });
        let vb: [[f64; 3]; 3] =
            std::array::from_fn(|a| std::array::from_fn(|d| before_v[(3 * w + a) * 3 + d] as f64));
        let va: [[f64; 3]; 3] =
            std::array::from_fn(|a| std::array::from_fn(|d| after_v[(3 * w + a) * 3 + d] as f64));
        let momentum_error = norm(sub(momentum(&va), momentum(&vb)));
        let angular_momentum_error =
            norm(sub(angular_momentum(&p, &va), angular_momentum(&p, &vb)));
        max_momentum_error = max_momentum_error.max(momentum_error);
        max_angular_momentum_error = max_angular_momentum_error.max(angular_momentum_error);
        assert!(momentum_error < 2.0e-5);
        assert!(angular_momentum_error < 3.0e-5);
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            let residual = dot(sub(p[j], p[i]), sub(va[j], va[i]));
            max_velocity_residual = max_velocity_residual.max(residual.abs());
            assert!(residual.abs() < 2.0e-5, "velocity residual {residual}");
        }
    }
    eprintln!(
        "SETTLE 64-water maxima: OH={max_oh_error:.3e} A HH={max_hh_error:.3e} A COM={max_com_error:.3e} A linear_momentum={max_momentum_error:.3e} angular_momentum={max_angular_momentum_error:.3e} velocity_residual={max_velocity_residual:.3e}"
    );
    let receipt = settle.capability_receipt();
    assert_eq!(receipt.graph_hot_path_dtoh_count, 0);
    assert!(receipt.preserves_rigid_rotation && receipt.device_failure_latch);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn settle_device_failure_latch_rejects_degenerate_water() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let mut settle = Settle::new_on_stream(context, stream.clone(), &[0], 3).unwrap();
    let collinear = vec![0.0f32, 0.0, 0.0, 0.9572, 0.0, 0.0, 1.9144, 0.0, 0.0];
    let d_old = stream.clone_htod(&collinear).unwrap();
    settle.save_positions(&d_old).unwrap();
    let mut d_new = stream.clone_htod(&collinear).unwrap();
    settle.reset_failure_latch().unwrap();
    settle.enqueue_apply(&mut d_new).unwrap();
    assert!(settle.readback_failure_latch().unwrap() > 0);
}

#[test]
fn rigid_water_contract_rejects_tip3p_tip3pfb_identity_mismatch() {
    let tip3p = RigidWaterParameters::tip3p();
    let tip3p_fb = RigidWaterParameters::tip3p_fb_openmm();
    let admission = tip3pfb_admission(1);

    assert_ne!(tip3p.model_id, tip3p_fb.model_id);
    assert_ne!(tip3p.parameter_sha256, tip3p_fb.parameter_sha256);
    assert!(tip3p
        .validate_admission(&tip3p_fb.model_id, &tip3p_fb.parameter_sha256)
        .is_err());
    assert!(tip3p_fb
        .validate_admission(&tip3p.model_id, &tip3p.parameter_sha256)
        .is_err());
    tip3p_fb
        .validate_admission(&tip3p_fb.model_id, &tip3p_fb.parameter_sha256)
        .unwrap();
    assert!(tip3p.validate_topology_admission(&admission, 1).is_err());
    tip3p_fb.validate_topology_admission(&admission, 1).unwrap();
    eprintln!(
        "rigid-water identities: TIP3P={} TIP3P-FB={} TIP3P-FB-source={}",
        tip3p.parameter_sha256, tip3p_fb.parameter_sha256, tip3p_fb.source_artifact_sha256
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn settle_tip3pfb_receipt_and_geometry_match_topology_contract() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let parameters = RigidWaterParameters::tip3p_fb_openmm();
    let admission = tip3pfb_admission(1);
    let half_angle = 1.887_546_4_f64 / 2.0;
    let oh = parameters.oh_distance_angstrom as f64;
    let trial = vec![
        0.0f32,
        0.0,
        0.0,
        (oh * half_angle.sin()) as f32,
        (oh * half_angle.cos()) as f32,
        0.0,
        (-oh * half_angle.sin()) as f32,
        (oh * half_angle.cos()) as f32,
        0.0,
    ];
    let mut settle = Settle::new_on_stream_with_admission(
        context,
        stream.clone(),
        &[0],
        3,
        parameters.clone(),
        admission.clone(),
    )
    .unwrap();
    settle
        .validate_water_contract(&parameters.model_id, &parameters.parameter_sha256)
        .unwrap();
    settle.validate_topology_water_contract(&admission).unwrap();
    let d_old = stream.clone_htod(&trial).unwrap();
    settle.save_positions(&d_old).unwrap();
    let mut d_pos = stream.clone_htod(&trial).unwrap();
    settle.reset_failure_latch().unwrap();
    settle.enqueue_apply(&mut d_pos).unwrap();
    assert_eq!(settle.readback_failure_latch().unwrap(), 0);
    let (max_oh, max_hh) = settle.check_constraints(&d_pos).unwrap();
    assert!(max_oh < 3.0e-5, "TIP3P-FB OH residual {max_oh}");
    assert!(max_hh < 3.0e-5, "TIP3P-FB HH residual {max_hh}");
    let receipt = settle.capability_receipt();
    assert_eq!(receipt.water_model_id, "TIP3P-FB");
    assert!(receipt.topology_contract_bound);
    assert_eq!(
        receipt.water_contract_sha256.as_deref(),
        Some(admission.contract_sha256.as_str())
    );
    assert_eq!(receipt.water_parameter_sha256, parameters.parameter_sha256);
    assert_eq!(
        receipt.water_source_artifact_sha256,
        parameters.source_artifact_sha256
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn pme_and_settle_hot_paths_capture_and_replay_on_supplied_stream() {
    let context = CudaContext::new(0).unwrap();
    // cudarc's automatic cross-stream event bookkeeping creates dependencies
    // outside a CUDA capture.  This test owns one stream and therefore safely
    // uses explicit single-stream ordering, as the resident graph does.
    unsafe { context.disable_event_tracking() };
    let stream = context.new_stream().unwrap();
    let parameters = RigidWaterParameters::tip3p_fb_openmm();
    let admission = tip3pfb_admission(1);
    let half_angle = 1.887_546_4_f64 / 2.0;
    let oh = parameters.oh_distance_angstrom as f64;
    let positions = vec![
        4.0f32,
        5.0,
        6.0,
        4.0 + (oh * half_angle.sin()) as f32,
        5.0 + (oh * half_angle.cos()) as f32,
        6.0,
        4.0 - (oh * half_angle.sin()) as f32,
        5.0 + (oh * half_angle.cos()) as f32,
        6.0,
    ];
    let charges = vec![-0.848_448_7f32, 0.424_224_35, 0.424_224_35];
    let dcharges = vec![0.01f32, -0.004, -0.006];
    let mut d_positions = stream.clone_htod(&positions).unwrap();
    let d_charges = stream.clone_htod(&charges).unwrap();
    let d_dcharges = stream.clone_htod(&dcharges).unwrap();
    let mut d_forces = stream.alloc_zeros::<f32>(positions.len()).unwrap();
    let mut pme = PME::new_on_stream(
        context.clone(),
        stream.clone(),
        3,
        [20.0, 21.0, 22.0],
        PmeSplitContract::external_real_space_and_exclusions(),
    )
    .unwrap();
    let mut settle = Settle::new_on_stream_with_admission(
        context,
        stream.clone(),
        &[0],
        3,
        parameters,
        admission.clone(),
    )
    .unwrap();
    settle.save_positions(&d_positions).unwrap();

    // Warm cuFFT and both module launch paths before capture.
    pme.enqueue_reciprocal_with_charge_derivatives(
        &d_positions,
        &d_charges,
        &d_dcharges,
        &mut d_forces,
    )
    .unwrap();
    settle.reset_failure_latch().unwrap();
    settle.enqueue_apply(&mut d_positions).unwrap();
    settle
        .enqueue_velocity_correction(&mut d_forces, &d_positions)
        .unwrap();
    pme.readback_reciprocal_result().unwrap();
    assert_eq!(settle.readback_failure_latch().unwrap(), 0);
    settle.reset_failure_latch().unwrap();

    stream
        .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        .unwrap();
    pme.enqueue_reciprocal_with_charge_derivatives(
        &d_positions,
        &d_charges,
        &d_dcharges,
        &mut d_forces,
    )
    .unwrap();
    settle.enqueue_apply(&mut d_positions).unwrap();
    settle
        .enqueue_velocity_correction(&mut d_forces, &d_positions)
        .unwrap();
    let graph = stream
        .end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
        .unwrap()
        .expect("captured graph");
    graph.launch().unwrap();

    let result = pme.readback_reciprocal_result().unwrap();
    assert!(result.reciprocal_energy.is_finite());
    assert!(result.reciprocal_dudlambda.unwrap().is_finite());
    assert_eq!(settle.readback_failure_latch().unwrap(), 0);
    assert_eq!(pme.capability_receipt().graph_hot_path_dtoh_count, 0);
    assert_eq!(settle.capability_receipt().graph_hot_path_dtoh_count, 0);
    assert_eq!(
        settle.capability_receipt().water_contract_sha256,
        Some(admission.contract_sha256)
    );
}

#[test]
fn ewald_beta_reference_value() {
    let beta = prism_gpu::pme::compute_ewald_beta(12.0, 1e-5);
    assert!((beta - 0.274_111_93).abs() < 1.0e-6);
}
