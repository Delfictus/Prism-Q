//! Production CUDA acceptance for the occupied-state A/B interaction union.
//! The tests launch `nhs_lambda_dual_endpoint_hamiltonian_v1` itself.  Scalar
//! formulas below are independent acceptance oracles, not execution fallbacks.

#![cfg(feature = "gpu")]

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use prism_nhs::fused_engine::{
    GpuLambdaDualAnglePlane, GpuLambdaDualAtomPlane, GpuLambdaDualBondPlane,
    GpuLambdaDualExceptionPlane, GpuLambdaDualTorsionPlane,
};
use std::path::Path;
use std::sync::Arc;

const COULOMB: f64 = 332.063_713_299;

fn bytes<T: Copy>(values: &[T]) -> Vec<u8> {
    if values.is_empty() {
        return Vec::new();
    }
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
            .to_vec()
    }
}

fn module() -> Option<(Arc<CudaStream>, Arc<CudaModule>)> {
    let context = match CudaContext::new(0) {
        Ok(context) => context,
        Err(error) => {
            eprintln!("CUDA unavailable; skipping dual-endpoint runtime acceptance: {error}");
            return None;
        }
    };
    let stream = context.default_stream();
    let ptx_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/ptx/nhs_amber_fused.ptx")
        .canonicalize()
        .expect("canonical rebuilt fused PTX path");
    let module = context
        .load_module(Ptx::from_file(ptx_path))
        .expect("load rebuilt fused PTX");
    Some((stream, module))
}

#[derive(Clone, Default)]
struct UnionCase {
    positions: Vec<f32>,
    dual_atom_indices: Vec<i32>,
    atom_planes: Vec<GpuLambdaDualAtomPlane>,
    bonds: Vec<GpuLambdaDualBondPlane>,
    angles: Vec<GpuLambdaDualAnglePlane>,
    torsions: Vec<GpuLambdaDualTorsionPlane>,
    exceptions: Vec<GpuLambdaDualExceptionPlane>,
    exclusion_list: Vec<i32>,
    exclusion_offsets: Vec<i32>,
    pairs14_list: Vec<i32>,
    pairs14_offsets: Vec<i32>,
}

fn upload_bytes(stream: &Arc<CudaStream>, data: &[u8]) -> CudaSlice<u8> {
    let mut device = stream.alloc_zeros::<u8>(data.len().max(1)).unwrap();
    if !data.is_empty() {
        stream.memcpy_htod(data, &mut device).unwrap();
    }
    device
}

fn upload_i32(stream: &Arc<CudaStream>, data: &[i32]) -> CudaSlice<i32> {
    let mut device = stream.alloc_zeros::<i32>(data.len().max(1)).unwrap();
    if !data.is_empty() {
        stream.memcpy_htod(data, &mut device).unwrap();
    }
    device
}

fn run_union(
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
    case: &UnionCase,
    lambda: f32,
) -> (f64, f64, Vec<f32>) {
    assert_eq!(case.positions.len() % 3, 0);
    let n_atoms = (case.positions.len() / 3) as i32;
    assert_eq!(case.atom_planes.len(), n_atoms as usize);
    assert_eq!(case.exclusion_offsets.len(), n_atoms as usize + 1);
    assert_eq!(case.pairs14_offsets.len(), n_atoms as usize + 1);

    let mut d_positions = stream.clone_htod(&case.positions).unwrap();
    let mut d_forces = stream.alloc_zeros::<f32>(case.positions.len()).unwrap();
    let d_dual_indices = upload_i32(stream, &case.dual_atom_indices);
    let d_atom_planes = upload_bytes(stream, &bytes(&case.atom_planes));
    let d_bonds = upload_bytes(stream, &bytes(&case.bonds));
    let d_angles = upload_bytes(stream, &bytes(&case.angles));
    let d_torsions = upload_bytes(stream, &bytes(&case.torsions));
    let d_exceptions = upload_bytes(stream, &bytes(&case.exceptions));
    let d_exclusion_list = upload_i32(stream, &case.exclusion_list);
    let d_exclusion_offsets = upload_i32(stream, &case.exclusion_offsets);
    let d_pairs14_list = upload_i32(stream, &case.pairs14_list);
    let d_pairs14_offsets = upload_i32(stream, &case.pairs14_offsets);
    let d_slots = stream.clone_htod(&[lambda]).unwrap();
    // Occupancy mask 3 = endpoint A occupied (bit0) | endpoint B occupied
    // (bit1); this is the same "one-element no-op sentinel" value production
    // uses for an ordinary, non-lambda-fragment-backed slot (see
    // fused_engine.rs's `lambda_endpoint_occupancy_masks = vec![3i32]`
    // default). Required: the kernel's `nhs_lambda_dual_endpoint_hamiltonian_v1`
    // signature takes this buffer as its 20th parameter, between
    // slot_values and n_slots; omitting it here previously desynchronized
    // every argument from that point on in the host-side kernelParams array
    // against the compiled kernel's 31-parameter layout, corrupting the
    // launch itself (a host-side fault, not a device one -- confirmed via
    // `compute-sanitizer --tool memcheck`, which reported 0 device memory
    // errors even though the process still crashed).
    let d_masks = upload_i32(stream, &[3i32]);
    let mut d_pe = stream.alloc_zeros::<f64>(n_atoms as usize).unwrap();
    let mut d_lambda_energy = stream.alloc_zeros::<f64>(n_atoms as usize).unwrap();
    let mut d_dudl = stream.alloc_zeros::<f64>(n_atoms as usize).unwrap();
    let n_dual_atoms = case.dual_atom_indices.len() as i32;
    let n_bonds = case.bonds.len() as i32;
    let n_angles = case.angles.len() as i32;
    let n_torsions = case.torsions.len() as i32;
    let n_exceptions = case.exceptions.len() as i32;
    let n_slots = 1i32;
    let alpha_lj = 0.5f32;
    let softcore_power = 1.0f32;
    let coupling_power = 1.0f32;
    let cutoff_sq = 144.0f32;
    let box_length = 30.0f32;
    let tasks = n_dual_atoms
        .max(n_bonds)
        .max(n_angles)
        .max(n_torsions)
        .max(n_exceptions);
    assert!(tasks > 0);
    unsafe {
        stream
            .launch_builder(function)
            .arg(&mut d_positions)
            .arg(&mut d_forces)
            .arg(&n_atoms)
            .arg(&d_dual_indices)
            .arg(&n_dual_atoms)
            .arg(&d_atom_planes)
            .arg(&d_bonds)
            .arg(&n_bonds)
            .arg(&d_angles)
            .arg(&n_angles)
            .arg(&d_torsions)
            .arg(&n_torsions)
            .arg(&d_exceptions)
            .arg(&n_exceptions)
            .arg(&d_exclusion_list)
            .arg(&d_exclusion_offsets)
            .arg(&d_pairs14_list)
            .arg(&d_pairs14_offsets)
            .arg(&d_slots)
            .arg(&d_masks)
            .arg(&n_slots)
            .arg(&alpha_lj)
            .arg(&softcore_power)
            .arg(&coupling_power)
            .arg(&cutoff_sq)
            .arg(&box_length)
            .arg(&box_length)
            .arg(&box_length)
            .arg(&mut d_pe)
            .arg(&mut d_lambda_energy)
            .arg(&mut d_dudl)
            .launch(LaunchConfig {
                grid_dim: ((tasks as u32).div_ceil(32), 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            })
            .unwrap();
    }
    stream.synchronize().unwrap();
    let mut pe = vec![0.0f64; n_atoms as usize];
    let mut lambda_energy = vec![0.0f64; n_atoms as usize];
    let mut dudl = vec![0.0f64; n_atoms as usize];
    let mut forces = vec![0.0f32; case.positions.len()];
    stream.memcpy_dtoh(&d_pe, &mut pe).unwrap();
    stream
        .memcpy_dtoh(&d_lambda_energy, &mut lambda_energy)
        .unwrap();
    stream.memcpy_dtoh(&d_dudl, &mut dudl).unwrap();
    stream.memcpy_dtoh(&d_forces, &mut forces).unwrap();
    let pe_sum = pe.iter().sum::<f64>();
    let lambda_sum = lambda_energy.iter().sum::<f64>();
    assert!((pe_sum - lambda_sum).abs() < 2.0e-8);
    (pe_sum, dudl.iter().sum(), forces)
}

fn lj(sigma: f64, epsilon: f64, displacement: [f64; 3]) -> f64 {
    let r = displacement
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let sr6 = (sigma / r).powi(6);
    4.0 * epsilon * (sr6 * sr6 - sr6)
}

fn run_lj_reference(
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
    displacement: [f32; 3],
    lambda: f32,
) -> (f64, f64, [f32; 3]) {
    run_lj_reference_with_presence(stream, function, displacement, lambda, 1, 1)
}

fn run_lj_reference_with_presence(
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
    displacement: [f32; 3],
    lambda: f32,
    present_a: i32,
    present_b: i32,
) -> (f64, f64, [f32; 3]) {
    let mut potential = stream.alloc_zeros::<f64>(1).unwrap();
    let mut force = stream.alloc_zeros::<f32>(3).unwrap();
    let mut dudl = stream.alloc_zeros::<f64>(1).unwrap();
    let sigma_a = 3.1f32;
    let epsilon_a = 0.17f32;
    let sigma_b = 3.6f32;
    let epsilon_b = 0.09f32;
    let alpha = 0.5f32;
    let power = 1.0f32;
    let cutoff = 12.0f32;
    unsafe {
        stream
            .launch_builder(function)
            .arg(&displacement[0])
            .arg(&displacement[1])
            .arg(&displacement[2])
            .arg(&lambda)
            .arg(&present_a)
            .arg(&sigma_a)
            .arg(&epsilon_a)
            .arg(&present_b)
            .arg(&sigma_b)
            .arg(&epsilon_b)
            .arg(&alpha)
            .arg(&power)
            .arg(&power)
            .arg(&cutoff)
            .arg(&mut potential)
            .arg(&mut force)
            .arg(&mut dudl)
            .launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            })
            .unwrap();
    }
    stream.synchronize().unwrap();
    let mut u = [0.0f64; 1];
    let mut f = [0.0f32; 3];
    let mut du = [0.0f64; 1];
    stream.memcpy_dtoh(&potential, &mut u).unwrap();
    stream.memcpy_dtoh(&force, &mut f).unwrap();
    stream.memcpy_dtoh(&dudl, &mut du).unwrap();
    (u[0], du[0], f)
}

#[test]
fn cuda_dual_endpoint_lj_has_physical_endpoints_and_central_derivatives() {
    let Some((stream, module)) = module() else {
        return;
    };
    let function = module
        .load_function("nhs_lambda_dual_endpoint_lj_reference_v3")
        .expect("dual endpoint LJ reference kernel");
    let d = [3.77f32, -0.31, 0.42];
    let ua = lj(3.1, 0.17, d.map(f64::from));
    let ub = lj(3.6, 0.09, d.map(f64::from));
    let (gpu_a, _, _) = run_lj_reference(&stream, &function, d, 0.0);
    let (gpu_b, _, _) = run_lj_reference(&stream, &function, d, 1.0);
    assert!((gpu_a - ua).abs() < 2.0e-6, "A endpoint {gpu_a} != {ua}");
    assert!((gpu_b - ub).abs() < 2.0e-6, "B endpoint {gpu_b} != {ub}");

    let lambda = 0.37f32;
    let h_lambda = 1.0e-3f32;
    let (u, analytic_dudl, force) = run_lj_reference(&stream, &function, d, lambda);
    let up = run_lj_reference(&stream, &function, d, lambda + h_lambda).0;
    let um = run_lj_reference(&stream, &function, d, lambda - h_lambda).0;
    let fd_dudl = (up - um) / (2.0 * f64::from(h_lambda));
    assert!((analytic_dudl - fd_dudl).abs() < 2.0e-5);
    assert!((u - ((1.0 - f64::from(lambda)) * ua + f64::from(lambda) * ub)).abs() < 2.0e-6);

    let h_x = 1.0e-3f32;
    let mut dp = d;
    let mut dm = d;
    dp[0] += h_x;
    dm[0] -= h_x;
    let fd_force_x = -(run_lj_reference(&stream, &function, dp, lambda).0
        - run_lj_reference(&stream, &function, dm, lambda).0)
        / (2.0 * f64::from(h_x));
    assert!((f64::from(force[0]) - fd_force_x).abs() < 3.0e-4);
}

#[test]
fn cuda_dual_endpoint_dummy_atoms_are_endpoint_zero_and_softcore_differentiable() {
    let Some((stream, module)) = module() else {
        return;
    };
    let function = module
        .load_function("nhs_lambda_dual_endpoint_lj_reference_v3")
        .expect("dual endpoint LJ reference kernel");
    let displacement = [3.63f32, 0.27, -0.18];
    let physical_b = lj(3.6, 0.09, displacement.map(f64::from));
    let insertion_a = run_lj_reference_with_presence(&stream, &function, displacement, 0.0, 0, 1).0;
    let insertion_b = run_lj_reference_with_presence(&stream, &function, displacement, 1.0, 0, 1).0;
    assert_eq!(insertion_a.to_bits(), 0.0f64.to_bits());
    assert!((insertion_b - physical_b).abs() < 2.0e-6);

    let lambda = 0.42f32;
    let h_lambda = 1.0e-3f32;
    let (_, analytic_dudl, force) =
        run_lj_reference_with_presence(&stream, &function, displacement, lambda, 0, 1);
    let up =
        run_lj_reference_with_presence(&stream, &function, displacement, lambda + h_lambda, 0, 1).0;
    let um =
        run_lj_reference_with_presence(&stream, &function, displacement, lambda - h_lambda, 0, 1).0;
    let fd_dudl = (up - um) / (2.0 * f64::from(h_lambda));
    assert!((analytic_dudl - fd_dudl).abs() < 2.0e-4);

    let h_x = 1.0e-3f32;
    let mut plus = displacement;
    let mut minus = displacement;
    plus[0] += h_x;
    minus[0] -= h_x;
    let fd_force = -(run_lj_reference_with_presence(&stream, &function, plus, lambda, 0, 1).0
        - run_lj_reference_with_presence(&stream, &function, minus, lambda, 0, 1).0)
        / (2.0 * f64::from(h_x));
    assert!((f64::from(force[0]) - fd_force).abs() < 5.0e-4);

    let deletion_b = run_lj_reference_with_presence(&stream, &function, displacement, 1.0, 1, 0).0;
    assert_eq!(deletion_b.to_bits(), 0.0f64.to_bits());
}

#[test]
fn production_dual_lj_honors_full_exclusions_and_exact_1_4_csr_ownership() {
    let Some((stream, module)) = module() else {
        return;
    };
    let function = module
        .load_function("nhs_lambda_dual_endpoint_hamiltonian_v1")
        .expect("production dual endpoint kernel");
    let dual = GpuLambdaDualAtomPlane {
        slot: 0,
        present_a: 1,
        present_b: 1,
        reserved: 0,
        sigma_a: 3.1,
        epsilon_a: 0.17,
        sigma_b: 3.6,
        epsilon_b: 0.09,
    };
    let receptor = GpuLambdaDualAtomPlane {
        slot: -1,
        present_a: 1,
        present_b: 1,
        reserved: 0,
        sigma_a: 3.2,
        epsilon_a: 0.12,
        sigma_b: 3.2,
        epsilon_b: 0.12,
    };
    let mut case = UnionCase {
        positions: vec![2.0, 2.1, 2.2, 5.4, 2.0, 2.4],
        dual_atom_indices: vec![0],
        atom_planes: vec![dual, receptor],
        exclusion_offsets: vec![0, 0, 0],
        pairs14_offsets: vec![0, 0, 0],
        ..Default::default()
    };
    let ordinary = run_union(&stream, &function, &case, 0.37).0;
    assert!(ordinary.abs() > 1.0e-5);

    case.exclusion_list = vec![1];
    case.exclusion_offsets = vec![0, 1, 1];
    let excluded = run_union(&stream, &function, &case, 0.37).0;
    assert_eq!(excluded.to_bits(), 0.0f64.to_bits());

    case.exclusion_list.clear();
    case.exclusion_offsets = vec![0, 0, 0];
    case.pairs14_list = vec![1];
    case.pairs14_offsets = vec![0, 1, 1];
    let exact_14_owned_elsewhere = run_union(&stream, &function, &case, 0.37).0;
    assert_eq!(exact_14_owned_elsewhere.to_bits(), 0.0f64.to_bits());
}

fn vec3(positions: &[f32], atom: usize) -> [f64; 3] {
    [
        positions[3 * atom] as f64,
        positions[3 * atom + 1] as f64,
        positions[3 * atom + 2] as f64,
    ]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
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

// Abramowitz-Stegun 7.1.26; independent host oracle with maximum absolute
// error below 1.5e-7, sufficient for the CUDA exception-composition gate.
fn erf_oracle(value: f64) -> f64 {
    let sign = if value < 0.0 { -1.0 } else { 1.0 };
    let x = value.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let polynomial =
        (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t;
    sign * (1.0 - polynomial * (-x * x).exp())
}

fn bonded_endpoints(case: &UnionCase) -> (f64, f64) {
    let mut endpoints = [0.0f64; 2];
    for term in &case.bonds {
        let r = norm(sub(
            vec3(&case.positions, term.j as usize),
            vec3(&case.positions, term.i as usize),
        ));
        if term.active_a != 0 {
            endpoints[0] += f64::from(term.k_a) * (r - f64::from(term.r0_a)).powi(2);
        }
        if term.active_b != 0 {
            endpoints[1] += f64::from(term.k_b) * (r - f64::from(term.r0_b)).powi(2);
        }
    }
    for term in &case.angles {
        let rij = sub(
            vec3(&case.positions, term.i as usize),
            vec3(&case.positions, term.j as usize),
        );
        let rkj = sub(
            vec3(&case.positions, term.k as usize),
            vec3(&case.positions, term.j as usize),
        );
        let theta = (dot(rij, rkj) / (norm(rij) * norm(rkj)))
            .clamp(-1.0, 1.0)
            .acos();
        if term.active_a != 0 {
            endpoints[0] += f64::from(term.force_k_a) * (theta - f64::from(term.theta0_a)).powi(2);
        }
        if term.active_b != 0 {
            endpoints[1] += f64::from(term.force_k_b) * (theta - f64::from(term.theta0_b)).powi(2);
        }
    }
    for term in &case.torsions {
        let pi = vec3(&case.positions, term.i as usize);
        let pj = vec3(&case.positions, term.j as usize);
        let pk = vec3(&case.positions, term.k as usize);
        let pl = vec3(&case.positions, term.l as usize);
        let v0 = sub(pi, pj);
        let v1 = sub(pk, pj);
        let v2 = sub(pk, pl);
        let cp0 = cross(v0, v1);
        let cp1 = cross(v1, v2);
        let mut phi = (dot(cp0, cp1) / (norm(cp0) * norm(cp1)))
            .clamp(-1.0, 1.0)
            .acos();
        if dot(v0, cp1) < 0.0 {
            phi = -phi;
        }
        let n = f64::from(term.periodicity);
        if term.active_a != 0 {
            endpoints[0] +=
                f64::from(term.force_k_a) * (1.0 + (n * phi - f64::from(term.phase_a)).cos());
        }
        if term.active_b != 0 {
            endpoints[1] +=
                f64::from(term.force_k_b) * (1.0 + (n * phi - f64::from(term.phase_b)).cos());
        }
    }
    (endpoints[0], endpoints[1])
}

#[test]
fn production_bond_angle_torsion_union_matches_endpoints_and_finite_differences() {
    let Some((stream, module)) = module() else {
        return;
    };
    let function = module
        .load_function("nhs_lambda_dual_endpoint_hamiltonian_v1")
        .expect("production dual endpoint kernel");
    let mut case = UnionCase {
        positions: vec![1.2, 1.0, 1.1, 2.4, 1.1, 1.0, 3.1, 2.0, 1.3, 4.0, 2.3, 2.1],
        atom_planes: vec![GpuLambdaDualAtomPlane::default(); 4],
        exclusion_offsets: vec![0; 5],
        pairs14_offsets: vec![0; 5],
        bonds: vec![GpuLambdaDualBondPlane {
            i: 0,
            j: 1,
            slot: 0,
            active_a: 1,
            active_b: 1,
            reserved: 0,
            r0_a: 1.15,
            k_a: 90.0,
            r0_b: 1.28,
            k_b: 73.0,
        }],
        // The angle disappears at B. Its retained reference geometry and
        // explicit active_b=0/k_b=0 are the authenticated zero dummy.
        angles: vec![GpuLambdaDualAnglePlane {
            i: 0,
            j: 1,
            k: 2,
            slot: 0,
            active_a: 1,
            active_b: 0,
            theta0_a: 1.92,
            force_k_a: 28.0,
            theta0_b: 1.92,
            force_k_b: 0.0,
        }],
        torsions: vec![GpuLambdaDualTorsionPlane {
            i: 0,
            j: 1,
            k: 2,
            l: 3,
            slot: 0,
            periodicity: 3,
            active_a: 1,
            active_b: 1,
            improper: 0,
            reserved: 0,
            phase_a: 0.2,
            force_k_a: 0.72,
            phase_b: -0.35,
            force_k_b: 1.13,
        }],
        ..Default::default()
    };
    let (ua, ub) = bonded_endpoints(&case);
    let (gpu_a, _, _) = run_union(&stream, &function, &case, 0.0);
    let (gpu_b, _, _) = run_union(&stream, &function, &case, 1.0);
    assert!(
        (gpu_a - ua).abs() < 2.0e-4,
        "A union endpoint {gpu_a} != {ua}"
    );
    assert!(
        (gpu_b - ub).abs() < 2.0e-4,
        "B union endpoint {gpu_b} != {ub}"
    );

    let lambda = 0.43f32;
    let h_lambda = 1.0e-3f32;
    let (u, analytic_dudl, forces) = run_union(&stream, &function, &case, lambda);
    let up = run_union(&stream, &function, &case, lambda + h_lambda).0;
    let um = run_union(&stream, &function, &case, lambda - h_lambda).0;
    let fd_dudl = (up - um) / (2.0 * f64::from(h_lambda));
    assert!((analytic_dudl - fd_dudl).abs() < 2.0e-3);
    assert!((u - ((1.0 - f64::from(lambda)) * ua + f64::from(lambda) * ub)).abs() < 2.0e-4);

    let coordinate = 8usize;
    let h_x = 2.0e-3f32;
    case.positions[coordinate] += h_x;
    let plus = run_union(&stream, &function, &case, lambda).0;
    case.positions[coordinate] -= 2.0 * h_x;
    let minus = run_union(&stream, &function, &case, lambda).0;
    let fd_force = -(plus - minus) / (2.0 * f64::from(h_x));
    assert!(
        (f64::from(forces[coordinate]) - fd_force).abs() < 1.5e-2,
        "analytic force {} != central FD {fd_force}",
        forces[coordinate]
    );
}

fn ewald_zero_exception_correction(
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
    displacement: [f32; 3],
    regular_q: f32,
    d_regular_q: f32,
    beta: f32,
) -> (f64, f64, [f32; 3]) {
    let qi = regular_q;
    let qj = 1.0f32;
    let dqi = d_regular_q;
    let dqj = 0.0f32;
    let zero = 0.0f32;
    let mode = 1i32;
    let box_length = 30.0f32;
    let mut potential = stream.alloc_zeros::<f64>(1).unwrap();
    let mut force = stream.alloc_zeros::<f32>(3).unwrap();
    let mut dudl = stream.alloc_zeros::<f64>(1).unwrap();
    unsafe {
        stream
            .launch_builder(function)
            .arg(&displacement[0])
            .arg(&displacement[1])
            .arg(&displacement[2])
            .arg(&box_length)
            .arg(&box_length)
            .arg(&box_length)
            .arg(&qi)
            .arg(&dqi)
            .arg(&qj)
            .arg(&dqj)
            .arg(&zero)
            .arg(&zero)
            .arg(&beta)
            .arg(&mode)
            .arg(&mut potential)
            .arg(&mut force)
            .arg(&mut dudl)
            .launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            })
            .unwrap();
    }
    stream.synchronize().unwrap();
    let mut u = [0.0f64; 1];
    let mut f = [0.0f32; 3];
    let mut du = [0.0f64; 1];
    stream.memcpy_dtoh(&potential, &mut u).unwrap();
    stream.memcpy_dtoh(&force, &mut f).unwrap();
    stream.memcpy_dtoh(&dudl, &mut du).unwrap();
    (u[0], du[0], f)
}

#[test]
fn production_exact_exception_composes_with_pme_correction_and_central_derivatives() {
    let Some((stream, module)) = module() else {
        return;
    };
    let union_function = module
        .load_function("nhs_lambda_dual_endpoint_hamiltonian_v1")
        .expect("production dual endpoint kernel");
    let ewald_function = module
        .load_function("nhs_lambda_fragment_ewald_pair_reference_v2")
        .expect("Ewald exception correction kernel");
    let mut case = UnionCase {
        positions: vec![2.0, 2.2, 2.4, 5.4, 2.0, 2.7],
        atom_planes: vec![GpuLambdaDualAtomPlane::default(); 2],
        exclusion_offsets: vec![0; 3],
        pairs14_list: vec![1, 0],
        pairs14_offsets: vec![0, 1, 2],
        exceptions: vec![GpuLambdaDualExceptionPlane {
            i: 0,
            j: 1,
            slot: 0,
            present_a: 1,
            present_b: 1,
            reserved: 0,
            charge_product_a: -0.083,
            sigma_a: 3.0,
            epsilon_a: 0.11,
            charge_product_b: -0.031,
            sigma_b: 3.45,
            epsilon_b: 0.07,
        }],
        ..Default::default()
    };
    let lambda = 0.41f32;
    let q_regular_a = -0.19f32;
    let q_regular_b = -0.12f32;
    let beta = 0.31f32;
    let displacement = [
        case.positions[0] - case.positions[3],
        case.positions[1] - case.positions[4],
        case.positions[2] - case.positions[5],
    ];
    let regular_q = (1.0 - lambda) * q_regular_a + lambda * q_regular_b;
    let d_regular_q = q_regular_b - q_regular_a;
    let (union_u, union_dudl, union_force) = run_union(&stream, &union_function, &case, lambda);
    let (correction_u, correction_dudl, correction_force) = ewald_zero_exception_correction(
        &stream,
        &ewald_function,
        displacement,
        regular_q,
        d_regular_q,
        beta,
    );
    let total_u = union_u + correction_u;
    let total_dudl = union_dudl + correction_dudl;
    let r = norm(displacement.map(f64::from));
    let q_exception = (1.0 - f64::from(lambda)) * -0.083 + f64::from(lambda) * -0.031;
    let expected_u = (1.0 - f64::from(lambda)) * lj(3.0, 0.11, displacement.map(f64::from))
        + f64::from(lambda) * lj(3.45, 0.07, displacement.map(f64::from))
        + COULOMB * (q_exception - f64::from(regular_q) * erf_oracle(f64::from(beta) * r)) / r;
    assert!(
        (total_u - expected_u).abs() < 8.0e-6,
        "composed {total_u} != {expected_u}"
    );

    // Independent endpoint references: exact scaled-1-4 LJ/Coulomb plus the
    // reciprocal exclusion correction, with no interpolation oracle shared
    // with the CUDA implementation.
    for (endpoint, sigma, epsilon, q_exception_endpoint, q_regular_endpoint) in [
        (0.0f32, 3.0, 0.11, -0.083, q_regular_a),
        (1.0f32, 3.45, 0.07, -0.031, q_regular_b),
    ] {
        let endpoint_union = run_union(&stream, &union_function, &case, endpoint).0;
        let endpoint_correction = ewald_zero_exception_correction(
            &stream,
            &ewald_function,
            displacement,
            q_regular_endpoint,
            d_regular_q,
            beta,
        )
        .0;
        let endpoint_reference = lj(sigma, epsilon, displacement.map(f64::from))
            + COULOMB
                * (q_exception_endpoint
                    - f64::from(q_regular_endpoint) * erf_oracle(f64::from(beta) * r))
                / r;
        assert!(
            (endpoint_union + endpoint_correction - endpoint_reference).abs() < 8.0e-6,
            "scaled-1-4 endpoint {endpoint} disagrees with independent reference"
        );
    }

    let h_lambda = 1.0e-3f32;
    let composed_at = |value: f32| {
        let union = run_union(&stream, &union_function, &case, value).0;
        let regular = (1.0 - value) * q_regular_a + value * q_regular_b;
        union
            + ewald_zero_exception_correction(
                &stream,
                &ewald_function,
                displacement,
                regular,
                d_regular_q,
                beta,
            )
            .0
    };
    let fd_dudl = (composed_at(lambda + h_lambda) - composed_at(lambda - h_lambda))
        / (2.0 * f64::from(h_lambda));
    assert!((total_dudl - fd_dudl).abs() < 2.0e-3);

    let h_x = 1.0e-3f32;
    case.positions[0] += h_x;
    let plus_d = [
        case.positions[0] - case.positions[3],
        case.positions[1] - case.positions[4],
        case.positions[2] - case.positions[5],
    ];
    let plus = run_union(&stream, &union_function, &case, lambda).0
        + ewald_zero_exception_correction(
            &stream,
            &ewald_function,
            plus_d,
            regular_q,
            d_regular_q,
            beta,
        )
        .0;
    case.positions[0] -= 2.0 * h_x;
    let minus_d = [
        case.positions[0] - case.positions[3],
        case.positions[1] - case.positions[4],
        case.positions[2] - case.positions[5],
    ];
    let minus = run_union(&stream, &union_function, &case, lambda).0
        + ewald_zero_exception_correction(
            &stream,
            &ewald_function,
            minus_d,
            regular_q,
            d_regular_q,
            beta,
        )
        .0;
    let fd_force_x = -(plus - minus) / (2.0 * f64::from(h_x));
    let analytic_force_x = f64::from(union_force[0] + correction_force[0]);
    assert!((analytic_force_x - fd_force_x).abs() < 2.0e-2);
}

#[test]
fn production_zero_exception_cancels_regular_pme_for_excluded_pair() {
    let Some((stream, module)) = module() else {
        return;
    };
    let union_function = module
        .load_function("nhs_lambda_dual_endpoint_hamiltonian_v1")
        .expect("production dual endpoint kernel");
    let ewald_function = module
        .load_function("nhs_lambda_fragment_ewald_pair_reference_v2")
        .expect("Ewald exception correction kernel");
    let mut case = UnionCase {
        positions: vec![1.7, 2.3, 2.9, 5.2, 2.0, 3.1],
        atom_planes: vec![GpuLambdaDualAtomPlane::default(); 2],
        exclusion_list: vec![1, 0],
        exclusion_offsets: vec![0, 1, 2],
        // The exact-exception CSR owns both exclusions and scaled 1-4 pairs;
        // the full exclusion CSR independently proves ordinary LJ omission.
        pairs14_list: vec![1, 0],
        pairs14_offsets: vec![0, 1, 2],
        exceptions: vec![GpuLambdaDualExceptionPlane {
            i: 0,
            j: 1,
            slot: 0,
            present_a: 1,
            present_b: 1,
            reserved: 0,
            charge_product_a: 0.0,
            sigma_a: 0.0,
            epsilon_a: 0.0,
            charge_product_b: 0.0,
            sigma_b: 0.0,
            epsilon_b: 0.0,
        }],
        ..Default::default()
    };
    let q_regular_a = -0.23f32;
    let q_regular_b = -0.11f32;
    let d_regular_q = q_regular_b - q_regular_a;
    let beta = 0.29f32;
    let displacement = [
        case.positions[0] - case.positions[3],
        case.positions[1] - case.positions[4],
        case.positions[2] - case.positions[5],
    ];
    let r = norm(displacement.map(f64::from));
    let composed = |lambda: f32, positions: &[f32]| {
        let mut local = case.clone();
        local.positions.copy_from_slice(positions);
        let d = [
            positions[0] - positions[3],
            positions[1] - positions[4],
            positions[2] - positions[5],
        ];
        let regular_q = (1.0 - lambda) * q_regular_a + lambda * q_regular_b;
        let (union_u, union_dudl, union_force) =
            run_union(&stream, &union_function, &local, lambda);
        let (correction_u, correction_dudl, correction_force) = ewald_zero_exception_correction(
            &stream,
            &ewald_function,
            d,
            regular_q,
            d_regular_q,
            beta,
        );
        (
            union_u + correction_u,
            union_dudl + correction_dudl,
            f64::from(union_force[0] + correction_force[0]),
        )
    };

    for (endpoint, q_regular_endpoint) in [(0.0f32, q_regular_a), (1.0f32, q_regular_b)] {
        let actual = composed(endpoint, &case.positions).0;
        let reference =
            -COULOMB * f64::from(q_regular_endpoint) * erf_oracle(f64::from(beta) * r) / r;
        assert!((actual - reference).abs() < 8.0e-6);
    }

    let lambda = 0.38f32;
    let (energy, analytic_dudl, analytic_force_x) = composed(lambda, &case.positions);
    let regular_q = (1.0 - lambda) * q_regular_a + lambda * q_regular_b;
    let reference = -COULOMB * f64::from(regular_q) * erf_oracle(f64::from(beta) * r) / r;
    assert!((energy - reference).abs() < 8.0e-6);
    let h_lambda = 1.0e-3f32;
    let fd_dudl = (composed(lambda + h_lambda, &case.positions).0
        - composed(lambda - h_lambda, &case.positions).0)
        / (2.0 * f64::from(h_lambda));
    assert!((analytic_dudl - fd_dudl).abs() < 2.0e-3);

    let h_x = 1.0e-3f32;
    let mut plus_positions = case.positions.clone();
    plus_positions[0] += h_x;
    let plus = composed(lambda, &plus_positions).0;
    let mut minus_positions = case.positions.clone();
    minus_positions[0] -= h_x;
    let minus = composed(lambda, &minus_positions).0;
    let fd_force_x = -(plus - minus) / (2.0 * f64::from(h_x));
    assert!((analytic_force_x - fd_force_x).abs() < 2.0e-2);
}
