//! Real topology, production fused-force/integrator ASC qualification.
//! Run explicitly with PRISM_ASC_TOPOLOGY, PRISM_ASC_EVIDENCE_DIR and pinned
//! PRISM4D_PTX_DIR. No CUDA skip: missing hardware/input fails this opt-in test.
//! Adjudication is a declared deterministic boundary fixture. This qualifies
//! library/device integration, not the CLI, observer decision, or campaign.
#![cfg(feature = "gpu")]
use cudarc::driver::{sys, CudaContext, DevicePtr};
use prism_nhs::{
    asc_device::AscDevice,
    asc_support::{asc_atom_order_sha256, AscSelectedSupportPolicy},
    fused_engine::{CryoUvProtocol, NhsAmberFusedEngine},
    input::PrismPrepTopology,
    interferometric_adjudicator::InterferometricAdjudicatorFfi,
};
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};
const TOPOLOGY_SHA256: &str = "166935fb39c69afb2575f3a6301dbf5523c292ea88def8c12758d417d7dc581d";
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn check(rc: sys::CUresult) {
    assert_eq!(rc, sys::CUresult::CUDA_SUCCESS);
}

#[test]
#[ignore = "explicit real GPU and hash-pinned MYC:MAX input required"]
fn selected_support_reaches_real_fused_kicks_and_graph_replay() {
    let input = PathBuf::from(std::env::var("PRISM_ASC_TOPOLOGY").expect("PRISM_ASC_TOPOLOGY"));
    let out =
        PathBuf::from(std::env::var("PRISM_ASC_EVIDENCE_DIR").expect("PRISM_ASC_EVIDENCE_DIR"));
    fs::create_dir_all(&out).unwrap();
    let bytes = fs::read(&input).unwrap();
    assert_eq!(hash(&bytes), TOPOLOGY_SHA256, "fixture identity changed");
    let topology = PrismPrepTopology::load(&input).unwrap();
    assert_eq!(topology.n_atoms, 2790);
    assert_eq!(topology.nonbonded_exceptions.len(), 7331);
    // Exact first physical residue from the real topology; no pocket/causal claim.
    let residue = topology.residue_ids[0];
    let policy_bytes = serde_json::to_vec_pretty(&serde_json::json!({
        "schema_kind":"prism_asc_selected_support_policy", "schema_version":"1.0.0",
        "topology_sha256":TOPOLOGY_SHA256, "atom_order_sha256":asc_atom_order_sha256(&topology).unwrap(),
        "source_kind":"REAL_TOPOLOGY_STRUCTURAL_RESIDUE_INTEGRATION_FIXTURE",
        "source_identity":"6G6K human MYC:MAX first physical residue",
        "source_evidence_sha256":TOPOLOGY_SHA256,
        "support_basis":"EXPLICIT_STRUCTURAL_ATOM_RESIDUE_MEMBERSHIP", "atoms":[],
        "residues":[{"physical_residue_id":residue,"residue_name":topology.residue_names[0],"chain_id":topology.chain_ids[0]}]
    })).unwrap();
    fs::write(out.join("selected_support_policy.json"), &policy_bytes).unwrap();
    let policy = AscSelectedSupportPolicy::from_bytes(&policy_bytes, &hash(&policy_bytes)).unwrap();
    let support = policy
        .validate_and_expand(&topology, TOPOLOGY_SHA256)
        .unwrap();
    assert!(support.atom_indices().len() > 1 && support.atom_indices().len() < topology.n_atoms);
    let mut bad = policy.clone();
    bad.topology_sha256 = "0".repeat(64);
    assert!(bad.validate_and_expand(&topology, TOPOLOGY_SHA256).is_err());
    let context = CudaContext::new(0).expect("GPU is required; cannot skip");
    let stream = context.new_stream().unwrap();
    let mut engine =
        NhsAmberFusedEngine::new_on_stream(context, stream.clone(), &topology, 32, 2.0).unwrap();
    engine.set_fused_inner_steps(1);
    engine.set_dt(0.0005);
    engine.set_adaptive_dt(false);
    // Production initializes the GPU-resident dt/thermal state via this API.
    let mut protocol = CryoUvProtocol::twin_equilibrium_observer();
    protocol.uv_burst_energy = 0.0;
    engine.set_cryo_uv_protocol(protocol).unwrap();
    engine.set_uv_enabled(false);
    let owner = AscDevice::new_selected_support(
        stream.clone(),
        0,
        &topology,
        TOPOLOGY_SHA256,
        &policy,
        0.05,
    )
    .unwrap();
    engine.set_asc_device(owner.clone()).unwrap();
    assert!(
        engine.set_asc_device(owner.clone()).is_err(),
        "replacement must fail"
    );
    let mut journal=prism_nhs::asc_device::measurement::AscMeasurementJournal::create(
        &out.join("isolated_asc_journal"),"real_topology_fixture",&owner).unwrap();
    let pre = engine.get_positions().unwrap();
    engine.step().expect("real unsteered initialization step");
    let initial = owner.snapshot().unwrap();
    assert_eq!(initial.completed_epoch, 1);
    for phase in &initial.phases {
        assert_eq!(phase.composed_atoms, support.atom_indices().len() as u64);
        assert_eq!(phase.consumed_atoms, support.atom_indices().len() as u64);
        assert_eq!(phase.nonzero_force_atoms, 0);
        assert_eq!(phase.nonzero_impulse_atoms, 0);
    }
    let initial_receipt = owner.receipt().unwrap();
    journal.append_endpoint(&owner,1,0).unwrap();
    // Test boundary input only. Native publish kernel computes the topology-bound
    // control and selected centroid; the fixture never touches forces or kicks.
    let mut adj_bytes = vec![0u8; std::mem::size_of::<InterferometricAdjudicatorFfi>()];
    let divergence = std::mem::offset_of!(InterferometricAdjudicatorFfi, current_divergence);
    let code = std::mem::offset_of!(InterferometricAdjudicatorFfi, adjudication_code);
    adj_bytes[divergence..divergence + 4].copy_from_slice(&1.0f32.to_ne_bytes());
    adj_bytes[code..code + 4].copy_from_slice(&1u32.to_ne_bytes());
    let mut adj = stream.alloc_zeros::<u8>(adj_bytes.len()).unwrap();
    stream.memcpy_htod(&adj_bytes, &mut adj).unwrap();
    let (adj_ptr, _guard) = adj.device_ptr(&stream);
    unsafe { owner.publish(&stream, adj_ptr as *const InterferometricAdjudicatorFfi) }.unwrap();
    engine.step().expect("real steered physical step");
    let mut receipts = vec![owner.receipt().unwrap()];
    // Capture the actual autonomous physical step; replay holds the previous
    // completed decision. No test kernels are introduced into this graph.
    let mut graph = std::ptr::null_mut();
    let mut exec = std::ptr::null_mut();
    let mut graph_nodes = 0usize;
    unsafe {
        cudarc::driver::set_capture_mode_active(true);
        check(sys::cuStreamBeginCapture_v2(
            stream.cu_stream(),
            sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        ));
        engine
            .step_autonomous_kernels(&stream)
            .expect("capture actual fused engine body");
        check(sys::cuStreamEndCapture(stream.cu_stream(), &mut graph));
        cudarc::driver::set_capture_mode_active(false);
        check(sys::cuGraphGetNodes(
            graph,
            std::ptr::null_mut(),
            &mut graph_nodes,
        ));
        assert!(
            graph_nodes > 7,
            "real physical graph must contain phase and supporting kernels"
        );
        check(sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0));
        for expected_epoch in 3..=5 {
            check(sys::cuGraphLaunch(exec, stream.cu_stream()));
            stream.synchronize().unwrap();
            let s = owner.snapshot().unwrap();
            assert_eq!(s.completed_epoch, expected_epoch);
            assert_eq!(s.active.producer_epoch, 1);
            receipts.push(owner.receipt().unwrap());
        }
        check(sys::cuGraphExecDestroy(exec));
        check(sys::cuGraphDestroy(graph));
    }
    for receipt in &receipts {
        assert_eq!(receipt["coverage_failures"], 0);
        assert_eq!(receipt["status_code"], 1);
        for phase in receipt["phases"].as_array().unwrap() {
            assert_eq!(phase["composed_atoms"], support.atom_indices().len());
            assert_eq!(phase["consumed_atoms"], support.atom_indices().len());
            assert_eq!(phase["nonzero_force_atoms"], support.atom_indices().len());
            assert!(phase["nonzero_impulse_atoms"].as_u64().unwrap() > 0);
        }
    }
    // Rows from steps2..4 are deliberately not persisted. Complete cumulative
    // work must nevertheless include the nonzero decision change at step2.
    journal.append_endpoint(&owner,5,1).unwrap();
    assert!(journal.append_endpoint(&owner,5,2).is_err());
    journal.finish(5).unwrap();
    let verified=prism_nhs::asc_device::measurement::verify_sealed_journal(
        &out.join("isolated_asc_journal"),"real_topology_fixture",0).unwrap();
    assert_eq!(verified["seal"]["completed_physical_steps"],5);
    assert!(prism_nhs::asc_device::measurement::verify_sealed_journal(
        &out.join("isolated_asc_journal"),"other_run",0).is_err());
    let tampered=out.join("tampered_journal_negative");fs::create_dir(&tampered).unwrap();
    for name in ["schema.json","seal.json","atom_rows.bin","endpoints.jsonl"] {
        fs::copy(out.join("isolated_asc_journal").join(name),tampered.join(name)).unwrap();
    }
    let mut bad_rows=fs::read(tampered.join("atom_rows.bin")).unwrap();bad_rows.pop();
    fs::write(tampered.join("atom_rows.bin"),bad_rows).unwrap();
    assert!(prism_nhs::asc_device::measurement::verify_sealed_journal(
        &tampered,"real_topology_fixture",0).is_err());
    let records:Vec<serde_json::Value>=fs::read_to_string(out.join("isolated_asc_journal/endpoints.jsonl"))
        .unwrap().lines().map(|line|serde_json::from_str(line).unwrap()).collect();
    assert_eq!(records.len(),2);
    assert_eq!(records[1]["unobserved_intermediate_atom_row_step_count"],3);
    let expected_work:f64=receipts.iter().map(|r|r["control_change_work_kcal_mol"].as_f64().unwrap()).sum();
    assert!(expected_work.abs()>0.0);
    assert_eq!(records[1]["interval_control_work_kcal_mol"].as_f64().unwrap().to_bits(),expected_work.to_bits());
    assert_eq!(records[1]["last_step_control_work_kcal_mol"].as_f64().unwrap(),0.0);
    let persisted=fs::read(out.join("isolated_asc_journal/atom_rows.bin")).unwrap();
    assert_eq!(persisted.len(),2*topology.n_atoms*320);
    assert_eq!(records[1]["row_sha256"],hash(&persisted[topology.n_atoms*320..]));
    let s = owner.snapshot().unwrap();
    assert_eq!(s.composed_steps, 5);
    assert_eq!(s.consumed_steps, 5);
    let mut rows = vec![0u8; topology.n_atoms * 320];
    unsafe {
        check(sys::cuMemcpyDtoH_v2(
            rows.as_mut_ptr().cast(),
            s.rows,
            rows.len(),
        ));
    }
    // ABI row offsets independently checked against prism_asc_decision.cuh:
    // impulse=112, force=160, velocity_delta=256, differential_energy=304.
    let read_f64 = |atom: usize, offset: usize| {
        f64::from_ne_bytes(
            rows[atom * 320 + offset..atom * 320 + offset + 8]
                .try_into()
                .unwrap(),
        )
    };
    let mut selected_kicks = [0usize; 2];
    for atom in 0..topology.n_atoms {
        for event in 0..4 {
            let offset = atom * 320 + event * 8;
            assert_eq!(
                u64::from_ne_bytes(rows[offset..offset + 8].try_into().unwrap()),
                5
            );
        }
        for phase in 0..2 {
            let impulse: Vec<f64> = (0..3)
                .map(|j| read_f64(atom, 112 + phase * 24 + j * 8))
                .collect();
            let force: Vec<f64> = (0..3)
                .map(|j| read_f64(atom, 160 + phase * 24 + j * 8))
                .collect();
            assert!(impulse.iter().chain(force.iter()).all(|v| v.is_finite()));
            if support.mask()[atom] == 0 {
                assert!(
                    impulse.iter().chain(force.iter()).all(|v| *v == 0.0),
                    "unselected ASC force/kick"
                );
            } else if impulse.iter().any(|v| *v != 0.0) {
                selected_kicks[phase] += 1;
            }
        }
    }
    assert!(selected_kicks.iter().all(|n| *n > 0));
    let post = engine.get_positions().unwrap();
    let velocities = engine.get_velocities().unwrap();
    assert!(post.iter().chain(velocities.iter()).all(|v| v.is_finite()));
    assert!(pre.iter().zip(post.iter()).any(|(a, b)| a != b));
    fs::write(out.join("asc_atom_rows.bin"), &rows).unwrap();
    let positions_bytes: Vec<u8> = post.iter().flat_map(|v| v.to_le_bytes()).collect();
    let velocity_bytes: Vec<u8> = velocities.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(out.join("positions_after.f32le"), &positions_bytes).unwrap();
    fs::write(out.join("velocities_after.f32le"), &velocity_bytes).unwrap();
    // Include actual allocation retirement after graph destruction in acceptance.
    // Deferred CudaSlice event failures must not be hidden by a passing test body.
    drop(engine);
    stream.synchronize().unwrap();
    stream.context().check_err().unwrap();
    fs::write(out.join("runtime_receipt.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "status":"IMPLEMENTATION_VERIFIED", "scope":"library/device integration only",
        "topology_sha256":TOPOLOGY_SHA256,"atoms":topology.n_atoms,"exact_nonbonded_exceptions":7331,
        "selected_atom_indices":support.atom_indices(),"selected_kick_atoms":selected_kicks,
        "physical_steps":5,"host_steps":2,"graph_replays":3,"both_phases_checked":true,
        "decision_boundary":"deterministic construct divergence=1.0; adjudicator producer not qualified",
        "row_stride":320,"rows_sha256":hash(&rows),"receipts":receipts,
        "initial_zero_control_receipt":initial_receipt,"captured_graph_nodes":graph_nodes,
        "positions_sha256":hash(&positions_bytes),"velocities_sha256":hash(&velocity_bytes)
    })).unwrap()).unwrap();
    eprintln!("ASC_FUSED_REAL_TOPOLOGY: 2790 atoms, {} selected, both real kicks, 2 host steps + 3 graph replays verified",support.atom_indices().len());
}
