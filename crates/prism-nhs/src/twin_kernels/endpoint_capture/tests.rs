//! Actual native physical-owner integration. Fixed descriptor memberships are
//! regression inputs; every physical spike population is produced by native MD.
use super::*;
use crate::captured_pipeline::{measured_kcc_causal_leads, PipelineConfig};
use crate::persistent_engine::{PersistentBatchConfig, PersistentNhsEngine};
use crate::rescue_controller::CoupledWindowBarrier;
use crate::rich_spike::RichSpike;
use sha2::{Digest, Sha256};

#[test]
#[ignore = "requires CUDA conditional graphs, native PTX and pinned full topology"]
fn native_four_physical_owners_shared_parent() {
    run_native_four_physical_owners_shared_parent(true,false,false);
}

#[test]
#[ignore = "legacy destructive-retirement diagnostic; expected sanitizer failure without retained-root control"]
fn native_four_physical_owners_legacy_retirement_diagnostic() {
    run_native_four_physical_owners_shared_parent(false,false,false);
}

#[test]
#[ignore = "actual GPU cancellation handoff; all native readbacks complete before retirement error admission"]
fn native_four_physical_owners_cancelled_handoff() {
    run_native_four_physical_owners_shared_parent(true,true,false);
}

#[test]
#[ignore = "actual four distinct seeded native physical owners plus recursive captured neighbor arguments"]
fn native_four_distinct_physical_owners_neighbor_arguments() {
    run_native_four_physical_owners_shared_parent(true,false,true);
}

fn run_native_four_physical_owners_shared_parent(production_lifetime:bool,cancelled_handoff:bool,distinct_seeds:bool) {
    let inspect_neighbor_arguments=distinct_seeds || std::env::var("PRISM_SHARED_PHYSICAL_TEST_NEIGHBOR_ARGUMENTS").as_deref()==Ok("1");

    let full_csr_owner_readback=std::env::var("PRISM_SHARED_PHYSICAL_TEST_NO_CSR_READBACK").as_deref()!=Ok("1");
    let context_publication=std::env::var("PRISM_SHARED_PHYSICAL_TEST_CONTEXT_PUBLICATION").as_deref()==Ok("1");
    let retain_local_roots=!production_lifetime && std::env::var("PRISM_SHARED_PHYSICAL_TEST_RETAIN_LOCAL_ROOTS").as_deref()==Ok("1");
    eprintln!("SHARED_PHYSICAL_DIAGNOSTIC {}",serde_json::json!({"serial_physical_branches":std::env::var("PRISM_SHARED_PHYSICAL_TEST_SERIAL_BRANCHES").as_deref()==Ok("1"),"full_csr_owner_readback":full_csr_owner_readback,"one_time_context_publication":context_publication,"retain_retired_local_roots_until_complete":retain_local_roots,"production_bounded_lifetime":production_lifetime,"cancelled_handoff_fixture":cancelled_handoff,"physical_steps_per_owner":70,"owners":4}));
    let path = std::env::var("PRISM_KCC_TEST_TOPOLOGY").unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(std::fs::read(&path).unwrap())),
        "861be5f35178fa5b5f15022bda7f6b6ffa9febf679d4ff1932669a4accf0b4d3"
    );
    let topology = Arc::new(crate::input::PrismPrepTopology::load(&path).unwrap());
    let config = PersistentBatchConfig {
        max_atoms: topology.n_atoms,
        grid_dim: 32,
        grid_spacing: 2.0,
        ..Default::default()
    };
    let module_owner = PersistentNhsEngine::new(&config).unwrap();
    let context = module_owner.cuda_context().clone();
    let module = module_owner.cuda_module().clone();
    let ring_module = context
        .load_module(cudarc::nvrtc::Ptx::from_file(
            std::env::var("PRISM_RING_TEST_PTX")
                .unwrap_or_else(|_| find_twin_ptx("ring_buffer.ptx").unwrap()),
        ))
        .unwrap();
    let rings: Vec<_> = (0..4usize)
        .map(|group| {
            let stream = context.new_stream().unwrap();
            let recipients: Vec<_> = (0..4).filter(|id| *id != group).collect();
            let ring = TwinRingBuffer::new_lossless_broadcast(
                &context,
                &stream,
                &ring_module,
                32,
                4,
                &recipients,
            )
            .unwrap();
            stream.synchronize().unwrap();
            Arc::new(Mutex::new(ring))
        })
        .collect();
    let coordinator = CapturedEndpointCoordinator::new(&context, rings.clone(), 4).unwrap();
    let boundary = CoupledWindowBarrier::new(4);
    let setup = Arc::new(Mutex::new(()));
    let populations = Arc::new(Mutex::new(vec![vec![0usize; 7]; 4]));
    let results = std::thread::scope(|scope| {
        let handles:Vec<_> = (0..4usize).map(|id| {
            let context=context.clone();let module=module.clone();let topology=topology.clone();
            let coordinator=coordinator.clone();let boundary=boundary.clone();let rings=rings.clone();
            let setup=setup.clone();let populations=populations.clone();let config=&config;
            scope.spawn(move || {
                let _participant=boundary.participant(id);
                context.bind_to_thread().unwrap();
                // Serialize fallible construction, not physical execution. Every
                // graph still owns its actual worker stream and allocations.
                let setup_guard=setup.lock().unwrap();
                let stream=context.new_stream().unwrap();
                let native=PersistentNhsEngine::new_on_stream(config,context.clone(),module,stream.clone()).unwrap();
                let mut engine=CudaOwned::new(native,stream.clone());
                coordinator.bind_owner(&mut engine);
                engine.load_topology(&topology).unwrap();
                let mut protocol=crate::fused_engine::CryoUvProtocol::standard();
                protocol.cold_hold_steps=30;protocol.ramp_steps=10;protocol.warm_hold_steps=10;
                protocol.ramp_down_steps=10;protocol.cold_return_steps=10;
                engine.set_cryo_uv_protocol(protocol).unwrap();
                let seed_evidence=if distinct_seeds {
                    let base=engine.get_positions().unwrap();
                    let seed=[42001u64,42003,42007,42011][id];
                    engine.reset_for_replica(seed).unwrap();
                    let after=engine.get_positions().unwrap();
                    let velocities=engine.fused_engine().unwrap().get_velocities().unwrap();
                    let hash=|values:&[f32]| {let mut h=Sha256::new();for value in values {h.update(value.to_le_bytes());}format!("{:x}",h.finalize())};
                    serde_json::json!({"seed":seed,"initializer":"native_reset_for_replica",
                        "base_position_sha256":hash(&base),"post_reset_position_sha256":hash(&after),
                        "post_reset_velocity_sha256":hash(&velocities),
                        "existing_initializer_scope":"seeded_MB_velocities_GPU_RNG_and_thermal_position_jitter"})
                } else {serde_json::Value::Null};
                if inspect_neighbor_arguments {engine.fused_engine_mut().unwrap().enable_neighbor_capture_diagnostic();}
                engine.set_ladd_enabled(true);engine.set_spike_accumulation(true);
                assert!(engine.set_spike_stream_id(id as u32));
                let initial=engine.get_positions().unwrap();
                let spikes:Vec<RichSpike>=(0..16usize).map(|atom| {
                    let mut s=RichSpike::zero();s.x=initial[atom*3];s.y=initial[atom*3+1];s.z=initial[atom*3+2];
                    s.cluster_id=(atom/8) as i32;s.residue_id=atom as i32;s
                }).collect();
                let bytes=unsafe {std::slice::from_raw_parts(spikes.as_ptr().cast::<u8>(),spikes.len()*std::mem::size_of::<RichSpike>())};
                let mut storage=CudaOwned::new(stream.clone_htod(bytes).unwrap(),stream.clone());
                let mut offsets=CudaOwned::new(stream.clone_htod(&[0u32,8,16]).unwrap(),stream.clone());
                let candidates=vec![(0..16u32).collect(),Vec::new()];
                let mut csr=CudaOwned::new(engine.prepare_kcc_causal_candidates(&candidates).unwrap(),stream.clone());
                let mut leads=CudaOwned::new(stream.clone_htod(&[u32::MAX;2]).unwrap(),stream.clone());
                let mut work=CudaOwned::new(stream.clone_htod(&[0f64]).unwrap(),stream.clone());
                let mut ghost=CudaOwned::new(crate::ghost_tile::GhostTileRing::allocate(14).unwrap(),stream.clone());
                coordinator.bind_owner(&mut storage);coordinator.bind_owner(&mut offsets);
                coordinator.bind_owner(&mut csr);coordinator.bind_owner(&mut leads);
                coordinator.bind_owner(&mut work);coordinator.bind_owner(&mut ghost);
                assert_eq!(unsafe{crate::sh_basis::ffi::prism_sh_basis_init(stream.cu_stream().cast())},0);
                let cfg=PipelineConfig {
            seed_layout: crate::rich_spike::NativeV2SeedLayout::default(),
                    d_spikes:storage.device_ptr(&stream).0 as *const RichSpike,
                    d_cluster_offsets:offsets.device_ptr(&stream).0 as *const u32,
                    n_clusters:2,d_k_lm:crate::sh_basis::k_lm_device_ptr().unwrap(),
                    initial_frame_id:0,source_stream_id:id as u32,diagnostic_stream_id:Some(id as u32),
                    asc:None,d_force_norm_mirror:0,governor_device:None,zstr:None,sisr:None,
                    noise_floor_override:None,mar_v2:None,d_dt:std::ptr::null_mut(),d_velocities:std::ptr::null_mut(),
                    g26_parent_cond_handle:0,f1_parent_cond_handle:0,
                    d_pe_components:engine.d_potential_energy_components_dev_ptr() as *const f64,
                    n_atoms_for_pe:topology.n_atoms as u32,d_external_work:work.device_ptr(&stream).0 as *mut f64,
                    ghost_tile_ring_dev:ghost.device_base,ghost_tile_max_records:ghost.max_records,
                    firehose_enable:1,body_unroll:1,d_kcc_lead:leads.device_ptr(&stream).0,
                    d_forces_anchor:engine.d_forces_dev_ptr(),d_masses:engine.d_masses_dev_ptr(),
                    force_burst_step:None,n_spikes:16,
                    d_current_md_step:engine.d_protocol_current_step_dev_ptr() as *const u32,branch_trace_dev:0,
                };
                let mut local=Some(engine.capture_resident_observation_graph(10,&cfg,&csr,&mut leads).unwrap());
                assert_eq!(local.as_ref().unwrap().captured_ladd_owner_keys_for_test()["parity_keys"][0][29],30,
                    "four-owner LADD boundary differs from native physical protocol");
                eprintln!("SHARED_LADD_HOST_OWNER {}",serde_json::json!({"stream_id":id,
                    "context":context.cu_ctx() as usize,"raw_owner_stream":stream.cu_stream() as usize,
                    "captured":local.as_ref().unwrap().captured_ladd_owner_keys_for_test()}));
                // Diagnostic ownership control only. These roots are never
                // replayed after replacement; retain until the existing final
                // all-worker completion barrier, then destroy normally.
                let mut retired_local_roots=Vec::new();
                let mut neighbor_argument_checks=Vec::new();
                if inspect_neighbor_arguments {
                    let actual=local.as_ref().unwrap().neighbor_arguments_for_test().unwrap();
                    let owner=engine.fused_engine().unwrap().neighbor_capture_owner_for_test().unwrap();
                    eprintln!("NEIGHBOR_CAPTURE_OWNER {}",serde_json::json!({"stream_id":id,"boundary":"initial_local","actual":actual,"owner":owner}));
                    crate::graph_capture::verify_neighbor_owner_for_test(&actual,&owner).unwrap();
                    assert_eq!(actual["kernels"].as_array().unwrap().len() as u64,owner["native_neighbor_enqueues_in_capture"].as_u64().unwrap());
                    neighbor_argument_checks.push(serde_json::json!({"boundary":"initial_local","actual":actual,"owner":owner}));
                }
                let live=local.as_ref().unwrap().pipeline().clone();
                let reference=live.frozen_baseline_readback().unwrap();
                assert_eq!(engine.device_protocol_current_step().unwrap(),0);
                assert_eq!(engine.get_positions().unwrap(),initial);
                assert_eq!(ghost.n_frames_written(),0);
                let publication=TwinSpikePublication::new(&stream,engine.spike_buffer_gpu().unwrap(),engine.spike_count_gpu().unwrap()).unwrap();
                let mut publication=CudaOwned::new(Some(publication),stream.clone());
                coordinator.bind_owner(&mut publication);
                coordinator.prepare_worker(id,&mut engine,&mut publication).unwrap();
                drop(setup_guard);
                boundary.wait(||false).unwrap();
                if context_publication {
                    // Diagnostic only: all four captures are complete before a
                    // single context fence. Physical branches remain parallel.
                    if id==0 {context.synchronize().unwrap();}
                    boundary.wait(||false).unwrap();
                }
                let mut handoffs=Vec::new();let mut window_receipts=Vec::new();let mut owner_checks=Vec::new();
                for endpoint in 1..=7u32 {
                    let active=coordinator.shared_resident_active().unwrap();
                    assert_eq!(active,endpoint>=5);
                    if endpoint==5 {
                        let before=live.rebind_state_sha256().unwrap();
                        let clock=ghost.n_frames_written();let payload=ghost.payload_bytes().to_vec();
                        let shared=coordinator.shared_resident_pipeline(id).unwrap();
                        assert!(Arc::ptr_eq(&live,&shared));
                        if inspect_neighbor_arguments {
                            let actual=coordinator.neighbor_arguments_for_test().unwrap();
                            let owner=engine.fused_engine().unwrap().neighbor_capture_owner_for_test().unwrap();
                            eprintln!("NEIGHBOR_CAPTURE_OWNER {}",serde_json::json!({"stream_id":id,"boundary":"shared_activation","actual":actual,"owner":owner}));
                            crate::graph_capture::verify_neighbor_owner_for_test(&actual,&owner).unwrap();
                            assert_eq!(actual["kernels"].as_array().unwrap().len() as u64,4*owner["native_neighbor_enqueues_in_capture"].as_u64().unwrap());
                            let pointers:std::collections::BTreeSet<_>=actual["kernels"].as_array().unwrap().iter()
                                .map(|k|k["arguments_u64_or_u32_bits"][0].as_u64().unwrap()).collect();
                            assert_eq!(pointers.len(),4,"shared physical owners alias positions");
                            neighbor_argument_checks.push(serde_json::json!({"boundary":"shared_activation","actual":actual,"owner":owner}));
                        }

                        assert_eq!(live.rebind_state_sha256().unwrap(),before);
                        assert_eq!(ghost.n_frames_written(),clock);assert_eq!(ghost.payload_bytes(),payload);
                        owner_checks.push(serde_json::json!({"boundary":"shared_activation","step":40,"state":before}));
                        if full_csr_owner_readback {
                        let csr_before=engine.fused_engine().unwrap().shared_exception_owner_diagnostic().unwrap();
                        eprintln!("SHARED_CSR_OWNER {}",serde_json::json!({"stream_id":id,"boundary":"before_local_root_retirement","owner":csr_before}));
                        if production_lifetime {
                            if !cancelled_handoff {
                                coordinator.retain_local_observation_root(id,
                                    LocalObservationRetirementBoundary::SharedActivation,&mut local).unwrap();
                            }
                        } else if retain_local_roots {retired_local_roots.push(local.take().unwrap());} else {drop(local.take());}
                        let csr_after=engine.fused_engine().unwrap().shared_exception_owner_diagnostic().unwrap();
                        assert_eq!(csr_before,csr_after,"local graph retirement changed exact CSR owner/content");
                        eprintln!("SHARED_CSR_OWNER {}",serde_json::json!({"stream_id":id,"boundary":"after_local_root_retirement","owner":csr_after}));
                        } else {
                            // Diagnostic control: preserve retirement/barrier without
                            // injecting any CSR readback or buffer tracking operation.
                            if production_lifetime {
                            if !cancelled_handoff {
                                coordinator.retain_local_observation_root(id,
                                    LocalObservationRetirementBoundary::SharedActivation,&mut local).unwrap();
                            }
                        } else if retain_local_roots {retired_local_roots.push(local.take().unwrap());} else {drop(local.take());}
                        }
                        boundary.wait(||false).unwrap();
                    }
                    let receipt=if active {
                        coordinator.execute_shared_resident_window(id,&mut engine,"native_four_owner_shared_test",||boundary.is_aborted()).unwrap()
                    } else {
                        engine.execute_resident_observation_window(local.as_mut().unwrap(),10,"native_four_owner_local_test").unwrap()
                    };
                    assert_eq!(receipt.start_device_step,(endpoint-1)*10);
                    assert_eq!(receipt.end_device_step,endpoint*10);
                    assert_eq!(receipt.graph_windows[0].graph_launches,1);
                    if endpoint>=6 {
                        let row=coordinator.verified_handoff_for_worker(id,(endpoint-1)*10).unwrap();
                        let values=row["native_u32_values"].as_array().unwrap();
                        let previous=populations.lock().unwrap()[id][endpoint as usize-2];
                        assert_eq!(values[0],(endpoint-1)*10);assert_eq!(values[1],(endpoint-1)*10);
                        for field in 2..=4 {assert_eq!(values[field],previous as u64);}
                        assert_eq!(values[5],0);assert_eq!(row["source_reset_before_next_physics"],true);
                        handoffs.push(row);
                    }
                    let kcc=engine.read_composed_kcc_endpoint(endpoint*10,true).unwrap();
                    let expected=measured_kcc_causal_leads(&candidates,&kcc,u64::from(endpoint*10)).unwrap()
                        .iter().map(|row|row.causal_lead_residue.unwrap_or(u32::MAX)).collect::<Vec<_>>();
                    assert_eq!(stream.clone_dtoh(&*leads).unwrap(),expected);
                    assert_eq!(ghost.n_frames_written(),endpoint*2);
                    assert!(engine.get_positions().unwrap().iter().all(|v|v.is_finite()));
                    let archived=engine.force_spike_sync_preserving_source().unwrap();
                    populations.lock().unwrap()[id][endpoint as usize-1]=archived;
                    window_receipts.push(serde_json::json!({"start_step":receipt.start_device_step,
                        "end_step":receipt.end_device_step,"native_archived_count":archived,"shared_parent":active}));
                    // All real producers completed before setup/coupling captures.
                    boundary.wait(||false).unwrap();
                    coordinator.prepare_worker(id,&mut engine,&mut publication).unwrap();
                    if endpoint==2 {
                        let _capture=setup.lock().unwrap();
                        let before=live.rebind_state_sha256().unwrap();
                        let payload=ghost.payload_bytes().to_vec();
                        let replacement=engine.capture_resident_observation_graph_with_live_v2(10,&cfg,&csr,&mut leads,live.clone()).unwrap();
                        if inspect_neighbor_arguments {
                            let actual=replacement.neighbor_arguments_for_test().unwrap();
                            let owner=engine.fused_engine().unwrap().neighbor_capture_owner_for_test().unwrap();
                            eprintln!("NEIGHBOR_CAPTURE_OWNER {}",serde_json::json!({"stream_id":id,"boundary":"local_recapture20","actual":actual,"owner":owner}));
                            crate::graph_capture::verify_neighbor_owner_for_test(&actual,&owner).unwrap();
                            assert_eq!(actual["kernels"].as_array().unwrap().len() as u64,owner["native_neighbor_enqueues_in_capture"].as_u64().unwrap());
                            neighbor_argument_checks.push(serde_json::json!({"boundary":"local_recapture20","actual":actual,"owner":owner}));
                        }

                        assert_eq!(replacement.captured_ladd_owner_keys_for_test()["parity_keys"][0][29],30);
                        assert!(Arc::ptr_eq(replacement.pipeline(),&live));
                        assert_eq!(live.rebind_state_sha256().unwrap(),before);
                        assert_eq!(ghost.payload_bytes(),payload);
                        owner_checks.push(serde_json::json!({"boundary":"local_recapture","step":20,"state":before}));
                        if production_lifetime {
                            // Failed host-only admission must keep the caller's root.
                            assert!(coordinator.retain_local_observation_root(id,
                                LocalObservationRetirementBoundary::SharedActivation,&mut local).is_err());
                            assert!(local.is_some());
                            coordinator.retain_local_observation_root(id,
                                LocalObservationRetirementBoundary::LocalRebind,&mut local).unwrap();
                            local=Some(replacement);
                            assert!(coordinator.retain_local_observation_root(id,
                                LocalObservationRetirementBoundary::LocalRebind,&mut local).is_err());
                            assert!(local.is_some());
                        } else if retain_local_roots {
                            retired_local_roots.push(local.replace(replacement).unwrap());
                        } else {local=Some(replacement);}
                    }
                    if endpoint as usize==id+1 {
                        let _capture=setup.lock().unwrap();
                        let before=live.rebind_state_sha256().unwrap();
                        let prepared=engine.prepare_resident_observation_parts(10,&cfg,&csr,&mut leads,live.clone()).unwrap();
                        assert_eq!(live.rebind_state_sha256().unwrap(),before);
                        coordinator.register_resident_worker(id,prepared).unwrap();
                    }
                    boundary.wait(||false).unwrap();
                    if active {
                        coordinator.defer_archived_endpoint(id,endpoint*10,archived).unwrap();
                        if endpoint==7 {
                            coordinator.drain_shared_endpoint(id,70,||boundary.is_aborted()).unwrap();
                            assert_eq!(engine.device_protocol_current_step().unwrap(),70);
                            assert_eq!(ghost.n_frames_written(),14);
                            let row=coordinator.verified_handoff_for_worker(id,70).unwrap();
                            let values=row["native_u32_values"].as_array().unwrap();
                            for field in 2..=4 {assert_eq!(values[field],archived as u64);}
                            assert_eq!(row["source_reset_before_next_physics"],false);handoffs.push(row);
                        }
                    } else {coordinator.exchange(id,endpoint*10,||boundary.is_aborted()).unwrap();}
                    if !active || endpoint==7 {
                        publication.as_ref().unwrap().validate_archived_population(&stream,
                            engine.spike_buffer_gpu().unwrap(),engine.spike_count_gpu().unwrap(),archived).unwrap();
                        engine.release_archived_spike_source().unwrap();
                        publication.as_mut().unwrap().acknowledge_source_reset(&stream,
                            engine.spike_buffer_gpu().unwrap(),engine.spike_count_gpu().unwrap()).unwrap();
                    }
                    boundary.wait(||false).unwrap();
                }
                assert_ne!(engine.get_positions().unwrap(),initial);
                assert_eq!(live.frozen_baseline_readback().unwrap()["frozen_sha256"],reference["frozen_sha256"]);
                let expected:usize=populations.lock().unwrap().iter().enumerate().filter(|(producer,_)|*producer!=id)
                    .flat_map(|(_,rows)|rows.iter()).sum();
                let mut inspected=0u64;
                for group in 0..4 {if group!=id {
                    let row=rings[group].lock().unwrap().recipient_receipt(&stream,id).unwrap();
                    assert_eq!(row.total_overwritten,0);assert_eq!(row.read_windows,7);
                    inspected+=row.total_inspected;
                }}
                assert_eq!(inspected,expected as u64);
                let telemetry=coordinator.execution_telemetry();
                assert_eq!(telemetry["shared_resident_parent"]["activation_step"],40);
                assert_eq!(telemetry["shared_resident_parent"]["parent_graph_launches"],3);
                assert_eq!(telemetry["shared_resident_parent"]["final_coupling_only_drain_completed"],true);
                assert_eq!(telemetry["shared_resident_parent"]["pending_archived_sources"],0);
                assert_eq!(retired_local_roots.len(),if retain_local_roots {2}else{0});
                if production_lifetime {
                    assert_eq!(telemetry["shared_resident_parent"]["retained_local_root_counts"],serde_json::json!(vec![if cancelled_handoff {1}else{2};4]));
                    assert_eq!(telemetry["shared_resident_parent"]["graph_retirement_completed"],false);
                }
                let mut result=serde_json::json!({"stream":id,"physical_steps":70,"ghost_records":14,
                    "retired_local_roots_retained_until_complete":retired_local_roots.len(),
                    "ghost_payload_sha256":format!("{:x}",Sha256::digest(ghost.payload_bytes())),
                    "seed_evidence":seed_evidence,"neighbor_argument_checks":neighbor_argument_checks,
                    "same_owner_checks":owner_checks,"physical_windows":window_receipts,
                    "native_source_handoffs":handoffs,"actual_recipient_records":inspected,
                    "coordinator":telemetry,"native_kcc_selection_equal":true});
                stream.synchronize().unwrap();
                // Every fallible readback finishes before any allocation owner
                // can retire the shared graph. The participant releases peers
                // on panic; no uncancellable fixture barrier is introduced.
                boundary.wait(||false).unwrap();
                if cancelled_handoff {
                    // Every physical replay/readback has completed. Keep the
                    // current local root in caller ownership until the actual
                    // production error path fences and retires global users.
                    coordinator.state.lock().unwrap().cancelled=Some("explicit completed-fixture cancellation".into());
                    let error=coordinator.retain_local_observation_root(id,
                        LocalObservationRetirementBoundary::SharedActivation,&mut local).unwrap_err();
                    assert!(local.is_some(),"successful cancellation fence preserves caller ownership");
                    let retired=coordinator.execution_telemetry();
                    assert_eq!(retired["shared_resident_parent"]["graph_retirement_completed"],true);
                    assert_eq!(retired["shared_resident_parent"]["released_local_root_count"],4);
                    assert_eq!(retired["shared_resident_parent"]["retained_local_root_counts"],serde_json::json!([0,0,0,0]));
                    result["cancelled_handoff"]=serde_json::json!({"error":format!("{error:#}"),
                        "input_root_still_owned":true,"shared_retirement_precedes_error_return":true,
                        "retirement":retired});
                }
                result
            })
        }).collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });
    if distinct_seeds {
        let base:std::collections::BTreeSet<_>=results.iter().map(|v|v["seed_evidence"]["base_position_sha256"].as_str().unwrap()).collect();
        let velocity:std::collections::BTreeSet<_>=results.iter().map(|v|v["seed_evidence"]["post_reset_velocity_sha256"].as_str().unwrap()).collect();
        assert_eq!(base.len(),1,"all seeded replicas must start from the same topology geometry");
        assert_eq!(velocity.len(),4,"native replica seeds must produce four distinct initial velocity fields");
    }
    context.synchronize().unwrap();
    context.check_err().unwrap();
    let retirement=coordinator.execution_telemetry();
    if production_lifetime {
        assert_eq!(retirement["shared_resident_parent"]["graph_retirement_completed"],true);
        assert_eq!(retirement["shared_resident_parent"]["released_local_root_count"],if cancelled_handoff {4}else{8});
        assert_eq!(retirement["shared_resident_parent"]["retained_local_root_counts"],serde_json::json!([0,0,0,0]));
    }
    println!(
        "{}",
        serde_json::json!({"schema":"prism.native_shared_physical_parent_fixture.v1",
        "scope":"four real physical owners; native 70-step scheduling, publication and lifetime validation per owner",
        "fixed_descriptor_memberships_are_regression_inputs":true,"actual_physical_spike_counts":*populations.lock().unwrap(),
        "active_asc_and_armed_g26":false,"distinct_physical_seeds":distinct_seeds,"neighbor_argument_inspection":inspect_neighbor_arguments,"production_bounded_lifetime":production_lifetime,"retirement":retirement,"workers":results})
    );
}
