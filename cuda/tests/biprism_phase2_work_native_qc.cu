// Native measurement controls on the retained actual KRAS 960/1461 complex.
// This is not a physical trajectory; it does not launch v10 or advance a clock.
#define main biprism_retained_native_qc_main
#include "biprism_v10_native_qc.cu"
#undef main
#include "nhs_biprism_phase2_work.cuh"
#include <sstream>
static_assert(sizeof(FoliatedStreamPtrTableFfi)==640,"post-join pointer-table ABI");
static_assert(sizeof(BiprismPhase2VertexWorkFfi)==224,"BPW3 Phase2 vertex readback ABI");
static_assert(sizeof(BiprismPhase2EdgeWorkFfi)==80,"Phase2 edge readback ABI");
static_assert(sizeof(BiprismPhase2WorkLedgerFfi)==240,"Phase2 ledger readback ABI");
template<class T> static void p2_save(const std::string& p,const std::vector<T>& v){
 std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T));
 if(!f)throw std::runtime_error("Phase2 control output write");
}
__global__ void phase2_control_corrupt_identity(BiprismPhase2EdgeWorkFfi* e){
 if(blockIdx.x==0&&threadIdx.x==0)e[0].energy_after+=2e-5;
}
static bool run_p2_case(const Input& x,const std::vector<ContactShellSimplicialStateFfi>& seed,
 const std::string& out,const std::string& name,uint32_t seed_target=0u){
 const bool missing=name=="cold_placeholder",cold_leaf=name=="cold_leaf_refusal",deformed=name=="deformed",boundary=name=="boundary",corrupt=name=="identity_refusal",low_plane=name=="low_plane";
 Device<float3> nx(8*x.n),nv(8*x.n),nf(8*x.n),ax(8*x.n),av(8*x.n),af(8*x.n);
 Device<float3> nl(8*x.n),ns(8*x.n),nsl(8*x.n),nr(8*x.n),nrl(8*x.n);
 Device<float3> al(8*x.n),as(8*x.n),asl(8*x.n),ar(8*x.n),arl(8*x.n);
 std::vector<float3> repeated;for(unsigned l=0;l<8;++l)repeated.insert(repeated.end(),x.positions.begin(),x.positions.end());
 nx.put(repeated);nv.zero();nf.zero();ax.zero();av.zero();af.zero();
 nl.zero();ns.put(repeated);nsl.zero();nr.put(repeated);nrl.zero();
 al.zero();as.zero();asl.zero();ar.zero();arl.zero();
 Device<BiprismCoordinateOwnerFfi> no(8),ao(8);
 std::vector<BiprismCoordinateOwnerFfi> native_owners(8),au_owners(8);
 FoliatedStreamPtrTableFfi native={},au={};for(unsigned l=0;l<8;++l){
  native.d_coords_xyz[l]=reinterpret_cast<float*>(nx.p+l*x.n);native.d_vels_xyz[l]=reinterpret_cast<float*>(nv.p+l*x.n);native.d_forces_xyz[l]=reinterpret_cast<float*>(nf.p+l*x.n);
  au.d_coords_xyz[l]=reinterpret_cast<float*>(ax.p+l*x.n);au.d_vels_xyz[l]=reinterpret_cast<float*>(av.p+l*x.n);au.d_forces_xyz[l]=reinterpret_cast<float*>(af.p+l*x.n);
  auto& n=native_owners[l];n={BP_COORDINATE_MAGIC,BP_COORDINATE_VERSION,l,x.n,23,
   BP_COORD_NATIVE_ANGSTROM,BP_COORD_ALL_INITIALIZED,native.d_coords_xyz[l],reinterpret_cast<float*>(nl.p+l*x.n),
   reinterpret_cast<float*>(ns.p+l*x.n),reinterpret_cast<float*>(nsl.p+l*x.n),reinterpret_cast<float*>(nr.p+l*x.n),reinterpret_cast<float*>(nrl.p+l*x.n),{0,0}};
  auto& a=au_owners[l];a={BP_COORDINATE_MAGIC,BP_COORDINATE_VERSION,l,x.n,23,
   BP_COORD_ATOMIC_BOHR,BP_COORD_ALL_INITIALIZED,au.d_coords_xyz[l],reinterpret_cast<float*>(al.p+l*x.n),
   reinterpret_cast<float*>(as.p+l*x.n),reinterpret_cast<float*>(asl.p+l*x.n),reinterpret_cast<float*>(ar.p+l*x.n),reinterpret_cast<float*>(arl.p+l*x.n),{0,0}};}
 no.put(native_owners);ao.put(au_owners);native.d_coordinate_owners=no.p;au.d_coordinate_owners=ao.p;
 Device<FoliatedStreamPtrTableFfi> nt(1),at(1);nt.put({native});at.put({au});Device<float> charges(x.n);charges.put(x.charges);
 Device<BiprismUnitBridgeStatusFfi> bridge(8);bridge.zero();Device<uint32_t> step(1);step.put({0});
 std::vector<BiprismConvergenceRecordFfi> cs(8);for(unsigned l=0;l<8;++l){cs[l]={};cs[l].abi_magic=BIPRISM_TAIL_ABI_MAGIC;cs[l].leaf_id=l;}
 Device<BiprismConvergenceRecordFfi> conv(8);conv.put(cs);
 Device<ContactShellSimplicialStateFfi> tile(8),old(8);tile.put(seed);old.zero();
 Device<double> accepted_phi(8*256),prepared_phi(8*256);accepted_phi.zero();prepared_phi.zero();
 Device<BiprismPotentialOwnerFfi> potential(8);std::vector<BiprismPotentialOwnerFfi> potential_rows(8);
 for(unsigned leaf=0;leaf<8;++leaf){auto& p=potential_rows[leaf];p={};p.abi_magic=BP_POTENTIAL_MAGIC;p.version=BP_POTENTIAL_VERSION;
  p.leaf_id=leaf;p.allocation_generation=23;p.topology_generation=1;
  p.tile_address=reinterpret_cast<uint64_t>(tile.p+leaf);p.coordinate_owner_address=reinterpret_cast<uint64_t>(ao.p+leaf);
  p.accepted_phi=accepted_phi.p+leaf*256;p.prepared_phi=prepared_phi.p+leaf*256;}
 Device<BiprismSpinorOwnerFfi> spinors(8);std::vector<BiprismSpinorOwnerFfi> spinor_rows(8);
 for(unsigned l=0;l<8;++l){auto& o=spinor_rows[l];const auto& p=potential_rows[l];o={};o.magic=BP_SPINOR_MAGIC;o.version=1;o.leaf=l;
  o.allocation_generation=p.allocation_generation;o.topology_generation=p.topology_generation;o.tile_address=p.tile_address;o.coordinate_owner_address=p.coordinate_owner_address;}
 spinors.put(spinor_rows);au.d_spinor_owners=spinors.p;
 potential.put(potential_rows);au.d_potential_owners=potential.p;at.put({au});
 Device<BiprismPhase2VertexWorkFfi> vr(8*256),vj(8*256);vr.zero();vj.zero();
 Device<BiprismPhase2EdgeWorkFfi> er(8*1536),ej(8*1536);er.zero();ej.zero();
 Device<BiprismPhase2WorkLedgerFfi> lr(8),lj(8);lr.zero();lj.zero();
 nhs_biprism_import_native_state_v1<<<8,256>>>(nt.p,at.p,bridge.p,step.p,x.n,7u);
 nhs_biprism_guard_unit_import_v1<<<1,256>>>(bridge.p,conv.p,step.p);rt(cudaGetLastError(),"Phase2 cold import");rt(cudaDeviceSynchronize(),"Phase2 cold import completion");
 std::vector<ContactShellSimplicialStateFfi> actual_seed=seed;
 if(seed_target){
  // Diagnostic-only fresh native discretization. Same physical atom slots and
  // imported high/low coordinates at every target; production target unchanged.
  for(const auto& b:bridge.get())if(b.sticky_fault||b.imported_fields!=7u)throw std::runtime_error("mesh sensitivity cold import refused");
  Device<uint8_t> elements(x.n);elements.put(x.elements);
  std::vector<uint32_t> chosen;std::vector<const float*> pointers;
  for(unsigned leaf=0;leaf<8;++leaf){pointers.push_back(au.d_coords_xyz[leaf]);
   for(unsigned u=0;u<256;++u)chosen.push_back(UNPACK_GLOBAL_ATOM_ID(seed[leaf].global_atom_idx[u]));}
  Device<uint32_t> selected(chosen.size());selected.put(chosen);
  Device<const float*> coordinates(8);coordinates.put(pointers);
  Device<BiprismBwScratchFfi> scratch(8);Device<BiprismBwStatusFfi> status(8);status.zero();tile.zero();
  nhs_seed_bowyer_watson_256_compensated_v1<<<8,256>>>(coordinates.p,charges.p,elements.p,selected.p,x.n,
   tile.p,scratch.p,status.p,8u,seed_target,ao.p);
  rt(cudaGetLastError(),"sensitivity BW");rt(cudaDeviceSynchronize(),"sensitivity BW completion");
  const auto states=status.get();actual_seed=tile.get();
  p2_save(out+".requested_target.u32le",std::vector<uint32_t>{seed_target});
  p2_save(out+".selected_atoms.u32le",chosen);p2_save(out+".bw_status.80b",states);
  p2_save(out+".seed_tiles.bin",actual_seed);
  p2_save(out+".coordinates_au.f32le",ax.get());p2_save(out+".coordinates_au_low.f32le",al.get());
  p2_save(out+".reference_au.f32le",ar.get());p2_save(out+".reference_au_low.f32le",arl.get());
  p2_save(out+".coordinates_native.f32le",nx.get());p2_save(out+".coordinates_native_low.f32le",nl.get());
  p2_save(out+".coordinate_owners_native.96b",no.get());p2_save(out+".coordinate_owners_au.96b",ao.get());
  p2_save(out+".bridge.64b",bridge.get());
  bool admitted=true;
  for(unsigned leaf=0;leaf<8;++leaf){const auto& s=states[leaf];const auto& t=actual_seed[leaf];
   admitted=admitted&&s.abi_magic==BPBW_STATUS_MAGIC&&s.leaf_id==leaf&&!s.error&&s.phase==5u
    &&s.inserted_vertices==256u&&s.published==1u&&s.final_tets>0u&&s.final_tets<=seed_target
    &&s.final_edges>0u&&s.final_edges<=1536u&&s.peel_passes<=BPBW_MAX_PEEL_PASSES;
   if(s.published){admitted=admitted&&biprism_qc::validate_tile_topology(t,x.n,x.groups).ok();
    for(unsigned u=0;u<256;++u)admitted=admitted&&UNPACK_GLOBAL_ATOM_ID(t.global_atom_idx[u])==chosen[leaf*256+u];}
   std::printf("mesh_target=%u leaf=%u seed_error=%u published=%u tets=%u edges=%u peel_passes=%u\n",
    seed_target,leaf,s.error,s.published,s.final_tets,s.final_edges,s.peel_passes);
  }
  if(!admitted){p2_save(out+".initial_step.u32le",step.get());return false;}
 }else p2_save(out+".seed_tiles.bin",seed);
 if(!missing)for(unsigned l=0;l<8;++l)nhs_biprism_simplicial_tile_compensated_v1<<<1,256>>>(au.d_coords_xyz[l],charges.p,tile.p,l,ao.p,conv.p);
 if(cold_leaf){auto invalid=tile.get();invalid[3].star1_wt[0]=0.f;tile.put(invalid);}
 if(seed_target){rt(cudaGetLastError(),"sensitivity Phase2");rt(cudaDeviceSynchronize(),"sensitivity Phase2 completion");p2_save(out+".phase2_tiles.bin",tile.get());}
 p2_save(out+".provisional_tiles.bin",tile.get());
 nhs_biprism_phase2_work_initialize_v1<<<8,256>>>(tile.p,at.p,charges.p,x.n,conv.p,vr.p,lr.p);
 nhs_biprism_phase2_work_initialize_publish_v1<<<1,256>>>(tile.p,at.p,conv.p,vr.p,lr.p,step.p);
 rt(cudaGetLastError(),"Phase2 cold qualification");rt(cudaDeviceSynchronize(),"Phase2 cold qualification completion");
 p2_save(out+".initial_tiles.bin",tile.get());p2_save(out+".initial_ledgers.240b",lr.get());
 p2_save(out+".initial_convergence.128b",conv.get());p2_save(out+".initial_step.u32le",step.get());
 p2_save(out+".initial_potential_owners.128b",potential.get());p2_save(out+".initial_phi_accepted.f64le",accepted_phi.get());
 p2_save(out+".initial_phi_prepared.f64le",prepared_phi.get());
 p2_save(out+".initial_vertices.256b",vr.get());
 if(missing||cold_leaf){for(unsigned l=0;l<8;++l)if(lr.get()[l].state==4u||!(conv.get()[l].terminal_fault_latch&BIPRISM_PHASE2_WORK_TERMINAL)
   ||potential.get()[l].state!=BP_POTENTIAL_COLD)throw std::runtime_error("cold invalid operator published an accepted leaf");
  for(double p:accepted_phi.get())if(p!=0.)throw std::runtime_error("cold refusal changed accepted bank");
  for(double p:prepared_phi.get())if(p!=0.)throw std::runtime_error("cold refusal changed prepared bank");
  std::printf("phase2_case=%s cold_accepted_publication_refused=8 clock=0 trajectory=0\n",name.c_str());return true;}
 if(seed_target){
  bool admitted=true;const auto ledgers=lr.get();const auto records=conv.get();
  for(unsigned leaf=0;leaf<8;++leaf){const auto& l=ledgers[leaf];const auto& c=records[leaf];
   admitted=admitted&&l.state==4u&&!l.sticky_fault&&std::isfinite(l.residual[0])&&l.residual[0]<=1e-6f
    &&!c.epoch_leaf_fault&&!c.epoch_cluster_fault&&!c.terminal_fault_latch&&!c.committed_substeps;
   std::printf("mesh_target=%u leaf=%u cold_operator_state=%u fault=%u residual=%.9g clock=0 trajectory=0\n",
    seed_target,leaf,l.state,l.sticky_fault,l.residual[0]);}
  if(step.get()[0]!=0u)throw std::runtime_error("mesh sensitivity advanced protocol clock");
  return admitted;
 }
 for(const auto& l:lr.get())if(l.state!=4u||l.sticky_fault||l.residual[0]>1e-6f)throw std::runtime_error("actual cold Phase2/DEC refused");
 if(deformed)for(unsigned leaf=0;leaf<8;++leaf)for(unsigned u=0;u<192;++u){
  const unsigned atom=UNPACK_GLOBAL_ATOM_ID(seed[leaf].global_atom_idx[u]);
  repeated[leaf*x.n+atom].x+=float(int(u%7)-3)*0.00025f;}
 if(low_plane){std::vector<float3> low(8*x.n,{0.f,0.f,0.f});
  for(unsigned leaf=0;leaf<8;++leaf)for(unsigned u=0;u<192;++u){const unsigned atom=UNPACK_GLOBAL_ATOM_ID(seed[leaf].global_atom_idx[u]);
   const float hi=repeated[leaf*x.n+atom].x;
   low[leaf*x.n+atom].x=.25f*(std::nextafter(hi,INFINITY)-hi);}
  nl.put(low);}
 nx.put(repeated);step.put({1});
 nhs_biprism_import_native_state_v1<<<8,256>>>(nt.p,at.p,bridge.p,step.p,x.n,7u);
 nhs_biprism_guard_unit_import_v1<<<1,256>>>(bridge.p,conv.p,step.p);rt(cudaDeviceSynchronize(),"Phase2 refresh import");
 if(boundary){
  // Explicit fixed-state boundary control, not a trajectory commit: retain the
  // prior prescribed-halo perturbation in its authoritative full bank and its
  // exact projection. Actual Phase2 restores the declared charge-based halo.
  auto t=tile.get();auto p=accepted_phi.get();for(unsigned leaf=0;leaf<8;++leaf){
   t[leaf].phi_dec[192]+=0.001f;p[leaf*256+192]=double(t[leaf].phi_dec[192]);}
  accepted_phi.put(p);tile.put(t);}
 p2_save(out+".before_potential_owners.128b",potential.get());p2_save(out+".before_phi_accepted.f64le",accepted_phi.get());
 p2_save(out+".before_phi_prepared.f64le",prepared_phi.get());
 p2_save(out+".before_tiles.bin",tile.get());p2_save(out+".coordinates_au.f32le",ax.get());
 p2_save(out+".coordinates_au_low.f32le",al.get());p2_save(out+".reference_au.f32le",ar.get());p2_save(out+".reference_au_low.f32le",arl.get());
 p2_save(out+".coordinates_native.f32le",nx.get());p2_save(out+".coordinates_native_low.f32le",nl.get());
 BiprismSharedStepFfi sh={};sh.abi_magic=BIPRISM_SHARED_STEP_MAGIC;sh.state=BIPRISM_SHARED_STEP_SELECTED;sh.scheduled_step=1;sh.generation=23;sh.policy=1;
 Device<BiprismSharedStepFfi> shared(1);shared.put({sh});
 BiprismPhase2WorkWindowFfi wi={BIPRISM_PHASE2_WORK_MAGIC,1,0,1,1,0,23,1,0};Device<BiprismPhase2WorkWindowFfi> window(1);window.put({wi});
 nhs_biprism_phase2_work_capture_v1<<<8,256>>>(tile.p,at.p,charges.p,bridge.p,conv.p,shared.p,x.n,.1f,old.p,vr.p,er.p,lr.p,window.p);
 for(unsigned l=0;l<8;++l)nhs_biprism_simplicial_tile_compensated_v1<<<1,256>>>(au.d_coords_xyz[l],charges.p,tile.p,l,ao.p,conv.p);
 nhs_biprism_phase2_work_measure_v1<<<8,256>>>(tile.p,at.p,charges.p,old.p,conv.p,shared.p,vr.p,er.p,lr.p);
 if(corrupt)phase2_control_corrupt_identity<<<1,1>>>(er.p);
 nhs_biprism_phase2_work_publish_v1<<<1,256>>>(conv.p,shared.p,vr.p,er.p,lr.p,window.p,vj.p,ej.p,lj.p,tile.p,at.p);
 rt(cudaGetLastError(),"Phase2 refresh measurement");rt(cudaDeviceSynchronize(),"Phase2 refresh completion");
 p2_save(out+".after_tiles.bin",tile.get());p2_save(out+".ledgers.240b",lr.get());p2_save(out+".vertices.256b",vr.get());p2_save(out+".edges.80b",er.get());
 p2_save(out+".after_potential_owners.128b",potential.get());p2_save(out+".after_phi_accepted.f64le",accepted_phi.get());
 p2_save(out+".after_phi_prepared.f64le",prepared_phi.get());
 p2_save(out+".journal.240b",lj.get());p2_save(out+".convergence.128b",conv.get());p2_save(out+".shared.128b",shared.get());p2_save(out+".window.48b",window.get());
 p2_save(out+".scheduled_step.u32le",step.get());
 for(unsigned l=0;l<8;++l){const auto a=lr.get()[l];const auto c=conv.get()[l];
  if(corrupt){if(!(a.sticky_fault&BIPRISM_PHASE2_WORK_IDENTITY)||!(c.terminal_fault_latch&BIPRISM_PHASE2_WORK_TERMINAL)||a.state==3u)throw std::runtime_error("identity fault was not collectively withheld");}
  else if(a.state!=3u||a.sticky_fault||a.completed_measurements!=1u||!biprism_phase2_identities_admitted(a.totals)||c.committed_substeps||c.terminal_fault_latch)throw std::runtime_error("actual refresh measurement refused");
  std::printf("phase2_case=%s leaf=%u tets=%u edges=%u state=%u fault=%u total_work=%.17g operator_identity=%.17g boundary_identity=%.17g residuals=%.9g,%.9g,%.9g\n",name.c_str(),l,a.tets,a.edges,a.state,a.sticky_fault,a.totals[12],a.totals[13],a.totals[14],a.residual[0],a.residual[1],a.residual[2]);}
 if(step.get()[0]!=1u||shared.get()[0].accepted_bodies!=0u)throw std::runtime_error("measurement changed physical clock");
 return true;
}
int main(int argc,char**argv){try{
 if(argc!=4&&argc!=5)throw std::runtime_error("usage: phase2_native_qc BPCASE02 actual_seed_tiles.bin output_prefix [--mesh-budgets=944,960,976]");
 std::vector<uint32_t> budgets;
 if(argc==5){const std::string option=argv[4],flag="--mesh-budgets=";
  if(option.compare(0,flag.size(),flag))throw std::runtime_error("unknown Phase2 option");
  std::istringstream in(option.substr(flag.size()));std::string token;
  while(std::getline(in,token,',')){if(token.empty()||token.find_first_not_of("0123456789")!=std::string::npos)throw std::runtime_error("invalid mesh target");
   const auto value=std::stoul(token);if(!value||value>1024u)throw std::runtime_error("mesh target outside native capacity");budgets.push_back(uint32_t(value));}
  if(budgets.size()!=3u||!(budgets[0]<960u&&budgets[1]==960u&&budgets[2]>960u))throw std::runtime_error("three ordered diagnostic targets must bracket production960");
 }
 const auto x=read_input(argv[1]);std::ifstream f(argv[2],std::ios::binary);auto seed=read_vec<ContactShellSimplicialStateFfi>(f,8);
 if(f.peek()!=std::char_traits<char>::eof())throw std::runtime_error("seed extent");
 for(unsigned l=0;l<8;++l){const auto&t=seed[l];if(t.num_active_tets!=960u||t.num_active_edges!=1461u||!biprism_qc::validate_tile_topology(t,x.n,x.groups).ok())throw std::runtime_error("actual retained 960/1461 complex required");
  for(unsigned e=0;e<t.num_active_edges;++e)if(t.star1_wt[e]!=0.f)throw std::runtime_error("cold seed is not exact zero-weight placeholder");
  for(unsigned u=0;u<256;++u)if(t.star0_vol[u]!=1.f)throw std::runtime_error("cold seed star0 not exact retained placeholder");}
 rt(cudaSetDevice(0),"device");rt(cudaFree(nullptr),"context");
 if(!budgets.empty()){bool passed=true;for(uint32_t budget:budgets)
   passed=run_p2_case(x,seed,std::string(argv[3])+".target"+std::to_string(budget),"mesh_sensitivity",budget)&&passed;
  std::printf("native_mesh_sensitivity_targets=3 all8_cold_operators=%u production_target=960 clocks_unchanged=1 physical_trajectory_acceptance=0\n",unsigned(passed));
  return passed?0:4;}
 for(const std::string name:{"cold_placeholder","cold_leaf_refusal","unchanged","deformed","boundary","identity_refusal","low_plane"})run_p2_case(x,seed,std::string(argv[3])+"."+name,name);
 std::puts("phase2_native_measurement_controls=7 core192=1 halo64=1 clocks_unchanged=1 physical_trajectory_acceptance=0");return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
