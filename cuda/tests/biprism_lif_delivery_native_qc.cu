// Isolated actual-input producer/delivery controls, not an accepted trajectory.
// The injected clean convergence records exercise this boundary in isolation.
#define main biprism_retained_native_qc_main
#include "biprism_v10_native_qc.cu"
#undef main
#include "nhs_biprism_unit_bridge.cuh"
#include <limits>

template<class T> void retain_delivery(const std::string& path,const std::vector<T>& data) {
    std::ofstream f(path,std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(T));
    if(!f)throw std::runtime_error("delivery evidence write");
}
static float mul_rn_host(float a,float b){volatile float r=a*b;return r;}
static float add_rn_host(float a,float b){volatile float r=a+b;return r;}

int main(int argc,char**argv){
 try{
    if(argc!=5)throw std::runtime_error("usage: delivery_qc BPCASE02 AU_PHASE2_TILES TRUE_INVERSE_AU OUTPUT_PREFIX");
    const auto input=read_input(argv[1]);const std::string out=argv[4];
    std::ifstream ft(argv[2],std::ios::binary);
    auto host_tiles=read_vec<ContactShellSimplicialStateFfi>(ft,8);
    if(ft.peek()!=std::char_traits<char>::eof())throw std::runtime_error("tile extent");
    std::ifstream fm(argv[3],std::ios::binary);auto inverse=read_vec<float>(fm,input.n);
    if(fm.peek()!=std::char_traits<char>::eof())throw std::runtime_error("mass extent");
    for(float m:inverse)if(!std::isfinite(m)||m<=0)throw std::runtime_error("physical inverse mass domain");
    for(const auto&t:host_tiles)
        if(!report("actual_AU_tile",biprism_qc::validate_tile_topology(t,input.n,input.groups)))
            throw std::runtime_error("tile topology refused");
    rt(cudaSetDevice(0),"device");rt(cudaFree(nullptr),"context");
    cudaFuncAttributes delivery_attributes={};int delivery_blocks=0;
    rt(cudaFuncGetAttributes(&delivery_attributes,nhs_deliver_persisted_lif_forces_v1),"delivery attributes");
    rt(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&delivery_blocks,nhs_deliver_persisted_lif_forces_v1,256,0),"delivery occupancy");
    std::printf("delivery_resource regs=%d local_bytes=%zu static_smem=%zu active_blocks=%d\n",delivery_attributes.numRegs,delivery_attributes.localSizeBytes,delivery_attributes.sharedSizeBytes,delivery_blocks);
    if(delivery_attributes.localSizeBytes!=0||delivery_blocks<2)throw std::runtime_error("delivery resource gate");
    Device<float3> coords(8*input.n),vel(8*input.n),force(8*input.n);
    Device<float3> native_coords(8*input.n),native_vel(8*input.n),native_force(8*input.n);
    std::vector<float3> repeated;for(unsigned r=0;r<8;++r)repeated.insert(repeated.end(),input.positions.begin(),input.positions.end());
    native_coords.put(repeated);native_vel.zero();native_force.zero();coords.zero();vel.zero();force.zero();
    Device<float> true_inverse(input.n),integrator_inverse(input.n);true_inverse.put(inverse);integrator_inverse.zero();
    Device<ContactShellSimplicialStateFfi> tile(8);tile.put(host_tiles);
    FoliatedStreamPtrTableFfi native={},atomic={};
    for(unsigned r=0;r<8;++r){
        const size_t off=size_t(r)*input.n;
        native.d_coords_xyz[r]=reinterpret_cast<float*>(native_coords.p+off);
        native.d_vels_xyz[r]=reinterpret_cast<float*>(native_vel.p+off);
        native.d_forces_xyz[r]=reinterpret_cast<float*>(native_force.p+off);
        atomic.d_coords_xyz[r]=reinterpret_cast<float*>(coords.p+off);
        atomic.d_vels_xyz[r]=reinterpret_cast<float*>(vel.p+off);
        atomic.d_forces_xyz[r]=reinterpret_cast<float*>(force.p+off);
        atomic.d_true_inv_masses[r]=true_inverse.p;
        atomic.d_inv_masses[r]=integrator_inverse.p; // Deliberately zero: physical masses must own the kick.
    }
    Device<FoliatedStreamPtrTableFfi> nt(1),at(1);nt.put({native});at.put({atomic});
    Device<BiprismUnitBridgeStatusFfi> bridge(8);bridge.zero();
    Device<uint32_t> step(1);step.put({23u});
    Device<BiprismConvergenceRecordFfi> conv(8);std::vector<BiprismConvergenceRecordFfi> records(8);
    for(unsigned r=0;r<8;++r){records[r]={};records[r].abi_magic=BIPRISM_TAIL_ABI_MAGIC;records[r].leaf_id=r;records[r].committed_substeps=4;records[r].w_x_lambda_trace=2.f;}
    conv.put(records);
    nhs_biprism_import_native_state_v1<<<8,256>>>(nt.p,at.p,bridge.p,step.p,input.n,7u);
    nhs_biprism_guard_unit_import_v1<<<1,256>>>(bridge.p,conv.p,step.p);
    rt(cudaGetLastError(),"actual input import");rt(cudaDeviceSynchronize(),"import completion");
    for(const auto&b:bridge.get())if(b.sticky_fault||b.imported_fields!=7u)throw std::runtime_error("input bridge refused");
    const auto positions=coords.get();
    for(unsigned r=0;r<8;++r)for(unsigned v=0;v<256;++v){
        const auto atom=UNPACK_GLOBAL_ATOM_ID(host_tiles[r].global_atom_idx[v]);
        const auto p=positions[size_t(r)*input.n+atom];const auto ref=host_tiles[r].ref_pos_q[v];
        if(p.x!=ref.x||p.y!=ref.y||p.z!=ref.z)throw std::runtime_error("tile/reference and imported source coordinates disagree");
    }
    float lo[3]={std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()};
    float hi[3]={-lo[0],-lo[1],-lo[2]};
    for(const auto&p:positions){const float a[3]={p.x,p.y,p.z};for(unsigned d=0;d<3;++d){lo[d]=std::min(lo[d],a[d]);hi[d]=std::max(hi[d],a[d]);}}
    SignalGridSpecFfi grid={};grid.inv_voxel_spacing=.125f;grid.total_voxels=1;
    for(unsigned d=0;d<3;++d){
        grid.origin_xyz[d]=lo[d]-8.f;const double dim=std::ceil((double(hi[d])-lo[d])/8.)+3.;
        if(!std::isfinite(dim)||dim<1.||dim>10000000.)throw std::runtime_error("grid dimension");
        grid.dim_xyz[d]=uint32_t(dim);
        const uint64_t size=uint64_t(grid.total_voxels)*grid.dim_xyz[d];
        if(size>10000000u)throw std::runtime_error("grid product");grid.total_voxels=uint32_t(size);
    }
    if(!grid.total_voxels||grid.total_voxels>10000000u)throw std::runtime_error("grid extent");
    Device<int32_t> hits(grid.total_voxels),coupled(grid.total_voxels),owner(grid.total_voxels),counts(grid.total_voxels),residues(input.n);
    hits.zero();coupled.zero();counts.zero();owner.put(std::vector<int32_t>(grid.total_voxels,-1));residues.put(input.residue_ids);
    Device<float> water(grid.total_voxels),exclusion(grid.total_voxels),voltage(8*256);water.zero();exclusion.zero();
    voltage.put(std::vector<float>(8*256,3.9f)); // Controlled firing state, equal on core and halo.
    Device<GpuSpikeEventCanonical96> queue(8*TILE_CORE_VERTICES);queue.zero();Device<uint32_t> qcount(1);qcount.zero();
    Device<BiprismLifCarryoverFfi> carry(8);carry.zero();
    nhs_voronoi_dual_lif_and_4grid_bypass_v2<<<8,256>>>(at.p,tile.p,conv.p,step.p,residues.p,voltage.p,
        hits.p,coupled.p,owner.p,counts.p,water.p,exclusion.p,queue.p,qcount.p,grid,uint32_t(queue.n),1.f,1.25f,.04f,carry.p,0u);
    rt(cudaGetLastError(),"producer");rt(cudaDeviceSynchronize(),"producer completion");
    const auto staged=carry.get();retain_delivery(out+".produced_carry.bin",staged);
    retain_delivery(out+".produced_queue.bin",queue.get());
    if(qcount.get()[0]==0||qcount.get()[0]>queue.n)throw std::runtime_error("empty or overflowing producer");
    for(unsigned r=0;r<8;++r){
        const auto&c=staged[r];if(c.fault||c.pending!=1||c.produced_step!=23||c.produced_committed_substeps!=4)throw std::runtime_error("carry production refused");
        double sum[3]={};double magnitude=0,halo_magnitude=0;
        for(unsigned v=0;v<256;++v){const float4 f=c.force[v];const double a[3]={f.x,f.y,f.z};
            for(unsigned d=0;d<3;++d){sum[d]+=a[d];magnitude+=std::abs(a[d]);if(v>=TILE_CORE_VERTICES)halo_magnitude+=std::abs(a[d]);}}
        std::printf("leaf=%u produced_sum=%.17g,%.17g,%.17g force_l1=%.17g halo_l1=%.17g\n",r,sum[0],sum[1],sum[2],magnitude,halo_magnitude);
        if(sum[0]!=0||sum[1]!=0||sum[2]!=0||!(magnitude>0)||!(halo_magnitude>0))throw std::runtime_error("produced core/halo force balance");
    }
    for(const auto&f:force.get())if(f.x!=0||f.y!=0||f.z!=0)throw std::runtime_error("carry-only producer changed force buffer");
    // Overwrite the complete force allocation; the delivery path may only read carry.
    force.put(std::vector<float3>(8*input.n,make_float3(17.f,-31.f,47.f)));
    const auto force_sentinel=force.get();const auto after_overwrite=carry.get();
    if(std::memcmp(staged.data(),after_overwrite.data(),8*sizeof(staged[0])))throw std::runtime_error("force clear overwrote carry");
    step.put({24u});for(auto&r:records)r.committed_substeps=8u;conv.put(records);
    const float dt=static_cast<float>(.0005*41341.37333517015);
    const auto deliver=[&](){nhs_deliver_persisted_lif_forces_v1<<<1,256>>>(at.p,carry.p,conv.p,step.p,input.n,dt);rt(cudaGetLastError(),"delivery");rt(cudaDeviceSynchronize(),"delivery completion");};
    // Each fault injection is an independent fixture, restored from the retained
    // producer snapshot. These resets do not erase an engine fault history.
    for(unsigned bad=0;bad<8;++bad){
        auto faulted=records;faulted[bad].epoch_cluster_fault=FAULT_PICARD_NONCONV;
        conv.put(faulted);carry.put(staged);vel.zero();deliver();
        for(const auto&v:vel.get())if(v.x!=0||v.y!=0||v.z!=0)throw std::runtime_error("one-leaf fault allowed velocity mutation");
        const auto held=carry.get();
        if(!(held[bad].fault&BIPRISM_CARRY_UPSTREAM_FAULT))throw std::runtime_error("upstream fault evidence missing");
        for(unsigned r=0;r<8;++r){auto expected=staged[r];expected.fault=held[r].fault;
            if(held[r].pending!=1||std::memcmp(&held[r],&expected,sizeof(expected)))throw std::runtime_error("faulted delivery changed carry payload");}
    }
    conv.put(records);bool momentum_ok=true;
    for(unsigned baseline=0;baseline<2;++baseline){
        std::vector<float3> initial(8*input.n,make_float3(0,0,0));
        if(baseline)for(size_t i=0;i<initial.size();++i)initial[i]=make_float3(float(int(i%7)-3)*0.000244140625f,float(int(i%11)-5)*0.0001220703125f,float(int(i%5)-2)*0.00048828125f);
        vel.put(initial);carry.put(staged);deliver();const auto result=vel.get();const auto consumed=carry.get();
        retain_delivery(out+".baseline"+std::to_string(baseline)+".vel_before.bin",initial);
        retain_delivery(out+".baseline"+std::to_string(baseline)+".vel_after.bin",result);
        auto expected=initial;
        for(unsigned r=0;r<8;++r){double dp[3]={};double max_error=0,kinetic_delta=0,requested_work=0,realized_work=0;
            for(unsigned v=0;v<256;++v){const auto atom=staged[r].atom_id[v];if(atom>=input.n)throw std::runtime_error("captured atom bounds");
                const size_t i=size_t(r)*input.n+atom;const auto f=staged[r].force[v];
                expected[i]=result[i]; // Validated below; all unrelated atoms still compare byte-for-byte.
                const double dv[3]={double(result[i].x)-initial[i].x,double(result[i].y)-initial[i].y,double(result[i].z)-initial[i].z};const double ff[3]={f.x,f.y,f.z};
                const double before[3]={initial[i].x,initial[i].y,initial[i].z},after[3]={result[i].x,result[i].y,result[i].z};
                for(unsigned d=0;d<3;++d){
                    const long double exact=std::fma((long double)dt*(long double)inverse[atom],(long double)ff[d],(long double)before[d]);
                    float lower=(float)exact,upper=lower;
                    if((long double)lower>exact)lower=std::nextafter(lower,-INFINITY);
                    if((long double)upper<exact)upper=std::nextafter(upper,INFINITY);
                    if(after[d]!=(double)lower&&after[d]!=(double)upper)throw std::runtime_error("delivered velocity outside adjacent physical update representations");
                    const double impulse=dv[d]/double(inverse[atom]);dp[d]+=impulse;max_error=std::max(max_error,std::abs(impulse-double(dt)*ff[d]));
                    const double vavg=.5*(before[d]+after[d]);
                    kinetic_delta+=.5*(after[d]*after[d]-before[d]*before[d])/double(inverse[atom]);
                    requested_work+=double(dt)*ff[d]*vavg;realized_work+=impulse*vavg;
                }
            }
            const double net=std::max({std::abs(dp[0]),std::abs(dp[1]),std::abs(dp[2])});
            std::printf("baseline=%u leaf=%u net_dp=%.17g max_atom_dp_error=%.17g momentum_gate=%d\n",baseline,r,net,max_error,net<=1.e-7);
            std::printf("baseline=%u leaf=%u kinetic_delta=%.17g requested_impulse_work=%.17g realized_impulse_work=%.17g represented_work_error=%.17g kinetic_identity_error=%.17g\n",baseline,r,kinetic_delta,requested_work,realized_work,realized_work-requested_work,kinetic_delta-realized_work);
            if(!std::isfinite(kinetic_delta)||!std::isfinite(requested_work)||!std::isfinite(realized_work))throw std::runtime_error("nonfinite impulse work");
            momentum_ok&=std::isfinite(net)&&net<=1.e-7;
            auto expected_carry=staged[r];expected_carry.pending=0;expected_carry.delivered_step=24u;
            if(std::memcmp(&expected_carry,&consumed[r],sizeof(expected_carry)))throw std::runtime_error("delivery acknowledgement/payload changed");
        }
        if(std::memcmp(expected.data(),result.data(),result.size()*sizeof(float3)))throw std::runtime_error("delivery changed unrelated atom");
        deliver();const auto twice=vel.get();const auto carry_twice=carry.get();
        if(std::memcmp(twice.data(),result.data(),result.size()*sizeof(float3))||std::memcmp(carry_twice.data(),consumed.data(),8*sizeof(consumed[0])))throw std::runtime_error("delivery applied twice");
        const auto force_after=force.get();if(std::memcmp(force_after.data(),force_sentinel.data(),force_after.size()*sizeof(float3)))throw std::runtime_error("delivery touched AMBER force buffer");
    }
    // A balanced impulse can have no admissible FP32 output pair at very large
    // baseline velocity. Such a leaf must cause collective refusal, not projection.
    for(unsigned bad=0;bad<8;++bad){
        auto impossible=staged;for(auto&f:impossible[bad].force)f=make_float4(0,0,0,0);
        impossible[bad].force[0].x=1.f;impossible[bad].force[1].x=-1.f;
        std::vector<float3> initial(8*input.n,make_float3(0,0,0));
        initial[size_t(bad)*input.n+impossible[bad].atom_id[1]].x=0x1p26f;
        vel.put(initial);carry.put(impossible);deliver();const auto held=carry.get();const auto unchanged=vel.get();
        if(std::memcmp(initial.data(),unchanged.data(),initial.size()*sizeof(float3)))throw std::runtime_error("infeasible leaf allowed partial physical delivery");
        if(!(held[bad].fault&BIPRISM_CARRY_NUMERIC_FAULT))throw std::runtime_error("infeasible representation did not produce numeric refusal");
        for(unsigned r=0;r<8;++r){auto expected=impossible[r];expected.fault=held[r].fault;
            if(std::memcmp(&held[r],&expected,sizeof(expected)))throw std::runtime_error("infeasible representation changed pending payload");}
    }
    const auto coords_after=coords.get();const auto tiles_after=tile.get();
    if(std::memcmp(coords_after.data(),positions.data(),positions.size()*sizeof(float3))||
       std::memcmp(tiles_after.data(),host_tiles.data(),8*sizeof(host_tiles[0])))throw std::runtime_error("LIF delivery mutated coordinate or tile state");
    std::printf("isolated_actual_AU_LIF_delivery=1 eight_leaf_fault_cases=8 adjacent_true_mass_update=1 infeasible_leaf_refusal_cases=8 once_only=1 momentum_gate=%d full_trajectory_accepted=0\n",momentum_ok);
    return momentum_ok?0:4;
 }catch(const std::exception&e){std::fprintf(stderr,"delivery_qc_failed: %s\n",e.what());return 1;}
}
