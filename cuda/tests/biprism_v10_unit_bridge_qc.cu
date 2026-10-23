// Native quantity/accepted-state diagnostic using the actual KRAS source input.
// Zero initial velocities/bath forces are explicit controls, not an AMBER run.
#include <cstdio>
#define main biprism_retained_native_qc_main
#include "biprism_v10_native_qc.cu"
#undef main
#include "nhs_biprism_unit_bridge.cuh"
#include <sstream>

template<class T> void retain_qc(const std::string& path,const std::vector<T>& data) {
    std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(T));
    if(!f)throw std::runtime_error("native unit evidence write");
}
int main(int argc,char** argv) {
    try {
        if(argc!=5)throw std::runtime_error("usage: unit_qc BPCASE02 selected.tsv true_inverse_au.f32le output_prefix");
        const auto x=read_input(argv[1]);const std::string out=argv[4];
        std::ifstream fids(argv[2]);std::string line;std::getline(fids,line);
        std::vector<uint32_t> ids;while(std::getline(fids,line)) {
            std::istringstream row(line);unsigned leaf,v,a,core;
            if(!(row>>leaf>>v>>a>>core))throw std::runtime_error("selection transport");
            if(leaf==0){if(v!=ids.size()||a>=x.n)throw std::runtime_error("selection identity");ids.push_back(a);}
        }
        if(ids.size()!=256)throw std::runtime_error("selection cardinality");
        std::ifstream fm(argv[3],std::ios::binary);auto inverse=read_vec<float>(fm,x.n);
        if(fm.peek()!=std::char_traits<char>::eof())throw std::runtime_error("physical inverse extent");
        for(float a:inverse)if(!std::isfinite(a)||a<=0)throw std::runtime_error("physical inverse domain");
        rt(cudaSetDevice(0),"device");rt(cudaFree(nullptr),"context");
        Device<float3> nx(8*x.n),nv(8*x.n),nf(8*x.n),ax(8*x.n),av(8*x.n),af(8*x.n);
        std::vector<float3> repeated;for(unsigned r=0;r<8;++r)repeated.insert(repeated.end(),x.positions.begin(),x.positions.end());
        nx.put(repeated);nv.zero();nf.zero();ax.zero();av.zero();af.zero();
        Device<float> mass(x.n),charges(x.n);mass.put(inverse);charges.put(x.charges);
        Device<uint8_t> elements(x.n);elements.put(x.elements);
        Device<uint32_t> masks(8*256);masks.zero();
        FoliatedStreamPtrTableFfi native={},atomic={};std::vector<const float*> coord_ptrs;
        Device<BiprismOrderedEdgesFfi> ordered_edges(8);ordered_edges.zero();
        atomic.d_ordered_edges=ordered_edges.p;
        for(unsigned r=0;r<8;++r){const size_t offset=r*x.n;
            native.d_coords_xyz[r]=reinterpret_cast<float*>(nx.p+offset);native.d_vels_xyz[r]=reinterpret_cast<float*>(nv.p+offset);native.d_forces_xyz[r]=reinterpret_cast<float*>(nf.p+offset);
            atomic.d_coords_xyz[r]=reinterpret_cast<float*>(ax.p+offset);atomic.d_vels_xyz[r]=reinterpret_cast<float*>(av.p+offset);atomic.d_forces_xyz[r]=reinterpret_cast<float*>(af.p+offset);
            atomic.d_inv_masses[r]=atomic.d_true_inv_masses[r]=mass.p;atomic.d_core_shake_mask[r]=masks.p+r*256;
            coord_ptrs.push_back(atomic.d_coords_xyz[r]);
        }
        Device<FoliatedStreamPtrTableFfi> nt(1),at(1);nt.put({native});at.put({atomic});
        Device<BiprismUnitBridgeStatusFfi> bridge(8);bridge.zero();Device<uint32_t> step(1);step.put({0u});
        Device<BiprismConvergenceRecordFfi> conv(8);std::vector<BiprismConvergenceRecordFfi> records(8);
        for(unsigned r=0;r<8;++r){records[r]={};records[r].abi_magic=BIPRISM_TAIL_ABI_MAGIC;records[r].leaf_id=r;}conv.put(records);
        nhs_biprism_import_native_state_v1<<<8,256>>>(nt.p,at.p,bridge.p,step.p,x.n,7u);
        nhs_biprism_guard_unit_import_v1<<<1,256>>>(bridge.p,conv.p,step.p);rt(cudaGetLastError(),"cold bridge");rt(cudaDeviceSynchronize(),"cold bridge wait");
        for(const auto&s:bridge.get())if(s.sticky_fault||s.imported_fields!=7u)throw std::runtime_error("cold quantity import refused");
        std::vector<uint32_t> selection;for(unsigned r=0;r<8;++r)selection.insert(selection.end(),ids.begin(),ids.end());
        Device<uint32_t> selected(selection.size());selected.put(selection);Device<const float*> coords(8);coords.put(coord_ptrs);
        Device<ContactShellSimplicialStateFfi> tile(8);tile.zero();Device<BiprismBwScratchFfi> scratch(8);Device<BiprismBwStatusFfi> status(8);status.zero();
        nhs_seed_bowyer_watson_256_v1<<<8,256>>>(coords.p,charges.p,elements.p,selected.p,x.n,tile.p,scratch.p,status.p,8u,960u);
        rt(cudaGetLastError(),"AU BW");rt(cudaDeviceSynchronize(),"AU BW wait");retain_qc(out+".bw_status.bin",status.get());retain_qc(out+".seed_tiles.bin",tile.get());
        for(const auto&s:status.get())if(s.error||!s.published)throw std::runtime_error("actual AU coordinate BW refused");
        for(const auto&t:tile.get())if(!report("actual_AU_seed_topology",biprism_qc::validate_tile_topology(t,x.n,x.groups)))throw std::runtime_error("AU mesh gate");
        rt(cudaFuncSetAttribute(nhs_biprism_variational_so3g_v10,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(VariationalClusterDsmemV10)),"v10 shared");
        cudaLaunchConfig_t config={};config.gridDim=dim3(8);config.blockDim=dim3(256);config.dynamicSmemBytes=sizeof(VariationalClusterDsmemV10);
        int active=0;rt(cudaOccupancyMaxActiveClusters(&active,nhs_biprism_variational_so3g_v10,&config),"cluster capacity");if(active<1)throw std::runtime_error("actual8 cluster unavailable");
        Device<InterferometricAdjudicatorFfi> adj(8);adj.zero();Device<uint32_t> arrivals(1),epochs(1);arrivals.zero();epochs.zero();
        const float dt_au=static_cast<float>(0.0005*41341.37333517015);
        std::printf("scope=controlled_native_v10_AU dt_ps=0.0005 dt_au=%.9g initial_velocity=0 native_bath_force=0 actual_physical_inverse=1 mode=FlexibleCoreVerified_GonzalezExactEnergy\n",dt_au);
        for(unsigned epoch=1;epoch<=3;++epoch) {
            step.put({epoch});
            nhs_biprism_import_native_state_v1<<<8,256>>>(nt.p,at.p,bridge.p,step.p,x.n,7u);
            nhs_biprism_guard_unit_import_v1<<<1,256>>>(bridge.p,conv.p,step.p);
            for(unsigned r=0;r<8;++r)nhs_biprism_simplicial_tile_v1<<<1,256>>>(atomic.d_coords_xyz[r],charges.p,tile.p,r);
            rt(cudaGetLastError(),"actual AU Phase2");rt(cudaDeviceSynchronize(),"AU Phase2 completion");
            retain_qc(out+".epoch"+std::to_string(epoch)+".phase2_tiles.bin",tile.get());
            nhs_biprism_variational_so3g_v10<<<8,256,sizeof(VariationalClusterDsmemV10)>>>(at.p,charges.p,tile.p,adj.p,conv.p,arrivals.p,epochs.p,dt_au,0.1f,FlexibleCoreVerified|GonzalezExactEnergyBit);
            rt(cudaGetLastError(),"positive v10");rt(cudaDeviceSynchronize(),"positive v10 completion");
            records=conv.get();retain_qc(out+".epoch"+std::to_string(epoch)+".conv.bin",records);
            bool accepted=true;
            for(unsigned r=0;r<8;++r){const auto&c=records[r];std::printf("epoch=%u leaf=%u committed=%u fault=%u picard=%.9g dec=%.9g energy=%.9g iterations=%u/%u\n",epoch,r,c.committed_substeps,c.terminal_fault_latch,c.last_picard_residual,c.last_dec_residual,c.step_energy_drift,c.picard_iters_used,c.dec_iters_used);
                accepted&=c.terminal_fault_latch==0u&&c.epoch_cluster_fault==0u&&c.committed_substeps==4u*epoch&&std::isfinite(c.step_energy_drift)&&c.step_energy_drift<=1e-6f;}
            if(!accepted){retain_qc(out+".refused_au_coords.bin",ax.get());return 4;}
            nhs_biprism_validate_unit_export_v1<<<1,256>>>(nt.p,at.p,tile.p,conv.p,bridge.p,step.p,x.n,1u);
            nhs_biprism_commit_unit_export_v1<<<8,256>>>(nt.p,at.p,tile.p,conv.p,bridge.p,step.p,x.n,1u);
            nhs_biprism_import_native_state_v1<<<8,256>>>(nt.p,at.p,bridge.p,step.p,x.n,2u);
            nhs_biprism_guard_unit_import_v1<<<1,256>>>(bridge.p,conv.p,step.p);
            // No pending LIF in this bridge control: zero velocity delta must be a no-op.
            const auto before=nv.get();
            nhs_biprism_validate_unit_export_v1<<<1,256>>>(nt.p,at.p,tile.p,conv.p,bridge.p,step.p,x.n,2u);
            nhs_biprism_commit_unit_export_v1<<<8,256>>>(nt.p,at.p,tile.p,conv.p,bridge.p,step.p,x.n,2u);
            const auto after=nv.get();if(std::memcmp(before.data(),after.data(),before.size()*sizeof(float3)))throw std::runtime_error("zero feedback roundtripped velocity");
            nhs_biprism_import_native_state_v1<<<8,256>>>(nt.p,at.p,bridge.p,step.p,x.n,3u);
            nhs_biprism_guard_unit_import_v1<<<1,256>>>(bridge.p,conv.p,step.p);rt(cudaDeviceSynchronize(),"endpoint bridge completion");
            for(const auto&s:bridge.get())if(s.sticky_fault||s.imported_fields!=31||s.exported_fields!=3)throw std::runtime_error("bridge endpoint refused");
        }
        retain_qc(out+".native_coords.bin",nx.get());retain_qc(out+".native_velocities.bin",nv.get());retain_qc(out+".bridge.bin",bridge.get());
        std::puts("native_positive_v10_AU_steps=3 bridge_all8=1 trajectory_acceptance=0 full_AMBER_schedule=0");return 0;
    }catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
