#pragma once
// Component control for the actual accepted-main -> LIF-delivery work path.
// The request rows below are explicit fixed-step diagnostic inputs. They do
// not represent an AMBER Director run or acceptance by the mixed-energy guard.
#include "nhs_biprism_shared_step.cuh"
#include "nhs_biprism_lif_work.cuh"
#include <memory>
#include <limits>

static_assert(sizeof(FoliatedStreamPtrTableFfi)==640,"post-join pointer-table ABI");
static_assert(offsetof(FoliatedStreamPtrTableFfi,d_shared_step)==512,"shared owner ABI");
static_assert(offsetof(FoliatedStreamPtrTableFfi,physical_step_mode)==520,"diagnostic mode ABI");
static_assert(sizeof(BiprismSharedStepJournalFfi)==896,"selected-time journal ABI");
static_assert(sizeof(BiprismLifWorkRowFfi)==464,"primitive work row ABI");
static_assert(sizeof(BiprismLifWorkLedgerFfi)==288,"primitive work ledger ABI");

template<class T> static void connected_save(const std::string& path,const std::vector<T>& data){
    std::ofstream f(path,std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(T));
    if(!f)throw std::runtime_error("connected-work raw output write");
}

class ConnectedWorkControl {
    uint32_t atoms_,bodies_;
    std::string prefix_;
    static constexpr uint64_t generation_=23u;
    Device<ProtocolState> protocols_{8};
    Device<ProtocolState*> protocol_ptrs_{8};
    Device<BiprismTerminalControlsFfi> baseline_{8},terminal_{8};
    Device<BiprismSharedStepWindowFfi> time_window_{1};
    Device<BiprismSharedStepJournalFfi> time_journal_;
    Device<BiprismCollectiveControlFfi> component_control_{1};
    Device<float> hmr_mass_,voltage_{8*256};
    Device<double> atom_work_;
    Device<BiprismLifWorkBindingFfi> bindings_{8};
    Device<BiprismLifWorkLedgerFfi> ledgers_{8},journal_ledgers_;
    Device<BiprismLifWorkRowFfi> rows_{8*256},journal_rows_;
    Device<BiprismLifWorkWindowFfi> work_window_{1};
    Device<int32_t> residues_;
    Device<GpuSpikeEventCanonical96> queue_;
    Device<uint32_t> count_{1};
    SignalGridSpecFfi grid_{};
    std::unique_ptr<Device<int32_t>> hits_,coupled_,owner_,counts_;
    std::unique_ptr<Device<float>> water_,exclusion_;
    bool produced_nonzero_=false,measured_nonzero_=false;
public:
    Device<BiprismSharedStepFfi> shared{1};
    Device<BiprismLifCarryoverFfi> carry{8};

    ConnectedWorkControl(const Input& input,uint32_t bodies,const std::string& prefix)
      :atoms_(input.n),bodies_(bodies),prefix_(prefix),time_journal_(bodies),hmr_mass_(input.n),
       atom_work_(size_t(8)*input.n),journal_ledgers_(size_t(8)*bodies),
       journal_rows_(size_t(8)*256*bodies),residues_(input.n),queue_(size_t(8)*192*bodies){
        if(!bodies||bodies>500)throw std::runtime_error("connected work body extent");
        hmr_mass_.put(input.masses);residues_.put(input.residue_ids);atom_work_.zero();
        voltage_.put(std::vector<float>(8*256,3.9f));
        carry.zero();queue_.zero();count_.zero();ledgers_.zero();rows_.zero();
        journal_ledgers_.zero();journal_rows_.zero();time_journal_.zero();component_control_.zero();
        BiprismSharedStepFfi s={};s.abi_magic=BIPRISM_SHARED_STEP_MAGIC;
        s.policy=BIPRISM_SHARED_STEP_COMMON_MIN;s.generation=generation_;shared.put({s});
        BiprismSharedStepWindowFfi tw={BIPRISM_SHARED_STEP_MAGIC,bodies,0u,bodies,1u,{0u,0u,0u}};
        time_window_.put({tw});
        BiprismLifWorkWindowFfi ww={BIPRISM_LIF_WORK_MAGIC,bodies,0u,bodies,1u,{0u,0u,0u}};
        work_window_.put({ww});
        std::vector<ProtocolState*> ptrs;std::vector<BiprismLifWorkBindingFfi> bindings;
        std::vector<BiprismTerminalControlsFfi> terminal(8);
        for(uint32_t leaf=0;leaf<8;++leaf){ptrs.push_back(protocols_.p+leaf);
            bindings.push_back({hmr_mass_.p,atom_work_.p+size_t(leaf)*atoms_});
            terminal[leaf]={};terminal[leaf].abi_magic=BIPRISM_SHARED_STEP_MAGIC;
            terminal[leaf].leaf_id=leaf;terminal[leaf].terminal_step=bodies_+1u;}
        protocol_ptrs_.put(ptrs);bindings_.put(bindings);terminal_.put(terminal);
        connected_save(prefix_+".lif_work.hmr_masses.f32le",input.masses);
        std::printf("connected_work_scope=actual_main_and_LIF_work_component diagnostic_request_dt_ps=0.0005 diagnostic_control_temperature=300 full_AMBER_schedule=0 mixed_guard_acceptance=0\n");
    }

    void initialize_grid(const std::vector<float3>& positions,const std::vector<float3>& lows){
        if(positions.size()!=size_t(8)*atoms_||lows.size()!=positions.size())throw std::runtime_error("connected grid population");
        double lo[3]={INFINITY,INFINITY,INFINITY},hi[3]={-INFINITY,-INFINITY,-INFINITY};
        for(size_t i=0;i<positions.size();++i){const auto& p=positions[i];const auto& l=lows[i];
            const double a[3]={double(p.x)+double(l.x),double(p.y)+double(l.y),double(p.z)+double(l.z)};
            for(unsigned k=0;k<3;++k){
                if(!std::isfinite(a[k]))throw std::runtime_error("connected grid nonfinite position");
                lo[k]=std::min(lo[k],a[k]);hi[k]=std::max(hi[k],a[k]);}}
        grid_.inv_voxel_spacing=.125f;grid_.total_voxels=1u;
        for(unsigned k=0;k<3;++k){grid_.origin_xyz[k]=float(lo[k]-8.);
            const double n=std::ceil((hi[k]-double(grid_.origin_xyz[k]))/8.)+2.;
            if(!std::isfinite(n)||n<1.||n>10000000.)throw std::runtime_error("connected grid dimension");
            grid_.dim_xyz[k]=uint32_t(n);const uint64_t size=uint64_t(grid_.total_voxels)*grid_.dim_xyz[k];
            if(size>10000000u)throw std::runtime_error("connected grid size");grid_.total_voxels=uint32_t(size);}
        hits_.reset(new Device<int32_t>(grid_.total_voxels));coupled_.reset(new Device<int32_t>(grid_.total_voxels));
        owner_.reset(new Device<int32_t>(grid_.total_voxels));counts_.reset(new Device<int32_t>(grid_.total_voxels));
        water_.reset(new Device<float>(grid_.total_voxels));exclusion_.reset(new Device<float>(grid_.total_voxels));
        hits_->zero();coupled_->zero();counts_->zero();water_->zero();exclusion_->zero();
        owner_->put(std::vector<int32_t>(grid_.total_voxels,-1));
        connected_save(prefix_+".lif_work.grid.bin",std::vector<SignalGridSpecFfi>{grid_});
    }

    float select(uint32_t epoch,const uint32_t* step,BiprismConvergenceRecordFfi* conv){
        std::vector<ProtocolState> p(8);std::vector<BiprismTerminalControlsFfi> b(8);
        for(uint32_t leaf=0;leaf<8;++leaf){p[leaf]={};p[leaf].current_step=epoch;p[leaf].total_steps=bodies_+1u;
            p[leaf].current_temperature=300.f;p[leaf].dt=.0005f;p[leaf].steering_uv_boost=1.f;
            p[leaf].uv_target_idx=-1;p[leaf].steering_focus_residue=-1;
            p[leaf].current_phase_bits=uint32_t((float(epoch)/float(bodies_+1u))*1024.f)&1023u;
            b[leaf]={};b[leaf].abi_magic=BIPRISM_SHARED_STEP_MAGIC;b[leaf].leaf_id=leaf;
            b[leaf].terminal_step=bodies_+1u;b[leaf].captured_step=epoch;b[leaf].temperature=300.f;
            b[leaf].requested_dt_ps=p[leaf].dt;b[leaf].uv_target_idx=-1;b[leaf].phase_bits=p[leaf].current_phase_bits;}
        protocols_.put(p);baseline_.put(b);
        nhs_biprism_shared_step_select_v1<<<1,1>>>(shared.p,protocol_ptrs_.p,baseline_.p,terminal_.p,
            step,generation_,conv,time_window_.p,time_journal_.p);
        rt(cudaGetLastError(),"connected native shared select");rt(cudaDeviceSynchronize(),"connected selected step completion");
        const auto s=shared.get()[0];
        if(s.state!=BIPRISM_SHARED_STEP_SELECTED||s.sticky_fault||s.scheduled_step!=epoch
            ||s.accepted_bodies+1u!=epoch)throw std::runtime_error("connected selected owner refused");
        connected_save(prefix_+".epoch"+std::to_string(epoch)+".selected_time.128b",std::vector<BiprismSharedStepFfi>{s});
        return s.outer_dt_au;
    }

    void capture(const FoliatedStreamPtrTableFfi* au,const FoliatedStreamPtrTableFfi* native,
        const ContactShellSimplicialStateFfi* tiles,const BiprismUnitBridgeStatusFfi* bridge,
        BiprismConvergenceRecordFfi* conv,const uint32_t* step,uint32_t mode,uint32_t epoch){
        nhs_biprism_lif_work_capture_v1<<<8,256>>>(bindings_.p,tiles,au,native,carry.p,bridge,conv,
            step,&shared.p->outer_dt_au,atoms_,mode,ledgers_.p,rows_.p,work_window_.p);
        rt(cudaGetLastError(),"connected work capture");rt(cudaDeviceSynchronize(),"connected work capture completion");
        connected_save(prefix_+".epoch"+std::to_string(epoch)+".captured_work.464b",rows_.get());
        connected_save(prefix_+".epoch"+std::to_string(epoch)+".carry_before_delivery.bin",carry.get());
        for(const auto& l:ledgers_.get())if(l.state!=BIPRISM_LIF_WORK_CAPTURED||l.sticky_fault)
            throw std::runtime_error("connected work capture refused");
    }

    void measure(const FoliatedStreamPtrTableFfi* au,const FoliatedStreamPtrTableFfi* native,
        const BiprismUnitBridgeStatusFfi* bridge,BiprismConvergenceRecordFfi* conv,const uint32_t* step,uint32_t epoch){
        nhs_biprism_lif_work_measure_v1<<<8,256>>>(au,native,carry.p,bridge,conv,step,ledgers_.p,rows_.p);
        nhs_biprism_lif_work_publish_v1<<<1,256>>>(bindings_.p,conv,ledgers_.p,rows_.p,work_window_.p,journal_ledgers_.p,journal_rows_.p);
        rt(cudaGetLastError(),"connected work measure/publish");rt(cudaDeviceSynchronize(),"connected work publication completion");
        const auto l=ledgers_.get();
        connected_save(prefix_+".epoch"+std::to_string(epoch)+".published_work.288b",l);
        connected_save(prefix_+".epoch"+std::to_string(epoch)+".published_work.464b",rows_.get());
        connected_save(prefix_+".epoch"+std::to_string(epoch)+".native_atom_work.f64le",atom_work_.get());
        for(uint32_t leaf=0;leaf<8;++leaf){const auto& v=l[leaf];
            if(v.state!=BIPRISM_LIF_WORK_SEALED||v.sticky_fault||v.leaf_id!=leaf||v.completed_steps!=epoch
                ||v.scheduled_step!=epoch||v.committed_substeps!=4u*epoch)throw std::runtime_error("connected work publication refused");
            for(unsigned k=0;k<17;++k)if(!std::isfinite(v.totals[k]))throw std::runtime_error("connected nonfinite work");
            for(unsigned k=0;k<3;++k)if(!std::isfinite(v.realized_lab_impulse[k])||std::abs(v.realized_lab_impulse[k])>1.e-7)
                throw std::runtime_error("connected physical momentum gate");
            measured_nonzero_=measured_nonzero_||v.totals[1]!=0.||v.totals[7]!=0.;}
    }

    void observe(const FoliatedStreamPtrTableFfi* au,ContactShellSimplicialStateFfi* tile,
        BiprismConvergenceRecordFfi* conv,const uint32_t* step,uint32_t epoch){
        if(!hits_)throw std::runtime_error("connected grid absent");
        if(epoch==1u){
            // Corrupt isolated diagnostic inputs, retain the refusals, and
            // verify all observation state stays unchanged on every leaf.
            FoliatedStreamPtrTableFfi table{};
            rt(cudaMemcpy(&table,au,sizeof(table),cudaMemcpyDeviceToHost),"observation precision table");
            std::vector<BiprismPotentialOwnerFfi> original_owners(8);
            std::vector<ContactShellSimplicialStateFfi> original_tiles(8);
            rt(cudaMemcpy(original_owners.data(),table.d_potential_owners,8*sizeof(original_owners[0]),cudaMemcpyDeviceToHost),"observation potential owners");
            rt(cudaMemcpy(original_tiles.data(),tile,8*sizeof(original_tiles[0]),cudaMemcpyDeviceToHost),"observation tiles");
            const auto original_carry=carry.get();
            const auto hit=hits_->get(),cpl=coupled_->get(),owner=owner_->get(),counts=counts_->get();
            const auto water=water_->get(),exclusion=exclusion_->get(),voltage=voltage_.get();
            const auto queue=queue_.get();
            const auto count=count_.get();
            const auto equal=[](const auto& a,const auto& b){return a.size()==b.size()
                &&std::memcmp(a.data(),b.data(),a.size()*sizeof(a[0]))==0;};
            for(uint32_t kind=0;kind<2u;++kind)for(uint32_t bad=0;bad<8u;++bad){
                auto owners=original_owners;auto tiles=original_tiles;
                if(kind==0u)++owners[bad].projection_step;
                else{uint32_t bits;std::memcpy(&bits,&tiles[bad].phi_dec[191],4);bits^=1u;
                    std::memcpy(&tiles[bad].phi_dec[191],&bits,4);}
                rt(cudaMemcpy(table.d_potential_owners,owners.data(),8*sizeof(owners[0]),cudaMemcpyHostToDevice),"inject observation owner");
                rt(cudaMemcpy(tile,tiles.data(),8*sizeof(tiles[0]),cudaMemcpyHostToDevice),"inject observation projection");
                carry.put(original_carry);
                nhs_voronoi_dual_lif_and_4grid_bypass_v2<<<8,256>>>(au,tile,conv,step,residues_.p,voltage_.p,
                    hits_->p,coupled_->p,owner_->p,counts_->p,water_->p,exclusion_->p,queue_.p,count_.p,grid_,
                    uint32_t(queue_.n),1.f,1.25f,.04f,carry.p,0u);
                rt(cudaGetLastError(),"observation precision refusal launch");rt(cudaDeviceSynchronize(),"observation precision refusal completion");
                auto expected=original_carry;for(auto& c:expected)c.fault|=BIPRISM_CARRY_BAD_INPUT;
                const auto refused=carry.get();
                connected_save(prefix_+".observation_precision_refusal_"+std::to_string(kind)+"_leaf"+std::to_string(bad)+".carry.bin",refused);
                const bool unchanged=equal(hit,hits_->get())&&equal(cpl,coupled_->get())&&equal(owner,owner_->get())
                    &&equal(counts,counts_->get())&&equal(water,water_->get())&&equal(exclusion,exclusion_->get())
                    &&equal(voltage,voltage_.get())&&equal(queue,queue_.get())&&equal(count,count_.get());
                const bool collective=equal(expected,refused);
                std::printf("observation_precision_refusal kind=%u leaf=%u all8_withheld=%u all_outputs_bitwise=%u\n",
                    kind,bad,unsigned(collective),unsigned(unchanged));
                if(!unchanged||!collective)throw std::runtime_error("potential observation refusal mutated outputs or missed a leaf");
            }
            rt(cudaMemcpy(table.d_potential_owners,original_owners.data(),8*sizeof(original_owners[0]),cudaMemcpyHostToDevice),"restore test-owned potential inputs");
            rt(cudaMemcpy(tile,original_tiles.data(),8*sizeof(original_tiles[0]),cudaMemcpyHostToDevice),"restore test-owned tile inputs");
            carry.put(original_carry);
        }
        nhs_voronoi_dual_lif_and_4grid_bypass_v2<<<8,256>>>(au,tile,conv,step,residues_.p,voltage_.p,
            hits_->p,coupled_->p,owner_->p,counts_->p,water_->p,exclusion_->p,queue_.p,count_.p,grid_,
            uint32_t(queue_.n),1.f,1.25f,.04f,carry.p,0u);
        rt(cudaGetLastError(),"connected actual LIF producer");rt(cudaDeviceSynchronize(),"connected observation completion");
        const auto c=carry.get();const auto n=count_.get()[0];
        connected_save(prefix_+".epoch"+std::to_string(epoch)+".produced_carry.bin",c);
        for(uint32_t leaf=0;leaf<8;++leaf){if(c[leaf].fault||c[leaf].pending!=1u||c[leaf].produced_step!=epoch
            ||c[leaf].produced_committed_substeps!=4u*epoch)throw std::runtime_error("connected actual producer refused");
            double sum[3]={};for(const auto& f:c[leaf].force){const double a[3]={f.x,f.y,f.z};
                for(unsigned k=0;k<3;++k){if(!std::isfinite(a[k]))throw std::runtime_error("connected nonfinite force");
                    sum[k]+=a[k];produced_nonzero_=produced_nonzero_||a[k]!=0.;}}
            if(sum[0]!=0.||sum[1]!=0.||sum[2]!=0.)throw std::runtime_error("connected unbalanced produced force");}
        if(!n||n>queue_.n)throw std::runtime_error("connected observation empty or overflowing");
    }

    void accept_component(const std::vector<BiprismConvergenceRecordFfi>& records,
        BiprismConvergenceRecordFfi* conv,const uint32_t* step,uint32_t epoch){
        // This descriptor records only the already checked native main+work
        // component. No native AMBER clock or mixed-energy success is asserted.
        if(records.size()!=8u)throw std::runtime_error("connected convergence extent");
        for(uint32_t leaf=0;leaf<8;++leaf){const auto& c=records[leaf];
            if(c.abi_magic!=BIPRISM_TAIL_ABI_MAGIC||c.leaf_id!=leaf||c.terminal_fault_latch
                ||c.epoch_leaf_fault||c.epoch_cluster_fault||c.committed_substeps!=4u*epoch)
                throw std::runtime_error("connected component admission refused");}
        BiprismCollectiveControlFfi c={};c.abi_magic=BIPRISM_COLLECTIVE_MAGIC;
        c.last_completed_step=epoch;c.successful_bodies=epoch;c.last_committed_substeps=4u*epoch;
        component_control_.put({c});
        // Retain the actual acceptance inputs and outputs before any refusal
        // throws. In particular, reason64 compares all128 selected bytes with
        // the journal; the earlier selection snapshot alone cannot prove that
        // those bytes stayed unchanged through main, delivery and observation.
        const auto capture_accept=[&](const char* phase){
            const auto state=shared.get();
            std::vector<BiprismConvergenceRecordFfi> convergence(8);
            std::vector<uint32_t> scheduled(1);
            rt(cudaMemcpy(convergence.data(),conv,convergence.size()*sizeof(convergence[0]),cudaMemcpyDeviceToHost),"connected acceptance convergence retention");
            rt(cudaMemcpy(scheduled.data(),step,sizeof(uint32_t),cudaMemcpyDeviceToHost),"connected acceptance clock retention");
            const std::string path=prefix_+".epoch"+std::to_string(epoch)+".time_accept."+phase;
            connected_save(path+".shared.128b",state);
            connected_save(path+".journal.896b",time_journal_.get());
            connected_save(path+".control.64b",component_control_.get());
            connected_save(path+".window.32b",time_window_.get());
            connected_save(path+".scheduled.u32le",scheduled);
            connected_save(path+".convergence.128b",convergence);
            std::printf("connected_time_accept epoch=%u phase=%s state=%u sticky_fault=%u scheduled=%u accepted_bodies=%u last_accepted_step=%u generation=%llu\n",
                epoch,phase,state[0].state,state[0].sticky_fault,scheduled[0],state[0].accepted_bodies,
                state[0].last_accepted_step,static_cast<unsigned long long>(state[0].generation));
            return state[0];
        };
        (void)capture_accept("before");
        nhs_biprism_shared_step_accept_v1<<<1,1>>>(shared.p,component_control_.p,conv,step,generation_,time_window_.p,time_journal_.p);
        rt(cudaGetLastError(),"connected native time accept");rt(cudaDeviceSynchronize(),"connected accepted time completion");
        const auto s=capture_accept("after");if(s.state!=BIPRISM_SHARED_STEP_ACCEPTED||s.sticky_fault
            ||s.last_accepted_step!=epoch||s.accepted_bodies!=epoch)
            throw std::runtime_error("connected time acceptance refused: sticky_fault="+std::to_string(s.sticky_fault));
    }

    void retain(const std::vector<uint32_t>& selected){
        connected_save(prefix_+".lif_work.rows.464b",journal_rows_.get());
        connected_save(prefix_+".lif_work.ledgers.288b",journal_ledgers_.get());
        connected_save(prefix_+".lif_work.window.32b",work_window_.get());
        connected_save(prefix_+".lif_work.selected_ids.u32le",selected);
        connected_save(prefix_+".shared_time.records.896b",time_journal_.get());
        connected_save(prefix_+".shared_time.window.32b",time_window_.get());
        connected_save(prefix_+".shared_time.terminal.48b",terminal_.get());
        connected_save(prefix_+".shared_time.final.128b",shared.get());
        connected_save(prefix_+".component_control.64b",component_control_.get());
        connected_save(prefix_+".lif_work.atom_work.kcal.f64le",atom_work_.get());
        connected_save(prefix_+".lif_work.carry.final.bin",carry.get());
        connected_save(prefix_+".lif_work.queue.96b",queue_.get());
        connected_save(prefix_+".lif_work.queue_count.u32le",count_.get());
        if(!produced_nonzero_||!measured_nonzero_)throw std::runtime_error("connected work nonzero coverage absent");
        std::printf("connected_actual_LIF_work=1 bodies=%u full_8x256_rows=1 actual_shared_selector_and_accept=1 nonzero_work=1 mixed_guard_acceptance=0 full_AMBER_trajectory=0\n",bodies_);
    }
};
