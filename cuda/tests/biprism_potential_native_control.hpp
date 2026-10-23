#pragma once
// Test-owned storage for the same persistent potential descriptor used by the
// production runtime. Only the actual all-leaf device publishers populate it.
// Neither bank aliases tile projections, numerical working memory or the other
// bank. Raw captures preserve prepared-versus-accepted identity on refusals.
#include "nhs_biprism_potential_precision.cuh"
#include "nhs_biprism_phase2_work.cuh"

struct CoupledPotentialOwners {
    Device<double> accepted{8u*256u},prepared{8u*256u};
    Device<BiprismPotentialOwnerFfi> owners{8};

    CoupledPotentialOwners(ContactShellSimplicialStateFfi* tiles,
        BiprismCoordinateOwnerFfi* coordinates,uint64_t generation) {
        accepted.zero();prepared.zero();
        std::vector<BiprismPotentialOwnerFfi> descriptors(8);
        for(uint32_t leaf=0;leaf<8u;++leaf){
            auto& o=descriptors[leaf];o={};
            o.abi_magic=BP_POTENTIAL_MAGIC;o.version=BP_POTENTIAL_VERSION;
            o.leaf_id=leaf;o.state=BP_POTENTIAL_COLD;
            o.allocation_generation=generation;o.topology_generation=1u;
            o.tile_address=reinterpret_cast<uint64_t>(tiles+leaf);
            o.coordinate_owner_address=reinterpret_cast<uint64_t>(coordinates+leaf);
            o.accepted_phi=accepted.p+256u*leaf;o.prepared_phi=prepared.p+256u*leaf;
        }
        owners.put(descriptors);
    }
    void retain(const std::string& prefix) const {
        retain_qc(prefix+".potential.owners.128b",owners.get());
        retain_qc(prefix+".potential.accepted.f64le",accepted.get());
        retain_qc(prefix+".potential.prepared.f64le",prepared.get());
    }
};

struct CoupledSpinorOwners {
    Device<BiprismSpinorOwnerFfi> owners{8};
    CoupledSpinorOwners(ContactShellSimplicialStateFfi* tiles,BiprismCoordinateOwnerFfi* coords,uint64_t generation) {
        std::vector<BiprismSpinorOwnerFfi> d(8);
        for(uint32_t l=0;l<8;++l){auto& o=d[l];o={};o.magic=BP_SPINOR_MAGIC;o.version=BP_SPINOR_VERSION;o.leaf=l;
            o.allocation_generation=generation;o.topology_generation=1u;o.tile_address=uint64_t(tiles+l);o.coordinate_owner_address=uint64_t(coords+l);}
        owners.put(d);
    }
    void retain(const std::string& prefix)const{retain_qc(prefix+".spinor.owners.bps1",owners.get());}
};

// Exercise the actual cold/refresh publications, including fixed-state work,
// in every coupled diagnostic. The mode4 fallback is explicitly a controlled
// work descriptor derived from the preceding native committed count. It never
// publishes a production accepted clock or supplies main's physical timestep.
struct CoupledPhase2Control {
    Device<ContactShellSimplicialStateFfi> snapshots{8};
    Device<BiprismPhase2VertexWorkFfi> vertices{8u*256u},vertex_journal;
    Device<BiprismPhase2EdgeWorkFfi> edges{8u*TILE_MAX_EDGES},edge_journal;
    Device<BiprismPhase2WorkLedgerFfi> ledgers{8},ledger_journal;
    Device<BiprismPhase2WorkWindowFfi> window{1};
    Device<BiprismSharedStepFfi> diagnostic_shared{1};
    const BiprismSharedStepFfi* selected=nullptr;

    explicit CoupledPhase2Control(uint32_t bodies):vertex_journal(size_t(bodies)*8u*256u),
        edge_journal(size_t(bodies)*8u*TILE_MAX_EDGES),ledger_journal(size_t(bodies)*8u) {
        snapshots.zero();vertices.zero();edges.zero();ledgers.zero();
        vertex_journal.zero();edge_journal.zero();ledger_journal.zero();diagnostic_shared.zero();
        BiprismPhase2WorkWindowFfi w={};w.magic=BIPRISM_PHASE2_WORK_MAGIC;
        w.capacity=w.requested_bodies=bodies;w.first_scheduled_step=1u;
        w.generation=23u;w.preparation_generation=1u;window.put({w});
    }
    void initialize(ContactShellSimplicialStateFfi* tiles,const FoliatedStreamPtrTableFfi* streams,
        const float* charges,uint32_t atoms,BiprismConvergenceRecordFfi* conv,const uint32_t* step) {
        nhs_biprism_phase2_work_initialize_v1<<<8,256>>>(tiles,streams,charges,atoms,conv,vertices.p,ledgers.p);
        nhs_biprism_phase2_work_initialize_publish_v1<<<1,256>>>(tiles,streams,conv,vertices.p,ledgers.p,step);
        rt(cudaGetLastError(),"potential cold solve/publication");rt(cudaDeviceSynchronize(),"potential cold publication completion");
        for(const auto& l:ledgers.get())if(l.sticky_fault||l.state!=4u)
            throw std::runtime_error("potential cold all-leaf publication refused");
    }
    void capture(uint32_t epoch,float dt,const BiprismSharedStepFfi* actual_selected,
        const ContactShellSimplicialStateFfi* tiles,const FoliatedStreamPtrTableFfi* streams,
        const float* charges,const BiprismUnitBridgeStatusFfi* bridge,
        BiprismConvergenceRecordFfi* conv,uint32_t atoms,float kappa) {
        selected=actual_selected;
        if(!selected){
            std::vector<BiprismConvergenceRecordFfi> actual(8);
            rt(cudaMemcpy(actual.data(),conv,actual.size()*sizeof(actual[0]),cudaMemcpyDeviceToHost),"diagnostic work count readback");
            for(uint32_t leaf=0;leaf<8u;++leaf)if(actual[leaf].leaf_id!=leaf||actual[leaf].terminal_fault_latch
                ||actual[leaf].epoch_cluster_fault||actual[leaf].committed_substeps!=4u*(epoch-1u))
                throw std::runtime_error("diagnostic work metadata requires actual preceding all-leaf commit");
            BiprismSharedStepFfi s={};s.abi_magic=BIPRISM_SHARED_STEP_MAGIC;s.state=BIPRISM_SHARED_STEP_SELECTED;
            s.scheduled_step=epoch;s.last_accepted_step=epoch-1u;s.accepted_bodies=epoch-1u;
            s.policy=BIPRISM_SHARED_STEP_COMMON_MIN;s.native_dt_ps=.0005f;s.outer_dt_au=dt;
            s.minimum_request_ps=s.maximum_request_ps=.0005f;s.generation=23u;
            for(uint32_t leaf=0;leaf<8u;++leaf){s.requested_dt_ps[leaf]=.0005f;s.protocol_control_step[leaf]=epoch;}
            diagnostic_shared.put({s});selected=diagnostic_shared.p;
        }
        nhs_biprism_phase2_work_capture_v1<<<8,256>>>(tiles,streams,charges,bridge,conv,selected,
            atoms,kappa,snapshots.p,vertices.p,edges.p,ledgers.p,window.p);
        rt(cudaGetLastError(),"actual Phase2 work capture");rt(cudaDeviceSynchronize(),"Phase2 work capture completion");
        for(const auto& l:ledgers.get())if(l.sticky_fault||l.state!=1u)
            throw std::runtime_error("actual Phase2 work capture refused");
    }
    void publish(ContactShellSimplicialStateFfi* tiles,const FoliatedStreamPtrTableFfi* streams,
        const float* charges,BiprismConvergenceRecordFfi* conv) {
        nhs_biprism_phase2_work_measure_v1<<<8,256>>>(tiles,streams,charges,snapshots.p,conv,selected,
            vertices.p,edges.p,ledgers.p);
        nhs_biprism_phase2_work_publish_v1<<<1,256>>>(conv,selected,vertices.p,edges.p,ledgers.p,window.p,
            vertex_journal.p,edge_journal.p,ledger_journal.p,tiles,streams);
        rt(cudaGetLastError(),"actual Phase2 work measure/publication");rt(cudaDeviceSynchronize(),"Phase2 work publication completion");
        for(const auto& l:ledgers.get())if(l.sticky_fault||l.state!=3u)
            throw std::runtime_error("actual Phase2 prepared publication refused");
    }
    void retain(const std::string& prefix) const {
        retain_qc(prefix+".phase2_work.vertices.bin",vertices.get());
        retain_qc(prefix+".phase2_work.edges.bin",edges.get());
        retain_qc(prefix+".phase2_work.ledgers.bin",ledgers.get());
        retain_qc(prefix+".phase2_work.snapshots.bin",snapshots.get());
        retain_qc(prefix+".phase2_work.window.bin",window.get());
    }
};
