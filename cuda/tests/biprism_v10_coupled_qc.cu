// Native quantity/accepted-state diagnostic using the actual KRAS source input.
// Compile against the active kernel headers; staged kernel replacements are not used.
// Zero initial velocities/bath forces are explicit controls, not an AMBER run.
#include <cstdio>
#if defined(BIPRISM_V10_DIAGNOSTIC_INITIAL_GUESS)
template<class Shared> __device__ __noinline__ void bp_v10_diagnostic_initial_guess(Shared*);
#endif
#define main biprism_retained_native_qc_main
#include "biprism_v10_native_qc.cu"
#undef main
#include "nhs_biprism_unit_bridge.cuh"
#include "biprism_v10_reciprocal_parity.cuh"
#include "biprism_connected_work_control.hpp"
#include "biprism_ordered_edges_native_control.hpp"
#include <sstream>
#include <map>
#include <set>
#if defined(BIPRISM_V10_DIAGNOSTIC_INITIAL_GUESS)
#include "biprism_v10_initial_guess_control.hpp"
#endif

template<class T> void retain_qc(const std::string& path,const std::vector<T>& data) {
    std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(T));
    if(!f)throw std::runtime_error("native unit evidence write");
}
#include "biprism_potential_native_control.hpp"

// These allocations are the diagnostic's actual persistent coordinate owners.
// Only cold construction initializes lows to zero. Subsequent import/export,
// snapshot and observation paths must preserve the full represented state.
struct CoupledCoordinateOwners {
    uint32_t atoms;
    Device<float3> native_low,native_start_hi,native_start_lo,native_ref_hi,native_ref_lo;
    Device<float3> atomic_low,atomic_start_hi,atomic_start_lo,atomic_ref_hi,atomic_ref_lo;
    Device<BiprismCoordinateOwnerFfi> native_owner{8},atomic_owner{8};
    CoupledCoordinateOwners(uint32_t n,const std::vector<float3>& initial,
        Device<float3>& native_high,Device<float3>& atomic_high):atoms(n),
        native_low(size_t(8)*n),native_start_hi(size_t(8)*n),native_start_lo(size_t(8)*n),
        native_ref_hi(size_t(8)*n),native_ref_lo(size_t(8)*n),atomic_low(size_t(8)*n),
        atomic_start_hi(size_t(8)*n),atomic_start_lo(size_t(8)*n),atomic_ref_hi(size_t(8)*n),atomic_ref_lo(size_t(8)*n) {
        static_assert(sizeof(float3)==3*sizeof(float),"dense full-N coordinate planes");
        native_low.zero();native_start_hi.put(initial);native_start_lo.zero();native_ref_hi.put(initial);native_ref_lo.zero();
        atomic_low.zero();atomic_start_hi.zero();atomic_start_lo.zero();atomic_ref_hi.zero();atomic_ref_lo.zero();
        std::vector<BiprismCoordinateOwnerFfi> native(8),atomic(8);
        for(uint32_t leaf=0;leaf<8u;++leaf){
            auto make=[&](uint32_t frame,float3* high,float3* low,float3* sh,float3* sl,float3* rh,float3* rl){
                BiprismCoordinateOwnerFfi o={};o.abi_magic=BP_COORDINATE_MAGIC;o.abi_version=BP_COORDINATE_VERSION;
                o.leaf_id=leaf;o.atom_count=n;o.generation=23u;o.unit_frame=frame;o.initialized_mask=BP_COORD_ALL_INITIALIZED;
                const size_t offset=size_t(leaf)*n;
                o.current_hi=reinterpret_cast<float*>(high+offset);o.current_lo=reinterpret_cast<float*>(low+offset);
                o.step_start_hi=reinterpret_cast<float*>(sh+offset);o.step_start_lo=reinterpret_cast<float*>(sl+offset);
                o.reference_hi=reinterpret_cast<float*>(rh+offset);o.reference_lo=reinterpret_cast<float*>(rl+offset);
                return o;
            };
            native[leaf]=make(BP_COORD_NATIVE_ANGSTROM,native_high.p,native_low.p,native_start_hi.p,native_start_lo.p,native_ref_hi.p,native_ref_lo.p);
            atomic[leaf]=make(BP_COORD_ATOMIC_BOHR,atomic_high.p,atomic_low.p,atomic_start_hi.p,atomic_start_lo.p,atomic_ref_hi.p,atomic_ref_lo.p);
        }
        native_owner.put(native);atomic_owner.put(atomic);
    }
    void snapshot_native_start(const Device<float3>& current) {
        rt(cudaMemcpy(native_start_hi.p,current.p,current.n*sizeof(float3),cudaMemcpyDeviceToDevice),"native start high snapshot");
        rt(cudaMemcpy(native_start_lo.p,native_low.p,native_low.n*sizeof(float3),cudaMemcpyDeviceToDevice),"native start low snapshot");
    }
    std::vector<float3> sidecar_snapshot() const {
        std::vector<float3> out;
        const Device<float3>* planes[]={&native_low,&native_start_hi,&native_start_lo,&native_ref_hi,&native_ref_lo,
            &atomic_low,&atomic_start_hi,&atomic_start_lo,&atomic_ref_hi,&atomic_ref_lo};
        for(const auto* plane:planes){const auto values=plane->get();out.insert(out.end(),values.begin(),values.end());}
        return out;
    }
    void retain(const std::string& prefix) const {
        retain_qc(prefix+".native_coords_low.bin",native_low.get());
        retain_qc(prefix+".native_start_high.bin",native_start_hi.get());retain_qc(prefix+".native_start_low.bin",native_start_lo.get());
        retain_qc(prefix+".native_reference_high.bin",native_ref_hi.get());retain_qc(prefix+".native_reference_low.bin",native_ref_lo.get());
        retain_qc(prefix+".au_coords_low.bin",atomic_low.get());
        retain_qc(prefix+".au_start_high.bin",atomic_start_hi.get());retain_qc(prefix+".au_start_low.bin",atomic_start_lo.get());
        retain_qc(prefix+".au_reference_high.bin",atomic_ref_hi.get());retain_qc(prefix+".au_reference_low.bin",atomic_ref_lo.get());
        retain_qc(prefix+".native_coordinate_owners.96b",native_owner.get());retain_qc(prefix+".au_coordinate_owners.96b",atomic_owner.get());
    }
};

// Optional asymmetric-ring diagnostic. The default three-step preparation and
// acceptance remain unchanged. The controls use raw represented spinors, with
// no normalization and no change to the common physical atom/edge map.
float asymmetric_float_bits(uint32_t bits) {
    float value;std::memcpy(&value,&bits,sizeof(value));return value;
}
long double asymmetric_population(const ContactShellSimplicialStateFfi& t,unsigned u) {
    const auto a=t.psi_vertex[u][0],b=t.psi_vertex[u][1];
    return ((long double)a.re*a.re+(long double)a.im*a.im)
         - ((long double)b.re*b.re+(long double)b.im*b.im);
}
double asymmetric_represented_population(const ContactShellSimplicialStateFfi& t,unsigned u) {
    // Independent host realization of the production FP64 FMA population from
    // the actual exported FP32 spinor; no production population helper is called.
    const auto a=t.psi_vertex[u][0],b=t.psi_vertex[u][1];
    return std::fma(double(a.re),double(a.re),double(a.im)*double(a.im))
         - std::fma(double(b.re),double(b.re),double(b.im)*double(b.im));
}
void initialize_asymmetric_spinors(std::vector<ContactShellSimplicialStateFfi>& tiles) {
    const float a=asymmetric_float_bits(0x3f3504f3u);
    const float c=asymmetric_float_bits(0x3f19999au);
    const float d=asymmetric_float_bits(0x3f4ccccdu);
    const cpx2_f32 pattern[8][2]={
        {{1.f,0.f},{0.f,0.f}}, {{0.f,0.f},{1.f,0.f}},
        {{a,0.f},{0.f,a}}, {{a,0.f},{0.f,-a}},
        {{c,0.f},{d,0.f}}, {{d,0.f},{0.f,c}},
        {{0.f,c},{d,0.f}}, {{d,0.f},{0.f,-c}}
    };
    if(tiles.size()!=8u)throw std::runtime_error("asymmetric control requires all eight leaves");
    for(unsigned leaf=0;leaf<8u;++leaf){
        long double max_norm=0.;
        for(unsigned u=0;u<TILE_CORE_VERTICES;++u){
            tiles[leaf].psi_vertex[u][0]=pattern[leaf][0];
            tiles[leaf].psi_vertex[u][1]=pattern[leaf][1];
            const auto p=pattern[leaf][0],q=pattern[leaf][1];
            const long double norm=(long double)p.re*p.re+(long double)p.im*p.im
                                  +(long double)q.re*q.re+(long double)q.im*q.im;
            max_norm=std::max(max_norm,std::abs(norm-1.L));
        }
        std::printf("asymmetric_initial_spinor leaf=%u psi=(%a,%a,%a,%a) raw_population=%.21Lg norm_defect=%.21Lg\n",
            leaf,double(pattern[leaf][0].re),double(pattern[leaf][0].im),
            double(pattern[leaf][1].re),double(pattern[leaf][1].im),
            asymmetric_population(tiles[leaf],0u),max_norm);
        if(max_norm>1.0e-7L)throw std::runtime_error("asymmetric input exceeds the unchanged norm gate");
    }
}
bool check_asymmetric_endpoint_ring(
    const std::vector<ContactShellSimplicialStateFfi>& tiles,
    const std::vector<FoliatedCouplingLeafFfi>& banks,const std::vector<BiprismSpinorOwnerFfi>& spinors,float kappa,unsigned epoch) {
    if(tiles.size()!=8u||banks.size()!=8u)return false;
    bool pass=true;long double global_delta=0.,global_work=0.,global_transfer=0.;
    long double max_transfer=0.,max_numerator=0.,max_population_error=0.;
    for(unsigned leaf=0;leaf<8u;++leaf){
        const unsigned prev=(leaf+7u)&7u,next=(leaf+1u)&7u;
        long double delta=0.,attributed=0.,work=0.,transfer=0.;bool identity=true;
        for(unsigned u=0;u<TILE_CORE_VERTICES;++u){
            const auto& p=banks[prev];const auto& b=banks[leaf];const auto& n=banks[next];
            const auto psi=spinors[leaf].accepted[u];const double represented=bp_ed_norm2(psi.s0)-bp_ed_norm2(psi.s1);
            identity=identity&&std::isfinite(represented)
                &&std::memcmp(&represented,&b.trial_population[u],sizeof(double))==0
                &&std::memcmp(&b.energy_weight[u],&tiles[leaf].star0_vol[u],sizeof(float))==0;
            max_population_error=std::max(max_population_error,
                std::abs((static_cast<long double>(psi.s0.re)*psi.s0.re+static_cast<long double>(psi.s0.im)*psi.s0.im-static_cast<long double>(psi.s1.re)*psi.s1.re-static_cast<long double>(psi.s1.im)*psi.s1.im)-(long double)b.trial_population[u]));
            const uint32_t atom=UNPACK_GLOBAL_ATOM_ID(tiles[leaf].global_atom_idx[u]);
            identity=identity&&atom==UNPACK_GLOBAL_ATOM_ID(tiles[prev].global_atom_idx[u])
                &&atom==UNPACK_GLOBAL_ATOM_ID(tiles[next].global_atom_idx[u]);
            const long double w=b.energy_weight[u],wp=p.energy_weight[u],wn=n.energy_weight[u];
            const long double sp0=p.start_population[u],sp1=p.trial_population[u];
            const long double si0=b.start_population[u],si1=b.trial_population[u];
            const long double sn0=n.start_population[u],sn1=n.trial_population[u];
            const long double operands[]={w,wp,wn,sp0,sp1,si0,si1,sn0,sn1};
            for(auto value:operands)identity=identity&&std::isfinite(value);
            identity=identity&&w>0.L&&wp>0.L&&wn>0.L;
            const long double jp=.5L*(long double)kappa*(wp+w),jn=.5L*(long double)kappa*(w+wn);
            const long double sp=.5L*(sp0+sp1),si=.5L*(si0+si1),sn=.5L*(sn0+sn1);
            const long double dp=sp1-sp0,di=si1-si0,dn=sn1-sn0;
            // Direct endpoint products are independent of the device's
            // endpoint-mean chain-expression implementation.
            delta-=jn*(si1*sn1-si0*sn0);
            attributed-=.5L*(jp*(sp1*si1-sp0*si0)+jn*(si1*sn1-si0*sn0));
            const long double numerator=-(jp*sp+jn*sn);
            work+=numerator*di;
            transfer+=.5L*(jp*(sp*di-si*dp)+jn*(sn*di-si*dn));
            max_numerator=std::max(max_numerator,std::abs(numerator));
        }
        const auto& f=banks[leaf].footer;
        const long double measured[]={f.owned_pair_energy_delta,f.attributed_coupling_energy_delta,
                                      f.coupling_discrete_work,f.physical_transfer_in};
        const long double expected[]={delta,attributed,work,transfer};
        long double error=0.;
        for(unsigned j=0;j<4u;++j){
            identity=identity&&std::isfinite(measured[j])&&std::isfinite(expected[j]);
            error=std::max(error,std::abs(measured[j]-expected[j]));
        }
        const long double local_chain=attributed-transfer-work;
        bool zero_control=true;
        if(kappa==0.f){
            for(auto value:measured)zero_control=zero_control&&value==0.L;
            for(auto value:expected)zero_control=zero_control&&value==0.L;
        }
        const bool leaf_pass=identity&&zero_control&&std::isfinite(error)
            &&error<=1.0e-6L&&std::abs(local_chain)<=1.0e-6L;
        std::printf("asymmetric_ring_endpoint epoch=%u leaf=%u kappa=%.9g population_weight_identity=%u delta=%.21Lg attributed=%.21Lg work=%.21Lg transfer=%.21Lg max_footer_error=%.21Lg local_chain=%.21Lg pass=%u\n",
            epoch,leaf,double(kappa),unsigned(identity),delta,attributed,work,transfer,error,local_chain,unsigned(leaf_pass));
        pass=pass&&leaf_pass;global_delta+=delta;global_work+=work;global_transfer+=transfer;
        max_transfer=std::max(max_transfer,std::abs(transfer));
    }
    pass=pass&&std::isfinite(global_transfer)&&std::abs(global_transfer)<=1.0e-6L
        &&std::isfinite(global_delta-global_work)&&std::abs(global_delta-global_work)<=1.0e-6L;
    std::printf("asymmetric_ring_global epoch=%u kappa=%.9g raw_population_error=%.21Lg max_generator_numerator=%.21Lg max_abs_leaf_transfer=%.21Lg transfer_sum=%.21Lg chain=%.21Lg pass=%u scope=actual_last_substep_endpoint_population_ledger\n",
        epoch,double(kappa),max_population_error,max_numerator,max_transfer,global_transfer,
        global_delta-global_work,unsigned(pass));
    return pass;
}
// This diagnostic fixes the admitted combinatorial domain, not its changing
// geometry, star0 or Phase2 weights. Compare original incidence in its actual
// stored order, including physical vertex identities and tetrahedron adjacency.
static bool fixed_960_domain_matches(const ContactShellSimplicialStateFfi& seed,
                                    const ContactShellSimplicialStateFfi& now) {
    const unsigned nt=seed.num_active_tets,ne=seed.num_active_edges;
    return nt==960u&&ne>0u&&ne<=TILE_MAX_EDGES
        &&now.num_active_tets==nt&&now.num_active_edges==ne
        &&std::memcmp(seed.global_atom_idx,now.global_atom_idx,sizeof(seed.global_atom_idx))==0
        &&std::memcmp(seed.tet_v,now.tet_v,nt*sizeof(seed.tet_v[0]))==0
        &&std::memcmp(seed.tet_edges,now.tet_edges,nt*sizeof(seed.tet_edges[0]))==0
        &&std::memcmp(seed.tet_opp_tet,now.tet_opp_tet,nt*sizeof(seed.tet_opp_tet[0]))==0
        &&std::memcmp(seed.edge_u8,now.edge_u8,ne*sizeof(seed.edge_u8[0]))==0;
}
// Exercise the compiled trace region with every other byte as a canary.
// No spinor, operator, visited mask or accepted endpoint may be overwritten.
static bool nuclear_history_lifetime_control() {
    auto state=std::unique_ptr<BiprismV10NumericalScratchFfi>(new BiprismV10NumericalScratchFfi);
    std::memset(state.get(),0x5a,sizeof(*state));
    const size_t begin=offsetof(BiprismV10NumericalScratchFfi,nuclear_history);
    const size_t end=begin+sizeof(state->nuclear_history);
    state->nuclear_history.magic=0x4e485034u;
    state->nuclear_history.substep=3u;state->nuclear_history.count=48u;state->nuclear_history.reserved=0u;
    for(uint32_t p=0u;p<48u;++p)for(uint32_t k=0u;k<3u;++k)state->nuclear_history.rows[p][k]=p*3u+k;
    const auto* bytes=reinterpret_cast<const unsigned char*>(state.get());bool ok=true;
    for(size_t j=0;j<sizeof(*state);++j)if((j<begin||j>=end)&&bytes[j]!=0x5au)ok=false;
    for(uint32_t tid=0;tid<192u;++tid)if(state->rank_search.visited_corners[tid]!=0x5a5a5a5au)ok=false;
    std::printf("nuclear_history_lifetime_control bytes=%zu preserved_live_c1_masks=%u pass=%u scope=actual_compiled_FFI_host_address_control\n",sizeof(*state),unsigned(ok),unsigned(ok));
    return ok;
}
int main(int argc,char** argv) {
    try {
        if(!nuclear_history_lifetime_control())throw std::runtime_error("numerical history clobbers live state");
        bool asymmetric=false,connected_work=false,fixed_domain_16=false;float selected_kappa=0.1f;unsigned epoch_count=3u;
        if(argc==6){
            const std::string option=argv[5];
            if(option=="--asymmetric-kappa=0"){selected_kappa=0.f;asymmetric=true;epoch_count=1u;}
            else if(option=="--asymmetric-kappa=0.1"){selected_kappa=0.1f;asymmetric=true;epoch_count=1u;}
            else if(option=="--connected-work")connected_work=true;
            else if(option=="--fixed-domain-16"){fixed_domain_16=true;epoch_count=16u;}
            else throw std::runtime_error("unknown native control option");
        }else if(argc!=5)throw std::runtime_error("usage: unit_qc BPCASE02 selected.tsv true_inverse_au.f32le output_prefix [--asymmetric-kappa=0|--asymmetric-kappa=0.1|--connected-work|--fixed-domain-16]");
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
#if defined(BIPRISM_V10_DIAGNOSTIC_INITIAL_GUESS)
        load_biprism_diagnostic_initial_guess();
#endif
        Device<float3> nx(8*x.n),nv(8*x.n),nf(8*x.n),ax(8*x.n),av(8*x.n),af(8*x.n);
        std::vector<float3> repeated;for(unsigned r=0;r<8;++r)repeated.insert(repeated.end(),x.positions.begin(),x.positions.end());
        nx.put(repeated);nv.zero();nf.zero();ax.zero();av.zero();af.zero();
        CoupledCoordinateOwners coordinate_state(x.n,repeated,nx,ax);
        Device<float> mass(x.n),charges(x.n);mass.put(inverse);charges.put(x.charges);
        Device<uint8_t> elements(x.n);elements.put(x.elements);
        Device<uint32_t> masks(8*256);masks.zero();
        FoliatedStreamPtrTableFfi native={},atomic={};std::vector<const float*> coord_ptrs;
        static_assert(sizeof(FoliatedStreamPtrTableFfi)==640,"coupled pointer ABI");
        static_assert(offsetof(FoliatedStreamPtrTableFfi,d_coordinate_owners)==528,"coordinate owner ABI");
        static_assert(offsetof(FoliatedStreamPtrTableFfi,d_ordered_edges)==536,"ordered edge owner ABI");
        static_assert(offsetof(FoliatedStreamPtrTableFfi,d_potential_owners)==544,"full potential owner ABI");
        static_assert(sizeof(BiprismOrderedEdgesFfi)==6656,"ordered edge scratch ABI");
        static_assert(sizeof(BiprismCoordinateOwnerFfi)==96,"coordinate descriptor ABI");
        static_assert(sizeof(FoliatedCouplingLeafFfi)==17792,"coupling bank ABI");
        static_assert(sizeof(BiprismV10NumericalScratchFfi)==1997440,"numerical scratch ABI");
        static_assert(offsetof(BiprismV10NumericalScratchFfi,working_phi)==38656,"full potential working suffix ABI");
        static_assert(offsetof(BiprismV10NumericalScratchFfi,peer_edge_index)==30976,"immutable peer map ABI");
        Device<FoliatedCouplingLeafFfi> exchange(8);exchange.zero();
        Device<BiprismV10NumericalScratchFfi> numerical(8);numerical.zero();
        Device<BiprismOrderedEdgesFfi> ordered_edges(8);ordered_edges.zero();
        std::unique_ptr<ConnectedWorkControl> work;
        if(connected_work)work.reset(new ConnectedWorkControl(x,epoch_count,out));
        native.physical_step_mode=4u;
        atomic.physical_step_mode=work?0u:4u;
        atomic.d_shared_step=work?work->shared.p:nullptr;
        native.d_coordinate_owners=coordinate_state.native_owner.p;
        atomic.d_coordinate_owners=coordinate_state.atomic_owner.p;
        atomic.d_ordered_edges=ordered_edges.p;
        for(unsigned r=0;r<8;++r){const size_t offset=r*x.n;
            native.d_coords_xyz[r]=reinterpret_cast<float*>(nx.p+offset);native.d_vels_xyz[r]=reinterpret_cast<float*>(nv.p+offset);native.d_forces_xyz[r]=reinterpret_cast<float*>(nf.p+offset);
            atomic.d_coords_xyz[r]=reinterpret_cast<float*>(ax.p+offset);atomic.d_vels_xyz[r]=reinterpret_cast<float*>(av.p+offset);atomic.d_forces_xyz[r]=reinterpret_cast<float*>(af.p+offset);
            atomic.d_inv_masses[r]=atomic.d_true_inv_masses[r]=mass.p;atomic.d_core_shake_mask[r]=masks.p+r*256;
            atomic.d_electronic_exchange[r]=exchange.p+r;
            atomic.d_numerical_scratch[r]=numerical.p+r;
            coord_ptrs.push_back(atomic.d_coords_xyz[r]);
        }
        Device<FoliatedStreamPtrTableFfi> nt(1),at(1);nt.put({native});at.put({atomic});
        Device<BiprismUnitBridgeStatusFfi> bridge(8);bridge.zero();Device<uint32_t> step(1);step.put({0u});
        Device<BiprismConvergenceRecordFfi> conv(8);std::vector<BiprismConvergenceRecordFfi> records(8);
        for(unsigned r=0;r<8;++r){records[r]={};records[r].abi_magic=BIPRISM_TAIL_ABI_MAGIC;records[r].leaf_id=r;}conv.put(records);
        nhs_biprism_import_native_state_v1<<<8,256>>>(nt.p,at.p,bridge.p,step.p,x.n,7u);
        nhs_biprism_guard_unit_import_v1<<<1,256>>>(bridge.p,conv.p,step.p);rt(cudaGetLastError(),"cold bridge");rt(cudaDeviceSynchronize(),"cold bridge wait");
        for(const auto&s:bridge.get())if(s.sticky_fault||s.imported_fields!=7u)throw std::runtime_error("cold quantity import refused");
        coordinate_state.retain(out+".cold");
        retain_qc(out+".cold.au_coords_high.bin",ax.get());retain_qc(out+".cold.native_coords_high.bin",nx.get());
        const auto immutable_native_reference_hi=coordinate_state.native_ref_hi.get(),immutable_native_reference_lo=coordinate_state.native_ref_lo.get();
        const auto immutable_atomic_reference_hi=coordinate_state.atomic_ref_hi.get(),immutable_atomic_reference_lo=coordinate_state.atomic_ref_lo.get();
        if(work)work->initialize_grid(ax.get(),coordinate_state.atomic_low.get());
        std::vector<uint32_t> selection;for(unsigned r=0;r<8;++r)selection.insert(selection.end(),ids.begin(),ids.end());
        Device<uint32_t> selected(selection.size());selected.put(selection);Device<const float*> coords(8);coords.put(coord_ptrs);
        Device<ContactShellSimplicialStateFfi> tile(8);tile.zero();Device<BiprismBwScratchFfi> scratch(8);Device<BiprismBwStatusFfi> status(8);status.zero();
        CoupledPotentialOwners potential_state(tile.p,coordinate_state.atomic_owner.p,23u);
        CoupledSpinorOwners spinor_state(tile.p,coordinate_state.atomic_owner.p,23u);
        CoupledPhase2Control phase2_work(epoch_count);
        atomic.d_potential_owners=potential_state.owners.p;atomic.d_spinor_owners=spinor_state.owners.p;at.put({atomic});
        nhs_seed_bowyer_watson_256_compensated_v1<<<8,256>>>(coords.p,charges.p,elements.p,selected.p,x.n,tile.p,scratch.p,status.p,8u,960u,coordinate_state.atomic_owner.p);
        rt(cudaGetLastError(),"AU BW");rt(cudaDeviceSynchronize(),"AU BW wait");retain_qc(out+".bw_status.bin",status.get());retain_qc(out+".seed_tiles.bin",tile.get());
        for(const auto&s:status.get())if(s.error||!s.published)throw std::runtime_error("actual AU coordinate BW refused");
        for(const auto&t:tile.get())if(!report("actual_AU_seed_topology",biprism_qc::validate_tile_topology(t,x.n,x.groups)))throw std::runtime_error("AU mesh gate");
        if(asymmetric){
            auto asymmetric_tiles=tile.get();initialize_asymmetric_spinors(asymmetric_tiles);
            tile.put(asymmetric_tiles);retain_qc(out+".asymmetric_initial_tiles.bin",asymmetric_tiles);
            std::printf("asymmetric_control enabled=1 outer_steps=1 kappa=%.9g preparation=fresh_actual_KRAS only_core_spinor_initialization_changed=1\n",double(selected_kappa));
        }
        // Assemble and qualify the actual cold operator before step1. The
        // following per-epoch refresh then has an explicit accepted baseline.
        for(unsigned r=0;r<8;++r)nhs_biprism_simplicial_tile_compensated_v1<<<1,256>>>(atomic.d_coords_xyz[r],charges.p,tile.p,r,coordinate_state.atomic_owner.p,conv.p);
        rt(cudaGetLastError(),"cold compensated Phase2");rt(cudaDeviceSynchronize(),"cold Phase2 completion");
        try{phase2_work.initialize(tile.p,at.p,charges.p,x.n,conv.p,step.p);}
        catch(...){phase2_work.retain(out+".cold");potential_state.retain(out+".cold");spinor_state.retain(out+".cold");throw;}
        phase2_work.retain(out+".cold");potential_state.retain(out+".cold");spinor_state.retain(out+".cold");
        retain_qc(out+".cold.au_stream_table.640b",at.get());
        // Construct a total physical-edge correspondence; never assume ordinal identity.
        const auto seed_tiles=tile.get();auto numerical_host=numerical.get();
        if(fixed_domain_16)for(const auto& t:seed_tiles)
            if(!fixed_960_domain_matches(t,t))throw std::runtime_error("fixed-domain control requires the actual 960-tetrahedron seed");
        std::vector<uint16_t> retained_peer_map(8*TILE_MAX_EDGES,0xffffu);
        for(unsigned r=0;r<8;++r){
            const auto& local=seed_tiles[r];const auto& peer=seed_tiles[(r+1)&7u];
            std::map<std::pair<uint32_t,uint32_t>,uint16_t> peer_edges;
            if(local.num_active_edges!=peer.num_active_edges)throw std::runtime_error("unequal peer edge population");
            for(unsigned v=0;v<TILE_CORE_VERTICES;++v)
                if(UNPACK_GLOBAL_ATOM_ID(local.global_atom_idx[v])!=UNPACK_GLOBAL_ATOM_ID(peer.global_atom_idx[v]))throw std::runtime_error("cross-leaf core identity mismatch");
            for(unsigned e=0;e<peer.num_active_edges;++e){
                const auto key=std::make_pair(UNPACK_GLOBAL_ATOM_ID(peer.global_atom_idx[peer.edge_u8[e][0]]),UNPACK_GLOBAL_ATOM_ID(peer.global_atom_idx[peer.edge_u8[e][1]]));
                if(!peer_edges.emplace(key,uint16_t(e)).second)throw std::runtime_error("duplicate peer physical edge");
            }
            std::set<uint16_t> assigned;
            for(unsigned e=0;e<TILE_MAX_EDGES;++e)numerical_host[r].peer_edge_index[e]=0xffffu;
            for(unsigned e=0;e<local.num_active_edges;++e){
                const auto key=std::make_pair(UNPACK_GLOBAL_ATOM_ID(local.global_atom_idx[local.edge_u8[e][0]]),UNPACK_GLOBAL_ATOM_ID(local.global_atom_idx[local.edge_u8[e][1]]));
                const auto it=peer_edges.find(key);if(it==peer_edges.end()||!assigned.insert(it->second).second)throw std::runtime_error("incomplete physical edge bijection");
                numerical_host[r].peer_edge_index[e]=it->second;retained_peer_map[r*TILE_MAX_EDGES+e]=it->second;
            }
            if(assigned.size()!=peer.num_active_edges)throw std::runtime_error("peer edge population not exhausted");
        }
        numerical.put(numerical_host);retain_qc(out+".peer_edge_map.bin",retained_peer_map);
        cudaFuncAttributes main_attributes{};rt(cudaFuncGetAttributes(&main_attributes,nhs_biprism_variational_so3g_v10),"main resource attributes");
        int resident_blocks=0;rt(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident_blocks,nhs_biprism_variational_so3g_v10,256,sizeof(VariationalClusterDsmemV10)),"main CTA occupancy");
        std::printf("main_resources regs=%d local_bytes=%zu static_smem=%zu dynamic_smem=%zu active_blocks=%d\n",main_attributes.numRegs,main_attributes.localSizeBytes,main_attributes.sharedSizeBytes,sizeof(VariationalClusterDsmemV10),resident_blocks);
        if(main_attributes.localSizeBytes!=0||resident_blocks<2)throw std::runtime_error("native main resource gate");
        rt(cudaFuncSetAttribute(nhs_biprism_variational_so3g_v10,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(VariationalClusterDsmemV10)),"v10 shared");
        cudaLaunchConfig_t config={};config.gridDim=dim3(8);config.blockDim=dim3(256);config.dynamicSmemBytes=sizeof(VariationalClusterDsmemV10);
        int active=0;rt(cudaOccupancyMaxActiveClusters(&active,nhs_biprism_variational_so3g_v10,&config),"cluster capacity");if(active<1)throw std::runtime_error("actual8 cluster unavailable");
        Device<InterferometricAdjudicatorFfi> adj(8);adj.zero();Device<uint32_t> arrivals(1),epochs(1);arrivals.zero();epochs.zero();
        const float dt_au=static_cast<float>(0.0005*41341.37333517015);
        std::printf("scope=controlled_native_v10_AU dt_ps=0.0005 dt_au=%.9g initial_velocity=0 native_bath_force=0 actual_physical_inverse=1 mode=FlexibleCoreVerified_GonzalezExactEnergy\n",dt_au);
        for(unsigned epoch=1;epoch<=epoch_count;++epoch) {
            step.put({epoch});
            const float selected_dt=work?work->select(epoch,step.p,conv.p):dt_au;
            coordinate_state.snapshot_native_start(nx);
            nhs_biprism_import_native_state_v1<<<8,256>>>(nt.p,at.p,bridge.p,step.p,x.n,7u);
            nhs_biprism_guard_unit_import_v1<<<1,256>>>(bridge.p,conv.p,step.p);
            retain_qc(out+".epoch"+std::to_string(epoch)+".pre_phase2_tiles.bin",tile.get());
            retain_qc(out+".epoch"+std::to_string(epoch)+".pre_phase2_au_coords.bin",ax.get());
            coordinate_state.retain(out+".epoch"+std::to_string(epoch)+".pre_phase2");
            potential_state.retain(out+".epoch"+std::to_string(epoch)+".pre_phase2");spinor_state.retain(out+".epoch"+std::to_string(epoch)+".pre_phase2");
            try{phase2_work.capture(epoch,selected_dt,work?work->shared.p:nullptr,tile.p,at.p,
                charges.p,bridge.p,conv.p,x.n,selected_kappa);}
            catch(...){phase2_work.retain(out+".epoch"+std::to_string(epoch)+".capture_refused");
                potential_state.retain(out+".epoch"+std::to_string(epoch)+".capture_refused");spinor_state.retain(out+".epoch"+std::to_string(epoch)+".capture_refused");throw;}
            for(unsigned r=0;r<8;++r)nhs_biprism_simplicial_tile_compensated_v1<<<1,256>>>(atomic.d_coords_xyz[r],charges.p,tile.p,r,coordinate_state.atomic_owner.p,conv.p);
            rt(cudaGetLastError(),"actual AU Phase2");rt(cudaDeviceSynchronize(),"AU Phase2 completion");
            try{phase2_work.publish(tile.p,at.p,charges.p,conv.p);}
            catch(...){phase2_work.retain(out+".epoch"+std::to_string(epoch)+".publish_refused");
                potential_state.retain(out+".epoch"+std::to_string(epoch)+".publish_refused");spinor_state.retain(out+".epoch"+std::to_string(epoch)+".publish_refused");throw;}
            phase2_work.retain(out+".epoch"+std::to_string(epoch));
            retain_qc(out+".epoch"+std::to_string(epoch)+".phase2_tiles.bin",tile.get());
            retain_qc(out+".epoch"+std::to_string(epoch)+".phase2_convergence.bin",conv.get());
            for(const auto& c:conv.get())if(c.terminal_fault_latch)throw std::runtime_error("compensated Phase2 explicit fault");
            if(fixed_domain_16){const auto prepared=tile.get();
                for(unsigned r=0;r<8u;++r)if(!fixed_960_domain_matches(seed_tiles[r],prepared[r]))
                    throw std::runtime_error("fixed-domain Phase2 incidence changed");}
            if(epoch==1u){
                const bool parity=work?
                    biprism_reciprocal_parity::run(at.p,tile.p,charges.p,x.n,double(selected_kappa),out+".epoch1",float(selected_dt*.25f)):
                    biprism_reciprocal_parity::run(at.p,tile.p,charges.p,x.n,double(selected_kappa),out+".epoch1");
                if(!parity)
                    throw std::runtime_error("reciprocal cache native parity");
                // Exact main entry controls on test-owned allocations. A failed
                // trial never rewrites the retained scientific input/old faults.
                const auto base_tiles=tile.get();const auto base_coords=ax.get(),base_vels=av.get(),base_forces=af.get();
                const auto base_native_coords=nx.get(),base_coordinate_sidecars=coordinate_state.sidecar_snapshot();
                const auto base_native_owners=coordinate_state.native_owner.get(),base_atomic_owners=coordinate_state.atomic_owner.get();
                const auto base_mass=mass.get(),base_charge=charges.get();
                const auto base_conv=conv.get();const auto base_exchange=exchange.get();
                const auto base_potential_owners=potential_state.owners.get();const auto base_spinor_owners=spinor_state.owners.get();
                const auto base_potential_accepted=potential_state.accepted.get(),base_potential_prepared=potential_state.prepared.get();
                const auto base_numerical=numerical.get();const auto base_adj=adj.get();
                const auto base_arrivals=arrivals.get(),base_epochs=epochs.get(),base_step=step.get();
                for(unsigned r=0;r<8u;++r)if(base_conv[r].abi_magic!=BIPRISM_TAIL_ABI_MAGIC||base_conv[r].leaf_id!=r||base_conv[r].terminal_fault_latch!=0u||base_conv[r].epoch_cluster_fault!=0u||base_conv[r].committed_substeps!=0u)
                    throw std::runtime_error("entry control requires clean initialized all-leaf preparation");
                retain_qc(out+".entry_control_before.tiles.bin",base_tiles);
                potential_state.retain(out+".entry_control_before");spinor_state.retain(out+".entry_control_before");
                retain_qc(out+".entry_control_before.coords.bin",base_coords);
                coordinate_state.retain(out+".entry_control_before");
                retain_qc(out+".entry_control_before.native_coords_high.bin",base_native_coords);
                retain_qc(out+".entry_control_before.velocities.bin",base_vels);
                retain_qc(out+".entry_control_before.forces.bin",base_forces);
                retain_qc(out+".entry_control_before.mass.bin",base_mass);
                retain_qc(out+".entry_control_before.charge.bin",base_charge);
                retain_qc(out+".entry_control_before.conv.bin",base_conv);
                retain_qc(out+".entry_control_before.adjudicator.bin",base_adj);
                retain_qc(out+".entry_control_before.exchange.bin",base_exchange);
                retain_qc(out+".entry_control_before.numerical.bin",base_numerical);
                retain_qc(out+".entry_control_before.barrier_arrivals.bin",base_arrivals);
                retain_qc(out+".entry_control_before.barrier_epoch.bin",base_epochs);
                retain_qc(out+".entry_control_before.protocol_step.bin",base_step);
                const auto equal_bytes=[](const auto& a,const auto& b){
                    return a.size()==b.size()&&std::memcmp(a.data(),b.data(),a.size()*sizeof(a[0]))==0;
                };
                std::printf("native_entry_controls kinds=6 leaves=8 pre_positive_main_launches=48 positive_input_restored=1\n");
                for(unsigned control=0;control<6u;++control)for(unsigned bad_leaf=0;bad_leaf<8u;++bad_leaf){
                    auto injected=base_conv;auto injected_numerical=base_numerical;auto injected_tiles=base_tiles;
                    auto injected_potential=base_potential_owners;auto injected_spinors=base_spinor_owners;
                    const uint32_t expected=control==0u?FAULT_BARRIER_TIMEOUT:FAULT_PICARD_NONCONV;
                    if(control==0u)injected[bad_leaf].terminal_fault_latch|=expected;
                    else if(control==1u)injected_numerical[bad_leaf].peer_edge_index[0]=0xffffu;
                    else if(control==2u){
                        // Finite norm four exercises the absolute-unit entry
                        // guard without relying on nonfinite-input refusal.
                        injected_tiles[bad_leaf].psi_vertex[0][0]={2.f,0.f};
                        injected_tiles[bad_leaf].psi_vertex[0][1]={0.f,0.f};
                        injected_spinors[bad_leaf].accepted[0]={{2.,0.},{0.,0.}};
                    }
                    else if(control==3u)++injected_potential[bad_leaf].prepared_from_committed_substeps;
                    else if(control==4u){
                        uint32_t bits;std::memcpy(&bits,&injected_tiles[bad_leaf].phi_dec[0],4);
                        bits^=1u;std::memcpy(&injected_tiles[bad_leaf].phi_dec[0],&bits,4);
                    } else {
                        // Checked compatibility projection must never become authority.
                        uint32_t bits;std::memcpy(&bits,&injected_tiles[bad_leaf].psi_vertex[0][0].re,4);
                        bits^=1u;std::memcpy(&injected_tiles[bad_leaf].psi_vertex[0][0].re,&bits,4);
                    }
                    conv.put(injected);tile.put(injected_tiles);numerical.put(injected_numerical);exchange.put(base_exchange);
                    potential_state.owners.put(injected_potential);spinor_state.owners.put(injected_spinors);
                    adj.put(base_adj);arrivals.put(base_arrivals);epochs.put(base_epochs);
                    const std::string prefix=out+".entry_control_"+std::to_string(control)+"_leaf"+std::to_string(bad_leaf);
                    retain_qc(prefix+".input_conv.bin",injected);
                    retain_qc(prefix+".input_numerical.bin",injected_numerical);
                    if(control==2u||control>=4u)retain_qc(prefix+".input_tiles.bin",injected_tiles);
                    retain_qc(prefix+".input_potential_owners.128b",injected_potential);retain_qc(prefix+".input_spinor_owners.bps1",injected_spinors);
                    nhs_biprism_variational_so3g_v10<<<8,256,sizeof(VariationalClusterDsmemV10)>>>(at.p,charges.p,tile.p,adj.p,conv.p,arrivals.p,epochs.p,selected_dt,selected_kappa,FlexibleCoreVerified|GonzalezExactEnergyBit);
                    rt(cudaGetLastError(),"entry fault exact-main launch");rt(cudaDeviceSynchronize(),"entry fault all-leaf completion");
                    const auto got_conv=conv.get();const auto got_tiles=tile.get();
                    const auto got_coords=ax.get(),got_vels=av.get(),got_forces=af.get();
                    retain_qc(prefix+".conv.bin",got_conv);retain_qc(prefix+".tiles.bin",got_tiles);
                    retain_qc(prefix+".coords.bin",got_coords);retain_qc(prefix+".velocities.bin",got_vels);retain_qc(prefix+".forces.bin",got_forces);
                    coordinate_state.retain(prefix);retain_qc(prefix+".native_coords_high.bin",nx.get());
                    potential_state.retain(prefix);spinor_state.retain(prefix);
                    retain_qc(prefix+".numerical.bin",numerical.get());retain_qc(prefix+".exchange.bin",exchange.get());
                    retain_qc(prefix+".barrier_epoch.bin",epochs.get());retain_qc(prefix+".barrier_arrivals.bin",arrivals.get());
                    const auto got_mass=mass.get(),got_charge=charges.get();const auto got_step=step.get();
                    retain_qc(prefix+".mass.bin",got_mass);retain_qc(prefix+".charge.bin",got_charge);retain_qc(prefix+".protocol_step.bin",got_step);retain_qc(prefix+".adjudicator.bin",adj.get());
                    bool spinor_unchanged=true;const auto got_spinors=spinor_state.owners.get();
                    for(unsigned l=0;l<8;++l)spinor_unchanged=spinor_unchanged&&std::memcmp(&got_spinors[l],&injected_spinors[l],6272)==0;
                    const bool physical=spinor_unchanged&&equal_bytes(injected_tiles,got_tiles)&&equal_bytes(base_coords,got_coords)&&equal_bytes(base_vels,got_vels)&&equal_bytes(base_forces,got_forces)
                        &&equal_bytes(base_mass,got_mass)&&equal_bytes(base_charge,got_charge)&&equal_bytes(base_step,got_step)
                        &&equal_bytes(base_native_coords,nx.get())&&equal_bytes(base_coordinate_sidecars,coordinate_state.sidecar_snapshot())
                        &&equal_bytes(base_native_owners,coordinate_state.native_owner.get())&&equal_bytes(base_atomic_owners,coordinate_state.atomic_owner.get())
                        &&equal_bytes(injected_potential,potential_state.owners.get())
                        &&equal_bytes(base_potential_accepted,potential_state.accepted.get())
                        &&equal_bytes(base_potential_prepared,potential_state.prepared.get());
                    bool collective=true;
                    for(unsigned r=0;r<8u;++r)collective=collective&&got_conv[r].terminal_fault_latch==expected&&got_conv[r].epoch_cluster_fault==expected
                        &&got_conv[r].committed_substeps==base_conv[r].committed_substeps;
                    std::printf("native_entry_fault control=%u injected_leaf=%u expected_mask=%u all8_withheld=%u persistent_state_bitwise=%u\n",control,bad_leaf,expected,unsigned(collective),unsigned(physical));
                    if(!physical||!collective)throw std::runtime_error("entry fault mutated persistent state or failed collective refusal");
                }
                // Quiescent reset of isolated diagnostic allocations only, after
                // retaining every failing-input and refusal state above.
                conv.put(base_conv);tile.put(base_tiles);exchange.put(base_exchange);numerical.put(base_numerical);
                potential_state.owners.put(base_potential_owners);spinor_state.owners.put(base_spinor_owners);
                adj.put(base_adj);arrivals.put(base_arrivals);epochs.put(base_epochs);
            }
            const auto entry_tiles=tile.get();const auto entry_coords=ax.get(),entry_coords_low=coordinate_state.atomic_low.get(),entry_vels=av.get(),entry_forces=af.get();
            const auto entry_convergence=conv.get();
            const auto entry_spinor_owners=spinor_state.owners.get();
            const auto entry_potential_owners=potential_state.owners.get();
            const auto entry_potential_accepted=potential_state.accepted.get(),entry_potential_prepared=potential_state.prepared.get();
            const auto entry_native_coords=nx.get(),entry_native_low=coordinate_state.native_low.get();
            retain_qc(out+".epoch"+std::to_string(epoch)+".entry_au_coords.bin",ax.get());
            coordinate_state.retain(out+".epoch"+std::to_string(epoch)+".entry");
            potential_state.retain(out+".epoch"+std::to_string(epoch)+".entry");spinor_state.retain(out+".epoch"+std::to_string(epoch)+".entry");
            retain_qc(out+".epoch"+std::to_string(epoch)+".entry.au_stream_table.640b",at.get());
            retain_qc(out+".epoch"+std::to_string(epoch)+".entry_native_coords.bin",entry_native_coords);
            retain_qc(out+".epoch"+std::to_string(epoch)+".entry_au_velocities.bin",av.get());
            cudaEvent_t main_begin,main_end;
            rt(cudaEventCreate(&main_begin),"main timing begin create");rt(cudaEventCreate(&main_end),"main timing end create");
            rt(cudaEventRecord(main_begin),"main timing begin record");
            nhs_biprism_variational_so3g_v10<<<8,256,sizeof(VariationalClusterDsmemV10)>>>(at.p,charges.p,tile.p,adj.p,conv.p,arrivals.p,epochs.p,selected_dt,selected_kappa,FlexibleCoreVerified|GonzalezExactEnergyBit);
            rt(cudaGetLastError(),"positive v10");rt(cudaEventRecord(main_end),"main timing end record");rt(cudaDeviceSynchronize(),"positive v10 completion");
            float main_elapsed_ms=0.f;rt(cudaEventElapsedTime(&main_elapsed_ms,main_begin,main_end),"main timing elapsed");
            rt(cudaEventDestroy(main_begin),"main timing begin destroy");rt(cudaEventDestroy(main_end),"main timing end destroy");
            std::printf("native_main_timing epoch=%u cuda_event_ms=%.9g scope=main_launch_only\n",epoch,main_elapsed_ms);
#if defined(BIPRISM_V10_DIAGNOSTIC_INITIAL_GUESS)
            {
                std::vector<uint32_t> status(8);
                rt(cudaMemcpyFromSymbol(status.data(),biprism_diagnostic_initial_guess,
                    status.size()*sizeof(uint32_t),offsetof(BiprismDiagnosticInitialGuess,status)),"diagnostic guess status");
                retain_qc(out+".epoch"+std::to_string(epoch)+".initial_guess_status.bin",status);
                for(unsigned r=0;r<8u;++r)std::printf("diagnostic_initial_guess epoch=%u leaf=%u status=%u autonomous_qualification=0\n",epoch,r,status[r]);
            }
#endif
            records=conv.get();retain_qc(out+".epoch"+std::to_string(epoch)+".conv.bin",records);
            retain_qc(out+".epoch"+std::to_string(epoch)+".exchange.bin",exchange.get());
            retain_qc(out+".epoch"+std::to_string(epoch)+".numerical.bin",numerical.get());
            const auto current_ordered_edges=ordered_edges.get();
            retain_qc(out+".epoch"+std::to_string(epoch)+".ordered_edges.bin",current_ordered_edges);
            retain_qc(out+".epoch"+std::to_string(epoch)+".tiles.bin",tile.get());
            retain_qc(out+".epoch"+std::to_string(epoch)+".au_coords.bin",ax.get());
            coordinate_state.retain(out+".epoch"+std::to_string(epoch));
            potential_state.retain(out+".epoch"+std::to_string(epoch));spinor_state.retain(out+".epoch"+std::to_string(epoch));
            retain_qc(out+".epoch"+std::to_string(epoch)+".au_stream_table.640b",at.get());
            retain_qc(out+".epoch"+std::to_string(epoch)+".au_velocities.bin",av.get());
            retain_qc(out+".epoch"+std::to_string(epoch)+".barrier_epoch.bin",epochs.get());
            retain_qc(out+".epoch"+std::to_string(epoch)+".barrier_arrivals.bin",arrivals.get());
            // Preserve the complete actual endpoint before any new diagnostic
            // can refuse an absent or corrupt traversal publication.
            for(unsigned r=0u;r<8u;++r){
                const auto& index=current_ordered_edges[r];const auto& t=entry_tiles[r];
                bool exact=biprism_ordered_edges_identity_matches(index,r,reinterpret_cast<uint64_t>(tile.p+r),
                    t.num_active_edges,entry_convergence[r].committed_substeps);
                for(unsigned u=0u;u<TILE_CORE_VERTICES;++u)
                    exact=exact&&biprism_ordered_edge_row_matches(index,&t.edge_u8[0][0],&t.edge_u8[0][1],2u,u);
                std::printf("ordered_edge_admission epoch=%u leaf=%u all192_rows_original_order=%u incidents=%u\n",epoch,r,unsigned(exact),index.incident_count);
                if(!exact)throw std::runtime_error("ordered edge rebuild identity/order mismatch");
            }
            if(!run_biprism_ordered_edge_native_control(at.p,tile.p,charges.p,
                out+".epoch"+std::to_string(epoch)+".ordered_parity"))
                throw std::runtime_error("ordered versus linear native arithmetic/control mismatch");
            const auto exchange_host=exchange.get();
            for(unsigned r=0;r<8;++r){const auto& f=exchange_host[r].footer;
                std::printf("coupled_ledger epoch=%u leaf=%u base=%.17g pair=%.17g transfer=%.17g global_energy=%.17g chain=%.17g electronic_eq=%.9g population=%.9g phase=%u fault=%u\n",epoch,r,f.base_energy_delta,f.owned_pair_energy_delta,f.physical_transfer_in,f.global_energy_delta,f.global_chain_defect,f.max_electronic_equation_residual,f.max_population_residual,f.publication_phase,f.local_fault_mask);
            }
            const auto endpoint_tiles=tile.get();const auto endpoint_spinor_owners=spinor_state.owners.get();bool accepted=true;
            if(fixed_domain_16)for(unsigned r=0;r<8u;++r){
                const bool same_domain=fixed_960_domain_matches(seed_tiles[r],endpoint_tiles[r]);
                std::printf("fixed_domain_admission epoch=%u leaf=%u tetrahedra=960 incidence_bitwise=%u\n",epoch,r,unsigned(same_domain));
                accepted=accepted&&same_domain;
            }
            std::array<bool,8> norm_pass{};
            for(unsigned r=0;r<8;++r){
                long double worst=0.,worst_entry=0.;
                for(unsigned v=0;v<TILE_CORE_VERTICES;++v){
                    const auto a=endpoint_spinor_owners[r].accepted[v].s0,b=endpoint_spinor_owners[r].accepted[v].s1;
                    const long double norm=(long double)a.re*a.re+(long double)a.im*a.im+(long double)b.re*b.re+(long double)b.im*b.im;
                    if(!std::isfinite(norm)){worst=INFINITY;break;}
                    worst=std::max(worst,std::abs(norm-1.0L));
                    const auto ea=entry_spinor_owners[r].accepted[v].s0,eb=entry_spinor_owners[r].accepted[v].s1;
                    const long double entry_norm=(long double)ea.re*ea.re+(long double)ea.im*ea.im+(long double)eb.re*eb.re+(long double)eb.im*eb.im;
                    if(!std::isfinite(entry_norm)){worst_entry=INFINITY;break;}
                    worst_entry=std::max(worst_entry,std::abs(norm-entry_norm));
                }
                norm_pass[r]=worst<=1.0e-7L&&worst_entry<=1.0e-7L;
                std::printf("represented_norm epoch=%u leaf=%u max_abs_from_one=%.21Lg max_abs_from_entry=%.21Lg pass=%u\n",epoch,r,worst,worst_entry,unsigned(norm_pass[r]));
            }
            for(unsigned r=0;r<8;++r){const auto&c=records[r];std::printf("epoch=%u leaf=%u committed=%u fault=%u picard=%.9g dec=%.9g energy=%.9g iterations=%u/%u\n",epoch,r,c.committed_substeps,c.terminal_fault_latch,c.last_picard_residual,c.last_dec_residual,c.step_energy_drift,c.picard_iters_used,c.dec_iters_used);
                const bool trace_range=std::isfinite(c.w_x_lambda_trace)&&c.w_x_lambda_trace>=-2.f&&c.w_x_lambda_trace<=2.f
                    &&std::isfinite(c.w_x_tau_trace)&&c.w_x_tau_trace>=-2.f&&c.w_x_tau_trace<=2.f;
                std::printf("wilson_trace_admission epoch=%u leaf=%u space_leaf=%.9g space_time=%.9g finite_closed_range_minus2_plus2=%u\n",
                    epoch,r,c.w_x_lambda_trace,c.w_x_tau_trace,unsigned(trace_range));
                accepted=accepted&&trace_range;
                accepted&=c.terminal_fault_latch==0u&&c.epoch_cluster_fault==0u&&c.committed_substeps==4u*epoch&&std::isfinite(c.step_energy_drift)&&c.step_energy_drift<=1e-6f&&std::isfinite(c.last_dec_residual)&&c.last_dec_residual<=1e-6f&&std::isfinite(c.last_picard_residual)&&c.last_picard_residual<=1e-7f&&std::isfinite(c.unitarity_defect)&&c.unitarity_defect<=1e-7f&&norm_pass[r];}
            long double total_transfer=0.,total_energy=0.,total_chain=0.;
            const auto& first=exchange_host[0].footer;
            for(unsigned r=0;r<8;++r){
                const auto& f=exchange_host[r].footer;
                const double energies[]={f.base_energy_delta,f.owned_pair_energy_delta,f.attributed_coupling_energy_delta,f.coupling_discrete_work,f.physical_transfer_in,f.balanced_local_defect,f.global_energy_delta,f.global_chain_defect};
                const float residuals[]={f.max_coordinate_residual,f.max_population_residual,f.max_electronic_equation_residual,f.max_norm_defect};
                bool ledger=f.abi_magic==BIPRISM_FOLIATED_COUPLING_ABI&&f.leaf_id==r&&uint64_t(f.accepted_substeps_at_entry)+4u==uint64_t(records[r].committed_substeps)&&f.substep==3u&&f.iteration>0u&&f.iteration==first.iteration&&f.publication_phase==4u&&f.local_fault_mask==0u&&f.convergence_vote==1u;
                for(auto a:f.reserved)ledger=ledger&&a==0u;
                for(auto a:energies)ledger=ledger&&std::isfinite(a);
                for(auto a:residuals)ledger=ledger&&std::isfinite(a)&&a>=0.f;
                ledger=ledger&&f.max_coordinate_residual<=1.0e-7f&&f.max_electronic_equation_residual<=1.0e-7f&&f.max_norm_defect<=1.0e-7f&&std::abs(f.global_energy_delta)<=1.0e-6&&std::abs(f.global_chain_defect)<=1.0e-6;
                ledger=ledger&&std::memcmp(&f.global_energy_delta,&first.global_energy_delta,sizeof(double))==0&&std::memcmp(&f.global_chain_defect,&first.global_chain_defect,sizeof(double))==0;
                ledger=ledger&&std::abs(f.attributed_coupling_energy_delta-f.physical_transfer_in-f.coupling_discrete_work)<=1.0e-6&&std::abs(f.balanced_local_defect-f.base_energy_delta-f.coupling_discrete_work)<=1.0e-6;
                total_transfer+=f.physical_transfer_in;total_energy+=(long double)f.base_energy_delta+f.owned_pair_energy_delta;total_chain+=(long double)f.owned_pair_energy_delta-f.coupling_discrete_work;
                std::printf("coupled_footer_admission epoch=%u leaf=%u pass=%u\n",epoch,r,unsigned(ledger));accepted=accepted&&ledger;
            }
            accepted=accepted&&std::isfinite(total_transfer)&&std::abs(total_transfer)<=1.0e-6L&&std::abs(total_energy)<=1.0e-6L&&std::abs(total_chain)<=1.0e-6L&&std::abs(total_energy-first.global_energy_delta)<=1.0e-6L&&std::abs(total_chain-first.global_chain_defect)<=1.0e-6L;
            const auto numerical_end=numerical.get();
            const auto endpoint_potential_owners=potential_state.owners.get();
            const auto endpoint_potential_accepted=potential_state.accepted.get(),endpoint_potential_prepared=potential_state.prepared.get();
            const auto endpoint_coords=ax.get(),endpoint_coords_low=coordinate_state.atomic_low.get(),endpoint_vels=av.get(),endpoint_forces=af.get();
            const auto same_float=[](float a,float b){return std::memcmp(&a,&b,sizeof(float))==0;};
            const auto same_double=[](double a,double b){return std::memcmp(&a,&b,sizeof(double))==0;};
            for(unsigned r=0;r<8u;++r){
                const auto& n=numerical_end[r];const auto& f=exchange_host[r].footer;
                // C4 restores scalar31 after its temporary C3 work diagnostic.
                // This label distinguishes the new raw schema from old captures.
                if(n.rank_diagnostic.trace.magic!=BP_V10_REPRESENTATION_TRACE_TAG&&std::isfinite(n.scalar[BP_NS_ATTEMPT_ENERGY]))
                    std::printf("represented_target epoch=%u leaf=%u centered_error=%.17g search_bias=%.17g raw_C1_epsilon=%.17g coordinate_round_work_reconstructed=%.17g schema=C4_search_bias_in_scalar31\n",
                        epoch,r,n.scalar[BP_NS_C1_GLOBAL_ERROR],n.scalar[BP_NS_C1_SEARCH_BIAS],n.scalar[BP_NS_C1_GLOBAL_ERROR]-n.scalar[BP_NS_C1_SEARCH_BIAS],n.scalar[BP_NS_MID_WORK_DEFECT]-n.scalar[BP_NS_MID_IDEAL_WORK_DEFECT]);
                const bool endpoint_qualified=records[r].terminal_fault_latch==0u
                    &&records[r].epoch_leaf_fault==0u&&records[r].epoch_cluster_fault==0u
                    &&records[r].committed_substeps==4u*epoch&&f.publication_phase==4u
                    &&f.convergence_vote==1u&&f.local_fault_mask==0u;
                if(!endpoint_qualified){
                    // These planes are deliberate C1/C3 aliases until the
                    // outer commit. Do not diagnose their bytes as an endpoint
                    // metric or velocity after a refused body.
                    const size_t first_atom=size_t(r)*x.n,bytes=size_t(x.n)*sizeof(float3);
                    bool unchanged=std::memcmp(&entry_tiles[r],&endpoint_tiles[r],sizeof(entry_tiles[r]))==0
                        &&std::memcmp(entry_coords.data()+first_atom,endpoint_coords.data()+first_atom,bytes)==0
                        &&std::memcmp(entry_coords_low.data()+first_atom,endpoint_coords_low.data()+first_atom,bytes)==0
                        &&std::memcmp(entry_vels.data()+first_atom,endpoint_vels.data()+first_atom,bytes)==0
                        &&std::memcmp(entry_forces.data()+first_atom,endpoint_forces.data()+first_atom,bytes)==0
                        &&std::memcmp(&entry_potential_owners[r],&endpoint_potential_owners[r],128)==0
                        &&std::memcmp(&entry_spinor_owners[r],&endpoint_spinor_owners[r],6272)==0
                        &&std::memcmp(entry_potential_accepted.data()+r*256u,endpoint_potential_accepted.data()+r*256u,2048)==0
                        &&std::memcmp(entry_potential_prepared.data()+r*256u,endpoint_potential_prepared.data()+r*256u,2048)==0;
                    for(unsigned e=0;e<TILE_MAX_EDGES;++e)unchanged=unchanged&&n.peer_edge_index[e]==retained_peer_map[r*TILE_MAX_EDGES+e];
                    std::printf("raw_endpoint_admission epoch=%u leaf=%u qualified=0 persistent_state_bitwise=%u metric_view=attempted_alias_not_endpoint\n",epoch,r,unsigned(unchanged));
                    if(n.scalar[BP_NS_C1_START_ENERGY]==double(BP_V10_NORM_REFUSAL_TAG)){
                        unsigned refused=0;bool capture_valid=true;
                        for(unsigned v=0;v<TILE_CORE_VERTICES;++v){
                            const auto flag=n.rank_choice.index[v];capture_valid=capture_valid&&flag<=1u;refused+=flag==0u;
                            for(unsigned k=0;k<3u;++k)capture_valid=capture_valid&&std::isfinite(n.conservative_force[k][v])&&std::isfinite(n.reciprocal_force[k][v]);
                            capture_valid=capture_valid&&bp_spinor_finite(endpoint_spinor_owners[r].substep_start[v]);
                        }
                        std::printf("norm_rounding_refusal epoch=%u leaf=%u substep=%u fp64_stored_refused_vertices=%u raw_capture_valid=%u schema=NRF4\n",epoch,r,f.substep,refused,unsigned(capture_valid));
                        if(!capture_valid)throw std::runtime_error("invalid represented-norm refusal capture");
                    }
                    const auto& trace=n.rank_diagnostic.trace;
                    if(trace.magic==BP_V10_REPRESENTATION_TRACE_TAG){
                        if(trace.count>17u)throw std::runtime_error("invalid represented-work trace count");
                        std::printf("represented_work_trace epoch=%u leaf=%u substep=%u stage=%u round=%u count=%u schema=RRF3_first_and_latest_rounds\n",epoch,r,f.substep,trace.stage,trace.round,trace.count);
                        for(unsigned j=0;j<trace.count;++j){
                            std::printf("represented_work_round epoch=%u leaf=%u slot=%u round=%u evaluated_error=%.17g predicted_local_best=%.17g global_winner=%u\n",epoch,r,j,trace.evaluation_round[j],trace.evaluated_error[j],trace.predicted_local_best[j],trace.winner[j]);
                        }
                    }
                    if(!unchanged)throw std::runtime_error("refused main changed persistent physical state");
                    accepted=false;continue;
                }
                bool scalar_ok=true;
                for(unsigned j=0;j<=28u;++j)scalar_ok=scalar_ok&&std::isfinite(n.scalar[j]);
                for(unsigned j:{6u,8u,13u,24u})scalar_ok=scalar_ok&&n.scalar[j]>=0.&&n.scalar[j]<=1.0e-7;
                for(unsigned j:{10u,11u,22u,23u})scalar_ok=scalar_ok&&std::abs(n.scalar[j])<=1.0e-6;
                scalar_ok=scalar_ok&&same_double(n.scalar[15],f.global_energy_delta)&&same_double(n.scalar[27],n.scalar[15])&&same_double(n.scalar[28],n.scalar[13]);
                std::vector<bool> core_atoms(x.n,false);
                bool state_ok=true,metric_ok=true;long double min_first=INFINITY,min_second=INFINITY,min_det=INFINITY;
                auto expected_potential_owner=entry_potential_owners[r];
                if(expected_potential_owner.state==BP_POTENTIAL_PREPARED)
                    expected_potential_owner.accepted_operator_generation=expected_potential_owner.prepared_operator_generation;
                expected_potential_owner.state=expected_potential_owner.projection_state=BP_POTENTIAL_ACCEPTED;
                expected_potential_owner.accepted_committed_substeps=expected_potential_owner.projection_committed_substeps=4u*epoch;
                expected_potential_owner.accepted_step=expected_potential_owner.projection_step=epoch;
                expected_potential_owner.projection_operator_generation=expected_potential_owner.accepted_operator_generation;
                state_ok=std::memcmp(&expected_potential_owner,&endpoint_potential_owners[r],128)==0
                    &&std::memcmp(entry_potential_prepared.data()+r*256u,endpoint_potential_prepared.data()+r*256u,2048)==0;
                auto expected_spinor=entry_spinor_owners[r];bp_spinor_publish(expected_spinor,4u*epoch,epoch);
                state_ok=state_ok&&std::memcmp(&expected_spinor,&endpoint_spinor_owners[r],128)==0;
                for(unsigned u=0;u<192u;++u){const auto p=endpoint_spinor_owners[r].accepted[u];
                    const auto a=endpoint_tiles[r].psi_vertex[u][0],b=endpoint_tiles[r].psi_vertex[u][1];
                    state_ok=state_ok&&bp_spinor_finite(p)&&fabs(bp_spinor_norm(p)-1.)<=1e-7
                        &&bp_spinor_projection(p,a.re,a.im,b.re,b.im)
                        &&std::memcmp(&p,&endpoint_spinor_owners[r].substep_start[u],32)==0;}
                const auto& entry_phi=entry_potential_owners[r].state==BP_POTENTIAL_PREPARED?entry_potential_prepared:entry_potential_accepted;
                for(unsigned v=0;v<256u;++v){const double phi=endpoint_potential_accepted[r*256u+v];
                    const float projection=float(phi);
                    state_ok=state_ok&&std::isfinite(phi)&&same_double(phi,n.working_phi[v])
                        &&same_float(projection,endpoint_tiles[r].phi_dec[v]);
                    if(v>=TILE_CORE_VERTICES)state_ok=state_ok&&same_double(phi,entry_phi[r*256u+v]);
                }
                auto expected_tile=entry_tiles[r];
                for(unsigned v=0;v<TILE_CORE_VERTICES;++v){
                    const unsigned atom=UNPACK_GLOBAL_ATOM_ID(entry_tiles[r].global_atom_idx[v]);
                    if(atom>=x.n||core_atoms[atom])throw std::runtime_error("accepted core identity duplicate/out of bounds");
                    core_atoms[atom]=true;const auto pos=endpoint_coords[r*x.n+atom],low=endpoint_coords_low[r*x.n+atom],vel=endpoint_vels[r*x.n+atom];
                    const auto& accepted=n.nuclear_position_start[v];
                    const auto& checked=n.nuclear_position_live[v];
                    state_ok=state_ok&&std::memcmp(&accepted,&checked,sizeof(accepted))==0
                        &&same_float(checked.x.hi,pos.x)&&same_float(checked.x.lo,low.x)
                        &&same_float(checked.y.hi,pos.y)&&same_float(checked.y.lo,low.y)
                        &&same_float(checked.z.hi,pos.z)&&same_float(checked.z.lo,low.z);
                    const long double expected[3]={static_cast<long double>(checked.x.hi)+checked.x.lo,
                        static_cast<long double>(checked.y.hi)+checked.y.lo,static_cast<long double>(checked.z.hi)+checked.z.lo};
                    const long double actual[3]={static_cast<long double>(pos.x)+low.x,
                        static_cast<long double>(pos.y)+low.y,static_cast<long double>(pos.z)+low.z};
                    for(unsigned k=0;k<3u;++k)state_ok=state_ok&&std::isfinite(expected[k])&&std::isfinite(actual[k])&&expected[k]==actual[k];
                    state_ok=state_ok&&same_float(n.endpoint_v[0][v],vel.x)&&same_float(n.endpoint_v[1][v],vel.y)&&same_float(n.endpoint_v[2][v],vel.z)
                        &&same_float(n.v_start[0][v],vel.x)&&same_float(n.v_start[1][v],vel.y)&&same_float(n.v_start[2][v],vel.z);
                    const long double xx=n.nuclear_metric[0][v],yy=n.nuclear_metric[1][v],zz=n.nuclear_metric[2][v],xy=n.nuclear_metric[3][v],xz=n.nuclear_metric[4][v],yz=n.nuclear_metric[5][v];
                    for(unsigned k=0;k<6u;++k)metric_ok=metric_ok&&same_float(n.endpoint_metric[k][v],float(n.nuclear_metric[k][v]));
                    const long double second=xx*yy-xy*xy,det=xx*(yy*zz-yz*yz)-xy*(xy*zz-yz*xz)+xz*(xy*yz-yy*xz);
                    metric_ok=metric_ok&&std::isfinite(xx)&&std::isfinite(yy)&&std::isfinite(zz)&&std::isfinite(xy)&&std::isfinite(xz)&&std::isfinite(yz)
                        &&std::isfinite(second)&&std::isfinite(det)&&xx>0.&&second>0.&&det>0.;
                    min_first=std::min(min_first,xx);min_second=std::min(min_second,second);min_det=std::min(min_det,det);
                    expected_tile.psi_vertex[v][0]=endpoint_tiles[r].psi_vertex[v][0];expected_tile.psi_vertex[v][1]=endpoint_tiles[r].psi_vertex[v][1];
                    expected_tile.phi_dec[v]=endpoint_tiles[r].phi_dec[v];
                }
                // Every tile field outside the admitted core spinor/potential and
                // every noncore physical atom must remain bitwise unchanged.
                state_ok=state_ok&&std::memcmp(&expected_tile,&endpoint_tiles[r],sizeof(expected_tile))==0;
                for(unsigned atom=0;atom<x.n;++atom){
                    const size_t j=r*x.n+atom;
                    state_ok=state_ok&&std::memcmp(&entry_forces[j],&endpoint_forces[j],sizeof(float3))==0;
                    if(!core_atoms[atom])state_ok=state_ok&&std::memcmp(&entry_coords[j],&endpoint_coords[j],sizeof(float3))==0
                        &&std::memcmp(&entry_coords_low[j],&endpoint_coords_low[j],sizeof(float3))==0
                        &&std::memcmp(&entry_vels[j],&endpoint_vels[j],sizeof(float3))==0;
                }
                for(unsigned e=0;e<TILE_MAX_EDGES;++e)state_ok=state_ok&&n.peer_edge_index[e]==retained_peer_map[r*TILE_MAX_EDGES+e];
                std::printf("raw_endpoint_admission epoch=%u leaf=%u scalar=%u exported_bits=%u metric_spd=%u min_gxx=%.21Lg min_minor2=%.21Lg min_det=%.21Lg coordinate_identity=immutable_high_plus_residual_equals_exported_high_plus_low\n",epoch,r,unsigned(scalar_ok),unsigned(state_ok),unsigned(metric_ok),min_first,min_second,min_det);
                accepted=accepted&&scalar_ok&&state_ok&&metric_ok;
            }
            if(asymmetric)accepted=check_asymmetric_endpoint_ring(endpoint_tiles,exchange_host,endpoint_spinor_owners,selected_kappa,epoch)&&accepted;
            const auto native_after_main=nx.get(),native_low_after_main=coordinate_state.native_low.get();
            accepted=accepted&&std::memcmp(entry_native_coords.data(),native_after_main.data(),entry_native_coords.size()*sizeof(float3))==0
                &&std::memcmp(entry_native_low.data(),native_low_after_main.data(),entry_native_low.size()*sizeof(float3))==0;
            if(!accepted){retain_qc(out+".refused_au_coords.bin",ax.get());retain_qc(out+".refused_au_coords_low.bin",coordinate_state.atomic_low.get());return 4;}
            nhs_biprism_validate_unit_export_v1<<<1,256>>>(nt.p,at.p,tile.p,conv.p,bridge.p,step.p,x.n,1u);
            nhs_biprism_commit_unit_export_v1<<<8,256>>>(nt.p,at.p,tile.p,conv.p,bridge.p,step.p,x.n,1u);
            nhs_biprism_import_native_state_v1<<<8,256>>>(nt.p,at.p,bridge.p,step.p,x.n,2u);
            nhs_biprism_guard_unit_import_v1<<<1,256>>>(bridge.p,conv.p,step.p);
            // The default bridge control remains a zero-delta no-op. The
            // explicit connected mode captures and delivers the actual prior
            // observation's pending force before measuring the exported work.
            const auto before=nv.get();
            if(work){
                work->capture(at.p,nt.p,tile.p,bridge.p,conv.p,step.p,FlexibleCoreVerified|GonzalezExactEnergyBit,epoch);
                nhs_deliver_persisted_lif_forces_v1<<<1,256>>>(at.p,work->carry.p,conv.p,step.p,x.n,selected_dt);
                rt(cudaGetLastError(),"connected actual delivery");
            }
            nhs_biprism_validate_unit_export_v1<<<1,256>>>(nt.p,at.p,tile.p,conv.p,bridge.p,step.p,x.n,2u);
            nhs_biprism_commit_unit_export_v1<<<8,256>>>(nt.p,at.p,tile.p,conv.p,bridge.p,step.p,x.n,2u);
            const auto after=nv.get();if(!work&&std::memcmp(before.data(),after.data(),before.size()*sizeof(float3)))throw std::runtime_error("zero feedback roundtripped velocity");
            if(work)work->measure(at.p,nt.p,bridge.p,conv.p,step.p,epoch);
            nhs_biprism_import_native_state_v1<<<8,256>>>(nt.p,at.p,bridge.p,step.p,x.n,3u);
            nhs_biprism_guard_unit_import_v1<<<1,256>>>(bridge.p,conv.p,step.p);rt(cudaDeviceSynchronize(),"endpoint bridge completion");
            for(const auto&s:bridge.get())if(s.sticky_fault||s.imported_fields!=31||s.exported_fields!=3)throw std::runtime_error("bridge endpoint refused");
            coordinate_state.retain(out+".epoch"+std::to_string(epoch)+".post_bridge");
            retain_qc(out+".epoch"+std::to_string(epoch)+".post_bridge.native_coords_high.bin",nx.get());
            retain_qc(out+".epoch"+std::to_string(epoch)+".post_bridge.au_coords_high.bin",ax.get());
            const auto equal_plane=[](const auto& a,const auto& b){return a.size()==b.size()&&std::memcmp(a.data(),b.data(),a.size()*sizeof(a[0]))==0;};
            const bool reference_preserved=equal_plane(immutable_native_reference_hi,coordinate_state.native_ref_hi.get())
                &&equal_plane(immutable_native_reference_lo,coordinate_state.native_ref_lo.get())
                &&equal_plane(immutable_atomic_reference_hi,coordinate_state.atomic_ref_hi.get())
                &&equal_plane(immutable_atomic_reference_lo,coordinate_state.atomic_ref_lo.get());
            uint64_t nonzero_native_low=0u,nonzero_au_low=0u;
            for(const auto& p:coordinate_state.native_low.get()){if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z))throw std::runtime_error("nonfinite native coordinate low");nonzero_native_low+=(p.x!=0.f)+(p.y!=0.f)+(p.z!=0.f);}
            for(const auto& p:coordinate_state.atomic_low.get()){if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z))throw std::runtime_error("nonfinite AU coordinate low");nonzero_au_low+=(p.x!=0.f)+(p.y!=0.f)+(p.z!=0.f);}
            std::printf("coordinate_continuity epoch=%u immutable_reference=%u nonzero_native_low_components=%llu nonzero_au_low_components=%llu\n",epoch,unsigned(reference_preserved),(unsigned long long)nonzero_native_low,(unsigned long long)nonzero_au_low);
            if(!reference_preserved||!nonzero_native_low||!nonzero_au_low)throw std::runtime_error("coordinate owner continuity/coverage gate");
            if(work){work->observe(at.p,tile.p,conv.p,step.p,epoch);work->accept_component(conv.get(),conv.p,step.p,epoch);}
        }
        if(work)work->retain(selection);
        retain_qc(out+".native_coords.bin",nx.get());retain_qc(out+".native_velocities.bin",nv.get());retain_qc(out+".bridge.bin",bridge.get());
        coordinate_state.retain(out+".final");
        if(work)std::puts("native_connected_work_v10_AU_steps=3 bridge_all8=1 actual_capture_delivery_measure_publish=1 trajectory_acceptance=0 full_AMBER_schedule=0 mixed_guard_acceptance=0");
        else if(asymmetric)std::printf("native_asymmetric_v10_AU_steps=1 kappa=%.9g bridge_all8=1 ring_endpoint_check=1 acceptance_scope=controlled_fixed_operator trajectory_acceptance=0 full_AMBER_schedule=0\n",double(selected_kappa));
        else if(fixed_domain_16)std::puts("native_fixed_domain_v10_AU_steps=16 committed_substeps_per_leaf=64 leaves=8 tetrahedra=960 fixed_combinatorial_domain=1 actual_Phase2_refresh=1 boundary_condition=prescribed_Dirichlet64 independent_boundary_work_readback=required trajectory_acceptance=0 full_AMBER_schedule=0");
        else std::puts("native_positive_v10_AU_steps=3 bridge_all8=1 acceptance_scope=per_epoch_fixed_operator refresh_work=retained_pre_post_operator_bytes_not_qualified trajectory_acceptance=0 full_AMBER_schedule=0");return 0;
    }catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
