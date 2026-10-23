// Actual-header native diagnostic. Run only through the canonical GPU admission
// and lease owner. --inspect performs host input/ABI checks without CUDA calls.
// This is not a production graph, physical trajectory, or acceptance substitute.
#include "nhs_biprism_voronoi_lif.cuh"
#include "nhs_biprism_bowyer_watson_seed.cuh"
#include "biprism_host_contract.hpp"
#include "biprism_lif_ledger_contract.hpp"
#include <cuda.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace {
void rt(cudaError_t e, const char* op) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(op)+": "+cudaGetErrorString(e));
}
void drv(CUresult e, const char* op) {
    if (e != CUDA_SUCCESS) {
        const char* s = nullptr; cuGetErrorString(e, &s);
        throw std::runtime_error(std::string(op)+": "+(s ? s : "unknown driver error"));
    }
}
template<class T> std::vector<T> read_vec(std::ifstream& in, size_t n) {
    std::vector<T> v(n);
    in.read(reinterpret_cast<char*>(v.data()), n*sizeof(T));
    if (!in) throw std::runtime_error("truncated transport input");
    return v;
}
struct Input {
    uint32_t n, center, hmr, n_residues;
    std::vector<float3> positions;
    std::vector<float> charges, masses;
    std::vector<uint8_t> elements;
    std::vector<int32_t> residue_ids;
    std::vector<int> offsets, exclusions;
    std::vector<biprism_qc::HConstraint> groups;
};
Input read_input(const char* path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("input open failed");
    const auto magic = read_vec<char>(in, 8);
    if (std::memcmp(magic.data(), "BPCASE02", 8)) throw std::runtime_error("wrong transport ABI");
    auto h = read_vec<uint32_t>(in, 6);
    if (h[0]<256 || h[0]>100000 || h[1]>=h[0] || h[2]>h[0]*1024ull || h[3]>h[0] || h[4]>1 || h[5]==0 || h[5]>h[0])
        throw std::runtime_error("input header bounds");
    Input x; x.n=h[0]; x.center=h[1]; x.hmr=h[4]; x.n_residues=h[5];
    x.positions=read_vec<float3>(in,x.n); x.charges=read_vec<float>(in,x.n); x.masses=read_vec<float>(in,x.n);
    x.elements=read_vec<uint8_t>(in,x.n); x.residue_ids=read_vec<int32_t>(in,x.n);
    x.offsets=read_vec<int>(in,x.n+1); x.exclusions=read_vec<int>(in,h[2]);
    for (uint32_t k=0;k<h[3];++k) {
        auto g=read_vec<uint32_t>(in,2);
        if (g[0]>=x.n || g[1]>3) throw std::runtime_error("constraint header bounds");
        auto hs=read_vec<uint32_t>(in,g[1]);
        for (auto a:hs) if(a>=x.n) throw std::runtime_error("constraint atom bounds");
        x.groups.push_back({g[0],hs});
    }
    if(in.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("extra input bytes");
    if(x.offsets.front()!=0 || x.offsets.back()!=int(x.exclusions.size())) throw std::runtime_error("CSR endpoints");
    for(uint32_t a=0;a<x.n;++a) {
        const auto p=x.positions[a];
        if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z)||!std::isfinite(x.charges[a])||
           !std::isfinite(x.masses[a])||x.masses[a]<=0||x.elements[a]>7) throw std::runtime_error("nonfinite/invalid native input");
        if(x.offsets[a]<0 || x.offsets[a]>x.offsets[a+1]) throw std::runtime_error("CSR ordering");
        if(x.residue_ids[a]<-1 || x.residue_ids[a]>=int32_t(x.n_residues)) throw std::runtime_error("residue identity outside original signed population");
    }
    for(auto a:x.exclusions) if(a<0 || uint32_t(a)>=x.n) throw std::runtime_error("CSR atom bounds");
    return x;
}
bool report(const char* gate, const biprism_qc::ContractReport& r) {
    std::printf("gate=%s accepted=%d total_failures=%zu\n",gate,r.ok(),size_t(r.total_failures));
    for(const auto& i:r.issues) std::printf("issue=%s subject=%u detail=%s\n",i.code.c_str(),i.subject,i.message.c_str());
    return r.ok();
}
// Atom membership is selected from the actual native neighbor graph. Whole
// constraint components are indivisible in the192 core; geometry remains GPU.
struct BwSelection {
    std::vector<uint32_t> ids, component, depth, parent_component, from_atom, to_atom;
};
BwSelection select_bw_population(const Input& x,const std::vector<int>& list,const std::vector<int>& counts) {
    if(counts.size()!=x.n||list.size()!=size_t(x.n)*1024)throw std::runtime_error("native neighbor storage mismatch");
    std::vector<uint32_t> parent(x.n);for(uint32_t a=0;a<x.n;++a)parent[a]=a;
    const auto find=[&](uint32_t a){while(parent[a]!=a){parent[a]=parent[parent[a]];a=parent[a];}return a;};
    for(const auto& g:x.groups)for(const auto h:g.hydrogens){uint32_t a=find(g.central),b=find(h);if(a!=b)parent[std::max(a,b)]=std::min(a,b);}
    std::vector<std::vector<uint32_t>> members(x.n);
    BwSelection out;out.component.resize(x.n);out.depth.assign(x.n,UINT32_MAX);out.parent_component.assign(x.n,UINT32_MAX);out.from_atom.assign(x.n,UINT32_MAX);out.to_atom.assign(x.n,UINT32_MAX);
    for(uint32_t a=0;a<x.n;++a){out.component[a]=find(a);members[out.component[a]].push_back(a);
        if(counts[a]<0||counts[a]>1024)throw std::runtime_error("native neighbor count outside actual capacity");
        for(int j=0;j<counts[a];++j){const int b=list[size_t(a)*1024+j];if(b<0||uint32_t(b)>=x.n)throw std::runtime_error("native neighbor atom outside input");}}
    const uint32_t start=out.component[x.center];out.depth[start]=0;
    std::vector<uint32_t> bfs{start};
    for(size_t q=0;q<bfs.size();++q){const uint32_t c=bfs[q];
        for(auto a:members[c])for(int j=0;j<counts[a];++j){const uint32_t b=uint32_t(list[size_t(a)*1024+j]),d=out.component[b];
            if(out.depth[d]==UINT32_MAX){out.depth[d]=out.depth[c]+1;out.parent_component[d]=c;out.from_atom[d]=a;out.to_atom[d]=b;bfs.push_back(d);}}}
    if(members[start].size()>TILE_CORE_VERTICES)throw std::runtime_error("center constraint component exceeds core capacity");
    std::sort(bfs.begin(),bfs.end(),[&](uint32_t a,uint32_t b){return std::make_pair(out.depth[a],a)<std::make_pair(out.depth[b],b);});
    const uint32_t remaining=TILE_CORE_VERTICES-uint32_t(members[start].size());
    std::vector<int> choice(remaining+1,-1),previous(remaining+1,-1);choice[0]=-2;
    for(uint32_t c:bfs){if(c==start)continue;const uint32_t size=uint32_t(members[c].size());if(size>remaining)continue;
        for(int sum=int(remaining);sum>=int(size);--sum)if(choice[sum]==-1&&choice[sum-int(size)]!=-1){choice[sum]=int(c);previous[sum]=sum-int(size);}
        if(choice[remaining]!=-1)break;}
    if(choice[remaining]==-1)throw std::runtime_error("reachable whole constraint components cannot fill exact192 core");
    std::vector<uint8_t> selected_component(x.n,0),selected_atom(x.n,0);selected_component[start]=1;
    for(int sum=int(remaining);sum>0;sum=previous[sum]){if(choice[sum]<0||previous[sum]<0)throw std::runtime_error("core selection witness invalid");selected_component[uint32_t(choice[sum])]=1;}
    out.ids.push_back(x.center);selected_atom[x.center]=1;
    for(uint32_t c:bfs)if(selected_component[c])for(uint32_t a:members[c])if(!selected_atom[a]){out.ids.push_back(a);selected_atom[a]=1;}
    if(out.ids.size()!=TILE_CORE_VERTICES)throw std::runtime_error("core selection cardinality mismatch");
    for(uint32_t c:bfs)for(uint32_t a:members[c])if(!selected_atom[a]&&out.ids.size()<TILE_TOTAL_VERTICES){out.ids.push_back(a);selected_atom[a]=1;}
    if(out.ids.size()!=TILE_TOTAL_VERTICES)throw std::runtime_error("reachable native graph has fewer than256 admissible atoms");
    std::vector<uint8_t> core(x.n,0);for(uint32_t i=0;i<TILE_CORE_VERTICES;++i)core[out.ids[i]]=1;
    for(const auto&g:x.groups){unsigned covered=core[g.central];for(auto h:g.hydrogens)covered+=core[h];if(covered&&covered!=1+g.hydrogens.size())throw std::runtime_error("selected core splits native constraint group");}
    std::printf("native_graph_selected=256 core_constraint_closed=192 halo=64 reachable_components=%zu source=native_neighbor_edges_and_actual_constraint_groups\n",bfs.size());
    return out;
}
void write_bw_selection(const BwSelection& selection,const char* tile_dump_path) {
    std::ofstream out(std::string(tile_dump_path)+".selection.tsv");if(!out)throw std::runtime_error("selection witness open");
    out<<"leaf\tvertex\tatom\tcore\tcomponent\tdepth\tparent_component\tdiscovery_from_atom\tdiscovery_to_atom\n";
    for(unsigned leaf=0;leaf<8;++leaf)for(unsigned v=0;v<256;++v){const uint32_t a=selection.ids[v],c=selection.component[a];
        out<<leaf<<'\t'<<v<<'\t'<<a<<'\t'<<(v<192)<<'\t'<<c<<'\t'<<selection.depth[c]<<'\t'<<selection.parent_component[c]<<'\t'<<selection.from_atom[c]<<'\t'<<selection.to_atom[c]<<'\n';}
    if(!out)throw std::runtime_error("selection witness write");
}

template<class F> struct KernelArguments;
template<class R,class... A> struct KernelArguments<R (*)(A...)> {
    using Types=std::tuple<A...>;
};
bool inspect_signal_grid_contract() {
    using Args=KernelArguments<decltype(&nhs_voronoi_dual_lif_and_4grid_bypass_v2)>::Types;
    using Hit=typename std::tuple_element<6,Args>::type;
    using Coupled=typename std::tuple_element<7,Args>::type;
    using Primary=typename std::tuple_element<8,Args>::type;
    using Count=typename std::tuple_element<9,Args>::type;
    using Residues=typename std::tuple_element<4,Args>::type;
    constexpr bool matches=std::tuple_size<Args>::value==21 &&
        std::is_same<Hit,int32_t*>::value && std::is_same<Count,int32_t*>::value &&
        std::is_same<Coupled,int32_t*>::value &&
        std::is_same<Primary,int32_t*>::value && std::is_same<Residues,const int32_t*>::value;
    // The native serializer and coupling producer allocate/consume CudaSlice<i32>.
    // Inspect the actual compiled candidate signature, not a duplicate declaration.
    std::printf("gate=integer_grid_storage accepted=%d candidate_float_pointer=%d primary_signed_i32=%d residue_map_signed_i32=%d\n",
                matches,std::is_same<Coupled,float*>::value,
                std::is_same<Primary,int32_t*>::value,std::is_same<Residues,const int32_t*>::value);
    return matches;
}
template<class T> struct Device {
    T* p=nullptr; size_t n=0;
    explicit Device(size_t count):n(count) { rt(cudaMalloc(reinterpret_cast<void**>(&p),std::max(size_t(1),n)*sizeof(T)),"cudaMalloc"); }
    Device(const Device&)=delete; Device& operator=(const Device&)=delete;
    ~Device() { if(p) { auto e=cudaFree(p); if(e!=cudaSuccess) std::fprintf(stderr,"cudaFree: %s\n",cudaGetErrorString(e)); } }
    void put(const std::vector<T>& v) { if(v.size()!=n)throw std::runtime_error("HtoD size"); rt(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice),"HtoD"); }
    std::vector<T> get() const { std::vector<T> v(n);rt(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost),"DtoH");return v; }
    void zero() {rt(cudaMemset(p,0,n*sizeof(T)),"memset");}
};
void launch_driver(CUfunction f, unsigned grid, void** args) {
    drv(cuLaunchKernel(f,grid,1,1,256,1,1,0,nullptr,args,nullptr),"cuLaunchKernel");
    drv(cuCtxSynchronize(),"cuCtxSynchronize");
}
struct Module {
    CUmodule m=nullptr;
    explicit Module(const char* path) {drv(cuModuleLoad(&m,path),"cuModuleLoad actual neighbor module");}
    ~Module() {if(m)cuModuleUnload(m);}
    CUfunction function(const char* name) {CUfunction f=nullptr;drv(cuModuleGetFunction(&f,m,name),name);return f;}
};

// Validator controls deliberately use constructed records, never a fabricated
// production run. All native kernel/readback gates below remain separate.
int host_ledger_controls() {
    struct Window {
        std::vector<GpuSpikeEventCanonical96> events;
        uint32_t count=2;
        std::vector<float> voltage;
        std::vector<ContactShellSimplicialStateFfi> tiles;
        std::vector<BiprismConvergenceRecordFfi> convergence;
        std::vector<int32_t> residues;
        std::vector<float3> coords,before,after;
        SignalGridSpecFfi grid{};
        std::vector<int32_t> hits,coupled,primary,primary_count;
        float threshold=1.25f;
    };
    const auto make_window=[]() {
        Window w;
        w.events.resize(2);w.voltage.resize(8*TILE_TOTAL_VERTICES);w.tiles.resize(8);w.convergence.resize(8);
        w.residues.resize(256);w.coords.resize(8*256,make_float3(.5f,.5f,.5f));
        w.before.resize(8*256,make_float3(0,0,0));w.after=w.before;
        for (unsigned a=0;a<256;++a) w.residues[a]=int32_t(a);
        for (auto& t:w.tiles) for (unsigned a=0;a<256;++a) {t.global_atom_idx[a]=PACK_ATOM_ELEM(a,0);t.star0_vol[a]=1;}
        for (auto& c:w.convergence)c.w_x_lambda_trace=2;
        w.grid.inv_voxel_spacing=1;w.grid.dim_xyz[0]=w.grid.dim_xyz[1]=w.grid.dim_xyz[2]=2;w.grid.total_voxels=8;
        w.hits.resize(8);w.coupled.resize(8);w.primary.resize(8,-1);w.primary_count.resize(8);
        w.hits[0]=9;w.coupled[0]=1;w.primary[0]=0;w.primary_count[0]=1;
        w.after[0].x=1;w.after[192].x=-1;
        for (unsigned a=0;a<2;++a) {
            auto& e=w.events[a];e={};
            e.pos_xyz[0]=e.pos_xyz[1]=e.pos_xyz[2]=.5f;
            e.step_tau=23;e.atom_or_site_id=a;e.residue_id=a;e.leaf_id=0;e.vertex_id=a;
            e.multiplicity_m=a?8:1;e.lif_threshold_used=w.threshold;
            e.q_discharged=float(e.multiplicity_m)*w.threshold;e.q_discharged_lo=0;
            e.r_post_reset=a?.5f:.25f;e.v_pre_reset=e.q_discharged+e.r_post_reset;
            e.spike_amplitude=e.q_discharged;e.voronoi_dual_vol=1;
            e.event_schema_magic=SPIKE_EVENT_V10_SCHEMA_MAGIC;e.voxel_cell_idx=0;
            e.coupled_force_xyz[0]=a?0.f:1.f;
            w.voltage[a]=e.r_post_reset;
        }
        return w;
    };
    unsigned checked=0;
    const auto check=[&](const char* name,const std::function<void(Window&)>& mutate,const char* expected_issue,bool strict=true) {
        auto w=make_window();mutate(w);
        const auto r=biprism_qc::validate_lif_window(w.events,w.count,w.voltage,w.tiles,w.convergence,w.residues,w.coords,
            w.before,w.after,w.grid,w.hits,w.coupled,w.primary,w.primary_count,23,w.threshold,strict);
        bool pass=expected_issue==nullptr ? r.contract.ok() : false;
        if (expected_issue) for (const auto& issue:r.contract.issues) if(issue.code==expected_issue) pass=true;
        std::printf("ledger_control=%s passed=%d failures=%zu plain_ledger_error=%.12Lg expanded_ledger_error=%.12Lg force_sum_change=%.12Lg\n",
            name,pass,size_t(r.contract.total_failures),r.max_single_float_ledger_error,r.max_expanded_ledger_error,r.max_force_sum_change);
        if(!pass)throw std::runtime_error(std::string("host ledger control failed: ")+name);
        ++checked;
    };
    check("valid_owner_conditioned_multiplicity",[](Window&){},nullptr);
    check("wrong_schema",[](Window&w){w.events[0].event_schema_magic=0;},"LIF_EVENT_SCHEMA");
    check("wrong_cell",[](Window&w){w.events[0].voxel_cell_idx=1;},"LIF_EVENT_CELL");
    check("wrong_product_witness",[](Window&w){w.events[0].q_lo_product_raw_bits=1;},"LIF_CANONICAL_PARTITION");
    check("wrong_quantum",[](Window&w){w.events[0].q_discharged+=.25f;},"LIF_QUOTIENT_OR_QUANTUM");
    check("wrong_membrane",[](Window&w){w.voltage[0]=0;},"LIF_MEMBRANE_LEDGER");
    check("negative_counter",[](Window&w){w.hits[0]=-1;},"LIF_MULTIPLICITY_GRIDS");
    check("misattributed_primary",[](Window&w){w.primary_count[0]=9;},"LIF_PRIMARY_ATTRIBUTION");
    check("dropped_halo_reaction",[](Window&w){w.after[192].x=0;},"LIF_REQUESTED_FORCE_BOUND");
    check("truncated_queue",[](Window&w){w.count=3;},"LIF_QUEUE_TRUNCATED");
    check("wrong_leaf",[](Window&w){w.events[0].leaf_id=8;},"LIF_EVENT_IDENTITY");
    check("duplicate_membrane",[](Window&w){w.events[1].vertex_id=0;},"LIF_EVENT_IDENTITY");
    check("unsupported_multiplicity",[](Window&w){w.events[0].multiplicity_m=65536;},"LIF_EVENT_IDENTITY");
    check("nonfinite_ledger",[](Window&w){w.events[0].v_pre_reset=std::numeric_limits<float>::infinity();},"LIF_LEDGER_DOMAIN");
    check("excluded_only_cell",[](Window&w){w.residues[0]=w.residues[1]=-1;w.events[0].residue_id=w.events[1].residue_id=-1;w.primary[0]=-1;w.primary_count[0]=0;},nullptr);
    check("mixed_excluded_attributable_cell",[](Window&w){w.residues[0]=-1;w.events[0].residue_id=-1;w.primary[0]=1;w.primary_count[0]=8;},nullptr);
    check("wrong_pocket",[](Window&w){w.events[0].pocket_tile_id=1;},"LIF_OBSERVABLE_PROVENANCE");
    check("wrong_dual_volume",[](Window&w){w.events[0].voronoi_dual_vol=2;},"LIF_OBSERVABLE_PROVENANCE");
    check("wrong_holonomy",[](Window&w){w.events[0].uv_or_holonomy_val=1;},"LIF_OBSERVABLE_PROVENANCE");
    check("nonfinite_matching_position",[](Window&w){w.coords[0].x=w.events[0].pos_xyz[0]=std::numeric_limits<float>::infinity();},"LIF_EVENT_POSITION");
    const auto wide_quantum=[](Window&w) {
        w.count=1;w.threshold=1.1f;auto&e=w.events[0];e.multiplicity_m=65535;e.lif_threshold_used=w.threshold;
        e.v_pre_reset=72088.5078125f;e.q_discharged=float(e.multiplicity_m)*w.threshold;
        e.q_discharged_lo=std::fma(float(e.multiplicity_m),w.threshold,-e.q_discharged);
        std::memcpy(&e.q_lo_product_raw_bits,&e.q_discharged_lo,4);
        e.r_post_reset=std::fma(-float(e.multiplicity_m),w.threshold,e.v_pre_reset);
        e.spike_amplitude=e.q_discharged;w.voltage[0]=e.r_post_reset;
        w.hits[0]=w.coupled[0]=w.primary_count[0]=65535;
    };
    check("expanded_quantum_ledger",wide_quantum,nullptr,false);
    check("single_float_absolute_limit",wide_quantum,"LIF_REQUESTED_SINGLE_FLOAT_BOUND");
    check("fp32_force_rounding_limit",[](Window&w){w.after[0].x=4096;w.after[192].x=-4096;w.after[193].x=-.000244140625f;},"LIF_REQUESTED_FORCE_BOUND");
    std::printf("host_ledger_controls_passed=%u gpu_execution=0\n",checked);
    return 0;
}

int native(const Input& x,const char* module_path,const char* tile_dump_path) {
    rt(cudaSetDevice(0),"cudaSetDevice");rt(cudaFree(nullptr),"initialize primary context");
    int cluster_launch=0;
    rt(cudaDeviceGetAttribute(&cluster_launch,cudaDevAttrClusterLaunch,0),"query clusterLaunch");
    std::printf("cluster_launch=%d actual_neighbor_module=%s\n",cluster_launch,module_path);
    Device<float3> coords(x.n);coords.put(x.positions);
    Device<float> charges(x.n);charges.put(x.charges);
    Device<uint8_t> elements(x.n);elements.put(x.elements);
    Device<int> exclusions(x.exclusions.size()),offsets(x.offsets.size());exclusions.put(x.exclusions);offsets.put(x.offsets);
    std::array<float,3> low={x.positions[0].x,x.positions[0].y,x.positions[0].z},high=low;
    for(auto p:x.positions) {float q[3]={p.x,p.y,p.z};for(int k=0;k<3;++k){low[k]=std::min(low[k],q[k]);high[k]=std::max(high[k],q[k]);}}
    int dims[3];float origin[3];
    for(int k=0;k<3;++k) {origin[k]=low[k]-10.f;const float extent=std::ceil((high[k]-low[k]+20.f)/10.f);if(!std::isfinite(extent)||extent<1||extent>32)throw std::runtime_error("native cell grid bound");dims[k]=int(extent);}
    int n=int(x.n),ncells=dims[0]*dims[1]*dims[2];
    // Match the actual native cell assignment before launching its bounded
    // 512-atom cells. This is an allocation guard, not a neighbor substitute.
    std::vector<unsigned> occupancy(ncells,0);
    for(auto p:x.positions) {
        const float q[3]={p.x,p.y,p.z}; int cell[3];
        for(int k=0;k<3;++k) cell[k]=std::max(0,std::min(dims[k]-1,int((q[k]-origin[k])*.1f)));
        const int id=cell[0]+dims[0]*(cell[1]+dims[1]*cell[2]);
        if(++occupancy[id]>512) throw std::runtime_error("input exceeds native cell capacity");
    }
    Device<int> cells(size_t(ncells)*512),counts(ncells),atom_cell(n),nlist(size_t(n)*1024),ncounts(n);
    Module module(module_path);
    auto reset=module.function("reset_cell_counts"),build=module.function("build_cell_list"),neighbors=module.function("nhs_build_neighbor_list_warp");
    void* rargs[]={&counts.p,&ncells};launch_driver(reset,(ncells+255)/256,rargs);
    void* cargs[]={&coords.p,&cells.p,&counts.p,&atom_cell.p,&origin[0],&origin[1],&origin[2],&dims[0],&dims[1],&dims[2],&n};
    launch_driver(build,(n+255)/256,cargs);
    const auto cell_counts=counts.get();for(auto c:cell_counts)if(c<0||c>512)throw std::runtime_error("native cell overflow");
    float cutoff_sq=144.f; // Actual native 10A cutoff + 2A neighbor buffer.
    void* nargs[]={&coords.p,&cells.p,&counts.p,&atom_cell.p,&exclusions.p,&offsets.p,&nlist.p,&ncounts.p,&dims[0],&dims[1],&dims[2],&n,&cutoff_sq};
    launch_driver(neighbors,(n+7)/8,nargs);
    auto host_list=nlist.get(),host_counts=ncounts.get();
    std::printf("native_neighbor_center=%u count=%d\n",x.center,host_counts.at(x.center));
    const auto selection=select_bw_population(x,host_list,host_counts);
    write_bw_selection(selection,tile_dump_path);
    std::vector<uint32_t> selected_ids;for(unsigned leaf=0;leaf<8;++leaf)selected_ids.insert(selected_ids.end(),selection.ids.begin(),selection.ids.end());
    Device<uint32_t> selected(selected_ids.size());selected.put(selected_ids);
    // This diagnostic starts all8 independent leaf buffers from the same actual
    // input coordinates; the production ABI accepts distinct per-leaf pointers.
    Device<const float*> seed_coords(8);seed_coords.put(std::vector<const float*>(8,reinterpret_cast<const float*>(coords.p)));
    Device<BiprismBwScratchFfi> bw_scratch(8);
    Device<BiprismBwStatusFfi> bw_status(8);bw_status.zero();
    Device<ContactShellSimplicialStateFfi> tiles(8);
    rt(cudaMemset(tiles.p,0xff,8*sizeof(ContactShellSimplicialStateFfi)),"poison tile allocation");
    int active_bw_blocks=0;rt(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_bw_blocks,nhs_seed_bowyer_watson_256_v1,256,0),"BW occupancy");
    std::printf("bw_active_blocks_per_sm=%d bw_scratch_per_leaf=%zu bw_status_bytes=%zu\n",active_bw_blocks,sizeof(BiprismBwScratchFfi),sizeof(BiprismBwStatusFfi));
    if(active_bw_blocks<2)throw std::runtime_error("native BW two-CTA occupancy gate");
    nhs_seed_bowyer_watson_256_v1<<<8,256>>>(seed_coords.p,charges.p,elements.p,selected.p,x.n,tiles.p,bw_scratch.p,bw_status.p,8u,960u);
    rt(cudaGetLastError(),"native BW seed launch");rt(cudaDeviceSynchronize(),"native BW seed completion");
    const auto statuses=bw_status.get();bool all_bw_published=true;
    {std::ofstream out(std::string(tile_dump_path)+".bw_status.bin",std::ios::binary);out.write(reinterpret_cast<const char*>(statuses.data()),statuses.size()*sizeof(statuses[0]));if(!out)throw std::runtime_error("BW status write");}
    for(unsigned leaf=0;leaf<8;++leaf){const auto&r=statuses[leaf];
        const bool accepted=r.abi_magic==BPBW_STATUS_MAGIC&&r.leaf_id==leaf&&r.error==BPBW_OK&&r.phase==5&&r.inserted_vertices==256&&r.published==1&&r.final_tets>0&&r.final_tets<=960&&r.final_edges>0&&r.final_edges<=TILE_MAX_EDGES;
        std::printf("gate=native_bowyer_publication leaf=%u accepted=%d error=%u phase=%u inserted=%u peak=%u unpeeled_tets=%u unpeeled_edges=%u final_tets=%u final_edges=%u super_attempt=%u uncertain=%u diagnostic_tet=%u diagnostic_vertex=%u\n",leaf,accepted,r.error,r.phase,r.inserted_vertices,r.peak_tets,r.unpeeled_tets,r.unpeeled_edges,r.final_tets,r.final_edges,r.super_attempt,r.uncertain_predicates,r.diagnostic_tet,r.diagnostic_vertex);
        all_bw_published=all_bw_published&&accepted;}
    if(!all_bw_published)throw std::runtime_error("native BW refused at leastone leaf; downstream launch withheld");
    auto before=tiles.get();
    {std::ofstream out(tile_dump_path,std::ios::binary);if(!out)throw std::runtime_error("tile dump open");out.write(reinterpret_cast<char*>(before.data()),8*sizeof(before[0]));if(!out)throw std::runtime_error("tile dump write");}
    bool valid=true;for(unsigned r=0;r<8;++r) valid=report("seeded_topology",biprism_qc::validate_tile_topology(before[r],x.n,x.groups))&&valid;
    if(!valid)return 3;
    nhs_biprism_simplicial_tile_v1<<<8,256>>>(reinterpret_cast<float*>(coords.p),charges.p,tiles.p,0);
    rt(cudaGetLastError(),"Phase2 launch");rt(cudaDeviceSynchronize(),"Phase2 completion");
    auto after=tiles.get();
    for(unsigned r=0;r<8;++r) {
        if(!report("Phase2_topology",biprism_qc::validate_tile_topology(after[r],x.n,x.groups)))
            throw std::runtime_error("Phase2 topology gate; refusing field access");
        for(unsigned v=0;v<TILE_TOTAL_VERTICES;++v)if(!std::isfinite(after[r].phi_dec[v])||!std::isfinite(after[r].star0_vol[v])||after[r].star0_vol[v]<=0)valid=false;
        for(unsigned e=0;e<after[r].num_active_edges;++e){
            if(!std::isfinite(after[r].star1_wt[e])||after[r].star1_wt[e]<=0)valid=false;
            const auto g=after[r].grad_star1_u[e];
            if(!std::isfinite(g.x)||!std::isfinite(g.y)||!std::isfinite(g.z))valid=false;
        }
    }
    if(!valid)throw std::runtime_error("Phase2 topology/finite gate");
    if(!cluster_launch)throw std::runtime_error("device does not admit required hardware cluster; no spinning fallback launch");
    rt(cudaFuncSetAttribute(nhs_biprism_variational_so3g_v10,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(VariationalClusterDsmemV10)),"set v10 shared memory");
    cudaLaunchConfig_t config={};config.gridDim=dim3(8);config.blockDim=dim3(256);config.dynamicSmemBytes=sizeof(VariationalClusterDsmemV10);
    int active_clusters=0;rt(cudaOccupancyMaxActiveClusters(&active_clusters,nhs_biprism_variational_so3g_v10,&config),"cluster occupancy");
    if(active_clusters<1)throw std::runtime_error("no resident cluster for actual v10");
    // Eight independent real-coordinate streams; diagnostic zero velocities/forces.
    // No physical-mass array is invented and FlexibleCoreVerified is never enabled.
    Device<float3> all_coords(size_t(n)*8),vel(size_t(n)*8),forces(size_t(n)*8);
    std::vector<float3> repeated;for(int r=0;r<8;++r)repeated.insert(repeated.end(),x.positions.begin(),x.positions.end());all_coords.put(repeated);vel.zero();forces.zero();
    Device<float> inv_mass(n);std::vector<float> inv(n);for(int a=0;a<n;++a)inv[a]=1.f/x.masses[a];inv_mass.put(inv);
    FoliatedStreamPtrTableFfi ptrs={};
    Device<BiprismOrderedEdgesFfi> ordered_edges(8);ordered_edges.zero();
    ptrs.d_ordered_edges=ordered_edges.p;
    for(int r=0;r<8;++r){ptrs.d_coords_xyz[r]=reinterpret_cast<float*>(all_coords.p+size_t(r)*n);ptrs.d_vels_xyz[r]=reinterpret_cast<float*>(vel.p+size_t(r)*n);ptrs.d_forces_xyz[r]=reinterpret_cast<float*>(forces.p+size_t(r)*n);ptrs.d_inv_masses[r]=inv_mass.p;}
    Device<FoliatedStreamPtrTableFfi> table(1);table.put(std::vector<FoliatedStreamPtrTableFfi>{ptrs});
    Device<BiprismConvergenceRecordFfi> conv(8);std::vector<BiprismConvergenceRecordFfi> records(8);
    for(unsigned r=0;r<8;++r){records[r]={};records[r].abi_magic=BIPRISM_TAIL_ABI_MAGIC;records[r].leaf_id=r;records[r].committed_substeps=17;}
    records[3].terminal_fault_latch=FAULT_PICARD_NONCONV;conv.put(records);
    Device<InterferometricAdjudicatorFfi> adj(8);rt(cudaMemset(adj.p,0xa5,8*sizeof(InterferometricAdjudicatorFfi)),"adjudicator sentinel");
    const auto adj_before=adj.get();Device<uint32_t> arrival(1),epoch(1);arrival.zero();epoch.zero();
    nhs_biprism_variational_so3g_v10<<<8,256,sizeof(VariationalClusterDsmemV10)>>>(table.p,charges.p,tiles.p,adj.p,conv.p,arrival.p,epoch.p,0.f,0.f,ConstrainedShakeSafe);
    rt(cudaGetLastError(),"v10 fault refusal launch");rt(cudaDeviceSynchronize(),"v10 fault refusal completion");
    const auto c_end=all_coords.get(),v_end=vel.get();const auto t_end=tiles.get();const auto a_end=adj.get();const auto rec_end=conv.get();
    if(std::memcmp(c_end.data(),repeated.data(),repeated.size()*sizeof(float3)))throw std::runtime_error("faulted v10 changed coordinates");
    for(auto v:v_end)if(v.x!=0||v.y!=0||v.z!=0)throw std::runtime_error("faulted v10 changed velocity");
    if(std::memcmp(t_end.data(),after.data(),8*sizeof(ContactShellSimplicialStateFfi)))throw std::runtime_error("faulted v10 changed persistent tile");
    for(int r=0;r<8;++r){if(!(rec_end[r].epoch_cluster_fault&FAULT_PICARD_NONCONV)||!(rec_end[r].terminal_fault_latch&FAULT_PICARD_NONCONV)||rec_end[r].committed_substeps!=17)throw std::runtime_error("fault collective/latch/clock mismatch");if(std::memcmp(&a_end[r],&adj_before[r],192))throw std::runtime_error("adjudicator active ABI overwritten");}
    for(auto f:forces.get())if(f.x!=0||f.y!=0||f.z!=0)throw std::runtime_error("faulted v10 changed force");
    Device<float> voltage(8*TILE_TOTAL_VERTICES);voltage.put(std::vector<float>(8*TILE_TOTAL_VERTICES,.5f));
    Device<GpuSpikeEventCanonical96> queue(8*TILE_CORE_VERTICES);rt(cudaMemset(queue.p,0xa5,queue.n*sizeof(GpuSpikeEventCanonical96)),"queue sentinel");const auto q0=queue.get();Device<uint32_t> qcount(1);qcount.zero();
    // Isolated candidate-layout buffers exercise the rejected-step write contract.
    // They are never passed to the incompatible production serializer.
    SignalGridSpecFfi grid={};
    grid.inv_voxel_spacing=.1f;
    for(unsigned k=0;k<3;++k){grid.origin_xyz[k]=origin[k];grid.dim_xyz[k]=uint32_t(dims[k]);}
    grid.total_voxels=uint32_t(ncells);
    Device<float> water(ncells),exclusion(ncells);
    Device<int32_t> hit(ncells),coupled(ncells),primary_id(ncells),primary_count(ncells),residue_ids(x.n);
    Device<uint32_t> protocol_step(1);
    Device<BiprismLifCarryoverFfi> carry(8);carry.zero();
    water.put(std::vector<float>(ncells,.125f));
    exclusion.put(std::vector<float>(ncells,.25f));
    coupled.zero();hit.zero();primary_count.zero();
    primary_id.put(std::vector<int32_t>(ncells,-1));
    residue_ids.put(x.residue_ids);
    protocol_step.put(std::vector<uint32_t>{23u});
    nhs_voronoi_dual_lif_and_4grid_bypass_v2<<<8,256>>>(
        table.p,tiles.p,conv.p,protocol_step.p,residue_ids.p,voltage.p,
        hit.p,coupled.p,primary_id.p,primary_count.p,water.p,exclusion.p,
        queue.p,qcount.p,grid,uint32_t(queue.n),.92f,1.25f,.04f,carry.p,1u);
    rt(cudaGetLastError(),"LIF fault refusal launch");rt(cudaDeviceSynchronize(),"LIF fault refusal completion");
    const auto q1=queue.get();for(float v:voltage.get())if(v!=.5f)throw std::runtime_error("faulted LIF changed voltage");
    if(qcount.get()[0]!=0||std::memcmp(q0.data(),q1.data(),q0.size()*sizeof(q0[0])))throw std::runtime_error("faulted LIF emitted event");
    for(auto f:forces.get())if(f.x!=0||f.y!=0||f.z!=0)throw std::runtime_error("faulted LIF changed force");
    for(float v:water.get())if(v!=.125f)throw std::runtime_error("faulted LIF changed water grid");
    for(float v:exclusion.get())if(v!=.25f)throw std::runtime_error("faulted LIF changed exclusion grid");
    for(int32_t v:coupled.get())if(v!=0)throw std::runtime_error("faulted LIF changed coupling grid");
    for(int32_t v:hit.get())if(v!=0)throw std::runtime_error("faulted LIF changed hit grid");
    for(int32_t v:primary_id.get())if(v!=-1)throw std::runtime_error("faulted LIF changed primary residue ID");
    for(int32_t v:primary_count.get())if(v!=0)throw std::runtime_error("faulted LIF changed primary residue count");
    if(protocol_step.get()[0]!=23u)throw std::runtime_error("LIF changed authoritative protocol clock");
    // A separate isolated observation window exercises the actual candidate
    // firing path after the fault-refusal checks. It starts with zero forces
    // and zero counts, and never supplies this packet to a native consumer.
    for (unsigned r=0;r<8;++r) {
        records[r]={};records[r].abi_magic=BIPRISM_TAIL_ABI_MAGIC;records[r].leaf_id=r;
        records[r].w_x_lambda_trace=2.f;records[r].committed_substeps=4u;
    }
    conv.put(records);
    std::vector<float> firing_voltage(8*TILE_TOTAL_VERTICES,0.f);
    for(unsigned r=0;r<8;++r)for(unsigned u=0;u<TILE_CORE_VERTICES;++u)firing_voltage[r*TILE_TOTAL_VERTICES+u]=3.9f;
    voltage.put(firing_voltage);forces.zero();qcount.zero();carry.zero();
    water.zero();exclusion.zero();hit.zero();coupled.zero();primary_count.zero();
    primary_id.put(std::vector<int32_t>(ncells,-1));
    const auto force_before=forces.get();
    size_t boundary_edges=0;
    for(const auto&t:t_end)for(unsigned e=0;e<t.num_active_edges;++e)
        if((t.edge_u8[e][0]<TILE_CORE_VERTICES)!=(t.edge_u8[e][1]<TILE_CORE_VERTICES))++boundary_edges;
    if(!boundary_edges)throw std::runtime_error("native LIF diagnostic has no core-halo edges to exercise");
    nhs_voronoi_dual_lif_and_4grid_bypass_v2<<<8,256>>>(
        table.p,tiles.p,conv.p,protocol_step.p,residue_ids.p,voltage.p,
        hit.p,coupled.p,primary_id.p,primary_count.p,water.p,exclusion.p,
        queue.p,qcount.p,grid,uint32_t(queue.n),1.f,1.25f,.04f,carry.p,1u);
    rt(cudaGetLastError(),"LIF multiplicity launch");rt(cudaDeviceSynchronize(),"LIF multiplicity completion");
    const uint32_t emitted=qcount.get()[0];
    if(!emitted)throw std::runtime_error("native LIF positive-path diagnostic emitted no records");
    // Retain actual emitted boundary arrays even when the unchanged force gate
    // refuses. This separates Q39->FP32 conversion from force-buffer addition.
    const auto carry_after = carry.get();
    const auto forces_after = forces.get();
    const auto queue_after = queue.get();
    const auto dump_boundary = [&](const char* suffix, const void* data, size_t bytes) {
        std::ofstream out(std::string(tile_dump_path) + suffix, std::ios::binary);
        out.write(reinterpret_cast<const char*>(data), bytes);
        if (!out) throw std::runtime_error("LIF boundary evidence write failed");
    };
    dump_boundary(".lif_carry.bin", carry_after.data(), carry_after.size()*sizeof(carry_after[0]));
    dump_boundary(".forces_after.bin", forces_after.data(), forces_after.size()*sizeof(forces_after[0]));
    dump_boundary(".lif_queue.bin", queue_after.data(), emitted*sizeof(queue_after[0]));
    const auto lif_report=biprism_qc::validate_lif_window(queue_after,emitted,voltage.get(),t_end,records,x.residue_ids,
        c_end,force_before,forces_after,grid,hit.get(),coupled.get(),primary_id.get(),primary_count.get(),23,1.25f,true);
    std::printf("native_lif_records=%u multiplicity=%llu core_halo_edges=%zu expanded_ledger_error=%.12Lg plain_ledger_error=%.12Lg force_sum_change=%.12Lg\n",
        emitted,static_cast<unsigned long long>(lif_report.total_multiplicity),boundary_edges,
        lif_report.max_expanded_ledger_error,lif_report.max_single_float_ledger_error,lif_report.max_force_sum_change);
    if(!report("native_lif_readback",lif_report.contract))throw std::runtime_error("native LIF readback gate");
    if(protocol_step.get()[0]!=23u)throw std::runtime_error("positive LIF path changed authoritative protocol clock");
    // Actual delivery refusal on this HMR input: no physical inverse masses
    // were supplied, so no velocity mutation or pending clear is admissible.
    if (x.hmr) {
        protocol_step.put(std::vector<uint32_t>{24u});
        for (auto& r:records) r.committed_substeps=8u;
        conv.put(records);
        const auto velocity_before=vel.get();
        nhs_deliver_persisted_lif_forces_v1<<<1,256>>>(table.p,carry.p,conv.p,protocol_step.p,x.n,1.f);
        rt(cudaGetLastError(),"LIF missing-physical-mass refusal launch");
        rt(cudaDeviceSynchronize(),"LIF missing-physical-mass refusal completion");
        const auto velocity_after=vel.get();
        if (std::memcmp(velocity_before.data(),velocity_after.data(),velocity_before.size()*sizeof(float3)))
            throw std::runtime_error("mass-refused LIF delivery changed velocity");
        for (const auto& r:carry.get()) if(r.pending!=1u || !(r.fault&BIPRISM_CARRY_BAD_INPUT))
            throw std::runtime_error("mass-refused LIF delivery lost pending state or fault evidence");
        std::puts("native_lif_missing_true_mass_refusal=1 positive_velocity_delivery=0");
    }
    std::puts("native_diagnostic_passed=1 scope=seed_Phase2_fault_refusal_and_isolated_emitted_record_consistency whole_window_accepted=0 trajectory_accepted=0 production_consumer_accepted=0");
    return 0;
}
}
int main(int argc,char** argv) {
    try {
        if(argc==2 && std::string(argv[1])=="--host-ledger-controls")return host_ledger_controls();
        if(argc<3)throw std::runtime_error("usage: --host-ledger-controls | --inspect CASE | --native CASE ACTUAL_NEIGHBOR_MODULE TILE_DUMP (canonical admission required)");
        const auto input=read_input(argv[2]);
        std::printf("input_atoms=%u center=%u constraints=%zu hmr=%u n_residues=%u tile_bytes=%zu main_smem=%zu Phase2_smem=%zu seeder_smem=%zu\n",input.n,input.center,input.groups.size(),input.hmr,input.n_residues,sizeof(ContactShellSimplicialStateFfi),sizeof(VariationalClusterDsmemV10),sizeof(SimplicialTilePhase2Smem),sizeof(SimplicialSeederSmem));
        const bool abi=report("canonical_spike_consumer",biprism_qc::validate_candidate_spike_abi());
        const bool grid=inspect_signal_grid_contract();
        if(std::string(argv[1])=="--inspect") {std::puts("gpu_execution=0");return abi&&grid?0:3;}
        if(std::string(argv[1])!="--native"||argc!=5)throw std::runtime_error("invalid diagnostic mode");
        // Mismatched event ABI prevents normal event delivery. Native tests
        // use isolated candidate storage and never call a production consumer.
        return native(input,argv[3],argv[4]);
    } catch(const std::exception& e) {std::fprintf(stderr,"diagnostic_failed: %s\n",e.what());return 1;}
}
