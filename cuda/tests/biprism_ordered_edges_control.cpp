#include "../../crates/prism-gpu/src/kernels/nhs_biprism_ordered_edges.cuh"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>
#if defined(BIPRISM_TEST_ACTUAL_DEC_ROWS)
#include "biprism_actual_dec_rows.hpp"
#endif

using Edge = std::array<uint8_t, 2>;
static size_t checks = 0;
static void require(bool value, const char* label) {
    ++checks;
    if (!value) throw std::runtime_error(label);
}
static uint64_t bits(double value) {
    uint64_t out; std::memcpy(&out, &value, sizeof(out)); return out;
}
static BiprismOrderedEdgesFfi make_index(const std::vector<Edge>& edges, uint32_t epoch=4u) {
    BiprismOrderedEdgesFfi index{};
    index.magic=BP_ORDERED_EDGES_MAGIC; index.version=BP_ORDERED_EDGES_VERSION;
    index.leaf=3u; index.tile_address=0x10000u; index.accepted_substeps_at_entry=epoch;
    index.num_edges=static_cast<uint32_t>(edges.size());
    index.publication_state=BP_ORDERED_EDGES_BUILDING;
    const auto* data=reinterpret_cast<const uint8_t*>(edges.data());
    for(uint32_t v=0;v<192;++v)
        index.offsets[v+1]=biprism_ordered_edge_count_row(data,data+1,2,index.num_edges,v);
    require(biprism_ordered_edge_prefix(index),"prefix capacity");
    for(uint32_t v=0;v<192;++v)
        require(biprism_ordered_edge_fill_row(index,data,data+1,2,v),"complete row fill");
    index.publication_state=BP_ORDERED_EDGES_READY;
    return index;
}
static void compare(const std::vector<Edge>& edges) {
    auto index=make_index(edges);
    const auto* data=reinterpret_cast<const uint8_t*>(edges.data());
    require(biprism_ordered_edges_identity_matches(index,3,0x10000,edges.size(),4),"identity");
    std::array<std::vector<uint16_t>,192> independent;
    for(size_t e=0;e<edges.size();++e) {
        const auto u=edges[e][0], v=edges[e][1];
        if(u<192) independent[u].push_back(e);
        if(v<192 && v!=u) independent[v].push_back(e);
    }
    std::vector<float> weights(edges.size());
    std::array<float,256> values{};
    for(size_t e=0;e<edges.size();++e) weights[e]=std::ldexp(float(1+(e%31)),int(e%35)-17);
    for(size_t v=0;v<values.size();++v) values[v]=std::ldexp(float(int(v%29)-14),int(v%19)-9);
    for(uint32_t v=0;v<192;++v) {
        require(biprism_ordered_edge_row_matches(index,data,data+1,2,v),"row exact completeness");
        const std::vector<uint16_t> actual(index.edge_index+index.offsets[v],index.edge_index+index.offsets[v+1]);
        require(actual==independent[v],"independent original ordinal population");
        double linear=0.0,indexed=0.0,diag_linear=1e-4,diag_indexed=1e-4;
        std::vector<uint16_t> outgoing_linear,outgoing_indexed;
        const double own=double(values[v]);
        for(uint32_t e=0;e<edges.size();++e) {
            const uint32_t u=edges[e][0], w=edges[e][1];
            if(u==v) linear+=double(weights[e])*(own-double(values[w]));
            else if(w==v) linear+=double(weights[e])*(own-double(values[u]));
            if(u==v||w==v) diag_linear+=double(weights[e]);
            if(u==v) outgoing_linear.push_back(e);
        }
        for(uint32_t k=index.offsets[v];k<index.offsets[v+1];++k) {
            const uint32_t e=index.edge_index[k],u=edges[e][0],w=edges[e][1];
            if(u==v) indexed+=double(weights[e])*(own-double(values[w]));
            else if(w==v) indexed+=double(weights[e])*(own-double(values[u]));
            if(u==v||w==v) diag_indexed+=double(weights[e]);
            if(u==v) outgoing_indexed.push_back(e);
        }
        require(bits(linear)==bits(indexed),"FP64 row terms and accumulation order");
        require(bits(diag_linear)==bits(diag_indexed),"FP64 diagonal accumulation order");
        require(outgoing_linear==outgoing_indexed,"outgoing provider call order/direction");
#if defined(BIPRISM_TEST_ACTUAL_DEC_ROWS)
        const auto actual_linear=biprism_dec_row_f64_accessor(data,data+1,2,
            BiprismDecArrayWeights{weights.data()},edges.size(),values.data(),v);
        const auto actual_indexed=biprism_dec_row_f64_traversal(data,data+1,2,
            BiprismDecArrayWeights{weights.data()},BiprismIncidentEdgeTraversal{&index},values.data(),v);
        require(bits(actual_linear)==bits(linear),"actual DEC linear row unchanged");
        require(bits(actual_indexed)==bits(linear),"actual DEC indexed row bitwise unchanged");
#endif
    }
    auto bad=index;
    bad.accepted_substeps_at_entry=8;
    require(!biprism_ordered_edges_identity_matches(bad,3,0x10000,edges.size(),4),"stale invocation identity");
    bad=index;bad.publication_state=BP_ORDERED_EDGES_BUILDING;
    require(!biprism_ordered_edges_identity_matches(bad,3,0x10000,edges.size(),4),"unpublished row refusal");
    bad=index;bad.tile_address+=128;
    require(!biprism_ordered_edges_identity_matches(bad,3,0x10000,edges.size(),4),"foreign tile refusal");
    for(uint32_t v=0;v<192;++v) if(independent[v].size()>1) {
        bad=index;std::swap(bad.edge_index[bad.offsets[v]],bad.edge_index[bad.offsets[v]+1]);
        require(!biprism_ordered_edge_row_matches(bad,data,data+1,2,v),"same-membership reordered row refusal");
        bad=index;--bad.offsets[v+1];
        require(!biprism_ordered_edge_row_matches(bad,data,data+1,2,v),"dropped incident edge refusal");
        break;
    }
    bad=index;bad.offsets[1]=3073;
    require(!biprism_ordered_edge_row_matches(bad,data,data+1,2,0),"overflow row refusal");
    auto rebuilt=make_index(edges,8);
    require(biprism_ordered_edges_identity_matches(rebuilt,3,0x10000,edges.size(),8),"fresh next invocation");
}
int main(int argc,char**argv) {
    try {
        compare({Edge{255,0},Edge{192,255},Edge{1,0},Edge{191,2},Edge{0,3}});
        compare({Edge{200,201}}); // Every core row is empty.
        std::vector<Edge> all_core;
        for(uint32_t u=0;u<192u&&all_core.size()<1536u;++u)
            for(uint32_t v=u+1u;v<192u&&all_core.size()<1536u;++v)
                all_core.push_back({static_cast<uint8_t>(u),static_cast<uint8_t>(v)});
        compare(all_core);
        auto full=make_index(all_core);
        require(full.incident_count==3072u,"maximum incident capacity retained");
        full.offsets[1]=1537u;
        require(!biprism_ordered_edge_prefix(full),"impossible row count refuses prefix");
        std::mt19937 rng(0xB1E1u);
        for(uint32_t count : {1u,31u,256u,1461u,1536u}) {
            std::set<std::pair<uint8_t,uint8_t>> seen;
            std::vector<Edge> edges;
            while(edges.size()<count) {
                uint8_t u=rng()%256,v=rng()%256;
                if(u==v||!seen.emplace(std::min(u,v),std::max(u,v)).second)continue;
                edges.push_back({u,v}); // Arbitrary direction and original ordinal order.
            }
            compare(edges);
            std::reverse(edges.begin(),edges.end());compare(edges);
        }
        // The retained native tile image is optional; no synthetic image is
        // substituted when it was not supplied. Each supplied file must be all8.
        for(int arg=1;arg<argc;++arg) {
            std::ifstream input(argv[arg],std::ios::binary);
            std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)),{});
            require(bytes.size()==8u*225408u,"native all-eight tile extent");
            for(size_t leaf=0;leaf<8;++leaf) {
                const auto* tile=bytes.data()+leaf*225408u;
                uint32_t count;std::memcpy(&count,tile+225284u,4);
                require(count>0&&count<=1536,"native edge count");
                std::vector<Edge> edges(count);
                std::memcpy(edges.data(),tile+29696u,count*2u);
                compare(edges);
            }
        }
        std::printf("ordered edge integer/sequence controls PASS: %zu assertions; native tile files=%d\n",checks,argc-1);
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
