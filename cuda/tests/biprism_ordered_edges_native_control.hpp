#pragma once
// Included after the real main/DEC headers and the diagnostic Device/rt owners.
// Component arithmetic controls use actual retained tile, weights and ordinal
// population. Every solver plane and corrupted index is a private allocation.
// No physical buffer, accepted counter, production scratch or main timer changes.
namespace biprism_ordered_native {
struct Result {
    double row_linear,row_indexed,diagonal_linear,diagonal_indexed;
    double phi_linear,phi_indexed;
    float residual_linear,residual_indexed;
    uint32_t iterations_linear,iterations_indexed,success_linear,success_indexed;
};
static_assert(sizeof(Result)==72,"ordered FP64-potential arithmetic control row");
static_assert(offsetof(Result,phi_linear)==32&&offsetof(Result,residual_linear)==48
    &&offsetof(Result,iterations_linear)==56,"ordered arithmetic control offsets");

// The diagnostic runs after the attempted main, including controlled refusals.
// Accepted output uses the accepted bank; a refused prepared body retains its
// prepared input. Neither case is silently reconstructed from the FP32 tile.
__host__ __device__ inline bool source_epoch_valid(const BiprismPotentialOwnerFfi& o,uint32_t entry) {
    if(entry>UINT32_MAX-4u)return false;
    if(o.state==BP_POTENTIAL_ACCEPTED)
        return (o.accepted_committed_substeps==entry||o.accepted_committed_substeps==entry+4u)
            &&bp_potential_accepted_valid(o,o.accepted_committed_substeps,o.accepted_step);
    return bp_potential_prepared_valid(o,entry,o.prepared_step);
}

__global__ void compare_arithmetic(const FoliatedStreamPtrTableFfi* streams,
    const ContactShellSimplicialStateFfi* tiles,const float* amber,
    uint32_t cold_guess,Result* output) {
    __shared__ double phi[256];
    __shared__ float residual[256],z[256],direction[256],product[256],diagonal[256],reduction[256];
    const uint32_t leaf=blockIdx.x,tid=threadIdx.x;
    const auto* tile=tiles+leaf;
    const auto* index=streams->d_ordered_edges+leaf;
    const auto* weights=streams->d_electronic_exchange[leaf]->live_weights;
    const auto* potential=streams->d_potential_owners?streams->d_potential_owners+leaf:nullptr;
    const auto* coordinate=streams->d_coordinate_owners?streams->d_coordinate_owners+leaf:nullptr;
    bool source_valid=bp_potential_owner_valid(potential,leaf,tile,coordinate)&&streams->d_spinor_owners;
    if(source_valid)source_valid=source_epoch_valid(*potential,index->accepted_substeps_at_entry)
        &&bp_potential_projection_value_valid(*potential,tid,tile->phi_dec[tid]);
    if(!__syncthreads_and(source_valid)) {
        Result refused{};refused.row_linear=refused.row_indexed=CUDART_NAN;
        refused.diagonal_linear=refused.diagonal_indexed=CUDART_NAN;
        refused.phi_linear=refused.phi_indexed=CUDART_NAN;
        refused.residual_linear=refused.residual_indexed=CUDART_INF_F;
        output[leaf*256u+tid]=refused;return;
    }
    const auto b=tid<192u?streams->d_spinor_owners[leaf].accepted[tid].s1:BiprismElectronicComplex<double>{0.,0.};
    const double q=tid<192u?__fma_rn(double(.15f),
        __fma_rn(double(b.re),double(b.re),double(b.im)*double(b.im)),
        double(amber[UNPACK_GLOBAL_ATOM_ID(tile->global_atom_idx[tid])])):0.;
    const double initial=cold_guess&&tid<192u?0.:bp_potential_input_phi(*potential)[tid];
    const auto weight=BiprismDecArrayWeightsT<double>{weights};
    phi[tid]=initial;residual[tid]=z[tid]=direction[tid]=product[tid]=diagonal[tid]=reduction[tid]=0.f;
    __syncthreads();
    Result result{};
    result.row_linear=biprism_dec_row_f64_traversal(&tile->edge_u8[0][0],&tile->edge_u8[0][1],2u,
        weight,BiprismLinearEdgeTraversal{tile->num_active_edges},phi,tid);
    result.row_indexed=biprism_dec_row_f64_traversal(&tile->edge_u8[0][0],&tile->edge_u8[0][1],2u,
        weight,BiprismIncidentEdgeTraversal{index},phi,tid);
    result.diagonal_linear=result.diagonal_indexed=1e-4;
    if(tid<192u){
        for(uint32_t e=0;e<tile->num_active_edges;++e)
            if(tile->edge_u8[e][0]==tid||tile->edge_u8[e][1]==tid)
                result.diagonal_linear+=double(weights[e]);
        for(uint32_t k=index->offsets[tid];k<index->offsets[tid+1u];++k){
            const uint32_t e=index->edge_index[k];
            if(tile->edge_u8[e][0]==tid||tile->edge_u8[e][1]==tid)
                result.diagonal_indexed+=double(weights[e]);
        }
    }
    __syncthreads();
    result.success_linear=biprism_dec_pcg_256_f64_accessor(&tile->edge_u8[0][0],&tile->edge_u8[0][1],2u,
        weight,tile->num_active_edges,q,phi,residual,z,direction,product,diagonal,reduction,
        result.residual_linear,result.iterations_linear);
    result.phi_linear=phi[tid];
    __syncthreads();
    phi[tid]=initial;residual[tid]=z[tid]=direction[tid]=product[tid]=diagonal[tid]=reduction[tid]=0.f;
    __syncthreads();
    result.success_indexed=biprism_dec_pcg_256_f64_indexed_accessor(&tile->edge_u8[0][0],&tile->edge_u8[0][1],2u,
        weight,tile->num_active_edges,index,q,phi,residual,z,direction,product,diagonal,reduction,
        result.residual_indexed,result.iterations_indexed);
    result.phi_indexed=phi[tid];
    output[leaf*256u+tid]=result;
}

__global__ void validate_index(const BiprismOrderedEdgesFfi* indices,
    const BiprismOrderedEdgesFfi* expected,const ContactShellSimplicialStateFfi* tiles,uint32_t* output) {
    const uint32_t leaf=blockIdx.x,tid=threadIdx.x;
    const auto& index=indices[leaf];const auto& wanted=expected[leaf];
    bool valid=biprism_ordered_edges_identity_matches(index,leaf,reinterpret_cast<uint64_t>(tiles+leaf),
        tiles[leaf].num_active_edges,wanted.accepted_substeps_at_entry);
    if(valid&&tid<192u) valid=biprism_ordered_edge_row_matches(index,
        &tiles[leaf].edge_u8[0][0],&tiles[leaf].edge_u8[0][1],2u,tid);
    const bool all=__syncthreads_and(valid);
    if(tid==0u)output[leaf]=all;
}

template<class T> inline bool same_bits(const T&a,const T&b){return std::memcmp(&a,&b,sizeof(T))==0;}
template<class T> inline void save(const std::string& path,const std::vector<T>& values){
    std::ofstream file(path,std::ios::binary);
    file.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(T));
    if(!file)throw std::runtime_error("ordered native control retention failed");
}
} // namespace biprism_ordered_native

inline bool run_biprism_ordered_edge_native_control(const FoliatedStreamPtrTableFfi* streams,
    const ContactShellSimplicialStateFfi* tiles,const float* amber,const std::string& prefix){
    using namespace biprism_ordered_native;
    FoliatedStreamPtrTableFfi table{};
    rt(cudaMemcpy(&table,streams,sizeof(table),cudaMemcpyDeviceToHost),"ordered control table");
    if(!table.d_ordered_edges||!table.d_potential_owners||!table.d_coordinate_owners)
        throw std::runtime_error("ordered control missing actual index/potential/coordinate owner");
    std::vector<BiprismPotentialOwnerFfi> potential(8);
    rt(cudaMemcpy(potential.data(),table.d_potential_owners,potential.size()*sizeof(potential[0]),cudaMemcpyDeviceToHost),"ordered potential owner input");
    auto read_potential_banks=[&](){
        std::vector<double> banks(2u*8u*256u);
        for(uint32_t leaf=0;leaf<8u;++leaf){
            const auto& o=potential[leaf];
            if(!o.accepted_phi||!o.prepared_phi||o.accepted_phi==o.prepared_phi)
                throw std::runtime_error("ordered potential bank pointer domain");
            rt(cudaMemcpy(banks.data()+leaf*256u,o.accepted_phi,256u*sizeof(double),cudaMemcpyDeviceToHost),"ordered accepted phi input");
            rt(cudaMemcpy(banks.data()+(8u+leaf)*256u,o.prepared_phi,256u*sizeof(double),cudaMemcpyDeviceToHost),"ordered prepared phi input");
        }
        return banks;
    };
    const auto potential_banks=read_potential_banks();
    save(prefix+".potential.owners.128b",potential);
    save(prefix+".potential.banks.accepted_then_prepared.f64le",potential_banks);
    save(prefix+".schema.u32le",std::vector<uint32_t>{0x3250444fu,2u,72u,8u,4u,8u,256u,0u});
    std::vector<BiprismOrderedEdgesFfi> actual(8);
    rt(cudaMemcpy(actual.data(),table.d_ordered_edges,actual.size()*sizeof(actual[0]),cudaMemcpyDeviceToHost),"ordered control input");
    Device<BiprismOrderedEdgesFfi> expected(8),corrupted(8);expected.put(actual);corrupted.put(actual);
    Device<uint32_t> status(8);status.zero();
    validate_index<<<8,256>>>(corrupted.p,expected.p,tiles,status.p);
    rt(cudaGetLastError(),"ordered baseline validation");rt(cudaDeviceSynchronize(),"ordered baseline validation completion");
    for(auto accepted:status.get())if(accepted!=1u)throw std::runtime_error("actual ordered index failed baseline validation");
    Device<Result> results(8u*256u);
    bool arithmetic_passed=true,negative_passed=true;uint32_t success[2]={0u,0u};
    for(uint32_t guess=0;guess<2u;++guess){
        compare_arithmetic<<<8,256>>>(streams,tiles,amber,guess,results.p);
        rt(cudaGetLastError(),"ordered linear/indexed arithmetic");rt(cudaDeviceSynchronize(),"ordered arithmetic completion");
        const auto rows=results.get();save(prefix+".guess"+std::to_string(guess)+".rows72.bin",rows);
        uint32_t mismatches=0u;
        for(uint32_t leaf=0;leaf<8u;++leaf){
            success[guess]+=rows[leaf*256u].success_linear!=0u;
            for(uint32_t v=0;v<256u;++v){const auto&r=rows[leaf*256u+v];
                const bool exact=std::isfinite(r.row_linear)&&std::isfinite(r.diagonal_linear)&&std::isfinite(r.phi_linear)
                    &&(!r.success_linear||std::isfinite(r.residual_linear))
                    &&same_bits(r.row_linear,r.row_indexed)&&same_bits(r.diagonal_linear,r.diagonal_indexed)
                    &&same_bits(r.phi_linear,r.phi_indexed)&&same_bits(r.residual_linear,r.residual_indexed)
                    &&r.iterations_linear==r.iterations_indexed&&r.success_linear==r.success_indexed;
                mismatches+=!exact;
            }
        }
        std::printf("ordered_native_arithmetic guess=%u all8_rows=%u bit_mismatches=%u linear_solved_leaves=%u private_solver_state=1 potential_bits=64 row_bytes=72 krylov_bits=32\n",
            guess,unsigned(rows.size()),mismatches,success[guess]);
        arithmetic_passed=arithmetic_passed&&mismatches==0u;
    }
    // Corrupt only private copies. The original device CSR and physical state
    // are unchanged; every non-target leaf must still validate.
    std::vector<uint32_t> negative_status;
    for(uint32_t control=0;control<5u;++control)for(uint32_t leaf=0;leaf<8u;++leaf){
        auto changed=actual;auto&index=changed[leaf];
        uint32_t vertex=0u;
        while(vertex<192u&&index.offsets[vertex+1u]-index.offsets[vertex]<2u)++vertex;
        if(vertex==192u)throw std::runtime_error("actual CSR lacks degree-two row for negative control");
        const uint32_t first=index.offsets[vertex];
        if(control==0u)std::swap(index.edge_index[first],index.edge_index[first+1u]);
        if(control==1u)index.edge_index[first]=index.edge_index[first+1u];
        if(control==2u)--index.offsets[vertex+1u];
        if(control==3u)index.accepted_substeps_at_entry^=4u;
        if(control==4u)index.publication_state=BP_ORDERED_EDGES_BUILDING;
        corrupted.put(changed);status.zero();
        validate_index<<<8,256>>>(corrupted.p,expected.p,tiles,status.p);
        rt(cudaGetLastError(),"ordered corrupt-index validation");rt(cudaDeviceSynchronize(),"ordered corrupt-index completion");
        const auto got=status.get();negative_status.insert(negative_status.end(),got.begin(),got.end());
        for(uint32_t other=0;other<8u;++other)negative_passed=negative_passed&&got[other]==uint32_t(other!=leaf);
    }
    save(prefix+".negative_status.u32le",negative_status);
    std::vector<BiprismOrderedEdgesFfi> unchanged(8);
    rt(cudaMemcpy(unchanged.data(),table.d_ordered_edges,unchanged.size()*sizeof(unchanged[0]),cudaMemcpyDeviceToHost),"ordered unchanged input");
    const bool retained=std::memcmp(actual.data(),unchanged.data(),actual.size()*sizeof(actual[0]))==0;
    std::vector<BiprismPotentialOwnerFfi> potential_after(8);
    rt(cudaMemcpy(potential_after.data(),table.d_potential_owners,potential_after.size()*sizeof(potential_after[0]),cudaMemcpyDeviceToHost),"ordered unchanged potential owner");
    const auto banks_after=read_potential_banks();
    const bool potential_retained=std::memcmp(potential.data(),potential_after.data(),potential.size()*sizeof(potential[0]))==0
        &&std::memcmp(potential_banks.data(),banks_after.data(),potential_banks.size()*sizeof(double))==0;
    const bool passed=arithmetic_passed&&negative_passed&&retained&&potential_retained;
    std::printf("ordered_native_potential_owner_unchanged=%u row_schema=ODP2_72 potential_bits=64\n",unsigned(potential_retained));
    std::printf("ordered_native_control pass=%u exact_rows_diagonal_PCG=%u corrupt_index_controls=40 corrupt_controls_passed=%u actual_index_unchanged=%u main_timing_excludes_controls=1\n",
        unsigned(passed),unsigned(arithmetic_passed),unsigned(negative_passed),unsigned(retained));
    return passed;
}
