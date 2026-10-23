#pragma once
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <string.h>

// New explicitly requested mixed-Hamiltonian acceptance owner. Native AMBER
// energy/work planes and 208/160-byte native governor records are untouched.
// Ordinary constraints, bridge rounding and lagged halo motion are NOT work.
static constexpr uint32_t BIPRISM_MIXED_MAGIC=0x424d4536u;
static constexpr uint32_t BIPRISM_MIXED_TERMINAL=0x01000000u;
static constexpr double BIPRISM_MIXED_ABSOLUTE_TOL=1e-6;
enum BiprismMixedStage:uint32_t {BP_MIX_START=0,BP_MIX_PRE_LIF=1,BP_MIX_POST_LIF=2,
    BP_MIX_POST_NONHAM=3,BP_MIX_POST_CA=4,BP_MIX_END=5,BP_MIX_HALO_COUNTERFACTUAL=6,BP_MIX_STAGES=7};
enum BiprismMixedFault:uint32_t {BP_MIX_DOMAIN=1,BP_MIX_EPOCH=2,BP_MIX_SOLVE=4,
    BP_MIX_NONFINITE=8,BP_MIX_REFERENCE=16,BP_MIX_ENERGY=32,BP_MIX_WORK=64};
struct alignas(16) BiprismMixedBindingFfi {
    const float* native_hmr_masses;
    const double* native_potential;
    const double* native_atom_work;
    const double* native_cumulative_work;
    uint32_t atoms,source;
    uint64_t reserved;
};
struct alignas(16) BiprismMixedVertexFfi {
    uint32_t packed_atom,leaf,local_vertex,stage;
    float native_x[3],native_v[3],x[3],v[3];
    float reference[3];
    uint32_t potential_alignment;
    double phi;
    float psi[4],omega,amber_charge,true_inverse_mass,propagation_inverse_mass,main_mass,native_hmr_mass;
    float metric[6];
    double charge,rho00,rho11,population,kinetic_metric,kinetic_native_core;
    float native_x_lo[3],x_lo[3],reference_lo[3];
    uint32_t potential_entry_alignment;
    double phi_entry;
    uint32_t coordinate_reserved[2];
    double metric_f64[6];
    double psi_precise[4];
};
struct alignas(16) BiprismMixedEdgeFfi {
    uint32_t leaf,edge,u,v;
    float base,reserved;
    double live;
    double E0,E1,V,S;
};
// values: U_native[kcal],K_native_all[kcal],K_native_core[kcal],K_g_core[Ha],
// onsite,edge,DEC,ring_anchor,owned_ring_pair,model_potential,C,E_mix[Ha],
// cumulative_native_work[kcal],current_atom_work[kcal],true_DEC_residual,
// core_spinor_norm_sum,K_g_true_mass[Ha],K_native_bath[kcal],K_lab_true_core[Ha],
// private_DEC_entry_refinement_delta[Ha],reserved[4]. The refinement diagnostic
// is excluded from physical energy/work credit. No virtual counterfactual is a
// physical endpoint.
struct alignas(16) BiprismMixedSampleFfi {
    uint32_t magic,leaf,stage,state,scheduled_step,committed_substeps,mode,sticky_fault;
    uint64_t generation,body_ordinal,preparation_generation;
    uint32_t dec_iterations,source;
    double values[24];
};
// values: native_declared_work,phase2_parameter_work,LIF_correction,
// postUV_nonham_correction,CA_correction,total_declared_work,energy_delta,
// local_residual,halo_geometry_counterfactual_delta,initial_parameter_prefix,
// LIF_native_work_identity,previous_energy,current_energy,previous_kinetic,
// current_kinetic,body_start_energy,uncredited_start_to_preLIF_C_change,
// uncredited_postCA_to_end_C_change,reserved[6]. Local residual is NOT gated;
// coupling transfers cancel only in the complete eight-leaf sum.
struct alignas(16) BiprismMixedLeafLedgerFfi {
    uint32_t magic,leaf,state,sticky_fault,scheduled_step,committed_substeps,mode,source;
    uint64_t generation,body_ordinal,preparation_generation,reserved;
    double values[24];
};
// values: initial_E,previous_E,current_E,declared_work,body_residual,
// cumulative_work,expected_E,run_residual,previous_K,current_K,
// initial_parameter_prefix,max_abs_body,max_abs_run,absolute_tolerance,
// sum_initial_parameter_prefix,reserved[9].
struct alignas(16) BiprismMixedGlobalLedgerFfi {
    uint32_t magic,state,sticky_fault,scheduled_step,committed_substeps,stage_mask,reserved[2];
    uint64_t generation,body_ordinal,preparation_generation,accepted_bodies;
    double values[24];
};
// Persistent accepted reference. It advances only inside the final collective
// guard after every existing owner is accepted; never reset at chunk boundaries.
struct alignas(16) BiprismMixedEnergyStateFfi {
    uint32_t magic,state,sticky_fault,last_accepted_step;
    uint64_t generation,accepted_bodies,preparation_generation,reserved[3];
    double totals[16];
    double previous_native_work[8],previous_leaf_energy[8],previous_leaf_kinetic[8];
    double reserved_tail[16];
};
struct alignas(16) BiprismMixedWindowFfi {
    uint32_t magic,capacity,first_body,requested_bodies,first_step,reserved;
    uint64_t generation,preparation_generation,reserved_tail;
};
static_assert(sizeof(BiprismMixedBindingFfi)==48,"mixed native binding");
static_assert(sizeof(BiprismMixedVertexFfi)==336,"BME6 mixed primitive vertex");
static_assert(offsetof(BiprismMixedVertexFfi,phi)==80,"BME4 full potential ABI");
static_assert(offsetof(BiprismMixedVertexFfi,native_x_lo)==200,"BME4 position expansion ABI");
static_assert(offsetof(BiprismMixedVertexFfi,charge)==152,"BME4 quadratic primitives");
static_assert(offsetof(BiprismMixedVertexFfi,phi_entry)==240,"BME4 entry potential ABI");
static_assert(offsetof(BiprismMixedVertexFfi,psi_precise)==304,"BME6 FP64 spinor");
static_assert(offsetof(BiprismMixedEdgeFfi,live)==24&&offsetof(BiprismMixedEdgeFfi,E0)==32,"BME6 full live-weight ABI");
static_assert(sizeof(BiprismMixedEdgeFfi)==64,"mixed primitive edge");
static_assert(sizeof(BiprismMixedSampleFfi)==256,"mixed stage sample");
static_assert(sizeof(BiprismMixedLeafLedgerFfi)==256,"mixed leaf ledger");
static_assert(sizeof(BiprismMixedGlobalLedgerFfi)==256,"mixed global ledger");
static_assert(sizeof(BiprismMixedEnergyStateFfi)==512,"mixed accepted reference");
static_assert(sizeof(BiprismMixedWindowFfi)==48,"mixed window");
#if defined(__CUDACC__)
#define BP_MIX_HD __host__ __device__
#else
#define BP_MIX_HD
#endif
BP_MIX_HD inline bool biprism_mixed_slot(const BiprismMixedWindowFfi& w,uint64_t body,uint32_t step,uint32_t& slot) {
    if(w.magic!=BIPRISM_MIXED_MAGIC||!w.capacity||w.capacity>500u||!w.requested_bodies||w.requested_bodies>w.capacity
        ||!w.generation||!w.preparation_generation||w.reserved||w.reserved_tail||body<w.first_body
        ||body-w.first_body>=w.requested_bodies||uint64_t(w.first_step)+body-w.first_body>=UINT32_MAX
        ||step!=uint64_t(w.first_step)+body-w.first_body)return false;
    slot=uint32_t(body-w.first_body);return true;
}
BP_MIX_HD inline bool biprism_mixed_energy_admitted(double body,double run) {
    return isfinite(body)&&isfinite(run)&&fabs(body)<=BIPRISM_MIXED_ABSOLUTE_TOL&&fabs(run)<=BIPRISM_MIXED_ABSOLUTE_TOL;
}
BP_MIX_HD inline bool biprism_mixed_same_float_bits(float a,float b) {
#if defined(__CUDA_ARCH__)
    return __float_as_uint(a)==__float_as_uint(b);
#else
    uint32_t x,y;memcpy(&x,&a,sizeof(x));memcpy(&y,&b,sizeof(y));return x==y;
#endif
}
// Metric realization depends on all selected positions, fixed references and
// ordered edge incidence. Velocities, masses and electronic state are sampled
// afresh. The metric formula and chi=.12f are part of this kernel's source pin.
BP_MIX_HD inline bool biprism_mixed_same_vertex_geometry(const BiprismMixedVertexFfi& a,
    const BiprismMixedVertexFfi& b) {
    if(a.packed_atom!=b.packed_atom||a.leaf!=b.leaf||a.local_vertex!=b.local_vertex)return false;
    for(uint32_t k=0;k<3u;++k)if(!biprism_mixed_same_float_bits(a.native_x[k],b.native_x[k])
        ||!biprism_mixed_same_float_bits(a.x[k],b.x[k])
        ||!biprism_mixed_same_float_bits(a.reference[k],b.reference[k])
        ||!biprism_mixed_same_float_bits(a.native_x_lo[k],b.native_x_lo[k])
        ||!biprism_mixed_same_float_bits(a.x_lo[k],b.x_lo[k])
        ||!biprism_mixed_same_float_bits(a.reference_lo[k],b.reference_lo[k]))return false;
    return true;
}
BP_MIX_HD inline bool biprism_mixed_same_edge_geometry(const BiprismMixedEdgeFfi& previous,
    uint32_t leaf,uint32_t edge,uint32_t active_count,uint32_t u,uint32_t v) {
    // Every inactive slot is explicitly zeroed by the producer. Comparing all
    // 1536 slots proves the prior and present active populations are identical.
    if(edge>=active_count)return previous.u==0u&&previous.v==0u;
    return u!=v&&previous.leaf==leaf&&previous.edge==edge&&previous.u==u&&previous.v==v;
}
BP_MIX_HD inline bool biprism_mixed_metric_source_valid(const BiprismMixedSampleFfi& source,
    const BiprismMixedSampleFfi& current,uint32_t source_stage,bool previous_body) {
    if(source.magic!=BIPRISM_MIXED_MAGIC||source.state!=1u||source.sticky_fault
        ||source.leaf!=current.leaf||source.stage!=source_stage||source.source!=current.source
        ||source.mode!=current.mode||source.generation!=current.generation
        ||source.preparation_generation!=current.preparation_generation
        ||source.committed_substeps!=current.committed_substeps)return false;
    if(previous_body)return current.body_ordinal>0u&&source.body_ordinal==current.body_ordinal-1u
        &&source.scheduled_step<UINT32_MAX&&source.scheduled_step+1u==current.scheduled_step;
    return source.body_ordinal==current.body_ordinal&&source.scheduled_step==current.scheduled_step;
}
BP_MIX_HD inline void biprism_mixed_energy_values(BiprismMixedSampleFfi& r,double kcal_to_hartree) {
    auto* v=r.values;v[9]=v[4]+v[5]+v[6]+v[7]+v[8];
    v[10]=v[3]-v[2]*kcal_to_hartree+v[9];
    v[11]=(v[0]+v[1])*kcal_to_hartree+v[10];
    v[17]=v[1]-v[2];
}
#undef BP_MIX_HD
#ifndef BIPRISM_MIXED_ENERGY_ABI_ONLY
#include "nhs_biprism_phase2_work.cuh"
#include "nhs_biprism_lif_work.cuh"
static constexpr double BP_MIX_KCAL_TO_HARTREE=BIPRISM_NATIVE_FORCE_TO_AU*BIPRISM_ANGSTROM_TO_BOHR;
struct BiprismMixedSmem {
    BiprismPhase2WorkSmem dec;
    float metric[6][192];
    uint32_t fault,metric_source,metric_required,metric_reused;
};
__device__ inline void bp_mix_fault(BiprismConvergenceRecordFfi* c,uint32_t bits) {
    for(uint32_t leaf=0;leaf<8;++leaf){atomicOr(&c[leaf].epoch_leaf_fault,BIPRISM_MIXED_TERMINAL);
        atomicOr(&c[leaf].terminal_fault_latch,BIPRISM_MIXED_TERMINAL);}
    (void)bits;
}
__device__ inline double bp_mix_quadratic(const double g[6],const float v[3],double m) {
    const double x=v[0],y=v[1],z=v[2];
    return .5*m*(x*__fma_rn(double(g[0]),x,__fma_rn(double(g[3]),y,double(g[4])*z))
        +y*__fma_rn(double(g[3]),x,__fma_rn(double(g[1]),y,double(g[5])*z))
        +z*__fma_rn(double(g[4]),x,__fma_rn(double(g[5]),y,double(g[2])*z)));
}
__device__ inline BiprismCoordinateExpansion3 bp_mix_position(const BiprismMixedVertexFfi& r) {
    return {{r.x[0],r.x_lo[0]},{r.x[1],r.x_lo[1]},{r.x[2],r.x_lo[2]}};
}
__device__ inline BiprismCoordinateExpansion3 bp_mix_reference(const BiprismMixedVertexFfi& r) {
    return {{r.reference[0],r.reference_lo[0]},{r.reference[1],r.reference_lo[1]},{r.reference[2],r.reference_lo[2]}};
}
// One leaf may be sampled by its native endpoint callback. All state reads are
// on the same owning stream; tile spinors/operator are immutable in these stages.
extern "C" __global__ __launch_bounds__(256,2) void nhs_biprism_mixed_sample_v1(
    const BiprismMixedBindingFfi* bindings,const ContactShellSimplicialStateFfi* tiles,
    const FoliatedStreamPtrTableFfi* native,const FoliatedStreamPtrTableFfi* au,const float* charges,
    const BiprismSharedStepFfi* shared,BiprismConvergenceRecordFfi* conv,
    const BiprismMixedWindowFfi* window,uint32_t leaf,uint32_t stage,uint32_t mode,float kappa,
    BiprismMixedSampleFfi* samples,BiprismMixedVertexFfi* vertices,BiprismMixedEdgeFfi* edges) {
    if(blockIdx.x||blockDim.x!=256u||leaf>=8u||stage>=BP_MIX_STAGES||!samples||!conv)return;
    const uint32_t tid=threadIdx.x;auto& sample=samples[stage*8u+leaf];
    __shared__ BiprismMixedSmem sm;
    if(tid==0){sm.fault=0;sm.metric_source=BP_MIX_STAGES;sm.metric_required=0u;sm.metric_reused=0u;uint32_t slot=0;
        if(!bindings||!tiles||!native||!au||!charges||!shared||!window||!vertices||!edges||!isfinite(kappa))sm.fault|=BP_MIX_DOMAIN;
        if(!sm.fault){const auto& b=bindings[leaf];
            if(!b.native_hmr_masses||!b.native_potential||!b.native_atom_work||!b.native_cumulative_work||b.reserved
                ||b.atoms<256u||b.atoms>0x01000000u||!native->d_coords_xyz[leaf]||!native->d_vels_xyz[leaf]
                ||!biprism_lif_work_coordinate_owners(au,native,leaf,b.atoms)
                ||!au->d_true_inv_masses[leaf]||!au->d_inv_masses[leaf]||!tiles[leaf].num_active_edges
                ||tiles[leaf].num_active_edges>1536u||shared->abi_magic!=BIPRISM_SHARED_STEP_MAGIC
                ||shared->state!=BIPRISM_SHARED_STEP_SELECTED||shared->sticky_fault
                ||!biprism_mixed_slot(*window,shared->accepted_bodies,shared->scheduled_step,slot)
                ||window->generation!=shared->generation||!shared->scheduled_step||shared->accepted_bodies>=0x3fffffffu)sm.fault|=BP_MIX_EPOCH;
            for(uint32_t j=0;j<8u;++j)if(conv[j].terminal_fault_latch||conv[j].epoch_leaf_fault||conv[j].epoch_cluster_fault
                ||conv[j].abi_magic!=BIPRISM_TAIL_ABI_MAGIC||conv[j].leaf_id!=j
                ||conv[j].committed_substeps!=4u*(shared->accepted_bodies+(stage==BP_MIX_START?0u:1u))
                ||!bp_p2_potential_owner_valid(au,tiles,j,bindings[j].atoms,window->preparation_generation)
                ||!bp_p2_spinor_valid(au,tiles,j,conv[j].committed_substeps,shared->scheduled_step-(stage==BP_MIX_START?1u:0u))
                ||!bp_potential_accepted_valid(au->d_potential_owners[j],conv[j].committed_substeps,
                    shared->scheduled_step-(stage==BP_MIX_START?1u:0u)))sm.fault|=BP_MIX_EPOCH;
            if(!sm.fault){sample={};sample.magic=BIPRISM_MIXED_MAGIC;sample.leaf=leaf;sample.stage=stage;
                sample.scheduled_step=shared->scheduled_step;sample.committed_substeps=conv[leaf].committed_substeps;
                sample.mode=mode;sample.generation=shared->generation;sample.body_ordinal=shared->accepted_bodies;
                sample.preparation_generation=window->preparation_generation;sample.source=b.source;
                if(stage==BP_MIX_POST_LIF||stage==BP_MIX_POST_NONHAM){sm.metric_source=BP_MIX_PRE_LIF;sm.metric_required=1u;}
                else if(stage==BP_MIX_END)sm.metric_source=BP_MIX_POST_CA;
                else if(stage==BP_MIX_START&&sample.body_ordinal)sm.metric_source=BP_MIX_END;
                if(sm.metric_source<BP_MIX_STAGES&&!biprism_mixed_metric_source_valid(
                    samples[sm.metric_source*8u+leaf],sample,sm.metric_source,stage==BP_MIX_START))sm.fault|=BP_MIX_EPOCH;
            }}
        if(sm.fault){sample.sticky_fault|=sm.fault;bp_mix_fault(conv,sm.fault);}
    }__syncthreads();if(sm.fault)return;
    const auto& tile=tiles[leaf];const auto& binding=bindings[leaf];
    auto* row=vertices+(stage*8u+leaf)*256u;auto* er=edges+(stage*8u+leaf)*1536u;
    auto& r=row[tid];r={};r.packed_atom=tile.global_atom_idx[tid];r.leaf=leaf;r.local_vertex=tid;r.stage=stage;
    const uint32_t atom=UNPACK_GLOBAL_ATOM_ID(r.packed_atom);bool ok=atom<binding.atoms;
    if(ok){const auto& start=vertices[leaf*256u+tid];
        if(stage==BP_MIX_HALO_COUNTERFACTUAL){const auto& s=samples[leaf];
            ok=s.magic==BIPRISM_MIXED_MAGIC&&s.state==1u&&s.generation==sample.generation&&s.body_ordinal==sample.body_ordinal
                &&start.packed_atom==r.packed_atom;}
        for(uint32_t k=0;k<3;++k){r.native_x[k]=native->d_coords_xyz[leaf][atom*3u+k];r.native_v[k]=native->d_vels_xyz[leaf][atom*3u+k];
            r.native_x_lo[k]=native->d_coordinate_owners[leaf].current_lo[atom*3u+k];
            if(stage==BP_MIX_HALO_COUNTERFACTUAL&&tid>=192u){r.native_x[k]=start.native_x[k];r.native_x_lo[k]=start.native_x_lo[k];}
            const auto q=bp_coord_scale(BiprismCoordinatePairF32{r.native_x[k],r.native_x_lo[k]},BIPRISM_ANGSTROM_TO_BOHR);
            r.x[k]=q.hi;r.x_lo[k]=q.lo;
            r.v[k]=float(double(r.native_v[k])*BIPRISM_NATIVE_VELOCITY_TO_AU);
            ok=ok&&bp_coord_finite(q)&&isfinite(r.v[k])&&isfinite(r.native_x[k])&&isfinite(r.native_x_lo[k])&&isfinite(r.native_v[k]);}
        r.reference[0]=tile.ref_pos_q[tid].x;r.reference[1]=tile.ref_pos_q[tid].y;r.reference[2]=tile.ref_pos_q[tid].z;
        for(uint32_t k=0;k<3;++k){const auto& owner=au->d_coordinate_owners[leaf];
            r.reference_lo[k]=owner.reference_lo[3u*atom+k];
            ok=ok&&isfinite(r.reference_lo[k])&&biprism_mixed_same_float_bits(r.reference[k],owner.reference_hi[3u*atom+k]);}
        r.psi[0]=tile.psi_vertex[tid][0].re;r.psi[1]=tile.psi_vertex[tid][0].im;r.psi[2]=tile.psi_vertex[tid][1].re;r.psi[3]=tile.psi_vertex[tid][1].im;
        const auto spin=bp_p2_spinor(au,leaf,tid);bp_p2_store_spinor(r.psi_precise,spin);
        ok=ok&&(tid>=192u||bp_spinor_projection(spin,r.psi[0],r.psi[1],r.psi[2],r.psi[3]));
        r.omega=tile.star0_vol[tid];r.amber_charge=charges[atom];r.true_inverse_mass=au->d_true_inv_masses[leaf][atom];
        r.propagation_inverse_mass=(mode&1u)?r.true_inverse_mass:au->d_inv_masses[leaf][atom];
        r.main_mass=1.f/fmaxf(r.propagation_inverse_mass,1e-6f);r.native_hmr_mass=binding.native_hmr_masses[atom];
        r.rho00=tid<192u?__fma_rn(r.psi_precise[0],r.psi_precise[0],r.psi_precise[1]*r.psi_precise[1]):1.;
        r.rho11=tid<192u?__fma_rn(r.psi_precise[2],r.psi_precise[2],r.psi_precise[3]*r.psi_precise[3]):0.;
        r.population=r.rho00-r.rho11;r.charge=__fma_rn(double(.15f),r.rho11,double(r.amber_charge));
        r.phi_entry=au->d_potential_owners[leaf].accepted_phi[tid];sm.dec.phi[tid]=r.phi_entry;
        ok=ok&&isfinite(r.omega)&&r.omega>0.f&&isfinite(r.charge)&&isfinite(sm.dec.phi[tid])&&isfinite(r.rho00)&&isfinite(r.rho11)
            &&bp_potential_projection_value_valid(au->d_potential_owners[leaf],tid,tile.phi_dec[tid])
            &&isfinite(r.true_inverse_mass)&&r.true_inverse_mass>0.f&&isfinite(r.propagation_inverse_mass)&&r.propagation_inverse_mass>0.f
            &&isfinite(r.main_mass)&&r.main_mass>0.f&&isfinite(r.native_hmr_mass)&&r.native_hmr_mass>0.f;
        for(int k=0;k<3;++k)ok=ok&&isfinite(r.reference[k]);
    }
    if(!__syncthreads_and(ok)){if(tid==0){sample.sticky_fault=BP_MIX_DOMAIN;bp_mix_fault(conv,BP_MIX_DOMAIN);}return;}
    const bool has_metric_source=sm.metric_source<BP_MIX_STAGES;
    const auto* previous_rows=has_metric_source?vertices+(sm.metric_source*8u+leaf)*256u:nullptr;
    const auto* previous_edges=has_metric_source?edges+(sm.metric_source*8u+leaf)*1536u:nullptr;
    bool same_geometry=has_metric_source&&biprism_mixed_same_vertex_geometry(r,previous_rows[tid]);
    for(uint32_t e=tid;e<1536u;e+=256u){er[e]={};if(e>=tile.num_active_edges)continue;
        const uint32_t u=tile.edge_u8[e][0],v=tile.edge_u8[e][1];auto& edge=er[e];edge.leaf=leaf;edge.edge=e;edge.u=u;edge.v=v;edge.base=tile.star1_wt[e];
        const auto& a=row[u];const auto& b=row[v];
        const auto d=bp_coord_difference(bp_mix_position(b),bp_mix_position(a));
        const auto ref=bp_coord_difference(bp_mix_reference(b),bp_mix_reference(a));
        edge.live=bp_geometry_live_weight_f64(edge.base,d,ref);sm.dec.weights[e]=edge.live;
        ok=ok&&u!=v&&isfinite(edge.base)&&edge.base>0.f&&isfinite(edge.live)&&edge.live>0.f;
    }
    if(has_metric_source)for(uint32_t e=tid;e<1536u;e+=256u)
        same_geometry=same_geometry&&biprism_mixed_same_edge_geometry(previous_edges[e],leaf,e,tile.num_active_edges,er[e].u,er[e].v);
    const bool all_same_geometry=__syncthreads_and(same_geometry);
    if(tid==0){sm.metric_reused=all_same_geometry?1u:0u;
        if(sm.metric_required&&!sm.metric_reused){sm.fault|=BP_MIX_WORK;sample.sticky_fault|=BP_MIX_WORK;bp_mix_fault(conv,BP_MIX_WORK);}}
    __syncthreads();if(sm.fault)return;
    // Match the main's original-edge ordered FP64 metric. Sampling can occur
    // before main publishes its invocation CSR, so this independent reader
    // scans the admitted edge list in its canonical ordinal order.
    if(tid<192u){
        for(uint32_t k=0;k<6u;++k)r.metric_f64[k]=sm.metric_reused?previous_rows[tid].metric_f64[k]:(k<3u?1.:0.);
        if(!sm.metric_reused){
            #pragma unroll 1
            for(uint32_t e=0;e<tile.num_active_edges;++e){
                const uint32_t u=er[e].u,v=er[e].v;if(u!=tid&&v!=tid)continue;
                const auto d=bp_coord_difference(bp_mix_position(row[u]),bp_mix_position(row[v]));
                const auto ref=bp_coord_difference(bp_mix_reference(row[u]),bp_mix_reference(row[v]));
                const double r2=bp_geometry_length_squared(d),R2=bp_geometry_length_squared(ref);
                const double c=double(.12f)*((r2-R2)/R2)*(1./r2);
                r.metric_f64[0]+=c*d.x*d.x;r.metric_f64[1]+=c*d.y*d.y;r.metric_f64[2]+=c*d.z*d.z;
                r.metric_f64[3]+=c*d.x*d.y;r.metric_f64[4]+=c*d.x*d.z;r.metric_f64[5]+=c*d.y*d.z;
            }
        }
    }else for(uint32_t k=0;k<6u;++k)r.metric_f64[k]=(k<3u?1.:0.);
    if(!__syncthreads_and(ok)){if(tid==0){sample.sticky_fault=BP_MIX_DOMAIN;bp_mix_fault(conv,BP_MIX_DOMAIN);}return;}
    float residual=CUDART_INF_F;uint32_t iterations=0;
    const bool solved=biprism_dec_pcg_256_f64(&tile.edge_u8[0][0],&tile.edge_u8[0][1],2u,sm.dec.weights,tile.num_active_edges,r.charge,
        sm.dec.phi,sm.dec.r,sm.dec.z,sm.dec.p,sm.dec.ap,sm.dec.diag,sm.dec.warp,residual,iterations);
    r.phi=sm.dec.phi[tid];if(tid==0){sample.dec_iterations=iterations;sample.values[14]=residual;}
    if(!solved){if(tid==0){sample.sticky_fault=BP_MIX_SOLVE;bp_mix_fault(conv,BP_MIX_SOLVE);}return;}
    for(uint32_t k=0;k<6;++k)r.metric[k]=float(r.metric_f64[k]);
    ok=biprism_lif_work_metric_spd(r.metric_f64);
    double native_u=0.,native_k=0.,atom_work=0.;
    for(uint32_t a=tid;a<binding.atoms;a+=256u){double v2=0;for(uint32_t k=0;k<3;++k){const double v=native->d_vels_xyz[leaf][3u*a+k];v2+=v*v;}
        const double mass=binding.native_hmr_masses[a],u=binding.native_potential[a],w=binding.native_atom_work[a];
        ok=ok&&isfinite(mass)&&mass>0&&isfinite(u)&&isfinite(w)&&isfinite(v2);
        native_u+=u;native_k+=.5*mass*v2/418.4;atom_work+=w;
    }
    double onsite=0.,edge_energy=0.,dec=r.charge*r.phi*(tid<192u?1.:0.);
    double dec_entry=r.charge*r.phi_entry*(tid<192u?1.:0.);
    double kg=0.,nk=0.,norm=0.,anchor=0.,pair=0.,truekg=0.,truelab=0.;
    if(tid<192u){const auto p=bp_v10_element(UNPACK_ELEM_CLASS(r.packed_atom));
        onsite=double(r.omega)*(r.rho00*p.h_s_valence+r.rho11*p.h_p_excited);
        kg=bp_mix_quadratic(r.metric_f64,r.v,double(r.main_mass));truekg=bp_mix_quadratic(r.metric_f64,r.v,1./double(r.true_inverse_mass));
        double nv2=0.,av2=0.;for(int k=0;k<3;++k){nv2+=double(r.native_v[k])*r.native_v[k];av2+=double(r.v[k])*r.v[k];}
        nk=.5*double(r.native_hmr_mass)*nv2/418.4;truelab=.5*av2/double(r.true_inverse_mass);norm=r.rho00+r.rho11;
        const auto& next=tiles[(leaf+1u)&7u];const auto spin=bp_p2_spinor(au,(leaf+1u)&7u,tid);const auto a=spin.s0,b=spin.s1;
        const double sn=__fma_rn(double(a.re),a.re,double(a.im)*a.im)-__fma_rn(double(b.re),b.re,double(b.im)*b.im);
        anchor=double(kappa)*r.omega;pair=-.5*double(kappa)*(double(r.omega)+double(next.star0_vol[tid]))*r.population*sn;
        ok=ok&&UNPACK_GLOBAL_ATOM_ID(next.global_atom_idx[tid])==atom&&isfinite(sn);
    }
    r.kinetic_metric=kg;r.kinetic_native_core=nk;
    __syncthreads(); // All represented phi values are published before edge reads.
    for(uint32_t e=tid;e<tile.num_active_edges;e+=256u){auto& z=er[e];const auto& a=row[z.u];const auto& b=row[z.v];
        BiprismElectronicEdgeProvider<double> p;
        const auto xa=bp_coord_value(bp_mix_position(a)),xb=bp_coord_value(bp_mix_position(b));
        const bool valid=bp_ed_evaluate_edge(bp_v10_element(UNPACK_ELEM_CLASS(a.packed_atom)),bp_v10_element(UNPACK_ELEM_CLASS(b.packed_atom)),
            BiprismVec64{xa.x,xa.y,xa.z},BiprismVec64{xb.x,xb.y,xb.z},a.charge,b.charge,p);
        ok=ok&&valid;if(valid){z.E0=p.E0;z.E1=p.E1;z.V=p.V01;z.S=p.S;
            // The actual provider owns density at oriented u; halo u is ground.
            const BiprismPsi64 psi=z.u<192u?BiprismPsi64{{a.psi_precise[0],a.psi_precise[1]},{a.psi_precise[2],a.psi_precise[3]}}:BiprismPsi64{{1.,0.},{0.,0.}};
            edge_energy+=bp_ed_edge_energy(p.E0,p.E1,p.V01,p.S,bp_ed_density(psi));
            const double d=double(a.phi)-double(b.phi);dec-=.5*double(z.live)*d*d;
            const double de=a.phi_entry-b.phi_entry;dec_entry-=.5*double(z.live)*de*de;
        }}
    const double values[20]={native_u,native_k,nk,kg,onsite,edge_energy,dec,anchor,pair,0.,0.,0.,0.,atom_work,0.,norm,truekg,0.,truelab,dec_entry};
    if(!__syncthreads_and(ok)){if(tid==0){sample.sticky_fault=BP_MIX_NONFINITE;bp_mix_fault(conv,BP_MIX_NONFINITE);}return;}
    for(uint32_t k=0;k<20;++k){if(k==9u||k==10u||k==11u||k==12u||k==14u||k==17u)continue;
        const double sum=bp_p2_sum(values[k],sm.dec.reductions);if(tid==0)sample.values[k]=sum;}
    if(tid==0){sample.values[19]=sample.values[6]-sample.values[19]; // Diagnostic only; no work/energy credit.
        sample.values[12]=*binding.native_cumulative_work;biprism_mixed_energy_values(sample,BP_MIX_KCAL_TO_HARTREE);
        for(uint32_t k=0;k<24;++k)if(!isfinite(sample.values[k]))sample.sticky_fault|=BP_MIX_NONFINITE;
        if(sample.values[3]<0.||sample.values[1]<0.||sample.values[17]<0.)sample.sticky_fault|=BP_MIX_DOMAIN;
        if(sample.sticky_fault)bp_mix_fault(conv,sample.sticky_fault);else{
            sample.state=1u;
            // Previous END rows and identity remain live until the START
            // sample has consumed the identical represented metric.
            if(stage==BP_MIX_START)for(uint32_t k=1;k<BP_MIX_STAGES;++k)samples[k*8u+leaf]={};
        }}
}

// Evaluate against the persistent accepted reference. The pending result is not
// committed until the caller's full collective guard succeeds.
extern "C" __global__ void nhs_biprism_mixed_evaluate_v1(
    const BiprismMixedSampleFfi* samples,const BiprismMixedVertexFfi* vertices,const BiprismPhase2WorkLedgerFfi* phase2,
    const BiprismLifWorkLedgerFfi* lif,const BiprismSharedStepFfi* shared,
    const BiprismMixedWindowFfi* window,BiprismMixedEnergyStateFfi* state,
    BiprismMixedLeafLedgerFfi* leaves,BiprismMixedGlobalLedgerFfi* pending,BiprismConvergenceRecordFfi* conv) {
    if(blockIdx.x||threadIdx.x||!state||!pending||!conv)return;
    uint32_t fault=state->sticky_fault,slot=0;*pending={};
    if(!samples||!vertices||!phase2||!lif||!shared||!window||!leaves||!biprism_mixed_slot(*window,shared->accepted_bodies,shared->scheduled_step,slot)
        ||window->generation!=shared->generation||shared->state!=BIPRISM_SHARED_STEP_SELECTED||shared->sticky_fault)fault|=BP_MIX_DOMAIN;
    if(!fault&&(state->accepted_bodies!=shared->accepted_bodies
        ||(state->accepted_bodies&&(state->magic!=BIPRISM_MIXED_MAGIC||state->state!=3u
            ||state->generation!=shared->generation||state->preparation_generation!=window->preparation_generation
            ||state->last_accepted_step+1u!=shared->scheduled_step))))fault|=BP_MIX_REFERENCE;
    if(fault){state->sticky_fault|=fault;pending->sticky_fault=fault;bp_mix_fault(conv,fault);return;}
    const bool first=state->accepted_bodies==0u;
    if(first&&(state->magic||state->state||state->generation||state->preparation_generation)){state->sticky_fault|=BP_MIX_REFERENCE;pending->sticky_fault=BP_MIX_REFERENCE;bp_mix_fault(conv,BP_MIX_REFERENCE);return;}
    pending->magic=BIPRISM_MIXED_MAGIC;pending->scheduled_step=shared->scheduled_step;pending->committed_substeps=4u*(shared->accepted_bodies+1u);
    pending->generation=shared->generation;pending->body_ordinal=shared->accepted_bodies;pending->preparation_generation=window->preparation_generation;
    pending->accepted_bodies=shared->accepted_bodies+1u;pending->stage_mask=(1u<<BP_MIX_STAGES)-1u;
    double startE=0.,endE=0.,previousE=0.,currentK=0.,previousK=0.,work=0.,prefix=0.;
    for(uint32_t leaf=0;leaf<8;++leaf){auto& l=leaves[leaf];l={};const auto& a=samples[leaf];const auto& z=samples[BP_MIX_END*8u+leaf];
        l.magic=BIPRISM_MIXED_MAGIC;l.leaf=leaf;l.scheduled_step=shared->scheduled_step;l.committed_substeps=pending->committed_substeps;l.mode=a.mode;l.source=a.source;
        l.generation=shared->generation;l.body_ordinal=shared->accepted_bodies;l.preparation_generation=window->preparation_generation;
        for(uint32_t s=0;s<BP_MIX_STAGES;++s){const auto& r=samples[s*8u+leaf];
            if(r.magic!=BIPRISM_MIXED_MAGIC||r.leaf!=leaf||r.stage!=s||r.state!=1u||r.sticky_fault||r.source!=a.source||r.mode!=a.mode
                ||r.scheduled_step!=shared->scheduled_step||r.generation!=shared->generation||r.body_ordinal!=shared->accepted_bodies
                ||r.preparation_generation!=window->preparation_generation||r.committed_substeps!=4u*(shared->accepted_bodies+(s==0u?0u:1u))
                ||r.dec_iterations>BIPRISM_DEC_MAX_ITERS||r.values[14]>1e-6||r.values[14]<0.)fault|=BP_MIX_EPOCH;
            for(uint32_t k=0;k<24;++k)if(!isfinite(r.values[k]))fault|=BP_MIX_NONFINITE;
        }
        const auto& p=phase2[leaf];const auto& f=lif[leaf];
        if(p.magic!=BIPRISM_PHASE2_WORK_MAGIC||p.leaf!=leaf||p.state!=3u||p.sticky_fault||!biprism_phase2_identities_admitted(p.totals)
            ||p.generation!=shared->generation||p.body_ordinal!=shared->accepted_bodies||p.scheduled_step!=shared->scheduled_step
            ||f.abi_magic!=BIPRISM_LIF_WORK_MAGIC||f.leaf_id!=leaf||f.state!=3u||f.sticky_fault||f.body_ordinal!=shared->accepted_bodies
            ||f.scheduled_step!=shared->scheduled_step||f.committed_substeps!=pending->committed_substeps
            ||conv[leaf].terminal_fault_latch||conv[leaf].epoch_leaf_fault||conv[leaf].epoch_cluster_fault)fault|=BP_MIX_EPOCH;
        auto* v=l.values;
        const auto& pre=samples[BP_MIX_PRE_LIF*8u+leaf];const auto& post=samples[BP_MIX_POST_LIF*8u+leaf];
        const auto& nonham=samples[BP_MIX_POST_NONHAM*8u+leaf];const auto& ca=samples[BP_MIX_POST_CA*8u+leaf];
        const double initial_prefix=first?a.values[13]:0.;
        const double prev_native=first?a.values[12]:state->previous_native_work[leaf];
        if(!first&&a.values[12]!=prev_native)fault|=BP_MIX_REFERENCE;
        v[0]=(z.values[12]-prev_native-initial_prefix)*BP_MIX_KCAL_TO_HARTREE;
        v[1]=p.totals[12];v[2]=post.values[10]-pre.values[10];
        v[3]=nonham.values[10]-post.values[10];v[4]=ca.values[10]-nonham.values[10];
        v[5]=v[0]+v[1]+v[2]+v[3]+v[4];
        v[9]=initial_prefix*BP_MIX_KCAL_TO_HARTREE;
        double native_lif_work=0.,sum_abs_lif_terms=0.;
        for(uint32_t aidx=0;aidx<256u;++aidx){
            const auto& before=vertices[(BP_MIX_PRE_LIF*8u+leaf)*256u+aidx];
            const auto& after=vertices[(BP_MIX_POST_LIF*8u+leaf)*256u+aidx];
            if(before.packed_atom!=after.packed_atom||before.native_hmr_mass!=after.native_hmr_mass)fault|=BP_MIX_WORK;
            for(uint32_t k=0;k<3;++k){
                if(before.native_x[k]!=after.native_x[k])fault|=BP_MIX_WORK;
                const double x=before.native_v[k],y=after.native_v[k];
                const double term=.5*double(before.native_hmr_mass)*(y-x)*(y+x)/418.4;
                native_lif_work+=term;sum_abs_lif_terms+=fabs(term);
            }
        }
        v[10]=native_lif_work-f.totals[7];
        // Forward-roundoff envelope for two sums of <=768 terms and their
        // products. This identity bound is not a physical-energy tolerance.
        const double neps=4096.*2.2204460492503131e-16;
        if(fabs(v[10])>(neps/(1.-neps))*sum_abs_lif_terms)fault|=BP_MIX_WORK;
        v[11]=first?a.values[11]:state->previous_leaf_energy[leaf];v[12]=z.values[11];
        v[13]=first?a.values[3]+a.values[17]*BP_MIX_KCAL_TO_HARTREE:state->previous_leaf_kinetic[leaf];
        v[14]=z.values[3]+z.values[17]*BP_MIX_KCAL_TO_HARTREE;v[15]=a.values[11];
        v[6]=v[12]-v[11];v[7]=v[6]-v[5];v[8]=z.values[10]-samples[BP_MIX_HALO_COUNTERFACTUAL*8u+leaf].values[10];
        v[16]=pre.values[10]-a.values[10]-p.totals[12];v[17]=z.values[10]-ca.values[10];
        for(uint32_t k=0;k<24;++k)if(!isfinite(v[k]))fault|=BP_MIX_NONFINITE;
        if(v[13]<0.||v[14]<0.)fault|=BP_MIX_REFERENCE;
        l.state=2u;l.sticky_fault=fault;startE+=a.values[11];endE+=z.values[11];previousE+=v[11];
        previousK+=v[13];currentK+=v[14];work+=v[5];prefix+=v[9];
    }
    auto* v=pending->values;v[0]=first?startE:state->totals[0];v[1]=previousE;v[2]=endE;v[3]=work;
    v[4]=(endE-previousE)-work;v[5]=(first?0.:state->totals[3])+work;v[6]=v[0]+v[5];v[7]=(endE-v[0])-v[5];
    v[8]=previousK;v[9]=currentK;v[10]=prefix;v[11]=fmax(first?0.:state->totals[7],fabs(v[4]));
    v[12]=fmax(first?0.:state->totals[8],fabs(v[7]));v[13]=BIPRISM_MIXED_ABSOLUTE_TOL;
    v[14]=(first?0.:state->totals[6])+prefix;
    for(uint32_t k=0;k<24;++k)if(!isfinite(v[k]))fault|=BP_MIX_NONFINITE;
    if(!biprism_mixed_energy_admitted(v[4],v[7]))fault|=BP_MIX_ENERGY;
    if(fault){state->sticky_fault|=fault;pending->sticky_fault=fault;for(uint32_t j=0;j<8;++j)leaves[j].sticky_fault|=fault;bp_mix_fault(conv,fault);return;}
    pending->state=2u;
}
// Called by the final serial collective guard only after all existing checks.
// If a later host readback fails, this attempted accepted prefix is retained and
// the run is terminal; no replay or silent reference reset is permitted.
__device__ inline bool biprism_mixed_commit_v1(BiprismMixedEnergyStateFfi* state,
    BiprismMixedGlobalLedgerFfi* pending,BiprismMixedLeafLedgerFfi* leaves,
    const BiprismMixedSampleFfi* samples,const BiprismMixedWindowFfi* window,
    BiprismMixedGlobalLedgerFfi* global_journal,BiprismMixedLeafLedgerFfi* leaf_journal,
    BiprismMixedSampleFfi* sample_journal,uint64_t body,uint32_t step) {
    uint32_t slot=0;
    if(!state||!pending||!leaves||!samples||!window||!global_journal||!leaf_journal||!sample_journal
        ||!biprism_mixed_slot(*window,body,step,slot)||state->sticky_fault||state->accepted_bodies!=body
        ||pending->magic!=BIPRISM_MIXED_MAGIC||pending->state!=2u||pending->sticky_fault||pending->body_ordinal!=body
        ||pending->scheduled_step!=step||pending->generation!=window->generation||pending->preparation_generation!=window->preparation_generation
        ||!biprism_mixed_energy_admitted(pending->values[4],pending->values[7]))return false;
    state->magic=BIPRISM_MIXED_MAGIC;state->state=3u;state->generation=pending->generation;
    state->preparation_generation=pending->preparation_generation;state->accepted_bodies=body+1u;state->last_accepted_step=step;
    state->totals[0]=pending->values[0];state->totals[1]=pending->values[2];state->totals[2]=pending->values[6];state->totals[3]=pending->values[5];
    if(!body)state->totals[4]=pending->values[8];state->totals[5]=pending->values[9];state->totals[6]=pending->values[14];
    state->totals[7]=pending->values[11];state->totals[8]=pending->values[12];
    for(uint32_t leaf=0;leaf<8u;++leaf){state->previous_native_work[leaf]=samples[BP_MIX_END*8u+leaf].values[12];
        state->previous_leaf_energy[leaf]=leaves[leaf].values[12];state->previous_leaf_kinetic[leaf]=leaves[leaf].values[14];
        leaves[leaf].state=3u;leaf_journal[uint64_t(slot)*8u+leaf]=leaves[leaf];}
    for(uint32_t i=0;i<BP_MIX_STAGES*8u;++i)sample_journal[uint64_t(slot)*BP_MIX_STAGES*8u+i]=samples[i];
    pending->state=3u;global_journal[slot]=*pending;return true;
}
#endif
