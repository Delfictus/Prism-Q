#pragma once
#include <stdint.h>
#include <stddef.h>
#include <math.h>

// Actual fixed-state operator-refresh measurement. This owner does not modify
// x, v, psi, the topology, reference geometry, or either physical clock.
// Its electronic-model work is not injected into the classical AMBER work plane.
struct alignas(16) BiprismPhase2VertexWorkFfi {
    uint32_t packed_atom,leaf,local_vertex,state;
    float position[3],reference[3],psi[4];
    float amber_charge,omega_before,omega_after;
    uint32_t potential_alignment;
    double phi_before,phi_same_halo,phi_after;
    float berry_before,berry_after;
    double charge,rho00,rho11,population;
    double onsite_delta,anchor_delta,pair_delta,boundary_work,stationarity_operator,stationarity_boundary;
    float onsite_eps_s,onsite_eps_p;
    float position_lo[3],reference_lo[3];
    double phi_entry;
    double psi_precise[4];
};
static_assert(offsetof(BiprismPhase2VertexWorkFfi,psi_precise)==224,"BPW5 FP64 spinor");
struct alignas(16) BiprismPhase2EdgeWorkFfi {
    uint32_t leaf,edge,u,v;
    float base_before,base_after;
    double live_before,live_after;
    double operator_cross_work,energy_before,energy_same_halo,energy_after;
};
struct alignas(16) BiprismPhase2WorkLedgerFfi {
    uint32_t magic,leaf,state,sticky_fault;
    uint32_t scheduled_step,entry_committed,edges,tets;
    uint64_t generation,preparation_generation,body_ordinal;
    double kappa;
    float residual[3];
    uint32_t iterations[3];
    double totals[16];
    uint64_t completed_measurements;
    uint32_t immutable_mismatches,reserved;
    // Diagnostic numerical refinement only. It is never physical work or
    // acceptance credit; totals[12] continues to contain only declared work.
    double entry_refinement_delta;
};
struct alignas(16) BiprismPhase2WorkWindowFfi {
    uint32_t magic,capacity,first_body,requested_bodies;
    uint32_t first_scheduled_step,reserved;
    uint64_t generation,preparation_generation,reserved_tail;
};
static_assert(sizeof(BiprismPhase2VertexWorkFfi)==256,"BPW5 Phase2 vertex record");
static_assert(offsetof(BiprismPhase2VertexWorkFfi,phi_before)==72,"BPW3 full potential ABI");
static_assert(offsetof(BiprismPhase2VertexWorkFfi,position_lo)==192,"BPW3 position expansion ABI");
static_assert(offsetof(BiprismPhase2VertexWorkFfi,charge)==104,"BPW3 vertex quadratic data");
static_assert(offsetof(BiprismPhase2VertexWorkFfi,phi_entry)==216,"BPW3 entry potential ABI");
static_assert(offsetof(BiprismPhase2EdgeWorkFfi,live_before)==24&&offsetof(BiprismPhase2EdgeWorkFfi,operator_cross_work)==40,"BPW5 full live-weight ABI");
static_assert(sizeof(BiprismPhase2EdgeWorkFfi)==80,"Phase2 edge record");
static_assert(sizeof(BiprismPhase2WorkLedgerFfi)==240,"Phase2 ledger");
static_assert(offsetof(BiprismPhase2WorkLedgerFfi,totals)==88,"Phase2 totals");
static_assert(offsetof(BiprismPhase2WorkLedgerFfi,entry_refinement_delta)==232,"BPW3 numerical diagnostic ABI");
static_assert(sizeof(BiprismPhase2WorkWindowFfi)==48,"Phase2 window");
static constexpr uint32_t BIPRISM_PHASE2_WORK_MAGIC=0x42505735u;
static constexpr uint32_t BIPRISM_PHASE2_WORK_TERMINAL=0x02000000u;
static constexpr uint32_t BIPRISM_PHASE2_WORK_DOMAIN=1u,BIPRISM_PHASE2_WORK_EPOCH=2u;
static constexpr uint32_t BIPRISM_PHASE2_WORK_SOLVE=4u,BIPRISM_PHASE2_WORK_CHANGED_STATE=8u;
static constexpr uint32_t BIPRISM_PHASE2_WORK_NONFINITE=16u,BIPRISM_PHASE2_WORK_IDENTITY=32u;
static constexpr double BIPRISM_PHASE2_IDENTITY_TOL=1e-6;
#if defined(__CUDACC__)
#define BP_P2_HD __host__ __device__
#else
#define BP_P2_HD
#endif
BP_P2_HD inline bool biprism_phase2_work_slot(const BiprismPhase2WorkWindowFfi& w,
    uint64_t body,uint32_t step,uint32_t& slot) {
    if(w.magic!=BIPRISM_PHASE2_WORK_MAGIC||!w.capacity||w.capacity>500u||!w.requested_bodies
        ||w.requested_bodies>w.capacity||!w.generation||!w.preparation_generation||w.reserved||w.reserved_tail
        ||body<w.first_body||body-w.first_body>=w.requested_bodies
        ||uint64_t(w.first_scheduled_step)+(body-w.first_body)>=UINT32_MAX
        ||step!=uint64_t(w.first_scheduled_step)+(body-w.first_body))return false;
    slot=uint32_t(body-w.first_body);return true;
}
BP_P2_HD inline void biprism_phase2_edge_energy(BiprismPhase2EdgeWorkFfi& e,
    const BiprismPhase2VertexWorkFfi& u,const BiprismPhase2VertexWorkFfi& v) {
    const double d0=double(u.phi_before)-double(v.phi_before);
    const double dm=double(u.phi_same_halo)-double(v.phi_same_halo);
    const double d1=double(u.phi_after)-double(v.phi_after);
    e.operator_cross_work=-.5*(double(e.live_after)-double(e.live_before))*d0*dm;
    e.energy_before=-.5*double(e.live_before)*d0*d0;
    e.energy_same_halo=-.5*double(e.live_after)*dm*dm;
    e.energy_after=-.5*double(e.live_after)*d1*d1;
}
BP_P2_HD inline void biprism_phase2_ring_delta(BiprismPhase2VertexWorkFfi& row,
    const BiprismPhase2VertexWorkFfi& next,double kappa) {
    const double dw=double(row.omega_after)-double(row.omega_before);
    const double dn=double(next.omega_after)-double(next.omega_before);
    row.anchor_delta=kappa*dw;
    row.pair_delta=-.5*kappa*(dw+dn)*row.population*next.population;
}
BP_P2_HD inline void biprism_phase2_finish_totals(double t[16]) {
    t[3]=t[1]-t[0];t[6]=t[2]-t[1];
    t[12]=(t[2]-t[0])+t[9]+t[10]+t[11];
    t[13]=t[3]-t[4]-t[5];t[14]=t[6]-t[7]-t[8];
}
// This pre-acceptance condition is shared by the measurement publisher and
// the final collective body guard. A later host check cannot rescue a clock.
BP_P2_HD inline bool biprism_phase2_identities_admitted(const double t[16]) {
    for(uint32_t k=0;k<16u;++k)if(!isfinite(t[k]))return false;
    return fabs(t[13])<=BIPRISM_PHASE2_IDENTITY_TOL
        &&fabs(t[14])<=BIPRISM_PHASE2_IDENTITY_TOL;
}
#undef BP_P2_HD
#ifndef BIPRISM_PHASE2_WORK_ABI_ONLY
#include "nhs_biprism_unit_bridge.cuh"
#include "nhs_biprism_shared_step_abi.cuh"
#include "nhs_biprism_coordinate_geometry.cuh"
#include "nhs_biprism_potential_precision.cuh"
static_assert(sizeof(ContactShellSimplicialStateFfi)==225408,"Phase2 snapshot extent");
static_assert(offsetof(ContactShellSimplicialStateFfi,phi_dec)==94208,"Phase2 immutable prefix");
static_assert(offsetof(ContactShellSimplicialStateFfi,num_active_tets)==225280,"Phase2 count identity");
static_assert(offsetof(ContactShellSimplicialStateFfi,active_pocket_id)==225292,"Phase2 pocket identity");

struct BiprismPhase2WorkSmem {
    double weights[TILE_MAX_EDGES];
    double phi[256];
    float r[256],z[256],p[256],ap[256],diag[256],warp[8];
    double reductions[8];
};
static_assert(sizeof(BiprismPhase2WorkSmem)<=24576,"bounded Phase2 measurement scratch");
__device__ inline void bp_p2_fault(BiprismPhase2WorkLedgerFfi& l,
    BiprismConvergenceRecordFfi* c,uint32_t leaf,uint32_t bits) {
    atomicOr(&l.sticky_fault,bits);
    atomicOr(&c[leaf].epoch_leaf_fault,BIPRISM_PHASE2_WORK_TERMINAL);
    atomicOr(&c[leaf].terminal_fault_latch,BIPRISM_PHASE2_WORK_TERMINAL);
}
__device__ inline double bp_p2_sum(double x,double* a) {
    const uint32_t lane=threadIdx.x&31u,warp=threadIdx.x>>5u;
    for(int d=16;d;d>>=1)x+=__shfl_xor_sync(0xffffffffu,x,d);
    if(lane==0)a[warp]=x;__syncthreads();
    if(warp==0){x=lane<8u?a[lane]:0.;for(int d=4;d;d>>=1)x+=__shfl_xor_sync(0xffffffffu,x,d);if(lane==0)a[0]=x;}
    __syncthreads();const double out=a[0];__syncthreads();return out;
}
__device__ inline double bp_p2_live_weight(const BiprismPhase2VertexWorkFfi& a,
    const BiprismPhase2VertexWorkFfi& b,float base) {
    const BiprismCoordinate3F64 d={
        bp_coord_difference(BiprismCoordinatePairF32{b.position[0],b.position_lo[0]},BiprismCoordinatePairF32{a.position[0],a.position_lo[0]}),
        bp_coord_difference(BiprismCoordinatePairF32{b.position[1],b.position_lo[1]},BiprismCoordinatePairF32{a.position[1],a.position_lo[1]}),
        bp_coord_difference(BiprismCoordinatePairF32{b.position[2],b.position_lo[2]},BiprismCoordinatePairF32{a.position[2],a.position_lo[2]})};
    const BiprismCoordinate3F64 ref={
        bp_coord_difference(BiprismCoordinatePairF32{b.reference[0],b.reference_lo[0]},BiprismCoordinatePairF32{a.reference[0],a.reference_lo[0]}),
        bp_coord_difference(BiprismCoordinatePairF32{b.reference[1],b.reference_lo[1]},BiprismCoordinatePairF32{a.reference[1],a.reference_lo[1]}),
        bp_coord_difference(BiprismCoordinatePairF32{b.reference[2],b.reference_lo[2]},BiprismCoordinatePairF32{a.reference[2],a.reference_lo[2]})};
    return bp_geometry_live_weight_f64(base,d,ref);
}
__device__ inline bool bp_p2_same_float(float a,float b){return __float_as_uint(a)==__float_as_uint(b);}
__device__ inline bool bp_p2_coordinate_owner_valid(const FoliatedStreamPtrTableFfi* streams,uint32_t leaf,uint32_t atoms) {
    if(!streams||leaf>=8u||!streams->d_coordinate_owners
        ||(reinterpret_cast<uintptr_t>(streams->d_coordinate_owners)&15u))return false;
    const auto* o=&streams->d_coordinate_owners[leaf];
    return bp_coord_owner_valid(o,leaf,atoms,o->generation,BP_COORD_ATOMIC_BOHR,
        BP_COORD_ALL_INITIALIZED,streams->d_coords_xyz[leaf]);
}
__device__ inline bool bp_p2_potential_owner_valid(const FoliatedStreamPtrTableFfi* streams,
    const ContactShellSimplicialStateFfi* tiles,uint32_t leaf,uint32_t atoms,uint64_t topology_generation=0u) {
    return tiles&&bp_p2_coordinate_owner_valid(streams,leaf,atoms)&&streams->d_potential_owners
        &&!(reinterpret_cast<uintptr_t>(streams->d_potential_owners)&127u)
        &&bp_potential_owner_valid(&streams->d_potential_owners[leaf],leaf,&tiles[leaf],
            &streams->d_coordinate_owners[leaf],topology_generation);
}
__device__ inline bool bp_p2_spinor_valid(const FoliatedStreamPtrTableFfi* streams,
    const ContactShellSimplicialStateFfi* tiles,uint32_t leaf,uint32_t committed,uint32_t step,bool cold=false) {
    if(!streams||!streams->d_spinor_owners||!streams->d_potential_owners||!streams->d_coordinate_owners)return false;
    const auto& p=streams->d_potential_owners[leaf];const auto& o=streams->d_spinor_owners[leaf];
    if(!bp_spinor_identity(&o,leaf,&tiles[leaf],&streams->d_coordinate_owners[leaf],p.allocation_generation,p.topology_generation))return false;
    return cold?(o.state==0u&&o.accepted_committed_substeps==0u&&o.accepted_step==0u)
        :bp_spinor_owner_valid(&o,leaf,&tiles[leaf],&streams->d_coordinate_owners[leaf],p.allocation_generation,p.topology_generation,committed,step);
}
__device__ inline BiprismSpinor64 bp_p2_spinor(const FoliatedStreamPtrTableFfi* streams,uint32_t leaf,uint32_t u) {
    return u<192u?streams->d_spinor_owners[leaf].accepted[u]:BiprismSpinor64{{1.,0.},{0.,0.}};
}
__device__ inline void bp_p2_store_spinor(double* out,BiprismSpinor64 p) {
    out[0]=p.s0.re;out[1]=p.s0.im;out[2]=p.s1.re;out[3]=p.s1.im;
}
__device__ inline bool bp_p2_same_double(double a,double b) {
    return __double_as_longlong(a)==__double_as_longlong(b);
}
// Preserve the existing Phase2 boundary equation and its FP32 rounding. The
// resulting double is prescribed boundary data, not a projected core solve.
__device__ inline double bp_p2_prescribed_halo(float charge) {
    return double(charge*0.07957747f);
}
__device__ inline double bp_p2_entry_refinement(
    const BiprismPhase2VertexWorkFfi* rows,const ContactShellSimplicialStateFfi& tile,
    const double* weights) {
    double entry=0.,solved=0.;
    for(uint32_t u=0;u<192u;++u){entry+=rows[u].charge*rows[u].phi_entry;solved+=rows[u].charge*rows[u].phi_before;}
    for(uint32_t e=0;e<tile.num_active_edges;++e){const uint32_t u=tile.edge_u8[e][0],v=tile.edge_u8[e][1];
        const double a=rows[u].phi_entry-rows[v].phi_entry,b=rows[u].phi_before-rows[v].phi_before;
        entry-=.5*double(weights[e])*a*a;solved-=.5*double(weights[e])*b*b;}
    return solved-entry;
}
__device__ inline bool bp_p2_capture_coordinate(BiprismPhase2VertexWorkFfi& v,
    const BiprismCoordinateOwnerFfi& owner,const float4& reference,uint32_t atom) {
    const auto p=bp_coord_load(owner,BiprismCoordinatePlane::Current,atom);
    const auto r=bp_coord_load(owner,BiprismCoordinatePlane::MeshReference,atom);
    if(!bp_coord_finite(p)||!bp_coord_finite(r)
        ||!bp_p2_same_float(r.x.hi,reference.x)||!bp_p2_same_float(r.y.hi,reference.y)
        ||!bp_p2_same_float(r.z.hi,reference.z))return false;
    v.position[0]=p.x.hi;v.position[1]=p.y.hi;v.position[2]=p.z.hi;
    v.position_lo[0]=p.x.lo;v.position_lo[1]=p.y.lo;v.position_lo[2]=p.z.lo;
    v.reference[0]=r.x.hi;v.reference[1]=r.y.hi;v.reference[2]=r.z.hi;
    v.reference_lo[0]=r.x.lo;v.reference_lo[1]=r.y.lo;v.reference_lo[2]=r.z.lo;
    return true;
}
__device__ inline bool bp_p2_state_identity(const ContactShellSimplicialStateFfi& old,
    const ContactShellSimplicialStateFfi& now,const BiprismPhase2VertexWorkFfi* rows,
    const BiprismCoordinateOwnerFfi& owner,const float* charges,uint32_t tid) {
    bool ok=old.num_active_edges==now.num_active_edges&&old.num_active_tets==now.num_active_tets
        &&old.active_pocket_id==now.active_pocket_id&&old.global_atom_idx[tid]==now.global_atom_idx[tid];
    const uint32_t atom=UNPACK_GLOBAL_ATOM_ID(old.global_atom_idx[tid]);
    if(atom>=owner.atom_count)return false;
    const auto& row=rows[tid];const float4 a=old.ref_pos_q[tid],b=now.ref_pos_q[tid];
    ok=ok&&bp_p2_same_float(a.x,b.x)&&bp_p2_same_float(a.y,b.y)&&bp_p2_same_float(a.z,b.z)&&bp_p2_same_float(a.w,b.w)
        &&bp_p2_same_float(row.amber_charge,charges[atom]);
    for(uint32_t k=0;k<3;++k)ok=ok&&bp_p2_same_float(row.position[k],owner.current_hi[3u*atom+k])
        &&bp_p2_same_float(row.position_lo[k],owner.current_lo[3u*atom+k])
        &&bp_p2_same_float(row.reference[k],owner.reference_hi[3u*atom+k])
        &&bp_p2_same_float(row.reference_lo[k],owner.reference_lo[3u*atom+k]);
    for(uint32_t k=0;k<2;++k)ok=ok&&bp_p2_same_float(old.psi_vertex[tid][k].re,now.psi_vertex[tid][k].re)
        &&bp_p2_same_float(old.psi_vertex[tid][k].im,now.psi_vertex[tid][k].im);
    for(uint32_t t=tid;t<old.num_active_tets;t+=256u){
        for(uint32_t k=0;k<4;++k)ok=ok&&old.tet_v[t][k]==now.tet_v[t][k]&&old.tet_opp_tet[t][k]==now.tet_opp_tet[t][k];
        for(uint32_t k=0;k<6;++k)ok=ok&&old.tet_edges[t][k]==now.tet_edges[t][k];
    }
    for(uint32_t e=tid;e<old.num_active_edges;e+=256u)ok=ok&&old.edge_u8[e][0]==now.edge_u8[e][0]&&old.edge_u8[e][1]==now.edge_u8[e][1];
    return ok;
}
// Cold initialization qualifies the actual Phase2 operator before any physical
// body. It writes only private candidate rows. A separate all-eight publisher
// establishes the accepted baseline after every leaf passes this solve.
extern "C" __global__ void nhs_biprism_phase2_work_initialize_v1(
    ContactShellSimplicialStateFfi* tiles,const FoliatedStreamPtrTableFfi* streams,
    const float* charges,uint32_t atoms,BiprismConvergenceRecordFfi* conv,
    BiprismPhase2VertexWorkFfi* vertices,BiprismPhase2WorkLedgerFfi* ledgers) {
    const uint32_t leaf=blockIdx.x,tid=threadIdx.x;
    if(leaf>=8u||blockDim.x!=256u||!conv||!ledgers)return;
    auto& l=ledgers[leaf];__shared__ BiprismPhase2WorkSmem sm;__shared__ uint32_t fault;
    if(tid==0){fault=(!tiles||!streams||!charges||!vertices||atoms<256u||atoms>0x01000000u
        ||l.magic||l.completed_measurements)?BIPRISM_PHASE2_WORK_DOMAIN:0u;
        if(!fault)for(uint32_t i=0;i<8;++i)if(!bp_p2_potential_owner_valid(streams,tiles,i,atoms)
            ||!bp_potential_cold_valid(streams->d_potential_owners[i])||!bp_p2_spinor_valid(streams,tiles,i,0u,0u,true)||conv[i].abi_magic!=BIPRISM_TAIL_ABI_MAGIC
            ||conv[i].leaf_id!=i||conv[i].committed_substeps||conv[i].epoch_leaf_fault||conv[i].epoch_cluster_fault
            ||conv[i].terminal_fault_latch||tiles[i].num_active_edges>TILE_MAX_EDGES||!tiles[i].num_active_edges
            ||tiles[i].num_active_tets>TILE_MAX_TETS||!tiles[i].num_active_tets)fault|=BIPRISM_PHASE2_WORK_EPOCH;
        if(!fault){l={};l.magic=BIPRISM_PHASE2_WORK_MAGIC;l.leaf=leaf;l.edges=tiles[leaf].num_active_edges;l.tets=tiles[leaf].num_active_tets;}
        if(fault)bp_p2_fault(l,conv,leaf,fault);}
    __syncthreads();if(fault)return;
    auto& tile=tiles[leaf];auto* rows=vertices+leaf*256u;auto& v=rows[tid];v={};
    const uint32_t atom=UNPACK_GLOBAL_ATOM_ID(tile.global_atom_idx[tid]);bool valid=atom<atoms;
    if(valid){v.packed_atom=tile.global_atom_idx[tid];v.leaf=leaf;v.local_vertex=tid;
        valid=bp_p2_capture_coordinate(v,streams->d_coordinate_owners[leaf],tile.ref_pos_q[tid],atom);
        const auto a=tile.psi_vertex[tid][0],psi=tile.psi_vertex[tid][1];
        bp_p2_store_spinor(v.psi_precise,{{double(a.re),double(a.im)},{double(psi.re),double(psi.im)}});
        for(unsigned j=0;j<4;++j)v.psi[j]=float(v.psi_precise[j]);
        valid=valid&&isfinite(bp_spinor_norm({{v.psi_precise[0],v.psi_precise[1]},{v.psi_precise[2],v.psi_precise[3]}}))
            &&(tid>=192u||fabs(bp_spinor_norm({{v.psi_precise[0],v.psi_precise[1]},{v.psi_precise[2],v.psi_precise[3]}})-1.)<=1e-7);
        v.amber_charge=charges[atom];
        v.rho11=tid<192u?__fma_rn(double(psi.re),double(psi.re),double(psi.im)*double(psi.im)):0.;
        v.charge=tid<192u?__fma_rn(double(.15f),v.rho11,double(v.amber_charge)):double(v.amber_charge);
        v.phi_entry=double(tile.phi_dec[tid]);sm.phi[tid]=v.phi_entry;
        valid=valid&&isfinite(v.charge)&&isfinite(sm.phi[tid])&&isfinite(tile.star0_vol[tid])&&tile.star0_vol[tid]>0.f;
        if(tid>=192u)valid=valid&&bp_p2_same_double(v.phi_entry,bp_p2_prescribed_halo(v.amber_charge));
        for(int k=0;k<3;++k)valid=valid&&isfinite(v.position[k])&&isfinite(v.reference[k]);
    }
    if(!__syncthreads_and(valid)){if(tid==0)bp_p2_fault(l,conv,leaf,BIPRISM_PHASE2_WORK_DOMAIN);return;}
    for(uint32_t e=tid;e<l.edges;e+=256u){const uint32_t u=tile.edge_u8[e][0],w=tile.edge_u8[e][1];
        sm.weights[e]=bp_p2_live_weight(rows[u],rows[w],tile.star1_wt[e]);
        // BW emits zero base weights. A clamped live-weight floor must never
        // make the seed placeholder qualify as the assembled cold operator.
        valid=valid&&u!=w&&isfinite(tile.star1_wt[e])&&tile.star1_wt[e]>0.f
            &&isfinite(sm.weights[e])&&sm.weights[e]>0.f;}
    if(!__syncthreads_and(valid)){if(tid==0)bp_p2_fault(l,conv,leaf,BIPRISM_PHASE2_WORK_DOMAIN);return;}
    float residual=CUDART_INF_F;uint32_t iterations=0;
    const bool solved=biprism_dec_pcg_256_f64(&tile.edge_u8[0][0],&tile.edge_u8[0][1],2u,sm.weights,l.edges,v.charge,
        sm.phi,sm.r,sm.z,sm.p,sm.ap,sm.diag,sm.warp,residual,iterations);
    v.phi_before=sm.phi[tid];
    if(tid==0){l.residual[0]=residual;l.iterations[0]=iterations;if(!solved)bp_p2_fault(l,conv,leaf,BIPRISM_PHASE2_WORK_SOLVE);}
    __syncthreads();if(!solved)return;
    v.phi_same_halo=v.phi_before;v.phi_after=v.phi_before;v.state=5u;
    __syncthreads();if(tid==0){l.entry_refinement_delta=bp_p2_entry_refinement(rows,tile,sm.weights);
        if(!isfinite(l.entry_refinement_delta))bp_p2_fault(l,conv,leaf,BIPRISM_PHASE2_WORK_NONFINITE);else l.state=5u;}
}

// Cold publication is a separate launch so a failing leaf cannot leave any
// accepted bank partially initialized. configured_step is the actual cold clock.
extern "C" __global__ void nhs_biprism_phase2_work_initialize_publish_v1(
    ContactShellSimplicialStateFfi* tiles,const FoliatedStreamPtrTableFfi* streams,
    BiprismConvergenceRecordFfi* conv,BiprismPhase2VertexWorkFfi* vertices,
    BiprismPhase2WorkLedgerFfi* ledgers,const uint32_t* configured_step) {
    const uint32_t tid=threadIdx.x;if(blockIdx.x||blockDim.x!=256u||!conv||!ledgers)return;
    __shared__ uint32_t fault,step;
    if(tid==0){fault=(!tiles||!streams||!vertices||!configured_step)?BIPRISM_PHASE2_WORK_DOMAIN:0u;step=0u;
        if(!fault){step=*configured_step;if(step>=UINT32_MAX-1u)fault|=BIPRISM_PHASE2_WORK_EPOCH;
            for(uint32_t leaf=0;leaf<8u;++leaf){const auto& l=ledgers[leaf];
                if(!streams->d_coordinate_owners||!bp_p2_potential_owner_valid(streams,tiles,leaf,streams->d_coordinate_owners[leaf].atom_count)
                    ||!bp_potential_cold_valid(streams->d_potential_owners[leaf])||!bp_p2_spinor_valid(streams,tiles,leaf,0u,0u,true)
                    ||conv[leaf].abi_magic!=BIPRISM_TAIL_ABI_MAGIC||conv[leaf].leaf_id!=leaf||conv[leaf].committed_substeps
                    ||conv[leaf].terminal_fault_latch||conv[leaf].epoch_leaf_fault||conv[leaf].epoch_cluster_fault
                    ||l.magic!=BIPRISM_PHASE2_WORK_MAGIC||l.leaf!=leaf||l.state!=5u||l.sticky_fault||l.completed_measurements
                    ||l.edges!=tiles[leaf].num_active_edges||l.tets!=tiles[leaf].num_active_tets
                    ||!isfinite(l.residual[0])||l.residual[0]<0.f||l.residual[0]>1e-6f
                    ||l.iterations[0]>BIPRISM_DEC_MAX_ITERS||!isfinite(l.entry_refinement_delta))fault|=BIPRISM_PHASE2_WORK_EPOCH;
            }}
    }
    __syncthreads();if(!fault)for(uint32_t leaf=0;leaf<8u;++leaf){const auto& v=vertices[leaf*256u+tid];
        if(v.state!=5u||v.leaf!=leaf||v.local_vertex!=tid||v.packed_atom!=tiles[leaf].global_atom_idx[tid]
            ||!isfinite(v.phi_before)||!isfinite(float(v.phi_before))||!isfinite(v.phi_entry)
            ||(tid>=192u&&!bp_p2_same_double(v.phi_before,v.phi_entry)))atomicOr(&fault,BIPRISM_PHASE2_WORK_DOMAIN);}
    __syncthreads();if(fault){if(tid<8u)bp_p2_fault(ledgers[tid],conv,tid,fault);return;}
    for(uint32_t leaf=0;leaf<8u;++leaf){auto& v=vertices[leaf*256u+tid];
        if(tid<192u)streams->d_spinor_owners[leaf].accepted[tid]={{v.psi_precise[0],v.psi_precise[1]},{v.psi_precise[2],v.psi_precise[3]}};
        streams->d_potential_owners[leaf].accepted_phi[tid]=v.phi_before;
        tiles[leaf].phi_dec[tid]=float(v.phi_before);v.state=4u;}
    __syncthreads();if(tid<8u){bp_potential_publish_cold(streams->d_potential_owners[tid],0u,step);
        bp_spinor_publish(streams->d_spinor_owners[tid],0u,step);
        ledgers[tid].scheduled_step=step;ledgers[tid].state=4u;}
}

// All coordinates and charges are current imported AU planes. A full old tile
// snapshot retains exact immutable identity through the actual Phase2 launch.
extern "C" __global__ void nhs_biprism_phase2_work_capture_v1(
    const ContactShellSimplicialStateFfi* tiles,const FoliatedStreamPtrTableFfi* streams,
    const float* charges,const BiprismUnitBridgeStatusFfi* bridge,
    BiprismConvergenceRecordFfi* conv,const BiprismSharedStepFfi* shared,
    uint32_t atoms,float kappa,ContactShellSimplicialStateFfi* snapshots,
    BiprismPhase2VertexWorkFfi* vertices,BiprismPhase2EdgeWorkFfi* edges,
    BiprismPhase2WorkLedgerFfi* ledgers,const BiprismPhase2WorkWindowFfi* window) {
    const uint32_t leaf=blockIdx.x,tid=threadIdx.x;
    if(leaf>=8u||blockDim.x!=256u||!conv||!ledgers)return;
    auto& l=ledgers[leaf];__shared__ BiprismPhase2WorkSmem sm;__shared__ uint32_t fault;
    if(tid==0){fault=0;uint32_t slot=0;
        if(!tiles||!streams||!charges||!bridge||!shared||!snapshots||!vertices||!edges||!window
            ||atoms<256u||atoms>0x01000000u||!isfinite(kappa))fault|=BIPRISM_PHASE2_WORK_DOMAIN;
        if(!fault){
            const uint64_t count=l.completed_measurements;
            if(l.magic!=BIPRISM_PHASE2_WORK_MAGIC||l.sticky_fault
                ||count!=shared->accepted_bodies||l.state!=(count?3u:4u)
                ||shared->abi_magic!=BIPRISM_SHARED_STEP_MAGIC||shared->state!=BIPRISM_SHARED_STEP_SELECTED
                ||shared->sticky_fault||shared->generation!=window->generation
                ||shared->accepted_bodies>=0x3fffffffu
                ||!biprism_phase2_work_slot(*window,shared->accepted_bodies,shared->scheduled_step,slot))fault|=BIPRISM_PHASE2_WORK_EPOCH;
            for(uint32_t i=0;i<8;++i){
                if(conv[i].terminal_fault_latch||conv[i].epoch_leaf_fault||conv[i].epoch_cluster_fault
                    ||conv[i].committed_substeps!=4u*shared->accepted_bodies
                    ||!bp_p2_potential_owner_valid(streams,tiles,i,atoms,window->preparation_generation)
                    ||!bp_potential_prepare_valid(streams->d_potential_owners[i],conv[i].committed_substeps,shared->scheduled_step)
                    ||!bp_p2_spinor_valid(streams,tiles,i,conv[i].committed_substeps,streams->d_potential_owners[i].accepted_step)
                    ||tiles[i].num_active_edges>TILE_MAX_EDGES||tiles[i].num_active_tets>TILE_MAX_TETS
                    ||!tiles[i].num_active_edges||!tiles[i].num_active_tets
                    ||bridge[i].abi_magic!=BIPRISM_BRIDGE_MAGIC||bridge[i].leaf_id!=i||bridge[i].sticky_fault
                    ||bridge[i].atom_count!=atoms||bridge[i].imported_step!=shared->scheduled_step
                    ||bridge[i].imported_fields!=7u)fault|=BIPRISM_PHASE2_WORK_EPOCH;
            }
            if(!fault){l={};l.magic=BIPRISM_PHASE2_WORK_MAGIC;l.leaf=leaf;l.scheduled_step=shared->scheduled_step;
                l.entry_committed=conv[leaf].committed_substeps;l.edges=tiles[leaf].num_active_edges;l.tets=tiles[leaf].num_active_tets;
                l.generation=shared->generation;l.preparation_generation=window->preparation_generation;l.body_ordinal=shared->accepted_bodies;
                l.kappa=double(kappa);l.completed_measurements=count;for(int k=0;k<3;++k)l.residual[k]=CUDART_INF_F;}
        }
        if(fault)bp_p2_fault(l,conv,leaf,fault);
    }
    __syncthreads();if(fault)return;
    const auto& tile=tiles[leaf];auto* row=vertices+leaf*256u;auto* er=edges+leaf*TILE_MAX_EDGES;
    const uint32_t atom=UNPACK_GLOBAL_ATOM_ID(tile.global_atom_idx[tid]);
    bool valid=atom<atoms;
    if(valid){
        auto& v=row[tid];v={};v.packed_atom=tile.global_atom_idx[tid];v.leaf=leaf;v.local_vertex=tid;
        valid=bp_p2_capture_coordinate(v,streams->d_coordinate_owners[leaf],tile.ref_pos_q[tid],atom);
        v.psi[0]=tile.psi_vertex[tid][0].re;v.psi[1]=tile.psi_vertex[tid][0].im;
        v.psi[2]=tile.psi_vertex[tid][1].re;v.psi[3]=tile.psi_vertex[tid][1].im;
        const auto spin=bp_p2_spinor(streams,leaf,tid);bp_p2_store_spinor(v.psi_precise,spin);
        valid=valid&&(tid>=192u||bp_spinor_projection(spin,v.psi[0],v.psi[1],v.psi[2],v.psi[3]));
        v.amber_charge=charges[atom];v.omega_before=tile.star0_vol[tid];v.berry_before=fmaxf(v.omega_before,1e-4f);
        v.rho00=tid<192u?__fma_rn(v.psi_precise[0],v.psi_precise[0],v.psi_precise[1]*v.psi_precise[1]):1.;
        v.rho11=tid<192u?__fma_rn(v.psi_precise[2],v.psi_precise[2],v.psi_precise[3]*v.psi_precise[3]):0.;
        const auto params=c_gfn_xtb_lut[UNPACK_ELEM_CLASS(v.packed_atom)&7u];v.onsite_eps_s=params.h_s_valence;v.onsite_eps_p=params.h_p_excited;
        v.population=v.rho00-v.rho11;v.charge=tid<192u?__fma_rn(double(.15f),v.rho11,double(v.amber_charge)):double(v.amber_charge);
        v.phi_entry=streams->d_potential_owners[leaf].accepted_phi[tid];
        valid=valid&&isfinite(v.charge)&&isfinite(v.rho00)&&isfinite(v.rho11)&&isfinite(v.omega_before)&&v.omega_before>0.f
            &&isfinite(v.phi_entry)&&bp_potential_projection_value_valid(streams->d_potential_owners[leaf],tid,tile.phi_dec[tid])
            &&isfinite(v.onsite_eps_s)&&isfinite(v.onsite_eps_p);
        for(int k=0;k<3;++k)valid=valid&&isfinite(v.position[k])&&isfinite(v.reference[k]);
        for(int k=0;k<4;++k)valid=valid&&isfinite(v.psi[k]);
        sm.phi[tid]=v.phi_entry;
    }
    if(!__syncthreads_and(valid)){if(tid==0)bp_p2_fault(l,conv,leaf,BIPRISM_PHASE2_WORK_DOMAIN);return;}
    for(uint32_t e=tid;e<l.edges;e+=256u){auto& a=er[e];a={};a.leaf=leaf;a.edge=e;a.u=tile.edge_u8[e][0];a.v=tile.edge_u8[e][1];
        a.base_before=tile.star1_wt[e];a.live_before=bp_p2_live_weight(row[a.u],row[a.v],a.base_before);sm.weights[e]=a.live_before;
        valid=valid&&a.u!=a.v&&isfinite(a.base_before)&&a.base_before>0.f&&isfinite(a.live_before)&&a.live_before>0.f;}
    if(!__syncthreads_and(valid)){if(tid==0)bp_p2_fault(l,conv,leaf,BIPRISM_PHASE2_WORK_NONFINITE);return;}
    float residual=CUDART_INF_F;uint32_t iterations=0;
    const bool solved=biprism_dec_pcg_256_f64(&tile.edge_u8[0][0],&tile.edge_u8[0][1],2u,sm.weights,l.edges,row[tid].charge,
        sm.phi,sm.r,sm.z,sm.p,sm.ap,sm.diag,sm.warp,residual,iterations);
    row[tid].phi_before=sm.phi[tid];
    if(tid==0){l.residual[0]=residual;l.iterations[0]=iterations;if(!solved)bp_p2_fault(l,conv,leaf,BIPRISM_PHASE2_WORK_SOLVE);}
    __syncthreads();if(!solved)return;
    auto* dst=reinterpret_cast<unsigned char*>(&snapshots[leaf]);const auto* src=reinterpret_cast<const unsigned char*>(&tile);
    for(uint32_t k=tid;k<sizeof(ContactShellSimplicialStateFfi);k+=256u)dst[k]=src[k];
    row[tid].state=1;__syncthreads();if(tid==0){l.entry_refinement_delta=bp_p2_entry_refinement(row,tile,sm.weights);
        if(!isfinite(l.entry_refinement_delta))bp_p2_fault(l,conv,leaf,BIPRISM_PHASE2_WORK_NONFINITE);else l.state=1;}
}

// A changed Dirichlet boundary is measured in a second, explicitly separated
// solve; it is never interpreted as a Neumann charge-neutrality condition.
extern "C" __global__ void nhs_biprism_phase2_work_measure_v1(
    const ContactShellSimplicialStateFfi* tiles,const FoliatedStreamPtrTableFfi* streams,
    const float* charges,const ContactShellSimplicialStateFfi* snapshots,
    BiprismConvergenceRecordFfi* conv,const BiprismSharedStepFfi* shared,
    BiprismPhase2VertexWorkFfi* vertices,BiprismPhase2EdgeWorkFfi* edges,
    BiprismPhase2WorkLedgerFfi* ledgers) {
    const uint32_t leaf=blockIdx.x,tid=threadIdx.x;
    if(leaf>=8u||blockDim.x!=256u||!conv||!ledgers)return;
    auto& l=ledgers[leaf];__shared__ BiprismPhase2WorkSmem sm;__shared__ uint32_t fault;
    if(tid==0){fault=(!tiles||!streams||!charges||!snapshots||!shared||!vertices||!edges
        ||!streams->d_coordinate_owners||(reinterpret_cast<uintptr_t>(streams->d_coordinate_owners)&15u))?BIPRISM_PHASE2_WORK_DOMAIN:0u;
        if(!fault){const auto& owner=streams->d_coordinate_owners[leaf];
            if(!bp_p2_potential_owner_valid(streams,tiles,leaf,owner.atom_count,l.preparation_generation)
                ||!bp_potential_prepare_valid(streams->d_potential_owners[leaf],l.entry_committed,l.scheduled_step))fault|=BIPRISM_PHASE2_WORK_DOMAIN;}
        if(!fault&&(l.magic!=BIPRISM_PHASE2_WORK_MAGIC||l.leaf!=leaf||l.state!=1u||l.sticky_fault
            ||shared->state!=BIPRISM_SHARED_STEP_SELECTED||shared->sticky_fault||shared->generation!=l.generation
            ||shared->scheduled_step!=l.scheduled_step||shared->accepted_bodies!=l.body_ordinal
            ||conv[leaf].committed_substeps!=l.entry_committed||conv[leaf].terminal_fault_latch))fault|=BIPRISM_PHASE2_WORK_EPOCH;
        if(fault)bp_p2_fault(l,conv,leaf,fault);}
    __syncthreads();if(fault)return;
    auto* vr=vertices+leaf*256u;auto* er=edges+leaf*TILE_MAX_EDGES;auto& v=vr[tid];
    const auto& tile=tiles[leaf];const auto& old=snapshots[leaf];
    bool same=bp_p2_state_identity(old,tile,vr,streams->d_coordinate_owners[leaf],charges,tid)
        &&bp_p2_same_double(v.phi_entry,streams->d_potential_owners[leaf].accepted_phi[tid]);
    if(!same)atomicAdd(&l.immutable_mismatches,1u);
    if(tid<192u){const auto spin=bp_p2_spinor(streams,leaf,tid);const auto& row=vr[tid];
        same=same&&bp_p2_same_double(spin.s0.re,row.psi_precise[0])&&bp_p2_same_double(spin.s0.im,row.psi_precise[1])
            &&bp_p2_same_double(spin.s1.re,row.psi_precise[2])&&bp_p2_same_double(spin.s1.im,row.psi_precise[3]);}
    if(!__syncthreads_and(same)){if(tid==0)bp_p2_fault(l,conv,leaf,BIPRISM_PHASE2_WORK_CHANGED_STATE);return;}
    v.omega_after=tile.star0_vol[tid];v.berry_after=fmaxf(v.omega_after,1e-4f);
    const double prescribed_halo=tid>=192u?bp_p2_prescribed_halo(v.amber_charge):0.;
    bool valid=isfinite(v.omega_after)&&v.omega_after>0.f&&isfinite(v.phi_before)
        &&(tid<192u||(isfinite(prescribed_halo)&&bp_p2_same_float(tile.phi_dec[tid],float(prescribed_halo))
            &&bp_p2_same_double(v.phi_before,v.phi_entry)));
    for(uint32_t e=tid;e<l.edges;e+=256u){auto& a=er[e];a.base_after=tile.star1_wt[e];
        a.live_after=bp_p2_live_weight(vr[a.u],vr[a.v],a.base_after);sm.weights[e]=a.live_after;
        valid=valid&&isfinite(a.base_after)&&a.base_after>0.f&&isfinite(a.live_after)&&a.live_after>0.f;}
    sm.phi[tid]=v.phi_before;
    if(!__syncthreads_and(valid)){if(tid==0)bp_p2_fault(l,conv,leaf,BIPRISM_PHASE2_WORK_NONFINITE);return;}
    float residual=CUDART_INF_F;uint32_t iterations=0;
    bool solved=biprism_dec_pcg_256_f64(&tile.edge_u8[0][0],&tile.edge_u8[0][1],2u,sm.weights,l.edges,v.charge,
        sm.phi,sm.r,sm.z,sm.p,sm.ap,sm.diag,sm.warp,residual,iterations);
    v.phi_same_halo=sm.phi[tid];
    if(tid==0){l.residual[1]=residual;l.iterations[1]=iterations;if(!solved)bp_p2_fault(l,conv,leaf,BIPRISM_PHASE2_WORK_SOLVE);}
    __syncthreads();if(!solved)return;
    if(tid>=192u)sm.phi[tid]=prescribed_halo;__syncthreads();
    solved=biprism_dec_pcg_256_f64(&tile.edge_u8[0][0],&tile.edge_u8[0][1],2u,sm.weights,l.edges,v.charge,
        sm.phi,sm.r,sm.z,sm.p,sm.ap,sm.diag,sm.warp,residual,iterations);
    v.phi_after=sm.phi[tid];
    if(tid==0){l.residual[2]=residual;l.iterations[2]=iterations;if(!solved)bp_p2_fault(l,conv,leaf,BIPRISM_PHASE2_WORK_SOLVE);}
    __syncthreads();if(!solved)return;
    double flux0=0.,fluxm=0.,flux1=0.;
    for(uint32_t e=0;e<l.edges;++e){const auto& a=er[e];if(a.u==tid||a.v==tid){const uint32_t peer=a.u==tid?a.v:a.u;
        flux0+=double(a.live_before)*(double(v.phi_before)-double(vr[peer].phi_before));
        fluxm+=double(a.live_after)*(double(v.phi_same_halo)-double(vr[peer].phi_same_halo));
        flux1+=double(a.live_after)*(double(v.phi_after)-double(vr[peer].phi_after));}}
    v.boundary_work=tid>=192u?-.5*(fluxm+flux1)*(double(v.phi_after)-double(v.phi_same_halo)):0.;
    v.stationarity_operator=tid<192u?.5*(double(v.phi_same_halo)-double(v.phi_before))*((v.charge-flux0)+(v.charge-fluxm)):0.;
    v.stationarity_boundary=tid<192u?.5*(double(v.phi_after)-double(v.phi_same_halo))*((v.charge-fluxm)+(v.charge-flux1)):0.;
    v.onsite_delta=tid<192u?(double(v.omega_after)-double(v.omega_before))*(double(v.onsite_eps_s)*v.rho00+double(v.onsite_eps_p)*v.rho11):0.;
    for(uint32_t e=tid;e<l.edges;e+=256u)biprism_phase2_edge_energy(er[e],vr[er[e].u],vr[er[e].v]);
    v.state=2;__syncthreads();if(tid==0)l.state=2;
}

// Separate launch supplies a true all-eight publication boundary for the ring
// parameter work and journal. Sealed means the refresh measurement completed;
// only the later collective physical gate can accept the outer body.
extern "C" __global__ void nhs_biprism_phase2_work_publish_v1(
    BiprismConvergenceRecordFfi* conv,const BiprismSharedStepFfi* shared,
    BiprismPhase2VertexWorkFfi* vertices,BiprismPhase2EdgeWorkFfi* edges,
    BiprismPhase2WorkLedgerFfi* ledgers,const BiprismPhase2WorkWindowFfi* window,
    BiprismPhase2VertexWorkFfi* vertex_journal,BiprismPhase2EdgeWorkFfi* edge_journal,
    BiprismPhase2WorkLedgerFfi* ledger_journal,ContactShellSimplicialStateFfi* tiles,
    const FoliatedStreamPtrTableFfi* streams) {
    const uint32_t tid=threadIdx.x;if(blockIdx.x||blockDim.x!=256u||!conv||!ledgers)return;
    __shared__ uint32_t fault,slot;
    if(tid==0){fault=(!shared||!vertices||!edges||!window||!vertex_journal||!edge_journal||!ledger_journal||!tiles||!streams
        ||!streams->d_coordinate_owners)?BIPRISM_PHASE2_WORK_DOMAIN:0u;
        if(!fault&&(shared->abi_magic!=BIPRISM_SHARED_STEP_MAGIC||shared->state!=BIPRISM_SHARED_STEP_SELECTED||shared->sticky_fault
            ||shared->generation!=window->generation||!biprism_phase2_work_slot(*window,shared->accepted_bodies,shared->scheduled_step,slot)))fault|=BIPRISM_PHASE2_WORK_EPOCH;
        if(!fault)for(uint32_t leaf=0;leaf<8;++leaf){const auto& l=ledgers[leaf];
            if(l.magic!=BIPRISM_PHASE2_WORK_MAGIC||l.leaf!=leaf||l.state!=2u||l.sticky_fault||l.immutable_mismatches
                ||l.generation!=shared->generation||l.preparation_generation!=window->preparation_generation
                ||l.scheduled_step!=shared->scheduled_step||l.body_ordinal!=shared->accepted_bodies
                ||l.completed_measurements!=shared->accepted_bodies||l.entry_committed!=conv[leaf].committed_substeps
                ||conv[leaf].terminal_fault_latch||conv[leaf].epoch_cluster_fault||conv[leaf].epoch_leaf_fault
                ||l.edges>1536u||l.tets>1024u||!isfinite(l.entry_refinement_delta)
                ||!bp_p2_potential_owner_valid(streams,tiles,leaf,streams->d_coordinate_owners[leaf].atom_count,window->preparation_generation)
                ||!bp_potential_prepare_valid(streams->d_potential_owners[leaf],l.entry_committed,l.scheduled_step))fault|=BIPRISM_PHASE2_WORK_EPOCH;
            for(uint32_t k=0;k<3;++k)if(!isfinite(l.residual[k])||l.residual[k]<0.f||l.residual[k]>1e-6f||l.iterations[k]>BIPRISM_DEC_MAX_ITERS)fault|=BIPRISM_PHASE2_WORK_SOLVE;
        }
    }
    __syncthreads();if(!fault)for(uint32_t leaf=0;leaf<8;++leaf){auto& v=vertices[leaf*256u+tid];
        const auto& next=vertices[((leaf+1u)&7u)*256u+tid];
        if(v.state!=2u||v.leaf!=leaf||v.local_vertex!=tid||v.packed_atom!=tiles[leaf].global_atom_idx[tid]
            ||v.potential_alignment||!isfinite(v.phi_entry)||!isfinite(v.phi_before)||!isfinite(v.phi_same_halo)
            ||!isfinite(v.phi_after)||!isfinite(float(v.phi_after))
            ||!bp_p2_same_double(v.phi_entry,streams->d_potential_owners[leaf].accepted_phi[tid]))atomicOr(&fault,BIPRISM_PHASE2_WORK_EPOCH);
        if(tid<192u){if(UNPACK_GLOBAL_ATOM_ID(v.packed_atom)!=UNPACK_GLOBAL_ATOM_ID(next.packed_atom))atomicOr(&fault,BIPRISM_PHASE2_WORK_CHANGED_STATE);
            biprism_phase2_ring_delta(v,next,ledgers[leaf].kappa);}
    }
    __syncthreads();
    if(tid==0&&!fault)for(uint32_t leaf=0;leaf<8;++leaf){auto& l=ledgers[leaf];auto* vr=vertices+leaf*256u;auto* er=edges+leaf*1536u;
        for(uint32_t k=0;k<16;++k)l.totals[k]=0.;
        for(uint32_t u=0;u<256;++u){const auto& v=vr[u];
            if(u<192u){l.totals[0]+=v.charge*double(v.phi_before);l.totals[1]+=v.charge*double(v.phi_same_halo);l.totals[2]+=v.charge*double(v.phi_after);}
            l.totals[5]+=v.stationarity_operator;l.totals[7]+=v.boundary_work;l.totals[8]+=v.stationarity_boundary;
            l.totals[9]+=v.onsite_delta;l.totals[10]+=v.anchor_delta;l.totals[11]+=v.pair_delta;
            if(u<192u)l.totals[15]+=fabs(double(v.berry_after)-double(v.berry_before));
        }
        for(uint32_t e=0;e<l.edges;++e){const auto& a=er[e];l.totals[0]+=a.energy_before;l.totals[1]+=a.energy_same_halo;l.totals[2]+=a.energy_after;l.totals[4]+=a.operator_cross_work;}
        biprism_phase2_finish_totals(l.totals);
        for(uint32_t k=0;k<16;++k)if(!isfinite(l.totals[k]))fault|=BIPRISM_PHASE2_WORK_NONFINITE;
        if(!biprism_phase2_identities_admitted(l.totals))fault|=BIPRISM_PHASE2_WORK_IDENTITY;
    }
    __syncthreads();if(fault){if(tid<8u)bp_p2_fault(ledgers[tid],conv,tid,fault);return;}
    // No failure branch follows the first prepared-bank write. Accepted data
    // remains untouched until the main kernel's later all-eight commit.
    for(uint32_t leaf=0;leaf<8;++leaf){const auto& v=vertices[leaf*256u+tid];
        streams->d_potential_owners[leaf].prepared_phi[tid]=v.phi_after;tiles[leaf].phi_dec[tid]=float(v.phi_after);}
    __syncthreads();if(tid<8u)bp_potential_publish_prepared(streams->d_potential_owners[tid],ledgers[tid].entry_committed,ledgers[tid].scheduled_step);
    __syncthreads();
    for(uint32_t leaf=0;leaf<8;++leaf){auto& l=ledgers[leaf];auto& v=vertices[leaf*256u+tid];v.state=3;
        vertex_journal[(uint64_t(slot)*8u+leaf)*256u+tid]=v;
        for(uint32_t e=tid;e<1536u;e+=256u){BiprismPhase2EdgeWorkFfi row={};if(e<l.edges)row=edges[leaf*1536u+e];
            edge_journal[(uint64_t(slot)*8u+leaf)*1536u+e]=row;}
        __syncthreads();if(tid==0){l.state=3;l.completed_measurements+=1u;ledger_journal[uint64_t(slot)*8u+leaf]=l;}
        __syncthreads();
    }
}
#endif
