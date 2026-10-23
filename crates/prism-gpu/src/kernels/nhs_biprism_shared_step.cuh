#ifndef BIPRISM_SHARED_STEP_CONTRACT_DEFINED
#define BIPRISM_SHARED_STEP_CONTRACT_DEFINED
#include "nhs_biprism_shared_step_abi.cuh"
#include "protocol_state.cuh"
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

// Journal baseline and actual post-modifier controls separately. Requests are
// captured before publication of the common dt; all physical consumers then
// use the single value in the shared record.
struct alignas(16) BiprismProtocolStepRowFfi {
    BiprismTerminalControlsFfi baseline;
    float current_temperature,effective_gamma,requested_dt_ps,uv_wavelength_nm;
    int32_t uv_burst_active,uv_target_idx;
    uint32_t phase_bits,steering_flags;
    float steering_uv_boost,steering_temp_bias;
    int32_t steering_focus_residue;
    uint32_t reserved;
};
struct alignas(128) BiprismSharedStepJournalFfi {
    BiprismSharedStepFfi step;
    BiprismProtocolStepRowFfi leaves[8];
};
struct alignas(16) BiprismSharedStepWindowFfi {
    uint32_t magic,capacity,first_body,requested_bodies;
    uint32_t first_scheduled_step,reserved[3]; // policy, min-dt bits, max-dt bits; all-zero is legacy common-min
};
static_assert(sizeof(BiprismProtocolStepRowFfi)==96,"protocol selected-step row");
static_assert(sizeof(BiprismSharedStepJournalFfi)==896,"shared time journal");
static_assert(sizeof(BiprismSharedStepWindowFfi)==32,"shared time window");
#if defined(__CUDACC__)
#define BPST_HD __host__ __device__
#else
#define BPST_HD
#endif
// The 32-byte window retains the immutable bound configuration. Policy 0
// is the historical unbounded common-min window; policy 2 is armed bounds.
BPST_HD inline float biprism_step_float_bits(uint32_t bits){float f;memcpy(&f,&bits,4);return f;}
BPST_HD inline bool biprism_shared_window_policy_valid(const BiprismSharedStepWindowFfi& w){
    if(w.reserved[0]==0u)return w.reserved[1]==0u&&w.reserved[2]==0u;
    const float lo=biprism_step_float_bits(w.reserved[1]),hi=biprism_step_float_bits(w.reserved[2]);
    return w.reserved[0]==BIPRISM_SHARED_STEP_BOUNDED_COMMON_MIN&&isfinite(lo)&&isfinite(hi)&&lo>0.f&&hi>=lo;
}
BPST_HD inline bool biprism_shared_bounds_match(const BiprismSharedStepFfi& s,const BiprismSharedStepWindowFfi* w){
    if(!w)return s.policy==BIPRISM_SHARED_STEP_COMMON_MIN;
    if(!biprism_shared_window_policy_valid(*w))return false;
    if(w->reserved[0]==0u)return s.policy==BIPRISM_SHARED_STEP_COMMON_MIN;
    const float lo=biprism_step_float_bits(w->reserved[1]),hi=biprism_step_float_bits(w->reserved[2]);
    return s.policy==BIPRISM_SHARED_STEP_BOUNDED_COMMON_MIN&&s.minimum_request_ps>=lo
        &&s.native_dt_ps==(s.minimum_request_ps<hi?s.minimum_request_ps:hi);
}
BPST_HD inline bool biprism_shared_window_slot(const BiprismSharedStepWindowFfi& w,
    uint32_t ordinal,uint32_t step,uint32_t& slot) {
    if(w.magic!=BIPRISM_SHARED_STEP_MAGIC||!w.capacity||w.capacity>500u||!w.requested_bodies
        ||w.requested_bodies>w.capacity||!biprism_shared_window_policy_valid(w)
        ||ordinal<w.first_body||ordinal-w.first_body>=w.requested_bodies
        ||uint64_t(w.first_scheduled_step)+(ordinal-w.first_body)>=UINT32_MAX
        ||step!=uint64_t(w.first_scheduled_step)+(ordinal-w.first_body))return false;
    slot=ordinal-w.first_body;return true;
}
BPST_HD inline bool biprism_shared_selected_valid(const BiprismSharedStepFfi& s,
    uint32_t step,uint64_t generation) {
    if(s.abi_magic!=BIPRISM_SHARED_STEP_MAGIC||s.state!=BIPRISM_SHARED_STEP_SELECTED||s.sticky_fault
        ||!generation||s.generation!=generation||(s.policy!=BIPRISM_SHARED_STEP_COMMON_MIN&&s.policy!=BIPRISM_SHARED_STEP_BOUNDED_COMMON_MIN)
        ||s.scheduled_step!=step||s.last_accepted_step==UINT32_MAX||step!=s.last_accepted_step+1u
        ||s.accepted_bodies>=0x3fffffffu||!isfinite(s.cumulative_accepted_time_ps)
        ||s.cumulative_accepted_time_ps<0.0||!isfinite(s.native_dt_ps)||s.native_dt_ps<=0.f
        ||!isfinite(s.outer_dt_au)||s.outer_dt_au<=0.f||s.terminal_frozen_mask&~255u)return false;
    float lo=s.requested_dt_ps[0],hi=lo;
    for(unsigned i=0;i<8;++i) {
        const float d=s.requested_dt_ps[i];
        if(!isfinite(d)||d<=0.f||!s.protocol_control_step[i]||s.protocol_control_step[i]>step)return false;
        if(d<lo)lo=d;if(d>hi)hi=d;
    }
    return (s.policy==BIPRISM_SHARED_STEP_COMMON_MIN?s.native_dt_ps==lo:s.native_dt_ps<=lo)
        &&s.minimum_request_ps==lo&&s.maximum_request_ps==hi
        &&s.outer_dt_au==float(double(s.native_dt_ps)*41341.37333517015);
}
BPST_HD inline uint32_t biprism_shared_float_bits(float value) {
#if defined(__CUDA_ARCH__)
    return __float_as_uint(value);
#else
    uint32_t bits;memcpy(&bits,&value,sizeof(bits));return bits;
#endif
}
BPST_HD inline uint64_t biprism_shared_double_bits(double value) {
#if defined(__CUDA_ARCH__)
    return static_cast<uint64_t>(__double_as_longlong(value));
#else
    uint64_t bits;memcpy(&bits,&value,sizeof(bits));return bits;
#endif
}
// All128 bytes are named fields. Compare their representations explicitly:
// the CUDA017 byte-loop lowering incorrectly expected state byte2 to be1
// after proving the complete uint32 state was1. Do not substitute numerical
// floating-point equality, which would lose signed-zero/payload identity.
static_assert(offsetof(BiprismSharedStepFfi,abi_magic)==0
    &&offsetof(BiprismSharedStepFfi,state)==4&&offsetof(BiprismSharedStepFfi,sticky_fault)==8
    &&offsetof(BiprismSharedStepFfi,scheduled_step)==12&&offsetof(BiprismSharedStepFfi,last_accepted_step)==16
    &&offsetof(BiprismSharedStepFfi,accepted_bodies)==20&&offsetof(BiprismSharedStepFfi,terminal_frozen_mask)==24
    &&offsetof(BiprismSharedStepFfi,policy)==28&&offsetof(BiprismSharedStepFfi,native_dt_ps)==32
    &&offsetof(BiprismSharedStepFfi,outer_dt_au)==36&&offsetof(BiprismSharedStepFfi,minimum_request_ps)==40
    &&offsetof(BiprismSharedStepFfi,maximum_request_ps)==44&&offsetof(BiprismSharedStepFfi,cumulative_accepted_time_ps)==48
    &&offsetof(BiprismSharedStepFfi,generation)==56&&offsetof(BiprismSharedStepFfi,requested_dt_ps)==64
    &&offsetof(BiprismSharedStepFfi,protocol_control_step)==96&&sizeof(BiprismSharedStepFfi)==128,
    "shared identity comparison covers every byte with no padding");
BPST_HD inline bool biprism_shared_identity_equal(const BiprismSharedStepFfi& a,
    const BiprismSharedStepFfi& b) {
    if(a.abi_magic!=b.abi_magic||a.state!=b.state||a.sticky_fault!=b.sticky_fault
        ||a.scheduled_step!=b.scheduled_step||a.last_accepted_step!=b.last_accepted_step
        ||a.accepted_bodies!=b.accepted_bodies||a.terminal_frozen_mask!=b.terminal_frozen_mask
        ||a.policy!=b.policy||a.generation!=b.generation
        ||biprism_shared_float_bits(a.native_dt_ps)!=biprism_shared_float_bits(b.native_dt_ps)
        ||biprism_shared_float_bits(a.outer_dt_au)!=biprism_shared_float_bits(b.outer_dt_au)
        ||biprism_shared_float_bits(a.minimum_request_ps)!=biprism_shared_float_bits(b.minimum_request_ps)
        ||biprism_shared_float_bits(a.maximum_request_ps)!=biprism_shared_float_bits(b.maximum_request_ps)
        ||biprism_shared_double_bits(a.cumulative_accepted_time_ps)!=biprism_shared_double_bits(b.cumulative_accepted_time_ps))return false;
    for(unsigned i=0;i<8;++i)
        if(biprism_shared_float_bits(a.requested_dt_ps[i])!=biprism_shared_float_bits(b.requested_dt_ps[i])
            ||a.protocol_control_step[i]!=b.protocol_control_step[i])return false;
    return true;
}
// No physical mutation on false. Pointer lifetime/alias uniqueness is owned by
// the cold host descriptor; exact leaf identity is checked below.
BPST_HD inline bool biprism_select_shared_step(
    BiprismSharedStepFfi& s,ProtocolState* const* protocols,
    const BiprismTerminalControlsFfi* baseline,const BiprismTerminalControlsFfi* terminal,
    uint32_t step,uint64_t generation,BiprismSharedStepJournalFfi& out,
    const BiprismSharedStepWindowFfi* window=nullptr) {
    if(!protocols||!baseline||!terminal||s.abi_magic!=BIPRISM_SHARED_STEP_MAGIC||s.sticky_fault
        ||(s.state!=0u&&s.state!=BIPRISM_SHARED_STEP_ACCEPTED)||!generation||s.generation!=generation
        ||(s.policy!=BIPRISM_SHARED_STEP_COMMON_MIN&&s.policy!=BIPRISM_SHARED_STEP_BOUNDED_COMMON_MIN)||s.last_accepted_step==UINT32_MAX
        ||step!=s.last_accepted_step+1u||step==UINT32_MAX||s.accepted_bodies>=0x3fffffffu
        ||!isfinite(s.cumulative_accepted_time_ps)||s.cumulative_accepted_time_ps<0)return false;
    BiprismSharedStepFfi candidate=s;candidate.state=BIPRISM_SHARED_STEP_SELECTED;
    candidate.scheduled_step=step;candidate.terminal_frozen_mask=0;
    for(uint32_t i=0;i<8;++i) {
        if(!protocols[i])return false;for(uint32_t j=0;j<i;++j)if(protocols[i]==protocols[j])return false;
        const auto& p=*protocols[i];const auto& b=baseline[i];const auto& t=terminal[i];
        const uint32_t control_step=step<p.total_steps?step:p.total_steps;
        if(!p.total_steps||p.total_steps>0x7fffffffu||p.current_step!=step
            ||(p.adaptive_dt_enabled!=0&&p.adaptive_dt_enabled!=1)
            ||b.abi_magic!=BIPRISM_SHARED_STEP_MAGIC||b.leaf_id!=i||b.terminal_step!=p.total_steps
            ||b.captured_step!=control_step||b.reserved||!isfinite(b.temperature)||b.temperature<=0
            ||!isfinite(b.effective_gamma)||b.effective_gamma<0||!isfinite(b.requested_dt_ps)||b.requested_dt_ps<=0
            ||!isfinite(b.uv_wavelength_nm)||(b.uv_burst_active!=0&&b.uv_burst_active!=1)
            ||t.abi_magic!=BIPRISM_SHARED_STEP_MAGIC||t.leaf_id!=i||t.terminal_step!=p.total_steps||t.reserved
            ||(step<p.total_steps?t.captured_step!=0:t.captured_step!=p.total_steps)
            ||p.dt!=b.requested_dt_ps||p.effective_gamma!=b.effective_gamma
            ||p.current_phase_bits!=b.phase_bits||p.uv_burst_active!=b.uv_burst_active
            ||p.uv_wavelength_nm!=b.uv_wavelength_nm||p.uv_target_idx!=b.uv_target_idx
            ||!isfinite(p.current_temperature)||p.current_temperature<=0
            ||!isfinite(p.steering_uv_boost)||!isfinite(p.steering_temp_bias))return false;
        if(step>=p.total_steps) {
            const unsigned char* x=reinterpret_cast<const unsigned char*>(&b);
            const unsigned char* y=reinterpret_cast<const unsigned char*>(&t);
            for(unsigned k=0;k<sizeof(b);++k)if(x[k]!=y[k])return false;
        }
        if(step>p.total_steps)candidate.terminal_frozen_mask|=1u<<i;
        candidate.requested_dt_ps[i]=p.dt;candidate.protocol_control_step[i]=control_step;
        out.leaves[i]={b,p.current_temperature,p.effective_gamma,p.dt,p.uv_wavelength_nm,
            p.uv_burst_active,p.uv_target_idx,p.current_phase_bits,uint32_t(p.steering_flags),
            p.steering_uv_boost,p.steering_temp_bias,p.steering_focus_residue,0u};
    }
    float lo=candidate.requested_dt_ps[0],hi=lo;
    for(unsigned i=1;i<8;++i){float d=candidate.requested_dt_ps[i];if(d<lo)lo=d;if(d>hi)hi=d;}
    candidate.native_dt_ps=lo;candidate.minimum_request_ps=lo;candidate.maximum_request_ps=hi;
    if(window&&window->reserved[0]==BIPRISM_SHARED_STEP_BOUNDED_COMMON_MIN){
        const float cap=biprism_step_float_bits(window->reserved[2]);
        candidate.native_dt_ps=lo<cap?lo:cap;
    }
    candidate.outer_dt_au=float(double(candidate.native_dt_ps)*41341.37333517015);
    if(!biprism_shared_selected_valid(candidate,step,generation)||!biprism_shared_bounds_match(candidate,window))return false;
    s=candidate;out.step=s;for(unsigned i=0;i<8;++i)protocols[i]->dt=s.native_dt_ps;return true;
}
BPST_HD inline bool biprism_shared_accept_valid(const BiprismSharedStepFfi& s,
    uint32_t step,uint64_t generation,uint32_t accepted_bodies,uint32_t committed,
    const BiprismSharedStepJournalFfi& journal,const BiprismSharedStepWindowFfi* window=nullptr) {
    if(!biprism_shared_selected_valid(s,step,generation)||!biprism_shared_bounds_match(s,window)||accepted_bodies!=s.accepted_bodies+1u
        ||committed!=accepted_bodies*4u)return false;
    if(!biprism_shared_identity_equal(s,journal.step))return false;
    const double next=s.cumulative_accepted_time_ps+double(s.native_dt_ps);
    if(!isfinite(next)||next<=s.cumulative_accepted_time_ps)return false;
    return true;
}
// Call only after common accepted-record validation on the one owning stream.
// No condition or allocation remains after the first reference is committed.
BPST_HD inline void biprism_publish_accepted_shared_step(BiprismSharedStepFfi& s,
    uint32_t step,uint32_t accepted_bodies,BiprismSharedStepJournalFfi& journal) {
    s.cumulative_accepted_time_ps+=double(s.native_dt_ps);s.accepted_bodies=accepted_bodies;
    s.last_accepted_step=step;s.state=BIPRISM_SHARED_STEP_ACCEPTED;journal.step=s;
}
BPST_HD inline bool biprism_accept_shared_step(BiprismSharedStepFfi& s,
    uint32_t step,uint64_t generation,uint32_t accepted_bodies,uint32_t committed,
    BiprismSharedStepJournalFfi& journal,const BiprismSharedStepWindowFfi* window=nullptr) {
    if(!biprism_shared_accept_valid(s,step,generation,accepted_bodies,committed,journal,window))return false;
    biprism_publish_accepted_shared_step(s,step,accepted_bodies,journal);return true;
}
#undef BPST_HD
#endif // BIPRISM_SHARED_STEP_CONTRACT_DEFINED
#if !defined(BIPRISM_SHARED_STEP_ABI_ONLY) && !defined(BIPRISM_SHARED_STEP_KERNELS_DEFINED)
#define BIPRISM_SHARED_STEP_KERNELS_DEFINED
#include "nhs_biprism_collective_control.cuh"
__device__ inline void biprism_shared_step_refuse(BiprismSharedStepFfi* s,
    BiprismConvergenceRecordFfi* c,uint32_t reason) {
    if(s){s->sticky_fault|=reason;s->state=3u;}
    if(c)for(unsigned i=0;i<8;++i){c[i].terminal_fault_latch|=0x04000000u;c[i].epoch_leaf_fault|=0x04000000u;}
}
extern "C" __global__ void nhs_biprism_shared_step_select_v1(
    BiprismSharedStepFfi* shared,ProtocolState* const* protocols,
    const BiprismTerminalControlsFfi* baseline,const BiprismTerminalControlsFfi* terminal,
    const uint32_t* scheduled,uint64_t generation,BiprismConvergenceRecordFfi* convergence,
    const BiprismSharedStepWindowFfi* window,BiprismSharedStepJournalFfi* journal) {
    if(blockIdx.x||threadIdx.x)return;uint32_t slot=0;
    if(!shared||!scheduled||!convergence||!window||!journal
        ||!biprism_shared_window_slot(*window,shared->accepted_bodies,*scheduled,slot)){
        biprism_shared_step_refuse(shared,convergence,2u);return;}
    for(unsigned i=0;i<8;++i)if(convergence[i].terminal_fault_latch||convergence[i].epoch_leaf_fault
        ||convergence[i].epoch_cluster_fault||convergence[i].committed_substeps!=shared->accepted_bodies*4u){
        biprism_shared_step_refuse(shared,convergence,4u);return;}
    if(!biprism_select_shared_step(*shared,protocols,baseline,terminal,*scheduled,generation,journal[slot],window))
        biprism_shared_step_refuse(shared,convergence,8u);
}
extern "C" __global__ void nhs_biprism_shared_step_accept_v1(
    BiprismSharedStepFfi* shared,BiprismCollectiveControlFfi* control,
    BiprismConvergenceRecordFfi* convergence,const uint32_t* scheduled,uint64_t generation,
    const BiprismSharedStepWindowFfi* window,BiprismSharedStepJournalFfi* journal) {
    if(blockIdx.x||threadIdx.x)return;uint32_t slot=0;
    if(!shared||!control||!convergence||!scheduled||!window||!journal||control->sticky_fault
        ||control->last_completed_step!=*scheduled
        ||!biprism_shared_window_slot(*window,shared->accepted_bodies,*scheduled,slot)){
        biprism_shared_step_refuse(shared,convergence,16u);return;}
    for(unsigned i=0;i<8;++i)if(convergence[i].terminal_fault_latch||convergence[i].epoch_leaf_fault
        ||convergence[i].epoch_cluster_fault||convergence[i].committed_substeps!=control->last_committed_substeps){
        biprism_shared_step_refuse(shared,convergence,32u);return;}
    if(!biprism_accept_shared_step(*shared,*scheduled,generation,control->successful_bodies,
        control->last_committed_substeps,journal[slot],window))biprism_shared_step_refuse(shared,convergence,64u);
}
#endif
