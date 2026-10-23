#pragma once
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#ifdef __CUDACC__
#define ASC_HD __host__ __device__
#else
#define ASC_HD
#endif
// This is the historical all-atom controller about the first relaxed tile's
// AABB midpoint. It does not identify a causal role, pocket or selected cluster.
enum : uint32_t {
    ASC_LEGACY_WHOLE_SYSTEM=1,
    ASC_CAUSAL_ROLE_UNSUPPORTED=2,
    ASC_SELECTED_STRUCTURAL_SUPPORT=3
};
enum : uint32_t { ASC_NO_PRIOR_DECISION=0, ASC_COMPLETE=1, ASC_BAD_DECISION=2,
                 ASC_INCOMPLETE_COVERAGE=3, ASC_NONFINITE=4, ASC_OPEN=5 };
struct alignas(16) PrismAscControlV1 {
    uint32_t source, mode, valid, status;
    uint64_t producer_epoch, producer_frame, support_identity, generation;
    float strength, center[3];
    uint64_t topology[4];
};
struct alignas(16) PrismAscPhaseV1 {
    uint64_t composed_atoms, consumed_atoms, nonzero_force_atoms, nonzero_impulse_atoms;
    double force[3], torque[3], impulse[3], velocity_delta[3], differential_kick_energy, potential;
};
struct alignas(16) PrismAscAtomV1 {
    uint64_t epoch[4], decision_epoch[4];
    double potential[2], control_work;
    float base_force[2][3];
    double impulse[2][3], force[2][3], torque[2][3], velocity_delta[2][3], differential_kick_energy[2];
};
// Additional allocation tail, after atom rows and before the selected mask.
// Does not change the established 736-byte owner or 320-byte row ABI.
struct alignas(16) PrismAscWorkCumulativeV1 {
    uint64_t magic, completed_steps, completed_epoch;
    double control_work_kcal_mol;
};
static_assert(sizeof(PrismAscWorkCumulativeV1)==32,"ASC work tail ABI");
static constexpr uint64_t PRISM_ASC_WORK_MAGIC_V1 = 0x415343574f524b31ULL;
struct alignas(16) PrismAscDeviceV1 {
    uint32_t abi, byte_size, source, n_atoms;
    uint64_t generation, physical_epoch, completed_epoch, published_epoch;
    uint64_t topology[4];
    float alpha;
    // `reserved` is selected_count in mode 3. Kept in place so ABI v1 remains
    // 736 bytes while the selected-support behavior is explicitly versioned by
    // mode and its v2 construction entry point.
    uint32_t mode, status, reserved;
    PrismAscControlV1 active, previous, pending;
    PrismAscPhaseV1 phases[2];
    double control_work;
    // `reserved2` is the measured policy-source identity in mode 3.
    uint64_t coverage_failures, decisions_produced, composed_steps, consumed_steps, reserved2;
    PrismAscAtomV1* rows;
    // `reserved3` is the device pointer to the immutable u8 membership mask.
    uint64_t reserved3;
};
static_assert(sizeof(PrismAscControlV1)==96,"ASC control ABI");
static_assert(sizeof(PrismAscPhaseV1)==144,"ASC phase ABI");
static_assert(sizeof(PrismAscAtomV1)==320,"ASC atom ABI");
static_assert(sizeof(PrismAscDeviceV1)==736,"ASC device ABI");
ASC_HD inline bool prism_asc_control_matches(const PrismAscControlV1& c, const PrismAscDeviceV1& s, uint64_t epoch) {
    const bool supported_mode=c.mode==ASC_LEGACY_WHOLE_SYSTEM || c.mode==ASC_SELECTED_STRUCTURAL_SUPPORT;
    const uint64_t expected_support=c.mode==ASC_SELECTED_STRUCTURAL_SUPPORT?s.reserved2:1;
    if(c.source!=s.source || !supported_mode || c.mode!=s.mode ||
       c.generation!=s.generation || c.support_identity!=expected_support || c.producer_epoch!=epoch ||
       c.producer_frame!=epoch || c.valid>1 || c.status!=ASC_COMPLETE) return false;
    for(int j=0;j<4;++j) if(c.topology[j]!=s.topology[j]) return false;
    return isfinite(c.strength)&&isfinite(c.center[0])&&isfinite(c.center[1])&&isfinite(c.center[2]);
}
ASC_HD inline bool prism_asc_atom_selected(const PrismAscDeviceV1& s,uint32_t i) {
    if(s.mode==ASC_LEGACY_WHOLE_SYSTEM)return true;
    if(s.mode!=ASC_SELECTED_STRUCTURAL_SUPPORT || s.reserved3==0)return false;
    return reinterpret_cast<const uint8_t*>(s.reserved3)[i]!=0;
}
ASC_HD inline void prism_asc_law(const PrismAscControlV1& c, const float x[3], double f[3], double* potential) {
    *potential=0.0;
    for(int j=0;j<3;++j) { double d=(double)x[j]-c.center[j]; f[j]=c.valid?(double)c.strength*d:0.0; *potential-=0.5*f[j]*d; }
}
ASC_HD inline double prism_asc_control_work(const PrismAscControlV1& before, const PrismAscControlV1& after, const float x[3]) {
    double f[3],a,b;prism_asc_law(before,x,f,&a);prism_asc_law(after,x,f,&b);return b-a;
}
// Compensated-coordinate consumers reconstruct x_hi + x_lo in FP64 before
// entering this law. The immutable published control retains its existing
// FP32 center/strength ABI; only the physical position input is widened.
ASC_HD inline void prism_asc_law_f64(const PrismAscControlV1& c, const double x[3],
                                    double f[3], double* potential) {
    *potential=0.0;
    for(int j=0;j<3;++j) { double d=x[j]-c.center[j]; f[j]=c.valid?(double)c.strength*d:0.0; *potential-=0.5*f[j]*d; }
}
ASC_HD inline double prism_asc_control_work_f64(const PrismAscControlV1& before,
    const PrismAscControlV1& after, const double x[3]) {
    double f[3],a,b;prism_asc_law_f64(before,x,f,&a);prism_asc_law_f64(after,x,f,&b);return b-a;
}
// Exact arithmetic order of the production force scaling/clamp. Used by the
// physical kick and the counterfactual ASC-consumption oracle alike.
ASC_HD inline void prism_asc_used_force(const float force[3], float scale, float limit, float used[3]) {
    float a=force[0]*scale,b=force[1]*scale,c=force[2]*scale;
    float mag=sqrtf(a*a+b*b+c*c);float q=mag>limit?limit/mag:1.0f;
    used[0]=a*q;used[1]=b*q;used[2]=c*q;
}
// Existing resident topology adjudicates at completed chunk boundaries. Controls
// are held between publications, never reinterpreted as fresh per-step decisions.
ASC_HD inline bool prism_asc_promote(PrismAscDeviceV1& d) {
    if(d.physical_epoch!=d.completed_epoch || d.physical_epoch==UINT64_MAX || d.status==ASC_BAD_DECISION)return false;
    PrismAscControlV1 next=d.active;
    if(d.published_epoch==0) {
        if(d.active.producer_epoch!=0 || d.active.valid!=0)return false;
        next={};
    }else if(d.published_epoch>d.active.producer_epoch) {
        if(d.published_epoch!=d.completed_epoch || !prism_asc_control_matches(d.pending,d,d.completed_epoch))return false;
        next=d.pending;
    }else if(d.published_epoch!=d.active.producer_epoch || !prism_asc_control_matches(d.active,d,d.published_epoch))return false;
    d.previous=d.active;d.active=next;
    d.status=d.published_epoch?ASC_OPEN:ASC_NO_PRIOR_DECISION;
    ++d.physical_epoch;return true;
}
ASC_HD inline void prism_asc_kick_difference(const float base[3],const float total[3],const float velocity[3],
    float scale,float limit,float mass,float dt,float conversion,double impulse[3],double dv[3],double* energy) {
    float with[3],without[3];prism_asc_used_force(total,scale,limit,with);prism_asc_used_force(base,scale,limit,without);
    float kick=0.5f*dt*conversion/mass;double dk=0;
    for(int j=0;j<3;++j){
        impulse[j]=0.5*(double)dt*((double)with[j]-(double)without[j]);
        float actual=velocity[j]+kick*with[j],counterfactual=velocity[j]+kick*without[j];
        dv[j]=(double)actual-counterfactual;dk+=(double)actual*actual-(double)counterfactual*counterfactual;
    }
    *energy=0.5*(double)mass*dk/conversion;
}
#ifdef __CUDACC__
__device__ inline bool prism_asc_phase_valid(PrismAscDeviceV1* s, uint32_t atoms) {
    if(s==nullptr)return false;
    const bool selected_ok=s->mode!=ASC_SELECTED_STRUCTURAL_SUPPORT ||
        (s->reserved>0 && s->reserved<=atoms && s->reserved3!=0);
    if(s->abi!=1 || s->byte_size!=sizeof(*s) || s->n_atoms!=atoms || s->physical_epoch==0 ||
       (s->mode!=ASC_LEGACY_WHOLE_SYSTEM && s->mode!=ASC_SELECTED_STRUCTURAL_SUPPORT) ||
       !selected_ok || s->status==ASC_BAD_DECISION) { asm volatile("trap;");return false; }
    return true;
}
__device__ inline PrismAscWorkCumulativeV1* prism_asc_work_tail(PrismAscDeviceV1* s) {
    return reinterpret_cast<PrismAscWorkCumulativeV1*>(s->rows+s->n_atoms);
}
__device__ inline void prism_asc_initialize_work_tail(PrismAscDeviceV1* s) {
    auto* w=prism_asc_work_tail(s);*w={};w->magic=PRISM_ASC_WORK_MAGIC_V1;
}
// Called only after all existing per-step coverage/finiteness checks pass.
// Isolated ASC control-change work; never the mixed external-work accumulator.
__device__ inline void prism_asc_commit_control_work(PrismAscDeviceV1* s) {
    auto* w=prism_asc_work_tail(s);
    if(w->magic!=PRISM_ASC_WORK_MAGIC_V1 || w->completed_steps==UINT64_MAX ||
       w->completed_epoch+1!=s->physical_epoch || w->completed_steps!=w->completed_epoch) {
        s->status=ASC_BAD_DECISION;asm volatile("trap;");return;
    }
    const double next=w->control_work_kcal_mol+s->control_work;
    if(!isfinite(next)){s->status=ASC_NONFINITE;asm volatile("trap;");return;}
    w->control_work_kcal_mol=next;++w->completed_steps;w->completed_epoch=s->physical_epoch;
}
// Called after AMBER assigned this atom's force and PE, once at each force phase.
__device__ inline void prism_asc_compose(PrismAscDeviceV1* s,uint32_t i,uint32_t atoms,uint32_t phase,
                                       const float x[3],float f[3],double* pe,double* work) {
    if(!prism_asc_phase_valid(s,atoms))return;
    PrismAscAtomV1& r=s->rows[i];
    if(r.epoch[phase]==s->physical_epoch){asm volatile("trap;");return;}
    const bool selected=prism_asc_atom_selected(*s,i);
    double a[3]={0.0,0.0,0.0},v=0.0;if(selected)prism_asc_law(s->active,x,a,&v);
    for(int j=0;j<3;++j){r.base_force[phase][j]=f[j];f[j]+=(float)a[j];r.force[phase][j]=a[j];}
    r.torque[phase][0]=(double)x[1]*a[2]-(double)x[2]*a[1];
    r.torque[phase][1]=(double)x[2]*a[0]-(double)x[0]*a[2];
    r.torque[phase][2]=(double)x[0]*a[1]-(double)x[1]*a[0];
    r.potential[phase]=v;
    if(pe)*pe+=v;
    if(phase==0){
        r.control_work=selected?prism_asc_control_work(s->previous,s->active,x):0.0;
        if(work) *work += r.control_work;
    }
    r.decision_epoch[phase]=s->active.producer_epoch;r.epoch[phase]=s->physical_epoch;
}
// Same force, potential and control-work accounting as prism_asc_compose,
// with the complete represented coordinate used for every geometric term.
__device__ inline void prism_asc_compose_f64(PrismAscDeviceV1* s,uint32_t i,uint32_t atoms,uint32_t phase,
    const double x[3],float f[3],double* pe,double* work) {
    if(!prism_asc_phase_valid(s,atoms))return;
    PrismAscAtomV1& r=s->rows[i];
    if(phase>1u || r.epoch[phase]==s->physical_epoch){asm volatile("trap;");return;}
    const bool selected=prism_asc_atom_selected(*s,i);
    double a[3]={0.0,0.0,0.0},v=0.0;if(selected)prism_asc_law_f64(s->active,x,a,&v);
    for(int j=0;j<3;++j){r.base_force[phase][j]=f[j];f[j]+=(float)a[j];r.force[phase][j]=a[j];}
    r.torque[phase][0]=x[1]*a[2]-x[2]*a[1];
    r.torque[phase][1]=x[2]*a[0]-x[0]*a[2];
    r.torque[phase][2]=x[0]*a[1]-x[1]*a[0];
    r.potential[phase]=v;
    if(pe)*pe+=v;
    if(phase==0){
        r.control_work=selected?prism_asc_control_work_f64(s->previous,s->active,x):0.0;
        if(work)*work+=r.control_work;
    }
    r.decision_epoch[phase]=s->active.producer_epoch;r.epoch[phase]=s->physical_epoch;
}
// Measure the difference between the actual pre-constraint half-kick and the
// same kick without ASC, holding base force, velocity, scaling and clamp fixed.
// This is not the final velocity after RATTLE, safety clipping or thermostat.
__device__ inline void prism_asc_measure_kick(PrismAscDeviceV1* s,uint32_t i,uint32_t atoms,uint32_t phase,
    const float x[3],const float total[3],const float velocity[3],float scale,float limit,float mass,float dt,float conversion) {
    if(!prism_asc_phase_valid(s,atoms))return;
    PrismAscAtomV1& r=s->rows[i];
    if(r.epoch[phase]!=s->physical_epoch || r.decision_epoch[phase]!=s->active.producer_epoch || r.epoch[phase+2]==s->physical_epoch){asm volatile("trap;");return;}
    prism_asc_kick_difference(r.base_force[phase],total,velocity,scale,limit,mass,dt,conversion,
        r.impulse[phase],r.velocity_delta[phase],&r.differential_kick_energy[phase]);
    r.decision_epoch[phase+2]=s->active.producer_epoch;r.epoch[phase+2]=s->physical_epoch;
}
__device__ inline void prism_asc_measure_kick_f64(PrismAscDeviceV1* s,uint32_t i,uint32_t atoms,uint32_t phase,
    const double x[3],const float total[3],const float velocity[3],float scale,float limit,float mass,float dt,float conversion) {
    // Kick attribution has no coordinate factor. Do not round x merely to call
    // the legacy overload; both paths execute the same force/velocity algebra.
    (void)x;
    if(!prism_asc_phase_valid(s,atoms))return;
    PrismAscAtomV1& r=s->rows[i];
    if(phase>1u || r.epoch[phase]!=s->physical_epoch || r.decision_epoch[phase]!=s->active.producer_epoch || r.epoch[phase+2]==s->physical_epoch){asm volatile("trap;");return;}
    prism_asc_kick_difference(r.base_force[phase],total,velocity,scale,limit,mass,dt,conversion,
        r.impulse[phase],r.velocity_delta[phase],&r.differential_kick_energy[phase]);
    r.decision_epoch[phase+2]=s->active.producer_epoch;r.epoch[phase+2]=s->physical_epoch;
}
#endif
#undef ASC_HD
