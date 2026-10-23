#pragma once
#include <stdint.h>
#include <math.h>
#ifdef __CUDACC__
#define PRISM_CLOCK_HD __host__ __device__ __forceinline__
#else
#define PRISM_CLOCK_HD inline
#endif
// Observer-only state: no pointers, no controller fields and no RNG ownership.
// This ABI is captured losslessly by the engine's existing checkpoint registry.
struct PrismObserverPhysicalClockStateV1 {
    uint64_t count, step_at_begin, capacity, status;
    double step_dt_ps, cumulative_time_ps;
    uint64_t pending, reserved;
};
struct PrismObserverPhysicalTimeRecordV1 {
    uint64_t step;
    double applied_dt_ps, cumulative_simulated_time_ps;
};
static_assert(sizeof(PrismObserverPhysicalClockStateV1)==64,"observer clock state ABI");
static_assert(sizeof(PrismObserverPhysicalTimeRecordV1)==24,"observer time record ABI");
// A rejected call latches an observer error but does not change elapsed time,
// committed count or payload records. The CUDA wrapper refuses further physics.
PRISM_CLOCK_HD bool prism_observer_clock_begin_v1(
    PrismObserverPhysicalClockStateV1* s,uint64_t step,float live_dt_ps) {
    if(!s || s->status) return false;
    uint64_t error=0;
    if(s->pending) error=1;
    else if(s->count>=s->capacity) error=2;
    else if(step!=s->count+1u) error=3;
    else if(!isfinite(live_dt_ps)||live_dt_ps<=0.0f) error=4;
    if(error){s->status=error;return false;}
    s->step_at_begin=step;
    s->step_dt_ps=(double)live_dt_ps; // exact widening of the integrator's f32 dt
    s->pending=1;
    return true;
}
PRISM_CLOCK_HD bool prism_observer_clock_commit_v1(
    PrismObserverPhysicalClockStateV1* s,PrismObserverPhysicalTimeRecordV1* records,
    uint64_t step,float live_dt_ps) {
    if(!s || s->status) return false;
    uint64_t error=0;
    if(!records || !s->pending)error=5;
    else if(s->count>=s->capacity)error=2;
    else if(step!=s->step_at_begin || step!=s->count+1u)error=3;
    else if((double)live_dt_ps!=s->step_dt_ps || !isfinite(live_dt_ps)||live_dt_ps<=0.0f)error=6;
    if(error){s->status=error;return false;}
#ifdef __CUDA_ARCH__
    const double next=__dadd_rn(s->cumulative_time_ps,s->step_dt_ps);
#else
    const double next=s->cumulative_time_ps+s->step_dt_ps;
#endif
    if(!isfinite(next)){s->status=7;return false;}
    records[s->count]={step,s->step_dt_ps,next};
    s->count+=1u;s->cumulative_time_ps=next;s->pending=0;
    return true;
}
#undef PRISM_CLOCK_HD
