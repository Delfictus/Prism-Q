// Native contract test of the exact pure functions used by the CUDA wrappers.
// CPU execution does not qualify actual CUDA graph dispatch or physics parity.
#include "../src/kernels/prism_observer_physical_clock.cuh"
#include <cassert>
#include <cstring>
#include <limits>
#include <iostream>

static void actual_dt_not_nominal_product() {
    PrismObserverPhysicalClockStateV1 s={};s.capacity=4;
    PrismObserverPhysicalTimeRecordV1 rows[4]={};
    // Both historical 6/4 fs and admitted-base 4/2 fs sequences are samples,
    // never a clock-side policy. The caller/integrator alone supplies live dt.
    const float actual[]={0.006f,0.004f,0.004f,0.002f};double expected=0.0;
    for(uint64_t i=0;i<4;++i){
        assert(prism_observer_clock_begin_v1(&s,i+1,actual[i]));
        assert(s.count==i && s.pending==1);
        assert(prism_observer_clock_commit_v1(&s,rows,i+1,actual[i]));
        expected+=(double)actual[i];
        assert(rows[i].step==i+1 && rows[i].applied_dt_ps==(double)actual[i]);
        assert(rows[i].cumulative_simulated_time_ps==expected);
    }
    assert(s.count==4 && s.pending==0 && s.status==0);
    assert(expected!=4.0*(double)actual[0]);
}
static void refusal_preserves_committed_prefix() {
    PrismObserverPhysicalClockStateV1 s={};s.capacity=1;
    PrismObserverPhysicalTimeRecordV1 rows[2]={{0xdead,17.0,19.0},{0xbeef,23.0,29.0}};
    assert(prism_observer_clock_begin_v1(&s,1,0.004f));
    assert(prism_observer_clock_commit_v1(&s,rows,1,0.004f));
    const auto before=rows[0];const double time=s.cumulative_time_ps;
    assert(!prism_observer_clock_begin_v1(&s,2,0.004f));assert(s.status==2);
    assert(s.count==1 && s.cumulative_time_ps==time && std::memcmp(&before,&rows[0],24)==0);
    assert(rows[1].step==0xbeef); // No ring wrap or overwrite.
}
static void duplicate_gap_and_invalid_dt_refused() {
    for(uint64_t step: {0ull,2ull}){
        PrismObserverPhysicalClockStateV1 s={};s.capacity=3;
        assert(!prism_observer_clock_begin_v1(&s,step,0.004f));assert(s.status==3 && s.count==0);
    }
    for(float dt: {0.0f,-0.004f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}){
        PrismObserverPhysicalClockStateV1 s={};s.capacity=3;
        assert(!prism_observer_clock_begin_v1(&s,1,dt));assert(s.status==4 && s.count==0 && s.cumulative_time_ps==0.0);
    }
    PrismObserverPhysicalClockStateV1 pending={};pending.capacity=3;
    assert(prism_observer_clock_begin_v1(&pending,1,0.004f));
    assert(!prism_observer_clock_begin_v1(&pending,1,0.004f));assert(pending.status==1 && pending.count==0);
}
static void applied_dt_change_and_incomplete_step_refused() {
    PrismObserverPhysicalClockStateV1 s={};s.capacity=3;
    PrismObserverPhysicalTimeRecordV1 rows[3]={};
    assert(prism_observer_clock_begin_v1(&s,1,0.006f));
    assert(!prism_observer_clock_commit_v1(&s,rows,1,0.004f));
    assert(s.status==6 && s.count==0 && rows[0].step==0 && s.cumulative_time_ps==0.0);
    PrismObserverPhysicalClockStateV1 no_begin={};no_begin.capacity=3;
    assert(!prism_observer_clock_commit_v1(&no_begin,rows,1,0.004f));assert(no_begin.status==5);
}
static void byte_exact_state_resume_and_independent_owners() {
    PrismObserverPhysicalClockStateV1 uninterrupted={};uninterrupted.capacity=3;
    PrismObserverPhysicalTimeRecordV1 a[3]={};
    assert(prism_observer_clock_begin_v1(&uninterrupted,1,0.006f));
    assert(prism_observer_clock_commit_v1(&uninterrupted,a,1,0.006f));
    PrismObserverPhysicalClockStateV1 resumed;PrismObserverPhysicalTimeRecordV1 b[3];
    std::memcpy(&resumed,&uninterrupted,sizeof resumed);std::memcpy(b,a,sizeof b);
    for(uint64_t step=2;step<=3;++step){const float dt=step==2?0.004f:0.002f;
        assert(prism_observer_clock_begin_v1(&uninterrupted,step,dt));assert(prism_observer_clock_commit_v1(&uninterrupted,a,step,dt));
        assert(prism_observer_clock_begin_v1(&resumed,step,dt));assert(prism_observer_clock_commit_v1(&resumed,b,step,dt));}
    assert(std::memcmp(a,b,sizeof a)==0 && std::memcmp(&resumed,&uninterrupted,sizeof resumed)==0);
    PrismObserverPhysicalClockStateV1 other={};other.capacity=3;
    assert(other.count==0 && other.cumulative_time_ps==0.0); // No global clock owner.
}
int main(){actual_dt_not_nominal_product();refusal_preserves_committed_prefix();duplicate_gap_and_invalid_dt_refused();applied_dt_change_and_incomplete_step_refused();byte_exact_state_resume_and_independent_owners();std::cout<<"observer physical clock CPU contract: PASS\n";}
