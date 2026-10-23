// Standalone root-feasibility diagnosis, never an autonomous QC receipt.
// A CPU initial guess grants no physical acceptance. The complete native
// equations and represented-state gates still run against the actual z_n.
// Earlier accepted electronic/DEC iterations are not bitwise reproducible.
// A nearby-state guard bounds seed applicability, NOT physical acceptance;
// no captured physical operand is copied over the actual live starting state.
#include <cstdlib>

struct BiprismDiagnosticInitialGuess {
    uint32_t magic, accepted_at_entry, substep;
    float h;
    double alpha[8];
    double start_x[8][3][192], start_p[8][3][192];
    double start_psi[8][4][192], trial_v[8][3][192];
    float start_v[8][3][192];
    uint32_t status[8];
};
static_assert(sizeof(BiprismDiagnosticInitialGuess)==178288, "diagnostic guess schema");
__device__ BiprismDiagnosticInitialGuess biprism_diagnostic_initial_guess;

inline void load_biprism_diagnostic_initial_guess() {
    const char* path=std::getenv("BIPRISM_DIAGNOSTIC_INITIAL_GUESS");
    if(!path)throw std::runtime_error("standalone CPU-guess diagnostic requires explicit input");
    std::ifstream in(path,std::ios::binary);
    auto data=std::make_unique<BiprismDiagnosticInitialGuess>();
    in.read(reinterpret_cast<char*>(data.get()),sizeof(*data));
    if(!in||in.peek()!=std::char_traits<char>::eof()||data->magic!=0x31474942u
       ||data->substep>=4u||!std::isfinite(data->h)||data->h<=0.f)
        throw std::runtime_error("invalid diagnostic initial guess");
    for(uint32_t l=0;l<8u;++l) {
        if(data->status[l]||!std::isfinite(data->alpha[l]))throw std::runtime_error("invalid guess status/alpha");
        for(uint32_t d=0;d<3u;++d)for(uint32_t u=0;u<192u;++u)
            if(!std::isfinite(data->trial_v[l][d][u]))throw std::runtime_error("nonfinite diagnostic guess");
    }
    rt(cudaMemcpyToSymbol(biprism_diagnostic_initial_guess,data.get(),sizeof(*data)),"standalone guess upload");
    std::printf("CPU_INITIAL_GUESS_DIAGNOSTIC=1 AUTONOMOUS_QUALIFICATION=0 expected_entry=%u substep=%u\n",
        data->accepted_at_entry,data->substep);
}

template<class Shared> __device__ __noinline__ void bp_v10_diagnostic_initial_guess(Shared* s) {
    const uint32_t tid=threadIdx.x,leaf=s->invocation.collective.leaf;
    auto& g=biprism_diagnostic_initial_guess;
    if(g.magic!=0x31474942u||s->invocation.bank->footer.accepted_substeps_at_entry!=g.accepted_at_entry
       ||s->invocation.collective.substep!=g.substep)return;
    bool valid=s->invocation.h==g.h;
    if(tid<192u) {
        const auto x=bp_v10_start_position(s,tid);
        valid=valid&&(fabs(double(x.x.hi)+double(x.x.lo)-g.start_x[leaf][0][tid])<=1e-5)
            &&(fabs(double(x.y.hi)+double(x.y.lo)-g.start_x[leaf][1][tid])<=1e-5)
            &&(fabs(double(x.z.hi)+double(x.z.lo)-g.start_x[leaf][2][tid])<=1e-5);
        const auto p=bp_ns_load3d(s->invocation.numerical->nuclear_start_momentum,tid);
        valid=valid&&fabs(p.x-g.start_p[leaf][0][tid])<=.005
            &&fabs(p.y-g.start_p[leaf][1][tid])<=.005&&fabs(p.z-g.start_p[leaf][2][tid])<=.005;
        const auto psi=bp_v10_reload_spinor(&bp_v10_starts(s)[tid]);
        valid=valid&&fabs(psi.s0.re-g.start_psi[leaf][0][tid])<=5e-5&&fabs(psi.s0.im-g.start_psi[leaf][1][tid])<=5e-5
            &&fabs(psi.s1.re-g.start_psi[leaf][2][tid])<=5e-5&&fabs(psi.s1.im-g.start_psi[leaf][3][tid])<=5e-5;
        const auto v=bp_ns_load3(s->invocation.numerical->v_start,tid);
        valid=valid&&fabs(double(v.x)-g.start_v[leaf][0][tid])<=1e-6
            &&fabs(double(v.y)-g.start_v[leaf][1][tid])<=1e-6&&fabs(double(v.z)-g.start_v[leaf][2][tid])<=1e-6;
    }
    if(!valid){atomicOr(&g.status[leaf],2u);atomicOr(&s->local_fault_accum,FAULT_PICARD_NONCONV);}
    __syncthreads();
    if(g.status[leaf]&2u)return;
    if(tid<192u)bp_ns_store3d(s->invocation.numerical->nuclear_trial_velocity,tid,
        {g.trial_v[leaf][0][tid],g.trial_v[leaf][1][tid],g.trial_v[leaf][2][tid]});
    if(tid==0u){s->dg_alpha_global=g.alpha[leaf];atomicOr(&g.status[leaf],1u);}
    __syncthreads();
}
