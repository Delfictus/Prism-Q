// CPU oracle and actual exported-kernel replay of retained acceptance inputs.
// Build with -DBIPRISM_ACCEPT_NATIVE -I${CUDA_HOME}/include -lcuda for native mode.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef BIPRISM_ACCEPT_NATIVE
#include <cuda.h>
#endif
using std::isfinite;
#define BIPRISM_SHARED_STEP_ABI_ONLY
#include "nhs_biprism_shared_step.cuh"

struct Inputs {
    BiprismSharedStepFfi shared{};
    BiprismSharedStepWindowFfi window{};
    std::vector<BiprismSharedStepJournalFfi> journal;
    uint32_t control[16]{},convergence[8][32]{},scheduled=0;
};
static void require(bool yes,const std::string& what){if(!yes)throw std::runtime_error(what);}
static void read_exact(const std::string& path,void* data,size_t bytes){
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    require(bool(f)&&f.tellg()==std::streamoff(bytes),"input extent: "+path);
    f.seekg(0);f.read(static_cast<char*>(data),bytes);require(bool(f),"input read: "+path);
}
static Inputs load(const std::string& p){
    Inputs x;read_exact(p+".shared.128b",&x.shared,128);
    read_exact(p+".window.32b",&x.window,32);
    require(x.window.capacity>0&&x.window.capacity<=500,"journal capacity");
    x.journal.resize(x.window.capacity);
    read_exact(p+".journal.896b",x.journal.data(),x.journal.size()*896);
    read_exact(p+".control.64b",x.control,64);
    read_exact(p+".convergence.128b",x.convergence,1024);
    read_exact(p+".scheduled.u32le",&x.scheduled,4);return x;
}
static unsigned slot(const Inputs& x){uint32_t s=0;
    require(biprism_shared_window_slot(x.window,x.shared.accepted_bodies,x.scheduled,s),"fixture slot");return s;}
// Retains the original CPU byte comparison as an independent identity oracle.
static bool byte_oracle(const Inputs& x){const auto&s=x.shared;
    return biprism_shared_selected_valid(s,x.scheduled,s.generation)
        &&biprism_shared_bounds_match(s,&x.window)
        &&x.control[3]==s.accepted_bodies+1u&&x.control[4]==x.control[3]*4u
        &&std::memcmp(&s,&x.journal[slot(x)].step,128)==0
        &&isfinite(s.cumulative_accepted_time_ps+double(s.native_dt_ps))
        &&s.cumulative_accepted_time_ps+double(s.native_dt_ps)>s.cumulative_accepted_time_ps;
}
static bool actual_predicate(const Inputs& x){return biprism_shared_accept_valid(x.shared,
    x.scheduled,x.shared.generation,x.control[3],x.control[4],x.journal[slot(x)],&x.window);}
static void cpu_controls(const Inputs& original){
    require(byte_oracle(original)&&actual_predicate(original),"retained tuple accepted by both CPU predicates");
    unsigned checks=1;
    for(unsigned byte=0;byte<128;++byte)for(unsigned bit=0;bit<8;++bit){
        auto x=original;reinterpret_cast<unsigned char*>(&x.journal[slot(x)].step)[byte]^=1u<<bit;
        require(!byte_oracle(x)&&!actual_predicate(x),"bit corruption "+std::to_string(byte)+":"+std::to_string(bit));++checks;
    }
    // Representation equality must retain signed zeros and every NaN payload,
    // even though semantic admission independently rejects nonfinite state.
    for(unsigned byte=0;byte<128;++byte){BiprismSharedStepFfi a{},b{};
        reinterpret_cast<unsigned char*>(&a)[byte]=0x80;b=a;
        require(biprism_shared_identity_equal(a,b),"equal arbitrary representation");
        reinterpret_cast<unsigned char*>(&b)[byte]^=1;
        require(!biprism_shared_identity_equal(a,b),"arbitrary byte identity");checks+=2;
    }
    auto accepted=original;const unsigned s=slot(accepted);
    require(biprism_accept_shared_step(accepted.shared,accepted.scheduled,accepted.shared.generation,
        accepted.control[3],accepted.control[4],accepted.journal[s],&accepted.window),"CPU commit");
    require(accepted.shared.cumulative_accepted_time_ps==original.shared.cumulative_accepted_time_ps+double(original.shared.native_dt_ps)
        &&accepted.shared.accepted_bodies==original.shared.accepted_bodies+1u
        &&accepted.shared.state==BIPRISM_SHARED_STEP_ACCEPTED
        &&!std::memcmp(&accepted.shared,&accepted.journal[s].step,128),"CPU exact accepted state");
    require(!biprism_shared_accept_valid(accepted.shared,accepted.scheduled,accepted.shared.generation,
        accepted.control[3],accepted.control[4],accepted.journal[s],&accepted.window),"CPU duplicate acceptance refused");
    std::printf("PASS CPU controls=%u retained tuple,1024 bit corruptions,256 arbitrary representation checks,commit/replay\n",checks+3);
}
#ifdef BIPRISM_ACCEPT_NATIVE
static void ck(CUresult r,const char* what){if(r!=CUDA_SUCCESS){const char* s=nullptr;cuGetErrorString(r,&s);
    throw std::runtime_error(std::string(what)+": "+(s?s:"CUDA error"));}}
struct Buffer {CUdeviceptr p=0;size_t n;explicit Buffer(size_t bytes):n(bytes){ck(cuMemAlloc(&p,n),"allocation");}
    ~Buffer(){if(p)cuMemFree(p);}void put(const void* v){ck(cuMemcpyHtoD(p,v,n),"upload");}
    void get(void* v){ck(cuMemcpyDtoH(v,p,n),"download");}};
static void native_controls(const Inputs& original,const char* module_path){
    ck(cuInit(0),"driver init");CUdevice dev;ck(cuDeviceGet(&dev,0),"device");CUcontext context;
    ck(cuDevicePrimaryCtxRetain(&context,dev),"retain context");ck(cuCtxSetCurrent(context),"current context");
    CUmodule module;ck(cuModuleLoad(&module,module_path),"exact module load");CUfunction accept;
    ck(cuModuleGetFunction(&accept,module,"nhs_biprism_shared_step_accept_v1"),"actual accept export");
    Buffer shared(128),control(64),conv(1024),scheduled(4),window(32),journal(original.journal.size()*896);
    unsigned cases=0;
    auto run=[&](Inputs x,bool expected,const std::string& name){
        const Inputs before=x;const unsigned index=slot(x);uint64_t generation=x.shared.generation;
        shared.put(&x.shared);control.put(x.control);conv.put(x.convergence);scheduled.put(&x.scheduled);
        window.put(&x.window);journal.put(x.journal.data());
        void* args[]={&shared.p,&control.p,&conv.p,&scheduled.p,&generation,&window.p,&journal.p};
        ck(cuLaunchKernel(accept,1,1,1,1,1,1,0,nullptr,args,nullptr),"actual accept launch");
        ck(cuCtxSynchronize(),"actual accept completion");
        shared.get(&x.shared);control.get(x.control);conv.get(x.convergence);scheduled.get(&x.scheduled);
        window.get(&x.window);journal.get(x.journal.data());
        require(!std::memcmp(x.control,before.control,64)&&x.scheduled==before.scheduled
            &&!std::memcmp(&x.window,&before.window,32),name+": control/clock/window unchanged");
        if(expected){Inputs oracle=before;
            require(biprism_accept_shared_step(oracle.shared,oracle.scheduled,generation,oracle.control[3],oracle.control[4],oracle.journal[index],&oracle.window),"CPU expected accepted");
            require(!std::memcmp(&x.shared,&oracle.shared,128)
                &&!std::memcmp(x.journal.data(),oracle.journal.data(),journal.n)
                &&!std::memcmp(x.convergence,before.convergence,1024),name+": exact native accepted state");
        }else{auto refused=before.shared;refused.sticky_fault|=64u;refused.state=3;
            require(!std::memcmp(&x.shared,&refused,128)
                &&!std::memcmp(x.journal.data(),before.journal.data(),journal.n),name+": no time/journal commit on refusal");
            for(unsigned leaf=0;leaf<8;++leaf){auto c=before; c.convergence[leaf][4]|=0x04000000u;c.convergence[leaf][2]|=0x04000000u;
                require(!std::memcmp(x.convergence[leaf],c.convergence[leaf],128),name+": exact collective terminal fault");}
        }
        std::printf("PASS native_accept %s state=%u sticky=%u accepted=%u time=%.17g\n",name.c_str(),x.shared.state,x.shared.sticky_fault,x.shared.accepted_bodies,x.shared.cumulative_accepted_time_ps);++cases;
    };
    run(original,true,"retained017");
    for(unsigned byte=0;byte<128;++byte){auto x=original;
        reinterpret_cast<unsigned char*>(&x.journal[slot(x)].step)[byte]^=1;
        require(!byte_oracle(x),"native negative CPU oracle");run(x,false,"journal_byte_"+std::to_string(byte));}
    auto continued=original;continued.shared.last_accepted_step=continued.scheduled;
    ++continued.scheduled;continued.shared.scheduled_step=continued.scheduled;++continued.shared.accepted_bodies;
    continued.shared.cumulative_accepted_time_ps+=double(continued.shared.native_dt_ps);
    ++continued.control[2];++continued.control[3];continued.control[4]+=4;
    for(auto& c:continued.convergence)c[5]+=4;
    continued.journal[slot(continued)].step=continued.shared;
    run(continued,true,"continuation");
    auto nonincreasing=original;nonincreasing.shared.cumulative_accepted_time_ps=1e30;
    nonincreasing.journal[slot(nonincreasing)].step=nonincreasing.shared;
    require(!byte_oracle(nonincreasing),"nonincreasing time oracle");run(nonincreasing,false,"nonincreasing_time");
    auto counter=original;++counter.control[3];run(counter,false,"accepted_counter_mismatch");
    auto stale=original;++stale.shared.generation;run(stale,false,"journal_generation_mismatch");
    std::printf("PASS native acceptance cases=%u; exact supplied module, no molecular trajectory claim\n",cases);
}
#endif
int main(int argc,char** argv){try{
    if(argc<2||argc>3){std::fprintf(stderr,"usage: %s RETAINED_BEFORE_PREFIX [ACTUAL_MODULE.ptx_or_cubin]\n",argv[0]);return 2;}
    const Inputs x=load(argv[1]);cpu_controls(x);
#ifdef BIPRISM_ACCEPT_NATIVE
    if(argc==3)native_controls(x,argv[2]);
#else
    require(argc==2,"native mode requires BIPRISM_ACCEPT_NATIVE build");
#endif
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
