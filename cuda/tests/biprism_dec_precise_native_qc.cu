// Actual retained operators, independent true-residual readback, two solvers.
// This numerical control does not confer coupled trajectory acceptance.
#include <cstdio>
#include <fstream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include "nhs_biprism_simplicial_tile.cuh"

struct DecPreciseCase {
    uint32_t edges,leaf;
    uint8_t uv[TILE_MAX_EDGES][2];
    double weight[TILE_MAX_EDGES],charge[256],initial_phi[256];
};
static_assert(sizeof(DecPreciseCase)==19464,"retained DEC control schema");
struct DecPreciseResult {uint32_t accepted,iterations;float residual;uint32_t reserved;};
struct DecPreciseWeights {
    const double* p;uint32_t control;
    __device__ __forceinline__ double operator()(uint32_t e)const {
        return e==0u&&control==1u?CUDART_NAN:(e==0u&&control==2u?-p[e]:p[e]);
    }
};
__global__ __launch_bounds__(256,2) void biprism_dec_precise_case(
    const DecPreciseCase* input,double* output,DecPreciseResult* result,
    uint32_t solver,uint32_t cold,uint32_t control
) {
    __shared__ double phi[256],direction[256];
    __shared__ float r[256],z[256],p[256],ap[256],inv[256],reduce[32];
    const uint32_t tid=threadIdx.x,leaf=blockIdx.x;const auto& c=input[leaf];
    phi[tid]=cold&&tid<TILE_CORE_VERTICES?0.:c.initial_phi[tid];
    __syncthreads();
    const double charge=c.charge[tid]+(control==3u?double(.15f)*.0123456789012345:0.);
    float residual=CUDART_INF_F;uint32_t iterations=0;
    const DecPreciseWeights weights{c.weight,control};
    const bool accepted=solver?
        biprism_dec_pcg_256_precise_traversal(&c.uv[0][0],&c.uv[0][1],2u,
            weights,c.edges,BiprismLinearEdgeTraversal{c.edges},charge,
            phi,direction,reduce,residual,iterations,1e-20):
        biprism_dec_pcg_256_traversal(&c.uv[0][0],&c.uv[0][1],2u,
            weights,c.edges,BiprismLinearEdgeTraversal{c.edges},charge,
            phi,r,z,p,ap,inv,reduce,residual,iterations,1e-20);
    output[leaf*256+tid]=phi[tid];
    if(tid==0u)result[leaf]={accepted?1u:0u,iterations,residual,0u};
}
void checked(cudaError_t e,const char* op){if(e!=cudaSuccess)throw std::runtime_error(std::string(op)+": "+cudaGetErrorString(e));}
int main(int argc,char** argv){
    try{
        if(argc!=3)throw std::runtime_error("usage: precise_dec_qc retained_cases.bin output_prefix");
        std::vector<DecPreciseCase> cases(8);
        std::ifstream in(argv[1],std::ios::binary);
        in.read(reinterpret_cast<char*>(cases.data()),cases.size()*sizeof(cases[0]));
        if(!in||in.peek()!=std::char_traits<char>::eof())throw std::runtime_error("case extent");
        for(uint32_t l=0;l<8;++l){const auto& c=cases[l];
            if(c.leaf!=l||!c.edges||c.edges>TILE_MAX_EDGES)throw std::runtime_error("case identity");
            for(uint32_t e=0;e<c.edges;++e)if(c.uv[e][0]==c.uv[e][1]||!std::isfinite(c.weight[e])||c.weight[e]<=0.)throw std::runtime_error("case edge");
            for(uint32_t v=0;v<256;++v)if(!std::isfinite(c.charge[v])||!std::isfinite(c.initial_phi[v]))throw std::runtime_error("case value");
        }
        DecPreciseCase* d;double* dp;DecPreciseResult* ds;
        checked(cudaMalloc(&d,cases.size()*sizeof(cases[0])),"allocate cases");
        checked(cudaMalloc(&dp,8*256*sizeof(double)),"allocate output");
        checked(cudaMalloc(&ds,8*sizeof(DecPreciseResult)),"allocate result");
        checked(cudaMemcpy(d,cases.data(),cases.size()*sizeof(cases[0]),cudaMemcpyHostToDevice),"upload");
        std::ofstream phi_out(std::string(argv[2])+".phi.bin",std::ios::binary);
        std::ofstream result_out(std::string(argv[2])+".result.bin",std::ios::binary);
        bool all=true;uint32_t old_refusals=0;
        for(uint32_t solver=0;solver<2;++solver)for(uint32_t cold=0;cold<2;++cold)for(uint32_t control=0;control<4;++control){
            biprism_dec_precise_case<<<8,256>>>(d,dp,ds,solver,cold,control);
            checked(cudaGetLastError(),"launch");checked(cudaDeviceSynchronize(),"complete");
            std::vector<double> phi(8*256);std::vector<DecPreciseResult> status(8);
            checked(cudaMemcpy(phi.data(),dp,phi.size()*sizeof(double),cudaMemcpyDeviceToHost),"read phi");
            checked(cudaMemcpy(status.data(),ds,status.size()*sizeof(status[0]),cudaMemcpyDeviceToHost),"read result");
            phi_out.write(reinterpret_cast<char*>(phi.data()),phi.size()*sizeof(double));
            result_out.write(reinterpret_cast<char*>(status.data()),status.size()*sizeof(status[0]));
            for(uint32_t l=0;l<8;++l){
                const auto& c=cases[l];long double rr=0.;bool halo=true;
                for(uint32_t v=0;v<TILE_CORE_VERTICES;++v){
                    long double row=0.;
                    for(uint32_t e=0;e<c.edges;++e){const uint32_t u=c.uv[e][0],w=c.uv[e][1];
                        if(u==v)row+=(long double)c.weight[e]*((long double)phi[l*256+v]-phi[l*256+w]);
                        else if(w==v)row+=(long double)c.weight[e]*((long double)phi[l*256+v]-phi[l*256+u]);}
                    const double q=c.charge[v]+(control==3u?double(.15f)*.0123456789012345:0.);
                    const long double r=(long double)q-row;rr+=r*r;
                }
                for(uint32_t v=TILE_CORE_VERTICES;v<256;++v)halo=halo&&phi[l*256+v]==c.initial_phi[v];
                const bool positive=control==0u||control==3u;
                const bool pass=positive?(status[l].accepted&&std::isfinite(rr)&&rr<=1e-20L&&halo&&status[l].iterations<=192u):!status[l].accepted;
                if(solver)all=all&&pass;
                else if(positive&&!status[l].accepted)++old_refusals;
                std::printf("solver=%u cold=%u control=%u leaf=%u accepted=%u iterations=%u native=%.9g independent=%.17g halo=%u pass=%u\n",
                    solver,cold,control,l,status[l].accepted,status[l].iterations,status[l].residual,double(sqrtl(rr)),halo,pass);
            }
        }
        if(!phi_out||!result_out)throw std::runtime_error("readback write");
        checked(cudaFree(d),"free cases");checked(cudaFree(dp),"free output");checked(cudaFree(ds),"free result");
        std::printf("PRECISE_DEC_CASES=64 OLD_POSITIVE_REFUSALS=%u NUMERICAL_CONTROL_PASS=%u COUPLED_ACCEPTANCE=0\n",old_refusals,all);
        return all?0:4;
    }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 2;}
}
