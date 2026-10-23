// Native numerical controls for the actual production Q39 conversion helper.
// Constructed antisymmetric vectors test the numerical operator, not a trajectory.
#include "nhs_biprism_voronoi_lif.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <stdexcept>

static void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
__global__ void exercise_balanced_conversion(const unsigned long long* in,
    float* out, uint32_t* metadata) {
    __shared__ unsigned long long q[256][3];
    __shared__ uint32_t shift, up, fault;
    const unsigned tid=threadIdx.x, b=blockIdx.x;
    for (unsigned a=0;a<3;++a) q[tid][a]=in[b*768+tid*3+a];
    if (!tid) fault=0;
    __syncthreads();
    for (unsigned a=0;a<3;++a) {
        const float v=biprism_balanced_q39_axis_f32(q,a,shift,up,fault);
        out[b*768+tid*3+a]=v;
        if (!tid) metadata[b*4+a]=shift;
        __syncthreads();
    }
    if (!tid) metadata[b*4+3]=fault;
}
int main() {
    try {
        constexpr unsigned cases=129;
        std::vector<long long> input(cases*768,0);
        uint64_t rng=0x6013c0de5eedULL;
        auto random=[&](){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;return rng;};
        for(unsigned b=1;b<cases-1;++b) {
            const unsigned scale=b%37;
            for(unsigned e=0;e<768;++e) {
                const unsigned u=random()%256,v=random()%256,a=random()%3;
                const long long k=(static_cast<long long>(random()%65537)-32768)*(1LL<<scale);
                input[b*768+3*u+a]+=k;input[b*768+3*v+a]-=k;
            }
        }
        // Rejection control: corrupt a zero vector's sum without overflowing.
        input[(cases-1)*768]=1;
        unsigned long long* di=nullptr;float* dout=nullptr;uint32_t* dm=nullptr;
        check(cudaMalloc(&di,input.size()*sizeof(long long)));
        check(cudaMalloc(&dout,input.size()*sizeof(float)));
        check(cudaMalloc(&dm,cases*4*sizeof(uint32_t)));
        check(cudaMemcpy(di,input.data(),input.size()*sizeof(long long),cudaMemcpyHostToDevice));
        exercise_balanced_conversion<<<cases,256>>>(di,dout,dm);
        check(cudaGetLastError());check(cudaDeviceSynchronize());
        std::vector<float> output(input.size());std::vector<uint32_t> meta(cases*4);
        check(cudaMemcpy(output.data(),dout,output.size()*sizeof(float),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(meta.data(),dm,meta.size()*sizeof(uint32_t),cudaMemcpyDeviceToHost));
        for(unsigned b=0;b<cases-1;++b) {
            if(meta[b*4+3])throw std::runtime_error("valid vector refused");
            for(unsigned a=0;a<3;++a) {
                __int128 sum=0;
                const long long quantum=1LL<<meta[b*4+a];
                for(unsigned v=0;v<256;++v) {
                    const size_t i=b*768+v*3+a;
                    const double represented=static_cast<double>(output[i])*549755813888.0;
                    const long long j=static_cast<long long>(represented),k=input[i];
                    if(represented!=static_cast<double>(j))throw std::runtime_error("output off Q39 lattice");
                    const long long delta=j>=k?j-k:k-j;
                    if(delta>=quantum)throw std::runtime_error("rounding quantum bound violated");
                    if((k==0&&j!=0)||(k<0&&j>0)||(k>0&&j<0))throw std::runtime_error("zero/sign changed");
                    sum+=j;
                }
                if(sum!=0)throw std::runtime_error("represented force vector does not close exactly");
            }
        }
        if(!meta[(cases-1)*4+3])throw std::runtime_error("nonzero-sum input accepted");
        check(cudaFree(di));check(cudaFree(dout));check(cudaFree(dm));
        std::printf("native_balanced_rounding_cases=128 axes=384 exact_closure=1 quantum_bound=1 zero_sign=1 malformed_sum_rejected=1\n");
        return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
}
