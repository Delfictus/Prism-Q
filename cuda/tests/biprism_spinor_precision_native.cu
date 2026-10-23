// Native storage/reload control for BPS1. The Python reader independently
// evaluates the unchanged frozen equation and all three norm predicates.
#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <cstdio>
#include "nhs_biprism_spinor_precision.cuh"
__global__ void witness(const double* in,BiprismSpinorOwnerFfi* owner) {
    if(threadIdx.x||blockIdx.x)return;
    BiprismElectronicCayleyResult<double> c;
    const BiprismSpinor64 start={{in[0],in[1]},{in[2],in[3]}};
    const BiprismElectronicHermitian<double> H={in[4],in[5],in[6],in[7]};
    if(!bp_ed_hermitian_cayley(start,H,in[8],in[9],c))return;
    owner->substep_start[95]=start;owner->candidate[95]=c.psi_np1;
    // Explicit component stores to the actual accepted storage layout.
    owner->accepted[95]=owner->candidate[95];bp_spinor_publish(*owner,4,1);
}
int main(int argc,char** argv) {
    if(argc!=3)return 2;
    std::vector<double> in(11);std::ifstream input(argv[1],std::ios::binary);
    if(!input.read(reinterpret_cast<char*>(in.data()),in.size()*8))return 2;
    double* d;BiprismSpinorOwnerFfi* o;
    if(cudaMalloc(&d,in.size()*8)!=cudaSuccess||cudaMalloc(&o,sizeof(*o))!=cudaSuccess)return 3;
    if(cudaMemcpy(d,in.data(),in.size()*8,cudaMemcpyHostToDevice)!=cudaSuccess
       ||cudaMemset(o,0,sizeof(*o))!=cudaSuccess)return 3;
    witness<<<1,1>>>(d,o);
    if(cudaGetLastError()!=cudaSuccess||cudaDeviceSynchronize()!=cudaSuccess)return 4;
    BiprismSpinorOwnerFfi h;
    if(cudaMemcpy(&h,o,sizeof(h),cudaMemcpyDeviceToHost)!=cudaSuccess)return 4;
    std::ofstream output(argv[2],std::ios::binary|std::ios::trunc);
    output.write(reinterpret_cast<const char*>(&h),sizeof(h));
    cudaFree(d);cudaFree(o);
    return !output||h.state!=1||h.accepted_committed_substeps!=4||h.accepted_step!=1;
}
