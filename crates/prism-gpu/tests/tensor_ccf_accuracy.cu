// Numerical integration test for the actual production WMMA CCF kernel.
// This validates the kernel's defined normalized dot product, not a campaign.
// Compile with nvcc -O3 -std=c++17 -arch=sm_120 this_file.cu -o test_ccf.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <vector>
#include "../src/kernels/tensor_ccf.cu"

static void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
template<class T> struct Buffer {
    T* ptr = nullptr;
    explicit Buffer(size_t count) { check(cudaMalloc(&ptr, count * sizeof(T))); }
    ~Buffer() { if (ptr) cudaFree(ptr); }
};

static bool run_case(int n, int bins, bool identical, std::ofstream& rows) {
    const int np = (n + 15) / 16 * 16, kp = (bins + 15) / 16 * 16;
    std::vector<half> a(np * kp), b(np * kp);
    // Exact dyadic inputs. For bins <= 4096, all squared/product sums
    // have integer numerators <= 31^2*4096 < 2^24 and denominator 64.
    // Thus FP32 accumulation is exact here; the 2e-6 absolute bound
    // covers the rounded sqrt/multiply/add/divide, including near zero.
    // Padding and row zero remain zero; unequal A/B catches transpose bugs.
    for (int r = 1; r < n; ++r) for (int k = 0; k < bins; ++k) {
        int x = ((r * 17 + k * 7 + (r * k) % 19) % 32);
        int y = ((r * 3 + k * 23 + (r * k) % 11) % 32);
        if ((r + k) % 5 == 0) x = 0;
        if ((r + 2 * k) % 3 == 0) y = 0;
        a[r * kp + k] = __float2half(float(x) / 8);
        b[r * kp + k] = identical ? a[r * kp + k] : __float2half(float(y) / 8);
    }
    Buffer<half> da(a.size()), db(b.size());
    Buffer<float> na(n), nb(n), out(n * n);
    check(cudaMemcpy(da.ptr, a.data(), a.size()*sizeof(half), cudaMemcpyHostToDevice));
    check(cudaMemcpy(db.ptr, b.data(), b.size()*sizeof(half), cudaMemcpyHostToDevice));
    check(cudaMemset(out.ptr, 0xff, n*n*sizeof(float)));
    compute_spike_norms<<<(n+127)/128,128>>>(da.ptr,na.ptr,n,kp);
    compute_spike_norms<<<(n+127)/128,128>>>(db.ptr,nb.ptr,n,kp);
    check(cudaGetLastError());
    tensor_ccf_compute<<<dim3((np+63)/64,(np+15)/16),dim3(32,4)>>>(
        da.ptr,db.ptr,out.ptr,na.ptr,nb.ptr,n,np,kp);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    std::vector<float> actual(n*n);
    check(cudaMemcpy(actual.data(),out.ptr,n*n*sizeof(float),cudaMemcpyDeviceToHost));
    std::vector<double> an(n,0),bn(n,0),reference(n*n,0);
    for(int r=0;r<n;++r) for(int k=0;k<bins;++k) {
        double av=__half2float(a[r*kp+k]),bv=__half2float(b[r*kp+k]);
        an[r]+=av*av;bn[r]+=bv*bv;
    }
    for(int r=0;r<n;++r) {an[r]=std::sqrt(an[r]);bn[r]=std::sqrt(bn[r]);}
    double max_abs=0, transpose_delta=0;size_t failures=0;
    for(int r=0;r<n;++r) for(int c=0;c<n;++c) {
        double dot=0;
        for(int k=0;k<bins;++k)
            dot+=double(__half2float(a[r*kp+k]))*double(__half2float(b[c*kp+k]));
        double expected=dot/(an[r]*bn[c]+double(1e-8f));
        reference[r*n+c]=expected;
        double err=std::abs(double(actual[r*n+c])-expected);
        max_abs=std::max(max_abs,err);
        if(!std::isfinite(actual[r*n+c]) || err>2e-6) ++failures;
        rows<<n<<'\t'<<bins<<'\t'<<identical<<'\t'<<r<<'\t'<<c<<'\t'
            <<actual[r*n+c]<<'\t'<<expected<<'\t'<<err<<'\n';
    }
    for(int r=0;r<n;++r) for(int c=0;c<n;++c)
        transpose_delta=std::max(transpose_delta,std::abs(reference[r*n+c]-reference[c*n+r]));
    if(!identical && n>2 && transpose_delta<1e-3) ++failures;
    // Stream-synchronized timing is a kernel microbenchmark, never MD steps/sec.
    cudaEvent_t start,stop;check(cudaEventCreate(&start));check(cudaEventCreate(&stop));
    check(cudaEventRecord(start));
    for(int repeat=0;repeat<100;++repeat)
        tensor_ccf_compute<<<dim3((np+63)/64,(np+15)/16),dim3(32,4)>>>(
            da.ptr,db.ptr,out.ptr,na.ptr,nb.ptr,n,np,kp);
    check(cudaEventRecord(stop));check(cudaEventSynchronize(stop));
    float ms=0;check(cudaEventElapsedTime(&ms,start,stop));
    check(cudaEventDestroy(start));check(cudaEventDestroy(stop));
    std::printf("{\"rows\":%d,\"bins\":%d,\"identical\":%s,\"max_absolute_error\":%.12g,\"transpose_sensitivity\":%.12g,\"failures\":%zu,\"ccf_kernel_mean_ms\":%.9g}\n",
        n,bins,identical?"true":"false",max_abs,transpose_delta,failures,ms/100);
    return failures==0;
}

int main(int argc,char** argv) {
    if(argc!=2) {std::fprintf(stderr,"usage: tensor_ccf_accuracy output.tsv\n");return 2;}
    try {
        std::ofstream rows(argv[1]);if(!rows) throw std::runtime_error("cannot open output");
        rows<<std::setprecision(17)<<"rows\tbins\tidentical\trow\tcol\tactual\treference\tabs_error\n";
        int count=0;check(cudaGetDeviceCount(&count));if(count<1) throw std::runtime_error("GPU required");
        bool ok=true;
        for(auto shape:std::vector<std::pair<int,int>>{{2,1},{17,17},{31,100},{65,127},{129,4096},{257,1024}})
            ok=run_case(shape.first,shape.second,false,rows)&&ok;
        ok=run_case(65,127,true,rows)&&ok;
        rows.flush();if(!rows) throw std::runtime_error("failed to write output");
        return ok?0:1;
    } catch(const std::exception& e) {std::fprintf(stderr,"ERROR: %s\n",e.what());return 2;}
}
