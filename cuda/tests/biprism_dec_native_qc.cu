#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <cmath>
#include <limits>
#include "nhs_biprism_simplicial_tile.cuh"

struct DecQcStatus { uint32_t accepted, iterations; float residual; uint32_t control; };
__global__ void check_actual_dec(const ContactShellSimplicialStateFfi* tile,
    float* output, DecQcStatus* status, uint32_t control) {
    __shared__ float phi[256], r[256], z[256], p[256], ap[256], diagonal[256], sums[8];
    __shared__ float weights[TILE_MAX_EDGES];
    const uint32_t tid=threadIdx.x, leaf=blockIdx.x;
    const auto& t=tile[leaf];
    for(uint32_t e=tid;e<t.num_active_edges;e+=256u)
        weights[e]=(e==0u && control==1u)?CUDART_NAN_F:
            (e==0u && control==2u)?-t.star1_wt[e]:t.star1_wt[e];
    phi[tid]=t.ref_pos_q[tid].w*0.07957747f;
    __syncthreads();
    float residual=CUDART_INF_F;uint32_t iterations=0u;
    // The electronic endpoint charge is formed in FP64. This control is
    // deliberately not representable in one float, while phi remains FP32.
    const double charge=double(t.ref_pos_q[tid].w)
        +(control==3u ? double(0.15f)*0.0123456789012345 : 0.0);
    const bool accepted=biprism_dec_pcg_256(
        &t.edge_u8[0][0],&t.edge_u8[0][1],2u,weights,t.num_active_edges,charge,
        phi,r,z,p,ap,diagonal,sums,residual,iterations);
    output[leaf*256u+tid]=phi[tid];
    if(tid==0u)status[leaf]={accepted?1u:0u,iterations,residual,control};
}
static void check(cudaError_t result,const char* operation) {
    if(result!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",operation,cudaGetErrorString(result));std::exit(2);}
}
int main(int argc,char** argv) {
    if(argc!=2){std::fprintf(stderr,"usage: dec_qc actual_phase2_tiles.bin\n");return 2;}
    std::vector<ContactShellSimplicialStateFfi> tiles(8);
    std::ifstream file(argv[1],std::ios::binary);
    file.read(reinterpret_cast<char*>(tiles.data()),tiles.size()*sizeof(tiles[0]));
    if(!file || file.peek()!=std::char_traits<char>::eof())return 2;
    for(const auto&t:tiles)if(t.num_active_edges==0||t.num_active_edges>TILE_MAX_EDGES)return 2;
    ContactShellSimplicialStateFfi* dt=nullptr;float* dp=nullptr;DecQcStatus* ds=nullptr;
    check(cudaMalloc(&dt,tiles.size()*sizeof(tiles[0])),"allocate tiles");
    check(cudaMalloc(&dp,8u*256u*sizeof(float)),"allocate phi");
    check(cudaMalloc(&ds,8u*sizeof(DecQcStatus)),"allocate status");
    check(cudaMemcpy(dt,tiles.data(),tiles.size()*sizeof(tiles[0]),cudaMemcpyHostToDevice),"upload tiles");
    bool all=true;
    for(uint32_t control=0;control<4;++control){
        check_actual_dec<<<8,256>>>(dt,dp,ds,control);
        check(cudaGetLastError(),"launch actual DEC");check(cudaDeviceSynchronize(),"actual DEC completion");
        std::vector<float> phi(8u*256u);std::vector<DecQcStatus> status(8);
        check(cudaMemcpy(phi.data(),dp,phi.size()*sizeof(float),cudaMemcpyDeviceToHost),"read phi");
        check(cudaMemcpy(status.data(),ds,status.size()*sizeof(DecQcStatus),cudaMemcpyDeviceToHost),"read status");
        for(unsigned leaf=0;leaf<8;++leaf){
            double residual2=0.0;bool halo=true;
            for(unsigned v=0;v<192;++v){
                double row=0.0;
                for(unsigned e=0;e<tiles[leaf].num_active_edges;++e){
                    unsigned u=tiles[leaf].edge_u8[e][0],w=tiles[leaf].edge_u8[e][1];
                    if(u==v)row+=double(tiles[leaf].star1_wt[e])*(double(phi[leaf*256+v])-double(phi[leaf*256+w]));
                    else if(w==v)row+=double(tiles[leaf].star1_wt[e])*(double(phi[leaf*256+v])-double(phi[leaf*256+u]));
                }
                const double charge=double(tiles[leaf].ref_pos_q[v].w)
                    +(control==3u ? double(0.15f)*0.0123456789012345 : 0.0);
                double residual=charge-row;residual2+=residual*residual;
            }
            for(unsigned v=192;v<256;++v)halo &= phi[leaf*256+v]==tiles[leaf].ref_pos_q[v].w*0.07957747f;
            bool pass=(control==0u||control==3u) ? status[leaf].accepted&&std::isfinite(residual2)
                &&residual2<=double(BIPRISM_DEC_TOL_SQ)&&status[leaf].iterations<=BIPRISM_DEC_MAX_ITERS&&halo
                : !status[leaf].accepted;
            all &= pass;
            std::printf("control=%u leaf=%u accepted=%u iterations=%u reported=%.9g true_residual=%.17g halo_unchanged=%u pass=%u\n",
                control,leaf,status[leaf].accepted,status[leaf].iterations,status[leaf].residual,std::sqrt(residual2),halo,pass);
        }
    }
    check(cudaFree(dt),"free tiles");check(cudaFree(dp),"free phi");check(cudaFree(ds),"free status");
    return all?0:1;
}
