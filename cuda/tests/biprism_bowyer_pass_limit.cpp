// Actual CPU_REFERENCE body regression; this does not execute a GPU kernel.
// Build from roles/engine:
//   g++ -std=c++17 -O2 -Icrates/prism-gpu/src/kernels \
//     cuda/tests/biprism_bowyer_pass_limit.cpp -o /tmp/biprism_bowyer_pass_limit
// Run with three exact retained files from /mnt/storage/BIPRISM_V10_RESOLUTION:
//   /tmp/biprism_bowyer_pass_limit \
//     failure_1_2_3_coupled_native_007_state.seed_tiles.bin \
//     failure_1_2_3_coupled_native_007_state.cold.au_reference_high.bin \
//     failure_1_2_3_coupled_native_007_state.cold.au_reference_low.bin
// Input SHA256, in argument order:
//   5ed733573d17f43fde83884016698271f9316791f725d1e15584a762de7e52db
//   b9a14e22d13a851ad3f9301c9325c3db8402135382dfb050eea48b2c0cd196dc
//   e4d9f0f07556af427f082f5b649020216857dd77d87c07238218a00c8817f85e
// Original exact-body execution log: failure_6_bw_peel_limit_cpu.txt in that
// directory, SHA256 7e0728c290afefb9c4bb5ac840ee152f105591f00bc8f51868271b5908a9d1a5.
// Native007 published the measured mesh, then refused the later main step.
// These tests cover source execution, unchanged mesh identity and pass bounds.
#define BIPRISM_BW_CPU_REFERENCE
#include "nhs_biprism_bowyer_watson_seed.cuh"
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <vector>
static std::vector<unsigned char> read(const char* path){
 std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("input open");
 return {std::istreambuf_iterator<char>(f),{}};
}
static uint32_t u32(const unsigned char*p){uint32_t v;std::memcpy(&v,p,4);return v;}
static float f32(const unsigned char*p){float v;std::memcpy(&v,p,4);return v;}
static void require(bool x,const char* s){if(!x)throw std::runtime_error(s);}
static bool unchanged_topology(const BiprismBwScratchFfi&a,const BiprismBwScratchFfi&b){
 return !std::memcmp(a.active,b.active,sizeof(a.active))&&!std::memcmp(a.faces,b.faces,sizeof(a.faces))
 &&!std::memcmp(a.adjacency,b.adjacency,sizeof(a.adjacency))&&!std::memcmp(a.vertex_degree,b.vertex_degree,sizeof(a.vertex_degree))
 &&!std::memcmp(a.edge_degree,b.edge_degree,sizeof(a.edge_degree))&&!std::memcmp(a.boundary_vertex_degree,b.boundary_vertex_degree,sizeof(a.boundary_vertex_degree))
 &&!std::memcmp(a.boundary_edge_degree,b.boundary_edge_degree,sizeof(a.boundary_edge_degree));
}
int main(int argc,char**argv){try{
 require(argc==4,"usage: tiles au_reference_hi au_reference_lo");
 static_assert(BPBW_MAX_PEEL_PASSES==256u,"public hard scan ceiling");
 const auto tiles=read(argv[1]),hi=read(argv[2]),lo=read(argv[3]);
 constexpr size_t tile_size=225408;
 require(tiles.size()==8*tile_size&&hi.size()==lo.size()&&hi.size()%(8*12)==0,"input extents");
 const size_t atoms=hi.size()/(8*12);
 auto scratch=std::make_unique<BiprismBwScratchFfi>();auto&s=*scratch;
 for(unsigned leaf=0;leaf<8;++leaf){const auto*t=tiles.data()+leaf*tile_size;
  require(u32(t+225280)==960&&u32(t+225284)==1461,"retained mesh shape");
  for(unsigned v=0;v<256;++v){const unsigned atom=u32(t+v*4)&0xffffffu;require(atom<atoms,"selected ID");
   double xyz[3];for(unsigned k=0;k<3;++k){const size_t off=((leaf*atoms+atom)*3+k)*4;xyz[k]=double(f32(hi.data()+off))+double(f32(lo.data()+off));}
   const BiprismBwPoint p{xyz[0],xyz[1],xyz[2]};
   if(leaf==0){s.points[v]=p;s.selected_ids[v]=atom;}
   else require(!std::memcmp(&s.points[v],&p,sizeof p)&&s.selected_ids[v]==atom,"eight actual input populations differ");
  }
  require(!std::memcmp(t+5120,tiles.data()+5120,960*4),"eight retained tet populations differ");
 }
 BiprismBwStatusFfi r{};require(bp_bw_construct(s,r,960,1536)==BPBW_OK,"actual input construct refusal");
 require(r.phase==5&&r.peel_passes==1&&r.removed_tets==558&&r.final_tets==960&&r.final_edges==1461&&r.published==0,"actual counters");
 unsigned k=0;for(unsigned i=0;i<r.unpeeled_tets;++i)if(s.active[i]){
  for(unsigned j=0;j<4;++j)require(bp_bw_vertex(s.tetrahedra[0][i],j)==tiles[5120+4*k+j],"retained ordered tet identity");++k;}
 require(k==960,"tet population");
 std::printf("PASS actual native007 full high+low population: all 8 inputs identical; one exact CPU body reconstruction matches all 8 ordered 960-tet populations; edges=1461 passes=1 removed=558\n");
 auto base=std::make_unique<BiprismBwScratchFfi>();for(unsigned v=0;v<256;++v)base->points[v]=s.points[v];
 for(unsigned i=0;i<960;++i){const auto*t=tiles.data()+5120+4*i;base->tetrahedra[0][i]={t[0],t[1],t[2],t[3],0};base->active[i]=1;}
 BiprismBwStatusFfi br{};require(bp_bw_build_incidence(*base,br,960)==BPBW_OK,"compact incidence");
 auto trial=std::make_unique<BiprismBwScratchFfi>(*base);BiprismBwStatusFfi rr{};rr.peel_passes=256;
 require(bp_bw_peel(*trial,rr,960,959,1536)==BPBW_PEEL_BUDGET_UNREACHABLE&&rr.peel_passes==256&&rr.removed_tets==0&&rr.published==0&&unchanged_topology(*base,*trial),"pass257 must refuse before topology mutation");
 std::puts("PASS pass257 refused at counter256; no active/incidence/degree/adjacency change, no publication");
 *trial=*base;rr={};rr.peel_passes=255;
 require(bp_bw_peel(*trial,rr,960,959,1536)==BPBW_OK&&rr.peel_passes==256&&rr.removed_tets==1&&rr.final_tets==959&&rr.published==0,"pass256 must remain allowed");
 std::puts("PASS pass256 allowed from counter255 and existing target ends scan after one removal");
 *trial=*base;rr={};rr.peel_passes=256;
 require(bp_bw_peel(*trial,rr,960,960,1536)==BPBW_OK&&rr.peel_passes==256&&rr.removed_tets==0&&rr.final_tets==960&&rr.final_edges==1461&&unchanged_topology(*base,*trial),"already satisfied targets must not require a pass");
 std::puts("PASS already satisfied capacity at counter256 succeeds without an extra pass");
 return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
