"""Actual metric gather versus a serial original-edge scatter; CPU only."""
import subprocess
import tempfile
import unittest
from pathlib import Path

K=Path(__file__).resolve().parents[2]/'crates/prism-gpu/src/kernels'


class OrderedMetric(unittest.TestCase):
    def test_incident_gather_preserves_every_represented_metric_term(self):
        src=(K/'nhs_biprism_variational_so3g_v10.cuh').read_text()
        begin=src.index('__device__ __forceinline__ sym3_f32 assemble_metric_to_reg_v10(')
        end=src.index('// The exchange is replay scratch only.',begin)
        function=src[begin:end]
        terms=function[function.index('        const uint8_t u ='):function.index('            my_g.xx+=')]
        code='''
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <random>
#include "'''+str(K/'nhs_biprism_ordered_edges.cuh')+'''"
#define __device__
#define __forceinline__ inline
#define TILE_CORE_VERTICES 192
inline void __syncthreads(){}
struct {uint32_t x;} threadIdx;
struct V {double x,y,z;};
struct sym3_f32 {float xx,yy,zz,xy,xz,yz;};
struct sym3_f64 {double xx,yy,zz,xy,xz,yz;};
struct Numerical {double nuclear_metric[6][192];};
void bp_ns_store_metric64(double a[6][192],unsigned u,sym3_f64 g){a[0][u]=g.xx;a[1][u]=g.yy;a[2][u]=g.zz;a[3][u]=g.xy;a[4][u]=g.xz;a[5][u]=g.yz;}
struct ContactShellSimplicialStateFfi {};
struct Streams {BiprismOrderedEdgesFfi* d_ordered_edges;};
struct VariationalClusterDsmemV10 {
    struct {struct {Streams* streams;uint32_t leaf;} collective;Numerical* numerical;} invocation;
    uint8_t edge_u8[1536][2];V x[256],ref[256];
    struct {sym3_f32 g_cov[192];} stage;
};
V diff(V a,V b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
V bp_v10_edge_difference(VariationalClusterDsmemV10* s,unsigned u,unsigned v){return diff(s->x[u],s->x[v]);}
V bp_v10_reference_difference(VariationalClusterDsmemV10* s,unsigned u,unsigned v){return diff(s->ref[u],s->ref[v]);}
double bp_geometry_length_squared(V a){return std::fma(a.x,a.x,std::fma(a.y,a.y,a.z*a.z))+1e-12;}
'''+function+'''
int main(){std::mt19937 rng(57056);std::uniform_real_distribution<double> d(-3.,3.);
 for(unsigned count: {1u,25u,1461u,1536u}){
  VariationalClusterDsmemV10 data{};Numerical numerical{};data.invocation.numerical=&numerical;auto* smem=&data;BiprismOrderedEdgesFfi index{};Streams streams{&index};
  smem->invocation.collective={&streams,0};index.num_edges=count;
  for(unsigned u=0;u<256;++u){smem->ref[u]={d(rng),d(rng),d(rng)};smem->x[u]=smem->ref[u];smem->x[u].x+=.01*d(rng);}
  for(unsigned e=0;e<count;++e){smem->edge_u8[e][0]=rng()%256;smem->edge_u8[e][1]=(smem->edge_u8[e][0]+1+rng()%255)%256;}
  for(unsigned u=0;u<192;++u)index.offsets[u+1]=biprism_ordered_edge_count_row(&smem->edge_u8[0][0],&smem->edge_u8[0][1],2,count,u);
  if(!biprism_ordered_edge_prefix(index))return 1;
  for(unsigned u=0;u<192;++u)if(!biprism_ordered_edge_fill_row(index,&smem->edge_u8[0][0],&smem->edge_u8[0][1],2,u))return 2;
  sym3_f64 reference[256];for(auto& g:reference)g={1,1,1,0,0,0};float chi_g=.12f;
  for(unsigned e=0;e<count;++e){
'''+terms+'''
   for(unsigned owner: {unsigned(u),unsigned(v)})if(owner<192){auto& g=reference[owner];g.xx+=s_xx;g.yy+=s_yy;g.zz+=s_zz;g.xy+=s_xy;g.xz+=s_xz;g.yz+=s_yz;}
  }
  for(unsigned u=0;u<256;++u){threadIdx.x=u;auto got=assemble_metric_to_reg_v10(smem,nullptr,count,chi_g);
   const auto ref=reference[u];const sym3_f32 projected={float(ref.xx),float(ref.yy),float(ref.zz),float(ref.xy),float(ref.xz),float(ref.yz)};
   if(std::memcmp(&got,&projected,sizeof(got)))return 3;
   if(u<192){const sym3_f64 actual={numerical.nuclear_metric[0][u],numerical.nuclear_metric[1][u],numerical.nuclear_metric[2][u],numerical.nuclear_metric[3][u],numerical.nuclear_metric[4][u],numerical.nuclear_metric[5][u]};
    if(std::memcmp(&actual,&ref,sizeof(ref)))return 4;}}
 }
 std::puts("ACTUAL_METRIC_GATHER_CPU_BIT_PARITY_PASS");
}
'''
        with tempfile.TemporaryDirectory(prefix='biprism_metric_cpu_') as tmp:
            p=Path(tmp);(p/'a.cpp').write_text(code)
            subprocess.run(['g++','-std=c++17','-O2','-fno-fast-math',str(p/'a.cpp'),'-o',str(p/'a')],check=True,capture_output=True)
            r=subprocess.run([str(p/'a')],check=True,capture_output=True,text=True)
            self.assertIn('BIT_PARITY_PASS',r.stdout)


if __name__=='__main__':unittest.main()
