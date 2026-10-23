"""Independent residual-space orthogonality controls for the actual helper."""
from pathlib import Path
import subprocess
import tempfile
import unittest

K=Path(__file__).resolve().parents[2]/'crates/prism-gpu/src/kernels/nhs_biprism_variational_so3g_v10.cuh'

class ImplicitArnoldi(unittest.TestCase):
    def test_next_vector_is_orthogonal_to_base_and_previous_output_space(self):
        s=K.read_text();a=s.index('__device__ __forceinline__ double bp_v10_arnoldi_component');b=s.index('// Fixed-base matrix-free',a)
        code='''#include <cmath>
#include <cassert>
#include <random>
#define __host__
#define __device__
#define __forceinline__ inline
#define __fma_rn std::fma
'''+s[a:b]+'''
int main(){
 std::mt19937_64 rng(145);std::uniform_real_distribution<double> u(-1.,1.);
 for(unsigned repeat=0;repeat<200;++repeat){
  double Q[4][12],f[12],r[12],co[4];
  for(auto &v:Q){for(double &x:v)x=u(rng);}
  for(unsigned j=0;j<4;++j){
   for(unsigned pass=0;pass<2;++pass)for(unsigned k=0;k<j;++k){double d=0.;for(unsigned i=0;i<12;++i)d+=Q[j][i]*Q[k][i];for(unsigned i=0;i<12;++i)Q[j][i]-=d*Q[k][i];}
   double n=0.;for(double x:Q[j])n+=x*x;for(double &x:Q[j])x/=std::sqrt(n);
  }
  for(unsigned i=0;i<12;++i)r[i]=f[i]=u(rng);
  for(unsigned j=0;j<4;++j){co[j]=0.;for(unsigned i=0;i<12;++i)co[j]+=Q[j][i]*f[i];for(unsigned i=0;i<12;++i)r[i]-=co[j]*Q[j][i];}
  double n=0.;for(double x:r)n+=x*x;
  double v[12];for(unsigned i=0;i<12;++i)v[i]=bp_v10_arnoldi_component(Q[3][i],r[i],co[3],n);
  double dot=0.;for(unsigned i=0;i<12;++i)dot+=v[i]*f[i];assert(std::abs(dot)<4e-15);
  for(unsigned j=0;j<3;++j){dot=0.;for(unsigned i=0;i<12;++i)dot+=v[i]*Q[j][i];assert(std::abs(dot)<4e-15);}
  // Compare direct projection off span(previous Q, base F), independently.
  double old_r[12],old_n=0.;for(unsigned i=0;i<12;++i){old_r[i]=r[i]+co[3]*Q[3][i];old_n+=old_r[i]*old_r[i];}
  for(unsigned i=0;i<12;++i)assert(std::abs(v[i]-(Q[3][i]-co[3]*old_r[i]/old_n))<3e-15);
 }
}
'''
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'a.cpp').write_text(code)
            subprocess.run(['g++','-std=c++17','-O2',str(p/'a.cpp'),'-o',str(p/'a')],check=True,capture_output=True)
            subprocess.run([str(p/'a')],check=True,capture_output=True)

if __name__=='__main__':unittest.main()
