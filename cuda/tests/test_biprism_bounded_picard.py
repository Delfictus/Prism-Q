"""Execute the actual bounded scalar predictor; CPU evidence only."""
from pathlib import Path
import subprocess
import tempfile
import unittest

K=Path(__file__).resolve().parents[2]/'crates/prism-gpu/src/kernels'

class BoundedPicard(unittest.TestCase):
    def test_actual_increment_bootstrap_secant_and_transient_bounds(self):
        source=(K/'nhs_biprism_variational_so3g_v10.cuh').read_text()
        start=source.index('__host__ __device__ __forceinline__ double bp_v10_bounded_alpha_increment(')
        end=source.index('// Only the NEXT trial',start)
        code = '#include <cmath>\n#include <limits>\nusing namespace std;\n#define __host__\n#define __device__\n#define __forceinline__ inline\n'+source[start:end]+r"""
int main(){
 if(bp_v10_bounded_alpha_increment(3.,0.,0.,0.,0.,false,false,.5)!=0.)return 1;
 if(bp_v10_bounded_alpha_increment(93.317,1.,0.,0.,0.,false,false,.5)!=.5)return 2;
 if(bp_v10_bounded_alpha_increment(-93.317,1.,0.,0.,0.,false,false,.5)!=-.5)return 3;
 // Exact local R(alpha)=.4-2alpha: Newton proposal .1 at alpha=.1.
 if(fabs(bp_v10_bounded_alpha_increment(.2,1.,.1,.4,0.,true,true,.25)-.1)>1e-15)return 4;
 // A near-zero or wrongly signed secant must never amplify the transient.
 if(bp_v10_bounded_alpha_increment(.2,1.,.1,.2,0.,true,true,.25)!=.2)return 5;
 if(bp_v10_bounded_alpha_increment(.2,1.,.1,.1,0.,true,true,.25)!=.2)return 6;
 for(int i=-10000;i<=10000;++i){
  const double x=bp_v10_bounded_alpha_increment(double(i),.001,.2,.1,.1,true,true,.25);
  if(!isfinite(x)||fabs(x)>.25)return 7;
 }
 return 0;
}
"""
        with tempfile.TemporaryDirectory(prefix='biprism_bounded_') as tmp:
            p=Path(tmp);(p/'test.cpp').write_text(code)
            subprocess.run(['g++','-std=c++17','-O2','-fno-fast-math',str(p/'test.cpp'),'-o',str(p/'test')],check=True,capture_output=True)
            subprocess.run([str(p/'test')],check=True,capture_output=True)

if __name__=='__main__':unittest.main()
