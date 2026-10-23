"""CPU controls for the actual early numerical search scalar, not MD evidence."""
import subprocess
import tempfile
import unittest
from pathlib import Path

K = Path(__file__).resolve().parents[2] / 'crates/prism-gpu/src/kernels'


class NuclearDirection(unittest.TestCase):
    def test_ascent_direction_reversal_and_invalid_inputs(self):
        source = (K / 'nhs_biprism_variational_so3g_v10.cuh').read_text()
        start = source.index('__host__ __device__ __forceinline__ double bp_v10_reverse_trial_weight(')
        end = source.index('// One numerical (velocity,alpha) proposal,', start)
        code = '''
#include <cmath>
#include <limits>
using std::isfinite;
#define __host__
#define __device__
#define __forceinline__ inline
''' + source[start:end] + '''
int main(){
 // R(v)=1+4v. Every positive Picard step is ascent; -1/4 is its root.
 double w=bp_v10_reverse_trial_weight(.125,.5,.25);
 if(w!=-.25||1.+4.*w!=0.)return 1;
 // A contracting direction already descends; no reversal is proposed.
 if(bp_v10_reverse_trial_weight(.125,-.5,.25)!=0.)return 2;
 // A near-zero directional slope may never produce an unbounded jump.
 if(bp_v10_reverse_trial_weight(.125,1e-30,1e-60)!=-.5)return 3;
 for(double bad: {0.,-1.,std::numeric_limits<double>::infinity(),
                  std::numeric_limits<double>::quiet_NaN()})
  if(bp_v10_reverse_trial_weight(.125,1.,bad)!=0.)return 4;
 // The exact retained 057 Epoch5 terminal residual model has positive
 // slope. This only checks its predicted search direction, not acceptance.
 const double base2=2.3103674323949908,cross=2.705064774722838e-6,
     difference2=3.336079346232954e-12,trial=1.0015468307968711e-7;
 w=bp_v10_reverse_trial_weight(trial,cross,difference2);
 double predicted=base2+2.*(w/trial)*cross+(w/trial)*(w/trial)*difference2;
 if(!(w<0.&&fabs(w)<.5&&predicted<.06*base2))return 5;
 // The displacement trust region applies to the magnitude for either sign.
 for(double direction: {-0.5,0.5}){
  double limited=copysign(fmin(fabs(direction),1./40.),direction);
  if(fabs(limited)!=.025||std::signbit(limited)!=std::signbit(direction))return 6;
 }
}
'''
        code = '#include <initializer_list>\n' + code
        with tempfile.TemporaryDirectory(prefix='biprism_direction_') as tmp:
            p = Path(tmp)
            (p / 'a.cpp').write_text(code)
            subprocess.run(['g++', '-std=c++17', '-O2', '-fno-fast-math',
                            str(p / 'a.cpp'), '-o', str(p / 'a')], check=True, capture_output=True)
            subprocess.run([str(p / 'a')], check=True, capture_output=True)


if __name__ == '__main__':
    unittest.main()
