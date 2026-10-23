"""Actual late-base action-revalidation predicate, independent of raw units."""
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

K=Path(__file__).resolve().parents[2]/'crates/prism-gpu/src/kernels/nhs_biprism_variational_so3g_v10.cuh'

class DirectionRevalidation(unittest.TestCase):
    def test_weak_accepted_step_rechecks_the_changed_jacobian(self):
        source=K.read_text()
        predicate=re.search(r'if\((q->accepted_trial_ratio[^\n]+)\)\s*q->direction_checked=0u;',source).group(1)
        code='''#include <cassert>
struct Q { double accepted_trial_ratio; };
struct Invocation { double nuclear_raw_residual; };
struct S { Invocation invocation; };
bool recheck(double ratio,double raw){Q value{ratio};S state{{raw}};auto*q=&value;auto*s=&state;
return '''+predicate+''';}
int main(){
    // Retained native143 accepted correction and native142 late correction.
    assert(recheck(1.2530504586759534e-9/1.350601945731024e-9,3.859844955015542e-5));
    assert(recheck(2.4396806238278803e-11/2.5033646079649258e-11,1.9084809367977347e-6));
    assert(recheck(.99,0.)); // A work-row defect can remain at zero nuclear dx.
    assert(!recheck(.5,1.));assert(!recheck(.9,1.));
}
'''
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'a.cpp').write_text(code)
            subprocess.run(['g++','-std=c++17','-O2',str(p/'a.cpp'),'-o',str(p/'a')],check=True,capture_output=True)
            subprocess.run([str(p/'a')],check=True,capture_output=True)

if __name__=='__main__':unittest.main()
