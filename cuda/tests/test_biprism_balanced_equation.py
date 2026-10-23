"""Exact CUDA proposal body against a retained native EQR2 refusal.

This proves frozen-H feasibility only. It cannot admit nonlinear GPU state.
"""
import ctypes
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest

import check_biprism_c1_refusal as C
from test_biprism_norm_lattice import extract, MAIN, ELECTRONIC

# Source: failure_1_2_3_coupled_native_058_fixed_domain_16_state.epoch4.numerical.bin
SOURCE_SHA256 = "3b50fabc47d05f68584a7ae7826627344bf0a3be3348e7de0d3aa625d7cb9427"
RECORD = bytes.fromhex("325251450200000004000000610000000300000001000000f80200000000000000000000b40200000c00000000000000d6f53bbf86ad2dbfe875bc3cdbe7373cddfb7f3f176bf5bb923c06bca049e13ae00026bc7bffef3f80847fe462ad7ebf68c48c726ac780bf08d283fd33295c3f8bb191b065ba20c056550e6b5bd0efbfbbcc9e7d1d91c13f00000000000000004463f12566028dbf00000080c919214000000020b2ab14400000000000000000a20396e1ffffef3f00000000000000000000000000000000836e92533d6f7f3e0000000000000000836e92533d6f7f3e570d44e15ac88e3f0024026bb367d13f00000000000000000000000000000000")

# Native062 full-covariance capture; SHA256 12f0269429a6ec7dccad28a5223e35536a7fc2a15e2aa5ac915a588a7264a7f1
RECORD_062 = bytes.fromhex("3252514502000000040000006100000003000000010000001b0300000000000000000000b40200000c00000000000000d6f53bbf86ad2dbfe675bc3cd6e7373cddfb7f3f456bf5bb6c3c06bc7b49e13ab1c100bc7bffef3f0047258068ad7ebf31c0552f6ac780bf20186c5a2f295c3f592116c765ba20c027471b405bd0efbf7dff5f7c1d91c13f00000000000000004afd39ff65028dbf000000a0c919214000000020b2ab1440000000000000000042af8ce1ffffef3f000000000000000000000000000000004713e26ebee07b3e00000000000000004713e26ebee07b3e4fc901ec5ac88e3ff030f06ab367d13f00000000000000000000000000000000")


class BalancedEquationProposal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='biprism_equation_balanced_')
        root = Path(cls.tmp.name)
        source = MAIN.read_text()
        names = ['bp_v10_promote','bp_v10_spinor_norm','bp_v10_bracket',
                 'bp_v10_norm_admitted','bp_v10_with_component','bp_v10_electronic_equation',
                 'bp_v10_equation_gram','bp_v10_component64','bp_v10_gram_solve3',
                 'bp_v10_proposal_vec','bp_v10_reload_h',
                 'bp_v10_propose_equation_balanced']
        fragments = '\n'.join(extract(source,n) for n in names)
        begin=source.index('struct BiprismEquationProposalScratch {')
        workspace=source[begin:source.index('};',begin)+2]
        electronic = ELECTRONIC.read_text().replace('#include <cuda_runtime.h>','')
        code = r'''
#include <cmath>
#include <cstdint>
#include <limits>
#define __host__
#define __device__
#define __forceinline__ inline
#define CUDART_INF std::numeric_limits<double>::infinity()
#define CUDART_INF_F std::numeric_limits<float>::infinity()
#define CUDART_NAN std::numeric_limits<double>::quiet_NaN()
#define VARIATIONAL_PICARD_TOL 1e-7f
inline double __fma_rn(double a,double b,double c){return std::fma(a,b,c);}
''' + electronic + r'''
using BiprismPsi64=BiprismElectronicSpinor<double>;
using BiprismPsi32=BiprismElectronicSpinor<float>;
using BiprismH64=BiprismElectronicHermitian<double>;
using BiprismVec64=BiprismElectronicVector<double>;
''' + workspace + r'''
inline BiprismPsi64 bp_v10_reload_spinor(const volatile BiprismPsi64* p) {return {{p->s0.re,p->s0.im},{p->s1.re,p->s1.im}};}
''' + fragments + r'''
extern "C" int propose(const double* a,const double* q,const double* h,
    double mass,double dt,double outer,double eq,float* out){
    BiprismPsi32 p{}; BiprismEquationProposalScratch workspace{};
    const bool ok=bp_v10_propose_equation_balanced({{a[0],a[1]},{a[2],a[3]}},
        {{q[0],q[1]},{q[2],q[3]}},{h[0],h[1],h[2],h[3]},mass,dt,outer,eq,p,workspace);
    out[0]=p.s0.re;out[1]=p.s0.im;out[2]=p.s1.re;out[3]=p.s1.im;return ok;
}
'''
        (root/'actual.cpp').write_text(code)
        subprocess.run(['g++','-std=c++17','-O2','-fno-fast-math','-shared','-fPIC',str(root/'actual.cpp'),'-o',str(root/'actual.so')],check=True,capture_output=True)
        cls.lib=ctypes.CDLL(str(root/'actual.so'))
        cls.lib.propose.argtypes=[ctypes.POINTER(ctypes.c_double)]*3+[ctypes.c_double]*4+[ctypes.POINTER(ctypes.c_float)]
        cls.lib.propose.restype=ctypes.c_int

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_retained_vertex97_joint_norm_equation_witness(self):
        for record in (RECORD,RECORD_062):
            with self.subTest(capture=hashlib.sha256(record).hexdigest()):
                r=C.parse_record(record,0,'equation')
                r['norm_start']=C.NORM.norm(r['start'])
                counts,prior=C.frozen_candidates(r)
                self.assertEqual(counts['joint'],0)
                self.assertIsNone(prior)
                H=(*r['H'][:3],r['H'][3]+r['connection'])
                out=(ctypes.c_float*4)()
                ok=self.lib.propose((ctypes.c_double*4)(*r['start']),
                    (ctypes.c_double*4)(*r['trial']),(ctypes.c_double*4)(*H),
                    r['omega'],r['h'],r['norm_outer'],r['equation_candidate'],out)
                self.assertEqual(ok,1)
                self.assertTrue(C.NORM.admitted(tuple(out),r['norm_start'],r['norm_outer'],absolute_one=True))
                actual=C.equation(r['start'],tuple(out),H,r['omega'],r['h'])
                self.assertLessEqual(actual,C.NATIVE_TOL)
                self.assertLess(actual,r['equation_candidate'])

    def test_impossible_norm_intersection_remains_refused(self):
        out=(ctypes.c_float*4)()
        self.assertEqual(self.lib.propose((ctypes.c_double*4)(1,0,0,0),
            (ctypes.c_double*4)(1,0,0,0),(ctypes.c_double*4)(1,1,0,0),
            1.,1.,1.001,1e-5,out),0)


if __name__=='__main__': unittest.main()
