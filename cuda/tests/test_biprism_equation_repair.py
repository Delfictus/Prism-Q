#!/usr/bin/env python3
"""Exact current scalar proposal on retained native033 inputs; CPU only.

The source helper ranks frozen-H witnesses. This test neither rebuilds the
nonlinear operator nor admits a trajectory. Native all-eight rebuilding and
the unchanged full equation/work/energy gates remain required.
"""
import ctypes
from dataclasses import replace
import hashlib
import math
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

import check_biprism_c1_refusal as C
import test_biprism_c1_native033_shell as FIXTURE
from test_biprism_norm_lattice import extract


def independent_best(record):
    choices = []
    for value in record.trial:
        lo, hi = FIXTURE.brackets(value)
        choices.append((FIXTURE.next32(lo, False), lo, hi, FIXTURE.next32(hi, True)))
    best = None
    for mask in range(256):
        p = tuple(choices[i][(mask >> (2*i)) & 3] for i in range(4))
        if tuple(map(FIXTURE.bits, p)) == tuple(map(FIXTURE.bits, record.candidate)):
            continue
        norm = FIXTURE.native_norm(p)
        if not all(abs(norm-target) <= 1e-7 for target in (record.norm_start, record.norm_outer, 1.)):
            continue
        equation = FIXTURE.native_equation(record, p)
        if equation > FIXTURE.EQUATION_TOL or not equation < record.native_equation:
            continue
        key = (equation, C.NORM.distance(p, record.trial), mask)
        if best is None or key < best[0]:
            best = (key, p)
    return best


class EquationRepairControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = C.MAIN.read_text()
        cls.source_sha = C.sha(C.MAIN)
        blobs = {key: (FIXTURE.ROOT/name).read_bytes() for key, (name, _) in FIXTURE.FILES.items()}
        cls.record = FIXTURE.decode_bound_record(blobs)
        names = ('bp_v10_promote', 'bp_v10_spinor_norm', 'bp_v10_electronic_equation',
                 'bp_v10_bracket', 'bp_v10_norm_admitted', 'bp_v10_shell1_bracket',
                 'bp_v10_propose_equation_shell', 'bp_v10_matching_s0_labels')
        actual = '\n'.join(extract(cls.source, name) for name in names)
        cls.helper_sha = hashlib.sha256(actual.encode()).hexdigest()
        electronic = C.ELECTRONIC.read_text()
        if C.sha(C.ELECTRONIC) != C.ELECTRONIC_SHA:
            raise ValueError('original electronic helper changed')
        electronic = electronic.replace('#include <cuda_runtime.h>', '')
        prelude = r'''
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
using std::isfinite;
#define __host__
#define __device__
#define __forceinline__ inline
#define __noinline__
#define CUDART_INF (std::numeric_limits<double>::infinity())
#define CUDART_INF_F (std::numeric_limits<float>::infinity())
#define VARIATIONAL_PICARD_TOL 1e-7f
static double __fma_rn(double a,double b,double c){return std::fma(a,b,c);}
static uint32_t __float_as_uint(float x){uint32_t bits;std::memcpy(&bits,&x,4);return bits;}
'''
        aliases = r'''
using BiprismPsi64=BiprismElectronicSpinor<double>;
using BiprismPsi32=BiprismElectronicSpinor<float>;
using BiprismH64=BiprismElectronicHermitian<double>;
'''
        wrappers = r'''
extern "C" int propose(const double* n,const double* t,const float* p,const double* H,
    double omega,double h,double outer,double prior,float* output,double* equation){
    BiprismPsi64 start={{n[0],n[1]},{n[2],n[3]}},trial={{t[0],t[1]},{t[2],t[3]}};
    BiprismPsi32 current={{p[0],p[1]},{p[2],p[3]}},result={{13.f,17.f},{19.f,23.f}};
    BiprismH64 matrix={H[0],H[1],H[2],H[3]};
    const bool ok=bp_v10_propose_equation_shell(start,trial,current,matrix,omega,h,outer,prior,result);
    output[0]=result.s0.re;output[1]=result.s0.im;output[2]=result.s1.re;output[3]=result.s1.im;
    *equation=ok?bp_v10_electronic_equation(start,bp_v10_promote(result),matrix,omega,h):-1.;
    return ok;
}
extern "C" uint32_t labels(const double* t,const float* p){
    return bp_v10_matching_s0_labels({{t[0],t[1]},{t[2],t[3]}},{{p[0],p[1]},{p[2],p[3]}});
}
'''
        cls.directory = tempfile.TemporaryDirectory(prefix='biprism_equation_repair_cpu_')
        path = Path(cls.directory.name)
        (path/'actual.cpp').write_text(prelude+electronic+aliases+actual+wrappers)
        result = subprocess.run(['g++', '-std=c++17', '-O2', '-fno-fast-math', '-ffp-contract=off',
                                 '-shared', '-fPIC', str(path/'actual.cpp'), '-o', str(path/'actual.so')],
                                capture_output=True, text=True)
        if result.returncode:
            raise ValueError(result.stdout+result.stderr)
        cls.lib = ctypes.CDLL(str(path/'actual.so'))
        d, f = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_float)
        cls.lib.propose.argtypes = [d,d,f,d,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,f,d]
        cls.lib.propose.restype = ctypes.c_int
        cls.lib.labels.argtypes = [d,f]; cls.lib.labels.restype = ctypes.c_uint32
        print('ACTUAL_MAIN_SHA256='+cls.source_sha, flush=True)
        print('ACTUAL_PROPOSAL_HELPERS_SHA256='+cls.helper_sha, flush=True)
        print('SCOPE=CPU exact-source proposal; no nonlinear or native trajectory admission', flush=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()
        if C.sha(C.MAIN) != cls.source_sha:
            raise ValueError('main source changed during scalar controls')

    def actual(self, record):
        d4, f4 = ctypes.c_double*4, ctypes.c_float*4
        output, equation = f4(), ctypes.c_double()
        ok = self.lib.propose(d4(*record.start),d4(*record.trial),f4(*record.candidate),d4(*record.H),
                              record.omega,record.h,record.norm_outer,record.native_equation,output,ctypes.byref(equation))
        if not ok:
            self.assertEqual(tuple(output),(13.,17.,19.,23.))
            self.assertEqual(equation.value,-1.)
            return None
        return tuple(output), equation.value

    def test_actual033_selects_the_largest_margin_shell_witness(self):
        best = independent_best(self.record)
        actual = self.actual(self.record)
        self.assertEqual(best[0][2],228)
        self.assertEqual(tuple(map(FIXTURE.bits,actual[0])),FIXTURE.EXPECTED[228])
        self.assertEqual(tuple(map(FIXTURE.bits,actual[0])),tuple(map(FIXTURE.bits,best[1])))
        self.assertEqual(actual[1],best[0][0])
        self.assertGreater(FIXTURE.EQUATION_TOL-actual[1],5e-9)
        self.assertNotEqual(FIXTURE.bits(actual[0][3]),FIXTURE.bits(self.record.candidate[3]))
        for margin in FIXTURE.margins(self.record,actual[0]): self.assertGreater(margin,0)

    def test_proposal_does_not_establish_rebuilt_h_admission(self):
        selected,_ = self.actual(self.record)
        changed = replace(self.record,H=(self.record.H[0],self.record.H[1]+1e-5,*self.record.H[2:]))
        self.assertGreater(FIXTURE.native_equation(changed,selected),FIXTURE.EQUATION_TOL)
        q0 = FIXTURE.native_norm((self.record.candidate[2],self.record.candidate[3],0.,0.))
        q1 = FIXTURE.native_norm((selected[2],selected[3],0.,0.))
        self.assertNotEqual(q0,q1)

    def test_no_feasible_or_invalid_proposal_preserves_output(self):
        cases = [replace(self.record,norm_outer=2.),replace(self.record,omega=0.),
                 replace(self.record,h=0.),replace(self.record,native_equation=FIXTURE.EQUATION_TOL),
                 replace(self.record,native_equation=math.nan),replace(self.record,H=(math.inf,*self.record.H[1:])),
                 replace(self.record,trial=(math.nan,*self.record.trial[1:])),
                 replace(self.record,candidate=(math.nan,*self.record.candidate[1:]))]
        for record in cases:
            with self.subTest(record=record): self.assertIsNone(self.actual(record))

    def test_labels_reset_only_to_matching_s0_bit_patterns(self):
        d4,f4=ctypes.c_double*4,ctypes.c_float*4
        trial=(1.,-0.,0.,0.)
        self.assertEqual(self.lib.labels(d4(*trial),f4(1.,-0.,.5,.25)),15)
        self.assertEqual(self.lib.labels(d4(*trial),f4(1.,0.,.5,.25)),0)
        self.assertEqual(self.lib.labels(d4(*trial),f4(FIXTURE.next32(1.,True),-0.,0.,0.)),0)
        selected,_=self.actual(self.record)
        expected=0
        for mask in range(4):
            values=(FIXTURE.brackets(self.record.trial[0])[mask&1],FIXTURE.brackets(self.record.trial[1])[(mask>>1)&1])
            if tuple(map(FIXTURE.bits,values))==tuple(map(FIXTURE.bits,selected[:2])): expected|=1<<mask
        self.assertEqual(self.lib.labels(d4(*self.record.trial),f4(*selected)),expected)

    def test_explicit_finite_update_accounting(self):
        # Exhaust the initial labels, then maximize each selected-row reset.
        # Each actual repair contributes one round and at most four new labels.
        moves,repairs=8*192*4,0
        self.assertEqual(moves,6144)
        for _ in range(32):
            repairs+=1; moves+=4
            rounds=moves+repairs
            self.assertEqual(moves,6144+4*repairs)
            self.assertEqual(rounds,6144+5*repairs)
        self.assertEqual(rounds,6304)
        self.assertIn('BP_V10_EQUATION_MAX_REPAIRS=4u*8u',self.source)
        self.assertIn('#define MAX_VARIATIONAL_PICARD_ITERS 48',self.source)
        self.assertIn('#define MAX_CONTINUOUS_ELECTRONIC_ITERS 24',self.source)
        self.assertIn('BP_V10_REPRESENTATION_MAX_UPDATES+5u*BP_V10_EQUATION_MAX_REPAIRS',self.source)


if __name__ == '__main__':
    unittest.main(verbosity=2)
