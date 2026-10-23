"""Independent MPFR controls for the actual scalar magnetic field primitive.

This is arithmetic evidence only; no coupled/native acceptance is inferred.
"""
import ctypes as C
import ctypes.util
import math
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import unittest

HEADER=Path(__file__).resolve().parents[2]/'crates/prism-gpu/src/kernels/nhs_biprism_magnetic_field_f64.cuh'
class Mpfr(C.Structure):
    _fields_=[('precision',C.c_long),('sign',C.c_int),('exponent',C.c_long),('limbs',C.POINTER(C.c_ulong))]

def ordered(x):
    u=struct.unpack('<Q',struct.pack('<d',x))[0]
    return (~u & ((1<<64)-1)) if u>>63 else u+(1<<63)

class MagneticFieldF64(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(prefix='bp_field_')
        p=Path(cls.tmp.name);src=p/'control.cpp'
        src.write_text('#include "'+str(HEADER)+'"\nextern "C" double evaluate(double x){return biprism_magnetic_f64::phase_plus_sine(x);}\n')
        subprocess.run(['g++','-std=c++17','-O2','-fno-fast-math','-shared','-fPIC',str(src),'-o',str(p/'control.so')],check=True,capture_output=True)
        cls.control=C.CDLL(str(p/'control.so'));cls.control.evaluate.argtypes=[C.c_double];cls.control.evaluate.restype=C.c_double
        library=C.util.find_library('mpfr')
        if not library:raise RuntimeError('MPFR is required; arithmetic proof cannot be skipped')
        cls.mp=C.CDLL(library);P=C.POINTER(Mpfr)
        for name,args,result in [('mpfr_init2',[P,C.c_long],None),('mpfr_clear',[P],None),('mpfr_set_d',[P,C.c_double,C.c_int],C.c_int),('mpfr_sin',[P,P,C.c_int],C.c_int),('mpfr_add',[P,P,P,C.c_int],C.c_int),('mpfr_get_d',[P,C.c_int],C.c_double)]:
            f=getattr(cls.mp,name);f.argtypes=args;f.restype=result
        cls.x=Mpfr();cls.y=Mpfr();cls.mp.mpfr_init2(C.byref(cls.x),256);cls.mp.mpfr_init2(C.byref(cls.y),256)
    @classmethod
    def tearDownClass(cls):
        cls.mp.mpfr_clear(C.byref(cls.x));cls.mp.mpfr_clear(C.byref(cls.y));cls.tmp.cleanup()
    def oracle(self,x):
        self.mp.mpfr_set_d(C.byref(self.x),x,0);self.mp.mpfr_sin(C.byref(self.y),C.byref(self.x),0)
        self.mp.mpfr_add(C.byref(self.y),C.byref(self.x),C.byref(self.y),0)
        return self.mp.mpfr_get_d(C.byref(self.y),0)
    def test_mpfr_exponents_reduction_boundaries_and_low_bits(self):
        rng=random.Random(134)
        cases=[0.,-0.,math.nextafter(0.,1.),-math.nextafter(0.,1.)]
        for exponent in range(-1074,1024):
            for sign in (-1.,1.):
                cases.append(sign*math.ldexp(1.+rng.random(),exponent))
        for _ in range(4096):
            cases.append(rng.uniform(-.2,.2))
            n=rng.randrange(-(1<<53),1<<53)
            center=n*(math.pi/2)
            cases.extend((math.nextafter(center,-math.inf),center,math.nextafter(center,math.inf)))
        for x in (2.**54,-2.**54,math.pi/4,-math.pi/4,math.pi/2,-math.pi/2):
            cases.extend((math.nextafter(x,-math.inf),x,math.nextafter(x,math.inf)))
        maximum=0
        for x in cases:
            y=self.control.evaluate(x);expected=self.oracle(x)
            ulp=abs(ordered(y)-ordered(expected));maximum=max(maximum,ulp)
            self.assertLessEqual(ulp,3,(x.hex(),y.hex(),expected.hex(),ulp))
        print('MPFR_MAGNETIC_FIELD',len(cases),'cases max_ULP',maximum,'native_acceptance=false',flush=True)
    def test_zero_sign_nonfinite_and_large_rounding_identity(self):
        self.assertEqual(math.copysign(1.,self.control.evaluate(-0.)),-1.)
        for x in (math.inf,-math.inf,math.nan):self.assertTrue(math.isnan(self.control.evaluate(x)))
        for x in (2.**54,-2.**54,2.**100,-2.**100,float.fromhex('0x1.fffffffffffffp1023')):
            self.assertEqual(self.control.evaluate(x),x)
    def test_original_fp32_phase_loses_resolved_response(self):
        x=.08593765432476688;y=x+2.**-34
        f32=lambda v:struct.unpack('<f',struct.pack('<f',v))[0]
        self.assertEqual(f32(x),f32(y))
        change=self.control.evaluate(y)-self.control.evaluate(x)
        self.assertGreater(change,1e-10)
        self.assertLess(abs(change-(self.oracle(y)-self.oracle(x))),1e-16)

if __name__=='__main__':unittest.main()
