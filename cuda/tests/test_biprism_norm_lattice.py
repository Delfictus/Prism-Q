#!/usr/bin/env python3
"""CPU tests of the exact current CUDA initializer against independent slices.

Only scalar helper bodies are host-compiled with explicit FMA and memcpy bit
casts. This is neither a CUDA build nor a native trajectory acceptance test.
"""
import ctypes
from fractions import Fraction as Q
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import random
import re
import struct
import subprocess
import tempfile
import unittest

import biprism_norm_lattice_reference as REF
import check_biprism_norm_refusal as OLD

def _env_root(name):
    """Absolute root from the environment; there is no compiled-in host default.
    When unset, a non-existent sentinel naming the variable is returned so every
    read fails closed; entry points call _require_env() first."""
    value = os.environ.get(name, '')
    return Path(value) if os.path.isabs(value) else Path('/nonexistent') / (name + '_UNSET')


def _require_env(*names):
    for name in names:
        if not os.path.isabs(os.environ.get(name, '')):
            raise SystemExit(f'{name} must name an absolute path (no compiled-in host default)')


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "crates/prism-gpu/src/kernels/nhs_biprism_variational_so3g_v10.cuh"
ELECTRONIC = ROOT / "crates/prism-gpu/src/kernels/nhs_biprism_electronic_discrete.cuh"
RAW_ROOT = _env_root("PRISM_BIPRISM_RESOLUTION_ROOT")
RAW = {
    "008": ("failure_1_2_3_coupled_native_008_state.epoch1.numerical.bin",
            "c71ca92952ac8b19c22e74612a4ff893b1cea828319d995e1a014413debbb7bd"),
    "011": ("failure_1_2_3_coupled_native_011_state.epoch2.numerical.bin",
            "18bef3c2f86ae48adbcdabb7ef58b591b8faeab5ba29c397c58baced85428626"),
}


def extract(source, name):
    match = re.search(r"(?m)^__device__ [^\n]*\b" + name + r"\s*\(", source)
    if match is None:
        raise ValueError("actual helper missing: " + name)
    brace = source.index("{", match.end())
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end]


def prior_and_pivot(trial, n, outer):
    # First prefer a norm centered at one within the unchanged triple bounds.
    # If all16 corners fail, the historical nearest shell/pivot order remains.
    for shell in (0, 1):
        if shell == 0:
            candidates = OLD.candidates(trial, 0)
        else:
            b = [OLD.bracket(x) for x in trial]
            a = [(OLD.next32(lo, False), lo, hi, OLD.next32(hi, True)) for lo, hi in b]
            candidates = (tuple(a[i][(mask >> (2*i)) & 3] for i in range(4))
                          for mask in range(256))
        best, error = None, math.inf
        for candidate in candidates:
            q = OLD.norm(candidate)
            if all(abs(q-t) <= OLD.TOL for t in (n, outer, 1.0)):
                e = abs(q-1.) if shell == 0 else OLD.distance(candidate, trial)
                if e < error:
                    best, error = candidate, e
        if best is not None:
            return best
    return REF.complete_norm_lattice(trial, n, outer)[0]


class NormLatticeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = MAIN.read_text()
        cls.source_sha = hashlib.sha256(MAIN.read_bytes()).hexdigest()
        names = ["bp_v10_promote", "bp_v10_spinor_norm", "bp_v10_bracket",
                 "bp_v10_norm_admitted", "bp_v10_shell1_bracket", "bp_v10_round_norm_shell1",
                 "bp_v10_spinor_component", "bp_v10_with_component", "bp_v10_slice_component",
                 "bp_v10_norm_below", "bp_v10_norm_above", "bp_v10_round_norm_interval",
                 "bp_v10_round_norm"]
        fragments = "\n".join(extract(cls.text, n) for n in names)
        cls.helper_sha = hashlib.sha256(fragments.encode()).hexdigest()
        cls.fragments = fragments
        for qualifier in ("__device__", "__forceinline__", "__noinline__"):
            fragments = fragments.replace(qualifier, "")
        prelude = r'''
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
using std::isfinite;using std::fabs;using std::nextafterf;
#define CUDART_INF (std::numeric_limits<double>::infinity())
#define CUDART_INF_F (std::numeric_limits<float>::infinity())
template<class R> struct Complex {R re,im;};
template<class R> struct Spinor {Complex<R> s0,s1;};
using BiprismPsi32=Spinor<float>;using BiprismPsi64=Spinor<double>;
static double __fma_rn(double a,double b,double c){return std::fma(a,b,c);}
static uint32_t __float_as_uint(float f){uint32_t u;std::memcpy(&u,&f,4);return u;}
static float __uint_as_float(uint32_t u){float f;std::memcpy(&f,&u,4);return f;}
template<class R> R bp_ed_norm2(Complex<R> a){return std::fma(a.re,a.re,a.im*a.im);}
'''
        # The original a416 helper defines this exact norm tree.
        cls.norm_expression = "return bp_ed_fma(a.re,a.re,a.im*a.im);"
        if cls.norm_expression not in ELECTRONIC.read_text():
            raise ValueError("original norm helper tree changed")
        wrappers = r'''
extern "C" int invoke(const double* t,double n,double o,float* out,int fast){
    BiprismPsi64 p={{t[0],t[1]},{t[2],t[3]}};
    BiprismPsi32 r={{13.f,17.f},{19.f,23.f}};
    bool ok=fast?bp_v10_round_norm(p,n,o,r):bp_v10_round_norm_interval(p,n,o,r);
    out[0]=r.s0.re;out[1]=r.s0.im;out[2]=r.s1.re;out[3]=r.s1.im;return ok;
}
extern "C" int norm_ok(const double* t,double n,double o){
    return bp_v10_norm_admitted({{t[0],t[1]},{t[2],t[3]}},n,o);
}
'''
        guard = re.search(r"\(!core\|\|fabs\(bp_v10_outer_norm\((?:tile|s->invocation\.tile),tid\)-1\.\)<=1e-7\)", cls.text)
        if guard is None:
            raise ValueError("actual entry absolute norm gate missing")
        guard = re.sub(r"bp_v10_outer_norm\((?:tile|s->invocation\.tile),tid\)", "value", guard.group())
        wrappers += '\nextern "C" int entry_ok(double value,bool core){return '+guard+';}\n'
        cls.directory = tempfile.TemporaryDirectory(prefix="biprism_norm_lattice_cpu_")
        p = Path(cls.directory.name)
        (p/"actual.cpp").write_text(prelude+fragments+wrappers)
        subprocess.run(["g++", "-std=c++17", "-O2", "-fno-fast-math", "-ffp-contract=off",
                        "-shared", "-fPIC", str(p/"actual.cpp"), "-o", str(p/"actual.so")],
                       check=True, capture_output=True, text=True)
        cls.lib = ctypes.CDLL(str(p/"actual.so"))
        cls.lib.invoke.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_double,
                                   ctypes.c_double, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        cls.lib.invoke.restype = ctypes.c_int
        cls.lib.norm_ok.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_double, ctypes.c_double]
        cls.lib.norm_ok.restype = ctypes.c_int
        cls.lib.entry_ok.argtypes = [ctypes.c_double, ctypes.c_bool]
        cls.lib.entry_ok.restype = ctypes.c_int
        print("ACTUAL_MAIN_SHA256="+cls.source_sha, flush=True)
        print("EXTRACTED_HELPERS_SHA256="+cls.helper_sha, flush=True)
        print("SCOPE=CPU exact-source scalar search versus independent reference; no nonlinear/native acceptance", flush=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()
        if hashlib.sha256(MAIN.read_bytes()).hexdigest() != cls.source_sha:
            raise ValueError("actual source changed during controls")

    def actual(self, trial, n, outer, fast=False):
        t = (ctypes.c_double*4)(*trial)
        out = (ctypes.c_float*4)()
        ok = self.lib.invoke(t, n, outer, out, fast)
        if not ok:
            self.assertEqual(tuple(out), (13., 17., 19., 23.))
            return None
        return tuple(out)

    def assert_same(self, a, b):
        self.assertEqual(a is None, b is None)
        if a is not None:
            self.assertEqual(tuple(map(OLD.bits, a)), tuple(map(OLD.bits, b)))

    def actual_rows(self, label):
        name, digest = RAW[label]
        raw = (RAW_ROOT/name).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), digest)
        return OLD.parse(raw)

    def test_actual_011_all_eight_failed_rows(self):
        failures = [r for r in self.actual_rows("011") if not r[-1]]
        self.assertEqual([(r[0], r[1]) for r in failures], [(l, 133) for l in range(8)])
        for leaf, vertex, trial, start, n, outer, flag in failures:
            reference, error, slices = REF.complete_norm_lattice(trial, n, outer)
            self.assertIsNotNone(reference)
            self.assert_same(self.actual(trial, n, outer), reference)
            self.assert_same(self.actual(trial, n, outer, True), reference)
            self.assertEqual(tuple(map(OLD.bits, reference)), (0xB9620884, 0xBF7FCAC4, 0xBD140457, 0x3C920FFA))
            exact = OLD.exact_norm(reference)
            self.assertTrue(all(abs(exact-Q(t)) <= Q(1,10_000_000) for t in (n, outer, 1.)))
            self.assertLess(sum(abs(a-b) for a,b in zip(reference, trial)), 1e-7)
            self.assertTrue(all(r.iterations <= 31 for _,_,r in slices))
            print(f"NATIVE011 leaf={leaf} vertex={vertex} norm={OLD.norm(reference):.17g} "
                  f"max_delta={max(abs(a-b) for a,b in zip(reference,trial)):.17g} "
                  f"l1_delta={sum(abs(a-b) for a,b in zip(reference,trial)):.17g}", flush=True)

    def test_actual_008_centered_fast_path_matches_independent_order(self):
        rows = [r for r in self.actual_rows("008") if not r[-1]]
        self.assertEqual(len(rows), 1)
        _,_,trial,_,n,outer,_ = rows[0]
        self.assert_same(self.actual(trial,n,outer), REF.complete_norm_lattice(trial,n,outer)[0])
        self.assert_same(self.actual(trial,n,outer,True), prior_and_pivot(trial,n,outer))

    def test_all_recorded_011_initializers(self):
        for _,_,trial,_,n,outer,_ in self.actual_rows("011"):
            self.assert_same(self.actual(trial,n,outer,True), prior_and_pivot(trial,n,outer))

    def test_random_slices_and_source_boundary_certificates(self):
        rng = random.Random(8912)
        for _ in range(80):
            p = [rng.uniform(-1,1) for _ in range(4)]
            scale = math.sqrt(sum(x*x for x in p))
            trial = tuple(x/scale for x in p)  # Generates a test input only.
            n, outer = 1+rng.uniform(-9e-8,9e-8), 1+rng.uniform(-9e-8,9e-8)
            reference, _, slices = REF.complete_norm_lattice(trial,n,outer)
            self.assert_same(self.actual(trial,n,outer), reference)
            for pivot, mask, result in slices:
                self.assertLessEqual(result.iterations,31)
                if result.candidate is None:
                    continue
                q = OLD.norm(result.candidate)
                self.assertTrue(all(abs(q-t)<=OLD.TOL for t in (n,outer,1.)))
                adjacent = result.magnitude_bits + (-1 if result.direction=="up" else 1)
                if result.direction=="nearest" or not 0<=adjacent<=REF.MAX_MAGNITUDE_BITS:
                    continue
                p = list(result.candidate)
                p[pivot] = math.copysign(REF.magnitude_from_bits(adjacent),trial[pivot])
                below,above = REF.norm_class(OLD.norm(p),(n,outer,1.))
                self.assertTrue(below if result.direction=="up" else above)

    def test_monotonicity_across_float_exponents_and_subnormals(self):
        indices = {0,1,2,0x7FFFFE,0x7FFFFF,0x800000,0x800001,REF.MAX_MAGNITUDE_BITS}
        for exponent in range(1,255):
            center=exponent<<23
            indices.update((center-1,center,center+1))
        for pivot in range(4):
            for sign in (-1.,1.):
                previous=-math.inf
                for index in sorted(indices):
                    p=[.3,-.2,.4,-.1];p[pivot]=sign*REF.magnitude_from_bits(index)
                    q=OLD.norm(p)
                    self.assertTrue(math.isfinite(q));self.assertGreaterEqual(q,previous)
                    previous=q

    def test_signed_zero_and_subnormal_pivot(self):
        for tiny in (-0., -REF.magnitude_from_bits(1), REF.magnitude_from_bits(1)):
            trial=(tiny,1.,0.,0.)
            result=REF.closest_in_slice(trial,trial,0,(1.,1.,1.))
            self.assertEqual(OLD.bits(result.candidate[0]),OLD.bits(tiny))
            self.assert_same(self.actual(trial,1.,1.),REF.complete_norm_lattice(trial,1.,1.)[0])

    def test_empty_intersection_and_unrepairable_slice(self):
        self.assertIsNone(REF.complete_norm_lattice((.5,.5,.5,.5),1.,1.000001)[0])
        self.assertIsNone(self.actual((.5,.5,.5,.5),1.,1.000001))
        result=REF.closest_in_slice((1.,2.,0.,0.),(1.,2.,0.,0.),0,(1.,1.,1.))
        self.assertIsNone(result.candidate)

    def test_finite_maximum_magnitude(self):
        maximum=REF.magnitude_from_bits(REF.MAX_MAGNITUDE_BITS)
        trial=(maximum,0.,0.,0.)
        result=REF.closest_in_slice(trial,trial,0,(1.,1.,1.))
        self.assertIsNotNone(result.candidate)
        self.assertLessEqual(result.iterations,31)
        self.assertTrue(math.isfinite(OLD.norm(result.candidate)))
        self.assert_same(self.actual(trial,1.,1.),REF.complete_norm_lattice(trial,1.,1.)[0])

    def test_nonfinite_targets_and_components_refuse(self):
        for bad in (math.nan,math.inf,-math.inf):
            self.assertIsNone(self.actual((bad,0.,0.,0.),1.,1.))
            self.assertIsNone(self.actual((1.,0.,0.,0.),bad,1.))
            self.assertIsNone(self.actual((1.,0.,0.,0.),1.,bad))

    def test_absolute_norm_and_actual_entry_guard(self):
        for p in ((0.,0.,0.,0.),(2.,0.,0.,0.),(OLD.f32(1.000001),0.,0.,0.)):
            n=OLD.norm(p)
            self.assertFalse(self.lib.norm_ok((ctypes.c_double*4)(*p),n,n))
            self.assertFalse(self.lib.entry_ok(n,True))
        for n in (0.,4.,1.+2e-7,1.-2e-7,math.inf,math.nan):
            self.assertFalse(self.lib.entry_ok(n,True))
        self.assertTrue(self.lib.entry_ok(1.,True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
