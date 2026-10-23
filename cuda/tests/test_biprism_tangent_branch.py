"""Compiled derivative extension; independent actions remain canonical."""
import ctypes
import math
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import unittest


class TangentBranchControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        p = Path(cls.tmp.name)
        kernels = Path(__file__).resolve().parents[2] / 'crates/prism-gpu/src/kernels'
        (p / 'control.cpp').write_text('''
#include "nhs_biprism_tangent_branch.cuh"
extern "C" bool scope(unsigned bank,unsigned phase,unsigned mode,unsigned tangent,unsigned refinement){
 return bp_v10_endpoint_branch_model(bank,phase,mode,tangent,refinement);
}
extern "C" double energy(double a,double b,double v,double S,double D,bool model){
 return bp_v10_endpoint_branch_energy(a,b,v,S,BiprismElectronicDensity<double>{.3,.7,.2,0.},D,model);
}
extern "C" double canonical(double a,double b,double v,double S){
 return bp_ed_edge_energy(a,b,v,S,BiprismElectronicDensity<double>{.3,.7,.2,0.});
}
extern "C" float energy32(float a,float b,float v,float S,float D,bool model){
 return bp_v10_endpoint_branch_energy(a,b,v,S,BiprismElectronicDensity<float>{.3f,.7f,.2f,0.f},D,model);
}
extern "C" float canonical32(float a,float b,float v,float S){
 return bp_ed_edge_energy(a,b,v,S,BiprismElectronicDensity<float>{.3f,.7f,.2f,0.f});
}
''')
        subprocess.run(['g++', '-std=c++17', '-O2', '-shared', '-fPIC',
                        '-I/usr/local/cuda/include', '-I' + str(kernels),
                        str(p / 'control.cpp'), '-o', str(p / 'control.so')],
                       check=True, capture_output=True)
        cls.lib = ctypes.CDLL(str(p / 'control.so'))
        cls.lib.scope.argtypes = [ctypes.c_uint] * 5
        cls.lib.scope.restype = ctypes.c_bool
        for suffix, scalar in (('', ctypes.c_double), ('32', ctypes.c_float)):
            f = getattr(cls.lib, 'energy' + suffix)
            f.argtypes = [scalar] * 5 + [ctypes.c_bool]
            f.restype = scalar
            f = getattr(cls.lib, 'canonical' + suffix)
            f.argtypes = [scalar] * 4
            f.restype = scalar

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_model_is_excluded_from_midpoint_actions_and_physical_trials(self):
        for bank in (0, 1, 3):
            for phase in range(3):
                for mode in range(5):
                    for tangent in range(3):
                        for refinement in range(9):
                            got = self.lib.scope(bank, phase, mode, tangent, refinement)
                            self.assertEqual(got, bank == 3 and phase == 2 and mode == 1
                                             and tangent != 0 and refinement >= 6)

    def test_disabled_model_preserves_canonical_bits_and_ignores_dead_storage(self):
        rng = random.Random(172)
        for suffix, fmt in (('', 'd'), ('32', 'f')):
            f = getattr(self.lib, 'energy' + suffix)
            canonical = getattr(self.lib, 'canonical' + suffix)
            for _ in range(2000):
                args = [rng.uniform(-3, 3) for _ in range(3)] + [rng.uniform(.9, 1.05)]
                self.assertEqual(struct.pack(fmt, f(*args, float('nan'), False)),
                                 struct.pack(fmt, canonical(*args)))

    def test_crossing_probes_differentiate_one_anchor_branch(self):
        floor = float(ctypes.c_float(.001).value)
        D = floor + 6.46e-13
        eps = 1e-8
        sp, sm = math.sqrt(1 - (D + eps)), math.sqrt(1 - (D - eps))
        plus = self.lib.energy(1., 1., 0., sp, D, True)
        minus = self.lib.energy(1., 1., 0., sm, D, True)
        derivative = (plus - minus) / (2 * eps)
        self.assertLess(abs(derivative / (-1 / (D * D)) - 1), 1e-7)
        mixed = (self.lib.canonical(1., 1., 0., sp)
                 - self.lib.canonical(1., 1., 0., sm)) / (2 * eps)
        self.assertGreater(abs(mixed / derivative - 1), .49)
        # The clamped side has zero D derivative, including equality at kink.
        for anchor in (floor, floor - 6.46e-13):
            self.assertEqual(self.lib.energy(1., 1., 0., sp, anchor, True),
                             self.lib.energy(1., 1., 0., sm, anchor, True))

    def test_invalid_extension_cannot_produce_an_admissible_energy(self):
        self.assertTrue(math.isnan(self.lib.energy(1., 1., 0., 1.1, .002, True)))
        self.assertTrue(math.isnan(self.lib.energy(1., 1., 0., .9, float('nan'), True)))


if __name__ == '__main__':
    unittest.main()
