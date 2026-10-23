"""Host compilation of the production primitive versus an exact rational solve.

This validates the frozen-metric map and kick/work identity, not a native
trajectory. Native resource and coupled C3/C4 controls remain separate gates.
"""
import ctypes
from fractions import Fraction as Q
import math
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import unittest

ROLE = Path(__file__).resolve().parents[2]
HEADER = ROLE / 'crates/prism-gpu/src/kernels/nhs_biprism_nuclear_cayley_f64.cuh'


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def mul(a, v):
    return [sum(x*y for x, y in zip(row, v)) for row in a]


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def solve(a, b):
    rows = [list(map(Q, row)) + [Q(rhs)] for row, rhs in zip(a, b)]
    for j in range(3):
        pivot = next(k for k in range(j, 3) if rows[k][j])
        rows[j], rows[pivot] = rows[pivot], rows[j]
        d = rows[j][j]
        rows[j] = [x/d for x in rows[j]]
        for k in range(3):
            if k != j:
                d = rows[k][j]
                rows[k] = [x-d*y for x, y in zip(rows[k], rows[j])]
    return [row[3] for row in rows]


def matrix(g):
    xx, yy, zz, xy, xz, yz = map(Q, g)
    return [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]


class NuclearCayleyF64(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='biprism_nuclear_cpu_')
        p = Path(cls.tmp.name)
        source = p/'control.cpp'
        source.write_text('#include "'+str(HEADER)+'"\n'+r'''
struct G { float xx,yy,zz,xy,xz,yz; };
struct GD { double xx,yy,zz,xy,xz,yz; };
struct V { double x,y,z; };
extern "C" void rotate(const float* g,const double* v,const double* t,double* out) {
    const auto r=biprism_nuclear_f64::cayley(
        G{g[0],g[1],g[2],g[3],g[4],g[5]},V{v[0],v[1],v[2]},V{t[0],t[1],t[2]});
    out[0]=r.x;out[1]=r.y;out[2]=r.z;
}
extern "C" void rotate64(const double* g,const double* v,const double* t,double* out) {
    const auto r=biprism_nuclear_f64::cayley(GD{g[0],g[1],g[2],g[3],g[4],g[5]},V{v[0],v[1],v[2]},V{t[0],t[1],t[2]});
    out[0]=r.x;out[1]=r.y;out[2]=r.z;
}
''')
        subprocess.run(['g++', '-std=c++17', '-O2', '-fno-fast-math', '-shared', '-fPIC', str(source), '-o', str(p/'control.so')], check=True, capture_output=True)
        cls.lib = ctypes.CDLL(str(p/'control.so'))
        cls.lib.rotate.argtypes = [ctypes.POINTER(ctypes.c_float)] + [ctypes.POINTER(ctypes.c_double)]*3
        cls.lib.rotate.restype = None
        cls.lib.rotate64.argtypes = [ctypes.POINTER(ctypes.c_double)]*4
        cls.lib.rotate64.restype = None

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def rotate(self, g, v, t):
        out = (ctypes.c_double*3)()
        self.lib.rotate((ctypes.c_float*6)(*g), (ctypes.c_double*3)(*v), (ctypes.c_double*3)(*t), out)
        return list(out)

    def test_full_metric_low_bits_enter_actual_cayley(self):
        g=[2.+2.**-30,3.-2.**-31,4.+2.**-29,.2,-.35,.4]
        v=[.01,-.02,.03];t=[.1,-.3,.2];out=(ctypes.c_double*3)()
        self.lib.rotate64((ctypes.c_double*6)(*g),(ctypes.c_double*3)(*v),(ctypes.c_double*3)(*t),out)
        G=matrix(g);x,y,z=map(Q,t);C=[[Q(0),z,-y],[-z,Q(0),x],[y,-x,Q(0)]]
        expected=solve([[G[j][k]-C[j][k] for k in range(3)] for j in range(3)],mul([[G[j][k]+C[j][k] for k in range(3)] for j in range(3)],list(map(Q,v))))
        self.assertLess(max(abs(out[i]-float(expected[i])) for i in range(3)),2e-16)
        rounded=self.rotate(list(map(f32,g)),v,t)
        self.assertGreater(max(abs(out[i]-rounded[i]) for i in range(3)),1e-12)

    def test_exact_equation_and_metric_energy(self):
        rng = random.Random(44045)
        for i in range(96):
            scale = 10.**((i % 9)-4)
            g = list(map(f32, [2*scale, 3*scale, 4*scale, .2*scale, -.35*scale, .4*scale]))
            v = [rng.uniform(-.1, .1) for _ in range(3)]
            t = [f32(rng.uniform(-1, 1)*scale*10.**((i % 7)-3)) for _ in range(3)]
            G, (x, y, z) = matrix(g), map(Q, t)
            C = [[Q(0), z, -y], [-z, Q(0), x], [y, -x, Q(0)]]
            a = [[G[j][k]-C[j][k] for k in range(3)] for j in range(3)]
            b = mul([[G[j][k]+C[j][k] for k in range(3)] for j in range(3)], list(map(Q, v)))
            exact = solve(a, b)
            actual = self.rotate(g, v, t)
            self.assertLess(max(abs(float(x)-y) for x, y in zip(exact, actual)), 2e-14)
            r = [x-y for x, y in zip(mul(a, list(map(Q, actual))), b)]
            bound = max(map(abs, b)) + max(abs(x) for row in a for x in row)*sum(map(abs, map(Q, actual)))
            self.assertLess(float(max(map(abs, r))/bound), 3e-14)
            e0 = dot(list(map(Q, v)), mul(G, list(map(Q, v))))
            e1 = dot(list(map(Q, actual)), mul(G, list(map(Q, actual))))
            self.assertLess(float(abs(e1-e0)/e0), 3e-13)
            reverse = self.rotate(g, actual, [-x for x in t])
            self.assertLess(max(abs(x-y) for x, y in zip(reverse, v)), 4e-14)

    def test_endpoint_conversion_does_not_remove_metric_transport_work(self):
        # Exact arithmetic counterexample to native159's midpoint-metric reuse.
        p=[Q(2),Q(0),Q(0)]; mass=Q(1)
        gm=matrix([1,1,1,0,0,0]); ge=matrix([2,2,2,0,0,0])
        vm=[x/mass for x in solve(ge,p)]
        represented=list(map(Q,map(f32,vm)))
        te=mass*dot(represented,mul(ge,represented))/2
        raw_endpoint=dot(p,solve(ge,p))/(2*mass)
        raw_midpoint=dot(p,solve(gm,p))/(2*mass)
        self.assertEqual(te-raw_endpoint,0)
        self.assertEqual(te-raw_midpoint,-1)
        # Removing conversion must leave the metric work in the smooth row.
        represented_work=te-raw_midpoint
        self.assertEqual(represented_work-(te-raw_endpoint),-1)
        self.assertEqual(represented_work-(te-raw_midpoint),0)

    def test_kick_rotation_kick_work_without_rounding_intermediates(self):
        g = list(map(f32, [1.75, 2.5, 3., .125, -.25, .375]))
        G = matrix(g)
        mass, h = 21000., f32(5.16767168)
        v0, force = [.037, -.021, .044], [.32, -.28, .11]
        p0 = [float(mass*x) for x in mul(G, list(map(Q, v0)))]
        pm = [x+.5*h*f for x, f in zip(p0, force)]
        vm = [float(x/mass) for x in solve(G, pm)]
        vp = self.rotate(g, vm, [.31, -.27, .47])
        pp = [float(mass*x)+.5*h*f for x, f in zip(mul(G, list(map(Q, vp))), force)]
        dx = [h*.5*(a+b) for a, b in zip(vm, vp)]
        t0 = .5/mass*float(dot(list(map(Q, p0)), solve(G, p0)))
        t1 = .5/mass*float(dot(list(map(Q, pp)), solve(G, pp)))
        self.assertLess(abs((t1-t0)-dot(force, dx)), 2e-12)
        # Negative control: quantizing the interior momentum is not an exact
        # work-preserving map, even though the same final gate is requested.
        rounded = list(map(f32, pp))
        tr = .5/mass*float(dot(list(map(Q, rounded)), solve(G, rounded)))
        self.assertGreater(abs((tr-t0)-dot(force, dx)), 1e-7)

    def test_zero_rotation_and_invalid_metric(self):
        v = [.03, -.07, .01]
        self.assertEqual(self.rotate([1., 2., 3., 0., 0., 0.], v, [0., 0., 0.]), v)
        self.assertTrue(all(math.isnan(x) for x in self.rotate([0., 0., 0., 0., 0., 0.], v, [1., 2., 3.])))


if __name__ == '__main__':
    unittest.main()
