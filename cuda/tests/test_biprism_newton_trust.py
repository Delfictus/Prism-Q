"""Compiled proposal algebra against an independent NumPy KKT oracle.

These tests cover only the private <=16-dimensional correction solver; native
C1-C4 reduction and physical acceptance are separate mandatory checks.
"""
import ctypes
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np

class NewtonTrustControls(unittest.TestCase):
    capacity=16
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();p=Path(cls.tmp.name)
        root=Path(__file__).resolve().parents[2]/'crates/prism-gpu/src/kernels'
        (p/'control.cpp').write_text('#include "nhs_biprism_newton_trust.cuh"\nextern "C" int solve(unsigned n,double r,const double* g,const double* c,double* y,double* l){double f['+str(cls.capacity**2)+'];return bp_newton_trust<'+str(cls.capacity)+'>(n,r,g,c,f,y,*l);}\n')
        subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC','-I'+str(root),str(p/'control.cpp'),'-o',str(p/'control.so')],check=True)
        cls.lib=ctypes.CDLL(str(p/'control.so'));d=ctypes.POINTER(ctypes.c_double)
        cls.lib.solve.argtypes=[ctypes.c_uint,ctypes.c_double,d,d,d,d]
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def solve(self,T,c,radius):
        n=len(c);g=np.zeros((self.capacity,self.capacity));g[:n,:n]=T.T@T;c=np.ascontiguousarray(c,dtype=np.float64);y=np.zeros(self.capacity);lam=ctypes.c_double()
        ptr=lambda a:a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self.assertEqual(self.lib.solve(n,radius**2,ptr(g),ptr(c),ptr(y),ctypes.byref(lam)),1)
        y=y[:n];l=lam.value;G=g[:n,:n]
        expected=np.linalg.solve(np.eye(n)+l*G,-c)
        np.testing.assert_allclose(y,expected,atol=5e-11,rtol=5e-11)
        self.assertLessEqual(np.linalg.norm(T@y),radius*(1+2e-10))
        self.assertLess(2*c@y+y@y,0.)
        if l>0:self.assertAlmostEqual(np.linalg.norm(T@y)/radius,1.,places=8)
        return y,l
    def test_unconstrained_newton_is_unchanged(self):
        T=np.diag([.01,1.,100.]);c=np.array([1.,-2.,.1]);y,l=self.solve(T,c,100.)
        np.testing.assert_array_equal(y,-c);self.assertEqual(l,0.)
    def test_rank_deficient_basis_remains_spd(self):
        T=np.array([[1.,2.,3.],[2.,4.,6.]]);self.solve(T,np.array([1.,-2.,.2]),.03)
    def test_regularization_retains_strong_response_lost_by_ray_scaling(self):
        # A stiff direction needs little physical motion; shortening every
        # component by the soft direction's bound discards that useful response.
        T=np.diag([.003,3.,.5]);c=np.array([1.7,.4,.8]);radius=.01
        y,l=self.solve(T,c,radius);ray=-c*radius/np.linalg.norm(T@c)
        self.assertLess(np.linalg.norm(c+y),.6*np.linalg.norm(c+ray))
        self.assertGreater(l,0.)
    def test_all_supported_dimensions_and_conditioned_metrics(self):
        rng=np.random.default_rng(927)
        for n in range(1,17):
            U,_=np.linalg.qr(rng.normal(size=(20,n)));V,_=np.linalg.qr(rng.normal(size=(n,n)))
            T=U@np.diag(np.geomspace(.003,3.,n))@V.T
            self.solve(T,rng.normal(size=n),.03)
class ExpandedNewtonTrustControls(NewtonTrustControls):
    capacity=64
    def test_expanded_dimensions_preserve_kkt_and_trust_bound(self):
        rng=np.random.default_rng(119)
        for n in (17,24,32,48,64):
            U,_=np.linalg.qr(rng.normal(size=(n+8,n)));V,_=np.linalg.qr(rng.normal(size=(n,n)))
            T=U@np.diag(np.geomspace(.003,3.,n))@V.T
            self.solve(T,rng.normal(size=n),.03)

if __name__=='__main__':unittest.main()
