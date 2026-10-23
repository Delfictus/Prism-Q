#!/usr/bin/env python3
"""CPU controls for the actual small numerical solve; no trajectory admission."""
import ctypes
import math
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np


class NuclearSecantControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='biprism_secant_cpu_')
        root = Path(cls.temp.name)
        kernels = Path(__file__).resolve().parents[2]/'crates/prism-gpu/src/kernels'
        (root/'actual.cpp').write_text('''
#include "nhs_biprism_nuclear_iteration.cuh"
extern "C" double root_next(double* state,double alpha,double residual,double dot,int late){
    BiprismCoordinateRootFfi w{state[0],state[1],state[2],state[3],uint32_t(state[4]),0};
    const double next=bp_coordinate_root_next(&w,alpha,residual,dot,late!=0);
    state[0]=w.left_alpha;state[1]=w.left_residual;state[2]=w.right_alpha;state[3]=w.right_residual;state[4]=w.flags;
    return next;
}
extern "C" double backtrack(double base,double trial,double cross){return bp_nuclear_backtrack_factor(base,trial,cross);}
extern "C" double endpoint_target(double current,double raw,double linear,double quadratic,double half,double defect){
    return bp_endpoint_rounding_target(current,raw,linear,quadratic,half,defect);
}
extern "C" int solve(const double* g,const double* b,unsigned n,double* out){
    BiprismNuclearSecantFfi w{};
    for(unsigned i=0;i<4;++i){w.rhs[i]=b[i];for(unsigned j=0;j<4;++j)w.gram[i][j]=g[4*i+j];}
    bool ok=bp_nuclear_secant_solve(&w,n);
    for(unsigned i=0;i<4;++i)out[i]=w.gamma[i];
    return ok;
}
''')
        subprocess.run(['g++','-O2','-std=c++17','-fno-fast-math','-ffp-contract=off',
                        '-shared','-fPIC','-I'+str(kernels),str(root/'actual.cpp'),
                        '-o',str(root/'actual.so')],check=True,capture_output=True)
        cls.lib = ctypes.CDLL(str(root/'actual.so'))
        ptr = ctypes.POINTER(ctypes.c_double)
        cls.lib.solve.argtypes=[ptr,ptr,ctypes.c_uint,ptr]
        cls.lib.solve.restype=ctypes.c_int
        cls.lib.root_next.argtypes=[ptr,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_int]
        cls.lib.root_next.restype=ctypes.c_double
        cls.lib.endpoint_target.argtypes=[ctypes.c_double]*6
        cls.lib.endpoint_target.restype=ctypes.c_double
        cls.lib.backtrack.argtypes=[ctypes.c_double]*3
        cls.lib.backtrack.restype=ctypes.c_double

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def solve(self, gram, rhs):
        n=len(rhs);g=np.zeros((4,4));b=np.zeros(4);g[:n,:n]=gram;b[:n]=rhs
        out=(ctypes.c_double*4)()
        ok=self.lib.solve((ctypes.c_double*16)(*g.ravel()),(ctypes.c_double*4)(*b),n,out)
        return bool(ok),np.array(out)[:n]

    def test_rejected_uphill_map_reverses_without_acceptance_credit(self):
        # R(x)=1+8*x: forward Picard motion is uphill. Use the evaluated
        # residual difference, then verify the original equation at the trial.
        base=1.;trial=9.;cross=base*(trial-base)
        weight=self.lib.backtrack(base*base,trial*trial,cross)
        self.assertGreaterEqual(weight,-.5);self.assertLess(weight,0.)
        self.assertLess(abs(1.+8.*weight),1e-14)
        self.assertEqual(self.lib.backtrack(1.,math.nan,2.),.5)
        self.assertEqual(self.lib.backtrack(1.,1.,0.),.5)

    def test_endpoint_target_closes_both_signed_quadratics(self):
        for linear in (10000.,-10000.):
            for defect in (1.3e-6,-1.3e-6):
                current=.2
                out=self.lib.endpoint_target(current,current,linear,800.,2.5,defect)
                delta=out-current
                self.assertLess(abs(defect+linear*delta+800.*delta*delta),2e-13)
                self.assertLessEqual(2.5*abs(delta),1e-7)

    def test_anisotropic_represented_candidate_uses_actual_velocity_energy(self):
        g=np.array([[1.3,.2,-.1],[.2,1.1,.04],[-.1,.04,1.5]])
        mass=1822.8885;half=2.5838358402252197
        raw=np.array([.075,-.035,.018])+np.array([1e-9,-2e-10,3e-10])
        current=raw.astype(np.float32).astype(float)
        defect=1.2025742947873686e-6
        kinetic=lambda v:.5*mass*float(v@g@v)
        initial=kinetic(current);best=(abs(defect),current)
        for component in range(3):
            target=self.lib.endpoint_target(current[component],raw[component],
                (mass*g@current)[component],.5*mass*g[component,component],half,defect)
            nearest=np.float32(target)
            for value in (nearest,np.nextafter(nearest,np.float32(-math.inf)),np.nextafter(nearest,np.float32(math.inf))):
                proposal=current.copy();proposal[component]=float(value)
                if half*np.max(np.abs(proposal-raw))>1e-7:continue
                score=abs(defect+kinetic(proposal)-initial)
                if score<best[0]:best=(score,proposal)
        self.assertLess(best[0],1e-7)
        represented_momentum=mass*g@best[1]
        reconstructed_error=np.linalg.solve(g,represented_momentum/mass)-raw
        self.assertLessEqual(half*np.max(np.abs(reconstructed_error)),1e-7)
        # A kinetic correction is only realized when these actual bits export.
        self.assertFalse(np.array_equal(best[1],current))
        np.testing.assert_array_equal(best[1],best[1].astype(np.float32).astype(float))

    def test_endpoint_target_cannot_jump_outside_reconstruction_ball(self):
        out=self.lib.endpoint_target(.2,.2,100.,5.,2.,5888.)
        self.assertLessEqual(2.*abs(out-.2),.990001e-7)
        # No feasible root in the representation ball: retain the failure.
        delta=out-.2
        self.assertGreater(abs(5888.+100.*delta+5.*delta*delta),5887.)
        self.assertTrue(math.isnan(self.lib.endpoint_target(.2,.2,100.,-5.,2.,1.)))
        self.assertTrue(math.isnan(self.lib.endpoint_target(.2,.2,100.,5.,0.,1.)))
        self.assertTrue(math.isnan(self.lib.endpoint_target(.2,.2,math.nan,5.,2.,1.)))

    def test_local_work_bracket_contracts_without_internal_work_credit(self):
        state=(ctypes.c_double*5)();alpha=0.
        # Increasingly sharp, decreasing coordinate defect; arbitrary external
        # kinetic/Berry terms are not arguments of this scalar predictor.
        for _ in range(64):
            residual=1.7-alpha*alpha*alpha
            if abs(residual)<1e-10:break
            next_alpha=self.lib.root_next(state,alpha,residual,.1,1)
            self.assertLessEqual(abs(next_alpha-alpha),.250000000000001)
            alpha=next_alpha
        self.assertLess(abs(1.7-alpha**3),1e-10)

    def test_changed_coupled_state_does_not_trap_the_scalar_in_an_old_bracket(self):
        # These old signs bracket 0, but the current physical functional has
        # moved to 0.01 as its nuclear/electronic state relaxed. Reusing the
        # old endpoints would bisect indefinitely toward an unrelated root.
        state=(ctypes.c_double*5)(-.001,.001,.001,-.001,3.)
        alpha=0.
        for _ in range(40):
            alpha=self.lib.root_next(state,alpha,.01-alpha,1.,1)
        self.assertLess(abs(alpha-.01),1e-12)
        self.assertGreater(alpha,.001)

    def test_near_zero_secant_denominator_cannot_explode_the_work_update(self):
        state=(ctypes.c_double*5)(0.,5888.,0.,0.,1.)
        out=self.lib.root_next(state,.01,5888.-1e-12,63.,1)
        self.assertLessEqual(abs(out-.01),.25)
        self.assertGreater(out,.01)

    def test_unrelaxed_and_nonfinite_trials_cannot_make_large_scalar_jump(self):
        state=(ctypes.c_double*5)()
        self.assertEqual(self.lib.root_next(state,0.,5888.,63.,0),.25)
        self.assertEqual(state[4],0.)
        self.assertEqual(self.lib.root_next(state,0.,1.,0.,1),0.)
        self.assertEqual(self.lib.root_next(state,1.,float('nan'),1.,1),1.)

    def test_actual_regularized_solve_matches_independent_linear_algebra(self):
        d=np.array([[1.,2.,-.5,0.],[3.,0.,1.,-.2],[.5,-1.,2.,4.],[1.,0.,0.,2.],[0.,1.,2.,3.]])
        r=np.array([2.,-3.,1.,.25,.75]);g=d.T@d;b=d.T@r
        ok,actual=self.solve(g,b);self.assertTrue(ok)
        expected=np.linalg.solve(g+np.eye(4)*np.max(np.diag(g))*1e-12,b)
        np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=2e-13)

    def test_mixed_direction_root_converges_without_changing_the_equation(self):
        # Positive Picard steps cannot contract all of these coupled modes.
        # The final assertion uses the original raw equation, not the history fit.
        a=np.array([[-3.,.4,0.],[0.,1.2,.1],[.2,0.,.5]])
        b=np.array([1.,-.5,.3]);expected=np.linalg.solve(np.eye(3)-a,b)
        x=np.zeros(3);previous=None;dx=[];dr=[]
        for _ in range(18):
            r=a@x+b-x
            if np.max(np.abs(r))<1e-10:break
            if previous is not None:
                dx.append(x-previous[0]);dr.append(r-previous[1]);dx=dx[-4:];dr=dr[-4:]
            direction=.5*r
            if dr:
                dd=np.column_stack(dr);xx=np.column_stack(dx)
                ok,gamma=self.solve(dd.T@dd,dd.T@r);self.assertTrue(ok)
                direction-=(xx+.5*dd)@gamma
            previous=(x.copy(),r.copy());x=x+direction
        self.assertLess(np.max(np.abs(a@x+b-x)),1e-10)
        np.testing.assert_allclose(x,expected,rtol=1e-10,atol=1e-10)

    def test_joint_work_velocity_columns_close_both_original_equations(self):
        # Two velocity modes and two leaf-local coordinate-work roots. A
        # coefficient change initially worsens velocity while reducing work.
        # Include both in the same norm; do not credit work as kinetic energy.
        a=np.array([[.5,.1,-.4,0.],[.2,.4,0.,-.3],
                    [1.2,0.,.2,.1],[0.,.9,.1,.3]])
        b=np.array([.1,-.2,.8,-.6]);weights=np.array([1.,1.,.15,.08])
        expected=np.linalg.solve(np.eye(4)-a,b)
        x=np.zeros(4);previous=None;dx=[];dr=[]
        for _ in range(24):
            residual=a@x+b-x
            if np.max(np.abs(residual))<1e-10:break
            if previous is not None:
                dx.append(x-previous[0]);dr.append(residual-previous[1]);dx=dx[-4:];dr=dr[-4:]
            direction=.5*residual
            if dr:
                d=np.column_stack(dr);step=np.column_stack(dx)
                ok,gamma=self.solve(d.T@(weights[:,None]*d),d.T@(weights*residual))
                self.assertTrue(ok);direction-=(step+.5*d)@gamma
            previous=(x.copy(),residual.copy())
            x+=direction/max(1.,np.max(np.abs(direction[2:]))/.25)
        np.testing.assert_allclose(x,expected,rtol=1e-9,atol=1e-10)
        self.assertLess(np.max(np.abs((np.eye(4)-a)@x-b)),1e-10)

    def test_projected_root_does_not_require_a_root_of_the_zero_alpha_subsystem(self):
        # R_v=1-alpha has no root while alpha is frozen at zero. The full
        # projected system R=(1-alpha, v-1) has the unique root (1,1).
        x=np.zeros(2);previous=None;dx=[];dr=[]
        for _ in range(24):
            r=np.array([1.-x[1],x[0]-1.])
            if np.max(np.abs(r))<1e-10:break
            if previous is not None:
                dx.append(x-previous[0]);dr.append(r-previous[1]);dx=dx[-4:];dr=dr[-4:]
            direction=.5*r
            if dr:
                d=np.column_stack(dr);step=np.column_stack(dx)
                ok,gamma=self.solve(d.T@d,d.T@r);self.assertTrue(ok)
                direction-=(step+.5*d)@gamma
            previous=(x.copy(),r.copy())
            x+=direction/max(1.,abs(direction[1])/.25)
        np.testing.assert_allclose(x,[1.,1.],atol=1e-10,rtol=0.)

    def test_collinear_history_is_finite_and_corrupt_history_is_refused(self):
        ok,gamma=self.solve(np.ones((4,4)),np.ones(4));self.assertTrue(ok)
        self.assertTrue(np.isfinite(gamma).all())
        self.assertAlmostEqual(float(gamma.sum()),1.,places=10)
        self.assertFalse(self.solve(np.array([[float('nan')]]),np.ones(1))[0])
        self.assertFalse(self.solve(np.array([[-1.]]),np.ones(1))[0])


if __name__=='__main__':unittest.main()
