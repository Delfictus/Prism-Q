"""Independent linearized-system controls; not a coupled-trajectory claim."""
import unittest
import numpy as np

def append(Q,T,df,dx):
    q,t=df.copy(),dx.copy()
    for _ in range(2):
        for a,b in zip(Q,T):
            c=float(a@q);q-=c*a;t-=c*b
    scale=float(np.linalg.norm(q))
    if scale<=1e-12*np.linalg.norm(df):return False
    Q.append(q/scale);T.append(t/scale);return True

def direction(f,Q,T):
    p=f.copy()
    for q,t in zip(Q,T):p-=(q@f)*(t+q)
    return p

class MultisecantControls(unittest.TestCase):
    def test_local_correction_must_pass_full_coupled_action_check(self):
        rng=np.random.default_rng(112)
        blocks=[rng.normal(size=(3,3))+np.diag([4.,7.,11.]) for _ in range(8)]
        diagonal=np.zeros((24,24))
        for l,A in enumerate(blocks):diagonal[3*l:3*l+3,3*l:3*l+3]=A
        ring=np.zeros_like(diagonal)
        for l in range(8):ring[3*l:3*l+3,3*((l+1)%8):3*((l+1)%8)+3]=np.eye(3)
        probes=rng.normal(size=(24,3));f=rng.normal(size=24)
        for coupling,expected in [(.0001,True),(2.,False)]:
            J=diagonal+coupling*ring;responses=J@probes;p=np.zeros(24);pred=np.zeros(24)
            for l in range(8):
                ix=slice(3*l,3*l+3);Q=[];T=[]
                for j in range(3):self.assertTrue(append(Q,T,responses[ix,j],probes[ix,j]))
                coeff=np.array(Q)@f[ix];p[ix]=-np.array(T).T@coeff;pred[ix]=-np.array(Q).T@coeff
            relative=np.linalg.norm(J@p-pred)/np.linalg.norm(pred)
            self.assertEqual(relative<=.1,expected)
            if expected:self.assertLess(np.linalg.norm(f+J@p),np.linalg.norm(f)*.1)

    def test_leaf_local_observed_inverse_matches_independent_blocks(self):
        rng=np.random.default_rng(91)
        probes=rng.normal(size=(4,4))
        for leaf in range(8):
            A=rng.normal(size=(4,4))+np.diag([3.+leaf,5.,-2.,.3])
            Q=[];T=[]
            for p in probes.T:self.assertTrue(append(Q,T,A@p,p))
            inverse=np.array(T).T@np.array(Q)
            np.testing.assert_allclose(inverse,np.linalg.inv(A),rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(A@inverse,np.eye(4),atol=1e-12)

    def test_actual_perturbations_are_orthogonal_after_anisotropic_preconditioning(self):
        rng=np.random.default_rng(19)
        V=np.linalg.qr(rng.normal(size=(12,12)))[0]
        H=np.eye(12)+1e6*np.ones((12,12))/12
        actual=H@V
        normalized=actual/np.linalg.norm(actual,axis=0)
        self.assertGreater(np.max(np.abs(normalized.T@normalized-np.eye(12))),.99)
        retained=[]
        for column in actual.T:
            x=column.copy()
            for _ in range(2):
                for d in retained:x-=(d@x)*d
            self.assertGreater(np.linalg.norm(x),1e-10)
            retained.append(x/np.linalg.norm(x))
        Q=np.array(retained).T
        np.testing.assert_allclose(Q.T@Q,np.eye(12),atol=1e-14)
        np.testing.assert_allclose(Q@(Q.T@actual),actual,rtol=2e-13,atol=2e-9)

    def test_recycled_preconditioner_measures_current_operator(self):
        old=np.array([[8.,2.,.1],[1.,-3.,.2],[.3,.1,.4]])
        current=old+np.outer([.2,-.4,.1],[.3,.2,-.1])
        Q=np.linalg.qr(old)[0];T=np.linalg.solve(old,Q);P=-np.eye(3)
        H=T@Q.T+P@(np.eye(3)-Q@Q.T)
        directions=H@np.eye(3)
        # Fresh measurements use the CURRENT operator on preconditioned
        # directions. Substitution of old Q values gives a different answer.
        measured=current@directions
        f=np.array([.4,-.2,.7])
        step=directions@np.linalg.lstsq(measured,-f,rcond=None)[0]
        np.testing.assert_allclose(current@step,-f,atol=4e-15)
        self.assertGreater(np.linalg.norm(current@(-H@f)+f),.01)

    def test_inverse_secant_action_matches_dense_rank_one_reference(self):
        rng=np.random.default_rng(731)
        for n,k in [(12,1),(12,5),(24,15)]:
            Q=np.linalg.qr(rng.normal(size=(n,k)))[0]
            T=rng.normal(size=(n,k))
            P=np.diag(-np.linspace(.4,1.7,n))
            y=rng.normal(size=n);step=rng.normal(size=n)
            H=T@Q.T+P@(np.eye(n)-Q@Q.T)
            reference=H+np.outer(step-H@y,y)/(y@y)
            c=Q.T@y;perpendicular=y-Q@c
            correction=step-P@y-(T-P@Q)@c
            updated=T+np.outer(correction,c)/(y@y)
            norm=np.linalg.norm(perpendicular)
            extra=perpendicular/norm
            Q=np.column_stack((Q,extra))
            T=np.column_stack((updated,P@extra+correction*norm/(y@y)))
            actual=T@Q.T+P@(np.eye(n)-Q@Q.T)
            np.testing.assert_allclose(actual,reference,rtol=2e-13,atol=2e-13)
            np.testing.assert_allclose(actual@y,step,rtol=2e-13,atol=2e-13)
            np.testing.assert_allclose(Q.T@Q,np.eye(k+1),atol=3e-15)

    def test_reused_inverse_does_not_authorize_a_nonlinear_step(self):
        # An exact secant is not the derivative at a distant state. Even an
        # algebraically valid inverse update must face actual globalization.
        f=lambda x:np.array([x[0]**3-1.,x[1]+.1*x[0]])
        base=np.array([.1,0.]);trial=np.array([.2,0.]);y=f(trial)-f(base)
        H=np.eye(2);H+=np.outer(trial-base-H@y,y)/(y@y)
        np.testing.assert_allclose(H@y,trial-base,atol=1e-15)
        candidate=trial-H@f(trial)
        self.assertGreater(np.linalg.norm(f(candidate)),np.linalg.norm(f(trial)))

    def test_physical_gate_row_scaling_and_relative_edge_guard(self):
        half=2.5
        velocity_gate=5e-9/half
        energy_weight=5e-9/(half*2e-7)
        self.assertAlmostEqual(energy_weight*2e-7,velocity_gate)
        # Independent polynomial roots, with a dense line-segment check.
        for change in [np.array([-2.,.1,0.]),np.array([1.,0.,0.]),np.array([0.,2.,0.])]:
            edge=np.array([1.,0.,0.]);r2=float(edge@edge);rd=float(edge@change);dd=float(change@change)
            disc=rd*rd-.75*r2*dd
            bound=1.
            if rd<0 and disc>0:bound=min(1.,.9*.75*r2/(-rd+np.sqrt(disc)))
            for w in np.linspace(0,bound,101):self.assertGreaterEqual(np.linalg.norm(edge+w*change),.5)
            if rd>=0:self.assertEqual(bound,1.)
    def test_fixed_base_direction_agrees_with_independent_nonlinear_jacobian(self):
        def residual(x):
            return np.array([4*x[0]+.8*x[1]+.2*x[0]*x[1]+1.,
                             -.3*x[0]-2*x[1]+.1*x[0]**2-2.])
        x=np.array([.25,-.4]);f=residual(x)
        J=np.array([[4.+.2*x[1],.8+.2*x[0]],[-.3+.2*x[0],-2.]])
        # The oracle builds the complete directional matrix and solves a
        # least-squares system independently of the device's MGS recurrence.
        errors=[]
        for radius in [1e-3,1e-4,1e-5]:
            directions=np.array([[.7,-.2],[.3,.9]]).T
            observed=np.column_stack([(residual(x+radius*d)-f)/radius for d in directions.T])
            step=directions@np.linalg.lstsq(observed,-f,rcond=None)[0]
            errors.append(np.linalg.norm(J@step+f))
            self.assertLess(np.linalg.norm(residual(x+step)),np.linalg.norm(f))
        self.assertLess(errors[2],errors[0]/90.)
        self.assertLess(errors[2],1e-5)

    def test_residual_denominator_is_fixed_even_at_zero_trial_chord(self):
        # Dividing by the current chord creates a pole at bootstrap. A fixed
        # positive physical scale changes conditioning only, never R=0.
        reference_dot=.04
        for current_dot in [0.,1e-14,.04,-.01]:
            def physical_residual(alpha):return 3.-alpha*current_dot
            scaled=physical_residual(2.)/reference_dot
            self.assertTrue(np.isfinite(scaled))
            self.assertEqual(scaled==0.,physical_residual(2.)==0.)

    def test_joint_motion_bound_is_independent_of_alpha_units(self):
        # The coefficient can be rescaled without changing a physical Newton
        # displacement. Bound the actual input and model-output motion instead.
        f=np.array([[.31,-.12,.04],[-.27,.08,.02]])
        p=np.array([[-.05,.07,.01],[.06,-.03,.01]])
        half=2.5
        maximum=half*max(np.linalg.norm(p,axis=1).max(),np.linalg.norm(p-f,axis=1).max())
        weight=min(1.,.05/maximum)
        for alpha_scale in [.001,1.,1000.]:
            scaled_alpha=67.*alpha_scale
            self.assertTrue(np.isfinite(weight*scaled_alpha/alpha_scale))
            self.assertLessEqual(weight*maximum,.05+1e-15)
        # A bounded private trial need not be converged or physically accepted.
        self.assertGreater(np.linalg.norm((1.-weight)*f),.1)
    def test_positive_picard_eigenvalue_is_solved_by_measured_secant(self):
        J=np.array([[4.,.8],[-.3,-2.]])
        x=np.array([.25,-.4]);f=J@x+np.array([1.,-2.]);Q=[];T=[]
        for step in [np.array([.07,.02]),np.array([-.01,.09])]:
            observed=(J@(x+step)+np.array([1.,-2.]))-f
            self.assertTrue(append(Q,T,observed,step))
        np.testing.assert_allclose(np.array(Q)@np.array(Q).T,np.eye(2),atol=4e-15)
        np.testing.assert_allclose(J@np.array(T).T,np.array(Q).T,atol=4e-15)
        p=direction(f,Q,T)
        self.assertLess(np.linalg.norm(J@p+f),2e-14)
        self.assertGreater(np.linalg.norm(J@(x+f)+np.array([1.,-2.])),np.linalg.norm(f))
    def test_alpha_scaling_changes_conditioning_not_root(self):
        J=np.array([[3.,.8],[-.2,-.6]]);S=np.diag([1.,.003]);A=S@J@np.linalg.inv(S)
        x=np.array([.01,.4]);f=J@x;Q=[];T=[]
        for step in [np.array([.1,0.]),np.array([0.,.2])]:append(Q,T,S@(J@step),S@step)
        p=np.linalg.solve(S,direction(S@f,Q,T))
        self.assertLess(np.linalg.norm(J@(x+p)),1e-13)
        w=.125
        self.assertGreater(np.linalg.norm(J@(x+w*p)),1e-6) # trust trial cannot pass as a root
    def test_dependent_observations_do_not_create_singular_division(self):
        Q=[];T=[];d=np.array([1.,2.,3.])
        self.assertTrue(append(Q,T,d,.1*d))
        self.assertFalse(append(Q,T,2*d,.2*d))
        self.assertTrue(np.all(np.isfinite(direction(d,Q,T))))

if __name__=='__main__':unittest.main()
