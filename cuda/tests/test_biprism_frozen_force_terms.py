"""Independent derivative checks for the retained-state CPU reconstruction.

These check the diagnostic reference, not native trajectory acceptance.
"""
import unittest
import numpy as np
from check_biprism_frozen_force_terms import provider,edge_energy_force,metric_force

class FrozenForceReferenceControls(unittest.TestCase):
    def test_radial_force_matches_energy_on_both_sides_of_overlap_floor(self):
        lut=np.array([[1.1,-.5,-.2,.4],[1.3,-.6,-.3,.5]])
        edge=np.array([[0,1]]);elem=np.array([0,1]);q=np.array([-.2,.3])
        rho=np.array([[.7,.3,.2,-.1]])
        direction=np.array([1.,-2.,3.]);direction/=np.linalg.norm(direction)
        for radius in (.02,.04,.17,.37,.49,1.4,5.):
            x=np.array([np.zeros(3),radius*direction])
            _,force=edge_energy_force(provider(x,edge,elem,q,lut),rho)
            for axis in range(3):
                step=1e-6;plus=x.copy();minus=x.copy()
                plus[0,axis]+=step;minus[0,axis]-=step
                ep=edge_energy_force(provider(plus,edge,elem,q,lut),rho)[0][0]
                em=edge_energy_force(provider(minus,edge,elem,q,lut),rho)[0][0]
                self.assertAlmostEqual(force[0,axis],-(ep-em)/(2*step),delta=2e-6*max(1.,abs(force[0,axis])))

    def test_metric_force_is_fixed_momentum_kinetic_derivative(self):
        edge=np.array([[0,1]]);x=np.zeros((256,3));x[1]=[.8,.3,-.1]
        ref=x.copy();ref[1]=[1.,.1,.2];mass=np.ones(256)*12.
        v=np.zeros((256,3));v[0]=[.07,.03,-.02];v[1]=[-.03,.02,.05]
        force,g=metric_force(x,ref,v,mass,edge)
        p=mass[:192,None]*np.einsum('nij,nj->ni',g,v[:192])
        def kinetic(y):
            _,gy=metric_force(y,ref,np.zeros_like(v),mass,edge)
            vy=np.linalg.solve(gy,p[...,None])[...,0]/mass[:192,None]
            return .5*np.sum(p*vy)
        for axis in range(3):
            plus=x.copy();minus=x.copy();step=1e-5
            plus[0,axis]+=step;minus[0,axis]-=step
            self.assertAlmostEqual(force[0,axis],-(kinetic(plus)-kinetic(minus))/(2*step),delta=1e-9)

if __name__=='__main__':unittest.main()
