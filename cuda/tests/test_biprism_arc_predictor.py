"""Compile actual private-continuation algebra; no physical admission authority."""
from pathlib import Path
import subprocess
import tempfile
import unittest

K = Path(__file__).resolve().parents[2]/'crates/prism-gpu/src/kernels/nhs_biprism_variational_so3g_v10.cuh'


class ArcPredictor(unittest.TestCase):
    def test_bounded_tangent_check_crossing_canonical_provider_branch(self):
        source=K.read_text()
        start=source.index('__device__ __forceinline__ bool bp_v10_action_probe_pending')
        end=source.index('__device__ __forceinline__ double bp_v10_arnoldi_component',start)
        code='''#include <cstdint>
#include <cassert>
#include <cmath>
#include <algorithm>
#define __device__
#define __forceinline__ inline
'''+source[start:end]+'''
int main(){
    // Native156-bound full C1-C4 CPU response: first three action probes
    // cross the endpoint overlap floor, fourth does not. This is reference
    // evidence for selecting a probe, not native trajectory acceptance.
    double errors[]={18.1605054329,17.2043236104,7.7369868918,.0003580976122};
    double best=INFINITY;
    for(uint32_t i=0;i<4u;++i){best=std::min(best,errors[i]*errors[i]);
        assert(bp_v10_action_probe_pending(i+1u,best,true)==(i<3u));}
    assert(best<=.01);
    // An unsuccessful model remains unsuccessful after the bounded check.
    assert(bp_v10_action_probe_pending(4u,1.,true));
    assert(bp_v10_action_probe_pending(5u,1.,true));
    assert(bp_v10_action_probe_pending(6u,1.,true));
    assert(bp_v10_action_probe_pending(7u,1.,true));
    assert(bp_v10_action_probe_pending(8u,1.,true));
    assert(!bp_v10_action_probe_pending(9u,1.,true));
    assert(!bp_v10_action_probe_pending(3u,1.,false));
    assert(!bp_v10_action_probe_pending(3u,.01,true));
    assert(bp_v10_action_probe_pending(2u,0.,true));
    // Native166-bound full-response control: the work row is nonlinear
    // above this interval and pair-rounding dominates below it. All points
    // are actual represented-map actions, not the derivative model itself.
    double window[]={18.2344,17.9218,15.1019,7.5186,2.8065,.0488714479237};
    double radius=1e-4;best=INFINITY;
    for(uint32_t i=0;i<6u;++i){
        best=std::min(best,window[i]*window[i]);
        bool pending=bp_v10_action_probe_pending(i+1u,best,true);
        assert(pending==(i<5u));
        if(pending)radius*=bp_v10_action_probe_scale(i+1u);
    }
    assert(std::abs(radius-3e-9)<1e-23);
    assert(best<=.01);
    assert(std::abs(radius*bp_v10_action_probe_scale(6u)-1e-9)<1e-23);
    // Native169's branch-bound response: positive1e-7 crosses the clamp,
    // reverse1e-7 remains local and validates the SAME signed derivative.
    double signed_radius=1e-4;
    for(uint32_t j=1;j<=7;++j)signed_radius*=bp_v10_action_probe_scale(j);
    assert(std::abs(signed_radius+1e-7)<1e-21);
    assert(!bp_v10_action_probe_pending(8u,.010244512671579324*.010244512671579324,true));
    signed_radius*=bp_v10_action_probe_scale(8u);
    assert(std::abs(signed_radius+1e-8)<1e-22);
    assert(!bp_v10_action_probe_pending(9u,1.,true));
}
'''
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'a.cpp').write_text(code)
            subprocess.run(['g++','-std=c++17','-O2',str(p/'a.cpp'),'-o',str(p/'a')],check=True,capture_output=True)
            subprocess.run([str(p/'a')],check=True,capture_output=True)

    def test_actual_bilinear_bound_and_original_scale_private_gate(self):
        source = K.read_text()
        fragments=[]
        for name,kind in [('bp_v10_arc_predictor_inverse_step','double'),('bp_v10_private_nuclear_converged','bool'),
                          ('bp_v10_private_coordinate_work_converged','bool')]:
            first=source.index('__device__ __forceinline__ '+kind+' '+name)
            end=source.index('{',first);depth=1
            while depth:
                end+=1
                depth+=(source[end]=='{')-(source[end]=='}')
            fragments.append(source[first:end+1])
        code = '''#include <cmath>
#include <cassert>
#include <algorithm>
#include <random>
#define __device__
#define __forceinline__ inline
using std::isfinite;
''' + '\n'.join(fragments) + '''
int main() {
    // Native157's private anchor passed represented work after rounding,
    // but its own continuous corrector row was still4.61e-3Ha.
    assert(!bp_v10_private_coordinate_work_converged(true,.004610688066804869));
    assert(bp_v10_private_coordinate_work_converged(true,1e-4));
    assert(bp_v10_private_coordinate_work_converged(true,-1e-4));
    assert(!bp_v10_private_coordinate_work_converged(true,NAN));
    assert(!bp_v10_private_coordinate_work_converged(true,INFINITY));
    assert(bp_v10_private_coordinate_work_converged(false,NAN));
    // Native142 fixed-domain: tiny h passed the old dimensional gate despite
    // a nonzero original-h-scale residual .0006352962184909576.
    assert(!bp_v10_private_nuclear_converged(8.512354436814181e-20,
                 6.924179882822237e-16,5.1676716804504395));
    assert(bp_v10_private_nuclear_converged(1e-7,.5,1.));
    assert(!bp_v10_private_nuclear_converged(1e-6,.5,1.));
    assert(!bp_v10_private_nuclear_converged(0.,0.,1.));
    assert(!bp_v10_private_nuclear_converged(NAN,1.,1.));
    assert(!bp_v10_private_nuclear_converged(0.,1.,INFINITY));
    // Linear-only, quadratic-only and mixed cases, including cancellation
    // between the two vectors that invalidates a linear-only step bound.
    assert(bp_v10_arc_predictor_inverse_step(0.,0.)==0.);
    assert(std::abs(1./bp_v10_arc_predictor_inverse_step(2.,0.)-.0125)<1e-16);
    assert(std::abs(1./bp_v10_arc_predictor_inverse_step(0.,4.)-std::sqrt(.025/4.))<1e-16);
    std::mt19937_64 rng(143);std::uniform_real_distribution<double> u(-2.,2.);
    for(unsigned i=0;i<10000;++i){
        double l[3]={u(rng),u(rng),u(rng)},q[3]={u(rng),u(rng),u(rng)};
        double ln=0.,qn=0.;for(unsigned j=0;j<3;++j){ln+=l[j]*l[j];qn+=q[j]*q[j];}
        double ds=1./bp_v10_arc_predictor_inverse_step(std::sqrt(ln),std::sqrt(qn));
        double x=0.;for(unsigned j=0;j<3;++j){double d=ds*l[j]+ds*ds*q[j];x+=d*d;}
        assert(std::sqrt(x)<=.025*(1.+8e-16));
    }
    // Measured native141 maxima: old ds=.88468 allows .05181 displacement.
    // A bound computed from these maxima must limit even aligned components.
    double old=.8846833611701148,L=.011490477831257654/old,Q=.06317376638715534/(old*old);
    double ds=1./bp_v10_arc_predictor_inverse_step(L,Q);
    assert(ds<old && ds*L+ds*ds*Q<=.025*(1.+8e-16));
}
'''
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'a.cpp').write_text(code)
            subprocess.run(['g++','-std=c++17','-O2',str(p/'a.cpp'),'-o',str(p/'a')],check=True,capture_output=True)
            subprocess.run([str(p/'a')],check=True,capture_output=True)


if __name__ == '__main__':
    unittest.main()
