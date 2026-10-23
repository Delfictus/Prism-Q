"""Real append-only ABI and replicated global-unknown controls."""
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import check_biprism_nuclear_budget as B

ROLE=Path(__file__).resolve().parents[2]
K=ROLE/'crates/prism-gpu/src/kernels/nhs_biprism_variational_so3g_v10.cuh'
R=ROLE/'crates/prism-nhs/src/biprism_v10_runtime.rs'

class ArcLayout(unittest.TestCase):
    def test_measured_completion_tail_cannot_change_prior_owners_or_limits(self):
        raw=bytearray(8*B.COMPLETION_STRIDE)
        h=[B.COMPLETION_MAGIC,0,3072,512,2500,58,0,2,480,32,
           struct.unpack('<I',struct.pack('<f',4.))[0],1,0,
           struct.unpack('<I',struct.pack('<f',2.))[0]]+list(struct.unpack('<II',struct.pack('<d',2.)))+[0]*16
        struct.pack_into('<32I',raw,B.OFFSET,*h)
        struct.pack_into('<d',raw,1680768+4608,123.456)
        for i in range(3072):
            struct.pack_into('<9d',raw,B.evaluation_offset(i),float(i),1.,2.,3.,4.,5.,.1,1700.,.1)
            struct.pack_into('<4d',raw,B.arc_history_offset(i),2.,.1,.05,1.)
        head,rows=B.read_trace(raw,0)
        indices=(191,192,1023,1024,1535,1536,3071)
        self.assertEqual([rows[i][0] for i in indices],list(map(float,indices)))
        self.assertEqual(B.evaluation_offset(1536),1837696)
        self.assertEqual(B.arc_history_offset(1536),1948288)
        self.assertEqual(struct.unpack_from('<d',raw,1680768+4608)[0],123.456)
        self.assertFalse(B.original_target_reached(head))
        for field,value in ((2,3073),(3,513),(0,B.TANGENT_MAGIC)):
            changed=bytearray(raw);struct.pack_into('<I',changed,B.OFFSET+4*field,value)
            with self.assertRaises(ValueError):B.read_trace(changed,0)

    def test_tangent_action_uses_split_error_fields_not_nuclear_residual(self):
        from check_biprism_nuclear_newton import format_action_check
        for phase in (5,8):
            text=format_action_check(1533,1,phase,4,3.,.04,4.,.001,.03)
            fields=dict(token.split('=',1) for token in text.split())
            self.assertEqual(int(fields['phase']),phase)
            self.assertEqual(int(fields['continuation_stage']),1)
            self.assertAlmostEqual(float(fields['independent_direction_relative_error']),.1)
            self.assertAlmostEqual(float(fields['velocity_error2']),.01)
            self.assertEqual(float(fields['velocity_Jp2']),3.)
            self.assertEqual(float(fields['scalar_error2']),.03)
            self.assertEqual(float(fields['scalar_Jp2']),1.)
            self.assertNotIn('raw_dx',fields)
            self.assertNotIn('actual_norm2',fields)

    def test_actual_corrector_progress_rejects_native150_weak_sequence(self):
        source=K.read_text()
        start=source.index('__device__ __forceinline__ uint32_t bp_v10_arc_slow_trials(')
        end=source.index('__device__ __forceinline__ double bp_v10_arc_predictor_inverse_step(',start)
        code='''#include <cassert>
#include <cmath>
#include <cstdint>
#define __device__
#define __forceinline__ inline
'''+source[start:end]+'''
int main(){
    uint32_t n=0;
    // Native150's late accepted ratios approach one; acceptance of a small
    // merit decrease must not reset the segment's stagnation detector.
    for(double r:{.9501,.989,.997,.9989799798777154})n=bp_v10_arc_slow_trials(n,r);
    assert(n==4u);
    assert(bp_v10_arc_slow_trials(n,.999)==4u);
    assert(bp_v10_arc_slow_trials(n,.6)==0u);
    n=bp_v10_arc_slow_trials(0u,.999);
    n=bp_v10_arc_slow_trials(n,.8);
    assert(n==0u);
    assert(bp_v10_arc_step_factor(4u)>1.);
    assert(bp_v10_arc_step_factor(7u)==1.);
    assert(bp_v10_arc_step_factor(9u)<1.);
    const double anchor=.97952606,tangent=.79538316,step=.06621056346350626;
    const double intercept=bp_v10_arc_target_intercept(anchor,tangent,step);
    assert(intercept>0.&&intercept<step);
    assert(std::abs(anchor+intercept*tangent-1.)<1e-15);
    assert(bp_v10_arc_target_intercept(.25,.1,.5)==.5);
    assert(bp_v10_arc_target_intercept(.9,-.1,.5)==.5);
    assert(bp_v10_arc_target_intercept(.9,0.,.5)==.5);
}
'''
        code='#include <initializer_list>\n'+code
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'progress.cpp').write_text(code)
            subprocess.run(['g++','-std=c++17',str(p/'progress.cpp'),'-o',str(p/'progress')],check=True,capture_output=True)
            subprocess.run([str(p/'progress')],check=True,capture_output=True)

    def test_actual_timestep_publication_retains_sub_float_correction(self):
        source=K.read_text()
        start=source.index('__device__ __forceinline__ double bp_v10_half(')
        end=source.index('__device__ __forceinline__ void bp_v10_record_evaluated_h',start)
        code='''#include <cassert>
#define __device__
#define __forceinline__ inline
struct VariationalClusterDsmemV10 {
    struct {struct {volatile double h;} collective;volatile float h,half;} invocation;
};
'''+source[start:end]+'''
int main(){
    VariationalClusterDsmemV10 s{};
    const double original=double(5.025624752044678f);
    bp_v10_publish_h(&s,original);
    assert(s.invocation.collective.h==original);
    assert(bp_v10_half(&s)==.5*original);
    const double correction=original-1e-8;
    bp_v10_publish_h(&s,correction);
    assert(double(s.invocation.h)==original);
    assert(s.invocation.collective.h==correction);
    assert(bp_v10_half(&s)==.5*correction);
    assert(bp_v10_half(&s)!=double(s.invocation.half));
}
'''
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'h.cpp').write_text(code)
            subprocess.run(['g++','-std=c++17',str(p/'h.cpp'),'-o',str(p/'h')],check=True,capture_output=True)
            subprocess.run([str(p/'h')],check=True,capture_output=True)

    def test_precise_private_h_cannot_round_into_original_acceptance(self):
        raw=bytearray(8*B.FORWARD_ARC_STRIDE)
        h=[B.PRECISE_H_MAGIC,1,1,0,0,0,0,1,0,0,
           struct.unpack('<I',struct.pack('<f',4.))[0],16,0,
           struct.unpack('<I',struct.pack('<f',4.))[0]]+[0]*18
        private=4.-1e-8
        h[14:16]=struct.unpack('<II',struct.pack('<d',private))
        struct.pack_into('<32I',raw,B.OFFSET,*h)
        struct.pack_into('<9d',raw,B.evaluation_offset(0),1.,2.,3.,4.,5.,6.,0.,16000.,0.)
        struct.pack_into('<4d',raw,B.arc_history_offset(0),private,0.,.1,0.)
        header,_=B.read_trace(raw,0)
        self.assertEqual(header[13],header[10])
        self.assertEqual(B.live_h(header),private)
        self.assertFalse(B.original_target_reached(header))
        # A different evaluated h must be rejected even when FP32 bits agree.
        struct.pack_into('<d',raw,B.OFFSET+56,4.-2e-8)
        with self.assertRaisesRegex(ValueError,'candidate timestep mismatch'):B.read_trace(raw,0)
        struct.pack_into('<d',raw,B.OFFSET+56,4.)
        struct.pack_into('<d',raw,B.arc_history_offset(0),4.)
        self.assertTrue(B.original_target_reached(B.read_trace(raw,0)[0]))
        # The old schema must never authorize the new reserved double.
        struct.pack_into('<I',raw,B.OFFSET,B.FORWARD_ARC_MAGIC)
        with self.assertRaisesRegex(ValueError,'arclength identity'):B.read_trace(raw,0)

    def test_forward_work_tail_preserves_all_prior_trace_offsets(self):
        raw=bytearray(8*B.FORWARD_ARC_STRIDE)
        h=[B.FORWARD_ARC_MAGIC,1,1536,256,1270,8,0,2,250,6,
           struct.unpack('<I',struct.pack('<f',4.))[0],1,0,
           struct.unpack('<I',struct.pack('<f',2.))[0]]+[0]*18
        struct.pack_into('<32I',raw,B.OFFSET,*h)
        struct.pack_into('<d',raw,1680768+4608,123.456)
        for i in range(1536):
            struct.pack_into('<9d',raw,B.evaluation_offset(i),float(i),1.,2.,3.,4.,5.,.1,1100.,.1)
            struct.pack_into('<4d',raw,B.arc_history_offset(i),2.,.1,.05,1.)
        head,rows=B.read_trace(raw,0)
        indices=(191,192,1023,1024,1535)
        self.assertEqual([rows[i][0] for i in indices],list(map(float,indices)))
        self.assertEqual(B.evaluation_offset(1024),1784448)
        self.assertEqual(B.arc_history_offset(1024),1821312)
        self.assertEqual(struct.unpack_from('<d',raw,1680768+4608)[0],123.456)
        self.assertFalse(B.original_target_reached(head))
        for field,value in ((2,1537),(3,257),(0,B.EXTENDED_ARC_MAGIC)):
            changed=bytearray(raw);struct.pack_into('<I',changed,B.OFFSET+4*field,value)
            with self.assertRaises(ValueError):B.read_trace(changed,0)

    def test_appended_history_boundary_and_bounded_candidate_identity(self):
        raw=bytearray(8*B.EXTENDED_ARC_STRIDE)
        h=[B.EXTENDED_ARC_MAGIC,1,1024,256,758,8,0,2,250,6,
           struct.unpack('<I',struct.pack('<f',4.))[0],1,0,
           struct.unpack('<I',struct.pack('<f',2.))[0]]+[0]*18
        struct.pack_into('<32I',raw,B.OFFSET,*h)
        # The old192-row prefix and both new suffixes remain separate from
        # every physical/preconditioner owner, including the arclength tangent.
        struct.pack_into('<d',raw,1680768+4608,123.456)
        for i in range(1024):
            struct.pack_into('<9d',raw,B.evaluation_offset(i),float(i),1.,2.,3.,4.,5.,.1,1100.,.1)
            struct.pack_into('<4d',raw,B.arc_history_offset(i),2.,.1,.05,1.)
        head,rows=B.read_trace(raw,0)
        self.assertEqual([rows[i][0] for i in (191,192,1023)],[191.,192.,1023.])
        self.assertEqual(struct.unpack_from('<d',raw,1680768+4608)[0],123.456)
        self.assertFalse(B.original_target_reached(head))
        for field,value in ((2,1025),(3,257)):
            changed=bytearray(raw);struct.pack_into('<I',changed,B.OFFSET+4*field,value)
            with self.assertRaises(ValueError):B.read_trace(changed,0)
        # An old192-entry source identity cannot authorize extended history.
        struct.pack_into('<I',raw,B.OFFSET,B.ARC_MAGIC)
        with self.assertRaises(ValueError):B.read_trace(raw,0)

    def test_tangent_rhs_does_not_replace_measured_difference_origin(self):
        source=K.read_text();start=source.index('__device__ __forceinline__ double bp_v10_arc_linear_rhs(')
        end=source.index('__device__ __forceinline__ void bp_v10_arc_predict_from_tangent',start)
        helper=source[start:end].replace('__device__','').replace('__forceinline__','inline')
        code='#include <cassert>\n#include <cmath>\n'+helper+"""
int main(){
  // F(a) is deliberately nonzero. J=(2,-3) must be measured against it,
  // never against the synthetic tangent right-hand side.
  const double base[2]={.125,-.75},jac[2]={2.,-3.},delta=.01;
  for(int c=0;c<2;++c){
    assert(bp_v10_arc_linear_rhs(false,base[c],c==1)==base[c]);
    assert(bp_v10_arc_linear_rhs(true,base[c],c==1)==(c?-1.:0.));
    double probe=base[c]+delta*jac[c];
    assert(std::abs((probe-base[c])/delta-jac[c])<1e-13);
    assert(std::abs((probe-bp_v10_arc_linear_rhs(true,base[c],c==1))/delta-jac[c])>1.);
  }
}
"""
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'rhs.cpp').write_text(code)
            subprocess.run(['g++','-std=c++17',str(p/'rhs.cpp'),'-o',str(p/'rhs')],check=True,capture_output=True)
            subprocess.run([str(p/'rhs')],check=True,capture_output=True)

    def test_tangent_candidate_identity_is_versioned(self):
        raw=bytearray(8*B.FORWARD_ARC_STRIDE)
        h=[B.TANGENT_MAGIC,0,1,0,1,0,0,0,0,0,
           struct.unpack('<I',struct.pack('<f',4.))[0],1,0,
           struct.unpack('<I',struct.pack('<f',2.))[0]]+list(struct.unpack('<II',struct.pack('<d',2.)))+[0]*16
        struct.pack_into('<32I',raw,B.OFFSET,*h)
        struct.pack_into('<9d',raw,B.evaluation_offset(0),.1,.2,.3,1.,2.,3.,.1,1702.,.1)
        struct.pack_into('<4d',raw,B.arc_history_offset(0),2.,-.1,.1,1.)
        self.assertEqual(B.read_trace(raw,0)[1][0][7],1702.)
        struct.pack_into('<I',raw,B.OFFSET,B.PRECISE_H_MAGIC)
        with self.assertRaisesRegex(ValueError,'continuation row stage'):B.read_trace(raw,0)

    def test_actual_cuda_and_rust_tail_layout(self):
        c=K.read_text();start=c.index('struct __align__(128) BiprismNuclearArcFfi');end=c.index('__device__ __forceinline__ void bp_v10_reset_tangent_probes',start)
        c='#include <cstdint>\n#include <cstddef>\n#define __align__(n) alignas(n)\n#define TILE_CORE_VERTICES 192\n#define BIPRISM_MAX_NUCLEAR_EVALUATIONS 192\n'+c[start:end]+'''\nstatic_assert(offsetof(BiprismNuclearArcFfi,history)==10896);
static_assert(offsetof(BiprismNuclearArcFfi,slow_trial_streak)==17040);
static_assert(offsetof(BiprismNuclearArcFfi,segment_accepted_trials)==17044);
static_assert(offsetof(BiprismNuclearArcFfi,tangent_mode)==17048);
static_assert(offsetof(BiprismNuclearArcFfi,tangent_probe_refinements)==17052);
int main() {return sizeof(BiprismNuclearArcFfi)!=17152;}
'''
        r=R.read_text();start=r.rindex('#[repr(C,align(128))]',0,r.index('pub struct BiprismNuclearArcFfi'));end=r.index('#[repr(C,align(128))]',start+1)
        r=r[start:end]+'''\nfn main(){assert_eq!(core::mem::offset_of!(BiprismNuclearArcFfi,history),10896);
assert_eq!(core::mem::offset_of!(BiprismNuclearArcFfi,slow_trial_streak),17040);
assert_eq!(core::mem::offset_of!(BiprismNuclearArcFfi,segment_accepted_trials),17044);
assert_eq!(core::mem::offset_of!(BiprismNuclearArcFfi,tangent_mode),17048);
assert_eq!(core::mem::offset_of!(BiprismNuclearArcFfi,tangent_probe_refinements),17052);}
'''
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'a.cpp').write_text(c);(p/'a.rs').write_text(r)
            for command in (['g++','-std=c++17',str(p/'a.cpp'),'-o',str(p/'cpp')],['rustc','--edition=2021',str(p/'a.rs'),'-o',str(p/'rs')],[str(p/'cpp')],[str(p/'rs')]):
                subprocess.run(command,check=True,capture_output=True)

    def test_evaluated_h_survives_private_branch_turn(self):
        raw=bytearray(8*B.ARC_STRIDE)
        header=[B.ARC_MAGIC,1,2,0,1,0,0,1,0,0,
                struct.unpack('<I',struct.pack('<f',5.1676716804504395))[0],1,0,
                struct.unpack('<I',struct.pack('<f',1.125))[0]]+[0]*18
        struct.pack_into('<32I',raw,B.OFFSET,*header)
        # Stage2 was evaluated before the arclength driver; a fold can lower h.
        for i,(stage,h) in enumerate([(2,1.25),(1,1.125)]):
            struct.pack_into('<9d',raw,B.OFFSET+128+72*i,1.,2.,3.,4.,5.,6.,.1,float(stage*1000+100),0.)
            struct.pack_into('<4d',raw,B.ARC_HISTORY+32*i,h,0.,.1,float(stage==1))
        out,_=B.read_trace(raw,0)
        self.assertEqual(B.live_h(out),1.125)
        self.assertFalse(B.original_target_reached(out))
        # Next-proposal metadata cannot masquerade as the evaluated C4 h.
        struct.pack_into('<I',raw,B.OFFSET+52,struct.unpack('<I',struct.pack('<f',1.5))[0])
        with self.assertRaisesRegex(ValueError,'candidate timestep mismatch'):B.read_trace(raw,0)

    def test_private_h_and_history_are_not_original_step_acceptance(self):
        header=[0]*32;header[0]=B.ARC_MAGIC;header[10]=struct.unpack('<I',struct.pack('<f',4.))[0]
        header[11]=16;header[13]=struct.unpack('<I',struct.pack('<f',3.999))[0]
        self.assertFalse(B.original_target_reached(header))
        header[13]=header[10];self.assertTrue(B.original_target_reached(header))
        header[11]=1;self.assertFalse(B.original_target_reached(header))

if __name__=='__main__':unittest.main()
