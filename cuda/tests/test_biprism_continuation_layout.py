"""Compile the real CUDA/Rust storage definitions and check disjoint owners."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROLE = Path(__file__).resolve().parents[2]
CUDA = ROLE / 'crates/prism-gpu/src/kernels/nhs_biprism_variational_so3g_v10.cuh'
RUST = ROLE / 'crates/prism-nhs/src/biprism_v10_runtime.rs'


class ContinuationLayout(unittest.TestCase):
    def test_continuous_coordinate_scope_excludes_actual_trials_and_actions(self):
        text=CUDA.read_text()
        begin=text.index('__host__ __device__ __forceinline__ bool bp_v10_continuous_tangent_probe(')
        end=text.index('__device__',begin+len('__host__ __device__'))
        source='''#include <cstdint>
#include <cassert>
#define __host__
#define __device__
#define __forceinline__ inline
'''+text[begin:end]+'''
int main(){
 for(uint32_t mode=0;mode<5;++mode)for(uint32_t tangent=0;tangent<3;++tangent)
  for(uint32_t refinement=0;refinement<10;++refinement){
   bool got=bp_v10_continuous_tangent_probe(mode,tangent,refinement);
   if(mode!=1u||tangent==0u||refinement<2u)assert(!got);
   else assert(got);
  }
}
'''
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'scope.cpp').write_text(source)
            subprocess.run(['g++','-std=c++17','-O2',str(p/'scope.cpp'),'-o',str(p/'scope')],check=True,capture_output=True)
            subprocess.run([str(p/'scope')],check=True,capture_output=True)

    def test_tangent_probe_state_does_not_inherit_corrector_restarts(self):
        text=CUDA.read_text()
        begin=text.index('struct __align__(128) BiprismNuclearQrFfi')
        end=text.index('struct __align__(128) BiprismV10NumericalScratchFfi',begin)
        source='''#include <cstdint>
#include <cstddef>
#include <memory>
#include <cassert>
#define __align__(n) alignas(n)
#define __device__
#define __forceinline__ inline
#define TILE_CORE_VERTICES 192
#define TILE_MAX_EDGES 1536
'''+text[begin:end]+'''
int main(){
    auto q=std::make_unique<BiprismNuclearQrFfi>();
    auto a=std::make_unique<BiprismNuclearArcFfi>();
    q->continuation_previous_velocity[2][191]=1.25;
    a->history[191][3]=17.;a->tangent_mode=1u;
    for(unsigned old_restarts=0;old_restarts<4;++old_restarts){
        q->linear_restarts=old_restarts;a->tangent_probe_refinements=7u;
        bp_v10_reset_tangent_probes(*q,*a);q->fd_radius=1e-6;
        assert(q->linear_restarts==0u&&a->tangent_probe_refinements==0u);
        // A direction breakdown restarts QR, but does not select paired FD.
        ++q->linear_restarts;
        assert(a->tangent_probe_refinements==0u&&q->fd_radius==1e-6);
        bp_v10_refine_tangent_probes(*q,*a);
        assert(a->tangent_probe_refinements==1u&&q->fd_radius==1e-8);
        ++q->linear_restarts;
        bp_v10_refine_tangent_probes(*q,*a);
        assert(a->tangent_probe_refinements==2u&&q->fd_radius==1e-10);
        bp_v10_refine_tangent_probes(*q,*a);
        assert(a->tangent_probe_refinements==3u&&q->fd_radius==.5e-10);
        bp_v10_refine_tangent_probes(*q,*a);
        assert(a->tangent_probe_refinements==4u&&q->fd_radius==.25e-10);
        bp_v10_refine_tangent_probes(*q,*a);
        assert(a->tangent_probe_refinements==5u&&q->fd_radius==1e-9);
        bp_v10_refine_tangent_probes(*q,*a);
        assert(a->tangent_probe_refinements==6u&&q->fd_radius==1e-7);
        assert(q->continuation_previous_velocity[2][191]==1.25);
        assert(a->history[191][3]==17.&&a->tangent_mode==1u);
    }
}
'''
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'probe.cpp').write_text(source)
            subprocess.run(['g++','-std=c++17','-O2',str(p/'probe.cpp'),'-o',str(p/'probe')],check=True,capture_output=True)
            subprocess.run([str(p/'probe')],check=True,capture_output=True)

    def test_actual_rust_appended_trace_extent(self):
        text=RUST.read_text()
        begin=text.rindex('#[repr(C,align(128))]',0,text.index('pub struct BiprismNuclearQrFfi'))
        end=text.index('/// This relation is stronger',begin)
        constants='\n'.join(line for line in text.splitlines() if line.startswith('pub const BIPRISM_NUMERICAL_SCRATCH_BYTES:'))
        source=constants+'\n'+text[begin:end]+'''
fn main(){
    assert_eq!(core::mem::size_of::<BiprismNumericalScratchFfi>(),1997440);
    assert_eq!(core::mem::offset_of!(BiprismNumericalScratchFfi,nuclear_evaluation_completion),1837696);
    assert_eq!(core::mem::offset_of!(BiprismNumericalScratchFfi,nuclear_arc_history_completion),1948288);
    assert_eq!(core::mem::offset_of!(BiprismNumericalScratchFfi,nuclear_arc),1680768);
}
'''
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'full.rs').write_text(source)
            subprocess.run(['rustc','--edition=2021',str(p/'full.rs'),'-o',str(p/'full')],check=True,capture_output=True)
            subprocess.run([str(p/'full')],check=True,capture_output=True)

    def test_actual_cxx_preconditioner_storage_cannot_overwrite_predictor(self):
        text = CUDA.read_text()
        begin = text.index('struct __align__(128) BiprismNuclearQrFfi')
        end = text.index('struct __align__(128) BiprismNuclearEvaluationFfi', begin)
        source = '''#include <cstdint>
#include <cstddef>
#include <cstring>
#include <memory>
#include <cassert>
#define __align__(n) alignas(n)
#define TILE_CORE_VERTICES 192
#define TILE_MAX_EDGES 1536
''' + text[begin:end] + '''
int main(){
    auto q=std::make_unique<BiprismNuclearQrFfi>();
    static_assert(sizeof(q->pre_dx)/sizeof(q->pre_dx[0])==63);
    static_assert(sizeof(q->dx)/sizeof(q->dx[0])==64);
    static_assert(sizeof(BiprismNuclearQrFfi)==1560064);
    static_assert(offsetof(BiprismNuclearQrFfi,tangent_endpoint_raw_D)==607696);
    static_assert(offsetof(BiprismNuclearQrFfi,tangent_endpoint_raw_D)==offsetof(BiprismNuclearQrFfi,trust_gram));
    for(unsigned k=0;k<3;++k)for(unsigned i=0;i<192;++i)
        q->continuation_previous_velocity[k][i]=0.25+k+i*0x1p-30;
    q->continuation_previous_alpha=19.739707080948182;
    q->continuation_previous_stage=15.;
    std::memset(q->pre_dx,0xa5,sizeof(q->pre_dx));
    std::memset(q->pre_df,0x5a,sizeof(q->pre_df));
    std::memset(q->pre_alpha_dx,0xa5,sizeof(q->pre_alpha_dx));
    std::memset(q->pre_alpha_df,0x5a,sizeof(q->pre_alpha_df));
    std::memset(q->tangent_endpoint_raw_D,0x4d,sizeof(q->tangent_endpoint_raw_D));
    for(unsigned k=0;k<3;++k)for(unsigned i=0;i<192;++i)
        assert(q->continuation_previous_velocity[k][i]==0.25+k+i*0x1p-30);
    assert(q->continuation_previous_alpha==19.739707080948182);
    assert(q->continuation_previous_stage==15.);
}
'''
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / 'control.cpp').write_text(source)
            subprocess.run(['g++', '-std=c++17', '-O2', str(p / 'control.cpp'),
                            '-o', str(p / 'control')], check=True, capture_output=True)
            subprocess.run([str(p / 'control')], check=True, capture_output=True)

    def test_actual_rust_extent_offsets_and_private_roundtrip(self):
        text = RUST.read_text()
        begin = text.rindex('#[repr(C,align(128))]', 0, text.index('pub struct BiprismNuclearQrFfi'))
        end = text.index('#[repr(C,align(128))]', begin + 1)
        source = text[begin:end] + '''
fn main(){unsafe{
    let layout=std::alloc::Layout::new::<BiprismNuclearQrFfi>();
    let p=std::alloc::alloc_zeroed(layout).cast::<BiprismNuclearQrFfi>();
    assert!(!p.is_null());
    (*p).continuation_previous_velocity[2][191]=1.0+2f64.powi(-30);
    (*p).continuation_previous_alpha=19.739707080948182;
    (*p).continuation_previous_stage=15.0;
    (*p).pre_dx.fill([[2.0;192];3]);
    (*p).pre_alpha_dx.fill(3.0);(*p).pre_alpha_df.fill(4.0);
    assert_eq!((*p).continuation_previous_velocity[2][191],1.0+2f64.powi(-30));
    assert_eq!((*p).continuation_previous_alpha,19.739707080948182);
    assert_eq!((*p).continuation_previous_stage,15.0);
    assert_eq!((*p).pre_dx.len(),63);assert_eq!((*p).dx.len(),64);
    std::alloc::dealloc(p.cast(),layout);
}}
'''
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / 'control.rs').write_text(source)
            subprocess.run(['rustc', '--edition=2021', '-O', str(p / 'control.rs'),
                            '-o', str(p / 'control')], check=True, capture_output=True)
            subprocess.run([str(p / 'control')], check=True, capture_output=True)


if __name__ == '__main__':
    unittest.main()
