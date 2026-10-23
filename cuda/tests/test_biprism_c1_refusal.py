#!/usr/bin/env python3
"""Diagnostic ABI/replay negatives and actual-declaration CPU controls."""
import copy
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

import check_biprism_c1_refusal as C


def fixture(kind, version=1, continuous_count=12):
    numerical = bytearray(C.LEAVES*C.STRIDE)
    exchange = bytearray(C.LEAVES*C.BANK)
    tiles = bytearray(C.LEAVES*C.TILE)
    conv = bytearray(C.LEAVES*128)
    v, n, H, omega, h, phi = 7, (1., 0., 0., 0.), (0., 0., 0., 0.), 1., 1., .25
    trial = (1., 1.1e-7 if kind == 'equation' else 1e-8, 0., 0.)
    endpoint = (1., C.NORM.f32(1.2e-7), 0., 0.) if kind == 'equation' else n
    for leaf in range(C.LEAVES):
        b, f, t = leaf*C.STRIDE, leaf*C.BANK+11520, leaf*C.TILE
        struct.pack_into('<I', tiles, t+4*v, v)
        struct.pack_into('<f', tiles, t+95232+4*v, omega)
        struct.pack_into('<4f', tiles, t+90112+16*v, *n)
        struct.pack_into('<f', tiles, t+1024+16*v+12, C.NORM.f32(.1))
        struct.pack_into('<d', numerical, b+C.PHI+8*v, phi)
        struct.pack_into('<d', exchange, leaf*C.BANK+8*v, 1.)
        source_endpoint = endpoint if kind == 'equation' else trial
        pop = C.NORM.norm(source_endpoint)
        struct.pack_into('<d', exchange, leaf*C.BANK+1536+8*v, pop)
        struct.pack_into('<12I', exchange, f+80, 0x46435031, leaf, 0, 0, 12, 2, 0, 5, 0, 0, 0, 0)
        struct.pack_into('<6I', conv, leaf*128, 0x42505631, leaf, 2, 2, 2, 0)
        offset, magic, iteration = (C.TAIL, C.EQR, 0) if kind == 'equation' else (C.PREFIX, C.CFR, 11)
        r = b+offset
        struct.pack_into('<12I', numerical, r, magic, 1, leaf, v, 0, 0, 12, iteration, 0, v, 0, 0)
        struct.pack_into('<4f', numerical, r+48, *n)
        trial_eq = C.equation(n, trial, H, omega, h)
        if kind == 'equation':
            residual = C.equation(n, endpoint, H, omega, h)
            struct.pack_into('<4I', numerical, b+C.PREFIX, C.RRF, 7, 1, 0)
            struct.pack_into('<4f', numerical, r+64, *endpoint)
            struct.pack_into('<4d', numerical, r+80, *trial)
            struct.pack_into('<4d', numerical, r+112, *H)
            values = (0., omega, h, C.NORM.norm(n), 1., C.NORM.norm(endpoint), C.NORM.norm(trial),
                      residual, trial_eq, residual, phi, C.NORM.f32(.1))
            struct.pack_into('<12d', numerical, r+144, *values)
            struct.pack_into('<d', numerical, b+C.SCALAR+13*8, residual)
            struct.pack_into('<d', exchange, f+16, residual)
        else:
            change = max(abs(x-y) for x, y in zip(trial, endpoint))
            struct.pack_into('<4d', numerical, r+64, *trial)
            struct.pack_into('<4d', numerical, r+96, *endpoint)
            struct.pack_into('<4d', numerical, r+128, *H)
            values = (0., omega, h, 1., C.NORM.norm(trial), C.NORM.norm(endpoint), change,
                      trial_eq, 0., change, trial_eq, phi)
            struct.pack_into('<12d', numerical, r+160, *values)
            struct.pack_into('<4I', numerical, b+C.TAIL, C.CHT, 1, 12, 0xfff)
            for k in range(12):
                struct.pack_into('<ddfI', numerical, b+C.TAIL+16+24*k, change, trial_eq, 1e-7, 0)
            struct.pack_into('<d', exchange, f+16, change)
    if version == 2:
        for leaf in range(C.LEAVES):
            base = leaf*C.STRIDE+(C.TAIL if kind == 'equation' else C.PREFIX)
            struct.pack_into('<2I', numerical, base, C.EQR2 if kind == 'equation' else C.CFR2, 2)
            values_offset = 144 if kind == 'equation' else 160
            for index in ((3, 5, 6, 8) if kind == 'equation' else (3, 4, 5, 7, 8)):
                struct.pack_into('<d', numerical, base+values_offset+8*index, 0.)
    if kind == 'continuous' and continuous_count == 24:
        if version != 2:
            raise ValueError('CHT2 requires CFR2')
        for leaf in range(C.LEAVES):
            base = leaf*C.STRIDE
            row = bytes(numerical[base+C.TAIL+16:base+C.TAIL+40])
            numerical[base+C.TAIL:base+C.TAIL+304] = bytes(304)
            struct.pack_into('<I', numerical, base+C.PREFIX+28, 23)
            struct.pack_into('<4I', numerical, base+C.HISTORY2, C.CHT2, 2, 24, 0xffffff)
            for k in range(24):
                numerical[base+C.HISTORY2+16+24*k:base+C.HISTORY2+40+24*k] = row
    return dict(numerical=numerical, exchange=exchange, tiles=tiles, convergence=conv)


def fixture_v3(kind):
    from test_biprism_spinor_owner import fixture as owner_fixture
    d=fixture(kind,version=2,continuous_count=24 if kind=='continuous' else 12)
    owners,_=owner_fixture(committed=0,step=0);d['spinors']=owners
    for l in range(8):
        b=l*C.STRIDE;o=b+(C.TAIL if kind=='equation' else C.PREFIX)
        old=C.parse_record(d['numerical'],o,kind);identity=list(struct.unpack_from('<12I',d['numerical'],o))
        identity[:2]=[C.EQR3 if kind=='equation' else C.CFR3,3]
        size=256 if kind=='equation' else 272;d['numerical'][o:o+size]=bytes(size)
        struct.pack_into('<12I',d['numerical'],o,*identity)
        if kind=='equation':
            for off,key in ((48,'start'),(80,'candidate'),(112,'trial'),(144,'H')):struct.pack_into('<4d',d['numerical'],o+off,*old[key])
            names=('connection','omega','h','norm_outer','equation_candidate','leaf_equation','phi','charge')
            struct.pack_into('<8d',d['numerical'],o+176,*(old[k] for k in names))
        else:
            for off,key in ((48,'start'),(80,'trial'),(112,'next'),(144,'H')):struct.pack_into('<4d',d['numerical'],o+off,*old[key])
            names=('connection','omega','h','change','equation_trial','equation_next','leaf_change','leaf_equation','phi')
            struct.pack_into('<9d',d['numerical'],o+176,*(old[k] for k in names))
        v=old['vertex'];a=l*C.SPINOR.SIZE
        for off in (128,6272):struct.pack_into('<4d',owners,a+off+32*v,*old['start'])
        struct.pack_into('<4d',owners,a+12416+32*v,*(old.get('candidate',old['trial'])))
    return d


class RefusalControls(unittest.TestCase):
    def test_fp64_equation_and_continuous_raw_records(self):
        for kind in ('equation','continuous'):
            d=fixture_v3(kind);records,_=C.inspect_buffers(**d,search=False)
            self.assertTrue(all(r['version']==3 for r in records))
            d['spinors'][6272+32*7:6272+32*7+8]=struct.pack('<d',.5)
            with self.assertRaises(ValueError):C.inspect_buffers(**d,search=False)

    def test_fp64_record_does_not_narrow_components(self):
        d=fixture_v3('equation');off=C.TAIL+80
        p=(1.+2.**-42,2.**-35,2.**-40,-2.**-44)
        struct.pack_into('<4d',d['numerical'],off,*p)
        parsed=C.parse_record(d['numerical'],C.TAIL,'equation')
        self.assertEqual(parsed['candidate'],p)
        self.assertNotEqual(parsed['candidate'][0],C.NORM.f32(p[0]))

    def test_dedicated_nuclear_history_preserves_exact_c1_record_offsets(self):
        for stride in (43008,66048,70656,72960,97536):
            d=fixture('equation',version=2)
            expected,_=C.inspect_buffers(**d,search=False)
            old=d['numerical']
            d['numerical']=b''.join(old[l*40704:(l+1)*40704]+bytes([67+l])*(stride-40704) for l in range(8))
            actual,_=C.inspect_buffers(**d,search=False,stride=stride)
            self.assertEqual(actual,expected)
            with self.assertRaises(ValueError):C.inspect_buffers(**d,search=False)

    def test_equation_record_replay(self):
        records, lines = C.inspect_buffers(**fixture('equation'), search=False)
        self.assertEqual(len(records), 8)
        self.assertTrue(all(r['equation_candidate'] > 1e-7 for r in records))
        self.assertIn('FULL_TRAJECTORY_ACCEPTED=false', lines)

    def test_continuous_preupdate_replay(self):
        records, lines = C.inspect_buffers(**fixture('continuous'), search=False)
        self.assertEqual(len(records), 8)
        self.assertTrue(all(r['equation_next'] == 0. and r['change'] > 1e-12 for r in records))
        self.assertEqual(sum(' iteration=' in line for line in lines), 96)

    def test_full24_history_preserves_terminal_and_all_iterations(self):
        records, lines = C.inspect_buffers(**fixture('continuous', 2, 24), search=False)
        self.assertTrue(all(r['iteration'] == 23 for r in records))
        self.assertEqual(sum(' iteration=' in line for line in lines), 192)
        self.assertEqual(sum('history=CHT2' in line for line in lines), 8)

    def test_cht2_version_count_mask_and_terminal_are_bound(self):
        changes = [(C.HISTORY2, '<I', C.CHT), (C.HISTORY2+4, '<I', 1),
                   (C.HISTORY2+8, '<I', 12), (C.HISTORY2+12, '<I', 0x7fffff),
                   (C.PREFIX+28, '<I', 11), (C.PREFIX+28, '<I', 24),
                   (C.HISTORY2+16+23*24, '<d', 2e-8),
                   (C.HISTORY2+16+20*24+20, '<I', 193)]
        for offset, fmt, value in changes:
            with self.subTest(offset=offset):
                data = fixture('continuous', 2, 24)
                struct.pack_into(fmt, data['numerical'], offset, value)
                with self.assertRaises(ValueError): C.inspect_buffers(**data, search=False)

    def test_rrf3_keeps_equation_capture_identity(self):
        data = fixture('equation', 2)
        for leaf in range(C.LEAVES):
            struct.pack_into('<I', data['numerical'], leaf*C.STRIDE+C.PREFIX, C.RRF3)
        records, _ = C.inspect_buffers(**data, search=False)
        self.assertEqual(len(records), 8)

    def test_rrf3_cannot_mix_leaf_schemas_or_claim_old_equation_record(self):
        data = fixture('equation', 2)
        struct.pack_into('<I', data['numerical'], C.PREFIX, C.RRF3)
        with self.assertRaisesRegex(ValueError, 'schemas disagree'):
            C.inspect_buffers(**data, search=False)
        data = fixture('equation', 1)
        for leaf in range(C.LEAVES):
            struct.pack_into('<I', data['numerical'], leaf*C.STRIDE+C.PREFIX, C.RRF3)
        with self.assertRaisesRegex(ValueError, 'RRF3 requires EQR2'):
            C.inspect_buffers(**data, search=False)

    def test_version2_derives_redundant_values_from_raw_state(self):
        for kind in ('equation', 'continuous'):
            with self.subTest(kind=kind):
                old, _ = C.inspect_buffers(**fixture(kind), search=False)
                new, lines = C.inspect_buffers(**fixture(kind, version=2), search=False)
                for a, b in zip(old, new):
                    for key in ('norm_start', 'norm_trial', 'equation_trial'):
                        self.assertEqual(a[key], b[key])
                    self.assertEqual(b['version'], 2)
                self.assertTrue(any(('kind=EQR2' if kind == 'equation' else 'kind=CFR2') in line for line in lines))

    def test_version2_reserved_and_saved_native_argmax_are_strict(self):
        for kind, indices in (('equation', (3, 5, 6, 8)), ('continuous', (3, 4, 5, 7, 8))):
            for index in indices:
                for value in (-0., 1.):
                    with self.subTest(kind=kind, index=index, value=value):
                        data = fixture(kind, version=2)
                        base = C.TAIL+144 if kind == 'equation' else C.PREFIX+160
                        struct.pack_into('<d', data['numerical'], base+8*index, value)
                        with self.assertRaises(ValueError): C.inspect_buffers(**data, search=False)
        data = fixture('equation', version=2)
        value = struct.unpack_from('<d', data['numerical'], C.TAIL+144+7*8)[0]
        import math
        struct.pack_into('<d', data['numerical'], C.TAIL+144+7*8, math.nextafter(value, 0.))
        with self.assertRaisesRegex(ValueError, 'saved native argmax'): C.inspect_buffers(**data, search=False)

    def test_population_and_epoch_corruptions(self):
        changes = [('numerical', C.TAIL+8, '<I', 1),
                   ('numerical', C.TAIL+12, '<I', 192),
                   ('numerical', C.TAIL+44, '<I', 1),
                   ('numerical', C.TAIL+240, '<Q', 1),
                   ('exchange', 11520+96, '<I', 13),
                   ('convergence', 20, '<I', 4),
                   ('tiles', 4*7, '<I', 8),
                   ('tiles', 95232+4*7, '<f', 2.),
                   ('numerical', C.PHI+8*7, '<d', .2500000001),
                   ('exchange', 1536+8*7, '<d', .9)]
        for key, offset, fmt, value in changes:
            with self.subTest(key=key, offset=offset):
                data = fixture('equation'); struct.pack_into(fmt, data[key], offset, value)
                with self.assertRaises(ValueError): C.inspect_buffers(**data, search=False)

    def test_primitive_tampering_is_not_self_consistency(self):
        for offset, value in ((112, .01), (144+7*8, 9e-8), (144+4*8, .99), (144+11*8, .11)):
            with self.subTest(offset=offset):
                data = fixture('equation'); struct.pack_into('<d', data['numerical'], C.TAIL+offset, value)
                with self.assertRaises(ValueError): C.inspect_buffers(**data, search=False)

    def test_history_and_last_trial_negatives(self):
        changes = [(C.TAIL+8, '<I', 11), (C.TAIL+12, '<I', 0x7ff),
                   (C.TAIL+16+11*24, '<d', 2e-8), (C.TAIL+16+20, '<I', 193),
                   (C.PREFIX+64+8, '<d', 2e-8), (C.PREFIX+96+8, '<d', 2e-8)]
        for offset, fmt, value in changes:
            with self.subTest(offset=offset):
                data=fixture('continuous'); struct.pack_into(fmt, data['numerical'], offset, value)
                with self.assertRaises(ValueError): C.inspect_buffers(**data, search=False)

    def test_complete_population_required(self):
        for name in ('numerical', 'exchange', 'tiles', 'convergence'):
            data=fixture('equation'); data[name]=data[name][:-1]
            with self.subTest(name=name), self.assertRaises(ValueError): C.inspect_buffers(**data, search=False)

    def test_frozen_equation_known_diagonal_cayley(self):
        # Exact algebraic map at H=diag(2,0), h=b=1: p1=-i*p0.
        self.assertEqual(C.equation((1.,0.,0.,0.), (0.,-1.,0.,0.), (2.,0.,0.,0.), 1.,1.), 0.)
        self.assertEqual(C.equation((1.,0.,0.,0.), (0.,1.,0.,0.), (2.,0.,0.,0.), 1.,1.), 2.)

    def test_candidate_oracle_keeps_norm_and_equation_separate(self):
        r = C.parse_record(fixture('equation')['numerical'], C.TAIL, 'equation')
        counts, best = C.frozen_candidates(r)
        self.assertGreater(counts['norm'], 0)
        self.assertEqual(counts['joint'], 0)
        self.assertIsNone(best)
        # A valid fixed-H witness never claims nonlinear rebuilt-H feasibility.
        r['trial']=(1.,0.,0.,0.)
        counts, best=C.frozen_candidates(r)
        self.assertGreater(counts['joint'], 0)
        self.assertEqual(best[1], 0.)

    def test_candidate_equation_uses_native_represented_threshold(self):
        r = C.parse_record(fixture('equation')['numerical'], C.TAIL, 'equation')
        r['trial'] = (1., C.NATIVE_TOL, 0., 0.)
        self.assertGreater(C.NATIVE_TOL, C.TOL)
        counts, best = C.frozen_candidates(r)
        self.assertGreater(counts['joint'], 0)
        self.assertEqual(best[0], 0.)
        self.assertEqual(best[1], C.NATIVE_TOL)
        self.assertEqual(best[3], r['trial'])
        self.assertTrue(C.NORM.admitted(best[3], r['norm_start'], r['norm_outer'], absolute_one=True))

    def test_output_cannot_replace_a_prior_replay(self):
        with tempfile.TemporaryDirectory(prefix='biprism_c1_output_') as directory:
            path = Path(directory)/'replay.txt'
            C.write_output(path, 'first observation\n')
            with self.assertRaises(FileExistsError):
                C.write_output(path, 'different observation\n')
            self.assertEqual(path.read_text(), 'first observation\n')

    def test_actual_declaration_layout_and_accessor(self):
        source = C.MAIN.read_text()
        first = source.index('enum BiprismV10ScalarSlot')
        last = source.index('static_assert(offsetof(BiprismV10NumericalScratchFfi,working_phi)', first)
        last = source.index('\n', last)
        start = source.index('__device__ __forceinline__ double bp_v10_diagnostic_component')
        end = source.index('\n}', start)+2
        pre = '''#include <cstdint>
#include <cstddef>
#include <cstring>
#define __align__(n) alignas(n)
#define __host__
#define __device__
#define __forceinline__ inline
#define TILE_CORE_VERTICES 192
#define TILE_TOTAL_VERTICES 256
#define TILE_MAX_EDGES 1536
#define MAX_VARIATIONAL_PICARD_ITERS 12
#define MAX_CONTINUOUS_ELECTRONIC_ITERS 24
struct float2{float x,y;};
template<class R>struct BiprismElectronicComplex{R re,im;};
template<class R>struct BiprismElectronicSpinor{BiprismElectronicComplex<R> s0,s1;};
template<class R>struct BiprismElectronicHermitian{R h00,h11,h01_re,h01_im;};
template<class R>struct BiprismElectronicVector{R x,y,z;};
struct BiprismPsi64{BiprismElectronicComplex<double> s0,s1;};
'''
        program = r'''
int main(){
  BiprismPsi64 p={{1.0000000000000002,-0.0},{0x1p-100,0x1p100}};
  double a[4]={p.s0.re,p.s0.im,p.s1.re,p.s1.im};
  for(unsigned i=0;i<4;++i){double q=bp_v10_diagnostic_component(p,i);if(std::memcmp(&a[i],&q,8))return 1;}
  BiprismV10NumericalScratchFfi state;
  std::memset(&state,0x5a,sizeof(state));
  state.nuclear_history.magic=0x4e485034u;
  for(unsigned row=0;row<48;++row)for(unsigned k=0;k<3;++k)state.nuclear_history.rows[row][k]=row*3u+k;
  const size_t begin=offsetof(BiprismV10NumericalScratchFfi,nuclear_history),end=begin+sizeof(state.nuclear_history);
  const auto* bytes=reinterpret_cast<const unsigned char*>(&state);
  for(size_t i=0;i<sizeof(state);++i)if((i<begin||i>=end)&&bytes[i]!=0x5au)return 4;
  for(unsigned lane=0;lane<192;++lane)if(state.rank_search.visited_corners[lane]!=0x5a5a5a5au)return 2;
  // The historical256-double mapping must fail this same live-mask canary.
  std::memset(&state,0x5a,sizeof(state));
  for(unsigned lane=0;lane<256;++lane){
    const double value=double(lane);
    std::memcpy(reinterpret_cast<unsigned char*>(&state)+offsetof(BiprismV10NumericalScratchFfi,next_mid_v)+8u*lane,&value,8u);
  }
  if(state.rank_search.visited_corners[0]==0x5a5a5a5au)return 3;
  std::memset(&state,0x5a,sizeof(state));
  for(unsigned row=0;row<BIPRISM_MAX_NUCLEAR_EVALUATIONS;++row)
    for(unsigned k=0;k<9;++k)bp_v10_evaluation_row(&state,row)[k]=row*9u+k;
  const size_t eval=offsetof(BiprismV10NumericalScratchFfi,nuclear_evaluations)+128;
  for(size_t i=0;i<sizeof(state);++i){
    bool row=(i>=eval&&i<eval+192*72)
      ||(i>=1697920&&i<1697920+832*72)
      ||(i>=1784448&&i<1784448+512*72)
      ||(i>=1837696&&i<1837696+1536*72);
    if(!row&&bytes[i]!=0x5au)return 5;
  }
  for(unsigned row=0;row<BIPRISM_MAX_NUCLEAR_EVALUATIONS;++row)
    for(unsigned k=0;k<9;++k)if(bp_v10_evaluation_row(&state,row)[k]!=row*9u+k)return 6;
  std::memset(&state,0x5a,sizeof(state));
  for(unsigned row=0;row<BIPRISM_MAX_NUCLEAR_EVALUATIONS;++row)
    for(unsigned k=0;k<4;++k)bp_v10_arc_history_row(&state,row)[k]=row*4u+k;
  for(size_t i=0;i<sizeof(state);++i){
    bool row=(i>=1691664&&i<1691664+192*32)
      ||(i>=1757824&&i<1757824+832*32)
      ||(i>=1821312&&i<1821312+512*32)
      ||(i>=1948288&&i<1948288+1536*32);
    if(!row&&bytes[i]!=0x5au)return 7;
  }
  static_assert(offsetof(BiprismV10EquationRefusalFfi,start)==48);
  static_assert(offsetof(BiprismV10EquationRefusalFfi,candidate_spinor)==80);
  static_assert(offsetof(BiprismV10EquationRefusalFfi,continuous)==112);
  static_assert(offsetof(BiprismV10EquationRefusalFfi,connection_im)==176);
  static_assert(offsetof(BiprismV10ContinuousRefusalFfi,trial)==80);
  static_assert(offsetof(BiprismV10ContinuousRefusalFfi,connection_im)==176);
  return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix='biprism_c1_refusal_cpu_') as directory:
            path=Path(directory)/'control.cpp'; binary=Path(directory)/'control'
            path.write_text('#include \"'+str(C.MAIN.parent/'nhs_biprism_coordinate_precision.cuh')+'\"\n#include \"'+str(C.MAIN.parent/'nhs_biprism_nuclear_iteration.cuh')+'\"\n'+'#include \"'+str(C.MAIN.parent/'nhs_biprism_nuclear_budget.cuh')+'\"\n'+pre+source[first:last]+'\n'+source[start:end]+program)
            result=subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-pedantic',str(path),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertEqual(subprocess.run([str(binary)]).returncode,0)


if __name__ == '__main__':
    unittest.main()
