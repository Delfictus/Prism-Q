#!/usr/bin/env python3
"""Bounded source/CPU controls for retained-potential scalar dispatch.

The CPU executable compiles the actual complete headers, but only executes
their noncollective row functions and the extracted potential update. CUDA
shuffle/barrier stand-ins are never executed. Native collective qualification
remains a separate required control.
"""
from pathlib import Path
import hashlib
import re
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
KERNELS = HERE.parents[1]/'crates/prism-gpu/src/kernels'
HEADER = KERNELS/'nhs_biprism_dec_pcg.cuh'
OLD_BLOB = '74f16cbef097f41267b6bccc1596fc95a9b3bdf8'
OLD_SHA = '19698481371739f3772894f536df524a629ab97753786f5c78b4c14d52f33c8d'


def body(text, name):
    begin = text.index('{', text.index(name+'('))
    i, depth = begin+1, 1
    while depth:
        depth += (text[i] == '{')-(text[i] == '}')
        i += 1
    return text[begin:i]


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.current = HEADER.read_text()
        cls.old = subprocess.check_output(['git', 'cat-file', 'blob', OLD_BLOB], cwd=HERE).decode()
        assert hashlib.sha256(cls.old.encode()).hexdigest() == OLD_SHA

    def test_old_row_and_wrapper_bodies_are_exact(self):
        for name in ('biprism_dec_sum_f64_256', 'biprism_dec_row_f64_traversal',
                     'biprism_dec_row_f64_accessor', 'biprism_dec_pcg_256_accessor',
                     'biprism_dec_pcg_256_indexed_accessor', 'biprism_dec_row_f64',
                     'biprism_dec_pcg_256'):
            self.assertEqual(body(self.old, name), body(self.current, name), name)

    def test_float_solver_reconstructs_exact_previous_body(self):
        current = body(self.current, 'biprism_dec_pcg_256_traversal')
        type_part = '''    using PotentialValue = typename std::remove_cv<PotentialScalar>::type;
    static_assert(std::is_same<PotentialValue,float>::value
        ||std::is_same<PotentialValue,double>::value,"DEC potential must be FP32 or retained FP64");
'''
        self.assertIn(type_part, current)
        current = current.replace(type_part, '', 1)
        branch = '''            if constexpr (std::is_same<PotentialValue,float>::value) {
                // Keep the historical FP32 update and its contraction exact.
                phi[tid] = __fmaf_rn(alpha, direction[tid], phi[tid]);
            } else {
                // Retain all potential bits through residual/work and commit.
                phi[tid] = __fma_rn(static_cast<double>(alpha),
                    static_cast<double>(direction[tid]),static_cast<double>(phi[tid]));
            }'''
        self.assertIn(branch, current)
        current = current.replace(branch, '            phi[tid] = __fmaf_rn(alpha, direction[tid], phi[tid]);', 1)
        # At the existing default, the new internal target reconstructs the
        # historical stopping predicates exactly. No operator/update changes.
        guard = '''    // An internal solve target may be tighter than the external admission
    // ceiling. It may never relax that ceiling or change the iteration cap.
    bool valid = isfinite(internal_tol_sq) && internal_tol_sq > 0.0
        && internal_tol_sq <= static_cast<double>(BIPRISM_DEC_TOL_SQ)
        && isfinite(phi[tid]) && (!core || isfinite(charge));'''
        self.assertIn(guard, current)
        current = current.replace(guard, '    bool valid = isfinite(phi[tid]) && (!core || isfinite(charge));')
        current = current.replace('true_rr <= internal_tol_sq', 'true_rr <= static_cast<double>(BIPRISM_DEC_TOL_SQ)')
        current = current.replace('static_cast<double>(rr) <= internal_tol_sq', 'rr <= BIPRISM_DEC_TOL_SQ')
        self.assertEqual(current, body(self.old, 'biprism_dec_pcg_256_traversal'))
        for declaration in ('constexpr uint32_t BIPRISM_DEC_MAX_ITERS = TILE_CORE_VERTICES;',
                            'constexpr uint32_t BIPRISM_DEC_RESTART_INTERVAL = 16u;',
                            'constexpr float BIPRISM_DEC_TOL_SQ = 1e-12f;'):
            self.assertIn(declaration, self.old)
            self.assertIn(declaration, self.current)

    def test_actual_typed_rows_and_update_retain_low_values(self):
        # Extract the actual compile-time update branch, not a copied formula.
        start = self.current.index('            if constexpr (std::is_same<PotentialValue,float>::value)')
        stop = self.current.index('            residual[tid] =', start)
        update = self.current[start:stop].replace('phi[tid]', 'value').replace('direction[tid]', 'direction')
        source = r'''
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <type_traits>
#include <vector>
#define __device__
#define __forceinline__ inline
#define FULL_WARP_MASK 0xffffffffu
constexpr uint32_t TILE_CORE_VERTICES=192u;
using std::isfinite;
struct { uint32_t x=0; } threadIdx;
inline void __syncthreads() { std::abort(); }
inline int __syncthreads_and(bool) { std::abort(); }
template<class T> inline T __shfl_xor_sync(uint32_t,T,int) { std::abort(); }
inline float __int_as_float(int x) { float y;std::memcpy(&y,&x,4);return y; }
inline int __float_as_int(float x) { int y;std::memcpy(&y,&x,4);return y; }
inline int __double2loint(double x) { uint64_t b;std::memcpy(&b,&x,8);return int(b); }
inline int __double2hiint(double x) { uint64_t b;std::memcpy(&b,&x,8);return int(b>>32); }
inline double __hiloint2double(int h,int l) { uint64_t b=(uint64_t(uint32_t(h))<<32)|uint32_t(l);double d;std::memcpy(&d,&b,8);return d; }
inline float cta_dot_product_256(float,float*) { std::abort(); }
inline float __fmaf_rn(float a,float b,float c) { return std::fma(a,b,c); }
inline double __fma_rn(double a,double b,double c) { return std::fma(a,b,c); }
#include "math_constants.h"
#include "nhs_biprism_ordered_edges.cuh"
namespace before {
#include "before.cuh"
}
namespace current {
#include "nhs_biprism_dec_pcg.cuh"
}
template<class T> T actual_update(T value,float alpha,float direction) {
 using PotentialValue=T;
@UPDATE@
 return value;
}
static uint64_t bits(double x) { uint64_t b;std::memcpy(&b,&x,8);return b; }
static void require(bool x,const char* m) { if(!x){std::fprintf(stderr,"FAIL %s\n",m);std::exit(1);} }
int main() {
 uint64_t checks=0;
 const uint8_t edge[6]={0,1,0,192,1,193};
 const float weight[3]={1.f,2.f,.5f};
 float f[256]={};double d[256]={};
 f[0]=f[1]=f[192]=f[193]=1.f;
 for(unsigned i=0;i<256;++i)d[i]=double(f[i]);
 for(unsigned v=0;v<256;++v) {
  const double a=before::biprism_dec_row_f64(edge,edge+1,2,weight,3,f,v);
  const double b=current::biprism_dec_row_f64(edge,edge+1,2,weight,3,f,v);
  const double c=current::biprism_dec_row_f64_traversal(edge,edge+1,2,
   current::BiprismDecArrayWeights{weight},BiprismLinearEdgeTraversal{3},
   static_cast<const volatile double*>(d),v);
  require(bits(a)==bits(b)&&bits(b)==bits(c),"promoted represented phi exact row identity");++checks;
 }
 d[0]+=std::ldexp(1.,-30);
 const double row=current::biprism_dec_row_f64_traversal(edge,edge+1,2,
  current::BiprismDecArrayWeights{weight},BiprismLinearEdgeTraversal{3},
  static_cast<const volatile double*>(d),0);
 require(row==3.*std::ldexp(1.,-30),"full low-only potential reaches both incident terms");++checks;
 require(current::biprism_dec_row_f64(edge,edge+1,2,weight,3,f,0)==0.,"compatibility projection discards the same low-only value");++checks;
 for(unsigned j=0;j<4096;++j) {
  const float old=std::ldexp(float(int(j%97)-48),int(j%23)-11);
  const float a=std::ldexp(float(1+j%31),int(j%17)-14);
  const float p=std::ldexp(float(int(j%61)-30),int(j%19)-15);
  const float x=actual_update(old,a,p),expected=std::fma(a,p,old);
  require(std::memcmp(&x,&expected,4)==0,"original FP32 update expression");++checks;
  const double old64=double(old)+std::ldexp(1.,int(j%23)-45);
  const double y=actual_update(old64,a,p),want=std::fma(double(a),double(p),old64);
  require(bits(y)==bits(want),"actual FP64 branch retains potential");++checks;
 }
 require(actual_update(1.,1.f,std::ldexp(1.f,-30))==1.+std::ldexp(1.,-30),"FP64 correction survives where FP32 stagnates");++checks;
 require(actual_update(1.f,1.f,std::ldexp(1.f,-30))==1.f,"old FP32 behavior preserved");++checks;
 std::printf("PASS %llu exact-source noncollective checks; collective stand-ins not executed\n",(unsigned long long)checks);
}
'''.replace('@UPDATE@', update)
        with tempfile.TemporaryDirectory(prefix='biprism-dec-scalar-') as name:
            temp = Path(name)
            (temp/'before.cuh').write_text(self.old)
            (temp/'math_constants.h').write_text('#pragma once\n#define CUDART_INF INFINITY\n#define CUDART_INF_F INFINITY\n')
            (temp/'control.cpp').write_text(source)
            command = ['g++', '-std=c++17', '-O2', '-ffp-contract=off', '-Wno-unknown-pragmas',
                       '-I'+str(temp), '-I'+str(KERNELS), str(temp/'control.cpp'), '-o', str(temp/'control')]
            built = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(built.returncode, 0, built.stdout+built.stderr)
            run = subprocess.run([str(temp/'control')], text=True, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
            print(run.stdout.strip())

    def test_native_control_actual_schema_and_epoch_predicate(self):
        control = (HERE/'biprism_ordered_edges_native_control.hpp').read_text()
        declarations = control[control.index('struct Result {'):control.index('__global__ void compare_arithmetic')]
        source = r'''
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#define __host__
#define __device__
#include "nhs_biprism_potential_precision.cuh"
@DECLARATIONS@
int main() {
 unsigned checks=0;
 auto need=[&](bool b){++checks;if(!b){std::fprintf(stderr,"failed epoch control %u\n",checks);std::exit(1);}};
 BiprismPotentialOwnerFfi a={};
 a.state=BP_POTENTIAL_ACCEPTED;a.accepted_committed_substeps=8;a.accepted_step=2;
 a.accepted_operator_generation=7;a.projection_state=BP_POTENTIAL_ACCEPTED;
 a.projection_committed_substeps=8;a.projection_step=2;a.projection_operator_generation=7;
 need(source_epoch_valid(a,8));
 a.accepted_committed_substeps=a.projection_committed_substeps=12;
 a.accepted_step=a.projection_step=3;need(source_epoch_valid(a,8));
 need(!source_epoch_valid(a,4));need(!source_epoch_valid(a,UINT32_MAX));
 a.state=BP_POTENTIAL_COLD;need(!source_epoch_valid(a,8));
 a.state=BP_POTENTIAL_PREPARED;a.accepted_committed_substeps=8;a.accepted_step=2;
 a.prepared_from_committed_substeps=8;a.prepared_step=3;a.prepared_operator_generation=8;
 a.projection_state=BP_POTENTIAL_PREPARED;a.projection_committed_substeps=8;
 a.projection_step=3;a.projection_operator_generation=8;need(source_epoch_valid(a,8));
 for(unsigned k=0;k<9;++k){auto b=a;
  if(k==0)++b.accepted_committed_substeps;if(k==1)++b.accepted_step;
  if(k==2)++b.prepared_from_committed_substeps;if(k==3)++b.prepared_step;
  if(k==4)++b.prepared_operator_generation;if(k==5)++b.projection_state;
  if(k==6)++b.projection_committed_substeps;if(k==7)++b.projection_step;
  if(k==8)++b.projection_operator_generation;need(!source_epoch_valid(b,8));
 }
 std::printf("PASS %u actual native-control schema/epoch checks; no CUDA execution\n",checks);
}
'''.replace('@DECLARATIONS@', declarations)
        with tempfile.TemporaryDirectory(prefix='biprism-potential-epoch-') as name:
            temp = Path(name)
            (temp/'control.cpp').write_text(source)
            built = subprocess.run(['g++', '-std=c++17', '-O2', '-I'+str(KERNELS),
                                    str(temp/'control.cpp'), '-o', str(temp/'control')],
                                   text=True, capture_output=True)
            self.assertEqual(built.returncode, 0, built.stdout+built.stderr)
            run = subprocess.run([str(temp/'control')], text=True, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
            print(run.stdout.strip())


if __name__ == '__main__':
    unittest.main(verbosity=2)
