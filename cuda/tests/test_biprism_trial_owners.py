"""Actual coordinate/Q39 primitives and parity-fixture owner initialization.

CPU evidence only. These controls do not stand in for native coupled acceptance.
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

K = Path(__file__).resolve().parents[2] / 'crates/prism-gpu/src/kernels'
MAIN = K / 'nhs_biprism_variational_so3g_v10.cuh'
PARITY = Path(__file__).with_name('biprism_v10_reciprocal_parity.cuh')


def fragment(source, name):
    start = source.index(name)
    start = source.rfind('\n', 0, start) + 1
    a = source.index('{', start)
    depth = 1
    b = a + 1
    while depth:
        depth += (source[b] == '{') - (source[b] == '}')
        b += 1
    return source[start:b]


class TrialOwners(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='biprism_trial_owners_')
        root = Path(cls.tmp.name)
        main, parity = MAIN.read_text(), PARITY.read_text()
        q39 = fragment(main, 'long long bp_v10_q39(')
        prepare = fragment(parity, 'void prepare_edges(')
        prepare = prepare[:prepare.index('    const uint32_t count =')] + '\n}'
        vertex = fragment(parity, 'BiprismCoordinateExpansion3 vertex_position(')
        pre = r'''
#include <algorithm>
#include <cmath>
#include <cstring>
using std::max; using std::isfinite;
#define __device__
#define __forceinline__ inline
#define __global__
#define __shared__
#define __align__(n) __attribute__((aligned(n)))
#define TILE_TOTAL_VERTICES 256
#define TILE_CORE_VERTICES 192
constexpr uint32_t kLeaves=8, FAULT_PICARD_NONCONV=2;
struct float3 {float x,y,z;};
struct Dim{unsigned x;}; Dim blockIdx{},threadIdx{};
struct FoliatedStreamPtrTableFfi{};
struct ContactShellSimplicialStateFfi{};
struct EdgeRecord{};
struct VertexRecord {float x[3],x_low[3];};
struct BiprismV10NumericalScratchFfi {BiprismCoordinateExpansion3 nuclear_position_live[256];};
struct VariationalClusterDsmemV10 {
    float3 x_mid[256];
    struct {ContactShellSimplicialStateFfi* tile;BiprismV10NumericalScratchFfi* numerical;
      struct {const FoliatedStreamPtrTableFfi* streams;unsigned leaf;} collective;
    } invocation;
    unsigned local_fault_accum;
};
alignas(128) unsigned char scratch_bytes[sizeof(VariationalClusterDsmemV10)];
inline void atomicOr(unsigned* p,unsigned v){*p|=v;}
inline long long __double2ll_rn(double x){return std::llrint(x);}
'''
        wrapper = r'''
extern "C" long long quantize(double value,unsigned terms,unsigned* fault){
    VariationalClusterDsmemV10 s{};const auto q=bp_v10_q39(value,terms,&s);*fault=s.local_fault_accum;return q;
}
extern "C" void advance(const float* x,double h,const double* v,float* out,double* delta){
    const BiprismCoordinateExpansion3 p={{x[0],x[1]},{x[2],x[3]},{x[4],x[5]}};
    const BiprismCoordinateExpansion3 q={bp_coord_fma(p.x,h,v[0]),bp_coord_fma(p.y,h,v[1]),bp_coord_fma(p.z,h,v[2])};
    std::memcpy(out,&q,24);const auto d=bp_coord_difference(q,p);delta[0]=d.x;delta[1]=d.y;delta[2]=d.z;
}
extern "C" int initialize(int missing){
    FoliatedStreamPtrTableFfi streams{};ContactShellSimplicialStateFfi tiles[8]{};
    BiprismV10NumericalScratchFfi numerical[8]{};VertexRecord vertices[2048]{};
    for(unsigned i=0;i<2048;++i)for(unsigned k=0;k<3;++k){vertices[i].x[k]=float(i+k);vertices[i].x_low[k]=std::ldexp(float(k+1),-30);}
    for(unsigned leaf=0;leaf<8;++leaf){
        std::memset(scratch_bytes,0,sizeof(scratch_bytes));blockIdx.x=leaf;
        for(unsigned tid=0;tid<256;++tid){threadIdx.x=tid;
            if(missing)prepare_edges_missing_bind(&streams,tiles,nullptr,vertices,nullptr,numerical);
            else prepare_edges(&streams,tiles,nullptr,vertices,nullptr,numerical);
        }
        auto* s=reinterpret_cast<VariationalClusterDsmemV10*>(scratch_bytes);
        if(s->invocation.numerical!=numerical+leaf)return 0;
        for(unsigned tid=0;tid<256;++tid){const auto expected=vertex_position(vertices[leaf*256+tid]);
            if(std::memcmp(&expected,&s->invocation.numerical->nuclear_position_live[tid],24))return 0;}
    }
    return 1;
}
'''
        missing = prepare.replace('void prepare_edges(', 'void prepare_edges_missing_bind(').replace('scratch->invocation.numerical=numerical+leaf;', '')
        assert prepare != missing and 'scratch->invocation.numerical=numerical+leaf;' in prepare
        cpp = root / 'controls.cpp'
        cpp.write_text('#include "'+str(K/'nhs_biprism_coordinate_precision.cuh')+'"\n'+pre+q39+vertex+prepare+missing+wrapper)
        subprocess.run(['g++','-std=c++17','-O2','-fno-fast-math','-shared','-fPIC',str(cpp),'-o',str(root/'controls.so')],check=True,capture_output=True)
        cls.lib = ctypes.CDLL(str(root/'controls.so'))
        cls.lib.quantize.argtypes = [ctypes.c_double,ctypes.c_uint,ctypes.POINTER(ctypes.c_uint)]
        cls.lib.quantize.restype = ctypes.c_longlong
        cls.lib.initialize.argtypes = [ctypes.c_int]
        cls.lib.advance.argtypes = [ctypes.POINTER(ctypes.c_float),ctypes.c_double,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_double)]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def q(self, x, terms):
        flag=ctypes.c_uint()
        value=self.lib.quantize(x,terms,ctypes.byref(flag))
        return value,flag.value

    def test_actual_fixture_initializes_every_owner_before_read(self):
        self.assertEqual(self.lib.initialize(0),1)
        self.assertEqual(self.lib.initialize(1),0, 'missing owner must fail before dereference')

    def test_incidence_accepts_localized_force_without_changing_rounding(self):
        force=-10114.0562121
        self.assertEqual(self.q(force,26),(round(force*2**39),0))
        self.assertEqual(self.q(force,1462),(0,2))
        for terms in (1,2,26,192,1462,3073):
            boundary=8388608./terms
            self.assertEqual(self.q(boundary,terms),(round(boundary*2**39),0))
            self.assertEqual(self.q(math.nextafter(boundary,math.inf),terms),(0,2))
        for x in (math.nan,math.inf,-math.inf):
            self.assertEqual(self.q(x,26),(0,2))

    def test_partial_sum_bound_including_integer_rounding(self):
        for degree in range(1,3073):
            self.assertLess(Q(2**62)+Q(degree+1,2),2**63)
        rng=random.Random(51054)
        for degree in (1,7,25,192,1461):
            total=0
            for _ in range(degree+1):
                # A neighbor may impose a larger denominator, never smaller.
                terms=degree+1+rng.randrange(100)
                value=rng.uniform(-1,1)*8388608./terms
                q,flag=self.q(value,terms);self.assertEqual(flag,0)
                total+=q;self.assertLess(abs(total),2**63)

    def test_full_pair_retains_sub_fp32_midpoint_changes(self):
        f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
        x=[f32(13.0),f32(2**-23),f32(-9.),f32(2**-25),f32(.2),0.]
        h=5.1676716804504395
        velocities=[.034,-.02,.012]
        previous=None
        for dv in (0.,2e-10):
            v=[velocities[0]+dv,*velocities[1:]];out=(ctypes.c_float*6)();delta=(ctypes.c_double*3)()
            self.lib.advance((ctypes.c_float*6)(*x),h,(ctypes.c_double*3)(*v),out,delta)
            for k in range(3):
                actual=Q(out[2*k])+Q(out[2*k+1]);start=Q(x[2*k])+Q(x[2*k+1]);exact=start+Q(h)*Q(v[k])
                self.assertLess(abs(actual-exact),Q(1,10**13))
                self.assertLess(abs(Q(delta[k])-(actual-start)),Q(1,10**15))
            if previous is not None:
                self.assertNotEqual(bytes(out),previous)
                self.assertEqual(f32(x[1]+h*velocities[0]),f32(x[1]+h*v[0]), 'old rounded residual cannot distinguish these trials')
            previous=bytes(out)

if __name__=='__main__':
    unittest.main()
