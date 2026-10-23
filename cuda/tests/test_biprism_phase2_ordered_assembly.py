"""Compile the actual Phase2 gather and check ordered represented sums on CPU."""
import ctypes
import random
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

HEADER=Path(__file__).resolve().parents[2]/'crates/prism-gpu/src/kernels/nhs_biprism_simplicial_tile.cuh'


class OrderedAssembly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(prefix='biprism_phase2_cpu_')
        p=Path(cls.tmp.name);src=HEADER.read_text()
        layout=src[src.index('struct __align__(16) SimplicialTilePhase2Smem {'):src.index('// =============================================================================================\n// PHASE 2 KERNEL')]
        gather=src[src.index('        const uint32_t count=min(256u,num_tets-base);'):src.index('        __syncthreads(); // All term readers')]
        body='''
#include <algorithm>
#include <cstdint>
#include <cstddef>
using std::min;
#define __align__(n) alignas(n)
#define TILE_TOTAL_VERTICES 256
#define TILE_MAX_EDGES 1536
struct alignas(16) float4 {float x,y,z,w;};
'''+layout+'''
struct Tile {uint8_t tet_v[1024][4];uint16_t tet_edges[1024][6];};
extern "C" void assemble(unsigned num_tets,unsigned num_edges,const uint8_t* vertices,const uint16_t* edges,
    const float* terms,float* out){
    Tile data{};auto* tile=&data;SimplicialTilePhase2Smem smem{};
    for(unsigned t=0;t<num_tets;++t){
        for(unsigned k=0;k<4;++k)tile->tet_v[t][k]=vertices[4*t+k];
        for(unsigned k=0;k<6;++k)tile->tet_edges[t][k]=edges[6*t+k];
    }
    for(unsigned tid=0;tid<256;++tid)smem.star0[tid]=1e-5f;
    for(unsigned base=0;base<num_tets;base+=256u){
        for(unsigned j=0;j<min(256u,num_tets-base);++j)
            for(unsigned k=0;k<7;++k)smem.tet_terms[k][j]=terms[7*(base+j)+k];
        for(unsigned tid=0;tid<256;++tid){
'''+gather+'''
        }
    }
    for(unsigned i=0;i<256;++i)out[i]=smem.star0[i];
    for(unsigned i=0;i<num_edges;++i)out[256+i]=smem.star1[i];
}
static_assert(offsetof(SimplicialTilePhase2Smem,tet_terms)>=offsetof(SimplicialTilePhase2Smem,star1)+sizeof(SimplicialTilePhase2Smem::star1),"live weights outside stage");
static_assert(offsetof(SimplicialTilePhase2Smem,edge_u)>=offsetof(SimplicialTilePhase2Smem,tet_terms)+sizeof(SimplicialTilePhase2Smem::tet_terms),"live incidence outside stage");
'''
        (p/'a.cpp').write_text(body)
        subprocess.run(['g++','-std=c++17','-O2','-fno-fast-math','-shared','-fPIC',str(p/'a.cpp'),'-o',str(p/'a.so')],check=True,capture_output=True)
        cls.lib=ctypes.CDLL(str(p/'a.so'))
        cls.lib.assemble.argtypes=[ctypes.c_uint,ctypes.c_uint,ctypes.POINTER(ctypes.c_uint8),ctypes.POINTER(ctypes.c_uint16),ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float)]

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_full_and_partial_batches_match_serial_tetrahedron_order(self):
        rng=random.Random(56055);f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
        for count in (1,255,256,257,960,1024):
            nedge=1536;v=[];e=[];w=[];expected=[f32(1e-5)]*256+[0.]*nedge
            for t in range(count):
                vt=rng.sample(range(256),4);et=rng.sample(range(nedge),6)
                wt=[f32(2.**rng.randint(-20,8)*rng.uniform(.5,1.)) for _ in range(7)]
                v+=vt;e+=et;w+=wt
                for u in vt:expected[u]=f32(expected[u]+wt[0])
                for k,edge in enumerate(et):expected[256+edge]=f32(expected[256+edge]+wt[1+k])
            out=(ctypes.c_float*(256+nedge))()
            self.lib.assemble(count,nedge,(ctypes.c_uint8*len(v))(*v),(ctypes.c_uint16*len(e))(*e),(ctypes.c_float*len(w))(*w),out)
            self.assertEqual(bytes(out),struct.pack('<'+str(len(expected))+'f',*expected),count)


if __name__=='__main__':unittest.main()
