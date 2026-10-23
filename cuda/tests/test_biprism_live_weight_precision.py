"""Storage/reader and independent radial checks at the native119 boundary.

These validate primitive precision, not a self-consistent trajectory.
"""
import math
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import biprism_exchange_layout as EX
from check_biprism_v10_dec_readback import precise_weight

K=Path(__file__).resolve().parents[2]/'crates/prism-gpu/src/kernels'

class LiveWeightPrecision(unittest.TestCase):
    def test_actual_rust_exchange_layout_and_full_precision_decode(self):
        runtime=(K.parents[2]/'prism-nhs/src/biprism_v10_runtime.rs').read_text()
        begin=runtime.index('/// Exact derived replay scratch ABI.')
        end=runtime.index('/// Validate the named final-substep publication',begin)
        source='const BIPRISM_ELECTRONIC_EXCHANGE_BYTES:usize=17792;\n'+runtime[begin:end]
        source+=r'''
fn main(){
    let value=310.01176446155296_f64;
    let mut bytes=vec![0u8;8*BIPRISM_ELECTRONIC_EXCHANGE_BYTES];
    let offset=3*BIPRISM_ELECTRONIC_EXCHANGE_BYTES+5376+363*8;
    bytes[offset..offset+8].copy_from_slice(&value.to_le_bytes());
    let read=decode_electronic_exchange(&bytes).unwrap();
    assert_eq!(read.leaves[3].live_weights[363].to_bits(),value.to_bits());
    assert_ne!(value,(value as f32) as f64);
    assert_eq!(read.raw_bytes,bytes);
    assert!(decode_electronic_exchange(&vec![0;8*11648]).is_err());
}
'''
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'control.rs').write_text(source)
            subprocess.run(['rustc','--edition=2021','-O',str(p/'control.rs'),'-o',str(p/'control')],check=True,capture_output=True)
            subprocess.run([str(p/'control')],check=True,capture_output=True)

    def test_versions_preserve_full_values_and_reject_reinterpretation(self):
        for size,(footer,fmt,magic) in EX.LAYOUTS.items():
            raw=bytearray(8*size)
            for leaf in range(8):
                struct.pack_into('<I',raw,leaf*size+footer+80,magic)
                struct.pack_into('<'+fmt,raw,leaf*size+5376,310.01176446155296)
            value=EX.weights(raw,3,1)[0]
            self.assertEqual(value,310.01176446155296 if fmt=='d' else 310.0117492675781)
            struct.pack_into('<I',raw,footer+80,0x46435033)
            with self.assertRaises(ValueError):EX.layout(raw)

    def test_compiled_radial_low_bits_and_cxx_record_storage(self):
        source=r'''
#include "nhs_biprism_coordinate_geometry.cuh"
#include "nhs_biprism_foliated_population.cuh"
#define BIPRISM_PHASE2_WORK_ABI_ONLY
#define BIPRISM_MIXED_ENERGY_ABI_ONLY
#include "nhs_biprism_phase2_work.cuh"
#include "nhs_biprism_mixed_energy.cuh"
#include <cstdio>
#include <cstring>
int main() {
    const BiprismCoordinate3F64 a={1.+0x1p-30,.3,-.2},b={1.,.3,-.2};
    const double w=bp_geometry_live_weight_f64(310.011749267578125f,a,b);
    FoliatedCouplingLeafFfi bank={},copy={};bank.live_weights[363]=w;
    std::memcpy(&copy,&bank,sizeof(bank));
    BiprismPhase2EdgeWorkFfi edge={};edge.live_before=w;edge.live_after=w+0x1p-30;
    BiprismPhase2VertexWorkFfi u={},v={};u.phi_before=u.phi_same_halo=u.phi_after=2.;
    biprism_phase2_edge_energy(edge,u,v);
    BiprismMixedEdgeFfi mixed={};mixed.live=w;
    if(copy.live_weights[363]!=w||mixed.live!=w||double(float(w))==w
        ||edge.operator_cross_work!=-2.*0x1p-30)return 2;
    std::printf("%a %zu %zu %zu\n",w,sizeof(bank),sizeof(edge),sizeof(mixed));
}
'''
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'control.cpp').write_text(source)
            subprocess.run(['g++','-std=c++17','-O2','-I'+str(K),'-I/usr/local/cuda/include',
                            str(p/'control.cpp'),'-o',str(p/'control')],check=True,capture_output=True)
            result=subprocess.check_output([str(p/'control')],text=True).split()
        zero=((0.,0.),)*3
        a=((1.,2.**-30),(.3,0.),(-.2,0.));b=((1.,0.),(.3,0.),(-.2,0.))
        expected=precise_weight(310.011749267578125,zero,a,zero,b,fp64=True)
        self.assertEqual(float.fromhex(result[0]),expected)
        self.assertEqual(list(map(int,result[1:])),[17792,80,64])

if __name__=='__main__':unittest.main()
