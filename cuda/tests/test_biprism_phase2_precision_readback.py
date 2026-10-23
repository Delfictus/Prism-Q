#!/usr/bin/env python3
"""Constructed reader controls; these bytes are not native execution evidence."""
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
import check_biprism_phase2_work_native as C


def owner_rows(state):
    rows=[]
    for leaf in range(8):
        b=bytearray(128)
        struct.pack_into('<4I6Q',b,0,0x31505042,1,leaf,state,23,1,
                         0x100000+leaf*C.TILE,0x400000+leaf*96,
                         0x500000+leaf*4096,0x500800+leaf*4096)
        if state:
            struct.pack_into('<4I2Q4I2Q',b,64,0,0,0,1 if state==2 else 0,
                             1,2 if state==2 else 0,state,0,1 if state==2 else 0,0,
                             2 if state==2 else 1,0)
        rows.append(b)
    return rows


def cold_rows():
    tile=bytearray(C.TILE);leaf=3;atoms=264
    ids=np.arange(8,264,dtype='<u4')[::-1].copy()
    packed=ids|np.uint32(2<<24)
    reference=(np.arange(atoms*3,dtype='<f4').reshape(atoms,3)-128)/np.float32(32)
    low=np.full((atoms,3),np.float32(2.**-30),dtype='<f4')
    refq=np.c_[reference[ids],np.full(256,np.float32(.125))].astype('<f4')
    psi=np.tile(np.array([.75,.125,.25,-.125],dtype='<f4'),(256,1))
    tile[:1024]=packed.tobytes();tile[1024:5120]=refq.tobytes();tile[90112:94208]=psi.tobytes()
    row=np.zeros(256,dtype=C.V)
    row['packed']=packed;row['leaf']=leaf;row['local']=np.arange(256);row['state']=4
    row['psi']=psi;row['psi_precise']=psi.astype('<f8')
    row['x']=row['ref']=refq[:,:3];row['xlo']=row['reflo']=low[ids];row['qamber']=refq[:,3]
    row['rho1'][:192]=[C.fma(float(z[2]),float(z[2]),float(z[3])*float(z[3])) for z in psi[:192]]
    row['q']=[C.fma(float(np.float32(.15)),float(row['rho1'][i]),float(refq[i,3])) if i<192 else float(refq[i,3]) for i in range(256)]
    return tile,row,reference,low,leaf


class Controls(unittest.TestCase):
    def call(self,rows,state,accepted=None,prepared=None):
        with tempfile.TemporaryDirectory() as tmp:
            p=str(Path(tmp)/'state')
            Path(p+'.sample_potential_owners.128b').write_bytes(b''.join(rows))
            a=np.zeros((8,256),dtype='<f8') if accepted is None else accepted
            b=np.zeros((8,256),dtype='<f8') if prepared is None else prepared
            a.tofile(p+'.sample_phi_accepted.f64le');b.tofile(p+'.sample_phi_prepared.f64le')
            return C.potential_state(p,'sample',state)

    def test_actual_abi_offsets(self):
        self.assertEqual(C.V.itemsize,256)
        self.assertEqual(C.V.fields['psi_precise'][1],224)
        self.assertEqual(C.V.fields['p0'][1],72)
        self.assertEqual(C.V.fields['entry'][1],216)
        self.assertEqual(C.LDT.itemsize,240)
        self.assertEqual(C.LDT.fields['refinement'][1],232)

    def test_full_potential_survives_reader_without_float_roundtrip(self):
        a=np.zeros((8,256),dtype='<f8');a[3,9]=1.+2.**-40
        _,read,_=self.call(owner_rows(1),1,a)
        self.assertEqual(read[3,9],a[3,9]);self.assertNotEqual(read[3,9],float(np.float32(a[3,9])))

    def test_cold_accepted_and_prepared_metadata(self):
        for state in (0,1,2):
            with self.subTest(state=state):self.call(owner_rows(state),state)

    def test_stale_step_generation_projection_and_reserved_refuse(self):
        for offset,fmt,value in ((4,'I',2),(8,'I',2),(16,'Q',22),(24,'Q',2),
                                 (68,'I',1),(76,'I',2),(88,'Q',3),(96,'I',1),
                                 (104,'I',0),(108,'I',1),(112,'Q',1),(120,'Q',1)):
            rows=owner_rows(2);struct.pack_into('<'+fmt,rows[3],offset,value)
            with self.subTest(offset=offset),self.assertRaises(AssertionError):self.call(rows,2)

    def test_cross_leaf_bank_alias_refuses(self):
        rows=owner_rows(1);rows[4][48:56]=rows[3][48:56]
        with self.assertRaises(AssertionError):self.call(rows,1)

    def test_partial_bank_overlap_refuses(self):
        rows=owner_rows(1);p=struct.unpack_from('<Q',rows[4],48)[0]
        struct.pack_into('<Q',rows[4],56,p+8)
        with self.assertRaises(AssertionError):self.call(rows,1)

    def test_nonzero_aligned_descriptor_bases_required(self):
        for field in (32,40):
            for base in (0,1):
                rows=owner_rows(1)
                for leaf,row in enumerate(rows):
                    struct.pack_into('<Q',row,field,base+leaf*(C.TILE if field==32 else 96))
                with self.subTest(field=field,base=base),self.assertRaises(AssertionError):self.call(rows,1)

    def test_valid_individual_snapshots_cannot_rebind_allocations(self):
        before=owner_rows(1);after=owner_rows(2)
        C.potential_allocation_unchanged(b''.join(before),b''.join(after))
        for field in (32,40,48,56):
            changed=owner_rows(2)
            for row in changed:
                value=struct.unpack_from('<Q',row,field)[0]
                struct.pack_into('<Q',row,field,value+0x1000000)
            self.call(changed,2)  # Each snapshot alone remains well formed.
            with self.subTest(field=field),self.assertRaises(AssertionError):
                C.potential_allocation_unchanged(b''.join(before),b''.join(changed))

    def test_exact_residual_uses_full_potential(self):
        phi=np.zeros(256,dtype='<f8');q=np.zeros(192,dtype='<f8')
        phi[0]=1.+2.**-40;q[0]=1.
        squared=C.exact_residual_squared(phi,q,[(0,192)],[1.])
        self.assertEqual(squared,C.Fraction(1,2**80))
        phi[0]=float(np.float32(phi[0]))
        self.assertEqual(C.exact_residual_squared(phi,q,[(0,192)],[1.]),0)

    def test_cold_geometry_and_actual_tile_spinor_binding(self):
        tile,row,reference,low,leaf=cold_rows()
        C.bind_cold_rows(tile,row,reference,low,leaf)
        self.assertTrue(C.same_f64(row['psi_precise'],row['psi'].astype('<f8')))  # Exact cold promotion.
        self.assertTrue(np.all(row['rho1'][192:]==0))

    def test_cold_high_low_and_charge_corruptions_refuse(self):
        for field in ('x','ref','xlo','reflo','qamber','rho1','q','psi_precise'):
            tile,row,reference,low,leaf=cold_rows()
            values=row[field];index=(17,1) if values.ndim==2 else 17
            values[index]=np.nextafter(values[index],np.array(np.inf,dtype=values.dtype))
            with self.subTest(field=field),self.assertRaises(AssertionError):
                C.bind_cold_rows(tile,row,reference,low,leaf)
        for plane in ('reference','low'):
            tile,row,reference,low,leaf=cold_rows()
            values=reference if plane=='reference' else low
            values[200,1]=np.nextafter(values[200,1],np.float32(np.inf))
            with self.subTest(plane=plane),self.assertRaises(AssertionError):
                C.bind_cold_rows(tile,row,reference,low,leaf)

    def test_cold_population_identity_corruptions_refuse(self):
        for field in ('packed','leaf','local','state','reserved'):
            tile,row,reference,low,leaf=cold_rows();row[field][17]+=1
            with self.subTest(field=field),self.assertRaises(AssertionError):
                C.bind_cold_rows(tile,row,reference,low,leaf)
        tile,row,reference,low,leaf=cold_rows()
        psi=np.frombuffer(tile,dtype='<f4',count=1024,offset=90112).reshape(256,4)
        psi[17,2]=np.nextafter(psi[17,2],np.float32(np.inf))
        with self.assertRaises(AssertionError):C.bind_cold_rows(tile,row,reference,low,leaf)

    def test_nonfinite_full_bank_refuses(self):
        for field in ('accepted','prepared'):
            values=np.zeros((8,256),dtype='<f8');values[7,255]=np.nan
            with self.subTest(field=field),self.assertRaises(AssertionError):
                self.call(owner_rows(2),2,**{field:values})

    def test_double_identity_detects_lost_low_information_and_signed_zero(self):
        self.assertFalse(C.same_f64([1.+2.**-40],[1.]))
        self.assertFalse(C.same_f64([-0.],[0.]))
        self.assertTrue(C.same_f32([1.+2.**-40],[1.]))


if __name__=='__main__':unittest.main()
