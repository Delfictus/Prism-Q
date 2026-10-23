"""Constructed ABI/rejection controls; native precision witness is separate."""
import struct
import unittest
import check_biprism_spinor_owner as S

def fixture(committed=4,step=1):
    raw=bytearray(8*S.SIZE);tiles=bytearray(8*225408)
    for l in range(8):
        base=l*S.SIZE
        struct.pack_into('<4I4Q2I',raw,base,0x31535042,1,l,1,23,1,0x100000+l*225408,0x400000+l*96,committed,step)
        for u in range(192):
            p=(1.,2.**-35,-2.**-40,2.**-44)
            for off in (128,6272,12416):struct.pack_into('<4d',raw,base+off+32*u,*p)
            struct.pack_into('<4f',tiles,l*225408+90112+16*u,*p)
    return raw,tiles

class OwnerControls(unittest.TestCase):
    def test_precise_store_and_projection(self):
        raw,tile=fixture();p=(1.-2.**-45,2.**-35,-2.**-40,2.**-44)
        for off in (128,6272,12416):struct.pack_into('<4d',raw,off,*p)
        owner=S.endpoint(raw,tile,4,1)[0]
        self.assertEqual(owner['accepted'][0],p)
        self.assertNotEqual(p[0],struct.unpack('<f',struct.pack('<f',p[0]))[0])
    def test_partial_or_unpublished_owner_refuses(self):
        raw,tile=fixture()
        with self.assertRaises(ValueError):S.endpoint(raw[:-1],tile,4,1)
        struct.pack_into('<I',raw,12,0)
        with self.assertRaises(ValueError):S.endpoint(raw,tile,4,1)
    def test_projection_stale_commit_and_start_refuse(self):
        for target,off,fmt,value in [('tile',90112,'f',.5),('raw',48,'I',3),('raw',52,'I',2),('raw',6272,'d',.5),('raw',56,'Q',1)]:
            raw,tile=fixture();struct.pack_into('<'+fmt,tile if target=='tile' else raw,off,value)
            with self.subTest(offset=off),self.assertRaises(ValueError):S.endpoint(raw,tile,4,1)
    def test_private_candidate_is_not_accepted_state(self):
        raw,tile=fixture();struct.pack_into('<d',raw,12416,5.)
        self.assertEqual(S.endpoint(raw,tile,4,1)[0]['accepted'][0][0],1.)

if __name__=='__main__':unittest.main()
