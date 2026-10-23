import struct
import unittest
import tempfile
import textwrap
from pathlib import Path
import check_biprism_nuclear_budget as C

def fixture():
    raw=bytearray(8*C.STRIDE)
    for leaf in range(8):
        h=[C.MAGIC,2,192,12,171,6,1,2,9,3]+[0]*22
        struct.pack_into('<32I',raw,leaf*C.STRIDE+C.OFFSET,*h)
        for i in range(192):struct.pack_into('<9d',raw,leaf*C.STRIDE+C.OFFSET+128+72*i,1./(i+1),2./(i+1),.5,1.,1.,1.,.1,100.,.1)
    return raw

class NuclearBudgetControls(unittest.TestCase):
    def test_reverse_tangent_actions_require_complete_same_model_schedule(self):
        weights=[1e-4]
        for scale in (.1,.1,.1,.1,.3,1./3.,-100.,.1):weights.append(weights[-1]*scale)
        for stage in (1,2):
            for column in (1,15,64):
                rows=[[0.,0.,0.,0.,0.,1.,w,float(stage*1000+800+column),0.] for w in weights]
                for stop in (7,8,9):C.validate_paired_probe_weights(rows[:stop])
                for field,value in ((7,stage*1000+500+column),(7,(3-stage)*1000+800+column),
                                    (7,stage*1000+800),(6,-1e-4),(5,2.)):
                    bad=[r[:] for r in rows];bad[7][field]=value
                    with self.assertRaises(ValueError):C.validate_paired_probe_weights(bad)
                with self.assertRaises(ValueError):C.validate_paired_probe_weights(rows[1:])
                with self.assertRaises(ValueError):C.validate_paired_probe_weights(rows[:6]+rows[7:])
                bad=[r[:] for r in rows];bad[8][6]*=2.
                with self.assertRaises(ValueError):C.validate_paired_probe_weights(bad)

    def test_launcher_checks_every_leaf_of_current_capture_extent(self):
        source=(Path(__file__).parent/'run_biprism_v10_coupled_qc.py').read_text()
        first=source.index('            records=(O/')
        last=source.index('            output=O/',first)
        block=compile(textwrap.dedent(source[first:last]),'<actual native launcher trace checks>','exec')
        raw=bytearray(8*C.COMPLETION_STRIDE);records=bytearray(8*128)
        for leaf in range(8):
            h=[C.COMPLETION_MAGIC,3,1,0,0,0,0,1,0,0]+[0]*22
            h[10]=h[13]=struct.unpack('<I',struct.pack('<f',4.))[0]
            h[11]=16;h[14:16]=struct.unpack('<II',struct.pack('<d',4.))
            struct.pack_into('<32I',raw,leaf*C.COMPLETION_STRIDE+C.OFFSET,*h)
            struct.pack_into('<9d',raw,leaf*C.COMPLETION_STRIDE+C.evaluation_offset(0),0.,0.,0.,0.,0.,0.,0.,16000.,0.)
            struct.pack_into('<4d',raw,leaf*C.COMPLETION_STRIDE+C.arc_history_offset(0),4.,0.,0.,0.)
            struct.pack_into('<I',records,leaf*128+24,1)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);record=p/'test_state.epoch1.conv.bin';record.write_bytes(records)
            def check(candidate):
                result={'native_nuclear_work_last_substep':[]}
                exec(block,dict(O=p,tag='test',epoch=1,numerical=candidate,NB=C,struct=struct,receipt=result))
                return result
            self.assertEqual(len(check(raw)['native_nuclear_work_last_substep']),8)
            # A single late leaf must not escape checks due to its extent.
            base=7*C.COMPLETION_STRIDE
            for offset,fmt,value in [(C.OFFSET+44,'I',1),(C.evaluation_offset(0),'d',6e-9),
                                     (C.evaluation_offset(0)+8,'d',2e-6),
                                     (C.evaluation_offset(0),'d',float('nan'))]:
                bad=raw[:];struct.pack_into('<'+fmt,bad,base+offset,value)
                with self.assertRaises((AssertionError,ValueError)):check(bad)
            struct.pack_into('<I',records,7*128+24,2);record.write_bytes(records)
            with self.assertRaises(AssertionError):check(raw)

    def test_terminal_tangent_is_bound_to_preceding_probe_and_full_unpublished_row(self):
        h=[C.COMPLETION_MAGIC,1,1402]+[0]*8+[1]
        rows=[[0.]*9 for _ in range(h[2])]
        rows[-2][7]=1763.;rows[-1]=[float('nan')]*9
        args=(64,2,1,1400,.0022875092822808927)
        got=C.terminal_tangent_refusal(h,rows,*args)
        self.assertFalse(got['current_C4_candidate'])
        self.assertEqual(got['evaluated_probe'],1400)
        for index,value in [(0,0.),(1,1.),(2,2.)]:
            bad=[r[:] for r in rows];bad[-1][index]=value
            with self.assertRaises(ValueError):C.terminal_tangent_refusal(h,bad,*args)
        for phase in (1762.,1263.,2763.):
            bad=[r[:] for r in rows];bad[-2][7]=phase
            with self.assertRaises(ValueError):C.terminal_tangent_refusal(h,bad,*args)
        for j,value in [(0,63),(1,1),(2,0),(3,1401),(4,1e-6),(4,float('nan'))]:
            bad=list(args);bad[j]=value
            with self.assertRaises(ValueError):C.terminal_tangent_refusal(h,rows,*bad)
    def test_terminal_tangent_breakdown_requires_its_measured_column_norm(self):
        h=[C.COMPLETION_MAGIC,1,1802]+[0]*8+[1]
        rows=[[0.]*9 for _ in range(h[2])]
        rows[-2][7]=1714.;rows[-1]=[float('nan')]*9
        args=(14,2,1,1800,.011880528385719786)
        got=C.terminal_tangent_refusal(h,rows,*args,6.605630830073122e-29)
        self.assertFalse(got['current_C4_candidate'])
        self.assertEqual(got['reason'],'derivative_breakdown')
        self.assertEqual(got['column'],14)
        for norm in (None,-1.,float('nan'),float('inf'),1.00001e-28):
            with self.assertRaises(ValueError):C.terminal_tangent_refusal(h,rows,*args,norm)
        for phase in (1713.,1763.,2714.,1214.):
            bad=[r[:] for r in rows];bad[-2][7]=phase
            with self.assertRaises(ValueError):C.terminal_tangent_refusal(h,bad,*args,1e-29)

    def test_real_private_stage_rounds_at_the_cuda_float_store(self):
        raw=fixture()
        for magic in (C.BOUNDED_CONTINUATION_MAGIC,C.KRYLOV64_MAGIC):
            struct.pack_into('<I',raw,C.OFFSET,magic)
            struct.pack_into('<III',raw,C.OFFSET+40,1084579217,6,0)
            h,_=C.read_trace(raw,0)
            self.assertEqual(C.live_h(h),1.9378769397735596)
            # Native135/136 stage6 is not the unrounded Python product.
            self.assertNotEqual(C.live_h(h),5.1676716804504395*(6./16.))
            self.assertFalse(C.original_target_reached(h))
            struct.pack_into('<I',raw,C.OFFSET+44,16)
            h,_=C.read_trace(raw,0)
            self.assertEqual(C.live_h(h),5.1676716804504395)
            self.assertTrue(C.original_target_reached(h))
    def test_krylov64_extent_keeps_previous_physical_prefix_and_history_version(self):
        raw=fixture();struct.pack_into('<I',raw,C.OFFSET,C.KRYLOV64_MAGIC)
        struct.pack_into('<fII',raw,C.OFFSET+40,4.,16,0)
        h,_=C.read_trace(raw,0);self.assertTrue(C.original_target_reached(h))
        old=bytearray(8*510720)
        for leaf in range(8):
            old[leaf*510720+496768:(leaf+1)*510720]=raw[leaf*C.STRIDE+C.OFFSET:(leaf+1)*C.STRIDE]
        with self.assertRaises(ValueError):C.read_trace(old,0)
        for leaf in range(8):struct.pack_into('<I',old,leaf*510720+496768,C.BOUNDED_CONTINUATION_MAGIC)
        self.assertEqual(C.read_trace(old,0)[1],C.read_trace(raw,0)[1])
    def test_continuation_cannot_claim_original_target_early(self):
        raw=fixture()
        for stage in range(5):
            struct.pack_into('<I',raw,C.OFFSET,C.CONTINUATION_MAGIC)
            struct.pack_into('<fII',raw,C.OFFSET+40,4.,stage,0)
            h,rows=C.read_trace(raw,0)
            self.assertEqual(C.live_h(h),[4.,.5,1.,2.,4.][stage])
            self.assertEqual(C.original_target_reached(h),stage in (0,4))
        struct.pack_into('<I',raw,C.OFFSET+48,1)
        self.assertFalse(C.original_target_reached(C.read_trace(raw,0)[0]))
    def test_bounded_continuation_preserves_old_schema_and_final_h(self):
        raw=fixture();struct.pack_into('<I',raw,C.OFFSET,C.BOUNDED_CONTINUATION_MAGIC)
        for stage in range(17):
            struct.pack_into('<fII',raw,C.OFFSET+40,4.,stage,0)
            h,rows=C.read_trace(raw,0)
            self.assertEqual(C.live_h(h),stage/4. if stage else 4.)
            self.assertEqual(C.original_target_reached(h),stage in (0,16))
    def test_every_evaluation_retained_after_old_48_boundary(self):
        raw=fixture();h,rows=C.read_trace(raw,7)
        self.assertEqual(len(rows),192);self.assertEqual(rows[191][0],1./192)
        self.assertEqual(h[8]+h[9],h[3]);self.assertNotEqual(rows[0],rows[48])
    def test_accounting_identity_and_bounds_reject(self):
        for field,value in [(2,193),(3,49),(4,170),(8,10),(10,1)]:
            raw=fixture();struct.pack_into('<I',raw,C.OFFSET+4*field,value)
            with self.assertRaises(ValueError):C.read_trace(raw,0)
    def test_unpublished_final_c4_is_not_an_energy_balance(self):
        raw=fixture();struct.pack_into('<9d',raw,C.OFFSET+128+72*191,*([float('nan')]*9))
        _,rows=C.read_trace(raw,0);self.assertNotEqual(rows[-1][1],rows[-1][1])
        struct.pack_into('<9d',raw,C.OFFSET+128,*([float('nan')]*9))
        with self.assertRaises(ValueError):C.read_trace(raw,0)
    def test_next_leaf_and_truncated_extent_do_not_alias(self):
        raw=fixture();struct.pack_into('<d',raw,C.STRIDE+C.OFFSET+128,123.)
        self.assertEqual(C.read_trace(raw,0)[1][0][0],1.)
        self.assertEqual(C.read_trace(raw,1)[1][0][0],123.)
        with self.assertRaises(ValueError):C.read_trace(raw[:-1],0)
