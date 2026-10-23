#!/usr/bin/env python3
"""Conservative-work bookkeeping and stage ownership, no native acceptance."""
from fractions import Fraction as Q
import random
import struct
import unittest
import check_biprism_c1_refusal as C


class WorkPartitionControls(unittest.TestCase):
    def test_internal_electronic_transfer_is_counted_once(self):
        rng=random.Random(41)
        for _ in range(100):
            dtmid,dtmetric,du,du_elec,bath,wcons,wberry,projection,rounding,pair=[
                Q(rng.randrange(-1000,1000),127) for _ in range(10)]
            used=wcons+wberry-projection+rounding
            mid=dtmid-used
            coord=dtmetric+du-du_elec+bath+wcons-projection
            joint=du_elec+wberry+pair
            direct=dtmetric+dtmid+du+bath+pair
            self.assertEqual(coord+mid+joint+rounding,direct)
            if du_elec:
                self.assertNotEqual(coord+mid+joint+rounding+du_elec,direct)

    def test_c1_refusal_never_relabels_old_c4_as_current(self):
        raw=bytearray(8*C.STRIDE)
        for leaf in range(8):
            struct.pack_into('<I',raw,leaf*C.STRIDE+C.PREFIX,C.RRF3)
            struct.pack_into('<d',raw,leaf*C.STRIDE+C.SCALAR+27*8,.25)
        lines=C.work_energy_lines({'numerical':raw})
        self.assertEqual(sum('current_C4_balance=EVIDENCE_NOT_FOUND' in x for x in lines),8)
        self.assertFalse(any(x.startswith('all8_C4_aggregate_partition') for x in lines))

    def test_large_refused_energy_is_not_a_host_recomposition_failure(self):
        raw=bytearray(8*C.STRIDE)
        for leaf in range(8):
            a=[0.]*32
            a[0]=2427.437191207;a[1]=71.123987001;a[2]=-1742.512301
            a[3]=-.04101777;a[5]=.00004301;a[10]=.003491331
            a[11]=2500.991987+leaf*.01497;a[12]=-.125870901
            a[16]=2495.910892+leaf*.097511;a[17]=.000019337
            a[20]=.000017228;a[27]=19959.9766
            struct.pack_into('<32d',raw,leaf*C.STRIDE+C.SCALAR,*a)
        lines=C.work_energy_lines({'numerical':raw})
        self.assertTrue(any(x.startswith('all8_C4_aggregate_partition total=199') for x in lines))
        self.assertTrue(any(x.startswith('host_partition_recomposition_error=') and
                            'scientific_acceptance=false' in x for x in lines))
        self.assertIn('FULL_TRAJECTORY_ACCEPTED=false',lines)

    def test_native168_cancellation_retains_leaf_operation_error_bound(self):
        # Exact retained scalar rows; no GPU acceptance can follow from this test.
        rows=[[228.03282552866185, -24.02136281441926, 244.89465353327182, 0.07434310302931257,
          0.5269204798151624, 0.0, 2.1611041762851899e-07, 0.003802689553348475, 5.4197066431522636e-08,
          -7.541957843765706e-17, 5.4069490860404325e-12, -1.1801987880900866e-14, -0.7332237704031286,
          7.140619058138796e-13, 0.0, 1.1121836585026301e-07, -0.6588806673685497, 0.631956551359672,
          0.6319565300704149, -0.026924180358874762, 0.6588807104292896, 1.1102230246251565e-16, 0.0, 0.0,
          3.219646771412954e-15, 0.0, 0.0, 1.1121836585026301e-07, 7.140619058138796e-13,
          1.1220034397415611e-14, -2.7028823623709286e-11, 0.0],
         [46.61245358651985, -0.2742250709854447, 39.537741731591595, 0.07046174646221764,
          0.2816985147030309, 0.0, 4.829016877635153e-08, 0.0016864730512878518, 7.1607509047501e-08,
          -2.294647949953861e-15, 5.999904355746317e-12, -1.1510778419502959e-12, -0.6892416274340576,
          7.087415866479322e-13, 0.0, 1.1121836585026301e-07, -0.6187798809668834, 0.6316267359845836,
          0.6317916436721279, 0.013011753018500377, 0.6187798906536275, 1.1102230246251565e-16, 0.0, 0.0,
          2.886579864025407e-15, 0.0, 0.0, 1.1121836585026301e-07, 7.087415866479322e-13,
          8.464297592106928e-15, -2.7028823623709286e-11, 0.0],
         [46.626007155754934, -0.2744817660348942, 39.54647906170946, 0.07035351714508081,
          0.2817207437519038, 0.0, 5.289734872349072e-08, 0.001875343100175839, 7.544539925762576e-08,
          3.411919527930226e-15, 2.6826890900368313e-12, -3.175880264990859e-12, -0.7014943597223464,
          7.131473953394091e-13, 0.0, 1.1121836585026301e-07, -0.6311408425776435, 0.6315281819630183,
          0.631577458973801, 0.00043660669114620827, 0.6311408522826547, 1.1102230246251565e-16, 0.0, 0.0,
          3.2156965978913377e-09, 0.0, 0.0, 1.1121836585026301e-07, 7.131473953394091e-13,
          -5.119432457793713e-15, -2.7028823623709286e-11, 0.0],
         [46.626414860886854, -0.2744896087741471, 39.54674249634762, 0.0703494516609158,
          0.2817214202146046, 0.0, 5.3201715204197244e-08, 0.002118598635263478, 9.3067496800749e-08,
          -1.0864401715360072e-15, 5.085276752368245e-12, -7.929224814683577e-14, -0.7018627246916367,
          7.088869482497686e-13, 0.0, 1.1121836585026301e-07, -0.6315132730258597, 0.6315247341890182,
          0.6315264580760183, 1.317531408169552e-05, 0.6315132827619365, 1.1102230246251565e-16, 0.0, 0.0,
          3.1086244689504383e-15, 0.0, 0.0, 1.1121836585026301e-07, 7.088869482497686e-13,
          2.4287450670347726e-14, -2.7028823623709286e-11, 0.0],
         [46.62642279785541, -0.27448972041645936, 39.546746282623246, 0.07034966350585953,
          0.2817214207848013, 0.0, 5.3210669648251496e-08, 0.00016425388785166247, 7.757064916709052e-08,
          -4.2302903731533305e-15, -2.102716546094057e-12, 1.7207733726633582e-10, -0.7018733096192591,
          7.138177702257178e-13, 0.0, 1.1121836585026301e-07, -0.6315236459437585, 0.6315246153999112,
          0.6315246747944647, 1.0189503289989247e-06, 0.6315236558441357, 1.1102230246251565e-16, 0.0, 0.0,
          3.552713678800501e-15, 0.0, 0.0, 1.1121836585026301e-07, 7.138177702257178e-13,
          1.0871460249057749e-14, -2.7028823623709286e-11, 0.0],
         [46.62641298916539, -0.27448952590711784, 39.546740681191366, 0.07034975396813964,
          0.28172141658154376, 0.0, 5.3199885783128215e-08, 5.689875722273384e-05, 2.6039997489786134e-08,
          6.273090242143613e-17, 5.950007609463548e-12, -4.2064884241955043e-13, -0.7018628081214047,
          7.087071441884706e-13, 0.0, 1.1121836585026301e-07, -0.6315130541476179, 0.6315280272000172,
          0.6315263212999642, 1.3257416025998379e-05, 0.6315130638839381, 1.1102230246251565e-16, 0.0, 0.0,
          2.886579864025407e-15, 0.0, 0.0, 1.1121836585026301e-07, 7.087071441884706e-13,
          3.2702751906318525e-14, -2.7028823623709286e-11, 0.0],
         [46.626006773769426, -0.27448178465770595, 39.5464793985035, 0.07035288494278145,
          0.28172075601896285, 0.0, 5.2896893976139836e-08, 0.0017589086288411946, 7.00913165508395e-08,
          -4.2191301809315204e-15, 5.719011975556896e-12, -6.786119838697111e-14, -0.7014937297826691,
          7.129474880898126e-13, 0.0, 1.1121836585026301e-07, -0.6311408448341282, 0.6316266493716916,
          0.6315773382858545, 0.00043648374056065754, 0.6311408545452938, 1.1102230246251565e-16, 0.0, 0.0,
          3.1086244689504383e-15, 0.0, 0.0, 1.1121836585026301e-07, 7.129474880898126e-13,
          6.417857097694592e-15, -2.7028823623709286e-11, 0.0],
         [46.61245459671823, -0.2742251423485649, 39.53774271546266, 0.07046165374995733,
          0.28169852652570726, 0.0, 4.82905306853354e-08, 0.001996001612102953, 8.03263088731974e-08,
          1.834438800568181e-15, 8.642248949344527e-13, -3.5858137533218275e-13, -0.6892413379168829,
          7.084513413172517e-13, 0.0, 1.1121836585026301e-07, -0.6187796841662632, 0.6319565087811578,
          0.6317915790764247, 0.013011885228230833, 0.6187796938481939, 1.1102230246251565e-16, 0.0, 0.0,
          2.886579864025407e-15, 0.0, 0.0, 1.1121836585026301e-07, 7.084513413172517e-13,
          9.997112712715984e-15, -2.7028823623709286e-11, 0.0]]
        raw=bytearray(8*C.STRIDE)
        for leaf,a in enumerate(rows):
            struct.pack_into("<32d",raw,leaf*C.STRIDE+C.SCALAR,*a)
        lines=C.work_energy_lines({"numerical":raw})
        self.assertTrue(any(x.startswith("all8_C4_aggregate_partition total=1.112183") for x in lines))
        self.assertIn("FULL_TRAJECTORY_ACCEPTED=false",lines)
        self.assertTrue(any("scientific_acceptance=false" in x for x in lines))

    def test_final_dec_refusal_without_c1_record_excludes_stale_c4_scalars(self):
        stride=106752;raw=bytearray(8*stride)
        for leaf in range(8):
            base=leaf*stride
            struct.pack_into('<4I',raw,base+40704,0x4e485034,1,2,0)
            struct.pack_into('<3d',raw,base+40704+16,.1,6138.,.4)
            struct.pack_into('<3d',raw,base+40704+40,*((float('nan'),)*3))
            # Finite values and an unrecognized diagnostic prefix must not
            # make previous-iteration C4 aggregates current evidence.
            struct.pack_into('<32d',raw,base+C.SCALAR,*([.25]*32))
        lines=C.work_energy_lines({'numerical':raw},stride)
        self.assertEqual(sum('current_C4_balance=EVIDENCE_NOT_FOUND' in x for x in lines),8)
        self.assertFalse(any('C4_exact_aggregate_partition' in x for x in lines))
        self.assertFalse(any(x.startswith('all8_C4_aggregate_partition') for x in lines))
        self.assertEqual(sum('nuclear_picard=0 raw_residual=' in x for x in lines),8)
        self.assertEqual(sum('nuclear_picard=1 C4_evaluated=false' in x for x in lines),8)

    def test_decoder_uses_pinned_source_iteration_bound(self):
        self.assertEqual(C.compiled_picard_limit('#define MAX_VARIATIONAL_PICARD_ITERS 12\n'),12)
        self.assertEqual(C.compiled_picard_limit('#define MAX_VARIATIONAL_PICARD_ITERS 24\n'),24)
        for s in ('', '#define MAX_VARIATIONAL_PICARD_ITERS x\n',
                  '#define MAX_VARIATIONAL_PICARD_ITERS 12\n#define MAX_VARIATIONAL_PICARD_ITERS 24\n'):
            with self.assertRaises(ValueError):C.compiled_picard_limit(s)

    def test_nuclear_history_cannot_alias_an_active_phase(self):
        raw=bytearray(8*C.STRIDE);tiles=bytearray(8*C.TILE)
        start=C.SCALAR-75*4
        struct.pack_into('<I2f',raw,start,0x4e485031,2.,2.)
        struct.pack_into('<6f',raw,start+12,1.e-3,.1,3.,1.e-5,.001,4.)
        struct.pack_into('<I',tiles,225284,1461)
        files={'numerical':raw,'tiles':tiles}
        lines=C.nuclear_history_lines(files)
        self.assertEqual(len(lines),3)
        self.assertIn('nuclear_picard=1',lines[-1])
        struct.pack_into('<I',tiles,225284,1462)
        with self.assertRaisesRegex(ValueError,'overlaps'):C.nuclear_history_lines(files)
        struct.pack_into('<I',tiles,225284,1461)
        for count in (-1.,1.5,25.,float('nan')):
            struct.pack_into('<f',raw,start+8,count)
            with self.assertRaises(ValueError):C.nuclear_history_lines(files)

    def test_rolling_history_preserves_late_iteration_chronology(self):
        raw=bytearray(8*C.STRIDE);tiles=bytearray(8*C.TILE)
        start=C.SCALAR-75*4
        struct.pack_into('<I2f',raw,start,0x4e485032,1.,48.)
        struct.pack_into('<I',tiles,225284,1461)
        for p in range(48):struct.pack_into('<3f',raw,start+12+12*(p%24),1e-8,float(p),float(-p))
        lines=C.nuclear_history_lines({'numerical':raw,'tiles':tiles})
        self.assertEqual(len(lines),25)
        self.assertIn('nuclear_picard=24',lines[1]);self.assertIn('global_energy=24',lines[1])
        self.assertIn('nuclear_picard=47',lines[-1]);self.assertIn('alpha_used=-47',lines[-1])

    def test_nhp3_skips_unevaluated_c4_slots_without_relabeling_stale_values(self):
        raw=bytearray(8*C.STRIDE);tiles=bytearray(8*C.TILE)
        start=C.SCALAR-75*4
        struct.pack_into('<I2f',raw,start,0x4e485033,1.,4.)
        struct.pack_into('<I',tiles,225284,1461)
        for p in range(4):
            values=(float('nan'),)*3 if p%2 else (1e-8,float(p),float(-p))
            struct.pack_into('<3f',raw,start+12+12*p,*values)
        lines=C.nuclear_history_lines({'numerical':raw,'tiles':tiles})
        self.assertIn('attempted_nuclear_iterations=4',lines[0])
        self.assertIn('C4_evaluated=false',lines[2]);self.assertIn('C4_evaluated=false',lines[4])
        self.assertIn('global_energy=2',lines[3])
        struct.pack_into('<f',raw,start+12+12,0.)
        with self.assertRaisesRegex(ValueError,'nonfinite'):C.nuclear_history_lines({'numerical':raw,'tiles':tiles})

    def test_complete_fp64_history_retains_first_and_last_iterations(self):
        stride=97536;raw=bytearray(8*stride);start=40704
        struct.pack_into('<4I',raw,start,0x4e485034,2,48,0)
        for p in range(48):
            values=(float('nan'),)*3 if p==7 else (1e-12*(p+1),float(p),float(-p))
            struct.pack_into('<3d',raw,start+16+24*p,*values)
        lines=C.nuclear_history_lines({'numerical':raw},stride)
        self.assertEqual(len(lines),49)
        self.assertIn('complete=true',lines[0])
        self.assertIn('nuclear_picard=0 ',lines[1])
        self.assertIn('C4_evaluated=false',lines[8])
        self.assertIn('nuclear_picard=47 ',lines[-1])
        self.assertIn('global_energy=47',lines[-1])
        struct.pack_into('<I',raw,start+8,49)
        with self.assertRaisesRegex(ValueError,'complete nuclear history'):
            C.nuclear_history_lines({'numerical':raw},stride)


if __name__=='__main__':unittest.main(verbosity=2)
