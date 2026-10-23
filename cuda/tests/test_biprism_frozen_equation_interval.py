"""Independent Cayley signs, conservative inclusion, and retained EQR2 exclusion."""
import unittest
from fractions import Fraction

import biprism_frozen_equation_interval as I
import check_biprism_c1_refusal as C


def vertex95():
    # Native086 fixed-domain, epoch4, leaf3, vertex95. Source SHA256:
    # 60dcb2c5f572563762c98aa61f921c865a9922b22248738941665a03db837e80
    # The command-line checker independently binds these to the raw record.
    return dict(h=5.1676716804504395, omega=.766565203666687,
                H=(-.5365931664835097,.008404575544527738,0.,0.),
                connection=-.0014583004076241992,
                start=(-.8445186614990234,.5355201363563538,
                       .001051425002515316,.0022989013232290745),
                norm_start=.9999999765050597, norm_outer=1.0000000759267418)


class FrozenIntervalControls(unittest.TestCase):
    def test_real_matrix_matches_complex_equation_with_both_offdiagonals(self):
        start=(.5,-.5,.5,.5)
        H=(-.4,.7,.3,-.2)
        A,b,_=I.cayley_system(start,H,.8,2.3)
        end=tuple(map(float,I.solve(A,b)))
        self.assertLess(C.equation(start,end,H,.8,2.3),1e-14)
        self.assertLess(abs(sum(x*x for x in end)-1.),1e-14)

    def test_exact_fp32_cayley_witness_is_never_excluded(self):
        record=dict(start=(1.,0.,0.,0.), H=(2.,0.,0.,0.),
                    connection=0.,omega=1.,h=1.,norm_start=1.,norm_outer=1.)
        result=I.exclude_frozen(record)
        self.assertEqual(result['centre'],[0,-1,0,0])
        self.assertFalse(result['all_fp32_slices_excluded'])
        witness=[s for s in result['slices'] if s['value']==-1][0]
        self.assertFalse(witness['excluded'])

    def test_all_possible_fp32_pivot_slices_miss_native_norm_interval(self):
        result=I.exclude_frozen(vertex95())
        self.assertEqual(result['pivot'],1)
        self.assertEqual(len(result['slices']),4)
        self.assertTrue(result['all_fp32_slices_excluded'])
        self.assertGreater(min(s['margin'] for s in result['slices']),Fraction(19,10**10))
        self.assertTrue(all(s['vertices']==8 for s in result['slices']))

    def test_norm_owner_changes_are_not_silently_ignored(self):
        record=vertex95()
        record['norm_outer']=1.
        self.assertFalse(I.exclude_frozen(record)['all_fp32_slices_excluded'])

    def test_outside_stated_roundoff_regime_refuses(self):
        for key,value in (('omega',0.),('h',129.),('norm_outer',float('nan'))):
            record=vertex95();record[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                I.exclude_frozen(record)


if __name__=='__main__':
    unittest.main()
