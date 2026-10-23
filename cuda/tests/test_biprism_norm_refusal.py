#!/usr/bin/env python3
"""CPU oracle controls; these fixtures are not actual native refusal captures."""
import importlib.util
from pathlib import Path
import struct
import unittest
from fractions import Fraction

S = importlib.util.spec_from_file_location("norm_refusal", Path(__file__).with_name("check_biprism_norm_refusal.py"))
C = importlib.util.module_from_spec(S)
S.loader.exec_module(C)


def fixture():
    raw = bytearray(C.LEAVES * C.STRIDE)
    for leaf in range(C.LEAVES):
        b = leaf * C.STRIDE
        struct.pack_into("<d", raw, b + 30720 + 29 * 8, C.TAG)
        for i in range(C.CORE):
            for off, value in zip((15360, 16896, 18432, 19968, 21504, 23040), (1., 0., 0., 0., 1., 1.)):
                struct.pack_into("<d", raw, b + off + i * 8, value)
            struct.pack_into("<f", raw, b + 9216 + i * 4, 1.)
            struct.pack_into("<I", raw, b + 8448 + i * 4, 1)
    return raw


def set_capture_row(raw, leaf, vertex, trial, start, outer, admitted):
    b = leaf * C.STRIDE
    for off, value in zip((15360, 16896, 18432, 19968, 21504, 23040), (*trial, C.norm(start), outer)):
        struct.pack_into("<d", raw, b + off + vertex * 8, value)
    for off, value in zip((9216, 9984, 10752, 11520), start):
        struct.pack_into("<f", raw, b + off + vertex * 4, value)
    struct.pack_into("<I", raw, b + 8448 + vertex * 4, int(admitted))


def actual_008_values():
    # Exact retained native008 NRF1 leaf5/vertex107. These literal regression
    # values do not replace binding a complete actual readback in the CLI.
    trial = tuple(float.fromhex(x) for x in (
        '-0x1.f66f21bf3cdebp-1', '0x1.893bffb13b9e4p-3',
        '0x1.8a076814d0c0cp-7', '-0x1.3a49eadbd8efep-9'))
    start = tuple(struct.unpack('<f', struct.pack('<I', x))[0]
                  for x in (0x3f4d4397, 0x3f18f962, 0x3a9c588a, 0xbb9ceb64))
    return trial, start


class Controls(unittest.TestCase):
    def test_nrf3_includes_pivot_and_absolute_one(self):
        raw = fixture()
        for leaf in range(C.LEAVES):
            struct.pack_into('<d', raw, leaf*C.STRIDE+30720+29*8, C.TAG_V3)
        trial = tuple(float.fromhex(x) for x in (
            '-0x1.c411072057000p-13', '-0x1.ff95883334782p-1',
            '-0x1.280883dffe94fp-5', '0x1.241ff33e0fbb4p-6'))
        start = tuple(struct.unpack('<f', struct.pack('<I', x))[0]
                      for x in (0xbf1f5b95,0x3f4850f7,0xbbfedc50,0xbc4fb373))
        set_capture_row(raw,0,133,trial,start,1.0000000058484633,True)
        # These artificial incompatible targets exercise reader refusal only;
        # native input admission would reject this nonunit physical state.
        set_capture_row(raw,0,0,(2.,0.,0.,0.),(2.,0.,0.,0.),4.,False)
        text = C.inspect(raw,2)
        self.assertIn('INITIALIZER_MODE=triple_bound_',text)
        self.assertIn('ALREADY_ADMITTED_BY_NATIVE_PIVOT_COMPLETION=1',text)
        self.assertIn('CAPTURED_INITIALIZER_FAILURES=1',text)
        self.assertIn('ABSOLUTE_ONE_NORM_LIMIT=1e-7 unchanged',text)

    def test_nrf3_cannot_reinterpret_old_failed_vote(self):
        raw = fixture()
        for leaf in range(C.LEAVES):
            struct.pack_into('<d',raw,leaf*C.STRIDE+30720+29*8,C.TAG_V3)
        trial,start=actual_008_values()
        set_capture_row(raw,5,107,trial,start,1.,False)
        with self.assertRaisesRegex(ValueError,'NRF3 initializer vote disagrees'):
            C.inspect(raw,2)

    def test_actual_008_finite_shell_candidate(self):
        trial, start = actual_008_values()
        self.assertIsNone(C.select(trial, C.norm(start), 1., 0)[0])
        candidate, error, count = C.select_shell1_source_order(trial, C.norm(start), 1.)
        self.assertEqual(tuple(C.bits(x) for x in candidate), (0xbf7b3790, 0x3e449e01, 0x3c4503b4, 0xbb1d24f5))
        self.assertEqual(count, 16)
        self.assertLess(error, 3.01e-15)
        self.assertLessEqual(abs(C.exact_norm(candidate) - C.exact_norm(start)), Fraction(1, 10_000_000))
        self.assertLessEqual(abs(C.exact_norm(candidate) - 1), Fraction(1, 10_000_000))

    def test_nrf2_vote_includes_already_successful_shell_fallback(self):
        raw = fixture()
        for leaf in range(C.LEAVES):
            struct.pack_into('<d', raw, leaf * C.STRIDE + 30720 + 29 * 8, C.TAG_V2)
        trial, start = actual_008_values()
        set_capture_row(raw, 5, 107, trial, start, 1., True)
        # A distinct impossible target exercises genuine unresolved refusal.
        set_capture_row(raw, 0, 0, (1., 0., 0., 0.), (1., 0., 0., 0.), 2., False)
        text = C.inspect(raw, 2)
        self.assertIn('ORIGINAL_16_CORNER_FAILURES=2', text)
        self.assertIn('ALREADY_ADMITTED_BY_NATIVE_SHELL1=1', text)
        self.assertIn('CAPTURED_INITIALIZER_FAILURES=1', text)
        self.assertIn('UNRESOLVED_WITHIN_SEARCH=1', text)

    def test_nrf2_wrong_failed_vote_is_refused(self):
        raw = fixture()
        for leaf in range(C.LEAVES):
            struct.pack_into('<d', raw, leaf * C.STRIDE + 30720 + 29 * 8, C.TAG_V2)
        trial, start = actual_008_values()
        set_capture_row(raw, 5, 107, trial, start, 1., False)
        with self.assertRaisesRegex(ValueError, 'NRF2 initializer vote disagrees'):
            C.inspect(raw, 2)

    def test_schema_mixture_refused(self):
        raw = fixture()
        struct.pack_into('<d', raw, C.STRIDE + 30720 + 29 * 8, C.TAG_V2)
        with self.assertRaisesRegex(ValueError, 'inconsistent NRF1/NRF2'):
            C.parse(raw)

    def test_existing_corners_can_miss_nearby_joint_norm_candidate(self):
        start = (1., 0.0003098107408732176, 0., 0.)
        trial = (-0.9694920135355538, -0.24512309494103698, 0., 0.)
        n = C.norm(start)
        self.assertLess(abs(C.norm(trial) - n), 5e-16)
        self.assertIsNone(C.select(trial, n, 1., 0)[0])
        candidate, error, count = C.select(trial, n, 1., 1)
        self.assertIsNotNone(candidate)
        self.assertGreater(count, 0)
        self.assertEqual(candidate[2:], (0., 0.))
        self.assertLess(error, 5e-16)
        self.assertLessEqual(abs(C.exact_norm(candidate) - C.exact_norm(start)), Fraction(1, 10_000_000))
        self.assertLessEqual(abs(C.exact_norm(candidate) - 1), Fraction(1, 10_000_000))

    def test_complete_capture_identity_and_original_votes(self):
        raw = fixture()
        self.assertEqual(len(C.parse(raw)), 1536)
        out = C.inspect(raw, 1)
        self.assertIn("ORIGINAL_16_CORNER_FAILURES=0", out)
        self.assertIn("COUPLED_EQUATION_WORK_ENERGY_NOT_EVALUATED=true", out)

    def test_missing_leaf_refused(self):
        with self.assertRaisesRegex(ValueError, "eight complete"):
            C.parse(fixture()[:-C.STRIDE])

    def test_wrong_tag_refused(self):
        raw = fixture()
        struct.pack_into("<d", raw, 30720 + 29 * 8, 0.)
        with self.assertRaisesRegex(ValueError, "NRF1"):
            C.parse(raw)

    def test_stale_source_norm_refused(self):
        raw = fixture()
        struct.pack_into("<d", raw, 21504, 1.00000001)
        with self.assertRaisesRegex(ValueError, "start norm mismatch"):
            C.parse(raw)

    def test_bad_flag_refused(self):
        raw = fixture()
        struct.pack_into("<I", raw, 8448, 2)
        with self.assertRaisesRegex(ValueError, "original vote"):
            C.parse(raw)

    def test_wrong_original_vote_refused(self):
        raw = fixture()
        struct.pack_into("<I", raw, 8448, 0)
        with self.assertRaisesRegex(ValueError, "vote disagrees"):
            C.inspect(raw, 1)

    def test_nonfinite_trial_refused(self):
        raw = fixture()
        struct.pack_into("<d", raw, 15360, float("nan"))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            C.parse(raw)

    def test_expansion_never_relaxes_incompatible_targets(self):
        for shell in range(3):
            self.assertIsNone(C.select((1., 0., 0., 0.), 1., 1.000001, shell)[0])


if __name__ == "__main__":
    print("CONTROL_SCOPE=CPU mathematical and parser controls; no native or trajectory acceptance")
    unittest.main(verbosity=2)
