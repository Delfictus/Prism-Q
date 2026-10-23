"""Retained exact-byte frozen witness regression, separate from GPU execution."""
import struct
import unittest

from check_biprism_spinor_witness import SIZE, inspect

INPUT = bytes.fromhex(
    '000000004c06ebbf00000020fb22e13f00000000ff39513f0000004025d5623f'
    '62b8a96ec52be1bf27d7cd086b36813f0000000000000000584abe238ee457bf'
    '000000c0b387e83f00000020b2ab14405ea561140000f03f')
ACCEPTED = bytes.fromhex('00279dd3ec3072bf5a393681e1ffefbf3d197ee38e0569bf606685787307133f')


def fixture():
    raw = bytearray(SIZE)
    struct.pack_into('<I', raw, 12, 1)
    struct.pack_into('<2I', raw, 48, 4, 1)
    raw[6272+32*95:6272+32*96] = INPUT[:32]
    for offset in (128, 12416):
        raw[offset+32*95:offset+32*96] = ACCEPTED
    return raw


class SpinorWitnessControls(unittest.TestCase):
    def test_retained_fp64_bits_clear_all_frozen_gates(self):
        _, equation, norms, tolerance = inspect(INPUT, fixture())
        self.assertLess(float(equation), 3e-16)
        self.assertGreater(min(float(tolerance-x) for x in norms), 5.7e-10)

    def test_fp32_round_trip_rejects_the_original_excluded_case(self):
        raw = fixture()
        fp32 = struct.unpack('<4f', struct.pack('<4f', *struct.unpack('<4d', ACCEPTED)))
        for offset in (128, 12416):
            struct.pack_into('<4d', raw, offset+32*95, *fp32)
        with self.assertRaises(AssertionError):
            inspect(INPUT, raw)

    def test_stored_accepted_state_must_equal_evaluated_candidate(self):
        raw = fixture()
        raw[128+32*95] ^= 1
        with self.assertRaises(AssertionError):
            inspect(INPUT, raw)

    def test_truncated_old_owner_layout_is_rejected(self):
        with self.assertRaises(AssertionError):
            inspect(INPUT, fixture()[:18560])


if __name__ == '__main__':
    unittest.main()
