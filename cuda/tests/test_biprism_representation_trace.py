#!/usr/bin/env python3
"""Parser and prediction controls; constructed traces are not native evidence."""
import importlib.util
import math
from pathlib import Path
import struct
import unittest

SPEC = importlib.util.spec_from_file_location("trace", Path(__file__).with_name("check_biprism_representation_trace.py"))
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


def fixture(last=39, stage=9, version=2):
    stride = C.PRECISE_STRIDE if version == 3 else C.STRIDE
    raw = bytearray(C.LEAVES * stride)
    rounds = list(range(last + 1)) if last <= 16 else [0] + list(range(last - 15, last + 1))
    for leaf in range(8):
        off = leaf * stride + C.OFFSET
        struct.pack_into("<4I", raw, off, {1:C.MAGIC_V1,2:C.MAGIC_V2,3:C.MAGIC_V3}[version],
                         stage, len(rounds), last + (stage == 6))
        struct.pack_into("<34d", raw, off + 16, *([math.nan] * 34))
        struct.pack_into("<17I", raw, off + 288, *([C.UINT_MAX] * 17))
        if version >= 2:
            struct.pack_into("<17I", raw, off + 356, *([C.UINT_MAX] * 17))
        for r in rounds:
            slot = r if version == 1 else (0 if r == 0 else 1 + ((r - 1) & 15))
            error = -.001 / (r + 1)
            best = .001 / (r + 2)
            winner = (r % 8) * 768 + r % 768
            if r == last and stage != 6:
                score = math.nan if stage in (7, 11, 12) else abs(error)
                winner = C.UINT_MAX
            else:
                score = best if leaf == r % 8 else abs(error)
            struct.pack_into("<d", raw, off + 16 + 8 * slot, error)
            struct.pack_into("<d", raw, off + 152 + 8 * slot, score)
            struct.pack_into("<I", raw, off + 288 + 4 * slot, winner)
            if version >= 2:
                struct.pack_into("<I", raw, off + 356 + 4 * slot, r)
    return raw


def repair_row(raw, round_id, selected=4*192*4+4*17):
    slot = 0 if round_id == 0 else 1+((round_id-1)&15)
    for leaf in range(8):
        offset = leaf*C.PRECISE_STRIDE+C.OFFSET
        struct.pack_into('<d',raw,offset+152+8*slot,math.nan)
        struct.pack_into('<I',raw,offset+288+4*slot,C.REPAIR_BIT|selected)


class Controls(unittest.TestCase):
    def test_original_consecutive_trace(self):
        text = C.inspect(fixture(16, 10, 1))
        self.assertIn("COMPARED_ADJACENT_UPDATES=16", text)
        self.assertIn("MAX_ACTUAL_VS_PREDICTED_ERROR=0", text)
        self.assertIn("FULL_SEARCH_HISTORY_RETAINED=true", text)

    def test_sparse_rounds_never_compared_across_gap(self):
        text = C.inspect(fixture())
        self.assertIn("COMPARED_ADJACENT_UPDATES=15", text)
        self.assertIn("UNRETAINED_ROUND_INTERVALS=[(1, 23)]", text)
        self.assertIn("MAX_ACTUAL_VS_PREDICTED_ERROR=0", text)
        self.assertIn("FULL_SEARCH_HISTORY_RETAINED=false", text)

    def test_final_full_recheck_at_6144(self):
        text = C.inspect(fixture(6144, 10))
        self.assertIn("TERMINAL_ROUND=6144 LAST_EVALUATED_ROUND=6144", text)
        self.assertIn("UNRETAINED_ROUND_INTERVALS=[(1, 6128)]", text)

    def test_repair_rows_have_no_linear_energy_prediction(self):
        raw = fixture(3, 11, 3)
        repair_row(raw, 1)
        text = C.inspect(raw)
        self.assertIn('EQUATION_REPAIR_ROWS_RETAINED=1', text)
        self.assertIn('COMPARED_ADJACENT_UPDATES=2', text)
        self.assertIn('equation_repair', text)
        self.assertIn('MAX_ACTUAL_VS_PREDICTED_ERROR=0', text)

    def test_repair_dec_refusal_and_full6164_bound(self):
        raw = fixture(2, 5, 3)
        repair_row(raw, 2)
        self.assertIn('STAGE=5', C.inspect(raw))
        self.assertIn('TERMINAL_ROUND=6164', C.inspect(fixture(6164, 10, 3)))
        with self.assertRaisesRegex(ValueError, 'search bound'):
            C.inspect(fixture(6165, 10, 3))

    def test_repair_identity_score_and_budget_negatives(self):
        for selected in (8*192*4, 4*17+1):
            raw = fixture(3, 11, 3); repair_row(raw, 1, selected)
            with self.assertRaisesRegex(ValueError, 'repair identity'): C.inspect(raw)
        raw = fixture(3, 11, 3); repair_row(raw, 1)
        struct.pack_into('<d', raw, C.OFFSET+152+8, 0.)
        with self.assertRaisesRegex(ValueError, 'fabricated energy prediction'): C.inspect(raw)
        raw = fixture(6, 11, 3)
        for r in range(5): repair_row(raw, r)
        with self.assertRaisesRegex(ValueError, 'four-proposal bound'): C.inspect(raw)
        raw = fixture(3, 11, 3); repair_row(raw, 3)
        with self.assertRaisesRegex(ValueError, 'following full evaluation'): C.inspect(raw)

    def test_old_schema_cannot_claim_repair_or_precise_layout(self):
        raw = fixture(3, 11, 3); repair_row(raw, 1)
        for leaf in range(8):
            struct.pack_into('<I', raw, leaf*C.PRECISE_STRIDE+C.OFFSET, C.MAGIC_V2)
        with self.assertRaisesRegex(ValueError, 'repair identity'): C.inspect(raw)
        raw = fixture(3, 11, 2)
        for leaf in range(8): struct.pack_into('<I',raw,leaf*C.STRIDE+C.OFFSET,C.MAGIC_V3)
        with self.assertRaisesRegex(ValueError, 'numerical layout'): C.inspect(raw)

    def test_early_acceptance_does_not_require_unevaluated_score(self):
        self.assertIn("COMPARED_ADJACENT_UPDATES=0", C.inspect(fixture(0, 11)))

    def test_equation_refusal_retains_error_but_not_ranking(self):
        self.assertIn("STAGE=7", C.inspect(fixture(3, 7)))

    def test_failed_next_h_build_has_no_error_sample(self):
        text = C.inspect(fixture(40, 6))
        self.assertIn("TERMINAL_ROUND=41 LAST_EVALUATED_ROUND=40", text)

    def test_missing_population_refused(self):
        with self.assertRaisesRegex(ValueError, "complete eight"):
            C.inspect(fixture()[:-C.STRIDE])

    def test_stale_round_id_refused(self):
        raw = fixture()
        struct.pack_into("<I", raw, C.OFFSET + 356 + 4, 1)
        with self.assertRaisesRegex(ValueError, "retained-round set"):
            C.inspect(raw)

    def test_wrong_ring_slot_refused(self):
        raw = fixture()
        a, b = struct.unpack_from("<2I", raw, C.OFFSET + 360)
        struct.pack_into("<2I", raw, C.OFFSET + 360, b, a)
        with self.assertRaisesRegex(ValueError, "wrong ring slot"):
            C.inspect(raw)

    def test_disagreed_leaf_error_refused(self):
        raw = fixture()
        struct.pack_into("<d", raw, C.STRIDE + C.OFFSET + 16, -.002)
        with self.assertRaisesRegex(ValueError, "error/winner disagreement"):
            C.inspect(raw)

    def test_winner_must_own_minimum(self):
        raw = fixture()
        for leaf in range(8):
            struct.pack_into("<I", raw, leaf * C.STRIDE + C.OFFSET + 288, 768)
        with self.assertRaisesRegex(ValueError, "strictly improved minimum"):
            C.inspect(raw)

    def test_missing_winner_before_unretained_gap_refused(self):
        raw = fixture()
        for leaf in range(8):
            struct.pack_into("<I", raw, leaf * C.STRIDE + C.OFFSET + 288, C.UINT_MAX)
        with self.assertRaisesRegex(ValueError, "missing winner"):
            C.inspect(raw)

    def test_partial_nan_score_refused(self):
        raw = fixture()
        struct.pack_into("<d", raw, C.OFFSET + 152, math.nan)
        with self.assertRaisesRegex(ValueError, "invalid local scores"):
            C.inspect(raw)

    def test_nonimproving_measurement_is_reported_not_hidden(self):
        raw = fixture(3)
        for leaf in range(8):
            off = leaf * C.STRIDE + C.OFFSET
            struct.pack_into("<d", raw, off + 16 + 8, -.002)
        text = C.inspect(raw)
        self.assertIn("OBSERVED_ADJACENT_STRICTLY_IMPROVING=false", text)
        self.assertIn("NEW_ACCEPTANCE_TOLERANCE_DEFINED=false", text)


if __name__ == "__main__":
    print("CONTROL_SCOPE=constructed trace parser controls; no native trajectory acceptance")
    unittest.main(verbosity=2)
