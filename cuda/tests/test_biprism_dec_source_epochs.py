#!/usr/bin/env python3
"""Exact source-identity controls; no native or numerical acceptance claim."""
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
import check_biprism_v10_dec_readback as C


class Controls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.linear = self.root/'linear.cuh'
        original = subprocess.check_output(['git', 'cat-file', 'blob',
            '2171486a4aa33c4d58fe805746d1e891f2307876'], cwd=C.ROOT)
        self.assertEqual(hashlib.sha256(original).hexdigest(), C.EXPECTED_DEC)
        self.linear.write_bytes(original)
        self.indexed = self.root/'indexed.cuh'
        indexed = subprocess.check_output(['git', 'cat-file', 'blob',
            '74f16cbef097f41267b6bccc1596fc95a9b3bdf8'], cwd=C.ROOT)
        self.assertEqual(hashlib.sha256(indexed).hexdigest(), C.EXPECTED_ORDERED_DEC)
        self.indexed.write_bytes(indexed)
        self.precise = C.KERNELS/'nhs_biprism_dec_pcg.cuh'
        self.ordered = C.KERNELS/'nhs_biprism_ordered_edges.cuh'
        self.potential = C.KERNELS/'nhs_biprism_potential_precision.cuh'

    def tearDown(self):
        self.tmp.cleanup()

    def test_historical_linear_needs_no_fabricated_index(self):
        self.assertEqual(C.dec_source_bindings(self.linear, self.root/'absent'),
                         ('historical_linear', []))

    def test_reviewed_indexed_source_binds_its_traversal(self):
        self.assertEqual(C.dec_source_bindings(self.indexed, self.ordered),
                         ('ordered_incidence_same_operator', [self.ordered]))

    def test_changed_solver_is_not_an_implicit_new_epoch(self):
        bad = self.root/'changed.cuh'; bad.write_bytes(self.indexed.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError, 'DEC source changed'):
            C.dec_source_bindings(bad, self.ordered)

    def test_changed_index_is_not_hidden_by_unchanged_solver(self):
        bad = self.root/'changed.cuh'; bad.write_bytes(self.ordered.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError, 'traversal source changed'):
            C.dec_source_bindings(self.indexed, bad)

    def test_missing_index_refuses(self):
        with self.assertRaises(FileNotFoundError):
            C.dec_source_bindings(self.indexed, self.root/'absent')

    def test_precise_source_requires_both_qualified_owners(self):
        self.assertEqual(C.dec_source_bindings(self.precise, self.ordered, self.potential),
                         ('ordered_incidence_retained_fp64_potential', [self.ordered, self.potential]))

    def test_changed_potential_owner_cannot_be_silently_reinterpreted(self):
        bad = self.root/'changed.cuh'; bad.write_bytes(self.potential.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError, 'potential owner source changed'):
            C.dec_source_bindings(self.precise, self.ordered, bad)

    def test_missing_potential_owner_refuses(self):
        with self.assertRaises(FileNotFoundError):
            C.dec_source_bindings(self.precise, self.ordered, self.root/'absent')

    def test_historical_indexed_needs_no_fabricated_precision_owner(self):
        self.assertEqual(C.dec_source_bindings(self.indexed, self.ordered, self.root/'absent'),
                         ('ordered_incidence_same_operator', [self.ordered]))


if __name__ == '__main__':
    print('CONTROL_SCOPE=source provenance only; exact operator oracle unchanged')
    unittest.main(verbosity=2)
