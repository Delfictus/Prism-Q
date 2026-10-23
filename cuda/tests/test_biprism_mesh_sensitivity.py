#!/usr/bin/env python3
"""Comparison/format controls only; these constructed inputs are not native runs."""
import copy
from fractions import Fraction as Q
import math
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest

import check_biprism_mesh_sensitivity as C
import run_biprism_phase2_work_qc as R


def population():
    return {'ids': list(range(256)), 'charge': [0.]*192,
            'coordinates': [(0., 0., 0.)]*256, 'halo_bits': bytes(256),
            'tets': {(0, 1, 2, 3)}, 'edge_set': {(0, 1)},
            'operator': C.sparse_operator([(0, 1)], [.5]), 'phi': [0.]*256,
            'star0': [1.]*256, 'volume_au': Q(1), 'volume_native': Q(1, 8),
            'functional': Q(0)}


class Controls(unittest.TestCase):
    def test_baseline_comparison_is_exact_zero(self):
        self.assertTrue(all(v == 0 for v in C.differences(population(), population()).values()))

    def test_operator_delta_includes_diagonal_and_both_offdiagonals(self):
        a = population(); b = copy.deepcopy(a)
        b['operator'] = C.sparse_operator([(0, 1)], [1.])
        self.assertEqual(C.differences(a, b)['operator_delta_frobenius'], 1.)

    def test_domain_change_is_measured_without_a_sensitivity_limit(self):
        a = population(); b = copy.deepcopy(a)
        b['tets'].add((0, 1, 2, 4)); b['edge_set'].add((0, 4))
        b['volume_au'] += Q(1000); b['volume_native'] += Q(125)
        b['functional'] = Q(10**10)
        d = C.differences(a, b)
        self.assertEqual(d['tetrahedra_added'], 1)
        self.assertEqual(d['edges_added'], 1)
        self.assertEqual(d['volume_delta_bohr3'], 1000)
        self.assertEqual(d['functional_delta'], 10**10)

    def test_physical_population_charge_geometry_and_halo_are_not_variable_budgets(self):
        for field in ('ids', 'charge', 'coordinates', 'halo_bits'):
            a = population(); b = copy.deepcopy(a)
            if field == 'ids': b[field][0] = 999
            elif field == 'charge': b[field][0] = .125
            elif field == 'coordinates': b[field][0] = (1., 0., 0.)
            else: b[field] = b'\x01'+b[field][1:]
            with self.subTest(field=field), self.assertRaises(ValueError):
                C.differences(a, b)

    def test_exact_tetrahedral_volume_and_orientation_independence(self):
        p = [(Q(0), Q(0), Q(0)), (Q(1), Q(0), Q(0)),
             (Q(0), Q(1), Q(0)), (Q(0), Q(0), Q(1))]
        self.assertEqual(C.tet_volume(p, (0, 1, 2, 3)), Q(1, 6))
        self.assertEqual(C.tet_volume(p, (0, 2, 1, 3)), Q(1, 6))
        p[3] = p[0]
        with self.assertRaisesRegex(ValueError, 'zero-volume'):
            C.tet_volume(p, (0, 1, 2, 3))

    def test_low_coordinate_is_used_before_volume(self):
        x = Q(64)
        p = [(x, x, x), (x+Q(1, 2**30), x, x),
             (x, x+Q(1, 2**30), x), (x, x, x+Q(1, 2**30))]
        self.assertEqual(C.tet_volume(p, (0, 1, 2, 3)), Q(1, 6*2**90))

    def test_cold_conversion_retains_the_low_product(self):
        for value in (0., 1., -1., 64., 128.125, -3.0625):
            high, low = C.scalar_import(value)
            self.assertTrue(math.isfinite(high) and math.isfinite(low))
            self.assertLessEqual(abs(float(Q(high)+Q(low)-Q(value)*Q(C.FACTOR))),
                                 max(abs(value), 1.)*1e-14)
        self.assertNotEqual(C.scalar_import(64.)[1], 0.)

    def test_nonzero_seed_halo_uses_actual_bw_charge_initialization(self):
        ids = list(range(256)); charges = [(i-128)/256 for i in ids]
        tile = bytearray(C.TILE)
        phi = b''.join(C.fbits(C.DEC.f32(q*C.DEC.f32(.07957747))) for q in charges)
        offset = C.DEC.OFF['phi']; tile[offset:offset+1024] = phi
        halo = C.seed_and_halo_identity(tile, tile, tile, ids, charges)
        self.assertEqual(halo, phi[4*192:])
        self.assertNotEqual(halo, bytes(256))

    def test_wrong_seed_and_each_halo_stage_refuse(self):
        ids = list(range(256)); charges = [.25]*256
        tile = bytearray(C.TILE); offset = C.DEC.OFF['phi']
        phi = C.fbits(C.DEC.f32(.25*C.DEC.f32(.07957747)))*256
        tile[offset:offset+1024] = phi
        for stage, vertex in ((0, 3), (0, 192), (1, 192), (2, 255)):
            rows = [bytearray(tile) for _ in range(3)]
            rows[stage][offset+4*vertex] ^= 1
            with self.subTest(stage=stage, vertex=vertex), self.assertRaises(ValueError):
                C.seed_and_halo_identity(*rows, ids, charges)
        zero = bytearray(C.TILE)
        with self.assertRaisesRegex(ValueError, 'seed potential'):
            C.seed_and_halo_identity(zero, zero, zero, ids, charges)

    def test_historical_source_replay_is_explicit_and_hash_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = Path(tmp)/'original'; old = Path(tmp)/'snapshot'; index = Path(tmp)/'index'
            original.write_bytes(b'old'); old.write_bytes(b'old'); digest = C.sha(old)
            self.assertEqual(C.bound_source(original, digest, {}), original)
            original.write_bytes(b'changed')
            with self.assertRaises(ValueError): C.bound_source(original, digest, {})
            index.write_text(f'{original}\t{old}\t{digest}\n')
            mapping = C.source_snapshots(index)
            self.assertEqual(C.bound_source(original, digest, mapping), old)
            with self.assertRaises(ValueError): C.bound_source(original, '0'*64, mapping)
            old.write_bytes(b'tampered')
            with self.assertRaises(ValueError): C.source_snapshots(index)

    def test_default_profile_preserves_six_cases_and_adds_collective_cold_refusal(self):
        args = SimpleNamespace(mesh_budgets=None)
        stem, binary, checker, entries, boundary = R.profile(args)
        self.assertEqual(stem, 'failure_7_phase2_native')
        self.assertEqual(checker, R.CHECKER)
        self.assertEqual(len(R.CASES), 7)
        self.assertIn('cold_leaf_refusal', R.CASES)
        self.assertIn('nhs_biprism_phase2_work_initialize_publish_v1', entries)
        self.assertNotIn('nhs_seed_bowyer_watson_256_compensated_v1', entries)
        self.assertEqual(len(R.expected_outputs(Path('/test'), 400)), 180)

    def test_sensitivity_mode_has_distinct_outputs_and_actual_bw_entry(self):
        args = SimpleNamespace(mesh_budgets=[944, 960, 976])
        stem, binary, checker, entries, boundary = R.profile(args)
        self.assertEqual(stem, 'failure_5_mesh_sensitivity')
        self.assertEqual(checker, R.MESH_CHECKER)
        self.assertIn('nhs_seed_bowyer_watson_256_compensated_v1', entries)
        out = R.mesh_expected_outputs(Path('/test'), 400, args.mesh_budgets)
        self.assertEqual(len(out), 69)
        self.assertEqual(out['/test.target960.bw_status.80b'], 640)
        self.assertEqual(out['/test.target944.initial_vertices.224b'], 8*256*224)
        self.assertEqual(out['/test.target944.initial_phi_accepted.f64le'], 8*256*8)


if __name__ == '__main__':
    print('CONTROL_SCOPE=reader/runner constructed controls only; no native execution')
    unittest.main(verbosity=2)
