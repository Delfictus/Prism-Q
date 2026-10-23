#!/usr/bin/env python3
"""Deterministic checker controls; constructed graphs are not native evidence."""
from fractions import Fraction as Q
from itertools import combinations
import struct
import unittest

import check_biprism_boundary_identity as C


def graph():
    phi = [0.0]*C.VERTICES
    for v, x in ((0, .75), (1, -.25), (192, .125), (193, -.5), (194, .25)):
        phi[v] = x
    edges = [(0, 1), (0, 192), (193, 0), (1, 193), (192, 193), (194, 193)]
    weights = [2., .5, 4., .25, 1.5, .125]
    charges = [0.0]*C.CORE
    charges[0], charges[1] = .375, -.625
    return charges, phi, edges, weights


def tile(tets):
    edges = sorted({tuple(sorted(e)) for t in tets for e in combinations(t, 2)})
    raw = bytearray(C.DEC.TILE_BYTES)
    struct.pack_into('<II', raw, C.TET_COUNT_OFFSET, len(tets), len(edges))
    for index, t in enumerate(tets):
        struct.pack_into('<4B', raw, C.TET_OFFSET+4*index, *t)
    for index, edge in enumerate(edges):
        struct.pack_into('<2B', raw, C.DEC.OFF['edges']+2*index, *edge)
    return bytes(raw), edges


class GraphControls(unittest.TestCase):
    def test_exact_graph_and_all_coordinate_basis_rows(self):
        result = C.graph_identities(*graph())
        self.assertEqual(result['basis_checks'], 256)
        self.assertEqual(len(result['halo_rows']), 64)
        self.assertEqual(len(result['cut_edges']), 3)
        self.assertEqual(len(result['pairings']), 4)

    def test_explicit_cut_orientation_and_internal_halo_divergence(self):
        result = C.graph_identities(*graph())
        rows = {row[0]: row[1:] for row in result['halo_rows']}
        # At192: incoming core flux=.5*(.75-.125)=5/16;
        # internal halo divergence=1.5*(.125-(-.5))=15/16.
        self.assertEqual(rows[192], (Q(5, 8), Q(5, 16), Q(15, 16)))
        # Stored edge193->0 must become the outward0->193 cut orientation.
        self.assertIn((2, 0, 193, Q(5)), result['cut_edges'])

    def test_reversing_all_stored_edges_preserves_physical_cut(self):
        q, phi, edges, weights = graph()
        a = C.graph_identities(q, phi, edges, weights)
        b = C.graph_identities(q, phi, [(v, u) for u, v in edges], weights)
        self.assertEqual(a, b)

    def test_equal_opposite_halo_forgery_cannot_hide_in_scalar_gauss_sum(self):
        args = graph()
        op = C.DEC.exact_operator(*args)
        before_sum = sum(op['rows'][C.CORE:])
        op['rows'][192] += Q(1, 1024)
        op['rows'][193] -= Q(1, 1024)
        self.assertEqual(sum(op['rows'][C.CORE:]), before_sum)
        with self.assertRaisesRegex(ValueError, 'coordinate-basis'):
            C.graph_identities(*args, operator=op)

    def test_omitting_halo_internal_divergence_is_detected(self):
        args = graph()
        result = C.graph_identities(*args)
        op = C.DEC.exact_operator(*args)
        before_sum = sum(op['rows'][C.CORE:])
        for vertex, _, cut, _ in result['halo_rows']:
            op['rows'][vertex] = -cut
        self.assertEqual(sum(op['rows'][C.CORE:]), before_sum)
        with self.assertRaisesRegex(ValueError, 'coordinate-basis'):
            C.graph_identities(*args, operator=op)
        self.assertNotEqual(result['pairings']['halo_nonconstant'][4], 0)

    def test_charge_residual_pairing_is_not_assumed(self):
        args = graph()
        op = C.DEC.exact_operator(*args)
        op['residual'][0] += Q(1, 64)
        with self.assertRaisesRegex(ValueError, 'charge/residual'):
            C.graph_identities(*args, operator=op)

    def test_screening_cannot_be_added_to_the_actual_operator(self):
        args = graph()
        op = C.DEC.exact_operator(*args)
        for v, phi in enumerate(args[1]):
            op['rows'][v] += Q.from_float(1e-4)*Q(phi)
        with self.assertRaisesRegex(ValueError, 'coordinate-basis'):
            C.graph_identities(*args, operator=op)

    def test_virtual_halo_delta_includes_quadratic_term(self):
        result = C.graph_identities(*graph())
        self.assertEqual(result['virtual_delta'], result['virtual_linear_work']+result['virtual_quadratic_term'])
        self.assertLess(result['virtual_quadratic_term'], 0)
        self.assertNotEqual(result['virtual_delta'], result['virtual_linear_work'])

    def test_forged_functional_is_detected(self):
        args = graph()
        op = C.DEC.exact_operator(*args)
        op['functional'] += Q(1, 1024)
        with self.assertRaisesRegex(ValueError, 'functional'):
            C.graph_identities(*args, operator=op)

    def test_invalid_weight_and_duplicate_edges_refused(self):
        for value in (0., -1., float('nan'), float('inf')):
            q, phi, edges, weights = graph()
            weights[1] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                C.graph_identities(q, phi, edges, weights)
        q, phi, edges, weights = graph()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            C.graph_identities(q, phi, edges+[edges[0]], weights+[weights[0]])

    def test_wrong_row_extent_refused(self):
        args = graph()
        op = C.DEC.exact_operator(*args)
        op['rows'].pop()
        with self.assertRaisesRegex(ValueError, 'row extent'):
            C.graph_identities(*args, operator=op)


class DomainControls(unittest.TestCase):
    def test_algebraic_halo_and_geometric_boundary_remain_distinct(self):
        raw, edges = tile([(0, 1, 192, 193), (1, 192, 193, 194)])
        info = C.retained_domain(raw, raw, edges)
        self.assertEqual(info, {'tetrahedra': 2, 'edges': 9, 'exposed_faces': 6,
                               'geometric_boundary_vertices': 5,
                               'core_on_geometric_boundary': 2,
                               'halo_on_geometric_boundary': 3,
                               'halo_off_geometric_boundary': 61})

    def test_retained_operator_mutation_refused(self):
        raw, edges = tile([(0, 1, 192, 193)])
        for offset in (0, C.TET_OFFSET, C.TET_EDGE_OFFSET, C.TET_OPPOSITE_OFFSET,
                       C.DEC.OFF['edges'], C.STAR0_OFFSET, C.DEC.OFF['base_weights'], C.TET_COUNT_OFFSET):
            changed = bytearray(raw)
            changed[offset] ^= 1
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                C.retained_domain(raw, bytes(changed), edges)

    def test_unrelated_graph_edge_cannot_be_called_mesh_incidence(self):
        raw, edges = tile([(0, 1, 192, 193)])
        with self.assertRaisesRegex(ValueError, 'edge population'):
            C.retained_domain(raw, raw, edges+[(2, 3)])

    def test_duplicate_and_three_incident_face_are_refused(self):
        raw, edges = tile([(0, 1, 192, 193)]*2)
        with self.assertRaisesRegex(ValueError, 'duplicate tetrahedron'):
            C.retained_domain(raw, raw, edges)
        raw, edges = tile([(0, 1, 192, 193), (0, 1, 192, 194), (0, 1, 192, 195)])
        with self.assertRaisesRegex(ValueError, 'face multiplicity'):
            C.retained_domain(raw, raw, edges)


if __name__ == '__main__':
    print('CONTROL_SCOPE=constructed exact graph/format controls only; no native or trajectory evidence')
    unittest.main(verbosity=2)
