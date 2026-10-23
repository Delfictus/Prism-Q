#!/usr/bin/env python3
"""Constructed ABI/adversarial controls, not native or molecular evidence."""
import math
import struct
import unittest
import check_biprism_v10_dec_readback as C


FMT = '<4I6Q4I2Q4I2Q'


def fixture(prepared=True, production=False):
    atoms, entry = 256, 8
    data = {k: bytearray(n) for k, n in {
        'potential_owner': 1024, 'entry_potential_owner': 1024,
        'potential_accepted': 16384, 'potential_prepared': 16384,
        'entry_potential_accepted': 16384, 'entry_potential_prepared': 16384,
        'coordinate_owner': 768, 'entry_coordinate_owner': 768,
        'stream_table': 640, 'entry_stream_table': 640,
        'ordered_edges': 8*6656, 'numerical': 8*40704,
        'entry': 8*C.TILE_BYTES, 'endpoint': 8*C.TILE_BYTES}.items()}
    coord_base, index_base, owner_base, tile_base = 0x1000000, 0x1100000, 0x1200000, 0x2000000
    struct.pack_into('<II3Q', data['stream_table'], 520, 0 if production else 4, 0,
                     coord_base, index_base, owner_base)
    for leaf in range(8):
        planes = [0x3000000+(6*leaf+k)*4096 for k in range(6)]
        scratch = 0x4000000+leaf*131072
        accepted_ptr, prepared_ptr = 0x5000000+leaf*8192, 0x5001000+leaf*8192
        struct.pack_into('<Q', data['stream_table'], leaf*8, planes[0])
        struct.pack_into('<Q', data['stream_table'], 448+leaf*8, scratch)
        struct.pack_into('<4IQ2I8Q', data['coordinate_owner'], leaf*96,
                         0x31504342, 1, leaf, atoms, 23, 2, 7, *planes, 0, 0)
        common = [0x31505042, 1, leaf, 2 if prepared else 1, 23, 11,
                  tile_base+leaf*C.TILE_BYTES, coord_base+leaf*96, accepted_ptr, prepared_ptr]
        if prepared:
            before = common+[entry, 2, entry, 3, 7, 8, 2, entry, 3, 0, 8, 0]
            after = common[:3]+[1]+common[4:]+[entry+4, 3, entry, 3, 8, 8, 1, entry+4, 3, 0, 8, 0]
        else:
            before = common+[entry, 2, 4, 2, 7, 7, 1, entry, 2, 0, 7, 0]
            after = common[:3]+[1]+common[4:]+[entry+4, 3, 4, 2, 7, 7, 1, entry+4, 3, 0, 7, 0]
        struct.pack_into(FMT, data['entry_potential_owner'], leaf*128, *before)
        struct.pack_into(FMT, data['potential_owner'], leaf*128, *after)
        initial = [.25+leaf/1024+v/1000000 for v in range(256)]
        final = [x+(2**-35 if v < 192 else 0) for v, x in enumerate(initial)]
        old_accepted = [x-(.001 if prepared and v < 192 else 0) for v, x in enumerate(initial)]
        for name, values in (('entry_potential_accepted', old_accepted),
                             ('entry_potential_prepared', initial),
                             ('potential_prepared', initial), ('potential_accepted', final)):
            struct.pack_into('<256d', data[name], leaf*2048, *values)
        struct.pack_into('<256d', data['numerical'], leaf*40704+38656, *final)
        for name, values in (('entry', initial), ('endpoint', final)):
            struct.pack_into('<256f', data[name], leaf*C.TILE_BYTES+C.OFF['phi'], *values)
            struct.pack_into('<I', data[name], leaf*C.TILE_BYTES+C.OFF['edge_count'], 1)
        struct.pack_into('<4IQ3I', data['ordered_edges'], leaf*6656,
                         0x31454942, 1, leaf, 1, tile_base+leaf*C.TILE_BYTES, entry, 2, 1)
    data['entry_stream_table'][:] = data['stream_table']
    data['entry_coordinate_owner'][:] = data['coordinate_owner']
    return data, atoms, entry


class Controls(unittest.TestCase):
    def test_dedicated_iteration_history_is_not_a_potential_suffix(self):
        for stride in (43008,66048,70656,72960,97536):
            d,n,count=fixture()
            expected=C.retained_potential_epoch(d,n,count)
            old=d['numerical']
            d['numerical']=b''.join(old[l*40704:(l+1)*40704]+bytes([37+l])*(stride-40704) for l in range(8))
            self.assertEqual(C.retained_potential_epoch(d,n,count,stride),expected)
            with self.assertRaises(ValueError):C.retained_potential_epoch(d,n,count)
            changed=bytearray(d['numerical']);changed[7*stride+38656]^=1;d['numerical']=changed
            with self.assertRaisesRegex(ValueError,'differs from evaluated'):C.retained_potential_epoch(d,n,count,stride)

    def test_complete_prepared_owner_and_full_bits(self):
        d, n, count = fixture(prepared=True, production=True)
        start, end, rows = C.retained_potential_epoch(d, n, count)
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[0]['entry_state'], 'prepared')
        self.assertNotEqual(end[0][0], C.f32(end[0][0]))
        self.assertEqual(struct.pack('<64d', *start[5][192:]), struct.pack('<64d', *end[5][192:]))

    def test_explicit_diagnostic_accepted_input(self):
        d, n, count = fixture(prepared=False)
        self.assertEqual(C.retained_potential_epoch(d, n, count)[2][0]['entry_state'], 'accepted')

    def test_production_cannot_consume_stale_accepted_bank(self):
        d, n, count = fixture(prepared=False, production=True)
        with self.assertRaisesRegex(ValueError, 'fresh prepared'):
            C.retained_potential_epoch(d, n, count)

    def test_all_extents_are_required(self):
        d, n, count = fixture()
        for key in ('potential_owner', 'potential_accepted', 'potential_prepared',
                    'entry_potential_owner', 'entry_potential_accepted', 'entry_potential_prepared',
                    'coordinate_owner', 'entry_coordinate_owner', 'stream_table',
                    'entry_stream_table', 'ordered_edges', 'numerical'):
            with self.subTest(key=key):
                bad = d.copy(); bad[key] = d[key][:-1]
                with self.assertRaisesRegex(ValueError, 'extent'):
                    C.retained_potential_epoch(bad, n, count)

    def test_low_only_endpoint_change_is_not_hidden_by_float_projection(self):
        d, n, count = fixture()
        before, = struct.unpack_from('<d', d['potential_accepted'])
        after = before+2**-40
        self.assertEqual(C.f32(before), C.f32(after))
        struct.pack_into('<d', d['potential_accepted'], 0, after)
        with self.assertRaisesRegex(ValueError, 'evaluated working state'):
            C.retained_potential_epoch(d, n, count)

    def test_halo_low_only_change_cannot_be_hidden_by_working_or_projection(self):
        d, n, count = fixture()
        value, = struct.unpack_from('<d', d['potential_accepted'], 192*8)
        struct.pack_into('<d', d['potential_accepted'], 192*8, value+2**-40)
        struct.pack_into('<d', d['numerical'], 38656+192*8, value+2**-40)
        with self.assertRaisesRegex(ValueError, 'fixed halo'):
            C.retained_potential_epoch(d, n, count)

    def test_prepared_owner_is_never_silently_committed(self):
        d, n, count = fixture()
        struct.pack_into('<I', d['potential_owner'], 12, 2)
        with self.assertRaisesRegex(ValueError, 'accepted potential chronology'):
            C.retained_potential_epoch(d, n, count)

    def test_malformed_source_epochs(self):
        for key, offset, fmt, value in (
            ('potential_owner', 64, '<I', 8),
            ('potential_owner', 80, '<Q', 9),
            ('potential_owner', 112, '<Q', 9),
            ('entry_potential_owner', 76, '<I', 4),
            ('entry_potential_owner', 88, '<Q', 9),
            ('entry_potential_owner', 108, '<I', 1),
            ('potential_owner', 24, '<Q', 12),
            ('potential_owner', 40, '<Q', 0x1000060),
            ('ordered_edges', 24, '<I', 4),
        ):
            with self.subTest(key=key, offset=offset):
                d, n, count = fixture(); struct.pack_into(fmt, d[key], offset, value)
                with self.assertRaises(ValueError):
                    C.retained_potential_epoch(d, n, count)

    def test_prepared_storage_not_overwritten(self):
        d, n, count = fixture()
        d['potential_prepared'][0] ^= 1
        with self.assertRaisesRegex(ValueError, 'prepared bank'):
            C.retained_potential_epoch(d, n, count)

    def test_cross_leaf_potential_alias_refuses(self):
        d, n, count = fixture()
        for key in ('potential_owner', 'entry_potential_owner'):
            struct.pack_into('<Q', d[key], 128+48, 0x5000000)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            C.retained_potential_epoch(d, n, count)

    def test_pointer_to_coordinate_plane_cannot_be_used_as_potential(self):
        d, n, count = fixture()
        for key in ('potential_owner', 'entry_potential_owner'):
            struct.pack_into('<Q', d[key], 48, 0x3000000)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            C.retained_potential_epoch(d, n, count)

    def test_actual_tile_projection_is_required(self):
        d, n, count = fixture()
        d['endpoint'][C.OFF['phi']] ^= 1
        with self.assertRaisesRegex(ValueError, 'compatibility projection'):
            C.retained_potential_epoch(d, n, count)

    def test_nonfinite_full_value_refuses_even_if_unused_bank(self):
        for name in ('entry_potential_accepted', 'entry_potential_prepared'):
            d, n, count = fixture()
            struct.pack_into('<d', d[name], 0, math.nan)
            if name == 'entry_potential_prepared':
                d['potential_prepared'][:] = d[name]
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                C.retained_potential_epoch(d, n, count)

    def test_exact_operator_consumes_fp64_low_bits(self):
        edges = [(u, 192) for u in range(192)]
        phi = [.375+2**-30]*192+[.125]*64
        exact = C.exact_operator([.25]*192, phi, edges, [1.]*192)
        projected = C.exact_operator([.25]*192, list(map(C.f32, phi)), edges, [1.]*192)
        self.assertEqual(projected['squared'], 0)
        self.assertEqual(exact['squared'], C.Q(192, 2**60))


class OrderedControls(unittest.TestCase):
    def setUp(self):
        self.phi = [.375]*192+[.125]*64
        self.edges = [(u, 192) for u in range(192)]
        self.weights = [1.]*192
        self.charge = [.25]*192
        records = []
        for leaf in range(8):
            for u in range(256):
                row, diagonal = (.25, 1.0001) if u < 192 else (0., .0001)
                records.append(struct.pack('<6d2f4I', row, row, diagonal, diagonal,
                                           self.phi[u], self.phi[u], 0., 0., 1, 1, 1, 1))
        self.data = {'ordered_control_schema': struct.pack('<8I', 0x3250444f, 2, 72, 8, 4, 8, 256, 0),
                     'ordered_control_0': bytearray(b''.join(records)),
                     'ordered_control_1': bytearray(b''.join(records))}

    def check(self):
        return C.ordered_potential_control_leaf(self.data, 0, self.charge, self.edges, self.weights, self.phi)

    def test_actual72_schema_and_exact_residual(self):
        results = self.check()
        self.assertEqual([v['source_success'] for v in results], [True, True])
        self.assertTrue(all(v['true_residual_l2'] == 0 for v in results))

    def test_old64_schema_is_not_silently_decoded(self):
        self.data['ordered_control_0'] = self.data['ordered_control_0'][:8*256*64]
        with self.assertRaisesRegex(ValueError, 'population extent'):
            self.check()

    def test_every_compared_field_is_checked(self):
        original = bytes(self.data['ordered_control_0'])
        for offset in (8, 24, 40, 52, 60, 68):
            self.data['ordered_control_0'][:] = original
            self.data['ordered_control_0'][offset] ^= 1
            with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, 'mismatch'):
                self.check()

    def test_both_solvers_cannot_agree_on_false_success(self):
        for u in range(192):
            struct.pack_into('<2d', self.data['ordered_control_0'], u*72+32, .375+1e-7, .375+1e-7)
        with self.assertRaisesRegex(ValueError, 'independent exact residual'):
            self.check()

    def test_private_refusal_is_retained_without_acceptance(self):
        for u in range(256):
            struct.pack_into('<2f4I', self.data['ordered_control_0'], u*72+48,
                             math.inf, math.inf, 192, 192, 0, 0)
            if u < 192:
                struct.pack_into('<2d', self.data['ordered_control_0'], u*72+32, .375+1e-7, .375+1e-7)
        result = self.check()[0]
        self.assertFalse(result['source_success'])
        self.assertFalse(result['strict_exact_residual_pass'])

    def test_halo_change_is_not_private_solver_freedom(self):
        struct.pack_into('<2d', self.data['ordered_control_0'], 192*72+32, .125+2**-40, .125+2**-40)
        with self.assertRaisesRegex(ValueError, 'full fixed halo'):
            self.check()

    def test_nonuniform_cta_decision_refuses(self):
        struct.pack_into('<2I', self.data['ordered_control_0'], 72+56, 2, 2)
        with self.assertRaisesRegex(ValueError, 'CTA decision'):
            self.check()


if __name__ == '__main__':
    print('CONTROL_SCOPE=constructed retained-potential ABI and exact oracle controls; no native execution')
    unittest.main(verbosity=2)
