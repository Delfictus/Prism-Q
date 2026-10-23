"""Read a refused native epoch without assigning acceptance to private state.

Exact retained primitives define the DEC equation. The independent dense solve
is an inner-accuracy sensitivity measurement, never a replacement native step.
"""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np
import check_biprism_spinor_owner as S
import check_biprism_v10_dec_readback as D


def inspect(execution, epoch):
    receipt = json.loads(execution.read_text())
    assert receipt['native_execution'] and receipt['sources_binary_inputs_unchanged']
    assert receipt['exit_code'] == 4 and receipt['new_kernel_fault_records'] == []
    prefix = receipt['command'][4] + f'.epoch{epoch}'

    def read(suffix):
        p = Path(prefix + suffix)
        raw = p.read_bytes()
        assert receipt['artifacts'][str(p)] == {
            'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        return raw

    numerical, tiles, exchange = (read(x) for x in ('.numerical.bin', '.tiles.bin', '.exchange.bin'))
    spinors, before = read('.spinor.owners.bps1'), read('.entry.spinor.owners.bps1')
    stride = len(numerical) // 8
    assert stride in (196992, 198144, 235136,349056,496768,510720,1680768,1697920,1784448,1837696,1997440) and len(numerical) == 8 * stride
    for a, b in [('.entry_au_coords.bin', '.au_coords.bin'),
                 ('.entry.au_coords_low.bin', '.au_coords_low.bin'),
                 ('.entry_au_velocities.bin', '.au_velocities.bin'),
                 ('.phase2_tiles.bin', '.tiles.bin'),
                 ('.entry.potential.owners.128b', '.potential.owners.128b'),
                 ('.entry.potential.accepted.f64le', '.potential.accepted.f64le'),
                 ('.entry.potential.prepared.f64le', '.potential.prepared.f64le'),
                 ('.entry.au_coordinate_owners.96b', '.au_coordinate_owners.96b'),
                 ('.entry.au_stream_table.640b', '.au_stream_table.640b')]:
        assert read(a) == read(b), (a, b)
        print('REFUSAL_BITWISE_UNCHANGED', a, b, 'all_leaves=8')
    for leaf in range(8):
        sl = slice(leaf * S.SIZE, leaf * S.SIZE + 6272)
        assert spinors[sl] == before[sl]
        o = S.read(spinors, leaf)
        assert o['committed'] == 4 * (epoch - 1)
        magic, substep, count, reserved = struct.unpack_from('<4I', numerical, leaf * stride + 40704)
        assert magic == 0x4e485034 and substep in range(4) and count == 48 and reserved == 0
        history = np.frombuffer(numerical, '<f8', 3 * count, leaf * stride + 40720).reshape(count, 3)
        assert np.isfinite(history).all()
        print('EVALUATED_C4_PREFIX', leaf, 'substep', substep, 'count', count,
              'accepted_spinor_prefix_unchanged', True, 'final', history[-1].tolist())
        tile = tiles[leaf * D.TILE_BYTES:(leaf + 1) * D.TILE_BYTES]
        _, edges, _, _ = D.parse_tile(tile)
        weights = struct.unpack_from('<%df' % len(edges), exchange, leaf * D.BANK_BYTES + D.OFF['live_weights'])
        phi = list(struct.unpack_from('<256d', numerical, leaf * stride + 38656))
        amber = [struct.unpack_from('<f', tile, 1024 + 16 * v + 12)[0] for v in range(192)]
        charges = [D.represented_charge(amber[v], o['candidate'][v][2:]) for v in range(192)]
        exact = D.exact_operator(charges, phi, edges, list(weights))
        A = np.zeros((192, 192))
        for (u, v), w in zip(edges, weights):
            if u < 192: A[u, u] += w
            if v < 192: A[v, v] += w
            if max(u, v) < 192: A[u, v] -= w; A[v, u] -= w
        residual = np.array([float(x) for x in exact['residual']])
        # This correction solves the ORIGINAL represented operator with its
        # fixed halo. Starting from an exact residual avoids dot-product noise.
        correction = np.linalg.solve(A, residual)
        eigmin = np.linalg.eigvalsh(A)[0]
        gain = .5 * float(residual @ correction)
        field_error = 0.
        for (u, v), w in zip(edges, weights):
            d = phi[u] - phi[v]
            dd = (correction[u] if u < 192 else 0.) - (correction[v] if v < 192 else 0.)
            field_error = max(field_error, abs(.5 * w * (2. * d * dd + dd * dd)))
        print('EXACT_ENDPOINT_DEC', leaf, 'l2', math.sqrt(float(exact['squared'])),
              'lambda_min', eigmin, 'potential_correction_l2', np.linalg.norm(correction),
              'legendre_energy_gain', gain, 'max_edge_field_coefficient_change', field_error)
    print('SCOPE: endpoint DEC sensitivity only; not a midpoint-force replay or production qualification')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--execution', type=Path, required=True)
    p.add_argument('--epoch', type=int, required=True)
    a = p.parse_args()
    inspect(a.execution, a.epoch)
