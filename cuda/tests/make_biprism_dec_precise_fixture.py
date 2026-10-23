"""Bind DEC numerical cases to retained native operators and electronic states.

The RHS is constructed explicitly from the retained start/candidate endpoints.
It is not claimed to be an unsaved continuous C1 trial's identical RHS.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import biprism_exchange_layout as EX
import check_biprism_spinor_owner as SO
import check_biprism_v10_dec_readback as D


def make(execution, epoch, output):
    rec = json.loads(Path(execution).read_text())
    assert rec['native_execution'] and rec['sources_binary_inputs_unchanged']
    assert not rec['new_kernel_fault_records']
    prefix = rec['command'][4]+f'.epoch{epoch}'
    def read(suffix):
        path = Path(prefix+'.'+suffix); raw = path.read_bytes()
        assert rec['artifacts'][str(path)] == dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        return raw
    tiles, owners, bank, numerical = (read(s) for s in
        ('tiles.bin', 'spinor.owners.bps1', 'exchange.bin', 'numerical.bin'))
    bank_stride, _, kind, _ = EX.layout(bank)
    assert kind == 'd' and len(numerical) == 8*1680768
    pieces = []
    for l in range(8):
        tile = tiles[l*D.TILE_BYTES:(l+1)*D.TILE_BYTES]
        _, edges, _, _ = D.parse_tile(tile)
        o = SO.read(owners, l)
        q = list(struct.unpack_from('<1024f', tile, 1024))[3::4]
        for v in range(192):
            q0 = D.represented_charge(q[v], o['start'][v][2:])
            q1 = D.represented_charge(q[v], o['candidate'][v][2:])
            q[v] = .5*(q0+q1)
        weights = struct.unpack_from('<1536d', bank, l*bank_stride+5376)
        phi = struct.unpack_from('<256d', numerical, l*1680768+38656)
        row = struct.pack('<II', len(edges), l)
        row += b''.join(struct.pack('<BB', *e) for e in edges)+bytes(2*(1536-len(edges)))
        row += struct.pack('<1536d256d256d', *weights, *q, *phi)
        assert len(row) == 19464
        pieces.append(row)
    raw = b''.join(pieces)
    with Path(output).open('xb') as f:
        f.write(raw)
    print('ACTUAL_OPERATOR_DERIVED_RHS', str(execution), 'epoch', epoch,
          'sha256', hashlib.sha256(raw).hexdigest(), 'bytes', len(raw),
          'EXACT_UNSAVED_C1_RHS=0 COUPLED_ACCEPTANCE=0')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execution', type=Path, required=True)
    p.add_argument('--epoch', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); make(a.execution, a.epoch, a.output)
