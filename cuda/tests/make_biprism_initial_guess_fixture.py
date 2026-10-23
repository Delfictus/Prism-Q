"""Bind a diagnostic initial guess to exact native substep-start operands.

Only the initial velocity guess and Gonzalez coefficient are supplied. The
native test bounds seed applicability and still evaluates C1-C4 on the actual
live starting operands; it never imports these captured physical operands.
This fixture is forbidden as evidence of autonomous production qualification.
"""
import argparse
import hashlib
import struct
from pathlib import Path

import numpy as np

from check_biprism_coupled_work_response import CoupledResponse


def write(execution, epoch, candidate, output):
    c = CoupledResponse(execution, epoch)
    with np.load(candidate, allow_pickle=False) as a:
        assert str(a['execution_sha256']) == hashlib.sha256(Path(execution).read_bytes()).hexdigest()
        assert str(a['numerical_sha256']) == hashlib.sha256(c.raw).hexdigest()
        assert int(a['epoch']) == epoch
        x, scales, h = a['x'].copy(), a['scales'].copy(), float(a['h'])
    original = struct.unpack('<f', struct.pack('<I', c.headers[0][10]))[0]
    assert h == original and x.shape == (4616,) and scales.shape == (8,)
    conv = c.read('conv.bin')
    entries = [struct.unpack_from('<I', conv, 128*l+20)[0] for l in range(8)]
    assert len(set(entries)) == 1 and all(row[1] == c.headers[0][1] for row in c.headers)
    p = np.array([c.arr(l, 43008, 576).reshape(3, 192) for l in range(8)])
    psi = np.stack((c.z0[..., 0].real, c.z0[..., 0].imag,
                    c.z0[..., 1].real, c.z0[..., 1].imag), axis=1)
    v0 = np.array([np.frombuffer(c.raw, '<f4', 576, l*c.stride+2304).reshape(3, 192)
                   for l in range(8)])
    pieces = [struct.pack('<IIIf', 0x31474942, entries[0], c.headers[0][1], h)]
    for plane in (x[4608:]/scales, c.start.transpose(0, 2, 1), p, psi,
                  x[:4608].reshape(8, 192, 3).transpose(0, 2, 1)):
        assert np.isfinite(plane).all()
        pieces.append(np.asarray(plane, dtype='<f8').tobytes(order='C'))
    pieces.extend((np.asarray(v0, dtype='<f4').tobytes(order='C'), bytes(32)))
    raw = b''.join(pieces)
    assert len(raw) == 178288
    with Path(output).open('xb') as out:
        out.write(raw)
    print('STANDALONE_INITIAL_GUESS', output, hashlib.sha256(raw).hexdigest(),
          'accepted_entry', entries[0], 'substep', c.headers[0][1],
          'NATIVE_ACCEPTANCE=0 AUTONOMOUS_QUALIFICATION=0')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execution', type=Path, required=True)
    p.add_argument('--epoch', type=int, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    write(a.execution, a.epoch, a.candidate, a.output)
