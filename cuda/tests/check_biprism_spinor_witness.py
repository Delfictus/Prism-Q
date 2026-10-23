"""Exact-dyadic checks of the native frozen Cayley storage/reload witness.

This covers the frozen equation and three norm gates, not coupled C1 admission.
The fixture kernel writes the real BPS1 accepted/start/candidate member layout;
it does not construct a complete production owner descriptor.
"""
import argparse
import hashlib
import math
import struct
from fractions import Fraction as Q
from pathlib import Path

from biprism_frozen_equation_interval import cayley_system
from check_biprism_spinor_owner import SIZE


def inspect(inputs, owner):
    assert len(inputs) == 88 and len(owner) == SIZE
    values = struct.unpack('<11d', inputs)
    assert all(math.isfinite(x) for x in values)
    start, H, omega, h, outer = values[:4], values[4:8], *values[8:]
    accepted = struct.unpack_from('<4d', owner, 128 + 32 * 95)
    stored_start = struct.unpack_from('<4d', owner, 6272 + 32 * 95)
    candidate = struct.unpack_from('<4d', owner, 12416 + 32 * 95)
    assert struct.pack('<4d', *stored_start) == inputs[:32]
    assert struct.pack('<4d', *accepted) == struct.pack('<4d', *candidate)
    assert struct.unpack_from('<I', owner, 12)[0] == 1
    assert struct.unpack_from('<2I', owner, 48) == (4, 1)
    assert all(math.isfinite(x) for x in accepted)
    matrix, rhs, scale = cayley_system(start, H, omega, h)
    equation = max(abs(sum(a * Q(x) for a, x in zip(row, accepted)) - b)
                   for row, b in zip(matrix, rhs)) / scale
    norm = sum(Q(x) ** 2 for x in accepted)
    references = (Q(1), sum(Q(x) ** 2 for x in start), Q(outer))
    defects = [abs(norm - x) for x in references]
    tolerance = Q(struct.unpack('<f', struct.pack('<f', 1e-7))[0])
    assert equation <= tolerance and all(x <= tolerance for x in defects)
    assert any(float(struct.unpack('<f', struct.pack('<f', x))[0]) != x for x in accepted)
    return accepted, equation, defects, tolerance


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--owner', required=True, type=Path)
    a = p.parse_args()
    inputs, owner = a.input.read_bytes(), a.owner.read_bytes()
    values, eq, norms, tol = inspect(inputs, owner)
    print('scope=FROZEN_CAYLEY_NATIVE_FP64_STORAGE_RELOAD coupled_C1_not_claimed')
    for name, data in ((str(a.input), inputs), (str(a.owner), owner)):
        print(name, 'bytes', len(data), 'sha256', hashlib.sha256(data).hexdigest())
    print('components_hex', [v.hex() for v in values])
    print('exact_dyadic_equation_residual', float(eq), 'margin', float(tol-eq))
    print('triple_norm_defects', [float(x) for x in norms], 'margins', [float(tol-x) for x in norms])
    print('FROZEN_PREDICATES_PASS=true STORED_RELOADED_WITHOUT_FP32_NARROWING=true')
