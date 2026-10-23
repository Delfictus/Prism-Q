"""Exact rational exclusion of FP32 states for a RETAINED FROZEN Cayley matrix.

This is a local representation feasibility diagnostic, not nonlinear coupled
admission. Freeze one FP32 component; allow the other three to be arbitrary
real numbers. Enumerate the vertices of the resulting equation polytope using
Fraction arithmetic. If even that larger set misses the norm interval, every
FP32 state on the slice is excluded. All FP32 pivot values allowed by the
inverse-matrix bound are considered. An inconclusive slice stays inconclusive.

The native equation and norm gates remain unchanged. The CPU exclusion expands
both by 1e-12 to cover FP64 evaluation roundoff in this bounded coefficient
regime; it therefore excludes a superset of the native feasible states.
"""
from fractions import Fraction as Q
from itertools import combinations
import math
import struct


def solve(matrix, rhs):
    a = [list(map(Q, row)) + [Q(b)] for row, b in zip(matrix, rhs)]
    for i in range(len(a)):
        pivot = next((j for j in range(i, len(a)) if a[j][i]), None)
        if pivot is None:
            return None
        a[i], a[pivot] = a[pivot], a[i]
        scale = a[i][i]
        a[i] = [x / scale for x in a[i]]
        for j in range(len(a)):
            if j != i:
                scale = a[j][i]
                a[j] = [x - scale*y for x, y in zip(a[j], a[i])]
    return [row[-1] for row in a]


def ordered_float(key):
    bits = key ^ 0x80000000 if key & 0x80000000 else key ^ 0xffffffff
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def lower_float_key(value):
    lo, hi = 0x00800000, 0xff800000  # finite range, with one-past upper sentinel
    while lo < hi:
        mid = (lo + hi)//2
        if Q(ordered_float(mid)) < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


def cayley_system(start, H, omega, h):
    a, d, b, c = map(Q, H)
    half, w = Q(h)/2, Q(omega)
    # H01=b+i*c. i*H in (Re0,Im0,Re1,Im1) coordinates.
    jh = [[0, -a, -c, -b], [a, 0, b, -c],
          [c, -b, 0, -d], [b, c, d, 0]]
    matrix = [[w*(i == j) + half*jh[i][j] for j in range(4)] for i in range(4)]
    rhs = [sum((w*(i == j) - half*jh[i][j])*Q(start[j]) for j in range(4)) for i in range(4)]
    return matrix, rhs, max(w, *(abs(half*x) for x in (a, d, b, c)))


def exclude_frozen(record):
    H = (*record['H'][:3], record['H'][3] + record['connection'])
    start, omega, h = record['start'], record['omega'], record['h']
    # Restrict the roundoff envelope to the stated finite, normal regime.
    values = (*start, *H, omega, h, record['norm_start'], record['norm_outer'])
    if not all(math.isfinite(x) and abs(x) <= 128 for x in values):
        raise ValueError('outside bounded frozen arithmetic regime')
    if not (1e-4 <= omega <= 128 and 1e-4 <= h <= 128):
        raise ValueError('outside bounded time/weight regime')
    if sum(x*x for x in start) > 1.000001:
        raise ValueError('start not within native unit-norm regime')
    matrix, rhs, scale = cayley_system(start, H, omega, h)
    centre = solve(matrix, rhs)
    inverse = list(zip(*(solve(matrix, [int(i == j) for i in range(4)]) for j in range(4))))
    tolerance = Q(struct.unpack('<f', struct.pack('<f', 1e-7))[0])
    envelope = Q(1e-12)
    bound = (tolerance + envelope)*scale
    radii = [bound*sum(map(abs, row)) for row in inverse]
    pivot = max(range(4), key=lambda i: abs(centre[i]))
    left, right = centre[pivot]-radii[pivot], centre[pivot]+radii[pivot]
    lo = lower_float_key(left)
    hi = lower_float_key(right)
    if hi < 0xff800000 and Q(ordered_float(hi)) == right:
        hi += 1
    if not 0 < hi-lo <= 32:
        raise ValueError('pivot population outside bounded enumeration')
    norm_lo = max(map(Q, (1., record['norm_start'], record['norm_outer'])))-tolerance-envelope
    norm_hi = min(map(Q, (1., record['norm_start'], record['norm_outer'])))+tolerance+envelope
    others = [j for j in range(4) if j != pivot]
    slices = []
    for key in range(lo, hi):
        value = Q(ordered_float(key))
        rows, limits = [], []
        for k in range(4):
            fixed_rhs = rhs[k]-matrix[k][pivot]*value
            for sign in (1, -1):
                rows.append([sign*matrix[k][j] for j in others])
                limits.append(bound+sign*fixed_rhs)
        vertices = []
        for indices in combinations(range(8), 3):
            point = solve([rows[i] for i in indices], [limits[i] for i in indices])
            if point is not None and all(sum(a*b for a, b in zip(row, point)) <= limit
                                         for row, limit in zip(rows, limits)):
                vertices.append(point)
        if not vertices:
            slices.append({'value': value, 'excluded': True, 'reason': 'empty_equation_polytope'})
            continue
        # Convex squared norm attains its maximum at a polytope vertex.
        upper = value*value + max(sum(x*x for x in point) for point in vertices)
        # Independent coordinate intervals give a conservative norm lower
        # bound. Loose intervals may only make the result inconclusive.
        lower = value*value
        for j in range(3):
            low = min(p[j] for p in vertices); high = max(p[j] for p in vertices)
            lower += 0 if low <= 0 <= high else min(low*low, high*high)
        excluded = upper < norm_lo or lower > norm_hi
        slices.append({'value': value, 'vertices': len(vertices), 'lower': lower, 'upper': upper,
                       'excluded': excluded,
                       'margin': max(norm_lo-upper, lower-norm_hi) if excluded else Q(0)})
    return {'centre': centre, 'pivot': pivot, 'norm_lo': norm_lo, 'norm_hi': norm_hi,
            'equation_roundoff_envelope': envelope, 'slices': slices,
            'all_fp32_slices_excluded': all(s['excluded'] for s in slices)}


def main():
    import argparse
    import contextlib
    import io
    import json
    import re
    from pathlib import Path
    from types import SimpleNamespace
    import check_biprism_c1_refusal as C
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dispatch', required=True, type=Path)
    parser.add_argument('--main-source', required=True, type=Path)
    parser.add_argument('--epoch', required=True, type=int)
    parser.add_argument('--leaf', required=True, type=int, choices=range(8))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    dispatch = json.loads(args.dispatch.read_text()); prefix = dispatch['command'][4]
    verified = io.StringIO()
    # Reuse the native source/binary/input/readback binding and complete C1
    # identity checks; no new unbound diagnostic import path is introduced.
    with contextlib.redirect_stdout(verified):
        C.run(SimpleNamespace(prefix=prefix, dispatch=args.dispatch, epoch=args.epoch,
                              main_source=args.main_source, output=None, work_energy=False))
    files = {}
    for key, suffix in (('numerical', 'numerical'), ('exchange', 'exchange'), ('tiles', 'tiles'), ('convergence', 'conv')):
        path = Path(f'{prefix}.epoch{args.epoch}.{suffix}.bin')
        data = path.read_bytes()
        C.require(dispatch['artifacts'][str(path)] == {'sha256': C.sha(path), 'bytes': len(data)}, 'changed native readback')
        files[key] = data
    source = args.main_source.read_text()
    stride = int(re.search(r'static_assert\(sizeof\(BiprismV10NumericalScratchFfi\)==(\d+)', source)[1])
    records, _ = C.inspect_buffers(**files, search=False, picard_limit=C.compiled_picard_limit(source), stride=stride)
    record = records[args.leaf]
    C.require(record['kind'] == 'equation', 'equation refusal required')
    result = exclude_frozen(record)
    H = (*record['H'][:3], record['H'][3]+record['connection'])
    C.require(C.equation(record['start'], tuple(map(float, result['centre'])), H, record['omega'], record['h']) < 1e-14,
              'independent Cayley matrix sign/scale mismatch')
    lines = [verified.getvalue().rstrip(), f'INTERVAL_CHECKER_SHA256={C.sha(__file__)}',
             f'FROZEN_MATRIX_ONLY=true COUPLED_NONLINEAR_INFEASIBILITY_PROVED=false leaf={args.leaf} vertex={record["vertex"]}',
             f'raw_record={record!r}', f'pivot={result["pivot"]} equation_envelope=1e-12 norm_envelope=1e-12',
             f'expanded_norm_interval=[{float(result["norm_lo"]):.17g},{float(result["norm_hi"]):.17g}]']
    for row in result['slices']:
        lines.append('pivot_slice '+ ' '.join(f'{k}={float(v):.17g}' if isinstance(v, Q) else f'{k}={v}' for k, v in row.items()))
    lines += [f'ALL_FP32_FROZEN_SLICES_EXCLUDED={result["all_fp32_slices_excluded"]}',
              'INVARIANT_CURRENTLY_SATISFIED=false; no native state or acceptance gate was changed']
    with args.output.open('x') as out:
        out.write('\n'.join(lines)+'\n')
    print('\n'.join(lines[-len(result['slices'])-4:]))


if __name__ == '__main__':
    main()
