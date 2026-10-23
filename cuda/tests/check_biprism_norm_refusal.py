#!/usr/bin/env python3
"""Reconstruct NRF1/NRF2/NRF3 refusals using their exact initializer contracts.

This is an offline representation oracle, not trajectory acceptance. It replays
the original 16 Cartesian brackets, then optionally explores outward neighbors
without normalizing a spinor or changing either 1e-7 norm bound. A candidate
found here still requires the actual coupled equation, charge/DEC, work and
energy admission in the native solver. NRF1 records original16 votes; NRF2
records votes after the conditional one-outward-neighbor fallback too. NRF3
also requires the absolute-one norm bound and conditionally completes the32
one-pivot lattice slices. Historical captures retain their original semantics.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import itertools
import math
from pathlib import Path
import struct
from fractions import Fraction

CORE, LEAVES, STRIDE = 192, 8, 38656
TAG, TAG_V2, TAG_V3 = 0x4E524631, 0x4E524632, 0x4E524633
TOL = 1e-7
LIBM = ctypes.CDLL("libm.so.6")
LIBM.fma.argtypes = [ctypes.c_double] * 3
LIBM.fma.restype = ctypes.c_double


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def next32(x, up):
    if x == 0.0:
        return struct.unpack("<f", struct.pack("<I", 1 if up else 0x80000001))[0]
    return struct.unpack("<f", struct.pack("<I", bits(x) + (1 if (x > 0) == up else -1)))[0]


def bracket(x):
    n = f32(x)
    if n == x:
        return n, n
    return (n, next32(n, True)) if n < x else (next32(n, False), n)


def norm(p):
    return LIBM.fma(p[0], p[0], p[1] * p[1]) + LIBM.fma(p[2], p[2], p[3] * p[3])


def exact_norm(p):
    return sum((Fraction(x) ** 2 for x in p), Fraction(0))


def admitted(p, n, outer, *, absolute_one=False):
    q = norm(p)
    return (math.isfinite(q) and abs(q - n) <= TOL and abs(q - outer) <= TOL
            and (not absolute_one or abs(q - 1.0) <= TOL))


def distance(p, trial):
    a, b, c, d = [x - y for x, y in zip(p, trial)]
    return LIBM.fma(a, a, LIBM.fma(b, b, LIBM.fma(c, c, d * d)))


def candidates(trial, shell):
    axes = []
    for x in trial:
        lo, hi = bracket(x)
        values = [lo, hi]
        for _ in range(shell):
            lo, hi = next32(lo, False), next32(hi, True)
            values.extend((lo, hi))
        unique = {bits(v): v for v in values if math.isfinite(v)}
        # Keep exact/bracket endpoints first. If the squared distance of a tiny
        # extra component underflows, the tie must not introduce that component
        # ahead of an already represented zero.
        axes.append(list(unique.values()))
    # Preserve the original bit-enumeration order for the zero-shell replay.
    if shell == 0:
        bounds = [bracket(x) for x in trial]
        for mask in range(16):
            yield tuple(bounds[i][(mask >> i) & 1] for i in range(4))
    else:
        yield from itertools.product(*axes)


def select(trial, n, outer, shell, *, absolute_one=False):
    best = None
    best_error = math.inf
    count = 0
    for p in candidates(trial, shell):
        if admitted(p, n, outer, absolute_one=absolute_one):
            count += 1
            error = distance(p, trial)
            if error < best_error:
                best, best_error = p, error
    return best, best_error, count


def select_shell1_source_order(trial, n, outer, *, absolute_one=False):
    """Independent replay of the native256-case order, including exact ties."""
    bounds = [bracket(x) for x in trial]
    axes = [(next32(lo, False), lo, hi, next32(hi, True)) for lo, hi in bounds]
    best, best_error, count = None, math.inf, 0
    for mask in range(256):
        p = tuple(axes[i][(mask >> (2 * i)) & 3] for i in range(4))
        if admitted(p, n, outer, absolute_one=absolute_one):
            count += 1
            error = distance(p, trial)
            if error < best_error:
                best, best_error = p, error
    return best, best_error, count


def capture_tag(raw):
    if len(raw) != LEAVES * STRIDE:
        raise ValueError("NRF1/NRF2/NRF3 requires all eight complete 38656-byte banks")
    tag = struct.unpack_from("<d", raw, 30720 + 29 * 8)[0]
    if tag not in (TAG, TAG_V2, TAG_V3):
        raise ValueError("leaf0: missing exact NRF1/NRF2/NRF3 tag")
    return int(tag)


def parse(raw):
    expected_tag = capture_tag(raw)
    rows = []
    for leaf in range(LEAVES):
        base = leaf * STRIDE
        tag = struct.unpack_from("<d", raw, base + 30720 + 29 * 8)[0]
        if tag != expected_tag:
            raise ValueError(f"leaf {leaf}: missing or inconsistent NRF1/NRF2/NRF3 tag")
        for vertex in range(CORE):
            trial = tuple(struct.unpack_from("<d", raw, base + off + vertex * 8)[0]
                          for off in (15360, 16896, 18432, 19968))
            n, outer = (struct.unpack_from("<d", raw, base + off + vertex * 8)[0]
                        for off in (21504, 23040))
            start = tuple(struct.unpack_from("<f", raw, base + off + vertex * 4)[0]
                          for off in (9216, 9984, 10752, 11520))
            flag = struct.unpack_from("<I", raw, base + 8448 + vertex * 4)[0]
            if not all(math.isfinite(x) for x in trial + start + (n, outer)):
                raise ValueError(f"leaf {leaf} vertex {vertex}: nonfinite capture")
            if struct.pack("<d", norm(start)) != struct.pack("<d", n):
                raise ValueError(f"leaf {leaf} vertex {vertex}: start norm mismatch")
            if flag not in (0, 1):
                raise ValueError(f"leaf {leaf} vertex {vertex}: invalid original vote")
            rows.append((leaf, vertex, trial, start, n, outer, flag))
    return rows


def inspect(raw, max_shell):
    rows = parse(raw)
    tag = capture_tag(raw)
    schema = {TAG:"NRF1", TAG_V2:"NRF2", TAG_V3:"NRF3"}[tag]
    absolute_one = tag == TAG_V3
    mode = {TAG:"original16", TAG_V2:"original16_then_conditional_shell1",
            TAG_V3:"triple_bound_original16_then_shell1_then_32_pivot_slices"}[tag]
    lines = [f"SCOPE=actual {schema} raw-state replay; bounded norm feasibility only",
             "INITIALIZER_MODE=" + mode,
             "NATIVE_TRAJECTORY_ACCEPTANCE=false", f"ROWS={len(rows)}",
             "NORM_LIMIT=1e-7 unchanged; source FP64 evaluation plus independent exact dyadic audit"]
    original_failures, failures, native_repaired, repaired, unresolved = 0, 0, 0, 0, 0
    pivot_repaired = 0
    max_trial_defect = 0.0
    for leaf, vertex, trial, start, n, outer, flag in rows:
        chosen, err, count = select(trial, n, outer, 0, absolute_one=absolute_one)
        if chosen is None:
            original_failures += 1
            if tag in (TAG_V2, TAG_V3):
                chosen, err, count = select_shell1_source_order(trial, n, outer, absolute_one=absolute_one)
                native_repaired += int(chosen is not None)
            if chosen is None and tag == TAG_V3:
                # Lazy import avoids a module cycle: the independent lattice
                # reference deliberately reuses only the scalar FMA primitives.
                from biprism_norm_lattice_reference import complete_norm_lattice
                chosen, err, _ = complete_norm_lattice(trial, n, outer)
                pivot_repaired += int(chosen is not None)
        if bool(chosen is not None) != bool(flag):
            raise ValueError(f"leaf {leaf} vertex {vertex}: {schema} initializer vote disagrees")
        max_trial_defect = max(max_trial_defect, abs(norm(trial) - n))
        if chosen is not None:
            continue
        failures += 1
        lines.append(f"FAILED leaf={leaf} vertex={vertex} n={n:.17g} outer={outer:.17g} "
                     f"trial_norm={norm(trial):.17g} trial={tuple(x.hex() for x in trial)} "
                     f"start_bits={tuple(hex(bits(x)) for x in start)}")
        if tag == TAG_V3:
            lines.append(f"PIVOT_SLICES leaf={leaf} vertex={vertex} slices=32 feasible=0 "
                         "full_4D_lattice_or_nonlinear_infeasibility_not_proved=true")
            unresolved += 1
            continue
        found = False
        for shell in range(1 if tag == TAG else 2, max_shell + 1):
            candidate, error, feasible_count = select(trial, n, outer, shell)
            if candidate is None:
                lines.append(f"SHELL leaf={leaf} vertex={vertex} radius={shell} feasible=0")
                continue
            exact = exact_norm(candidate)
            exact_n, exact_o = exact_norm(start), Fraction(outer)
            exact_defect = max(abs(exact - exact_n), abs(exact - exact_o))
            lines.append(f"SHELL leaf={leaf} vertex={vertex} radius={shell} feasible={feasible_count} "
                         f"distance2={error:.17g} max_component_change={max(abs(a-b) for a,b in zip(candidate,trial)):.17g} "
                         f"candidate_bits={tuple(hex(bits(x)) for x in candidate)} "
                         f"source_norm={norm(candidate):.17g} exact_joint_norm_defect={float(exact_defect):.17g} "
                         f"exact_joint_norm_within_decimal_1e-7={exact_defect <= Fraction(1,10_000_000)}")
            found = True
            break
        repaired += int(found)
        unresolved += int(not found)
    lines.extend((f"ORIGINAL_16_CORNER_FAILURES={original_failures}",
                  f"ALREADY_ADMITTED_BY_NATIVE_SHELL1={native_repaired}",
                  f"CAPTURED_INITIALIZER_FAILURES={failures}",
                  f"BOUNDED_NORM_CANDIDATES_FOUND={repaired}",
                  f"UNRESOLVED_WITHIN_SEARCH={unresolved}",
                  f"MAX_CONTINUOUS_TRIAL_NORM_CHANGE={max_trial_defect:.17g}",
                  "COUPLED_EQUATION_WORK_ENERGY_NOT_EVALUATED=true"))
    if tag == TAG_V3:
        lines.append(f"ALREADY_ADMITTED_BY_NATIVE_PIVOT_COMPLETION={pivot_repaired}")
        lines.append("ABSOLUTE_ONE_NORM_LIMIT=1e-7 unchanged")
    return "\n".join(lines) + "\n"


def inspect_fp64(raw,spinors):
    import check_biprism_spinor_owner as S
    stride=len(raw)//8
    stride=len(raw)//8
    if len(raw)!=8*stride or stride not in (106752,196992,198144,235136,349056,496768,510720,1680768,1697920,1784448,1837696,1997440):raise ValueError('NRF4 numerical extent')
    lines=['schema=NRF4 FP64 stored initializer; no FP32 candidate search'];failures=0
    for leaf in range(8):
        base=leaf*stride
        if struct.unpack_from('<d',raw,base+30720+29*8)[0]!=0x4e524634:raise ValueError('NRF4 all-leaf tag')
        owner=S.read(spinors,leaf)
        for u in range(192):
            trial=tuple(struct.unpack_from('<d',raw,base+off+u*8)[0] for off in (15360,16896,18432,19968))
            start=owner['start'][u];outer=norm(owner['accepted'][u])
            n=struct.unpack_from('<d',raw,base+21504+8*u)[0]
            saved_outer=struct.unpack_from('<d',raw,base+23040+8*u)[0]
            flag=struct.unpack_from('<I',raw,base+8448+4*u)[0]
            if norm(start)!=n or outer!=saved_outer:raise ValueError('NRF4 exact owner/norm binding')
            expected=admitted(trial,n,outer,absolute_one=True)
            if flag!=int(expected):raise ValueError('NRF4 source predicate mismatch')
            if not expected:
                failures+=1;lines.append(f'leaf={leaf} vertex={u} start_hex={[x.hex() for x in start]} trial_hex={[x.hex() for x in trial]} norm={norm(trial):.17g} start_norm={n:.17g} outer_norm={outer:.17g}')
    return '\n'.join(lines+[f'FP64_INITIALIZER_FAILURES={failures}','COUPLED_ACCEPTANCE=false'])+'\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--numerical", type=Path, required=True)
    p.add_argument("--spinors",type=Path)
    p.add_argument("--max-shell", type=int, default=2, choices=range(1, 5))
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    raw = args.numerical.read_bytes()
    text = (f"NUMERICAL_PATH={args.numerical.resolve()}\nNUMERICAL_SHA256={hashlib.sha256(raw).hexdigest()}\n"
            f"CHECKER_SHA256={hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}\n" + (inspect_fp64(raw,args.spinors.read_bytes()) if args.spinors else inspect(raw,args.max_shell)))
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
