#!/usr/bin/env python3
"""CPU controls of the independent checker; these fixtures are not native output."""
import copy
from decimal import Decimal, localcontext
import importlib.util
import math
from pathlib import Path
import struct
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("work_checker", HERE / "check_biprism_native_lif_work_rows.py")
C = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = C
spec.loader.exec_module(C)
passed = 0


def d(value):
    return Decimal.from_float(float(value))


def independent_decimal_values(row, dt):
    """Expanded kinetic forms in Decimal, not the checker's Fraction dot oracle."""
    with localcontext() as ctx:
        ctx.prec = 110
        inv, _, mm, mh = map(d, row["masses"])
        mass = 1 / inv
        g0, g1, g2, g3, g4, g5 = map(d, row["metric"])
        matrix = ((g0, g3, g4), (g3, g1, g5), (g4, g5, g2))
        a, b, n0, n1, vm, force = [tuple(map(d, row[k])) for k in ("v0", "v1", "n0", "n1", "vmain", "force")]
        factor = d(C.VELOCITY_TO_AU)
        na, nb = [tuple(t * factor for t in v) for v in (n0, n1)]
        def dot(x, y):
            return sum((u * v for u, v in zip(x, y)), Decimal(0))
        def mul(x):
            return tuple(dot(r, x) for r in matrix)
        def kinetic(x, m, metric=False):
            return m * dot(x, mul(x) if metric else x) / 2
        def delta(x, y):
            return tuple(v - u for u, v in zip(x, y))
        avg = tuple((u + v) / 2 for u, v in zip(a, b))
        request = tuple(d(dt) * f for f in force)
        realized = tuple(mass * v for v in delta(a, b))
        covariant = tuple(mass * v for v in mul(delta(a, b)))
        klab = kinetic(b, mass) - kinetic(a, mass)
        km = kinetic(b, mass, True) - kinetic(a, mass, True)
        nlab = kinetic(nb, mass) - kinetic(na, mass)
        nm = kinetic(nb, mass, True) - kinetic(na, mass, True)
        hmr = (kinetic(n1, mh) - kinetic(n0, mh)) / d(418.4)
        phmr = (kinetic(n1, mh) - kinetic(n0, mh)) / d(C.f32(418.4))
        core = row["identity"][2] < C.CORE
        main = kinetic(b, mm, True) - kinetic(a, mm, True) if core else Decimal(0)
        pre_true = kinetic(a, mass, True) - kinetic(vm, mass, True) if core else Decimal(0)
        pre_main = kinetic(a, mm, True) - kinetic(vm, mm, True) if core else Decimal(0)
        values = (dot(request, avg), dot(realized, avg), klab, dot(mul(request), avg), dot(covariant, avg), km,
                  main, hmr, nlab, nlab - klab, nm, nm - km, main - km if core else Decimal(0),
                  phmr, phmr - hmr, pre_true, pre_main)
        return tuple(map(float, values)), tuple(map(float, request + realized + covariant))


def pack_row(row):
    out = bytearray(C.ROW_BYTES)
    struct.pack_into("<8I", out, 0, *row["identity"])
    for key, off, fmt in (("masses", 32, "4f"), ("xau", 48, "3f"), ("xnative", 60, "3f"),
                          ("vmain", 72, "3f"), ("v0", 88, "3f"), ("v1", 100, "3f"),
                          ("n0", 112, "3f"), ("n1", 124, "3f"), ("force", 136, "3f"),
                          ("metric", 148, "6f"), ("metric", 416, "6d"), ("impulses", 176, "9d"), ("values", 248, "17d"),
                          ("xau_low", 384, "3f"), ("xnative_low", 396, "3f")):
        struct.pack_into("<" + fmt, out, off, *row[key])
    struct.pack_into("<I", out, 84, row["scope"])
    return out


def fixture(count=2, mode=3):
    rows = bytearray()
    ledgers = bytearray(count * C.LEAVES * C.LEDGER_BYTES)
    shared = bytearray(count * C.SHARED_BYTES)
    ids = [list(range(10000 * leaf, 10000 * leaf + C.VERTICES)) for leaf in range(C.LEAVES)]
    terms = [1, 2, 100, 1, 2, 100, 1, 2]
    requests = [C.f32((leaf + 1) * .00005) for leaf in range(C.LEAVES)]
    native_dt = min(requests)
    dt = C.f32(native_dt * C.TIME_TO_AU)
    snapshots = []
    time_sum = 0.
    for leaf, terminal in enumerate(terms):
        captured = terminal if count >= terminal else 0
        phase = 0 if not captured else int(C.f32(C.f32(float(terminal)) / C.f32(float(terminal))) * 1024.) & 1023
        snapshots.append({"abi_magic": C.SHARED_MAGIC, "leaf_id": leaf, "terminal_step": terminal,
                          "captured_step": captured, "temperature": 300. if captured else 0.,
                          "effective_gamma": 1. if captured else 0., "requested_dt_ps": requests[leaf] if captured else 0.,
                          "uv_wavelength_nm": 280. if captured else 0., "uv_burst_active": 0, "uv_target_idx": -1 if captured else 0,
                          "phase_bits": phase, "reserved": 0})
    for body in range(count):
        step, commit, pending = body + 1, (body + 1) * 4, int(body > 0)
        time_sum += native_dt
        off = body * C.SHARED_BYTES
        frozen = sum(1 << leaf for leaf, terminal in enumerate(terms) if step > terminal)
        struct.pack_into("<8I4fdQ8f8I", shared, off, C.SHARED_MAGIC, 2, 0, step, step, body + 1, frozen, 1,
                         native_dt, dt, native_dt, max(requests), time_sum, 77, *requests, *(min(step, t) for t in terms))
        for leaf in range(C.LEAVES):
            target, control = terms[leaf], min(step, terms[leaf])
            phase = int(C.f32(C.f32(float(control)) / C.f32(float(target))) * 1024.) & 1023
            base = (C.SHARED_MAGIC, leaf, target, control, 300., 1., requests[leaf], 280., 0, -1, phase, 0)
            struct.pack_into("<4I4f2i2I", shared, off + 128 + leaf * 96, *base)
            struct.pack_into("<4f2i2I2fiI", shared, off + 176 + leaf * 96,
                             301., 1., requests[leaf], 280., 0, -1, phase, 1, .125, 1., -1, 0)
            sums = [0.] * 26
            for vertex in range(C.VERTICES):
                sign = 1 if vertex % 2 == 0 else -1
                inv = C.f32(.03 if (vertex // 2) % 2 else .125)
                prop = inv if mode & 1 else C.f32(5e-7 if (vertex // 2) % 2 else .1)
                mass = C.round_f32_exact(1 / C.F(max(prop, C.f32(1e-6))))
                n0 = tuple(C.f32(sign * (axis + 1) * .03125) for axis in range(3))
                v0 = tuple(C.f32(v * C.VELOCITY_TO_AU) for v in n0)
                force = tuple(C.f32(sign * (axis + 1) * 2**-18) if pending else 0. for axis in range(3))
                v1 = tuple(C.round_f32_exact(C.F(a) + C.F(dt) * C.F(inv) * C.F(f)) for a, f in zip(v0, force))
                n1 = tuple(C.f32(n + (b - a) / C.VELOCITY_TO_AU) for n, a, b in zip(n0, v0, v1)) if pending else n0
                metric = (1.5, 2., 1.25, .125, .0625, -.25) if vertex < C.CORE else (1., 1., 1., 0., 0., 0.)
                vm = tuple(C.f32(a * 1.001) for a in v0) if vertex < C.CORE else v0
                row = {"identity": (ids[leaf][vertex], leaf, vertex, 3, step, commit, body if pending else 0, pending),
                       "masses": (inv, prop, mass, 12.25), "xau": (1., 2., 3.), "xnative": (.5, 1., 1.5),
                       "vmain": vm, "scope": ((1 if mode & 1 else 2) if vertex < C.CORE else 0),
                       "v0": v0, "v1": v1, "n0": n0, "n1": n1, "force": force, "metric": metric,
                       "xau_low": (C.f32(1e-8), C.f32(-2e-8), C.f32(3e-8)),
                       "xnative_low": (C.f32(-1e-8), C.f32(2e-8), C.f32(-3e-8)), "reserved": 0}
                row["values"], row["impulses"] = independent_decimal_values(row, dt)
                rows.extend(pack_row(row))
                for k, value in enumerate(row["values"] + row["impulses"]):
                    sums[k] += value
            loff = (body * C.LEAVES + leaf) * C.LEDGER_BYTES
            struct.pack_into("<12IQQd", ledgers, loff, C.WORK_MAGIC, leaf, 3, 0, step, commit,
                             body if pending else 0, body if body > 1 else 0, pending, mode, 0, 0, body + 1, body, dt)
            struct.pack_into("<17d9dQ", ledgers, loff + 72, *sums, 0)
    wi = {"schema": "biprism.lif.actual_work.v2", "window": {"abi_magic": C.WORK_MAGIC, "capacity": count,
          "first_body": 0, "requested_steps": count, "first_scheduled_step": 1, "reserved": [0, 0, 0]},
          "attempted_current_body": False, "row_bytes": C.ROW_BYTES, "ledger_bytes": C.LEDGER_BYTES,
          "reserved_row_slots": count * C.LEAVES * C.VERTICES, "ledger_slots": count * C.LEAVES,
          "summary_fields": list(C.NAMES)}
    si = {"schema": "biprism.shared_physical_time.v1", "record_bytes": C.SHARED_BYTES,
          "window": {"magic": C.SHARED_MAGIC, "capacity": count, "first_body": 0, "requested_bodies": count,
                     "first_scheduled_step": 1, "reserved": [0, 0, 0]}, "generation": 77,
          "before_time_ps": 0., "terminal_steps": terms, "terminal_snapshots": snapshots, "records": count}
    return rows, ledgers, shared, wi, si, ids


def pass_case(name, call):
    global passed
    result = call()
    passed += 1
    print(f"PASS {name}")
    return result


def refusal(name, call, match=None):
    global passed
    try:
        call()
    except C.Refusal as exc:
        if match is not None:
            assert match in str(exc), (name, match, str(exc))
        passed += 1
        print(f"PASS refused {name}: {exc}")
        return
    raise AssertionError(f"negative control accepted: {name}")


def main():
    start = time.monotonic()
    print("CONTROL_CLASS=synthetic CPU checker controls; no GPU or accepted trajectory")
    source = HERE / "check_biprism_native_lif_work_rows.py"
    print("CHECKER_SHA256=" + C.sha256_file(source))
    data = fixture()
    result = pass_case("complete two-body 8x256, anisotropic core, pending/no-pending, terminal hold", lambda: C.check_buffers(*data))
    assert result["max_exact_lab_net_momentum"] == 0 and result["full_c4_qualification_inferred"] is False
    constrained = fixture(1, 2)
    pass_case("constrained propagation mass and represented reciprocal floor", lambda: C.check_buffers(*constrained))
    # Bounded policy keeps every native request and applies the exact selected
    # minimum/cap. These are shared-journal controls, not simulated work rows.
    bounded = bytearray(constrained[2])
    bounded_index = copy.deepcopy(constrained[4])
    minimum, maximum = C.f32(.0005), C.f32(.002)
    requests = [C.f32(.003 + leaf*.0001) for leaf in range(8)]
    bounded_index["window"]["reserved"] = [2, C.f32bits(minimum), C.f32bits(maximum)]
    struct.pack_into("<I", bounded, 28, 2)
    struct.pack_into("<4f", bounded, 32, maximum, C.f32(maximum*C.TIME_TO_AU), min(requests), max(requests))
    struct.pack_into("<d", bounded, 48, maximum)
    struct.pack_into("<8f", bounded, 64, *requests)
    for leaf in range(8):
        struct.pack_into("<f", bounded, 128+leaf*96+24, requests[leaf])
        struct.pack_into("<f", bounded, 128+leaf*96+48+8, requests[leaf])
        snapshot = bounded_index["terminal_snapshots"][leaf]
        if snapshot["captured_step"]:
            snapshot["requested_dt_ps"] = requests[leaf]
    pass_case("bounded exact common cap preserves raw requests", lambda: C.check_shared(bounded, bounded_index))
    below = bytearray(bounded)
    struct.pack_into("<f", below, 64, C.f32(.00025))
    struct.pack_into("<f", below, 40, C.f32(.00025))
    refusal("bounded native request below minimum", lambda: C.check_shared(below, bounded_index), "below configured")
    wrong_cap = bytearray(bounded)
    struct.pack_into("<f", wrong_cap, 32, C.f32(.001))
    refusal("bounded chosen below cap but not exact minimum", lambda: C.check_shared(wrong_cap, bounded_index), "common minimum")
    wrong_policy = bytearray(bounded)
    struct.pack_into("<I", wrong_policy, 28, 1)
    refusal("bounded record with legacy policy", lambda: C.check_shared(wrong_policy, bounded_index), "unaccepted")
    for name, words in (("unknown bounded policy", [3, 0, 0]),
                        ("reversed bounded interval", [2, C.f32bits(maximum), C.f32bits(minimum)]),
                        ("nonfinite bounded interval", [2, C.f32bits(minimum), 0x7f800000])):
        index = copy.deepcopy(bounded_index)
        index["window"]["reserved"] = words
        refusal(name, lambda index=index: C.check_shared(bounded, index))
    row = C.parse_row(data[0], C.LEAVES * C.VERTICES * C.ROW_BYTES)
    ledger = C.parse_ledger(data[1], C.LEAVES * C.LEDGER_BYTES)
    args = (ledger, data[5][0][0], 0, 0, 2, 8)
    for k in range(17):
        corrupt = copy.deepcopy(row)
        values = list(corrupt["values"])
        values[k] += .01
        corrupt["values"] = tuple(values)
        refusal("independent work field " + C.NAMES[k], lambda c=corrupt: C.check_row(c, *args), "row work")
    for k in range(9):
        corrupt = copy.deepcopy(row)
        values = list(corrupt["impulses"])
        values[k] += .01
        corrupt["impulses"] = tuple(values)
        refusal(f"independent impulse field {k}", lambda c=corrupt: C.check_row(c, *args), "row impulse")
    for name, key, value in (("indefinite metric", "metric", (1., 1., 1., 2., 0., 0.)),
                             ("nonfinite primitive", "force", (math.nan, 0., 0.)),
                             ("nonfinite AU position low", "xau_low", (math.nan, 0., 0.)),
                             ("nonfinite native position low", "xnative_low", (0., math.inf, 0.)),
                             ("nonzero coordinate reserved field", "coordinate_reserved", (0, 1)),
                             ("wrong main mass", "masses", (row["masses"][0], row["masses"][1], 123., 12.25)),
                             ("unbracketed velocity", "v1", (1., 2., 3.)),
                             ("wrong bridge", "n0", (3., 2., 1.)),
                             ("wrong native delta export", "n1", (3., 2., 1.)),
                             ("wrong core scope", "scope", 0)):
        corrupt = copy.deepcopy(row)
        corrupt[key] = value
        refusal(name, lambda c=corrupt: C.check_row(c, *args))
    for k in range(9):
        ledgers = bytearray(data[1])
        struct.pack_into("<d", ledgers, 208 + 8*k, .01)
        refusal(f"aggregate impulse {k}", lambda l=ledgers: C.check_buffers(data[0], l, *data[2:]), "aggregate")
    corrupt_rows, corrupt_ledgers = bytearray(data[0]), bytearray(data[1])
    old = struct.unpack_from("<d", corrupt_rows, 248)[0]
    struct.pack_into("<d", corrupt_rows, 248, old + .01)
    old_sum = struct.unpack_from("<d", corrupt_ledgers, 72)[0]
    struct.pack_into("<d", corrupt_ledgers, 72, old_sum + .01)
    refusal("matching forged row and aggregate", lambda: C.check_buffers(corrupt_rows, corrupt_ledgers, *data[2:]), "row work")
    # Honest primitive-based fields and sums do not excuse an adjacent-float
    # delivery whose physical momentum exceeds the unchanged absolute gate.
    momentum_rows, momentum_ledgers = bytearray(data[0]), bytearray(data[1])
    body_offset = C.LEAVES * C.VERTICES * C.ROW_BYTES
    for vertex in (0, 1):
        off = body_offset + vertex * C.ROW_BYTES
        changed = C.parse_row(momentum_rows, off)
        sign = 1 if vertex == 0 else -1
        changed["force"] = (C.f32(sign * 32768.125), 0., 0.)
        endpoints = []
        for axis in range(3):
            target = C.F(changed["v0"][axis]) + C.F(ledger["dt"]) * C.F(changed["masses"][0]) * C.F(changed["force"][axis])
            lo, hi = C.bracket_f32(target)
            value = C.round_f32_exact(target)
            if vertex == 0 and axis == 0:
                assert lo != hi
                value = hi if value == lo else lo
            endpoints.append(value)
        changed["v1"] = tuple(endpoints)
        changed["n1"] = tuple(C.f32(n + (b-a)/C.VELOCITY_TO_AU) for n, a, b in zip(changed["n0"], changed["v0"], changed["v1"]))
        changed["values"], changed["impulses"] = independent_decimal_values(changed, ledger["dt"])
        momentum_rows[off:off+C.ROW_BYTES] = pack_row(changed)
    sums = [0.] * 26
    for vertex in range(C.VERTICES):
        rr = C.parse_row(momentum_rows, body_offset + vertex*C.ROW_BYTES)
        for k, value in enumerate(rr["values"] + rr["impulses"]):
            sums[k] += value
    struct.pack_into("<26d", momentum_ledgers, C.LEAVES*C.LEDGER_BYTES + 72, *sums)
    refusal("honest ledger but infeasible physical momentum", lambda: C.check_buffers(momentum_rows, momentum_ledgers, *data[2:]), "physical net momentum")
    # All work values are recomputed honestly after a forged native export.
    # Primitive work accounting alone would accept it; the source-expression
    # bridge check must reject it before those internally consistent numbers.
    forged = copy.deepcopy(row)
    forged["n1"] = tuple(C.f32(x + .01) for x in forged["n1"])
    forged["values"], forged["impulses"] = independent_decimal_values(forged, ledger["dt"])
    refusal("honest work for wrong native export", lambda: C.check_row(forged, *args), "kind-2 delta export")
    refusal("dropped row", lambda: C.check_buffers(data[0][:-C.ROW_BYTES], *data[1:]), "complete")
    ids = copy.deepcopy(data[5]); ids[0][1] = ids[0][0]
    refusal("duplicate selected ID", lambda: C.check_buffers(*data[:5], ids), "unique")
    ids = copy.deepcopy(data[5]); ids[0][0] += 900000
    refusal("wrong selected ID", lambda: C.check_buffers(*data[:5], ids), "identity")
    for name, offset, fmt, value in (("wrong generation", 56, "Q", 78), ("shared unaccepted", 4, "I", 1),
                                    ("wrong chosen minimum", 32, "f", C.f32(.5)),
                                    ("wrong accepted time", 48, "d", .5),
                                    ("wrong accepted count", 20, "I", 2),
                                    ("terminal baseline changed", C.SHARED_BYTES + 128 + 16, "f", 301.)):
        shared = bytearray(data[2]); struct.pack_into("<"+fmt, shared, offset, value)
        refusal(name, lambda s=shared: C.check_buffers(data[0], data[1], s, *data[3:]))
    for name, offset, fmt, value in (("wrong ledger dt", 64, "d", .5), ("wrong committed count", 20, "I", 8),
                                    ("leaf unaccepted", 8, "I", 2), ("mixed pending", 32, "I", 1)):
        ledgers = bytearray(data[1]); struct.pack_into("<"+fmt, ledgers, offset, value)
        refusal(name, lambda l=ledgers: C.check_buffers(data[0], l, *data[2:]))
    wi = copy.deepcopy(data[3]); wi["attempted_current_body"] = True
    refusal("attempted current storage", lambda: C.check_buffers(*data[:3], wi, *data[4:]), "schema/attempt")
    si = copy.deepcopy(data[4]); si["window"]["first_body"] = 1
    refusal("mismatched shared work window", lambda: C.check_buffers(*data[:4], si, data[5]), "window mismatch")
    three = fixture(3)
    pass_case("three-body delivery continuity", lambda: C.check_buffers(*three))
    broken_history = bytearray(three[1])
    struct.pack_into("<I", broken_history, 2*C.LEAVES*C.LEDGER_BYTES + 28, 1)
    refusal("discontinuous delivered-step history", lambda: C.check_buffers(three[0], broken_history, *three[2:]), "history")
    refusal("strict JSON Infinity", lambda: C.strict_json('{"v":Infinity}'), "nonfinite")
    refusal("strict JSON duplicate", lambda: C.strict_json('{"v":1,"v":2}'), "duplicate")
    with tempfile.TemporaryDirectory(prefix="biprism_work_checker_control_") as tmp:
        root = Path(tmp)
        p = root / "manifest.txt"
        p.write_text('{"schema":"biprism.lif.work.readback.v2","evidence_class":"synthetic"}')
        refusal("synthetic manifest cannot qualify native", lambda: C.check_manifest(p, C.sha256_file(p)), "actual native")
        refusal("wrong external manifest hash", lambda: C.check_manifest(p, "0" * 64), "external hash")
        artifact = root / "bytes.bin"; artifact.write_bytes(b"actual control bytes")
        entry = {"path": str(artifact), "bytes": artifact.stat().st_size, "sha256": C.sha256_file(artifact)}
        artifact.write_bytes(b"changed control byte")
        refusal("mutated bound artifact", lambda: C.verified_path(entry, root), "identity mismatch")
    pass_case("exact halfway adjacent rounding ties to even", lambda: (
        assert_equal(C.round_f32_exact((C.F(1.) + C.F(C.next_f32(1., True))) / 2), 1.),
        assert_equal(C.bracket_f32(C.F(1.)), (1., 1.))))
    print(f"CONTROLS_PASSED={passed}")
    print(f"ELAPSED_SECONDS={time.monotonic()-start:.6f}")
    print("NATIVE_WORK_ROWS_CHECKED=0")
    print("FULL_C4_QUALIFICATION_INFERRED=false")


def assert_equal(actual, expected):
    assert actual == expected, (actual, expected)


if __name__ == "__main__":
    main()
