#!/usr/bin/env python3
"""Independently check accepted native LIF work records, without CUDA execution.

Input ABI: complete body/leaf/vertex ordered 464-byte BLW3 rows, 288-byte leaf
ledgers, and the accepted 896-byte shared-step journal. All byte order is LE.
The numerical oracle uses exact rational arithmetic on represented primitives;
it does not infer physics from agreement between reported rows and summaries.

CLI requires --manifest and its separately supplied --manifest-sha256. Manifest
schema is ``biprism.lif.work.readback.v2`` with evidence_class
``actual_native_readback``. ``artifacts`` maps rows, ledgers, work_index,
shared_steps, shared_index, selected_ids, execution, build, binary, device_module to objects
{path, sha256, bytes}. selected_ids is 8*256 unpacked u32 atom IDs. ``sources``
is a list of {role,path,sha256,bytes}; required roles: work_cuda, work_host,
shared_cuda, shared_abi, shared_host, unit_bridge, main, delivery,
coordinate_precision.

``receipt_bindings`` supplies RFC6901 JSON pointers (no inferred field names):
execution_exit, execution_new_xids, execution_native, execution_binary_sha256,
execution_device_module_sha256,
execution_source_hashes, execution_output_hashes, execution_generation,
execution_first_body, execution_body_count, execution_first_scheduled_step,
build_exit, build_binary_sha256, build_device_module_sha256,
build_source_hashes. Hash populations may be
nested maps/lists; every required hash must occur as a complete string value.
The execution must record exit 0, zero new Xids, the selected binary, source
population, exact produced artifacts and window identities. This binds existing
actual receipts; it neither creates a dispatch nor upgrades synthetic controls.

No full-C4 qualification is inferred from work rows. Accepted coupled main,
DEC, spinor, endpoint and resource evidence remain separate requirements.
Control fixtures call check_buffers directly and are explicitly not native data.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import mmap
from pathlib import Path
import struct
import sys
from fractions import Fraction as F
from functools import lru_cache

ROW_BYTES, LEDGER_BYTES, SHARED_BYTES = 464, 288, 896
WORK_MAGIC, SHARED_MAGIC = 0x424C5733, 0x42505354
CORE, VERTICES, LEAVES = 192, 256, 8
VELOCITY_TO_AU = 4.571028907499172e-5
TIME_TO_AU = 41341.37333517015
MOMENTUM_LIMIT = F(1, 10_000_000)
UNIT_ROUNDOFF = F(1, 1 << 53)
GAMMA128 = 128 * UNIT_ROUNDOFF / (1 - 128 * UNIT_ROUNDOFF)
GAMMA256 = 256 * UNIT_ROUNDOFF / (1 - 256 * UNIT_ROUNDOFF)
MIN_DOUBLE = F(1, 1 << 1074)
NAMES = (
    "requested_lab_work_hartree", "realized_lab_work_hartree",
    "true_lab_kinetic_delta_hartree", "requested_true_metric_work_hartree",
    "realized_true_metric_work_hartree", "true_metric_kinetic_delta_hartree",
    "core_main_mass_metric_kinetic_delta_hartree", "native_guard_hmr_kinetic_delta_kcal_mol",
    "native_true_lab_kinetic_delta_hartree", "lab_delivery_bridge_difference_hartree",
    "native_true_metric_kinetic_delta_hartree", "metric_delivery_bridge_difference_hartree",
    "core_main_vs_true_metric_mass_difference_hartree", "native_phase_hmr_kinetic_delta_kcal_mol",
    "native_phase_vs_guard_work_difference_kcal_mol", "core_pre_lif_true_metric_gap_hartree",
    "core_pre_lif_main_metric_gap_hartree",
)


class Refusal(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise Refusal(message)


def strict_json(data):
    def duplicate_pairs(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, f"duplicate JSON key: {key}")
            out[key] = value
        return out
    def bad_constant(value):
        raise Refusal(f"nonfinite JSON constant: {value}")
    return json.loads(data, object_pairs_hook=duplicate_pairs, parse_constant=bad_constant)


def f32(value):
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def f32bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def f64bits(value):
    return struct.pack("<d", value)


def next_f32(value, upward):
    require(math.isfinite(value), "next-float input is nonfinite")
    if value == 0.0:
        return struct.unpack("<f", struct.pack("<I", 1 if upward else 0x80000001))[0]
    bits = f32bits(value)
    bits += 1 if (value > 0) == upward else -1
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def bracket_f32(value: F):
    """Exact neighbors of a rational target, correcting possible double rounding."""
    near = f32(float(value))
    require(math.isfinite(near), "requested represented target exceeds finite FP32")
    q = F(near)
    if q == value:
        return near, near
    lo, hi = (near, next_f32(near, True)) if q < value else (next_f32(near, False), near)
    require(math.isfinite(lo) and math.isfinite(hi) and F(lo) <= value <= F(hi),
            "finite adjacent FP32 target bracket unavailable")
    return lo, hi


def round_f32_exact(value: F):
    lo, hi = bracket_f32(value)
    dl, dh = value - F(lo), F(hi) - value
    if dl < dh:
        return lo
    if dh < dl:
        return hi
    return lo if not (f32bits(lo) & 1) else hi


def finite(values, label):
    require(all(math.isfinite(x) for x in values), f"nonfinite {label}")


@lru_cache(maxsize=8192)
def exact_metric(values):
    xx, yy, zz, xy, xz, yz = map(F, values)
    require(xx > 0 and xx * yy - xy * xy > 0 and
            xx * (yy * zz - yz * yz) - xy * (xy * zz - xz * yz) +
            xz * (xy * yz - yy * xz) > 0, "accepted metric is not exactly SPD")
    return ((xx, xy, xz), (xy, yy, yz), (xz, yz, zz))


def dot(a, b):
    return sum((x * y for x, y in zip(a, b)), F(0))


def mv(matrix, vector):
    return tuple(dot(row, vector) for row in matrix)


def dot_absolute(a, b):
    return sum((abs(x * y) for x, y in zip(a, b)), F(0))


def metric_absolute(matrix, vector, other):
    return sum((abs(matrix[i][j] * vector[j] * other[i])
                for i in range(3) for j in range(3)), F(0))


def parse_row(buf, offset=0):
    ints = struct.unpack_from("<8I", buf, offset)
    return {
        "identity": ints,
        "masses": struct.unpack_from("<4f", buf, offset + 32),
        "xau": struct.unpack_from("<3f", buf, offset + 48),
        "xnative": struct.unpack_from("<3f", buf, offset + 60),
        "vmain": struct.unpack_from("<3f", buf, offset + 72),
        "scope": struct.unpack_from("<I", buf, offset + 84)[0],
        "v0": struct.unpack_from("<3f", buf, offset + 88),
        "v1": struct.unpack_from("<3f", buf, offset + 100),
        "n0": struct.unpack_from("<3f", buf, offset + 112),
        "n1": struct.unpack_from("<3f", buf, offset + 124),
        "force": struct.unpack_from("<3f", buf, offset + 136),
        "metric_projection": struct.unpack_from("<6f", buf, offset + 148),
        "metric": struct.unpack_from("<6d", buf, offset + 416),
        "reserved": struct.unpack_from("<I", buf, offset + 172)[0],
        "impulses": struct.unpack_from("<9d", buf, offset + 176),
        "values": struct.unpack_from("<17d", buf, offset + 248),
        "xau_low": struct.unpack_from("<3f", buf, offset + 384),
        "xnative_low": struct.unpack_from("<3f", buf, offset + 396),
        "coordinate_reserved": struct.unpack_from("<2I", buf, offset + 408),
    }


def parse_ledger(buf, offset=0):
    return {
        "u": struct.unpack_from("<12I", buf, offset),
        "completed": struct.unpack_from("<Q", buf, offset + 48)[0],
        "body": struct.unpack_from("<Q", buf, offset + 56)[0],
        "dt": struct.unpack_from("<d", buf, offset + 64)[0],
        "values": struct.unpack_from("<17d", buf, offset + 72),
        "impulses": struct.unpack_from("<9d", buf, offset + 208),
        "reserved": struct.unpack_from("<Q", buf, offset + 280)[0],
    }


def exact_row_quantities(row, dt):
    """Physical formulas on recorded primitives, independent of reported fields.

    Each returned scale is the sum of absolute uncombined monomials. There are
    fewer than 128 rounded basic operations along every source expression path,
    including reciprocal mass and the represented-main-mass ratio. gamma_128
    times this scale bounds FP64 arithmetic (fused or unfused). A 128-subnormal
    allowance covers gradual-underflow rounding. This is an arithmetic bound,
    not an energy/momentum acceptance tolerance; momentum uses exact fractions.
    """
    inv, _, mm, mh = map(F, row["masses"])
    m = 1 / inv
    dt = F(dt)
    scale_v = F(VELOCITY_TO_AU)
    g = exact_metric(tuple(row["metric"]))
    a, b, n0, n1, vm, force = (tuple(map(F, row[k])) for k in ("v0", "v1", "n0", "n1", "vmain", "force"))
    dv = tuple(y - x for x, y in zip(a, b))
    avg = tuple((y + x) / 2 for x, y in zip(a, b))
    dn_raw = tuple(y - x for x, y in zip(n0, n1))
    navg_raw = tuple((y + x) / 2 for x, y in zip(n0, n1))
    dn = tuple(x * scale_v for x in dn_raw)
    navg = tuple(x * scale_v for x in navg_raw)
    pre = tuple(x - y for x, y in zip(a, vm))
    pre_avg = tuple((x + y) / 2 for x, y in zip(a, vm))
    req = tuple(dt * x for x in force)
    lab = tuple(m * x for x in dv)
    cov = tuple(m * x for x in mv(g, dv))
    impulse = req + lab + cov
    impulse_scale = tuple(abs(x) for x in req + lab) + tuple(
        m * sum((abs(g[i][j] * dv[j]) for j in range(3)), F(0)) for i in range(3))
    klab, kmetric = dot(lab, avg), dot(cov, avg)
    nlab = m * dot(dn, navg)
    nmetric = m * dot(mv(g, dn), navg)
    hmr = mh * dot(dn_raw, navg_raw) / F(418.4)
    hmr_phase = mh * dot(dn_raw, navg_raw) / F(f32(418.4))
    core = row["identity"][2] < CORE
    kmain = mm * dot(mv(g, dv), avg) if core else F(0)
    ptrue = m * dot(mv(g, pre), pre_avg) if core else F(0)
    pmain = mm * dot(mv(g, pre), pre_avg) if core else F(0)
    values = (dot(req, avg), klab, klab, dot(mv(g, req), avg), kmetric, kmetric,
              kmain, hmr, nlab, nlab - klab, nmetric, nmetric - kmetric,
              kmain - kmetric if core else F(0), hmr_phase, hmr_phase - hmr,
              ptrue, pmain)
    sl = m * dot_absolute(dv, avg)
    sm = m * metric_absolute(g, dv, avg)
    snl = m * dot_absolute(dn, navg)
    snm = m * metric_absolute(g, dn, navg)
    sh = mh * dot_absolute(dn_raw, navg_raw) / F(418.4)
    shp = mh * dot_absolute(dn_raw, navg_raw) / F(f32(418.4))
    skmain = mm * metric_absolute(g, dv, avg) if core else F(0)
    scales = (dot_absolute(req, avg), sl, sl, metric_absolute(g, req, avg), sm, sm,
              skmain, sh, snl, snl + sl, snm, snm + sm, skmain + sm if core else F(0),
              shp, shp + sh, m * metric_absolute(g, pre, pre_avg) if core else F(0),
              mm * metric_absolute(g, pre, pre_avg) if core else F(0))
    bounds = tuple(GAMMA128 * x + 128 * MIN_DOUBLE for x in scales)
    ibounds = tuple(GAMMA128 * x + 128 * MIN_DOUBLE for x in impulse_scale)
    return values, bounds, impulse, ibounds


def compare_quantities(actual, expected, bounds, label):
    finite(actual, label)
    max_error = F(0)
    for k, (a, e, bound) in enumerate(zip(actual, expected, bounds)):
        error = abs(F(a) - e)
        require(error <= bound, f"{label}[{k}] differs from primitive oracle: error={float(error):.17g}, arithmetic_bound={float(bound):.17g}")
        max_error = max(max_error, error)
    return max_error


def check_row(row, ledger, expected_atom, leaf, vertex, step, commit):
    atom, rl, rv, state, rs, rc, produced, pending = row["identity"]
    lu = ledger["u"]
    require((atom, rl, rv, state, rs, rc, produced, pending) ==
            (expected_atom, leaf, vertex, 3, step, commit, lu[6], lu[8]), "row population/accepted identity mismatch")
    require(row["reserved"] == 0, "nonzero row reserved field")
    require(row["coordinate_reserved"] == (0, 0), "nonzero coordinate reserved field")
    require(row["metric_projection"] == tuple(f32(x) for x in row["metric"]), "metric projection differs from FP64 authority")
    for key in ("masses", "xau", "xnative", "xau_low", "xnative_low", "vmain", "v0", "v1", "n0", "n1", "force", "metric"):
        finite(row[key], key)
    inv, prop, main_mass, hmr = row["masses"]
    require(min(inv, prop, main_mass, hmr) > 0, "nonpositive mass representation")
    require(f32bits(main_mass) == f32bits(round_f32_exact(1 / F(max(prop, f32(1e-6))))),
            "main mass does not match actual FP32 reciprocal/floor")
    if lu[9] & 1:
        require(f32bits(prop) == f32bits(inv), "flexible mode propagation inverse differs from true inverse")
    scope = (1 if lu[9] & 1 else 2) if vertex < CORE else 0
    require(row["scope"] == scope, "pre-LIF scope identity mismatch")
    if vertex >= CORE:
        require(row["metric"] == (1., 1., 1., 0., 0., 0.) and row["vmain"] == row["v0"],
                "halo metric/main-velocity contract mismatch")
    exact_metric(tuple(row["metric"]))
    for axis in range(3):
        require(row["v0"][axis] == f32(float(row["n0"][axis]) * VELOCITY_TO_AU),
                "pre-delivery AU/native bridge expression mismatch")
        dv_au = float(row["v1"][axis]) - float(row["v0"][axis])
        native_expected = row["n0"][axis] if dv_au == 0. else f32(
            float(row["n0"][axis]) + dv_au / VELOCITY_TO_AU)
        require(f32bits(row["n1"][axis]) == f32bits(native_expected),
                "native velocity does not match actual kind-2 delta export expression")
        if not pending:
            require(row["force"][axis] == 0 and row["v1"][axis] == row["v0"][axis] and
                    row["n1"][axis] == row["n0"][axis], "nonpending delivery changed state/force")
        target = F(row["v0"][axis]) + F(ledger["dt"]) * F(inv) * F(row["force"][axis])
        lo, hi = bracket_f32(target)
        require(row["v1"][axis] in (lo, hi), "delivered AU velocity is outside exact adjacent target bracket")
    expected, bounds, impulse, ibounds = exact_row_quantities(row, ledger["dt"])
    ev = compare_quantities(row["values"], expected, bounds, "row work")
    ei = compare_quantities(row["impulses"], impulse, ibounds, "row impulse")
    return expected, bounds, impulse, ibounds, max(ev, ei)


def terminal_bytes(t):
    return struct.pack("<4I4f2i2I", t["abi_magic"], t["leaf_id"], t["terminal_step"], t["captured_step"],
                       t["temperature"], t["effective_gamma"], t["requested_dt_ps"], t["uv_wavelength_nm"],
                       t["uv_burst_active"], t["uv_target_idx"], t["phase_bits"], t["reserved"])


def check_shared(buf, index):
    require(index["schema"] == "biprism.shared_physical_time.v1" and index["record_bytes"] == SHARED_BYTES,
            "shared journal schema/stride mismatch")
    w = index["window"]
    n = w["requested_bodies"]
    require(w["magic"] == SHARED_MAGIC and 1 <= n <= w["capacity"] <= 500,
            "invalid shared journal window")
    words = w["reserved"]
    require(len(words) == 3 and all(type(x) is int and 0 <= x <= 0xffffffff for x in words),
            "invalid shared selection words")
    bounds = None
    if words != [0, 0, 0]:
        require(words[0] == 2, "unknown shared selection policy")
        bounds = struct.unpack("<2f", struct.pack("<2I", *words[1:]))
        finite(bounds, "shared configured bounds")
        require(0 < bounds[0] <= bounds[1], "invalid shared configured bounds")
    require(len(buf) == n * SHARED_BYTES and index["records"] == n, "incomplete shared journal")
    require(isinstance(index["generation"], int) and index["generation"] > 0, "invalid shared generation")
    terms, snapshots = index["terminal_steps"], index["terminal_snapshots"]
    require(len(terms) == len(snapshots) == LEAVES and all(isinstance(t, int) and 0 < t <= 0x7fffffff for t in terms),
            "invalid terminal population")
    last_step = w["first_scheduled_step"] + n - 1
    require(0 < w["first_scheduled_step"] <= last_step < 0xffffffff, "invalid scheduled step range")
    terminal = []
    for leaf, (t, target) in enumerate(zip(snapshots, terms)):
        require((t["abi_magic"], t["leaf_id"], t["terminal_step"], t["captured_step"], t["reserved"]) ==
                (SHARED_MAGIC, leaf, target, target if last_step >= target else 0, 0), "terminal snapshot identity mismatch")
        terminal.append(terminal_bytes(t))
    time = index["before_time_ps"]
    require(math.isfinite(time) and time >= 0, "nonfinite/negative initial accepted time")
    if w["first_body"] == 0:
        require(time == 0 and w["first_scheduled_step"] == 1, "cold accepted chronology mismatch")
    selected = []
    for i in range(n):
        off, step, body = i * SHARED_BYTES, w["first_scheduled_step"] + i, w["first_body"] + i + 1
        u = struct.unpack_from("<8I", buf, off)
        native, au, lo, hi = struct.unpack_from("<4f", buf, off + 32)
        cumul, gen = struct.unpack_from("<dQ", buf, off + 48)
        requests = struct.unpack_from("<8f", buf, off + 64)
        controls = struct.unpack_from("<8I", buf, off + 96)
        require(body <= 0x3fffffff and u[:6] == (SHARED_MAGIC, 2, 0, step, step, body)
                and u[7] == (2 if bounds is not None else 1) and gen == index["generation"],
                "unaccepted/wrong-generation shared body")
        finite((native, au, lo, hi, cumul) + requests, "shared dt/time")
        require(lo == min(requests) and hi == max(requests) and lo > 0,
                "raw shared request extrema mismatch")
        if bounds is not None:
            require(lo >= bounds[0], "native safety request below configured minimum")
        chosen = min(lo, bounds[1]) if bounds is not None else lo
        require(f32bits(native) == f32bits(chosen) and
                f32bits(au) == f32bits(f32(float(native) * TIME_TO_AU)), "shared dt is not the represented common minimum")
        next_time = time + float(native)
        require(next_time > time and f64bits(cumul) == f64bits(next_time), "accepted physical time mismatch")
        time = next_time
        mask = 0
        for leaf in range(LEAVES):
            p = off + 128 + leaf * 96
            b = struct.unpack_from("<4I4f2i2I", buf, p)
            c = struct.unpack_from("<4f2i2I2fiI", buf, p + 48)
            control = min(step, terms[leaf])
            phase = int(f32(f32(float(control)) / f32(float(terms[leaf]))) * 1024.) & 0x3ff
            require(b[:4] == (SHARED_MAGIC, leaf, terms[leaf], control) and b[11] == c[11] == 0 and controls[leaf] == control,
                    "protocol control identity mismatch")
            require(b[10] == c[6] == phase and c[2] == b[6] == requests[leaf] and c[1] == b[5] and
                    c[3] == b[7] and c[4:6] == b[8:10] and b[8] in (0, 1), "baseline/current protocol mismatch")
            finite(b[4:8] + c[:4] + c[8:10], "protocol controls")
            require(b[4] > 0 and c[0] > 0 and b[5] >= 0, "inadmissible protocol controls")
            if step >= terms[leaf]:
                require(bytes(buf[p:p+48]) == terminal[leaf], "terminal exogenous baseline changed")
            if step > terms[leaf]:
                mask |= 1 << leaf
        require(u[6] == mask, "terminal held mask mismatch")
        selected.append(au)
    return selected


def check_buffers(rows, ledgers, shared, work_index, shared_index, selected_ids):
    require(work_index["schema"] == "biprism.lif.actual_work.v2" and not work_index["attempted_current_body"] and
            work_index["row_bytes"] == ROW_BYTES and work_index["ledger_bytes"] == LEDGER_BYTES,
            "work schema/attempt/stride mismatch")
    require(work_index.get("summary_fields") == list(NAMES), "work scalar field population mismatch")
    w, sw = work_index["window"], shared_index["window"]
    require(w is not None and w["abi_magic"] == WORK_MAGIC and w["reserved"] == [0, 0, 0], "invalid work window")
    require((w["capacity"], w["first_body"], w["requested_steps"], w["first_scheduled_step"]) ==
            (sw["capacity"], sw["first_body"], sw["requested_bodies"], sw["first_scheduled_step"]), "work/time window mismatch")
    dts = check_shared(shared, shared_index)
    n = len(dts)
    require(len(rows) == n * LEAVES * VERTICES * ROW_BYTES and len(ledgers) == n * LEAVES * LEDGER_BYTES,
            "work journal is not complete 8x256 population")
    require(work_index["reserved_row_slots"] == n * LEAVES * VERTICES and work_index["ledger_slots"] == n * LEAVES,
            "work index population mismatch")
    require(len(selected_ids) == LEAVES and all(len(x) == VERTICES and len(set(x)) == VERTICES for x in selected_ids),
            "selected atom population is not unique within each leaf")
    max_error, max_dp = F(0), F(0)
    largest_requested_realized_difference = F(0)
    mode_seen = None
    delivered_after = [None] * LEAVES
    previous_produced = [None] * LEAVES
    for i, dt in enumerate(dts):
        pending_seen = None
        for leaf in range(LEAVES):
            ledger = parse_ledger(ledgers, (i * LEAVES + leaf) * LEDGER_BYTES)
            u = ledger["u"]
            body, step = w["first_body"] + i, w["first_scheduled_step"] + i
            commit = (body + 1) * 4
            require(u[:6] == (WORK_MAGIC, leaf, 3, 0, step, commit) and u[8] in (0, 1) and u[9] <= 3 and
                    u[10:] == (0, 0) and ledger["reserved"] == 0 and ledger["body"] == body and
                    ledger["completed"] == body + 1 and f64bits(ledger["dt"]) == f64bits(float(dt)), "leaf ledger accepted identity mismatch")
            if mode_seen is None:
                mode_seen = u[9]
            require(u[9] == mode_seen, "integration mode changed inside the journal")
            if pending_seen is None:
                pending_seen = u[8]
            require(u[8] == pending_seen, "mixed eight-leaf pending delivery")
            require(u[7] < step and (u[6] + 1 == step if u[8] else u[6] <= step), "carry production/delivery chronology mismatch")
            if body == 0:
                require(u[6] == u[7] == u[8] == 0, "cold body has an impossible earlier delivery")
            if delivered_after[leaf] is not None:
                require(u[7] == delivered_after[leaf] and u[6] >= previous_produced[leaf],
                        "delivery history does not continue the preceding accepted body")
            delivered_after[leaf] = step if u[8] else u[7]
            previous_produced[leaf] = u[6]
            exact_sum = [F(0)] * 26
            bound_sum = [F(0)] * 26
            abs_reported = [F(0)] * 26
            reported_sum = [0.] * 26
            for vertex in range(VERTICES):
                off = ((i * LEAVES + leaf) * VERTICES + vertex) * ROW_BYTES
                row = parse_row(rows, off)
                try:
                    vals, bounds, impulse, ibounds, error = check_row(row, ledger, selected_ids[leaf][vertex], leaf, vertex, step, commit)
                except Refusal as exc:
                    raise Refusal(f"body={body} leaf={leaf} vertex={vertex}: {exc}") from exc
                max_error = max(max_error, error)
                largest_requested_realized_difference = max(largest_requested_realized_difference, abs(vals[0] - vals[2]))
                for k, (e, b, a) in enumerate(zip(vals + impulse, bounds + ibounds, row["values"] + row["impulses"])):
                    exact_sum[k] += e
                    bound_sum[k] += b
                    abs_reported[k] += abs(F(a))
                    reported_sum[k] += a
            for axis in range(3):
                momentum = abs(exact_sum[20 + axis])
                require(momentum <= MOMENTUM_LIMIT, f"body={body} leaf={leaf} axis={axis}: exact physical net momentum exceeds 1e-7: {float(momentum):.17g}")
                max_dp = max(max_dp, momentum)
            actual = ledger["values"] + ledger["impulses"]
            finite(actual, "aggregate work/impulse")
            for k, (a, summed) in enumerate(zip(actual, reported_sum)):
                require(f64bits(a) == f64bits(summed), f"body={body} leaf={leaf}: aggregate[{k}] differs from actual ordered row sum")
            aggregate_bounds = [b + GAMMA256 * s + 256 * MIN_DOUBLE for b, s in zip(bound_sum, abs_reported)]
            max_error = max(max_error, compare_quantities(actual, exact_sum, aggregate_bounds, "aggregate primitive work/impulse"))
    return {
        "bodies": n, "rows": n * LEAVES * VERTICES, "ledgers": n * LEAVES,
        "generation": shared_index["generation"], "first_body": w["first_body"],
        "first_scheduled_step": w["first_scheduled_step"],
        "max_exact_lab_net_momentum": float(max_dp),
        "max_reported_vs_exact_primitive_error": float(max_error),
        "max_row_requested_vs_realized_lab_work": float(largest_requested_realized_difference),
        "full_c4_qualification_inferred": False,
    }


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verified_path(entry, base):
    path = Path(entry["path"])
    if not path.is_absolute():
        path = base / path
    require(path.is_file(), f"required artifact missing: {path}")
    require(path.stat().st_size == entry["bytes"] and sha256_file(path) == entry["sha256"], f"artifact identity mismatch: {path}")
    return path


def pointer(obj, location):
    require(isinstance(location, str) and (location == "" or location.startswith("/")), "invalid receipt JSON pointer")
    try:
        for part in location.split("/")[1:]:
            key = part.replace("~1", "/").replace("~0", "~")
            obj = obj[int(key)] if isinstance(obj, list) else obj[key]
        return obj
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        raise Refusal(f"receipt field missing: {location}") from exc


def string_leaves(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from string_leaves(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from string_leaves(item)


def check_manifest(path, expected_hash):
    require(sha256_file(path) == expected_hash, "manifest external hash mismatch")
    manifest = strict_json(path.read_text())
    require(manifest["schema"] == "biprism.lif.work.readback.v2" and manifest["evidence_class"] == "actual_native_readback",
            "manifest does not designate actual native readback")
    roles = {"work_cuda", "work_host", "shared_cuda", "shared_abi", "shared_host", "unit_bridge", "main", "delivery", "coordinate_precision"}
    require(len(manifest["sources"]) >= len(roles) and roles <= {x["role"] for x in manifest["sources"]}, "required source identities missing")
    require(len({x["role"] for x in manifest["sources"]}) == len(manifest["sources"]), "duplicate source role")
    for source in manifest["sources"]:
        verified_path(source, path.parent)
    artifacts = manifest["artifacts"]
    expected_roles = {"rows", "ledgers", "work_index", "shared_steps", "shared_index", "selected_ids", "execution", "build", "binary", "device_module"}
    require(expected_roles <= artifacts.keys(), "required readback/dispatch/build artifact missing")
    paths = {role: verified_path(artifacts[role], path.parent) for role in expected_roles}
    execution, build = (strict_json(paths[k].read_text()) for k in ("execution", "build"))
    bindings = manifest["receipt_bindings"]
    def bound(name):
        return pointer(execution if name.startswith("execution_") else build, bindings[name])
    for name in ("execution_exit", "build_exit", "execution_new_xids"):
        value = bound(name)
        require(isinstance(value, int) and not isinstance(value, bool) and value == 0, f"native build/dispatch receipt failed: {name}")
    require(bound("execution_native") is True, "execution receipt does not identify actual native execution")
    binary_hash = artifacts["binary"]["sha256"]
    require(bound("execution_binary_sha256") == binary_hash == bound("build_binary_sha256"), "build/dispatch binary identity mismatch")
    module_hash = artifacts["device_module"]["sha256"]
    require(bound("execution_device_module_sha256") == module_hash == bound("build_device_module_sha256"),
            "build/dispatch device-module identity mismatch")
    source_hashes = {x["sha256"] for x in manifest["sources"]}
    require(source_hashes <= set(string_leaves(bound("build_source_hashes"))) and
            source_hashes <= set(string_leaves(bound("execution_source_hashes"))), "build/dispatch source population incomplete")
    outputs = {artifacts[k]["sha256"] for k in ("rows", "ledgers", "work_index", "shared_steps", "shared_index", "selected_ids")}
    require(outputs <= set(string_leaves(bound("execution_output_hashes"))), "dispatch does not bind actual complete output population")
    wi, si = (strict_json(paths[k].read_text()) for k in ("work_index", "shared_index"))
    expected = {"execution_generation": si["generation"], "execution_first_body": si["window"]["first_body"],
                "execution_body_count": si["window"]["requested_bodies"],
                "execution_first_scheduled_step": si["window"]["first_scheduled_step"]}
    for key, value in expected.items():
        require(bound(key) == value, f"dispatch/window identity mismatch: {key}")
    ids_bytes = paths["selected_ids"].read_bytes()
    require(len(ids_bytes) == LEAVES * VERTICES * 4, "selected ID binary extent mismatch")
    ids_flat = struct.unpack("<2048I", ids_bytes)
    ids = [ids_flat[i*VERTICES:(i+1)*VERTICES] for i in range(LEAVES)]
    with contextlib.ExitStack() as stack:
        buffers = []
        for role in ("rows", "ledgers", "shared_steps"):
            file = stack.enter_context(paths[role].open("rb"))
            require(paths[role].stat().st_size > 0, f"empty actual artifact: {role}")
            buffers.append(stack.enter_context(mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)))
        result = check_buffers(*buffers, wi, si, ids)
    # Refuse a source/output mutation that occurred while the independent pass ran.
    for source in manifest["sources"]:
        verified_path(source, path.parent)
    for role in expected_roles:
        verified_path(artifacts[role], path.parent)
    require(sha256_file(path) == expected_hash, "manifest changed during verification")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, help="Optional plain-text execution log; stdout is always emitted.")
    args = parser.parse_args(argv)
    try:
        result = check_manifest(args.manifest, args.manifest_sha256)
        lines = ["PASS: independent native LIF work readback", *(f"{k}={v}" for k, v in result.items()),
                 "scope=complete accepted work/time journal, exact primitive arithmetic and physical lab momentum",
                 "not_inferred=full C4 dynamics, main energy/spinor/DEC acceptance, restart, sampling or model validity"]
        code = 0
    except (Refusal, KeyError, TypeError, ValueError, OSError, struct.error, OverflowError) as exc:
        lines = [f"REFUSED: {exc}", "No native work-owner qualification was issued."]
        code = 1
    output = "\n".join(lines) + "\n"
    sys.stdout.write(output)
    if args.output:
        require(not any(x in args.output.name for x in ("_shadow.json", "_proposal.json", "_adjudication.json")), "prohibited output name")
        args.output.write_text(output)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
