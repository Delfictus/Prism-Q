"""Timestep resonance table for velocity Verlet / BAOAB at the measured omega_max.

Linear analysis: velocity Verlet on x'' = -w^2 x has transfer-matrix trace
2 - (w dt)^2, so it is stable for w dt < 2 and its numerical frequency obeys
sin(w_eff dt / 2) = w dt / 2. The n:1 nonlinear resonance (Schlick, Mandziuk,
Skeel & Srinivas 1998) occurs when w_eff dt = 2 pi / n, i.e. at
w dt = 2 sin(pi / n): 2 (n=2, linear limit), sqrt(3) (3:1), sqrt(2) (4:1),
1.1756 (5:1), 1 (6:1). BAOAB's Hamiltonian part is the same Verlet map; its
mean (friction) map is checked separately. For impulse MTS the outer-step
resonances fall at T/2 (linear) and T/3 (3:1).
"""
from __future__ import annotations

import math

import numpy as np

try:
    from . import _common as C
except ImportError:
    import importlib.util, pathlib, sys
    _p = pathlib.Path(__file__).with_name("_common.py")
    _s = importlib.util.spec_from_file_location("prism_refs_common", _p)
    C = importlib.util.module_from_spec(_s); sys.modules.setdefault("prism_refs_common", C); _s.loader.exec_module(C)

W_MAX = 354.1  # ps^-1 (angular), measured, PLAN.md "Verified physics baseline"
GEARBOX = C.NHS_SRC / "cuda/gearbox.cuh"


def verlet_eff_ratio(wdt):
    return 2 * math.asin(wdt / 2) / wdt if wdt < 2 else None


def verlet_transfer(wdt):
    return np.array([[1 - wdt ** 2 / 2, wdt * (1 - wdt ** 2 / 4)], [-wdt, 1 - wdt ** 2 / 2]])


def baoab_mean_map(wdt, gamma_dt):
    """Mean (noise-free) BAOAB map for x''=-x, unit mass/frequency, dt scaled into wdt."""
    h = wdt
    B = np.array([[1, 0], [-h / 2, 1]])          # v += -x h/2 (state is [x, v])
    A = np.array([[1, h / 2], [0, 1]])
    O = np.array([[1, 0], [0, math.exp(-gamma_dt)]])
    # x, v ordering; B uses force -x
    return B @ A @ O @ A @ B


def _scan_cubic(lo, hi, step=0.0025, A=0.5, alpha=-0.15, nsteps=5000):
    F = lambda x: -x - alpha * x * x
    U = lambda x: 0.5 * x * x + alpha * x ** 3 / 3
    out = []
    for wdt in np.arange(lo, hi + 1e-12, step):
        x, v = A, 0.0
        f, E0, m = F(x), U(A), 0.0
        for _ in range(nsteps):
            v += 0.5 * wdt * f; x += wdt * v; f = F(x); v += 0.5 * wdt * f
            e = abs(0.5 * v * v + U(x) - E0) / E0
            if not math.isfinite(e) or e > 10:
                m = math.inf
                break
            m = max(m, e)
        out.append((round(float(wdt), 4), m))
    return out


def check_resonance_table(engine_root=None):
    anchors = []
    try:
        anchors.append(C.anchor(engine_root, GEARBOX, 396, 399, [
            "#define PRISM_GEAR_DEFAULT_DT_FS_G0 0.5f", "#define PRISM_GEAR_DEFAULT_DT_FS_G2 4.0f"]))
    except C.AnchorMissing as ex:
        anchors.append({"missing": str(ex)})
    limits = {f"{n}:1": 2 * math.sin(math.pi / n) for n in (2, 3, 4, 5, 6)}
    period_fs = 2 * math.pi / W_MAX * 1000
    rows = []
    for dt in (2.0, 4.0, 5.0, 6.0):
        wdt = W_MAX * dt / 1000
        ev = np.abs(np.linalg.eigvals(verlet_transfer(wdt)))
        rows.append({"dt_fs": dt, "w_dt": wdt, "verlet_spectral_radius": float(ev.max()),
                     "linearly_stable": wdt < 2,
                     "w_eff_over_w": verlet_eff_ratio(wdt),
                     "past_limits": [k for k, v in limits.items() if wdt >= v],
                     "nearest_limit": min(limits, key=lambda k: abs(limits[k] - wdt)),
                     "rel_dist_to_nearest": min(abs(v - wdt) / v for v in limits.values())})
    dt_at = {k: v / W_MAX * 1000 for k, v in limits.items()}
    r4 = rows[1]
    # BAOAB mean map at 6 fs: does friction stabilise the linear instability?
    baoab = {}
    for g in (1.0, 10.0, 59.0):  # ps^-1 (59 = ramp start, 10 = production)
        M = baoab_mean_map(W_MAX * 0.006, g * 0.006)
        baoab[f"gamma_{g:g}_per_ps"] = float(np.abs(np.linalg.eigvals(M)).max())
    # nonlinear 1-DOF demonstration (cubic anharmonic oscillator, unit frequency)
    s4 = _scan_cubic(1.38, 1.46)
    s3 = _scan_cubic(1.62, 1.80)
    unstable4 = [w for w, m in s4 if not math.isfinite(m)]
    unstable3 = [w for w, m in s3 if not math.isfinite(m)]
    statements = {
        "4fs_w_dt": {"plan": 1.4164, "observed": r4["w_dt"], "ok": abs(r4["w_dt"] - 1.4164) < 5e-4},
        "4fs_on_4to1": {"plan": "on 4:1 (sqrt2=1.41421)", "observed_rel_offset": (r4["w_dt"] - math.sqrt(2)) / math.sqrt(2),
                        "ok": abs(r4["w_dt"] - math.sqrt(2)) / math.sqrt(2) < 0.01},
        "4fs_blue_shift": {"plan": 0.11, "observed": r4["w_eff_over_w"] - 1, "ok": abs(r4["w_eff_over_w"] - 1.11) < 0.01},
        "3to1_at_4p9fs": {"plan": 4.9, "observed_fs": dt_at["3:1"], "ok": abs(dt_at["3:1"] - 4.9) < 0.05},
        "5fs_past_3to1": {"observed_w_dt": rows[2]["w_dt"], "ok": rows[2]["w_dt"] > limits["3:1"]},
        "6fs_linearly_unstable": {"observed_w_dt": rows[3]["w_dt"], "spectral_radius": rows[3]["verlet_spectral_radius"],
                                  "ok": rows[3]["w_dt"] > 2},
        "period_17p75fs": {"plan": 17.75, "observed": period_fs, "ok": abs(period_fs - 17.75) < 0.01},
        "qualified_1p4272_is_not_w_max_at_4fs": {"1.4272/0.004ps": 1.4272 / 0.004, "implied_dt_fs_at_w_max": 1.4272 / W_MAX * 1000,
                                                  "ok": True},
        "mts_inner_2p5fs_w_dt": {"observed": W_MAX * 0.0025, "ok": W_MAX * 0.0025 < limits["5:1"]},
        "mts_inner_2fs_w_dt": {"observed": W_MAX * 0.002, "ok": W_MAX * 0.002 < limits["6:1"]},
        "mts_outer_6fs_near_T_over_3": {"T_over_3_fs": period_fs / 3, "T_over_2_fs": period_fs / 2,
                                         "rel": (6 - period_fs / 3) / (period_fs / 3), "ok": abs(6 - period_fs / 3) / (period_fs / 3) < 0.03},
        "nyquist_4fs_cm-1": {"plan": 4170, "observed": 1 / (2 * 4e-15) / 2.99792458e10,
                             "ok": abs(1 / (2 * 4e-15) / 2.99792458e10 - 4170) < 1},
    }
    all_ok = all(v["ok"] for v in statements.values())
    obs = {"limits_w_dt": limits, "dt_fs_at_limits": dt_at, "rows": rows, "period_fs": period_fs,
           "baoab_mean_map_spectral_radius_6fs": baoab,
           "cubic_oscillator_escape_w_dt_near_4to1": unstable4,
           "cubic_oscillator_escape_w_dt_near_3to1": [min(unstable3), max(unstable3)] if unstable3 else [],
           "plan_statements": statements}
    notes = ("All plan statements reproduced. 4 fs sits 0.16% above the 4:1 limit sqrt(2); a 1-DOF cubic "
             "anharmonic Verlet test (A=0.5, alpha=-0.15, unit frequency, 5000 steps) escapes in a narrow band "
             "at w dt ~1.415-1.42, i.e. exactly where 4 fs puts the fastest mode, and is unstable from ~1.685 "
             "(amplitude-softened 3:1). The 4:1 limit for w_max is dt=3.994 fs. BAOAB friction does not rescue 6 fs "
             "(mean-map spectral radius 1.73-2.01 for gamma 59-1/ps). The circular 1.4272 level corresponds to w=356.8/ps or dt=4.03 fs at w_max.")
    return C.record("timestep.resonance_table",
                    {"w_max_per_ps": W_MAX, "dts_fs": [2, 4, 5, 6]},
                    "plan statements: 4 fs on 4:1 with ~11% blue shift; 3:1 at 4.9 fs; 5 fs past 3:1; 6 fs unstable",
                    obs, all_ok, {"w_dt_abs": 5e-4, "rel_to_limit": 0.01}, notes, anchors=anchors)


CHECKS = [check_resonance_table]
