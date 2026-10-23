"""Built-in fallback reference checks for the Prism-Q review bundle (stdlib only).

These are minimal, independent references so the generator is testable on its own.
Checks with the same `name` in any non-`builtin_` module under scripts/physics_refs/
supersede these. Contract (shared with the physics-refs modules):

    fn(engine_root: pathlib.Path | None = None) -> dict(
        name, inputs, expected, observed, pass: bool | None, tolerance, notes[, defect_id])

pass=None with notes starting "NOT_BUILT" means the source under test does not exist yet.
`defect_id` marks a check that is EXPECTED to fail while that defect is OPEN.
"""
from __future__ import annotations

import math
import re
from fractions import Fraction
from pathlib import Path

DEC_FILE = "crates/prism-gpu/src/kernels/nhs_biprism_simplicial_tile.cuh"
FUSED_FILE = "crates/prism-gpu/src/kernels/nhs_amber_fused.cu"
BRIDGE_FILE = "crates/prism-gpu/src/kernels/nhs_biprism_unit_bridge.cuh"
GEAR_FILE = "crates/prism-nhs/src/cuda/gearbox.cuh"

# CODATA 2022 (exact SI where defined)
E_CHARGE = 1.602176634e-19        # C (exact)
EPS0 = 8.8541878188e-12           # F/m (CODATA 2022)
N_A = 6.02214076e23               # 1/mol (exact)
H_PLANCK = 6.62607015e-34         # J s (exact)
C_LIGHT = 299792458.0             # m/s (exact)
KCAL = 4184.0                     # J (thermochemical)
HARTREE_J = 4.3597447222060e-18   # J (CODATA 2022)
BOHR_A = 0.529177210544           # Angstrom (CODATA 2022)
AU_TIME_S = 2.4188843265864e-17   # s (CODATA 2022)
EV_J = 1.602176634e-19


def _read(engine_root: Path | None, rel: str) -> str | None:
    if engine_root is None:
        return None
    p = Path(engine_root) / rel
    return p.read_text(errors="replace") if p.is_file() else None


# --------------------------------------------------------------------------- DEC

def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _source_normals(p):
    """Face normals exactly as defined in evaluate_diagonal_dec_tet_weights (exact rationals)."""
    e01, e02, e03 = _sub(p[1], p[0]), _sub(p[2], p[0]), _sub(p[3], p[0])
    e12, e13 = _sub(p[2], p[1]), _sub(p[3], p[1])
    return {0: _cross(e13, e12), 1: _cross(e02, e03), 2: _cross(e03, e01), 3: _cross(e01, e02)}, _dot(e01, _cross(e02, e03))


def _p1_stiffness(p):
    """Independent P1 stiffness K_ij = V grad(phi_i).grad(phi_j) via the inverse Jacobian (exact)."""
    J = [[p[k][d] - p[0][d] for k in (1, 2, 3)] for d in range(3)]  # columns e01, e02, e03
    det = (J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0])
           + J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]))
    cof = [[(J[(r + 1) % 3][(c + 1) % 3] * J[(r + 2) % 3][(c + 2) % 3]
             - J[(r + 1) % 3][(c + 2) % 3] * J[(r + 2) % 3][(c + 1) % 3]) for c in range(3)] for r in range(3)]
    inv = [[cof[c][r] / det for c in range(3)] for r in range(3)]  # inv = adj^T / det -> rows = grad(lambda_k)
    g = {1: inv[0], 2: inv[1], 3: inv[2]}
    g[0] = tuple(-(g[1][d] + g[2][d] + g[3][d]) for d in range(3))
    V = abs(det) / 6
    return {(i, j): V * _dot(g[i], g[j]) for i in range(4) for j in range(4)}, V


def dec_p1fem_edge_weight_mapping(engine_root: Path | None = None) -> dict:
    tet = [(Fraction(0), Fraction(0), Fraction(0)), (Fraction(3), Fraction(0), Fraction(0)),
           (Fraction(1, 2), Fraction(2), Fraction(0)), (Fraction(1), Fraction(1, 3), Fraction(5, 2))]
    N, det6 = _source_normals(tet)
    K, V = _p1_stiffness(tet)
    # independent derivation agrees with the face-normal identity K_ij = N_i.N_j / (36 V)?
    identity_ok = all(K[(i, j)] == _dot(N[i], N[j]) / (36 * V) for i in range(4) for j in range(4) if i != j)
    correct = {f"w{i}{j}": f"N{i}.N{j}" for i in range(4) for j in range(i + 1, 4)}
    src = _read(engine_root, DEC_FILE)
    if src is None:
        return {"name": "dec_p1fem_edge_weight_mapping", "inputs": {"tet": [list(map(str, v)) for v in tet]},
                "expected": correct, "observed": None, "pass": None, "tolerance": "exact",
                "notes": f"NOT_BUILT: {DEC_FILE} not present"}
    rx = re.compile(r"out_w(\d)(\d)\s*=\s*fmaxf?\(\s*dot_N(\d)_N(\d)")
    found = rx.findall(src)
    blocks = [found[i:i + 6] for i in range(0, len(found), 6)]
    observed = [{f"w{a}{b}": f"N{c}.N{d}" for a, b, c, d in blk} for blk in blocks]
    mapping_ok = bool(observed) and all(o == correct for o in observed)
    w_exact = {f"w{i}{j}": float(-K[(i, j)]) for i in range(4) for j in range(i + 1, 4)}
    w_src = {}
    if observed:
        for k, v in observed[0].items():
            a, b = int(v[1]), int(v[4])
            w_src[k] = float(-_dot(N[a], N[b]) / (36 * V))
    return {
        "name": "dec_p1fem_edge_weight_mapping",
        "defect_id": "D-DEC-OPP-EDGE",
        "inputs": {"tet_vertices": [[str(c) for c in v] for v in tet], "source": DEC_FILE,
                   "assignment_blocks_found": len(blocks)},
        "expected": {"mapping": correct, "weights_exact_unregularised": w_exact},
        "observed": {"mapping_per_function": observed, "weights_from_source_mapping": w_src,
                     "face_normal_identity_holds": identity_ok, "signed_6V": str(det6)},
        "pass": bool(mapping_ok and identity_ok),
        "tolerance": "exact (rational arithmetic)",
        "notes": "Independent P1 stiffness from the inverse Jacobian; source mapping parsed from out_wXY = fmax(dot_Na_Nb ...). "
                 "A mismatch reproduces the opposite-edge assignment (01<->23, 02<->13, 03<->12).",
    }


# --------------------------------------------------------------------------- units

def unit_conversions(engine_root: Path | None = None) -> dict:
    kC = E_CHARGE ** 2 / (4 * math.pi * EPS0 * 1e-10) * N_A / KCAL
    ha_kcal = HARTREE_J * N_A / KCAL
    e280_ev = H_PLANCK * C_LIGHT / 280e-9 / EV_J
    e280_kcal = H_PLANCK * C_LIGHT / 280e-9 * N_A / KCAL
    a_to_bohr = 1.0 / BOHR_A
    ps_to_au = 1e-12 / AU_TIME_S
    observed: dict = {}
    src = _read(engine_root, FUSED_FILE)
    if src:
        m = re.search(r"#define\s+COULOMB_CONSTANT\s+([0-9.eE+-]+)f?", src)
        observed["COULOMB_CONSTANT"] = float(m.group(1)) if m else None
    br = _read(engine_root, BRIDGE_FILE)
    if br:
        for key in ("BIPRISM_ANGSTROM_TO_BOHR", "BIPRISM_PS_TO_ATOMIC_TIME"):
            m = re.search(key + r"\s*=\s*([0-9.eE+-]+)", br)
            observed[key] = float(m.group(1)) if m else None
    expected = {"COULOMB_CONSTANT": kC, "BIPRISM_ANGSTROM_TO_BOHR": a_to_bohr, "BIPRISM_PS_TO_ATOMIC_TIME": ps_to_au,
                "hartree_kcal_per_mol": ha_kcal, "E_photon_280nm_eV": e280_ev, "E_photon_280nm_kcal_per_mol": e280_kcal}
    tol = {"COULOMB_CONSTANT": 2e-6, "BIPRISM_ANGSTROM_TO_BOHR": 1e-9, "BIPRISM_PS_TO_ATOMIC_TIME": 1e-9}
    rel = {k: (abs(observed[k] - expected[k]) / expected[k] if observed.get(k) is not None else None) for k in tol}
    if not observed:
        return {"name": "unit_conversions", "inputs": {}, "expected": expected, "observed": None, "pass": None,
                "tolerance": tol, "notes": "NOT_BUILT: engine sources with unit constants not present"}
    ok = all(rel[k] is not None and rel[k] <= tol[k] for k in tol if k in observed)
    return {"name": "unit_conversions", "inputs": {"codata": "2022", "kcal": "4184 J"}, "expected": expected,
            "observed": {"source_constants": observed, "relative_error": rel}, "pass": ok,
            "tolerance": {"relative": tol},
            "notes": "k_C reference: AMBER uses 332.0522173, CHARMM 332.0716; the CODATA 2022 value is ~332.0637. "
                     "Bridge constants are compared with CODATA 2022 a0 and atomic time; a mismatch at ~1e-10 relative "
                     "indicates an older CODATA vintage (record it, do not silently change physics)."}


# --------------------------------------------------------------------------- integrator

def omega_dt_resonance(engine_root: Path | None = None) -> dict:
    w = 354.1  # ps^-1, measured omega_max
    rows = []
    for dt_fs in (0.5, 2.0, 4.0, 5.0, 6.0):
        x = w * dt_fs * 1e-3
        stable = x < 2.0
        weff = (2 / (dt_fs * 1e-3)) * math.asin(x / 2) if stable else float("nan")
        rows.append({"dt_fs": dt_fs, "omega_dt": round(x, 6), "linearly_stable": stable,
                     "blue_shift_pct": round(100 * (weff / w - 1), 3) if stable else None})
    res = {n: 2 * math.sin(math.pi / n) for n in (2, 3, 4, 5, 6)}
    dt_at = {f"{n}:1": round(1e3 * v / w, 4) for n, v in res.items()}
    gears = None
    g = _read(engine_root, GEAR_FILE)
    if g:
        gears = [float(x) for x in re.findall(r"#define\s+PRISM_GEAR_DEFAULT_DT_FS_G\d\s+([0-9.]+)f", g)]
    x4 = w * 4e-3
    expected = {"omega_dt_4fs": 1.4164, "blue_shift_4fs_pct": 11.1, "dt_3to1_fs": 4.89, "gears_fs": [0.5, 2.0, 4.0, 0.0]}
    observed = {"table": rows, "resonance_dt_fs": dt_at, "gears_fs_in_source": gears,
                "omega_dt_4fs": round(x4, 4), "nearest_resonance_4fs": min(res, key=lambda n: abs(res[n] - x4))}
    ok = (abs(x4 - 1.4164) < 1e-3 and abs(rows[2]["blue_shift_pct"] - 11.1) < 0.2 and abs(dt_at["3:1"] - 4.89) < 0.01
          and (gears is None or gears == expected["gears_fs"]))
    return {"name": "omega_dt_resonance", "inputs": {"omega_max_ps^-1": w, "period_fs": round(2e3 * math.pi / w, 2)},
            "expected": expected, "observed": observed, "pass": ok, "tolerance": "1e-3 in omega*dt; 0.2% shift",
            "notes": "Velocity-Verlet n:1 resonance at omega*dt = 2 sin(pi/n). 4 fs sits on 4:1 (sqrt 2). A PASS reproduces "
                     "the plan's resonance table; it does not qualify any timestep."}


# --------------------------------------------------------------------------- electronic

def cayley_vs_su2_phase(engine_root: Path | None = None) -> dict:
    gap = 0.38  # Ha
    au_per_fs = 1e-15 / AU_TIME_S
    rows = []
    for dt_fs in (0.5, 2.0, 4.0):
        h = dt_fs * au_per_fs / 4  # 4 inner substeps
        theta_c = 2 * math.atan(gap * h / 2)
        eff = theta_c / h
        rows.append({"dt_fs": dt_fs, "substep_au": round(h, 4), "exact_phase_rad": round(gap * h, 6),
                     "cayley_phase_rad": round(theta_c, 6), "effective_gap_Ha": round(eff, 6),
                     "compression": round(gap / eff, 4)})
    src = _read(engine_root, "crates/prism-gpu/src/kernels/nhs_biprism_electronic_discrete.cuh")
    present = bool(src and "bp_ed_hermitian_cayley" in src)
    r4 = rows[2]
    ok_reproduces = abs(r4["effective_gap_Ha"] - 0.070) < 0.002
    return {"name": "cayley_vs_su2_phase", "defect_id": "D-ELEC-CAYLEY-GAP",
            "inputs": {"onsite_gap_Ha": gap, "inner_substeps": 4, "omega": 1.0, "source_present": present},
            "expected": {"phase_error_rad_per_ps_max": 1e-3, "effective_gap_Ha": gap},
            "observed": {"table": rows, "reproduces_plan_0p070_Ha_at_4fs": ok_reproduces},
            "pass": all(abs(r["effective_gap_Ha"] - gap) / gap < 1e-3 for r in rows),
            "tolerance": "relative 1e-3 on the effective gap (H1 gate: <= 1e-3 rad/ps)",
            "notes": "Eigenphase of the Cayley map (I + i h H/2)^-1 (I - i h H/2) is 2 atan(E h/2) versus exact E h. "
                     "Expected to fail until the interaction-picture propagator (H1) lands."}


def coupling_charge_conservation(engine_root: Path | None = None) -> dict:
    return {"name": "coupling_charge_conservation", "inputs": {}, "expected": "sum(dq) = 0 to 1e-12 e over 1e5 steps",
            "observed": None, "pass": None, "tolerance": 1e-12,
            "notes": "NOT_BUILT: the H2 charge-conserving electronic coupling is not built; the current form "
                     "q = q_amber + 0.15*rho11 creates charge by construction (D-CHARGE-CREATE)."}


def qnanl_roundtrip(engine_root: Path | None = None) -> dict:
    return {"name": "qnanl_roundtrip", "inputs": {}, "expected": "byte-exact round trip + bit-flip refusal",
            "observed": None, "pass": None, "tolerance": "exact",
            "notes": "NOT_BUILT: no independent (out-of-process, non-Rust) PRSQNB01 decoder or committed fixture exists yet "
                     "(G-Q2). The Rust codec's own round-trip tests are not independent references."}


CHECKS = [dec_p1fem_edge_weight_mapping, unit_conversions, omega_dt_resonance, cayley_vs_su2_phase,
          coupling_charge_conservation, qnanl_roundtrip]
