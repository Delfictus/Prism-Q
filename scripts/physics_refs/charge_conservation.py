"""Charge conservation of the electronic charge coupling.

Engine forms:
  * BIPRISM mixed sample: r.charge = fma(double(.15f), rho11, double(q_amber))
    (nhs_biprism_mixed_energy.cuh:263; plan cites 261-263) and bp_ed_endpoint_data
    q_n = fma(0.15f, r11, q_amber) (nhs_biprism_electronic_discrete.cuh:140-141).
    Sum_i q_i = Q_amber + 0.15 * Sum_i rho11_i: charge is created, not moved.
  * AMBER UV path: uv_excited_charge (nhs_amber_fused.cu:2709-2720) is affine
    about the set mean (conserving) but falls back to q_ground*scale when the
    set-mean pointer is null (selected at :3118-3119 when
    exact_nonbonded_exceptions_enabled < 0).
Reference: an edge transfer dq_u = -dq_v conserves Sum q to round-off.
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

F015 = float(np.float32(0.15))  # double(.15f) as written in the engine
ME = C.KERNELS / "nhs_biprism_mixed_energy.cuh"
ED = C.KERNELS / "nhs_biprism_electronic_discrete.cuh"
AF = C.KERNELS / "nhs_amber_fused.cu"
TRP_RING = ("CG", "CD1", "HD1", "CD2", "NE1", "HE1", "CE2", "CE3", "HE3", "CZ2", "HZ2", "CZ3", "HZ3", "CH2", "HH2")
TRP_DIPOLE_RATIO_SQRT = 1.690
# AMBER TRP indole charges as carried by the frozen IL2 topology (residue 109); used when no topology is present.
TRP_RING_Q = {"CG": -0.1415, "CD1": -0.1638, "HD1": 0.2062, "CD2": 0.1243, "NE1": -0.3418, "HE1": 0.3412,
              "CE2": 0.138, "CE3": -0.2387, "HE3": 0.17, "CZ2": -0.2601, "HZ2": 0.1572, "CZ3": -0.1972,
              "HZ3": 0.1447, "CH2": -0.1134, "HH2": 0.1417}


def _charges():
    topo, path, sha = C.load_topology("IL2")
    if topo is None:
        rng = np.random.default_rng(5)
        q = rng.normal(scale=0.3, size=2052)
        q -= q.mean()
        return q, {"source": "synthetic neutral (IL2 topology not found)", "path": path}, None
    return np.array(topo["charges"], float), {"source": "IL2 frozen May topology", "path": path, "sha256": sha}, topo


def check_mixed_charge(engine_root=None):
    try:
        anchors = [C.anchor(engine_root, ME, 263, 263, ["r.charge=__fma_rn(double(.15f),r.rho11,double(r.amber_charge));"]),
                   C.anchor(engine_root, ED, 140, 141, ["out.q_n=bp_ed_fma(R(0.15f),r0.r11,q_amber);",
                                                        "out.q_np1=bp_ed_fma(R(0.15f),r1.r11,q_amber);"])]
    except C.AnchorMissing as ex:
        return C.not_built("charge.biprism_rho11_charge_creation", str(ex))
    q, src, _ = _charges()
    core = q[:192]  # one 256-vertex tile has 192 core rows (tid<192)
    rng = np.random.default_rng(3)
    rows = {}
    for label, rho in (("thermal_like_rho11_1e-3", rng.uniform(0, 2e-3, 192)),
                       ("uv_excited_10pct_atoms_rho0.5", np.where(rng.uniform(size=192) < 0.1, 0.5, 0.0)),
                       ("all_excited_rho1", np.ones(192))):
        q_eng = np.fma(F015, rho, core) if hasattr(np, "fma") else F015 * rho + core
        dQ = math.fsum(q_eng) - math.fsum(core)
        rows[label] = {"sum_rho11": float(rho.sum()), "dQ_e": dQ, "predicted_0.15_sum_rho11": F015 * float(rho.sum())}
    worst = max(abs(r["dQ_e"]) for r in rows.values())
    tol = 1e-12
    obs = {"charges": src, "tile_core_atoms": 192, "Q_amber_core": math.fsum(core), "cases": rows,
           "double(.15f)": F015}
    notes = ("Sum of engine charges = Q_amber + 0.15*Sum rho11 exactly (dQ equals the prediction): every rise in "
             "excited population creates net charge, e.g. +0.15 e per fully excited atom; across 192 atoms at "
             "rho11=1 that is +28.8 e. Also the factor is double(0.15f)=0.15000000596. H2 gate requires <=1e-12 e.")
    return C.record("charge.biprism_rho11_charge_creation", {"tile_atoms": "first 192 IL2 atoms"},
                    "Sum q conserved (|dQ| <= 1e-12 e) when rho11 changes", obs,
                    (worst <= tol) if C.anchors_ok(anchors) else None, tol, notes,
                    defect_id="D-CHARGE-CREATE", anchors=anchors)


def check_uv_null_fallback(engine_root=None):
    try:
        anchors = [C.anchor(engine_root, AF, 2709, 2720, ["return q_ground * scale;", "return q_mean + (q_ground - q_mean) * scale;"]),
                   C.anchor(engine_root, AF, 3118, 3119, ["exact_nonbonded_exceptions_enabled < 0 ? nullptr : d_aromatic_charge_set_mean"])]
    except C.AnchorMissing as ex:
        return C.not_built("charge.uv_excited_charge_null_fallback", str(ex))
    q, src, topo = _charges()
    if topo is None:
        qr = np.array([TRP_RING_Q[a] for a in TRP_RING])
        src = {"source": "embedded AMBER TRP indole charges (IL2 topology not found)"}
    else:
        idx = [i for i, (r, a) in enumerate(zip(topo["residue_names"], topo["atom_names"])) if r == "TRP" and a in TRP_RING]
        qr = q[idx]
    mean = qr.mean()
    out = {}
    for pop in (0.25, 1.0):
        s = 1.0 + (TRP_DIPOLE_RATIO_SQRT - 1.0) * pop
        q_null = qr * s
        q_aff = mean + (qr - mean) * s
        out[f"pop_{pop}"] = {"scale": s, "dQ_null_fallback": math.fsum(q_null) - math.fsum(qr),
                             "dQ_affine": math.fsum(q_aff) - math.fsum(qr)}
    worst_null = max(abs(v["dQ_null_fallback"]) for v in out.values())
    worst_aff = max(abs(v["dQ_affine"]) for v in out.values())
    tol = 1e-6  # float32 engine arithmetic; affine form meets it, fallback misses by orders
    obs = {"charges": src, "trp_ring_atoms": int(len(qr)), "sum_ring_charge": math.fsum(qr), "cases": out,
           "max_abs_dQ_affine": worst_aff, "max_abs_dQ_null_fallback": worst_null}
    notes = ("Ring set = IL2 TRP indole atoms incl. ring H (an assumption about the mapped set). The affine form "
             "conserves the monopole for any scale; the null-pointer fallback scales the monopole by `scale` and "
             "creates charge whenever the set's net charge is nonzero. Plan H2: fallback becomes a refusal.")
    return C.record("charge.uv_excited_charge_null_fallback", {"residue": "IL2 TRP", "dipole_ratio_sqrt": TRP_DIPOLE_RATIO_SQRT},
                    "Sum q over the rescaled set unchanged by the UV rescale on every reachable branch", obs,
                    (worst_null <= tol and worst_aff <= tol) if C.anchors_ok(anchors) else None, tol, notes,
                    defect_id="D-CHARGE-UV-NULL-FALLBACK", anchors=anchors)


def check_edge_transfer_reference(engine_root=None):
    q, src, _ = _charges()
    n = 192
    rng = np.random.default_rng(17)
    pairs = set()
    while len(pairs) < 600:
        u, v = rng.integers(0, n, 2)
        if u != v:
            pairs.add((min(u, v), max(u, v)))
    E = np.array(sorted(pairs))
    u, v = E[:, 0], E[:, 1]
    qq = q[:n].copy()
    Q0 = math.fsum(qq)
    steps = 100_000
    worst = 0.0
    for s in range(steps):
        f = rng.normal(scale=1e-3, size=len(E))
        qq += np.bincount(v, f, minlength=n) - np.bincount(u, f, minlength=n)
        if s % 1000 == 999:
            worst = max(worst, abs(math.fsum(qq) - Q0))
    tol = 1e-12
    obs = {"atoms": n, "edges": len(E), "steps": steps, "flux_sigma_e": 1e-3,
           "max_abs_dQ_e": worst, "final_dQ_e": math.fsum(qq) - Q0}
    return C.record("charge.edge_transfer_reference", {"charges": src},
                    "dq_u = -dq_v keeps |Sum q - Q0| <= 1e-12 e over 1e5 steps (H2 gate)", obs, worst <= tol, tol,
                    "Reference-only: the conserving form the H2 change adopts. FP64 per-atom accumulation, "
                    "exactly-rounded total (math.fsum); residual is round-off of the per-atom adds.")


CHECKS = [check_mixed_charge, check_uv_null_fallback, check_edge_transfer_reference]
