"""DEC / P1-FEM cotangent edge weights on non-regular tetrahedra.

Continuous law: for a linear (P1) element with barycentric functions l_i, the
element stiffness is K_ij = V grad(l_i).grad(l_j). With N_i the area-weighted
normal (|N_i| = 2 * area) of the face OPPOSITE vertex i, all oriented
consistently, grad(l_i) = -N_i_out/(6V), so the positive DEC edge weight
(dual-area / primal-length, star1) is

    w_ij = -K_ij = -N_i.N_j / (36 V) = l_kl cot(theta_kl) / 6

where kl is the edge opposite ij. The engine evaluator
(nhs_biprism_simplicial_tile.cuh:268-273 float, 334-339 compensated FP64)
assigns w01 <- N2.N3, w02 <- N1.N3, w03 <- N1.N2, w12 <- N0.N3, w13 <- N0.N2,
w23 <- N0.N1, i.e. the weight of the OPPOSITE edge.

Three independent references are used: barycentric gradients from the inverse
Jacobian, the dihedral cotangent formula, and exact rational arithmetic
(fractions.Fraction) for the assembled star0/star1.
"""
from __future__ import annotations

import itertools
from fractions import Fraction

import numpy as np

try:
    from . import _common as C
except ImportError:  # loaded by file path (review-bundle loader)
    import importlib.util, pathlib, sys
    _p = pathlib.Path(__file__).with_name("_common.py")
    _s = importlib.util.spec_from_file_location("prism_refs_common", _p)
    C = importlib.util.module_from_spec(_s); sys.modules.setdefault("prism_refs_common", C); _s.loader.exec_module(C)

EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
SWAP = {(0, 1): (2, 3), (0, 2): (1, 3), (0, 3): (1, 2), (1, 2): (0, 3), (1, 3): (0, 2), (2, 3): (0, 1)}
TILE = C.KERNELS / "nhs_biprism_simplicial_tile.cuh"
BW = C.KERNELS / "nhs_biprism_bowyer_watson_seed.cuh"


def _anchors(engine_root):
    return [
        C.anchor(engine_root, TILE, 268, 273, [
            "out_w01 = fmaxf(dot_N2_N3 * inv_36V_reg, spd_floor);",
            "out_w23 = fmaxf(dot_N0_N1 * inv_36V_reg, spd_floor);"]),
        C.anchor(engine_root, TILE, 334, 339, [
            "out_w01 = fmax(dot_N2_N3 * inv_36V_reg, spd_floor);",
            "out_w23 = fmax(dot_N0_N1 * inv_36V_reg, spd_floor);"]),
        C.anchor(engine_root, TILE, 236, 239, [
            "const float3 N3 = {e01.y * e02.z - e01.z * e02.y",
            "const float3 N0 = {e13.y * e12.z - e13.z * e12.y"]),
        C.anchor(engine_root, TILE, 688, 689, [
            "const uint8_t eu[6] = {v0, v0, v0, v1, v1, v2};",
            "const uint8_t ev[6] = {v1, v2, v3, v2, v3, v3};"]),
        C.anchor(engine_root, BW, 440, 440, [
            "for(uint32_t a=0;a<4;++a)for(uint32_t b=a+1;b<4;++b)tile.tet_edges[t][k++]"]),
    ]


# ---------------------------------------------------------------- engine transcription
def engine_weights(x, X=None, sigma0=1e-4, mapping="engine", floor=True, reg=True):
    """Transcription of evaluate_diagonal_dec_tet_weights_compensated (FP64).

    x, X: (4,3) current / mesh-reference coordinates. mapping="engine" keeps
    the source assignment; "fixed" applies the correct one. floor/reg allow the
    separate effects of the SPD floor and the sigma0 regulariser to be shown.
    Returns (warped_vol, dict edge->weight).
    """
    x = np.asarray(x, float)
    X = x if X is None else np.asarray(X, float)
    e01, e02, e03 = x[1] - x[0], x[2] - x[0], x[3] - x[0]
    e12, e13, e23 = x[2] - x[1], x[3] - x[1], x[3] - x[2]
    E01, E02, E03 = X[1] - X[0], X[2] - X[0], X[3] - X[0]
    N3 = np.cross(e01, e02)
    N2 = np.cross(e03, e01)
    N1 = np.cross(e02, e03)
    N0 = np.cross(e13, e12)
    det6V = float(e01 @ N1)
    det6V_ref = float(np.linalg.det(np.array([E01, E02, E03])))
    V_T = abs(det6V) / 6.0
    V_ref = max(abs(det6V_ref) / 6.0, 1e-5)
    J = min(max(V_T / V_ref, 0.05), 20.0)
    l2 = sum(float(e @ e) for e in (e01, e02, e03, e12, e13, e23))
    l_rms2 = l2 / 6.0
    denom = V_T * V_T + (sigma0 * sigma0) * l_rms2 ** 3 if reg else V_T * V_T
    inv36 = (V_T / max(denom, 1e-16)) / 36.0 * J
    warped = max(V_T * J, 1e-5)
    fl = 1e-3 * np.sqrt(l_rms2) if floor else -np.inf
    N = (N0, N1, N2, N3)
    d = {(i, j): -float(N[i] @ N[j]) for i, j in EDGES}
    w = {}
    for e in EDGES:
        src = SWAP[e] if mapping == "engine" else e  # engine: w01 <- N2.N3, ...
        w[e] = max(d[src] * inv36, fl)
    return warped, w


# ---------------------------------------------------------------- independent references
def p1_weights_gradients(x):
    """w_ij = -V grad(l_i).grad(l_j) from the inverse Jacobian (no normals)."""
    x = np.asarray(x, float)
    Jm = np.array([x[1] - x[0], x[2] - x[0], x[3] - x[0]])  # rows
    V = abs(np.linalg.det(Jm)) / 6.0
    G = np.linalg.inv(Jm)  # columns are grad l1..l3
    g = [-(G[:, 0] + G[:, 1] + G[:, 2]), G[:, 0], G[:, 1], G[:, 2]]
    return V, {(i, j): -V * float(g[i] @ g[j]) for i, j in EDGES}


def p1_weights_cotangent(x):
    """w_ij = l_kl cot(theta_kl)/6, theta_kl = interior dihedral angle at edge kl."""
    x = np.asarray(x, float)
    out = {}
    for i, j in EDGES:
        k, l = [m for m in range(4) if m not in (i, j)]
        ekl = x[l] - x[k]
        lkl = np.linalg.norm(ekl)
        u = ekl / lkl
        # perpendicular components of the two other vertices relative to edge kl
        a = (x[i] - x[k]) - ((x[i] - x[k]) @ u) * u
        b = (x[j] - x[k]) - ((x[j] - x[k]) @ u) * u
        cos_t = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
        sin_t = np.linalg.norm(np.cross(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
        out[(i, j)] = lkl * (cos_t / sin_t) / 6.0
    return out


def exact_weights(xq):
    """Exact rational -N_i.N_j/(36V) and V for rational coordinates."""
    def sub(a, b): return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    def cross(a, b): return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    def dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    e01, e02, e03 = sub(xq[1], xq[0]), sub(xq[2], xq[0]), sub(xq[3], xq[0])
    e12, e13 = sub(xq[2], xq[1]), sub(xq[3], xq[1])
    N = [cross(e13, e12), cross(e02, e03), cross(e03, e01), cross(e01, e02)]
    V = abs(dot(e01, N[1])) / 6
    return V, {(i, j): -dot(N[i], N[j]) / (36 * V) for i, j in EDGES}


def _tets(seed=20261023, n=64):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        x = rng.normal(size=(4, 3)) * np.array([1.0, 1.3, 0.8]) + rng.normal(size=3)
        Jm = np.array([x[1] - x[0], x[2] - x[0], x[3] - x[0]])
        V = abs(np.linalg.det(Jm)) / 6
        lmax = max(np.linalg.norm(x[a] - x[b]) for a, b in EDGES)
        if V / lmax ** 3 > 0.01:  # exclude near-degenerate slivers for the identity test
            out.append(x)
    return out


def _dirichlet(w, x, a):
    u = np.asarray(x) @ a
    return sum(w[(i, j)] * (u[i] - u[j]) ** 2 for i, j in EDGES)


# ---------------------------------------------------------------- checks
def check_reference_selftest(engine_root=None):
    """Correct mapping == inverse-Jacobian P1 == cotangent formula; linear energy exact."""
    tets = _tets()
    errs_grad, errs_cot, errs_energy = [], [], []
    a = np.array([0.3, -1.1, 0.7])
    for x in tets:
        V, wg = p1_weights_gradients(x)
        wc = p1_weights_cotangent(x)
        _, wf = engine_weights(x, mapping="fixed", floor=False, reg=False)
        scale = max(abs(v) for v in wg.values())
        errs_grad.append(max(abs(wf[e] - wg[e]) for e in EDGES) / scale)
        errs_cot.append(max(abs(wc[e] - wg[e]) for e in EDGES) / scale)
        errs_energy.append(abs(_dirichlet(wf, x, a) - V * float(a @ a)) / (V * float(a @ a)))
    obs = {"n_tets": len(tets), "max_rel_err_normals_vs_gradients": max(errs_grad),
           "max_rel_err_cotangent_vs_gradients": max(errs_cot),
           "max_rel_err_linear_dirichlet_energy": max(errs_energy)}
    tol = 1e-10
    ok = all(v <= tol for k, v in obs.items() if k.startswith("max"))
    return C.record("dec.reference_p1_selftest",
                    {"tets": "64 seeded non-regular tetrahedra (seed 20261023, shape V/lmax^3>0.01)",
                     "linear_field_a": a.tolist()},
                    "three independent P1 forms agree; sum_e w_e (du_e)^2 == V|a|^2 for linear u",
                    obs, ok, tol,
                    "Reference-only. w_ij = -N_i.N_j/(36V), N_i = normal of face opposite vertex i.")


def check_engine_mapping(engine_root=None):
    try:
        anchors = _anchors(engine_root)
    except C.AnchorMissing as ex:
        return C.not_built("dec.engine_edge_weight_mapping", str(ex))
    tets = _tets()
    a = np.array([0.3, -1.1, 0.7])
    err_engine, err_swap_identity, energy_engine = [], [], []
    for x in tets:
        V, wg = p1_weights_gradients(x)
        _, we = engine_weights(x, floor=False)  # engine mapping, engine regulariser, no floor
        scale = max(abs(v) for v in wg.values())
        err_engine.append(max(abs(we[e] - wg[e]) for e in EDGES) / scale)
        err_swap_identity.append(max(abs(we[e] - wg[SWAP[e]]) for e in EDGES) / scale)
        energy_engine.append(abs(_dirichlet(we, x, a) - V * float(a @ a)) / (V * float(a @ a)))
    # regular tetrahedron: every edge has the same weight so the swap is invisible
    reg = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    _, wr_e = engine_weights(reg, floor=False)
    _, wr_g = p1_weights_gradients(reg)
    regular_err = max(abs(wr_e[e] - wr_g[e]) for e in EDGES) / max(wr_g.values())
    # one worked example
    x0 = tets[0]
    _, we0 = engine_weights(x0)
    _, wg0 = p1_weights_gradients(x0)
    example = {f"w{i}{j}": {"engine": we0[(i, j)], "correct": wg0[(i, j)]} for i, j in EDGES}
    obs = {
        "max_rel_err_engine_vs_correct": max(err_engine),
        "median_rel_err_engine_vs_correct": float(np.median(err_engine)),
        "max_rel_err_engine_vs_correct_of_opposite_edge": max(err_swap_identity),
        "max_rel_err_linear_dirichlet_energy_engine": max(energy_engine),
        "regular_tet_rel_err": regular_err,
        "worked_example_tet0": {"coords": x0.tolist(), "weights": example},
    }
    tol = 1e-6  # regulariser sigma0=1e-4 perturbs weights at ~1e-8 relative
    passed = max(err_engine) <= tol
    notes = ("Engine weight of edge ij equals the correct weight of the opposite edge kl to "
             f"{max(err_swap_identity):.1e} (only the sigma0=1e-4 regulariser differs): an exact 01<->23, "
             "02<->13, 03<->12 swap. Invisible on a regular tetrahedron (all six weights equal), so "
             "regular-mesh tests cannot catch it. Floor disabled here to isolate the mapping; see star1 check.")
    if not C.anchors_ok(anchors):
        notes = "TRANSCRIPTION_STALE: cited source no longer matches. " + notes
        passed = None
    return C.record("dec.engine_edge_weight_mapping",
                    {"tets": "64 seeded non-regular tetrahedra", "sigma0": 1e-4},
                    "engine w_ij == -N_i.N_j/(36V) (opposite-face normals of i and j)",
                    obs, passed, tol, notes, defect_id="D-DEC-OPP-EDGE", anchors=anchors)


def _kuhn_mesh():
    """Perturbed 3x3x3 vertex grid (8 cubes x 6 Kuhn tets = 48 tets), rational coords.
    Vertex 13 is the single interior vertex."""
    rng = np.random.default_rng(7)
    idx = lambda i, j, k: i + 3 * j + 9 * k
    xq = []
    for k, j, i in itertools.product(range(3), range(3), range(3)):
        p = [Fraction(int(v)) for v in (i, j, k)]
        pert = rng.integers(-12, 13, size=3)  # up to +-12/64 = 0.19 of a cell
        xq.append([p[m] * Fraction(3, 2) + Fraction(int(pert[m]), 64) for m in range(3)])
    tets = []
    for i, j, k in itertools.product(range(2), repeat=3):
        c = [idx(i + a, j + b, k + cc) for cc in (0, 1) for b in (0, 1) for a in (0, 1)]
        # cube corners by bit (a,b,c): c[a + 2b + 4c]; Kuhn: paths 000->111 along axes
        for perm in itertools.permutations(range(3)):
            v = [0, 0, 0]
            path = [c[0]]
            for ax in perm:
                v[ax] = 1
                path.append(c[v[0] + 2 * v[1] + 4 * v[2]])
            tets.append(path)
    return xq, tets


def _assemble(xq, tets, weight_fn):
    star0, star1 = {}, {}
    for t in tets:
        xs = [xq[v] for v in t]
        vol, w = weight_fn(xs)
        for v in t:
            star0[v] = star0.get(v, 0) + vol / 4
        for (a, b) in EDGES:
            key = tuple(sorted((t[a], t[b])))
            star1[key] = star1.get(key, 0) + w[(a, b)]
    return star0, star1


def check_star_exact(engine_root=None):
    try:
        anchors = _anchors(engine_root) + [
            C.anchor(engine_root, TILE, 1022, 1022, ["smem.tet_terms[0][tid]=0.25f*warped_vol;"])]
    except C.AnchorMissing as ex:
        return C.not_built("dec.star0_star1_exact_recompute", str(ex))
    xq, tets = _kuhn_mesh()
    ex0, ex1 = _assemble(xq, tets, exact_weights)
    xf = [[float(c) for c in p] for p in xq]
    def eng(mapping, floor):
        return lambda xs: engine_weights(np.array([[float(c) for c in p] for p in xs]), mapping=mapping, floor=floor)
    en0, en1 = _assemble(xq, tets, eng("engine", True))
    fx0, fx1 = _assemble(xq, tets, eng("fixed", True))
    nf0, nf1 = _assemble(xq, tets, eng("fixed", False))
    total_exact = sum(ex0.values())
    keys = sorted(ex1)
    s1max = max(abs(float(ex1[k])) for k in keys)
    rel = lambda d: max(abs(d[k] - float(ex1[k])) for k in keys) / s1max
    star0_err = max(abs(en0[v] - float(ex0[v])) / float(ex0[v]) for v in ex0)
    # linear precision at the interior vertex 13: sum_j w_ij (x_j - x_i) == 0 exactly (P1)
    def lin(st1, exact=False):
        acc = [Fraction(0)] * 3 if exact else np.zeros(3)
        for (a, b), w in st1.items():
            if 13 in (a, b):
                o = b if a == 13 else a
                for m in range(3):
                    if exact:
                        acc[m] += w * (xq[o][m] - xq[13][m])
                    else:
                        acc[m] += w * (xf[o][m] - xf[13][m])
        return [float(v) for v in acc]
    lin_exact = lin(ex1, True)
    neg = sum(1 for k in keys if ex1[k] < 0)
    obs = {
        "n_vertices": len(xq), "n_tets": len(tets), "n_edges": len(keys),
        "sum_star0_exact": str(total_exact), "sum_star0_engine": sum(en0.values()),
        "max_rel_err_star0_engine": star0_err,
        "max_rel_err_star1_engine_as_is": rel(en1),
        "max_rel_err_star1_mapping_fixed_with_floor": rel(fx1),
        "max_rel_err_star1_mapping_fixed_no_floor": rel(nf1),
        "exact_negative_star1_edges": neg,
        "interior_linear_precision_exact": lin_exact,
        "interior_linear_precision_engine_as_is": lin(en1),
        "interior_linear_precision_mapping_fixed_with_floor": lin(fx1),
        "interior_linear_precision_mapping_fixed_no_floor": lin(nf1),
    }
    tol = 1e-6
    passed = rel(en1) <= tol and star0_err <= tol
    notes = ("Exact rational star0/star1 on a perturbed Kuhn mesh (48 tets). star0 (barycentric V/4) "
             "matches; star1 is wrong as-is. Linear precision at the interior vertex is exact (0) for the "
             "correct operator and fails for the engine operator. The SPD floor fmax(w,1e-3*l_rms) is a "
             "separate defect whenever exact weights are negative (non-Delaunay / obtuse dihedrals): it "
             "breaks linear precision even after the mapping fix. The dual is barycentric (V/4), not Voronoi.")
    if not C.anchors_ok(anchors):
        notes = "TRANSCRIPTION_STALE: " + notes
        passed = None
    return C.record("dec.star0_star1_exact_recompute",
                    {"mesh": "3x3x3 grid, spacing 3/2, rational perturbation k/64 (|k|<=12), Kuhn split"},
                    "engine star0 and star1 equal exact rational values; interior linear precision == 0",
                    obs, passed, tol, notes, defect_id="D-DEC-OPP-EDGE", anchors=anchors)


CHECKS = [check_reference_selftest, check_engine_mapping, check_star_exact]
