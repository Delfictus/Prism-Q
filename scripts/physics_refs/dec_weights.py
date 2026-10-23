"""DEC / P1-FEM cotangent edge weights on non-regular tetrahedra.

Continuous law: for a linear (P1) element with barycentric functions l_i, the
element stiffness is K_ij = V grad(l_i).grad(l_j). With N_i the area-weighted
normal (|N_i| = 2 * area) of the face OPPOSITE vertex i, all oriented
consistently, grad(l_i) = -N_i_out/(6V), so the DEC edge weight
(dual-area / primal-length, star1) is

    w_ij = -K_ij = -N_i.N_j / (36 V) = l_kl cot(theta_kl) / 6

where kl is the edge opposite ij (weights are legitimately negative for obtuse
dihedrals; the element matrix is PSD regardless).

Two engine source variants are recognised by their text and transcribed:
  legacy_opposite_edge (canonical engine before P3-DEC, nhs_biprism_simplicial_tile.cuh
      268-273/334-339 pre-fix): w01 <- N2.N3 ... w23 <- N0.N1 (opposite edge),
      per-tet floor fmax(w, 1e-3 l_rms), Tikhonov guard V/(V^2 + s0^2 l_rms^6),
      assembled floor fmax(star1, 1e-4).
  p3_signed (lane/p3-dec-20261023): w_ij <- k_ij = -N_i.N_j, no floor unless
      -DBIPRISM_DEC_LEGACY_POSITIVE_FLOOR=1 (deprecated, reachable), guard
      V/max(V^2, s0^2 l_rms^6), signed assembled star1 (bp_dec_assembled_star1).
Any other text gives TRANSCRIPTION_STALE. The transcription keeps the source's
float32 literals (double(1.0f/6.0f), double(1e-5f), ...) and rounds outputs to
FP32 as the compensated evaluator does.

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
_f = lambda v: float(np.float32(v))  # value of a C float literal promoted to double

VARIANTS = {
    "legacy_opposite_edge": {"mapping": "swap", "floor": True, "guard": "tikhonov", "assembled_floor": True},
    "p3_signed": {"mapping": "own", "floor": False, "guard": "max", "assembled_floor": False},
}


def _common_anchors(engine_root):
    return [
        C.anchor(engine_root, BW, 440, 440, [
            "for(uint32_t a=0;a<4;++a)for(uint32_t b=a+1;b<4;++b)tile.tet_edges[t][k++]"]),
    ]


def _variant_anchors(engine_root, variant):
    if variant == "legacy_opposite_edge":
        return [
            C.anchor(engine_root, TILE, 268, 273, ["out_w01 = fmaxf(dot_N2_N3 * inv_36V_reg, spd_floor);",
                                                   "out_w23 = fmaxf(dot_N0_N1 * inv_36V_reg, spd_floor);"]),
            C.anchor(engine_root, TILE, 334, 339, ["out_w01 = fmax(dot_N2_N3 * inv_36V_reg, spd_floor);",
                                                   "out_w23 = fmax(dot_N0_N1 * inv_36V_reg, spd_floor);"]),
            C.anchor(engine_root, TILE, 320, 320, ["const double reg_denom = __fma_rn(V_T, V_T, (sigma_0 * sigma_0) * (l_rms2 * l_rms2 * l_rms2));"]),
            C.anchor(engine_root, TILE, 236, 239, ["const float3 N3 = {e01.y * e02.z - e01.z * e02.y",
                                                   "const float3 N0 = {e13.y * e12.z - e13.z * e12.y"]),
            C.anchor(engine_root, TILE, 688, 689, ["const uint8_t eu[6] = {v0, v0, v0, v1, v1, v2};",
                                                   "const uint8_t ev[6] = {v1, v2, v3, v2, v3, v3};"]),
            C.anchor(engine_root, TILE, 1022, 1022, ["smem.tet_terms[0][tid]=0.25f*warped_vol;"]),
            C.anchor(engine_root, TILE, 1060, 1060, ["const float star1_base=fmaxf(smem.star1[e],1e-4f);"]),
        ]
    return [
        C.anchor(engine_root, TILE, 233, 236, ["#define BIPRISM_DEC_LEGACY_POSITIVE_FLOOR 0", "#define BIPRISM_DEC_SIGMA0 1e-4f"]),
        C.anchor(engine_root, TILE, 245, 251, ["return fmax(w, double(1e-3f) * sqrt(l_rms2));", "(void)l_rms2; return w;"]),
        C.anchor(engine_root, TILE, 253, 259, ["return fmaxf(star1, 1e-4f);", "return star1;"]),
        C.anchor(engine_root, TILE, 365, 368, ["const BiprismCoordinate3F64 N3 = {e01.y * e02.z - e01.z * e02.y",
                                               "const BiprismCoordinate3F64 N0 = {e13.y * e12.z - e13.z * e12.y"]),
        C.anchor(engine_root, TILE, 383, 387, ["const double guard2    = (sigma_0 * sigma_0) * (l_rms2 * l_rms2 * l_rms2);",
                                               "const double reg_denom = fmax(V_T * V_T, guard2);",
                                               "const double inv_36V_reg = (V_T / fmax(reg_denom, double(1e-16f))) * double(1.0f / 36.0f) * J_piola;"]),
        C.anchor(engine_root, TILE, 390, 402, ["const double k01 = -(N0.x * N1.x + N0.y * N1.y + N0.z * N1.z);",
                                               "const double k23 = -(N2.x * N3.x + N2.y * N3.y + N2.z * N3.z);",
                                               "out_w01 = bp_dec_edge_weight_policy(k01 * inv_36V_reg, l_rms2);",
                                               "out_w23 = bp_dec_edge_weight_policy(k23 * inv_36V_reg, l_rms2);"]),
        C.anchor(engine_root, TILE, 324, 342, ["const float reg_denom = fmaxf(V_T * V_T, guard2);",
                                               "const float k01 = -(N0.x * N1.x + N0.y * N1.y + N0.z * N1.z);",
                                               "out_w01 = bp_dec_edge_weight_policy(k01 * inv_36V_reg, l_rms2);"]),
        C.anchor(engine_root, TILE, 769, 770, ["const uint8_t eu[6] = {v0, v0, v0, v1, v1, v2};",
                                               "const uint8_t ev[6] = {v1, v2, v3, v2, v3, v3};"]),
        C.anchor(engine_root, TILE, 1099, 1103, ["BIPRISM_DEC_SIGMA0,warped_vol,w01,w02,w03,w12,w13,w23);",
                                                 "smem.tet_terms[0][tid]=0.25f*warped_vol;"]),
        C.anchor(engine_root, TILE, 1141, 1141, ["const float star1_base=bp_dec_assembled_star1(smem.star1[e]);"]),
    ]


def detect_variant(engine_root):
    """Return (variant or None, anchors). Raises C.AnchorMissing if the file is absent."""
    tried = []
    for v in ("p3_signed", "legacy_opposite_edge"):
        a = _variant_anchors(engine_root, v) + _common_anchors(engine_root)
        if C.anchors_ok(a):
            return v, a
        tried += [dict(x, variant_tried=v) for x in a]
    return None, tried


# ---------------------------------------------------------------- engine transcription
def engine_weights(x, X=None, variant="legacy_opposite_edge", sigma0=_f(1e-4), mapping=None, floor=None,
                   guard=None, round_f32=True):
    """Transcription of the compensated FP64 evaluator for the given source variant.

    mapping: "swap" (legacy, opposite edge) or "own" (k_ij = -N_i.N_j); floor:
    per-tet fmax(w, 1e-3 l_rms); guard: "tikhonov" V/(V^2+g), "max" V/max(V^2,g),
    or "none". Overrides allow isolating each effect. Returns (warped_vol, {edge: w}).
    """
    P = dict(VARIANTS[variant])
    if mapping is not None: P["mapping"] = mapping
    if floor is not None: P["floor"] = floor
    if guard is not None: P["guard"] = guard
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
    det6V_ref = float(E01 @ np.cross(E02, E03))
    V_T = abs(det6V) * _f(1.0 / 6.0)
    V_ref = max(abs(det6V_ref) * _f(1.0 / 6.0), _f(1e-5))
    J = min(max(V_T / V_ref, _f(0.05)), _f(20.0))
    l2 = sum(float(e @ e) for e in (e01, e02, e03, e12, e13, e23))
    l_rms2 = l2 * _f(1.0 / 6.0)
    g = (sigma0 * sigma0) * (l_rms2 * l_rms2 * l_rms2)
    denom = {"tikhonov": V_T * V_T + g, "max": max(V_T * V_T, g), "none": V_T * V_T}[P["guard"]]
    inv36 = (V_T / max(denom, _f(1e-16))) * _f(1.0 / 36.0) * J
    warped = max(V_T * J, _f(1e-5))
    fl = _f(1e-3) * np.sqrt(l_rms2) if P["floor"] else -np.inf
    N = (N0, N1, N2, N3)
    k = {(i, j): -float(N[i] @ N[j]) for i, j in EDGES}
    rnd = _f if round_f32 else float
    w = {}
    for e in EDGES:
        src = SWAP[e] if P["mapping"] == "swap" else e
        w[e] = rnd(max(k[src] * inv36, fl))
    return rnd(warped), w


# ---------------------------------------------------------------- independent references
def p1_weights_normals(x):
    """Pure formula w_ij = -N_i.N_j/(36V) in FP64 (no engine literals, guard, floor or rounding)."""
    x = np.asarray(x, float)
    e01, e02, e03, e12, e13 = x[1] - x[0], x[2] - x[0], x[3] - x[0], x[2] - x[1], x[3] - x[1]
    N = (np.cross(e13, e12), np.cross(e02, e03), np.cross(e03, e01), np.cross(e01, e02))
    V = abs(float(e01 @ N[1])) / 6.0
    return V, {(i, j): -float(N[i] @ N[j]) / (36.0 * V) for i, j in EDGES}


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
        _, wf = p1_weights_normals(x)
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
    name = "dec.engine_edge_weight_mapping"
    try:
        variant, anchors = detect_variant(engine_root)
    except C.AnchorMissing as ex:
        return C.not_built(name, str(ex))
    if variant is None:
        return C.record(name, {}, None, None, None, None,
                        "TRANSCRIPTION_STALE: neither the legacy nor the P3 evaluator text was found",
                        defect_id="D-DEC-OPP-EDGE", anchors=anchors)
    tets = _tets()
    a = np.array([0.3, -1.1, 0.7])
    err_engine, err_swap_identity, energy_engine = [], [], []
    for x in tets:
        V, wg = p1_weights_gradients(x)
        _, we = engine_weights(x, variant=variant, floor=False)  # isolate the mapping
        scale = max(abs(v) for v in wg.values())
        err_engine.append(max(abs(we[e] - wg[e]) for e in EDGES) / scale)
        err_swap_identity.append(max(abs(we[e] - wg[SWAP[e]]) for e in EDGES) / scale)
        energy_engine.append(abs(_dirichlet(we, x, a) - V * float(a @ a)) / (V * float(a @ a)))
    reg = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    _, wr_e = engine_weights(reg, variant=variant, floor=False)
    _, wr_g = p1_weights_gradients(reg)
    regular_err = max(abs(wr_e[e] - wr_g[e]) for e in EDGES) / max(wr_g.values())
    x0 = tets[0]
    _, we0 = engine_weights(x0, variant=variant)
    _, wg0 = p1_weights_gradients(x0)
    example = {f"w{i}{j}": {"engine": we0[(i, j)], "correct": wg0[(i, j)]} for i, j in EDGES}
    obs = {
        "source_variant": variant,
        "max_rel_err_engine_vs_correct": max(err_engine),
        "median_rel_err_engine_vs_correct": float(np.median(err_engine)),
        "max_rel_err_engine_vs_correct_of_opposite_edge": max(err_swap_identity),
        "max_rel_err_linear_dirichlet_energy_engine": max(energy_engine),
        "regular_tet_rel_err": regular_err,
        "worked_example_tet0": {"coords": x0.tolist(), "weights": example},
    }
    tol = 1e-6  # FP32 output rounding (~6e-8) and float32 literals; legacy Tikhonov guard adds ~1e-5
    passed = max(err_engine) <= tol
    if variant == "legacy_opposite_edge":
        notes = ("Legacy source: engine weight of edge ij equals the correct weight of the opposite edge kl to "
                 f"{max(err_swap_identity):.1e} (only the Tikhonov guard differs): an exact 01<->23, 02<->13, "
                 "03<->12 swap. Invisible on a regular tetrahedron. Per-tet floor disabled here to isolate the mapping.")
    else:
        notes = ("P3 signed source: w_ij <- -N_i.N_j/(36V) with the max() sliver guard (exactly 1/V for "
                 "V >= sigma0 l_rms^3, true for all 64 tets). Residual is FP32 output rounding and float32 literals.")
    return C.record(name, {"tets": "64 seeded non-regular tetrahedra", "sigma0": _f(1e-4)},
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


def _lin_residual(st1, xq, centre, exact=False):
    """sum_j w_cj (x_j - x_c) at an interior vertex, plus the scale sum_j |w_cj| |x_j - x_c|."""
    acc = [Fraction(0)] * 3 if exact else np.zeros(3)
    scale = 0.0
    for (a, b), w in st1.items():
        if centre in (a, b):
            o = b if a == centre else a
            d = [xq[o][m] - xq[centre][m] for m in range(3)]
            for m in range(3):
                acc[m] += w * (d[m] if exact else float(d[m]))
            scale += abs(float(w)) * float(sum(float(v) ** 2 for v in d)) ** 0.5
    return [float(v) for v in acc], scale


def check_star_exact(engine_root=None):
    name = "dec.star0_star1_exact_recompute"
    try:
        variant, anchors = detect_variant(engine_root)
    except C.AnchorMissing as ex:
        return C.not_built(name, str(ex))
    if variant is None:
        return C.record(name, {}, None, None, None, None,
                        "TRANSCRIPTION_STALE: neither the legacy nor the P3 evaluator text was found",
                        defect_id="D-DEC-OPP-EDGE", anchors=anchors)
    xq, tets = _kuhn_mesh()
    ex0, ex1 = _assemble(xq, tets, exact_weights)
    keys = sorted(ex1)

    def eng(**kw):
        fn = lambda xs: engine_weights(np.array([[float(c) for c in p] for p in xs]), variant=variant, **kw)
        s0, s1 = _assemble(xq, tets, fn)
        return s0, s1

    def assembled(s1, floor):  # bp_dec_assembled_star1 / legacy fmaxf(star1, 1e-4f), stored as float
        return {k: _f(max(v, _f(1e-4)) if floor else v) for k, v in s1.items()}

    P = VARIANTS[variant]
    en0, en1_raw = eng()
    en1 = assembled(en1_raw, P["assembled_floor"])
    s1max = max(abs(float(ex1[k])) for k in keys)
    rel = lambda d: max(abs(d[k] - float(ex1[k])) for k in keys) / s1max
    star0_err = max(abs(en0[v] - float(ex0[v])) / float(ex0[v]) for v in ex0)
    lin_e, _ = _lin_residual(ex1, xq, 13, exact=True)
    lin_g, lin_scale = _lin_residual(en1, xq, 13)
    lin_rel = max(abs(v) for v in lin_g) / lin_scale
    # isolate effects (both variants): mapping, per-tet floor, assembled floor
    alt = {}
    for label, kw, af in (("mapping_own_with_legacy_floors (= P3 built with -DBIPRISM_DEC_LEGACY_POSITIVE_FLOOR=1)",
                           dict(mapping="own", floor=True), True),
                          ("mapping_own_no_floors", dict(mapping="own", floor=False), False)):
        _, s1 = eng(**kw)
        s1 = assembled(s1, af)
        r, sc = _lin_residual(s1, xq, 13)
        alt[label] = {"max_rel_err_star1": rel(s1), "interior_linear_precision": r,
                      "interior_linear_precision_rel": max(abs(v) for v in r) / sc}
    obs = {
        "source_variant": variant,
        "n_vertices": len(xq), "n_tets": len(tets), "n_edges": len(keys),
        "sum_star0_exact": str(sum(ex0.values())), "sum_star0_engine": sum(en0.values()),
        "max_rel_err_star0_engine": star0_err,
        "max_rel_err_star1_engine_as_is": rel(en1),
        "exact_negative_star1_edges": sum(1 for k in keys if ex1[k] < 0),
        "interior_linear_precision_exact": lin_e,
        "interior_linear_precision_engine_as_is": lin_g,
        "interior_linear_precision_engine_rel": lin_rel,
        "effect_isolation": alt,
    }
    tol = 1e-6
    passed = rel(en1) <= tol and star0_err <= tol and lin_rel <= tol
    notes = ("Exact rational star0/star1 on a perturbed Kuhn mesh (48 tets, 17/98 exact weights negative). "
             "star0 (barycentric V/4) matches in both variants. Legacy: star1 wrong (opposite-edge map) and "
             "both floors (per-tet fmax(w,1e-3 l_rms), assembled fmax(star1,1e-4)) break linear precision even "
             "after the map fix. P3 signed: star1 and interior linear precision match exact to FP32 rounding; the "
             "deprecated legacy-floor build (-DBIPRISM_DEC_LEGACY_POSITIVE_FLOOR=1) is reported for reference and "
             "still breaks linear precision. The dual is barycentric (V/4), not Voronoi.")
    return C.record(name, {"mesh": "3x3x3 grid, spacing 3/2, rational perturbation k/64 (|k|<=12), Kuhn split"},
                    "engine star0/star1 == exact rational values; interior linear precision residual <= 1e-6 "
                    "of sum|w||dx|", obs, passed, tol, notes, defect_id="D-DEC-OPP-EDGE", anchors=anchors)


CHECKS = [check_reference_selftest, check_engine_mapping, check_star_exact]
