"""Neighbour list: engine cell scheme vs brute-force O(N^2) reference.

Engine: CELL_SIZE = 10 A (nhs_amber_fused.cu:167), grid origin = initial
bounding-box min - 10 A, n = ceil((extent+20)/10) (fused_engine.rs:12272-12305 at engine 78a2482+; was 12355-12388),
cell index by truncation with clamping (build_cell_list, nhs_amber_fused.cu
~11704-11717), 27-cell stencil (nhs_neighbor_warp.cuh:42-51 base, 172-181 compensated), list cutoff
r^2 <= (10 A)^2 * 1.44 = (12 A)^2 (fused_engine.rs:16673). The potential cuts at
10 A (NB_CUTOFF); the 10-12 A buffer exists so the list can be reused for
20 steps. A pair with r in (10,12] whose atoms sit two cells apart along any
axis is never examined by the 27-cell stencil, so it is missed.
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

CELL = np.float32(10.0)
CELL_INV = np.float32(0.1)
F32 = np.float32
AF = C.KERNELS / "nhs_amber_fused.cu"
NW = C.KERNELS / "nhs_neighbor_warp.cuh"
FE = C.NHS_SRC / "fused_engine.rs"


def _anchors(engine_root):
    return [
        C.anchor(engine_root, AF, 165, 171, ["#define NB_CUTOFF 10.0f", "#define CELL_SIZE 10.0f", "#define CELL_SIZE_INV 0.1f"]),
        C.anchor(engine_root, AF, 11704, 11717, ["int ix = (int)((x - origin_x) * CELL_SIZE_INV);",
                                                 "ix = max(0, min(ix, nx - 1));", "int cell_idx = ix + iy * nx + iz * nx * ny;"]),
        C.anchor(engine_root, NW, 42, 51, ["for (int dz=-1; dz<=1; ++dz) {", "if (z<0 || z>=nz) continue;",
                                           "for (int dx=-1; dx<=1; ++dx) {"]),
        C.anchor(engine_root, NW, 172, 181, ["for (int dz=-1; dz<=1; ++dz) {", "for (int dx=-1; dx<=1; ++dx) {"]),
        C.anchor(engine_root, FE, 12272, 12305, ["const CELL_SIZE: f32 = 10.0;", "let cell_padding = CELL_SIZE;",
                                                 "let value = ((extent[axis] / CELL_SIZE).ceil() as i64).max(1);"]),
        C.anchor(engine_root, FE, 16671, 16674, ["let cutoff_sq_with_buffer = self.cutoff * self.cutoff * 1.44;"]),
    ]


def grid_from_bbox(pos0):
    lo, hi = pos0.min(0).astype(F32), pos0.max(0).astype(F32)
    origin = lo - CELL
    extent = (hi - lo) + F32(2) * CELL
    n = np.maximum(np.ceil(extent / CELL).astype(np.int64), 1)
    return origin, n


def cell_pairs(pos, origin, n, cutoff_sq):
    """Transcribed cell scheme; returns set of (i<j) pairs kept, and max cell occupancy."""
    p = pos.astype(F32)
    g = ((p - origin.astype(F32)) * CELL_INV).astype(F32)
    idx = np.trunc(g).astype(np.int64)                     # (int) cast truncates toward zero
    idx = np.clip(idx, 0, n - 1)
    cid = idx[:, 0] + idx[:, 1] * n[0] + idx[:, 2] * n[0] * n[1]
    cells = {}
    for a, c in enumerate(cid):
        cells.setdefault(int(c), []).append(a)
    kept = set()
    for c, members in cells.items():
        cx, cy, cz = c % n[0], (c // n[0]) % n[1], c // (n[0] * n[1])
        cand = []
        for dz in (-1, 0, 1):
            z = cz + dz
            if z < 0 or z >= n[2]:
                continue
            for dy in (-1, 0, 1):
                y = cy + dy
                if y < 0 or y >= n[1]:
                    continue
                for dx in (-1, 0, 1):
                    x = cx + dx
                    if x < 0 or x >= n[0]:
                        continue
                    cand.extend(cells.get(int(x + y * n[0] + z * n[0] * n[1]), []))
        if not cand:
            continue
        cand = np.array(cand)
        for i in members:
            d = p[cand] - p[i]
            r2 = (d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2]).astype(F32)
            for j in cand[~(r2 > cutoff_sq)]:
                if j != i:
                    kept.add((min(i, int(j)), max(i, int(j))))
    occ = max(len(m) for m in cells.values())
    return kept, occ


def brute_pairs(pos, cutoff_sq):
    p = pos.astype(F32)
    out, r_of = set(), {}
    N = len(p)
    for i in range(N - 1):
        d = p[i + 1:] - p[i]
        r2 = (d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2]).astype(F32)
        js = np.nonzero(~(r2 > cutoff_sq))[0]
        for j, rr in zip(js + i + 1, r2[js]):
            out.add((i, int(j)))
            r_of[(i, int(j))] = math.sqrt(float(rr))
    return out, r_of


def _analyse(pos):
    cutoff_sq = F32(F32(10.0) * F32(10.0)) * F32(1.44)
    origin, n = grid_from_bbox(pos)
    kept, occ = cell_pairs(pos, origin, n, cutoff_sq)
    ref, r_of = brute_pairs(pos, cutoff_sq)
    missed = ref - kept
    extra = kept - ref
    rs = np.array([r_of[m] for m in missed]) if missed else np.zeros(0)
    buf = [k for k in ref if r_of[k] > 10.0]
    return {"n_atoms": int(len(pos)), "grid": n.tolist(), "cutoff_sq": float(cutoff_sq),
            "pairs_le_12A_bruteforce": len(ref), "pairs_kept_cell_scheme": len(kept),
            "pairs_in_buffer_10_12A": len(buf),
            "missed_total": len(missed), "missed_le_10A": int((rs <= 10.0).sum()),
            "missed_10_10p25A": int(((rs > 10.0) & (rs <= 10.25)).sum()),
            "missed_10_10p5A": int(((rs > 10.0) & (rs <= 10.5)).sum()),
            "missed_10_11A": int(((rs > 10.0) & (rs <= 11.0)).sum()),
            "missed_10_12A": int(((rs > 10.0) & (rs <= 12.0)).sum()),
            "missed_fraction_of_buffer_pairs": (len(missed) / len(buf)) if buf else 0.0,
            "spurious_pairs": len(extra), "max_cell_occupancy": occ}


def synthetic_cloud(n=600, radius=11.3, dmin=1.0, seed=20261023):
    """Deterministic protein-density point cloud (~0.1 atoms/A^3), offset from the origin."""
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        if p @ p > radius * radius:
            continue
        if pts and np.min(np.sum((np.array(pts) - p) ** 2, 1)) < dmin * dmin:
            continue
        pts.append(p)
    return np.array(pts) + np.array([31.7, -12.3, 7.9])


def check_neighbor_cells(engine_root=None):
    try:
        anchors = _anchors(engine_root)
    except C.AnchorMissing as ex:
        return C.not_built("neighbor.cell_scheme_vs_bruteforce", str(ex))
    systems, inputs = {}, {}
    systems["synthetic_600"] = _analyse(synthetic_cloud())
    inputs["synthetic_600"] = "seeded uniform sphere r=11.3 A, dmin 1 A, seed 20261023"
    for name in ("IL2", "1BTL"):
        topo, path, sha = C.load_topology(name)
        if topo is None:
            inputs[name] = f"INPUT_NOT_FOUND: {path}"
            continue
        pos = np.array(topo["positions"], float).reshape(-1, 3)
        inputs[name] = {"path": path, "sha256": sha, "n_atoms": int(len(pos))}
        systems[name] = _analyse(pos)
        if name == "IL2":
            # grid fixed at init; the solute then drifts 25 A along x past the 10 A padding
            origin, n = grid_from_bbox(pos)
            moved = pos + np.array([25.0, 0.0, 0.0])
            cutoff_sq = F32(F32(10.0) * F32(10.0)) * F32(1.44)
            kept, occ = cell_pairs(moved, origin, n, cutoff_sq)
            ref, _ = brute_pairs(moved, cutoff_sq)
            systems["IL2_drift_25A_x_fixed_grid"] = {"missed_total": len(ref - kept), "spurious_pairs": len(kept - ref),
                                                     "max_cell_occupancy": occ, "MAX_ATOMS_PER_CELL": 512,
                                                     "occupancy_at_rest": systems["IL2"]["max_cell_occupancy"]}
    real = [k for k in ("IL2", "1BTL") if k in systems]
    missed_le10 = sum(systems[k]["missed_le_10A"] for k in systems if "missed_le_10A" in systems[k])
    missed_buf = sum(systems[k]["missed_10_12A"] for k in systems if "missed_10_12A" in systems[k])
    obs = {"systems": systems, "real_systems_evaluated": real, "missed_within_potential_cutoff_total": missed_le10,
           "missed_in_buffer_total": missed_buf}
    passed = missed_buf == 0 and missed_le10 == 0
    notes = ("Brute force uses the engine's FP32 r^2 and the same <= comparison. No pair within the 10 A potential "
             "cutoff is missed at build time; pairs in the 10-12 A buffer two cells apart are, so a pair that closes "
             "inside 10 A during the 20-step reuse window (no displacement trigger) is absent from the force sum "
             "until the next rebuild. Clamping is not a miss mechanism (it makes cells closer); after drift past the "
             "fixed 10 A padding it piles atoms into face cells (occupancy reported vs the 512 trap) and the list "
             "is over-inclusive. Fix options: cell size >= list cutoff (12 A), or a 5x5x5 stencil, plus a "
             "half-skin displacement trigger.")
    if not real:
        notes = "INPUT_NOT_FOUND for IL2/1BTL (synthetic system still evaluated). " + notes
    return C.record("neighbor.cell_scheme_vs_bruteforce", inputs,
                    "cell-scheme pair set == brute-force pair set for r <= 12 A (the list cutoff)",
                    obs, passed if C.anchors_ok(anchors) else None, 0, notes,
                    defect_id="D-NBL-MISS-10-12", anchors=anchors)


CHECKS = [check_neighbor_cells]
