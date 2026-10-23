"""UV kick RNG: engine LCG seeding vs counter-based Philox4x32-10 keying.

Engine (nhs_amber_fused.cu ~11068, nhs_uv_pump_probe_step) seeds
apply_vibrational_transfer (nhs_excited_state.cuh:502-560) with
    seed = timestep * n_aromatics + arom_idx
and draws rng = rng*1103515245 + 12345 (mod 2^32). The seed carries no stream,
replica or purpose term, so every stream with the same protocol step and
aromatic set draws identical kick directions.

Reference: Philox4x32-10 (Salmon et al., SC'11; Random123), verified against
the Random123 known-answer vectors, keyed by (seed, stream) with counter
(atom, step_lo, step_hi16|purpose8|replica_epoch8) -- an injective packing for
seed,stream < 2^32, atom < 2^32, step < 2^48, purpose < 2^8, epoch < 2^8.
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

M32 = 0xFFFFFFFF
AF = C.KERNELS / "nhs_amber_fused.cu"
EX = C.KERNELS / "nhs_excited_state.cuh"


# ----------------------------------------------------------------- engine LCG
def engine_kicks(seed, n_neighbors, ring_normal=(0.0, 0.0, 1.0)):
    """Transcription of the direction choice in apply_vibrational_transfer."""
    rng = seed & M32
    out = []
    for _ in range(n_neighbors):
        rng = (rng * 1103515245 + 12345) & M32
        r = (rng & 0x7FFFFFFF) / float(0x7FFFFFFF)
        if r < 0.6:
            d = ring_normal if not ((rng >> 16) & 1) else tuple(-c for c in ring_normal)
        else:
            rng = (rng * 1103515245 + 12345) & M32
            rx = ((rng >> 0) & 0xFF) / 127.5 - 1.0
            ry = ((rng >> 8) & 0xFF) / 127.5 - 1.0
            rz = ((rng >> 16) & 0xFF) / 127.5 - 1.0
            ln = math.sqrt(rx * rx + ry * ry + rz * rz)
            d = (rx / ln, ry / ln, rz / ln)
        out.append(d)
    return out


# ----------------------------------------------------------------- Philox4x32-10
PHILOX_M0, PHILOX_M1 = 0xD2511F53, 0xCD9E8D57
PHILOX_W0, PHILOX_W1 = 0x9E3779B9, 0xBB67AE85


def philox4x32_10(ctr, key):
    c0, c1, c2, c3 = ctr
    k0, k1 = key
    for r in range(10):
        if r:
            k0 = (k0 + PHILOX_W0) & M32
            k1 = (k1 + PHILOX_W1) & M32
        p0 = PHILOX_M0 * c0
        p1 = PHILOX_M1 * c2
        hi0, lo0 = p0 >> 32, p0 & M32
        hi1, lo1 = p1 >> 32, p1 & M32
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
    return (c0, c1, c2, c3)


def keyed_draw(seed, stream, replica_epoch, atom, step, purpose):
    assert 0 <= seed < 2 ** 32 and 0 <= stream < 2 ** 32 and 0 <= atom < 2 ** 32
    assert 0 <= step < 2 ** 48 and 0 <= purpose < 2 ** 8 and 0 <= replica_epoch < 2 ** 8
    ctr = (atom, step & M32, ((step >> 32) << 16) | (purpose << 8) | replica_epoch, 0)
    return philox4x32_10(ctr, (seed, stream))


KAT = [  # Random123 kat_vectors, philox4x32 10 rounds
    ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    ((M32, M32, M32, M32), (M32, M32), (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
    ((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0),
     (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1)),
]


def check_uv_lcg_streams(engine_root=None):
    try:
        anchors = [C.anchor(engine_root, AF, 11060, 11072, ["timestep * n_aromatics + arom_idx,"]),
                   C.anchor(engine_root, AF, 11018, 11018, ["const int timestep = (int)d_protocol->current_step;"]),
                   C.anchor(engine_root, EX, 530, 560, ["rng = rng * 1103515245 + 12345;",
                                                        "float r = (float)(rng & 0x7FFFFFFF) / (float)0x7FFFFFFF;",
                                                        "if ((rng >> 16) & 1) kick_dir"])]
    except C.AnchorMissing as ex:
        return C.not_built("rng.uv_lcg_stream_identity", str(ex))
    n_streams, n_arom, n_nb, steps = 8, 12, 20, 200
    identical, total = 0, 0
    for t in range(1000, 1000 + steps):
        for a in range(n_arom):
            seqs = [engine_kicks(t * n_arom + a, n_nb) for _ in range(n_streams)]  # seed has no stream term
            total += 1
            identical += all(s == seqs[0] for s in seqs)
    # serial structure across consecutive seeds: first draw r(s+1)-r(s) mod 1 is constant
    r1 = [(((s * 1103515245 + 12345) & M32) & 0x7FFFFFFF) / float(0x7FFFFFFF) for s in range(5000, 5400)]
    inc = np.mod(np.diff(r1), 1.0)
    # with per-stream Philox keying, how many (t,a) give identical first draws across streams?
    ph_ident = 0
    for t in range(1000, 1000 + steps):
        for a in range(n_arom):
            draws = {keyed_draw(42, s, 0, a, t, 1) for s in range(n_streams)}
            ph_ident += len(draws) < n_streams
    obs = {"streams": n_streams, "aromatics": n_arom, "neighbors": n_nb, "steps": steps,
           "fraction_identical_kick_sequences_across_streams": identical / total,
           "first_draw_increment_between_consecutive_seeds": {"mean": float(inc.mean()), "std": float(inc.std())},
           "philox_keyed_collisions_across_streams": ph_ident}
    tol = 0.0
    notes = ("seed = timestep*n_aromatics + arom_idx has no stream/replica/purpose term, so all streams sharing "
             "current_step draw identical kick directions (fraction 1.0): stream-to-stream UV responses are not "
             "independent samples. Consecutive seeds give first draws that advance by a constant 0.5139 mod 1 "
             "(std 0), a deterministic lattice. Philox keyed by (seed, stream) gives no cross-stream collisions.")
    return C.record("rng.uv_lcg_stream_identity",
                    {"model": "8 streams, 12 aromatics, 20 neighbours, 200 steps"},
                    "kick sequences differ across streams (fraction identical == 0)",
                    obs, (identical / total <= tol) if C.anchors_ok(anchors) else None, tol, notes,
                    defect_id="D-RNG-UV-LCG-STREAM", anchors=anchors)


def check_philox_kat(engine_root=None):
    res = []
    for ctr, key, exp in KAT:
        got = philox4x32_10(ctr, key)
        res.append({"ctr": [hex(c) for c in ctr], "key": [hex(k) for k in key],
                    "expected": [hex(e) for e in exp], "observed": [hex(g) for g in got], "ok": got == exp})
    return C.record("rng.philox4x32_10_known_answer", {"vectors": "Random123 kat_vectors philox4x32_10"},
                    "bit-exact Random123 outputs", res, all(r["ok"] for r in res), 0,
                    "Reference implementation used by the stream-independence test and the H0 target keying.")


def check_philox_stream_independence(engine_root=None):
    n_streams, n_atoms, n_steps = 8, 64, 64
    U = np.empty((n_streams, n_atoms * n_steps * 4))
    for s in range(n_streams):
        vals = []
        for step in range(n_steps):
            for atom in range(n_atoms):
                vals.extend(keyed_draw(20261023, s, 0, atom, step, 1))
        U[s] = (np.array(vals, dtype=np.float64) + 0.5) / 2 ** 32
    N = U.shape[1]
    corr = np.corrcoef(U)
    off = np.abs(corr[~np.eye(n_streams, dtype=bool)])
    bins = np.floor(U.ravel() * 64).astype(int)
    counts = np.bincount(bins, minlength=64)
    expct = U.size / 64
    chi2 = float(((counts - expct) ** 2 / expct).sum())
    # same key, different purpose / replica epoch must also decorrelate
    alt = np.array([keyed_draw(20261023, 0, 0, a, st, 2)[0] for st in range(n_steps) for a in range(n_atoms)], float)
    base = np.array([keyed_draw(20261023, 0, 0, a, st, 1)[0] for st in range(n_steps) for a in range(n_atoms)], float)
    c_purpose = abs(float(np.corrcoef(alt, base)[0, 1]))
    det = keyed_draw(1, 2, 3, 4, 5, 6) == keyed_draw(1, 2, 3, 4, 5, 6)
    # chi2 with 63 dof: mean 63, sd ~11.2; bound at mean + 5 sd
    bound_corr = 5 / math.sqrt(N)
    ok = off.max() < bound_corr and chi2 < 63 + 5 * math.sqrt(126) and c_purpose < 5 / math.sqrt(len(alt)) and det
    obs = {"samples_per_stream": N, "max_abs_cross_stream_corr": float(off.max()), "corr_bound_5sigma": bound_corr,
           "uniformity_chi2_63dof": chi2, "purpose_decorrelation_abs_corr": c_purpose, "deterministic_replay": det}
    return C.record("rng.philox_stream_independence",
                    {"key": "(seed, stream)", "counter": "(atom, step_lo32, step_hi16<<16|purpose<<8|replica_epoch, 0)",
                     "streams": n_streams, "atoms": n_atoms, "steps": n_steps},
                    "cross-stream |corr| < 5/sqrt(N); chi2 < mean+5sd; purpose keys decorrelate; bitwise replay",
                    obs, ok, {"corr": bound_corr, "chi2": 63 + 5 * math.sqrt(126)},
                    "Reference keying for H0 (counter-based, order-independent, stream/replica/purpose separated).")


CHECKS = [check_uv_lcg_streams, check_philox_kat, check_philox_stream_independence]
