"""Electronic two-level propagator: engine Cayley vs exact SU(2) exponential.

Engine map (nhs_biprism_electronic_discrete.cuh:197-231, bp_ed_hermitian_cayley):
    (omega I + i h H/2) psi1 = (omega I - i h H/2) psi0,  hbar = 1, atomic units,
called with H = omega * diag(eps_s, eps_p) (+0.15 phi on h11,
bp_ed_onsite_dec_hamiltonian :148-150) and h = selected_dt/4 in atomic time
(nhs_biprism_variational_so3g_v10.cuh:4363, four committed substeps per body,
nhs_biprism_mixed_energy.cuh:383). Onsite energies come from c_gfn_xtb_lut
(nhs_biprism_xtb_lut.cuh:21-30).

For eigenvalue E the Cayley substep multiplies by exp(-2i atan(hE/2)), so the
effective energy is (2/h) atan(hE/2): compressed for |hE| >~ 1 and NOT
invariant under a constant energy shift. The exact propagator is
exp(-i h H) (closed-form SU(2) x U(1)).
"""
from __future__ import annotations

import cmath
import math

import numpy as np

try:
    from . import _common as C
except ImportError:
    import importlib.util, pathlib, sys
    _p = pathlib.Path(__file__).with_name("_common.py")
    _s = importlib.util.spec_from_file_location("prism_refs_common", _p)
    C = importlib.util.module_from_spec(_s); sys.modules.setdefault("prism_refs_common", C); _s.loader.exec_module(C)

PS_TO_AU = 41341.37333517015  # engine unit bridge; CODATA check lives in units.py
FS_TO_AU = PS_TO_AU / 1000.0
SUBSTEPS = 4
DTS_FS = (0.5, 2.0, 4.0)
# c_gfn_xtb_lut (h_s_valence, h_p_excited) in Hartree
LUT = {"C": (-0.558, -0.392), "N": (-0.682, -0.445), "O": (-0.825, -0.510), "H": (-0.500, -0.120)}
ED = C.KERNELS / "nhs_biprism_electronic_discrete.cuh"


def _anchors(engine_root):
    return [
        C.anchor(engine_root, ED, 197, 231, [
            "const R a=R(.5)*dt*H.h00,d=R(.5)*dt*H.h11,tr=R(.5)*dt*H.h01_re,ti=R(.5)*dt*H.h01_im;",
            "const BiprismElectronicComplex<R> A={w,aa},B={-ii,rr},C={ii,rr},D={w,dd};",
            "const BiprismElectronicSpinor<R> p1={bp_ed_sub(bp_ed_scale(y0,R(2)),p0.s0),bp_ed_sub(bp_ed_scale(y1,R(2)),p0.s1)};"]),
        C.anchor(engine_root, ED, 148, 150, ["return {omega*eps_s,bp_ed_fma(R(0.15f),phi_bar,omega*eps_p),R(0),R(0)};"]),
        C.anchor(engine_root, C.KERNELS / "nhs_biprism_variational_so3g_v10.cuh", 4363, 4363,
                 ["s->invocation.h=selected_dt*.25f;"]),
        C.anchor(engine_root, C.KERNELS / "nhs_biprism_xtb_lut.cuh", 22, 25, [
            "{1.240f, -0.500f, -0.120f, 0.472f}, // 0: H", "{1.625f, -0.558f, -0.392f, 0.380f}, // 1: C",
            "{1.950f, -0.682f, -0.445f, 0.435f}, // 2: N", "{2.275f, -0.825f, -0.510f, 0.495f}, // 3: O"]),
    ]


# ------------------------------------------------------------------ transcription
def cayley_engine(p0, h00, h11, h01_re, h01_im, omega, dt):
    """Literal FP64 transcription of bp_ed_hermitian_cayley (scaled Cramer solve)."""
    s0, s1 = p0
    a, d, tr, ti = .5 * dt * h00, .5 * dt * h11, .5 * dt * h01_re, .5 * dt * h01_im
    scale = max(omega, abs(a), abs(d), abs(tr), abs(ti))
    w, aa, dd, rr, ii = omega / scale, a / scale, d / scale, tr / scale, ti / scale
    A, B, Cc, D = complex(w, aa), complex(-ii, rr), complex(ii, rr), complex(w, dd)
    det = A * D - B * Cc
    ds = max(abs(det.real), abs(det.imag))
    dr, di = det.real / ds, det.imag / ds
    den = dr * dr + di * di
    b0, b1 = s0 * w, s1 * w
    n_0 = D * b0 - B * b1
    n_1 = A * b1 - Cc * b0
    n0r, n0i, n1r, n1i = n_0.real / ds, n_0.imag / ds, n_1.real / ds, n_1.imag / ds
    y0 = complex((n0r * dr + n0i * di) / den, (n0i * dr - n0r * di) / den)
    y1 = complex((n1r * dr + n1i * di) / den, (n1i * dr - n1r * di) / den)
    return (2 * y0 - s0, 2 * y1 - s1)


def expm_2x2(h00, h11, h01, t):
    """exp(-i t H) for Hermitian 2x2 H, closed form."""
    m = .5 * (h00 + h11)
    dz = .5 * (h00 - h11)
    r = math.sqrt(dz * dz + abs(h01) ** 2)
    c, s = math.cos(t * r), (math.sin(t * r) / r if r > 0 else t)
    ph = cmath.exp(-1j * t * m)
    return ((ph * complex(c, -s * dz), ph * (-1j * s * h01)),
            (ph * (-1j * s * h01.conjugate()), ph * complex(c, s * dz)))


def _apply(U, p):
    return (U[0][0] * p[0] + U[0][1] * p[1], U[1][0] * p[0] + U[1][1] * p[1])


def effective_gap(e0, e1, dt_fs, omega=1.0):
    """Effective level splitting of SUBSTEPS engine-Cayley substeps per nuclear step."""
    h = dt_fs * FS_TO_AU / SUBSTEPS
    ph = []
    for k, E in enumerate((e0, e1)):
        p = (1 + 0j, 0j) if k == 0 else (0j, 1 + 0j)
        tot = 0.0
        for _ in range(SUBSTEPS):
            q = cayley_engine(p, omega * e0, omega * e1, 0.0, 0.0, omega, h)
            tot += -cmath.phase(q[k] / p[k])
            p = q
        ph.append(tot)
    numeric = (ph[1] - ph[0]) / (dt_fs * FS_TO_AU)
    analytic = (2 / h) * (math.atan(h * e1 / 2) - math.atan(h * e0 / 2))
    return numeric, analytic


# ------------------------------------------------------------------ checks
def check_transcription_selftest(engine_root=None):
    rng = np.random.default_rng(11)
    worst_solve, worst_norm, worst_omega = 0.0, 0.0, 0.0
    for _ in range(200):
        h00, h11 = rng.normal(scale=0.5, size=2)
        h01 = complex(*rng.normal(scale=0.05, size=2))
        dt = float(rng.uniform(1, 60))
        om = float(rng.uniform(0.2, 5))
        p = rng.normal(size=2) + 1j * rng.normal(size=2)
        p = p / np.linalg.norm(p)
        q = cayley_engine((p[0], p[1]), om * h00, om * h11, om * h01.real, om * h01.imag, om, dt)
        H = np.array([[h00, h01], [h01.conjugate(), h11]])
        M = np.eye(2) + 0.5j * dt * H
        ref = np.linalg.solve(M, (np.eye(2) - 0.5j * dt * H) @ p)
        worst_solve = max(worst_solve, float(np.max(np.abs(np.array(q) - ref))))
        worst_norm = max(worst_norm, abs(abs(q[0]) ** 2 + abs(q[1]) ** 2 - 1))
        q1 = cayley_engine((p[0], p[1]), h00, h11, h01.real, h01.imag, 1.0, dt)
        worst_omega = max(worst_omega, float(np.max(np.abs(np.array(q) - np.array(q1)))))
    obs = {"max_abs_diff_vs_dense_solve": worst_solve, "max_norm_defect": worst_norm,
           "max_abs_diff_omega_scaling": worst_omega}
    tol = 1e-12
    return C.record("electronic.cayley_transcription_selftest",
                    {"cases": "200 seeded Hermitian 2x2, omega in [0.2,5], dt in [1,60] au"},
                    "transcription == (I + i dt H/2)^-1 (I - i dt H/2); unitary; H=omega*h makes omega cancel",
                    obs, all(v <= tol for v in obs.values()), tol,
                    "Reference-only: validates the transcription that the gap/Rabi/LZ checks use.")


def check_effective_gap(engine_root=None):
    try:
        anchors = _anchors(engine_root)
    except C.AnchorMissing as ex:
        return C.not_built("electronic.cayley_effective_gap", str(ex))
    table = []
    for el, (es, ep) in LUT.items():
        gap = ep - es
        for dt in DTS_FS:
            plan_num, plan_an = effective_gap(0.0, gap, dt)
            eng_num, eng_an = effective_gap(es, ep, dt)
            table.append({"element": el, "gap_Ha": round(gap, 6), "dt_fs": dt,
                          "h_substep_au": dt * FS_TO_AU / SUBSTEPS,
                          "eff_gap_ground_at_zero_Ha": plan_num, "analytic_ground_at_zero": plan_an,
                          "eff_gap_engine_absolute_onsite_Ha": eng_num, "analytic_engine": eng_an,
                          "compression_ground_at_zero": gap / plan_num,
                          "compression_engine_absolute": gap / eng_num,
                          "phase_error_rad_per_ps_engine": abs(gap - eng_num) * PS_TO_AU})
    h4 = next(r for r in table if r["element"] == "H" and r["dt_fs"] == 4.0)
    worst = max(r["phase_error_rad_per_ps_engine"] for r in table)
    obs = {"table": table,
           "plan_claim_0p38Ha_4fs": {"plan_states_Ha": 0.070, "reproduced_ground_at_zero_Ha": h4["eff_gap_ground_at_zero_Ha"],
                                     "compression": h4["compression_ground_at_zero"],
                                     "engine_absolute_onsite_Ha": h4["eff_gap_engine_absolute_onsite_Ha"],
                                     "engine_compression": h4["compression_engine_absolute"]},
           "max_phase_error_rad_per_ps": worst}
    tol = 1e-3  # H1 gate: Rabi/onsite phase <= 1e-3 rad/ps
    notes = ("Plan figure reproduced only in the ground-at-zero convention (0.38 Ha -> "
             f"{h4['eff_gap_ground_at_zero_Ha']:.4f} Ha at 4 fs, {h4['compression_ground_at_zero']:.2f}x). "
             "The engine passes ABSOLUTE onsite energies (eps_s, eps_p both negative) and Cayley is not "
             "shift-invariant, so the realised splitting is smaller still "
             f"({h4['eff_gap_engine_absolute_onsite_Ha']:.4f} Ha, {h4['compression_engine_absolute']:.1f}x for H at 4 fs). "
             "Even 0.5 fs fails the 1e-3 rad/ps gate by orders of magnitude.")
    if not C.anchors_ok(anchors):
        notes = "TRANSCRIPTION_STALE: " + notes
    return C.record("electronic.cayley_effective_gap",
                    {"substeps": SUBSTEPS, "dts_fs": DTS_FS, "lut_Ha": LUT, "omega": 1.0, "phi_bar": 0.0},
                    "effective splitting == exact gap within 1e-3 rad/ps",
                    obs, (worst <= tol) if C.anchors_ok(anchors) else None, tol, notes,
                    defect_id="D-ELEC-CAYLEY-GAP", anchors=anchors)


def _rabi_case(es, ep, V, dt_fs, t_ps=1.0):
    h = dt_fs * FS_TO_AU / SUBSTEPS
    nsub = int(round(t_ps * PS_TO_AU / h))
    lam = np.linalg.eigvalsh(np.array([[es, V], [V, ep]]))
    om_exact = lam[1] - lam[0]
    om_cay = (2 / h) * (math.atan(h * lam[1] / 2) - math.atan(h * lam[0] / 2))
    U = expm_2x2(es, ep, complex(V), h)
    p, q = (1 + 0j, 0j), (1 + 0j, 0j)
    worst = 0.0
    for _ in range(nsub):
        p = cayley_engine(p, es, ep, V, 0.0, 1.0, h)
        q = _apply(U, q)
        worst = max(worst, abs(abs(p[1]) ** 2 - abs(q[1]) ** 2))
    return om_exact, om_cay, worst


def check_rabi(engine_root=None):
    try:
        anchors = _anchors(engine_root)
    except C.AnchorMissing as ex:
        return C.not_built("electronic.rabi_phase_and_population", str(ex))
    V = 0.01
    rows = []
    for el in ("C", "H"):
        es, ep = LUT[el]
        for dt in DTS_FS:
            oe, oc, pop = _rabi_case(es, ep, V, dt)
            oe0, oc0, pop0 = _rabi_case(0.0, ep - es, V, dt)
            rows.append({"element": el, "dt_fs": dt, "V_Ha": V,
                         "rabi_phase_err_rad_per_ps_engine": abs(oc - oe) * PS_TO_AU,
                         "max_pop_err_1ps_engine": pop,
                         "rabi_phase_err_rad_per_ps_ground_at_zero": abs(oc0 - oe0) * PS_TO_AU,
                         "max_pop_err_1ps_ground_at_zero": pop0,
                         "exact_su2_phase_err": 0.0})
    worst_phase = max(r["rabi_phase_err_rad_per_ps_engine"] for r in rows)
    worst_pop = max(r["max_pop_err_1ps_engine"] for r in rows)
    obs = {"rows": rows, "max_phase_err_rad_per_ps": worst_phase, "max_population_err": worst_pop}
    tol = {"phase_rad_per_ps": 1e-3, "population": 1e-4}
    passed = worst_phase <= tol["phase_rad_per_ps"] and worst_pop <= tol["population"]
    notes = ("Static two-level H=[[eps_s,V],[V,eps_p]] for 1 ps. Cayley preserves eigenvectors, so the "
             "mixing AMPLITUDE of a static H is exact; the error is the Rabi phase (effective splitting), "
             "which then corrupts populations after dephasing. Closed-form SU(2) exponential is exact here.")
    return C.record("electronic.rabi_phase_and_population",
                    {"elements": ["C", "H"], "V_Ha": V, "duration_ps": 1.0},
                    "H1 gate: Rabi phase <= 1e-3 rad/ps and populations <= 1e-4 at 0.5/2/4 fs",
                    obs, passed if C.anchors_ok(anchors) else None, tol, notes,
                    defect_id="D-ELEC-CAYLEY-GAP", anchors=anchors)


def _lz_run(g, v, T, h, offset, method):
    """Diabatic H(t)=[[v t/2+off, g],[g, -v t/2+off]], start |0>, t in [-T,T]. Returns P(stay |0>)."""
    n = int(round(2 * T / h))
    h = 2 * T / n
    p = (1 + 0j, 0j)
    t = -T
    if method == "cayley":
        for _ in range(n):
            tm = t + h / 2
            p = cayley_engine(p, v * tm / 2 + offset, -v * tm / 2 + offset, g, 0.0, 1.0, h)
            t += h
    elif method == "magnus2":
        for _ in range(n):
            tm = t + h / 2
            p = _apply(expm_2x2(v * tm / 2 + offset, -v * tm / 2 + offset, complex(g), h), p)
            t += h
    elif method == "cf4":  # 4th-order commutator-free Magnus (two exponentials)
        c1, c2 = 0.5 - math.sqrt(3) / 6, 0.5 + math.sqrt(3) / 6
        a1, a2 = 0.25 + math.sqrt(3) / 6, 0.25 - math.sqrt(3) / 6
        for _ in range(n):
            t1, t2 = t + c1 * h, t + c2 * h
            H1 = (v * t1 / 2, -v * t1 / 2, g)
            H2 = (v * t2 / 2, -v * t2 / 2, g)
            A = tuple(a1 * x + a2 * y for x, y in zip(H1, H2))
            B = tuple(a2 * x + a1 * y for x, y in zip(H1, H2))
            p = _apply(expm_2x2(A[0] + offset / 2, A[1] + offset / 2, complex(A[2]), h), p)
            p = _apply(expm_2x2(B[0] + offset / 2, B[1] + offset / 2, complex(B[2]), h), p)
            t += h
    return abs(p[0]) ** 2


def check_landau_zener(engine_root=None):
    try:
        anchors = _anchors(engine_root)
    except C.AnchorMissing as ex:
        return C.not_built("electronic.landau_zener", str(ex))
    g, v = 0.002, 2.5e-5          # Ha, Ha/au  -> 2*pi*g^2/v = 1.0053
    T = 0.2 / (v / 2)             # sweep until |v t/2| = 0.2 Ha (16000 au, 387 fs)
    P_analytic = math.exp(-2 * math.pi * g * g / v)
    ref = _lz_run(g, v, T, 1.0, 0.0, "cf4")
    ref_half = _lz_run(g, v, T, 2.0, 0.0, "cf4")
    rows = []
    for dt in DTS_FS:
        h = dt * FS_TO_AU / SUBSTEPS
        pc = _lz_run(g, v, T, h, 0.0, "cayley")
        pc_off = _lz_run(g, v, T, h, -0.5, "cayley")
        pm = _lz_run(g, v, T, h, 0.0, "magnus2")
        rows.append({"dt_fs": dt, "h_au": h,
                     "P_cayley_traceless": pc, "err_cayley_traceless": abs(pc - ref),
                     "P_cayley_engine_offset_-0.5Ha": pc_off, "err_cayley_engine_offset": abs(pc_off - ref),
                     "P_su2_midpoint_fix_preview": pm, "err_su2_midpoint": abs(pm - ref)})
    worst = max(max(r["err_cayley_traceless"], r["err_cayley_engine_offset"]) for r in rows)
    obs = {"P_LZ_analytic_infinite_time": P_analytic, "P_reference_cf4_h1au": ref,
           "reference_convergence_h1_vs_h2": abs(ref - ref_half),
           "finite_window_minus_analytic": ref - P_analytic, "rows": rows, "max_cayley_err": worst}
    tol = 1e-4
    notes = ("Reference: 4th-order commutator-free Magnus with exact SU(2) exponentials, h=1 au "
             "(converged vs h=2 au as reported); analytic P=exp(-2 pi g^2/v) differs from the finite "
             "+-0.2 Ha window by finite-time oscillations only. The -0.5 Ha offset case reproduces the "
             "engine convention (absolute onsite energies); exact dynamics are offset-invariant, Cayley is "
             "not. su2_midpoint is the H1 fix preview (closed-form SU(2) exponential at the same substep).")
    return C.record("electronic.landau_zener",
                    {"g_Ha": g, "v_Ha_per_au": v, "T_au": T, "dts_fs": DTS_FS, "substeps": SUBSTEPS},
                    "H1 gate: Landau-Zener transition probability within 1e-4 of the converged reference at 0.5/2/4 fs",
                    obs, (worst <= tol) if C.anchors_ok(anchors) else None, tol, notes,
                    defect_id="D-ELEC-CAYLEY-GAP", anchors=anchors)


def _driven_mean_rho11(e0, e1, V0, wd, step, tend, method):
    n = int(round(tend / step)); step = tend / n
    p, t, acc = (1 + 0j, 0j), 0.0, 0.0
    for _ in range(n):
        tm = t + step / 2
        c = V0 * math.sin(wd * tm)
        if method == "cayley":
            p = cayley_engine(p, e0, e1, c, 0.0, 1.0, step)
        else:
            p = _apply(expm_2x2(e0, e1, complex(c), step), p)
        t += step
        acc += abs(p[1]) ** 2
    return acc / n


def check_driven_mixing(engine_root=None):
    """Off-resonant mixing under a coupling modulated in the nuclear band.

    rho11 (time-averaged excited population) is what the engine turns into
    charge (q = q_amber + 0.15 rho11), so it is the consequential observable.
    """
    try:
        anchors = _anchors(engine_root)
    except C.AnchorMissing as ex:
        return C.not_built("electronic.driven_offresonant_mixing", str(ex))
    V0, dt, tend = 0.01, 4.0, 0.25 * PS_TO_AU
    h = dt * FS_TO_AU / SUBSTEPS
    w_max = 354.1  # rad/ps, measured fastest nuclear mode
    rows, worst = [], 0.0
    for el in ("C", "H"):
        es, ep = LUT[el]
        for frac in (0.25, 0.5, 1.0):
            wd = frac * w_max / PS_TO_AU
            ex = _driven_mean_rho11(es, ep, V0, wd, 0.5, tend, "exact")
            ce = _driven_mean_rho11(es, ep, V0, wd, h, tend, "cayley")
            cz = _driven_mean_rho11(0.0, ep - es, V0, wd, h, tend, "cayley")
            rows.append({"element": el, "drive_rad_per_ps": frac * w_max, "mean_rho11_exact": ex,
                         "mean_rho11_cayley_engine_abs": ce, "ratio_engine_abs": ce / ex,
                         "mean_rho11_cayley_ground_at_zero": cz, "ratio_ground_at_zero": cz / ex})
            worst = max(worst, abs(ce / ex - 1))
    # spurious resonance: drive at the Cayley effective gap (ground-at-zero, 0.38 Ha, 4 fs)
    geff, _ = effective_gap(0.0, 0.38, dt)
    gabs, _ = effective_gap(*LUT["H"], dt)
    res_ex = _driven_mean_rho11(0.0, 0.38, V0, geff, 0.5, tend, "exact")
    res_c = _driven_mean_rho11(0.0, 0.38, V0, geff, h, tend, "cayley")
    obs = {"rows": rows, "max_rel_dev_engine_abs_in_nuclear_band": worst,
           "H_engine_abs_eff_gap": {"Ha": gabs, "rad_per_ps": gabs * PS_TO_AU, "cm-1": gabs * 219474.6313632},
           "spurious_resonance_at_eff_gap": {"drive_rad_per_ps": geff * PS_TO_AU, "mean_rho11_exact": res_ex,
                                             "mean_rho11_cayley": res_c, "ratio": res_c / res_ex},
           "plan_29x_claim": {"(0.38/gap_eff)^2": (0.38 / geff) ** 2,
                              "verdict": "not a uniform factor: inflation depends on drive frequency vs gap_eff; "
                                         "see rows and spurious_resonance"}}
    tol = 0.05
    notes = ("Coupling V0*sin(w t) with w in the nuclear band (<= 354.1 rad/ps), 0.25 ps, start in |0>. "
             "Exact: closed-form SU(2) at 0.5 au. In the engine convention the H-atom Cayley gap at 4 fs is "
             f"{gabs * 219474.6313632:.0f} cm-1, inside the vibrational band, so nuclear motion can drive spurious "
             "near-resonant electronic population. The plan's '29x inflated mixing' is not a fixed factor; it is "
             "frequency dependent: ~1.00-1.03 in ground-at-zero, 0.14-26x in the engine convention across the nuclear band, "
             "and ~10^3x when the drive hits gap_eff (spurious resonance).")
    return C.record("electronic.driven_offresonant_mixing",
                    {"V0_Ha": V0, "dt_fs": dt, "window_ps": 0.25, "drive_fracs_of_w_max": [0.25, 0.5, 1.0]},
                    "mean rho11 under Cayley within 5% of exact for drives in the nuclear band",
                    obs, (worst <= tol) if C.anchors_ok(anchors) else None, tol, notes,
                    defect_id="D-ELEC-CAYLEY-GAP", anchors=anchors)


CHECKS = [check_transcription_selftest, check_effective_gap, check_rabi, check_landau_zener, check_driven_mixing]
