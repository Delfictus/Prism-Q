"""Unit and constant references (CODATA 2022) against engine constants.

Exact SI: e, h, c, N_A, k_B. Measured CODATA 2022: alpha, E_h, a_0. The same
CODATA 2022 profile is served by the prism-lab operation
physical-types-semantics-units:constants (profile CODATA-2022-prism-v1,
constants_sha256 f1524195...ddd9) for N_A/e/h/c/k_B/a_0; alpha and E_h are
taken from the CODATA 2022 listing (physics.nist.gov/cuu/Constants).
"""
from __future__ import annotations

import math

try:
    from . import _common as C
except ImportError:
    import importlib.util, pathlib, sys
    _p = pathlib.Path(__file__).with_name("_common.py")
    _s = importlib.util.spec_from_file_location("prism_refs_common", _p)
    C = importlib.util.module_from_spec(_s); sys.modules.setdefault("prism_refs_common", C); _s.loader.exec_module(C)

# CODATA 2022
E_CHARGE = 1.602176634e-19
H_PLANCK = 6.62607015e-34
HBAR = H_PLANCK / (2 * math.pi)
C_LIGHT = 299792458.0
N_A = 6.02214076e23
K_B = 1.380649e-23
ALPHA = 7.2973525643e-3
E_H = 4.3597447222060e-18      # J
A_0 = 5.29177210544e-11        # m
CAL = 4.184                    # J (thermochemical)

KCAL_MOL = CAL * 1000 / N_A    # J per (kcal/mol)
HA_TO_KCAL = E_H / KCAL_MOL
EV_TO_KCAL = E_CHARGE / KCAL_MOL
HC_EV_NM = H_PLANCK * C_LIGHT / E_CHARGE * 1e9
AU_TIME_S = HBAR / E_H
PS_TO_AU = 1e-12 / AU_TIME_S
COULOMB_K = ALPHA * HBAR * C_LIGHT * N_A / 1e-10 / (CAL * 1000)   # e^2/(4 pi eps0) in kcal A / mol
EPS_TO_SIGMA_A2 = 1000 * math.log(10) / N_A * 1e16                # (M^-1 cm^-1) -> A^2
KCALMOL_TO_AMU_A2_PS2 = CAL * 1000 / 1e-3 / (1e-10 / 1e-12) ** 2   # 418.4
WAVELENGTHS = (280, 274, 258, 254, 211)

EX = C.KERNELS / "nhs_excited_state.cuh"
AF = C.KERNELS / "nhs_amber_fused.cu"
UB = C.KERNELS / "nhs_biprism_unit_bridge.cuh"

# transcribed band model, nhs_excited_state.cuh:67-87 (lambda_max nm, eps_max M^-1cm^-1, 1/e half-width nm)
BANDS = {"TRP": (280.0, 5600.0, 15.0), "TYR": (274.0, 1490.0, 12.0), "PHE": (258.0, 200.0, 10.0), "SS": (250.0, 300.0, 20.0)}
NEFF = {"TRP": 9.0, "TYR": 10.0, "PHE": 9.0, "SS": 2.0}
ENG = {"COULOMB_CONSTANT": 332.0636, "HC_EV_NM": 1239.84, "EV_TO_KCAL_MOL": 23.06, "EPSILON_TO_SIGMA_FACTOR": 3.823e-5,
       "ENERGY_VEL_PREFACTOR": 28.93, "CALIBRATED_PHOTON_FLUENCE": 0.024, "UV_PHOTON_ENERGY_EV": 4.428,
       "BIPRISM_ANGSTROM_TO_BOHR": 1.8897261259077824, "BIPRISM_NATIVE_VELOCITY_TO_AU": 4.571028907499172e-5,
       "BIPRISM_NATIVE_FORCE_TO_AU": 0.0008432975634897884, "BIPRISM_PS_TO_ATOMIC_TIME": 41341.37333517015}


def f32(x):
    import struct
    return struct.unpack("f", struct.pack("f", x))[0]


def _anchors(engine_root):
    return [
        C.anchor(engine_root, AF, 66, 66, ["#define COULOMB_CONSTANT 332.0636f"]),
        C.anchor(engine_root, EX, 37, 40, ["#define EPSILON_TO_SIGMA_FACTOR  3.823e-5f", "#define CALIBRATED_PHOTON_FLUENCE  0.024f"]),
        C.anchor(engine_root, EX, 67, 87, ["#define BAND_W_TO_SIGMA   0.70710678f", "#define TRP_LAMBDA_MAX    280.0f",
                                           "#define TYR_EPSILON_MAX   1490.0f", "#define SS_BANDWIDTH      20.0f"]),
        C.anchor(engine_root, EX, 90, 98, ["#define UV_PHOTON_ENERGY_EV    4.428f", "#define EV_TO_KCAL_MOL         23.06f",
                                           "#define NEFF_TRP  9.0f"]),
        C.anchor(engine_root, EX, 108, 108, ["#define ENERGY_VEL_PREFACTOR   28.93f"]),
        C.anchor(engine_root, EX, 209, 231, ["return 1239.84f / wavelength_nm;",
                                             "float p_absorb = sigma * CALIBRATED_PHOTON_FLUENCE;",
                                             "float e_photon_kcal = e_photon_ev * EV_TO_KCAL_MOL;",
                                             "return e_photon_kcal * p_absorb * eta;"]),
        C.anchor(engine_root, UB, 9, 12, ["BIPRISM_ANGSTROM_TO_BOHR = 1.8897261259077824;",
                                          "BIPRISM_PS_TO_ATOMIC_TIME = 41341.37333517015;"]),
    ]


def sigma_band(chrom, lam):
    lmax, emax, w = BANDS[chrom]
    s = w * 0.70710678
    return emax * math.exp(-((lam - lmax) ** 2) / (2 * s * s)) * ENG["EPSILON_TO_SIGMA_FACTOR"]


def e_dep_engine(chrom, lam):
    """Transcription of compute_deposited_energy_wavelength (kcal/mol)."""
    return (ENG["HC_EV_NM"] / lam) * ENG["EV_TO_KCAL_MOL"] * sigma_band(chrom, lam) * ENG["CALIBRATED_PHOTON_FLUENCE"] * 1.0


def check_constants(engine_root=None):
    try:
        anchors = _anchors(engine_root)
    except C.AnchorMissing as ex:
        return C.not_built("units.constants_codata2022", str(ex))
    ref = {"COULOMB_CONSTANT": COULOMB_K, "HC_EV_NM": HC_EV_NM, "EV_TO_KCAL_MOL": EV_TO_KCAL,
           "EPSILON_TO_SIGMA_FACTOR": EPS_TO_SIGMA_A2, "ENERGY_VEL_PREFACTOR": math.sqrt(2 * KCALMOL_TO_AMU_A2_PS2),
           "UV_PHOTON_ENERGY_EV": HC_EV_NM / 280.0,
           "BIPRISM_ANGSTROM_TO_BOHR": 1e-10 / A_0,
           "BIPRISM_NATIVE_VELOCITY_TO_AU": 100.0 / (A_0 / AU_TIME_S),
           "BIPRISM_NATIVE_FORCE_TO_AU": (1 / HA_TO_KCAL) * (A_0 / 1e-10),
           "BIPRISM_PS_TO_ATOMIC_TIME": PS_TO_AU}
    tol = {"COULOMB_CONSTANT": 1e-6, "HC_EV_NM": 2e-6, "EV_TO_KCAL_MOL": 5e-5, "EPSILON_TO_SIGMA_FACTOR": 5e-4,
           "ENERGY_VEL_PREFACTOR": 5e-4, "UV_PHOTON_ENERGY_EV": 5e-4, "BIPRISM_ANGSTROM_TO_BOHR": 1e-12,
           "BIPRISM_NATIVE_VELOCITY_TO_AU": 1e-10, "BIPRISM_NATIVE_FORCE_TO_AU": 1e-10, "BIPRISM_PS_TO_ATOMIC_TIME": 1e-12}
    rows, ok = {}, True
    for k, r in ref.items():
        e = ENG[k]
        rel = abs(e - r) / abs(r)
        row = {"engine": e, "codata2022": r, "rel": rel, "tol": tol[k], "ok": rel <= tol[k]}
        if k == "COULOMB_CONSTANT":
            row["engine_float32"] = f32(e)
            row["rel_float32"] = abs(f32(e) - r) / r
            row["amber_332.0522173_rel"] = abs(332.0522173 - r) / r
            row["charmm_332.0716_rel"] = abs(332.0716 - r) / r
        rows[k] = row
        ok = ok and row["ok"]
    obs = {"constants": rows, "Ha_to_kcal_mol": HA_TO_KCAL, "eV_to_kcal_mol": EV_TO_KCAL,
           "kcal_mol_to_amu_A2_ps2": KCALMOL_TO_AMU_A2_PS2, "atomic_time_fs": AU_TIME_S * 1e15}
    notes = ("332.0636 is alpha*hbar*c*N_A/(1 A * 4184 J/kcal) to 2e-7 (CODATA 2022 gives "
             f"{COULOMB_K:.5f}); it is not the AMBER 332.0522 (18.2223^2) value, so energies differ from "
             "AMBER/pmemd by 3.5e-5 relative. EV_TO_KCAL_MOL=23.06 is truncated (2.4e-5 rel). Bridge constants "
             "match CODATA 2022.")
    return C.record("units.constants_codata2022", {"profile": "CODATA 2022"},
                    "engine constants equal CODATA 2022 derivations within per-constant tolerance",
                    obs, ok if C.anchors_ok(anchors) else None, tol, notes, anchors=anchors)


def check_photon_and_edep(engine_root=None):
    try:
        anchors = _anchors(engine_root)
    except C.AnchorMissing as ex:
        return C.not_built("units.photon_energy_and_edep", str(ex))
    photons = {lam: {"eV": HC_EV_NM / lam, "kcal_mol": HC_EV_NM / lam * EV_TO_KCAL, "Ha": HC_EV_NM / lam * E_CHARGE / E_H,
                     "engine_kcal_mol": ENG["HC_EV_NM"] / lam * ENG["EV_TO_KCAL_MOL"]} for lam in WAVELENGTHS}
    edep = {}
    for ch in BANDS:
        edep[ch] = {}
        for lam in WAVELENGTHS:
            s = sigma_band(ch, lam)
            p = s * ENG["CALIBRATED_PHOTON_FLUENCE"]
            e = e_dep_engine(ch, lam)
            edep[ch][lam] = {"sigma_A2": s, "p_absorb": p, "E_dep_kcal_mol": e, "E_dep_Ha": e / HA_TO_KCAL,
                             "dT_K_over_Neff": e / (1.5 * NEFF[ch] * K_B * N_A / (CAL * 1000))}
    trp = edep["TRP"][280]["dT_K_over_Neff"]
    dims = {
        "sigma": "eps[M^-1 cm^-1] * (1000 cm^3/L) * ln10 / N_A -> cm^2; *1e16 -> A^2  (factor 3.8235e-5)",
        "p_absorb": "sigma[A^2] * F[photons/A^2] -> dimensionless; requires F in photons/A^2 (declared in comment :33,:39)",
        "E_dep": "E_photon[kcal/mol photons] * p * eta -> kcal/mol per chromophore (expected molar energy per pulse)",
        "consistent": True,
    }
    obs = {"photon_energies": photons, "E_dep": edep, "dimensional_analysis": dims,
           "TRP_280_dT_K_with_Neff9": trp, "engine_target_comment": "TRP @ 280nm -> dT ~ 20K (:34)",
           "211nm_sigma_all_chromophores_A2": {ch: edep[ch][211]["sigma_A2"] for ch in BANDS}}
    ok = dims["consistent"] and abs(trp - 20) < 1.0
    notes = ("E_dep = E_photon*sigma*F*eta is dimensionally consistent iff F is photons/A^2 (as the comment states); "
             "0.024 has no recorded provenance (it is calibrated to the dT~20 K target, reproduced here). Single "
             "Gaussian bands: at 211 nm every chromophore sigma is ~0 (TRP 1.4e-10 A^2, TYR 6e-14 A^2) although Trp has a ~33,000 "
             "M^-1cm^-1 B-band near 218 nm and the peptide bond absorbs strongly below 230 nm, so the 211/254/258 nm "
             "channels deposit essentially nothing for TRP/TYR: a calibration defect for multi-wavelength action "
             "spectra, not a unit defect.")
    return C.record("units.photon_energy_and_edep",
                    {"wavelengths_nm": WAVELENGTHS, "bands": BANDS, "F_photons_per_A2": ENG["CALIBRATED_PHOTON_FLUENCE"]},
                    "E_dep units kcal/mol; TRP@280 dT ~20 K with N_eff=9 as the source comment states",
                    obs, ok if C.anchors_ok(anchors) else None, {"dT_K": 1.0}, notes,
                    anchors=anchors)


CHECKS = [check_constants, check_photon_and_edep]
