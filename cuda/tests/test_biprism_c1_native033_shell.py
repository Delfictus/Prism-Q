#!/usr/bin/env python3
"""Frozen actual-native033 EQR2 regression, evaluated on CPU only.

The retained leaf4/vertex17 refusal has no joint norm/equation witness among
the original four s0 corners with s1 fixed, or the full sixteen corners. Its
one-outward-neighbor shell has seven witnesses. This test independently checks
that fact with exact rational arithmetic and the original FP64/FMA expression
order. It does not rebuild charge, DEC, H or work, admit a nonlinear candidate,
execute CUDA, or qualify a repaired source/trajectory.

Run directly with --artifact-root pointing to the retained flat artifact set.
The original path strings inside the pinned receipts remain unchanged even if
that set is copied elsewhere. Missing or changed fixture files are errors, not
skips. All outputs are plaintext; no fixture/source/output files are modified.
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
from fractions import Fraction as Q
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import unittest

def _env_root(name):
    """Absolute root from the environment; there is no compiled-in host default.
    When unset, a non-existent sentinel naming the variable is returned so every
    read fails closed; entry points call _require_env() first."""
    value = os.environ.get(name, '')
    return Path(value) if os.path.isabs(value) else Path('/nonexistent') / (name + '_UNSET')


def _require_env(*names):
    for name in names:
        if not os.path.isabs(os.environ.get(name, '')):
            raise SystemExit(f'{name} must name an absolute path (no compiled-in host default)')



ROOT = _env_root('PRISM_BIPRISM_RESOLUTION_ROOT')
ORIGINAL_ROOT = ROOT
# The engine source-tree path recorded in the historical 033 receipts (receipt keys).
ENGINE = str(_env_root('PRISM_BIPRISM_RECORDED_ENGINE_ROOT'))
MAIN = ENGINE + '/crates/prism-gpu/src/kernels/nhs_biprism_variational_so3g_v10.cuh'
ELECTRONIC = ENGINE + '/crates/prism-gpu/src/kernels/nhs_biprism_electronic_discrete.cuh'
PREFIX = 'failure_1_2_3_coupled_native_033_asymmetric_kappa_zero_state'
FILES = {
    'dispatch': ('failure_1_2_3_coupled_native_033_asymmetric_kappa_zero_dispatch.json',
                 '93c7c8a10ccdff86820495e80188857b319e8eaab1fbe1d06c3ff1ca013e2126'),
    'build': ('failure_1_2_3_coupled_native_build_033.json',
              '572bdf60c79d8fbc09f5606a98443ae8096723bb9438a6d08e3913378969a4d2'),
    'main': ('failure_1_2_3_native_033_main_source.cuh',
             '957921af47d0fcd40e0503948a4571ba989ffbd55d228744fc48d716c3393c3a'),
    'numerical': (PREFIX + '.epoch1.numerical.bin',
                  '4b05d1d51b9b74c7319d8af18a111ab8244b1a76a9fb4fd15e4eeece0784a3c8'),
    'exchange': (PREFIX + '.epoch1.exchange.bin',
                 '968ae99e8d0071c249fa6ad20cbbc7a110239afbdf865ed7373151bed2c6b1d8'),
    'tiles': (PREFIX + '.epoch1.tiles.bin',
              '4096351bfd7448b5a426a9aa8c208ba4f4ce4db6914117417455f0d0a9260b0f'),
    'convergence': (PREFIX + '.epoch1.conv.bin',
                    '2ce67285b8bf621d4be23509d8b6c6bf530b35935974b47843c1d25b33a40439'),
}
BINARY_SHA = '2ebc8f6a1524c6de40b4d8447121c23efb5eef3cf490b791fc4f260b089ad9f0'
ELECTRONIC_SHA = 'a416c303a31602bc1ae077cc17874a6b59d5048a792726a334b30265ef19eedc'
STRIDE, TILE, BANK = 40704, 225408, 11648
LEAF, VERTEX, EQR_OFFSET = 4, 17, 13480
NORM_TOL = 1e-7  # Original binary64 norm tolerance.
LIBM = ctypes.CDLL('libm.so.6')
LIBM.fma.argtypes = [ctypes.c_double] * 3
LIBM.fma.restype = ctypes.c_double
FMA = LIBM.fma


def require(ok, message):
    if not ok:
        raise ValueError(message)


def bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def from_bits(value):
    return struct.unpack('<f', struct.pack('<I', value))[0]


def f32(x):
    return from_bits(bits(x))


EQUATION_TOL = f32(1e-7)  # Original promoted FP32 equation tolerance.


def bitwise64(a, b):
    return struct.pack('<d', a) == struct.pack('<d', b)


def validate_fixture(blobs):
    """Bind original source/build/dispatch/raw bytes before mathematical use."""
    for key, (_, expected) in FILES.items():
        require(hashlib.sha256(blobs[key]).hexdigest() == expected,
                'frozen fixture digest mismatch: ' + key)
    dispatch, build = (json.loads(blobs[k]) for k in ('dispatch', 'build'))
    require(dispatch['native_execution'] is True and dispatch['native_exit_code'] == 4,
            'fixture must remain the actual controlled native refusal')
    require(dispatch['sources_binary_inputs_unchanged'] is True
            and dispatch['gpu_recovery_none_after'] is True
            and dispatch['new_kernel_fault_records'] == [], 'native provenance/health changed')
    require(build['exit_code'] == 0 and build['sources_unchanged'] is True
            and build['header_overlays'] is False, 'actual-source build required')
    require(dispatch['build_receipt'] == {
        'path': str(ORIGINAL_ROOT / FILES['build'][0]), 'sha256': FILES['build'][1]},
        'dispatch/build identity mismatch')
    require(build['input_hashes'][MAIN] == dispatch['input_hashes'][MAIN] == FILES['main'][1],
            'executed main/source snapshot mismatch')
    require(build['input_hashes'][ELECTRONIC] == dispatch['input_hashes'][ELECTRONIC] == ELECTRONIC_SHA,
            'original electronic provider mismatch')
    require(build['binary_sha256'] == dispatch['input_hashes'][dispatch['command'][0]] == BINARY_SHA,
            'recorded actual executable mismatch')
    require(dispatch['command'][4] == str(ORIGINAL_ROOT / PREFIX)
            and dispatch['command'][5:] == ['--asymmetric-kappa=0'], 'wrong native case')
    require(all(dispatch['main_compiler_resources'].get(k) == 0
                for k in ('stack', 'spill_stores', 'spill_loads', 'cumulative_stack')),
            'historical native resource gate changed')
    for key, extent in (('numerical', 8*STRIDE), ('exchange', 8*BANK),
                        ('tiles', 8*TILE), ('convergence', 8*128)):
        require(len(blobs[key]) == extent, 'incomplete native population: ' + key)
        require(dispatch['artifacts'][str(ORIGINAL_ROOT / FILES[key][0])] == {
            'sha256': FILES[key][1], 'bytes': extent}, 'raw dispatch binding mismatch: ' + key)


def native_norm(p):
    return FMA(p[0], p[0], p[1]*p[1]) + FMA(p[2], p[2], p[3]*p[3])


@dataclass(frozen=True)
class Record:
    start: tuple
    candidate: tuple
    trial: tuple
    H: tuple
    omega: float
    h: float
    norm_start: float
    norm_outer: float
    native_equation: float


def decode_bound_record(blobs):
    validate_fixture(blobs)
    raw = blobs['numerical']
    # Keep the all-eight refusal population/commit identity, even though the
    # exhaustive local candidate enumeration concerns the offending leaf4.
    for leaf in range(8):
        base = leaf*STRIDE
        ident = struct.unpack_from('<12I', raw, base+EQR_OFFSET)
        require(ident[:3] == (0x45515232, 2, leaf) and 0 <= ident[3] < 192,
                'EQR2 leaf/vertex identity')
        require(ident[4:9] == (3, 0, 405, 0, 0) and ident[10:] == (0, 0),
                'EQR2 refusal chronology')
        require(struct.unpack_from('<4I', raw, base+13056) == (0x52524632, 7, 1, 0),
                'EQR2 must remain the original round0 stage7 refusal')
        conv = struct.unpack_from('<32I', blobs['convergence'], leaf*128)
        require(conv[:2] == (0x42505631, leaf) and conv[4] & 2 and conv[5] == 0,
                'refusal must not have committed an outer body')
        footer = leaf*BANK + 11520
        meta = struct.unpack_from('<12I', blobs['exchange'], footer+80)
        require(meta[:5] == (0x46435031, leaf, 0, 3, 405)
                and meta[5] & 2 and meta[7] == 5 and meta[8:] == (0, 0, 0, 0),
                'exchange refusal identity')
    off = LEAF*STRIDE + EQR_OFFSET
    ident = struct.unpack_from('<12I', raw, off)
    require(ident[3] == VERTEX and ident[9] == 17, 'wrong selected physical vertex')
    start, candidate = (struct.unpack_from('<4f', raw, off+x) for x in (48, 64))
    trial = struct.unpack_from('<4d', raw, off+80)
    energy_h = struct.unpack_from('<4d', raw, off+112)
    v = struct.unpack_from('<12d', raw, off+144)
    require(all(bitwise64(v[j], 0.) for j in (3, 5, 6, 8))
            and raw[off+240:off+256] == bytes(16), 'EQR2 reserved slots')
    H = (*energy_h[:3], energy_h[3]+v[0])
    require(all(math.isfinite(x) for x in (*start, *candidate, *trial, *H, *v)),
            'nonfinite frozen primitive')
    require(v[1] > 0 and v[2] > 0, 'invalid Berry mass/timestep')
    tile = blobs['tiles'][LEAF*TILE:(LEAF+1)*TILE]
    require(struct.unpack_from('<I', tile, 4*VERTEX)[0] == ident[9], 'tile physical ID')
    require(bitwise64(v[1], max(struct.unpack_from('<f', tile, 95232+4*VERTEX)[0], f32(1e-4))),
            'actual tile Berry mass')
    require(bitwise64(v[4], native_norm(struct.unpack_from('<4f', tile, 90112+16*VERTEX))),
            'outer norm must be the unchanged physical input')
    require(bitwise64(v[7], v[9])
            and bitwise64(v[7], struct.unpack_from('<d', raw, LEAF*STRIDE+30720+13*8)[0])
            and bitwise64(v[7], struct.unpack_from('<d', blobs['exchange'], LEAF*BANK+11520+16)[0]),
            'selected native equation/leaf maximum/footer mismatch')
    return Record(start, candidate, trial, H, v[1], v[2], native_norm(start), v[4], v[7])


def next32(x, up):
    if x == 0.:
        return from_bits(1 if up else 0x80000001)
    return from_bits(bits(x) + (1 if (x > 0) == up else -1))


def brackets(x):
    n = f32(x)
    if n == x:
        return n, n
    return (n, next32(n, True)) if n < x else (next32(n, False), n)


def domains(r):
    bounds = [brackets(x) for x in r.trial]
    shell = [(next32(lo, False), lo, hi, next32(hi, True)) for lo, hi in bounds]
    return {
        'original4': [(m, tuple(bounds[i][(m >> i) & 1] for i in range(2))+r.candidate[2:])
                      for m in range(4)],
        'full16': [(m, tuple(bounds[i][(m >> i) & 1] for i in range(4))) for m in range(16)],
        'shell256': [(m, tuple(shell[i][(m >> (2*i)) & 3] for i in range(4))) for m in range(256)],
    }


def exact_equation(r, p):
    """Exact scaled Cayley residual on represented inputs, with no FP rounding.

    The same recorded total H and coefficient semantics are used. This is an
    independent rational linear-algebra reference, not a native FMA replay.
    """
    a, d, b, c = map(Q, r.H)
    half_h, omega = Q(r.h)/2, Q(r.omega)
    scale = max(omega, *(abs(half_h*x) for x in (a, d, b, c)))
    x, y, z, w = (Q(t)+Q(q) for t, q in zip(r.start, p))
    dx, dy, dz, dw = (Q(q)-Q(t) for t, q in zip(r.start, p))
    hs = (a*x+b*z-c*w, a*y+b*w+c*z, b*x+c*y+d*z, b*y-c*x+d*w)
    return max(map(abs, ((omega*dx-half_h*hs[1])/scale,
                         (omega*dy+half_h*hs[0])/scale,
                         (omega*dz-half_h*hs[3])/scale,
                         (omega*dw+half_h*hs[2])/scale)))


def native_equation(r, p):
    a, d, b, c = r.H
    scale = max(r.omega, *(abs(.5*r.h*x) for x in r.H))
    x, y, z, w = (t+q for t, q in zip(r.start, p))
    hs = (a*x+FMA(b, z, -c*w), a*y+FMA(b, w, c*z),
          FMA(b, x, c*y)+d*z, FMA(b, y, -c*x)+d*w)
    dx, dy, dz, dw = (q-t for t, q in zip(r.start, p))
    aa, bb = r.omega/scale, .5*r.h/scale
    return max(map(abs, (FMA(aa, dx, -bb*hs[1]), FMA(aa, dy, bb*hs[0]),
                         FMA(aa, dz, -bb*hs[3]), FMA(aa, dw, bb*hs[2]))))


def margins(r, p):
    norm = sum((Q(x)**2 for x in p), Q(0))
    # The targets are the source-evaluated FP64 start/outer norms. Keep all
    # three bounds, including absolute one, without silent normalization.
    norm_margin = min(Q(NORM_TOL)-abs(norm-Q(target))
                      for target in (1., r.norm_start, r.norm_outer))
    return norm_margin, Q(EQUATION_TOL)-exact_equation(r, p)


EXPECTED = {
    164: (0x3e844f9d, 0xbef86c73, 0x3f4b686d, 0xbe840066),
    228: (0x3e844f9d, 0xbef86c73, 0x3f4b686d, 0xbe840065),
    229: (0x3e844f9e, 0xbef86c73, 0x3f4b686d, 0xbe840065),
    232: (0x3e844f9d, 0xbef86c72, 0x3f4b686d, 0xbe840065),
    233: (0x3e844f9e, 0xbef86c72, 0x3f4b686d, 0xbe840065),
    234: (0x3e844f9f, 0xbef86c72, 0x3f4b686d, 0xbe840065),
    235: (0x3e844fa0, 0xbef86c72, 0x3f4b686d, 0xbe840065),
}
NORM_DENOMINATOR = 9444732965739290427392
EQUATION_DENOMINATOR = 8980556929388844068841019968743014400
EXPECTED_MARGIN_NUMERATORS = {
    164: (34824506545641, 611042828480324419770021339),
    228: (179961744299497, 45489215461963326669344634331),
    229: (34484264605161, 45486974292941574754851019739),
    232: (453106405447145, 7207584255199379012891604443),
    233: (307628925752809, 7205343086177627098397989851),
    234: (162151429281257, 7203101917155875183904375259),
    235: (16673916032489, 7200860748134123269410760667),
}


class Native033ShellRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blobs = {key: (ROOT/name).read_bytes() for key, (name, _) in FILES.items()}
        cls.record = decode_bound_record(cls.blobs)
        cls.domains = domains(cls.record)

    def test_original_capture_is_still_a_refusal(self):
        self.assertEqual(tuple(map(bits, self.record.candidate)),
                         (0x3e844f9e, 0xbef86c72, 0x3f4b686d, 0xbe840067))
        self.assertTrue(bitwise64(native_equation(self.record, self.record.candidate),
                                  self.record.native_equation))
        nm, em = margins(self.record, self.record.candidate)
        self.assertGreaterEqual(nm, 0)
        self.assertLess(em, 0)

    def test_exact_original4_full16_shell_counts(self):
        for name, expected in (('original4', (4, 1, 0)), ('full16', (16, 11, 0)),
                               ('shell256', (256, 95, 7))):
            with self.subTest(domain=name):
                population = self.domains[name]
                self.assertEqual(len({tuple(map(bits, p)) for _, p in population}), len(population))
                ms = [margins(self.record, p) for _, p in population]
                actual = len(population), sum(n >= 0 for n, _ in ms), sum(n >= 0 and e >= 0 for n, e in ms)
                self.assertEqual(actual, expected)

    def test_all_seven_witness_bits_and_exact_margins(self):
        found = {}
        for index, p in self.domains['shell256']:
            nm, em = margins(self.record, p)
            if nm >= 0 and em >= 0:
                found[index] = tuple(map(bits, p))
                nn, en = EXPECTED_MARGIN_NUMERATORS[index]
                self.assertEqual(nm, Q(nn, NORM_DENOMINATOR))
                self.assertEqual(em, Q(en, EQUATION_DENOMINATOR))
                self.assertNotEqual(bits(p[3]), bits(self.record.candidate[3]))
        self.assertEqual(found, EXPECTED)

    def test_original_equation_only_witnesses_fail_absolute_one(self):
        found = []
        for index, p in self.domains['full16']:
            nm, em = margins(self.record, p)
            if em >= 0:
                found.append(index)
                self.assertLess(nm, 0)
                self.assertGreater(sum(Q(x)**2 for x in p)-1, Q(NORM_TOL))
        self.assertEqual(found, [12, 13])

    def test_fma_replay_and_rational_reference_agree_on_every_decision(self):
        for name, population in self.domains.items():
            for index, p in population:
                with self.subTest(domain=name, index=index):
                    nm, em = margins(self.record, p)
                    n = native_norm(p)
                    native_norm_ok = all(abs(n-target) <= NORM_TOL
                                         for target in (1., self.record.norm_start, self.record.norm_outer))
                    e = native_equation(self.record, p)
                    self.assertEqual(nm >= 0, native_norm_ok)
                    self.assertEqual(em >= 0, e <= EQUATION_TOL)
                    # Arithmetic replay envelope only; never used for either gate.
                    self.assertLessEqual(abs(e-float(exact_equation(self.record, p))),
                                         128*sys.float_info.epsilon)

    def test_changed_raw_is_rejected_before_enumeration(self):
        changed = dict(self.blobs)
        raw = bytearray(changed['numerical'])
        raw[LEAF*STRIDE+EQR_OFFSET+64] ^= 1
        changed['numerical'] = bytes(raw)
        with self.assertRaisesRegex(ValueError, 'digest mismatch: numerical'):
            decode_bound_record(changed)

    def test_mismatched_source_is_rejected_before_enumeration(self):
        changed = dict(self.blobs)
        changed['main'] += b'\n// a different source epoch\n'
        with self.assertRaisesRegex(ValueError, 'digest mismatch: main'):
            decode_bound_record(changed)


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.artifact_root.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Native033ShellRegression))
    if not result.wasSuccessful():
        return 1
    r = Native033ShellRegression.record
    print('SCOPE=frozen actual033 refusal regression; CPU exact-rational and FP64/FMA replay')
    print('NATIVE_RERUN=false NONLINEAR_ADMISSION=false REPAIRED_SOURCE_QUALIFIED=false')
    print('FIXTURE_DISPATCH_SHA256='+FILES['dispatch'][1]+' EXECUTED_MAIN_SHA256='+FILES['main'][1])
    for name, population in Native033ShellRegression.domains.items():
        ms = [margins(r, p) for _, p in population]
        print(f'{name}: unique={len(population)} norm={sum(n >= 0 for n, _ in ms)} '
              f'joint={sum(n >= 0 and e >= 0 for n, e in ms)}')
    for index, p in Native033ShellRegression.domains['shell256']:
        nm, em = margins(r, p)
        if nm >= 0 and em >= 0:
            print(f'witness={index} bits='+','.join(f'{bits(x):08x}' for x in p)
                  +f' exact_limiting_norm_margin={nm} exact_equation_margin={em}'
                  +f' cpu_fma_equation={native_equation(r, p):.17g}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
