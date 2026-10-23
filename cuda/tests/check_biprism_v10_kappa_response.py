#!/usr/bin/env python3
"""Compare guarded fresh asymmetric kappa controls from actual raw readbacks.

No GPU work, solve, state repair or normalization is performed. A nonzero
gauge-invariant density response is required, together with independently
reconstructed nonzero ring work/transfer. A fresh repeated kappa=0 run is
required for qualification against the observed repeat envelope; this single
repeat is not a statistical or universal nondeterminism bound.
"""
from pathlib import Path
from fractions import Fraction as Q
import argparse
import hashlib
import json
import math
import biprism_exchange_layout as EX
import struct
import check_biprism_spinor_owner as SPINOR

LEAVES, CORE, VERTICES = 8, 192, 256
TILE_BYTES, BANK_BYTES, CONV_BYTES = 225408, 11648, 128
PSI, OMEGA, FOOTER = 90112, 95232, 11520
TOL, NORM_TOL = Q(1, 10**6), Q(1, 10**7)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def require_resources(resources):
    require(all(resources.get(k) == 0 for k in
                ('stack', 'cumulative_stack', 'spill_stores', 'spill_loads'))
            and 0 < resources.get('registers', 0) <= 128, 'main/callee resource gate')


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def bits64(x):
    return struct.pack('<d', x)


def density(psi):
    """Exact gauge-invariant density of four represented binary32 components."""
    require(len(psi) == 4 and all(math.isfinite(x) for x in psi), 'nonfinite spinor')
    a, b, c, d = map(Q.from_float, psi)
    return a*a+b*b, c*c+d*d, a*c+b*d, b*c-a*d


def represented_population(psi):
    d = density(psi)
    # The two squared norms use RN64 FMA in production. Each FP32 square is
    # exact in binary64; Fraction->float supplies the one final RN rounding.
    return float(d[0])-float(d[1])


class Files:
    def __init__(self):
        self.identities = {}

    def read(self, path, expected=None):
        path = Path(path).resolve()
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if expected is not None:
            require(digest == expected, 'hash mismatch: '+str(path))
        old = self.identities.get(str(path))
        identity = {'sha256': digest, 'bytes': len(data)}
        require(old is None or old == identity, 'input changed while checking: '+str(path))
        self.identities[str(path)] = identity
        return data

    def document(self, path, expected=None):
        def bad_constant(value):
            raise ValueError('nonfinite JSON token: '+value)
        return json.loads(self.read(path, expected), parse_constant=bad_constant)

    def binding(self, binding):
        require(isinstance(binding, dict) and set(('path', 'sha256')) <= binding.keys(), 'missing file binding')
        return self.document(binding['path'], binding['sha256'])

    def unchanged(self):
        for path, value in list(self.identities.items()):
            self.read(path, value['sha256'])


def reconstruct_ring(start, end, weights, kappa):
    """Direct endpoint products, independent of the device midpoint chain."""
    require(len(start) == len(end) == len(weights) == LEAVES, 'ring extent')
    result = []
    for leaf in range(LEAVES):
        prev, nxt = (leaf-1) % LEAVES, (leaf+1) % LEAVES
        require(all(len(a[leaf]) == CORE for a in (start, end, weights)), 'ring core extent')
        delta = attributed = work = transfer = Q(0)
        max_numerator = Q(0)
        for u in range(CORE):
            p0, p1 = start[prev][u], end[prev][u]
            i0, i1 = start[leaf][u], end[leaf][u]
            n0, n1 = start[nxt][u], end[nxt][u]
            jp = kappa*(weights[prev][u]+weights[leaf][u])/2
            jn = kappa*(weights[leaf][u]+weights[nxt][u])/2
            p, i, n = (p0+p1)/2, (i0+i1)/2, (n0+n1)/2
            dp, di, dn = p1-p0, i1-i0, n1-n0
            delta -= jn*(i1*n1-i0*n0)
            attributed -= (jp*(p1*i1-p0*i0)+jn*(i1*n1-i0*n0))/2
            numerator = -(jp*p+jn*n)
            work += numerator*di
            transfer += (jp*(p*di-i*dp)+jn*(n*di-i*dn))/2
            max_numerator = max(max_numerator, abs(numerator))
        require(attributed-transfer-work == 0, 'exact local ring chain failed')
        result.append((delta, attributed, work, transfer, max_numerator))
    require(sum((r[3] for r in result), Q(0)) == 0, 'exact ring transfer closure failed')
    require(sum((r[0]-r[2] for r in result), Q(0)) == 0, 'exact global ring chain failed')
    return result


def response_metrics(zero, coupled, repeat=None):
    require(len(zero) == len(coupled) == LEAVES*CORE, 'endpoint density population')
    if repeat is not None:
        require(len(repeat) == len(zero), 'repeat density population')
    changed = outside = population_changed = 0
    largest = repeat_largest = Q(0)
    first = None
    for i, (a, b) in enumerate(zip(zero, coupled)):
        require(len(a) == len(b) == 4, 'density component extent')
        population_changed += (a[0]-a[1]) != (b[0]-b[1])
        for c in range(4):
            diff = abs(b[c]-a[c])
            changed += diff != 0
            largest = max(largest, diff)
            if repeat is not None:
                baseline = abs(repeat[i][c]-a[c])
                repeat_largest = max(repeat_largest, baseline)
                # Both baselines must be farther from the coupled endpoint
                # than their own exact observed separation. No epsilon floor.
                qualifies = diff > baseline and abs(b[c]-repeat[i][c]) > baseline
                outside += qualifies
                if qualifies and first is None:
                    first = {'leaf': i//CORE, 'vertex': i % CORE, 'component': c,
                             'difference': float(diff), 'observed_repeat_difference': float(baseline)}
    return {'changed_density_components': changed, 'changed_population_vertices': population_changed,
            'components_beyond_both_observed_repeat_envelopes': outside,
            'maximum_exact_density_difference': float(largest),
            'maximum_observed_repeat_density_difference': float(repeat_largest),
            'first_qualified_component': first}


def check_precision_planes(planes):
    require(len(planes) == 13, 'paired coordinate/velocity plane population')
    extent = len(planes[0])
    require(extent >= 8*256*12 and extent % (8*12) == 0 and
            all(len(p) == extent for p in planes),
            'paired full coordinate/velocity plane extent')
    for plane in planes:
        require(all(math.isfinite(v[0]) for v in struct.iter_unpack('<f', plane)),
                'nonfinite paired coordinate authority')


def check_dec_source_bindings(files, dec, check, expected):
    """Bind device dependencies and the independent reader by identity, not order."""
    command = check.get('command', [])
    require(len(command) >= 2 and isinstance(check.get('checker_sha256'), str),
            'missing independent DEC checker identity')
    checker = str(Path(command[1]).resolve())
    reader_count = 0
    seen = set()
    for source in dec['source_bindings']:
        path = str(Path(source['path']).resolve())
        require(path not in seen, 'duplicate DEC source binding')
        seen.add(path)
        if path == checker:
            reader_count += 1
            require(source['sha256'] == check['checker_sha256'], 'DEC checker binding differs')
        else:
            require(expected.get(path) == source['sha256'], 'DEC helper not in executed build')
        files.read(path, source['sha256'])
    require(reader_count == 1, 'independent DEC reader binding is absent')


def load_run(files, path, mode, is_repeat=False):
    receipt = files.document(path)
    for name in ('sources_binary_inputs_unchanged', 'gpu_recovery_none_after',
                 'independent_dec_all_epochs_passed', 'coupled_diagnostic_accepted'):
        require(receipt.get(name) is True, 'run did not pass '+name)
    require(receipt.get('exit_code') == receipt.get('native_exit_code') == 0, 'run exit code')
    require(receipt.get('expected_epochs') == 1 and receipt.get('diagnostic_mode') == mode, 'wrong asymmetric mode/extent')
    require(receipt.get('baseline_repeat', False) is is_repeat, 'repeat dispatch identity')
    require(receipt.get('new_kernel_fault_records') == [], 'new kernel fault')
    require(receipt.get('full_trajectory_accepted') is False, 'control improperly promoted to trajectory')
    resources = receipt.get('main_compiler_resources', {})
    require_resources(resources)
    dispatch = files.binding(receipt['dispatch_receipt'])
    for name in ('command', 'input_hashes', 'build_receipt', 'diagnostic_mode', 'expected_epochs',
                 'native_exit_code', 'main_compiler_resources', 'boot_id'):
        require(dispatch.get(name) == receipt.get(name), 'dispatch/execution mismatch: '+name)
    require(dispatch.get('baseline_repeat', False) is is_repeat, 'dispatch repeat mismatch')
    command = receipt['command']
    require(len(command) == 6 and command[5] == mode, 'unexpected native CLI')
    binary, case, selected, mass, prefix = command[:5]
    expected = receipt['input_hashes']
    require(all(p in expected for p in (binary, case, selected, mass)), 'unbound binary/physical input')
    for source, digest in expected.items():
        files.read(source, digest)
    build = files.binding(receipt['build_receipt'])
    require(build.get('exit_code') == 0 and build.get('sources_unchanged') is True
            and build.get('header_overlays') is False, 'build was not canonical and successful')
    require(build.get('binary_sha256') == expected[binary], 'build/execution binary mismatch')
    for source, digest in build['input_hashes'].items():
        require(expected.get(source) == digest, 'source missing from dispatch: '+source)
    checks = receipt.get('independent_dec_checks', [])
    require(len(checks) == 1 and checks[0].get('epoch') == 1 and checks[0].get('exit_code') == 0,
            'missing accepted one-epoch independent DEC check')
    dec = files.binding(checks[0]['result'])
    require(dec.get('status') == 'PASS' and dec.get('epoch') == 1 and dec.get('entry_substeps') == 0
            and len(dec.get('leaves', [])) == LEAVES and all(x.get('pass') is True for x in dec['leaves']),
            'independent DEC result not accepted')
    require(dec.get('execution_receipt') == {'path': receipt['dispatch_receipt']['path'],
            'sha256': receipt['dispatch_receipt']['sha256'],
            'bytes': files.identities[str(Path(receipt['dispatch_receipt']['path']).resolve())]['bytes']},
            'independent DEC was not bound to this dispatch')
    check_dec_source_bindings(files, dec, checks[0], expected)

    def raw(suffix):
        filename = prefix+suffix
        binding = dispatch['artifacts'].get(filename)
        require(binding is not None, 'unbound raw output: '+filename)
        data = files.read(filename, binding['sha256'])
        require(len(data) == binding['bytes'], 'raw output byte extent: '+filename)
        return data

    initial = raw('.asymmetric_initial_tiles.bin')
    entry = raw('.epoch1.phase2_tiles.bin')
    endpoint = raw('.epoch1.tiles.bin')
    exchange = raw('.epoch1.exchange.bin')
    convergence = raw('.epoch1.conv.bin')
    require(len(initial) == len(entry) == len(endpoint) == LEAVES*TILE_BYTES, 'eight-tile byte extent')
    bank_stride,footer_offset,_,bank_magic=EX.layout(exchange)
    require(len(exchange) == LEAVES*bank_stride and len(convergence) == LEAVES*CONV_BYTES, 'bank/record byte extent')
    # Paired runs must start from the complete coordinate authority. Equal
    # high projections alone can conceal a different physical input.
    physical_entry = [raw(s) for s in ('.epoch1.entry_au_coords.bin',
                                     '.epoch1.entry.au_coords_low.bin',
                                     '.epoch1.entry.au_start_high.bin', '.epoch1.entry.au_start_low.bin',
                                     '.epoch1.entry.au_reference_high.bin', '.epoch1.entry.au_reference_low.bin',
                                     '.epoch1.entry_native_coords.bin', '.epoch1.entry.native_coords_low.bin',
                                     '.epoch1.entry.native_start_high.bin', '.epoch1.entry.native_start_low.bin',
                                     '.epoch1.entry.native_reference_high.bin', '.epoch1.entry.native_reference_low.bin',
                                     '.epoch1.entry_au_velocities.bin',
                                     '.entry_control_before.forces.bin', '.peer_edge_map.bin')]
    check_precision_planes(physical_entry[:13])
    precise_input=raw('.epoch1.entry.spinor.owners.bps1')
    precise_endpoint=raw('.epoch1.spinor.owners.bps1')
    owners=SPINOR.endpoint(precise_endpoint,endpoint,4,1)
    physical_entry.append(b''.join(precise_input[l*SPINOR.SIZE+128:l*SPINOR.SIZE+6272] for l in range(8)))
    start, end, weights, densities, recorded, sequences, atom_rows = [], [], [], [], [], [], []
    for leaf in range(LEAVES):
        tile = endpoint[leaf*TILE_BYTES:(leaf+1)*TILE_BYTES]
        before = entry[leaf*TILE_BYTES:(leaf+1)*TILE_BYTES]
        ids = tuple(i & 0xffffff for i in struct.unpack_from('<256I', tile))
        require(len(set(ids)) == VERTICES and tile[:1024] == before[:1024], 'atom map changed/duplicated')
        atom_rows.append(ids[:CORE])
        bank = exchange[leaf*bank_stride:(leaf+1)*bank_stride]
        foot = struct.unpack_from('<8d4f12I', bank, footer_offset)
        require(all(math.isfinite(x) for x in foot[:12]), 'nonfinite coupling footer')
        identity = foot[12:20]
        require(identity[0] == bank_magic and identity[1] == leaf and identity[2] == 0
                and identity[3] == 3 and identity[4] > 0 and identity[5:] == (0, 1, 4)
                and not any(foot[20:]), 'unaccepted coupling footer')
        sequences.append(identity[4])
        rec = struct.unpack_from('<8I', convergence, leaf*CONV_BYTES)
        require(rec[0] == 0x42505631 and rec[1] == leaf and not any(rec[2:5]) and rec[5] == 4,
                'unaccepted convergence record')
        require(abs(Q.from_float(foot[6])) <= TOL and abs(Q.from_float(foot[7])) <= TOL,
                'global energy/chain gate')
        require(all(0 <= x <= 1e-7 for x in (foot[8], foot[10], foot[11])), 'footer residual gate')
        s0 = struct.unpack_from('<192d', bank, 0)
        s1 = struct.unpack_from('<192d', bank, 1536)
        omega = struct.unpack_from('<192f', bank, 3072)
        require(all(math.isfinite(x) for x in s0+s1+omega) and all(x > 0 for x in omega), 'ring scalar domain')
        require(bank[3072:3840] == tile[OMEGA:OMEGA+CORE*4], 'actual energy-weight bit mismatch')
        for u in range(CORE):
            psi=owners[leaf]['accepted'][u]
            psi0=SPINOR.read(precise_input,leaf)['accepted'][u]
            d, d0 = density(psi), density(psi0)
            require(abs(d[0]+d[1]-1) <= NORM_TOL and abs(d[0]+d[1]-d0[0]-d0[1]) <= NORM_TOL,
                    'represented endpoint norm gate')
            require(bits64(represented_population(psi)) == bits64(s1[u]), 'bank is not exported spinor population')
            densities.append(d)
        start.append(list(map(Q.from_float, s0)));end.append(list(map(Q.from_float, s1)))
        weights.append(list(map(Q.from_float, omega)))
        recorded.append(tuple(Q.from_float(x) for x in foot[1:5]))
    require(len(set(sequences)) == 1 and all(ids == atom_rows[0] for ids in atom_rows), 'ring epoch/correspondence')
    kappa = Q.from_float(f32(.1)) if mode.endswith('=0.1') else Q(0)
    ring = reconstruct_ring(start, end, weights, kappa)
    for leaf, (measured, expected_ring) in enumerate(zip(recorded, ring)):
        require(all(abs(a-b) <= TOL for a, b in zip(measured, expected_ring[:4])),
                'independent direct endpoint ring mismatch at leaf '+str(leaf))
        if kappa == 0:
            require(all(x == 0 for x in measured+expected_ring[:4]), 'kappa=0 has coupling work')
    nonzero_work = any(row[2] != 0 or row[3] != 0 for row in ring)
    nonzero_recorded = any(row[2] != 0 or row[3] != 0 for row in recorded)
    return {'receipt': str(Path(path).resolve()), 'binary_sha256': expected[binary],
            'source_hashes': build['input_hashes'], 'selected_sources': build['selected_sources'],
            'physical_input_hashes': [expected[p] for p in (case, selected, mass)],
            'initial': initial, 'entry': entry, 'physical_entry': physical_entry,
            'densities': densities, 'ring': ring, 'nonzero_ring_work_or_transfer': nonzero_work,
            'nonzero_recorded_ring_work_or_transfer': nonzero_recorded,
            'boot_id': receipt['boot_id'], 'mode': mode, 'repeat': is_repeat}


def compare_runs(zero, coupled, repeat=None):
    for other in ([coupled]+([repeat] if repeat is not None else [])):
        for key in ('binary_sha256', 'source_hashes', 'selected_sources', 'physical_input_hashes',
                    'initial', 'entry', 'physical_entry', 'boot_id'):
            require(zero[key] == other[key], 'fresh controls differ in '+key)
    metrics = response_metrics(zero['densities'], coupled['densities'],
                               repeat['densities'] if repeat is not None else None)
    independent_work = (coupled['nonzero_ring_work_or_transfer']
                        and coupled['nonzero_recorded_ring_work_or_transfer'])
    accepted = repeat is not None and independent_work and metrics['components_beyond_both_observed_repeat_envelopes'] > 0
    return {'status': 'PASS' if accepted else 'REFUSED', 'actual_endpoint_response_qualified': accepted,
            'nonzero_independently_reconstructed_kappa_work_or_transfer': independent_work,
            'zero_repeat_present': repeat is not None, **metrics,
            'coupled_leaf_ring_terms': [[float(v) for v in row] for row in coupled['ring']],
            'scope': 'Fresh same-binary/input asymmetric fixed-operator control. Exact represented density response beyond one observed zero-repeat envelope, with independent last-substep ring work. No universal nondeterminism bound, standard nonadiabatic benchmark, or full AMBER trajectory acceptance.'}


def self_test():
    # Deliberately synthetic checker controls, never a native physics claim.
    planes = [bytes(8*256*12)] * 13
    check_precision_planes(planes)
    invalid_low = bytearray(planes[1])
    struct.pack_into('<f', invalid_low, 0, math.nan)
    for invalid in ([planes[0], planes[1][:-4], *planes[2:]],
                    [planes[0], bytes(invalid_low), *planes[2:]]):
        try:
            check_precision_planes(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('malformed coordinate low plane accepted')
    resources = {'stack': 0, 'cumulative_stack': 0, 'spill_stores': 0,
                 'spill_loads': 0, 'registers': 128}
    require_resources(resources)
    for invalid in ({k: v for k, v in resources.items() if k != 'cumulative_stack'},
                    {**resources, 'cumulative_stack': 32}):
        try:
            require_resources(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('missing or nonzero cumulative stack accepted')
    zero = [(Q(1), Q(0), Q(0), Q(0))]*(LEAVES*CORE)
    change = zero.copy();change[0] = (Q(3, 4), Q(1, 4), Q(1, 4), Q(0))
    assert response_metrics(zero, zero, zero)['components_beyond_both_observed_repeat_envelopes'] == 0
    assert response_metrics(zero, change, zero)['components_beyond_both_observed_repeat_envelopes'] == 3
    assert response_metrics(zero, change, change)['components_beyond_both_observed_repeat_envelopes'] == 0
    repeat = zero.copy();repeat[0] = (Q(7, 8), Q(1, 8), Q(1, 8), Q(0))
    assert response_metrics(zero, change, repeat)['components_beyond_both_observed_repeat_envelopes'] == 0
    assert density((.6, .8, 0., 0.)) == density((-.6, -.8, -0., -0.))
    a = [[Q(leaf-3, 8)]*CORE for leaf in range(LEAVES)]
    b = [row.copy() for row in a];b[2][0] += Q(1, 1024)
    weights = [[Q(1+leaf, 8)]*CORE for leaf in range(LEAVES)]
    ring = reconstruct_ring(a, b, weights, Q.from_float(f32(.1)))
    assert any(r[2] != 0 or r[3] != 0 for r in ring)
    assert all(all(v == 0 for v in r) for r in reconstruct_ring(a, b, weights, Q(0)))
    assert all(all(v == 0 for v in r[:4]) for r in reconstruct_ring(a, a, weights, Q.from_float(f32(.1))))
    x = {'binary_sha256': 'test-only', 'source_hashes': {}, 'selected_sources': {},
         'physical_input_hashes': [], 'initial': b'fixed', 'entry': b'fixed',
         'physical_entry': [], 'boot_id': 'test-only', 'densities': zero,
         'ring': ring, 'nonzero_ring_work_or_transfer': False,
         'nonzero_recorded_ring_work_or_transfer': False}
    y = {**x, 'densities': change, 'nonzero_ring_work_or_transfer': True,
         'nonzero_recorded_ring_work_or_transfer': True}
    assert compare_runs(x, y, x)['status'] == 'PASS'
    assert compare_runs(x, y)['status'] == 'REFUSED'
    assert compare_runs(x, {**y, 'nonzero_ring_work_or_transfer': False}, x)['status'] == 'REFUSED'
    assert compare_runs(x, {**y, 'nonzero_recorded_ring_work_or_transfer': False}, x)['status'] == 'REFUSED'
    assert compare_runs(x, {**y, 'densities': zero}, x)['status'] == 'REFUSED'
    for key in ('binary_sha256', 'source_hashes', 'physical_input_hashes', 'initial', 'entry', 'physical_entry'):
        try:
            compare_runs(x, {**y, key: 'changed'}, x)
        except ValueError:
            pass
        else:
            raise AssertionError('changed '+key+' accepted')
    return {'status': 'PASS', 'checker_control_count': 25,
            'scope': 'Exact algebra/adversarial checker controls only; native readbacks tested=0'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zero', type=Path)
    parser.add_argument('--coupled', type=Path)
    parser.add_argument('--zero-repeat', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    try:
        if args.self_test:
            result = self_test()
        else:
            require(args.zero is not None and args.coupled is not None, '--zero and --coupled execution receipts required')
            files = Files()
            zero = load_run(files, args.zero, '--asymmetric-kappa=0')
            coupled = load_run(files, args.coupled, '--asymmetric-kappa=0.1')
            repeat = load_run(files, args.zero_repeat, '--asymmetric-kappa=0', True) if args.zero_repeat else None
            result = compare_runs(zero, coupled, repeat)
            files.unchanged()
            result['verified_files'] = files.identities
            result['execution_receipts'] = [r['receipt'] for r in (zero, coupled, repeat) if r is not None]
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result['status'] == 'PASS' else 2
    except (ValueError, KeyError, TypeError, OSError, struct.error) as error:
        print(json.dumps({'status': 'REFUSED', 'error': str(error),
                          'actual_endpoint_response_qualified': False}, indent=2))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
