#!/usr/bin/env python3
"""Assemble the operator's seven flat, source-derived forensic packets.

This is an archive assembler, not a GPU runner or a scientific certifier. It
never creates a successful engine status. --check-only writes nothing, and an
explicit --production-absent reason always prevents archive publication.

The required --evidence file is a real evidence selection, not a generated
example. Schema prism.biprism.forensic_inputs.v1 has:

* files: ID -> {path, sha256, bytes, original_path?}; paths are absolute.
* accepted_build: file ID of the selected coupled build receipt.
* compiler_stderr: file ID of that build's actual .stderr resource log.
* native_controls: ID -> {kind: coupled|phase2, receipt: file ID,
  sources: [current source IDs], mode?: exact coupled diagnostic_mode}.
  CLI --native-control ID=PATH must select this exact population.
* checks: ID -> {control: native-control ID, artifact: file ID,
  kind: json|kv, assertions: [...]}. JSON assertions have path (a list of
  explicit keys/indices), op (eq|le|ge), value. KV assertions have prefix,
  select (exact field strings), count (exact positive count), eq and positive
  field lists/maps. No shell command or arbitrary expression is accepted.
* failures: exactly seven objects {id, invariant, fragments, checks, evidence,
  open_requirements}. Each fragment has label, baseline_kind
  (historical_failure|immediate_predecessor), before/current {file,start,end}.
  Bounds are inclusive original line numbers. Whole source bytes and exact
  excerpts are retained, along with the generated before/current diff.
* production_recorder: file ID of committed run_biprism_v10_production.py.
* production_receipt: file ID matching --production-evidence exactly, required
  when production evidence is supplied. Its expected digest is never inferred.

Each failure needs actual native checks; optional unresolved scientific scope
stays OPEN. Mandatory accepted build/native/production gates are never waived.
Current code fragments must match a consumed selected build source identity.
The producer's native status is preserved, not renamed by this tool.

Publication writes failure_1.md ... failure_7.md, flat raw files,
AUDIT_SUMMARY.md and ARCHIVE_RECEIPT.json, then a flat ZIP. The receipt excludes
its own digest explicitly; the ZIP digest is printed, not embedded cyclically.
Use --verify ZIP for portable byte/excerpt verification without original paths.
"""
from __future__ import annotations

import argparse
import contextlib
import difflib
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import zipfile

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



ROLE = Path(__file__).resolve().parents[1]
MAIN = ROLE / 'crates/prism-gpu/src/kernels/nhs_biprism_variational_so3g_v10.cuh'
DEFAULT_OUTPUT = _env_root('PRISM_BIPRISM_RESOLUTION_ROOT') / 'FORENSIC_SOURCE_AUDIT_20261018'
TITLES = {
    1: 'Electronic Generator vs Retained Potential & 0-ULP Provider Lockstep',
    2: 'Discrete Gonzalez Projection, Picard Convergence & EQR2 Closure',
    3: 'Joint Electronic Endpoint Iteration, Midpoint Density & Reciprocal Berry Work',
    4: 'Bounded DEC PCG & sm120 Zero-Spill Hardware',
    5: 'Seeded Contact-Domain Geometry 960 Tetrahedra / 1461 Edges & Phase2 Operator Refresh Bridge',
    6: 'Kappa PTX Load & Collective Eight-CTA Execution',
    7: 'Stream Table 640 / LIF Realized Momentum Work / E2E 40000 Outputs',
}
REQUIRED_MODES = {'--asymmetric-kappa=0', '--connected-work', '--fixed-domain-16'}
ALLOWED_MODES = REQUIRED_MODES | {'--asymmetric-kappa=0.1', 'default_three_epoch'}
ZERO_RESOURCE_FIELDS = ('stack', 'cumulative_stack', 'spill_stores', 'spill_loads')
HEX = re.compile(r'[0-9a-f]{64}\Z')
# Typed companion columns are a lossless view of these exact raw ABI fields.
# Source: biprism_spike.rs / biprism_spike_arrow.rs, V10S96 + NOB2_64.
LEDGER_FIELDS = {
    'step_tau': ('p', 'I', 12), 'leaf_id': ('p', 'H', 64), 'vertex_id': ('p', 'H', 66),
    'atom_or_site_id': ('p', 'I', 16), 'pocket_tile_id': ('p', 'I', 32),
    'residue_id': ('p', 'i', 36), 'voxel_cell_idx': ('p', 'i', 68),
    'atom_x': ('p', 'f', 0), 'atom_y': ('p', 'f', 4), 'atom_z': ('p', 'f', 8),
    'spike_amplitude': ('p', 'f', 20), 'holonomy_deficit': ('p', 'f', 24),
    'voronoi_dual_vol': ('p', 'f', 28), 'multiplicity_m': ('p', 'I', 40),
    'v_pre_reset': ('p', 'f', 44), 'q_discharged': ('p', 'f', 48),
    'q_discharged_lo': ('p', 'f', 52), 'r_post_reset': ('p', 'f', 56),
    'lif_threshold_used': ('p', 'f', 60), 'q_lo_product_raw_bits': ('p', 'I', 88),
    'produced_force_x': ('p', 'f', 76), 'produced_force_y': ('p', 'f', 80),
    'produced_force_z': ('p', 'f', 84), 'event_schema_magic': ('p', 'I', 72),
    'atom_x_low': ('o', 'f', 32), 'atom_y_low': ('o', 'f', 36), 'atom_z_low': ('o', 'f', 40),
    'coordinate_magic': ('o', 'I', 44), 'coordinate_generation': ('o', 'Q', 48),
    'coordinate_unit_frame': ('o', 'I', 56),
}


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def integer(value, expected):
    """JSON booleans are not native exit codes, counters or resource values."""
    return type(value) is int and value == expected


def read_pinned(pin):
    raw = Path(pin['path']).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == pin['sha256'], f'input changed: {pin["path"]}')
    return raw


def strict_json(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, f'duplicate JSON key: {key}')
            out[key] = value
        return out
    def invalid(value):
        raise AuditError(f'nonfinite JSON constant: {value}')
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    def finite(item):
        if type(item) is float:
            require(math.isfinite(item), 'nonfinite JSON numeric value')
        elif isinstance(item, dict):
            for v in item.values(): finite(v)
        elif isinstance(item, list):
            for v in item: finite(v)
    finite(value)
    return value


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stable_pin(path, expected=None, size=None):
    p = Path(path)
    require(p.is_absolute(), f'absolute input path required: {p}')
    before = p.lstat()
    require(stat.S_ISREG(before.st_mode), f'input is not a regular nonsymlink file: {p}')
    # A symlink in an ancestor also hides the selected source identity.
    require(p.resolve() == p, f'input path has an alias: {p}')
    h = digest(p)
    after = p.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    require(identity(before) == identity(after), f'input changed while hashing: {p}')
    require(expected is None or (type(expected) is str and HEX.fullmatch(expected) and h == expected), f'hash mismatch: {p}')
    require(size is None or type(size) is int and size == after.st_size, f'size mismatch: {p}')
    return {'path': str(p), 'sha256': h, 'bytes': after.st_size}


class Inputs:
    """Every packaged byte has one checked source; no recursive SHA guessing."""
    def __init__(self):
        self.files = {}
        self.ids = {}

    def add(self, pin, key=None):
        require(isinstance(pin, dict) and {'path', 'sha256'} <= pin.keys(), 'explicit file pin required')
        rec = stable_pin(pin['path'], pin['sha256'], pin.get('bytes', pin.get('size')))
        rec['original_path'] = pin.get('original_path', rec['path'])
        require(Path(rec['original_path']).is_absolute(), 'original source path must be absolute')
        path = rec['path']
        if path in self.files:
            require(self.files[path]['sha256'] == rec['sha256'], f'conflicting pins: {path}')
        else:
            safe = re.sub('[^A-Za-z0-9_.-]', '_', Path(path).name)[:100]
            rec['archive_name'] = f'raw_{rec["sha256"]}_{safe}'
            self.files[path] = rec
        if key is not None:
            require(key not in self.ids, f'duplicate input ID: {key}')
            self.ids[key] = path
        return self.files[path]

    def item(self, key):
        require(key in self.ids, f'unknown evidence ID: {key}')
        return self.files[self.ids[key]]

    def read(self, key):
        rec = self.item(key)
        return read_pinned(rec)

    def document(self, key):
        return strict_json(self.read(key))

    def reference(self, pin):
        return self.add(pin)

    def retain_artifacts(self, receipt):
        artifacts = receipt.get('artifacts')
        require(isinstance(artifacts, dict) and artifacts, 'native artifact population is absent')
        for path, pin in artifacts.items():
            require('path' not in pin or pin['path'] == path, 'artifact key/path mismatch')
            self.add({**pin, 'path': path})

    def retain_hashes(self, population):
        require(isinstance(population, dict) and population, 'consumed input population is absent')
        for path, sha in population.items():
            self.add({'path': path, 'sha256': sha})


def committed_script(path):
    p = Path(path).resolve()
    root = Path(subprocess.check_output(['git', '-C', str(p.parent), 'rev-parse', '--show-toplevel'], text=True).strip())
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    rel = p.relative_to(root).as_posix()
    raw = subprocess.check_output(['git', '-C', str(root), 'show', f'{commit}:{rel}'])
    require(raw == p.read_bytes(), f'assembler/validator must match committed source: {p}')
    return {'repository': str(root), 'commit': commit, 'path': rel, 'sha256': hashlib.sha256(raw).hexdigest()}


def load_pinned_module(pin, name):
    """Execute the checked source bytes without reading or writing its bytecode cache."""
    raw = read_pinned(pin)
    module_spec = importlib.util.spec_from_file_location(name, pin['path'])
    module = importlib.util.module_from_spec(module_spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        exec(compile(raw, pin['path'], 'exec'), module.__dict__)
    finally:
        sys.dont_write_bytecode = previous
    return module


def validate_build(inputs, key):
    b = inputs.document(key)
    require(integer(b.get('exit_code'), 0) and b.get('sources_unchanged') is True, 'selected native build failed or changed')
    require(b.get('header_overlays') is False, 'native build has source overlays')
    command = b['command']
    require('-arch=sm_120' in command and command.count('-o') == 1, 'actual sm120 compiler command required')
    binary = command[command.index('-o') + 1]
    inputs.add({'path': binary, 'sha256': b['binary_sha256']})
    inputs.retain_hashes(b.get('input_hashes'))
    return b


def native_resources(raw):
    """Parse the actual selected translation-unit report, not a claimed count."""
    text = raw.decode('utf-8')
    symbol = 'nhs_biprism_variational_so3g_v10'
    start = f"Compiling entry function '{symbol}'"
    require(text.count(start) == 1, 'native main compiler report absent or ambiguous')
    block = text.split(start, 1)[1].split('Compiling entry function ', 1)[0]
    require(re.search(r"for 'sm_120'", block) is not None, 'native compiler report is not sm120')
    headers = [s.strip() for s in re.findall(r'Function properties for ([^\n]+)', block)]
    rows = [{'symbol': name.strip(), 'stack': int(stack), 'spill_stores': int(stores), 'spill_loads': int(loads)}
            for name, stack, stores, loads in re.findall(
                r'Function properties for ([^\n]+)\s*\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads', block)]
    # ptxas may repeat a callee symbol. Preserve every emitted row and check it;
    # deduplication could hide a later nonzero resource row.
    require(len(rows) > 1 and headers == [r['symbol'] for r in rows],
            'native main/callee compiler coverage is incomplete')
    require(all(r['stack'] == r['spill_stores'] == r['spill_loads'] == 0 for r in rows), 'actual native compiler stack/spill gate failed')
    main = block.split('Function properties for '+symbol, 1)[1].split('Function properties for ', 1)[0]
    used = re.findall(r'Used (\d+) registers([^\n]*)', main)
    require(len(used) == 1 and 0 < int(used[0][0]) <= 128, 'actual native main register gate failed')
    cumulative = re.findall(r'(\d+) bytes cumulative stack size', used[0][1])
    require(len(cumulative) <= 1 and all(int(n) == 0 for n in cumulative), 'actual native cumulative stack is nonzero')
    return {'main': {'registers': int(used[0][0]), 'stack': 0, 'spill_stores': 0, 'spill_loads': 0, 'cumulative_stack': 0},
            'callees': [r for r in rows if r['symbol'] != symbol]}


def source_bindings(inputs, ids, build, receipt):
    require(isinstance(ids, list) and ids, 'native current-source bindings absent')
    for key in ids:
        source = inputs.item(key)
        path = source['original_path']
        require(build['input_hashes'].get(path) == source['sha256'] == receipt['input_hashes'].get(path),
                f'current source is not consumed by this native control: {path}')


def validate_native(inputs, definition, cli_path, accepted_build):
    rec = inputs.item(definition['receipt'])
    require(Path(cli_path).absolute() == Path(rec['path']), 'CLI native receipt differs from evidence selection')
    r = inputs.document(definition['receipt'])
    require(r.get('native_execution') is True and integer(r.get('native_exit_code'), 0) and integer(r.get('exit_code'), 0),
            'native control did not complete successfully')
    require(r.get('sources_binary_inputs_unchanged') is True and r.get('gpu_recovery_none_after') is True
            and r.get('new_kernel_fault_records') == [] and bool(r.get('boot_id')), 'native source/health gate failed')
    require(r.get('production_admission_reset') is False and r.get('installed_state_mutation') is False,
            'diagnostic altered production/installed state')
    build_pin = inputs.reference(r['build_receipt'])
    b = strict_json(read_pinned(build_pin))
    require(integer(b.get('exit_code'), 0) and b.get('sources_unchanged') is True and b.get('header_overlays') is False,
            'native control build is not accepted')
    binary = r['command'][0]
    require(b['binary_sha256'] == r['input_hashes'].get(binary), 'control binary/build mismatch')
    inputs.add({'path': binary, 'sha256': b['binary_sha256']})
    dispatch = inputs.reference(r['dispatch_receipt'])
    d = strict_json(read_pinned(dispatch))
    for field in ('command', 'input_hashes', 'native_exit_code', 'native_execution', 'boot_id',
                  'sources_binary_inputs_unchanged', 'artifacts', 'gpu_recovery_none_after', 'new_kernel_fault_records'):
        require(field in d and exact_value(d[field], r.get(field)), f'execution/dispatch mismatch: {field}')
    source_bindings(inputs, definition['sources'], b, r)
    inputs.retain_hashes(r['input_hashes'])
    if definition['kind'] == 'coupled':
        require(build_pin['sha256'] == accepted_build['sha256'], 'old coupled build cannot qualify selected main')
        require(r.get('diagnostic_mode') == definition['mode'] and definition['mode'] in ALLOWED_MODES, 'native mode mismatch')
        require(any(inputs.item(s)['original_path'] == str(MAIN) for s in definition['sources']), 'coupled main binding absent')
        resource = r['main_compiler_resources']
        require(all(integer(resource.get(k), 0) for k in ZERO_RESOURCE_FIELDS), 'main stack/spill/cumulative gate failed')
        require(type(resource.get('registers')) is int and 0 < resource['registers'] <= 128, 'main register gate failed')
        for item in r['main_device_callee_resources']:
            require(all(integer(item.get(k), 0) for k in ('stack', 'spill_stores', 'spill_loads')), 'device callee has stack/spills')
        require(r['main_device_callee_resources'], 'device callee resource population is absent')
        require(r.get('coupled_diagnostic_accepted') is True and r.get('independent_dec_all_epochs_passed') is True,
                'native or independent all-epoch acceptance absent')
        epochs = r['expected_epochs']
        expected = 16 if definition['mode'] == '--fixed-domain-16' else (1 if definition['mode'].startswith('--asymmetric-kappa=') else 3)
        require(integer(epochs, expected), 'native accepted epoch population differs')
        require([x['epoch'] for x in r['independent_dec_checks']] == list(range(1, epochs + 1)), 'DEC epoch coverage differs')
        for check in r['independent_dec_checks']:
            require(integer(check['exit_code'], 0) and check.get('error') is None, 'independent DEC check failed')
            inputs.reference(check['result'])
        if definition['mode'] == '--connected-work':
            require(r.get('independent_lif_work_passed') is True and integer(r['independent_lif_work_check']['exit_code'], 0),
                    'realized LIF work check is absent')
            check = r['independent_lif_work_check']
            require(check['command'].count('--output') == 1, 'LIF work output binding is ambiguous')
            inputs.reference({'path': check['command'][check['command'].index('--output')+1], 'sha256': check['output_sha256']})
        if definition['mode'] == '--fixed-domain-16':
            require(r.get('fixed_domain_readback_passed') is True, 'fixed-domain64-substep check is absent')
            inputs.reference(r['fixed_domain_readback'])
        prefix = r['command'][4]
        for epoch in range(1, epochs + 1):
            p = f'{prefix}.epoch{epoch}.conv.bin'
            pin = r['artifacts'].get(p)
            require(pin is not None, 'all-eight raw convergence is absent')
            inputs.add({'path': p, **pin})
            raw = read_pinned(inputs.files[p])
            require(len(raw) == 8 * 128, 'convergence extent differs')
            for leaf in range(8):
                words = struct.unpack_from('<32I', raw, leaf * 128)
                require(words[0] == 0x42505631 and words[1] == leaf and words[2:5] == (0, 0, 0) and words[5] == 4 * epoch,
                        f'unaccepted/partial native epoch {epoch}, leaf {leaf}')
    elif definition['kind'] == 'phase2':
        require(r.get('phase2_measurement_controls_passed') is True, 'Phase2 independent acceptance absent')
        require(r.get('expected_cases') == ['cold_placeholder', 'cold_leaf_refusal', 'unchanged', 'deformed',
                                          'boundary', 'identity_refusal', 'low_plane'], 'Phase2 seven-control population differs')
        require(integer(r['independent_check']['exit_code'], 0) and r['independent_check']['outputs_unchanged'] is True,
                'Phase2 independent readback failed')
        for pin in independent_references(r):
            inputs.reference(pin)
    else:
        raise AuditError('unsupported native control kind')
    inputs.retain_artifacts(r)
    return r


def exact_value(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(exact_value(actual[k], expected[k]) for k in actual)
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(exact_value(a, b) for a, b in zip(actual, expected))
    return actual == expected


def independent_references(native):
    # An independent result may be published after the raw dispatch artifact map.
    refs = [v.get('result') for v in native.get('independent_dec_checks', [])]
    refs.append(native.get('fixed_domain_readback'))
    if native.get('independent_lif_work_passed') is True:
        lif = native['independent_lif_work_check']
        refs.append({'path': lif['command'][lif['command'].index('--output')+1], 'sha256': lif['output_sha256']})
    if native.get('phase2_measurement_controls_passed') is True:
        dispatch = native['dispatch_receipt']['path']
        require(dispatch.endswith('_dispatch.json'), 'Phase2 dispatch suffix differs from actual runner')
        prefix = dispatch[:-len('_dispatch.json')] + '_independent'
        for stream in ('stdout', 'stderr'):
            refs.append({'path': prefix + '.' + stream,
                         'sha256': native['independent_check'][stream+'_sha256']})
    return [pin for pin in refs if pin is not None]


def check_assertions(inputs, check, native):
    item = inputs.item(check['artifact'])
    bound = native['artifacts'].get(item['path'])
    if bound is None:
        bound = next((v for v in independent_references(native) if v.get('path') == item['path']), None)
    require(bound is not None and bound['sha256'] == item['sha256'], 'check input is not a bound native output')
    assertions = check['assertions']
    require(isinstance(assertions, list) and assertions, 'empty scientific check')
    raw = inputs.read(check['artifact'])
    if check['kind'] == 'json':
        doc = strict_json(raw)
        for assertion in assertions:
            value = doc
            require(isinstance(assertion['path'], list) and assertion['path'], 'explicit JSON path required')
            for token in assertion['path']:
                value = value[token]
            op, expected = assertion['op'], assertion['value']
            if op == 'eq':
                require(exact_value(value, expected), f'JSON predicate differs: {assertion["path"]}')
            else:
                require(type(value) in (int, float) and type(expected) in (int, float)
                        and math.isfinite(value) and math.isfinite(expected), 'finite numeric predicate required')
                require(op in ('le', 'ge'), 'unsupported predicate operation')
                require(value <= expected if op == 'le' else value >= expected, 'numeric predicate failed')
    elif check['kind'] == 'kv':
        lines = raw.decode('utf-8').splitlines()
        for assertion in assertions:
            prefix, select = assertion['prefix'], assertion['select']
            require(prefix and not any(c.isspace() for c in prefix), 'exact native record prefix required')
            records = []
            for line in lines:
                if line.split(' ', 1)[0] != prefix:
                    continue
                fields = {}
                for token in line.split()[1:]:
                    if '=' in token:
                        k, v = token.split('=', 1)
                        require(k not in fields, 'duplicate native field')
                        fields[k] = v
                if all(fields.get(k) == str(v) for k, v in select.items()):
                    records.append(fields)
            count = assertion['count']
            require(type(count) is int and count > 0 and len(records) == count, 'native record coverage differs')
            require(assertion.get('eq') or assertion.get('positive'), 'native predicate is empty')
            for fields in records:
                for key, value in assertion.get('eq', {}).items():
                    require(fields.get(key) == str(value), f'native {prefix}.{key} differs')
                for key in assertion.get('positive', []):
                    value = float(fields[key])
                    require(math.isfinite(value) and value > 0, f'native {prefix}.{key} was not exercised')
    else:
        raise AuditError('unsupported evidence check kind')


def excerpt(inputs, definition):
    rec = inputs.item(definition['file'])
    lines = inputs.read(definition['file']).splitlines(keepends=True)
    start, end = definition['start'], definition['end']
    require(type(start) is int and type(end) is int and 1 <= start <= end <= len(lines), 'invalid source line interval')
    data = b''.join(lines[start - 1:end])
    require(data.strip() and b'\0' not in data, 'empty/binary code excerpt')
    data.decode('utf-8')
    return rec, data


def validate_failures(inputs, spec, checked, consumed_sources):
    failures = spec['failures']
    require(len(failures) == 7 and all(integer(f['id'], n) for n, f in enumerate(failures, 1)),
            'exact ordered seven-failure population required')
    states = []
    for f in failures:
        require(isinstance(f['invariant'], str) and f['invariant'].strip(), 'invariant missing')
        require(f['fragments'] and f['checks'] and f['evidence'], 'narrative-only packet refused')
        require(all(x in checked for x in f['checks']), 'packet names an unvalidated check')
        for key in f['evidence']:
            inputs.item(key)
        for fragment in f['fragments']:
            require(fragment['baseline_kind'] in ('historical_failure', 'immediate_predecessor') and fragment['label'],
                    'exact preceding/failing baseline classification required')
            excerpt(inputs, fragment['before'])
            current, _ = excerpt(inputs, fragment['current'])
            require(consumed_sources.get(current['original_path']) == current['sha256'],
                    f'current fragment has no selected build identity: {current["original_path"]}')
        require(isinstance(f['open_requirements'], list) and all(isinstance(v, str) and v for v in f['open_requirements']),
                'open requirements must remain explicit')
        states.append({'failure': f['id'], 'title': TITLES[f['id']],
                       'invariant': f['invariant'], 'status': 'OPEN' if f['open_requirements'] else 'SATISFIED',
                       'open_requirements': f['open_requirements'], 'checks': f['checks']})
    return states


def signal_grid(path, scope, run_id):
    raw = Path(path).read_bytes()
    require(raw[:8] == b'PRSGD001' and struct.unpack_from('<III', raw, 8) == (1, 0x01020304, 0xffffffff),
            'four-grid envelope/header differs')
    offset = 20
    for expected in (run_id, 'eight_leaf_aggregate'):
        n, = struct.unpack_from('<Q', raw, offset); offset += 8
        require(n <= 65536 and raw[offset:offset+n].decode() == expected, 'grid run/aggregate identity differs')
        offset += n
    count, stride, size = struct.unpack_from('<QQQ', raw, offset); offset += 24
    dim, voxels = struct.unpack_from('<QQ', raw, offset)
    require(dim > 0 and dim ** 3 == voxels == count == scope['voxel_count'] and stride == 4
            and size == 16 + 16 * count and len(raw) == offset + size + 16 == scope['serialized_bytes'],
            'four-grid extent differs')
    h = 0xcbf29ce484222325
    for byte in memoryview(raw)[offset:offset+size]:
        h = ((h ^ byte) * 0x100000001b3) & 0xffffffffffffffff
    require(struct.unpack_from('<Q', raw, offset+size)[0] == h == scope['payload_fnv1a64']
            and raw[-8:] == b'PRSGDEND' and scope['through_completed_step'] == 40000, 'grid checksum/clock differs')
    return {'voxels': count, 'planes': 4, 'payload_fnv1a64': h}


def arrow_rows(path, required, columns=None, metadata_out=None):
    try:
        import pyarrow as pa
    except ImportError as error:
        raise AuditError('actual Arrow validation requires pyarrow; no file-presence substitute') from error
    with pa.memory_map(str(path), 'r') as source:
        try:
            reader = pa.ipc.open_file(source)
            batches = (reader.get_batch(i) for i in range(reader.num_record_batches))
        except pa.ArrowInvalid:
            source.seek(0)
            reader = pa.ipc.open_stream(source)
            batches = iter(reader)
        require(len(reader.schema.names) == len(set(reader.schema.names)) and set(required) <= set(reader.schema.names),
                'Arrow schema is missing/duplicating native identity columns')
        types = {'spike_id': pa.uint64(), 'replica_seed': pa.uint64(), 'stream_id': pa.uint8(),
                 'timestep': pa.int32(), 'spike_source': pa.int32(), 'step_tau': pa.uint32(),
                 'leaf_id': pa.uint16(), 'raw_v10s_96': pa.binary(), 'raw_nob2_64': pa.binary()}
        kinds = {'f': pa.float32(), 'I': pa.uint32(), 'H': pa.uint16(), 'i': pa.int32(), 'Q': pa.uint64()}
        types.update({name: kinds[fmt] for name, (_, fmt, _) in LEDGER_FIELDS.items()})
        require(all(reader.schema.field(name).type == types[name] for name in required), 'Arrow identity ABI types differ')
        if metadata_out is not None:
            metadata_out.update(reader.schema.metadata or {})
        if columns is not None:
            require(len(reader.schema) == columns, 'native V5 column population differs')
        for batch in batches:
            table = {name: batch.column(batch.schema.get_field_index(name)).to_pylist() for name in required}
            for i in range(batch.num_rows):
                row = {name: values[i] for name, values in table.items()}
                require(all(v is not None for v in row.values()), 'null Arrow identity/raw field')
                yield row


def paired_arrow(directory, config, expected_raw=None, leaf_only=None, counts=None, stream_hashes=None,
                 step_window=(1, 40000), coordinate_generations=None):
    nmeta, lmeta = {}, {}
    native = arrow_rows(directory/'spike_events.arrow',
                        ('spike_id', 'replica_seed', 'stream_id', 'timestep', 'spike_source'), 30, nmeta)
    ledger = arrow_rows(directory/'voronoi_ledger.arrow',
                        ('spike_id', 'replica_seed', 'raw_v10s_96', 'raw_nob2_64', *LEDGER_FIELDS), 34, lmeta)
    count = 0
    with contextlib.ExitStack() as stack:
        raw = stack.enter_context(expected_raw[0].open('rb')) if expected_raw else None
        obs = stack.enter_context(expected_raw[1].open('rb')) if expected_raw else None
        for i, (a, b) in enumerate(itertools.zip_longest(native, ledger)):
            require(a is not None and b is not None, 'paired Arrow row counts differ')
            leaf, step = b['leaf_id'], b['step_tau']
            require(type(leaf) is int and 0 <= leaf < 8 and (leaf_only is None or leaf == leaf_only), 'Arrow leaf identity differs')
            seed, stream = config['leaves'][leaf]['seed'], config['leaves'][leaf]['stream_id']
            require(a['spike_id'] == b['spike_id'] == i and a['replica_seed'] == b['replica_seed'] == seed
                    and a['stream_id'] == stream and a['timestep'] == step and a['spike_source'] == 6
                    and type(step) is int and step_window[0] <= step <= step_window[1], 'Arrow seed/source/row/step join differs')
            packet, observation = b['raw_v10s_96'], b['raw_nob2_64']
            require(isinstance(packet, bytes) and len(packet) == 96 and isinstance(observation, bytes) and len(observation) == 64,
                    'retained event/observation ABI differs')
            for name, (owner, fmt, offset) in LEDGER_FIELDS.items():
                raw_field = (packet if owner == 'p' else observation)[offset:offset+struct.calcsize(fmt)]
                value = b[name]
                require(fmt != 'f' or math.isfinite(value), 'nonfinite typed ledger value')
                require(struct.pack('<'+fmt, value) == raw_field, f'typed ledger field differs from raw bytes: {name}')
            require(struct.unpack_from('<I', packet, 12)[0] == step == struct.unpack_from('<I', observation)[0]
                    and struct.unpack_from('<H', packet, 64)[0] == leaf == struct.unpack_from('<H', observation, 4)[0],
                    'Arrow/raw packet identity differs')
            require(struct.unpack_from('<H', packet, 66)[0] == struct.unpack_from('<H', observation, 6)[0]
                    and struct.unpack_from('<I', packet, 16)[0] == struct.unpack_from('<I', observation, 8)[0]
                    and struct.unpack_from('<i', packet, 68)[0] == struct.unpack_from('<i', observation, 12)[0],
                    'event/observation vertex, atom or cell differs')
            coordinate_generation = struct.unpack_from('<Q', observation, 48)[0]
            require(struct.unpack_from('<I', packet, 72)[0] == 0x56313053 and packet[92:96] == b'\0'*4
                    and struct.unpack_from('<I', observation, 44)[0] == 0x4e4f4232
                    and coordinate_generation > 0
                    and struct.unpack_from('<II', observation, 56) == (2, 0), 'event/coordinate schema or generation differs')
            require(coordinate_generations is None or coordinate_generation == coordinate_generations[leaf],
                    'observation coordinate allocation generation differs from retained owner')
            if raw:
                require(raw.read(96) == packet and obs.read(64) == observation, 'paired Arrow changed raw queue bytes')
            if counts is not None:
                counts[leaf] += 1
            if stream_hashes is not None:
                stream_hashes[leaf].update(packet)
                stream_hashes[leaf].update(observation)
            count += 1
        if raw:
            require(raw.read(1) == obs.read(1) == b'', 'raw queue rows omitted by paired Arrow')
    for key in (b'prism.queue_owner', b'prism.queue_generation', b'prism.leaf_replica_seeds'):
        require(key in nmeta and nmeta[key] == lmeta.get(key), 'paired Arrow export identity metadata differs')
    require(nmeta[b'prism.queue_owner'] and nmeta[b'prism.queue_generation'].isdigit(), 'invalid paired export owner/generation')
    require(exact_value(strict_json(nmeta[b'prism.leaf_replica_seeds']), [l['seed'] for l in config['leaves']]),
            'Arrow metadata seed population differs')
    for key, value in {b'prism.schema': b'voronoi_lif_ledger_v2', b'prism.event_frame': b'atomic_bohr',
                       b'prism.position_unit': b'bohr', b'prism.force_unit': b'hartree/bohr',
                       b'prism.dual_volume_unit': b'bohr^3', b'prism.native_position_unit': b'angstrom'}.items():
        require(lmeta.get(key) == value, 'companion coordinate/unit metadata differs')
    return count


def validate_delivery_outputs(root, config):
    """Exact consumer identity/extent checks; no substitute Hamiltonian oracle."""
    counts = [0] * 8
    generations = []
    for leaf in range(8):
        identities = []
        for frame, label in ((1, 'native'), (2, 'atomic')):
            image = strict_json((root/f'leaf{leaf:02}.{label}_coordinates_terminal.index.json').read_bytes())
            identity = image['identity']
            require(image['schema'] == 'biprism.coordinate_owner.v1' and integer(identity['version'], 1)
                    and integer(identity['leaf_id'], leaf) and integer(identity['unit_frame'], frame)
                    and type(identity['generation']) is int and identity['generation'] > 0
                    and type(identity['atom_count']) is int and identity['atom_count'] >= 256,
                    'terminal coordinate allocation identity differs')
            identities.append(identity)
        require(identities[0]['generation'] == identities[1]['generation']
                and identities[0]['atom_count'] == identities[1]['atom_count'], 'native/AU coordinate identities differ')
        generations.append(identities[0]['generation'])
    chunk_hashes = [hashlib.sha256() for _ in range(8)]
    for begin in range(0, 40000, 500):
        chunk = root/f'chunk_{begin:08}'
        paired_arrow(chunk, config, (chunk/'events.v10s96', chunk/'native_observations.64b'), counts=counts,
                     stream_hashes=chunk_hashes, step_window=(begin+1, begin+500), coordinate_generations=generations)
    leaf_hashes = [hashlib.sha256() for _ in range(8)]
    for leaf in range(8):
        path = root/f'leaf{leaf:02}'
        count = paired_arrow(path, config, leaf_only=leaf, stream_hashes=leaf_hashes, coordinate_generations=generations)
        require(count == counts[leaf], 'whole-leaf Arrow population differs from accepted chunks')
        require(leaf_hashes[leaf].digest() == chunk_hashes[leaf].digest(), 'whole-leaf packet stream differs from accepted chunks')
        validate_pathb(path, count)
    grid = signal_grid(root/'eight_leaf_signal_grids.bin',
                       strict_json((root/'signal_grid_scope.json').read_bytes()), config['run_id'])
    return {'event_rows_by_leaf': counts, 'packet_stream_sha256_by_leaf': [h.hexdigest() for h in chunk_hashes], 'grid': grid,
            'scope': 'full paired raw/Arrow leaf and seed identity, PathB population, four-grid envelope; native physical gates remain separate'}


def validate_pathb(path, count):
    selection = strict_json((path/'source_population_selection.json').read_bytes())
    analysis = strict_json((path/'pathb_analysis.json').read_bytes())
    require(integer(selection['unfiltered_rows'], count), 'PathB source population differs')
    selected = selection['selected_rows']
    require(type(selected) is int and 0 <= selected <= count and integer(analysis['native_ingest_rows'], selected),
            'PathB ingest population differs')
    sites = strict_json((path/'binding_sites.json').read_bytes())
    require(isinstance(sites['cluster_ids'], list) and len(sites['cluster_ids']) == selected
            and isinstance(sites['sites'], list), 'binding-site clustering population differs')
    if selected == 0:
        require(analysis['status'] == 'NO_EMITTED_POPULATION', 'empty PathB scope differs')
    else:
        require(analysis['status'] == 'NATIVE_ANALYSIS_EXECUTED', 'PathB native execution absent')
        state = analysis['native_state']
        data = (path/'pathb.native').read_bytes()
        offset, start, stride = state['event_count_offset'], state['event_buffer_offset'], state['event_stride_bytes']
        require(all(type(x) is int and x >= 0 for x in (offset, start, stride, state['config_size_bytes']))
                and state['endianness'] == 'little' and offset == state['config_size_bytes'] + 8 and start == offset + 4 and stride > 0
                and integer(state['event_count'], selected) and data[:4] == b'SDST'
                and struct.unpack_from('<I', data, 4)[0] == 1
                and struct.unpack_from('<I', data, offset)[0] == selected and len(data) >= start + selected * stride,
                'PathB serialized population/extent differs')


def production_receipt_pin(inputs, spec, path):
    require('production_receipt' in spec, 'explicit production receipt selection is required')
    pin = inputs.item(spec['production_receipt'])
    require(Path(path).absolute() == Path(pin['path']), 'CLI production receipt differs from evidence selection')
    return pin


def bind_production_inventory(inputs, receipt, root, extent):
    """Retain exactly the inventory/result independently validated at the run root."""
    for field in ('artifact_inventory', 'execution_result'):
        expected_path = str(root / (field + '.json'))
        selected, validated = receipt[field], extent[field]
        require(selected['path'] == validated['path'] == expected_path,
                f'production {field} is not the validated run-root file')
        require(selected['sha256'] == validated['sha256'],
                f'production {field} differs from validated bytes')
        selected_pin = inputs.reference(selected)
        validated_pin = inputs.reference(validated)
        require(selected_pin['bytes'] == validated_pin['bytes'], f'production {field} extent differs')


def validate_production(inputs, spec, path, current_main_sha):
    pin = production_receipt_pin(inputs, spec, path)
    receipt = strict_json(read_pinned(pin))
    require(receipt.get('schema') == 'prism.biprism.production_execution.v1', 'unsupported production recorder schema')
    require(receipt.get('native_execution') is True and integer(receipt.get('native_exit_code'), 0) and integer(receipt.get('exit_code'), 0),
            'actual production invocation did not succeed')
    require(receipt.get('sources_binary_inputs_unchanged') is True and receipt.get('boot_unchanged') is True
            and receipt.get('gpu_recovery_none_after') is True and receipt.get('new_kernel_fault_records') == [],
            'production source/health gate failed')
    recorder = inputs.item(spec['production_recorder'])
    require(Path(recorder['path']) == ROLE/'scripts/run_biprism_v10_production.py', 'unexpected production recorder owner')
    committed_script(recorder['path'])
    require(receipt['input_hashes'].get(recorder['path']) == recorder['sha256'], 'production recorder version differs')
    module = load_pinned_module(recorder, 'biprism_production_recorder_for_audit')
    config_pin = inputs.reference(receipt['config'])
    config = strict_json(read_pinned(config_pin))
    require(integer(config['full_step_budget'], 40000) and integer(config['chunk_steps'], 500) and len(config['leaves']) == 8,
            'production experiment is not40000/500/eight leaves')
    binary = inputs.reference(receipt['binary'])
    require(receipt['command'] == [binary['path'], 'biprism-run', '--config', config_pin['path']],
            'production command is not the actual requested entrypoint')
    build_pin = inputs.reference(receipt['build_receipt'])
    build = strict_json(read_pinned(build_pin))
    require(integer(build['exit_code'], 0) and build['sources_unchanged'] is True and build['governed_population_passed'] is True,
            'actual governed production build absent')
    require(build['binary_after']['sha256'] == binary['sha256'], 'production binary does not match actual build')
    require(build['source_after'][str(MAIN)]['sha256'] == current_main_sha, 'production uses a different main epoch')
    inputs.reference(receipt['trust_manifest'])
    inputs.retain_hashes(receipt['input_hashes'])
    for field in ('stdout', 'stderr'):
        inputs.reference(receipt[field])
    require(receipt.get('health_commands'), 'raw production health evidence absent')
    for check in receipt['health_commands']:
        require(integer(check.get('exit_code'), 0), 'production health command failed')
        inputs.reference(check['stdout']); inputs.reference(check['stderr'])
    resource = inputs.reference(receipt['resource_audit']['receipt'])
    audit = strict_json(read_pinned(resource))
    ptx = inputs.reference(audit['ptx'])
    require(receipt['input_hashes'].get(ptx['path']) == ptx['sha256'], 'resource PTX is not a production input')
    resource_pins = {}
    resource_result = module.verify_resource_audit(resource['path'], resource['sha256'], build_pin, build, ptx, resource_pins)
    inputs.retain_hashes(resource_pins)
    extent = module.validate_production_outputs(config, config_pin['sha256'])
    root = Path(config['output_root'])
    bind_production_inventory(inputs, receipt, root, extent)
    inventory = strict_json(read_pinned(inputs.files[receipt['artifact_inventory']['path']]))
    for item in inventory['files']:
        rel = Path(item['path'])
        require(not rel.is_absolute() and '..' not in rel.parts, 'production inventory path escapes its run')
        inputs.add({'path': str(root/rel), 'sha256': item['sha256'], 'bytes': item['bytes']})
    delivery = validate_delivery_outputs(root, config)
    require(receipt.get('engine_accepted_full_run') is True and receipt.get('output_extent_validated') is True,
            'producer did not accept the full production run')
    return {'extent': extent, 'delivery': delivery, 'resources': resource_result, 'receipt': pin}, build


def render_packets(inputs, spec, statuses):
    outputs, slices = {}, []
    for f, state in zip(spec['failures'], statuses):
        name = f'failure_{f["id"]}.md'
        buf = bytearray(f'# {TITLES[f["id"]]}\n\nInvariant: {f["invariant"]}\n\nStatus: {state["status"]}.\n\n'.encode())
        if state['open_requirements']:
            buf.extend(('Open requirements: '+'; '.join(state['open_requirements'])+'\n\n').encode())
        buf.extend(b'The following source blocks preserve exact bytes; line numbers refer to original files.\n\n')
        for fragment in f['fragments']:
            buf.extend(f'## {fragment["label"]}\n\nBaseline: {fragment["baseline_kind"]}.\n\n'.encode())
            pair = []
            for side in ('before', 'current'):
                definition = fragment[side]
                rec, raw = excerpt(inputs, definition)
                h = hashlib.sha256(raw).hexdigest()
                buf.extend((f'{side}: `{rec["original_path"]}` lines {definition["start"]}–{definition["end"]}. '
                            f'Whole SHA256 `{rec["sha256"]}`; excerpt SHA256 `{h}`. '
                            f'[Complete raw source]({rec["archive_name"]}).\n\n').encode())
                fence = b'`' * max(3, 1 + max((len(x) for x in re.findall(b'`+', raw)), default=0))
                buf.extend(fence+b'text\n')
                offset = len(buf); buf.extend(raw)
                slices.append({'document': name, 'offset': offset, 'bytes': len(raw), 'sha256': h,
                               'source': rec['archive_name'], 'first_line': definition['start'], 'last_line': definition['end']})
                if not raw.endswith(b'\n'):
                    buf.extend(b'\n')
                buf.extend(fence+b'\n\n')
                pair.append((rec, raw.decode('utf-8')))
            delta = ''.join(difflib.unified_diff(pair[0][1].splitlines(True), pair[1][1].splitlines(True),
                        fromfile=pair[0][0]['original_path']+f':{fragment["before"]["start"]}',
                        tofile=pair[1][0]['original_path']+f':{fragment["current"]["start"]}'))
            fence = '`' * max(3, 1 + max((len(x) for x in re.findall(r'`+', delta)), default=0))
            buf.extend((fence+'diff\n'+delta+'\n'+fence+'\n\n').encode())
        buf.extend(b'## Bound raw evidence\n\n')
        for key in f['evidence']:
            rec = inputs.item(key)
            buf.extend(f'- [{key}]({rec["archive_name"]}) — `{rec["path"]}`; SHA256 `{rec["sha256"]}`; {rec["bytes"]} bytes.\n'.encode())
        buf.extend(('\nParsed checks: '+', '.join(f['checks'])+'.\n').encode())
        outputs[name] = bytes(buf)
    return outputs, slices


def publish(inputs, spec, statuses, validation, output, archive, assembler_commit):
    require(validation.get('production') is not None and not validation.get('production_absence_reason'),
            'absent production prevents archive publication')
    require(output != archive, 'output directory and ZIP must be distinct')
    require(not output.exists() and not archive.exists(), 'refusing to replace an existing delivery or archive')
    require(output.parent.is_dir() and archive.parent == output.parent, 'output and ZIP must share an existing parent')
    generated, slices = render_packets(inputs, spec, statuses)
    summary = ['# Biprism v10 forensic source audit\n\n',
               'Source, native-control and production-output evidence retain their separate scopes. '
               'This assembler does not assign an engine implementation status or prove restart/experimental validity.\n\n']
    for row in statuses:
        summary.append(f'- [Failure {row["failure"]}: {row["title"]}](failure_{row["failure"]}.md): {row["status"]}.\n')
    summary.append('\nEvery raw source, baseline, selected receipt and consumed output is in this same directory. '
                   'ARCHIVE_RECEIPT.json binds exact payloads and embedded source intervals.\n')
    generated['AUDIT_SUMMARY.md'] = ''.join(summary).encode()
    created_output = False
    with tempfile.TemporaryDirectory(prefix='.biprism_audit_', dir=output.parent) as temp:
        stage = Path(temp)/'flat'; stage.mkdir()
        for rec in inputs.files.values():
            target = stage/rec['archive_name']
            if target.exists():
                require(digest(target) == rec['sha256'], 'flat-name collision')
                continue
            with Path(rec['path']).open('rb') as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst, 1024*1024)
            require(digest(target) == rec['sha256'] and target.stat().st_size == rec['bytes'], 'input changed during retention')
        for name, raw in generated.items():
            (stage/name).write_bytes(raw)
        payloads = [{'name': p.name, 'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(stage.iterdir())]
        receipt = {'schema': 'prism.biprism.forensic_archive.v1', 'assembler': assembler_commit,
                   'engine_status_synthesized': False, 'native_execution_by_assembler': False,
                   'production_absent': False, 'validation': validation, 'failures': statuses,
                   'source_map': list(inputs.files.values()), 'excerpts': slices, 'payloads': payloads,
                   'self_exclusion': 'ARCHIVE_RECEIPT.json excludes its own digest; ZIP SHA256 is returned on stdout.'}
        (stage/'ARCHIVE_RECEIPT.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
        zip_path = Path(temp)/'archive.zip'
        with zipfile.ZipFile(zip_path, 'x', zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
            for path in sorted(stage.iterdir()):
                z.write(path, path.name)
        verify_archive(zip_path)
        # Exclusive destination creation prevents replacing concurrent or old evidence.
        try:
            output.mkdir(); created_output = True
            for path in stage.iterdir():
                os.replace(path, output/path.name)
            os.link(zip_path, archive)  # exclusive, same filesystem; never overwrite
        except BaseException:
            if created_output and not archive.exists():
                shutil.rmtree(output)
            raise
    return {'archive': str(archive), 'sha256': digest(archive), 'bytes': archive.stat().st_size,
            'directory': str(output), 'files': len(payloads)+1, 'engine_status_synthesized': False}


def verify_archive(path):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        require(len(names) == len(set(names)) and all('/' not in n and '\\' not in n and n not in ('.', '..') for n in names),
                'archive is not a unique flat population')
        receipt = strict_json(z.read('ARCHIVE_RECEIPT.json'))
        require(receipt['schema'] == 'prism.biprism.forensic_archive.v1', 'archive receipt schema differs')
        required = {f'failure_{n}.md' for n in TITLES} | {'AUDIT_SUMMARY.md'}
        require(required <= set(names) and receipt.get('engine_status_synthesized') is False
                and receipt.get('native_execution_by_assembler') is False and receipt.get('production_absent') is False
                and receipt['validation'].get('production') is not None, 'archive scope/production binding differs')
        require(len(receipt['payloads']) == len({r['name'] for r in receipt['payloads']}), 'duplicate receipt payload')
        require(set(names) == {'ARCHIVE_RECEIPT.json'} | {r['name'] for r in receipt['payloads']}, 'archive payload population differs')
        for rec in receipt['payloads']:
            h = hashlib.sha256(); size = 0
            with z.open(rec['name']) as source:
                for block in iter(lambda: source.read(1024*1024), b''):
                    h.update(block); size += len(block)
            require(h.hexdigest() == rec['sha256'] and size == rec['bytes'], 'archive payload hash/extent differs')
        payload = {r['name']: r for r in receipt['payloads']}
        for rec in receipt['source_map']:
            target = payload.get(rec['archive_name'])
            require(target is not None and target['sha256'] == rec['sha256'] and target['bytes'] == rec['bytes'],
                    'portable source map differs from retained raw bytes')
        for rec in receipt['excerpts']:
            require(all(type(rec[k]) is int for k in ('offset', 'bytes', 'first_line', 'last_line'))
                    and rec['document'] in required and rec['source'] in payload and rec['offset'] >= 0 and rec['bytes'] > 0
                    and 1 <= rec['first_line'] <= rec['last_line'], 'invalid portable excerpt interval')
            document = z.read(rec['document'])
            source_lines = z.read(rec['source']).splitlines(True)
            require(rec['last_line'] <= len(source_lines) and rec['offset']+rec['bytes'] <= len(document),
                    'portable excerpt interval exceeds retained bytes')
            raw = document[rec['offset']:rec['offset']+rec['bytes']]
            source = b''.join(source_lines[rec['first_line']-1:rec['last_line']])
            require(raw == source and hashlib.sha256(raw).hexdigest() == rec['sha256'], 'embedded source excerpt differs')
    return {'verified_payloads': len(names), 'flat': True, 'native_execution': False}


def load_and_validate(args):
    inputs = Inputs()
    inputs.add(stable_pin(Path(args.evidence).absolute(), args.evidence_sha256), 'audit_inputs')
    spec = inputs.document('audit_inputs')
    require(spec['schema'] == 'prism.biprism.forensic_inputs.v1', 'unsupported evidence selection schema')
    for key, pin in spec['files'].items():
        inputs.add(pin, key)
    accepted = inputs.item(spec['accepted_build'])
    require(Path(args.accepted_build).absolute() == Path(accepted['path']), 'selected accepted build differs')
    build = validate_build(inputs, spec['accepted_build'])
    compiler = inputs.item(spec['compiler_stderr'])
    require(Path(compiler['path']) == Path(accepted['path']).with_suffix('.stderr'), 'compiler log is not the selected build log')
    resources = native_resources(inputs.read(spec['compiler_stderr']))
    cli = {}
    for value in args.native_control:
        key, sep, path = value.partition('=')
        require(sep and key not in cli, 'native controls require unique ID=PATH arguments')
        cli[key] = path
    require(set(cli) == set(spec['native_controls']), 'CLI and declared native-control populations differ')
    controls = {key: validate_native(inputs, definition, cli[key], accepted)
                for key, definition in spec['native_controls'].items()}
    for key, definition in spec['native_controls'].items():
        if definition['kind'] == 'coupled':
            require(controls[key]['main_compiler_resources'] == resources['main']
                    and controls[key]['main_device_callee_resources'] == resources['callees'],
                    'native receipt resources differ from actual selected compiler output')
    modes = {v['mode'] for v in spec['native_controls'].values() if v['kind'] == 'coupled'}
    require(REQUIRED_MODES <= modes, 'mandatory kappa0/connected/fixed64 native controls absent')
    require(any(v['kind'] == 'phase2' for v in spec['native_controls'].values()), 'actual Phase2 controls absent')
    checked = {}
    for key, check in spec['checks'].items():
        require(check['control'] in controls, 'check has no actual native owner')
        check_assertions(inputs, check, controls[check['control']])
        checked[key] = {'status': 'SATISFIED', 'control': check['control'], 'artifact': check['artifact'],
                        'kind': check['kind'], 'assertions': check['assertions']}
    production = None
    consumed = dict(build['input_hashes'])
    if args.production_evidence:
        production, prod_build = validate_production(inputs, spec, args.production_evidence, build['input_hashes'][str(MAIN)])
        for path, pin in prod_build['source_after'].items():
            if path in consumed:
                require(consumed[path] == pin['sha256'], 'native and production source epochs differ')
            consumed[path] = pin['sha256']
    states = validate_failures(inputs, spec, checked, consumed)
    validation = {'checks': checked, 'native_compiler_resources': resources,
                  'native_receipts': {k: inputs.item(v['receipt']) for k, v in spec['native_controls'].items()},
                  'production': production, 'production_absence_reason': args.production_absent}
    return inputs, spec, states, validation


def main(argv=None):
    # --check-only also imports the Arrow validator; no import may create cache
    # files as a side effect of this read-only execution mode.
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--verify', type=Path, help='verify an existing flat archive; no original paths required')
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--evidence-sha256')
    parser.add_argument('--accepted-build', type=Path)
    parser.add_argument('--native-control', action='append', default=[])
    production = parser.add_mutually_exclusive_group()
    production.add_argument('--production-evidence', type=Path)
    production.add_argument('--production-absent', help='explicit missing production gate; check-only, never publish')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        if args.verify:
            print(json.dumps(verify_archive(args.verify), sort_keys=True)); return 0
        require(args.evidence and args.evidence_sha256 and args.accepted_build and args.native_control,
                'evidence pin, accepted build and native-control arguments are required')
        require(args.production_evidence or args.production_absent, 'explicit production evidence or absence is required')
        require(not args.production_absent or args.check_only, 'absent production prevents archive publication')
        inputs, spec, states, validation = load_and_validate(args)
        if args.check_only:
            print(json.dumps({'archive_written': False, 'failure_status': states,
                              'production_present': validation['production'] is not None,
                              'production_absence_reason': args.production_absent}, indent=2, allow_nan=False))
            return 2 if args.production_absent else 0
        commit = committed_script(__file__)
        inputs.add(stable_pin(Path(__file__).resolve()), 'assembler_source')
        output = args.output.absolute()
        print(json.dumps(publish(inputs, spec, states, validation, output, output.with_suffix('.zip'), commit), indent=2))
        return 0
    except (AuditError, OSError, KeyError, IndexError, TypeError, ValueError, struct.error,
            zipfile.BadZipFile, subprocess.CalledProcessError) as error:
        print(f'FORENSIC_ARCHIVE_REFUSED: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
