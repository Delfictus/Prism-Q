#!/usr/bin/env python3
"""Record one real, nonprivileged eight-leaf Biprism production invocation.

This wrapper never builds, resets admission/fault state, changes scientific
settings, retries a refused body, or manufactures outputs. The only engine
command is nhs_rt_full biprism-run --config CONFIG. --describe validates the
provided immutable inputs without a GPU query, launch, or output write.

Resource input schema: prism.biprism.production_resource.v1, with started_utc,
ended_utc, build_receipt/ptx/cubin {path,sha256}, and ptxas/cuobjdump records
{command,exit_code,stdout:{path,sha256},stderr:{path,sha256}}. The raw compiler
and SASS reports are parsed here; a supplied PASS boolean is not sufficient.
This offline audit does not replace the engine's actual loaded-kernel gate.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import struct
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
ENGINE = Path(__file__).resolve().parents[1]
MAIN = 'nhs_biprism_variational_so3g_v10'
SCHEMA = 'prism.biprism.production_execution.v1'
RESOURCE_SCHEMA = 'prism.biprism.production_resource.v1'
REQUIRED_MODES = {'default_three_epoch', '--connected-work', '--fixed-domain-16',
                  '--asymmetric-kappa=0', '--asymmetric-kappa=0 --repeat',
                  '--asymmetric-kappa=0.1'}
spec = importlib.util.spec_from_file_location('biprism_config_owner', ENGINE / 'scripts/generate_biprism_v10_run_config.py')
config_owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_owner)
Refusal = config_owner.Refusal
require = config_owner.require
strict_json = config_owner.strict_json


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def identity(path, expected=None):
    """Hash one nonsymlink regular file while checking its open/path identity."""
    path = Path(path)
    require(path.is_absolute(), f'input path must be absolute: {path}')
    fields = ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode), f'not a regular file: {path}')
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
        after, current = os.fstat(stream.fileno()), path.lstat()
        require(all(getattr(before, k) == getattr(after, k) == getattr(current, k) for k in fields),
                f'file changed while reading: {path}')
    result = {'path': str(path), 'sha256': digest.hexdigest(), 'size': before.st_size,
              'mtime_ns': before.st_mtime_ns}
    if expected is not None:
        require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected) is not None,
                f'invalid SHA256: {path}')
        require(result['sha256'] == expected, f'SHA256 differs: {path}')
    return result


def pinned_json(path, expected):
    pin = identity(path, expected)
    value = strict_json(Path(path).read_bytes())
    require(identity(path) == pin, f'JSON changed during read: {path}')
    return value, pin


def require_pin(value, pins):
    require(isinstance(value, dict) and {'path', 'sha256'} <= value.keys(), 'missing file identity')
    pin = identity(value['path'], value['sha256'])
    if 'size' in value:
        require(value['size'] == pin['size'], f'file size differs: {pin["path"]}')
    pins[pin['path']] = pin['sha256']
    return pin


def parse_resources(text):
    entry = f"Compiling entry function '{MAIN}'"
    require(text.count(entry) == 1, 'missing or ambiguous production main resource block')
    block = text.split(entry, 1)[1].split('Compiling entry function ', 1)[0]
    require(re.search(r"for 'sm_120'", block) is not None, 'resource target is not sm_120')
    headers = [x.strip() for x in re.findall(r'Function properties for ([^\n]+)', block)]
    rows = [{'symbol': name.strip(), 'stack': int(stack), 'spill_stores': int(stores), 'spill_loads': int(loads)}
            for name, stack, stores, loads in re.findall(
                r'Function properties for ([^\n]+)\s*\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads', block)]
    require(headers == [r['symbol'] for r in rows] and headers.count(MAIN) == 1,
            'incomplete/ambiguous main or callee resource report')
    require(len(rows) > 1 and len(set(headers)) == len(headers), 'missing or duplicate device callees')
    require(all(r['stack'] == r['spill_stores'] == r['spill_loads'] == 0 for r in rows),
            'main or device callee has stack/spills')
    main_text = block.split(f'Function properties for {MAIN}', 1)[1].split('Function properties for ', 1)[0]
    used = re.findall(r'Used (\d+) registers[^\n]*', main_text)
    shared = re.findall(r'(\d+) bytes smem', main_text)
    cumulative = re.findall(r'(\d+) bytes cumulative stack size', main_text)
    # ptxas omits the static-SMEM field when it is zero. Dynamic launch
    # memory is independently source-bound below, not present in this row.
    require(len(used) == 1 and len(shared) <= 1 and int(used[0]) <= 128,
            'main register/shared memory gate refused')
    require(all(int(x) == 0 for x in cumulative), 'nonzero cumulative main stack')
    return {'registers': int(used[0]), 'static_shared_bytes': int(shared[0]) if shared else 0,
            'cumulative_stack': 0, 'functions': rows}


def launch_shared_bytes(host, device):
    host_values = re.findall(r'pub const BIPRISM_DYNAMIC_SMEM_BYTES:\s*u32\s*=\s*([0-9_]+);', host)
    device_values = re.findall(r'static_assert\s*\(\s*sizeof\(VariationalClusterDsmemV10\)\s*==\s*([0-9]+)', device)
    require(len(host_values) == len(device_values) == 1 and
            int(host_values[0].replace('_', '')) == int(device_values[0]),
            'actual host/device dynamic shared-memory contract differs')
    return int(device_values[0])


def validate_sass(text, resources, elf_symbols):
    headers = list(re.finditer(r'^\s*Function\s*:\s*(\S+)\s*$', text, re.M))
    blocks = {}
    for i, match in enumerate(headers):
        name = match.group(1)
        require(name not in blocks, f'duplicate SASS function: {name}')
        blocks[name] = text[match.end():headers[i + 1].start() if i + 1 < len(headers) else len(text)]
    selected = [row['symbol'] for row in resources['functions']]
    require(MAIN in selected and MAIN in blocks, 'SASS omits the production main')
    # cuobjdump embeds CUDA terminal device functions in the entry's section.
    # They are not separate Function headings. Bind their actual ELF ownership
    # instead of either demanding nonexistent headings or dropping callee proof.
    local_rows = re.findall(r'^STT_FUNC\s+STB_LOCAL\s+STV_DEFAULT\s+\$'+re.escape(MAIN)+r'\$(\S+)\s*$', elf_symbols, re.M)
    require(len(local_rows) == len(set(local_rows)), 'ambiguous duplicate ELF callee')
    require(set(local_rows) == set(selected)-{MAIN}, 'ELF local-callee population differs from compiler resources')
    require(len(re.findall(r'^STT_FUNC\s+STB_GLOBAL\s+STO_ENTRY\s+'+re.escape(MAIN)+r'\s*$', elf_symbols, re.M)) == 1,
            'missing or ambiguous ELF main entry')
    # The entire entry section, including every embedded callee, is inspected.
    for name in {MAIN} | (set(selected) & set(blocks)):
        require(re.search(r'/\*[0-9a-fA-F]+\*/\s+(?:@!?P\d+\s+)?\w', blocks[name]) is not None,
                f'SASS function has no decoded instructions: {name}')
        require(re.search(r'\b(?:LDL|STL)(?:\.|\b)', blocks[name]) is None,
                f'local load/store in actual SASS: {name}')
    return selected


def verify_resource_audit(path, expected, build_pin, build, ptx_pin, pins):
    audit, pin = pinned_json(path, expected)
    require(audit.get('schema') == RESOURCE_SCHEMA, 'wrong resource audit schema')
    require(audit.get('build_receipt') == {k: build_pin[k] for k in ('path', 'sha256')},
            'resource audit belongs to another production build')
    require(audit.get('ptx') == {k: ptx_pin[k] for k in ('path', 'sha256')},
            'resource audit is not the actual governed Biprism PTX')
    start = dt.datetime.fromisoformat(audit['started_utc'].replace('Z', '+00:00'))
    end = dt.datetime.fromisoformat(audit['ended_utc'].replace('Z', '+00:00'))
    build_end = dt.datetime.fromisoformat(build['ended_utc'].replace('Z', '+00:00'))
    require(start.tzinfo is not None and end >= start >= build_end, 'resource audit predates this completed build')
    cubin = require_pin(audit['cubin'], pins)
    require(Path(cubin['path']).read_bytes()[:4] == b'\x7fELF', 'resource cubin is not ELF')
    require(start.timestamp() <= cubin['mtime_ns'] / 1e9 <= end.timestamp() + 2, 'cubin outside audit interval')
    expected_commands = {
        'ptxas': ['/usr/local/cuda/bin/ptxas', '-arch=sm_120', '-m64', '-O3', '--generate-line-info',
                  '--verbose', ptx_pin['path'], '-o', cubin['path']],
        'cuobjdump': ['/usr/local/cuda/bin/cuobjdump', '--dump-sass', cubin['path']],
        'elf_symbols': ['/usr/local/cuda/bin/cuobjdump', '--dump-elf-symbols', cubin['path']],
    }
    logs = {}
    for name, command in expected_commands.items():
        row = audit[name]
        require(row.get('command') == command and type(row.get('exit_code')) is int and row['exit_code'] == 0,
                f'actual {name} command/exit differs')
        streams = []
        for channel in ('stdout', 'stderr'):
            item = require_pin(row[channel], pins)
            streams.append(Path(item['path']).read_text())
        logs[name] = '\n'.join(streams)
    resources = parse_resources(logs['ptxas'])
    sources = build['source_after']
    host_path = str(ENGINE / 'crates/prism-nhs/src/biprism_v10_runtime.rs')
    device_path = str(ENGINE / 'crates/prism-gpu/src/kernels/nhs_biprism_variational_so3g_v10.cuh')
    layout_pins = [require_pin(sources[path], pins) for path in (host_path, device_path)]
    require([p['path'] for p in layout_pins] == [host_path, device_path], 'resource source key/path mismatch')
    resources['dynamic_shared_bytes'] = launch_shared_bytes(Path(host_path).read_text(), Path(device_path).read_text())
    resources['total_shared_bytes'] = resources['static_shared_bytes'] + resources['dynamic_shared_bytes']
    require(resources['total_shared_bytes'] <= 32768, 'actual static plus dynamic SMEM exceeds 32 KiB')
    symbols = validate_sass(logs['cuobjdump'], resources, logs['elf_symbols'])
    pins[pin['path']] = pin['sha256']
    return {'receipt': pin, 'cubin': cubin, 'resources': resources, 'launch_layout_sources': layout_pins, 'sass_functions': symbols,
            'loaded_jit_attributes': 'separate actual engine admission; offline cubin is not loaded-byte attestation'}


def native_control_identity(record):
    """Identify one direct control and its immutable build, before any launch."""
    mode = record.get('diagnostic_mode')
    repeat = record.get('baseline_repeat')
    require(type(repeat) is bool, 'missing native repeat identity')
    require(not repeat or mode == '--asymmetric-kappa=0', 'repeat is not a zero-kappa baseline')
    key = mode + ' --repeat' if repeat and isinstance(mode, str) else mode
    require(key in REQUIRED_MODES, 'unexpected prerequisite mode')
    command = record.get('command')
    require(isinstance(command, list) and all(isinstance(x, str) for x in command),
            'missing direct native command')
    suffix = [] if mode == 'default_three_epoch' else [mode]
    require(len(command) == 5 + len(suffix) and command[5:] == suffix,
            'native control command/mode mismatch')
    require(record.get('memory_instrumentation') is None and
            record.get('launcher_command') == command,
            'instrumented or wrapped control is not a production prerequisite')
    expected_epochs = 16 if mode == '--fixed-domain-16' else (1 if mode.startswith('--asymmetric-') else 3)
    require(record.get('expected_epochs') == expected_epochs, 'native control epoch extent differs')
    build = record.get('build_receipt', {})
    binary_sha = record.get('input_hashes', {}).get(command[0])
    require(isinstance(build.get('path'), str) and isinstance(build.get('sha256'), str) and
            isinstance(binary_sha, str), 'missing immutable native build/binary identity')
    return key, (build['path'], build['sha256'], command[0], binary_sha)


def verify_native_prerequisites(arguments, production_sources, pins):
    found, results, builds = set(), [], set()
    for path, digest in arguments:
        record, pin = pinned_json(path, digest)
        mode, build_identity = native_control_identity(record)
        require(mode not in found, 'repeated prerequisite mode')
        found.add(mode)
        builds.add(build_identity)
        require(len(builds) == 1, 'native controls do not use one immutable build/binary')
        for key in ('native_execution', 'sources_binary_inputs_unchanged', 'gpu_recovery_none_after',
                    'independent_dec_all_epochs_passed', 'coupled_diagnostic_accepted'):
            require(record.get(key) is True, f'failed native prerequisite {key}: {path}')
        require(type(record.get('exit_code')) is int and record['exit_code'] == 0 and
                type(record.get('native_exit_code')) is int and record['native_exit_code'] == 0 and
                record.get('new_kernel_fault_records') == [], 'native prerequisite failed or faulted')
        if mode == '--connected-work':
            require(record.get('independent_lif_work_passed') is True, 'connected work readback did not pass')
            check = record['independent_lif_work_check']
            require(check['exit_code'] == 0 and check['command'].count('--output') == 1,
                    'connected work checker command/exit differs')
            require_pin({'path': check['command'][1], 'sha256': check['checker_sha256']}, pins)
            require_pin({'path': check['command'][check['command'].index('--output') + 1],
                         'sha256': check['output_sha256']}, pins)
        if mode == '--fixed-domain-16':
            require(record.get('expected_epochs') == 16 and record.get('fixed_domain_readback_passed') is True,
                    '64-substep domain readback did not pass')
            require_pin(record['fixed_domain_readback'], pins)
        source_map = record.get('input_hashes', {})
        required = [str(ENGINE / 'crates/prism-gpu/src/kernels' / name) for name in (
            'nhs_biprism_variational_so3g_v10.cuh', 'nhs_biprism_electronic_discrete.cuh',
            'nhs_biprism_dec_pcg.cuh', 'nhs_biprism_potential_precision.cuh')]
        require(all(k in source_map and source_map[k] == production_sources[k]['sha256'] for k in required),
                'native numerical sources differ from production build')
        for file, sha in source_map.items():
            require_pin({'path': file, 'sha256': sha}, pins)
            if file in production_sources:
                require(sha == production_sources[file]['sha256'], 'native/production dependency mismatch')
        build_pin = require_pin(record['build_receipt'], pins)
        build, _ = pinned_json(build_pin['path'], build_pin['sha256'])
        require(not record.get('diagnostic_initial_guess', False)
                and not build.get('diagnostic_initial_guess', False)
                and build.get('autonomous_qualification_eligible', True) is True,
                'externally supplied solver guess cannot qualify an autonomous production run')
        require(build.get('exit_code') == 0 and build.get('sources_unchanged') is True and
                build.get('binary_sha256') == build_identity[3] and
                build.get('command', [])[-2:] == ['-o', build_identity[2]],
                'native build receipt does not bind the executed binary')
        dispatch_pin = require_pin(record['dispatch_receipt'], pins)
        dispatch, _ = pinned_json(dispatch_pin['path'], dispatch_pin['sha256'])
        for key in ('command', 'launcher_command', 'memory_instrumentation', 'input_hashes',
                    'build_receipt', 'diagnostic_mode', 'baseline_repeat', 'expected_epochs',
                    'native_exit_code', 'boot_id'):
            require(dispatch.get(key) == record.get(key), 'dispatch/execution identity differs: ' + key)
        require(record.get('artifacts'), 'native prerequisite has no actual retained artifacts')
        for file, artifact in record['artifacts'].items():
            actual = require_pin({'path': file, 'sha256': artifact['sha256']}, pins)
            require(actual['size'] == artifact['bytes'], 'native artifact extent differs')
        checks = record.get('independent_dec_checks', [])
        require(len(checks) == record['expected_epochs'] and all(c['exit_code'] == 0 for c in checks),
                'native independent epoch coverage differs')
        for check in checks:
            require_pin(check['result'], pins)
        pins[pin['path']] = pin['sha256']
        results.append({'mode': mode, 'receipt': pin})
    require(found == REQUIRED_MODES, 'missing required current-source native prerequisite modes')
    return results


def validate_config_extent(config):
    require(config.get('schema') == 'prism.biprism.native_run.v1', 'wrong actual config schema')
    require(config.get('full_step_budget') == 40000 and config.get('chunk_steps') == 500,
            'production requires unchanged 40000/500/80 extent')
    leaves = config.get('leaves', [])
    require(len(leaves) == 8 and [x['stream_id'] for x in leaves] == list(range(8)), 'wrong physical leaf population')
    require(len({x['seed'] for x in leaves}) == 8, 'physical replica seeds are not distinct')
    require(config.get('hold_terminal_controls') is True and config.get('director_adaptive_dt') is True and
            config.get('governor') == 'armed_adaptive' and config.get('adaptive_dt_bounds') ==
            {'dt_min_ps': 0.0005, 'dt_max_ps': 0.002}, 'physical clock/control policy differs')
    require(config.get('device_ordinal') == 0, 'this serialized runner binds device ordinal zero')
    require(config['modules']['trust_class'] == 'source_owned_dev_hash_bound', 'different module trust profile')
    require(Path(config['modules']['ptx_directory']) == ENGINE / 'target/ptx', 'config does not select canonical PTX')
    require(type(config.get('physical_time_generation')) is int and config['physical_time_generation'] > 0,
            'missing accepted-time generation')


def collect_pins(value, pins):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            require_pin(value, pins)
        for child in value.values():
            collect_pins(child, pins)
    elif isinstance(value, list):
        for child in value:
            collect_pins(child, pins)


def prepare(args):
    config, config_pin = pinned_json(args.config, args.config_sha256)
    validate_config_extent(config)
    output = Path(config['output_root'])
    require(output.is_absolute() and not output.exists() and not output.is_symlink(), 'production output root is not unused')
    build, build_pin = pinned_json(args.build_record, args.build_record_sha256)
    require(build.get('sources_unchanged') is True and build.get('runner_exit_code') == 0 and
            build.get('governed_population_passed') is True, 'actual production build was not completed cleanly')
    pins = {config_pin['path']: config_pin['sha256'], build_pin['path']: build_pin['sha256']}
    require_pin({'path': str(args.gpu_build_output), 'sha256': args.gpu_build_output_sha256}, pins)
    trust = config['modules']['trust_manifest']
    modules, binary = config_owner.verify_build(ENGINE, args.build_record, args.gpu_build_output,
                                                Path(trust['path']), config['modules']['trust_class'])
    require(modules == config['modules'], 'config modules differ from fresh actual build/trust')
    require(build.get('source_before') == build.get('source_after'), 'source identities changed in production build')
    for name, pin in build['source_after'].items():
        require(name == pin['path'], 'build source key/path mismatch')
        require_pin(pin, pins)
    require_pin(build['governed_population'], pins)
    collect_pins(config, pins)
    if config.get('harmonic_source_binding'):
        binding, _ = pinned_json(**{'path': config['harmonic_source_binding']['path'],
                                  'expected': config['harmonic_source_binding']['sha256']})
        collect_pins(binding, pins)
    ptx = identity(ENGINE / 'target/ptx/nhs_biprism_native.ptx', modules['biprism_sha256'])
    manifest = strict_json(config_owner.read_file(trust['path'], trust['sha256']))
    for name, digest in manifest['governed_ptx_membership']['per_kernel_sha256'].items():
        for directory in (ENGINE / 'target/ptx', args.gpu_build_output.parent / 'out/ptx'):
            require_pin({'path': str(directory / name), 'sha256': digest}, pins)
            sidecar = identity(directory / (name + '.sha256'))
            pins[sidecar['path']] = sidecar['sha256']
    resources = verify_resource_audit(args.resource_audit, args.resource_audit_sha256, build_pin, build, ptx, pins)
    prerequisites = verify_native_prerequisites(args.native_prerequisite, build['source_after'], pins)
    archive_output = require_pin({'path': str(args.prismcode_build_output), 'sha256': args.prismcode_build_output_sha256}, pins)
    require(args.prismcode_build_output.is_relative_to(ENGINE / 'target/release/build'), 'PrismCode output is outside actual release build')
    archives = re.findall(r'^cargo:rustc-env=PRISMCODE_CUDA_STATIC_LIB=(.+)$', Path(archive_output['path']).read_text(), re.M)
    require(len(archives) == 1 and Path(archives[0]).is_relative_to(ENGINE / 'target/release/build'),
            'missing/ambiguous actual PrismCode static archive')
    archive = identity(archives[0]); pins[archive['path']] = archive['sha256']
    require('PRISMOS_QNANL_FORCE_KERNEL_PACKET' not in os.environ, 'external forced QNANL packet is selected')
    child_env = {'PRISM4D_PTX_DIR': str(ENGINE / 'target/ptx'), 'PRISM_PTX_DIR': str(ENGINE / 'target/ptx'),
                 'PRISMCODE_CUDA_STATIC_LIB': archive['path'], 'CUDA_VISIBLE_DEVICES': '0'}
    for path in (Path(__file__), ENGINE / 'scripts/generate_biprism_v10_run_config.py'):
        p = identity(path); pins[p['path']] = p['sha256']
    pins[binary['path']] = binary['sha256']
    return {'config_value': config, 'config': config_pin, 'build_receipt': build_pin,
            'trust_manifest': identity(trust['path'], trust['sha256']), 'binary': binary,
            'resource_audit': resources, 'native_prerequisites': prerequisites,
            'prismcode_static_archive': archive, 'input_hashes': pins, 'environment_overrides': child_env,
            'command': [binary['path'], 'biprism-run', '--config', str(args.config)], 'cwd': str(ENGINE)}


def bits32(x):
    return struct.pack('<f', x)


def bits64(x):
    return struct.pack('<d', x)


def validate_time_chunk(raw, index, config, begin, before_time):
    require(len(raw) == 500 * 896 and index.get('schema') == 'biprism.shared_physical_time.v1' and
            index.get('record_bytes') == 896 and index.get('records') == 500, 'shared clock record extent/schema differs')
    w = index['window']
    lo, hi = (struct.unpack('<f', bits32(config['adaptive_dt_bounds'][key]))[0] for key in ('dt_min_ps', 'dt_max_ps'))
    require(w == {'magic': 0x42505354, 'capacity': 500, 'first_body': begin, 'requested_bodies': 500,
                  'first_scheduled_step': begin + 1, 'reserved': [2, struct.unpack('<I', bits32(lo))[0],
                                                               struct.unpack('<I', bits32(hi))[0]]}, 'shared clock window differs')
    require(index['generation'] == config['physical_time_generation'] and
            bits64(index['before_time_ps']) == bits64(before_time), 'shared clock continuity differs')
    terminal_keys = ('cold_hold_steps', 'ramp_steps', 'warm_hold_steps', 'ramp_down_steps', 'cold_return_steps')
    terminals = [sum(leaf['protocol'][key] for key in terminal_keys) for leaf in config['leaves']]
    require(index['terminal_steps'] == terminals and all(0 < t <= 40000 for t in terminals),
            'terminal extents differ from the configured physical protocols')
    clocks = []
    for i in range(500):
        row = raw[i * 896:(i + 1) * 896]
        magic, state, fault, step, last, bodies, mask, policy = struct.unpack_from('<8I', row)
        expected = begin + i + 1
        require((magic, state, fault, step, last, bodies, policy) ==
                (0x42505354, 2, 0, expected, expected, expected, 2), 'shared accepted identity differs')
        native, atomic, minimum, maximum = struct.unpack_from('<4f', row, 32)
        time, generation = struct.unpack_from('<dQ', row, 48)
        requests = struct.unpack_from('<8f', row, 64)
        require(generation == config['physical_time_generation'] and all(math.isfinite(x) and x >= lo for x in requests),
                'shared request/generation invalid')
        chosen = min(min(requests), hi)
        require(bits32(native) == bits32(chosen) and bits32(minimum) == bits32(min(requests)) and
                bits32(maximum) == bits32(max(requests)) and bits32(atomic) == bits32(chosen * 41341.37333517015),
                'shared selected timestep differs')
        before_time += native
        require(math.isfinite(time) and bits64(time) == bits64(before_time), 'shared accepted time recurrence differs')
        controls = struct.unpack_from('<8I', row, 96)
        require(len(terminals) == 8 and controls == tuple(min(expected, t) for t in terminals) and
                mask == sum(1 << leaf for leaf, t in enumerate(terminals) if expected > t), 'terminal-control chronology differs')
        clocks.append((expected, bits64(native), bits64(time)))
    return before_time, clocks


def validate_production_outputs(config, config_sha256):
    """Read-only file identity, extent and accepted-time checks; no scientific proxy.

    Arrow payloads, grids, PathB and the independent scientific work/energy
    consumers retain their own deeper validation in the forensic assembler.
    """
    validate_config_extent(config)
    root = Path(config['output_root'])
    require(root.is_dir() and not root.is_symlink(), 'missing actual production output directory')
    inventory_path, result_path = root / 'artifact_inventory.json', root / 'execution_result.json'
    inventory, inventory_pin = pinned_json(inventory_path, identity(inventory_path)['sha256'])
    terminal, result_pin = pinned_json(result_path, identity(result_path)['sha256'])
    require(inventory['run_id'] == config['run_id'] and inventory['configuration_sha256'] == config_sha256,
            'production inventory run/config identity differs')
    members = {}
    for item in inventory['files']:
        rel = Path(item['path'])
        require(not rel.is_absolute() and rel.parts and all(p not in ('.', '..') for p in rel.parts) and
                str(rel) not in members, 'invalid/duplicate output inventory member')
        path = root / rel
        require(all(not p.is_symlink() for p in (path, *path.parents) if p == root or root in p.parents),
                'symlink in output member path')
        member = identity(path, item['sha256'])
        require(member['size'] == item['bytes'], 'output inventory extent differs')
        members[str(rel)] = member
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p != inventory_path}
    require(actual == set(members), 'unlisted/missing production artifacts')
    require(members['run_configuration.json']['sha256'] == config_sha256, 'executed config bytes differ')
    require(terminal.get('accepted_full_run') is True and terminal.get('status') == 'IMPLEMENTATION_VERIFIED' and
            terminal.get('requested_physical_steps_per_leaf') == 40000 and terminal.get('completed_physical_steps_per_leaf') == 40000 and
            terminal.get('total_leaf_physical_steps') == 320000 and terminal.get('graph_launches') == 80 and
            terminal.get('retained_chunks') == 80, 'engine did not accept the complete production extent')
    analysis = terminal['analysis']
    require(len(analysis) == 8 and [x['leaf_id'] for x in analysis] == list(range(8)) and
            all(x['stream_id'] == config['leaves'][i]['stream_id'] and x['replica_seed'] == config['leaves'][i]['seed']
                for i, x in enumerate(analysis)), 'terminal physical leaf/seed population differs')
    expected_dirs = {f'chunk_{begin:08}' for begin in range(0, 40000, 500)}
    require({p.name for p in root.glob('chunk_*')} == expected_dirs, 'production chunk population differs')
    time = 0.0
    for begin in range(0, 40000, 500):
        directory = root / f'chunk_{begin:08}'
        outcome = strict_json((directory / 'execution.json').read_bytes())
        require(outcome.get('accepted') is True and outcome.get('refusal') is None and
                outcome.get('completed_before') == begin and outcome.get('requested_steps') == 500,
                'unaccepted/mismatched physical chunk')
        index = strict_json((directory / 'shared_physical_time.index.json').read_bytes())
        time, clocks = validate_time_chunk((directory / 'shared_physical_time.896b').read_bytes(), index, config, begin, time)
        for leaf in range(8):
            native = strict_json((directory / f'leaf{leaf:02}.physical_time.json').read_bytes())
            require(len(native) == 500 and all((r['step'], bits64(r['applied_dt_ps']), bits64(r['cumulative_simulated_time_ps'])) == expected
                                             for r, expected in zip(native, clocks)), 'native/shared accepted clocks differ')
    return {'artifact_inventory': inventory_pin, 'execution_result': result_pin,
            'accepted_steps_per_leaf': 40000, 'physical_leaf_count': 8, 'retained_chunks': 80,
            'cumulative_accepted_time_ps': time, 'hashed_artifacts': len(members),
            'scope': 'actual inventory bytes, engine accepted extent, and all native/shared clock rows; deeper scientific/Arrow/grid/PathB validation remains separate'}


def write_json(path, value, exclusive=False):
    raw = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    if exclusive:
        with path.open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    else:
        fd, temp = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            os.replace(temp, path)
        finally:
            if os.path.exists(temp): os.unlink(temp)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def health_rows(text):
    rows = list(csv.reader(text.splitlines(), skipinitialspace=True))
    require(len(rows) == 1 and len(rows[0]) == 6, 'ambiguous GPU health population')
    require(rows[0][3].strip() == 'None', 'GPU recovery action is not None')
    return rows[0]


def gpu_lock_path():
    return f'/tmp/prism-foundation-gpu-{os.getuid()}.lock'


def run(args, prepared):
    receipt_path = args.receipt
    require(receipt_path.is_absolute() and receipt_path.parent.is_dir() and receipt_path.suffix == '.json', 'invalid receipt destination')
    require(not receipt_path.exists() and not receipt_path.is_symlink(), 'refusing to overwrite an execution receipt')
    root = Path(prepared['config_value']['output_root'])
    require(not receipt_path.is_relative_to(root), 'external recorder must be outside engine output tree')
    stem = receipt_path.with_suffix('')
    paths = {name: Path(str(stem) + suffix) for name, suffix in (('stdout', '.stdout.log'), ('stderr', '.stderr.log'))}
    require(not any(p.exists() or p.is_symlink() for p in paths.values()), 'refusing to overwrite streamed logs')
    record = {k: v for k, v in prepared.items() if k != 'config_value'}
    record.update(schema=SCHEMA, started_utc=now(), native_execution=False, native_exit_code=None, exit_code=None,
                  sources_binary_inputs_unchanged=False, new_kernel_fault_records=None, gpu_recovery_none_after=False,
                  boot_unchanged=False, output_extent_validated=False, independent_full_consumer_validation=None,
                  installed_state_mutation=False, production_admission_reset=False, historical_faults_preserved=True,
                  privileged_lease_claim=False, cwd=str(ENGINE), output_root=str(root), errors=[])
    write_json(receipt_path, record, exclusive=True)
    lock_fd, process, before = None, None, None

    def capture(name, command):
        out, err = Path(str(stem) + '.' + name + '.stdout'), Path(str(stem) + '.' + name + '.stderr')
        require(not out.exists() and not err.exists(), 'refusing to overwrite health evidence')
        try:
            completed = subprocess.run(command, capture_output=True, timeout=30, check=False)
            code, so, se = completed.returncode, completed.stdout, completed.stderr
        except subprocess.TimeoutExpired as exc:
            code, so, se = 124, exc.stdout or b'', exc.stderr or b''
        for path, data in ((out, so), (err, se)):
            with path.open('xb') as stream:
                stream.write(data); stream.flush(); os.fsync(stream.fileno())
        record.setdefault('health_commands', []).append({'command': command, 'exit_code': code,
                                                       'stdout': identity(out), 'stderr': identity(err)})
        require(code == 0, f'health command failed: {name}')
        return so.decode(errors='strict')

    def journal(name):
        text = capture(name, ['journalctl', '-k', '-b', '--grep', 'NVRM|Xid', '-o', 'json', '--no-pager'])
        rows = [strict_json(line) for line in text.splitlines() if line.startswith('{')]
        require(all(isinstance(r.get('__CURSOR'), str) for r in rows), 'kernel journal has no stable cursor')
        return rows

    health_command = ['nvidia-smi', '--query-gpu=name,uuid,driver_version,gpu_recovery_action,memory.used,utilization.gpu', '--format=csv,noheader']
    try:
        lock_path = gpu_lock_path()
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record['cooperative_gpu_lock'] = lock_path
        record['boot_id'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        before = journal('journal_before')
        record['historical_kernel_records_retained'] = len(before)
        record['gpu_before'] = health_rows(capture('gpu_before', health_command))
        apps = capture('compute_before', ['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader'])
        rows = [r for r in csv.reader(apps.splitlines(), skipinitialspace=True) if r]
        require(all(len(row) == 3 and row[1].strip() == '/usr/libexec/gnome-remote-desktop-daemon' for row in rows),
                'another GPU compute process is present')
        require(not root.exists(), 'output root appeared before launch')
        for path, digest in prepared['input_hashes'].items(): identity(path, digest)
        config_owner.require_unchanged_executable(ENGINE, prepared['binary'], 'before production launch')
        env = os.environ.copy(); env.update(prepared['environment_overrides'])
        with paths['stdout'].open('xb') as stdout, paths['stderr'].open('xb') as stderr:
            process = subprocess.Popen(prepared['command'], cwd=ENGINE, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            record.update(native_execution=True, pid=process.pid, launched_utc=now())
            write_json(receipt_path, record)
            print(json.dumps({'pid': process.pid, 'receipt': str(receipt_path), 'stdout': str(paths['stdout']), 'stderr': str(paths['stderr'])}), flush=True)
            try:
                record['native_exit_code'] = process.wait()
            except KeyboardInterrupt:
                os.killpg(process.pid, signal.SIGINT)
                record['native_exit_code'] = process.wait()
                record['errors'].append('operator interrupted the owned engine process')
            stdout.flush(); stderr.flush(); os.fsync(stdout.fileno()); os.fsync(stderr.fileno())
    except Exception as exc:
        record['errors'].append(f'{type(exc).__name__}: {exc}')
    finally:
        # A failure never becomes an admission reset, a retry, or a lost exit.
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            record['native_exit_code'] = process.wait()
        if before is not None:
            try:
                after = journal('journal_after')
                old = {r['__CURSOR'] for r in before}
                require(old <= {r['__CURSOR'] for r in after}, 'kernel journal coverage was lost during execution')
                record['new_kernel_fault_records'] = [r for r in after if r['__CURSOR'] not in old]
                record['boot_unchanged'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip() == record['boot_id']
                record['gpu_after'] = health_rows(capture('gpu_after', health_command))
                record['gpu_recovery_none_after'] = True
                require(record['gpu_after'][1:3] == record['gpu_before'][1:3], 'GPU identity/driver changed')
            except Exception as exc:
                record['errors'].append(f'post-run health: {exc}')
        try:
            for path, digest in prepared['input_hashes'].items(): identity(path, digest)
            config_owner.require_unchanged_executable(ENGINE, prepared['binary'], 'during production execution')
            record['sources_binary_inputs_unchanged'] = True
        except Exception as exc:
            record['errors'].append(f'post-run identity: {exc}')
        for key, path in paths.items():
            if path.is_file(): record[key] = identity(path)
        for key, name in (('artifact_inventory', 'artifact_inventory.json'), ('execution_result', 'execution_result.json')):
            path = root / name
            if path.is_file():
                try: record[key] = identity(path)
                except Exception as exc: record['errors'].append(f'output pin: {exc}')
        if record['native_exit_code'] == 0:
            try:
                record['output_validation'] = validate_production_outputs(prepared['config_value'], prepared['config']['sha256'])
                record['output_extent_validated'] = True
            except Exception as exc:
                record['errors'].append(f'actual output validation: {exc}')
        clean = (record['native_execution'] and record['native_exit_code'] == 0 and not record['errors']
                 and record['sources_binary_inputs_unchanged'] and record['boot_unchanged']
                 and record['gpu_recovery_none_after'] and record['new_kernel_fault_records'] == []
                 and record['output_extent_validated'])
        record['engine_accepted_full_run'] = bool(clean)
        record['scope'] = 'actual direct engine execution and bounded output chronology; independent complete scientific/consumer qualification is separate'
        record['exit_code'] = 0 if clean else (record['native_exit_code'] if record['native_exit_code'] not in (None, 0) else 2)
        record['ended_utc'] = now()
        write_json(receipt_path, record)
        if lock_fd is not None:
            fcntl.flock(lock_fd, fcntl.LOCK_UN); os.close(lock_fd)
    print(json.dumps({'receipt': str(receipt_path), 'native_exit_code': record['native_exit_code'], 'exit_code': record['exit_code']}), flush=True)
    return record['exit_code'] if record['exit_code'] >= 0 else 128 - record['exit_code']


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'build-record', 'gpu-build-output', 'resource-audit', 'prismcode-build-output'):
        p.add_argument('--' + name, type=Path, required=True)
        p.add_argument('--' + name + '-sha256', required=True)
    p.add_argument('--native-prerequisite', nargs=2, action='append', default=[], metavar=('EXECUTION_JSON', 'SHA256'))
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--describe', action='store_true')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        prepared = prepare(args)
        if args.describe:
            print(json.dumps({k: v for k, v in prepared.items() if k not in ('config_value', 'input_hashes')}, indent=2))
            return 0
        return run(args, prepared)
    except (Refusal, OSError, KeyError, ValueError) as exc:
        print(f'production recorder refused before launch: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
