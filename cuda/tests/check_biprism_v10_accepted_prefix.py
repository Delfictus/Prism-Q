#!/usr/bin/env python3
"""Bind an accepted epoch's unchanged DEC oracle to its actual native dispatch.

A later controlled refusal (exit4) does not erase an earlier complete accepted
epoch. This wrapper preserves that refusal and never labels the full trajectory
accepted. It performs no native execution and does not reinterpret failed rows.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import uuid

import check_biprism_v10_dec_readback as DEC


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def dispatch_domain(d, epoch):
    require(d.get('native_execution') is True, 'actual native execution required')
    require(type(d.get('native_exit_code')) is int and d['native_exit_code'] in (0, 4),
            'only success or retained controlled scientific refusal is supported')
    require(type(d.get('exit_code')) is int and d['exit_code'] == d['native_exit_code'],
            'dispatch/native exit disagreement')
    if d['native_exit_code'] == 4:
        require(d.get('full_trajectory_accepted') is False, 'controlled refusal cannot claim full trajectory acceptance')
    require(d.get('sources_binary_inputs_unchanged') is True, 'dispatch source/input mutation')
    require(d.get('gpu_recovery_none_after') is True, 'post-run GPU recovery')
    require(d.get('new_kernel_fault_records') == [], 'new kernel fault records')
    require(type(d.get('expected_epochs')) is int and type(epoch) is int
            and 1 <= epoch <= d['expected_epochs'], 'epoch outside dispatch')
    require(isinstance(d.get('boot_id'), str), 'missing actual boot identity')
    uuid.UUID(d['boot_id'])
    resources = d.get('main_compiler_resources', {})
    for key in ('stack', 'cumulative_stack', 'spill_stores', 'spill_loads'):
        require(type(resources.get(key)) is int and resources[key] == 0, 'resource gate: ' + key)
    command = d.get('command', [])
    require(isinstance(command, list) and len(command) >= 5 and all(isinstance(x, str) for x in command),
            'complete native command required')


def check(execution, epoch, main_source=None, dec_source=None, ordered_source=None, potential_source=None,
          ordered_control_source=None):
    execution = execution.resolve()
    execution_hash = sha(execution)
    d = json.loads(execution.read_text())
    dispatch_domain(d, epoch)
    pin = d.get('build_receipt', {})
    build_path = Path(pin['path'])
    require(sha(build_path) == pin.get('sha256'), 'build receipt hash mismatch')
    build = json.loads(build_path.read_text())
    require(build.get('exit_code') == 0 and build.get('sources_unchanged') is True,
            'actual stable successful build required')
    command = d['command']
    binary = Path(command[0])
    binary_hash = sha(binary)
    require(binary_hash == build.get('binary_sha256') == d.get('input_hashes', {}).get(str(binary)),
            'native binary/dispatch/build mismatch')
    compiled_main = DEC.KERNELS/'nhs_biprism_variational_so3g_v10.cuh'
    compiled_dec = DEC.KERNELS/'nhs_biprism_dec_pcg.cuh'
    compiled_ordered = DEC.KERNELS/'nhs_biprism_ordered_edges.cuh'
    compiled_potential = DEC.KERNELS/'nhs_biprism_potential_precision.cuh'
    compiled_control = DEC.ROOT/'biprism_ordered_edges_native_control.hpp'
    args = SimpleNamespace(prefix=Path(command[4]), epoch=epoch, entry_substeps=None,
                           bp_case=Path(command[1]), main_source=Path(main_source).resolve() if main_source else compiled_main,
                           dec_source=Path(dec_source).resolve() if dec_source else compiled_dec,
                           ordered_source=Path(ordered_source).resolve() if ordered_source else compiled_ordered,
                           potential_source=Path(potential_source).resolve() if potential_source else compiled_potential,
                           ordered_control_source=Path(ordered_control_source).resolve() if ordered_control_source else compiled_control,
                           electronic_source=DEC.KERNELS/'nhs_biprism_electronic_discrete.cuh',
                           execution_receipt=execution)
    result = DEC.check_epoch(args)
    require(result['status'] == 'PASS', 'requested epoch did not pass the complete DEC/identity oracle')
    # Native source identities and every primitive read by the oracle must
    # occur in the actual build/dispatch population, not merely a nearby file.
    for item in result['source_bindings']:
        if Path(item['path']).resolve() == Path(DEC.__file__).resolve():
            continue  # Independent reader code is retained under its own hash.
        source_paths = {args.main_source: compiled_main, args.dec_source: compiled_dec,
                        args.ordered_source: compiled_ordered, args.potential_source: compiled_potential,
                        args.ordered_control_source: compiled_control}
        compiled_path = str(source_paths.get(Path(item['path']), Path(item['path'])))
        require(build.get('input_hashes', {}).get(compiled_path) == item['sha256']
                and d.get('input_hashes', {}).get(compiled_path) == item['sha256'],
                'native source is not build/dispatch-bound: ' + item['path'])
    for item in result['input_bindings']:
        path = item['path']
        recorded = d.get('artifacts', {}).get(path)
        if recorded is not None:
            require(recorded == {'sha256': item['sha256'], 'bytes': item['bytes']},
                    'native output identity mismatch: ' + path)
        else:
            require(d.get('input_hashes', {}).get(path) == item['sha256'],
                    'physical input identity mismatch: ' + path)
    for item in result['source_bindings'] + result['input_bindings']:
        require(sha(item['path']) == item['sha256'], 'source/readback changed while checking')
    require(sha(execution) == execution_hash and sha(build_path) == pin['sha256']
            and sha(binary) == binary_hash, 'dispatch/build/binary changed while checking')
    lines = [
        'SCOPE=accepted epoch DEC/identity prefix only; no whole-trajectory or full C4 inference',
        'PREFIX_DEC_STATUS=PASS', 'FULL_TRAJECTORY_ACCEPTED=false',
        f'ACTUAL_NATIVE_EXIT_CODE={d["native_exit_code"]}',
        f'EPOCH={epoch} ENTRY_SUBSTEPS={result["entry_substeps"]} LEAVES=8',
        f'EXECUTION={execution} SHA256={execution_hash}',
        f'BUILD={build_path} SHA256={pin["sha256"]}',
        f'BINARY={binary} SHA256={binary_hash}',
        f'BOOT_ID={d["boot_id"]}', f'ORACLE_SHA256={sha(DEC.__file__)}',
        f'WRAPPER_SHA256={sha(__file__)}',
        f'EXECUTED_MAIN_CANONICAL_PATH={compiled_main}',
        f'READ_MAIN_SOURCE={args.main_source} SHA256={sha(args.main_source)}',
        f'READ_DEC_SOURCE={args.dec_source} SHA256={sha(args.dec_source)}',
        'SOURCE_SNAPSHOT_IS_IDENTITY_REPLAY_NOT_A_NATIVE_BUILD_OVERLAY=true']
    for leaf in result['leaves']:
        lines.append(f'LEAF={leaf["leaf"]} TRUE_DEC_L2={leaf["true_residual_l2"]:.17g}'
                     ' ACCEPTED_COUNTER_FOOTER=PASS FULL_COORDINATE_WEIGHT_BITS=PASS FIXED_HALO=PASS')
    for item in result['source_bindings'] + result['input_bindings']:
        lines.append(f'BOUND_FILE={item["path"]} SHA256={item["sha256"]} BYTES={item["bytes"]}')
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execution', required=True, type=Path)
    p.add_argument('--epoch', required=True, type=int)
    p.add_argument('--main-source', type=Path,
                   help='Optional retained exact main bytes; hash must match the original compiled canonical path')
    p.add_argument('--dec-source', type=Path,
                   help='Optional retained exact DEC bytes; hash must match the original compiled canonical path')
    p.add_argument('--ordered-source', type=Path,
                   help='Optional retained exact ordered-traversal bytes for an indexed DEC build')
    p.add_argument('--potential-source', type=Path,
                   help='Optional retained exact full-potential owner bytes for a precise DEC build')
    p.add_argument('--ordered-control-source', type=Path,
                   help='Optional retained exact ODP2 native-control bytes for a precise DEC build')
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    text = check(args.execution, args.epoch, args.main_source, args.dec_source, args.ordered_source,
                 args.potential_source, args.ordered_control_source)
    args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
