#!/usr/bin/env python3
"""Repeat a retained native binary/input tuple without qualifying current source.

Only the output prefix changes. The original immutable build and dispatch
receipts remain the source lineage; an edited source tree is not represented as
the source of this historical binary.
"""
import datetime
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('dispatch', type=Path)
parser.add_argument('output_prefix', type=Path)
parser.add_argument('--pc-profile', action='store_true', help='One-pass hardware PC sampling of the first positive main launch; never source qualification.')
options = parser.parse_args()
original_path, prefix = options.dispatch, options.output_prefix
original = json.loads(original_path.read_text())
build_path = Path(original['build_receipt']['path'])
assert sha(build_path) == original['build_receipt']['sha256']
build = json.loads(build_path.read_text())
assert build['exit_code'] == 0 and build['sources_unchanged']
assert original['native_execution'] and original['diagnostic_mode'] == 'default_three_epoch'
resources = original['main_compiler_resources']
assert resources['registers'] <= 128
assert all(resources[k] == 0 for k in ('stack', 'cumulative_stack', 'spill_stores', 'spill_loads'))
command = original['command'].copy()
assert len(command) == 5 and sha(command[0]) == build['binary_sha256']
pins = {p: original['input_hashes'][p] for p in command[:4]}
assert all(sha(p) == h for p, h in pins.items())
assert not list(prefix.parent.glob(prefix.name + '*')), 'Repeat output already exists'
command[4] = str(prefix) + '_state'
profile_tool = Path('/usr/local/cuda/bin/ncu')
executed_command = command
if options.pc_profile:
    # This harness executes sixteen deliberate main-kernel refusal controls
    # before the first positive epoch. Do not add counter/SourceCounters sets:
    # a single PC-sampling pass avoids replaying the collective eight-CTA body.
    original_stdout = original_path.with_name(original_path.name.removesuffix('_dispatch.json') + '.stdout')
    assert sha(original_stdout) == original['artifacts'][str(original_stdout)]['sha256']
    entry_lines = [line for line in original_stdout.read_text().splitlines()
        if line.startswith('native_entry_fault ')]
    assert len(entry_lines) == 16 and all('all8_withheld=1' in line for line in entry_lines), 'Positive-launch offset must match the retained harness'
    executed_command = [str(profile_tool), '--config-file', '0', '--rename-kernels', '0',
        '--target-processes', 'application-only', '--kernel-name-base', 'function',
        '--kernel-name', 'nhs_biprism_variational_so3g_v10', '--launch-skip', '16',
        '--launch-count', '1', '--metrics', 'group:smsp__pcsamp_warp_stall_reasons',
        '--warp-sampling-max-passes', '1', '--warp-sampling-interval', '13',
        '--clock-control', 'none', '--cache-control', 'none', '--import-source', '0',
        '--export', str(prefix), *command]
lock_path = f'/tmp/prism-foundation-gpu-{os.getuid()}.lock'
lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

def capture(name, args):
    r = subprocess.run(args, capture_output=True, text=True, timeout=30)
    Path(str(prefix) + '_' + name + '.stdout').write_text(r.stdout)
    Path(str(prefix) + '_' + name + '.stderr').write_text(r.stderr)
    assert r.returncode == 0, (name, r.returncode)
    return r.stdout

def journal(name):
    return [json.loads(s) for s in capture(name, ['journalctl', '-k', '-b', '--grep',
        'NVRM|Xid', '-o', 'json', '--no-pager']).splitlines() if s.startswith('{')]

health_args = ['nvidia-smi', '--query-gpu=name,uuid,driver_version,gpu_recovery_action,memory.used,utilization.gpu', '--format=csv,noheader']
before = journal('journal_before')
assert ', None,' in capture('gpu_before', health_args)
apps = capture('compute_before', ['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader'])
assert all('/usr/libexec/gnome-remote-desktop-daemon' in s for s in apps.splitlines() if s.strip())
receipt = dict(started_utc=now(), command=command, original_dispatch=str(original_path),
    original_dispatch_sha256=sha(original_path), build_receipt=str(build_path),
    build_receipt_sha256=sha(build_path), input_hashes=pins, script_sha256=sha(__file__),
    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
    authorization='Operator authorized native GPU repairs and independent diagnosis.',
    scope='Historical exact binary/input repeat; output prefix is the sole changed argument.',
    main_compiler_resources=resources, active_source_qualification=False,
    full_trajectory_accepted=False, installed_state_mutation=False)
if options.pc_profile:
    receipt.update(executed_command=executed_command,
        scope='Historical exact binary/input PC-sampling profile; one pass requested, no numerical qualification inferred.',
        profiling_tool={'path': str(profile_tool), 'sha256': sha(profile_tool)},
        requested_sampling_passes=1, expected_prior_main_launches=16,
        measured_sampling_passes=None)
receipt_path = Path(str(prefix) + '_execution.json')
receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
try:
    with Path(str(prefix) + '.stdout').open('w') as out, Path(str(prefix) + '.stderr').open('w') as err:
        try:
            r = subprocess.run(executed_command, cwd=build['canonical_source_tree'],
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0'}, stdout=out, stderr=err, timeout=180)
            receipt['native_exit_code'] = r.returncode
        except subprocess.TimeoutExpired:
            receipt.update(native_exit_code=124, timed_out=True)
    after = journal('journal_after')
    old = {r.get('__CURSOR') for r in before}
    receipt.update(ended_utc=now(), native_execution=True,
        new_kernel_fault_records=[r for r in after if r.get('__CURSOR') not in old],
        gpu_recovery_none_after=', None,' in capture('gpu_after', health_args),
        binary_inputs_unchanged=all(sha(p) == h for p, h in pins.items()))
    if options.pc_profile:
        # Keep distinct metric identities: aggregated sampling passes and
        # kernel replay passes need not be reported as the same value.
        report = Path(str(prefix) + '.ncu-rep')
        if report.exists():
            raw = capture('profile_raw', [str(profile_tool), '--import', str(report), '--page', 'raw', '--csv'])
            rows = list(csv.DictReader(raw.splitlines()))
            data = [row for row in rows if row.get('Process ID', '').isdigit()]
            receipt['profile_metric_rows'] = len(data)
            receipt['profile_metrics'] = [{key: row[key] for key in (
                'Kernel Name', 'profiler__replayer_passes', 'smsp__pcsamp_aggregated_passes',
                'smsp__pcsamp_buffer_overflow', 'smsp__pcsamp_dropped_bytes') if key in row} for row in data]
            receipt['requested_one_sampling_pass_verified'] = len(data) == 1 and float(data[0].get('smsp__pcsamp_aggregated_passes', 'nan')) == 1
finally:
    receipt['artifacts'] = {str(p): {'sha256': sha(p), 'bytes': p.stat().st_size}
        for p in prefix.parent.glob(prefix.name + '*') if p.is_file() and p != receipt_path}
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    fcntl.flock(lock_fd, fcntl.LOCK_UN)
    os.close(lock_fd)
print(json.dumps({k: receipt[k] for k in ('native_exit_code', 'binary_inputs_unchanged', 'gpu_recovery_none_after')}))
sys.exit(receipt['native_exit_code'])
