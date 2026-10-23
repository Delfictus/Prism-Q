#!/usr/bin/env python3
"""Run a separately named, non-qualifying native initial-guess experiment.

Uses the same GPU ownership, health, fault-history and compiler checks as the
ordinary diagnostic. A supplied CPU guess never qualifies an autonomous run.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROLE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROLE/'scripts'))
import run_biprism_v10_production as production


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--guess', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert a.output.name.startswith('failure_2_initial_guess_diagnostic_')
    assert not list(a.output.parent.glob(a.output.name+'*')), 'immutable experiment prefix'
    build, baseline = (json.loads(path.read_text()) for path in (a.build, a.baseline))
    assert build['exit_code'] == 0 and build['sources_unchanged']
    assert build['diagnostic_initial_guess'] and not build['autonomous_qualification_eligible']
    binary = Path(build['command'][-1])
    assert sha(binary) == build['binary_sha256']
    expected = dict(build['input_hashes'])
    assert all(sha(path) == digest for path, digest in expected.items()), 'bound source changed'
    resources = production.parse_resources(a.build.with_suffix('.stderr').read_text())
    def capture(label, command):
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        for stream in ('stdout', 'stderr'):
            Path(str(a.output)+'_'+label+'.'+stream).write_text(getattr(result, stream))
        return result
    sass = capture('sass', ['/usr/local/cuda/bin/cuobjdump', '--dump-sass', str(binary)])
    symbols = capture('symbols', ['/usr/local/cuda/bin/cuobjdump', '--dump-elf-symbols', str(binary)])
    assert sass.returncode == symbols.returncode == 0
    production.validate_sass(sass.stdout, resources, symbols.stdout)
    # Terminal callees are embedded in the entry's SASS section. Their
    # identities are in ELF symbols, already reconciled with ptxas above.
    assert any('bp_v10_diagnostic_initial_guess' in row['symbol']
               for row in resources['functions'])
    shared = production.launch_shared_bytes(
        (ROLE/'crates/prism-nhs/src/biprism_v10_runtime.rs').read_text(),
        (ROLE/'crates/prism-gpu/src/kernels/nhs_biprism_variational_so3g_v10.cuh').read_text())
    assert shared == 31616
    lock_path = f'/tmp/prism-foundation-gpu-{os.getuid()}.lock'
    lock_fd = os.open(lock_path, os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW, 0o600)
    fcntl.flock(lock_fd, fcntl.LOCK_EX|fcntl.LOCK_NB)
    def journal(label):
        r = capture(label, ['journalctl', '-k', '-b', '--grep', 'NVRM|Xid', '-o', 'json', '--no-pager'])
        assert r.returncode == 0
        return [json.loads(s) for s in r.stdout.splitlines() if s.startswith('{')]
    before = journal('journal_before')
    health_command = ['nvidia-smi', '--query-gpu=name,uuid,driver_version,gpu_recovery_action,memory.used,utilization.gpu', '--format=csv,noheader']
    health = capture('gpu_before', health_command)
    assert health.returncode == 0 and ', None,' in health.stdout
    apps = capture('compute_before', ['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader'])
    assert apps.returncode == 0 and all('/usr/libexec/gnome-remote-desktop-daemon' in line
                                       for line in apps.stdout.splitlines() if line.strip())
    command = baseline['command'].copy()
    for path in command[1:4]:
        assert sha(path) == baseline['input_hashes'][path]
        expected[path] = sha(path)
    command[0], command[4] = str(binary), str(a.output)+'_state'
    expected.update({str(path): sha(path) for path in (binary, a.guess, a.build, a.baseline, Path(__file__))})
    receipt_path = Path(str(a.output)+'_execution.json')
    now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    receipt = dict(started_utc=now(), command=command, input_hashes=expected,
                   build_receipt={'path':str(a.build), 'sha256':sha(a.build)},
                   expected_epochs=baseline['expected_epochs'],
                   main_compiler_resources=dict(resources['functions'][0],
                       cumulative_stack=resources['cumulative_stack']),
                   native_execution=True, autonomous_qualification_eligible=False,
                   coupled_diagnostic_accepted=False, full_trajectory_accepted=False,
                   production_admission_reset=False, historical_faults_preserved=True,
                   installed_state_mutation=False, diagnostic_initial_guess=True,
                   boundary='CPU-supplied nearby-state private initial guess; all actual native C1-C4 gates unchanged. No physical starting operand is imported. Not an autonomous workload qualification.',
                   authorization='Standing operator authorization for native GPU gap-closure diagnostics.',
                   boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                   cooperative_gpu_lock=lock_path, compiler_resources=resources,
                   shared_bytes=shared, environment={'CUDA_VISIBLE_DEVICES': '0',
                   'BIPRISM_DIAGNOSTIC_INITIAL_GUESS': str(a.guess)})
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
    env = os.environ.copy(); env.update(receipt['environment'])
    with Path(str(a.output)+'.stdout').open('x') as out, Path(str(a.output)+'.stderr').open('x') as err:
        try:
            receipt['exit_code'] = subprocess.run(command, cwd=ROLE, env=env,
                stdout=out, stderr=err, timeout=180).returncode
        except subprocess.TimeoutExpired:
            receipt.update(exit_code=124, timed_out=True)
    after = journal('journal_after'); old = {record.get('__CURSOR') for record in before}
    health = capture('gpu_after', health_command)
    receipt.update(ended_utc=now(), native_exit_code=receipt['exit_code'],
        new_kernel_fault_records=[record for record in after if record.get('__CURSOR') not in old],
        gpu_recovery_none_after=health.returncode == 0 and ', None,' in health.stdout,
        sources_binary_inputs_unchanged=all(sha(path) == digest for path, digest in expected.items()),
        artifacts={str(path): dict(sha256=sha(path), bytes=path.stat().st_size)
                   for path in a.output.parent.glob(a.output.name+'*') if path.is_file() and path != receipt_path})
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(exit_code=receipt['exit_code'],
        new_kernel_fault_records=len(receipt['new_kernel_fault_records']),
        receipt=str(receipt_path), autonomous_qualification=False)))
    return receipt['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
