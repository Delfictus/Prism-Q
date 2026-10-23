"""CPU admission-rejection controls; constructed files are not GPU evidence."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'run_biprism_v10_production.py'
SPEC = importlib.util.spec_from_file_location('production_prerequisites_under_test', SOURCE)
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


class NativePrerequisites(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='biprism_prerequisite_cpu_')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.binary = self.file('binary', b'CPU fixture, not an executable')
        self.build = self.document('build', {
            'exit_code': 0, 'sources_unchanged': True,
            'binary_sha256': self.binary['sha256'], 'command': ['nvcc', '-o', self.binary['path']]})
        self.production = {}
        for name in ('nhs_biprism_variational_so3g_v10.cuh', 'nhs_biprism_electronic_discrete.cuh',
                     'nhs_biprism_dec_pcg.cuh', 'nhs_biprism_potential_precision.cuh'):
            pin = R.identity(R.ENGINE / 'crates/prism-gpu/src/kernels' / name)
            self.production[pin['path']] = pin

    def file(self, name, raw):
        path = self.root / name
        path.write_bytes(raw)
        return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}

    def document(self, name, value):
        return self.file(name, json.dumps(value, sort_keys=True).encode())

    def records(self):
        result = []
        for index, key in enumerate(sorted(R.REQUIRED_MODES)):
            repeat = key.endswith(' --repeat')
            mode = key.removesuffix(' --repeat')
            epochs = 16 if mode == '--fixed-domain-16' else (1 if mode.startswith('--asymmetric-') else 3)
            command = [self.binary['path'], '/case', '/selection', '/mass', '/output']
            if mode != 'default_three_epoch': command.append(mode)
            artifact = self.file(f'artifact{index}', b'CPU fixture')
            record = dict(diagnostic_mode=mode, baseline_repeat=repeat, expected_epochs=epochs,
                          command=command, launcher_command=command.copy(), memory_instrumentation=None,
                          build_receipt=self.build, boot_id='cpu-test', native_execution=True,
                          sources_binary_inputs_unchanged=True, gpu_recovery_none_after=True,
                          independent_dec_all_epochs_passed=True, coupled_diagnostic_accepted=True,
                          exit_code=0, native_exit_code=0, new_kernel_fault_records=[],
                          input_hashes={p: v['sha256'] for p, v in self.production.items()},
                          artifacts={artifact['path']: {'sha256': artifact['sha256'], 'bytes': 11}},
                          independent_dec_checks=[{'exit_code': 0, 'result': artifact} for _ in range(epochs)])
            record['input_hashes'][self.binary['path']] = self.binary['sha256']
            if mode == '--fixed-domain-16':
                record.update(fixed_domain_readback_passed=True, fixed_domain_readback=artifact)
            if mode == '--connected-work':
                record.update(independent_lif_work_passed=True, independent_lif_work_check={
                    'exit_code': 0, 'command': ['python', artifact['path'], '--output', artifact['path']],
                    'checker_sha256': artifact['sha256'], 'output_sha256': artifact['sha256']})
            record['dispatch_receipt'] = self.document(f'dispatch{index}', record)
            result.append(record)
        return result

    def verify(self, records):
        args = []
        for index, record in enumerate(records):
            pin = self.document(f'record{index}', record)
            args.append((pin['path'], pin['sha256']))
        return R.verify_native_prerequisites(args, self.production, {})

    def test_complete_six_mode_matrix_and_zero_repeat_identity(self):
        self.assertEqual({r['mode'] for r in self.verify(self.records())}, R.REQUIRED_MODES)

    def test_every_mode_is_required(self):
        records = self.records()
        for i in range(6):
            with self.subTest(i=i), self.assertRaisesRegex(R.Refusal, 'missing required'):
                self.verify(records[:i] + records[i+1:])
        with self.assertRaisesRegex(R.Refusal, 'repeated prerequisite'):
            self.verify(records + [records[0]])

    def test_instrumented_or_wrapped_receipt_is_refused(self):
        for change in ({'memory_instrumentation': 'compute-sanitizer memcheck'},
                       {'launcher_command': ['compute-sanitizer']}, {'launcher_command': None}):
            records = self.records(); records[0].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(R.Refusal, 'instrumented or wrapped'):
                self.verify(records)

    def test_same_source_with_different_build_is_refused(self):
        records = self.records()
        alternate = self.document('different-build', {'sources_unchanged': True})
        records[-1]['build_receipt'] = alternate
        with self.assertRaisesRegex(R.Refusal, 'one immutable build'):
            self.verify(records)

    def test_mode_extent_and_dispatch_mutations_are_refused(self):
        for field, value, reason in (('expected_epochs', 2, 'epoch extent'),
                                    ('baseline_repeat', True, 'repeat is not'),
                                    ('boot_id', 'different', 'dispatch/execution')):
            records = self.records()
            row = next(x for x in records if x['diagnostic_mode'] == '--connected-work')
            row[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(R.Refusal, reason):
                self.verify(records)

    def test_receipt_cannot_relabel_failed_build_as_success(self):
        records = self.records()
        failed = self.document('failed-build', {'exit_code': 1, 'sources_unchanged': True})
        for row in records: row['build_receipt'] = copy.deepcopy(failed)
        with self.assertRaisesRegex(R.Refusal, 'does not bind'):
            self.verify(records)

    def test_forensic_tangent_stop_cannot_qualify_production(self):
        records = self.records()
        build = json.loads(Path(self.build['path']).read_text())
        build.update(diagnostic_tangent_capture=True,
                     autonomous_qualification_eligible=False)
        capture = self.document('tangent-capture-build', build)
        for row in records:
            row['build_receipt'] = copy.deepcopy(capture)
        # Even fabricated all-green native rows cannot promote a build that
        # deliberately stops before completing the original workload.
        with self.assertRaisesRegex(R.Refusal, 'cannot qualify'):
            self.verify(records)


if __name__ == '__main__': unittest.main()
