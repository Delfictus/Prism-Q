"""CPU input-contract checks; no production configuration or GPU execution.

The authority checks read the retained actual KRAS records. Numeric values in
the scalar-validation controls are test inputs only and are never serialized
as run settings. A passing test does not establish the pending production
build/module or native-run gates.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

def _env_root(name):
    """Absolute root from the environment; there is no compiled-in host default.
    When unset, a non-existent sentinel naming the variable is returned so every
    read of host data fails closed."""
    value = os.environ.get(name, '')
    return Path(value) if os.path.isabs(value) else Path('/nonexistent') / (name + '_UNSET')




SCRIPT = Path(__file__).resolve().parents[1] / 'generate_biprism_v10_run_config.py'
SPEC = importlib.util.spec_from_file_location('biprism_run_config_generator', SCRIPT)
GEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GEN)


class ActualAuthorityChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.authority = GEN.load_harmonic_binding(GEN.load_authorities(),
            _env_root('PRISM_BIPRISM_RESOLUTION_ROOT') / 'failure_7_harmonic_source_binding.json',
            'febe0f98dc250f6e86fde953065ff66d1b51cbdb2329a40f20b4345d06f3a15d')

    def test_all_retained_authorities_bind(self):
        self.assertEqual(len(self.authority['baseline_pins']), 27)
        self.assertEqual(self.authority['atom_count'], 2685)
        self.assertEqual(self.authority['topology']['sha256'],
                         'bc19e5f455552b6619b34e23fff55425c8c60f040168e89f21bbd028d527ff19')

    def test_original_durations_and_all_five_wavelengths(self):
        fields = ['cold_hold_steps', 'ramp_steps', 'warm_hold_steps', 'ramp_down_steps', 'cold_return_steps']
        self.assertEqual([sum(p[k] for k in fields) for p in self.authority['protocols']],
                         [35000, 35000, 40000, 40000, 37000, 37000, 34000, 34000])
        for protocol in self.authority['protocols']:
            self.assertEqual(protocol['scan_wavelengths'], [280.0, 274.0, 258.0, 254.0, 211.0])
            self.assertEqual(protocol['current_step'], 0)
            self.assertEqual(protocol['stepped_holds'], [])

    def test_source_grid_dt_asc_and_separate_budgets(self):
        self.assertEqual([self.authority['grid_dimension'], self.authority['grid_spacing_angstrom']], [128, .75])
        self.assertEqual(self.authority['dt_ps'], 0.004000000189989805)
        self.assertEqual(self.authority['asc'], {'kind': 'explicit_legacy_whole_system', 'alpha': 0.004999999888241291})
        self.assertEqual(GEN.TARGET_TETS, 960)
        self.assertEqual(GEN.FULL_STEPS, 40000)
        self.assertEqual(GEN.FULL_STEPS // GEN.CHUNK_STEPS, 80)
        self.assertEqual(GEN.QUEUE_CAPACITY, 768000)

    def test_centers_are_actual_eight_leaf_inputs_not_inferred_pockets(self):
        proof = self.authority['center_authority']
        self.assertEqual(proof['center_atom'], [0] * 8)
        self.assertEqual(proof['atom_identity'],
                         {'atom': 0, 'name': 'N', 'residue_id': 0, 'residue_name': 'MET'})
        self.assertEqual(proof['scope'], 'Explicit diagnostic center, not an inferred active pocket')
        self.assertEqual(sorted(proof['historical_exit_codes'].values()), [1, 4, 4])
        self.assertTrue(proof['production_geometry_requalification_required'])

    def test_operator_config_uses_real_adaptive_schema_and_keeps_full_run(self):
        # Pure assembly of the operator's declared parameters. Actual module
        # binding/publication still requires verify_build and the real binary.
        science = dict(center_atom=[0] * 8, kappa_foliated=.05,
                       lif_decay=.95, lif_threshold=1., lif_kick=.01)
        with tempfile.TemporaryDirectory() as directory:
            config = GEN.build_config(self.authority, science, 'contract-control',
                                      Path(directory) / 'unused-run', 1, 0, {})
        self.assertEqual(config['governor'], 'armed_adaptive')
        self.assertEqual(config['harmonic_source_binding'], self.authority['harmonic_source_binding'])
        self.assertEqual(config['adaptive_dt_bounds'], {'dt_min_ps': .0005, 'dt_max_ps': .002})
        self.assertEqual(config['numerical']['dt_ps'], .002)
        self.assertTrue(config['director_adaptive_dt'])
        self.assertTrue(config['hold_terminal_controls'])
        self.assertEqual([config['grid_dimension'], config['grid_spacing_angstrom']], [64, 1.])
        self.assertEqual(config['full_step_budget'], 40000)
        self.assertEqual(config['chunk_steps'], 500)
        self.assertEqual(config['target_tetrahedra'], 960)
        self.assertEqual(config['max_seed_peel_passes'], 256)
        self.assertEqual(config['clustering_epsilon_angstrom'], 5.)
        for key in ['picard_residual', 'unitarity_defect']:
            self.assertEqual(config['numerical'][key], 1e-7)
        for key in ['dec_residual', 'absolute_energy_drift']:
            self.assertEqual(config['numerical'][key], 1e-6)
        self.assertEqual([row['center_atom'] for row in config['leaves']], [0] * 8)
        self.assertEqual([row['protocol'] for row in config['leaves']], self.authority['protocols'])
        self.assertEqual([row['asc'] for row in config['leaves']], [self.authority['asc']] * 8)
        self.assertFalse(any('radius' in key for key in config))

    def test_operator_authority_keeps_superseded_values_and_unbound_radius(self):
        record = GEN.operator_authority(self.authority, None)
        self.assertEqual(record['grid']['superseded_retained_dimension'], 128)
        self.assertEqual(record['grid']['superseded_retained_spacing_angstrom'], .75)
        self.assertEqual(record['time']['superseded_retained_initial_request_ps'], .004000000189989805)
        self.assertEqual(record['mesh']['requested_radius'], 8.5)
        self.assertIn('UNRESOLVED', record['mesh']['radius_binding'])
        self.assertIn('kernel pre-scan limit BPBW_MAX_PEEL_PASSES=256', record['mesh']['peel_binding'])
        self.assertEqual(record['mesh']['peel_kernel_source'], GEN.pin(GEN.ENGINE / GEN.SEED_SOURCE))
        self.assertEqual(record['accepted_steps']['full_step_budget_field'], 40000)
        self.assertEqual(record['harmonic_execution']['historical_producer_verification'], 'UNKNOWN')
        self.assertEqual(record['harmonic_execution']['bond_scale'], .5)
        self.assertEqual(record['harmonic_execution']['angle_scale'], .5)
        self.assertEqual(record['harmonic_execution']['binding_status'], 'BOUND')

    def test_original_topology_requires_explicit_harmonic_binding(self):
        a = dict(self.authority)
        a.pop('harmonic_source_binding')
        science = dict(center_atom=[0] * 8, kappa_foliated=.05, lif_decay=.95, lif_threshold=1., lif_kick=.01)
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(GEN.Refusal, 'harmonic source binding'):
            GEN.build_config(a, science, 'contract-control', Path(directory) / 'unused', 1, 0, {})

    def test_harmonic_binding_refuses_changed_target_or_extra_claim(self):
        actual = Path(self.authority['harmonic_source_binding']['path'])
        for change in ['target', 'claim', 'receipt']:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                value = GEN.strict_json(actual.read_bytes())
                if change == 'target':
                    value['target_topology']['sha256'] = '0' * 64
                elif change == 'claim':
                    value['historical_producer_verification'] = 'VERIFIED'
                else:
                    value['preparation_receipt']['sha256'] = '0' * 64
                path = Path(directory) / 'binding.json'
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    GEN.load_harmonic_binding(self.authority, path, GEN.sha(path.read_bytes()))

    def test_generator_refuses_unbound_center_substitution(self):
        science = dict(center_atom=list(range(8)), kappa_foliated=.05,
                       lif_decay=.95, lif_threshold=1., lif_kick=.01)
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(GEN.Refusal, 'center authority'):
            GEN.build_config(self.authority, science, 'contract-control',
                             Path(directory) / 'unused-run', 1, 0, {})

    def test_selection_duplicate_or_center_substitution_refused(self):
        raw = GEN.read_file(GEN.TASK / GEN.CENTER_SELECTION, GEN.CENTER_SELECTION_SHA)
        topology = GEN.strict_json(GEN.read_file(Path(self.authority['topology']['path']),
                                               self.authority['topology']['sha256']))
        for changed in [raw.replace(b'0\t0\t0\t1\t', b'0\t0\t1\t1\t', 1),
                        raw.replace(b'0\t1\t1\t1\t', b'0\t0\t1\t1\t', 1)]:
            with self.assertRaises(GEN.Refusal):
                GEN.centers_from_selection(changed, 0, topology)

    def test_corrupt_phase_boundary_refused(self):
        state = GEN.strict_json((GEN.BASELINE / 'NATIVE_OUTPUT/kras_stream0_protocol_state.json').read_bytes())
        plan = GEN.strict_json((GEN.BASELINE / 'NATIVE_OUTPUT/native_differential_protocol_plan.json').read_bytes())
        state['warm_hold_end'] += 1
        with self.assertRaisesRegex(GEN.Refusal, 'boundaries disagree'):
            GEN.protocol_from_records(state, plan['streams'][0], 0)

    def test_corrupt_retained_authority_refused(self):
        path = GEN.BASELINE / 'launch.json'
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / 'launch.json'
            changed.write_bytes(path.read_bytes() + b'\n')
            with self.assertRaisesRegex(GEN.Refusal, 'SHA256 mismatch'):
                GEN.read_file(changed, GEN.BASELINE_HASHES['launch.json'])

    def test_inspect_declares_missing_science_without_run(self):
        result = subprocess.run([sys.executable, '-B', str(SCRIPT), '--inspect'],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 2, result.stderr)
        result = json.loads(result.stdout)
        self.assertEqual(result['missing_scientific_keys'], sorted(GEN.SELECTION_KEYS))
        self.assertFalse(result['device_initialized'])
        self.assertEqual(result['missing_input_bindings'], ['harmonic_source_binding'])
        self.assertEqual(result['status'], 'INPUT_BINDING_ONLY')
        self.assertEqual(result['grid'], [64, 1.])
        self.assertEqual(result['initial_native_dt_request_ps'], .002)
        self.assertEqual(result['operator_authority']['mesh']['requested_radius'], 8.5)


class NumericInputChecks(unittest.TestCase):
    def setUp(self):
        # Pure admission inputs only; these are not operator run settings.
        self.values = dict(center_atom=list(range(8)), kappa_foliated=1,
                           lif_decay=.5, lif_threshold=1, lif_kick=1)

    def test_finite_representable_values_accepted(self):
        GEN.validate_science(self.values, 2685)

    def test_every_missing_key_refused(self):
        for key in GEN.SELECTION_KEYS:
            with self.subTest(key=key):
                value = dict(self.values)
                del value[key]
                with self.assertRaises(GEN.Refusal):
                    GEN.validate_science(value, 2685)

    def test_unknown_key_and_invalid_centers_refused(self):
        cases = [dict(self.values, extra=1), dict(self.values, center_atom=[0] * 7),
                 dict(self.values, center_atom=[2685] * 8), dict(self.values, center_atom=[True] * 8)]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(GEN.Refusal):
                GEN.validate_science(case, 2685)

    def test_finite_fp32_underflow_cannot_disable_required_owners(self):
        for key in ['kappa_foliated', 'lif_threshold', 'lif_kick']:
            for value in [0.0, 1e-300]:
                with self.subTest(key=key, value=value), self.assertRaises(GEN.Refusal):
                    GEN.validate_science(dict(self.values, **{key: value}), 2685)

    def test_nonfinite_overflow_boolean_and_out_of_domain_refused(self):
        for key in GEN.SELECTION_KEYS - {'center_atom'}:
            for value in [float('nan'), float('inf'), 1e100, 10**1000, True]:
                with self.subTest(key=key, value=repr(value)[:32]), self.assertRaises(GEN.Refusal):
                    GEN.validate_science(dict(self.values, **{key: value}), 2685)
        for value in [-.1, 1.1]:
            with self.assertRaises(GEN.Refusal):
                GEN.validate_science(dict(self.values, lif_decay=value), 2685)

    def test_duplicate_and_nonfinite_json_refused(self):
        for raw in ['{"lif_kick":1,"lif_kick":2}', '{"lif_kick":NaN}', '{"lif_kick":Infinity}']:
            with self.assertRaises(GEN.Refusal):
                GEN.strict_json(raw)


class BuildAdmissionChecks(unittest.TestCase):
    def setUp(self):
        # Metadata controls exercise the gate only; no module/build result is
        # fabricated and verify_build/publish_validated are not invoked here.
        self.metadata = {'command': ['cargo', 'build', '-p', 'prism-nhs', '--bin', 'nhs_rt_full',
                                     '--release', '--no-default-features', '--features', 'gpu,ensemble,v2_ignition'],
                         'cwd': str(GEN.ENGINE), 'exit_code': 0, 'sources_unchanged': True,
                         'started_utc': '2026-10-18T00:00:00Z', 'ended_utc': '2026-10-18T00:01:00Z'}

    def test_complete_build_metadata_shape(self):
        start, end = GEN.validate_build_record(self.metadata, GEN.ENGINE)
        self.assertEqual(end - start, 60)

    def test_check_test_failure_changed_sources_and_wrong_workspace_refused(self):
        cases = []
        for command in ['check', 'test']:
            record = copy.deepcopy(self.metadata)
            record['command'][1] = command
            cases.append(record)
        cases.extend([dict(self.metadata, exit_code=1), dict(self.metadata, exit_code=False),
                      dict(self.metadata, sources_unchanged=False), dict(self.metadata, cwd='/tmp')])
        for record in cases:
            with self.subTest(record=record), self.assertRaises(GEN.Refusal):
                GEN.validate_build_record(record, GEN.ENGINE)

    def test_missing_release_feature_or_wrong_interval_refused(self):
        cases = []
        record = copy.deepcopy(self.metadata)
        record['command'].remove('--release')
        cases.append(record)
        record = copy.deepcopy(self.metadata)
        record['command'][-1] = 'gpu'
        cases.append(record)
        cases.append(dict(self.metadata, ended_utc='2026-10-17T23:59:00Z'))
        for record in cases:
            with self.subTest(record=record), self.assertRaises(GEN.Refusal):
                GEN.validate_build_record(record, GEN.ENGINE)

    def test_actual_failed_check_record_is_not_a_build(self):
        path = GEN.TASK / 'INTERNAL_QC/v10_engine_integration/governed_module/actual_crate_001/EXECUTION.json'
        record = GEN.strict_json(GEN.read_file(path))
        with self.assertRaises(GEN.Refusal):
            GEN.validate_build_record(record, GEN.ENGINE)


class BinaryFreshnessChecks(unittest.TestCase):
    """File/transaction controls only; no test file is a production build.

    The validator response is mocked to exercise publication and mutation
    refusal. The complete build/module gate and real executable validation
    remain mandatory in the generator's main path.
    """
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.engine = Path(self.directory.name) / 'engine'
        self.binary = self.engine / 'target/release/nhs_rt_full'
        self.binary.parent.mkdir(parents=True)
        self.binary.write_bytes(b'identity-control-only; not a production executable\n')
        self.binary.chmod(0o755)
        self.identity = GEN.executable_identity(self.engine)
        self.record = {'binary_after': {key: self.identity[key]
                                      for key in ('path', 'sha256', 'size', 'mtime_ns')}}
        self.destination = Path(self.directory.name) / 'configuration.json'
        self.config = {'output_root': str(Path(self.directory.name) / 'unused-run')}

    def validator_response(self, argv, **kwargs):
        self.assertEqual(argv[:3], [str(self.binary), 'biprism-run', '--config'])
        self.assertEqual(argv[-1], '--validate-only')
        self.assertEqual(kwargs['cwd'], self.engine)
        raw = Path(argv[3]).read_bytes()
        return subprocess.CompletedProcess(argv, 0, json.dumps({
            'status': 'INPUTS_VERIFIED', 'device_initialized': False,
            'configuration_sha256': GEN.sha(raw), 'physical_steps_per_leaf': GEN.FULL_STEPS}), '')

    def assert_not_published(self):
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.destination.parent.glob('.biprism-config-*')), [])
        self.assertFalse(Path(self.config['output_root']).exists())

    def test_exact_build_binary_identity_accepted(self):
        self.assertEqual(GEN.verify_build_binary(self.engine, self.record), self.identity)

    def test_each_build_binary_identity_field_must_match(self):
        changed = {'path': str(self.binary.parent / 'other'), 'sha256': '0' * 64,
                   'size': self.identity['size'] + 1, 'mtime_ns': self.identity['mtime_ns'] + 1}
        for key, value in changed.items():
            record = copy.deepcopy(self.record)
            record['binary_after'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(GEN.Refusal, f'{key} differs'):
                GEN.verify_build_binary(self.engine, record)

    def test_missing_and_malformed_build_binary_identity_refused(self):
        for record in [{}, {'binary_after': None}, {'binary_after': []}]:
            with self.subTest(record=record), self.assertRaises(GEN.Refusal):
                GEN.verify_build_binary(self.engine, record)
        for key, invalid in [('path', None), ('sha256', 'G' * 64), ('size', True),
                             ('size', 0), ('mtime_ns', 1.5), ('mtime_ns', -1)]:
            record = copy.deepcopy(self.record)
            record['binary_after'][key] = invalid
            with self.subTest(key=key, value=invalid), self.assertRaises(GEN.Refusal):
                GEN.verify_build_binary(self.engine, record)
        for key in ('path', 'sha256', 'size', 'mtime_ns'):
            record = copy.deepcopy(self.record)
            del record['binary_after'][key]
            with self.subTest(missing=key), self.assertRaises(GEN.Refusal):
                GEN.verify_build_binary(self.engine, record)

    def test_same_size_replacement_cannot_reuse_recorded_timestamp(self):
        previous = self.binary.stat()
        data = self.binary.read_bytes()
        self.binary.write_bytes(data[:-1] + b'!')
        os.utime(self.binary, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        with self.assertRaisesRegex(GEN.Refusal, 'sha256 differs'):
            GEN.verify_build_binary(self.engine, self.record)

    def test_symlink_is_not_the_recorded_executable(self):
        target = self.binary.with_name('actual-file')
        self.binary.rename(target)
        self.binary.symlink_to(target)
        with self.assertRaises((OSError, GEN.Refusal)):
            GEN.verify_build_binary(self.engine, self.record)

    def test_unchanged_validator_identity_preserves_exclusive_publication(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(GEN.subprocess, 'run', side_effect=self.validator_response) as run:
            result = GEN.publish_validated(self.config, self.destination, self.engine, self.identity)
            self.assertEqual(result['status'], 'INPUTS_VERIFIED')
            self.assertEqual(json.loads(self.destination.read_bytes()), self.config)
            original = self.destination.read_bytes()
            with self.assertRaisesRegex(GEN.Refusal, 'absolute and unused'):
                GEN.publish_validated(self.config, self.destination, self.engine, self.identity)
            self.assertEqual(self.destination.read_bytes(), original)
            self.assertEqual(run.call_count, 1)
        self.assertEqual(list(self.destination.parent.glob('.biprism-config-*')), [])
        self.assertFalse(Path(self.config['output_root']).exists())

    def test_binary_changed_before_validation_is_never_launched(self):
        previous = self.binary.stat()
        os.utime(self.binary, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1))
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(GEN.subprocess, 'run') as run, \
                self.assertRaisesRegex(GEN.Refusal, 'before --validate-only'):
            GEN.publish_validated(self.config, self.destination, self.engine, self.identity)
        run.assert_not_called()
        self.assert_not_published()

    def test_binary_changed_during_validation_is_not_published(self):
        for mutation in ('content', 'mtime', 'inode'):
            self.identity = GEN.executable_identity(self.engine)
            def validate_then_mutate(argv, **kwargs):
                response = self.validator_response(argv, **kwargs)
                previous = self.binary.stat()
                if mutation == 'content':
                    data = self.binary.read_bytes()
                    self.binary.write_bytes(data[:-1] + b'!')
                    os.utime(self.binary, ns=(previous.st_atime_ns, previous.st_mtime_ns))
                elif mutation == 'mtime':
                    os.utime(self.binary, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1))
                else:
                    replacement = self.binary.with_name('replacement')
                    replacement.write_bytes(self.binary.read_bytes())
                    replacement.chmod(previous.st_mode & 0o777)
                    os.utime(replacement, ns=(previous.st_atime_ns, previous.st_mtime_ns))
                    replacement.replace(self.binary)
                return response
            with self.subTest(mutation=mutation), mock.patch.dict(os.environ, {}, clear=True), \
                    mock.patch.object(GEN.subprocess, 'run', side_effect=validate_then_mutate), \
                    self.assertRaisesRegex(GEN.Refusal, 'during --validate-only'):
                GEN.publish_validated(self.config, self.destination, self.engine, self.identity)
            self.assert_not_published()

    def test_failed_validator_does_not_publish(self):
        refused = subprocess.CompletedProcess([], 2, '', 'actual validator refusal control')
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(GEN.subprocess, 'run', return_value=refused), \
                self.assertRaisesRegex(GEN.Refusal, 'validator refused'):
            GEN.publish_validated(self.config, self.destination, self.engine, self.identity)
        self.assert_not_published()


if __name__ == '__main__':
    unittest.main()
