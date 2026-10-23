#!/usr/bin/env python3
"""Temporary-fixture controls for the archive assembler, never native evidence.

No GPU, production configuration or operator archive is created by this suite.
The small synthetic records below test rejection and preservation behavior only.
"""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zipfile

SOURCE = Path(__file__).resolve().parents[1] / 'assemble_biprism_v10_forensic_audit.py'
SPEC = importlib.util.spec_from_file_location('forensic_assembler_under_test', SOURCE)
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


def dump(path, value):
    path.write_text(json.dumps(value, allow_nan=False))
    return A.stable_pin(path)


class EvidenceControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def file(self, name, raw):
        path = self.root / name
        path.write_bytes(raw)
        return A.stable_pin(path)

    def test_strict_json_duplicate_and_nonfinite(self):
        for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}', '{"x":[1e999]}'):
            with self.subTest(raw=raw), self.assertRaises(A.AuditError):
                A.strict_json(raw)
        self.assertEqual(A.strict_json('{"x":1}'), {'x': 1})

    def test_hash_extent_and_alias_refusal(self):
        rec = self.file('bytes', b'actual bytes')
        self.assertEqual(A.stable_pin(rec['path'], rec['sha256'], rec['bytes']), rec)
        with self.assertRaises(A.AuditError): A.stable_pin(rec['path'], '0'*64)
        with self.assertRaises(A.AuditError): A.stable_pin(rec['path'], size=True)
        alias = self.root / 'alias'; alias.symlink_to(rec['path'])
        with self.assertRaises(A.AuditError): A.stable_pin(alias)
        Path(rec['path']).write_bytes(b'changed')
        with self.assertRaises(A.AuditError): A.read_pinned(rec)

    def test_json_numbers_are_not_boolean_statuses(self):
        self.assertFalse(A.integer(False, 0)); self.assertFalse(A.integer(True, 1))
        self.assertFalse(A.exact_value(False, 0)); self.assertFalse(A.exact_value(1, 1.0))
        self.assertFalse(A.exact_value({'a': [True]}, {'a': [1]}))
        self.assertTrue(A.integer(0, 0))

    def test_pinned_module_loading_never_writes_bytecode(self):
        pin = self.file('validator.py', b'import sys\nobserved = sys.dont_write_bytecode\nvalue = 17\n')
        with patch.object(A.sys, 'dont_write_bytecode', False):
            module = A.load_pinned_module(pin, 'test_only_validator')
            self.assertEqual(module.value, 17)
            self.assertTrue(module.observed)
            self.assertFalse(A.sys.dont_write_bytecode)
        self.assertFalse((self.root/'__pycache__').exists())
        Path(pin['path']).write_bytes(b'value = 18\n')
        with self.assertRaises(A.AuditError): A.load_pinned_module(pin, 'test_only_validator')
        broken = self.file('broken_validator.py', b'raise RuntimeError("test only")\n')
        with patch.object(A.sys, 'dont_write_bytecode', False), self.assertRaises(RuntimeError):
            try:
                A.load_pinned_module(broken, 'test_only_broken_validator')
            finally:
                self.assertFalse(A.sys.dont_write_bytecode)
        self.assertFalse((self.root/'__pycache__').exists())

    def test_production_receipt_requires_explicit_selected_pin(self):
        inputs = A.Inputs()
        selected = dump(self.root/'production.json', {'fixture_only': True})
        inputs.add(selected, 'production')
        spec = {'production_receipt': 'production'}
        self.assertEqual(A.production_receipt_pin(inputs, spec, selected['path'])['sha256'], selected['sha256'])
        with self.assertRaisesRegex(A.AuditError, 'explicit production receipt'):
            A.production_receipt_pin(inputs, {}, selected['path'])
        other = dump(self.root/'different_production.json', {'fixture_only': True})
        with self.assertRaisesRegex(A.AuditError, 'CLI production receipt'):
            A.production_receipt_pin(inputs, spec, other['path'])

    def test_production_inventory_and_result_match_validated_root(self):
        root = self.root/'run'; root.mkdir()
        receipt = {key: dump(root/(key+'.json'), {'fixture_only': key})
                   for key in ('artifact_inventory', 'execution_result')}
        A.bind_production_inventory(A.Inputs(), receipt, root, receipt)
        for key in receipt:
            alternate = self.file(key+'_elsewhere.json', Path(receipt[key]['path']).read_bytes())
            for kind in ('selected_path', 'validated_path', 'selected_sha', 'validated_sha', 'size'):
                selected, validated = copy.deepcopy(receipt), copy.deepcopy(receipt)
                if kind == 'selected_path': selected[key] = alternate
                if kind == 'validated_path': validated[key] = alternate
                if kind == 'selected_sha': selected[key]['sha256'] = '0'*64
                if kind == 'validated_sha': validated[key]['sha256'] = '0'*64
                if kind == 'size': validated[key]['bytes'] += 1
                with self.subTest(key=key, kind=kind), self.assertRaises(A.AuditError):
                    A.bind_production_inventory(A.Inputs(), selected, root, validated)

    def test_phase2_post_dispatch_independent_log_binding(self):
        inputs = A.Inputs()
        stdout = self.file('p2_independent.stdout', b'phase2_readback pass=1 actual_leaves=8\n')
        stderr = self.file('p2_independent.stderr', b'')
        inputs.add(stdout, 'check_stdout')
        native = {'artifacts': {}, 'phase2_measurement_controls_passed': True,
                  'dispatch_receipt': {'path': str(self.root/'p2_dispatch.json')},
                  'independent_check': {'stdout_sha256': stdout['sha256'], 'stderr_sha256': stderr['sha256']}}
        check = {'artifact': 'check_stdout', 'kind': 'kv', 'assertions': [
            {'prefix': 'phase2_readback', 'select': {}, 'count': 1,
             'eq': {'pass': '1'}, 'positive': ['actual_leaves']}]}
        self.assertEqual(A.independent_references(native), [
            {'path': stdout['path'], 'sha256': stdout['sha256']},
            {'path': stderr['path'], 'sha256': stderr['sha256']}])
        A.check_assertions(inputs, check, native)
        for kind in ('wrong_prefix', 'wrong_hash', 'wrong_suffix'):
            changed = copy.deepcopy(native)
            if kind == 'wrong_prefix': changed['dispatch_receipt']['path'] = str(self.root/'other_dispatch.json')
            if kind == 'wrong_hash': changed['independent_check']['stdout_sha256'] = '0'*64
            if kind == 'wrong_suffix': changed['dispatch_receipt']['path'] = str(self.root/'p2_execution.json')
            with self.subTest(kind=kind), self.assertRaises(A.AuditError):
                A.check_assertions(inputs, check, changed)

    def test_raw_predicate_binding_and_nonzero_coverage(self):
        inputs = A.Inputs()
        rec = self.file('native.stdout', b'coordinate_parity precision=32 max_ulp=0 value_mismatches=0 actual_edges=10 pass=1\n')
        inputs.add(rec, 'native_stdout')
        native = {'artifacts': {rec['path']: rec}}
        check = {'artifact': 'native_stdout', 'kind': 'kv', 'assertions': [
            {'prefix': 'coordinate_parity', 'select': {'precision': '32'}, 'count': 1,
             'eq': {'max_ulp': '0', 'value_mismatches': '0', 'pass': '1'}, 'positive': ['actual_edges']}]}
        A.check_assertions(inputs, check, native)
        for alteration in ('unbound', 'mismatch', 'missing_precision', 'empty_coverage'):
            c, n = copy.deepcopy(check), copy.deepcopy(native)
            if alteration == 'unbound': n['artifacts'] = {}
            if alteration == 'mismatch': c['assertions'][0]['eq']['max_ulp'] = '1'
            if alteration == 'missing_precision': c['assertions'][0]['select']['precision'] = '64'
            if alteration == 'empty_coverage': c['assertions'][0]['positive'] = ['max_ulp']
            with self.subTest(alteration=alteration), self.assertRaises(A.AuditError):
                A.check_assertions(inputs, c, n)

    def native_fixture(self, change=None, after_dispatch=None):
        """Test-only receipt graph; never passed off as an executed control."""
        inputs = A.Inputs()
        main = self.file('main.cuh', b'current source\n')
        binary = self.file('native_binary', b'test-only binary bytes')
        inputs.add(main, 'main')
        hashes = {main['path']: main['sha256'], binary['path']: binary['sha256']}
        build = {'exit_code': 0, 'sources_unchanged': True, 'header_overlays': False,
                 'command': ['nvcc', '-arch=sm_120', main['path'], '-o', binary['path']],
                 'binary_sha256': binary['sha256'], 'input_hashes': hashes}
        build_pin = dump(self.root/'build.json', build); inputs.add(build_pin, 'build')
        prefix = str(self.root/'state')
        conv = bytearray(1024)
        for leaf in range(8): struct.pack_into('<6I', conv, leaf*128, 0x42505631, leaf, 0, 0, 0, 4)
        conv_pin = self.file('state.epoch1.conv.bin', conv)
        dec_pin = dump(self.root/'dec.json', {'passed': True})
        receipt = {'native_execution': True, 'native_exit_code': 0, 'exit_code': 0,
                   'sources_binary_inputs_unchanged': True, 'gpu_recovery_none_after': True,
                   'new_kernel_fault_records': [], 'boot_id': 'test-only-boot',
                   'production_admission_reset': False, 'installed_state_mutation': False,
                   'build_receipt': build_pin, 'command': [binary['path'], 'case', 'selection', 'mass', prefix],
                   'input_hashes': hashes, 'diagnostic_mode': '--asymmetric-kappa=0',
                   'main_compiler_resources': {'registers': 128, 'stack': 0, 'cumulative_stack': 0, 'spill_stores': 0, 'spill_loads': 0},
                   'main_device_callee_resources': [{'symbol': 'callee', 'stack': 0, 'spill_stores': 0, 'spill_loads': 0}],
                   'coupled_diagnostic_accepted': True, 'independent_dec_all_epochs_passed': True,
                   'expected_epochs': 1, 'independent_dec_checks': [{'epoch': 1, 'exit_code': 0, 'error': None, 'result': dec_pin}],
                   'artifacts': {conv_pin['path']: conv_pin}}
        if change: change(receipt, conv_pin)
        receipt['dispatch_receipt'] = dump(self.root/'dispatch.json', receipt)
        if after_dispatch: after_dispatch(receipt)
        execution = dump(self.root/'execution.json', receipt); inputs.add(execution, 'execution')
        definition = {'kind': 'coupled', 'receipt': 'execution', 'sources': ['main'], 'mode': '--asymmetric-kappa=0'}
        with patch.object(A, 'MAIN', Path(main['path'])):
            return A.validate_native(inputs, definition, execution['path'], build_pin)

    def test_actual_resource_field_names_are_required(self):
        self.native_fixture()
        def wrong_keys(r, _):
            r['main_device_callee_resources'] = [{'symbol': 'callee', 'stack': 0, 'stores': 0, 'loads': 0}]
        with self.assertRaisesRegex(A.AuditError, 'callee'):
            self.native_fixture(wrong_keys)

    def test_native_partial_leaf_and_cumulative_stack_refuse(self):
        def bad_leaf(r, pin):
            raw = bytearray(Path(pin['path']).read_bytes()); struct.pack_into('<I', raw, 7*128+20, 0)
            Path(pin['path']).write_bytes(raw)
            r['artifacts'][pin['path']] = A.stable_pin(pin['path'])
        def callee_spill(r, _): r['main_device_callee_resources'][0]['spill_loads'] = 4
        def no_cumulative(r, _): del r['main_compiler_resources']['cumulative_stack']
        def false_exit(r, _): r['exit_code'] = False
        def false_stack(r, _): r['main_compiler_resources']['stack'] = False
        def wrong_mode(r, _): r['diagnostic_mode'] = 'asymmetric_kappa_zero'
        for change in (bad_leaf, callee_spill, no_cumulative, false_exit, false_stack, wrong_mode):
            with self.subTest(change=change.__name__), self.assertRaises(A.AuditError): self.native_fixture(change)
        for word in (2, 3, 4):
            def nonzero_fault(r, pin, word=word):
                raw = bytearray(Path(pin['path']).read_bytes()); struct.pack_into('<I', raw, 4*128+4*word, 1)
                Path(pin['path']).write_bytes(raw)
                r['artifacts'][pin['path']] = A.stable_pin(pin['path'])
            with self.subTest(fault_word=word), self.assertRaises(A.AuditError): self.native_fixture(nonzero_fault)

    def test_enclosing_receipt_cannot_replace_dispatch_artifacts(self):
        def replaced(receipt):
            pin = next(iter(receipt['artifacts'].values()))
            raw = Path(pin['path']).read_bytes()+b'replaced after dispatch'
            Path(pin['path']).write_bytes(raw)
            receipt['artifacts'][pin['path']] = A.stable_pin(pin['path'])
        with self.assertRaisesRegex(A.AuditError, 'dispatch mismatch: artifacts'):
            self.native_fixture(after_dispatch=replaced)

    def test_actual_ptxas_block_requires_all_function_resources(self):
        log = ("ptxas info : Compiling entry function 'nhs_biprism_variational_so3g_v10' for 'sm_120'\n"
               'ptxas info : Function properties for nhs_biprism_variational_so3g_v10\n'
               '    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n'
               'ptxas info : Used 128 registers, 0 bytes cumulative stack size\n'
               'ptxas info : Function properties for callee\n'
               '    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n')
        self.assertEqual(A.native_resources(log.encode())['main']['registers'], 128)
        duplicate = ('ptxas info : Function properties for callee\n'
                     '    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n')
        self.assertEqual(len(A.native_resources((log+duplicate).encode())['callees']), 2)
        with self.assertRaises(A.AuditError):
            A.native_resources((log+duplicate.replace('0 bytes spill loads', '4 bytes spill loads')).encode())
        for bad in (log.replace('0 bytes cumulative', '8 bytes cumulative'),
                    log.replace('0 bytes spill loads', '4 bytes spill loads'),
                    log.replace("'sm_120'", "'sm_90'"), log+'ptxas info : Function properties for missing\n'):
            with self.subTest(bad=bad), self.assertRaises(A.AuditError): A.native_resources(bad.encode())

    def packet_fixture(self):
        inputs = A.Inputs()
        before = self.file('before.cuh', b'line one\r\nold `value`\r\nno newline')
        current = self.file('current.cuh', b'line one\nnew ```value```\nno newline')
        raw = self.file('evidence.stdout', b'raw immutable test log\n')
        for key, rec in [('before', before), ('current', current), ('raw', raw)]: inputs.add(rec, key)
        spec = {'failures': [{'id': n, 'invariant': f'Test-only invariant {n}',
                  'fragments': [{'label': 'Actual byte interval', 'baseline_kind': 'historical_failure',
                      'before': {'file': 'before', 'start': 2, 'end': 3},
                      'current': {'file': 'current', 'start': 2, 'end': 3}}],
                  'evidence': ['raw'], 'checks': ['checked'], 'open_requirements': []} for n in range(1, 8)]}
        states = A.validate_failures(inputs, spec, {'checked': {}}, {current['path']: current['sha256']})
        return inputs, spec, states

    def test_exact_source_intervals_and_open_scope(self):
        inputs, spec, states = self.packet_fixture()
        spec['failures'][0]['open_requirements'] = ['Independent molecular benchmark not supplied']
        states = A.validate_failures(inputs, spec, {'checked': {}}, {inputs.item('current')['path']: inputs.item('current')['sha256']})
        self.assertEqual(states[0]['status'], 'OPEN')
        documents, slices = A.render_packets(inputs, spec, states)
        for row in slices:
            embedded = documents[row['document']][row['offset']:row['offset']+row['bytes']]
            self.assertEqual(hashlib.sha256(embedded).hexdigest(), row['sha256'])
        self.assertIn(b'old `value`\r\nno newline', documents['failure_1.md'])
        self.assertIn(b'new ```value```\nno newline', documents['failure_1.md'])
        for interval in ({'file': 'before', 'start': 0, 'end': 3}, {'file': 'before', 'start': 1, 'end': 99},
                         {'file': 'before', 'start': True, 'end': 3}):
            with self.assertRaises(A.AuditError): A.excerpt(inputs, interval)

    def test_flat_archive_reopens_without_original_sources_and_refuses_corruption(self):
        inputs, spec, states = self.packet_fixture()
        output, archive = self.root/'output', self.root/'output.zip'
        # This library-level temp fixture tests byte packaging, not clinical,
        # native or production acceptance. CLI gates are not bypassed in use.
        validation = {'production': {'unit_test_fixture_only': True}}
        A.publish(inputs, spec, states, validation, output, archive, {'unit_test_fixture_only': True})
        original = archive.read_bytes()
        for rec in inputs.files.values(): Path(rec['path']).unlink()
        self.assertTrue(A.verify_archive(archive)['flat'])
        with self.assertRaises(A.AuditError): A.publish(inputs, spec, states, validation, output, archive, {})
        self.assertEqual(archive.read_bytes(), original)
        altered = self.root/'altered.zip'
        with zipfile.ZipFile(archive) as source, zipfile.ZipFile(altered, 'w') as dest:
            for name in source.namelist():
                dest.writestr(name, source.read(name) + (b'corruption' if name == 'failure_1.md' else b''))
        with self.assertRaises(A.AuditError): A.verify_archive(altered)

    def test_absent_production_never_publishes(self):
        inputs, spec, states = self.packet_fixture()
        with self.assertRaises(A.AuditError):
            A.publish(inputs, spec, states, {'production': None}, self.root/'output', self.root/'output.zip', {})
        self.assertFalse((self.root/'output').exists())
        argv = ['--evidence', '/unused', '--evidence-sha256', '0'*64, '--accepted-build', '/unused',
                '--native-control', 'a=/unused', '--production-absent', 'Actual production not executed']
        with contextlib.redirect_stderr(io.StringIO()): self.assertEqual(A.main(argv), 2)
        self.assertFalse((self.root/'output.zip').exists())

    def test_output_directory_cannot_alias_zip_destination(self):
        inputs, spec, states = self.packet_fixture()
        output = self.root/'same.zip'
        with self.assertRaisesRegex(A.AuditError, 'must be distinct'):
            A.publish(inputs, spec, states, {'production': {'unit_test_fixture_only': True}},
                      output, output.with_suffix('.zip'), {})
        self.assertFalse(output.exists())

    def test_portable_excerpt_metadata_cannot_exceed_source_or_document(self):
        inputs, spec, states = self.packet_fixture()
        archive = self.root/'original.zip'
        A.publish(inputs, spec, states, {'production': {'unit_test_fixture_only': True}},
                  self.root/'original', archive, {'unit_test_fixture_only': True})
        changes = [('last_line', 999999), ('offset', 999999), ('bytes', 999999)]
        changes += [(key, value) for key in ('offset', 'bytes', 'first_line', 'last_line')
                    for value in (True, 1.0)]
        for index, (key, value) in enumerate(changes):
            altered = self.root/f'metadata_{index}.zip'
            with zipfile.ZipFile(archive) as source, zipfile.ZipFile(altered, 'w') as dest:
                receipt = json.loads(source.read('ARCHIVE_RECEIPT.json'))
                receipt['excerpts'][0][key] = value
                for name in source.namelist():
                    raw = json.dumps(receipt).encode() if name == 'ARCHIVE_RECEIPT.json' else source.read(name)
                    dest.writestr(name, raw)
            with self.subTest(key=key, value=value), self.assertRaisesRegex(A.AuditError, 'excerpt interval'):
                A.verify_archive(altered)


class OutputControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        import pyarrow as pa
        self.pa = pa
        self.config = {'leaves': [{'seed': 100+i, 'stream_id': i} for i in range(8)]}

    def tearDown(self):
        self.tmp.cleanup()

    def write_pair(self, directory, items, raw=True, change_native=None, change_ledger=None):
        pa = self.pa
        directory.mkdir(exist_ok=True)
        native, ledger, packets, observations = [], [], [], []
        for i, (leaf, step, marker) in enumerate(items):
            p, o = bytearray(96), bytearray(64)
            struct.pack_into('<I', p, 12, step); struct.pack_into('<I', p, 16, 201)
            struct.pack_into('<HHi', p, 64, leaf, 17, 26); p[40] = marker
            struct.pack_into('<I', p, 72, 0x56313053)
            struct.pack_into('<IHHIi', o, 0, step, leaf, 17, 201, 26)
            struct.pack_into('<IQII', o, 44, 0x4e4f4232, 1, 2, 0)
            packets.append(bytes(p)); observations.append(bytes(o))
            native.append({'spike_id': i, 'replica_seed': 100+leaf, 'stream_id': leaf, 'timestep': step, 'spike_source': 6})
            ledger.append({'spike_id': i, 'replica_seed': 100+leaf, 'step_tau': step, 'leaf_id': leaf,
                           'raw_v10s_96': bytes(p), 'raw_nob2_64': bytes(o)})
            for name, (owner, fmt, offset) in A.LEDGER_FIELDS.items():
                ledger[-1][name] = struct.unpack_from('<'+fmt, p if owner == 'p' else o, offset)[0]
        if change_native: change_native(native)
        if change_ledger: change_ledger(ledger)
        ntypes = [pa.uint64(), pa.uint64(), pa.uint8(), pa.int32(), pa.int32()]
        kinds = {'f': pa.float32(), 'I': pa.uint32(), 'H': pa.uint16(), 'i': pa.int32(), 'Q': pa.uint64()}
        ltypes = [pa.uint64(), pa.uint64(), pa.binary(), pa.binary()]+[kinds[fmt] for _, fmt, _ in A.LEDGER_FIELDS.values()]
        for name, rows, names, types in (
            ('spike_events.arrow', native, ['spike_id', 'replica_seed', 'stream_id', 'timestep', 'spike_source'], ntypes),
            ('voronoi_ledger.arrow', ledger, ['spike_id', 'replica_seed', 'raw_v10s_96', 'raw_nob2_64', *A.LEDGER_FIELDS], ltypes)):
            columns = {key: pa.array([r[key] for r in rows], type=kind) for key, kind in zip(names, types)}
            if name.startswith('spike_'):
                columns.update({f'unused_{i}': pa.array([0]*len(rows), type=pa.int32()) for i in range(25)})
            table = pa.table(columns)
            meta = {b'prism.queue_owner': b'fixture-only', b'prism.queue_generation': b'0',
                    b'prism.leaf_replica_seeds': json.dumps([x['seed'] for x in self.config['leaves']]).encode()}
            if name == 'voronoi_ledger.arrow':
                meta.update({b'prism.schema': b'voronoi_lif_ledger_v2', b'prism.event_frame': b'atomic_bohr',
                             b'prism.position_unit': b'bohr', b'prism.force_unit': b'hartree/bohr',
                             b'prism.dual_volume_unit': b'bohr^3', b'prism.native_position_unit': b'angstrom'})
            table = table.replace_schema_metadata(meta)
            with pa.OSFile(str(directory/name), 'wb') as target, pa.ipc.new_file(target, table.schema) as writer:
                writer.write_table(table)
        if raw:
            (directory/'events.v10s96').write_bytes(b''.join(packets))
            (directory/'native_observations.64b').write_bytes(b''.join(observations))

    def test_paired_arrow_joins_actual_raw_and_per_leaf_seed(self):
        p = self.root/'paired'; self.write_pair(p, [(0, 1, 2), (7, 2, 3)])
        counts = [0]*8
        self.assertEqual(A.paired_arrow(p, self.config, (p/'events.v10s96', p/'native_observations.64b'), counts=counts), 2)
        self.assertEqual(counts, [1, 0, 0, 0, 0, 0, 0, 1])
        with self.assertRaises(A.AuditError): A.paired_arrow(p, self.config, coordinate_generations=[2]*8)
        with self.assertRaises(A.AuditError): A.paired_arrow(p, self.config, step_window=(501, 1000))
        self.write_pair(p, [(7, 2, 3)], change_native=lambda rows: rows[0].update(replica_seed=100))
        with self.assertRaises(A.AuditError): A.paired_arrow(p, self.config)

    def test_raw_observation_identity_and_modified_packet_refuse(self):
        p = self.root/'paired'
        for off in (6, 8, 12):
            def corrupt(rows, off=off):
                raw = bytearray(rows[0]['raw_nob2_64']); raw[off] ^= 1; rows[0]['raw_nob2_64'] = bytes(raw)
            self.write_pair(p, [(1, 4, 7)], change_ledger=corrupt)
            with self.subTest(offset=off), self.assertRaises(A.AuditError): A.paired_arrow(p, self.config)
        self.write_pair(p, [(1, 4, 7)])
        (p/'events.v10s96').write_bytes(bytes(96))
        with self.assertRaises(A.AuditError): A.paired_arrow(p, self.config, (p/'events.v10s96', p/'native_observations.64b'))

    def test_typed_ledger_cannot_diverge_from_preserved_raw_bytes(self):
        p = self.root/'paired'
        for field, value in (('produced_force_x', 0.5), ('coordinate_generation', 2), ('vertex_id', 18)):
            self.write_pair(p, [(1, 4, 7)], change_ledger=lambda rows, f=field, v=value: rows[0].update({f: v}))
            with self.subTest(field=field), self.assertRaisesRegex(A.AuditError, 'typed ledger field differs'):
                A.paired_arrow(p, self.config)

    def test_whole_leaf_raw_stream_preserves_order_and_content(self):
        a, b = self.root/'chunk', self.root/'leaf'
        self.write_pair(a, [(1, 3, 1), (1, 4, 2)])
        self.write_pair(b, [(1, 3, 1), (1, 4, 2)])
        first, second = [hashlib.sha256() for _ in range(8)], [hashlib.sha256() for _ in range(8)]
        A.paired_arrow(a, self.config, stream_hashes=first); A.paired_arrow(b, self.config, leaf_only=1, stream_hashes=second)
        self.assertEqual(first[1].digest(), second[1].digest())
        self.write_pair(b, [(1, 3, 1), (1, 4, 3)])
        changed = [hashlib.sha256() for _ in range(8)]; A.paired_arrow(b, self.config, stream_hashes=changed)
        self.assertNotEqual(first[1].digest(), changed[1].digest())

    def test_zero_emission_keeps_valid_empty_population(self):
        p = self.root/'empty'; self.write_pair(p, [])
        self.assertEqual(A.paired_arrow(p, self.config, (p/'events.v10s96', p/'native_observations.64b')), 0)

    def test_four_grid_payload_checksum_and_extent(self):
        payload = struct.pack('<QQ4i', 1, 1, -1, 2, -3, 4)
        h = 0xcbf29ce484222325
        for byte in payload: h = ((h ^ byte)*0x100000001b3) & 0xffffffffffffffff
        raw = b'PRSGD001'+struct.pack('<III', 1, 0x01020304, 0xffffffff)
        for name in ('test', 'eight_leaf_aggregate'):
            text = name.encode(); raw += struct.pack('<Q', len(text))+text
        raw += struct.pack('<QQQ', 1, 4, len(payload))+payload+struct.pack('<Q', h)+b'PRSGDEND'
        path = self.root/'grids'; path.write_bytes(raw)
        scope = {'voxel_count': 1, 'serialized_bytes': len(raw), 'payload_fnv1a64': h, 'through_completed_step': 40000}
        self.assertEqual(A.signal_grid(path, scope, 'test')['planes'], 4)
        corrupted = bytearray(raw); corrupted[-17] ^= 1; path.write_bytes(corrupted)
        with self.assertRaises(A.AuditError): A.signal_grid(path, scope, 'test')

    def test_pathb_native_extent_endianness_and_selected_population(self):
        p = self.root/'leaf'; p.mkdir()
        dump(p/'source_population_selection.json', {'unfiltered_rows': 3, 'selected_rows': 2})
        dump(p/'binding_sites.json', {'sites': [], 'cluster_ids': [-1, -1]})
        state = {'endianness': 'little', 'config_size_bytes': 8, 'event_count_offset': 16,
                 'event_buffer_offset': 20, 'event_stride_bytes': 16, 'event_count': 2}
        analysis = {'status': 'NATIVE_ANALYSIS_EXECUTED', 'native_ingest_rows': 2, 'native_state': state}
        dump(p/'pathb_analysis.json', analysis)
        raw = b'SDST'+struct.pack('<I', 1)+bytes(8)+struct.pack('<I', 2)+bytes(32)
        (p/'pathb.native').write_bytes(raw)
        A.validate_pathb(p, 3)
        analysis['native_state']['endianness'] = 'big'; dump(p/'pathb_analysis.json', analysis)
        with self.assertRaises(A.AuditError): A.validate_pathb(p, 3)
        analysis['native_state']['endianness'] = 'little'; dump(p/'pathb_analysis.json', analysis)
        (p/'pathb.native').write_bytes(raw[:-1])
        with self.assertRaises(A.AuditError): A.validate_pathb(p, 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
