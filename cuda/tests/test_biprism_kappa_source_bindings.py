"""CPU source-binding rejection controls; no new native observations."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

P = Path(__file__).with_name('check_biprism_v10_kappa_response.py')
S = importlib.util.spec_from_file_location('kappa_bindings', P)
R = importlib.util.module_from_spec(S)
S.loader.exec_module(R)


class DecBindings(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='kappa_binding_cpu_')
        self.addCleanup(self.tmp.cleanup)
        self.bindings = []
        for name in ('main.cuh', 'reader.py', 'ordered.cuh'):
            p = Path(self.tmp.name) / name
            p.write_bytes(name.encode())
            self.bindings.append({'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
        self.check = {'command': ['python', self.bindings[1]['path']],
                      'checker_sha256': self.bindings[1]['sha256']}
        self.expected = {x['path']: x['sha256'] for x in (self.bindings[0], self.bindings[2])}

    def run_check(self, bindings):
        R.check_dec_source_bindings(R.Files(), {'source_bindings': bindings}, self.check, self.expected)

    def test_reader_can_be_first_middle_or_last(self):
        for bindings in (self.bindings, self.bindings[::-1], self.bindings[1:]+self.bindings[:1],
                         [self.bindings[0],self.bindings[2],self.bindings[1]]):
            self.run_check(bindings)

    def test_last_dependency_is_never_skipped(self):
        bindings = copy.deepcopy(self.bindings)
        bindings[-1]['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'not in executed build'): self.run_check(bindings)

    def test_reader_binding_missing_duplicate_or_wrong_digest_is_refused(self):
        bad = copy.deepcopy(self.bindings); bad[1]['sha256'] = '0'*64
        for bindings in (self.bindings[::2], self.bindings+[self.bindings[1]], bad):
            with self.subTest(bindings=bindings), self.assertRaises(ValueError): self.run_check(bindings)

    def test_current_file_mutation_is_detected(self):
        Path(self.bindings[-1]['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'): self.run_check(self.bindings)


if __name__ == '__main__': unittest.main()
