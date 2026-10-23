"""Exact-dyadic residual verification of all actual-operator FP64 CG controls."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import math
from fractions import Fraction as Q

from check_biprism_v10_dec_readback import exact_operator


def inspect(execution):
    rec = json.loads(Path(execution).read_text())
    assert rec['native_execution'] and rec['sources_binary_inputs_unchanged']
    assert rec['exit_code'] == 0 and not rec['new_kernel_fault_records']
    assert rec['numerical_control_only'] and not rec['coupled_diagnostic_accepted']
    def artifact(path):
        p = Path(path); raw = p.read_bytes()
        assert rec['artifacts'][str(p)] == dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        return raw
    fixture = Path(rec['command'][1]); cases = fixture.read_bytes()
    assert hashlib.sha256(cases).hexdigest() == rec['input_hashes'][str(fixture)]
    assert len(cases) == 8*19464
    phi = artifact(rec['command'][2]+'.phi.bin')
    result = artifact(rec['command'][2]+'.result.bin')
    assert len(phi) == 128*256*8 and len(result) == 128*16
    maximum = 0.; count = 0
    for cold in range(2):
        for control in range(4):
            for leaf in range(8):
                index = ((1*2+cold)*4+control)*8+leaf
                ok, iters, reported, reserved = struct.unpack_from('<IIfI', result, index*16)
                assert not reserved and 0 <= iters <= 192
                state = list(struct.unpack_from('<256d', phi, index*256*8))
                base = leaf*19464
                edges, stored_leaf = struct.unpack_from('<II', cases, base)
                assert stored_leaf == leaf and 0 < edges <= 1536
                uv = list(struct.iter_unpack('<BB', cases[base+8:base+8+2*edges]))
                w = list(struct.unpack_from(f'<{edges}d', cases, base+3080))
                q = list(struct.unpack_from('<256d', cases, base+15368))
                before = list(struct.unpack_from('<256d', cases, base+17416))
                assert state[192:] == before[192:]
                if control in (1, 2):
                    assert not ok
                    continue
                if control == 3:
                    factor = struct.unpack('<f', struct.pack('<f', .15))[0]
                    q = [v+factor*.0123456789012345 for v in q]
                exact = exact_operator(q[:192], state, uv, w)['squared']
                assert ok and exact <= Q(1, 10**20), (cold, control, leaf, float(exact))
                rr = math.sqrt(float(exact)); maximum = max(maximum, rr); count += 1
                print('FP64_DEC_EXACT', cold, control, leaf, 'iterations', iters,
                      'native', reported, 'exact_dyadic_l2', rr)
    print('EXACT_POSITIVE_CASES', count, 'NEGATIVE_CASES', 32,
          'MAX_TRUE_RESIDUAL', maximum, 'COUPLED_ACCEPTANCE=0')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execution', required=True, type=Path)
    inspect(p.parse_args().execution)
