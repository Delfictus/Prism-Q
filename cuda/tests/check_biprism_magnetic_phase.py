#!/usr/bin/env python3
"""Check retained GPU magnetic scalars against MPFR, independent of libdevice.

An arithmetic pass does not admit the coupled workload or its trajectory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

from test_biprism_magnetic_field_f64 import MagneticFieldF64, ordered


def inspect(execution):
    path = Path(execution)
    receipt = json.loads(path.read_text())
    assert receipt['native_execution'] is True
    assert receipt['new_kernel_fault_records'] == []
    assert receipt['sources_binary_inputs_unchanged'] is True
    prefix = receipt['command'][4] + '.epoch1.magnetic_phase.'
    arrays = {}
    for name in ('input', 'actual', 'libdevice'):
        p = Path(prefix + name + '.f64le')
        raw = p.read_bytes()
        assert receipt['artifacts'][str(p)] == {
            'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        assert len(raw) % 8 == 0
        arrays[name] = struct.unpack('<' + str(len(raw) // 8) + 'd', raw)
    assert len(arrays['input']) == len(arrays['actual']) == len(arrays['libdevice']) > 0
    MagneticFieldF64.setUpClass()
    oracle = MagneticFieldF64()
    maximum = 0
    try:
        for index, (x, actual) in enumerate(zip(arrays['input'], arrays['actual'])):
            expected = oracle.oracle(x)
            error = abs(ordered(actual) - ordered(expected))
            maximum = max(maximum, error)
            assert error <= 3, (index, x.hex(), actual.hex(), expected.hex(), error)
    finally:
        MagneticFieldF64.tearDownClass()
    print('NATIVE_MAGNETIC_MPFR_PASS', len(arrays['input']), 'max_ULP', maximum,
          'execution_sha256', hashlib.sha256(path.read_bytes()).hexdigest())
    print('SCOPE magnetic_scalar_arithmetic_only native_physical_acceptance=false')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execution', required=True)
    inspect(parser.parse_args().execution)
