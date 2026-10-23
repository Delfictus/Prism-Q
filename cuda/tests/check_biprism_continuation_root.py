"""Independently re-evaluate a CPU root against its bound native starting state.

This is a numerical diagnosis. It cannot qualify a native candidate, grant a
physical commit, or replace the native represented-endpoint rounding stage.
"""
import argparse
import hashlib
import struct
from pathlib import Path

import numpy as np

from check_biprism_coupled_work_response import CoupledResponse


def inspect(execution, epoch, candidate):
    c = CoupledResponse(execution, epoch)
    c.h = struct.unpack('<f', struct.pack('<I', c.headers[0][10]))[0]
    with np.load(candidate, allow_pickle=False) as saved:
        if str(saved['execution_sha256']) != hashlib.sha256(Path(execution).read_bytes()).hexdigest():
            raise ValueError('CPU candidate is not bound to this native execution')
        if str(saved['numerical_sha256']) != hashlib.sha256(c.raw).hexdigest():
            raise ValueError('CPU candidate is not bound to this retained state')
        if int(saved['epoch']) != epoch or float(saved['h']) != c.h:
            raise ValueError('CPU candidate epoch/original timestep mismatch')
        x, scale = saved['x'].copy(), saved['scales'].copy()
    if x.shape != (4616,) or scale.shape != (8,) or not np.isfinite(x).all() or not (scale > 0.).all():
        raise ValueError('complete finite all-eight candidate required')
    v, alpha = x[:4608].reshape(8, 192, 3), x[4608:]/scale
    result = {}
    for represented in (False, True):
        r, work, z, d = c.nuclear_map(v, alpha, represented_velocity=represented, diagnostics=True)
        raw = .5*c.h*np.max(abs(r))
        norm = np.max(abs(np.sum(abs(z)**2, axis=-1)-np.sum(abs(c.z0)**2, axis=-1)))
        values = dict(raw_nuclear=float(raw), max_coordinate_work=float(np.max(abs(work))),
                      max_midpoint_work=float(np.max(abs(d['midpoint_defect']))),
                      direct_total_energy=float(d['direct_energy'].sum()),
                      joint_electronic_berry_foliated=float(d['joint_electronic_berry_foliated'].sum()),
                      velocity_lag=float(d['velocity_lag'].sum()),
                      max_partition_error=float(np.max(abs(d['partition_error']))),
                      minimum_midpoint_metric=float(d['metric_minimum'][:, 0].min()),
                      minimum_endpoint_metric=float(d['metric_minimum'][:, 1].min()),
                      norm_change=float(norm))
        label = 'rounded_endpoint_velocity' if represented else 'smooth_fp64_endpoint_velocity'
        result[label] = values
        print(label, values)
        if values['max_partition_error'] > 1e-8:
            raise ValueError('independent direct-energy/partition mismatch')
    smooth = result['smooth_fp64_endpoint_velocity']
    if not (smooth['raw_nuclear'] <= 5e-9 and smooth['max_coordinate_work'] <= 2e-7
            and smooth['max_midpoint_work'] <= 2e-7 and abs(smooth['direct_total_energy']) <= 1e-6
            and abs(smooth['joint_electronic_berry_foliated']) <= 1e-6
            and smooth['norm_change'] <= 1e-6 and smooth['minimum_midpoint_metric'] > 0.
            and smooth['minimum_endpoint_metric'] > 0.):
        raise ValueError('the smooth CPU root did not meet the stated numerical prediction')
    print('CPU_ORIGINAL_H_ROOT_RECONSTRUCTED=1 NATIVE_ACCEPTANCE=0 PHYSICAL_COMMITS=0')
    print('Native represented-spinor, endpoint rounding, layout, and original-workload controls remain required.')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execution', type=Path, required=True)
    p.add_argument('--epoch', type=int, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    a = p.parse_args()
    inspect(a.execution, a.epoch, a.candidate)
