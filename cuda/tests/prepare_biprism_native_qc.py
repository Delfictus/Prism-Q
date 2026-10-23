#!/usr/bin/env python3
"""Lossless identity-bound transport of real topology inputs to native QC.

This prepares data only. It does not synthesize neighbors, tessellations,
velocities, trajectories, physical masses, or scientific acceptance.
"""
import argparse
import hashlib
import json
import math
import pathlib
import struct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topology', type=pathlib.Path, required=True)
    ap.add_argument('--center', type=int, required=True)
    ap.add_argument('--output', type=pathlib.Path, required=True)
    args = ap.parse_args()
    raw = args.topology.read_bytes()
    t = json.loads(raw)
    n = t['n_atoms']
    if not isinstance(n, int) or not 256 <= n <= 100000:
        raise ValueError('diagnostic atom count must be within 256..100000')
    if not 0 <= args.center < n:
        raise ValueError('center outside original atom population')
    n_residues = t['n_residues']
    if not isinstance(n_residues, int) or not 1 <= n_residues <= n:
        raise ValueError('invalid original residue population')
    for key, count in [('positions', 3*n), ('charges', n), ('masses', n), ('elements', n), ('exclusions', n), ('residue_ids', n)]:
        if len(t[key]) != count:
            raise ValueError(f'{key}: unexpected population length')
    if any(not isinstance(r, int) or not -1 <= r < n_residues for r in t['residue_ids']):
        raise ValueError('residue identity outside original signed population')
    for key in ['positions', 'charges', 'masses']:
        if not all(math.isfinite(x) for x in t[key]):
            raise ValueError(f'{key}: nonfinite source value')
    if not all(x > 0 for x in t['masses']):
        raise ValueError('nonpositive source mass')
    classes = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'P': 4, 'S': 5, 'F': 6, 'CL': 6}
    unknown = sorted({str(e) for e in t['elements'] if not isinstance(e, str) or e.upper() not in classes})
    if unknown:
        raise ValueError(f'element mapping requires an explicit model definition: {unknown}')
    elements = bytes(classes[e.upper()] for e in t['elements'])
    offsets, exclusions = [0], []
    for row in t['exclusions']:
        if any(not isinstance(v, int) or not 0 <= v < n for v in row):
            raise ValueError('exclusion atom identity outside original population')
        exclusions.extend(row)
        offsets.append(len(exclusions))
    groups = []
    for g in t['h_clusters']:
        central = g['central_atom']
        hs = g['hydrogen_atoms'][:g['n_hydrogens']]
        if not 0 <= central < n or not 0 <= g['n_hydrogens'] <= 3 or len(hs) != g['n_hydrogens']:
            raise ValueError('invalid native constraint group')
        if any(not 0 <= h < n for h in hs):
            raise ValueError('constraint hydrogen outside source population')
        groups.append((central, hs))
    flags = int(bool(t.get('hmr_applied', False)))
    blob = bytearray(struct.pack('<8s6I', b'BPCASE02', n, args.center, len(exclusions), len(groups), flags, n_residues))
    for key in ['positions', 'charges', 'masses']:
        blob += struct.pack(f'<{len(t[key])}f', *t[key])
    blob += elements
    blob += struct.pack(f'<{n}i', *t['residue_ids'])
    blob += struct.pack(f'<{len(offsets)}i', *offsets)
    blob += struct.pack(f'<{len(exclusions)}i', *exclusions)
    for central, hs in groups:
        blob += struct.pack('<2I', central, len(hs))
        blob += struct.pack(f'<{len(hs)}I', *hs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as f:
        f.write(blob)
    receipt = {
        'scope': 'Original topology serialized to native FP32 input; no neighbor or mesh generation',
        'source': {'path': str(args.topology.resolve()), 'sha256': hashlib.sha256(raw).hexdigest()},
        'transport': {'path': str(args.output.resolve()), 'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)},
        'script_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        'n_atoms': n, 'center_atom_id': args.center, 'center_scope': 'Explicit diagnostic center, not an inferred active pocket',
        'positions_unit': 'angstrom as in native topology', 'charges_unit': 'native topology charge units',
        'hmr_applied': bool(flags), 'masses_preserved': True, 'physical_mass_array_created': False,
        'constraint_groups': len(groups), 'exclusion_entries': len(exclusions),
        'residue_identity': 'Original signed residue_ids preserved; -1 remains unassigned; no atom-index fallback',
        'n_residues': n_residues, 'transport_abi': 'BPCASE02',
        'element_classes': classes, 'unmapped_elements': 'rejected before transport',
        'rounding': 'One serialization to IEEE binary32 for native float arrays; original JSON retained by source hash',
        'gpu_execution': False, 'scientific_acceptance': False,
    }
    with pathlib.Path(str(args.output)+'.json').open('x') as f:
        json.dump(receipt, f, indent=2)
        f.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
