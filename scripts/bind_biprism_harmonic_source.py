#!/usr/bin/env python3
"""Bind an unchanged KRAS input to exact-row, actually prepared harmonic semantics.

No topology is written or replaced. The output is an explicit input manifest
for biprism-run. Historical producer identity remains UNKNOWN; the current
OpenMM full-k/half-energy contract lowers once to native no-half coefficients.
No CUDA, provider rerun or physical execution is performed by this script.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import struct

def _env_root(name):
    """Absolute root from the environment; there is no compiled-in host default.
    When unset, a non-existent sentinel naming the variable is returned so every
    read fails closed; entry points call _require_env() first."""
    value = os.environ.get(name, '')
    return Path(value) if os.path.isabs(value) else Path('/nonexistent') / (name + '_UNSET')


def _require_env(*names):
    for name in names:
        if not os.path.isabs(os.environ.get(name, '')):
            raise SystemExit(f'{name} must name an absolute path (no compiled-in host default)')


BASE = _env_root('PRISM_SINGLE_CHAIN10_ROOT')
FRESH = BASE / 'INPUTS_MAY_FOUR_FRESH_1010_V1'
SOURCES = {
    'target_topology': (BASE / 'BASELINES/1010_MAY_FROZEN_20261016/INPUTS/KRAS/kras.topology.json',
                        'bc19e5f455552b6619b34e23fff55425c8c60f040168e89f21bbd028d527ff19'),
    'authority_topology': (FRESH / 'KRAS_A/KRAS_A.topology.json',
                           '4592ef511dc34174e409baebe96e9885de290dde48ce29c96ce6aad502def9bc'),
    'preparation_receipt': (FRESH / 'KRAS_A/KRAS_A.topology_prep_receipt.json',
                            'a0fc830c2fdce19a4b10684967ca982749ab5b93e26307eef555e061c0c3498f'),
    'preparation_start': (FRESH / 'OPERATOR_CONTROL/KRAS_A/02.START.json',
                          'a8f59042154b6724aa4c328d4843206e13aa30dcb7c8c0548ac2adfc5c6b6b48'),
    'preparation_execution': (FRESH / 'OPERATOR_CONTROL/KRAS_A/02.END.json',
                              '7e84383687521fac9a37bb0619a4b0fb751c90f23c2103d45f0bf5f88b837aa0'),
}
ATOM_IDENTITY = ('n_atoms', 'n_residues', 'n_chains', 'elements', 'atom_names',
                 'residue_names', 'residue_ids', 'chain_ids', 'ca_indices',
                 'residue_to_atom_indices', 'pdb_atom_serial_to_topo_index')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def strict_json(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, 'duplicate JSON key: ' + key)
            out[key] = value
        return out
    def constant(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()


def read(pin):
    require(set(pin) == {'path', 'sha256'}, 'invalid pinned file fields')
    path = Path(pin['path'])
    require(path.is_absolute() and stat.S_ISREG(path.lstat().st_mode), 'pin must name a regular absolute file')
    raw = path.read_bytes()
    require(sha(raw) == pin['sha256'], 'pinned source changed: ' + str(path))
    return raw


def contract_hash(value, field='contract_sha256'):
    body = dict(value)
    expected = body.pop(field)
    require(sha(canonical(body)) == expected, 'provider contract digest mismatch')


def harmonic_rows(topology, name):
    indices, values = (('i', 'j'), ('r0', 'k')) if name == 'bonds' else (('i', 'j', 'k_idx'), ('theta0', 'force_k'))
    rows = topology[name]
    out = bytearray(name.encode() + b'\0' + struct.pack('<Q', len(rows)))
    for i, row in enumerate(rows):
        out += struct.pack('<Q', i)
        for key in indices:
            require(type(row[key]) is int and 0 <= row[key] < topology['n_atoms'], 'harmonic atom index outside population')
            out += struct.pack('<Q', row[key])
        for key in values:
            out += struct.pack('<d', row[key])
    return bytes(out)


def derive_binding(pins):
    raw = {key: read(value) for key, value in pins.items()}
    docs = {key: strict_json(value) for key, value in raw.items()}
    target, authority = docs['target_topology'], docs['authority_topology']
    require('harmonic_parameter_contract' not in target, 'target already has a harmonic contract')
    for key in ATOM_IDENTITY:
        require(key in target and key in authority and target[key] == authority[key], 'atom identity differs: ' + key)
    digests = {}
    for name in ('bonds', 'angles'):
        actual = harmonic_rows(target, name)
        require(actual == harmonic_rows(authority, name), 'ordered harmonic index/f64 rows differ: ' + name)
        digests[name + '_f64_rows_sha256'] = sha(actual)
    h = authority['harmonic_parameter_contract']
    contract_hash(h)
    require(h['schema_kind'] == 'prismos_harmonic_parameter_contract' and h['schema_version'] == 1,
            'unsupported harmonic contract')
    require(h['bond_convention'] == h['angle_convention'] == 'OPENMM_FULL_K_HALF_ENERGY',
            'provider contract is not full-k/half-energy')
    require(h['bond_count'] == len(target['bonds']) and h['angle_count'] == len(target['angles']),
            'harmonic contract population mismatch')
    require(h['bond_coefficient_unit'] == 'kcal/mol/angstrom^2'
            and h['angle_coefficient_unit'] == 'kcal/mol/radian^2'
            and h['bond_equilibrium_unit'] == 'angstrom' and h['angle_equilibrium_unit'] == 'radian'
            and h['bond_uses_periodic_boundary_conditions'] is False
            and h['angle_uses_periodic_boundary_conditions'] is False, 'unsupported harmonic units/boundary')
    provenance = authority['topology_provenance']
    contract_hash(provenance)
    require(h['source_provider'] == 'OpenMM/' + provenance['openmm_version'] + ':HarmonicBondForce+HarmonicAngleForce',
            'provider version mismatch')
    producer = {'path': provenance['producer'], 'sha256': provenance['producer_sha256']}
    source_pdb = {'path': provenance['source_pdb'], 'sha256': provenance['source_pdb_sha256']}
    read(producer)
    read(source_pdb)
    resources = provenance['forcefield_resources']
    require(sha(canonical(resources['resources'])) == resources['resource_closure_sha256'],
            'forcefield resource closure digest mismatch')
    bound_resources = []
    names = set()
    for row in resources['resources']:
        require(row['resource'] not in names, 'duplicate forcefield resource')
        names.add(row['resource'])
        pin = {'path': str(Path(resources['openmm_data_root']) / row['resource']), 'sha256': row['sha256']}
        require(len(read(pin)) == row['size_bytes'], 'forcefield resource size mismatch')
        bound_resources.append({'resource': row['resource'], 'file': pin})
    require(bound_resources, 'forcefield resource closure absent')
    receipt = docs['preparation_receipt']
    contract_hash(receipt, 'receipt_payload_sha256')
    require(receipt['schema_kind'] == 'prismos_topology_preparation_receipt' and receipt['schema_version'] == 2
            and receipt['status'] == 'PASS', 'preparation not accepted')
    require(receipt['topology_provenance_sha256'] == provenance['contract_sha256']
            and receipt['scientific_parameter_topology_sha256']
            == authority['scientific_parameter_topology_contract']['scientific_parameter_topology_sha256'],
            'preparation scientific/provenance identity mismatch')
    require(any(row['role'] == 'topology' and row['path'] == pins['authority_topology']['path']
                and row['sha256'] == pins['authority_topology']['sha256']
                and row['size_bytes'] == len(raw['authority_topology']) for row in receipt['artifacts']),
            'preparation receipt does not bind the authority bytes')
    start, execution = docs['preparation_start'], docs['preparation_execution']
    require(start['event'] == 'CHILD_STARTED' and execution['event'] == 'CHILD_FINISHED'
            and execution['exit_code'] == 0 and execution['pid'] > 0
            and all(start[key] == execution[key] for key in ('pid', 'stage', 'started_utc')), 'actual preparation failed/unmatched')
    argv = start['argv']
    require(any(argv[i:i+3] == [producer['path'], source_pdb['path'], pins['authority_topology']['path']]
                for i in range(len(argv) - 2)), 'actual preparation command mismatch')
    for key in ('authority_topology', 'preparation_receipt'):
        require(any(row['path'] == pins[key]['path'] and row['sha256'] == pins[key]['sha256']
                    and row['bytes'] == len(raw[key]) for row in execution['outputs']), 'actual preparation output mismatch')
    return dict(schema='prism.harmonic_source_binding.v1', **pins, producer=producer, source_pdb=source_pdb,
                forcefield_resources=bound_resources, **digests, historical_producer_verification='UNKNOWN')


def verify_binding(pin, target_pin):
    manifest = strict_json(read(pin))
    require(manifest['target_topology'] == target_pin, 'binding targets different topology')
    pins = {key: manifest[key] for key in SOURCES}
    require(manifest == derive_binding(pins), 'binding manifest differs from exact sources')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(args.output.is_absolute() and args.output.parent.is_dir(), 'output parent must exist and be absolute')
    pins = {key: {'path': str(path), 'sha256': digest} for key, (path, digest) in SOURCES.items()}
    binding = derive_binding(pins)
    raw = (json.dumps(binding, indent=2, sort_keys=True) + '\n').encode()
    with args.output.open('xb') as file:
        file.write(raw)
        file.flush()
        os.fsync(file.fileno())
    fd = os.open(args.output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    print(json.dumps({'binding': {'path': str(args.output), 'sha256': sha(raw)},
                      'bond_count': 2706, 'angle_count': 4884,
                      'historical_producer_verification': 'UNKNOWN',
                      'native_no_half_lowering': {'bond_scale': .5, 'angle_scale': .5},
                      'topology_replaced': False, 'native_execution': False}, indent=2))


if __name__ == '__main__':
    main()
