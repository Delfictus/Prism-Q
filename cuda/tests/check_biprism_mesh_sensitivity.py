#!/usr/bin/env python3
"""Quantify actual native retained-domain/DEC differences across mesh budgets.

Every target is a fresh BW + Phase2 + cold DEC initialization on the same
physical atoms/coordinates. Domain changes are measured, never described as
fixed-domain refinement. There is no sensitivity acceptance threshold, order
of convergence claim, physical clock advance or trajectory acceptance.
"""
import argparse
from fractions import Fraction as Q
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import uuid

# The root runner uses isolated Python. Import only these build-pinned sibling
# readers, not a caller's working directory or an ambient module search path.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_biprism_boundary_identity as B
import check_biprism_v10_dec_readback as DEC

FACTOR = 1.8897261259077824
EXPECTED_PHASE2 = '020798119f892ce5ae0cd8305cc773a558b0a2b299b7e02cbbc5568ac03de82b'
EXPECTED_PRECISE_PHASE2 = '79df130090915fa21605fa99d8cfe7c51a7e1f36b1afeba1e2868e27cd1788a2'
EXPECTED_BW = '255d925e1c37fb03d880a51365b43f1bf331adb130c2825171d7e4f0274f4a8a'
CORE, VERTICES, LEAVES, TILE = 192, 256, 8, DEC.TILE_BYTES


def require(ok, why):
    if not ok:
        raise ValueError(why)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fbits(x):
    return struct.pack('<f', x)


def source_snapshots(path):
    """Explicit historical-build replay; never silently fall back to other bytes."""
    result = {}
    if path is not None:
        for line in Path(path).read_text().splitlines():
            original, snapshot, digest = line.split('\t')
            require(original not in result and len(digest) == 64, 'duplicate/invalid source snapshot')
            require(sha(snapshot) == digest, 'source snapshot bytes changed: '+snapshot)
            result[original] = (Path(snapshot), digest)
    return result


def bound_source(path, digest, snapshots):
    path = Path(path)
    selected, declared = snapshots.get(str(path), (path, digest))
    require(declared == digest and sha(selected) == digest, 'compiled source identity changed: '+str(path))
    return selected


def seed_and_halo_identity(seed, provisional, endpoint, ids, charges):
    """Actual BW cold potential, followed by unchanged prescribed halo values."""
    phi_offset = DEC.OFF['phi']
    factor = DEC.f32(0.07957747)
    expected = b''.join(fbits(DEC.f32(charges[atom]*factor)) for atom in ids)
    require(seed[phi_offset:phi_offset+4*VERTICES] == expected,
            'seed potential differs from represented charge times BW factor')
    first, last = phi_offset+4*CORE, phi_offset+4*VERTICES
    halo = endpoint[first:last]
    require(halo == provisional[first:last] == seed[first:last], 'prescribed cold halo changed')
    return halo


def scalar_import(native):
    """Cold zero-low scale: exact FMA product remainder and declared FP32 pair."""
    product = Q(native)*Q(FACTOR)
    high64 = float(product)
    remainder = float(product-Q(high64))
    high = DEC.f32(high64)
    low = DEC.f32(float(Q(high64)-Q(high)+Q(remainder)))
    return high, low


def tet_volume(points, tet):
    a, b, c, d = (points[v] for v in tet)
    u, v, w = ([p[k]-a[k] for k in range(3)] for p in (b, c, d))
    determinant = (u[0]*(v[1]*w[2]-v[2]*w[1])
                   - u[1]*(v[0]*w[2]-v[2]*w[0])
                   + u[2]*(v[0]*w[1]-v[1]*w[0]))
    require(determinant != 0, 'zero-volume retained tetrahedron')
    return abs(determinant)/6


def sparse_operator(edges, weights):
    result = {}
    for (u, v), weight in zip(edges, weights):
        w = Q(weight)
        for key, value in (((u, u), w), ((v, v), w), ((u, v), -w), ((v, u), -w)):
            result[key] = result.get(key, Q(0))+value
    return result


def differences(base, candidate):
    """Quantitative differences; none of these values is an acceptance limit."""
    require(base['ids'] == candidate['ids'] and base['charge'] == candidate['charge'],
            'comparison changes physical nodes or source charge')
    require(base['coordinates'] == candidate['coordinates'], 'comparison changes physical coordinates')
    require(base['halo_bits'] == candidate['halo_bits'], 'comparison changes prescribed halo values')
    l0, l1 = base['operator'], candidate['operator']
    op_squared = sum(((l1.get(k, Q(0))-l0.get(k, Q(0)))**2 for k in l0.keys() | l1.keys()), Q(0))
    p0, p1 = list(map(Q, base['phi'][:CORE])), list(map(Q, candidate['phi'][:CORE]))
    omega_delta = [Q(a)-Q(b) for a, b in zip(candidate['star0'], base['star0'])]
    return {'tetrahedra_added': len(candidate['tets']-base['tets']),
            'tetrahedra_removed': len(base['tets']-candidate['tets']),
            'edges_added': len(candidate['edge_set']-base['edge_set']),
            'edges_removed': len(base['edge_set']-candidate['edge_set']),
            'volume_delta_bohr3': candidate['volume_au']-base['volume_au'],
            'volume_delta_angstrom3': candidate['volume_native']-base['volume_native'],
            'star0_delta_l1': sum(map(abs, omega_delta), Q(0)),
            'operator_delta_frobenius': math.sqrt(float(op_squared)),
            'phi_core_delta_l2': math.sqrt(float(sum(((a-b)**2 for a, b in zip(p1, p0)), Q(0)))),
            'phi_core_delta_linf': float(max(abs(a-b) for a, b in zip(p1, p0))),
            'functional_delta': candidate['functional']-base['functional']}


def check(execution, budgets, snapshot_index=None):
    require(len(budgets) == 3 and 0 < budgets[0] < budgets[1] == 960 < budgets[2] <= 1024,
            'diagnostic targets must bracket960')
    execution = Path(execution).resolve()
    dispatch_hash = sha(execution)
    d = json.loads(execution.read_text())
    require(d.get('native_execution') is True and type(d.get('native_exit_code')) is int
            and d['native_exit_code'] == 0, 'successful actual native dispatch required')
    require(d.get('sources_binary_inputs_unchanged') is True and d.get('boot_unchanged') is True
            and d.get('gpu_recovery_none_after') is True and d.get('new_kernel_fault_records') == [],
            'dispatch identity or GPU health refusal')
    uuid.UUID(d['boot_id'])
    require(d.get('mesh_budgets') == budgets and d.get('physical_trajectory_acceptance') is False,
            'diagnostic scope/target identity mismatch')
    command = d['command']
    require(len(command) == 5 and command[4] == '--mesh-budgets='+','.join(map(str, budgets)),
            'native command target mismatch')
    pin = d['build_receipt']; build_path = Path(pin['path'])
    require(sha(build_path) == pin['sha256'], 'build receipt hash mismatch')
    build = json.loads(build_path.read_text())
    require(build.get('exit_code') == 0 and build.get('sources_unchanged') is True
            and build.get('header_overlays') is False and build.get('invoked_entry_reports_present') is True
            and build.get('mesh_budgets') == budgets, 'actual canonical build required')
    require(sha(command[0]) == build['binary_sha256'] == d['input_hashes'].get(command[0]),
            'binary build/dispatch identity mismatch')
    snapshots = source_snapshots(snapshot_index)
    require(all(p in build['input_hashes'] for p in snapshots), 'snapshot not in actual build population')
    # These sources define every byte offset and numerical tree used below.
    phase2_digest = build['input_hashes'].get(str(DEC.KERNELS/'nhs_biprism_phase2_work.cuh'))
    require(phase2_digest in (EXPECTED_PHASE2, EXPECTED_PRECISE_PHASE2), 'unreviewed Phase2 potential realization')
    precise = phase2_digest == EXPECTED_PRECISE_PHASE2
    known = {DEC.KERNELS/'nhs_biprism_simplicial_tile.cuh': B.EXPECTED_TILE_LAYOUT_SOURCE,
             DEC.KERNELS/'nhs_biprism_phase2_work.cuh': phase2_digest,
             DEC.KERNELS/'nhs_biprism_bowyer_watson_seed.cuh': EXPECTED_BW,
             DEC.KERNELS/'nhs_biprism_dec_pcg.cuh': DEC.EXPECTED_PRECISE_DEC if precise else DEC.EXPECTED_DEC,
             DEC.KERNELS/'nhs_biprism_electronic_discrete.cuh': DEC.EXPECTED_ELECTRONIC,
             DEC.KERNELS/'nhs_biprism_coordinate_geometry.cuh': DEC.EXPECTED_GEOMETRY,
             DEC.KERNELS/'nhs_biprism_coordinate_precision.cuh': DEC.EXPECTED_COORDINATE}
    if precise:
        known[DEC.KERNELS/'nhs_biprism_potential_precision.cuh'] = DEC.EXPECTED_POTENTIAL
        known[DEC.KERNELS/'nhs_biprism_ordered_edges.cuh'] = DEC.EXPECTED_ORDERED_EDGES
    for path, digest in known.items():
        require(sha(bound_source(path, digest, snapshots)) == digest == build['input_hashes'].get(str(path))
                == d['input_hashes'].get(str(path)), 'unreviewed numerical/layout source: '+str(path))
    source_pins = {}
    # Imported oracle implementations must still be the exact built versions.
    # Only this explicitly identified analysis program can be a repaired reader.
    for path in (Path(B.__file__), Path(DEC.__file__)):
        digest = sha(path)
        require(build['input_hashes'].get(str(path)) == digest == d['input_hashes'].get(str(path)),
                'independent reader is not build/dispatch-bound: '+str(path))
        source_pins[str(path)] = digest
    original_reader_digest = build['input_hashes'][str(Path(__file__))]
    original_reader = bound_source(Path(__file__), original_reader_digest, snapshots)
    source_pins[str(Path(__file__))] = sha(__file__)
    # Reconcile the full source population in the real build, including actual
    # import/Phase2 helpers, rather than accepting hashes for only this reader.
    for path, digest in build['input_hashes'].items():
        selected = bound_source(path, digest, snapshots)
        require(d['input_hashes'].get(path) == digest,
                'compiled source population changed: '+path)
        source_pins[str(selected)] = digest
    if snapshot_index is not None:
        source_pins[str(Path(snapshot_index))] = sha(snapshot_index)
    pins = {str(execution): dispatch_hash, str(build_path): pin['sha256'],
            command[0]: build['binary_sha256'], **source_pins}

    def read(path):
        path = Path(path)
        raw, digest = B.bound_bytes(path, d)
        pins[str(path)] = digest
        return raw

    input_raw = read(command[1]); old_seed = read(command[2])
    require(input_raw[:8] == b'BPCASE02' and len(old_seed) == 8*TILE, 'physical input extent/format')
    atoms, = struct.unpack_from('<I', input_raw, 8)
    amber = DEC.read_charges(command[1])
    require(atoms == len(amber), 'charge population')
    native_input = struct.unpack_from('<'+str(3*atoms)+'f', input_raw, 32)
    elements = input_raw[32+20*atoms:32+21*atoms]
    require(len(elements) == atoms and all(e < 8 for e in elements), 'physical element classes')
    original_selected = [DEC.parse_tile(old_seed[l*TILE:(l+1)*TILE])[0] for l in range(8)]
    leaves_by_target = {}
    common_planes = None
    lines = ['MESH_SENSITIVITY=actual native cold operators; quantitative comparison only',
             'PRODUCTION_TARGET=960 UNCHANGED=true PHYSICAL_STEPS=0 TRAJECTORY_ACCEPTED=false',
             'DOMAIN_SCOPE=retained tetrahedral subcomplex; changing target can change domain',
             'NO_FIXED_DOMAIN_REFINEMENT_OR_CONVERGENCE_ORDER_OR_FULL_HULL_EQUIVALENCE_CLAIM=true',
             'NO_NEW_SENSITIVITY_ACCEPTANCE_LIMIT=true',
             f'EXECUTION={execution} SHA256={dispatch_hash}', f'BUILD={build_path} SHA256={pin["sha256"]}',
             f'BINARY={command[0]} SHA256={build["binary_sha256"]}', f'BOOT_ID={d["boot_id"]}',
             f'ORIGINAL_DISPATCH_READER={original_reader} SHA256={original_reader_digest}',
             f'CURRENT_ANALYSIS_READER={Path(__file__)} SHA256={sha(__file__)}',
             f'HISTORICAL_SOURCE_REPLAY={snapshot_index is not None} NEW_GPU_EXECUTION=false']
    if snapshot_index is not None:
        lines.append(f'SOURCE_SNAPSHOT_INDEX={snapshot_index} SHA256={sha(snapshot_index)}')
    for target in budgets:
        prefix = command[3]+'.target'+str(target)
        row_size = 224 if precise else 208
        raw = {name: read(prefix+'.'+suffix) for name, suffix in {
            'target': 'requested_target.u32le', 'ids': 'selected_atoms.u32le',
            'status': 'bw_status.80b', 'seed': 'seed_tiles.bin', 'phase2': 'phase2_tiles.bin',
            'endpoint': 'initial_tiles.bin', 'ledger': 'initial_ledgers.240b',
            'rows': f'initial_vertices.{row_size}b', 'conv': 'initial_convergence.128b',
            'step': 'initial_step.u32le', 'bridge': 'bridge.64b',
            'native_owner': 'coordinate_owners_native.96b', 'au_owner': 'coordinate_owners_au.96b'}.items()}
        extents = {'target': 4, 'ids': 8*256*4, 'status': 8*80, 'seed': 8*TILE,
                   'phase2': 8*TILE, 'endpoint': 8*TILE, 'ledger': 8*240, 'rows': 8*256*row_size,
                   'conv': 8*128, 'step': 4, 'bridge': 8*64, 'native_owner': 8*96, 'au_owner': 8*96}
        require(all(len(raw[n]) == size for n, size in extents.items()), 'native output extent')
        full_phi = None
        if precise:
            raw['potential_owner'] = read(prefix+'.initial_potential_owners.128b')
            raw['accepted_phi'] = read(prefix+'.initial_phi_accepted.f64le')
            raw['prepared_phi'] = read(prefix+'.initial_phi_prepared.f64le')
            require(len(raw['potential_owner']) == 8*128 and len(raw['accepted_phi']) == len(raw['prepared_phi']) == 8*2048,
                    'retained potential extent')
            full_phi = struct.unpack('<2048d', raw['accepted_phi'])
            require(all(map(math.isfinite, full_phi)) and not any(raw['prepared_phi']), 'cold full potential or prepared bank')
            bank_intervals=[]
            for leaf in range(8):
                o=raw['potential_owner'][leaf*128:(leaf+1)*128]
                require(struct.unpack_from('<4I2Q',o)==(0x31505042,1,leaf,1,23,1)
                        and struct.unpack_from('<4I2Q4I2Q',o,64)==(0,0,0,0,1,0,1,0,0,0,1,0),
                        'cold accepted potential identity')
                for p in struct.unpack_from('<2Q',o,48):
                    require(p and p%8==0 and p+2048<2**64,'full potential pointer')
                    bank_intervals.append((p,p+2048))
            bank_intervals.sort()
            require(all(a[1]<=b[0] for a,b in zip(bank_intervals,bank_intervals[1:])), 'full potential bank alias')
        require(struct.unpack('<I', raw['target'])[0] == target and raw['step'] == bytes(4),
                'target/zero-clock identity')
        planes = {}
        for name in ('coordinates_au', 'coordinates_au_low', 'reference_au', 'reference_au_low',
                     'coordinates_native', 'coordinates_native_low'):
            block = read(prefix+'.'+name+'.f32le')
            require(len(block) == 8*atoms*12, 'coordinate extent: '+name)
            values = struct.unpack('<'+str(24*atoms)+'f', block)
            require(all(map(math.isfinite, values)), 'nonfinite coordinate plane')
            planes[name] = values
        if common_planes is None:
            common_planes = planes
        else:
            require(planes == common_planes, 'budget variation changed physical coordinate inputs')
        require(planes['coordinates_native'] == native_input*8
                and all(v == 0 for v in planes['coordinates_native_low']), 'cold native input differs')
        require(planes['coordinates_au'] == planes['reference_au']
                and planes['coordinates_au_low'] == planes['reference_au_low'], 'cold mesh reference differs')
        require(planes['coordinates_au'] == planes['coordinates_au'][:3*atoms]*8
                and planes['coordinates_au_low'] == planes['coordinates_au_low'][:3*atoms]*8,
                'identical native leaves did not receive identical atomic coordinates')
        for x, high, low in zip(native_input, planes['coordinates_au'][:3*atoms], planes['coordinates_au_low'][:3*atoms]):
            expected = scalar_import(x)
            require(fbits(high) == fbits(expected[0]) and fbits(low) == fbits(expected[1]),
                    'native to atomic coordinate expansion mismatch')
        intervals = []
        for frame, key in ((1, 'native_owner'), (2, 'au_owner')):
            for leaf in range(8):
                o = raw[key][leaf*96:(leaf+1)*96]
                require(struct.unpack_from('<4I', o) == (0x31504342, 1, leaf, atoms)
                        and struct.unpack_from('<QII', o, 16) == (23, frame, 7)
                        and not any(o[80:]), 'coordinate owner metadata')
                for p in struct.unpack_from('<6Q', o, 32):
                    require(p and p % 4 == 0 and p+12*atoms < 2**64, 'coordinate pointer domain')
                    intervals.append((p, p+12*atoms))
        intervals.sort()
        require(all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:])), 'coordinate owner plane alias')
        leaves = []
        for leaf in range(8):
            status = struct.unpack_from('<20I', raw['status'], leaf*80)
            require(status[:5] == (0x42574231, leaf, 0, 5, 256) and status[16] == 1
                    and 0 < status[8] <= target and 0 < status[9] <= 1536
                    and status[10] <= 256 and status[12] == 0 and not any(status[17:]), 'BW publication refused')
            conv = struct.unpack_from('<32I', raw['conv'], leaf*128)
            require(conv[:6] == (0x42505631, leaf, 0, 0, 0, 0), 'cold convergence/clock refusal')
            bridge = struct.unpack_from('<16I', raw['bridge'], leaf*64)
            require(bridge[:5] == (0x42554231, leaf, 0, 0, 7) and bridge[8:12] == (1, 0, atoms, 0),
                    'actual cold import not admitted')
            ledger = raw['ledger'][leaf*240:(leaf+1)*240]
            require(struct.unpack_from('<8I', ledger) == (0x42505733 if precise else 0x42505732, leaf, 4, 0, 0, 0, status[9], status[8]),
                    'cold Phase2/DEC ledger identity')
            reported, = struct.unpack_from('<f', ledger, 64)
            require(math.isfinite(reported) and reported <= DEC.f32(1e-6), 'reported cold DEC refusal')
            seed = raw['seed'][leaf*TILE:(leaf+1)*TILE]
            provisional = raw['phase2'][leaf*TILE:(leaf+1)*TILE]
            endpoint = raw['endpoint'][leaf*TILE:(leaf+1)*TILE]
            ids, edges, psi, phi = DEC.parse_tile(endpoint)
            if precise:
                phi = list(full_phi[leaf*256:(leaf+1)*256])
                require(endpoint[DEC.OFF['phi']:DEC.OFF['phi']+1024] == b''.join(fbits(p) for p in phi),
                        'legacy tile differs from full accepted projection')
            require(ids == original_selected[leaf] == list(struct.unpack_from('<256I', raw['ids'], leaf*1024)),
                    'native mesh changes physical selected IDs')
            require(list(struct.unpack_from('<256I', endpoint)) ==
                    [atom | (elements[atom] << 24) for atom in ids], 'physical element/atom mapping differs')
            require(seed[:94208] == endpoint[:94208] and seed[225280:225288] == endpoint[225280:225288],
                    'Phase2 changes native topology/references/spinors')
            domain = B.retained_domain(provisional, endpoint, edges)
            require(domain['tetrahedra'] == status[8] and domain['edges'] == status[9], 'published BW count disagreement')
            halo = seed_and_halo_identity(seed, provisional, endpoint, ids, amber)
            star0 = struct.unpack_from('<256f', endpoint, B.STAR0_OFFSET)
            bases = struct.unpack_from('<'+str(len(edges))+'f', endpoint, DEC.OFF['base_weights'])
            require(all(math.isfinite(w) and w > 0 for w in star0+bases), 'assembled operator is not positive finite')
            cp = [DEC.coordinate_at(planes['coordinates_au'], planes['coordinates_au_low'], leaf, a, atoms) for a in ids]
            np = [DEC.coordinate_at(planes['coordinates_native'], planes['coordinates_native_low'], leaf, a, atoms) for a in ids]
            weights = [DEC.precise_weight(w, cp[u], cp[v], cp[u], cp[v]) for (u, v), w in zip(edges, bases)]
            charges = [DEC.represented_charge(amber[ids[v]], psi[v][2:4]) for v in range(CORE)]
            exact = DEC.exact_operator(charges, phi, edges, weights)
            require(exact['squared'] <= Q.from_float(DEC.TOL_SQ) and exact['squared'] <= Q(1, 10**12),
                    'unchanged true DEC residual gate failed')
            for v in range(256):
                row = raw['rows'][(leaf*256+v)*row_size:(leaf*256+v+1)*row_size]
                packed, = struct.unpack_from('<I', endpoint, 4*v)
                require(struct.unpack_from('<4I', row) == (packed, leaf, v, 4), 'actual cold vertex identity')
                require(row[16:28] == struct.pack('<3f', *(p[0] for p in cp[v]))
                        and row[192 if precise else 176:204 if precise else 188] == struct.pack('<3f', *(p[1] for p in cp[v])), 'cold row coordinate mismatch')
                q, = struct.unpack_from('<d', row, 104 if precise else 88)
                require(q == (charges[v] if v < CORE else amber[ids[v]]), 'cold row charge mismatch')
                require((row[72:80] == struct.pack('<d',phi[v])) if precise else (row[68:72] == fbits(phi[v])), 'cold row potential mismatch')
                if precise:
                    entry,=struct.unpack_from('<d',row,216)
                    provisional_phi,=struct.unpack_from('<f',provisional,DEC.OFF['phi']+4*v)
                    require(entry==float(provisional_phi) and row[68:72]==bytes(4), 'cold provisional entry identity')
            if precise:
                require(struct.pack('<64d',*phi[192:]) == b''.join(struct.pack('<d',x[0]) for x in struct.iter_unpack('<f',halo)),
                        'full accepted halo changed prescribed boundary')
                entry_phi=[struct.unpack_from('<d',raw['rows'],(leaf*256+v)*row_size+216)[0] for v in range(256)]
                entry=DEC.exact_operator(charges,entry_phi,edges,weights)
                diagnostic,=struct.unpack_from('<d',ledger,232)
                expected=float(exact['functional']-entry['functional'])
                require(math.isfinite(diagnostic) and abs(diagnostic-expected)<=1e-9*(1+abs(expected)),
                        'private entry refinement diagnostic differs')
            tets = list(struct.iter_unpack('<4B', endpoint[B.TET_OFFSET:B.TET_OFFSET+4*status[8]]))
            points_au = [tuple(Q(hi)+Q(lo) for hi, lo in point) for point in cp]
            points_native = [tuple(Q(hi)+Q(lo) for hi, lo in point) for point in np]
            volume_au = sum((tet_volume(points_au, t) for t in tets), Q(0))
            volume_native = sum((tet_volume(points_native, t) for t in tets), Q(0))
            leaf_data = {'ids': ids, 'charge': charges, 'coordinates': cp, 'halo_bits': halo,
                         'tets': {tuple(sorted(t)) for t in tets}, 'edge_set': {tuple(sorted(e)) for e in edges},
                         'operator': sparse_operator(edges, weights), 'phi': phi, 'star0': star0,
                         'volume_au': volume_au, 'volume_native': volume_native, 'functional': exact['functional']}
            leaves.append(leaf_data)
            lines.append(f'TARGET={target} LEAF={leaf} TETS={status[8]} EDGES={status[9]} PEEL_PASSES={status[10]} '
                         f'EXPOSED_FACES={domain["exposed_faces"]} BOUNDARY_VERTICES={domain["geometric_boundary_vertices"]} '
                         f'VOLUME_BOHR3={float(volume_au):.17g} VOLUME_ANGSTROM3={float(volume_native):.17g} '
                         f'STAR0_SUM={sum(star0):.17g} DEC_TRUE_L2={math.sqrt(float(exact["squared"])):.17g} '
                         f'DEC_FUNCTIONAL={float(exact["functional"]):.17g}')
        leaves_by_target[target] = leaves
    for target in budgets:
        for leaf, state in enumerate(leaves_by_target[target]):
            delta = differences(leaves_by_target[960][leaf], state)
            lines.append(f'DIFFERENCE TARGET={target} BASELINE=960 LEAF={leaf} '+
                         ' '.join(f'{key}={float(value):.17g}' for key, value in delta.items()))
    for path, digest in pins.items():
        require(sha(path) == digest, 'source/raw input changed while quantifying: '+path)
    lines.append('ALL8_COLD_OPERATORS_PER_TARGET=PASS SENSITIVITY_LIMIT_APPLIED=false')
    lines.append('CHECKER_SHA256='+sha(__file__))
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution', required=True, type=Path)
    parser.add_argument('--budgets', required=True, nargs=3, type=int)
    parser.add_argument('--source-snapshots', type=Path,
                        help='Explicit tab-separated original-path/snapshot-path/SHA256 map for historical readback replay')
    args = parser.parse_args()
    print(check(args.execution, args.budgets, args.source_snapshots), end='')


if __name__ == '__main__':
    main()
