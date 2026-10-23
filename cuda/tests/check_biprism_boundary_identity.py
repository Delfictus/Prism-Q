#!/usr/bin/env python3
"""Check graph boundary identities on one provenance-bound accepted native epoch.

The 192 free core nodes and 64 prescribed halo nodes define an algebraic
Dirichlet partition. They are not asserted to coincide with the geometric
boundary of the retained tetrahedral complex. This reader performs no solve,
mesh change, native execution or physical trajectory acceptance.
"""
import argparse
from collections import Counter
from fractions import Fraction as Q
import hashlib
from itertools import combinations
import json
from pathlib import Path
import biprism_exchange_layout as EX
import struct
import check_biprism_spinor_owner as SPINOR

import check_biprism_v10_accepted_prefix as PREFIX
import check_biprism_v10_dec_readback as DEC


EXPECTED_ORACLE = '1b821d0e38ebb479fec3b4c18f6db5afe41957975ed05f5a429e7073b15cb8da'
EXPECTED_PREFIX = '92a19786825cb8c91ec5684c2588dd32304ba04eb717341e51cbaa91050b6ae5'
EXPECTED_TILE_LAYOUT_SOURCE = 'e89d6957b3fd37090ad5d24c8a80c54590e63e9f0298d32d4eca7303f03a8b2e'
TET_OFFSET, TET_COUNT_OFFSET, MAX_TETS = 5120, 225280, 1024
TET_EDGE_OFFSET, TET_OPPOSITE_OFFSET, STAR0_OFFSET = 9216, 21504, 95232
CORE, VERTICES, LEAVES = DEC.CORE, DEC.VERTICES, DEC.LEAVES


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rational(value):
    return f'{value.numerator}/{value.denominator}'


def retained_domain(before, after, edges):
    """Read incidence and boundary membership, without a new geometry claim."""
    require(len(before) == len(after) == DEC.TILE_BYTES, 'domain tile extent')
    count, = struct.unpack_from('<I', after, TET_COUNT_OFFSET)
    require(0 < count <= MAX_TETS, 'active tetrahedron count')
    require(before[TET_COUNT_OFFSET:TET_COUNT_OFFSET+8]
            == after[TET_COUNT_OFFSET:TET_COUNT_OFFSET+8], 'active domain counts changed')
    spans = [('physical IDs', 0, 4*VERTICES),
             ('tetrahedra', TET_OFFSET, 4*count),
             ('tetrahedron edge incidence', TET_EDGE_OFFSET, 12*count),
             ('tetrahedron adjacency', TET_OPPOSITE_OFFSET, 8*count),
             ('edge incidence', DEC.OFF['edges'], 2*len(edges)),
             ('vertex volumes', STAR0_OFFSET, 4*VERTICES),
             ('base edge weights', DEC.OFF['base_weights'], 4*len(edges))]
    for label, offset, size in spans:
        require(before[offset:offset+size] == after[offset:offset+size],
                'retained operator changed: '+label)
    tets = list(struct.iter_unpack('<4B', after[TET_OFFSET:TET_OFFSET+4*count]))
    require(all(len(set(t)) == 4 for t in tets), 'degenerate tetrahedron incidence')
    require(len({tuple(sorted(t)) for t in tets}) == count, 'duplicate tetrahedron incidence')
    tet_edges = {tuple(sorted(e)) for t in tets for e in combinations(t, 2)}
    require(len(set(tuple(sorted(e)) for e in edges)) == len(edges)
            and tet_edges == {tuple(sorted(e)) for e in edges},
            'DEC edge population differs from retained tetrahedral incidence')
    faces = Counter(tuple(sorted(face)) for t in tets for face in combinations(t, 3))
    require(all(m in (1, 2) for m in faces.values()), 'nonmanifold face multiplicity')
    exposed = [f for f, multiplicity in faces.items() if multiplicity == 1]
    boundary = set(v for face in exposed for v in face)
    return {'tetrahedra': count, 'edges': len(edges), 'exposed_faces': len(exposed),
            'geometric_boundary_vertices': len(boundary),
            'core_on_geometric_boundary': sum(v < CORE for v in boundary),
            'halo_on_geometric_boundary': sum(v >= CORE for v in boundary),
            'halo_off_geometric_boundary': sum(v not in boundary for v in range(CORE, VERTICES))}


def graph_identities(charges, phi, edges, weights, operator=None):
    """Exact dyadic checks; an optional supplied oracle enables corruption controls.

    Edge orientation is the stored u->v direction. A cut flux is separately
    reoriented core->halo, independent of that storage direction. A virtual
    halo perturbation below is an exact rational probe, not an FP32 state write.
    """
    actual = DEC.exact_operator(charges, phi, edges, weights)
    op = actual if operator is None else operator
    require(len(op['rows']) == VERTICES and len(op['residual']) == CORE,
            'oracle row extent')
    pf, wf, qf = list(map(Q, phi)), list(map(Q, weights)), list(map(Q, charges))
    incident = [[] for _ in range(VERTICES)]
    cut_received = [Q(0) for _ in range(VERTICES-CORE)]
    halo_internal = [Q(0) for _ in range(VERTICES-CORE)]
    cut_edges = []
    for index, ((u, v), w) in enumerate(zip(edges, wf)):
        flux = w*(pf[u]-pf[v])
        incident[u].append(flux)
        incident[v].append(-flux)
        if u >= CORE and v >= CORE:
            halo_internal[u-CORE] += flux
            halo_internal[v-CORE] -= flux
        elif (u < CORE) != (v < CORE):
            core, halo, outward = (u, v, flux) if u < CORE else (v, u, -flux)
            cut_received[halo-CORE] += outward
            cut_edges.append((index, core, halo, outward))
    # Every coordinate basis vector is checked, not only a cancellation-prone
    # global Gauss sum. The independent oracle remains the source of Lphi/r.
    for vertex, values in enumerate(incident):
        require(sum(values, Q(0)) == op['rows'][vertex],
                f'coordinate-basis integration-by-parts failed at vertex {vertex}')
    for vertex in range(CORE):
        require(op['rows'][vertex] == qf[vertex]-op['residual'][vertex],
                f'core charge/residual pairing failed at vertex {vertex}')
    for local in range(VERTICES-CORE):
        require(op['rows'][CORE+local] == -cut_received[local]+halo_internal[local],
                f'halo cut/internal reaction failed at vertex {CORE+local}')

    probes = {
        'constant': [Q(1) for _ in range(VERTICES)],
        'mixed_dyadic': [Q((37*v % 31)-15, 16) for v in range(VERTICES)],
        'halo_nonconstant': [Q(0) if v < CORE else Q((19*v % 23)-11, 32)
                             for v in range(VERTICES)],
        'represented_phi': pf,
    }
    pairings = {}
    for name, eta in probes.items():
        edge_pair = sum((w*(eta[u]-eta[v])*(pf[u]-pf[v])
                         for (u, v), w in zip(edges, wf)), Q(0))
        vertex_pair = sum((a*b for a, b in zip(eta, op['rows'])), Q(0))
        core_pair = sum((eta[v]*(qf[v]-op['residual'][v]) for v in range(CORE)), Q(0))
        halo_pair = sum((eta[v]*op['rows'][v] for v in range(CORE, VERTICES)), Q(0))
        # The halo pair has an internal H-H edge contribution whenever eta_H
        # is nonconstant. It must not be replaced by a scalar cut-flux sum.
        cut_pair = -sum((eta[CORE+j]*f for j, f in enumerate(cut_received)), Q(0))
        hh_pair = sum((w*(eta[u]-eta[v])*(pf[u]-pf[v])
                       for (u, v), w in zip(edges, wf) if u >= CORE and v >= CORE), Q(0))
        require(edge_pair == vertex_pair == core_pair+halo_pair,
                'Green identity failed: '+name)
        require(halo_pair == cut_pair+hh_pair, 'halo flux pairing failed: '+name)
        pairings[name] = (edge_pair, core_pair, halo_pair, cut_pair, hh_pair)

    delta = [Q(0) if v < CORE else Q((17*(v-CORE) % 29)-14, 256)
             for v in range(VERTICES)]
    def functional(values):
        return sum((qf[v]*values[v] for v in range(CORE)), Q(0)) - sum(
            (w*(values[u]-values[v])**2 for (u, v), w in zip(edges, wf)), Q(0))/2
    start = functional(pf)
    require(start == op['functional'], 'fixed-halo functional oracle mismatch')
    virtual_delta = functional([a+b for a, b in zip(pf, delta)])-start
    reaction_work = -sum((delta[v]*op['rows'][v] for v in range(CORE, VERTICES)), Q(0))
    quadratic = -sum((w*(delta[u]-delta[v])**2 for (u, v), w in zip(edges, wf)), Q(0))/2
    require(virtual_delta == reaction_work+quadratic, 'virtual Dirichlet perturbation identity failed')
    return {'basis_checks': VERTICES, 'cut_edges': cut_edges,
            'halo_rows': [(CORE+j, op['rows'][CORE+j], cut_received[j], halo_internal[j])
                          for j in range(VERTICES-CORE)],
            'pairings': pairings, 'virtual_delta': virtual_delta,
            'virtual_linear_work': reaction_work, 'virtual_quadratic_term': quadratic}


def bound_bytes(path, dispatch):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    recorded = dispatch.get('artifacts', {}).get(str(path))
    if recorded is None:
        require(dispatch.get('input_hashes', {}).get(str(path)) == digest,
                'unbound physical input: '+str(path))
    else:
        require(recorded == {'sha256': digest, 'bytes': len(raw)},
                'unbound native output: '+str(path))
    return raw, digest


def check(execution, epoch, main_source=None):
    execution = Path(execution).resolve()
    execution_hash = sha(execution)
    require(sha(DEC.__file__) == EXPECTED_ORACLE and sha(PREFIX.__file__) == EXPECTED_PREFIX,
            'independent oracle/prefix source changed; review required')
    dispatch = json.loads(execution.read_text())
    tile_source = DEC.KERNELS/'nhs_biprism_simplicial_tile.cuh'
    build = json.loads(Path(dispatch['build_receipt']['path']).read_text())
    require(sha(tile_source) == EXPECTED_TILE_LAYOUT_SOURCE
            and dispatch.get('input_hashes', {}).get(str(tile_source)) == EXPECTED_TILE_LAYOUT_SOURCE
            and build.get('input_hashes', {}).get(str(tile_source)) == EXPECTED_TILE_LAYOUT_SOURCE,
            'tetrahedral layout source differs from the reviewed compiled/dispatch identity')
    native_main = Path(main_source).resolve() if main_source else DEC.KERNELS/'nhs_biprism_variational_so3g_v10.cuh'
    source_paths = [Path(__file__), Path(DEC.__file__), Path(PREFIX.__file__), native_main, tile_source,
                    DEC.KERNELS/'nhs_biprism_dec_pcg.cuh',
                    DEC.KERNELS/'nhs_biprism_electronic_discrete.cuh',
                    DEC.KERNELS/'nhs_biprism_coordinate_geometry.cuh',
                    DEC.KERNELS/'nhs_biprism_coordinate_precision.cuh',
                    Path(dispatch['build_receipt']['path']), Path(dispatch['command'][0])]
    pins = {str(path): sha(path) for path in source_paths}
    pins[str(execution)] = execution_hash
    # Preserve the controlled later failure and reuse every accepted-counter,
    # full hi+lo live-weight, fixed-halo and dispatch/build provenance gate.
    prefix_text = PREFIX.check(execution, epoch, main_source)
    prefix = dispatch['command'][4]+'.epoch'+str(epoch)
    files = {name: Path(prefix+suffix) for name, suffix in {
        'entry': '.phase2_tiles.bin', 'endpoint': '.tiles.bin', 'exchange': '.exchange.bin'}.items()}
    files['bp_case'] = Path(dispatch['command'][1])
    files['potential']=Path(prefix+'.potential.accepted.f64le')
    if 'd_spinor_owners;' in native_main.read_text():
        files['spinors']=Path(prefix+'.spinor.owners.bps1')
    data = {}
    for name, path in files.items():
        data[name], pins[str(path)] = bound_bytes(path, dispatch)
    amber = DEC.read_charges(files['bp_case'])
    lines = [prefix_text.rstrip(), 'BOUNDARY_IDENTITY_STATUS=PASS',
             'BOUNDARY_SCOPE=retained weighted graph, algebraic Dirichlet subset and exact virtual probes',
             'NO_CONTINUUM_FACE_FLUX_OR_FULL_HULL_OR_DISCRETIZATION_ACCURACY_CLAIM=true',
             'NO_MOLECULAR_NONADIABATIC_BENCHMARK_OR_WHOLE_TRAJECTORY_ACCEPTANCE=true',
             'PRESCRIBED_HALO_IS_NOT_ASSUMED_TO_EQUAL_GEOMETRIC_BOUNDARY=true',
             'TEMPORAL_HALO_PHI_CHANGE=zero bits within this accepted main epoch only',
             'PHASE2_OPERATOR_REFRESH_AND_PHYSICAL_BOUNDARY_WORK=separate owners, not inferred here',
             'VIRTUAL_PERTURBATION=exact rational delta on halo, fixed core/charges/weights; no state write',
             'FLUX_ORIENTATION=core to halo; halo reaction=-incoming_cut_flux+internal_halo_divergence',
             'HALO_ROW_FIELDS=local_vertex physical_atom reaction incoming_cut_flux internal_halo_divergence',
             'PAIRING_FIELDS=edge_pair core_pair halo_pair negative_cut_pair internal_halo_pair']
    lines.append(f'BOUNDARY_LAYOUT_SOURCE={tile_source} SHA256={EXPECTED_TILE_LAYOUT_SOURCE}')
    for leaf in range(LEAVES):
        before = data['entry'][leaf*DEC.TILE_BYTES:(leaf+1)*DEC.TILE_BYTES]
        after = data['endpoint'][leaf*DEC.TILE_BYTES:(leaf+1)*DEC.TILE_BYTES]
        ids, edges, psi, phi = DEC.parse_tile(after)
        require(len(data['potential'])==8*2048,'complete precise boundary potential')
        phi=struct.unpack_from('<256d',data['potential'],leaf*2048)
        if 'spinors' in data:
            psi=SPINOR.read(data['spinors'],leaf)['accepted']+psi[CORE:]
        domain = retained_domain(before, after, edges)
        weights = EX.weights(data['exchange'],leaf,len(edges))
        charges = [DEC.represented_charge(amber[ids[v]], psi[v][2:4]) for v in range(CORE)]
        checked = graph_identities(charges, phi, edges, weights)
        lines.append(f'LEAF={leaf} ALGEBRAIC_CORE={CORE} ALGEBRAIC_HALO={VERTICES-CORE} '
                     + ' '.join(f'{k.upper()}={v}' for k, v in domain.items())
                     + f' BASIS_CHECKS={checked["basis_checks"]} CUT_EDGES={len(checked["cut_edges"])}')
        for vertex, reaction, cut, hh in checked['halo_rows']:
            lines.append(f'HALO LEAF={leaf} VERTEX={vertex} ATOM={ids[vertex]} '
                         f'REACTION={rational(reaction)} CUT_IN={rational(cut)} HH_DIV={rational(hh)}')
        for name, values in checked['pairings'].items():
            lines.append(f'GREEN LEAF={leaf} PROBE={name} '+ ' '.join(map(rational, values)))
        lines.append(f'VIRTUAL LEAF={leaf} DELTA_S={rational(checked["virtual_delta"])} '
                     f'LINEAR={rational(checked["virtual_linear_work"])} '
                     f'QUADRATIC={rational(checked["virtual_quadratic_term"])}')
    for path, digest in pins.items():
        require(sha(path) == digest, 'boundary input/reader changed while checking: '+path)
    lines.append('BOUNDARY_CHECKER_SHA256='+pins[str(Path(__file__))])
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution', required=True, type=Path)
    parser.add_argument('--epoch', required=True, type=int)
    parser.add_argument('--main-source', type=Path)
    parser.add_argument('--output', type=Path, help='Exclusive new plaintext output; stdout is always written')
    args = parser.parse_args()
    result = check(args.execution, args.epoch, args.main_source)
    if args.output is not None:
        with args.output.open('x') as target:
            target.write(result)
    print(result, end='')


if __name__ == '__main__':
    main()
