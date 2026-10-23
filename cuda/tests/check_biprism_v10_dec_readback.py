#!/usr/bin/env python3
"""Independent exact-dyadic residual of the actual exported DEC state.

No solve is performed and no state is repaired. This checks the retained graph,
actual live FP32 weights, authoritative retained phi and source-defined FP64 endpoint
charge. The weights are independently reconstructed from the complete hi+lo
endpoint/reference geometry. The diagonal 1e-4 is PRECONDITIONING ONLY.
"""
from pathlib import Path
from fractions import Fraction as Q
import argparse
import hashlib
import json
import math
import re
import biprism_exchange_layout as EX
import struct
import check_biprism_spinor_owner as SPINOR

ROOT = Path(__file__).resolve().parent
KERNELS = ROOT.parents[1] / 'crates/prism-gpu/src/kernels'
CORE, VERTICES, LEAVES, MAX_EDGES = 192, 256, 8, 1536
TILE_BYTES, BANK_BYTES, RECORD_BYTES = 225408, 11648, 128
OFF = {'ids': 0, 'edges': 29696, 'psi': 90112, 'phi': 94208,
       'edge_count': 225284, 'base_weights': 96256, 'live_weights': 5376, 'footer': 11520}


def identity(path):
    path = Path(path)
    raw = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


ALPHA = f32(.15)
TOL_SQ = f32(1e-12)
EXPECTED_DEC = 'f29ed2d7ef1dd31ecca3a9d6fb21b50028528c182fb317b6cab900d3643aa0f2'
# Preserve the historical linear source identity. The indexed realization
# traverses the same incident ordinals and retains every operator term.
EXPECTED_ORDERED_DEC = '19698481371739f3772894f536df524a629ab97753786f5c78b4c14d52f33c8d'
EXPECTED_PRECISE_DEC = 'd3a25a01a70ba5a1b6f66e7fb36787fa915edb37831baedd4a8a37b49cb24449'
# Same exact operator and residual oracle; optional tighter INTERNAL solve
# target, still bounded by the unchanged external true-residual ceiling.
EXPECTED_INTERNAL_TARGET_DEC = 'f4cb5dbf4552de12288c93b3797e63d476649afc003f53e1d4111edde64bbea0'
# FP64 Krylov vectors/reductions with reliable residual replacement. Native
# actual-operator controls retain the192-iteration cap and1e-10 internal
# target. This source epoch changes no term in the independent operator below.
EXPECTED_FP64_KRYLOV_DEC = '18dfce17d400d9b04389575948783ebabff949c127b6faa0ff89e068e9015c01'
EXPECTED_ORDERED_EDGES = 'c4f6c1267fa2a375b88fcc9fffab6020cf1937a3f87652b1464cd590ee462c25'
EXPECTED_POTENTIAL = 'c57468e9193277e92cb4e607699f66b27c6bfd95f5b6fd5f2c3f088010ad0888'
EXPECTED_ORDERED_CONTROL = '648f3cce75ee6b1e851d0af92e9393ee7856b6753edbc5b961bb5217146a99fd'
EXPECTED_ELECTRONIC = 'a416c303a31602bc1ae077cc17874a6b59d5048a792726a334b30265ef19eedc'
EXPECTED_GEOMETRY = '25eb751cbd5d2a573f576cefb181d6fc1e36e02e6b6b7f5a16947a0d07937a48'
EXPECTED_COORDINATE = 'ff4117e4af7b24ace9f4b11efbd04db62f77bbbc70a544d3a4c6f74305a5361d'


def dec_source_bindings(dec_source, ordered_source=None, potential_source=None):
    """Select only reviewed source epochs; an indexed source needs its index code."""
    digest = identity(dec_source)['sha256']
    if digest == EXPECTED_DEC:
        return 'historical_linear', []
    if digest in (EXPECTED_ORDERED_DEC, EXPECTED_PRECISE_DEC, EXPECTED_INTERNAL_TARGET_DEC, EXPECTED_FP64_KRYLOV_DEC):
        ordered_source = Path(ordered_source) if ordered_source else KERNELS/'nhs_biprism_ordered_edges.cuh'
        if identity(ordered_source)['sha256'] != EXPECTED_ORDERED_EDGES:
            raise ValueError('ordered DEC traversal source changed; independent review required')
        if digest in (EXPECTED_PRECISE_DEC, EXPECTED_INTERNAL_TARGET_DEC, EXPECTED_FP64_KRYLOV_DEC):
            potential_source = Path(potential_source) if potential_source else KERNELS/'nhs_biprism_potential_precision.cuh'
            if identity(potential_source)['sha256'] != EXPECTED_POTENTIAL:
                raise ValueError('retained potential owner source changed; independent review required')
            return 'ordered_incidence_retained_fp64_potential', [ordered_source, potential_source]
        return 'ordered_incidence_same_operator', [ordered_source]
    raise ValueError('qualified DEC source changed; independent review required')


def finite_json(value):
    if math.isfinite(value):
        return value
    return 'NaN' if math.isnan(value) else ('+Infinity' if value > 0 else '-Infinity')


def represented_charge(amber, psi1):
    """Correctly rounded binary64 FMA sequence from represented binary32 psi."""
    a, b = (Q.from_float(float(v)) for v in psi1)
    # Both binary32 squares are exact binary64 products. Their norm is one
    # binary64 FMA rounding. Fraction->float uses nearest, ties to even.
    population = float(a*a + Q(float(b*b)))
    return float(Q.from_float(ALPHA)*Q.from_float(population) + Q.from_float(float(amber)))


def fma64(a, b, c):
    return float(Q(a)*Q(b)+Q(c))


def precise_weight(base, current_u, current_v, reference_u, reference_v, fp64=False):
    """Independent declared FP64 radial tree with one FP32 output boundary.

    A position is three (high, low) pairs. Differences are formed exactly from
    these primitives before the FP64 geometry, never by adding in FP32.
    """
    if not all(math.isfinite(x) for p in (current_u,current_v,reference_u,reference_v) for pair in p for x in pair):
        raise ValueError('nonfinite coordinate expansion')
    if not math.isfinite(base) or base <= 0:
        raise ValueError('invalid base weight')
    def length2(a,b):
        e=[float((Q(y[0])+Q(y[1]))-(Q(x[0])+Q(x[1]))) for x,y in zip(a,b)]
        return fma64(e[0],e[0],fma64(e[1],e[1],e[2]*e[2]))+f32(1e-12)
    r=math.sqrt(length2(current_u,current_v))
    r0=math.sqrt(length2(reference_u,reference_v))
    value=float(max(base,f32(1e-4)))*(r0/r)
    return value if fp64 else f32(value)


def coordinate_at(high,low,leaf,atom,atoms):
    index=3*(leaf*atoms+atom)
    return tuple((high[index+k],low[index+k]) for k in range(3))


def exact_operator(charges, phi, edges, weights):
    if len(charges) != CORE or len(phi) != VERTICES or len(edges) != len(weights):
        raise ValueError('wrong represented DEC dimensions')
    if not all(math.isfinite(v) for v in charges + phi + weights):
        raise ValueError('nonfinite DEC input')
    if not all(w > 0 for w in weights):
        raise ValueError('nonpositive DEC edge weight')
    if len(set(tuple(sorted(e)) for e in edges)) != len(edges):
        raise ValueError('duplicate undirected DEC edge')
    if any(u == v or not (0 <= u < VERTICES and 0 <= v < VERTICES) for u, v in edges):
        raise ValueError('invalid DEC edge')
    pf = list(map(Q.from_float, phi))
    rows = [Q(0) for _ in range(VERTICES)]
    quadratic = Q(0)
    for (u, v), w in zip(edges, weights):
        diff = pf[u] - pf[v]
        flux = Q.from_float(w)*diff
        rows[u] += flux
        rows[v] -= flux
        quadratic += flux*diff
    residual = [Q.from_float(charges[u])-rows[u] for u in range(CORE)]
    squared = sum((v*v for v in residual), Q(0))
    qphi = sum((Q.from_float(charges[u])*pf[u] for u in range(CORE)), Q(0))
    return {'rows': rows, 'residual': residual, 'squared': squared,
            'functional': qphi-quadratic/2,
            'charge_sum': sum(map(Q.from_float, charges), Q(0)),
            'halo_reaction': sum(rows[CORE:], Q(0))}


def parse_tile(raw):
    if len(raw) != TILE_BYTES:
        raise ValueError('tile byte extent')
    ids = [v & 0x00ffffff for v in struct.unpack_from('<256I', raw, OFF['ids'])]
    if len(set(ids)) != VERTICES:
        raise ValueError('tile atom identity is not unique')
    ne, = struct.unpack_from('<I', raw, OFF['edge_count'])
    if not 0 < ne <= MAX_EDGES:
        raise ValueError('active edge count')
    edges = list(struct.iter_unpack('<BB', raw[OFF['edges']:OFF['edges']+2*ne]))
    psi = list(struct.iter_unpack('<4f', raw[OFF['psi']:OFF['psi']+16*VERTICES]))
    phi = list(struct.unpack_from('<256f', raw, OFF['phi']))
    return ids, edges, psi, phi


def potential_owner(raw):
    """Decode the source-bound BPP1 owner without interpreting pointers as data."""
    if len(raw) != 128:
        raise ValueError('potential owner extent')
    names = ('magic', 'version', 'leaf', 'state', 'allocation', 'topology', 'tile',
             'coordinate', 'accepted_ptr', 'prepared_ptr', 'accepted_count',
             'accepted_step', 'prepared_count', 'prepared_step', 'accepted_generation',
             'prepared_generation', 'projection_state', 'projection_count',
             'projection_step', 'reserved0', 'projection_generation', 'reserved1')
    return dict(zip(names, struct.unpack('<4I6Q4I2Q4I2Q', raw)))


def retained_potential_epoch(data, atoms, entry_count, numerical_bytes=40704):
    """Bind accepted FP64 values to entry authority, actual scratch and owner ABI.

    This is representation/chronology validation. The exact graph residual is
    evaluated separately, using the returned full binary64 potential values.
    """
    def require(test, why):
        if not test:
            raise ValueError(why)
    extents = {'potential_owner': 8*128, 'entry_potential_owner': 8*128,
               'potential_accepted': 8*2048, 'potential_prepared': 8*2048,
               'entry_potential_accepted': 8*2048, 'entry_potential_prepared': 8*2048,
               'coordinate_owner': 8*96, 'entry_coordinate_owner': 8*96,
               'stream_table': 640, 'entry_stream_table': 640,
               'ordered_edges': 8*6656, 'numerical': 8*numerical_bytes}
    require(numerical_bytes in (40704,43008,66048,70656,72960,97536,106752,143872,196992,198144,235136,349056,496768,510720,1680768,1697920,1784448,1837696,1997440), 'unsupported full-potential numerical ABI')
    for name, size in extents.items():
        require(len(data[name]) == size, 'precise potential readback extent: '+name)
    require(data['stream_table'] == data['entry_stream_table'], 'stream allocation changed inside main')
    require(data['coordinate_owner'] == data['entry_coordinate_owner'], 'coordinate owner changed inside main')
    table = data['stream_table']
    mode, reserved = struct.unpack_from('<2I', table, 520)
    coordinate_base, ordered_base, potential_base = struct.unpack_from('<3Q', table, 528)
    spinor_base=struct.unpack_from('<Q',table,552)[0]
    require(mode in (0,4) and reserved==0 and (not spinor_base or spinor_base%128==0) and not any(table[560:]), 'precise table mode/reserved identity')
    require(coordinate_base and coordinate_base % 16 == 0 and ordered_base and ordered_base % 128 == 0
            and potential_base and potential_base % 128 == 0, 'precise table owner pointer domain')
    ranges = [(coordinate_base, 8*96, 'coordinate descriptors'),
              (ordered_base, 8*6656, 'ordered descriptors'),
              (potential_base, 8*128, 'potential descriptors')]
    if 'spinor_owner' in data:
        require(spinor_base!=0, 'missing physical spinor table pointer')
        ranges.append((spinor_base,8*SPINOR.SIZE,'spinor owner'))
    initial_values, endpoint_values, details = [], [], []
    topology = set(); allocation = set(); steps = set(); generations = set()
    tile_base = None
    for leaf in range(8):
        before = potential_owner(data['entry_potential_owner'][leaf*128:(leaf+1)*128])
        after = potential_owner(data['potential_owner'][leaf*128:(leaf+1)*128])
        coord = struct.unpack_from('<4IQ2I8Q', data['coordinate_owner'], leaf*96)
        require(coord[:4] == (0x31504342, 1, leaf, atoms) and coord[4] > 0
                and coord[5:7] == (2, 7) and coord[-2:] == (0, 0), 'coordinate/potential owner identity')
        require(struct.unpack_from('<Q', table, leaf*8)[0] == coord[7], 'table/current-coordinate pointer mismatch')
        for pointer in coord[7:13]:
            require(pointer and pointer % 4 == 0, 'coordinate pointer domain')
            ranges.append((pointer, atoms*12, 'coordinate plane'))
        scratch, = struct.unpack_from('<Q', table, 448+leaf*8)
        require(scratch and scratch % 128 == 0, 'numerical pointer domain')
        ranges.append((scratch, numerical_bytes, 'numerical scratch'))
        immutable = ('magic', 'version', 'leaf', 'allocation', 'topology', 'tile',
                     'coordinate', 'accepted_ptr', 'prepared_ptr', 'reserved0', 'reserved1')
        require(all(before[k] == after[k] for k in immutable), 'potential allocation identity changed')
        for o in (before, after):
            require((o['magic'], o['version'], o['leaf']) == (0x31505042, 1, leaf)
                    and o['allocation'] == coord[4] and o['topology'] > 0
                    and o['coordinate'] == coordinate_base+leaf*96
                    and o['reserved0'] == o['reserved1'] == 0, 'potential descriptor identity')
            require(o['state'] in (1, 2) and o['accepted_generation'] > 0, 'unpublished potential owner')
        if 'spinor_owner' in data:
            for name,pot in (('entry_spinor_owner',before),('spinor_owner',after)):
                owner=SPINOR.read(data[name],leaf)
                require(all(owner[k]==pot[k] for k in ('allocation','topology','tile','coordinate')), 'spinor/potential identity mismatch')
                require(owner['committed']==pot['accepted_count'] and owner['step']==pot['accepted_step'], 'spinor/potential accepted chronology mismatch')
        topology.add(after['topology']); allocation.add(after['allocation'])
        if tile_base is None:
            tile_base = after['tile']
            require(tile_base and tile_base % 128 == 0, 'tile pointer domain')
            ranges.append((tile_base, 8*TILE_BYTES, 'tiles'))
        require(after['tile'] == tile_base+leaf*TILE_BYTES, 'potential tile population layout')
        index = data['ordered_edges'][leaf*6656:(leaf+1)*6656]
        ih = struct.unpack_from('<4IQ3I', index)
        edge_count, = struct.unpack_from('<I', data['endpoint'], leaf*TILE_BYTES+OFF['edge_count'])
        require(ih[:4] == (0x31454942, 1, leaf, edge_count) and ih[4] == after['tile']
                and ih[5] == entry_count and ih[6] == 2 and not any(index[36:64]),
                'potential/ordered invocation identity')
        for key in ('accepted_ptr', 'prepared_ptr'):
            require(after[key] and after[key] % 8 == 0, 'potential data pointer domain')
            ranges.append((after[key], 2048, 'potential bank'))
        require(before['accepted_count'] == entry_count, 'potential entry accepted counter')
        if before['state'] == 2:
            step = before['prepared_step']
            require(0 < step < 0xffffffff and before['accepted_step'] == step-1
                    and before['prepared_count'] == entry_count
                    and before['accepted_generation'] < 0xffffffffffffffff
                    and before['prepared_generation'] == before['accepted_generation']+1,
                    'prepared potential chronology')
            selected = 'entry_potential_prepared'; generation = before['prepared_generation']
        else:
            require(mode == 4 and before['accepted_step'] < 0xfffffffe,
                    'production main requires a fresh prepared potential')
            step = before['accepted_step']+1
            selected = 'entry_potential_accepted'; generation = before['accepted_generation']
        require((before['projection_state'], before['projection_count'], before['projection_step'], before['projection_generation'])
                == (before['state'], entry_count, step if before['state'] == 2 else step-1, generation),
                'entry projection identity')
        require((after['state'], after['accepted_count'], after['accepted_step'], after['accepted_generation'])
                == (1, entry_count+4, step, generation), 'accepted potential chronology')
        require((after['projection_state'], after['projection_count'], after['projection_step'], after['projection_generation'])
                == (1, entry_count+4, step, generation), 'accepted projection identity')
        require(all(before[k] == after[k] for k in ('prepared_count', 'prepared_step', 'prepared_generation')),
                'main rewrote prepared identity')
        begin, end = leaf*2048, (leaf+1)*2048
        require(data['entry_potential_prepared'][begin:end] == data['potential_prepared'][begin:end],
                'main rewrote prepared bank')
        for name in ('potential_accepted', 'potential_prepared', 'entry_potential_accepted', 'entry_potential_prepared'):
            require(all(map(math.isfinite, struct.unpack_from('<256d', data[name], begin))), 'nonfinite retained potential bank')
        initial = data[selected][begin:end]; final = data['potential_accepted'][begin:end]
        require(initial[192*8:] == final[192*8:], 'full-precision fixed halo changed')
        require(final == data['numerical'][leaf*numerical_bytes+38656:leaf*numerical_bytes+40704],
                'accepted potential differs from evaluated working state')
        p0 = list(struct.unpack('<256d', initial)); p1 = list(struct.unpack('<256d', final))
        for name, values in (('entry', p0), ('endpoint', p1)):
            rounded = b''.join(struct.pack('<f', f32(v)) for v in values)
            start = leaf*TILE_BYTES+OFF['phi']
            require(rounded == data[name][start:start+1024], 'tile compatibility projection mismatch')
        initial_values.append(p0); endpoint_values.append(p1)
        steps.add(step); generations.add(generation)
        details.append({'state': 'accepted', 'entry_state': 'prepared' if before['state'] == 2 else 'accepted',
                        'accepted_step': step, 'operator_generation': generation,
                        'full_potential_matches_evaluated_working_bits': True,
                        'compatibility_projection_bits': True})
    for pointer, size, _ in ranges:
        require(0 < pointer <= 2**64-size, 'potential/coordinate allocation range overflow')
    ordered_ranges = sorted((pointer, pointer+size, label) for pointer, size, label in ranges)
    require(all(a[1] <= b[0] for a, b in zip(ordered_ranges, ordered_ranges[1:])),
            'potential/coordinate/working allocations overlap')
    require(all(len(s) == 1 for s in (topology, allocation, steps, generations)), 'all8 potential generation/step mismatch')
    return initial_values, endpoint_values, details


def ordered_potential_control_leaf(data, leaf, charges, edges, weights, endpoint_phi):
    """Read ODP2 private solves; a private refusal never becomes acceptance."""
    if data['ordered_control_schema'] != struct.pack('<8I', 0x3250444f, 2, 72, 8, 4, 8, 256, 0):
        raise ValueError('ordered FP64 control schema')
    results = []
    for guess in range(2):
        raw = data['ordered_control_'+str(guess)]
        if len(raw) != 8*256*72:
            raise ValueError('ordered FP64 control population extent')
        rows = [raw[(leaf*256+v)*72:(leaf*256+v+1)*72] for v in range(256)]
        phi = []
        decision = None
        for row in rows:
            v = struct.unpack('<6d2f4I', row)
            if not all(math.isfinite(x) for x in v[:6]):
                raise ValueError('nonfinite ordered FP64 row/diagonal/potential')
            pairs = ((0, 8, 8), (16, 24, 8), (32, 40, 8), (48, 52, 4), (56, 60, 4), (64, 68, 4))
            if any(row[a:a+n] != row[b:b+n] for a, b, n in pairs):
                raise ValueError('ordered/linear FP64 control mismatch')
            if v[8] > 192 or v[10] not in (0, 1) or (v[10] and (not math.isfinite(v[6]) or v[6] < 0)):
                raise ValueError('ordered private solver status/iteration domain')
            current = row[48:52]+row[56:60]+row[64:68]
            if decision is not None and current != decision:
                raise ValueError('ordered private solver CTA decision disagreement')
            decision = current
            phi.append(v[4])
        if struct.pack('<64d', *phi[192:]) != struct.pack('<64d', *endpoint_phi[192:]):
            raise ValueError('ordered private solver changed full fixed halo')
        oracle = exact_operator(charges, phi, edges, weights)
        squared = oracle['squared']
        true_pass = squared <= Q.from_float(TOL_SQ) and squared <= Q(1, 10**12)
        record = struct.unpack('<6d2f4I', rows[0])
        if record[10] and not true_pass:
            raise ValueError('ordered private success fails independent exact residual')
        results.append({'guess': 'zero_core' if guess else 'retained_full_potential',
                        'source_success': bool(record[10]), 'iterations': record[8],
                        'linear_indexed_bits_equal': True, 'fixed_halo_bits_equal': True,
                        'true_residual_l2': math.sqrt(float(squared)),
                        'strict_exact_residual_pass': true_pass,
                        'scope': 'Private diagnostic solve only; no physical acceptance or state write'})
    return results


def read_charges(path):
    raw = Path(path).read_bytes()
    if raw[:8] != b'BPCASE02' or len(raw) < 32:
        raise ValueError('BPCASE02 input required')
    n, center, nx, ng, hmr, nr = struct.unpack_from('<6I', raw, 8)
    if not 256 <= n <= 100000 or center >= n or ng > n or hmr > 1 or not 0 < nr <= n:
        raise ValueError('BPCASE02 header domain')
    begin = 32+12*n
    if len(raw) < begin+4*n:
        raise ValueError('truncated original charge plane')
    charges = list(struct.unpack_from('<'+str(n)+'f', raw, begin))
    if not all(math.isfinite(v) for v in charges):
        raise ValueError('nonfinite original charge plane')
    return charges


def check_epoch(args):
    prefix = str(args.prefix)+'.epoch'+str(args.epoch)
    dec_realization, extra_sources = dec_source_bindings(args.dec_source,
        getattr(args, 'ordered_source', None), getattr(args, 'potential_source', None))
    precise = dec_realization == 'ordered_incidence_retained_fp64_potential'
    files = {name: Path(prefix+suffix) for name, suffix in {
        'entry': '.phase2_tiles.bin', 'endpoint': '.tiles.bin',
        'exchange': '.exchange.bin', 'convergence': '.conv.bin',
        'coords': '.au_coords.bin', 'coords_low': '.au_coords_low.bin',
        'reference': '.au_reference_high.bin', 'reference_low': '.au_reference_low.bin',
        'entry_coords': '.entry_au_coords.bin', 'entry_coords_low': '.entry.au_coords_low.bin',
        'entry_reference': '.entry.au_reference_high.bin',
        'entry_reference_low': '.entry.au_reference_low.bin'}.items()}
    if precise:
        for name, suffix in {
            'potential_owner': '.potential.owners.128b',
            'potential_accepted': '.potential.accepted.f64le',
            'potential_prepared': '.potential.prepared.f64le',
            'entry_potential_owner': '.entry.potential.owners.128b',
            'entry_potential_accepted': '.entry.potential.accepted.f64le',
            'entry_potential_prepared': '.entry.potential.prepared.f64le',
            'coordinate_owner': '.au_coordinate_owners.96b',
            'entry_coordinate_owner': '.entry.au_coordinate_owners.96b',
            'stream_table': '.au_stream_table.640b',
            'entry_stream_table': '.entry.au_stream_table.640b',
            'ordered_edges': '.ordered_edges.bin', 'numerical': '.numerical.bin',
            'ordered_control_schema': '.ordered_parity.schema.u32le',
            'ordered_control_owners': '.ordered_parity.potential.owners.128b',
            'ordered_control_banks': '.ordered_parity.potential.banks.accepted_then_prepared.f64le',
            'ordered_control_0': '.ordered_parity.guess0.rows72.bin',
            'ordered_control_1': '.ordered_parity.guess1.rows72.bin'}.items():
            files[name] = Path(prefix+suffix)
    spinor_migration='d_spinor_owners;' in Path(args.main_source).read_text()
    if spinor_migration:
        files['spinor_owner']=Path(prefix+'.spinor.owners.bps1')
        files['entry_spinor_owner']=Path(prefix+'.entry.spinor.owners.bps1')
    data = {name: path.read_bytes() for name, path in files.items()}
    bank_bytes, footer_offset, weight_format, bank_magic = EX.layout(data['exchange'])
    for name, stride in [('entry', TILE_BYTES), ('endpoint', TILE_BYTES),
                         ('exchange', bank_bytes), ('convergence', RECORD_BYTES)]:
        if len(data[name]) != LEAVES*stride:
            raise ValueError(name+' does not contain exactly eight leaves')
    amber = read_charges(args.bp_case)
    atoms=len(amber)
    planes={}
    for name in ('coords','coords_low','reference','reference_low','entry_coords','entry_coords_low','entry_reference','entry_reference_low'):
        if len(data[name]) != 8*atoms*12:
            raise ValueError('coordinate plane extent: '+name)
        planes[name]=struct.unpack('<'+str(24*atoms)+'f',data[name])
        if not all(map(math.isfinite,planes[name])):
            raise ValueError('nonfinite full coordinate plane: '+name)
    if data['reference']!=data['entry_reference'] or data['reference_low']!=data['entry_reference_low']:
        raise ValueError('mesh reference expansion changed inside accepted main')
    main = args.main_source.read_text()
    dec = args.dec_source.read_text()
    electronic = args.electronic_source.read_text()
    geometry_source=KERNELS/'nhs_biprism_coordinate_geometry.cuh'
    coordinate_source=KERNELS/'nhs_biprism_coordinate_precision.cuh'
    if identity(geometry_source)['sha256']!=EXPECTED_GEOMETRY or identity(coordinate_source)['sha256']!=EXPECTED_COORDINATE:
        raise ValueError('coordinate/radial realization changed; independent review required')
    if identity(args.electronic_source)['sha256'] != EXPECTED_ELECTRONIC:
        raise ValueError('qualified operator helper hash changed; independent review required')
    if 'positive-weight Dirichlet graph Laplacian' not in dec or 'double diagonal = 1e-4;' not in dec:
        raise ValueError('unrecognized DEC contract; review changed source before using this checker')
    if 'row += static_cast<double>(weights(e)) * (own - static_cast<double>(value[v]))' not in dec:
        raise ValueError('DEC row source changed')
    if 'bp_v10_publish_charge' not in main or 'R(0.15f)' not in electronic:
        raise ValueError('endpoint charge source changed')
    if precise:
        native_control_source = getattr(args, 'ordered_control_source', None) or ROOT/'biprism_ordered_edges_native_control.hpp'
        if identity(native_control_source)['sha256'] != EXPECTED_ORDERED_CONTROL:
            raise ValueError('ordered FP64 native-control source changed; independent review required')
        extra_sources.append(Path(native_control_source))
    if spinor_migration:
        extra_sources.append(KERNELS/'nhs_biprism_spinor_precision.cuh')
    expected_entry = args.entry_substeps if args.entry_substeps is not None else 4*(args.epoch-1)
    if not 0 <= expected_entry <= 0xffffffff-4:
        raise ValueError('accepted counter overflow domain')
    numerical_bytes=40704
    if precise:
        sizes=re.findall(r'static_assert\(sizeof\(BiprismV10NumericalScratchFfi\)\s*==\s*(\d+)',main)
        if len(sizes)!=1 or int(sizes[0]) not in (40704,43008,66048,70656,72960,97536,106752,143872,196992,198144,235136,349056,496768,510720,1680768,1697920,1784448,1837696,1997440):
            raise ValueError('compiled full-potential numerical ABI assertion')
        numerical_bytes=int(sizes[0])
    full_phi = retained_potential_epoch(data, atoms, expected_entry, numerical_bytes) if precise else None
    spinors=SPINOR.endpoint(data['spinor_owner'],data['endpoint'],expected_entry+4,args.epoch) if spinor_migration else None
    if precise and (data['ordered_control_owners'] != data['potential_owner']
                    or data['ordered_control_banks'] != data['potential_accepted']+data['potential_prepared']):
        raise ValueError('ordered diagnostic input differs from actual retained endpoint owner/banks')
    results = []
    sequences = []
    all_pass = True
    for leaf in range(LEAVES):
        before = data['entry'][leaf*TILE_BYTES:(leaf+1)*TILE_BYTES]
        after = data['endpoint'][leaf*TILE_BYTES:(leaf+1)*TILE_BYTES]
        ids0, edges0, _, phi0 = parse_tile(before)
        ids, edges, psi, phi = parse_tile(after)
        if spinors:
            psi=spinors[leaf]["accepted"]+psi[CORE:]
            old=SPINOR.read(data["entry_spinor_owner"],leaf)
            if old["committed"]!=expected_entry:raise ValueError("spinor entry chronology")
        if precise:
            phi0, phi = full_phi[0][leaf], full_phi[1][leaf]
        if ids != ids0 or edges != edges0 or any(a >= len(amber) for a in ids):
            raise ValueError('atom/edge population changed within the Hamiltonian body')
        bank = data['exchange'][leaf*bank_bytes:(leaf+1)*bank_bytes]
        weights = list(struct.unpack_from('<'+str(len(edges))+weight_format, bank, OFF['live_weights']))
        bases=struct.unpack_from('<'+str(len(edges))+'f',after,OFF['base_weights'])
        cp=[coordinate_at(planes['coords'],planes['coords_low'],leaf,a,atoms) for a in ids]
        rp=[coordinate_at(planes['reference'],planes['reference_low'],leaf,a,atoms) for a in ids]
        expected_weights=[precise_weight(b,cp[u],cp[v],rp[u],rp[v],fp64=weight_format=='d') for (u,v),b in zip(edges,bases)]
        weight_identity=all(struct.pack('<'+weight_format,a)==struct.pack('<'+weight_format,b) for a,b in zip(weights,expected_weights))
        halo_coordinate_fixed=all(
            data[name][(leaf*atoms+a)*12:(leaf*atoms+a+1)*12]==data[entry_name][(leaf*atoms+a)*12:(leaf*atoms+a+1)*12]
            for a in ids[CORE:] for name,entry_name in (('coords','entry_coords'),('coords_low','entry_coords_low')))
        record = data['convergence'][leaf*RECORD_BYTES:(leaf+1)*RECORD_BYTES]
        rec = struct.unpack_from('<8I', record)
        # abi, leaf, entry, substep, publication sequence, local fault, vote, phase
        footer = struct.unpack_from('<8I', bank, footer_offset+80)
        reserved = struct.unpack_from('<4I', bank, footer_offset+112)
        identity_ok = (rec[0] == 0x42505631 and rec[1] == leaf and not any(rec[2:5])
                       and rec[5] == expected_entry+4 and footer[0] == bank_magic
                       and footer[1] == leaf and footer[2] == expected_entry
                       and footer[3] == 3 and footer[4] > 0 and footer[5] == 0
                       and footer[6] == 1 and footer[7] == 4 and not any(reserved))
        sequences.append(footer[4])
        halo_phi_before = b''.join(struct.pack('<d', x) for x in phi0[CORE:])
        halo_phi_after = b''.join(struct.pack('<d', x) for x in phi[CORE:])
        halo_fixed = halo_phi_before == halo_phi_after
        charge = [represented_charge(amber[ids[u]], psi[u][2:4]) for u in range(CORE)]
        ordered_control = ordered_potential_control_leaf(data, leaf, charge, edges, weights, phi) if precise else None
        actual = exact_operator(charge, phi, edges, weights)
        squared = actual['squared']
        source_pass = squared <= Q.from_float(TOL_SQ)
        decimal_pass = squared <= Q(1, 10**12)
        passed = identity_ok and halo_fixed and halo_coordinate_fixed and weight_identity and source_pass and decimal_pass
        all_pass &= passed
        residue = actual['residual']
        gauss = actual['charge_sum']+actual['halo_reaction']
        assert gauss == sum(residue, Q(0))
        results.append({
            'leaf': leaf, 'active_edges': len(edges), 'accepted_epoch_identity': identity_ok,
            'fixed_halo_phi_bit_identity': halo_fixed,
            'potential_authority': full_phi[2][leaf] if precise else {'state': 'historical_fp32_tile'},
            'ordered_private_solver_controls': ordered_control,
            'fixed_halo_coordinate_pair_bit_identity': halo_coordinate_fixed,
            'full_coordinate_live_weight_bit_identity': weight_identity,
            'true_residual_l2': math.sqrt(float(squared)),
            'true_residual_linf': float(max(map(abs, residue))),
            'true_residual_squared_exact': {'numerator': str(squared.numerator), 'denominator': str(squared.denominator)},
            'core_residuals': [float(v) for v in residue],
            'source_squared_gate_pass': source_pass, 'strict_decimal_l2_1e_minus_6_pass': decimal_pass,
            'reported_solver_residual': finite_json(struct.unpack_from('<f', record, 36)[0]),
            'reported_solver_iterations': rec[7],
            'fixed_halo_functional': float(actual['functional']),
            'core_charge_sum': float(actual['charge_sum']),
            'halo_reaction_sum': float(actual['halo_reaction']),
            'core_charge_plus_halo_reaction': float(gauss),
            'all_vertex_neumann_neutrality_required': False,
            'pass': passed})
    same_sequence = len(set(sequences)) == 1
    all_pass &= same_sequence
    return {
        'status': 'PASS' if all_pass else 'REFUSED',
        'scope': 'Independent DEC endpoint residual and fixed-halo/accepted-epoch identity only; not whole-trajectory, electronic equation, metric, native execution or operator-refresh acceptance.',
        'operator': 'L_CC phi_C + L_CH phi_H = q_C; no screening mass term',
        'charge': 'q_C=RN64(double(0.15f)*RN64(re(psi1)^2+im(psi1)^2)+double(amber)); the multiply-add is rounded once',
        'precision': 'Exact rational rows and squared residual for the represented FP32 weights, '
            +('retained FP64 potential' if precise else 'historical FP32 tile potential')
            +' and source-defined binary64 charge. No solver recurrence or reported residual is used for acceptance.',
        'source_gate_squared': TOL_SQ, 'strict_decimal_l2_gate': '1e-6',
        'preconditioner_diagonal_is_not_screening': True,
        'dec_source_realization': dec_realization,
        'same_publication_sequence_all_leaves': same_sequence,
        'epoch': args.epoch, 'entry_substeps': expected_entry,
        'source_bindings': [identity(p) for p in (args.main_source, args.dec_source, args.electronic_source, geometry_source, coordinate_source, Path(__file__), *extra_sources)],
        'input_bindings': [identity(p) for p in [args.bp_case, *files.values()]],
        'execution_receipt': identity(args.execution_receipt) if args.execution_receipt else None,
        'build_binding_requirement': 'The caller must reconcile these source hashes with the actual executed binary receipt; this checker does not infer that relationship from filenames.',
        'leaves': results}


def self_test():
    # Exact graph control, explicitly not native molecular execution.
    edges = [(u, CORE) for u in range(CORE)]
    phi = [.375]*CORE+[.125]*(VERTICES-CORE)
    q = [.25]*CORE
    w = [1.]*CORE
    exact = exact_operator(q, phi, edges, w)
    assert exact['squared'] == 0
    assert exact['charge_sum']+exact['halo_reaction'] == 0
    wrong_screened = sum((Q.from_float(1e-4)*Q.from_float(p))**2 for p in phi[:CORE])
    assert wrong_screened > Q.from_float(TOL_SQ)
    phi_bad = phi.copy(); phi_bad[3] = .5
    assert exact_operator(q, phi_bad, edges, w)['squared'] > Q.from_float(TOL_SQ)
    for bad in (float('nan'), 0., -1.):
        badw = w.copy(); badw[5] = bad
        try:
            exact_operator(q, phi, edges, badw)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid weight was accepted')
    assert represented_charge(.25, (f32(.6), f32(.8))) != .25+float(.15)
    # An explicit lower-level charge coefficient guard: .15f and decimal .15
    # are different operators after promotion to binary64.
    assert Q.from_float(ALPHA) != Q(3, 20)
    a=((64.,0.),(64.,0.),(64.,0.))
    b=((65.,0.),(64.,0.),(64.,0.))
    changed=((65.,f32(1e-6)),(64.,0.),(64.,0.))
    assert precise_weight(1.,a,b,a,b)==1.
    assert precise_weight(1.,a,changed,a,b)!=1.
    # A common exact translation must not erase low-only geometry.
    aa=((64.,0.),(64.,0.),(64.,0.))
    bb=((64.,f32(1e-7)),(64.,0.),(64.,0.))
    cc=((64.,f32(2e-7)),(64.,0.),(64.,0.))
    assert precise_weight(1.,aa,cc,aa,bb)!=1.
    bad=((64.,float('nan')),(64.,0.),(64.,0.))
    try:
        precise_weight(1.,aa,bad,aa,bb)
    except ValueError:
        pass
    else:
        raise AssertionError('nonfinite low component accepted')
    return {'scope': 'Checker-only exact algebra and malformed-input controls; not native acceptance',
            'control_count': 13, 'status': 'PASS', 'screening_negative_residual_l2': math.sqrt(float(wrong_screened)),
            'source_alpha': ALPHA, 'source_tol_squared': TOL_SQ,
            'script': identity(Path(__file__))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--prefix', type=Path)
    p.add_argument('--epoch', type=int, default=1)
    p.add_argument('--entry-substeps', type=int)
    p.add_argument('--bp-case', type=Path)
    p.add_argument('--main-source', type=Path, default=KERNELS/'nhs_biprism_variational_so3g_v10.cuh')
    p.add_argument('--dec-source', type=Path, default=KERNELS/'nhs_biprism_dec_pcg.cuh')
    p.add_argument('--ordered-source', type=Path,
                   help='Optional exact traversal source snapshot for a reviewed indexed DEC epoch')
    p.add_argument('--potential-source', type=Path,
                   help='Optional exact potential-owner source snapshot for a reviewed retained-FP64 epoch')
    p.add_argument('--ordered-control-source', type=Path,
                   help='Optional exact ODP2 native-control source snapshot for a retained-FP64 epoch')
    p.add_argument('--electronic-source', type=Path, default=KERNELS/'nhs_biprism_electronic_discrete.cuh')
    p.add_argument('--execution-receipt', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not args.self_test and (args.prefix is None or args.bp_case is None or args.epoch < 1):
        p.error('--prefix, --bp-case and positive --epoch are required')
    result = self_test() if args.self_test else check_epoch(args)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': identity(args.output), 'status': result['status']}))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
