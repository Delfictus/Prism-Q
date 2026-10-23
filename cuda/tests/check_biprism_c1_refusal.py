#!/usr/bin/env python3
"""Decode actual EQR1/2 and CFR1/2 C1 refusals without admitting a trajectory.

The primitive replay evaluates the recorded frozen Hermitian matrix. Alternative
represented candidates are local feasibility evidence only: charge, DEC, the
nonlinear Hamiltonian, reciprocal work and full endpoint energy still require
native rebuilding and the unchanged production gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import biprism_exchange_layout as EX
import struct
import check_biprism_spinor_owner as SPINOR
import sys

import check_biprism_norm_refusal as NORM
import biprism_norm_lattice_reference as LATTICE

ENGINE = Path(__file__).resolve().parents[2]
MAIN = ENGINE / 'crates/prism-gpu/src/kernels/nhs_biprism_variational_so3g_v10.cuh'
ELECTRONIC = MAIN.with_name('nhs_biprism_electronic_discrete.cuh')
ELECTRONIC_SHA = 'a416c303a31602bc1ae077cc17874a6b59d5048a792726a334b30265ef19eedc'
LEAVES, CORE, STRIDE, TILE, BANK = 8, 192, 40704, 225408, 11648
PREFIX, TAIL, SCALAR, PHI = 13056, 13480, 30720, 38656
EQR, CFR, CHT, RRF = 0x45515231, 0x43465231, 0x43485431, 0x52524632
EQR3, CFR3 = 0x45515233, 0x43465233
EQR2, CFR2 = 0x45515232, 0x43465232
RRF3, CHT2, HISTORY2 = 0x52524633, 0x43485432, 11520
TOL = 1e-7
NATIVE_TOL = NORM.f32(TOL)
FMA = NORM.LIBM.fma
IDENTITY = ('magic', 'version', 'leaf', 'vertex', 'substep', 'picard', 'sequence',
            'iteration', 'flags', 'packed_atom', 'accepted_at_entry', 'reserved')
EQ_VALUES = ('connection', 'omega', 'h', 'norm_start', 'norm_outer', 'norm_candidate',
             'norm_trial', 'equation_candidate', 'equation_trial', 'leaf_equation', 'phi', 'charge')
CONT_VALUES = ('connection', 'omega', 'h', 'norm_start', 'norm_trial', 'norm_next',
               'change', 'equation_trial', 'equation_next', 'leaf_change', 'leaf_equation', 'phi')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same(a, b):
    return struct.pack('<d', a) == struct.pack('<d', b)


def replay_close(a, b, label):
    # This checks implementation replay, never relaxes an admission threshold.
    # Native compiler contraction can differ from the independent scalar CPU
    # tree. Retain both quantities and require a small FP64 evaluation envelope.
    require(math.isfinite(a) and math.isfinite(b), f'{label}: nonfinite replay')
    require(abs(a-b) <= 128*sys.float_info.epsilon*(1+abs(a)+abs(b)), f'{label}: primitive replay mismatch')


def replay_or_derive(record, key, value, label):
    if record['version']==3:
        require(math.isfinite(value),f'{label}: nonfinite FP64 derived value');record[key]=value
    elif record['version'] == 1:
        replay_close(record[key], value, label)
    else:
        # Version2 deliberately omits this redundant GPU calculation. Its raw
        # slot is reserved +0; the independent CPU value is labeled as derived.
        require(same(record[key], 0.), f'{label}: nonzero version2 reserved slot')
        require(math.isfinite(value), f'{label}: nonfinite derived value')
        record[key] = value


def apply_h(H, p):
    a, d, b, c = H
    x, y, z, w = p
    return (a*x + FMA(b, z, -c*w), a*y + FMA(b, w, c*z),
            FMA(b, x, c*y) + d*z, FMA(b, y, -c*x) + d*w)


def equation(start, endpoint, H, omega, h):
    scale = max(omega, *(abs(.5*h*x) for x in H))
    require(math.isfinite(scale) and scale > 0, 'invalid frozen equation scale')
    hs = apply_h(H, tuple(x+y for x, y in zip(start, endpoint)))
    delta = tuple(y-x for x, y in zip(start, endpoint))
    a, b = omega/scale, .5*h/scale
    residual = (FMA(a, delta[0], -b*hs[1]), FMA(a, delta[1], b*hs[0]),
                FMA(a, delta[2], -b*hs[3]), FMA(a, delta[3], b*hs[2]))
    return max(map(abs, residual))


def parse_record(raw, offset, kind):
    size = 256 if kind == 'equation' else 272
    require(offset >= 0 and offset+size <= len(raw), 'truncated C1 diagnostic record')
    result = dict(zip(IDENTITY, struct.unpack_from('<12I', raw, offset)))
    result['kind'] = kind
    if result['version']==3:
        result['start']=struct.unpack_from('<4d',raw,offset+48)
        if kind=='equation':
            result['candidate']=struct.unpack_from('<4d',raw,offset+80)
            result['trial']=struct.unpack_from('<4d',raw,offset+112)
            result['H']=struct.unpack_from('<4d',raw,offset+144)
            names=('connection','omega','h','norm_outer','equation_candidate','leaf_equation','phi','charge')
            result.update(zip(names,struct.unpack_from('<8d',raw,offset+176)))
            require(raw[offset+240:offset+256]==bytes(16),'EQR3 reserved tail')
        else:
            result['trial']=struct.unpack_from('<4d',raw,offset+80)
            result['next']=struct.unpack_from('<4d',raw,offset+112)
            result['H']=struct.unpack_from('<4d',raw,offset+144)
            names=('connection','omega','h','change','equation_trial','equation_next','leaf_change','leaf_equation','phi')
            result.update(zip(names,struct.unpack_from('<9d',raw,offset+176)))
            require(raw[offset+248:offset+272]==bytes(24),'CFR3 reserved tail')
        return result
    result['start'] = struct.unpack_from('<4f', raw, offset+48)
    if kind == 'equation':
        result['candidate'] = struct.unpack_from('<4f', raw, offset+64)
        result['trial'] = struct.unpack_from('<4d', raw, offset+80)
        result['H'] = struct.unpack_from('<4d', raw, offset+112)
        result.update(zip(EQ_VALUES, struct.unpack_from('<12d', raw, offset+144)))
    else:
        result['trial'] = struct.unpack_from('<4d', raw, offset+64)
        result['next'] = struct.unpack_from('<4d', raw, offset+96)
        result['H'] = struct.unpack_from('<4d', raw, offset+128)
        result.update(zip(CONT_VALUES, struct.unpack_from('<12d', raw, offset+160)))
    require(raw[offset+size-16:offset+size] == bytes(16), 'nonzero C1 diagnostic reserved tail')
    return result


def frozen_candidates(record):
    """Bounded original/shell/pivot witnesses, not nonlinear feasibility proof."""
    trial, n, outer = record['trial'], record['norm_start'], record['norm_outer']
    H = (*record['H'][:3], record['H'][3]+record['connection'])
    seen, candidates = set(), []
    for shell in (0, 1):
        candidates.extend((f'shell{shell}', p) for p in NORM.candidates(trial, shell))
    _, _, slices = LATTICE.complete_norm_lattice(trial, n, outer)
    candidates.extend((f'pivot{pivot}:{mask}', result.candidate)
                      for pivot, mask, result in slices if result.candidate is not None)
    admitted = []
    counts = {'unique': 0, 'norm': 0, 'joint': 0}
    for domain, p in candidates:
        key = tuple(NORM.bits(x) for x in p)
        if key in seen:
            continue
        seen.add(key); counts['unique'] += 1
        if not NORM.admitted(p, n, outer, absolute_one=True):
            continue
        counts['norm'] += 1
        residual = equation(record['start'], p, H, record['omega'], record['h'])
        if residual <= NATIVE_TOL:
            counts['joint'] += 1
            admitted.append((NORM.distance(p, trial), residual, domain, p))
    admitted.sort(key=lambda x: (x[0], x[1], tuple(NORM.bits(v) for v in x[3])))
    return counts, admitted[0] if admitted else None


def compiled_picard_limit(source):
    values = re.findall(r'^#define MAX_VARIATIONAL_PICARD_ITERS ([1-9][0-9]*)$', source, re.M)
    require(len(values) == 1, 'missing or ambiguous compiled nuclear Picard bound')
    return int(values[0])


def inspect_buffers(numerical, exchange, tiles, convergence, *, search=True, picard_limit=12, stride=STRIDE,spinors=None):
    require(len(numerical) == LEAVES*stride, 'complete compiled numerical population required')
    bank_stride,footer_offset,_,bank_magic=EX.layout(exchange)
    require(len(exchange) == LEAVES*bank_stride and len(tiles) == LEAVES*TILE and len(convergence) == LEAVES*128,
            'complete eight-leaf auxiliary population required')
    records, lines = [], ['SCOPE=actual C1 refusal; frozen-H primitive replay only',
                         'FULL_TRAJECTORY_ACCEPTED=false', 'NONLINEAR_CANDIDATE_ADMISSION=false']
    for leaf in range(LEAVES):
        base = leaf*stride
        magic = struct.unpack_from('<I', numerical, base+PREFIX)[0]
        if magic in (CFR, CFR2, CFR3):
            kind, offset = 'continuous', PREFIX
        else:
            require(magic in (RRF, RRF3), f'leaf{leaf}: missing CFR1/2 or RRF2/3 prefix')
            trace = struct.unpack_from('<4I', numerical, base+PREFIX)
            require(trace[1] == 7 and 1 <= trace[2] <= 17, 'equation capture outside stage7')
            kind, offset = 'equation', TAIL
        r = parse_record(numerical, base+offset, kind)
        r['trace_schema'] = magic if kind == 'equation' else None
        require((r['magic'], r['version']) in (((EQR, 1), (EQR2, 2), (EQR3,3)) if kind == 'equation'
                                             else ((CFR, 1), (CFR2, 2), (CFR3,3))), 'C1 diagnostic version')
        if kind == 'equation' and magic == RRF3:
            require((r['magic'],r['version']) in ((EQR2,2),(EQR3,3)), 'RRF3 requires EQR2/3')
        require(r['leaf'] == leaf and 0 <= r['vertex'] < CORE and r['reserved'] == 0 and r['flags'] == 0,
                'C1 diagnostic identity or nonfinite selected vertex')
        require(0 <= r['substep'] < 4 and 0 <= r['picard'] < picard_limit and r['sequence'] > 0, 'C1 substep/Picard identity')
        v = r['vertex']; tile = tiles[leaf*TILE:(leaf+1)*TILE]
        packed = struct.unpack_from('<I', tile, 4*v)[0]
        omega = max(struct.unpack_from('<f', tile, 95232+4*v)[0], NORM.f32(1e-4))
        require(r['packed_atom'] == packed and same(r['omega'], omega) and r['h'] > 0, 'C1 selected vertex/weight binding')
        footer = leaf*bank_stride+footer_offset
        ident = struct.unpack_from('<12I', exchange, footer+80)
        require(ident[:5] == (bank_magic, leaf, r['accepted_at_entry'], r['substep'], r['sequence']), 'C1 footer epoch mismatch')
        require(ident[5] & 2 and ident[7] == 5 and ident[8:] == (0, 0, 0, 0), 'C1 footer is not a retained refusal')
        conv = struct.unpack_from('<32I', convergence, leaf*128)
        require(conv[0] == 0x42505631 and conv[1] == leaf and conv[4] & 2 and conv[5] == r['accepted_at_entry'], 'C1 convergence identity/commit mismatch')
        require(same(r['phi'], struct.unpack_from('<d', numerical, base+PHI+8*v)[0]), 'C1 full working potential binding')
        require(all(math.isfinite(x) for key in ('start', 'trial', 'H') for x in r[key]), 'nonfinite C1 primitive')
        require(all(math.isfinite(r[k]) for k in ('connection', 'omega', 'h', 'phi')), 'nonfinite C1 coefficient')
        H = (*r['H'][:3], r['H'][3]+r['connection'])
        for key in ('start', 'trial'):
            replay_or_derive(r, 'norm_'+key, NORM.norm(r[key]), f'{key} norm')
        trial_eq = equation(r['start'], r['trial'], H, r['omega'], r['h'])
        replay_or_derive(r, 'equation_trial', trial_eq, 'trial frozen-H equation')
        bank_start = struct.unpack_from('<d', exchange, leaf*bank_stride+8*v)[0]
        pop = FMA(r['start'][0], r['start'][0], r['start'][1]**2)-FMA(r['start'][2], r['start'][2], r['start'][3]**2)
        require(same(pop, bank_start), 'C1 start population mismatch')
        published = struct.unpack_from('<d', exchange, footer+16)[0]
        if r['version']==3:
            owner=SPINOR.read(spinors,leaf)
            require(owner['committed']==r['accepted_at_entry'], 'C1 FP64 owner chronology')
            require(struct.pack('<4d',*r['start'])==struct.pack('<4d',*owner['start'][v]), 'C1 exact start/owner binding')
            if kind=='equation':
                require(struct.pack('<4d',*r['candidate'])==struct.pack('<4d',*owner['candidate'][v]), 'C1 exact candidate/owner binding')
        if kind == 'equation':
            require(r['iteration'] == trace[3], 'EQR1/2 versus RRF2/3 round mismatch')
            require(all(map(math.isfinite, r['candidate'])), 'nonfinite represented candidate')
            replay_or_derive(r, 'norm_candidate', NORM.norm(r['candidate']), 'candidate norm')
            outer = struct.unpack_from('<4f', tile, 90112+16*v) if r['version']<3 else SPINOR.read(spinors,leaf)['accepted'][v]
            require(same(r['norm_outer'], NORM.norm(outer)), 'outer norm is not the retained physical input')
            residual = equation(r['start'], r['candidate'], H, r['omega'], r['h'])
            replay_close(r['equation_candidate'], residual, 'candidate frozen-H equation')
            if r['version'] >= 2:
                require(same(r['leaf_equation'], r['equation_candidate']), 'saved native argmax versus leaf maximum')
            else:
                replay_close(r['leaf_equation'], r['equation_candidate'], 'selected versus leaf maximum')
            require(same(r['leaf_equation'], published), 'EQR1 source/footer equation mismatch')
            require(same(r['leaf_equation'], struct.unpack_from('<d', numerical, base+SCALAR+13*8)[0]), 'EQR1 source/scalar equation mismatch')
            endpoint = r['candidate']
            amber = struct.unpack_from('<f', tile, 1024+16*v+12)[0]
            charge = .5*(FMA(NORM.f32(.15), FMA(r['start'][2], r['start'][2], r['start'][3]**2), amber)
                         + FMA(NORM.f32(.15), FMA(endpoint[2], endpoint[2], endpoint[3]**2), amber))
            require(same(charge, r['charge']), 'EQR1 full represented mean charge mismatch')
            line = f'leaf={leaf} kind=EQR{r["version"]} vertex={v} substep={r["substep"]} round={r["iteration"]} native_equation={r["equation_candidate"]:.17g} cpu_frozen_trial_equation={trial_eq:.17g}'
            if search and r['version']<3:
                counts, best = frozen_candidates(r)
                line += f' unique_candidates={counts["unique"]} norm_feasible={counts["norm"]} frozen_joint_feasible={counts["joint"]}'
                if best:
                    line += f' best_domain={best[2]} best_equation={best[1]:.17g} best_bits={tuple(hex(NORM.bits(x)) for x in best[3])}'
            lines.append(line)
        else:
            # Absolute terminal iteration selects the supported schema; a stale
            # prefix tag can never turn a historical12-row record into24 rows.
            if r['iteration'] == 11:
                history_offset, count, expected = TAIL, 12, (CHT, 1, 12, 0xfff)
            else:
                require(r['version'] in (2,3) and r['iteration'] == 23, 'CFR1/2 unsupported continuous iteration cap')
                history_offset, count, expected = HISTORY2, 24, (CHT2, 2, 24, 0xffffff)
            header = struct.unpack_from('<4I', numerical, base+history_offset)
            require(header == expected, 'CHT1/2 incomplete or invalid iteration history')
            history = [struct.unpack_from('<ddfI', numerical, base+history_offset+16+24*k) for k in range(count)]
            require(all(math.isfinite(x) and x >= 0 for row in history for x in row[:3]) and all(row[3] <= 192 for row in history), 'CHT1/2 invalid residual/iteration data')
            require(same(history[-1][0], r['leaf_change']) and same(history[-1][1], r['leaf_equation']), 'CFR1/2 versus CHT1/2 last iteration mismatch')
            require(same(r['leaf_change'], published), 'CFR1 source/footer change mismatch')
            change = max(abs(a-b) for a, b in zip(r['trial'], r['next']))
            require(same(change, r['change']) and same(r['change'], r['leaf_change']), 'CFR1 worst-change witness mismatch')
            replay_or_derive(r, 'norm_next', NORM.norm(r['next']), 'next norm')
            next_eq = equation(r['start'], r['next'], H, r['omega'], r['h'])
            replay_or_derive(r, 'equation_next', next_eq, 'next frozen-H equation')
            endpoint = r['trial']
            lines.append(f'leaf={leaf} kind=CFR{r["version"]} history=CHT{expected[1]} vertex={v} substep={r["substep"]} local_change={change:.17g} cpu_trial_equation={trial_eq:.17g} cpu_next_fixed_H_equation={next_eq:.17g}')
            lines.extend(f'leaf={leaf} iteration={k} change={row[0]:.17g} equation={row[1]:.17g} dec_residual={row[2]:.9g} dec_iterations={row[3]}' for k, row in enumerate(history))
        population = FMA(endpoint[0], endpoint[0], endpoint[1]**2)-FMA(endpoint[2], endpoint[2], endpoint[3]**2)
        require(same(population, struct.unpack_from('<d', exchange, leaf*bank_stride+1536+8*v)[0]), 'C1 trial population mismatch')
        records.append(r)
    require(all((r['kind'], r['version'], r['trace_schema'], r['substep'], r['picard'], r['sequence'], r['iteration'], r['accepted_at_entry']) ==
                (records[0]['kind'], records[0]['version'], records[0]['trace_schema'], records[0]['substep'], records[0]['picard'], records[0]['sequence'], records[0]['iteration'], records[0]['accepted_at_entry']) for r in records), 'all-leaf refusal epochs or schemas disagree')
    if records[0]['kind'] == 'continuous':
        require(max(r['change'] for r in records) > 1e-12 or max(r['leaf_equation'] for r in records) > NATIVE_TOL, 'recorded continuous target would have converged')
    lines.append('CHECKER_PASS=true; this is diagnostic integrity, not trajectory acceptance')
    return records, lines


def write_output(path, text):
    # Replays are separate observations; never replace a prior result.
    with Path(path).open('x') as output:
        output.write(text)


def nuclear_history_lines(files, stride=STRIDE):
    """Optional FP32 diagnostic history in proven inactive phase slots only."""
    raw=files['numerical'];lines=[]
    for leaf in range(LEAVES):
        if stride in (510720,1680768,1697920,1784448,1837696,1997440):
            from check_biprism_nuclear_budget import read_trace
            if struct.unpack_from('<I',raw,leaf*stride+(496768 if stride==510720 else 1666816))[0]==0:
                continue  # Entry refusal before a substep started.
            h,rows=read_trace(raw,leaf,stride)
            lines.append(f'leaf={leaf} nuclear_history_substep={h[1]} attempted_nuclear_iterations={h[2]} '
                         f'nonlinear_trials={h[3]} derivative_evaluations={h[4]} action_checks={h[5]} '
                         f'target_replays={h[6]} base_evaluations={h[7]} accepted_private_bases={h[8]} '
                         f'rejected_private_trials={h[9]} values=retained_native_FP64 complete=true')
            for p,row in enumerate(rows):
                dx,energy,alpha=row[:3]
                if all(map(math.isnan,(dx,energy,alpha))):
                    lines.append(f'leaf={leaf} nuclear_picard={p} C4_evaluated=false reason=electronic_trial_refused_or_incomplete')
                else:
                    lines.append(f'leaf={leaf} nuclear_picard={p} raw_residual={dx:.17g} global_energy={energy:.17g} alpha_used={alpha:.17g}')
            continue
        if stride >= 43008:
            history=leaf*stride+40704
            magic,sub,count,reserved=struct.unpack_from('<4I',raw,history)
            if magic==0x4e485034:
                require(sub in range(4) and count in range(49) and reserved==0, 'invalid complete nuclear history')
                lines.append(f'leaf={leaf} nuclear_history_substep={sub} attempted_nuclear_iterations={count} '
                             'values=retained_native_FP64 complete=true')
                for p in range(count):
                    dx,energy,alpha=struct.unpack_from('<3d',raw,history+16+24*p)
                    if all(map(math.isnan,(dx,energy,alpha))):
                        lines.append(f'leaf={leaf} nuclear_picard={p} C4_evaluated=false reason=electronic_trial_refused_or_incomplete')
                        continue
                    require(all(map(math.isfinite,(dx,energy,alpha))) and dx>=0, 'nonfinite complete nuclear history')
                    lines.append(f'leaf={leaf} nuclear_picard={p} raw_residual={dx:.17g} '
                                 f'global_energy={energy:.17g} alpha_used={alpha:.17g}')
                continue
        start=leaf*stride+SCALAR-75*4
        magic=struct.unpack_from('<I',raw,start)[0]
        if magic not in (0x4e485031,0x4e485032,0x4e485033):
            continue
        require('tiles' in files, 'nuclear history requires physical active-edge owner')
        edges=struct.unpack_from('<I',files['tiles'],leaf*TILE+225284)[0]
        require(0<edges<=1536-75, 'nuclear history overlaps active physical phases')
        sub,count=struct.unpack_from('<2f',raw,start+4)
        require(sub in range(4) and count in range(49 if magic in (0x4e485032,0x4e485033) else 25), 'invalid nuclear history identity/count')
        lines.append(f'leaf={leaf} nuclear_history_substep={int(sub)} attempted_nuclear_iterations={int(count)} '
                     'values=FP32_diagnostic_projections_of_native_FP64_reductions')
        for p in range(max(0,int(count)-24),int(count)):
            dx,energy,alpha=struct.unpack_from('<3f',raw,start+12+12*(p%24))
            if magic==0x4e485033 and all(map(math.isnan,(dx,energy,alpha))):
                lines.append(f'leaf={leaf} nuclear_picard={p} C4_evaluated=false reason=electronic_trial_refused_or_incomplete')
                continue
            require(all(map(math.isfinite,(dx,energy,alpha))) and dx>=0, 'nonfinite nuclear history')
            lines.append(f'leaf={leaf} nuclear_picard={p} raw_residual={dx:.9g} '
                         f'global_energy={energy:.9g} alpha_used={alpha:.9g}')
    return lines


def work_energy_lines(files, stride=STRIDE):
    """Decode retained C4 aggregates without inventing discarded force terms.

    These are native aggregate values, not a CPU re-evaluation at an inferred
    midpoint. C1 refusal supersedes C4 scratch, so its earlier scalars cannot
    be combined into a purported current-candidate work balance.
    """
    blob = files['numerical']
    require(len(blob) == LEAVES*stride, 'complete numerical population required')
    lines = ['SCOPE=retained native work aggregates; unavailable term owners are explicit',
             'FULL_TRAJECTORY_ACCEPTED=false',
             'NO_ACCEPTANCE_TOLERANCE_CHANGED=true']
    lines.extend(nuclear_history_lines(files,stride))
    totals = [0., 0., 0., 0., 0.]
    recomposition_operand_ulps = 0.
    complete = True
    for leaf in range(LEAVES):
        base = leaf*stride
        prefix = struct.unpack_from('<I', blob, base+PREFIX)[0]
        a = struct.unpack_from('<32d', blob, base+SCALAR)
        c1 = prefix in (CFR, CFR2, CFR3, RRF, RRF3)
        incomplete_c4 = False
        if stride in (510720,1680768,1697920,1784448,1837696,1997440):
            from check_biprism_nuclear_budget import read_trace
            if struct.unpack_from('<I',blob,base+(496768 if stride==510720 else 1666816))[0]==0:incomplete_c4=True
            else:
                _,rows=read_trace(blob,leaf,stride)
                incomplete_c4=not rows or all(map(math.isnan,rows[-1][:3]))
        elif stride >= 43008:
            magic, _, count, _ = struct.unpack_from('<4I', blob, base+40704)
            if magic == 0x4e485034:
                # nuclear_history_lines already validated the complete header
                # and every row. A final NaN row was published at Picard entry
                # and never replaced by a completed C4 result. DEC can refuse
                # without emitting a CFR/RRF record, so prefix alone is not a
                # sufficient ownership check for the old aggregate scalars.
                incomplete_c4 = count == 0 or all(map(math.isnan, struct.unpack_from(
                    '<3d', blob, base+40704+16+24*(count-1))))
        if c1 or incomplete_c4:
            complete = False
            stage = 'C1' if c1 else 'pre_C4_refusal'
            reason = 'C1_aliases_supersede_endpoint_state' if c1 else 'final_NHP4_row_has_no_C4_publication'
            lines.append(f'leaf={leaf} current_stage={stage} current_C4_balance=EVIDENCE_NOT_FOUND '
                         f'reason={reason} prior_C4_scalar_not_current_energy={a[27]:.17g}')
            for name in ('err_kin', 'err_metric', 'err_dec', 'err_xtb_berry', 'err_foliated'):
                lines.append(f'leaf={leaf} {name}=EVIDENCE_NOT_FOUND reason=no_current_C4_candidate')
            continue
        require(all(math.isfinite(a[i]) for i in (0,1,2,3,4,5,10,11,12,15,16,17,20,27)),
                f'leaf{leaf}: nonfinite C4 aggregate')
        # DT_METRIC = (T_end - T_mid1) - (T_start - T_mid0).
        # MID_WORK = T_mid1 - T_mid0 - F_used.dot(dx), evaluated on GPU.
        kinetic = a[1]+a[10]
        potential_delta = a[16]-a[0]-a[5]
        used_work = (a[0]-a[1])-a[10]
        electronic_work = a[12]+a[3]+a[17]
        foliated_chain = a[17]-a[20]
        # The remaining small closure includes the actual FP32 used-force
        # representation and separately rounded reductions. Do not label it
        # an individual DEC/xTB/metric physical term.
        other = (a[16]+a[17])-a[11]-a[10]-electronic_work
        values = (a[16]+a[17], a[11], a[10], electronic_work, other)
        # Bound the actual leaf-level additions/subtractions before cancellation
        # between leaves. Native168 has O(1) owned transfers but O(1e-7) net
        # energy; the ULPs of the final totals cannot bound those operations.
        recomposition_operand_ulps += sum(math.ulp(a[i]) for i in (16,17,11,10,12,3))
        recomposition_operand_ulps += sum(math.ulp(x) for x in values)
        totals = [x+y for x,y in zip(totals, values)]
        lines.extend([
            f'leaf={leaf} current_stage=C4 err_kin={kinetic:.17g} source=DT_METRIC_plus_MID_WORK_DEFECT',
            f'leaf={leaf} T_delta={a[0]:.17g} U_delta_derived={potential_delta:.17g} F_used_dot_dx_derived={used_work:.17g}',
            f'leaf={leaf} err_metric=EVIDENCE_NOT_FOUND reason=no_separate_U_metric_potential_or_component_force_capture '
            f'metric_kinetic_transport={a[1]:.17g}',
            f'leaf={leaf} err_dec=EVIDENCE_NOT_FOUND reason=component_force_and_original_C2_midpoint_not_retained',
            f'leaf={leaf} err_xtb_berry=EVIDENCE_NOT_FOUND reason=separate_xTB_energy_and_C1_xTB_work_not_retained',
            f'leaf={leaf} err_foliated={foliated_chain:.17g} source=owned_pair_delta_minus_discrete_work '
            f'physical_transfer_in={a[19]:.17g} coordinate_force=0_frozen_operator_weight',
            f'leaf={leaf} C4_exact_aggregate_partition total={values[0]:.17g} coordinate_projection_defect={values[1]:.17g} '
            f'midpoint_kinetic_work_defect={values[2]:.17g} joint_electronic_Berry_foliated_work={values[3]:.17g} '
            f'force_representation_and_reduction_closure={values[4]:.17g}',
        ])
    if complete:
        lines.append('all8_C4_aggregate_partition '+ ' '.join(f'{name}={value:.17g}' for name,value in zip(
            ('total','coordinate_projection_defect','midpoint_kinetic_work_defect',
             'joint_electronic_Berry_foliated_work','force_representation_and_reduction_closure'),totals)))
        # This is host recomposition roundoff, not an energy acceptance gate.
        # A fixed1e-12 bound is smaller than one FP64 ULP for large refused
        # energies. Account for the eight leaf sums and chained subtractions;
        # retain the actual physical defects above without any adjustment.
        recomposition = abs(totals[0]-math.fsum(totals[1:]))
        recomposition_bound = 16.*(recomposition_operand_ulps+sum(math.ulp(x) for x in totals))
        require(recomposition <= recomposition_bound, 'aggregate partition reconstruction mismatch')
        lines.append(f'host_partition_recomposition_error={recomposition:.17g} '
                     f'fp64_operation_bound={recomposition_bound:.17g} scientific_acceptance=false')
    lines.append('INDEPENDENT_FIVE_TERM_NATIVE_CAPTURE_COMPLETE=false')
    return lines


def run(args):
    dispatch_path = Path(args.dispatch).resolve(); dispatch = json.loads(dispatch_path.read_text())
    require(dispatch.get('native_execution') is True and dispatch.get('native_exit_code') == 4, 'actual controlled native refusal required')
    require(dispatch.get('sources_binary_inputs_unchanged') is True and dispatch.get('gpu_recovery_none_after') is True
            and dispatch.get('new_kernel_fault_records') == [], 'unstable or hardware-faulted native dispatch')
    require(isinstance(dispatch.get('boot_id'), str) and dispatch['boot_id'], 'missing actual boot identity')
    command = dispatch['command']; prefix = Path(command[4]).resolve()
    require(str(prefix) == str(Path(args.prefix).resolve()), 'prefix does not match actual native command')
    build_ref = dispatch['build_receipt']; build_path = Path(build_ref['path'])
    require(sha(build_path) == build_ref['sha256'], 'changed actual build receipt')
    build = json.loads(build_path.read_text())
    require(build.get('exit_code') == 0 and build.get('sources_unchanged') is True and build.get('header_overlays') is False, 'unqualified actual source build')
    require(sha(command[0]) == build['binary_sha256'] == dispatch['input_hashes'][command[0]], 'actual binary identity mismatch')
    for p in command[1:4]:
        require(sha(p) == dispatch['input_hashes'][p], 'changed native physical input')
    source = Path(args.main_source).resolve()
    require(sha(source) == build['input_hashes'][str(MAIN)] == dispatch['input_hashes'][str(MAIN)], 'main source is not the compiled diagnostic source')
    require(build['input_hashes'][str(ELECTRONIC)] == dispatch['input_hashes'][str(ELECTRONIC)] == ELECTRONIC_SHA, 'original electronic helper changed')
    resource = dispatch['main_compiler_resources']
    require(all(resource.get(k) == 0 for k in ('stack', 'spill_stores', 'spill_loads', 'cumulative_stack')), 'native resource contract failed')
    require(1 <= args.epoch <= dispatch['expected_epochs'], 'epoch outside actual dispatch scope')
    files = {}
    for key, suffix in (('numerical', 'numerical.bin'), ('exchange', 'exchange.bin'), ('tiles', 'tiles.bin'), ('convergence', 'conv.bin')):
        path = Path(f'{prefix}.epoch{args.epoch}.{suffix}'); data = path.read_bytes(); pin = dispatch['artifacts'].get(str(path))
        require(pin == {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}, f'unbound actual {key} readback')
        files[key] = data
    if 'd_spinor_owners;' in source.read_text():
        p=Path(f'{prefix}.epoch{args.epoch}.spinor.owners.bps1')
        require(dispatch['artifacts'].get(str(p))=={'sha256':sha(p),'bytes':p.stat().st_size},'unbound FP64 spinor capture')
        files['spinors']=p.read_bytes()
    sizes=re.findall(r'static_assert\(sizeof\(BiprismV10NumericalScratchFfi\)\s*==\s*(\d+)',source.read_text())
    require(len(sizes)==1 and int(sizes[0]) in (40704,43008,66048,70656,72960,97536,106752,143872,196992,198144,235136,349056,496768,510720,1680768,1697920,1784448,1837696,1997440),'compiled numerical ABI assertion')
    stride=int(sizes[0])
    if getattr(args, 'work_energy', False):
        lines = work_energy_lines(files,stride)
    else:
        _, lines = inspect_buffers(**files, picard_limit=compiled_picard_limit(source.read_text()), stride=stride)
    lines[:0] = [f'CHECKER_SHA256={sha(__file__)} NORM_REFERENCE_SHA256={sha(NORM.__file__)} LATTICE_REFERENCE_SHA256={sha(LATTICE.__file__)}',
                 f'DISPATCH={dispatch_path} SHA256={sha(dispatch_path)}', f'MAIN_SOURCE={source} SHA256={sha(source)}',
                 f'BINARY_SHA256={build["binary_sha256"]} EPOCH={args.epoch} NATIVE_EXIT_CODE=4']
    text = '\n'.join(lines)+'\n'
    if args.output:
        write_output(args.output, text)
    print(text, end='')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', required=True); parser.add_argument('--dispatch', required=True)
    parser.add_argument('--epoch', type=int, default=1); parser.add_argument('--main-source', default=str(MAIN))
    parser.add_argument('--output')
    parser.add_argument('--work-energy', action='store_true',
                        help='decode retained C4 aggregate work, explicitly report unavailable separate physical terms')
    run(parser.parse_args())
