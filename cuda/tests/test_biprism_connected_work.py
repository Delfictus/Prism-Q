#!/usr/bin/env python3
"""CPU-only adapter controls. Constructed buffers are never native evidence."""
import copy
import hashlib
from pathlib import Path
import struct
import sys
import tempfile

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
import check_biprism_connected_work as C
W = C.W
passed = 0


def check(name, call):
    global passed
    result = call()
    passed += 1
    print('PASS', name)
    return result


def refuse(name, call, contains=None):
    global passed
    try:
        call()
    except (W.Refusal, ValueError, KeyError) as error:
        if contains is not None and contains not in str(error):
            raise AssertionError((name, str(error), contains))
        passed += 1
        print('PASS refusal', name)
        return
    raise AssertionError('accepted negative control: '+name)


def fixture():
    rows = bytearray(3*8*256*464)
    ledgers = bytearray(3*8*288)
    shared = bytearray(3*896)
    terminals = bytearray(8*48)
    dt = W.f32(.0005)
    au = W.f32(dt*W.TIME_TO_AU)
    time = 0.
    for leaf in range(8):
        struct.pack_into('<4I', terminals, leaf*48, W.SHARED_MAGIC, leaf, 4, 0)
    for body in range(3):
        step = body+1
        time += dt
        struct.pack_into('<8I4fdQ8f8I', shared, body*896,
                         W.SHARED_MAGIC, 2, 0, step, step, step, 0, 1,
                         dt, au, dt, dt, time, 23, *([dt]*8), *([step]*8))
        for leaf in range(8):
            off = body*896+128+leaf*96
            struct.pack_into('<4I4f2i2I', shared, off, W.SHARED_MAGIC, leaf, 4, step,
                             300., 0., dt, 0., 0, -1, step*256, 0)
            struct.pack_into('<4f2i2I2fiI', shared, off+48, 300., 0., dt, 0., 0, -1,
                             step*256, 0, 1., 0., -1, 0)
            struct.pack_into('<12IQQd', ledgers, (body*8+leaf)*288,
                             W.WORK_MAGIC, leaf, 3, 0, step, step*4,
                             body if body else 0, body if body > 1 else 0,
                             int(body > 0), 3, 0, 0, step, body, au)
            for vertex in range(256):
                r = ((body*8+leaf)*256+vertex)*464
                struct.pack_into('<8I4f', rows, r, vertex, leaf, vertex, 3, step, step*4,
                                 body if body else 0, int(body > 0), .125, .125, 8., 12.25)
                struct.pack_into('<I', rows, r+84, int(vertex < 192))
                struct.pack_into('<6f', rows, r+148, 1., 1., 1., 0., 0., 0.)
                struct.pack_into('<6d', rows, r+416, 1., 1., 1., 0., 0., 0.)
    ww = struct.pack('<8I', W.WORK_MAGIC, 3, 0, 3, 1, 0, 0, 0)
    tw = struct.pack('<8I', W.SHARED_MAGIC, 3, 0, 3, 1, 0, 0, 0)
    final = bytes(shared[-896:-768])
    component = struct.pack('<16I', 0x42434331, 0, 3, 3, 12, *([0]*11))
    return [rows, ledgers, shared, ww, tw, bytes(terminals), final, component]


f = fixture()
wi, si = check('raw windows derive exact indices', lambda: C.derive_indices(*f))
check('derived accepted time chronology', lambda: W.check_shared(f[2], si))
ids = [tuple(range(256)) for _ in range(8)]
result = check('adapter indices invoke unchanged full numerical checker on zero-work CPU fixture',
               lambda: W.check_buffers(f[0], f[1], f[2], wi, si, ids))
assert result['rows'] == 6144 and result['full_c4_qualification_inferred'] is False

def mutated(arg, off, fmt, *values):
    b = list(f)
    raw = bytearray(b[arg])
    struct.pack_into(fmt, raw, off, *values)
    b[arg] = raw
    return b

refuse('unaccepted shared record cannot manufacture accepted index',
       lambda: C.derive_indices(*mutated(2, 4, '<I', 1)), 'not accepted')
refuse('unsealed work record cannot manufacture accepted index',
       lambda: C.derive_indices(*mutated(1, 8, '<I', 2)), 'not sealed')
refuse('partial row population', lambda: C.derive_indices(f[0][:-464], *f[1:]), 'incomplete')
refuse('wrong generation', lambda: C.derive_indices(*mutated(2, 56, '<Q', 24)), 'generation')
refuse('false native AMBER clock in component descriptor',
       lambda: C.derive_indices(*mutated(7, 24, '<I', 3)), 'false native-AMBER')
refuse('final owner mismatch', lambda: C.derive_indices(*mutated(6, 48, '<d', 3.)), 'final time')
refuse('reserved window word', lambda: C.derive_indices(*mutated(3, 28, '<I', 1)), 'window')
refuse('warm window cannot be relabeled cold', lambda: C.derive_indices(*mutated(3, 8, '<I', 2)), 'window')
bad = mutated(2, 896+56, '<Q', 24)
_, badsi = C.derive_indices(*bad)
refuse('generation transition inside complete journal', lambda: W.check_shared(bad[2], badsi), 'generation')
badrow = mutated(0, 248, '<d', 1e-30)
refuse('sealed metadata never substitutes for primitive work oracle',
       lambda: W.check_buffers(badrow[0], f[1], f[2], wi, si, ids), 'primitive oracle')

tsv = 'leaf\tvertex\tatom\tcore\n'+''.join(f'{l}\t{v}\t{v}\t{int(v<192)}\n' for l in range(8) for v in range(256))
idraw = struct.pack('<2048I', *range(256), *range(256), *range(256), *range(256),
                    *range(256), *range(256), *range(256), *range(256))
check('selected IDs tied to all actual TSV rows', lambda: C.selected_population(idraw, tsv.encode(), 256))
refuse('duplicate TSV vertex', lambda: C.selected_population(idraw, (tsv+'0\t0\t0\t1\n').encode(), 256), 'duplicate')
badid = bytearray(idraw);struct.pack_into('<I', badid, 4, 0)
refuse('selected ID remap', lambda: C.selected_population(badid, tsv.encode(), 256), 'differ')

case = bytearray(32+20*256);case[:8] = b'BPCASE02';struct.pack_into('<I', case, 8, 256)
hmr = struct.pack('<256f', *([12.25]*256));case[32+16*256:] = hmr
inverse = struct.pack('<256f', *([.125]*256));atomwork = bytes(8*256*8)
check('actual-input mass and zero-initialized atom-work binding',
      lambda: C.check_input_and_publication(f[0], ids, case, inverse, hmr, atomwork))
badmass = bytearray(inverse);struct.pack_into('<f', badmass, 0, .25)
refuse('true inverse substitution', lambda: C.check_input_and_publication(f[0], ids, case, badmass, hmr, atomwork), 'recorded mass')
badwork = bytearray(atomwork);struct.pack_into('<d', badwork, 16, 1.)
refuse('native publisher extra work', lambda: C.check_input_and_publication(f[0], ids, case, inverse, hmr, badwork), 'once-per-body')

endpoint_outputs = {}
for epoch in range(1, 4):
    ns = bytearray(8*38656)
    for leaf in range(8):
        for vertex in range(192):
            for axis in range(3):
                struct.pack_into('<f', ns, leaf*38656+34048+axis*768+vertex*4, 1.)
    endpoint_outputs[f'.epoch{epoch}.numerical.bin'] = bytes(ns)
    endpoint_outputs[f'.epoch{epoch}.au_coords.bin'] = bytes(8*256*12)
    endpoint_outputs[f'.epoch{epoch}.au_coords_low.bin'] = bytes(8*256*12)
    endpoint_outputs[f'.epoch{epoch}.post_bridge.native_coords_high.bin'] = bytes(8*256*12)
    endpoint_outputs[f'.epoch{epoch}.post_bridge.native_coords_low.bin'] = bytes(8*256*12)
    for frame,label in ((1,'native'),(2,'au')):
        owners=bytearray(8*96)
        for leaf in range(8):
            pointers=[0x100000+frame*0x100000+k*8*256*12+leaf*256*12 for k in range(6)]
            struct.pack_into('<4IQ2I8Q',owners,leaf*96,0x31504342,1,leaf,256,23,frame,7,*pointers,0,0)
        for stage in ('','entry.','post_bridge.'):
            endpoint_outputs[f'.epoch{epoch}.{stage}{label}_coordinate_owners.96b']=bytes(owners)
check('recorded metric/main velocity/coordinates bound to C4 primitives',
      lambda: C.check_endpoint_primitives(endpoint_outputs.__getitem__, f[0], ids, 256))
precise_outputs = dict(endpoint_outputs)
for epoch in range(1, 4):
    original = endpoint_outputs[f'.epoch{epoch}.numerical.bin']
    precise_outputs[f'.epoch{epoch}.numerical.bin'] = b''.join(
        original[leaf*38656:(leaf+1)*38656]+bytes(2048) for leaf in range(8))
check('FP64 potential suffix preserves exact C4 metric/velocity offsets on all eight leaves',
      lambda: C.check_endpoint_primitives(precise_outputs.__getitem__, f[0], ids, 256, 40704))
refuse('expanded numerical bank cannot be decoded with historical stride',
       lambda: C.check_endpoint_primitives(precise_outputs.__getitem__, f[0], ids, 256), 'extent')
bad_precise = dict(precise_outputs)
changed = bytearray(bad_precise['.epoch2.numerical.bin'])
struct.pack_into('<f', changed, 7*40704+34048+191*4, 2.)
bad_precise['.epoch2.numerical.bin'] = bytes(changed)
refuse('full-precision stride still detects leaf7 final-core metric corruption',
       lambda: C.check_endpoint_primitives(bad_precise.__getitem__, f[0], ids, 256, 40704), 'metric')
aliased = dict(endpoint_outputs)
for epoch in range(1, 4):
    for stage in ('', 'entry.', 'post_bridge.'):
        native_owners = aliased[f'.epoch{epoch}.{stage}native_coordinate_owners.96b']
        au_owners = bytearray(aliased[f'.epoch{epoch}.{stage}au_coordinate_owners.96b'])
        for leaf in range(8):
            au_owners[leaf*96+32:leaf*96+80] = native_owners[leaf*96+32:leaf*96+80]
        aliased[f'.epoch{epoch}.{stage}au_coordinate_owners.96b'] = bytes(au_owners)
refuse('native and AU coordinate frames alias',
       lambda: C.check_endpoint_primitives(aliased.__getitem__, f[0], ids, 256), 'native and AU')
for off, value, name in ((148, 2., 'metric'), (72, .001, 'main velocity'), (48, 1., 'coordinate'), (384, 1e-8, 'coordinate low'), (396, 1e-8, 'native coordinate')):
    bad = bytearray(f[0]);struct.pack_into('<f', bad, off, value)
    refuse('self-consistent but wrong recorded '+name,
           lambda bad=bad: C.check_endpoint_primitives(endpoint_outputs.__getitem__, bad, ids, 256), name)

# These are routing controls only: the requested forces below intentionally do
# not modify the zero-velocity fixture and are not passed as valid physics rows.
route_rows = bytearray(f[0]); outputs = {}; previous = bytes(8*C.CARRY_BYTES)
for body in range(3):
    epoch = body+1
    outputs[f'.epoch{epoch}.carry_before_delivery.bin'] = previous
    conv = bytearray(8*128)
    produced = bytearray(8*C.CARRY_BYTES)
    for leaf in range(8):
        struct.pack_into('<6I', conv, leaf*128, 0x42505631, leaf, 0, 0, 0, epoch*4)
        off = leaf*C.CARRY_BYTES
        struct.pack_into('<8I256I', produced, off, 1, epoch, epoch if body else 0, 0, epoch*4,
                         0, 0, 0, *range(256))
        struct.pack_into('<4f', produced, off+1056, 1., 0., 0., 0.)
        struct.pack_into('<4f', produced, off+1072, -1., 0., 0., 0.)
        if body:
            for vertex, sign in ((0, 1.), (1, -1.)):
                struct.pack_into('<f', route_rows, ((body*8+leaf)*256+vertex)*464+136, sign)
    outputs[f'.epoch{epoch}.conv.bin'] = conv
    outputs[f'.epoch{epoch}.produced_carry.bin'] = bytes(produced)
    outputs[f'.epoch{epoch}.published_work.464b'] = bytes(route_rows[body*8*256*464:(body+1)*8*256*464])
    outputs[f'.epoch{epoch}.published_work.288b'] = f[1][body*8*288:(body+1)*8*288]
    selected = bytearray(f[2][body*896:body*896+128])
    struct.pack_into('<I', selected, 4, 1)
    struct.pack_into('<II', selected, 16, body, body)
    struct.pack_into('<d', selected, 48, body*float(W.f32(.0005)))
    outputs[f'.epoch{epoch}.selected_time.128b'] = selected
    previous = bytes(produced)
outputs['.lif_work.carry.final.bin'] = previous
check('actual carry and accepted-component routing control',
      lambda: C.check_carry_and_publication(outputs.__getitem__, route_rows, f[1], f[2], ids))
badoutputs = dict(outputs)
badoutputs['.epoch2.carry_before_delivery.bin'] = bytes(8*C.CARRY_BYTES)
refuse('producer-to-delivery carry substitution',
       lambda: C.check_carry_and_publication(badoutputs.__getitem__, route_rows, f[1], f[2], ids), 'carry changed')
badrows = bytearray(route_rows);struct.pack_into('<f', badrows, 8*256*464+136, 2.)
badoutputs = dict(outputs)
badoutputs['.epoch2.published_work.464b'] = bytes(badrows[8*256*464:16*256*464])
refuse('requested force differs from actual carry',
       lambda: C.check_carry_and_publication(badoutputs.__getitem__, badrows, f[1], f[2], ids), 'requested force')
badoutputs = dict(outputs)
raw = bytearray(badoutputs['.epoch1.produced_carry.bin']);struct.pack_into('<f', raw, 1056, 2.)
badoutputs['.epoch1.produced_carry.bin'] = raw
refuse('unbalanced actual producer output',
       lambda: C.check_carry_and_publication(badoutputs.__getitem__, route_rows, f[1], f[2], ids), 'balanced')

execution = dict(exit_code=0, native_exit_code=0, native_execution=True,
                 sources_binary_inputs_unchanged=True, gpu_recovery_none_after=True,
                 new_kernel_fault_records=[], diagnostic_mode='--connected-work', baseline_repeat=False,
                 full_trajectory_accepted=False, expected_epochs=3,
                 boot_id='6b17f83c-c1d9-4003-a7ea-9b4422d00f42',
                 started_utc='2026-10-18T00:00:00+00:00', ended_utc='2026-10-18T00:00:01+00:00',
                 main_compiler_resources=dict(stack=0,cumulative_stack=0,spill_stores=0,spill_loads=0,registers=126))
build = dict(exit_code=0,sources_unchanged=True,header_overlays=False,canonical_source_tree=str(C.ENGINE))
check('complete actual-dispatch field contract', lambda: C.receipt_fields(execution, build))
for field, value in [('native_execution',False), ('boot_id',''), ('new_kernel_fault_records',[{}]),
                     ('sources_binary_inputs_unchanged',False),('full_trajectory_accepted',True),
                     ('diagnostic_mode','default_three_epoch'),('expected_epochs',True),('exit_code',False)]:
    e = copy.deepcopy(execution);e[field] = value
    refuse('dispatch '+field, lambda e=e: C.receipt_fields(e, build))
e = copy.deepcopy(execution);del e['main_compiler_resources']['cumulative_stack']
refuse('missing cumulative callee resource gate', lambda: C.receipt_fields(e, build), 'cumulative_stack')
refuse('noncanonical overlay build', lambda: C.receipt_fields(execution, dict(build,header_overlays=True)), 'canonical')
refuse('duplicate JSON key', lambda: W.strict_json('{"state":2,"state":1}'), 'duplicate')
refuse('nonfinite receipt token', lambda: W.strict_json('{"x":Infinity}'), 'nonfinite')
with tempfile.TemporaryDirectory(prefix='biprism-connected-cpu-') as tmp:
    path = Path(tmp)/'source.cuh';path.write_bytes(b'original source\n')
    files = C.Files();digest = hashlib.sha256(path.read_bytes()).hexdigest()
    check('exact source/readback identity', lambda: files.read(path, digest, 16))
    path.write_bytes(b'changed source\n')
    refuse('source mutation after accepted read', files.unchanged, 'hash mismatch')

print(f'CPU_ADAPTER_CONTROLS_PASS={passed}')
print('native_execution=0 constructed_buffers_are_not_native_work_evidence=1')

# Full covariance is authoritative even when the diagnostic FP32 view is unchanged.
metric64_outputs=dict(endpoint_outputs)
for epoch in range(1,4):
    src=endpoint_outputs[f'.epoch{epoch}.numerical.bin']; dst=bytearray(8*106752)
    for leaf in range(8):
        dst[leaf*106752:leaf*106752+38656]=src[leaf*38656:(leaf+1)*38656]
        for axis in range(3):
            for vertex in range(192): struct.pack_into('<d',dst,leaf*106752+97536+axis*1536+vertex*8,1.)
    metric64_outputs[f'.epoch{epoch}.numerical.bin']=bytes(dst)
check('FP64 metric suffix bound to all accepted work rows',lambda:C.check_endpoint_primitives(metric64_outputs.__getitem__,f[0],ids,256,106752))
corrupt=dict(metric64_outputs);d=bytearray(corrupt['.epoch2.numerical.bin'])
struct.pack_into('<d',d,7*106752+97536+191*8,1.+2.**-40)
corrupt['.epoch2.numerical.bin']=bytes(d)
refuse('sub-FP32 metric substitution cannot pass accepted-work binding',lambda:C.check_endpoint_primitives(corrupt.__getitem__,f[0],ids,256,106752),'FP64 metric')
