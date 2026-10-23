"""Independent NHE5/6/7 full-evaluation reader; no physical admission authority."""
import math
import struct

STRIDE=1680768
OFFSET=1666816
MAGIC=0x3545484e
CONTINUATION_MAGIC=0x3645484e
BOUNDED_CONTINUATION_MAGIC=0x3745484e
KRYLOV64_MAGIC=0x3845484e
ARC_MAGIC=0x3945484e
ARC_STRIDE=1697920
ARC_HISTORY=1680768+10896
EXTENDED_ARC_MAGIC=0x4145484e
EXTENDED_ARC_STRIDE=1784448
FORWARD_ARC_MAGIC=0x4245484e
FORWARD_ARC_STRIDE=1837696
PRECISE_H_MAGIC=0x4345484e
TANGENT_MAGIC=0x4445484e
COMPLETION_MAGIC=0x4545484e
COMPLETION_STRIDE=1997440
ARC_MAGICS=(ARC_MAGIC,EXTENDED_ARC_MAGIC,FORWARD_ARC_MAGIC,PRECISE_H_MAGIC,TANGENT_MAGIC,COMPLETION_MAGIC)

def limits(magic):
    return (3072,512) if magic==COMPLETION_MAGIC else (1536,256) if magic in (FORWARD_ARC_MAGIC,PRECISE_H_MAGIC,TANGENT_MAGIC) else (1024,256) if magic==EXTENDED_ARC_MAGIC else (192,48)

def evaluation_offset(i):
    return OFFSET+128+72*i if i<192 else 1697920+72*(i-192) if i<1024 else 1784448+72*(i-1024) if i<1536 else 1837696+72*(i-1536)

def arc_history_offset(i):
    return ARC_HISTORY+32*i if i<192 else 1757824+32*(i-192) if i<1024 else 1821312+32*(i-1024) if i<1536 else 1948288+32*(i-1536)

def original_target_reached(header):
    full=16 if header[0] in (BOUNDED_CONTINUATION_MAGIC,KRYLOV64_MAGIC)+ARC_MAGICS else 4
    if header[0] in ARC_MAGICS and live_h(header)!=struct.unpack('<f',struct.pack('<I',header[10]))[0]:return False
    return header[0]==MAGIC or (header[11] in (0,full) and header[12]==0)

def live_h(header):
    if header[0] in (PRECISE_H_MAGIC,TANGENT_MAGIC,COMPLETION_MAGIC):return struct.unpack('<d',struct.pack('<II',*header[14:16]))[0]
    if header[0] in ARC_MAGICS:return struct.unpack('<f',struct.pack('<I',header[13]))[0]
    if header[0] not in (CONTINUATION_MAGIC,BOUNDED_CONTINUATION_MAGIC,KRYLOV64_MAGIC):return None
    original=struct.unpack('<f',struct.pack('<I',header[10]))[0]
    factor=(header[11]/16. if header[11] else 1.) if header[0] in (BOUNDED_CONTINUATION_MAGIC,KRYLOV64_MAGIC) else {0:1.,1:.125,2:.25,3:.5,4:1.}[header[11]]
    # The producer stores this product into invocation.h (float). Multiplying
    # the decoded operands only in Python FP64 invents a different timestep
    # at non-power-of-two stages, changing C1 and the work row reconstruction.
    return struct.unpack('<f',struct.pack('<f',original*factor))[0]

def terminal_tangent_refusal(header,rows,count,mode,tangent,iteration,projected,column_norm2=None):
    """Identify the failed tangent, never an unpublished physical C4 state."""
    capacity=count==64
    breakdown=(0<=count<64 and column_norm2 is not None
               and math.isfinite(column_norm2) and 0.<=column_norm2<=1e-28)
    column=63 if capacity else count
    if (header[0] not in ARC_MAGICS or len(rows)!=header[2] or len(rows)<2
        or not all(math.isnan(x) for x in rows[-1])
        or not all(math.isfinite(x) for row in rows[:-1] for x in row)
        or not (capacity or breakdown) or mode!=2 or tangent not in (1,2) or iteration!=header[2]-2
        or not math.isfinite(projected) or projected<=1e-6
        or rows[-2][7]!=1000*header[11]+700+column):
        raise ValueError('unbound terminal tangent refusal identity')
    return dict(substep=header[1],evaluated_probe=iteration,column=column,stage=header[11],
                reason='column_capacity' if capacity else 'derivative_breakdown',column_norm2=column_norm2,
                projected_residual_squared=projected,required_max=1e-6,
                following_unpublished_row=header[2]-1,current_C4_candidate=False)

def validate_paired_probe_weights(rows):
    """Bind signed FD pairs and the bounded reverse tangent-action checks.

    Both actual C4 evaluations remain separate rows. Require the preceding
    positive sample from the same immutable base/column. Only original-h
    derivatives and private tangent derivatives permit symmetric FD pairs.
    Private tangent actions may additionally follow their seven positive
    checks with -100 then .1 weight multipliers (reverse1e-7 and1e-8).
    These are action measurements, never actual nonlinear corrections.
    """
    for i,row in enumerate(rows):
        if not math.isfinite(row[6]) or row[6]>=0.:continue
        stage,phase=divmod(int(row[7]),1000)
        if stage in (1,2) and 801<=phase<=864:
            first=i
            while first and rows[first-1][7]==row[7]:first-=1
            group=rows[first:i+1]
            scales=(.1,.1,.1,.1,.3,1./3.,-100.,.1)
            if (len(group) not in (8,9) or group[0][6]<=0.
                or any(g[7]!=row[7] or g[5]!=row[5] for g in group)
                or any(not math.isclose(group[j+1][6],group[j][6]*scales[j],rel_tol=1e-14,abs_tol=0.)
                       for j in range(len(group)-1))):
                raise ValueError('unbound reverse tangent action weight')
            continue
        paired_kind=(stage==16 and 100<=phase<164) or (stage in (1,2) and 700<=phase<764)
        if (not paired_kind or not i or rows[i-1][7]!=row[7]
            or rows[i-1][6]!=-row[6] or rows[i-1][5]!=row[5]):
            raise ValueError('unpaired or non-derivative negative probe weight')

def read_trace(raw,leaf,stride=None):
    if stride is None:stride=len(raw)//8
    if stride not in (510720,STRIDE,ARC_STRIDE,EXTENDED_ARC_STRIDE,FORWARD_ARC_STRIDE,COMPLETION_STRIDE) or len(raw)!=8*stride or leaf not in range(8):
        raise ValueError('complete NHE5 population/extent required')
    base=leaf*stride+(496768 if stride==510720 else OFFSET)
    h=struct.unpack_from('<32I',raw,base)
    magic,sub,count,trials,probes,checks,replays,bases,accepted,rejected=h[:10]
    if magic==KRYLOV64_MAGIC and stride!=STRIDE:raise ValueError('Krylov64 trace extent')
    if magic==ARC_MAGIC and stride!=ARC_STRIDE:raise ValueError('arclength trace extent')
    if magic==EXTENDED_ARC_MAGIC and stride!=EXTENDED_ARC_STRIDE:raise ValueError('extended arclength trace extent')
    if magic in (FORWARD_ARC_MAGIC,PRECISE_H_MAGIC,TANGENT_MAGIC) and stride!=FORWARD_ARC_STRIDE:raise ValueError("forward arclength trace extent")
    if magic==COMPLETION_MAGIC and stride!=COMPLETION_STRIDE:raise ValueError("completion arclength trace extent")
    evaluation_limit,trial_limit=limits(magic)
    if magic not in (MAGIC,CONTINUATION_MAGIC,BOUNDED_CONTINUATION_MAGIC,KRYLOV64_MAGIC)+ARC_MAGICS or sub not in range(4) or not 0<=count<=evaluation_limit or trials>trial_limit:
        raise ValueError('invalid NHE5 identity or work bound')
    if magic==MAGIC:
        if any(h[10:]):raise ValueError('NHE5 reserved fields')
    elif magic in ARC_MAGICS:
        if h[11] not in (0,1,2,16) or h[12] not in (0,2,17) or any(h[16:] if magic in (PRECISE_H_MAGIC,TANGENT_MAGIC,COMPLETION_MAGIC) else h[14:]):raise ValueError('arclength identity')
        if not math.isfinite(live_h(h)) or live_h(h)<=0.:raise ValueError('invalid evaluated arclength timestep')
        if magic in (PRECISE_H_MAGIC,TANGENT_MAGIC,COMPLETION_MAGIC) and struct.pack('<f',live_h(h))!=struct.pack('<I',h[13]):raise ValueError('evaluated timestep projection mismatch')
    else:
        limit=16 if magic in (BOUNDED_CONTINUATION_MAGIC,KRYLOV64_MAGIC) else 4
        if h[11]>limit or h[12]>limit or any(h[13:]):raise ValueError('continuation identity')
        step=live_h(h)
        if not math.isfinite(step) or step<=0.:raise ValueError('invalid original timestep')
    if count!=trials+probes+checks+replays+bases or trials!=accepted+rejected:
        raise ValueError('Newton work classifications do not conserve evaluation count')
    rows=[struct.unpack_from('<9d',raw,leaf*stride+evaluation_offset(i) if magic in (EXTENDED_ARC_MAGIC,FORWARD_ARC_MAGIC,PRECISE_H_MAGIC,TANGENT_MAGIC,COMPLETION_MAGIC) else base+128+72*i) for i in range(count)]
    for i,row in enumerate(rows):
        if all(math.isnan(x) for x in row[:3]):
            if i!=count-1:raise ValueError('incomplete historical C4 evaluation')
        elif not all(math.isfinite(x) for x in row[:3]) or row[0]<0:
            raise ValueError('invalid C4 trace row')
        if not all(math.isnan(x) for x in row[3:]) and not all(math.isfinite(x) for x in row[3:]):
            raise ValueError('partially published Newton row')
        if magic in ARC_MAGICS and not all(math.isnan(x) for x in row[:3]):
            ah,defect,ds,active=struct.unpack_from('<4d',raw,leaf*stride+arc_history_offset(i))
            if not all(math.isfinite(x) for x in (ah,defect,ds,active)) or ah<=0. or active not in (0.,1.):raise ValueError('arclength evaluated row')
            if i==count-1 and ah!=live_h(h):raise ValueError('arclength candidate timestep mismatch')
        if magic in (CONTINUATION_MAGIC,BOUNDED_CONTINUATION_MAGIC,KRYLOV64_MAGIC)+ARC_MAGICS and math.isfinite(row[7]):
            stage,phase=divmod(int(row[7]),1000)
            invalid_stage=stage not in (0,1,2,16) if magic in ARC_MAGICS else stage>h[11]
            if row[7]!=int(row[7]) or invalid_stage or phase//100 not in ((0,1,2,4,5,6,7,8,9) if magic in (TANGENT_MAGIC,COMPLETION_MAGIC) else (0,1,2,4,5,9)):
                raise ValueError('invalid continuation row stage')
    validate_paired_probe_weights(rows)
    return h,rows
