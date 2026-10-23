"""BPS1 physical spinor storage reader; never reconstructs it from tile FP32."""
import math
import struct
from fractions import Fraction as Q
SIZE=21632
CORE=192

def require(ok, why):
    if not ok: raise ValueError(why)

def norm(p):
    # Match the declared binary64 FMA tree independently with exact rationals.
    return float(Q(p[0])*Q(p[0])+Q(float(p[1]*p[1])))+float(Q(p[2])*Q(p[2])+Q(float(p[3]*p[3])))

def read(raw,leaf):
    require(len(raw)==8*SIZE and leaf in range(8),'complete BPS1 population')
    b=raw[leaf*SIZE:(leaf+1)*SIZE]
    magic,version,l,state,allocation,topology,tile,coord,committed,step=struct.unpack_from('<4I4Q2I',b)
    require((magic,version,l,state)==(0x31535042,1,leaf,1),'BPS1 accepted schema/identity')
    require(all((allocation,topology,tile,coord)) and tile%128==0 and coord%16==0 and not any(b[56:128]),'BPS1 generation/address/reserved')
    accepted=[struct.unpack_from('<4d',b,128+32*u) for u in range(CORE)]
    require(all(math.isfinite(x) for p in accepted for x in p),'BPS1 nonfinite accepted state')
    require(all(abs(norm(p)-1.)<=1e-7 for p in accepted),'BPS1 accepted norm')
    return dict(accepted=accepted,committed=committed,step=step,allocation=allocation,topology=topology,tile=tile,coordinate=coord,
                start=[struct.unpack_from('<4d',b,6272+32*u) for u in range(CORE)],candidate=[struct.unpack_from('<4d',b,12416+32*u) for u in range(CORE)])

def endpoint(raw,tiles,committed,step):
    require(len(tiles)==8*225408,'BPS1 tile population')
    result=[]
    for l in range(8):
        o=read(raw,l);require(o['committed']==committed and o['step']==step,'BPS1 accepted chronology')
        for u,p in enumerate(o['accepted']):
            require(struct.pack('<4f',*p)==tiles[l*225408+90112+16*u:l*225408+90112+16*(u+1)],'BPS1 projection mismatch')
            require(struct.pack('<4d',*p)==struct.pack('<4d',*o['start'][u]),'BPS1 accepted/substep endpoint mismatch')
        result.append(o)
    return result
