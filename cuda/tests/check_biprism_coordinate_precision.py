#!/usr/bin/env python3
"""Exact-input reference for the shared high/low coordinate helpers.

Preparation binds actual retained native004 input bits. Verification reads the
actual helper output, using exact rational arithmetic independently of its
TwoSum/FMA implementation. It does not assert a corrected engine trajectory.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as Q
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import struct

FMT_IN=struct.Struct('<ffddd')
FMT_OUT=struct.Struct('<ffddffffffffd')
assert FMT_IN.size==32 and FMT_OUT.size==64
FACTOR=1.8897261259077824


def f32(v):
    return struct.unpack('<f',struct.pack('<f',v))[0]


def bits(v):
    return struct.unpack('<I',struct.pack('<f',v))[0]


def adjacent(v,up):
    if v==0:return struct.unpack('<f',struct.pack('<I',1 if up else 0x80000001))[0]
    b=bits(v)+(1 if (v>0)==up else -1)
    return struct.unpack('<f',struct.pack('<I',b))[0]


def rn32(q):
    v=f32(float(q))
    assert math.isfinite(v)
    return min((v,adjacent(v,False),adjacent(v,True)),key=lambda x:(abs(Q(x)-q),bits(x)&1))


def pairq(p):
    return Q(p[0])+Q(p[1])


def ulp32(v):
    if v==0:return Q(1,2**149)
    a=adjacent(v,True);b=adjacent(v,False)
    gaps=[abs(Q(z)-Q(v)) for z in (a,b) if math.isfinite(z)]
    return max(gaps)


def bound(q,p):
    # The exact-pair output quantization plus a conservative FP64 arithmetic
    # bound. The actual represented native-input cases also obey the stricter
    # scientific absolute coordinate budget and exact reference-pair check.
    return 2*ulp32(p[1])+4*Q(math.ulp(float(q)))


def read_bound(receipt,path):
    p=Path(path);b=p.read_bytes();h=hashlib.sha256(b).hexdigest()
    meta=receipt['artifacts'][str(p)]
    assert meta['sha256']==h and meta['bytes']==len(b),p
    return b


def cases(execution):
    r=json.loads(execution.read_text())
    assert r['native_execution'] is True and r['native_exit_code']==4
    assert r['sources_binary_inputs_unchanged'] is True and r['new_kernel_fault_records']==[]
    prefix=r['command'][4]+'.epoch1'
    raw=read_bound(r,prefix+'.numerical.bin')
    h,=struct.unpack('<f',read_bound(r,prefix+'.berry_parity.h.f32le'))
    assert len(raw)==8*38656
    data=[]
    for leaf in range(8):
        x=struct.unpack_from('<576f',raw,leaf*38656)
        v=struct.unpack_from('<576f',raw,leaf*38656+9216)
        data.extend((xx,0.,h,vv,FACTOR) for xx,vv in zip(x,v))
    actual=len(data)
    # Explicit continuation references, not purported accepted native states.
    for hi,lo,hh,v,factor in list(data):
        target=Q(hi)+Q(hh)*Q(v);newhi=rn32(target);newlo=rn32(target-Q(newhi))
        data.append((newhi,newlo,hh,v,factor))
    rng=random.Random(0xBC0104)
    for e in range(-120,101,5):
        for sign in (-1.,1.):
            hi=f32(math.ldexp(sign*(1.+rng.random()),e))
            lo=f32(math.ldexp(rng.uniform(-.24,.24),e-23))
            data.append((hi,lo,f32(.25+rng.random()),f32(math.ldexp(rng.uniform(-1,1),e-8)),FACTOR))
    for hi in (0.,-0.,math.ldexp(1.,-149),-math.ldexp(1.,-149),1.,-1.,2**30,-2**30):
        for v in (0.,math.ldexp(1.,-149),1e-12,-1e-12):
            data.append((f32(hi),0.,1.,f32(v),FACTOR))
    # Product cancellation and a half-ULP boundary with a nonzero low part.
    data += [(f32(1.),f32(2**-25),1.+2**-27,1.-2**-27,FACTOR),
             (f32(-1.),0.,1.+2**-27,1.-2**-27,FACTOR),
             (f32(100.),f32(2**-30),5.1676716804504395,f32(-1e-10),FACTOR)]
    return data,actual


def verify(data,actual,raw):
    assert len(raw)==64*len(data)
    maxima={k:Q(0) for k in ('next','scale','back','four','difference')}
    exact_pairs=0;recovered_nonzero=0
    for i,(inp,record) in enumerate(zip(data,FMT_OUT.iter_unpack(raw))):
        hi,lo,h,v,factor=inp
        nh,nl,value,difference,sh,sl,bh,bl,fh,fl,ah,al,divided_value=record
        assert all(math.isfinite(x) for x in record),(i,'nonfinite helper output')
        start=Q(hi)+Q(lo);target=start+Q(h)*Q(v)
        p=(nh,nl);error=abs(pairq(p)-target)
        assert error<=bound(target,p),(i,'fma expansion error',float(error),float(bound(target,p)))
        assert value==float(pairq(p)),(i,'FP64 reconstruction')
        expected_diff=pairq(p)-start
        assert abs(Q(difference)-expected_diff)<=4*Q(math.ulp(float(expected_diff))),(i,'compensated difference')
        if i<actual:
            refhi=rn32(target);reflo=rn32(target-Q(refhi))
            exact_pairs+=bits(nh)==bits(refhi) and bits(nl)==bits(reflo)
            assert bits(nh)==bits(refhi) and bits(nl)==bits(reflo),(i,'actual-input exact reference split')
            assert error<Q(1,10**7),(i,'unchanged absolute coordinate criterion')
            recovered_nonzero+=nl!=0.
            maxima['next']=max(maxima['next'],error)
        expected_scale=pairq(p)*Q(factor)
        se=abs(pairq((sh,sl))-expected_scale)
        assert se<=bound(expected_scale,(sh,sl)),(i,'scale compensation')
        expected_back=pairq((sh,sl))/Q(factor)
        be=abs(pairq((bh,bl))-expected_back)
        assert be<=bound(expected_back,(bh,bl)),(i,'division compensation')
        fe=abs(pairq((fh,fl))-(start+4*Q(h)*Q(v)))
        assert fe<=8*bound(start+4*Q(h)*Q(v),(fh,fl)),(i,'four update accumulation')
        add_target=start+Q(float(Q(h)*Q(v)))
        assert abs(pairq((ah,al))-add_target)<=bound(add_target,(ah,al)),(i,'add reference')
        divided_target=pairq(p)/Q(factor)
        assert abs(Q(divided_value)-divided_target)<=8*bound(divided_target,(rn32(divided_target),rn32(divided_target-Q(rn32(divided_target))))),(i,'divided reconstruction')
        if i<actual:
            maxima['scale']=max(maxima['scale'],se);maxima['back']=max(maxima['back'],be)
            maxima['four']=max(maxima['four'],fe)
            maxima['difference']=max(maxima['difference'],abs(Q(difference)-expected_diff))
    assert recovered_nonzero>0
    print('actual004_input_components',actual,'all_reference_records',len(data))
    print('actual_exact_reference_pairs',exact_pairs,'nonzero_low_parts',recovered_nonzero)
    for key,value in maxima.items():print('actual_max_abs_'+key,float(value))
    # Negative oracle checks: replacing lows by zero or changing a high bit
    # must be detectable on real source-derived cases.
    row=list(FMT_OUT.unpack_from(raw,0));row[1]=0.
    t=Q(data[0][0])+Q(data[0][2])*Q(data[0][3])
    assert abs(Q(row[0])-t)>Q(1,10**7),'negative control must exercise lost physical displacement'
    print('negative_premature_float_collapse DETECTED')
    print('PASS CPU/exact-input coordinate foundation; no native trajectory acceptance')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--execution',required=True,type=Path)
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--output',type=Path)
    a=p.parse_args();data,actual=cases(a.execution)
    expected=b''.join(FMT_IN.pack(*row) for row in data)
    if a.output is None:
        with a.input.open('xb') as f:f.write(expected)
        print('prepared',len(data),'actual',actual,'input_sha256',hashlib.sha256(expected).hexdigest())
    else:
        assert a.input.read_bytes()==expected,'reference input mutation'
        raw=a.output.read_bytes();print('output_sha256',hashlib.sha256(raw).hexdigest())
        verify(data,actual,raw)


if __name__=='__main__':main()
