#!/usr/bin/env python3
"""Actual native040 round-zero refusal: frozen-H feasibility, not admission.

The 512-byte fixture is the exact EQR2 record at numerical offsets
1*40704+13480 and 2*40704+13480 of the pinned native readback.  The relaxed
linear Cayley polytope is wider than the native 1e-7 equation gate by 1e-12
in each scaled component, well above binary64 expression-rounding error.
All other coordinates remain continuous in the bound.  Infeasibility here
cannot be repaired by enumerating more FP32 points at the same H.
"""
from fractions import Fraction as Q
import hashlib
import itertools
from pathlib import Path
import struct
import unittest

import check_biprism_c1_refusal as C
import test_biprism_c1_native033_shell as F

FIXTURE = Path(__file__).with_name('fixtures')/'biprism_native040_vertex130.bin'
FIXTURE_SHA = 'd43322e11ef8f8af5f6b8ba431865291893a384189b1926e18371148a6058c30'
RAW_SHA = 'dd2032694b83dc182a8c30038b47ecf82f704dd1130f0dfce50ce59a173f7caa'
DISPATCH_SHA = 'db78bf49f7ffa297e9e0bbb89a7d7c8cbf34818ec2aeba29b6222d14e61b3651'
BUILD_SHA = '357b11b944d5d003b3301022e26a07e24066b3e3bfecf20aef6c194fbf480a35'
MAIN_SHA = '1ba553f36b92371ccd58a03476c8b9e1dfd0183d148309384d76f93b384b30ad'


def dot(a,b): return sum((x*y for x,y in zip(a,b)),Q(0))


def solve(A,b):
    """Exact rational elimination; the fixture geometry needs no tolerance."""
    rows=[list(a)+[v] for a,v in zip(A,b)];n=len(b)
    for k in range(n):
        pivot=next((i for i in range(k,n) if rows[i][k]),None)
        if pivot is None: raise ValueError('singular')
        rows[k],rows[pivot]=rows[pivot],rows[k]
        d=rows[k][k];rows[k]=[v/d for v in rows[k]]
        for i in range(n):
            if i!=k:
                d=rows[i][k];rows[i]=[a-d*b for a,b in zip(rows[i],rows[k])]
    return [r[-1] for r in rows]


def records():
    raw=FIXTURE.read_bytes()
    if len(raw)!=512 or hashlib.sha256(raw).hexdigest()!=FIXTURE_SHA:
        raise ValueError('actual EQR2 fixture changed')
    result=[]
    for i in range(2):
        r=C.parse_record(raw,256*i,'equation')
        r['norm_start']=F.native_norm(r['start']);r['norm_candidate']=F.native_norm(r['candidate'])
        result.append(r)
    return result


def norm_bounds_at_fixed_h(r):
    # Clear denominators by using the unscaled Hermitian Cayley equation.
    H=(*r['H'][:3],r['H'][3]+r['connection']);a,d,b,c=map(Q,H)
    om=Q(r['omega']);half=Q(r['h'])/2
    A=[[om,-half*a,-half*c,-half*b],[half*a,om,half*b,-half*c],
       [half*c,-half*b,om,-half*d],[half*b,half*c,half*d,om]]
    start=list(map(Q,r['start']));trial=list(map(Q,r['trial']))
    rhs=[2*om*start[i]-dot(row,start) for i,row in enumerate(A)]
    scale=max(om,*(abs(half*v) for v in (a,d,b,c)))
    tol=(Q(C.NATIVE_TOL)+Q(1,10**12))*scale
    center=solve(A,rhs)
    cols=[solve(A,[Q(int(i==j)) for i in range(4)]) for j in range(4)]
    radius=sum(abs(cols[j][1])*tol for j in range(4))
    left,right=center[1]-radius,center[1]+radius
    value=F.next32(F.f32(float(left)),False);branches=[]
    while Q(value)<=right:
        if Q(value)>=left:
            fixed=Q(value)-trial[1]
            offset=[dot(row,trial)-v+row[1]*fixed for row,v in zip(A,rhs)]
            L=[[row[i] for i in (0,2,3)] for row in A]
            L+= [[-v for v in row] for row in L]
            B=[tol-v for v in offset]+[tol+v for v in offset]
            vertices=[]
            for chosen in itertools.combinations(range(8),3):
                try: p=solve([L[j] for j in chosen],[B[j] for j in chosen])
                except ValueError: continue
                if all(dot(row,p)<=bound for row,bound in zip(L,B)):
                    vertices.append([p[0],fixed,p[1],p[2]])
            if vertices:
                n=dot(trial,trial)
                # Convex norm maximum is at a vertex. Its tangent supplies a
                # lower bound throughout the polytope (not vertex-min norm).
                low=n+min(2*dot(trial,v) for v in vertices)
                high=max(dot([x+y for x,y in zip(trial,v)],
                             [x+y for x,y in zip(trial,v)]) for v in vertices)
                branches.append((value,low,high))
        value=F.next32(value,True)
    return branches


class Native040Vertex130(unittest.TestCase):
    def test_exact_round_zero_identity_and_native_equation(self):
        for i,r in enumerate(records()):
            self.assertEqual((r['magic'],r['version'],r['leaf'],r['vertex'],r['substep'],
                              r['picard'],r['iteration'],r['accepted_at_entry']),
                             (C.EQR2,2,i+1,130,1,2,0,12))
            H=(*r['H'][:3],r['H'][3]+r['connection'])
            self.assertAlmostEqual(C.equation(r['start'],r['candidate'],H,r['omega'],r['h']),
                                   r['equation_candidate'],delta=2e-14)
            self.assertGreater(r['equation_candidate'],3.3e-7)
            self.assertLess(C.equation(r['start'],r['trial'],H,r['omega'],r['h']),4e-10)
            counts,witness=C.frozen_candidates(r)
            self.assertGreater(counts['norm'],0);self.assertEqual(counts['joint'],0)
            self.assertIsNone(witness)

    def test_even_continuous_other_components_cannot_close_norm_at_fixed_h(self):
        for r in records():
            low=max(Q(t)-Q(1e-7) for t in (1.,r['norm_start'],r['norm_outer']))-Q(1,10**15)
            high=min(Q(t)+Q(1e-7) for t in (1.,r['norm_start'],r['norm_outer']))+Q(1,10**15)
            bounds=norm_bounds_at_fixed_h(r);self.assertEqual(len(bounds),4)
            for dominant,lower,upper in bounds:
                self.assertTrue(upper<low or lower>high)
                print(f"NATIVE040 leaf={r['leaf']} vertex=130 dominant={dominant:.17g} "
                      f"relaxed_norm=[{float(lower):.17g},{float(upper):.17g}] "
                      f"required=[{float(low):.17g},{float(high):.17g}]",flush=True)

    def test_exact_fixture_byte_corruption_refuses(self):
        raw=bytearray(FIXTURE.read_bytes());raw[0]^=1
        self.assertNotEqual(hashlib.sha256(raw).hexdigest(),FIXTURE_SHA)


if __name__=='__main__': unittest.main(verbosity=2)
