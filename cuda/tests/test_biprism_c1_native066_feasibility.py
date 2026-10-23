"""Exact-rational relaxation of the retained native066 vertex13 EQR2 gate.

This is a frozen-H certificate for ONE rejected numerical trial. It does not
prove that the nonlinear timestep lacks a solution. Three components are
allowed to be real, so rejection of this superset rules out every FP32 tuple
at this Hamiltonian. Equation and norm rounding are enclosed conservatively.
"""
from fractions import Fraction as F
import itertools
import unittest
import check_biprism_c1_refusal as C

# Retained raw source: native066 fixed_domain_16 epoch5, leaf0 EQR2 offset13480.
RECORD=bytes.fromhex("3252514502000000000000000d00000000000000030000004f00000000000000000000000d0000001000000000000000a90b1bbff5b44bbf87636abad18b163bf1fe7f3fcdb0a3bbd1772bbb9a6b393a7c9a793bdeffef3f80e39f7cda7574bf7d819a04fa6e65bf28f99f4a732d473f23303d4607cc51c0c20ebf85f89810408a9624a02bf0bb3f0000000000000000a1f04823e0e2a1bf000000a0bcee564000000020b2ab14400000000000000000a8ee72190000f03f00000000000000000000000000000000040dc3ea4387913e0000000000000000040dc3ea4387913ed1bca93eeac16f3f3d8f5b6edee69d3f00000000000000000000000000000000")


def dot(a,b):return sum((x*y for x,y in zip(a,b)),F(0))


def inverse(matrix):
    n=len(matrix)
    rows=[list(r)+[F(i==j) for j in range(n)] for i,r in enumerate(matrix)]
    for col in range(n):
        pivot=next(i for i in range(col,n) if rows[i][col])
        rows[col],rows[pivot]=rows[pivot],rows[col]
        a=rows[col][col];rows[col]=[v/a for v in rows[col]]
        for i in range(n):
            if i!=col:
                b=rows[i][col];rows[i]=[v-b*w for v,w in zip(rows[i],rows[col])]
    return [r[n:] for r in rows]


def relaxed_sections(record):
    a,d,b,c=(*record['H'][:3],record['H'][3]+record['connection'])
    scale=max(record['omega'],*(abs(.5*record['h']*x) for x in (a,d,b,c)))
    A=F(record['omega']/scale);B=F(.5*record['h']/scale)
    a,d,b,c=map(F,(a,d,b,c))
    R=[[0,-a,-c,-b],[a,0,b,-c],[c,-b,0,-d],[b,c,d,0]]
    M=[[F(i==j)*A+B*R[i][j] for j in range(4)] for i in range(4)]
    N=[[F(i==j)*A-B*R[i][j] for j in range(4)] for i in range(4)]
    inv=inverse(M);start=list(map(F,record['start']))
    rhs=[dot(row,start) for row in N];z=[dot(row,rhs) for row in inv]
    # Inputs are O(1), |H|<72, h<6, scale>90. A conservative
    # 100*epsilon64*16 bound encloses the scaled source-order products,
    # FMA/add operations. Widen further to 1e-12 in the equation and 1e-13
    # in squared norm; both exceed direct operation-count error bounds.
    assert max(map(abs,record['start']))<=1.000001
    assert max(abs(float(x)) for x in (a,d,b,c))<72
    assert record['h']<6 and scale>90
    tol=F(float(C.NATIVE_TOL))+F('1e-12')
    radius=sum(map(abs,inv[0]))*tol
    norm_start=C.NORM.norm(record['start'])
    lower=F(max(1.,norm_start,record['norm_outer']))-F(1e-7)-F('1e-13')
    upper=F(min(1.,norm_start,record['norm_outer']))+F(1e-7)+F('1e-13')
    # Enumerate every FP32 major-component value allowed by the full real
    # equation box. Remaining components need not be representable.
    x=C.NORM.f32(float(z[0]-radius))
    while F(x)>z[0]-radius:x=C.NORM.next32(x,False)
    sections=[]
    while F(x)<=z[0]+radius:
        vertices=[]
        for free in range(4):
            if not inv[0][free]:continue
            fixed=[j for j in range(4) if j!=free]
            for signs in itertools.product((-1,1),repeat=3):
                r=[F(0)]*4
                for j,sign in zip(fixed,signs):r[j]=sign*tol
                r[free]=(F(x)-z[0]-dot(inv[0],r))/inv[0][free]
                if abs(r[free])<=tol:
                    vertices.append([z[i]+dot(inv[i],r) for i in range(4)])
        if vertices:
            # Convex squared norm attains its maximum at a polytope vertex.
            # Dropping ||delta||^2 supplies a certified lower bound on min norm.
            high=max(dot(v,v) for v in vertices)
            low=dot(z,z)+min(2*dot(z,[v[i]-z[i] for i in range(4)]) for v in vertices)
            sections.append((x,low,high,lower,upper))
        x=C.NORM.next32(x,True)
    return sections


class Native066Feasibility(unittest.TestCase):
    def test_frozen_h_equation_box_has_no_joint_fp32_norm_endpoint(self):
        r=C.parse_record(RECORD,0,'equation')
        self.assertEqual((r['leaf'],r['vertex']),(0,13))
        sections=relaxed_sections(r)
        self.assertEqual(len(sections),4)
        for x,low,high,lower,upper in sections:
            with self.subTest(dominant_component=x):
                self.assertTrue(high<lower or low>upper)
                margin=max(lower-high,low-upper)
                self.assertGreater(margin,F('3e-10'))

    def test_nearest_fp32_witness_passes_equation_but_fails_norm(self):
        r=C.parse_record(RECORD,0,'equation');p=tuple(C.NORM.f32(x) for x in r['trial'])
        H=(*r['H'][:3],r['H'][3]+r['connection'])
        self.assertLess(C.equation(r['start'],p,H,r['omega'],r['h']),C.NATIVE_TOL)
        self.assertFalse(C.NORM.admitted(p,C.NORM.norm(r['start']),r['norm_outer'],absolute_one=True))


if __name__=='__main__':unittest.main()
