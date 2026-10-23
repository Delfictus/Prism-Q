#!/usr/bin/env python3
"""Independent primitive-row checks for actual Phase2 native measurement output.

Default --retained-reference checks the actual retained 960/1461 complex and
cold placeholders on CPU. --native-prefix checks the seven-case CUDA harness
readback; it never creates a launch or promotes the result to a trajectory.
The external runner must retain exact source/build/device/dispatch identities.
"""
from pathlib import Path
import argparse,ctypes,hashlib,math,re
from fractions import Fraction
import numpy as np
R=Path('/mnt/storage/prism-biprism-implementation-20261017/INTERNAL_QC/v10_engine_integration/reduction_synchronization/native_qualification_001')
TILE=225408;CORE=192;VERT=256;EDGES=1536
M=ctypes.CDLL('libm.so.6');M.fma.argtypes=[ctypes.c_double]*3;M.fma.restype=ctypes.c_double
fma=M.fma
M.fmaf.argtypes=[ctypes.c_float]*3;M.fmaf.restype=ctypes.c_float
M.sqrtf.argtypes=[ctypes.c_float];M.sqrtf.restype=ctypes.c_float
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arr(raw,off,dt,n):return np.frombuffer(raw,dtype=dt,count=n,offset=off).copy()
def graph(raw):
 nt,ne=arr(raw,225280,'<u4',2);assert(nt,ne)==(960,1461)
 edge=arr(raw,29696,'u1',3072).reshape(1536,2)[:ne].astype(int)
 # Verify the exact retained incidence population rather than trusting the count.
 t=arr(raw,5120,'u1',4096).reshape(1024,4)[:nt];te=arr(raw,9216,'<u2',6144).reshape(1024,6)[:nt]
 assert len(set(map(tuple,edge)))==ne and np.all(edge[:,0]<edge[:,1])
 pairs=np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
 for vs,es in zip(t,te):assert np.array_equal(np.sort(vs[pairs],axis=1),edge[es])
 return edge

def matrix(edge,w):
 L=np.zeros((256,256));u,v=edge.T
 np.add.at(L,(u,u),w);np.add.at(L,(v,v),w);np.add.at(L,(u,v),-w);np.add.at(L,(v,u),-w)
 return L

def solve(L,q,h):return np.r_[np.linalg.solve(L[:CORE,:CORE],q[:CORE]-L[:CORE,CORE:]@h),h]
def energy(L,q,p):return q[:CORE]@p[:CORE]-.5*p@L@p
def exact_residual_squared(phi,q,pairs,weights):
 p=list(map(lambda x:Fraction(float(x)),phi));r=list(map(lambda x:Fraction(float(x)),q[:CORE]))
 for (u,v),w in zip(pairs,weights):
  flux=Fraction(float(w))*(p[u]-p[v])
  if u<CORE:r[u]-=flux
  if v<CORE:r[v]+=flux
 return sum((x*x for x in r),Fraction(0))

def retained():
 seed=(R/'state.seed_tiles.bin').read_bytes();post=(R/'state.epoch1.phase2_tiles.bin').read_bytes()
 assert len(seed)==len(post)==8*TILE
 for leaf in range(8):
  a=seed[leaf*TILE:(leaf+1)*TILE];b=post[leaf*TILE:(leaf+1)*TILE];edges=graph(a);assert np.array_equal(edges,graph(b))
  assert np.all(arr(a,96256,'<f4',1461)==0) and np.all(arr(a,95232,'<f4',256)==1)
  assert a[:94208]==b[:94208]
  w=arr(b,96256,'<f4',1461).astype(float);assert np.all(np.isfinite(w)) and np.all(w>0)
  q=arr(b,1024,'<f4',1024).reshape(256,4)[:,3].astype(float);h=arr(b,94208,'<f4',256)[192:].astype(float)
  L=matrix(edges,w);p=solve(L,q,h);r=q-L@p;flux=L@p
  E=energy(L,q,p);reduced=.5*q[:CORE]@p[:CORE]-.5*p[CORE:]@flux[CORE:]
  assert abs(E-reduced)<1e-11 and np.linalg.norm(r[:CORE])<1e-12
  # Finite, non-neutral Dirichlet population: halo reactions carry its flux.
  assert abs(q[:CORE].sum()+flux[CORE:].sum())<1e-11
  h1=h.copy();h1[0]+=float(np.float32(.001));p1=solve(L,q,h1)
  W=-.5*(L@p+L@p1)[CORE:]@(h1-h)
  assert abs(energy(L,q,p1)-E-W)<1e-11
  print('retained_leaf',leaf,'seed_zero_weights_refused',1461,'star0_placeholder',256,'core_true_residual',np.linalg.norm(r[:CORE]),'boundary_work',W,'gauss',q[:CORE].sum()+flux[CORE:].sum())
 print('retained_seed_sha',sha(R/'state.seed_tiles.bin'),'retained_phase2_sha',sha(R/'state.epoch1.phase2_tiles.bin'),'GPU_execution=False')

V=np.dtype({'names':['packed','leaf','local','state','x','ref','psi','qamber','w0','w1','reserved','p0','pm','p1','b0','b1','q','rho0','rho1','population','onsite','anchor','pair','boundary','opcorr','bcorr','epss','epsp','xlo','reflo','entry','psi_precise'],
 'formats':['<u4']*4+[('<f4',(3,)),('<f4',(3,)),('<f4',(4,))]+['<f4']*3+['<u4']+['<f8']*3+['<f4']*2+['<f8']*10+['<f4']*2+[('<f4',(3,)),('<f4',(3,)),'<f8',('<f8',(4,))],
 'offsets':[0,4,8,12,16,28,40,56,60,64,68,72,80,88,96,100,104,112,120,128,136,144,152,160,168,176,184,188,192,204,216,224],'itemsize':256})
E=np.dtype({'names':['leaf','edge','u','v','base0','base1','live0','live1','cross','E0','Em','E1'],'formats':['<u4']*4+['<f4']*2+['<f8']*6,'offsets':list(range(0,24,4))+[24,32,40,48,56,64],'itemsize':80})
LDT=np.dtype({'names':['magic','leaf','state','fault','step','entry','ne','nt','generation','prep','body','kappa','res','iters','totals','count','mismatch','reserved','refinement'],
 'formats':['<u4']*8+['<u8']*3+['<f8',('<f4',(3,)),('<u4',(3,)),('<f8',(16,)),'<u8','<u4','<u4','<f8'],
 'offsets':[0,4,8,12,16,20,24,28,32,40,48,56,64,76,88,216,224,228,232],'itemsize':240})
def close(a,b):return np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.all(np.abs(np.asarray(a)-b)<=1e-9*(1+np.abs(a)+np.abs(b)))
def same_f32(a,b):
 a=np.asarray(a,dtype='<f4');b=np.asarray(b,dtype='<f4')
 return a.shape==b.shape and np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.array_equal(a.view('<u4'),b.view('<u4'))
def same_f64(a,b):
 a=np.asarray(a,dtype='<f8');b=np.asarray(b,dtype='<f8')
 return a.shape==b.shape and np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.array_equal(a.view('<u8'),b.view('<u8'))

def potential_state(prefix,stage,state):
 raw=Path(prefix+'.'+stage+'_potential_owners.128b').read_bytes();assert len(raw)==8*128
 accepted=np.fromfile(prefix+'.'+stage+'_phi_accepted.f64le',dtype='<f8').reshape(8,256)
 prepared=np.fromfile(prefix+'.'+stage+'_phi_prepared.f64le',dtype='<f8').reshape(8,256)
 assert np.all(np.isfinite(accepted)) and np.all(np.isfinite(prepared))
 intervals=[];tile_base=coord_base=None
 for leaf in range(8):
  o=raw[leaf*128:(leaf+1)*128];u=arr(o,0,'<u4',32);q=arr(o,0,'<u8',16)
  assert tuple(u[:4])==(0x31505042,1,leaf,state) and tuple(q[2:4])==(23,1)
  assert u[27]==0 and q[15]==0
  if tile_base is None:
   tile_base=int(q[4]);coord_base=int(q[5]);assert tile_base and tile_base%128==0 and coord_base and coord_base%16==0
  assert int(q[4])==tile_base+leaf*TILE and int(q[5])==coord_base+leaf*96
  for p in map(int,q[6:8]):
   assert p and p%8==0 and p+2048<2**64;intervals.append((p,p+2048))
  if state==0:assert not any(o[64:])
  else:
   assert tuple(u[16:18])==(0,0) and q[10]==1
   if state==1:assert tuple(u[18:20])==(0,0) and q[11]==0 and tuple(u[24:27])==(1,0,0) and q[14]==1
   else:assert tuple(u[18:20])==(0,1) and q[11]==2 and tuple(u[24:27])==(2,0,1) and q[14]==2
 intervals.sort();assert all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:]))
 return raw,accepted,prepared

def potential_allocation_unchanged(before,after):
 assert len(before)==len(after)==8*128
 for leaf in range(8):
  a=before[leaf*128:(leaf+1)*128];b=after[leaf*128:(leaf+1)*128]
  assert all(a[start:end]==b[start:end] for start,end in ((0,12),(16,64),(108,112),(120,128))), 'potential allocation identity changed'

def bind_cold_rows(tile,row,reference,reference_low,leaf):
 """Bind cold current==reference geometry to the harness's immutable planes.

 Cold initialization promotes the original tile values once into BPS1.
 Verify the exact promoted row before evaluating its charge.
 """
 assert len(tile)==TILE and row.shape==(VERT,) and 0<=leaf<8
 assert reference.ndim==2 and reference.shape[1]==3 and reference.shape[0]>=VERT
 assert reference_low.shape==reference.shape and np.all(np.isfinite(reference)) and np.all(np.isfinite(reference_low))
 packed=arr(tile,0,'<u4',VERT);ids=packed&0xffffff
 assert np.all(ids<reference.shape[0]) and len(set(map(int,ids)))==VERT
 assert np.array_equal(row['packed'],packed) and np.all(row['leaf']==leaf)
 assert np.array_equal(row['local'],np.arange(VERT)) and np.all(row['state']==4) and np.all(row['reserved']==0)
 refq=arr(tile,1024,'<f4',1024).reshape(VERT,4)
 for field in ('x','ref'):
  assert same_f32(row[field],refq[:,:3]) and same_f32(row[field],reference[ids]), 'cold geometry high-plane binding'
 for field in ('xlo','reflo'):
  assert same_f32(row[field],reference_low[ids]), 'cold geometry low-plane binding'
 assert same_f32(row['qamber'],refq[:,3]), 'cold charge binding'
 psi=arr(tile,90112,'<f4',1024).reshape(VERT,4)
 assert np.all(np.isfinite(psi))
 assert same_f64(row['psi_precise'],psi.astype('<f8')) and same_f32(row['psi'],psi), 'cold spinor promotion'
 rho1=np.array([fma(float(z[2]),float(z[2]),float(z[3])*float(z[3])) if i<CORE else 0. for i,z in enumerate(psi)])
 q=np.array([fma(float(np.float32(.15)),rho1[i],float(refq[i,3])) if i<CORE else float(refq[i,3]) for i in range(VERT)])
 assert same_f64(row['rho1'],rho1) and same_f64(row['q'],q), 'cold represented charge binding'
 return q

def live_weight(x,xlo,ref,reflo,pairs,base):
 """Independent exact-dyadic endpoint differences, FP64 radial evaluation."""
 def distance(a,al,b,bl):
  d=[float(Fraction(float(b[k]))+Fraction(float(bl[k]))-Fraction(float(a[k]))-Fraction(float(al[k]))) for k in range(3)]
  square=fma(d[0],d[0],fma(d[1],d[1],d[2]*d[2]))+float(np.float32(1e-12))
  return math.sqrt(square)
 out=[]
 for (u,v),weight in zip(pairs,base):
  r=distance(x[u],xlo[u],x[v],xlo[v]);r0=distance(ref[u],reflo[u],ref[v],reflo[v])
  out.append(float(max(weight,np.float32(1e-4)))*(r0/r))
 return np.asarray(out,dtype='<f8')

def read_onsite_lut(path):
 text=Path(path).read_text()
 body=re.search(r'c_gfn_xtb_lut\s*\[8\]\s*=\s*\{(.*?)\};',text,re.S)
 assert body is not None,'actual provider LUT definition absent'
 rows=[]
 for fields in re.findall(r'\{([^{}]+)\}',body.group(1)):
  values=[float(x.strip().removesuffix('f')) for x in fields.split(',')]
  assert len(values)==4
  rows.append(values)
 assert len(rows)==8
 return np.asarray(rows,dtype='<f4')

def bind_primitives(prefix,rows,edges,lut):
 before=Path(prefix+'.before_tiles.bin').read_bytes();after=Path(prefix+'.after_tiles.bin').read_bytes()
 assert len(before)==len(after)==8*TILE
 coords=np.fromfile(prefix+'.coordinates_au.f32le',dtype='<f4')
 assert coords.size%24==0 and coords.size//24>=256 and np.all(np.isfinite(coords))
 coords=coords.reshape(8,-1,3)
 def plane(suffix):
  p=np.fromfile(prefix+suffix,dtype='<f4');assert p.size==coords.size and np.all(np.isfinite(p))
  return p.reshape(coords.shape)
 lows=plane('.coordinates_au_low.f32le');reference=plane('.reference_au.f32le');reference_low=plane('.reference_au_low.f32le')
 native=plane('.coordinates_native.f32le');native_low=plane('.coordinates_native_low.f32le')
 if prefix.endswith('.low_plane'):
  unchanged=prefix.removesuffix('.low_plane')+'.unchanged'
  assert Path(prefix+'.coordinates_native.f32le').read_bytes()==Path(unchanged+'.coordinates_native.f32le').read_bytes()
  assert np.any(native_low!=0), 'low-only physical displacement was lost'
 before_owner,accepted,before_prepared=potential_state(prefix,'before',1)
 after_owner,accepted_after,prepared=potential_state(prefix,'after',2)
 potential_allocation_unchanged(before_owner,after_owner)
 assert same_f64(accepted,accepted_after),'preparation changed accepted full potential'
 for leaf in range(8):
  a=before[leaf*TILE:(leaf+1)*TILE];b=after[leaf*TILE:(leaf+1)*TILE]
  va=rows[leaf];ea=edges[leaf,:1461];pairs=graph(a)
  assert np.array_equal(pairs,graph(b))
  # Exact identity includes topology, references, spatial/leaf matrices and
  # represented spinors; only the actual Phase2 operator outputs may differ.
  assert a[:94208]==b[:94208] and a[225280:225288]==b[225280:225288] and a[225292:225296]==b[225292:225296]
  packed=arr(a,0,'<u4',256);ids=packed&0xffffff
  assert np.all(ids<coords.shape[1]) and len(set(map(int,ids)))==256
  refq=arr(a,1024,'<f4',1024).reshape(256,4)
  psi=arr(a,90112,'<f4',1024).reshape(256,4)
  assert np.array_equal(va['packed'],packed)
  assert same_f32(va['x'],coords[leaf,ids]) and same_f32(va['xlo'],lows[leaf,ids])
  assert same_f32(va['ref'],refq[:,:3]) and same_f32(va['ref'],reference[leaf,ids])
  assert same_f32(va['reflo'],reference_low[leaf,ids]) and np.all(va['reserved']==0)
  assert same_f32(va['psi_precise'],psi) and same_f32(va['psi'],psi) and same_f32(va['qamber'],refq[:,3])
  assert same_f32(va['w0'],arr(a,95232,'<f4',256)) and same_f32(va['w1'],arr(b,95232,'<f4',256))
  assert same_f32(va['b0'],np.maximum(va['w0'],np.float32(1e-4))) and same_f32(va['b1'],np.maximum(va['w1'],np.float32(1e-4)))
  assert same_f64(va['entry'],accepted[leaf]) and same_f64(va['p1'],prepared[leaf])
  assert same_f32(accepted[leaf],arr(a,94208,'<f4',256)) and same_f32(prepared[leaf],arr(b,94208,'<f4',256))
  assert same_f64(va['p0'][192:],accepted[leaf,192:]) and same_f64(va['pm'][192:],accepted[leaf,192:])
  prescribed=(va['qamber'][192:]*np.float32(.07957747)).astype('<f8')
  assert same_f64(va['p1'][192:],prescribed)
  element=(packed>>24)&7
  assert same_f32(va['epss'],lut[element,1]) and same_f32(va['epsp'],lut[element,2])
  assert np.array_equal(ea['leaf'],np.full(1461,leaf,dtype='<u4')) and np.array_equal(ea['edge'],np.arange(1461,dtype='<u4'))
  assert np.array_equal(np.c_[ea['u'],ea['v']],pairs)
  assert same_f32(ea['base0'],arr(a,96256,'<f4',1461)) and same_f32(ea['base1'],arr(b,96256,'<f4',1461))
  assert same_f64(ea['live0'],live_weight(va['x'],va['xlo'],va['ref'],va['reflo'],pairs,ea['base0']))
  assert same_f64(ea['live1'],live_weight(va['x'],va['xlo'],va['ref'],va['reflo'],pairs,ea['base1']))
 return before,after

def native(prefix,xtb_source):
 lut=read_onsite_lut(xtb_source)
 print('phase2_primitive_provider_source',str(Path(xtb_source).resolve()),'sha256',sha(xtb_source))
 for case in ['cold_placeholder','cold_leaf_refusal','unchanged','deformed','boundary','identity_refusal','low_plane']:
  p=str(prefix)+'.'+case
  cold=np.fromfile(p+'.initial_ledgers.240b',dtype=LDT);assert len(cold)==8
  initial_conv=np.fromfile(p+'.initial_convergence.128b',dtype='<u4').reshape(8,32)
  assert np.array_equal(initial_conv[:,0],np.full(8,0x42505631,dtype='<u4')) and np.array_equal(initial_conv[:,1],np.arange(8,dtype='<u4'))
  assert np.all(initial_conv[:,5]==0) and Path(p+'.initial_step.u32le').read_bytes()==bytes(4)
  if case in ('cold_placeholder','cold_leaf_refusal'):
   assert np.all(cold['state']!=4) and np.all(cold['fault']!=0) and np.all((initial_conv[:,4]&0x02000000)!=0)
   _,a,b=potential_state(p,'initial',0);assert not np.any(a) and not np.any(b)
   continue
  assert np.all(cold['state']==4) and np.all(cold['fault']==0) and np.all(cold['res'][:,0]<=1e-6)
  assert np.all(initial_conv[:,2:5]==0)
  _,initial_phi,initial_prepared=potential_state(p,'initial',1);assert not np.any(initial_prepared)
  initial_rows=np.fromfile(p+'.initial_vertices.256b',dtype=V).reshape(8,256)
  assert same_f64(initial_rows['p0'],initial_phi)
  provisional=Path(p+'.provisional_tiles.bin').read_bytes();initialized=Path(p+'.initial_tiles.bin').read_bytes()
  assert len(provisional)==len(initialized)==8*TILE
  reference=np.fromfile(p+'.reference_au.f32le',dtype='<f4')
  reference_low=np.fromfile(p+'.reference_au_low.f32le',dtype='<f4')
  assert reference.size%24==0 and reference.size//24>=VERT and reference_low.size==reference.size
  reference=reference.reshape(8,-1,3);reference_low=reference_low.reshape(reference.shape)
  for leaf in range(8):
   old=provisional[leaf*TILE:(leaf+1)*TILE];now=initialized[leaf*TILE:(leaf+1)*TILE]
   row=initial_rows[leaf];pairs=graph(now);assert np.array_equal(pairs,graph(old))
   assert old[:94208]==now[:94208] and old[95232:]==now[95232:]
   q=bind_cold_rows(now,row,reference[leaf],reference_low[leaf],leaf)
   assert same_f64(row['entry'],arr(old,94208,'<f4',256).astype('<f8'))
   assert same_f64(initial_phi[leaf,192:],row['entry'][192:]) and same_f32(initial_phi[leaf],arr(now,94208,'<f4',256))
   weights=live_weight(row['x'],row['xlo'],row['ref'],row['reflo'],pairs,arr(now,96256,'<f4',1461))
   exact=exact_residual_squared(initial_phi[leaf],q,pairs,weights)
   assert exact<=Fraction(1,10**12) and exact<=Fraction(float(np.float32(1e-12))), 'cold full-potential exact residual refusal'
   operator=matrix(pairs,weights.astype(float))
   refinement=energy(operator,q,initial_phi[leaf])-energy(operator,q,row['entry'])
   assert close(cold[leaf]['refinement'],refinement), 'cold numerical refinement diagnostic mismatch'
  led=np.fromfile(p+'.ledgers.240b',dtype=LDT);assert len(led)==8
  shared=Path(p+'.shared.128b').read_bytes();assert len(shared)==128 and int.from_bytes(shared[20:24],'little')==0
  assert Path(p+'.scheduled_step.u32le').read_bytes()==b'\x01\x00\x00\x00'
  conv=np.fromfile(p+'.convergence.128b',dtype='<u4').reshape(8,32)
  assert np.array_equal(conv[:,0],initial_conv[:,0]) and np.array_equal(conv[:,1],initial_conv[:,1]) and np.all(conv[:,5]==0)
  if case=='identity_refusal':
   assert np.all((led['fault']&32)!=0) and np.all(led['state']!=3)
   assert np.all((conv[:,4]&0x02000000)!=0)
   assert not any(Path(p+'.journal.240b').read_bytes())
   for suffix in ('potential_owners.128b','phi_accepted.f64le','phi_prepared.f64le'):
    assert Path(p+'.before_'+suffix).read_bytes()==Path(p+'.after_'+suffix).read_bytes(),'refused publication changed potential owner/bank'
   continue
  assert np.all(conv[:,2:5]==0)
  assert Path(p+'.journal.240b').read_bytes()==Path(p+'.ledgers.240b').read_bytes()
  rows=np.fromfile(p+'.vertices.256b',dtype=V).reshape(8,256);edges=np.fromfile(p+'.edges.80b',dtype=E).reshape(8,1536)
  bind_primitives(p,rows,edges,lut)
  populations=[]
  for v in rows:
   pop=np.array([fma(float(a[0]),float(a[0]),float(a[1])*float(a[1]))-fma(float(a[2]),float(a[2]),float(a[3])*float(a[3])) for a in v['psi_precise']])
   pop[192:]=1.
   assert np.array_equal(v['population'].view('<u8'),pop.astype('<f8').view('<u8'))
   populations.append(pop)
  for leaf in range(8):
   l=led[leaf];v=rows[leaf];e=edges[leaf,:1461];nxt=rows[(leaf+1)&7]
   assert l['magic']==0x42505735 and l['leaf']==leaf and l['state']==3 and l['fault']==0 and l['count']==1 and l['generation']==23 and l['prep']==1 and l['entry']==0
   assert l['ne']==1461 and l['nt']==960 and l['step']==1 and l['body']==0 and l['mismatch']==0 and l['reserved']==0 and np.isfinite(l['refinement'])
   assert np.array_equal(v['local'],np.arange(256)) and np.all(v['state']==3) and np.all(v['leaf']==leaf)
   assert np.array_equal(v['packed'][:192]&0xffffff,nxt['packed'][:192]&0xffffff)
   rho0=np.array([fma(float(a[0]),float(a[0]),float(a[1])*float(a[1])) for a in v['psi_precise']]);rho1=np.array([fma(float(a[2]),float(a[2]),float(a[3])*float(a[3])) for a in v['psi_precise']]);rho0[192:]=1;rho1[192:]=0
   q=np.array([fma(float(np.float32(.15)),rho1[i],float(v['qamber'][i])) if i<192 else float(v['qamber'][i]) for i in range(256)])
   assert close(v['q'],q) and close(v['rho0'],rho0) and close(v['rho1'],rho1)
   P=np.array([v[k].astype(float) for k in ('p0','pm','p1')]);pair=np.c_[e['u'],e['v']].astype(int)
   L0=matrix(pair,e['live0'].astype(float));L1=matrix(pair,e['live1'].astype(float));flux=np.array([L0@P[0],L1@P[1],L1@P[2]]);res=q-flux
   assert np.all(np.linalg.norm(res[:,:192],axis=1)<=1e-6)
   for potential,weights in ((P[0],e['live0']),(P[1],e['live1']),(P[2],e['live1'])):
    squared=exact_residual_squared(potential,q,pair,weights)
    assert squared<=Fraction(1,10**12) and squared<=Fraction(float(np.float32(1e-12))), 'full-potential exact residual refusal'
   assert same_f64(v['p0'][192:],v['pm'][192:])
   t=np.zeros(16);t[:3]=[energy(L0,q,P[0]),energy(L1,q,P[1]),energy(L1,q,P[2])]
   t[4]=-.5*P[0]@(L1-L0)@P[1];t[5]=.5*(P[1]-P[0])[:192]@(res[0]+res[1])[:192]
   t[7]=-.5*(P[2]-P[1])[192:]@(flux[1]+flux[2])[192:];t[8]=.5*(P[2]-P[1])[:192]@(res[1]+res[2])[:192]
   dw=v['w1'].astype(float)-v['w0'];t[9]=np.sum(dw[:192]*(v['epss'][:192]*rho0[:192]+v['epsp'][:192]*rho1[:192]))
   t[10]=float(l['kappa'])*np.sum(dw[:192]);dn=nxt['w1'].astype(float)-nxt['w0'];t[11]=np.sum(-.5*float(l['kappa'])*(dw[:192]+dn[:192])*populations[leaf][:192]*populations[(leaf+1)&7][:192])
   t[3]=t[1]-t[0];t[6]=t[2]-t[1];t[12]=t[2]-t[0]+sum(t[9:12]);t[13]=t[3]-t[4]-t[5];t[14]=t[6]-t[7]-t[8];t[15]=np.sum(np.abs(v['b1'][:192].astype(float)-v['b0'][:192]))
   assert close(t,l['totals']) and abs(t[13])<=1e-6 and abs(t[14])<=1e-6
   assert close(l['refinement'],t[0]-energy(L0,q,v['entry']))
   print('native_case',case,'leaf',leaf,'operator_boundary_work',t[12],'identity',t[13],t[14],'residual',np.linalg.norm(res[:,:192],axis=1).tolist())
 print('native_measurement_readback_pass=True physical_trajectory_acceptance=False')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--retained-reference',action='store_true');a.add_argument('--native-prefix');a.add_argument('--xtb-source');x=a.parse_args()
 if x.retained_reference:retained()
 if x.native_prefix:
  if not x.xtb_source:a.error('--native-prefix requires the build-bound --xtb-source')
  native(x.native_prefix,x.xtb_source)
 if not x.retained_reference and not x.native_prefix:a.error('choose --retained-reference and/or --native-prefix')
