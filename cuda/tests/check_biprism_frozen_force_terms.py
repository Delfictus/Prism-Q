"""Independent CPU reconstruction of the last evaluated C2 force, not a trajectory.

Uses the retained measured input basis to recover a last derivative probe that
has already been replaced by the next private proposal. No GPU state is changed.
"""
import argparse, hashlib, json, re, struct
from pathlib import Path
import numpy as np
import check_biprism_nuclear_budget as NB
import check_biprism_spinor_owner as SO
import check_biprism_v10_dec_readback as D
import biprism_exchange_layout as EX

K=Path(__file__).resolve().parents[2]/'crates/prism-gpu/src/kernels'

def density(p):
    z=p.reshape((-1,2,2));s=z[:,:,0]+1j*z[:,:,1]
    cross=s[:,0]*s[:,1].conj()
    return np.c_[abs(s[:,0])**2,abs(s[:,1])**2,cross.real,cross.imag],s

def provider(x,edges,elem,charge,lut):
    u,v=edges.T;e=x[v]-x[u];r=np.sqrt(np.sum(e*e,axis=1)+float(np.float32(1e-10)))
    pu,pv=lut[elem[u]],lut[elem[v]];zeta=2*pu[:,0]*pv[:,0]/(pu[:,0]+pv[:,0]);zr=zeta*r
    c=float(np.float32(.33333333));ex=np.exp(-zr);S=(1+zr+c*zr*zr)*ex
    ds=zeta*zr*((2*c-1)-c*zr)*ex;gamma=1/np.sqrt((1/pu[:,3]+1/pv[:,3])**2+r*r)
    dg=-r*gamma**3;dq=charge[v]-charge[u]
    E0=pu[:,1]-.5*dq*gamma;E1=pv[:,2]+.5*dq*gamma;k=-.875*(pu[:,1]+pv[:,2]);V=k*S
    invD=1/np.maximum(1-S*S,float(np.float32(.001)))
    n=e/r[:,None]
    return S,E0,E1,V,gamma,invD,ds,k,.5*dq*dg,n

def edge_energy_force(p,rho):
    S,E0,E1,V,g,invD,ds,k,de,n=p
    N=rho[:,0]*E0+rho[:,1]*E1+2*rho[:,2]*V
    # Independent scalar radial derivative, then oriented Cartesian deposition.
    dN=(-rho[:,0]+rho[:,1])*de+2*rho[:,2]*k*ds
    radial=dN*invD+np.where(1-S*S>float(np.float32(.001)),2*N*S*invD**2*ds,0.)
    return N*invD,radial[:,None]*n

def metric_force(x,reference,velocity,mass,edges):
    u,v=edges.T;e=x[v]-x[u];E=reference[v]-reference[u]
    l2=np.sum(e*e,axis=1)+float(np.float32(1e-12));l20=np.sum(E*E,axis=1)+float(np.float32(1e-12))
    r=np.sqrt(l2);n=e/r[:,None];chi=float(np.float32(.12));vu=velocity[u];vv=velocity[v]
    xu=np.sum(vu*n,axis=1);xv=np.sum(vv*n,axis=1)
    perpendicular=mass[u,None]*xu[:,None]*(vu-xu[:,None]*n)+mass[v,None]*xv[:,None]*(vv-xv[:,None]*n)
    radial=-chi*(mass[u]*xu*xu+mass[v]*xv*xv)/np.maximum(l20,float(np.float32(1e-8)))
    angular=-chi*((l2-l20)/np.maximum(l20,float(np.float32(1e-8))))/np.maximum(r,float(np.float32(1e-6)))
    F=radial[:,None]*e+angular[:,None]*perpendicular
    g=np.tile(np.eye(3),(192,1,1));terms=chi*((l2-l20)/l20)/l2
    outer=terms[:,None,None]*e[:,:,None]*e[:,None,:]
    for t in range(len(edges)):
        if u[t]<192:g[u[t]]+=outer[t]
        if v[t]<192:g[v[t]]+=outer[t]
    return F,g

def accumulate(edges,F):
    total=np.zeros((256,3));np.add.at(total,edges[:,0],F);np.add.at(total,edges[:,1],-F);return total

def inspect(execution,epoch):
    receipt=json.loads(Path(execution).read_text());assert receipt['exit_code']==4 and receipt['native_execution']
    prefix=receipt['command'][4]+f'.epoch{epoch}'
    def read(suffix):
        p=prefix+'.'+suffix;raw=Path(p).read_bytes();assert receipt['artifacts'][p]==dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());return raw
    raw,tiles,spinors,bank=map(read,['numerical.bin','tiles.bin','spinor.owners.bps1','exchange.bin'])
    stride=len(raw)//8;assert stride in (510720,1680768,1697920,1784448,1837696,1997440)
    params=K/'nhs_biprism_xtb_lut.cuh';source=params.read_text().split('c_gfn_xtb_lut[8] = {')[1].split('};')[0]
    lut=np.array([[np.float32(x) for x in re.findall(r'([-+\d.]+)f',s)] for s in source.splitlines() if '{' in s],dtype=float);assert lut.shape==(8,4)
    assert hashlib.sha256((K/'nhs_biprism_electronic_discrete.cuh').read_bytes()).hexdigest()==D.EXPECTED_ELECTRONIC
    refs=np.frombuffer(read('au_reference_high.bin'),'<f4').astype(float)+np.frombuffer(read('au_reference_low.bin'),'<f4')
    refs=refs.reshape(8,-1,3)
    massp=receipt['command'][3];massraw=Path(massp).read_bytes();assert hashlib.sha256(massraw).hexdigest()==receipt['input_hashes'][massp]
    masses=(np.float32(1.)/np.maximum(np.frombuffer(massraw,'<f4'),np.float32(1e-6))).astype(float)
    bank_stride,_,weight_format,_=EX.layout(bank)
    states=[SO.read(spinors,l) for l in range(8)];omegas=np.array([np.frombuffer(bank,'<f4',192,l*bank_stride+3072).astype(float) for l in range(8)])
    pops=np.array([.5*((density(np.array(s['start']))[0][:,0]-density(np.array(s['start']))[0][:,1])+(density(np.array(s['candidate']))[0][:,0]-density(np.array(s['candidate']))[0][:,1])) for s in states])
    kappa=float(np.float32(.1))
    for leaf in range(8):
        b=leaf*stride;head,trace=NB.read_trace(raw,leaf);stage,phase=divmod(int(trace[-1][7]),1000);mode,j=divmod(phase,100)
        assert mode in (1,2),'this reconstruction requires an evaluated derivative probe or actual Newton trial'
        q=b+106752
        arr=lambda off,n:np.frombuffer(raw,'<f8',n,b+off).copy()
        base=arr(88320,576).reshape(3,192).T
        off=((168424 if stride==510720 else 673768)+j*576*8) if mode==1 else (156928 if stride==510720 else 600064)
        direction=np.frombuffer(raw,'<f8',576,q+off).reshape(3,192).T
        v=base+trace[-1][6]*direction;h=NB.live_h(head);assert h is not None
        start=np.frombuffer(raw,'<f4',192*6,b+72960).astype(float).reshape(192,3,2).sum(axis=2)
        endpoint=np.frombuffer(raw,'<f4',256*6,b+77568).astype(float).reshape(256,3,2).sum(axis=2)
        x=endpoint.copy();x[:192]=start+.5*h*v
        tile=tiles[leaf*D.TILE_BYTES:(leaf+1)*D.TILE_BYTES];ids,edge,_,_=D.parse_tile(tile);edge=np.array(edge);u,w=edge.T
        packed=np.frombuffer(tile,'<u4',256);elem=(packed>>24)&7;mass=masses[ids];reference=refs[leaf,ids]
        s0=np.array(states[leaf]['start']);s1=np.array(states[leaf]['candidate']);rho0,z0=density(s0);rho1,z1=density(s1);rho=np.tile([1.,0.,0.,0.],(256,1));rho[:192]=.5*(rho0+rho1)
        amber=np.frombuffer(tile,'<f4',1024,1024).reshape(256,4)[:,3].astype(float);charge=amber.copy();charge[:192]+=float(np.float32(.15))*rho[:192,1]
        e=x[w]-x[u];E=reference[w]-reference[u];l2=np.sum(e*e,axis=1)+float(np.float32(1e-12));l20=np.sum(E*E,axis=1)+float(np.float32(1e-12))
        basew=np.frombuffer(tile,'<f4',len(edge),96256).astype(float)
        weight=np.maximum(basew,float(np.float32(1e-4)))*(np.sqrt(l20)/np.sqrt(l2))
        if weight_format=='f':weight=weight.astype(np.float32).astype(float)
        phi=np.frombuffer(read('potential.prepared.f64le'),'<f8').reshape(8,256)[leaf].copy()
        A=np.zeros((192,192));rhs=charge[:192].copy()
        for (a,c),ww in zip(edge,weight):
            if a<192:A[a,a]+=ww
            if c<192:A[c,c]+=ww
            if a<192 and c<192:A[a,c]-=ww;A[c,a]-=ww
            elif a<192:rhs[a]+=ww*phi[c]
            elif c<192:rhs[c]+=ww*phi[a]
        phi[:192]=np.linalg.solve(A,rhs)
        velocity=np.zeros((256,3));velocity[:192]=v
        Fm,g=metric_force(x,reference,velocity,mass,edge)
        Fd=(.5*(phi[u]-phi[w])**2*weight/l2)[:,None]*e
        p=provider(x,edge,elem,charge,lut);Ux,Fx=edge_energy_force(p,rho[u]);S,E0,E1,V,gamma,invD,ds,k,de,n=p
        H=np.zeros((192,2,2),complex);H[:,0,0]=omegas[leaf]*lut[elem[:192],1];H[:,1,1]=omegas[leaf]*lut[elem[:192],2]+float(np.float32(.15))*phi[:192]
        fol=-.5*kappa*((omegas[(leaf-1)%8]+omegas[leaf])*pops[(leaf-1)%8]+(omegas[leaf]+omegas[(leaf+1)%8])*pops[(leaf+1)%8])
        H[:,0,0]+=fol;H[:,1,1]-=fol
        chain=.5*gamma*(rho[u,1]-rho[u,0])*invD
        for t,(a,c) in enumerate(edge):
            if a<192:H[a,0,0]+=E0[t]*invD[t];H[a,1,1]+=E1[t]*invD[t]-float(np.float32(.15))*chain[t];H[a,0,1]+=V[t]*invD[t];H[a,1,0]+=V[t]*invD[t]
            if c<192:H[c,1,1]+=float(np.float32(.15))*chain[t]
        pm=.5*(z0+z1);hp=np.einsum('nij,nj->ni',H,pm);yp=np.c_[-1j*pm[:,1],1j*pm[:,0]]
        zz=np.zeros(256);zz[:192]=np.sum(hp.conj()*yp,axis=1).imag/np.maximum(omegas[leaf],float(np.float32(1e-4)))
        kap=(.5*weight*invD/np.sqrt((E1-E0)**2+4*V*V+float(np.float32(5e-5))**2))[:,None]*(-k*ds)[:,None]*n
        Fb=(2*(zz[u]-zz[w]))[:,None]*kap
        forces=[accumulate(edge,f)[:192] for f in (Fm,Fd,Fx,Fb)]
        native=arr(66048,576).reshape(3,192).T
        reconstructed=sum(forces)-trace[-1][2]*h*v
        error=np.max(abs(native-reconstructed))
        midpoint=arr(61440,576).reshape(3,192).T
        print('FORCE_RECONSTRUCTION',leaf,'h_fraction',h/struct.unpack('<f',struct.pack('<I',head[10]))[0], 'native_max',np.max(abs(native)),'max_abs_difference',error,'CPU_not_native_qualification',True)
        for vertex in (0,1,2,3):
            print('VERTEX',leaf,vertex,'metric',forces[0][vertex].tolist(),'DEC',forces[1][vertex].tolist(),'xTB',forces[2][vertex].tolist(),'reciprocal',forces[3][vertex].tolist(),'projection',(-trace[-1][2]*h*v[vertex]).tolist(),'native',native[vertex].tolist())
        ranked=np.argsort(np.linalg.norm(Fx,axis=1))[-3:][::-1]
        for t in ranked:print('DOMINANT_EDGE',leaf,int(t),'vertices',edge[t].tolist(),'r',np.linalg.norm(e[t]),'xTB_energy',Ux[t],'xTB_force_magnitude',np.linalg.norm(Fx[t]),'overlap_denominator',1-S[t]**2)
        print('MIDPOINT_SPD',leaf,float(np.linalg.eigvalsh(g).min()),'current_map_residual',float(np.max(abs(.5*h*(midpoint-v)))))
        # Isolate the declared FP32 magnetic-field boundary. This counterfactual
        # keeps the exact native input velocity and metric fixed; it is NOT a
        # new self-consistent electronic or nuclear solution.
        tau64,tau32=magnetic_rotation(x,velocity,edge,elem,charge,lut,mass,h)
        vm=arr(47616,576).reshape(3,192).T;vp=arr(52224,576).reshape(3,192).T
        def rotated(tau):
            C=np.zeros((192,3,3));tx,ty,tz=tau[:192].T
            C[:,0,1]=tz;C[:,0,2]=-ty;C[:,1,0]=-tz;C[:,1,2]=tx;C[:,2,0]=ty;C[:,2,1]=-tx
            return np.linalg.solve(g-C,np.einsum('nij,nj->ni',g+C,vm)[...,None])[...,0]
        r64,r32=rotated(tau64),rotated(tau32)
        for vertex in (57,61):
            print('MAGNETIC_PRECISION',leaf,vertex,'tau64',tau64[vertex].tolist(),
                'tau32',tau32[vertex].tolist(),'native_rotation_to_cpu32_midpoint',(.25*h*(vp[vertex]-r32[vertex])).tolist(),
                'cpu32_to_cpu64_midpoint',(.25*h*(r32[vertex]-r64[vertex])).tolist())
        mapped=x.copy();mapped[:192]=start+.5*h*midpoint
        mapped_l2=np.sum((mapped[w]-mapped[u])**2,axis=1)+float(np.float32(1e-12))
        mapped_w=np.maximum(basew,float(np.float32(1e-4)))*(np.sqrt(l20)/np.sqrt(mapped_l2))
        if weight_format=='f':mapped_w=mapped_w.astype(np.float32).astype(float)
        changes=np.flatnonzero(mapped_w!=weight)
        print('WEIGHT_ROUNDING_JUMPS',leaf,changes.tolist())
        for k in changes:
            print('WEIGHT_ROUNDING_EDGE',leaf,int(k),edge[k].tolist(),'old',weight[k],'mapped',mapped_w[k])
        # Resolve phi under just the changed represented weights, holding all
        # x, rho, velocities and other physics fixed. The response through DEC,
        # H_energy and reciprocal connection is included; C1 is not re-solved.
        A2=A.copy();rhs2=rhs.copy()
        for k in changes:
            a,c=edge[k];dw=mapped_w[k]-weight[k]
            if a<192:A2[a,a]+=dw
            if c<192:A2[c,c]+=dw
            if a<192 and c<192:A2[a,c]-=dw;A2[c,a]-=dw
            elif a<192:rhs2[a]+=dw*phi[c]
            elif c<192:rhs2[c]+=dw*phi[a]
        phi2=phi.copy();phi2[:192]=np.linalg.solve(A2,rhs2)
        fd2=(.5*(phi2[u]-phi2[w])**2*mapped_w/l2)[:,None]*e
        H2=H.copy();H2[:,1,1]+=float(np.float32(.15))*(phi2[:192]-phi[:192])
        hp2=np.einsum('nij,nj->ni',H2,pm);zz2=zz.copy()
        zz2[:192]=np.sum(hp2.conj()*yp,axis=1).imag/np.maximum(omegas[leaf],float(np.float32(1e-4)))
        fb2=(2*(zz2[u]-zz2[w]))[:,None]*kap*(mapped_w/weight)[:,None]
        dforce=accumulate(edge,fd2-Fd+fb2-Fb)[:192]
        dvm=np.linalg.solve(g,(.5*h*dforce/mass[:192,None])[...,None])[...,0]
        C=np.zeros((192,3,3));tx,ty,tz=tau32[:192].T
        C[:,0,1]=tz;C[:,0,2]=-ty;C[:,1,0]=-tz;C[:,1,2]=tx;C[:,2,0]=ty;C[:,2,1]=-tx
        dvp=np.linalg.solve(g-C,np.einsum('nij,nj->ni',g+C,dvm)[...,None])[...,0]
        weight_jump_response=.25*h*(dvm+dvp)
        print('WEIGHT_ONLY_FIXED_RHO_MIDPOINT_RESPONSE',leaf,'max',np.max(abs(weight_jump_response)),
            'vertex57',weight_jump_response[57].tolist(),'native_residual_vertex57',(.5*h*(midpoint-v)[57]).tolist())
    print('SCOPE fixed native candidate C2 reconstruction; electronic elimination not re-solved; no physical timestep accepted')

def magnetic_rotation(x,velocity,edge,elem,charge,lut,mass,h,*,field_nodes_f64=False):
    """Independent vectorized quadrature, same stored nodes/weights as CUDA.

    FP32 sine here is NumPy's correctly rounded library path, not CUDA __sinf;
    CPU32 therefore bounds a rounding sensitivity, not native bitwise parity.
    """
    u,v=edge.T;e=x[v]-x[u];vr=velocity[v]-velocity[u]
    field_dtype=np.float64 if field_nodes_f64 else np.float32
    b64=np.zeros_like(e);b32=np.zeros_like(e,dtype=field_dtype)
    for node,w in ((-.3872983346,.27777778),(0.,.44444444),(.3872983346,.27777778)):
        eq=e+float(np.float32(node))*h*vr
        # Provider only depends on each oriented displacement; flatten edges.
        ep=np.zeros((2*len(edge),3));ep[1::2]=eq;edges=np.arange(2*len(edge)).reshape(-1,2)
        p=provider(ep,edges,elem[edge].reshape(-1),charge[edge].reshape(-1),lut)
        S,E0,E1,V,gamma,invD,ds,k,de,n=p
        grad=-k[:,None]*ds[:,None]*n
        phase=h*np.sum(grad*vr,axis=1)/np.sqrt((E1-E0)**2+4*V*V+float(np.float32(2.5e-9)))
        r2=np.sum(eq*eq,axis=1)+float(np.float32(1e-12))
        amp=float(np.float32(w))*(np.sin(phase)+phase)/(np.sqrt(r2)*r2)
        b64+=amp[:,None]*eq
        ph32=phase.astype(np.float32)
        amp32=float(np.float32(w))*(np.sin(ph32)+ph32).astype(float)/(np.sqrt(r2)*r2)
        b32+=(amp32[:,None]*eq).astype(field_dtype)
    tau64=np.zeros((256,3));tau32=np.zeros((256,3),field_dtype)
    for k,(a,b) in enumerate(edge):
        for vertex,sgn in ((a,1.),(b,-1.)):
            if vertex<192:
                tau64[vertex]+=(sgn*.5*h/mass[vertex])*b64[k]
                coefficient=(sgn*.5*h/mass[vertex] if field_nodes_f64 else
                             np.float32(sgn*.5)*np.float32(h)/np.float32(mass[vertex]))
                tau32[vertex]+=coefficient*b32[k]
    return tau64,tau32.astype(float)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execution',type=Path,required=True);p.add_argument('--epoch',type=int,required=True);a=p.parse_args();inspect(a.execution,a.epoch)
