#!/usr/bin/env python3
"""Run one immutable, resource-qualified actual-source coupled diagnostic.

No installed release, production admission state, or prior fault is modified.
The retained receipt distinguishes this control from the complete AMBER schedule.
"""
import os
from pathlib import Path
import datetime,fcntl,hashlib,json,os,re,subprocess,sys
import biprism_exchange_layout as EX

def _require_root(name):
    """Absolute root from the environment; there is no compiled-in host default."""
    value = os.environ.get(name, '')
    if not os.path.isabs(value):
        raise SystemExit(f'{name} must name an absolute path (no compiled-in host default)')
    return Path(value)


def diagnostic_cli(argv):
    attempt=argv[0] if argv else '001'
    assert len(attempt)==3 and attempt.isdigit()
    assert len(argv)<=3, 'usage: run_biprism_v10_coupled_qc.py ATTEMPT [--asymmetric-kappa=0|--asymmetric-kappa=0.1|--connected-work|--fixed-domain-16] [--repeat]'
    mode=argv[1] if len(argv)>=2 else None
    assert mode in (None,'--asymmetric-kappa=0','--asymmetric-kappa=0.1','--connected-work','--fixed-domain-16'), 'Unknown diagnostic mode'
    repeat=len(argv)==3
    assert not repeat or (argv[2]=='--repeat' and mode=='--asymmetric-kappa=0'), 'Repeat control must use kappa=0'
    epochs=16 if mode=='--fixed-domain-16' else (1 if mode and mode!='--connected-work' else 3)
    tag='failure_1_2_3_coupled_native_'+attempt
    if mode=='--connected-work': tag+='_connected_work'
    elif mode=='--fixed-domain-16': tag+='_fixed_domain_16'
    elif mode: tag+='_asymmetric_kappa_'+('zero' if mode.endswith('=0') else 'point1')
    if repeat: tag+='_repeat'
    return attempt,mode,repeat,epochs,tag


def fixed_domain_identity(raw):
    """Combinatorial identity only; Phase2 may change volumes/base weights."""
    import struct
    assert len(raw)==225408, 'fixed-domain tile extent'
    nt,ne=struct.unpack_from('<2I',raw,225280)
    assert nt==960 and 0<ne<=1536, 'fixed-domain active population'
    spans=((0,1024),(5120,4*nt),(9216,12*nt),(21504,8*nt),(29696,2*ne),(225280,8))
    return tuple(raw[offset:offset+size] for offset,size in spans)


def fixed_domain_norm_energy(entry,endpoint,convergence,exchange,epoch,entry_spinors,end_spinors):
    """Exact represented norms and independently decoded existing energy gates.

    Energy values are actual native reports/ledger terms; this does not claim
    an independent recomputation of the unretained four-substep Hamiltonian.
    """
    import math,struct
    import check_biprism_spinor_owner as S
    from fractions import Fraction as Q
    checked=S.endpoint(end_spinors,endpoint,4*epoch,epoch)
    assert len(entry)==len(endpoint)==8*225408 and len(convergence)==8*128 and len(exchange) in (8*11648,8*17792)
    bank_stride,footer_offset,_,bank_magic=EX.layout(exchange)
    f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
    total_energy=total_chain=total_transfer=Q(0)
    globals0=None;sequence=None;worst_norm=Q(0);worst_change=Q(0);reported_energy=0.
    for leaf in range(8):
        old=S.read(entry_spinors,leaf)
        assert old['committed']==4*(epoch-1) and old['step']==epoch-1
        before=tuple(x for p in old['accepted'] for x in p)
        after=tuple(x for p in checked[leaf]['accepted'] for x in p)
        assert all(map(math.isfinite,before+after)), 'nonfinite represented spinor'
        for vertex in range(192):
            a=sum((Q(x)**2 for x in before[4*vertex:4*vertex+4]),Q(0))
            b=sum((Q(x)**2 for x in after[4*vertex:4*vertex+4]),Q(0))
            worst_norm=max(worst_norm,abs(b-1));worst_change=max(worst_change,abs(b-a))
        rec=convergence[leaf*128:(leaf+1)*128]
        magic,owner,local,cluster,sticky,count=struct.unpack_from('<6I',rec)
        assert (magic,owner,local,cluster,sticky,count)==(0x42505631,leaf,0,0,0,4*epoch), 'accepted all-leaf counter/fault identity'
        picard,dec,energy,_,wilson_leaf,wilson_time,norm=struct.unpack_from('<7f',rec,32)
        assert all(map(math.isfinite,(picard,dec,energy,norm,wilson_leaf,wilson_time)))
        assert 0<=picard<=f32(1e-7) and 0<=dec<=f32(1e-6) and 0<=energy<=f32(1e-6) and 0<=norm<=f32(1e-7)
        assert -2<=wilson_leaf<=2 and -2<=wilson_time<=2
        reported_energy=max(reported_energy,energy)
        foot=exchange[leaf*bank_stride+footer_offset:(leaf+1)*bank_stride]
        e=struct.unpack_from('<8d',foot);r=struct.unpack_from('<4f',foot,64)
        m,l,c,sub,seq,fault,vote,phase=struct.unpack_from('<8I',foot,80)
        assert (m,l,c,sub,fault,vote,phase)==(bank_magic,leaf,4*(epoch-1),3,0,1,4)
        assert seq>0 and not any(foot[112:]), 'footer sequence/reserved'
        if sequence is None: sequence=seq
        assert seq==sequence, 'all-leaf footer sequence'
        assert all(map(math.isfinite,e+r)) and all(x>=0 for x in r)
        assert r[0]<=f32(1e-7) and r[2]<=f32(1e-7) and r[3]<=f32(1e-7)
        if globals0 is None: globals0=foot[48:64]
        assert foot[48:64]==globals0, 'all-leaf reported global energy identity'
        assert abs(e[6])<=1e-6 and abs(e[7])<=1e-6
        assert abs(Q(e[2])-Q(e[4])-Q(e[3]))<=Q(1,10**6)
        assert abs(Q(e[5])-Q(e[0])-Q(e[3]))<=Q(1,10**6)
        total_energy+=Q(e[0])+Q(e[1]);total_chain+=Q(e[1])-Q(e[3]);total_transfer+=Q(e[4])
    assert worst_norm<=Q(1,10**7) and worst_change<=Q(1,10**7), 'exact represented norm gate'
    assert all(abs(x)<=Q(1,10**6) for x in (total_energy,total_chain,total_transfer)), 'reconstructed global energy/chain/transfer gate'
    ge,gc=struct.unpack('<2d',globals0)
    assert abs(total_energy-Q(ge))<=Q(1,10**6) and abs(total_chain-Q(gc))<=Q(1,10**6)
    return {'max_exact_norm_defect':float(worst_norm),'max_exact_norm_change':float(worst_change),
            'max_native_reported_body_energy_drift':reported_energy,
            'last_substep_energy_delta':float(total_energy),'last_substep_chain_defect':float(total_chain)}


def fixed_domain_phase2_leaf(v,next_v,e,ledger,old,new,planes,accepted,prepared,epoch,leaf,lut):
    """Recompute the actual fixed-state BPW3 work; no solve or new work credit."""
    import numpy as np
    from fractions import Fraction as Q
    import check_biprism_phase2_work_native as P
    import check_biprism_v10_dec_readback as D
    ne=len(e);packed=P.arr(old,0,'<u4',256);ids=packed&0xffffff
    assert fixed_domain_identity(old)==fixed_domain_identity(new)
    assert old[:94208]==new[:94208], 'Phase2 changed fixed physical/operator-reference inputs'
    assert (int(ledger['magic']),int(ledger['leaf']),int(ledger['state']),int(ledger['fault']))==(0x42505735,leaf,3,0)
    assert (int(ledger['step']),int(ledger['entry']),int(ledger['body']),int(ledger['count']))==(epoch,4*(epoch-1),epoch-1,epoch)
    assert (int(ledger['generation']),int(ledger['prep']),int(ledger['ne']),int(ledger['nt']))==(23,1,ne,960)
    assert not ledger['mismatch'] and not ledger['reserved'] and np.isfinite(ledger['refinement'])
    assert float(ledger['kappa'])==float(np.float32(.1)), 'fixed diagnostic kappa identity'
    assert np.array_equal(v['packed'],packed) and np.all(v['leaf']==leaf) and np.all(v['state']==3)
    assert np.array_equal(v['local'],np.arange(256)) and not np.any(v['reserved'])
    assert np.array_equal(v['packed'][:192]&0xffffff,next_v['packed'][:192]&0xffffff)
    refq=P.arr(old,1024,'<f4',1024).reshape(256,4);psi=P.arr(old,90112,'<f4',1024).reshape(256,4)
    assert P.same_f32(v['psi_precise'],psi) and P.same_f32(v['psi'],psi) and P.same_f32(v['qamber'],refq[:,3])
    for field,plane in (('x','coords'),('xlo','low'),('ref','ref'),('reflo','reflo')):
        assert P.same_f32(v[field],planes[plane][leaf,ids]), 'Phase2 full-coordinate row identity: '+field
    assert P.same_f32(v['ref'],refq[:,:3])
    assert P.same_f32(v['w0'],P.arr(old,95232,'<f4',256)) and P.same_f32(v['w1'],P.arr(new,95232,'<f4',256))
    assert P.same_f32(v['b0'],np.maximum(v['w0'],np.float32(1e-4))) and P.same_f32(v['b1'],np.maximum(v['w1'],np.float32(1e-4)))
    assert P.same_f64(v['entry'],accepted) and P.same_f64(v['p1'],prepared), 'Phase2 full-potential bank binding'
    assert P.same_f64(v['p0'][192:],accepted[192:]) and P.same_f64(v['pm'][192:],accepted[192:])
    assert P.same_f64(v['p1'][192:],(v['qamber'][192:]*np.float32(.07957747)).astype('<f8'))
    element=(packed>>24)&7
    assert P.same_f32(v['epss'],lut[element,1]) and P.same_f32(v['epsp'],lut[element,2])
    _,pairs,_,_=D.parse_tile(new);pairs=np.asarray(pairs,dtype=int)
    assert np.array_equal(e['leaf'],np.full(ne,leaf)) and np.array_equal(e['edge'],np.arange(ne))
    assert np.array_equal(np.c_[e['u'],e['v']],pairs)
    assert P.same_f32(e['base0'],P.arr(old,96256,'<f4',ne)) and P.same_f32(e['base1'],P.arr(new,96256,'<f4',ne))
    for suffix in ('0','1'):
        assert P.same_f64(e['live'+suffix],P.live_weight(v['x'],v['xlo'],v['ref'],v['reflo'],pairs,e['base'+suffix])), 'Phase2 actual live-weight identity'
    rho0=np.array([P.fma(float(a[0]),float(a[0]),float(a[1])*float(a[1])) for a in v['psi_precise']]);rho0[192:]=1.
    rho1=np.array([P.fma(float(a[2]),float(a[2]),float(a[3])*float(a[3])) for a in v['psi_precise']]);rho1[192:]=0.
    q=np.array([D.represented_charge(v['qamber'][u],v['psi_precise'][u][2:4]) if u<192 else float(v['qamber'][u]) for u in range(256)])
    assert P.same_f64(v['q'],q) and P.same_f64(v['rho0'],rho0) and P.same_f64(v['rho1'],rho1)
    assert P.same_f64(v['population'],rho0-rho1)
    potentials=[list(map(float,v[name])) for name in ('p0','pm','p1')]
    weights=[list(map(float,e[name])) for name in ('live0','live1')]
    pairs=[tuple(map(int,p)) for p in pairs]
    charge=list(map(float,q[:192]))
    ops=[D.exact_operator(charge,p,pairs,w) for p,w in zip(potentials,(weights[0],weights[1],weights[1]))]
    for op in ops:
        assert op['squared']<=Q(1,10**12) and op['squared']<=Q(float(np.float32(1e-12))), 'Phase2 exact true-residual gate'
    p0,pm,p1=[list(map(Q,p)) for p in potentials];w0,w1=[list(map(Q,w)) for w in weights]
    t=[Q(0)]*16;t[:3]=[op['functional'] for op in ops]
    t[4]=-sum(((b-a)*(p0[u]-p0[vv])*(pm[u]-pm[vv]) for (u,vv),a,b in zip(pairs,w0,w1)),Q(0))/2
    t[5]=sum(((pm[u]-p0[u])*(ops[0]['residual'][u]+ops[1]['residual'][u]) for u in range(192)),Q(0))/2
    t[7]=-sum(((p1[u]-pm[u])*(ops[1]['rows'][u]+ops[2]['rows'][u]) for u in range(192,256)),Q(0))/2
    t[8]=sum(((p1[u]-pm[u])*(ops[1]['residual'][u]+ops[2]['residual'][u]) for u in range(192)),Q(0))/2
    dw=[Q(float(b))-Q(float(a)) for a,b in zip(v['w0'],v['w1'])]
    dn=[Q(float(b))-Q(float(a)) for a,b in zip(next_v['w0'],next_v['w1'])]
    t[9]=sum((dw[u]*(Q(float(v['epss'][u]))*Q(float(rho0[u]))+Q(float(v['epsp'][u]))*Q(float(rho1[u]))) for u in range(192)),Q(0))
    k=Q(float(ledger['kappa']));t[10]=k*sum(dw[:192],Q(0))
    t[11]=-k*sum(((dw[u]+dn[u])*Q(float(v['population'][u]))*Q(float(next_v['population'][u])) for u in range(192)),Q(0))/2
    t[3]=t[1]-t[0];t[6]=t[2]-t[1];t[12]=t[2]-t[0]+sum(t[9:12],Q(0));t[13]=t[3]-t[4]-t[5];t[14]=t[6]-t[7]-t[8]
    t[15]=sum((abs(Q(float(b))-Q(float(a))) for a,b in zip(v['b0'][:192],v['b1'][:192])),Q(0))
    assert t[13]==t[14]==0, 'exact finite-solve operator/Dirichlet boundary identity'
    assert P.close(np.array(list(map(float,t))),ledger['totals']), 'BPW3 primitive work reconstruction'
    assert abs(float(ledger['totals'][13]))<=1e-6 and abs(float(ledger['totals'][14]))<=1e-6
    entry=D.exact_operator(charge,list(map(float,v['entry'])),pairs,weights[0])
    refinement=t[0]-entry['functional'];assert P.close(float(refinement),float(ledger['refinement']))
    # Refinement is retained separately; it is never added to actual t[12].
    return {'actual_operator_work':float(t[4]),'operator_residual_correction':float(t[5]),
            'actual_Dirichlet_boundary_work':float(t[7]),'boundary_residual_correction':float(t[8]),
            'actual_total_parameter_work':float(t[12]),'numerical_refinement_diagnostic':float(refinement),
            'exact_operator_identity':str(t[13]),'exact_boundary_identity':str(t[14])}


def check_fixed_domain_readback(dispatch,build,checks,reader_pins):
    """All16 actual epochs, exact primitives, bound files; no additional launch."""
    import math,struct
    import numpy as np
    import check_biprism_boundary_identity as B
    import check_biprism_phase2_work_native as P
    import check_biprism_v10_dec_readback as D
    assert dispatch['diagnostic_mode']=='--fixed-domain-16' and dispatch['expected_epochs']==16
    assert dispatch['native_exit_code']==0 and len(checks)==16 and all(c['exit_code']==0 for c in checks)
    assert all(sha(p)==h for p,h in reader_pins.items()), 'fixed-domain reader source changed'
    bound={}
    def raw(path):
        path=str(path);data=Path(path).read_bytes();digest=hashlib.sha256(data).hexdigest()
        assert dispatch['artifacts'].get(path)=={'sha256':digest,'bytes':len(data)}, 'unbound fixed-domain output: '+path
        bound[path]=digest;return data
    prefix=dispatch['command'][4]
    seed=raw(prefix+'.seed_tiles.bin');assert len(seed)==8*D.TILE_BYTES
    cold_phi=raw(prefix+'.cold.potential.accepted.f64le')
    previous_phi=cold_phi
    previous_spinors=raw(prefix+'.cold.spinor.owners.bps1')
    amber=D.read_charges(Path(dispatch['command'][1]));atoms=len(amber)
    assert sha(dispatch['command'][1])==dispatch['input_hashes'][dispatch['command'][1]]
    # c_gfn_xtb_lut is an actual dependency of the compiled Biprism module.
    provider=D.KERNELS/'nhs_biprism_xtb_lut.cuh'
    assert sha(provider)==build['input_hashes'][str(provider)]==dispatch['input_hashes'][str(provider)]
    lut=P.read_onsite_lut(provider)
    lines=['FIXED_DOMAIN_DIAGNOSTIC_EPOCHS=16 SUBSTEPS_PER_LEAF=64 LEAVES=8 TETRAHEDRA=960',
           'SCOPE=fixed combinatorial domain; geometry and actual Phase2 operators may evolve',
           'DIRICHLET=64 prescribed algebraic vertices, not a Neumann or full geometric-boundary claim',
           'ENERGY=existing native reports and ledger gates, not independent full-Hamiltonian reconstruction',
           'VIRTUAL_BOUNDARY_PROBES=diagnostic only; actual BPW3 work and refinement remain separate',
           'FULL_40000_STEP_PRODUCTION_ACCEPTANCE=false']
    for epoch,check in enumerate(checks,1):
        assert check['epoch']==epoch and sha(check['result']['path'])==check['result']['sha256']
        result=json.loads(Path(check['result']['path']).read_text())
        assert result['status']=='PASS' and result['epoch']==epoch and result['entry_substeps']==4*(epoch-1)
        for item in result['source_bindings']:
            assert sha(item['path'])==item['sha256']
            if Path(item['path']).resolve()!=Path(D.__file__).resolve():
                assert build['input_hashes'][item['path']]==dispatch['input_hashes'][item['path']]==item['sha256']
        for item in result['input_bindings']:
            if item['path'] in dispatch['artifacts']:
                assert dispatch['artifacts'][item['path']]=={'sha256':item['sha256'],'bytes':item['bytes']}
            else: assert dispatch['input_hashes'][item['path']]==item['sha256']
            assert sha(item['path'])==item['sha256']
        pre=prefix+'.epoch'+str(epoch)
        entry=raw(pre+'.phase2_tiles.bin');end=raw(pre+'.tiles.bin');old=raw(pre+'.pre_phase2_tiles.bin')
        exchange=raw(pre+'.exchange.bin');conv=raw(pre+'.conv.bin')
        import check_biprism_spinor_owner as S
        entry_spinors=raw(pre+'.entry.spinor.owners.bps1');end_spinors=raw(pre+'.spinor.owners.bps1')
        pre_spinors=raw(pre+'.pre_phase2.spinor.owners.bps1')
        for l in range(8):
            a=l*S.SIZE
            assert pre_spinors[a:a+6272]==previous_spinors[a:a+6272], 'accepted spinor continuation'
        assert entry_spinors==pre_spinors, 'Phase2 changed physical spinor owner'
        energy=fixed_domain_norm_energy(entry,end,conv,exchange,epoch,entry_spinors,end_spinors)
        previous_spinors=end_spinors
        accepted_before=raw(pre+'.pre_phase2.potential.accepted.f64le')
        assert accepted_before==previous_phi, 'accepted full-potential continuation across epochs'
        assert raw(pre+'.entry.potential.accepted.f64le')==accepted_before, 'Phase2 changed accepted bank'
        phi_raw=raw(pre+'.potential.accepted.f64le');prepared_raw=raw(pre+'.entry.potential.prepared.f64le')
        assert raw(pre+'.potential.prepared.f64le')==prepared_raw, 'main rewrote prepared bank'
        phis=np.frombuffer(phi_raw,dtype='<f8').reshape(8,256)
        accepted=np.frombuffer(accepted_before,dtype='<f8').reshape(8,256)
        prepared=np.frombuffer(prepared_raw,dtype='<f8').reshape(8,256)
        vertices=np.frombuffer(raw(pre+'.phase2_work.vertices.bin'),dtype=P.V).reshape(8,256)
        edges=np.frombuffer(raw(pre+'.phase2_work.edges.bin'),dtype=P.E).reshape(8,1536)
        ledgers=np.frombuffer(raw(pre+'.phase2_work.ledgers.bin'),dtype=P.LDT);assert len(ledgers)==8
        assert raw(pre+'.phase2_work.snapshots.bin')==old, 'actual Phase2 capture snapshot'
        planes={name:np.frombuffer(raw(pre+suffix),dtype='<f4').reshape(8,atoms,3) for name,suffix in
                (('coords','.pre_phase2_au_coords.bin'),('low','.pre_phase2.au_coords_low.bin'),
                 ('ref','.pre_phase2.au_reference_high.bin'),('reflo','.pre_phase2.au_reference_low.bin'))}
        for leaf in range(8):
            a=entry[leaf*D.TILE_BYTES:(leaf+1)*D.TILE_BYTES];b=end[leaf*D.TILE_BYTES:(leaf+1)*D.TILE_BYTES]
            old_leaf=old[leaf*D.TILE_BYTES:(leaf+1)*D.TILE_BYTES];initial=seed[leaf*D.TILE_BYTES:(leaf+1)*D.TILE_BYTES]
            assert fixed_domain_identity(a)==fixed_domain_identity(b)==fixed_domain_identity(old_leaf)==fixed_domain_identity(initial)
            ids,pairs,psi,_=D.parse_tile(b);domain=B.retained_domain(a,b,pairs)
            psi=S.read(end_spinors,leaf)['accepted']+psi[192:]
            assert P.same_f64(vertices[leaf]['psi_precise'][:192],S.read(entry_spinors,leaf)['accepted']), 'Phase2 precise spinor row/source binding'
            weights=EX.weights(exchange,leaf,len(pairs))
            charge=[D.represented_charge(amber[ids[u]],psi[u][2:4]) for u in range(192)]
            assert P.same_f64(phis[leaf,192:],prepared[leaf,192:]), 'within-main prescribed halo changed'
            graph=B.graph_identities(charge,list(map(float,phis[leaf])),pairs,weights)
            work=fixed_domain_phase2_leaf(vertices[leaf],vertices[(leaf+1)&7],edges[leaf,:len(pairs)],
                    ledgers[leaf],old_leaf,a,planes,accepted[leaf],prepared[leaf],epoch,leaf,lut)
            lines.append('EPOCH='+str(epoch)+' LEAF='+str(leaf)+' DOMAIN='+json.dumps(domain,sort_keys=True)
                         +' NORM_ENERGY='+json.dumps(energy,sort_keys=True)+' PHASE2='+json.dumps(work,sort_keys=True))
            lines.append('DIRICHLET_EXACT '+json.dumps({'epoch':epoch,'leaf':leaf,
                'basis_checks':graph['basis_checks'],'cut_edges':len(graph['cut_edges']),
                'halo_rows':[[v,*map(str,values)] for v,*values in graph['halo_rows']],
                'Green_pairings':{name:list(map(str,values)) for name,values in graph['pairings'].items()},
                'virtual_delta':str(graph['virtual_delta']),'virtual_linear':str(graph['virtual_linear_work']),
                'virtual_quadratic':str(graph['virtual_quadratic_term'])},sort_keys=True))
        previous_phi=phi_raw
    assert all(sha(p)==h for p,h in bound.items()) and all(sha(p)==h for p,h in reader_pins.items())
    lines.extend('BOUND_FILE='+p+' SHA256='+h for p,h in sorted({**bound,**reader_pins}.items()))
    lines.append('FIXED_DOMAIN_READBACK_STATUS=PASS')
    return '\n'.join(lines)+'\n'

O=_require_root('PRISM_BIPRISM_RESOLUTION_ROOT')
T=_require_root('PRISM_BIPRISM_TASK_ROOT')
R=Path(__file__).resolve().parents[2]  # engine root (this file is cuda/tests/*.py)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
native_memcheck='--memcheck' in sys.argv[1:]
assert sys.argv[1:].count('--memcheck')<=1
native_tangent_capture='--tangent-model-capture' in sys.argv[1:]
assert sys.argv[1:].count('--tangent-model-capture')<=1
attempt,mode,repeat,epoch_count,tag=diagnostic_cli([a for a in sys.argv[1:] if a not in ('--memcheck','--tangent-model-capture')])
if native_tangent_capture:
    assert mode=='--fixed-domain-16' and not repeat and not native_memcheck
    tag='failure_2_tangent_capture_'+attempt
if native_memcheck:tag+='_memcheck'
connected_work=mode=='--connected-work'
fixed_domain_16=mode=='--fixed-domain-16'
build_tag=('failure_2_tangent_capture_build_' if native_tangent_capture else 'failure_1_2_3_coupled_native_build_')+attempt
build_path=O/(build_tag+'.json')
build=json.loads(build_path.read_text());assert build['exit_code']==0 and build['sources_unchanged']
assert build.get('diagnostic_tangent_capture',False)==native_tangent_capture
assert build.get('autonomous_qualification_eligible',True)==(not native_tangent_capture)
binary=O/(('biprism_tangent_capture_' if native_tangent_capture else 'biprism_coupled_native_qc_')+attempt)
assert sha(binary)==build['binary_sha256']
compiler=(O/(build_tag+'.stderr')).read_text()
pattern=r'Function properties for nhs_biprism_variational_so3g_v10\s*\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\s*\nptxas info\s*: Used (\d+) registers([^\n]*)'
match=re.search(pattern,compiler);assert match,'No actual main resource report'
stack,stores,loads,regs=map(int,match.groups()[:4])
cumulative_match=re.search(r'(\d+) bytes cumulative stack size',match.group(5))
cumulative_stack=int(cumulative_match.group(1)) if cumulative_match else stack
assert stack==stores==loads==cumulative_stack==0 and regs<=128,'Main/callee resource gate refused'
def main_helper_resources(compiler_output):
    """Include emitted v10 device callees, not just the entry's stack report."""
    entry="Compiling entry function 'nhs_biprism_variational_so3g_v10'"
    assert compiler_output.count(entry)==1, 'Missing or ambiguous v10 entry resource block'
    block=compiler_output.split(entry,1)[1].split('Compiling entry function ',1)[0]
    headers=[name.strip() for name in re.findall(r'Function properties for ([^\n]+)',block)
             if name.strip()!='nhs_biprism_variational_so3g_v10']
    pattern=(r'Function properties for ([^\n]+)\s*\n\s*'
             r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads')
    rows=[{'symbol':name.strip(),'stack':int(frame),'spill_stores':int(write),'spill_loads':int(read)}
          for name,frame,write,read in re.findall(pattern,block)
          if name.strip()!='nhs_biprism_variational_so3g_v10']
    assert headers==[row['symbol'] for row in rows], 'Incomplete v10 device-callee resource reports'
    return rows
helper_resources=main_helper_resources(compiler)
assert helper_resources,'Missing actual v10 device-callee resource reports'
assert all(h['stack']==h['spill_stores']==h['spill_loads']==0 for h in helper_resources), \
    'A v10 device callee failed the zero-stack/zero-spill gate'
case=T/'INTERNAL_QC/v10_multiplicity_revision/native/host_controls/KRAS_INPUT.bpqc'
selected=T/'INTERNAL_QC/v10_bowyer_native/implementation/direct_qualification_001/seeded_tiles.bin.selection.tsv'
mass=T/'INTERNAL_QC/v10_engine_integration/math/kras_true_inv_masses_atomic.f32le'
expected={str(binary):build['binary_sha256'],str(case):'eff6c982d8cdbd9f77e6e4ecf37852fb33ca486712747079e3a0e42efaf5bd54',str(selected):'b32acda57f813d8b24b013826a3f8cc635917e1c02a95c6e1004a63ca40d69f9',str(mass):'37e070f483d07f186cddc1d8ac2cdf2a8339b57d0bf6884742864b54bfedc772'}
expected[str(R/'cuda/tests/check_biprism_nuclear_budget.py')]=sha(R/'cuda/tests/check_biprism_nuclear_budget.py')
expected[str(R/'cuda/tests/biprism_exchange_layout.py')]=sha(R/'cuda/tests/biprism_exchange_layout.py')
expected.update(build['input_hashes']);assert all(sha(p)==h for p,h in expected.items())
assert build['canonical_source_tree']==str(R) and build['header_overlays'] is False
selected_sources=build['selected_sources']
for name,path in selected_sources.items():
    assert Path(path)==R/'crates/prism-gpu/src/kernels'/name
    assert path in build['input_hashes']
dec_checker=R/'cuda/tests/check_biprism_v10_dec_readback.py'
dec_checker_sha256=sha(dec_checker)
# Reviewed schema extensions: dedicated solver history and FP64 nuclear transit
# follow the unchanged256-double potential plane. The FP64 Krylov source epoch
# is added explicitly; the disjoint arclength tail keeps every prior offset.
# NHEB adds a disjoint512-row tail for the measured original-h closure experiment;
# every NHEA byte offset and the256-trial bound are preserved.
# NHEE appends measured native155 remaining-work capacity (3072/512);
# every NHED offset and every physical residual/operator/chronology gate stays.
assert dec_checker_sha256=='9f9775025c06ff1cc2531a520e118357ae1b34c8ddba868f8f60937295e77687'
expected[str(R/'cuda/tests/check_biprism_spinor_owner.py')]='328dd9c110988b9e07a30871e0ceebf052bae64a1520ce61e890d8cd3313feb2'
assert all(sha(p)==h for p,h in expected.items())
fixed_domain_reader_pins={}
if fixed_domain_16:
    for filename,digest in (
        ('check_biprism_boundary_identity.py','2573300febdd1daa7f145b45e1a761234bb6330c78a590daa78da413bad1f99e'),
        ('check_biprism_phase2_work_native.py','001697f66b79a918a2db5ff60574466b826d1751d5e683e655e9c8d6d4406316'),
        ('check_biprism_v10_accepted_prefix.py','92a19786825cb8c91ec5684c2588dd32304ba04eb717341e51cbaa91050b6ae5'),
        ('check_biprism_v10_dec_readback.py',dec_checker_sha256)):
        path=R/'cuda/tests'/filename
        assert sha(path)==digest, 'fixed-domain reader changed: '+filename
        fixed_domain_reader_pins[str(path)]=digest
    expected.update(fixed_domain_reader_pins)
receipt_path=O/(tag+'_execution.json');assert not receipt_path.exists()
lock_path=f'/tmp/prism-foundation-gpu-{os.getuid()}.lock'
lock_fd=os.open(lock_path,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
fcntl.flock(lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
def capture(name,args):
    r=subprocess.run(args,capture_output=True,text=True,timeout=30)
    (O/(tag+'_'+name+'.stdout')).write_text(r.stdout);(O/(tag+'_'+name+'.stderr')).write_text(r.stderr)
    return r
def journal(name):
    r=capture(name,['journalctl','-k','-b','--grep','NVRM|Xid','-o','json','--no-pager']);assert r.returncode==0
    return [json.loads(s) for s in r.stdout.splitlines() if s.startswith('{')]
before=journal('journal_before')
health_args=['nvidia-smi','--query-gpu=name,uuid,driver_version,gpu_recovery_action,memory.used,utilization.gpu','--format=csv,noheader']
health=capture('gpu_before',health_args);assert health.returncode==0 and ', None,' in health.stdout
apps=capture('compute_before',['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'])
assert apps.returncode==0 and all('/usr/libexec/gnome-remote-desktop-daemon' in s for s in apps.stdout.splitlines() if s.strip())
command=[str(binary),str(case),str(selected),str(mass),str(O/(tag+'_state'))]
if mode:
    command.append(mode)
receipt={'started_utc':now(),'command':command,'input_hashes':expected,'build_receipt':{'path':str(build_path),'sha256':sha(build_path)},'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'authorization':'Operator requested actual native GPU gap closure; standing nonprivileged diagnostic authority.','boundary':'Exact selected coupled header epoch on actual KRAS AU input, 0.5fs, true inverse masses, zero initial velocity and bath force. Per-epoch fixed-operator control; no full AMBER schedule/trajectory acceptance.','cooperative_gpu_lock':lock_path,'privileged_lease_claim':False,'installed_state_mutation':False,'production_admission_reset':False,'historical_faults_preserved':True,'full_trajectory_accepted':False,'script_sha256':sha(__file__),'main_compiler_resources':{'registers':regs,'stack':stack,'cumulative_stack':cumulative_stack,'spill_stores':stores,'spill_loads':loads}}
receipt.update(diagnostic_mode=mode or 'default_three_epoch',expected_epochs=epoch_count,baseline_repeat=repeat)
receipt.update(diagnostic_tangent_capture=native_tangent_capture,autonomous_qualification_eligible=not native_tangent_capture)
receipt['main_device_callee_resources']=helper_resources
if connected_work:
    receipt['boundary']='Actual KRAS coupled main and actual LIF producer/delivery/kinetic-work component, native common-min selector and time publication, three 0.5fs requests. No full AMBER schedule or mixed-energy guard acceptance is asserted.'
if fixed_domain_16:
    receipt['boundary']='Sixteen actual 0.5fs diagnostic bodies/64substeps per leaf on the same960-tetrahedron combinatorial KRAS domain. Actual Phase2 geometry/operator refresh and fixed-halo Dirichlet work are retained; this is not the40000-step production experiment or a full AMBER/mixed-energy trajectory.'
    receipt['fixed_domain_reader_pins']=fixed_domain_reader_pins
launcher_command=command
if native_memcheck:
    sanitizer='/usr/local/cuda/bin/compute-sanitizer'
    launcher_command=[sanitizer,'--tool','memcheck','--error-exitcode','97']+command
    expected[sanitizer]=sha(sanitizer)
receipt.update(launcher_command=launcher_command,memory_instrumentation='compute-sanitizer memcheck' if native_memcheck else None)
receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
with (O/(tag+'.stdout')).open('w') as stdout,(O/(tag+'.stderr')).open('w') as stderr:
    try:
        result=subprocess.run(launcher_command,cwd=R,env=env,stdout=stdout,stderr=stderr,timeout=960 if fixed_domain_16 or native_memcheck else 180)
        receipt['exit_code']=result.returncode
    except subprocess.TimeoutExpired:receipt.update(exit_code=124,timed_out=True)
after=journal('journal_after');old={r.get('__CURSOR') for r in before};new=[r for r in after if r.get('__CURSOR') not in old]
health_after=capture('gpu_after',health_args)
receipt.update(ended_utc=now(),new_kernel_fault_records=new,historical_kernel_records_retained=len(before),sources_binary_inputs_unchanged=all(sha(p)==h for p,h in expected.items()),gpu_recovery_none_after=health_after.returncode==0 and ', None,' in health_after.stdout,artifacts={str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in O.glob(tag+'*') if p.is_file() and p!=receipt_path})
# Freeze the dispatch evidence before binding independent readback checks to it;
# the enclosing execution receipt can then add results without a circular hash.
receipt['native_exit_code']=receipt['exit_code']
receipt['native_execution']=True
dispatch_path=O/(tag+'_dispatch.json')
assert not dispatch_path.exists()
dispatch_path.write_text(json.dumps(receipt,indent=2)+'\n')
receipt['dispatch_receipt']={'path':str(dispatch_path),'sha256':sha(dispatch_path)}
checks=[]
receipt['native_nuclear_work_last_substep']=[]
dispatch_ok=(not native_tangent_capture and receipt['native_exit_code']==0 and not new
    and receipt['sources_binary_inputs_unchanged'] and receipt['gpu_recovery_none_after'])
try:
    if dispatch_ok:
        assert sha(dec_checker)==dec_checker_sha256
        for epoch in range(1,epoch_count+1):
            import check_biprism_nuclear_budget as NB
            import struct
            numerical_path=O/(tag+'_state.epoch'+str(epoch)+'.numerical.bin')
            numerical=numerical_path.read_bytes()
            records=(O/(tag+'_state.epoch'+str(epoch)+'.conv.bin')).read_bytes()
            for leaf in range(8):
                header,rows=NB.read_trace(numerical,leaf)
                assert NB.original_target_reached(header), 'private continuation is not physical acceptance'
                assert header[1]==3 and header[2]==struct.unpack_from('<I',records,leaf*128+24)[0]
                assert rows[-1][0]<=5e-9 and abs(rows[-1][1])<=1e-6
                receipt['native_nuclear_work_last_substep'].append({'epoch':epoch,'leaf':leaf,
                    'residual_evaluations':header[2],'nonlinear_trials':header[3],
                    'derivative_evaluations':header[4],'action_checks':header[5],
                    'target_replays':header[6],'base_evaluations':header[7],
                    'accepted_private_bases':header[8],'rejected_private_trials':header[9]})
            output=O/(tag+'_independent_dec_epoch'+str(epoch)+'.json')
            assert not output.exists()
            check_command=[sys.executable,str(dec_checker),'--prefix',str(O/(tag+'_state')),
                '--epoch',str(epoch),'--bp-case',str(case),
                '--main-source',selected_sources['nhs_biprism_variational_so3g_v10.cuh'],
                '--dec-source',selected_sources['nhs_biprism_dec_pcg.cuh'],
                '--electronic-source',selected_sources['nhs_biprism_electronic_discrete.cuh'],
                '--execution-receipt',str(dispatch_path),'--output',str(output)]
            error=None
            try:
                result=subprocess.run(check_command,capture_output=True,text=True,timeout=90)
                check_code,check_out,check_err=result.returncode,result.stdout,result.stderr
            except subprocess.TimeoutExpired as exc:
                check_code,error=124,'Independent checker exceeded 90 seconds'
                check_out,check_err=exc.stdout or '',exc.stderr or ''
                if isinstance(check_out,bytes):check_out=check_out.decode(errors='replace')
                if isinstance(check_err,bytes):check_err=check_err.decode(errors='replace')
            except OSError as exc:
                check_code,error=127,repr(exc)
                check_out,check_err='',repr(exc)+'\n'
            (O/(tag+'_independent_dec_epoch'+str(epoch)+'.stdout')).write_text(check_out)
            (O/(tag+'_independent_dec_epoch'+str(epoch)+'.stderr')).write_text(check_err)
            checks.append({'epoch':epoch,'command':check_command,'exit_code':check_code,'error':error,
                'checker_sha256':dec_checker_sha256,'result':{'path':str(output),'sha256':sha(output)} if output.exists() else None})
except Exception as exc:
    receipt['post_run_validation_error']=repr(exc)
finally:
    receipt['independent_dec_checks']=checks
    receipt['independent_dec_all_epochs_passed']=(dispatch_ok and len(checks)==epoch_count
        and all(c['exit_code']==0 for c in checks) and 'post_run_validation_error' not in receipt)
    receipt['independent_lif_work_passed']=None
    if connected_work and receipt['independent_dec_all_epochs_passed']:
        checker=R/'cuda/tests/check_biprism_connected_work.py'
        output=O/(tag+'_independent_lif_work.txt')
        command=[sys.executable,str(checker),'--execution',str(dispatch_path),
                 '--build',str(build_path),'--output',str(output)]
        try:
            assert not output.exists()
            result=subprocess.run(command,capture_output=True,text=True,timeout=180)
            code,out,err=result.returncode,result.stdout,result.stderr
        except (OSError,subprocess.TimeoutExpired,AssertionError) as exc:
            code,out,err=1,'',repr(exc)+'\n'
        (O/(tag+'_independent_lif_work.stdout')).write_text(out)
        (O/(tag+'_independent_lif_work.stderr')).write_text(err)
        receipt['independent_lif_work_check']={'command':command,'exit_code':code,
            'checker_sha256':sha(checker) if checker.exists() else None,
            'output_sha256':sha(output) if output.exists() else None}
        receipt['independent_lif_work_passed']=code==0
    receipt['fixed_domain_readback_passed']=None
    if fixed_domain_16 and receipt['independent_dec_all_epochs_passed']:
        output=O/(tag+'_fixed_domain_readback.txt')
        try:
            assert not output.exists()
            text=check_fixed_domain_readback(json.loads(dispatch_path.read_text()),build,checks,fixed_domain_reader_pins)
            with output.open('x') as f: f.write(text)
            receipt['fixed_domain_readback_passed']=True
            receipt['fixed_domain_readback']={'path':str(output),'sha256':sha(output),'bytes':output.stat().st_size}
        except Exception as exc:
            receipt['fixed_domain_readback_passed']=False
            receipt['fixed_domain_readback_error']=repr(exc)
    receipt['coupled_diagnostic_accepted']=(receipt['independent_dec_all_epochs_passed']
        and (not connected_work or receipt['independent_lif_work_passed'] is True)
        and (not fixed_domain_16 or receipt['fixed_domain_readback_passed'] is True))
    if receipt['native_exit_code']==0 and not receipt['coupled_diagnostic_accepted']:
        receipt['exit_code']=2
    receipt['validation_ended_utc']=now()
    try:
        receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    finally:
        fcntl.flock(lock_fd,fcntl.LOCK_UN);os.close(lock_fd)
print(json.dumps({'exit_code':receipt['exit_code'],'new_kernel_fault_records':len(new),'receipt':str(receipt_path)}),flush=True)
sys.exit(receipt['exit_code'])
