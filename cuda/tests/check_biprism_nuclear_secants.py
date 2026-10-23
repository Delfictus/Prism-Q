"""Read the actual QR predictor; actual residual gates remain separate."""
import argparse,hashlib,json,struct
from pathlib import Path
import numpy as np
STRIDE=196992;BASE=106752

def check(execution,epoch):
    d=json.loads(execution.read_text());assert d['native_execution'] and d['sources_binary_inputs_unchanged']
    assert d['new_kernel_fault_records']==[] and d['native_exit_code'] in (0,4)
    p=Path(d['command'][4]+'.epoch'+str(epoch)+'.numerical.bin');raw=p.read_bytes()
    assert d['artifacts'][str(p)]=={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    assert len(raw)==8*STRIDE
    def a(l,off,count):return np.frombuffer(raw,dtype='<f8',count=count,offset=l*STRIDE+BASE+off)
    count=None;columns=[];paired=[];actual=projected=0.
    for l in range(8):
        n,have,valid,it=struct.unpack_from('<4I',raw,l*STRIDE+BASE+87808)
        assert n<=8 and have==valid==1 and it<48
        if count is None:count=n
        assert count==n
        columns.append(np.c_[a(l,36864,4608).reshape(8,576),a(l,73792,8)][:n])
        paired.append(np.c_[a(l,0,4608).reshape(8,576),a(l,73728,8)][:n])
        h=a(l,87824,288).reshape(48,6)[:it+1]
        assert np.all(np.isfinite(h)) and np.all(h[:,0:4]>=0) and np.all(h[:,3]<=1)
        assert np.all(h[:,1]<=h[:,0]*(1.+1e-10)+1e-25), 'orthogonal model residual must not increase'
        if l==0:
            for i,row in enumerate(h):print('qr_iteration',i,' '.join(f'{k}={x:.17g}' for k,x in zip(['actual_norm2','subspace_unresolved2','secant_norm2','trust_weight','columns','raw_dx'],row)))
    Q=np.concatenate(columns,axis=1);T=np.concatenate(paired,axis=1)
    assert np.all(np.isfinite(Q)) and np.all(np.isfinite(T))
    defect=float(np.max(np.abs(Q@Q.T-np.eye(count)))) if count else 0.
    print('ORTHOGONALITY_MAX',defect,'COLUMNS',count,'TRUE_JACOBIAN_OR_PRODUCTION_ACCEPTANCE_CLAIM=false')
    assert defect<1e-8
    return defect

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execution',type=Path,required=True);p.add_argument('--epoch',type=int,required=True)
    a=p.parse_args();check(a.execution,a.epoch)
