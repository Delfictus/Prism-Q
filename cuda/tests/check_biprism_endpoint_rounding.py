#!/usr/bin/env python3
"""Reconstruct FP32 endpoint-velocity kinetic rounding from native raw owners.
CPU reconstruction only; never grants trajectory or scientific acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def reconstruct(execution, epoch):
    dispatch=json.loads(Path(execution).read_text())
    assert dispatch['native_execution'] and dispatch['sources_binary_inputs_unchanged']
    prefix=dispatch['command'][4]+f'.epoch{epoch}'
    def artifact(suffix):
        name=prefix+'.'+suffix;data=Path(name).read_bytes()
        assert dispatch['artifacts'][name]=={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
        return data
    raw=artifact('numerical.bin');tiles=artifact('tiles.bin');stride=len(raw)//8
    assert stride in (97536,106752,143872,196992,198144,235136,349056,496768,510720,1680768,1697920,1784448,1837696,1997440) and len(raw)==stride*8
    mass_path=dispatch['command'][3];mass_bytes=Path(mass_path).read_bytes()
    assert hashlib.sha256(mass_bytes).hexdigest()==dispatch['input_hashes'][mass_path]
    inv=np.frombuffer(mass_bytes,dtype='<f4');total=0.
    for leaf in range(8):
        base=leaf*stride
        dtype,offset=('<f4',34048) if stride==97536 else ('<f8',97536)
        g=np.frombuffer(raw,dtype=dtype,count=6*192,offset=base+offset).astype(np.float64).reshape(6,192)
        gm=np.empty((192,3,3));gm[:,0,0]=g[0];gm[:,1,1]=g[1];gm[:,2,2]=g[2]
        gm[:,0,1]=gm[:,1,0]=g[3];gm[:,0,2]=gm[:,2,0]=g[4];gm[:,1,2]=gm[:,2,1]=g[5]
        ids=np.frombuffer(tiles,dtype='<u4',count=192,offset=leaf*225408)&0xffffff
        masses=(np.float32(1.)/np.maximum(inv[ids],np.float32(1.e-6))).astype(np.float64)
        endp=np.frombuffer(raw,dtype='<f8',count=3*192,offset=base+56832).reshape(3,192).T
        represented=np.frombuffer(raw,dtype='<f4',count=3*192,offset=base+11520).astype(np.float64).reshape(3,192).T
        rawv=np.linalg.solve(gm,endp[...,None])[...,0]/masses[:,None]
        rounding=.5*np.sum(masses*np.einsum('ij,ijk,ik->i',represented-rawv,gm,represented+rawv))
        assert np.isfinite(rounding)
        total+=rounding
        print(f'epoch={epoch} leaf={leaf} endpoint_representation_kinetic_delta_ha={rounding:.17g}')
    print(f'all8_endpoint_representation_kinetic_delta_ha={total:.17g} native_acceptance=false scope=CPU_linalg_reconstruction')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution',required=True,type=Path)
    parser.add_argument('--epoch',required=True,type=int)
    args=parser.parse_args();reconstruct(args.execution,args.epoch)
