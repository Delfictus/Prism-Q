"""Independently read fixed-base Newton probes/trials; never grants acceptance."""
import argparse,hashlib,json,struct
from pathlib import Path
import numpy as np
STRIDE=196992;QR=106752

def format_action_check(index, stage, phase, columns, actual, linear, base_norm, weight, dx):
    """Phases5/8 share split action-error fields, not nonlinear residuals."""
    assert phase in (5,8)
    return (f'newton_iteration={index} phase={phase} columns={columns} probe_weight={weight:.17g} '
            f'independent_direction_relative_error={(linear/base_norm)**.5:.17g} '
            f'velocity_error2={max(0.,linear-dx):.17g} velocity_Jp2={actual:.17g} '
            f'scalar_error2={dx:.17g} scalar_Jp2={max(0.,base_norm-actual):.17g} '
            f'continuation_stage={stage}')

def inspect(execution,epoch):
    receipt=json.loads(execution.read_text())
    assert receipt['native_execution'] and receipt['sources_binary_inputs_unchanged']
    assert receipt['new_kernel_fault_records']==[]
    p=Path(receipt['command'][4]+f'.epoch{epoch}.numerical.bin');raw=p.read_bytes()
    assert receipt['artifacts'][str(p)]=={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    stride=len(raw)//8
    assert stride in (196992,198144,235136,349056,496768,510720,1680768,1697920,1784448,1837696,1997440) and len(raw)==8*stride
    capacity=64 if stride in (1680768,1697920,1784448,1837696,1997440) else (16 if stride>=349056 else 8)
    # Offsets are asserted by both the compiled CUDA and Rust layouts.
    off=({'control':605248,'newton':607568,'mode':607632,'df':294912,'alpha_df':590336,'history':605264,'input_basis':673768,'input_alpha':968680} if capacity==64 else dict(control=161728,newton=164048,mode=164112,df=73728,alpha_df=147584,
              history=161744,input_basis=168424,input_alpha=242152) if capacity==16 else
         dict(control=87808,newton=90128,mode=90192,df=36864,alpha_df=73792,
              history=87824,input_basis=91368,input_alpha=128232))
    Q=[];V=[];count=None;block_identity=None
    for leaf in range(8):
        base=leaf*stride+QR
        a=lambda off,n:np.frombuffer(raw,dtype='<f8',offset=base+off,count=n)
        n,have,valid,it=struct.unpack_from('<4I',raw,base+off['control'])
        assert n<=capacity and it<(3072 if stride==1997440 else 1536 if stride==1837696 else 1024 if stride==1784448 else 192 if stride in (510720,1680768,1697920,1784448,1837696,1997440) else 48)
        transition_reset=(stride in (510720,1680768,1697920,1784448,1837696,1997440) and n==0 and have==0 and valid==0)
        if not transition_reset:assert have==1 and valid in (1,2,3,4)
        if count is None:count=n
        assert count==n
        scalars=a(off['newton'],8)
        assert np.isfinite(scalars[0]) and scalars[0]>0 and np.isfinite(scalars[4]) and scalars[4]>0
        mode,restarts,probes,accepted,target_refresh=struct.unpack_from('<5I',raw,base+off['mode'])
        block=struct.unpack_from('<I',raw,base+390008)[0] if stride==496768 else 0
        assert block in (0,1)
        if block_identity is None:block_identity=block
        assert block_identity==block
        assert mode in (0,1,2,4,6,7) and target_refresh in (0,1)
        if valid==4:
            # Accepted private trial; fixed-h inverse rebuilding was skipped.
            assert stride in (1697920,1784448,1837696,1997440) and mode==1 and n==0
            assert struct.unpack_from("<I",raw,leaf*stride+1680768+10880)[0]==1
        inverse_updates,inverse_schema=struct.unpack_from('<2I',raw,base+242280) if stride==349056 else (0,0)
        inverse_updates=inverse_updates if inverse_schema==0x31494e42 else 0
        Q.append(np.c_[a(off['df'],capacity*576).reshape(capacity,576),a(off['alpha_df'],capacity)][:n])
        arc_active=stride in (1697920,1784448,1837696,1997440) and struct.unpack_from('<I',raw,leaf*stride+1680768+10880)[0]==1
        if arc_active:
            # One replicated global h unknown: eight weighted copies, not
            # eight independent timestep columns. Match the producer's1/8 dot.
            qh=np.frombuffer(raw,'<f8',64,leaf*stride+1680768+9816)[:n]
            Q[-1]=np.c_[Q[-1],np.sqrt(.125)*qh]
        if stride>=235136 and inverse_updates==0 and not transition_reset:
            vn=n+1 if mode==1 and n<capacity else n
            V.append(np.c_[a(off['input_basis'],capacity*576).reshape(capacity,576),a(off['input_alpha'],capacity)][:vn])
            if arc_active:
                vh=np.frombuffer(raw,'<f8',64,leaf*stride+1680768+10328)[:vn]
                V[-1]=np.c_[V[-1],np.sqrt(.125)*vh]
        if stride in (510720,1680768,1697920,1784448,1837696,1997440):
            from check_biprism_nuclear_budget import read_trace
            header,rows=read_trace(raw,leaf,stride);history=np.array(rows)[:,3:]
            if transition_reset:
                assert int(rows[-1][7])%1000==900 and header[11]>0
                print('PRIVATE_CONTINUATION_RESET',leaf,'stage',header[11],
                      'nonlinear_trials',header[3],'no_live_Newton_basis',True,
                      'original_step_accepted',False)
        else:
            history=a(off['history'],288).reshape(48,6)[:it+1]
        if not np.all(np.isfinite(history)):
            # The terminal tangent gate faults after the preceding C4. The
            # current producer enters one more iteration before consuming the
            # collective fault, leaving a deliberately unpublished final row.
            # Do not call its stale C4 scalars a newly evaluated candidate.
            tangent=struct.unpack_from('<I',raw,leaf*stride+1680768+17048)[0]
            projected=a(605232,1)[0]
            from check_biprism_nuclear_budget import terminal_tangent_refusal
            terminal=terminal_tangent_refusal(header,rows,n,mode,tangent,it,projected,a(605208,1)[0])
            print('TERMINAL_TANGENT_LINEAR_REFUSAL',leaf,'substep',header[1],
                  'evaluated_probe',it,'column',terminal['column'],'reason',terminal['reason'],
                  'column_norm2',terminal['column_norm2'],'projected_residual_squared',projected,
                  'required_max',1e-6,'following_unpublished_row',header[2]-1,
                  'current_C4_candidate',False,'original_h_reached',False)
            history=history[:-1]
        assert np.all(np.isfinite(history)) and np.all(history[:,:3]>=0)
        # read_trace separately validates every signed +/- derivative pair.
        if stride<510720:assert np.all(history[:,3]>=0)
        if leaf==0:
            for index,(actual,linear,base_norm,weight,phase,dx) in enumerate(history):
                stage,phase=divmod(int(phase),1000)
                phase,columns=divmod(phase,100)
                if phase==9:
                    print(f'newton_iteration={index} continuation_stage={stage} transition=private_initialization no_physical_commit=1')
                    continue
                if phase in (5,8):
                    print(format_action_check(index,stage,phase,columns,actual,linear,base_norm,weight,dx))
                    continue
                msg=f'newton_iteration={index} continuation_stage={stage} phase={phase} columns={columns} actual_norm2={actual:.17g} base_norm2={base_norm:.17g} entering_linear_residual2={linear:.17g} trial_or_probe_weight={weight:.17g} raw_dx={dx:.17g}'
                if phase==4:
                    msg+=f' independent_direction_relative_error={(linear/base_norm)**.5:.17g}'
                if phase==2:
                    assert weight<=1 and linear<=base_norm*(1.+1e-10)+1e-25
                    prediction=linear if stride>=198144 else (1-weight)**2*(base_norm-linear)+linear
                    decrease=base_norm-actual
                    ratio=decrease/(base_norm-prediction) if base_norm>prediction else float('nan')
                    msg+=f' predicted_norm2={prediction:.17g} actual_reduction={decrease:.17g} agreement={ratio:.17g}'
                print(msg)
            print('FIXED_BASE_NEWTON', 'accepted_private_bases',accepted,'derivative_probes',probes,'model_restarts',restarts,'pending_mode',mode,'fd_radius',scalars[4])
            print('INVERSE_SECANT_UPDATES',inverse_updates,'scope=solver_private_action_requires_actual_trial_validation')
    matrix=np.concatenate(Q,axis=1)
    if block_identity:
        defect=0.
        for leaf_matrix in Q:
            gram=leaf_matrix@leaf_matrix.T
            active=(np.diag(gram)>.5).astype(float)
            defect=max(defect,float(np.max(np.abs(gram-np.diag(active)))))
    else:
        defect=float(np.max(np.abs(matrix@matrix.T-np.eye(count)))) if count else 0.
    assert np.isfinite(defect) and defect<1e-8
    print('MEASURED_COLUMN_ORTHOGONALITY_MAX',defect,'BLOCK_MODEL',block_identity,'NATIVE_PHYSICAL_ACCEPTANCE',False)
    if V:
        matrix=np.concatenate(V,axis=1)
        defect=float(np.max(np.abs(matrix@matrix.T-np.eye(len(matrix)))))
        assert defect<1e-8
        print('ARNOLDI_INPUT_ORTHOGONALITY_MAX',defect)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execution',type=Path,required=True);p.add_argument('--epoch',type=int,required=True)
    a=p.parse_args();inspect(a.execution,a.epoch)
