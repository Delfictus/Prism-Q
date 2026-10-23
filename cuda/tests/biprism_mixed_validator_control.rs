// Host-contract inputs compiled with the actual production Rust validator.
// No CUDA execution or molecular acceptance is inferred.

fn sample_kinetic(leaf:u32,stage:u32,body:u32,delta:f64,work:f64,stationary:bool)->MixedSample {
    let mut s=MixedSample {magic:MIXED_MAGIC,leaf,stage,state:1,scheduled_step:body+1,
        committed_substeps:4*(body+u32::from(stage!=0)),mode:2,generation:73,
        body_ordinal:u64::from(body),preparation_generation:41,source:90+leaf,
        dec_iterations:1,..Default::default()};
    // Independent constant model terms and positive physical kinetic reference.
    s.values[0]=10. + delta/KCAL_TO_HARTREE;
    s.values[1]=5.;s.values[2]=2.;s.values[3]=0.01;
    s.values[4]=0.02;s.values[5]=0.03;s.values[6]=-0.01;
    s.values[7]=0.004;s.values[8]=-0.003;
    s.values[9]=s.values[4..9].iter().sum();
    s.values[10]=s.values[3]-s.values[2]*KCAL_TO_HARTREE+s.values[9];
    s.values[11]=(s.values[0]+s.values[1])*KCAL_TO_HARTREE+s.values[10];
    s.values[12]=work/KCAL_TO_HARTREE;s.values[15]=192.;
    s.values[16]=0.01;s.values[17]=3.;s.values[18]=0.01;
    if stationary {
        for k in [1,2,3,16,17,18] {s.values[k]=0.;}
        s.values[10]=s.values[9];s.values[11]=s.values[0]*KCAL_TO_HARTREE+s.values[10];
    }
    s
}
fn sample(leaf:u32,stage:u32,body:u32,delta:f64,work:f64)->MixedSample {
    sample_kinetic(leaf,stage,body,delta,work,false)
}
fn fixture(deltas:&[f64],works:&[f64])->MixedReadback {
    fixture_kinetic(deltas,works,false)
}
fn fixture_kinetic(deltas:&[f64],works:&[f64],stationary:bool)->MixedReadback {
    assert_eq!(deltas.len(),works.len());let n=deltas.len();
    let mut r=MixedReadback {window:MixedWindow {magic:MIXED_MAGIC,capacity:500,
        requested_bodies:n as u32,first_step:1,generation:73,preparation_generation:41,
        ..Default::default()},before:MixedEnergyState::default(),after:MixedEnergyState::default(),
        globals:vec![],leaves:vec![],samples:vec![]};
    let initial=sample_kinetic(0,0,0,0.,0.,stationary).values[11]*8.;
    let mut prior=initial;let mut all_delta=0.;let mut all_work=0.;
    let mut maxbody:f64=0.;let mut maxrun:f64=0.;
    for body in 0..n {
        let old_delta=all_delta;let old_work=all_work;
        all_delta+=deltas[body];all_work+=works[body];
        let begin=r.samples.len();
        for stage in 0..7 {for leaf in 0..8 {r.samples.push(sample_kinetic(leaf,stage,body as u32,
            if stage==0 {old_delta/8.} else {all_delta/8.},
            if stage==0 {old_work/8.} else {all_work/8.},stationary));}}
        let mut end=0.;let mut ptotal=0.;let mut declared=0.;let mut k=0.;
        for leaf in 0..8 {
            let a=r.samples[begin+leaf];let z=r.samples[begin+5*8+leaf];
            let mut l=MixedLeafLedger {magic:MIXED_MAGIC,leaf:leaf as u32,state:3,
                scheduled_step:body as u32+1,committed_substeps:4*(body as u32+1),mode:2,
                source:90+leaf as u32,generation:73,body_ordinal:body as u64,
                preparation_generation:41,..Default::default()};
            l.values[0]=(z.values[12]-a.values[12])*KCAL_TO_HARTREE;
            l.values[5]=l.values[0];l.values[6]=z.values[11]-a.values[11];
            l.values[7]=l.values[6]-l.values[5];
            l.values[11]=a.values[11];l.values[12]=z.values[11];
            l.values[13]=a.values[3]+a.values[17]*KCAL_TO_HARTREE;
            l.values[14]=z.values[3]+z.values[17]*KCAL_TO_HARTREE;
            l.values[15]=a.values[11];
            end+=z.values[11];ptotal+=a.values[11];declared+=l.values[5];k+=l.values[14];
            r.after.previous_native_work[leaf]=z.values[12];
            r.after.previous_leaf_energy[leaf]=z.values[11];
            r.after.previous_leaf_kinetic[leaf]=l.values[14];r.leaves.push(l);
        }
        let mut g=MixedGlobalLedger {magic:MIXED_MAGIC,state:3,scheduled_step:body as u32+1,
            committed_substeps:4*(body as u32+1),stage_mask:127,generation:73,
            body_ordinal:body as u64,preparation_generation:41,accepted_bodies:body as u64+1,
            ..Default::default()};
        g.values[0]=initial;g.values[1]=ptotal;g.values[2]=end;g.values[3]=declared;
        g.values[4]=(end-ptotal)-declared;g.values[5]=all_work;
        g.values[6]=initial+all_work;g.values[7]=(end-initial)-all_work;
        g.values[8]=k;g.values[9]=k;g.values[13]=MIXED_TOL;
        maxbody=maxbody.max(g.values[4].abs());maxrun=maxrun.max(g.values[7].abs());
        g.values[11]=maxbody;g.values[12]=maxrun;
        assert!(close(prior,ptotal));prior=end;r.globals.push(g);
    }
    r.after.magic=MIXED_MAGIC;r.after.state=3;r.after.last_accepted_step=n as u32;
    r.after.generation=73;r.after.preparation_generation=41;r.after.accepted_bodies=n as u64;
    r.after.totals[0]=initial;r.after.totals[1]=prior;r.after.totals[2]=initial+all_work;
    r.after.totals[3]=all_work;r.after.totals[4]=r.globals[0].values[8];
    r.after.totals[5]=r.globals[n-1].values[9];
    r.after.totals[7]=maxbody;r.after.totals[8]=maxrun;r
}
fn reject(name:&str,edit:impl FnOnce(&mut MixedReadback)) {
    let mut r=fixture(&[0.,0.],&[0.,0.]);edit(&mut r);
    match validate_mixed_readback(&r){Ok(())=>panic!("accepted malformed control: {name}"),
        Err(e)=>println!("PASS reject {name}: {e}")}
}
fn main(){
    assert_eq!(core::mem::size_of::<MixedVertex>(),336);
    assert_eq!(core::mem::size_of::<MixedEdge>(),64);
    assert_eq!(core::mem::size_of::<MixedEnergyState>(),512);
    validate_mixed_readback(&fixture(&[0.,0.],&[0.,0.])).unwrap();
    println!("PASS accept constant complete two-body reference");
    validate_mixed_readback(&fixture(&[0.001,0.002],&[0.001,0.002])).unwrap();
    println!("PASS accept explicitly accounted nonzero work");
    let mut later=fixture(&[0.,0.],&[0.,0.]);
    later.before=fixture(&[0.],&[0.]).after;later.window.first_body=1;
    later.window.first_step=2;later.window.requested_bodies=1;
    later.globals.drain(..1);later.leaves.drain(..8);later.samples.drain(..56);
    validate_mixed_readback(&later).unwrap();
    println!("PASS accept second window with retained prior reference");
    validate_mixed_readback(&fixture_kinetic(&[0.,0.],&[0.,0.],true)).unwrap();
    println!("PASS accept finite stationary mixed state without normalized-kinetic assumptions");
    let mut stationary_later=fixture_kinetic(&[0.,0.],&[0.,0.],true);
    stationary_later.before=fixture_kinetic(&[0.],&[0.],true).after;
    stationary_later.window.first_body=1;stationary_later.window.first_step=2;
    stationary_later.window.requested_bodies=1;stationary_later.globals.drain(..1);
    stationary_later.leaves.drain(..8);stationary_later.samples.drain(..56);
    validate_mixed_readback(&stationary_later).unwrap();
    println!("PASS accept retained zero-kinetic reference in later window");
    reject("missing stage",|r|{r.samples.pop();});
    reject("empty ring",|r|{r.samples.fill(MixedSample::default());});
    reject("leaf permutation",|r|{r.leaves.swap(0,1);});
    reject("stage permutation",|r|{r.samples.swap(0,8);});
    reject("stale sample generation",|r|{r.samples[0].generation-=1;});
    reject("stale preparation",|r|{r.samples[8].preparation_generation-=1;});
    reject("native source disagreement",|r|{r.samples[8].source+=1;});
    reject("mode disagreement",|r|{r.samples[8].mode^=1;});
    reject("wrong committed count",|r|{r.samples[0].committed_substeps+=4;});
    reject("failed true DEC residual",|r|{r.samples[8].values[14]=1.0001e-6;});
    reject("negative residual",|r|{r.samples[8].values[14]=-1.;});
    reject("NaN sample",|r|{r.samples[8].values[6]=f64::NAN;});
    reject("infinite global work",|r|{r.globals[0].values[3]=f64::INFINITY;});
    reject("replaced correction",|r|{r.samples[8].values[10]+=0.01;});
    reject("hidden Phase2 work",|r|{r.leaves[0].values[1]+=0.001;});
    reject("omitted native work",|r|{r.leaves[0].values[0]+=0.001;});
    reject("native work reset",|r|{r.samples[56].values[12]+=0.001;});
    reject("persistent reference reset",|r|{r.after.totals[0]=0.;});
    reject("persistent partial commit",|r|{r.after.accepted_bodies=1;});
    reject("pending global not committed",|r|{r.globals[0].state=2;});
    reject("sticky fault",|r|{r.after.sticky_fault=1;});
    reject("changed physical tolerance",|r|{r.globals[0].values[13]=1e-5;});
    reject("nonzero cold reference",|r|{r.before.state=3;});
    reject("nonfinite retained expected energy",|r|{r.after.totals[2]=f64::NAN;});
    reject("changed retained peak drift",|r|{r.after.totals[7]=0.2;});
    reject("unreconciled global prefix",|r|{r.globals[0].values[10]=0.1;});
    assert!(validate_mixed_readback(&fixture(&[1.01e-6],&[0.])).is_err());
    println!("PASS reject actual body residual above1e-6");
    let accumulated=fixture(&[0.6e-6,0.6e-6],&[0.,0.]);
    assert!(accumulated.globals.iter().all(|g|g.values[4].abs()<1e-6));
    assert!(validate_mixed_readback(&accumulated).is_err());
    println!("PASS reject cumulative residual although each body is below1e-6");
    let mut diagnostic=fixture(&[0.,0.],&[0.,0.]);
    for (i,s) in diagnostic.samples.iter_mut().enumerate(){s.values[19]=(i as f64+1.)*0.125;}
    validate_mixed_readback(&diagnostic).unwrap();
    println!("PASS finite private refinement remains outside declared work");
    reject("nonfinite private refinement",|r|{r.samples[8].values[19]=f64::NAN;});
    let mut row=MixedVertex::default();row.phi=1.+2.0f64.powi(-40);row.phi_entry=1.+2.0f64.powi(-39);
    row.psi_precise=[1.0,2.0f64.powi(-40),0.,0.];
    let raw=unsafe{core::slice::from_raw_parts((&row as *const MixedVertex).cast::<u8>(),336)};
    assert_eq!(u64::from_le_bytes(raw[80..88].try_into().unwrap()),row.phi.to_bits());
    assert_eq!(u64::from_le_bytes(raw[240..248].try_into().unwrap()),row.phi_entry.to_bits());
    assert_ne!(row.phi,f64::from(row.phi as f32));
    assert_eq!(u64::from_le_bytes(raw[312..320].try_into().unwrap()),row.psi_precise[1].to_bits());
    println!("PASS BME6 full-potential row extent, offsets and precision");
}
