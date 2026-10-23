//! Residue-anchored connected cavity component replay for recorded V1 frames.
//! Uses the same occupied/axial-wall/nearest-component predicates as the
//! physical-lining evaluator, without face attribution or site ranking.
use super::*;
use clap::Parser;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
pub struct Args {
    #[arg(long)] pub topology: PathBuf,
    #[arg(long)] pub trajectory: PathBuf,
    #[arg(long)] pub output: PathBuf,
    #[arg(long, default_value_t = 0)] pub stream: usize,
    #[arg(long, default_value_t = 1.4)] pub solvent_probe_radius_a: f64,
}

fn position(frame: &[u8], atom: usize) -> [f64;3] {
    std::array::from_fn(|d| f32::from_le_bytes(frame[12*atom+4*d..12*atom+4*d+4].try_into().unwrap()) as f64)
}
/// Exact connected-component volume predicate used by the physical-lining
/// evaluator, restricted to volume so full face provenance need not be
/// reconstructed for every recorded frame.
fn component_volume(atoms: &[([f64;3],f64)], spacing:f64, phase:f64,
                    config:&CoordinateCavityConfig, metric:&PhysicalSpatialMetric) -> Result<f64> {
    let n=(2.0*config.local_half_width_a/spacing).ceil() as usize+1;
    let size=n.checked_mul(n).and_then(|x|x.checked_mul(n)).context("component grid overflow")?;
    let origin=-config.local_half_width_a+phase*spacing;
    let terminal=origin+(n-1) as f64*spacing;
    let max_radius=atoms.iter().map(|(_,r)|*r).fold(0.0,f64::max);
    metric.require_unique_periodic_local_window(origin,terminal,max_radius)?;
    let index=|x:usize,y:usize,z:usize|(x*n+y)*n+z;
    let mut occupied=vec![false;size];
    for (pos,radius) in atoms {
        if (0..3).any(|d|pos[d]+radius<origin || pos[d]-radius>terminal){continue}
        let lo=(0..3).map(|d|(((pos[d]-radius-origin)/spacing).floor() as isize).clamp(0,n as isize-1) as usize).collect::<Vec<_>>();
        let hi=(0..3).map(|d|(((pos[d]+radius-origin)/spacing).ceil() as isize).clamp(0,n as isize-1) as usize).collect::<Vec<_>>();
        for x in lo[0]..=hi[0] {for y in lo[1]..=hi[1] {for z in lo[2]..=hi[2] {
            let p=[origin+x as f64*spacing,origin+y as f64*spacing,origin+z as f64*spacing];
            if norm2(metric.displacement(p,*pos))<=radius*radius {occupied[index(x,y,z)]=true}
        }}}
    }
    let dirs=[(1isize,0isize,0isize),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)];
    let horizon=(config.ray_horizon_a/spacing).ceil() as isize;
    let mut candidate=vec![false;size];
    for x in 0..n {for y in 0..n {for z in 0..n {
        let at=index(x,y,z); if occupied[at]{continue}
        let hits=dirs.map(|(dx,dy,dz)| (1..=horizon).any(|step| {
            let xx=x as isize+dx*step;let yy=y as isize+dy*step;let zz=z as isize+dz*step;
            xx>=0&&yy>=0&&zz>=0&&xx<n as isize&&yy<n as isize&&zz<n as isize&&occupied[index(xx as usize,yy as usize,zz as usize)]
        }));
        let opposed=(hits[0]&&hits[1])||(hits[2]&&hits[3])||(hits[4]&&hits[5]);
        candidate[at]=hits.iter().filter(|h|**h).count()>=config.minimum_blocked_axial_directions as usize
            &&(!config.require_opposed_wall_pair||opposed);
    }}}
    let center_coord=((0.0-origin)/spacing).round() as isize;
    let mut seed=None;let mut best=isize::MAX;
    for x in 0..n {for y in 0..n {for z in 0..n {
        if candidate[index(x,y,z)] {
            let d=(x as isize-center_coord).pow(2)+(y as isize-center_coord).pow(2)+(z as isize-center_coord).pow(2);
            if d<best {best=d;seed=Some((x,y,z))}
        }
    }}}
    let Some(start)=seed else {return Ok(0.0)};
    let mut visited=vec![false;size];let mut q=VecDeque::new();
    visited[index(start.0,start.1,start.2)]=true;q.push_back(start);
    let mut count=0u64;
    while let Some((x,y,z))=q.pop_front(){
        count+=1;
        for (dx,dy,dz) in dirs {
            let xx=x as isize+dx;let yy=y as isize+dy;let zz=z as isize+dz;
            if xx<0||yy<0||zz<0||xx>=n as isize||yy>=n as isize||zz>=n as isize{continue}
            let next=index(xx as usize,yy as usize,zz as usize);
            if candidate[next]&&!visited[next]{visited[next]=true;q.push_back((xx as usize,yy as usize,zz as usize))}
        }
    }
    Ok(count as f64*spacing.powi(3))
}

pub fn run(args:Args)->Result<()> {
    let top_raw=read_regular_nofollow(&args.topology,"topology")?;
    let mut top:PrismPrepTopology=serde_json::from_slice(&top_raw)?;
    crate::ghost_coordinate_segment::normalize_legacy_nonperiodic_boundary(&mut top)?;
    top.validate_analysis_residue_axis().map_err(anyhow::Error::msg)?;
    let metric=PhysicalSpatialMetric::from_topology(&top)?;
    if !matches!(metric.authority.mode,PhysicalCavitySpatialMetricMode::DeclaredNonperiodicEuclidean){
        bail!("recorded interface replay requires the declared nonperiodic metric")
    }
    let ids=top.atom_analysis_residue_ids.as_ref().context("analysis atom map absent")?;
    let group=|resid:i32| -> Vec<usize> {ids.iter().enumerate().filter_map(|(i,r)|
        (*r==resid && top.elements[i]!="H").then_some(i)).collect()};
    let sources=[("L193",195),("F197",199)];
    let targets=[("pre_alpha7_I273",vec![275]),("pre_alpha7_E274",vec![276]),
        ("pre_alpha7_G275",vec![277]),("pre_alpha7_A276",vec![278]),
        ("pre_alpha7_K277",vec![279]),("pre_alpha7_273_277",(275..=279).collect()),
        ("WPD_all",vec![182,183,184]),("C216",vec![218])];
    let mut pairs=Vec::new();
    for (source_name,source_id) in sources {
        let source_atoms=group(source_id);
        if source_atoms.is_empty(){bail!("{source_name} has no heavy atoms")}
        for (target_name,target_ids) in &targets {
            let target_atoms=target_ids.iter().flat_map(|r|group(*r)).collect::<Vec<_>>();
            if target_atoms.is_empty(){bail!("{target_name} has no heavy atoms")}
            pairs.push((format!("{source_name}--{target_name}"),source_atoms.clone(),target_atoms));
        }
    }
    let ca=|resid:i32| -> Result<usize> {ids.iter().enumerate().find_map(|(i,r)|
        (*r==resid && top.atom_names[i]=="CA").then_some(i)).context("required body anchor CA missing")};
    let anchors=[ca(195)?,ca(199)?,ca(275)?];
    // Contacts use heavy atoms; the native solvent-exclusion field uses every
    // receptor atom with its topology LJ radius, excluding only tag residues.
    let eligible=ids.iter().enumerate().filter_map(|(i,r)|
        (*r>=9).then_some(i)).collect::<Vec<_>>();
    let config=CoordinateCavityConfig::default();config.validate()?;
    let identity_map=receptor_atom_identity_map(&top)?;
    let topology_sha=sha256_bytes(&top_raw);
    let mapping_sha=top.analysis_residue_contract.as_ref().context("analysis contract absent")?
        .atom_analysis_residue_ids_sha256.clone();
    let estimator_sha=canonical_sha256(&config)?;
    if (args.solvent_probe_radius_a-config.solvent_probe_radius_a).abs()>1e-12 {
        bail!("this replay uses the native declared solvent probe radius")
    }
    let mut reader=BufReader::new(std::fs::File::open(&args.trajectory)?);
    let mut hdr=[0u8;24];reader.read_exact(&mut hdr)?;
    if &hdr[..8]!=b"PRISM4D\0"||u32::from_le_bytes(hdr[8..12].try_into().unwrap())!=1
        ||u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize!=top.n_atoms {
        bail!("trajectory V1 header/topology mismatch")
    }
    let interval=u32::from_le_bytes(hdr[16..20].try_into().unwrap()) as usize;
    if interval!=10 {bail!("unexpected recorded frame cadence")}
    let size=std::fs::metadata(&args.trajectory)?.len() as usize;
    let stride=top.n_atoms.checked_mul(12).context("frame stride overflow")?;
    let count=(size-24)/stride;let trailing=(size-24)%stride;
    let mut frame=vec![0u8;stride];
    let mut writer=BufWriter::new(std::fs::File::create(&args.output)?);
    writeln!(writer,"stream,frame_index,step,pair,anchor_min_heavy_distance_a,component_volume_mean_a3,component_volume_min_a3,component_volume_max_a3,body_fit_rmsd_a,body_fit_max_residual_a,grid_volumes_a3")?;
    let mut reference=None;
    for i in 0..count {
        reader.read_exact(&mut frame)?;
        let current=anchors.iter().map(|a|position(&frame,*a)).collect::<Vec<_>>();
        let ref_points=reference.get_or_insert_with(||current.clone());
        let fit=crate::fixed_cadence_coordinate::measure_body_frame_fit(ref_points,&current,[0.0;3])?;
        let transformed=eligible.iter().map(|atom|{
            let pos=fit.point_in_reference(position(&frame,*atom));
            let sigma=top.lj_params[*atom].sigma;
            if !sigma.is_finite()||sigma<=0.0{bail!("invalid LJ sigma at atom {atom}")}
            Ok((*atom,pos,0.5*2f64.powf(1.0/6.0)*sigma+config.solvent_probe_radius_a))
        }).collect::<Result<Vec<_>>>()?;
        let maximal_grid_extension=config.grid_spacings_a.iter().copied().fold(0.0,f64::max);
        for (pair,source_atoms,target_atoms) in &pairs {
            let mut min_distance_sq=f64::INFINITY;
            let mut closest=([0.0;3],[0.0;3]);
            for &a in source_atoms {for &b in target_atoms {
                let pa=position(&frame,a);let pb=position(&frame,b);
                let d=norm2(metric.displacement(pa,pb));
                if d<min_distance_sq {min_distance_sq=d;closest=(pa,pb)}
            }}
            let current_center=std::array::from_fn(|d|(closest.0[d]+closest.1[d])*0.5);
            let center_ref=fit.point_in_reference(current_center);
            let mut atoms=Vec::new();let mut parity_atoms=Vec::new();
            for (atom,pos,radius) in &transformed {
                let rel=std::array::from_fn(|d|pos[d]-center_ref[d]);
                if (0..3).all(|d|rel[d].abs()<=config.local_half_width_a+maximal_grid_extension+radius){
                    atoms.push((rel,*radius));
                    if i==0 {parity_atoms.push(local_receptor_atom(&identity_map,*atom,rel,*radius)?)}
                }
            }
            let mut volumes=Vec::new();
            for spacing in &config.grid_spacings_a {for phase in &config.grid_phase_offsets {
                volumes.push(component_volume(&atoms,*spacing,*phase,&config,&metric)?);
            }}
            if i==0 {
                let observation_sha="f".repeat(64);
                let payload_sha=sha256_bytes(&frame);
                let source=PhysicalGridSourceBinding {region_frame_observation:true,
                    ghost_record_sha256:&observation_sha,coordinate_frame_sha256:&payload_sha,
                    topology_sha256:&topology_sha,atom_residue_mapping_sha256:&mapping_sha,
                    estimator_sha256:&estimator_sha};
                let mut k=0;
                for spacing in &config.grid_spacings_a {for phase in &config.grid_phase_offsets {
                    let full=evaluate_grid_with_physical_lining(&parity_atoms,*spacing,*phase,
                        k as u64,&config,&metric,&source)?.estimate.volume_a3;
                    if (full-volumes[k]).abs()>1e-12 {
                        bail!("{pair} volume-only component differs from native physical lining grid {k}: {full} vs {}",volumes[k]);
                    }
                    k+=1;
                }}
                eprintln!("stream {} {pair} first-frame four-grid native cavity parity PASS",args.stream);
            }
            let mean=volumes.iter().sum::<f64>()/volumes.len() as f64;
            let low=volumes.iter().copied().fold(f64::INFINITY,f64::min);
            let high=volumes.iter().copied().fold(f64::NEG_INFINITY,f64::max);
            writeln!(writer,"{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},\"{}\"",args.stream,i,(i+1)*interval,pair,
                min_distance_sq.sqrt(),mean,low,high,fit.rmsd,fit.max_residual,
                volumes.iter().map(|x|format!("{x:.6}")).collect::<Vec<_>>().join(";"))?;
        }
        if i>0&&i%1000==0 {eprintln!("stream {} cavity {}/{}",args.stream,i,count)}
    }
    writer.flush()?;
    eprintln!("stream {} cavity complete frames {} trailing bytes {}",args.stream,count,trailing);
    Ok(())
}
