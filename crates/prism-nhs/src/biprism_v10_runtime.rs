//! Owned device state and the frozen v10 / Phase2 / Bowyer-Watson launch ABIs.
//!
//! The physical engine owns propagation, units, clock and rollback policy. This
//! module retains its borrowed native buffers, owns separate atomic-unit working
//! planes, and never converts an AMBER pointer into a v10 pointer implicitly.
//! Capture methods require the physical owner to order its explicit unit bridge
//! and exclusive core propagation. Enqueue success is not scientific acceptance.

use crate::biprism_lif_runtime::BiprismLifDeviceState;
use crate::biprism_coordinates::{BiprismCoordinateOwnerFfi, COORDINATE_MAGIC,
    COORDINATE_VERSION, NATIVE_ANGSTROM, ATOMIC_BOHR};
use cudarc::driver::{
    sys, CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DeviceRepr,
    LaunchConfig, PushKernelArg,
};
use std::{any::Any, collections::HashSet, sync::Arc};

pub const BIPRISM_LEAVES: usize = 8;
pub const BIPRISM_VERTICES: usize = 256;
pub const BIPRISM_CORE_VERTICES: usize = 192;
pub const BIPRISM_TILE_BYTES: usize = 225_408;
pub const BIPRISM_CONVERGENCE_BYTES: usize = 128;
pub const BIPRISM_ADJUDICATOR_BYTES: usize = 256;
pub const BIPRISM_DYNAMIC_SMEM_BYTES: u32 = 31_616;
pub const BIPRISM_ELECTRONIC_EXCHANGE_BYTES:usize=17_792;
pub const BIPRISM_ELECTRONIC_EXCHANGE_MAGIC:u32=0x4643_5032;
pub const BIPRISM_MAX_NUCLEAR_EVALUATIONS:u32=3072;
pub const BIPRISM_MAX_NUCLEAR_TRIALS:u32=512;
pub const BIPRISM_NUMERICAL_SCRATCH_BYTES:usize=1997440;
pub const BIPRISM_WORKING_PHI_OFFSET:usize=38656;
pub const BIPRISM_POTENTIAL_MAGIC:u32=0x3150_5042;
pub const BIPRISM_ORDERED_EDGES_BYTES:usize=6656;
pub const BIPRISM_PEER_EDGE_MAP_OFFSET:usize=30976;
pub const BIPRISM_CONVERGENCE_MAGIC: u32 = 0x4250_5631;
pub const BIPRISM_BW_SCRATCH_BYTES: usize = 1_461_760;
pub const BIPRISM_BW_STATUS_BYTES: usize = 80;
pub const BIPRISM_BW_MAGIC: u32 = 0x4257_4231;
pub const BIPRISM_UNIT_BRIDGE_MAGIC: u32 = 0x4255_4231;
const TILE_COUNT_OFFSET: usize = 225_280;

/// Exact CUDA ABI: eight eight-entry pointer arrays, including separately owned replay scratch.
#[repr(C, align(128))]
#[derive(Clone, Copy, Debug, Default)]
pub struct FoliatedStreamPtrTableFfi {
    pub coords: [u64; BIPRISM_LEAVES],
    pub velocities: [u64; BIPRISM_LEAVES],
    pub forces: [u64; BIPRISM_LEAVES],
    pub inverse_masses: [u64; BIPRISM_LEAVES],
    pub true_inverse_masses: [u64; BIPRISM_LEAVES],
    pub core_shake_mask: [u64; BIPRISM_LEAVES],
    pub d_electronic_exchange:[u64;BIPRISM_LEAVES],
    pub d_numerical_scratch:[u64;8],
    pub d_shared_step:u64,
    pub physical_step_mode:u32,
    pub reserved_coordinate_alignment:u32,
    pub d_coordinate_owners:u64,
    pub d_ordered_edges:u64,
    pub d_potential_owners:u64,
    pub d_spinor_owners:u64,
    pub reserved_shared_tail:[u32;20],
}
// Every byte, including the alignment tail, is initialized explicitly.
unsafe impl DeviceRepr for FoliatedStreamPtrTableFfi {}

pub const BIPRISM_SPINOR_MAGIC:u32=0x31535042;
pub const BIPRISM_SPINOR_OWNER_BYTES:usize=21632;
#[repr(C,align(128))]
#[derive(Clone,Copy,Debug,PartialEq)]
pub struct BiprismSpinorOwnerFfi {
    pub magic:u32,pub version:u32,pub leaf:u32,pub state:u32,
    pub allocation_generation:u64,pub topology_generation:u64,
    pub tile_address:u64,pub coordinate_owner_address:u64,
    pub accepted_committed_substeps:u32,pub accepted_step:u32,pub reserved:[u64;9],
    pub accepted:[[f64;4];192],pub substep_start:[[f64;4];192],pub candidate:[[f64;4];192],
    pub proposed_s0:[[f64;2];192],
}
impl Default for BiprismSpinorOwnerFfi {fn default()->Self{Self{
    magic:0,version:0,leaf:0,state:0,allocation_generation:0,topology_generation:0,
    tile_address:0,coordinate_owner_address:0,accepted_committed_substeps:0,accepted_step:0,reserved:[0;9],
    accepted:[[0.;4];192],substep_start:[[0.;4];192],candidate:[[0.;4];192],proposed_s0:[[0.;2];192]}}}
unsafe impl DeviceRepr for BiprismSpinorOwnerFfi {}
const _:()={assert!(core::mem::size_of::<BiprismSpinorOwnerFfi>()==BIPRISM_SPINOR_OWNER_BYTES);
    assert!(core::mem::offset_of!(BiprismSpinorOwnerFfi,accepted)==128);
    assert!(core::mem::offset_of!(BiprismSpinorOwnerFfi,substep_start)==6272);
    assert!(core::mem::offset_of!(BiprismSpinorOwnerFfi,candidate)==12416);
    assert!(core::mem::offset_of!(BiprismSpinorOwnerFfi,proposed_s0)==18560);};

/// Persistent FP64 potential and a separately prepared operator. Only the
/// existing all-eight main gate advances the accepted bank after initialization.
#[repr(C,align(128))]
#[derive(Clone,Copy,Debug,Default,PartialEq,Eq,serde::Serialize)]
pub struct BiprismPotentialOwnerFfi {
    pub abi_magic:u32,pub version:u32,pub leaf_id:u32,pub state:u32,
    pub allocation_generation:u64,pub topology_generation:u64,
    pub tile_address:u64,pub coordinate_owner_address:u64,
    pub accepted_phi:u64,pub prepared_phi:u64,
    pub accepted_committed_substeps:u32,pub accepted_step:u32,
    pub prepared_from_committed_substeps:u32,pub prepared_step:u32,
    pub accepted_operator_generation:u64,pub prepared_operator_generation:u64,
    pub projection_state:u32,pub projection_committed_substeps:u32,pub projection_step:u32,pub reserved0:u32,
    pub projection_operator_generation:u64,pub reserved1:u64,
}
unsafe impl DeviceRepr for BiprismPotentialOwnerFfi {}
pub struct BiprismPotentialReadback {
    pub owners:Vec<BiprismPotentialOwnerFfi>,
    pub accepted:Vec<f64>,pub prepared:Vec<f64>,pub working:Vec<f64>,
}
impl BiprismPotentialReadback {
    pub fn index(&self)->serde_json::Value {serde_json::json!({
        "schema":"biprism.potential_precision.v1","owner_bytes":128,"leaf_count":8,"values_per_leaf":256,
        "value_type":"IEEE754 binary64 little endian","owners":self.owners,
        "states":{"cold":0,"accepted":1,"prepared":2},
        "accepted_scope":"cold qualified baseline or existing all-eight main commit; complete physical body acceptance is separate",
        "prepared_scope":"qualified Phase2 operator/boundary refresh for a specific scheduled step, accepted bank unchanged",
        "working_scope":"current main attempt replay scratch; not accepted potential authority",
        "compatibility_projection":"tile.phi_dec is RN32(full selected bank), never the precise source",
        "scientific_restart":false})}
    pub fn owner_bytes(&self)->Result<&[u8],String>{potential_bytes(&self.owners)}
    pub fn accepted_bytes(&self)->Result<&[u8],String>{potential_bytes(&self.accepted)}
    pub fn prepared_bytes(&self)->Result<&[u8],String>{potential_bytes(&self.prepared)}
    pub fn working_bytes(&self)->Result<&[u8],String>{potential_bytes(&self.working)}
}
fn potential_bytes<T>(values:&[T])->Result<&[u8],String>{
    if !cfg!(target_endian="little"){return Err("potential readback requires little endian".into());}
    Ok(unsafe{core::slice::from_raw_parts(values.as_ptr().cast(),core::mem::size_of_val(values))})
}
/// Production refreshes the operator before every main invocation. Scalar
/// component diagnostics may explicitly continue the accepted operator instead.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum PotentialEndpointProfile { PreparedProduction, ScalarDiagnostic }
pub fn validate_accepted_potential(read:&BiprismPotentialReadback,tiles:&[u8],committed:u32,step:u32,topology:u64,
    profile:PotentialEndpointProfile)->Result<(),String>{
    if read.owners.len()!=8||read.accepted.len()!=2048||read.prepared.len()!=2048||read.working.len()!=2048
        ||tiles.len()!=8*BIPRISM_TILE_BYTES||topology==0||committed%4!=0{
        return Err("potential readback extent/generation/commit count".into());}
    for (leaf,o) in read.owners.iter().enumerate(){
        if o.abi_magic!=BIPRISM_POTENTIAL_MAGIC||o.version!=1||o.leaf_id!=leaf as u32||o.state!=1
            ||o.allocation_generation==0||o.topology_generation!=topology||o.tile_address==0||o.coordinate_owner_address==0
            ||o.accepted_phi==0||o.prepared_phi==0||o.accepted_phi==o.prepared_phi||o.reserved0!=0||o.reserved1!=0
            ||o.accepted_committed_substeps!=committed||o.accepted_step!=step||o.accepted_operator_generation==0
            ||o.projection_state!=1||o.projection_committed_substeps!=committed||o.projection_step!=step
            ||o.projection_operator_generation!=o.accepted_operator_generation{return Err(format!("potential leaf{leaf} accepted identity mismatch"));}
        let prepared_used=o.prepared_operator_generation!=0;
        if committed==0||!prepared_used {
            // Cold publication leaves the zeroed prepared bank untouched. Only
            // an explicitly selected scalar diagnostic may keep using that
            // initialized operator after a main commit.
            if o.prepared_from_committed_substeps!=0||o.prepared_step!=0||prepared_used
                ||o.accepted_operator_generation!=1
                ||(committed!=0&&profile!=PotentialEndpointProfile::ScalarDiagnostic)
                ||step<committed/4 {
                return Err(format!("potential leaf{leaf} cold/scalar history mismatch"));
            }
        } else {
            let elapsed=committed.checked_sub(o.prepared_from_committed_substeps)
                .ok_or_else(||format!("potential leaf{leaf} prepared commit is in the future"))?;
            let scheduled=step.checked_sub(o.prepared_step).and_then(|n|n.checked_add(1))
                .ok_or_else(||format!("potential leaf{leaf} prepared step is in the future"))?;
            if elapsed<4||elapsed%4!=0||o.prepared_from_committed_substeps%4!=0
                ||o.prepared_step==0||o.prepared_operator_generation<=1
                ||o.prepared_operator_generation!=o.accepted_operator_generation
                ||elapsed/4!=scheduled
                ||(profile==PotentialEndpointProfile::PreparedProduction&&elapsed!=4) {
                return Err(format!("potential leaf{leaf} prepared/accepted history mismatch"));
            }
        }
        for vertex in 0..256 {let value=read.accepted[leaf*256+vertex];
            let offset=leaf*BIPRISM_TILE_BYTES+94208+vertex*4;
            let projection=u32::from_le_bytes(tiles[offset..offset+4].try_into().unwrap());
            if !value.is_finite()||!(value as f32).is_finite()||(value as f32).to_bits()!=projection{
                return Err(format!("potential leaf{leaf}/vertex{vertex} full value/projection mismatch"));}
            if committed>0&&value.to_bits()!=read.working[leaf*256+vertex].to_bits(){
                return Err(format!("potential leaf{leaf}/vertex{vertex} accepted/working commit mismatch"));}
            let prepared=read.prepared[leaf*256+vertex];
            if !prepared.is_finite()||!(prepared as f32).is_finite()
                ||(!prepared_used&&prepared.to_bits()!=0)
                ||(prepared_used&&vertex>=192&&prepared.to_bits()!=value.to_bits()) {
                return Err(format!("potential leaf{leaf}/vertex{vertex} prepared value/halo mismatch"));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod potential_readback_tests {
    use super::*;
    fn fixture(committed:u32,step:u32)->(BiprismPotentialReadback,Vec<u8>){
        let mut r=BiprismPotentialReadback{owners:Vec::new(),accepted:vec![0.;2048],
            prepared:vec![0.;2048],working:vec![0.;2048]};
        let mut tiles=vec![0u8;8*BIPRISM_TILE_BYTES];
        for leaf in 0..8 {
            let operator=if committed==0{1}else{3};
            r.owners.push(BiprismPotentialOwnerFfi{abi_magic:BIPRISM_POTENTIAL_MAGIC,version:1,
                leaf_id:leaf as u32,state:1,allocation_generation:17,topology_generation:29,
                tile_address:0x100000+(leaf*BIPRISM_TILE_BYTES) as u64,
                coordinate_owner_address:0x400000+(leaf*96) as u64,
                accepted_phi:0x500000+(leaf*4096) as u64,prepared_phi:0x600000+(leaf*4096) as u64,
                accepted_committed_substeps:committed,accepted_step:step,accepted_operator_generation:operator,
                prepared_from_committed_substeps:committed.saturating_sub(4),
                prepared_step:if committed==0{0}else{step},
                prepared_operator_generation:if committed==0{0}else{operator},
                projection_state:1,projection_committed_substeps:committed,projection_step:step,
                projection_operator_generation:operator,..Default::default()});
            for vertex in 0..256 {let i=leaf*256+vertex;
                let value=1.+i as f64/4096.+2.0f64.powi(-40);
                r.accepted[i]=value;r.working[i]=value;
                if committed!=0 {r.prepared[i]=if vertex<192{value+2.0f64.powi(-32)}else{value};}
                let offset=leaf*BIPRISM_TILE_BYTES+94208+vertex*4;
                tiles[offset..offset+4].copy_from_slice(&(value as f32).to_le_bytes());
            }
        }
        (r,tiles)
    }
    fn production(r:&BiprismPotentialReadback,t:&[u8],committed:u32,step:u32)->Result<(),String>{
        validate_accepted_potential(r,t,committed,step,29,PotentialEndpointProfile::PreparedProduction)
    }
    #[test]
    fn complete_full_precision_commit_and_cold_native_clock(){
        let (r,t)=fixture(12,74);
        production(&r,&t,12,74).unwrap();
        assert_ne!(r.accepted[0],f64::from(r.accepted[0] as f32));
        assert_eq!(potential_bytes(&r.owners).unwrap().len(),8*128);
        let (mut cold,t)=fixture(0,71);cold.working.fill(0.);
        production(&cold,&t,0,71).unwrap();
    }
    #[test]
    fn prepared_state_cannot_be_reported_as_accepted(){
        let (mut r,t)=fixture(4,72);r.owners[5].state=2;
        assert!(production(&r,&t,4,72).is_err());
    }
    #[test]
    fn compatibility_projection_cannot_replace_full_potential(){
        let (mut r,t)=fixture(4,72);r.accepted[17]=f64::from(r.accepted[17] as f32);
        assert!(production(&r,&t,4,72).is_err());
    }
    #[test]
    fn stale_identity_and_partial_population_refuse(){
        for edit in 0..8 {let(mut r,t)=fixture(4,72);match edit {
            0=>r.owners[7].topology_generation-=1,1=>r.owners[7].accepted_step-=1,
            2=>r.owners[7].accepted_committed_substeps-=4,3=>r.owners[7].projection_operator_generation-=1,
            4=>r.owners[7].leaf_id=0,5=>r.owners[7].reserved1=1,
            6=>{r.accepted.pop();},_=>{r.owners.pop();}}
            assert!(production(&r,&t,4,72).is_err(),"edit{edit}");
        }
    }
    #[test]
    fn altered_projection_and_nonfinite_full_value_refuse(){
        let (r,mut t)=fixture(4,72);t[94208]^=1;
        assert!(production(&r,&t,4,72).is_err());
        let(mut r,t)=fixture(4,72);r.accepted[0]=f64::NAN;r.working[0]=f64::NAN;
        assert!(production(&r,&t,4,72).is_err());
    }
    #[test]
    fn prepared_bank_and_chronology_corruptions_refuse(){
        for edit in 0..9 {let(mut r,t)=fixture(12,74);match edit {
            0=>r.prepared[0]=f64::NAN,1=>r.prepared[0]=f64::MAX,
            2=>r.owners[7].prepared_step-=1,3=>r.owners[7].prepared_step+=1,
            4=>r.owners[7].prepared_operator_generation-=1,
            5=>r.owners[7].prepared_from_committed_substeps+=4,
            6=>r.owners[7].prepared_from_committed_substeps-=4,
            7=>r.owners[7].prepared_from_committed_substeps=7,
            _=>r.prepared[7*256+192]+=2.0f64.powi(-40)}
            assert!(production(&r,&t,12,74).is_err(),"edit{edit}");
        }
    }
    #[test]
    fn prepared_halo_requires_exact_bits(){
        let(mut r,mut t)=fixture(4,72);let i=192;
        r.accepted[i]=0.;r.working[i]=0.;r.prepared[i]=-0.;
        t[94208+i*4..94208+i*4+4].copy_from_slice(&0.0f32.to_le_bytes());
        assert!(production(&r,&t,4,72).is_err());
        r.prepared[i]=0.;production(&r,&t,4,72).unwrap();
    }
    #[test]
    fn cold_history_cannot_contain_prepared_metadata_or_values(){
        for edit in 0..5 {let(mut r,t)=fixture(0,71);match edit {
            0=>r.owners[0].prepared_step=71,1=>r.owners[0].prepared_operator_generation=1,
            2=>r.owners[0].prepared_from_committed_substeps=4,
            3=>r.prepared[0]=2.0f64.powi(-40),_=>r.prepared[0]=-0.}
            assert!(production(&r,&t,0,71).is_err(),"edit{edit}");
        }
    }
    #[test]
    fn scalar_diagnostic_history_is_explicit_and_still_checked(){
        let(mut r,t)=fixture(0,71);
        for o in &mut r.owners {o.accepted_committed_substeps=8;o.accepted_step=73;
            o.projection_committed_substeps=8;o.projection_step=73;}
        validate_accepted_potential(&r,&t,8,73,29,PotentialEndpointProfile::ScalarDiagnostic).unwrap();
        assert!(production(&r,&t,8,73).is_err());
        let(mut r,t)=fixture(4,72);
        for o in &mut r.owners {o.accepted_committed_substeps=12;o.accepted_step=74;
            o.projection_committed_substeps=12;o.projection_step=74;}
        validate_accepted_potential(&r,&t,12,74,29,PotentialEndpointProfile::ScalarDiagnostic).unwrap();
        assert!(production(&r,&t,12,74).is_err());
        r.owners[0].prepared_step=73;
        assert!(validate_accepted_potential(&r,&t,12,74,29,PotentialEndpointProfile::ScalarDiagnostic).is_err());
    }
}


/// Device-built traversal scratch. The main kernel invalidates and rebuilds
/// every row from its current admitted topology on every invocation. This is
/// neither a cached mesh authority nor physical continuation state.
#[repr(C, align(128))]
#[derive(Clone, Copy, Debug)]
pub struct BiprismOrderedEdgesFfi {
    pub magic:u32,
    pub version:u32,
    pub leaf:u32,
    pub num_edges:u32,
    pub tile_address:u64,
    pub accepted_substeps_at_entry:u32,
    pub publication_state:u32,
    pub incident_count:u32,
    pub reserved:[u32;7],
    pub offsets:[u16;193],
    pub edge_index:[u16;3072],
    pub padding:[u8;62],
}

/// Exact derived replay scratch ABI. This is not a durable physical checkpoint.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FoliatedCouplingFooterFfi {
    pub base_energy_delta:f64, pub owned_pair_energy_delta:f64,
    pub attributed_coupling_energy_delta:f64, pub coupling_discrete_work:f64,
    pub physical_transfer_in:f64, pub balanced_local_defect:f64,
    pub global_energy_delta:f64, pub global_chain_defect:f64,
    pub max_coordinate_residual:f32, pub max_population_residual:f32,
    pub max_electronic_equation_residual:f32, pub max_norm_defect:f32,
    pub abi_magic:u32, pub leaf_id:u32, pub accepted_substeps_at_entry:u32,
    pub substep:u32, pub iteration:u32, pub local_fault_mask:u32,
    pub convergence_vote:u32, pub publication_phase:u32, pub reserved:[u32;4],
}
#[repr(C,align(128))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FoliatedCouplingLeafFfi {
    pub start_population:[f64;192], pub trial_population:[f64;192],
    pub energy_weight:[f32;192], pub reciprocal_response:[f64;192],
    pub live_weights:[f64;1536], pub footer:FoliatedCouplingFooterFfi,
}
const _:()={
    assert!(core::mem::size_of::<FoliatedCouplingFooterFfi>()==128);
    assert!(core::mem::offset_of!(FoliatedCouplingFooterFfi,max_coordinate_residual)==64);
    assert!(core::mem::offset_of!(FoliatedCouplingFooterFfi,abi_magic)==80);
    assert!(core::mem::offset_of!(FoliatedCouplingFooterFfi,reserved)==112);
    assert!(core::mem::size_of::<FoliatedCouplingLeafFfi>()==BIPRISM_ELECTRONIC_EXCHANGE_BYTES);
    assert!(core::mem::align_of::<FoliatedCouplingLeafFfi>()==128);
    assert!(core::mem::offset_of!(FoliatedCouplingLeafFfi,trial_population)==1536);
    assert!(core::mem::offset_of!(FoliatedCouplingLeafFfi,energy_weight)==3072);
    assert!(core::mem::offset_of!(FoliatedCouplingLeafFfi,reciprocal_response)==3840);
    assert!(core::mem::offset_of!(FoliatedCouplingLeafFfi,live_weights)==5376);
    assert!(core::mem::offset_of!(FoliatedCouplingLeafFfi,footer)==17664);
};
/// Unfiltered eight-leaf storage. Phases1/2/3 are temporary,4 names an accepted
/// final substep,5 is a faulted outer body. An endpoint phase alone is not commit.
pub struct BiprismElectronicExchangeReadback {pub raw_bytes:Vec<u8>,pub leaves:Vec<FoliatedCouplingLeafFfi>}
impl BiprismElectronicExchangeReadback {
    pub fn footers(&self)->[FoliatedCouplingFooterFfi;8] {std::array::from_fn(|i|self.leaves[i].footer)}
}
pub fn decode_electronic_exchange(bytes:&[u8])->Result<BiprismElectronicExchangeReadback,String>{
    if bytes.len()!=8*BIPRISM_ELECTRONIC_EXCHANGE_BYTES{return Err("electronic exchange extent differs from eight17792B banks".into());}
    let leaves=bytes.chunks_exact(BIPRISM_ELECTRONIC_EXCHANGE_BYTES).map(|b|{
        // All fields are numeric scalars/arrays; every bit pattern is a valid
        // Rust representation. Finite values and production identity are checked separately.
        unsafe {core::ptr::read_unaligned(b.as_ptr().cast::<FoliatedCouplingLeafFfi>())}
    }).collect();
    Ok(BiprismElectronicExchangeReadback{raw_bytes:bytes.to_vec(),leaves})
}
/// Validate the named final-substep publication against the independently read
/// outer commit. iteration is the publication sequence, not the Picard count.
/// Local physical transfer may be nonzero; this never imposes leaf-local energy conservation.
/// max_population_residual is finite/nonnegative diagnostic telemetry, not an
/// additional convergence gate. The actual equation/norm/coordinate gates remain explicit.
pub fn validate_electronic_exchange(
    records:&[FoliatedCouplingFooterFfi;8],committed:u32,mode:u32,limits:V10AcceptanceLimits
)->Result<(),String>{
    if committed<4||committed%4!=0||mode&!3!=0{return Err("invalid electronic exchange commit extent".into());}
    let first=records[0];
    for (leaf,f) in records.iter().enumerate(){
        let energies=[f.base_energy_delta,f.owned_pair_energy_delta,f.attributed_coupling_energy_delta,
            f.coupling_discrete_work,f.physical_transfer_in,f.balanced_local_defect,f.global_energy_delta,f.global_chain_defect];
        let residuals=[f.max_coordinate_residual,f.max_population_residual,f.max_electronic_equation_residual,f.max_norm_defect];
        if f.abi_magic!=BIPRISM_ELECTRONIC_EXCHANGE_MAGIC||f.leaf_id!=leaf as u32
            ||f.accepted_substeps_at_entry.checked_add(4)!=Some(committed)||f.substep!=3
            ||f.iteration==0||f.iteration!=first.iteration||f.publication_phase!=4
            ||f.local_fault_mask!=0||f.convergence_vote!=1||f.reserved!=[0;4]
            ||energies.iter().any(|x|!x.is_finite())||residuals.iter().any(|x|!x.is_finite()||*x<0.)
            ||f.max_coordinate_residual>limits.picard_residual||f.max_electronic_equation_residual>limits.picard_residual
            ||f.max_norm_defect>limits.unitarity_defect||f.global_chain_defect.abs()>f64::from(limits.absolute_energy_drift)
            ||(mode&2!=0&&f.global_energy_delta.abs()>f64::from(limits.absolute_energy_drift))
            ||f.global_energy_delta.to_bits()!=first.global_energy_delta.to_bits()
            ||f.global_chain_defect.to_bits()!=first.global_chain_defect.to_bits(){
            return Err(format!("electronic exchange leaf{leaf} has no matching accepted endpoint publication"));
        }
    }
    let tolerance=f64::from(limits.absolute_energy_drift);
    let energy:f64=records.iter().map(|f|f.base_energy_delta+f.owned_pair_energy_delta).sum();
    let chain:f64=records.iter().map(|f|f.owned_pair_energy_delta-f.coupling_discrete_work).sum();
    if !energy.is_finite()||!chain.is_finite()||chain.abs()>tolerance
        ||(mode&2!=0&&energy.abs()>tolerance)||(energy-first.global_energy_delta).abs()>tolerance
        ||(chain-first.global_chain_defect).abs()>tolerance
        ||records.iter().any(|f|(f.attributed_coupling_energy_delta-f.physical_transfer_in-f.coupling_discrete_work).abs()>tolerance
            ||(f.balanced_local_defect-f.base_energy_delta-f.coupling_discrete_work).abs()>tolerance){return Err("electronic exchange redundant ledger fields disagree".into());}
    let transfer:f64=records.iter().map(|f|f.physical_transfer_in).sum();
    if !transfer.is_finite()||transfer.abs()>f64::from(limits.absolute_energy_drift){return Err("global physical transfer closure refused".into());}
    Ok(())
}

#[repr(C, align(128))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BiprismConvergenceRecordFfi {
    pub abi_magic: u32,
    pub leaf_id: u32,
    pub epoch_leaf_fault: u32,
    pub epoch_cluster_fault: u32,
    pub terminal_fault_latch: u32,
    pub committed_substeps: u32,
    pub picard_iters_used: u32,
    pub dec_iters_used: u32,
    pub last_picard_residual: f32,
    pub last_dec_residual: f32,
    pub step_energy_drift: f32,
    pub w_xx_trace: f32,
    pub w_x_lambda_trace: f32,
    pub w_x_tau_trace: f32,
    pub unitarity_defect: f32,
    pub pulay_force_norm: f32,
    pub reserved: [u32; 16],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BiprismBwStatusFfi {
    pub magic: u32,
    pub leaf_id: u32,
    pub error: u32,
    pub phase: u32,
    pub inserted_vertices: u32,
    pub peak_tets: u32,
    pub unpeeled_tets: u32,
    pub unpeeled_edges: u32,
    pub final_tets: u32,
    pub final_edges: u32,
    pub peel_passes: u32,
    pub removed_tets: u32,
    pub uncertain_predicates: u32,
    pub super_attempt: u32,
    pub diagnostic_tet: u32,
    pub diagnostic_vertex: u32,
    pub published: u32,
    pub reserved: [u32; 3],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BiprismUnitBridgeStatusFfi {
    pub abi_magic: u32,
    pub leaf_id: u32,
    pub sticky_fault: u32,
    pub imported_step: u32,
    pub imported_fields: u32,
    pub export_ready_step: u32,
    pub export_ready_kind: u32,
    pub exported_fields: u32,
    pub import_calls: u32,
    pub export_calls: u32,
    pub atom_count: u32,
    pub last_core_committed: u32,
    pub reserved: [u32; 4],
}

const _: () = {
    assert!(core::mem::size_of::<FoliatedStreamPtrTableFfi>() == 640);
    assert!(core::mem::offset_of!(FoliatedStreamPtrTableFfi, true_inverse_masses) == 256);
    assert!(core::mem::offset_of!(FoliatedStreamPtrTableFfi, core_shake_mask) == 320);
    assert!(core::mem::offset_of!(FoliatedStreamPtrTableFfi,d_electronic_exchange)==384);
    assert!(core::mem::offset_of!(FoliatedStreamPtrTableFfi,d_numerical_scratch)==448);
    assert!(core::mem::offset_of!(FoliatedStreamPtrTableFfi,d_shared_step)==512);
    assert!(core::mem::offset_of!(FoliatedStreamPtrTableFfi,d_coordinate_owners)==528);
    assert!(core::mem::offset_of!(FoliatedStreamPtrTableFfi,d_ordered_edges)==536);
    assert!(core::mem::offset_of!(FoliatedStreamPtrTableFfi,d_potential_owners)==544);
    assert!(core::mem::offset_of!(FoliatedStreamPtrTableFfi,d_spinor_owners)==552);
    assert!(core::mem::offset_of!(FoliatedStreamPtrTableFfi,reserved_shared_tail)==560);
    assert!(core::mem::size_of::<BiprismPotentialOwnerFfi>()==128);
    assert!(core::mem::align_of::<BiprismPotentialOwnerFfi>()==128);
    assert!(core::mem::offset_of!(BiprismPotentialOwnerFfi,allocation_generation)==16);
    assert!(core::mem::offset_of!(BiprismPotentialOwnerFfi,accepted_phi)==48);
    assert!(core::mem::offset_of!(BiprismPotentialOwnerFfi,accepted_committed_substeps)==64);
    assert!(core::mem::offset_of!(BiprismPotentialOwnerFfi,accepted_operator_generation)==80);
    assert!(core::mem::offset_of!(BiprismPotentialOwnerFfi,projection_state)==96);
    assert!(core::mem::offset_of!(BiprismPotentialOwnerFfi,projection_operator_generation)==112);
    assert!(core::mem::size_of::<BiprismOrderedEdgesFfi>()==BIPRISM_ORDERED_EDGES_BYTES);
    assert!(core::mem::align_of::<BiprismOrderedEdgesFfi>()==128);
    assert!(core::mem::offset_of!(BiprismOrderedEdgesFfi,offsets)==64);
    assert!(core::mem::offset_of!(BiprismOrderedEdgesFfi,edge_index)==450);
    assert!(core::mem::size_of::<BiprismConvergenceRecordFfi>() == 128);
    assert!(core::mem::offset_of!(BiprismConvergenceRecordFfi, committed_substeps) == 20);
    assert!(core::mem::offset_of!(BiprismConvergenceRecordFfi, last_picard_residual) == 32);
    assert!(core::mem::offset_of!(BiprismConvergenceRecordFfi, reserved) == 64);
    assert!(core::mem::size_of::<BiprismBwStatusFfi>() == 80);
    assert!(core::mem::offset_of!(BiprismBwStatusFfi, published) == 64);
    assert!(BIPRISM_BW_SCRATCH_BYTES % 128 == 0);
    assert!(core::mem::size_of::<BiprismUnitBridgeStatusFfi>() == 64);
    assert!(core::mem::offset_of!(BiprismUnitBridgeStatusFfi, atom_count) == 40);
    assert!(core::mem::offset_of!(BiprismUnitBridgeStatusFfi, last_core_committed) == 44);
    assert!(core::mem::offset_of!(BiprismUnitBridgeStatusFfi, reserved) == 48);
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BiprismIdentity {
    pub topology_sha256: String,
    pub atom_map_sha256: String,
    /// Identity of the reviewed native-to-v10 quantity bridge; not a conversion.
    pub unit_bridge_sha256: String,
    /// Identity of the verified physical-mass derivation, including HMR history.
    pub physical_mass_sha256: String,
}
impl BiprismIdentity {
    pub fn validate(&self) -> Result<(), String> {
        for (label, digest) in [
            ("topology", &self.topology_sha256),
            ("atom map", &self.atom_map_sha256),
            ("unit bridge", &self.unit_bridge_sha256),
            ("physical masses", &self.physical_mass_sha256),
        ] {
            if digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(format!("invalid {label} source digest"));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceRegion {
    pub address: u64,
    pub bytes: usize,
}
impl DeviceRegion {
    fn checked_end(self) -> Result<u64, String> {
        if self.address == 0 || self.address % 4 != 0 || self.bytes == 0 {
            return Err("invalid or unaligned physical device region".into());
        }
        self.address
            .checked_add(u64::try_from(self.bytes).map_err(|_| "device extent overflow")?)
            .ok_or_else(|| "device address overflow".into())
    }
    fn overlaps(self, other: Self) -> Result<bool, String> {
        Ok(self.address < other.checked_end()? && other.address < self.checked_end()?)
    }
}

/// Retains the actual native allocation owner and stream. This is deliberately
/// not Clone. The caller must hold its physical launch-domain exclusion while
/// any captured kernel can touch these regions; Arc retention alone is not a lock.
pub struct BorrowedPhysicalLeaf {
    pub native_coordinates: DeviceRegion,
    pub native_velocities: DeviceRegion,
    pub native_forces: DeviceRegion,
    pub native_stream: Arc<CudaStream>,
    pub atom_count: u32,
    pub identity: BiprismIdentity,
    pub stream_id: u8,
    pub coordinate_owner: BiprismCoordinateOwnerFfi,
    _allocation_owner: Arc<dyn Any>,
}
impl BorrowedPhysicalLeaf {
    /// # Safety
    /// The allocation owner is deliberately thread-bound: the native engine retains
    /// thread-affine CUDA graph/capture state and must not gain Send/Sync here.
    /// `allocation_owner` must actually own every declared region. All regions
    /// must belong to `native_stream.context()`, cover atom_count*3 f32 values,
    /// remain stable and be exclusively scheduled through the physical owner.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new(
        allocation_owner: Arc<dyn Any>,
        native_stream: Arc<CudaStream>,
        atom_count: u32,
        identity: BiprismIdentity,
        stream_id: u8,
        native_coordinates: DeviceRegion,
        native_velocities: DeviceRegion,
        native_forces: DeviceRegion,
        coordinate_owner: BiprismCoordinateOwnerFfi,
    ) -> Result<Self, String> {
        identity.validate()?;
        let required = (atom_count as usize)
            .checked_mul(12)
            .ok_or("atom extent overflow")?;
        if !(256..=0x0100_0000).contains(&atom_count) {
            return Err("atom count is outside the packed 24-bit tile domain".into());
        }
        for region in [native_coordinates, native_velocities, native_forces] {
            region.checked_end()?;
            if region.bytes < required {
                return Err("physical plane is shorter than atom_count*3".into());
            }
        }
        coordinate_owner.validate(coordinate_owner.leaf_id, atom_count, NATIVE_ANGSTROM,
            native_coordinates.address)?;
        Ok(Self {
            native_coordinates,
            native_velocities,
            native_forces,
            native_stream,
            atom_count,
            identity,
            stream_id,
            coordinate_owner,
            _allocation_owner: allocation_owner,
        })
    }
}

pub fn validate_disjoint_regions(regions: &[DeviceRegion]) -> Result<(), String> {
    for (i, region) in regions.iter().enumerate() {
        region.checked_end()?;
        for prior in &regions[..i] {
            if region.overlaps(*prior)? {
                return Err(format!("physical device regions overlap at index {i}"));
            }
        }
    }
    Ok(())
}

/// One existing physical protocol clock, retained without changing its value.
/// The physical owner supplies the sole-writer and captured ordering contract.
pub struct BorrowedProtocolClock {
    region: DeviceRegion,
    context: Arc<CudaContext>,
    _allocation_owner: Arc<dyn Any + Send + Sync>,
}
impl BorrowedProtocolClock {
    /// # Safety
    /// The owner must own this stable, aligned u32 in the given CUDA context.
    /// It must outlive every captured replay and have exactly one authorized
    /// physical clock writer. An Arc keeps storage alive; it does not serialize it.
    pub unsafe fn new(
        allocation_owner: Arc<dyn Any + Send + Sync>,
        context: Arc<CudaContext>,
        region: DeviceRegion,
    ) -> Result<Self, String> {
        region.checked_end()?;
        if region.bytes != 4 {
            return Err("the protocol clock binding must be exactly one u32".into());
        }
        Ok(Self {
            region,
            context,
            _allocation_owner: allocation_owner,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum UnitExportKind {
    /// Accepted 192-core coordinates and velocities, before endpoint AMBER force.
    CoreState = 1,
    /// All selected 256 velocity deltas after native refresh and LIF delivery.
    DeliveredVelocity = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnitBridgeCheckpoint {
    FullImport,
    CoreExport,
    VelocityRefresh,
    DeliveredVelocityExport,
    ObservationRefresh,
}

#[derive(Clone, Debug)]
pub struct ConstraintGroup {
    pub atoms: Vec<u32>,
}

/// Values supplied by the verified mass/unit mapper. No mass reconstruction or
/// HMR reversal takes place here. Masks use 1 for the selected 192 core atoms.
pub struct LeafMassAndSelection {
    pub inverse_masses_au: Vec<f32>,
    pub true_inverse_masses_au: Vec<f32>,
    pub core_shake_mask: Vec<u32>,
    pub selected_atom_ids: [u32; BIPRISM_VERTICES],
}
pub fn validate_mass_and_selection(
    input: &LeafMassAndSelection,
    atom_count: u32,
    constraints: &[ConstraintGroup],
) -> Result<(), String> {
    let n = atom_count as usize;
    if input.inverse_masses_au.len() != n
        || input.true_inverse_masses_au.len() != n
        || input.core_shake_mask.len() != n
    {
        return Err("mass/mask array length does not equal the topology atom count".into());
    }
    if input
        .inverse_masses_au
        .iter()
        .chain(&input.true_inverse_masses_au)
        .any(|x| !x.is_finite() || *x <= 0.0)
    {
        return Err("inverse masses must be finite and strictly positive".into());
    }
    let mut selected = HashSet::with_capacity(256);
    for atom in input.selected_atom_ids {
        if atom >= atom_count || !selected.insert(atom) {
            return Err("selected atom IDs must be bounded and unique".into());
        }
    }
    let core: HashSet<u32> = input.selected_atom_ids[..192].iter().copied().collect();
    for (atom, &mask) in input.core_shake_mask.iter().enumerate() {
        if mask != u32::from(core.contains(&(atom as u32))) {
            return Err("core mask does not equal the first 192 selected atoms".into());
        }
    }
    for group in constraints {
        let mut unique = HashSet::new();
        if group.atoms.is_empty()
            || group
                .atoms
                .iter()
                .any(|&a| a >= atom_count || !unique.insert(a))
        {
            return Err("constraint group is empty, duplicated or outside the atom map".into());
        }
        if group.atoms.iter().any(|a| core.contains(a))
            && !group.atoms.iter().all(|a| core.contains(a))
        {
            return Err("selected core splits a native constraint group".into());
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct GovernedModuleIdentity {
    pub member: String,
    pub image_sha256: String,
}

pub struct BiprismV10Functions {
    _module: Arc<CudaModule>,
    context: Arc<CudaContext>,
    pub identity: GovernedModuleIdentity,
    v10: CudaFunction,
    phase2: CudaFunction,
    bowyer_watson: CudaFunction,
    import_native_state: CudaFunction,
    guard_unit_import: CudaFunction,
    validate_unit_export: CudaFunction,
    commit_unit_export: CudaFunction,
}
impl BiprismV10Functions {
    /// # Safety
    /// The existing governed module loader must supply the actual admitted image
    /// and its owning context. Metadata strings alone do not attest loaded bytes.
    pub unsafe fn from_governed_module(
        module: Arc<CudaModule>,
        context: Arc<CudaContext>,
        identity: GovernedModuleIdentity,
    ) -> Result<Self, String> {
        if identity.member.trim().is_empty()
            || identity.image_sha256.len() != 64
            || !identity.image_sha256.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err("loaded governed module identity is required".into());
        }
        context.bind_to_thread().map_err(|e| e.to_string())?;
        let v10 = module
            .load_function("nhs_biprism_variational_so3g_v10")
            .map_err(|e| e.to_string())?;
        let phase2 = module
            .load_function("nhs_biprism_simplicial_tile_compensated_v1")
            .map_err(|e| e.to_string())?;
        let bowyer_watson = module
            .load_function("nhs_seed_bowyer_watson_256_compensated_v1")
            .map_err(|e| e.to_string())?;
        let import_native_state = module
            .load_function("nhs_biprism_import_native_state_v1")
            .map_err(|e| e.to_string())?;
        let guard_unit_import = module
            .load_function("nhs_biprism_guard_unit_import_v1")
            .map_err(|e| e.to_string())?;
        let validate_unit_export = module
            .load_function("nhs_biprism_validate_unit_export_v1")
            .map_err(|e| e.to_string())?;
        let commit_unit_export = module
            .load_function("nhs_biprism_commit_unit_export_v1")
            .map_err(|e| e.to_string())?;
        Ok(Self {
            _module: module,
            context,
            identity,
            v10,
            phase2,
            bowyer_watson,
            import_native_state,
            guard_unit_import,
            validate_unit_export,
            commit_unit_export,
        })
    }
}

#[derive(Clone, Debug)]
pub struct KernelResources {
    pub symbol: &'static str,
    pub registers_per_thread: i32,
    pub static_shared_bytes: i32,
    pub local_bytes_per_thread: i32,
    pub constant_bytes: i32,
    pub max_threads_per_block: i32,
    pub binary_version: i32,
    pub ptx_version: i32,
    pub active_blocks_per_sm: u32,
}
#[derive(Clone, Debug)]
pub struct BiprismResourceReport {
    pub cluster_launch_supported: bool,
    pub required_cluster: [i32; 3],
    pub cluster_size_must_be_set: i32,
    pub potential_cluster_size: u32,
    pub active_clusters: u32,
    pub shared_memory_per_block_optin: i32,
    pub v10: KernelResources,
    pub phase2: KernelResources,
    pub bowyer_watson: KernelResources,
    pub import_native_state: KernelResources,
    pub guard_unit_import: KernelResources,
    pub validate_unit_export: KernelResources,
    pub commit_unit_export: KernelResources,
}
fn kernel_resources(
    f: &CudaFunction,
    symbol: &'static str,
    dynamic: u32,
) -> Result<KernelResources, String> {
    let r = KernelResources {
        symbol,
        registers_per_thread: f.num_regs().map_err(|e| e.to_string())?,
        static_shared_bytes: f.shared_size_bytes().map_err(|e| e.to_string())?,
        local_bytes_per_thread: f.local_size_bytes().map_err(|e| e.to_string())?,
        constant_bytes: f.const_size_bytes().map_err(|e| e.to_string())?,
        max_threads_per_block: f.max_threads_per_block().map_err(|e| e.to_string())?,
        binary_version: f.binary_version().map_err(|e| e.to_string())?,
        ptx_version: f.ptx_version().map_err(|e| e.to_string())?,
        active_blocks_per_sm: f
            .occupancy_max_active_blocks_per_multiprocessor(256, dynamic as usize, None)
            .map_err(|e| e.to_string())?,
    };
    if r.max_threads_per_block < 256 || r.active_blocks_per_sm == 0 {
        return Err(format!(
            "{symbol} cannot launch its required 256-thread CTA"
        ));
    }
    Ok(r)
}
fn require_not_capturing(stream: &CudaStream) -> Result<(), String> {
    let mut status = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
    let code = unsafe { sys::cuStreamIsCapturing(stream.cu_stream(), &mut status) };
    if code != sys::CUresult::CUDA_SUCCESS {
        return Err(format!("query capture status: {code:?}"));
    }
    if status != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
        return Err(
            "allocation, preparation readback and resource queries require an uncaptured stream"
                .into(),
        );
    }
    Ok(())
}
impl BiprismV10Functions {
    pub fn inspect_resources(&self, stream: &CudaStream) -> Result<BiprismResourceReport, String> {
        require_not_capturing(stream)?;
        if stream.context().cu_ctx() != self.context.cu_ctx() {
            return Err("kernel and stream contexts differ".into());
        }
        self.context.bind_to_thread().map_err(|e| e.to_string())?;
        let supports = self
            .context
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH)
            .map_err(|e| e.to_string())?;
        if supports != 1 {
            return Err("v10 requires native eight-CTA cluster launch support".into());
        }
        use sys::CUfunction_attribute as A;
        let attr = |a| self.v10.get_attribute(a).map_err(|e| e.to_string());
        let required_cluster = [
            attr(A::CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH)?,
            attr(A::CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT)?,
            attr(A::CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH)?,
        ];
        if required_cluster != [8, 1, 1] {
            return Err(format!(
                "loaded v10 cluster ABI is {required_cluster:?}, expected [8,1,1]"
            ));
        }
        let cluster_size_must_be_set = attr(A::CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET)?;
        let max_shared = self
            .context
            .attribute(
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            )
            .map_err(|e| e.to_string())?;
        let static_shared = self.v10.shared_size_bytes().map_err(|e| e.to_string())?;
        if static_shared < 0
            || max_shared < 0
            || static_shared as u64 + BIPRISM_DYNAMIC_SMEM_BYTES as u64 > max_shared as u64
        {
            return Err("v10 static plus dynamic shared memory exceeds the device limit".into());
        }
        self.v10
            .set_attribute(
                A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                BIPRISM_DYNAMIC_SMEM_BYTES as i32,
            )
            .map_err(|e| e.to_string())?;
        let cfg = v10_config();
        let potential_cluster_size = self
            .v10
            .occupancy_max_potential_cluster_size(cfg, stream)
            .map_err(|e| e.to_string())?;
        let active_clusters = self
            .v10
            .occupancy_max_active_clusters(cfg, stream)
            .map_err(|e| e.to_string())?;
        if potential_cluster_size < 8 || active_clusters == 0 {
            return Err("device cannot schedule the required v10 cluster".into());
        }
        Ok(BiprismResourceReport {
            cluster_launch_supported: true,
            required_cluster,
            cluster_size_must_be_set,
            potential_cluster_size,
            active_clusters,
            shared_memory_per_block_optin: max_shared,
            v10: kernel_resources(
                &self.v10,
                "nhs_biprism_variational_so3g_v10",
                BIPRISM_DYNAMIC_SMEM_BYTES,
            )?,
            phase2: kernel_resources(&self.phase2, "nhs_biprism_simplicial_tile_compensated_v1", 0)?,
            bowyer_watson: kernel_resources(
                &self.bowyer_watson,
                "nhs_seed_bowyer_watson_256_compensated_v1",
                0,
            )?,
            import_native_state: kernel_resources(
                &self.import_native_state,
                "nhs_biprism_import_native_state_v1",
                0,
            )?,
            guard_unit_import: kernel_resources(
                &self.guard_unit_import,
                "nhs_biprism_guard_unit_import_v1",
                0,
            )?,
            validate_unit_export: kernel_resources(
                &self.validate_unit_export,
                "nhs_biprism_validate_unit_export_v1",
                0,
            )?,
            commit_unit_export: kernel_resources(
                &self.commit_unit_export,
                "nhs_biprism_commit_unit_export_v1",
                0,
            )?,
        })
    }
}
fn v10_config() -> LaunchConfig {
    LaunchConfig {
        grid_dim: (8, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: BIPRISM_DYNAMIC_SMEM_BYTES,
    }
}
fn pointer<T>(slice: &CudaSlice<T>, stream: &Arc<CudaStream>) -> u64 {
    slice.device_ptr(stream).0
}
fn completed_dtoh<T: DeviceRepr + Default + Clone>(
    stream: &Arc<CudaStream>, slice: &CudaSlice<T>,
) -> Result<Vec<T>, String> {
    require_not_capturing(stream)?;
    let mut values=vec![T::default();slice.len()];
    let copy=stream.memcpy_dtoh(slice,&mut values).map_err(|e|e.to_string());
    let drain=stream.synchronize().map_err(|e|e.to_string());
    if let Err(e)=copy{return Err(format!("device readback failed: {e}; drain={drain:?}"));}
    drain?;Ok(values)
}
fn completed_htod<T:DeviceRepr>(stream:&Arc<CudaStream>,values:&[T])->Result<CudaSlice<T>,String>{
    require_not_capturing(stream)?;
    let mut result=unsafe{stream.alloc(values.len())}.map_err(|e|e.to_string())?;
    let copy=stream.memcpy_htod(values,&mut result).map_err(|e|e.to_string());
    let drain=stream.synchronize().map_err(|e|e.to_string());
    if let Err(e)=copy{return Err(format!("device upload failed: {e}; drain={drain:?}"));}
    drain?;Ok(result)
}

pub fn decode_convergence(words: &[u32]) -> Result<[BiprismConvergenceRecordFfi; 8], String> {
    if words.len() != 8 * 32 {
        return Err("convergence readback must contain all eight 128-byte records".into());
    }
    Ok(std::array::from_fn(|leaf| {
        let w = &words[leaf * 32..(leaf + 1) * 32];
        BiprismConvergenceRecordFfi {
            abi_magic: w[0],
            leaf_id: w[1],
            epoch_leaf_fault: w[2],
            epoch_cluster_fault: w[3],
            terminal_fault_latch: w[4],
            committed_substeps: w[5],
            picard_iters_used: w[6],
            dec_iters_used: w[7],
            last_picard_residual: f32::from_bits(w[8]),
            last_dec_residual: f32::from_bits(w[9]),
            step_energy_drift: f32::from_bits(w[10]),
            w_xx_trace: f32::from_bits(w[11]),
            w_x_lambda_trace: f32::from_bits(w[12]),
            w_x_tau_trace: f32::from_bits(w[13]),
            unitarity_defect: f32::from_bits(w[14]),
            pulay_force_norm: f32::from_bits(w[15]),
            reserved: std::array::from_fn(|i| w[16 + i]),
        }
    }))
}

/// Explicit numerical limits belong to the actual physical acceptance owner.
#[derive(Clone, Copy, Debug)]
pub struct V10AcceptanceLimits {
    pub picard_residual: f32,
    pub dec_residual: f32,
    pub absolute_energy_drift: f32,
    pub unitarity_defect: f32,
}
pub fn validate_convergence(
    records: &[BiprismConvergenceRecordFfi; 8],
    previous_committed: u32,
    successful_outer_steps: u32,
    limits: V10AcceptanceLimits,
) -> Result<u32, String> {
    let bounds = [
        limits.picard_residual,
        limits.dec_residual,
        limits.absolute_energy_drift,
        limits.unitarity_defect,
    ];
    if bounds.iter().any(|x| !x.is_finite() || *x < 0.0)
        || successful_outer_steps == 0
        || previous_committed % 4 != 0
    {
        return Err("invalid convergence acceptance extent or limits".into());
    }
    let expected = successful_outer_steps
        .checked_mul(4)
        .and_then(|x| previous_committed.checked_add(x))
        .ok_or("convergence counter overflow")?;
    for (leaf, r) in records.iter().enumerate() {
        let telemetry = [
            r.last_picard_residual,
            r.last_dec_residual,
            r.step_energy_drift,
            r.w_xx_trace,
            r.w_x_lambda_trace,
            r.w_x_tau_trace,
            r.unitarity_defect,
            r.pulay_force_norm,
        ];
        if r.abi_magic != BIPRISM_CONVERGENCE_MAGIC
            || r.leaf_id != leaf as u32
            || r.epoch_leaf_fault | r.epoch_cluster_fault | r.terminal_fault_latch != 0
            || r.committed_substeps != expected
            || telemetry.iter().any(|x| !x.is_finite())
            || r.picard_iters_used == 0
            || r.picard_iters_used > BIPRISM_MAX_NUCLEAR_EVALUATIONS
            || r.dec_iters_used > 192
            || r.last_picard_residual < 0.0
            || r.last_dec_residual < 0.0
            || r.unitarity_defect < 0.0
            || r.last_picard_residual > limits.picard_residual
            || r.last_dec_residual > limits.dec_residual
            || r.step_energy_drift.abs() > limits.absolute_energy_drift
            || r.unitarity_defect > limits.unitarity_defect
        {
            return Err(format!(
                "v10 leaf {leaf} failed the supplied acceptance contract"
            ));
        }
    }
    Ok(expected)
}

/// These are different allocations from the native AMBER planes. Parent-owned
/// bridge kernels must import/export them with the reviewed physical conversion.
pub struct AtomicUnitWorkingPlanes {
    coordinates: [CudaSlice<f32>; 8],
    coordinate_low: [CudaSlice<f32>; 8],
    step_start_high: [CudaSlice<f32>; 8],
    step_start_low: [CudaSlice<f32>; 8],
    reference_high: [CudaSlice<f32>; 8],
    reference_low: [CudaSlice<f32>; 8],
    velocities: [CudaSlice<f32>; 8],
    forces: [CudaSlice<f32>; 8],
}
#[derive(Clone, Copy, Debug)]
pub struct BiprismDeviceViews {
    pub potential_owners:u64,
    pub coordinate_owners: u64,
    pub native_coordinate_owners: u64,
    pub stream_table: u64,
    pub native_stream_table: u64,
    pub unit_bridge_status: u64,
    pub coordinates_by_leaf: u64,
    pub tiles: u64,
    pub convergence: u64,
    pub adjudicator: u64,
    pub selected_atom_ids: u64,
    pub charges: u64,
    pub element_classes: u64,
    pub atom_residues: u64,
    pub core_masks: [u64; 8],
    pub inverse_masses_au: [u64; 8],
    pub true_inverse_masses_au: [u64; 8],
    pub working_coordinates: [u64; 8],
    pub working_velocities: [u64; 8],
    pub working_forces: [u64; 8],
    pub electronic_exchange:[u64;8],
    pub numerical_scratch:[u64;8],
    pub ordered_edges:u64,
}

#[derive(Clone, Copy, Debug)]
pub struct V10Step {
    pub dt_outer_au: f32,
    pub kappa_foliated: f32,
    pub integration_mode: u32,
}
impl V10Step {
    pub fn validate(self) -> Result<(), String> {
        if !self.dt_outer_au.is_finite()
            || self.dt_outer_au <= 0.0
            || !self.kappa_foliated.is_finite()
            || self.integration_mode & !3 != 0
        {
            return Err(
                "invalid v10 step arguments; timestep must already be in the reviewed v10 units"
                    .into(),
            );
        }
        Ok(())
    }
}

/// Native geometry gate input. The private allocation/generation fields bind it
/// to the particular BW enqueue that produced these immutable readback bytes.
pub struct BiprismPreparationReadback {
    statuses: [BiprismBwStatusFfi; 8],
    tile_bytes: Vec<u8>,
    selected_positions_au: [[[f32; 3]; 256]; 8],
    selected_positions_au_low: [[[f32;3];256];8],
    target_tets: u32,
    allocation: Arc<()>,
    generation: u64,
}
impl BiprismPreparationReadback {
    pub fn statuses(&self) -> &[BiprismBwStatusFfi; 8] {
        &self.statuses
    }
    pub fn tile_bytes(&self) -> &[u8] {
        &self.tile_bytes
    }
    pub fn selected_positions_au(&self) -> &[[[f32; 3]; 256]; 8] {
        &self.selected_positions_au
    }
    pub fn selected_positions_au_low(&self)->&[[[f32;3];256];8]{&self.selected_positions_au_low}
    pub fn represented_selected_positions_au(&self)->[[[f64;3];256];8]{
        std::array::from_fn(|l|std::array::from_fn(|v|std::array::from_fn(|d|
            f64::from(self.selected_positions_au[l][v][d])+f64::from(self.selected_positions_au_low[l][v][d]))))
    }
    pub fn target_tets(&self) -> u32 {
        self.target_tets
    }
}

pub struct BiprismV10DeviceState {
    stream: Arc<CudaStream>,
    functions: BiprismV10Functions,
    pub resources: BiprismResourceReport,
    leaves: [BorrowedPhysicalLeaf; 8],
    identity: BiprismIdentity,
    atom_count: u32,
    working: AtomicUnitWorkingPlanes,
    table: CudaSlice<FoliatedStreamPtrTableFfi>,
    native_table: CudaSlice<FoliatedStreamPtrTableFfi>,
    coordinate_owners: CudaSlice<BiprismCoordinateOwnerFfi>,
    native_coordinate_owners: CudaSlice<BiprismCoordinateOwnerFfi>,
    potential_owners:CudaSlice<BiprismPotentialOwnerFfi>,
    spinor_owners:CudaSlice<BiprismSpinorOwnerFfi>,
    accepted_potential:[CudaSlice<f64>;8],
    prepared_potential:[CudaSlice<f64>;8],
    // Eight distinct retained allocations. Kernels overwrite derived scratch per epoch.
    electronic_exchange:[CudaSlice<u64>;8],
    numerical_scratch:[CudaSlice<u64>;8],
    ordered_edges:CudaSlice<u64>,
    peer_edge_map:Option<[[u16;1536];8]>,
    bridge_status: CudaSlice<u32>,
    protocol_clock: Option<BorrowedProtocolClock>,
    shared_step_owner:Option<Arc<CudaSlice<crate::biprism_shared_step::BiprismSharedStepFfi>>>,
    coords_by_leaf: CudaSlice<u64>,
    inverse_masses_au: [CudaSlice<f32>; 8],
    true_inverse_masses_au: [CudaSlice<f32>; 8],
    core_masks: [CudaSlice<u32>; 8],
    charges: CudaSlice<f32>,
    elements: CudaSlice<u8>,
    residues: CudaSlice<i32>,
    selected: CudaSlice<u32>,
    selected_host: [[u32; 256]; 8],
    elements_host: Vec<u8>,
    tiles: CudaSlice<u8>,
    convergence: CudaSlice<u32>,
    adjudicator: CudaSlice<u32>,
    barrier_arrivals: CudaSlice<u32>,
    barrier_epoch: CudaSlice<u32>,
    bw_scratch: CudaSlice<u32>,
    bw_status: CudaSlice<u32>,
    lif: BiprismLifDeviceState,
    prepared: bool,
    phase2_enqueued: bool,
    full_import_enqueued: bool,
    preparation_allocation: Arc<()>,
    preparation_generation: u64,
}

impl BiprismV10DeviceState {
    #[allow(clippy::too_many_arguments)]
    pub fn allocate(
        stream: Arc<CudaStream>,
        functions: BiprismV10Functions,
        leaves: [BorrowedPhysicalLeaf; 8],
        mass_and_selection: [LeafMassAndSelection; 8],
        charges: &[f32],
        elements: &[u8],
        residues: &[i32],
        constraints: &[ConstraintGroup],
        lif_capacity: u32,
        export_identity: String,
    ) -> Result<Self, String> {
        require_not_capturing(&stream)?;
        let n = leaves[0].atom_count;
        let identity = leaves[0].identity.clone();
        let mut regions = Vec::with_capacity(24);
        let mut stream_ids = HashSet::new();
        for (leaf_index, (leaf, mass)) in leaves.iter().zip(&mass_and_selection).enumerate() {
            if leaf.atom_count != n
                || leaf.identity != identity
                || leaf.native_stream.context().cu_ctx() != stream.context().cu_ctx()
                || !stream_ids.insert(leaf.stream_id)
            {
                return Err("all eight physical leaves must have compatible identities/context and unique stream IDs".into());
            }
            regions.extend([
                leaf.native_coordinates,
                leaf.native_velocities,
                leaf.native_forces,
            ]);
            leaf.coordinate_owner.validate(leaf_index as u32,n,NATIVE_ANGSTROM,leaf.native_coordinates.address)?;
            for address in [leaf.coordinate_owner.current_lo,leaf.coordinate_owner.step_start_hi,leaf.coordinate_owner.step_start_lo,leaf.coordinate_owner.reference_hi,leaf.coordinate_owner.reference_lo] {
                regions.push(DeviceRegion{address,bytes:n as usize*12});
            }
            validate_mass_and_selection(mass, n, constraints)?;
        }
        validate_disjoint_regions(&regions)?;
        if charges.len() != n as usize
            || elements.len() != n as usize
            || residues.len() != n as usize
            || charges.iter().any(|x| !x.is_finite())
            || elements.iter().any(|&e| e >= 8)
            || residues.iter().any(|&r| r < -1)
        {
            return Err(
                "shared charge/element/residue planes do not match the atom contract".into(),
            );
        }
        let resources = functions.inspect_resources(&stream)?;
        let n3 = (n as usize)
            .checked_mul(3)
            .ok_or("working-plane extent overflow")?;
        let alloc_f32 = || -> Result<[CudaSlice<f32>; 8], String> {
            let values = (0..8)
                .map(|_| stream.alloc_zeros(n3).map_err(|e| e.to_string()))
                .collect::<Result<Vec<_>, _>>()?;
            values
                .try_into()
                .map_err(|_| "working-plane leaf extent".into())
        };
        let working = AtomicUnitWorkingPlanes {
            coordinates: alloc_f32()?,
            coordinate_low: alloc_f32()?,
            step_start_high: alloc_f32()?,
            step_start_low: alloc_f32()?,
            reference_high: alloc_f32()?,
            reference_low: alloc_f32()?,
            velocities: alloc_f32()?,
            forces: alloc_f32()?,
        };
        let inverse_masses_au: [CudaSlice<f32>; 8] = mass_and_selection
            .iter()
            .map(|m| {
                completed_htod(&stream,&m.inverse_masses_au)
            })
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| "inverse-mass leaf extent")?;
        let true_inverse_masses_au: [CudaSlice<f32>; 8] = mass_and_selection
            .iter()
            .map(|m| {
                completed_htod(&stream,&m.true_inverse_masses_au)
            })
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| "physical inverse-mass leaf extent")?;
        let core_masks: [CudaSlice<u32>; 8] = mass_and_selection
            .iter()
            .map(|m| {
                completed_htod(&stream,&m.core_shake_mask)
            })
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| "mask leaf extent")?;
        let electronic_exchange:[CudaSlice<u64>;8]=(0..8).map(|_|stream.alloc_zeros(BIPRISM_ELECTRONIC_EXCHANGE_BYTES/8).map_err(|e|e.to_string())).collect::<Result<Vec<_>,_>>()?.try_into().map_err(|_|"electronic exchange leaf extent")?;
        let numerical_scratch:[CudaSlice<u64>;8]=(0..8).map(|_|stream.alloc_zeros(BIPRISM_NUMERICAL_SCRATCH_BYTES/8).map_err(|e|e.to_string())).collect::<Result<Vec<_>,_>>()?.try_into().map_err(|_|"numerical scratch leaf extent")?;
        // One array of eight align128 records, exclusively written by the main
        // CTA for that leaf. Allocation remains alive throughout graph replay.
        let ordered_edges:CudaSlice<u64>=stream.alloc_zeros(
            BIPRISM_LEAVES*BIPRISM_ORDERED_EDGES_BYTES/8).map_err(|e|e.to_string())?;
        if pointer(&ordered_edges,&stream)%128!=0{return Err("ordered edge allocation alignment".into());}
        let coordinate_values:[BiprismCoordinateOwnerFfi;8]=std::array::from_fn(|i| BiprismCoordinateOwnerFfi {
            magic:COORDINATE_MAGIC,version:COORDINATE_VERSION,leaf_id:i as u32,atom_count:n,
            generation:leaves[i].coordinate_owner.generation,unit_frame:ATOMIC_BOHR,initialized_mask:7,
            current_hi:pointer(&working.coordinates[i],&stream),current_lo:pointer(&working.coordinate_low[i],&stream),
            step_start_hi:pointer(&working.step_start_high[i],&stream),step_start_lo:pointer(&working.step_start_low[i],&stream),
            reference_hi:pointer(&working.reference_high[i],&stream),reference_lo:pointer(&working.reference_low[i],&stream),reserved:[0;2]});
        for (i,o) in coordinate_values.iter().enumerate(){o.validate(i as u32,n,ATOMIC_BOHR,o.current_hi)?;}
        let native_coordinate_values:[BiprismCoordinateOwnerFfi;8]=std::array::from_fn(|i|leaves[i].coordinate_owner);
        let coordinate_owners=completed_htod(&stream,&coordinate_values)?;
        let native_coordinate_owners=completed_htod(&stream,&native_coordinate_values)?;
        let tiles:CudaSlice<u8>=stream.alloc_zeros(8*BIPRISM_TILE_BYTES).map_err(|e|e.to_string())?;
        let alloc_phi=||->Result<[CudaSlice<f64>;8],String>{(0..8).map(|_|stream.alloc_zeros(256).map_err(|e|e.to_string()))
            .collect::<Result<Vec<_>,_>>()?.try_into().map_err(|_|"potential bank population".into())};
        let accepted_potential=alloc_phi()?;let prepared_potential=alloc_phi()?;
        let potential_values:[BiprismPotentialOwnerFfi;8]=std::array::from_fn(|i|BiprismPotentialOwnerFfi{
            abi_magic:BIPRISM_POTENTIAL_MAGIC,version:1,leaf_id:i as u32,
            allocation_generation:coordinate_values[i].generation,topology_generation:0,
            tile_address:pointer(&tiles,&stream)+(i*BIPRISM_TILE_BYTES) as u64,
            coordinate_owner_address:pointer(&coordinate_owners,&stream)+(i*96) as u64,
            accepted_phi:pointer(&accepted_potential[i],&stream),prepared_phi:pointer(&prepared_potential[i],&stream),
            ..Default::default()});
        let potential_owners=completed_htod(&stream,&potential_values)?;
        let spinor_values:Vec<BiprismSpinorOwnerFfi>=potential_values.iter().enumerate().map(|(i,p)|BiprismSpinorOwnerFfi{
            magic:BIPRISM_SPINOR_MAGIC,version:1,leaf:i as u32,allocation_generation:p.allocation_generation,
            topology_generation:p.topology_generation,tile_address:p.tile_address,coordinate_owner_address:p.coordinate_owner_address,
            ..Default::default()}).collect();
        let spinor_owners=completed_htod(&stream,&spinor_values)?;
        if pointer(&potential_owners,&stream)%128!=0{return Err("potential owner alignment".into());}
        let table_value = FoliatedStreamPtrTableFfi {
            coords: std::array::from_fn(|i| pointer(&working.coordinates[i], &stream)),
            velocities: std::array::from_fn(|i| pointer(&working.velocities[i], &stream)),
            forces: std::array::from_fn(|i| pointer(&working.forces[i], &stream)),
            inverse_masses: std::array::from_fn(|i| pointer(&inverse_masses_au[i], &stream)),
            true_inverse_masses: std::array::from_fn(|i| {
                pointer(&true_inverse_masses_au[i], &stream)
            }),
            core_shake_mask: std::array::from_fn(|i| pointer(&core_masks[i], &stream)),
            d_electronic_exchange:std::array::from_fn(|i|pointer(&electronic_exchange[i],&stream)),
            d_numerical_scratch:std::array::from_fn(|i|pointer(&numerical_scratch[i],&stream)),
            d_shared_step:0,physical_step_mode:4,reserved_coordinate_alignment:0,
            d_coordinate_owners:pointer(&coordinate_owners,&stream),
            d_ordered_edges:pointer(&ordered_edges,&stream),d_potential_owners:pointer(&potential_owners,&stream),d_spinor_owners:pointer(&spinor_owners,&stream),reserved_shared_tail:[0;20],
        };
        let selected_host = std::array::from_fn(|i| mass_and_selection[i].selected_atom_ids);
        let selected_flat = selected_host.iter().flatten().copied().collect::<Vec<_>>();
        let table_host = [table_value];
        let table = completed_htod(&stream,&table_host)?;
        let native_table_host = [FoliatedStreamPtrTableFfi {
            coords: std::array::from_fn(|i| leaves[i].native_coordinates.address),
            velocities: std::array::from_fn(|i| leaves[i].native_velocities.address),
            forces: std::array::from_fn(|i| leaves[i].native_forces.address),
            inverse_masses: [0; 8],
            true_inverse_masses: [0; 8],
            core_shake_mask: [0; 8],
            d_electronic_exchange:[0;8],d_numerical_scratch:[0;8],d_shared_step:0,physical_step_mode:4,reserved_coordinate_alignment:0,
            d_coordinate_owners:pointer(&native_coordinate_owners,&stream),
            d_ordered_edges:0,d_potential_owners:0,d_spinor_owners:0,reserved_shared_tail:[0;20],
        }];
        let native_table = completed_htod(&stream,&native_table_host)?;
        let bridge_status = stream.alloc_zeros(8 * 16).map_err(|e| e.to_string())?;
        let coords_by_leaf = completed_htod(&stream,&table_value.coords)?;
        let selected = completed_htod(&stream,&selected_flat)?;
        let charges_device = completed_htod(&stream,charges)?;
        let elements_device = completed_htod(&stream,elements)?;
        let residues_device = completed_htod(&stream,residues)?;
        // Cold identity only: no fault, numerical result or committed step is
        // fabricated by initialization. The selected main consumes this ABI.
        let mut convergence_host=[0u32;8*32];
        for leaf in 0..8 {convergence_host[leaf*32]=BIPRISM_CONVERGENCE_MAGIC;
            convergence_host[leaf*32+1]=leaf as u32;}
        let convergence=completed_htod(&stream,&convergence_host)?;
        let adjudicator = stream.alloc_zeros(8 * 64).map_err(|e| e.to_string())?;
        let barrier_arrivals = stream.alloc_zeros(1).map_err(|e| e.to_string())?;
        let barrier_epoch = stream.alloc_zeros(1).map_err(|e| e.to_string())?;
        let bw_scratch = stream
            .alloc_zeros(8 * BIPRISM_BW_SCRATCH_BYTES / 4)
            .map_err(|e| e.to_string())?;
        let bw_status = stream
            .alloc_zeros(8 * BIPRISM_BW_STATUS_BYTES / 4)
            .map_err(|e| e.to_string())?;
        let lif = BiprismLifDeviceState::allocate(
            stream.clone(),
            lif_capacity,
            std::array::from_fn(|i| leaves[i].stream_id),
            export_identity,
        )?;
        // Retain every pageable source above until the upload stream completes.
        stream.synchronize().map_err(|e| e.to_string())?;
        Ok(Self {
            stream,
            functions,
            resources,
            leaves,
            identity,
            atom_count: n,
            working,
            table,
            native_table,
            coordinate_owners,native_coordinate_owners,
            potential_owners,spinor_owners,accepted_potential,prepared_potential,
            electronic_exchange,
            numerical_scratch,ordered_edges,peer_edge_map:None,
            bridge_status,
            protocol_clock: None,
            shared_step_owner:None,
            coords_by_leaf,
            inverse_masses_au,
            true_inverse_masses_au,
            core_masks,
            charges: charges_device,
            elements: elements_device,
            residues: residues_device,
            selected,
            selected_host,
            elements_host: elements.to_vec(),
            tiles,
            convergence,
            adjudicator,
            barrier_arrivals,
            barrier_epoch,
            bw_scratch,
            bw_status,
            lif,
            prepared: false,
            phase2_enqueued: false,
            full_import_enqueued: false,
            preparation_allocation: Arc::new(()),
            preparation_generation: 0,
        })
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
    pub fn atom_count(&self) -> u32 {
        self.atom_count
    }
    pub fn identity(&self) -> &BiprismIdentity {
        &self.identity
    }
    pub fn native_leaves(&self) -> &[BorrowedPhysicalLeaf; 8] {
        &self.leaves
    }
    /// Bind the existing protocol owner once, before any bridge enqueue. Replacing
    /// this pointer would invalidate captured nodes and their epoch provenance.
    pub fn bind_protocol_clock(&mut self, clock: BorrowedProtocolClock) -> Result<(), String> {
        require_not_capturing(&self.stream)?;
        if self.protocol_clock.is_some() || self.full_import_enqueued || self.prepared {
            return Err("protocol clock is already bound or preparation has started".into());
        }
        if clock.context.cu_ctx() != self.stream.context().cu_ctx() {
            return Err("protocol clock and v10 state contexts differ".into());
        }
        for leaf in &self.leaves {
            for plane in [
                leaf.native_coordinates,
                leaf.native_velocities,
                leaf.native_forces,
            ] {
                if clock.region.overlaps(plane)? {
                    return Err("protocol clock overlaps a native physical plane".into());
                }
            }
        }
        self.protocol_clock = Some(clock);
        Ok(())
    }
    /// Cold binding of a retained shared control allocation. Native bridge tables
    /// retain a null shared owner; only actual AU propagation consumes it.
    pub(crate) fn bind_shared_physical_step(&mut self,record:Arc<CudaSlice<crate::biprism_shared_step::BiprismSharedStepFfi>>)->Result<(),String>{
        require_not_capturing(&self.stream)?;
        if self.shared_step_owner.is_some()||self.full_import_enqueued||self.prepared||record.len()!=1{return Err("shared physical owner binding is not cold/single-use".into());}
        let address=pointer(&*record,&self.stream);if address==0||address%128!=0{return Err("invalid shared control allocation".into());}
        let mut table=completed_dtoh(&self.stream,&self.table)?;
        if table.len()!=1||table[0].d_shared_step!=0||table[0].physical_step_mode!=4||table[0].reserved_shared_tail!=[0;20]||table[0].d_spinor_owners!=pointer(&self.spinor_owners,&self.stream)||table[0].d_ordered_edges!=pointer(&self.ordered_edges,&self.stream)||table[0].d_potential_owners!=pointer(&self.potential_owners,&self.stream)||table[0].reserved_coordinate_alignment!=0||table[0].d_coordinate_owners!=pointer(&self.coordinate_owners,&self.stream){return Err("shared table preimage mismatch".into());}
        table[0].d_shared_step=address;table[0].physical_step_mode=0;
        let queued=self.stream.memcpy_htod(&table,&mut self.table).map_err(|e|e.to_string());
        let drained=self.stream.synchronize().map_err(|e|e.to_string());queued?;drained?;
        self.shared_step_owner=Some(record);Ok(())
    }
    pub fn protocol_step_pointer(&self) -> Result<u64, String> {
        self.protocol_clock
            .as_ref()
            .map(|clock| clock.region.address)
            .ok_or_else(|| "a retained physical protocol clock must be bound before capture".into())
    }
    /// Immutable preparation metadata. All active edges map bijectively by
    /// ordered physical endpoints; inactive slots contain0xffff, never phase0.
    pub fn peer_edge_maps(&self)->Option<&[[u16;1536];8]>{if self.prepared{self.peer_edge_map.as_ref()}else{None}}
    pub fn peer_edge_map_generation(&self)->Option<u64>{self.peer_edge_maps().map(|_|self.preparation_generation)}
    pub fn selected_atom_ids(&self) -> &[[u32; 256]; 8] {
        &self.selected_host
    }
    pub fn lif_mut(&mut self) -> &mut BiprismLifDeviceState {
        &mut self.lif
    }
    fn cold_potential_owner(&self,leaf:usize,topology:u64)->BiprismPotentialOwnerFfi {
        BiprismPotentialOwnerFfi{abi_magic:BIPRISM_POTENTIAL_MAGIC,version:1,leaf_id:leaf as u32,
            allocation_generation:self.leaves[leaf].coordinate_owner.generation,topology_generation:topology,
            tile_address:pointer(&self.tiles,&self.stream)+(leaf*BIPRISM_TILE_BYTES) as u64,
            coordinate_owner_address:pointer(&self.coordinate_owners,&self.stream)+(leaf*96) as u64,
            accepted_phi:pointer(&self.accepted_potential[leaf],&self.stream),
            prepared_phi:pointer(&self.prepared_potential[leaf],&self.stream),..Default::default()}
    }
    /// Reads fixed retained allocations even after refusal. Descriptor pointers
    /// are recorded as evidence, never followed to download arbitrary memory.
    pub fn read_spinor_state(&self)->Result<Vec<BiprismSpinorOwnerFfi>,String>{
        require_not_capturing(&self.stream)?;completed_dtoh(&self.stream,&self.spinor_owners)
    }
    pub fn read_potential_state(&self)->Result<BiprismPotentialReadback,String>{
        require_not_capturing(&self.stream)?;
        let owners=completed_dtoh(&self.stream,&self.potential_owners)?;
        let mut accepted=Vec::with_capacity(2048);let mut prepared=Vec::with_capacity(2048);let mut working=Vec::with_capacity(2048);
        for leaf in 0..8 {
            accepted.extend(completed_dtoh(&self.stream,&self.accepted_potential[leaf])?);
            prepared.extend(completed_dtoh(&self.stream,&self.prepared_potential[leaf])?);
            let scratch=completed_dtoh(&self.stream,&self.numerical_scratch[leaf])?;
            if scratch.len()!=BIPRISM_NUMERICAL_SCRATCH_BYTES/8{return Err("numerical potential scratch extent".into());}
            working.extend(scratch[BIPRISM_WORKING_PHI_OFFSET/8..BIPRISM_WORKING_PHI_OFFSET/8+256].iter().map(|v|f64::from_bits(*v)));
        }
        Ok(BiprismPotentialReadback{owners,accepted,prepared,working})
    }
    pub fn validate_potential_endpoint(&self,read:&BiprismPotentialReadback,committed:u32,step:u32)->Result<(),String>{
        let profile=if self.shared_step_owner.is_some(){PotentialEndpointProfile::PreparedProduction}
            else{PotentialEndpointProfile::ScalarDiagnostic};
        validate_accepted_potential(read,&self.read_tiles()?,committed,step,self.preparation_generation,profile)?;
        let spinors=self.read_spinor_state()?;let tiles=self.read_tiles()?;
        if spinors.len()!=8{return Err("spinor owner population".into());}
        for (leaf,o) in spinors.iter().enumerate(){let p=&read.owners[leaf];
            if o.magic!=BIPRISM_SPINOR_MAGIC||o.version!=1||o.leaf!=leaf as u32||o.state!=1
                ||o.allocation_generation!=p.allocation_generation||o.topology_generation!=p.topology_generation
                ||o.tile_address!=p.tile_address||o.coordinate_owner_address!=p.coordinate_owner_address
                ||o.accepted_committed_substeps!=committed||o.accepted_step!=step||o.reserved!=[0;9]{return Err("spinor accepted identity mismatch".into());}
            for (u,psi) in o.accepted.iter().enumerate(){let norm=psi[0].mul_add(psi[0],psi[1]*psi[1])+psi[2].mul_add(psi[2],psi[3]*psi[3]);
                if psi.iter().any(|x|!x.is_finite())||(norm-1.).abs()>1e-7{return Err("spinor accepted norm/nonfinite".into());}
                for (j,x) in psi.iter().enumerate(){let off=leaf*BIPRISM_TILE_BYTES+90112+u*16+j*4;
                    if (*x as f32).to_bits()!=u32::from_le_bytes(tiles[off..off+4].try_into().unwrap()){return Err("spinor compatibility projection changed".into());}}
            }
        }
        for (leaf,o) in read.owners.iter().enumerate(){let expected=self.cold_potential_owner(leaf,self.preparation_generation);
            if o.allocation_generation!=expected.allocation_generation||o.tile_address!=expected.tile_address
                ||o.coordinate_owner_address!=expected.coordinate_owner_address||o.accepted_phi!=expected.accepted_phi
                ||o.prepared_phi!=expected.prepared_phi{return Err(format!("potential leaf{leaf} allocation identity changed"));}
        }
        Ok(())
    }
    /// Borrowed addresses remain owned by self; captured users must retain self.
    pub fn device_views(&self) -> BiprismDeviceViews {
        BiprismDeviceViews {
            potential_owners:pointer(&self.potential_owners,&self.stream),
            coordinate_owners: pointer(&self.coordinate_owners,&self.stream),
            native_coordinate_owners: pointer(&self.native_coordinate_owners,&self.stream),
            stream_table: pointer(&self.table, &self.stream),
            native_stream_table: pointer(&self.native_table, &self.stream),
            unit_bridge_status: pointer(&self.bridge_status, &self.stream),
            coordinates_by_leaf: pointer(&self.coords_by_leaf, &self.stream),
            tiles: pointer(&self.tiles, &self.stream),
            convergence: pointer(&self.convergence, &self.stream),
            adjudicator: pointer(&self.adjudicator, &self.stream),
            selected_atom_ids: pointer(&self.selected, &self.stream),
            charges: pointer(&self.charges, &self.stream),
            element_classes: pointer(&self.elements, &self.stream),
            atom_residues: pointer(&self.residues, &self.stream),
            core_masks: std::array::from_fn(|i| pointer(&self.core_masks[i], &self.stream)),
            inverse_masses_au: std::array::from_fn(|i| {
                pointer(&self.inverse_masses_au[i], &self.stream)
            }),
            true_inverse_masses_au: std::array::from_fn(|i| {
                pointer(&self.true_inverse_masses_au[i], &self.stream)
            }),
            working_coordinates: std::array::from_fn(|i| {
                pointer(&self.working.coordinates[i], &self.stream)
            }),
            working_velocities: std::array::from_fn(|i| {
                pointer(&self.working.velocities[i], &self.stream)
            }),
            working_forces: std::array::from_fn(|i| pointer(&self.working.forces[i], &self.stream)),
            electronic_exchange:std::array::from_fn(|i|pointer(&self.electronic_exchange[i],&self.stream)),
            numerical_scratch:std::array::from_fn(|i|pointer(&self.numerical_scratch[i],&self.stream)),
            ordered_edges:pointer(&self.ordered_edges,&self.stream),
        }
    }

    pub fn read_coordinate_snapshots(&self)->Result<Vec<crate::biprism_coordinates::BiprismCoordinateSnapshot>,String>{
        require_not_capturing(&self.stream)?;
        let owners=completed_dtoh(&self.stream,&self.coordinate_owners)?;
        let mut images=Vec::with_capacity(8);
        for leaf in 0..8 {
            owners[leaf].validate(leaf as u32,self.atom_count(),ATOMIC_BOHR,pointer(&self.working.coordinates[leaf],&self.stream))?;
            let image=crate::biprism_coordinates::BiprismCoordinateSnapshot{identity:owners[leaf].into(),
                current_high:completed_dtoh(&self.stream,&self.working.coordinates[leaf])?,
                current_low:completed_dtoh(&self.stream,&self.working.coordinate_low[leaf])?,
                step_start_high:completed_dtoh(&self.stream,&self.working.step_start_high[leaf])?,
                step_start_low:completed_dtoh(&self.stream,&self.working.step_start_low[leaf])?,
                reference_high:completed_dtoh(&self.stream,&self.working.reference_high[leaf])?,
                reference_low:completed_dtoh(&self.stream,&self.working.reference_low[leaf])?};
            image.validate()?;images.push(image);
        }
        Ok(images)
    }

    /// Import all native coordinates, velocities and forces, then run the
    /// all-eight-leaf guard before frozen v10 can observe the working planes.
    /// Fresh allocation accepts the existing clock value; no reset is performed.
    /// # Safety
    /// The physical owner must order all native producers before this stream,
    /// serialize the one protocol clock writer, and retain this owner through
    /// replay. Pointer-table writes are indirect and require that exclusivity.
    pub unsafe fn capture_import_native_state(&mut self) -> Result<(), String> {
        let protocol_step = self.protocol_step_pointer()?;
        let fields = 7u32;
        self.full_import_enqueued = false;
        self.stream
            .launch_builder(&self.functions.import_native_state)
            .arg(&self.native_table)
            .arg(&self.table)
            .arg(&mut self.bridge_status)
            .arg(&protocol_step)
            .arg(&self.atom_count)
            .arg(&fields)
            .launch(LaunchConfig {
                grid_dim: (8, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|e| format!("native full-state import: {e}"))?;
        self.stream
            .launch_builder(&self.functions.guard_unit_import)
            .arg(&self.bridge_status)
            .arg(&mut self.convergence)
            .arg(&protocol_step)
            .launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|e| format!("collective unit-import guard: {e}"))?;
        self.full_import_enqueued = true;
        Ok(())
    }

    /// Refresh the current full-N native velocity baseline after accepted core
    /// export and before the separate LIF delivery. The ordered collective guard
    /// derives marker15 from core export and stamps all leaves on a failed refresh.
    /// # Safety
    /// Order the current physical velocity producer before this stream and LIF
    /// delivery after it, without concurrent native/AU writes or clock changes.
    pub unsafe fn capture_refresh_native_velocities(&mut self) -> Result<(), String> {
        if !self.prepared {
            return Err("velocity refresh requires prepared physical state".into());
        }
        let protocol_step = self.protocol_step_pointer()?;
        let fields = 2u32;
        self.full_import_enqueued = false;
        self.stream
            .launch_builder(&self.functions.import_native_state)
            .arg(&self.native_table)
            .arg(&self.table)
            .arg(&mut self.bridge_status)
            .arg(&protocol_step)
            .arg(&self.atom_count)
            .arg(&fields)
            .launch(LaunchConfig {
                grid_dim: (8, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|e| format!("native velocity refresh: {e}"))?;
        self.stream
            .launch_builder(&self.functions.guard_unit_import)
            .arg(&self.bridge_status)
            .arg(&mut self.convergence)
            .arg(&protocol_step)
            .launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|e| format!("collective velocity-refresh guard: {e}"))?;
        Ok(())
    }

    /// Refresh all native positions and velocities after physical post-step
    /// owners (including RATTLE), before LIF observes core and halo together.
    /// The bridge requires both prior exports and publishes marker31.
    /// # Safety
    /// All post-step coordinate/velocity writes must precede this stream. LIF
    /// must follow it and use this same retained protocol clock and allocation.
    pub unsafe fn capture_refresh_observation_state(&mut self) -> Result<(), String> {
        if !self.prepared {
            return Err("observation refresh requires prepared physical state".into());
        }
        let protocol_step = self.protocol_step_pointer()?;
        let fields = 3u32;
        self.full_import_enqueued = false;
        self.stream
            .launch_builder(&self.functions.import_native_state)
            .arg(&self.native_table)
            .arg(&self.table)
            .arg(&mut self.bridge_status)
            .arg(&protocol_step)
            .arg(&self.atom_count)
            .arg(&fields)
            .launch(LaunchConfig {
                grid_dim: (8, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|e| format!("post-step observation state refresh: {e}"))?;
        self.stream
            .launch_builder(&self.functions.guard_unit_import)
            .arg(&self.bridge_status)
            .arg(&mut self.convergence)
            .arg(&protocol_step)
            .launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|e| format!("collective observation-refresh guard: {e}"))?;
        Ok(())
    }

    /// Validate all leaves before any native write, then enqueue the exact
    /// collective commit. Device status enforces epoch, counter and duplicate
    /// export rules. Enqueue success is not proof that this export was accepted.
    /// # Safety
    /// CoreState belongs after accepted v10 and before endpoint AMBER force.
    /// DeliveredVelocity belongs after current native-V refresh and successful
    /// LIF delivery, before RATTLE. The physical owner must preserve that order,
    /// exclusively schedule all planes, and retain this state through replay.
    pub unsafe fn capture_export_native_state(
        &mut self,
        kind: UnitExportKind,
    ) -> Result<(), String> {
        if !self.prepared {
            return Err("native export requires prepared physical state".into());
        }
        let protocol_step = self.protocol_step_pointer()?;
        let kind = kind as u32;
        self.stream
            .launch_builder(&self.functions.validate_unit_export)
            .arg(&self.native_table)
            .arg(&self.table)
            .arg(&self.tiles)
            .arg(&self.convergence)
            .arg(&mut self.bridge_status)
            .arg(&protocol_step)
            .arg(&self.atom_count)
            .arg(&kind)
            .launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|e| format!("collective unit-export validation: {e}"))?;
        self.stream
            .launch_builder(&self.functions.commit_unit_export)
            .arg(&self.native_table)
            .arg(&self.table)
            .arg(&self.tiles)
            .arg(&self.convergence)
            .arg(&mut self.bridge_status)
            .arg(&protocol_step)
            .arg(&self.atom_count)
            .arg(&kind)
            .launch(LaunchConfig {
                grid_dim: (8, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|e| format!("collective unit-export commit: {e}"))?;
        Ok(())
    }

    /// Execute the actual native BW producer and retain every status, including
    /// failures. A return value is a readback, not permission to propagate.
    /// # Safety
    /// All AU coordinate planes must have been imported by the reviewed bridge,
    /// with all native writers ordered before this exclusive preparation stream.
    pub unsafe fn prepare_bowyer_watson(
        &mut self,
        target_tets: u32,
    ) -> Result<BiprismPreparationReadback, String> {
        require_not_capturing(&self.stream)?;
        if target_tets == 0 || target_tets > 1024 {
            return Err("BW target must fit the frozen 1024-tet capacity".into());
        }
        if !self.full_import_enqueued {
            return Err("BW preparation requires the native full-import bridge".into());
        }
        let potential=completed_dtoh(&self.stream,&self.potential_owners)?;
        if potential.len()!=8||potential.iter().enumerate().any(|(leaf,o)|*o!=self.cold_potential_owner(leaf,o.topology_generation)){
            return Err("reseed would discard initialized potential authority".into());
        }
        let imported = self.read_unit_bridge_status()?;
        validate_unit_bridge_status(
            &imported,
            self.atom_count,
            imported[0].imported_step,
            imported[0].last_core_committed,
            UnitBridgeCheckpoint::FullImport,
        )?;
        let generation = self
            .preparation_generation
            .checked_add(1)
            .ok_or("preparation generation exhausted")?;
        self.prepared = false;
        self.peer_edge_map = None;
        self.phase2_enqueued = false;
        self.preparation_generation = generation;
        let leaves = 8u32;
        self.stream
            .launch_builder(&self.functions.bowyer_watson)
            .arg(&self.coords_by_leaf)
            .arg(&self.charges)
            .arg(&self.elements)
            .arg(&self.selected)
            .arg(&self.atom_count)
            .arg(&mut self.tiles)
            .arg(&mut self.bw_scratch)
            .arg(&mut self.bw_status)
            .arg(&leaves)
            .arg(&target_tets)
            .arg(&self.coordinate_owners)
            .launch(LaunchConfig {
                grid_dim: (8, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|e| format!("BW preparation launch: {e}"))?;
        let statuses = self.read_bw_status()?;
        let tile_bytes = self.read_tiles()?;
        let mut selected_positions_au = [[[0.0; 3]; 256]; 8];
        let mut selected_positions_au_low = [[[0.0;3];256];8];
        for leaf in 0..8 {
            let xyz = completed_dtoh(&self.stream, &self.working.coordinates[leaf])?;
            let low = completed_dtoh(&self.stream,&self.working.coordinate_low[leaf])?;
            let reference = completed_dtoh(&self.stream,&self.working.reference_high[leaf])?;
            let reference_low = completed_dtoh(&self.stream,&self.working.reference_low[leaf])?;
            for (v, &a) in self.selected_host[leaf].iter().enumerate() {
                for d in 0..3 { let i=3*a as usize+d;
                    if xyz[i].to_bits()!=reference[i].to_bits() || low[i].to_bits()!=reference_low[i].to_bits() {
                        return Err(format!("cold mesh current/reference split differs at leaf{leaf}/atom{a}/axis{d}"));
                    }
                    selected_positions_au[leaf][v][d]=xyz[i];selected_positions_au_low[leaf][v][d]=low[i];
                }
            }
        }
        Ok(BiprismPreparationReadback {
            statuses,
            tile_bytes,
            selected_positions_au,
            selected_positions_au_low,
            target_tets,
            allocation: self.preparation_allocation.clone(),
            generation,
        })
    }

    /// The supplied callback must run the physical owner's complete native mesh
    /// acceptance on these exact statuses, coordinates and tile bytes. Mechanical
    /// publication and identity checks alone do not establish mesh validity.
    pub fn accept_preparation<F>(
        &mut self,
        readback: BiprismPreparationReadback,
        native_mesh_validator: F,
    ) -> Result<(), String>
    where
        F: FnOnce(&BiprismPreparationReadback) -> Result<(), String>,
    {
        require_not_capturing(&self.stream)?;
        if !Arc::ptr_eq(&readback.allocation, &self.preparation_allocation)
            || readback.generation != self.preparation_generation
            || self.prepared
        {
            return Err("stale, foreign or already accepted tile preparation receipt".into());
        }
        validate_bw_status(&readback.statuses, readback.target_tets)?;
        validate_tile_identity(
            &readback.tile_bytes,
            &self.selected_host,
            &self.elements_host,
        )?;
        if readback
            .selected_positions_au
            .iter().chain(readback.selected_positions_au_low.iter())
            .flatten()
            .flatten()
            .any(|x| !x.is_finite())
        {
            return Err("prepared AU coordinates contain a nonfinite value".into());
        }
        native_mesh_validator(&readback)?;
        let mut potential=completed_dtoh(&self.stream,&self.potential_owners)?;
        if potential.len()!=8||potential.iter().enumerate().any(|(leaf,o)|*o!=self.cold_potential_owner(leaf,o.topology_generation)){
            return Err("potential preparation is not cold or allocation identity changed".into());
        }
        let maps=build_peer_edge_bijections(&readback.tile_bytes)?;
        for leaf in 0..8 {
            let destination=pointer(&self.numerical_scratch[leaf],&self.stream)+BIPRISM_PEER_EDGE_MAP_OFFSET as u64;
            let rc=unsafe{sys::cuMemcpyHtoDAsync_v2(destination,maps[leaf].as_ptr().cast(),3072,self.stream.cu_stream())};
            // Complete this row before another fallible launch or source-buffer
            // scope exit. The immutable metadata is published only after all8.
            let completed=self.stream.synchronize().map_err(|e|e.to_string());
            if rc!=sys::CUresult::CUDA_SUCCESS{return Err(format!("peer edge map upload leaf{leaf}: {rc:?}; completion={completed:?}"));}
            completed?;
        }
        let mut spinors=completed_dtoh(&self.stream,&self.spinor_owners)?;
        for (leaf,o) in spinors.iter_mut().enumerate(){let p=&potential[leaf];
            let expected=BiprismSpinorOwnerFfi{magic:BIPRISM_SPINOR_MAGIC,version:1,leaf:leaf as u32,
                allocation_generation:p.allocation_generation,topology_generation:p.topology_generation,
                tile_address:p.tile_address,coordinate_owner_address:p.coordinate_owner_address,..Default::default()};
            if *o!=expected{return Err("spinor cold allocation/preparation identity changed".into());}
            o.topology_generation=readback.generation;
        }
        let queued=self.stream.memcpy_htod(&spinors,&mut self.spinor_owners).map_err(|e|e.to_string());
        let drained=self.stream.synchronize().map_err(|e|e.to_string());queued?;drained?;
        for o in &mut potential{o.topology_generation=readback.generation;}
        let queued=self.stream.memcpy_htod(&potential,&mut self.potential_owners).map_err(|e|e.to_string());
        let drained=self.stream.synchronize().map_err(|e|e.to_string());queued?;drained?;
        self.peer_edge_map=Some(maps);
        self.prepared = true;
        Ok(())
    }

    /// # Safety
    /// Call only after the BW/native geometry gate and the actual quantity bridge
    /// have established each leaf's working coordinates. Exclusive state ownership
    /// and cross-stream dependencies are the enclosing physical owner's contract.
    pub unsafe fn capture_phase2_per_leaf(&mut self) -> Result<(), String> {
        if !self.prepared {
            return Err("Phase2 requires accepted native tile preparation".into());
        }
        self.phase2_enqueued = false;
        for leaf in 0u32..8 {
            self.stream
                .launch_builder(&self.functions.phase2)
                .arg(&self.working.coordinates[leaf as usize])
                .arg(&self.charges)
                .arg(&mut self.tiles)
                .arg(&leaf)
                .arg(&self.coordinate_owners)
                .arg(&self.convergence)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| format!("Phase2 leaf {leaf}: {e}"))?;
        }
        self.phase2_enqueued = true;
        Ok(())
    }

    /// # Safety
    /// The actual captured physical owner must import AU working planes, order
    /// exclusive core propagation, synchronize all eight leaves, and retain this
    /// owner through replay. This method neither advances the protocol clock nor
    /// performs AMBER integration, unit conversion, RATTLE or whole-step rollback.
    pub unsafe fn capture_v10(&mut self, step: V10Step) -> Result<(), String> {
        step.validate()?;
        if !self.prepared || !self.phase2_enqueued || !self.full_import_enqueued {
            return Err("v10 requires accepted tile preparation, all eight Phase2 launches and a full unit import".into());
        }
        self.stream
            .launch_builder(&self.functions.v10)
            .arg(&self.table)
            .arg(&self.charges)
            .arg(&mut self.tiles)
            .arg(&mut self.adjudicator)
            .arg(&mut self.convergence)
            .arg(&mut self.barrier_arrivals)
            .arg(&mut self.barrier_epoch)
            .arg(&step.dt_outer_au)
            .arg(&step.kappa_foliated)
            .arg(&step.integration_mode)
            .launch(v10_config())
            .map_err(|e| format!("v10 eight-leaf cluster launch: {e}"))?;
        Ok(())
    }

    /// Unfiltered diagnostics remain available after a fault; no counter/fault
    /// reset is offered that could erase a rejected trajectory's terminal latch.
    /// Raw banks remain readable on refusal and are never reset to manufacture a valid footer.
    pub fn read_electronic_exchange(&self)->Result<BiprismElectronicExchangeReadback,String>{
        require_not_capturing(&self.stream)?;
        let mut bytes=Vec::with_capacity(8*BIPRISM_ELECTRONIC_EXCHANGE_BYTES);
        for bank in &self.electronic_exchange {
            let words=completed_dtoh(&self.stream,bank)?;
            self.stream.synchronize().map_err(|e|e.to_string())?;
            for word in words{bytes.extend_from_slice(&word.to_ne_bytes());}
        }
        decode_electronic_exchange(&bytes)
    }
    pub fn read_convergence(&self) -> Result<[BiprismConvergenceRecordFfi; 8], String> {
        decode_convergence(&completed_dtoh(&self.stream, &self.convergence)?)
    }
    pub fn read_tiles(&self) -> Result<Vec<u8>, String> {
        completed_dtoh(&self.stream, &self.tiles)
    }
    pub fn read_adjudicator_words(&self) -> Result<Vec<u32>, String> {
        completed_dtoh(&self.stream, &self.adjudicator)
    }
    pub fn read_bw_status(&self) -> Result<[BiprismBwStatusFfi; 8], String> {
        decode_bw_status(&completed_dtoh(&self.stream, &self.bw_status)?)
    }
    pub fn read_unit_bridge_words(&self) -> Result<Vec<u32>, String> {
        completed_dtoh(&self.stream, &self.bridge_status)
    }
    pub fn read_unit_bridge_status(&self) -> Result<[BiprismUnitBridgeStatusFfi; 8], String> {
        decode_unit_bridge_status(&self.read_unit_bridge_words()?)
    }
}

pub fn decode_unit_bridge_status(words: &[u32]) -> Result<[BiprismUnitBridgeStatusFfi; 8], String> {
    if words.len() != 8 * 16 {
        return Err("unit bridge status must contain all eight 64-byte records".into());
    }
    Ok(std::array::from_fn(|leaf| {
        let w = &words[leaf * 16..leaf * 16 + 16];
        BiprismUnitBridgeStatusFfi {
            abi_magic: w[0],
            leaf_id: w[1],
            sticky_fault: w[2],
            imported_step: w[3],
            imported_fields: w[4],
            export_ready_step: w[5],
            export_ready_kind: w[6],
            exported_fields: w[7],
            import_calls: w[8],
            export_calls: w[9],
            atom_count: w[10],
            last_core_committed: w[11],
            reserved: [w[12], w[13], w[14], w[15]],
        }
    }))
}

/// Validate a completed readback at a named bridge boundary. This checks the
/// exact published protocol/commit counters; it does not validate physical
/// trajectory accuracy or replace the convergence and mesh acceptance gates.
pub fn validate_unit_bridge_status(
    statuses: &[BiprismUnitBridgeStatusFfi; 8],
    atom_count: u32,
    protocol_step: u32,
    committed_substeps: u32,
    checkpoint: UnitBridgeCheckpoint,
) -> Result<(), String> {
    if !(256..=0x0100_0000).contains(&atom_count) || committed_substeps % 4 != 0 {
        return Err("invalid unit bridge atom or convergence extent".into());
    }
    let (fields, exports, ready) = match checkpoint {
        UnitBridgeCheckpoint::FullImport => (7, 0, 0),
        UnitBridgeCheckpoint::CoreExport => (7, 1, 1),
        UnitBridgeCheckpoint::VelocityRefresh => (15, 1, 1),
        UnitBridgeCheckpoint::DeliveredVelocityExport => (15, 3, 2),
        UnitBridgeCheckpoint::ObservationRefresh => (31, 3, 2),
    };
    for (leaf, s) in statuses.iter().enumerate() {
        if s.abi_magic != BIPRISM_UNIT_BRIDGE_MAGIC
            || s.leaf_id != leaf as u32
            || s.sticky_fault != 0
            || s.atom_count != atom_count
            || s.imported_step != protocol_step
            || s.last_core_committed != committed_substeps
            || s.imported_fields != fields
            || s.exported_fields != exports
            || s.export_ready_kind != ready
            || (ready != 0 && s.export_ready_step != protocol_step)
            || s.import_calls == 0
            || (exports != 0 && s.export_calls == 0)
            || s.import_calls != statuses[0].import_calls
            || s.export_calls != statuses[0].export_calls
            || s.reserved != [0; 4]
        {
            return Err(format!(
                "unit bridge leaf {leaf} failed checkpoint {checkpoint:?}: {s:?}"
            ));
        }
    }
    Ok(())
}

pub fn decode_bw_status(words: &[u32]) -> Result<[BiprismBwStatusFfi; 8], String> {
    if words.len() != 8 * 20 {
        return Err("BW status must contain all eight 80-byte records".into());
    }
    Ok(std::array::from_fn(|leaf| {
        let w = &words[leaf * 20..leaf * 20 + 20];
        BiprismBwStatusFfi {
            magic: w[0],
            leaf_id: w[1],
            error: w[2],
            phase: w[3],
            inserted_vertices: w[4],
            peak_tets: w[5],
            unpeeled_tets: w[6],
            unpeeled_edges: w[7],
            final_tets: w[8],
            final_edges: w[9],
            peel_passes: w[10],
            removed_tets: w[11],
            uncertain_predicates: w[12],
            super_attempt: w[13],
            diagnostic_tet: w[14],
            diagnostic_vertex: w[15],
            published: w[16],
            reserved: [w[17], w[18], w[19]],
        }
    }))
}
pub fn validate_bw_status(
    statuses: &[BiprismBwStatusFfi; 8],
    target_tets: u32,
) -> Result<(), String> {
    if target_tets == 0 || target_tets > 1024 {
        return Err("invalid BW target extent".into());
    }
    for (leaf, s) in statuses.iter().enumerate() {
        if s.magic != BIPRISM_BW_MAGIC
            || s.leaf_id != leaf as u32
            || s.error != 0
            || s.phase != 5
            || s.inserted_vertices != 256
            || s.published != 1
            || s.final_tets == 0
            || s.final_tets > target_tets
            || s.final_edges == 0
            || s.final_edges > 1536
            || s.reserved != [0; 3]
        {
            return Err(format!(
                "BW leaf {leaf} did not publish complete accepted output: {s:?}"
            ));
        }
    }
    Ok(())
}

/// Mechanical readback checks precede the caller's stronger native geometry
/// validation. They deliberately do not label face/edge incidence as Delaunay.
pub fn validate_tile_identity(
    bytes: &[u8],
    selected: &[[u32; 256]; 8],
    elements: &[u8],
) -> Result<(), String> {
    if bytes.len() != 8 * BIPRISM_TILE_BYTES {
        return Err("tile readback byte extent mismatch".into());
    }
    for leaf in 0..8 {
        let tile = &bytes[leaf * BIPRISM_TILE_BYTES..(leaf + 1) * BIPRISM_TILE_BYTES];
        let word = |off| u32::from_le_bytes(tile[off..off + 4].try_into().unwrap());
        for vertex in 0..256 {
            let atom = selected[leaf][vertex] as usize;
            if atom >= elements.len()
                || word(4 * vertex) != (atom as u32 | ((elements[atom] as u32) << 24))
            {
                return Err(format!(
                    "tile {leaf} vertex {vertex} lost its selected atom/class identity"
                ));
            }
        }
        let tets = word(TILE_COUNT_OFFSET);
        let edges = word(TILE_COUNT_OFFSET + 4);
        if tets == 0 || tets > 1024 || edges == 0 || edges > 1536 {
            return Err(format!(
                "tile {leaf} active counts are empty or exceed the frozen ABI"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn selection() -> LeafMassAndSelection {
        LeafMassAndSelection {
            inverse_masses_au: vec![1.0; 256],
            true_inverse_masses_au: vec![1.0; 256],
            core_shake_mask: (0..256).map(|i| u32::from(i < 192)).collect(),
            selected_atom_ids: std::array::from_fn(|i| i as u32),
        }
    }
    #[test]
    fn biprism_v10_rejects_alias_and_overflow() {
        assert!(validate_disjoint_regions(&[
            DeviceRegion {
                address: 0x1000,
                bytes: 0x100
            },
            DeviceRegion {
                address: 0x1100,
                bytes: 0x100
            }
        ])
        .is_ok());
        assert!(validate_disjoint_regions(&[
            DeviceRegion {
                address: 0x1000,
                bytes: 0x100
            },
            DeviceRegion {
                address: 0x10fc,
                bytes: 4
            }
        ])
        .is_err());
        assert!(validate_disjoint_regions(&[DeviceRegion {
            address: u64::MAX - 3,
            bytes: 8
        }])
        .is_err());
    }
    #[test]
    fn biprism_v10_selection_requires_constraint_closure() {
        let mut s = selection();
        assert!(
            validate_mass_and_selection(&s, 256, &[ConstraintGroup { atoms: vec![0, 1] }]).is_ok()
        );
        assert!(validate_mass_and_selection(
            &s,
            256,
            &[ConstraintGroup {
                atoms: vec![191, 192]
            }]
        )
        .is_err());
        s.selected_atom_ids[255] = 0;
        assert!(validate_mass_and_selection(&s, 256, &[]).is_err());
        s = selection();
        s.true_inverse_masses_au[0] = f32::NAN;
        assert!(validate_mass_and_selection(&s, 256, &[]).is_err());
    }
    #[test]
    fn biprism_v10_readback_requires_full_equal_epoch_and_finite_values() {
        let mut words = vec![0u32; 256];
        for leaf in 0..8 {
            let w = &mut words[leaf * 32..(leaf + 1) * 32];
            w[0] = BIPRISM_CONVERGENCE_MAGIC;
            w[1] = leaf as u32;
            w[5] = 8;
            w[6] = 2;
        }
        let limits = V10AcceptanceLimits {
            picard_residual: 1e-7,
            dec_residual: 1e-6,
            absolute_energy_drift: 1e-5,
            unitarity_defect: 1e-5,
        };
        let r = decode_convergence(&words).unwrap();
        assert_eq!(validate_convergence(&r, 4, 1, limits).unwrap(), 8);
        words[6] = BIPRISM_MAX_NUCLEAR_EVALUATIONS;
        assert!(validate_convergence(&decode_convergence(&words).unwrap(), 4, 1, limits).is_ok());
        words[6] = BIPRISM_MAX_NUCLEAR_EVALUATIONS+1;
        assert!(validate_convergence(&decode_convergence(&words).unwrap(), 4, 1, limits).is_err());
        words[6] = 2;
        words[7] = 192;
        assert!(validate_convergence(&decode_convergence(&words).unwrap(), 4, 1, limits).is_ok());
        words[7] = 193;
        assert!(validate_convergence(&decode_convergence(&words).unwrap(), 4, 1, limits).is_err());
        words[7] = 192;
        words[9] = (1.01e-6f32).to_bits();
        assert!(validate_convergence(&decode_convergence(&words).unwrap(), 4, 1, limits).is_err());
        words[7] = 0;
        words[9] = 0;
        words[3 * 32 + 4] = 2;
        assert!(validate_convergence(&decode_convergence(&words).unwrap(), 4, 1, limits).is_err());
        words[3 * 32 + 4] = 0;
        words[7 * 32 + 5] = 4;
        assert!(validate_convergence(&decode_convergence(&words).unwrap(), 4, 1, limits).is_err());
        words[7 * 32 + 5] = 8;
        words[10] = f32::NAN.to_bits();
        assert!(validate_convergence(&decode_convergence(&words).unwrap(), 4, 1, limits).is_err());
        assert!(decode_convergence(&words[..255]).is_err());
    }
    #[test]
    fn biprism_v10_step_rejects_unbounded_parameters() {
        assert!(V10Step {
            dt_outer_au: 1.0,
            kappa_foliated: 0.0,
            integration_mode: 1
        }
        .validate()
        .is_ok());
        assert!(V10Step {
            dt_outer_au: 0.0,
            kappa_foliated: 0.0,
            integration_mode: 1
        }
        .validate()
        .is_err());
        assert!(V10Step {
            dt_outer_au: 1.0,
            kappa_foliated: 0.0,
            integration_mode: 4
        }
        .validate()
        .is_err());
    }
    #[test]
    fn biprism_v10_bw_refuses_one_leaf_partial_publication() {
        let mut w = vec![0; 160];
        for leaf in 0..8 {
            let s = &mut w[leaf * 20..leaf * 20 + 20];
            s[0] = BIPRISM_BW_MAGIC;
            s[1] = leaf as u32;
            s[3] = 5;
            s[4] = 256;
            s[8] = 960;
            s[9] = 1400;
            s[16] = 1;
        }
        assert!(validate_bw_status(&decode_bw_status(&w).unwrap(), 960).is_ok());
        w[3 * 20 + 16] = 0;
        assert!(validate_bw_status(&decode_bw_status(&w).unwrap(), 960).is_err());
        w[3 * 20 + 16] = 1;
        w[7 * 20 + 9] = 1537;
        assert!(validate_bw_status(&decode_bw_status(&w).unwrap(), 960).is_err());
        w[7 * 20 + 9] = 1400;
        w[6 * 20 + 2] = 2;
        assert!(validate_bw_status(&decode_bw_status(&w).unwrap(), 960).is_err());
    }
    #[test]
    fn biprism_v10_tile_identity_cannot_substitute_selected_atoms() {
        let ids = std::array::from_fn(|_| std::array::from_fn(|i| i as u32));
        let mut bytes = vec![0u8; 8 * BIPRISM_TILE_BYTES];
        for leaf in 0..8 {
            let b = &mut bytes[leaf * BIPRISM_TILE_BYTES..(leaf + 1) * BIPRISM_TILE_BYTES];
            for v in 0..256 {
                b[4 * v..4 * v + 4].copy_from_slice(&(v as u32).to_le_bytes());
            }
            b[TILE_COUNT_OFFSET..TILE_COUNT_OFFSET + 4].copy_from_slice(&960u32.to_le_bytes());
            b[TILE_COUNT_OFFSET + 4..TILE_COUNT_OFFSET + 8].copy_from_slice(&1400u32.to_le_bytes());
        }
        assert!(validate_tile_identity(&bytes, &ids, &vec![0; 256]).is_ok());
        bytes[7 * BIPRISM_TILE_BYTES + 4..7 * BIPRISM_TILE_BYTES + 8]
            .copy_from_slice(&0u32.to_le_bytes());
        assert!(validate_tile_identity(&bytes, &ids, &vec![0; 256]).is_err());
    }
    #[test]
    fn biprism_v10_unit_bridge_requires_fresh_velocity_and_all_leaf_commit() {
        let mut words = vec![0; 128];
        for leaf in 0..8 {
            let w = &mut words[leaf * 16..leaf * 16 + 16];
            w[0] = BIPRISM_UNIT_BRIDGE_MAGIC;
            w[1] = leaf as u32;
            w[3] = 73;
            w[4] = 7;
            w[8] = 1;
            w[10] = 256;
        }
        let check = |w: &[u32], count, checkpoint| {
            validate_unit_bridge_status(
                &decode_unit_bridge_status(w).unwrap(),
                256,
                73,
                count,
                checkpoint,
            )
        };
        assert!(check(&words, 0, UnitBridgeCheckpoint::FullImport).is_ok());
        for leaf in 0..8 {
            let w = &mut words[leaf * 16..leaf * 16 + 16];
            w[5] = 73;
            w[6] = 1;
            w[7] = 1;
            w[9] = 1;
            w[11] = 4;
        }
        assert!(check(&words, 4, UnitBridgeCheckpoint::CoreExport).is_ok());
        assert!(check(&words, 4, UnitBridgeCheckpoint::VelocityRefresh).is_err());
        for leaf in 0..8 {
            words[leaf * 16 + 4] = 15;
            words[leaf * 16 + 8] = 2;
        }
        assert!(check(&words, 4, UnitBridgeCheckpoint::VelocityRefresh).is_ok());
        for leaf in 0..8 {
            words[leaf * 16 + 6] = 2;
            words[leaf * 16 + 7] = 3;
            words[leaf * 16 + 9] = 2;
        }
        assert!(check(&words, 4, UnitBridgeCheckpoint::DeliveredVelocityExport).is_ok());
        assert!(check(&words, 4, UnitBridgeCheckpoint::ObservationRefresh).is_err());
        let mut observation_words = words.clone();
        for leaf in 0..8 {
            observation_words[leaf * 16 + 4] = 31;
            observation_words[leaf * 16 + 8] = 3;
        }
        assert!(check(
            &observation_words,
            4,
            UnitBridgeCheckpoint::ObservationRefresh
        )
        .is_ok());
        observation_words[7 * 16 + 4] = 15;
        assert!(check(
            &observation_words,
            4,
            UnitBridgeCheckpoint::ObservationRefresh
        )
        .is_err());
        words[3 * 16 + 2] = 4;
        assert!(check(&words, 4, UnitBridgeCheckpoint::DeliveredVelocityExport).is_err());
        words[3 * 16 + 2] = 0;
        words[7 * 16 + 11] = 0;
        assert!(check(&words, 4, UnitBridgeCheckpoint::DeliveredVelocityExport).is_err());
        words[7 * 16 + 11] = 4;
        words[7 * 16 + 3] = 72;
        assert!(check(&words, 4, UnitBridgeCheckpoint::DeliveredVelocityExport).is_err());
        assert!(decode_unit_bridge_status(&words[..127]).is_err());
    }
}

/// Actual numerical SoA layout. The peer-map region at30976 is immutable
/// preparation metadata. The final matrix region retains exact accepted C4
/// assembly bits, ordered xx,yy,zz,xy,xz,yz, and requires matching final
/// publication/commit identity before work consumption. Other regions are replay scratch.
#[repr(C,align(128))]
pub struct BiprismNuclearQrFfi {
    pub dx:[[[f64;192];3];64],pub df:[[[f64;192];3];64],
    pub alpha_dx:[f64;64],pub alpha_df:[f64;64],
    pub work_dx:[[f64;192];3],pub work_df:[[f64;192];3],pub direction:[[f64;192];3],
    pub coefficient:[f64;64],pub scalars:[f64;8],pub control:[u32;4],pub history:[[f64;6];48],
    pub newton_scalars:[f64;8],pub newton_control:[u32;5],
    pub trust_radius:f64,pub initial_trial_weight:f64,pub backtracks:u32,pub direction_checked:u32,
    pub model_slope:f64,pub model_curvature:f64,pub trust_gram:[f64;4096],pub trust_factor:[f64;4096],pub trust_y:[f64;64],
    pub trust_norm2:f64,pub regularization:f64,pub trust_status:u32,
    pub input_basis:[[[f64;192];3];64],pub input_alpha:[f64;64],
    pub pre_dx:[[[f64;192];3];63],pub continuation_previous_velocity:[[f64;192];3],
    pub pre_df:[[[f64;192];3];64],
    pub pre_alpha_dx:[f64;63],pub continuation_previous_alpha:f64,
    pub pre_alpha_df:[f64;63],pub continuation_previous_stage:f64,pub pre_count:u32,
    pub direction_error_best:f64,
    pub accepted_trial_ratio:f64,
}
const _:()=assert!(core::mem::size_of::<BiprismNuclearQrFfi>()==1560064);
const _:()=assert!(core::mem::offset_of!(BiprismNuclearQrFfi,df)==294912);
const _:()=assert!(core::mem::offset_of!(BiprismNuclearQrFfi,alpha_df)==590336);
const _:()=assert!(core::mem::offset_of!(BiprismNuclearQrFfi,control)==605248);
const _:()=assert!(core::mem::offset_of!(BiprismNuclearQrFfi,history)==605264);
const _:()=assert!(core::mem::offset_of!(BiprismNuclearQrFfi,newton_scalars)==607568);
const _:()=assert!(core::mem::offset_of!(BiprismNuclearQrFfi,newton_control)==607632);
const _:()=assert!(core::mem::offset_of!(BiprismNuclearQrFfi,input_basis)==673768);
const _:()=assert!(core::mem::offset_of!(BiprismNuclearQrFfi,input_alpha)==968680);
const _:()=assert!(core::mem::offset_of!(BiprismNuclearQrFfi,pre_dx)==969192);
const _:()=assert!(core::mem::offset_of!(BiprismNuclearQrFfi,continuation_previous_velocity)==1259496);
const _:()=assert!(core::mem::offset_of!(BiprismNuclearQrFfi,pre_df)==1264104);
const _:()=assert!(core::mem::offset_of!(BiprismNuclearQrFfi,continuation_previous_alpha)==1559520);
const _:()=assert!(core::mem::offset_of!(BiprismNuclearQrFfi,continuation_previous_stage)==1560032);
const _:()=assert!(core::mem::offset_of!(BiprismNuclearQrFfi,pre_count)==1560040);
const _:()=assert!(core::mem::offset_of!(BiprismNuclearQrFfi,direction_error_best)==1560048);
const _:()=assert!(core::mem::offset_of!(BiprismNuclearQrFfi,accepted_trial_ratio)==1560056);
#[repr(C,align(128))]
pub struct BiprismNuclearEvaluationFfi {
    pub header:[u32;10],pub original_h:f32,
    pub continuation_stage:u32,pub next_stage:u32,pub reserved:[u32;19],
    pub rows:[[f64;9];192],
}
const _:()=assert!(core::mem::size_of::<BiprismNuclearEvaluationFfi>()==13952);
const _:()=assert!(core::mem::offset_of!(BiprismNuclearEvaluationFfi,original_h)==40);
const _:()=assert!(core::mem::offset_of!(BiprismNuclearEvaluationFfi,rows)==128);
#[repr(C,align(128))]
pub struct BiprismNuclearArcFfi {
    pub anchor_v:[[f64;192];3],pub tangent_v:[[f64;192];3],
    pub anchor_alpha:f64,pub tangent_alpha:f64,pub anchor_h:f64,pub tangent_h:f64,pub step:f64,
    pub current_h:f64,pub base_h:f64,pub base_defect:f64,pub work_dx:f64,pub work_df:f64,pub direction:f64,
    pub dx:[f64;64],pub df:[f64;64],pub input:[f64;64],
    pub defect:f64,pub norm:f64,pub dot:f64,pub scale:f64,pub reference_row:f64,
    pub active:u32,pub roots:u32,pub last_root_trial:u32,pub reserved:u32,
    pub history:[[f64;4];192],
    pub slow_trial_streak:u32,pub segment_accepted_trials:u32,pub tangent_mode:u32,
    pub tangent_probe_refinements:u32,
}
const _:()=assert!(core::mem::size_of::<BiprismNuclearArcFfi>()==17152);
const _:()=assert!(core::mem::offset_of!(BiprismNuclearArcFfi,slow_trial_streak)==17040);
const _:()=assert!(core::mem::offset_of!(BiprismNuclearArcFfi,tangent_mode)==17048);
const _:()=assert!(core::mem::offset_of!(BiprismNuclearArcFfi,tangent_probe_refinements)==17052);
#[repr(C,align(128))]
pub struct BiprismNumericalScratchFfi {
    pub x_start:[[f32;192];3],pub v_start:[[f32;192];3],pub p_start:[[f32;192];3],
    pub next_mid_x:[[f32;192];3],pub next_mid_v:[[f32;192];3],pub endpoint_v:[[f32;192];3],
    pub kinetic_start:[f64;192],pub conservative_force:[[f64;192];3],pub reciprocal_force:[[f64;192];3],
    pub accepted_phase:[f32;1536],pub cta_footer:[u8;256],pub peer_edge_index:[u16;1536],
    pub endpoint_metric:[[f32;192];6],
    pub working_phi:[f64;256],
    pub nuclear_previous_residual:[[f32;192];3],
    pub nuclear_start_momentum:[[f64;192];3],
    pub nuclear_rotation_input:[[f64;192];3],
    pub nuclear_rotation_output:[[f64;192];3],
    pub nuclear_endpoint_momentum:[[f64;192];3],
    pub nuclear_midpoint_velocity:[[f64;192];3],
    pub nuclear_used_force:[[f64;192];3],
    pub nuclear_base_velocity:[[f32;192];3],
    pub nuclear_position_start:[[f32;6];192],
    pub nuclear_position_live:[[f32;6];256],
    pub nuclear_trial_velocity:[[f64;192];3],
    pub nuclear_base_velocity_f64:[[f64;192];3],
    pub nuclear_base_direction:[[f64;192];3],
    pub nuclear_metric:[[f64;192];6],
    pub nuclear_secant:BiprismNuclearQrFfi,
    pub nuclear_evaluations:BiprismNuclearEvaluationFfi,
    pub nuclear_arc:BiprismNuclearArcFfi,
    pub nuclear_evaluation_extension:[[f64;9];832],
    pub nuclear_arc_history_extension:[[f64;4];832],
    pub nuclear_evaluation_tail:[[f64;9];512],
    pub nuclear_arc_history_tail:[[f64;4];512],
    pub nuclear_evaluation_completion:[[f64;9];1536],
    pub nuclear_arc_history_completion:[[f64;4];1536],
}
const _:()={
    assert!(core::mem::size_of::<BiprismNumericalScratchFfi>()==BIPRISM_NUMERICAL_SCRATCH_BYTES);
    assert!(core::mem::offset_of!(BiprismNumericalScratchFfi,nuclear_evaluations)==1666816);
    assert!(core::mem::offset_of!(BiprismNumericalScratchFfi,nuclear_arc)==1680768);
    assert!(core::mem::offset_of!(BiprismNumericalScratchFfi,nuclear_evaluation_extension)==1697920);
    assert!(core::mem::offset_of!(BiprismNumericalScratchFfi,nuclear_arc_history_extension)==1757824);
    assert!(core::mem::offset_of!(BiprismNumericalScratchFfi,nuclear_evaluation_tail)==1784448);
    assert!(core::mem::offset_of!(BiprismNumericalScratchFfi,nuclear_arc_history_tail)==1821312);
    assert!(core::mem::offset_of!(BiprismNumericalScratchFfi,nuclear_evaluation_completion)==1837696);
    assert!(core::mem::offset_of!(BiprismNumericalScratchFfi,nuclear_arc_history_completion)==1948288);
    assert!(core::mem::align_of::<BiprismNumericalScratchFfi>()==128);
    assert!(core::mem::offset_of!(BiprismNumericalScratchFfi,v_start)==2304);
    assert!(core::mem::offset_of!(BiprismNumericalScratchFfi,p_start)==4608);
    assert!(core::mem::offset_of!(BiprismNumericalScratchFfi,next_mid_x)==6912);
    assert!(core::mem::offset_of!(BiprismNumericalScratchFfi,next_mid_v)==9216);
    assert!(core::mem::offset_of!(BiprismNumericalScratchFfi,endpoint_v)==11520);
    assert!(core::mem::offset_of!(BiprismNumericalScratchFfi,kinetic_start)==13824);
    assert!(core::mem::offset_of!(BiprismNumericalScratchFfi,conservative_force)==15360);
    assert!(core::mem::offset_of!(BiprismNumericalScratchFfi,reciprocal_force)==19968);
    assert!(core::mem::offset_of!(BiprismNumericalScratchFfi,accepted_phase)==24576);
    assert!(core::mem::offset_of!(BiprismNumericalScratchFfi,cta_footer)==30720);
    assert!(core::mem::offset_of!(BiprismNumericalScratchFfi,peer_edge_index)==30976);
    assert!(core::mem::offset_of!(BiprismNumericalScratchFfi,endpoint_metric)==34048);
    assert!(core::mem::offset_of!(BiprismNumericalScratchFfi,working_phi)==38656);
    assert!(core::mem::offset_of!(BiprismNumericalScratchFfi,nuclear_previous_residual)==40704);
    assert!(core::mem::offset_of!(BiprismNumericalScratchFfi,nuclear_start_momentum)==43008);
    assert!(core::mem::offset_of!(BiprismNumericalScratchFfi,nuclear_midpoint_velocity)==61440);
    assert!(core::mem::offset_of!(BiprismNumericalScratchFfi,nuclear_used_force)==66048);
    assert!(core::mem::offset_of!(BiprismNumericalScratchFfi,nuclear_base_velocity)==70656);
    assert!(core::mem::offset_of!(BiprismNumericalScratchFfi,nuclear_position_start)==72960);
    assert!(core::mem::offset_of!(BiprismNumericalScratchFfi,nuclear_position_live)==77568);
    assert!(core::mem::offset_of!(BiprismNumericalScratchFfi,nuclear_trial_velocity)==83712);
    assert!(core::mem::offset_of!(BiprismNumericalScratchFfi,nuclear_base_velocity_f64)==88320);
    assert!(core::mem::offset_of!(BiprismNumericalScratchFfi,nuclear_base_direction)==92928);
    assert!(core::mem::offset_of!(BiprismNumericalScratchFfi,nuclear_metric)==97536);
};
/// This relation is stronger than the separate192-core population guard. It
/// covers every active oriented edge, including halo endpoints, without changing
/// the source population, direction, count or Wilson averaging denominator.
pub fn build_peer_edge_bijections(tiles:&[u8])->Result<[[u16;1536];8],String>{
    if tiles.len()!=8*BIPRISM_TILE_BYTES{return Err("peer edge map tile extent".into());}
    let mut edges:Vec<Vec<(u32,u32)>>=Vec::with_capacity(8);
    let mut indices=Vec::with_capacity(8);
    for leaf in 0..8 {
        let tile=&tiles[leaf*BIPRISM_TILE_BYTES..(leaf+1)*BIPRISM_TILE_BYTES];
        let n=u32::from_le_bytes(tile[225284..225288].try_into().unwrap())as usize;
        if !(1..=1536).contains(&n){return Err(format!("leaf{leaf} active edge extent{n}"));}
        let mut row=Vec::with_capacity(n);let mut index=std::collections::HashMap::new();
        let mut undirected=std::collections::HashSet::new();
        for e in 0..n {
            let u=tile[29696+2*e]as usize;let v=tile[29697+2*e]as usize;
            let gu=u32::from_le_bytes(tile[u*4..u*4+4].try_into().unwrap())&0x00ff_ffff;
            let gv=u32::from_le_bytes(tile[v*4..v*4+4].try_into().unwrap())&0x00ff_ffff;
            if u==v||gu==gv||!undirected.insert((gu.min(gv),gu.max(gv)))||index.insert((gu,gv),e as u16).is_some(){return Err(format!("leaf{leaf} duplicate/degenerate oriented physical edge{e}: ({gu},{gv})"));}
            row.push((gu,gv));
        }
        edges.push(row);indices.push(index);
    }
    let mut maps=[[u16::MAX;1536];8];
    for leaf in 0..8 {let next=(leaf+1)&7;
        if edges[leaf].len()!=edges[next].len(){return Err(format!("peer edge population differs leaf{leaf}={} next{next}={}",edges[leaf].len(),edges[next].len()));}
        let mut used=std::collections::HashSet::new();
        for (e,key)in edges[leaf].iter().enumerate(){let peer=*indices[next].get(key).ok_or_else(||format!("missing oriented physical peer leaf{leaf} edge{e}={key:?} on leaf{next}; explicit common topology required"))?;
            if !used.insert(peer){return Err("peer edge mapping is not bijective".into());}maps[leaf][e]=peer;}
    }Ok(maps)
}
