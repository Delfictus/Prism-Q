//! Native CUDA module symbols are shared scientific owners, separate from
//! instance allocations. Capture only with every producer held and synchronized;
//! restore after all fresh constructors and before releasing any producer.
use std::collections::BTreeMap;
use anyhow::{ensure,Result};
use std::sync::Arc;
use cudarc::driver::CudaStream;
#[derive(serde::Serialize,serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeModuleContinuationV1 {
    schema_version:u32,
    #[serde(with="crate::checkpoint_codec::byte_map")]
    regions:BTreeMap<String,Vec<u8>>,
}
type Size=unsafe extern "C" fn()->usize;
type Capture=unsafe extern "C" fn(*mut u8,usize,*mut std::ffi::c_void)->i32;
type Restore=unsafe extern "C" fn(*const u8,usize,*mut std::ffi::c_void)->i32;
extern "C" {
 fn prism_checkpoint_ghost_tile_kernel_bytes()->usize;
 fn prism_checkpoint_ghost_tile_kernel_capture(bytes:*mut u8,length:usize,stream:*mut std::ffi::c_void)->i32;
 fn prism_checkpoint_ghost_tile_kernel_restore(bytes:*const u8,length:usize,stream:*mut std::ffi::c_void)->i32;
 fn prism_checkpoint_so3_project_bytes()->usize;
 fn prism_checkpoint_so3_project_capture(bytes:*mut u8,length:usize,stream:*mut std::ffi::c_void)->i32;
 fn prism_checkpoint_so3_project_restore(bytes:*const u8,length:usize,stream:*mut std::ffi::c_void)->i32;
 fn prism_checkpoint_symmetry_consensus_bytes()->usize;
 fn prism_checkpoint_symmetry_consensus_capture(bytes:*mut u8,length:usize,stream:*mut std::ffi::c_void)->i32;
 fn prism_checkpoint_symmetry_consensus_restore(bytes:*const u8,length:usize,stream:*mut std::ffi::c_void)->i32;
 fn prism_checkpoint_gearbox_bytes()->usize;
 fn prism_checkpoint_gearbox_capture(bytes:*mut u8,length:usize,stream:*mut std::ffi::c_void)->i32;
 fn prism_checkpoint_gearbox_restore(bytes:*const u8,length:usize,stream:*mut std::ffi::c_void)->i32;
 fn prism_checkpoint_sh_basis_bytes()->usize;
 fn prism_checkpoint_sh_basis_capture(bytes:*mut u8,length:usize,stream:*mut std::ffi::c_void)->i32;
 fn prism_checkpoint_sh_basis_restore(bytes:*const u8,length:usize,stream:*mut std::ffi::c_void)->i32;
 fn prism_checkpoint_adjudicator_bytes()->usize;
 fn prism_checkpoint_adjudicator_capture(bytes:*mut u8,length:usize,stream:*mut std::ffi::c_void)->i32;
 fn prism_checkpoint_adjudicator_restore(bytes:*const u8,length:usize,stream:*mut std::ffi::c_void)->i32;
}
fn owners()->[(&'static str,Size,Capture,Restore);6] {[
 ("ghost_tile_kernel",prism_checkpoint_ghost_tile_kernel_bytes,prism_checkpoint_ghost_tile_kernel_capture,prism_checkpoint_ghost_tile_kernel_restore),
 ("so3_project",prism_checkpoint_so3_project_bytes,prism_checkpoint_so3_project_capture,prism_checkpoint_so3_project_restore),
 ("symmetry_consensus",prism_checkpoint_symmetry_consensus_bytes,prism_checkpoint_symmetry_consensus_capture,prism_checkpoint_symmetry_consensus_restore),
 ("gearbox",prism_checkpoint_gearbox_bytes,prism_checkpoint_gearbox_capture,prism_checkpoint_gearbox_restore),
 ("sh_basis",prism_checkpoint_sh_basis_bytes,prism_checkpoint_sh_basis_capture,prism_checkpoint_sh_basis_restore),
 ("adjudicator",prism_checkpoint_adjudicator_bytes,prism_checkpoint_adjudicator_capture,prism_checkpoint_adjudicator_restore),
 ]}
/// # Safety
/// Every CUDA stream that can read/write these module symbols must be held
/// after completion. No concurrent module constructor may update a symbol.
pub unsafe fn continuation_allocation_plan()->Result<Vec<(String,u64)>> {
    let mut rows=Vec::new();
    for(name,size,_,_)in owners(){let n=size();ensure!(n>0 && n<=1024*1024,"invalid native module state extent {name}");rows.push((name.into(),u64::try_from(n)?));}
    Ok(rows)
}

/// Caller holds every native producer through capture and commit.
pub unsafe fn capture_quiescent(stream:&Arc<CudaStream>)->Result<NativeModuleContinuationV1> {
    stream.context().bind_to_thread()?;stream.synchronize()?;
    let mut regions=BTreeMap::new();
    for (name,size,capture,_) in owners(){
        let n=size();ensure!(n>0 && n<=1024*1024,"invalid native module state extent {name}");
        let mut bytes=vec![0u8;n];let rc=capture(bytes.as_mut_ptr(),n,stream.cu_stream() as _);
        ensure!(rc==0,"native module state capture {name} failed: CUDA {rc}");regions.insert(name.into(),bytes);
    }Ok(NativeModuleContinuationV1{schema_version:1,regions})
}
/// # Safety
/// Every fresh owner and graph constructor must have completed and every
/// producer must remain held through the final all-worker restore barrier.
pub unsafe fn restore_quiescent(stream:&Arc<CudaStream>,saved:&NativeModuleContinuationV1)->Result<()> {
    stream.context().bind_to_thread()?;stream.synchronize()?;
    ensure!(saved.schema_version==1 && saved.regions.len()==owners().len(),"native module state population changed");
    for(name,size,_,_)in owners(){ensure!(saved.regions.get(name).is_some_and(|v|v.len()==size()),"native module state extent changed: {name}");}
    for(name,_,_,restore)in owners(){let bytes=&saved.regions[name];let rc=restore(bytes.as_ptr(),bytes.len(),stream.cu_stream() as _);
        ensure!(rc==0,"native module state restore {name} failed: CUDA {rc}");}
    Ok(())
}
