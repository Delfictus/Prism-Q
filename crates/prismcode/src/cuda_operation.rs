//! Process-local serialization and observed provenance for direct CUDA calls.
//!
//! This object retains the existing FFI operation mutex. It does not authorize
//! GPU ownership or consult a scheduler, privileged helper, or kernel journal.

use crate::binary_runtime::{PcError, PcResult};
use crate::operand_ledger::sha256_file_hex;
use serde_json::{json, Value};
use std::sync::{Mutex, MutexGuard};

static CUDA_OPERATION: Mutex<()> = Mutex::new(());

pub(crate) struct CudaOperation {
    pub(crate) provenance: Value,
    _operation: MutexGuard<'static, ()>,
}

impl CudaOperation {
    pub(crate) fn begin() -> PcResult<Self> {
        let operation = CUDA_OPERATION.lock()
            .map_err(|_| PcError::Gpu("previous CUDA operation panicked".into()))?;
        // Provenance is observational: inability to read metadata must not
        // turn into a replacement GPU admission policy. Hash the running ELF
        // through procfs so replacement of its filesystem path cannot rebind it.
        let executable = std::env::current_exe();
        let digest = sha256_file_hex(std::path::Path::new("/proc/self/exe"));
        let provenance = json!({
            "schema_kind": "prismos_cuda_operation_provenance",
            "schema_version": 1,
            "execution_mode": "direct",
            "process_id": std::process::id(),
            "executable_path": executable.as_ref().ok(),
            "executable_sha256": digest.as_ref().ok(),
            "executable_path_error": executable.err().map(|error| error.to_string()),
            "executable_sha256_error": digest.err().map(|error| error.to_string()),
            "arbiter_invoked": false,
            "gpu_lease_required": false,
        });
        Ok(Self { provenance, _operation: operation })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_operation_observes_running_process_without_a_lease() {
        let operation = CudaOperation::begin().unwrap();
        assert_eq!(operation.provenance["process_id"], std::process::id());
        assert_eq!(operation.provenance["executable_path"],
            std::env::current_exe().unwrap().to_string_lossy().as_ref());
        assert_eq!(operation.provenance["executable_sha256"],
            sha256_file_hex(std::path::Path::new("/proc/self/exe")).unwrap());
        assert_eq!(operation.provenance["execution_mode"], "direct");
        assert_eq!(operation.provenance["gpu_lease_required"], false);
        assert_eq!(operation.provenance["arbiter_invoked"], false);
        assert!(operation.provenance.get("lease_id").is_none());
        assert!(operation.provenance.get("appliance_authority").is_none());
        assert!(operation.provenance.get("status").is_none());
    }

    #[test]
    fn operation_serialization_is_released_on_drop() {
        let operation = CudaOperation::begin().unwrap();
        assert!(matches!(CUDA_OPERATION.try_lock(), Err(std::sync::TryLockError::WouldBlock)));
        drop(operation);
        // Re-enter the public operation path to prove that the previous guard
        // released the mutex, even if a different test acquires it first.
        assert!(CudaOperation::begin().is_ok());
    }
}
