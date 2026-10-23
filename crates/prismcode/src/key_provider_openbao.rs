use crate::binary_runtime::{PcError, PcResult};
use crate::key_envelope::{
    KeyContext, KeyEnvelope, KeyRotationReport, RotationPolicy, SignContext, SignatureEnvelope,
};
use crate::key_provider::{KeyProviderHealth, PrismKeyProvider, ProviderState};

#[derive(Debug, Default, Clone)]
pub struct OpenBaoProvider;

impl PrismKeyProvider for OpenBaoProvider {
    fn provider_id(&self) -> &str {
        "prism-openbao-or-vault"
    }

    fn provider_state(&self) -> ProviderState {
        ProviderState::PrismVaultUnbound
    }

    fn health_check(&self) -> PcResult<KeyProviderHealth> {
        Err(PcError::Security(
            "OPENBAO_OR_VAULT is not bound on this host".to_string(),
        ))
    }

    fn wrap_dek(&self, _dek: &[u8], _context: &KeyContext) -> PcResult<KeyEnvelope> {
        Err(PcError::Security(
            "OPENBAO_OR_VAULT is not bound on this host".to_string(),
        ))
    }

    fn unwrap_dek(&self, _envelope: &KeyEnvelope, _context: &KeyContext) -> PcResult<Vec<u8>> {
        Err(PcError::Security(
            "OPENBAO_OR_VAULT is not bound on this host".to_string(),
        ))
    }

    fn sign_digest(&self, _digest: &[u8], _context: &SignContext) -> PcResult<SignatureEnvelope> {
        Err(PcError::Security(
            "OPENBAO_OR_VAULT is not bound on this host".to_string(),
        ))
    }

    fn verify_signature(&self, _signature: &SignatureEnvelope, _digest: &[u8]) -> PcResult<()> {
        Err(PcError::Security(
            "OPENBAO_OR_VAULT is not bound on this host".to_string(),
        ))
    }

    fn rotate_key(&self, _key_id: &str, _policy: &RotationPolicy) -> PcResult<KeyRotationReport> {
        Err(PcError::Security(
            "OPENBAO_OR_VAULT is not bound on this host".to_string(),
        ))
    }
}
