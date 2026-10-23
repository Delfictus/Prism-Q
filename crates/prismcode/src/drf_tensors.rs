use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DrfTensor {
    pub tensor_id: u64,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub layout: String,
    pub compression_codec_ref: String,
    pub payload: Vec<f32>,
    pub canonical_or_derived: String,
    pub quantization_policy: String,
    pub reconstruction_hash: String,
    pub claims_canonical_lossless: bool,
    pub tensor_hash: String,
}

impl DrfTensor {
    pub fn new(tensor_id: u64, shape: Vec<u64>, payload: Vec<f32>, codec: &str) -> Self {
        Self::new_with_state(
            tensor_id,
            shape,
            payload,
            codec,
            "DRF-CANONICAL",
            "NONE_CANONICAL",
            true,
        )
    }

    pub fn new_with_state(
        tensor_id: u64,
        shape: Vec<u64>,
        payload: Vec<f32>,
        codec: &str,
        canonical_or_derived: &str,
        quantization_policy: &str,
        claims_canonical_lossless: bool,
    ) -> Self {
        let mut bytes = Vec::with_capacity(payload.len() * 4);
        for value in &payload {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(canonical_or_derived.as_bytes());
        bytes.extend_from_slice(quantization_policy.as_bytes());
        let reconstruction_hash = blake3::hash(
            format!(
                "DRF_RECONSTRUCT:{tensor_id}:{canonical_or_derived}:{quantization_policy}:{}",
                payload.len()
            )
            .as_bytes(),
        )
        .to_hex()
        .to_string();
        Self {
            tensor_id,
            dtype: "F32".to_string(),
            shape,
            layout: "SOA".to_string(),
            compression_codec_ref: codec.to_string(),
            payload,
            canonical_or_derived: canonical_or_derived.to_string(),
            quantization_policy: quantization_policy.to_string(),
            reconstruction_hash,
            claims_canonical_lossless,
            tensor_hash: blake3::hash(&bytes).to_hex().to_string(),
        }
    }

    pub fn element_count(&self) -> u64 {
        self.shape.iter().product()
    }

    pub fn recompute_tensor_hash(&self) -> String {
        let mut bytes = Vec::with_capacity(self.payload.len() * 4);
        for value in &self.payload {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(self.canonical_or_derived.as_bytes());
        bytes.extend_from_slice(self.quantization_policy.as_bytes());
        blake3::hash(&bytes).to_hex().to_string()
    }
}
