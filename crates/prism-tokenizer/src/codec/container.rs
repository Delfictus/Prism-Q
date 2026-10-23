//! §5 — codec-inverting front-end. NO byte reaches a quantizer without first being decoded
//! and seal-checked (INV-1). QNANL (`PRSQNB01`) decoding delegates to the engine's
//! authoritative `prismcode` decoder (single source of truth); the spike packet (`PRSPCZ01`)
//! and byte-plane transpose live in an engine *binary* upstream, so they are reimplemented
//! here verbatim against the grounded layout.

use super::records::{GpuSpikeEvent, GPU_SPIKE_EVENT_BYTES};
use super::CodecError;
use std::io::Cursor;

pub const QNANL_MAGIC: &[u8; 8] = b"PRSQNB01";
pub const SPIKE_MAGIC: &[u8; 8] = b"PRSPCZ01";
const QNANL_FIXED_HEADER: usize = 152;
const SPIKE_FIXED_HEADER: usize = 8 + 4 + 8 * 4; // magic + version + 4×u64 lengths

#[inline]
fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
#[inline]
fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
#[inline]
fn u64_at(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes([
        b[o],
        b[o + 1],
        b[o + 2],
        b[o + 3],
        b[o + 4],
        b[o + 5],
        b[o + 6],
        b[o + 7],
    ])
}

// --------------------------------------------------------------------------- QNANL (PRSQNB01)

/// A decoded QNANL artifact. `verified` reflects seal+hash re-verification by the authoritative
/// decoder (INV-6 integrity, not authenticity). On failure `payload` is `None` and the record
/// is quarantined (INV-3) — the stream is never aborted (§5.1).
#[derive(Clone, Debug)]
pub struct QnanlDecoded {
    pub schema_version: u32,
    pub family_code: u16,
    pub verified: bool,
    pub payload: Option<Vec<u8>>,
    /// BLAKE3 of the uncompressed payload (header field; → sidecar, never vocabulary).
    pub raw_hash: [u8; 32],
    /// BLAKE3 of the NANL-Q4 seal (header field; → sidecar).
    pub seal_sha: [u8; 32],
}

/// Decode a QNANL `PRSQNB01` packet. The header is parsed locally for routing metadata
/// (family/version/seal sha); the heavy unseal→zstd⁻¹→hash-verify is delegated to the
/// engine's `prismcode::qnanl_artifact::decode_packet_bytes` so the tokenizer cannot diverge
/// from the engine format (INV-1).
pub fn decode_qnanl(packet: &[u8]) -> Result<QnanlDecoded, CodecError> {
    if packet.len() < QNANL_FIXED_HEADER || &packet[0..8] != QNANL_MAGIC {
        return Err(CodecError::Format("not a PRSQNB01 QNANL packet".into()));
    }
    let schema_version = u32_at(packet, 8);
    let family_code = u16_at(packet, 12);
    let mut raw_hash = [0u8; 32];
    raw_hash.copy_from_slice(&packet[56..88]);
    let mut seal_sha = [0u8; 32];
    seal_sha.copy_from_slice(&packet[120..152]);

    // Authoritative decode (verifies seal + all hashes). Ok ⇒ integrity_verified; Err ⇒ quarantine.
    let (verified, payload) = match prismcode::qnanl_artifact::decode_packet_bytes(packet) {
        Ok(bytes) => (true, Some(bytes)),
        Err(_) => (false, None),
    };
    Ok(QnanlDecoded {
        schema_version,
        family_code,
        verified,
        payload,
        raw_hash,
        seal_sha,
    })
}

// ----------------------------------------------------------------- byte-plane transpose (§5)

/// Inverse byte-plane transpose: lane-major → record-major (verbatim from
/// `nhs_rt_full::untranspose_fixed_stride_payload`).
pub fn untranspose(
    transposed: &[u8],
    record_count: usize,
    stride: usize,
) -> Result<Vec<u8>, CodecError> {
    if record_count == 0 {
        return Ok(Vec::new());
    }
    if stride == 0 {
        return Err(CodecError::Format("stride cannot be zero".into()));
    }
    if transposed.len() != record_count * stride {
        return Err(CodecError::Format(format!(
            "transposed length {} != record_count {} * stride {}",
            transposed.len(),
            record_count,
            stride
        )));
    }
    let mut out = vec![0u8; transposed.len()];
    let mut read = 0usize;
    for lane in 0..stride {
        for record in 0..record_count {
            out[record * stride + lane] = transposed[read];
            read += 1;
        }
    }
    Ok(out)
}

/// Forward byte-plane transpose (record-major → lane-major). Test/round-trip support; the
/// engine applies this before zstd.
pub fn transpose(
    payload: &[u8],
    record_count: usize,
    stride: usize,
) -> Result<Vec<u8>, CodecError> {
    if record_count == 0 {
        return Ok(Vec::new());
    }
    if stride == 0 {
        return Err(CodecError::Format("stride cannot be zero".into()));
    }
    if payload.len() != record_count * stride {
        return Err(CodecError::Format("payload length mismatch".into()));
    }
    let mut out = vec![0u8; payload.len()];
    let mut write = 0usize;
    for lane in 0..stride {
        for record in 0..record_count {
            out[write] = payload[record * stride + lane];
            write += 1;
        }
    }
    Ok(out)
}

// --------------------------------------------------------------- spike packet (PRSPCZ01)

/// A decoded NANL spike packet (A2 NANL_SPIKE_PACKET → A1 records).
#[derive(Clone, Debug)]
pub struct SpikePacketDecoded {
    pub record_count: usize,
    pub stride: usize,
    pub events: Vec<GpuSpikeEvent>,
}

/// Decode a `PRSPCZ01` spike packet: parse header → zstd⁻¹ of the compressed body → inverse
/// byte-plane transpose → typed `GpuSpikeEvent` records. The codec stack is
/// `byte-plane transpose → SECTION_ZSTD`; the NANL-Q4 seal is a sibling file (referenced by
/// path in the header), not embedded in this packet.
pub fn decode_spike_packet(packet: &[u8]) -> Result<SpikePacketDecoded, CodecError> {
    if packet.len() < SPIKE_FIXED_HEADER || &packet[0..8] != SPIKE_MAGIC {
        return Err(CodecError::Format("not a PRSPCZ01 spike packet".into()));
    }
    let header_len = u64_at(packet, 12) as usize;
    let prefix_len = u64_at(packet, 20) as usize;
    let suffix_len = u64_at(packet, 28) as usize;
    let compressed_len = u64_at(packet, 36) as usize;

    let header_start = SPIKE_FIXED_HEADER;
    let header_end = header_start + header_len;
    let compressed_start = header_end + prefix_len + suffix_len;
    let compressed_end = compressed_start + compressed_len;
    if compressed_end != packet.len() {
        return Err(CodecError::Format(format!(
            "spike packet length mismatch: declared end {compressed_end} != actual {}",
            packet.len()
        )));
    }

    let header: serde_json::Value = serde_json::from_slice(&packet[header_start..header_end])
        .map_err(|e| CodecError::Format(format!("spike header JSON: {e}")))?;
    let record_count = header
        .get("record_count")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| CodecError::Format("spike header missing record_count".into()))?
        as usize;
    let stride = header
        .get("payload_stride")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| CodecError::Format("spike header missing payload_stride".into()))?
        as usize;

    let compressed = &packet[compressed_start..compressed_end];
    let transposed = zstd::stream::decode_all(Cursor::new(compressed))
        .map_err(|e| CodecError::Codec(format!("SECTION_ZSTD decode: {e}")))?;
    let payload = untranspose(&transposed, record_count, stride)?;

    let mut events = Vec::with_capacity(record_count);
    for record in payload.chunks(stride.max(1)) {
        if record.len() >= GPU_SPIKE_EVENT_BYTES {
            if let Some(ev) = GpuSpikeEvent::from_le_bytes(record) {
                events.push(ev);
            }
        }
    }
    Ok(SpikePacketDecoded {
        record_count,
        stride,
        events,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn untranspose_inverts_transpose() {
        let record_count = 7;
        let stride = GPU_SPIKE_EVENT_BYTES;
        let payload: Vec<u8> = (0..(record_count * stride))
            .map(|i| (i % 251) as u8)
            .collect();
        let t = transpose(&payload, record_count, stride).unwrap();
        assert_ne!(t, payload, "transpose should reorder");
        let back = untranspose(&t, record_count, stride).unwrap();
        assert_eq!(back, payload);
    }

    #[test]
    fn spike_packet_round_trips_synthetic() {
        // Build N records → record-major bytes → transpose → zstd → assemble PRSPCZ01 → decode.
        let stride = GPU_SPIKE_EVENT_BYTES;
        let n = 5usize;
        let mut events = Vec::new();
        let mut payload = Vec::new();
        for k in 0..n {
            let mut ev = GpuSpikeEvent::zeroed();
            ev.timestep = k as i32;
            ev.spike_source = (k % 5) as i32 + 1;
            ev.wavelength_nm = 280.0;
            ev.aromatic_residue_id = if k == 0 { -1 } else { k as i32 };
            ev.phase_bits = (k * 17) as u32;
            payload.extend_from_slice(&ev.to_le_bytes());
            events.push(ev);
        }
        let transposed = transpose(&payload, n, stride).unwrap();
        let compressed = zstd::stream::encode_all(Cursor::new(&transposed), 3).unwrap();
        let header = serde_json::to_vec(&serde_json::json!({
            "schema_kind": "prismcode_nanl_spike_packet_header",
            "record_count": n,
            "payload_stride": stride,
        }))
        .unwrap();

        let mut packet = Vec::new();
        packet.extend_from_slice(SPIKE_MAGIC);
        packet.extend_from_slice(&1u32.to_le_bytes());
        packet.extend_from_slice(&(header.len() as u64).to_le_bytes());
        packet.extend_from_slice(&0u64.to_le_bytes()); // prefix_len
        packet.extend_from_slice(&0u64.to_le_bytes()); // suffix_len
        packet.extend_from_slice(&(compressed.len() as u64).to_le_bytes());
        packet.extend_from_slice(&header);
        packet.extend_from_slice(&compressed);

        let decoded = decode_spike_packet(&packet).unwrap();
        assert_eq!(decoded.record_count, n);
        assert_eq!(decoded.stride, stride);
        assert_eq!(decoded.events, events);
    }

    #[test]
    fn rejects_wrong_magic() {
        assert!(decode_qnanl(b"NOTQNANL........").is_err());
        assert!(decode_spike_packet(&[0u8; 64]).is_err());
    }

    /// §14.1 round-trip gate on a REAL sealed artifact: decode + seal re-verify must succeed.
    #[test]
    fn decodes_real_qnanl_fixture_with_seal_verified() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../OPERATOR_CONTROL/PRISMOS_PATHB_ACTIVE_MATERIALIZATION.json.prismqnanl");
        if !root.exists() {
            eprintln!("skip: real QNANL fixture not present at {}", root.display());
            return;
        }
        let bytes = std::fs::read(&root).unwrap();
        assert_eq!(&bytes[0..8], QNANL_MAGIC, "fixture must be PRSQNB01");
        let d = decode_qnanl(&bytes).unwrap();
        assert!(
            d.verified,
            "real fixture seal must re-verify (INV-6 integrity)"
        );
        assert!(
            d.payload.is_some(),
            "verified artifact must yield a decoded payload"
        );
        assert!(!d.payload.unwrap().is_empty());
    }
}
