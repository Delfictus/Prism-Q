//! PRSPK001 — the REAL on-disk spike-binary envelope (P1-8). The engine writes `*_spikes.bin`
//! as a `PRSPK001` envelope (schema header + record-major raw `GpuSpikeEvent` payload + FNV-1a
//! checksum + `PRSPKEND` trailer) — NOT the `PRSPCZ01` zstd packet. So real spike data is only
//! reachable through this decoder.
//!
//! INTEGRITY ≠ SEAL (INV-6): PRSPK001 uses FNV-1a (a fast non-cryptographic checksum, no BLAKE3
//! seal). `integrity_ok` therefore means "bytes intact per FNV", explicitly NOT a seal and NOT
//! authenticity. Tokens routed from PRSPK001 carry verified=integrity, authenticated=0.
//!
//! Format (LE), mirroring `nhs_rt_full::md_evidence_write_header`/`parse_spike_envelope_layout`:
//! ```text
//!  [0..8]   magic "PRSPK001"        [8..12]  schema_version u32   [12..16] endian 0x01020304
//!  [16..20] stream_id u32           [20..28] run_id_len u64 + run_id
//!  +8 stem_len u64 + stem           record_count u64  byte_stride u64  payload_size u64
//!  payload[payload_size]  (record-major, stride bytes/record)  fnv1a_64 u64  trailer "PRSPKEND"
//! ```

use super::records::{known_record_stride, GpuSpikeEvent, GPU_SPIKE_EVENT_BYTES};
use super::CodecError;

pub const PRSPK001_MAGIC: &[u8; 8] = b"PRSPK001";
pub const PRSPK001_TRAILER: &[u8; 8] = b"PRSPKEND";

#[derive(Clone, Debug)]
pub struct Prspk001Decoded {
    pub stream_id: u32,
    pub record_count: usize,
    pub stride: usize,
    pub events: Vec<GpuSpikeEvent>,
    /// Some(true/false) when FNV was checked over the full payload; None for a prefix decode.
    pub integrity_ok: Option<bool>,
}

#[inline]
fn fnv1a_64(bytes: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
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

struct Header {
    stream_id: u32,
    record_count: u64,
    stride: u64,
    payload_size: u64,
    payload_offset: usize,
}

/// Parse the variable-length header (only the header bytes need be present, so a small prefix of a
/// large file suffices). Does NOT require the full payload/trailer.
fn parse_header(raw: &[u8]) -> Result<Header, CodecError> {
    if raw.len() < 8 || &raw[0..8] != PRSPK001_MAGIC {
        return Err(CodecError::Format("not a PRSPK001 envelope".into()));
    }
    let need = |end: usize, what: &str| -> Result<(), CodecError> {
        if end > raw.len() {
            Err(CodecError::Format(format!(
                "PRSPK001 truncated reading {what}"
            )))
        } else {
            Ok(())
        }
    };
    let stream_id = u32_at(raw, 16); // schema_version@8, endian@12, stream_id@16
    let mut c = 20usize;
    need(c + 8, "run_id_len")?;
    let run_id_len = u64_at(raw, c) as usize;
    c += 8 + run_id_len;
    need(c + 8, "stem_len")?;
    let stem_len = u64_at(raw, c) as usize;
    c += 8 + stem_len;
    need(c + 24, "record_count/stride/payload_size")?;
    let record_count = u64_at(raw, c);
    let stride = u64_at(raw, c + 8);
    let payload_size = u64_at(raw, c + 16);
    c += 24;
    if record_count.checked_mul(stride) != Some(payload_size) {
        return Err(CodecError::Format(
            "PRSPK001 payload_size != record_count*stride".into(),
        ));
    }
    Ok(Header {
        stream_id,
        record_count,
        stride,
        payload_size,
        payload_offset: c,
    })
}

/// Decode records, VALIDATING the envelope's declared `byte_stride`.
///
/// The envelope carries `byte_stride = size_of::<GpuSpikeEvent>()` from the
/// writing engine, so it is the authoritative statement of the payload layout.
/// Any stride other than the 96 B record is refused rather than parsed at an
/// assumed width: reading 96 bytes out of a foreign-width record and stepping
/// forward by the header stride yields plausible-looking physics with no
/// diagnostic, which is exactly what this check exists to make impossible.
fn decode_records(
    raw: &[u8],
    h: &Header,
    max_records: usize,
) -> Result<Vec<GpuSpikeEvent>, CodecError> {
    let stride = h.stride as usize;
    if !known_record_stride(stride) {
        return Err(CodecError::Format(format!(
            "PRSPK001 declares byte_stride={stride}, which is not the GpuSpikeEvent record \
             ({GPU_SPIKE_EVENT_BYTES} B); refusing to parse an unknown record layout"
        )));
    }
    let mut events = Vec::new();
    let mut k = 0usize;
    while k < max_records && k < h.record_count as usize {
        let off = h.payload_offset + k * stride;
        if off + stride > raw.len() {
            break; // prefix ran out
        }
        match GpuSpikeEvent::from_le_bytes_with_stride(&raw[off..], stride) {
            Some(ev) => events.push(ev),
            None => {
                return Err(CodecError::Format(format!(
                    "PRSPK001 record {k} at offset {off} failed to decode at stride {stride}"
                )))
            }
        }
        k += 1;
    }
    Ok(events)
}

/// Full decode: parse header, verify the FNV-1a checksum + trailer over the whole payload, decode
/// all records. Use for complete buffers.
pub fn decode_prspk001(raw: &[u8]) -> Result<Prspk001Decoded, CodecError> {
    let h = parse_header(raw)?;
    let payload_end = h
        .payload_offset
        .checked_add(h.payload_size as usize)
        .ok_or_else(|| CodecError::Format("PRSPK001 payload_end overflow".into()))?;
    if payload_end + 16 > raw.len() {
        return Err(CodecError::Format(
            "PRSPK001 truncated before checksum/trailer".into(),
        ));
    }
    if &raw[payload_end + 8..payload_end + 16] != PRSPK001_TRAILER {
        return Err(CodecError::Format("PRSPK001 trailer mismatch".into()));
    }
    let stored = u64_at(raw, payload_end);
    let integrity_ok = Some(fnv1a_64(&raw[h.payload_offset..payload_end]) == stored);
    let events = decode_records(raw, &h, h.record_count as usize)?;
    Ok(Prspk001Decoded {
        stream_id: h.stream_id,
        record_count: h.record_count as usize,
        stride: h.stride as usize,
        events,
        integrity_ok,
    })
}

/// Prefix decode: parse the header and decode up to `max_records` from whatever bytes are present
/// (a small prefix of a large file). FNV is NOT checked (`integrity_ok=None`) — only the header +
/// the requested records are needed, so a multi-hundred-MB file need not be read in full.
pub fn decode_prspk001_head(raw: &[u8], max_records: usize) -> Result<Prspk001Decoded, CodecError> {
    let h = parse_header(raw)?;
    let events = decode_records(raw, &h, max_records)?;
    Ok(Prspk001Decoded {
        stream_id: h.stream_id,
        record_count: h.record_count as usize,
        stride: h.stride as usize,
        events,
        integrity_ok: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::path::PathBuf;

    fn build_prspk001(events: &[GpuSpikeEvent], corrupt: bool) -> Vec<u8> {
        let stride = GPU_SPIKE_EVENT_BYTES as u64;
        let n = events.len() as u64;
        let mut payload = Vec::new();
        for e in events {
            payload.extend_from_slice(&e.to_le_bytes());
        }
        let mut v = Vec::new();
        v.extend_from_slice(PRSPK001_MAGIC);
        v.extend_from_slice(&1u32.to_le_bytes()); // schema_version
        v.extend_from_slice(&0x0102_0304u32.to_le_bytes()); // endian
        v.extend_from_slice(&7u32.to_le_bytes()); // stream_id
        let run_id = b"run-x";
        v.extend_from_slice(&(run_id.len() as u64).to_le_bytes());
        v.extend_from_slice(run_id);
        let stem = b"stem-y";
        v.extend_from_slice(&(stem.len() as u64).to_le_bytes());
        v.extend_from_slice(stem);
        v.extend_from_slice(&n.to_le_bytes());
        v.extend_from_slice(&stride.to_le_bytes());
        v.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        v.extend_from_slice(&payload);
        let mut cksum = fnv1a_64(&payload);
        if corrupt {
            cksum ^= 1;
        }
        v.extend_from_slice(&cksum.to_le_bytes());
        v.extend_from_slice(PRSPK001_TRAILER);
        v
    }

    fn sample_events(n: usize) -> Vec<GpuSpikeEvent> {
        (0..n)
            .map(|k| {
                let mut e = GpuSpikeEvent::zeroed();
                e.timestep = k as i32;
                e.spike_source = (k % 5) as i32 + 1;
                e.wavelength_nm = 280.0;
                e.position = [k as f32, 1.0, 2.0];
                e.aromatic_residue_id = if k == 0 { -1 } else { k as i32 };
                e
            })
            .collect()
    }

    #[test]
    fn synthetic_prspk001_full_round_trip_and_integrity() {
        let evs = sample_events(6);
        let bytes = build_prspk001(&evs, false);
        let d = decode_prspk001(&bytes).unwrap();
        assert_eq!(d.stream_id, 7);
        assert_eq!(d.record_count, 6);
        assert_eq!(d.stride, GPU_SPIKE_EVENT_BYTES);
        assert_eq!(d.events, evs);
        assert_eq!(d.integrity_ok, Some(true), "FNV must verify");
    }

    #[test]
    fn corrupt_checksum_is_detected_not_panicked() {
        let bytes = build_prspk001(&sample_events(3), true);
        let d = decode_prspk001(&bytes).unwrap();
        assert_eq!(
            d.integrity_ok,
            Some(false),
            "corrupt FNV → integrity_ok=false"
        );
    }

    #[test]
    fn truncated_and_wrong_magic_error() {
        assert!(decode_prspk001(&[0u8; 40]).is_err());
        assert!(decode_prspk001(b"NOTPK001....").is_err());
    }

    /// P1-8 real-data proof: decode a REAL engine `*_spikes.bin` (PRSPK001) — reading only a small
    /// prefix (header + first records), so the 738MB file is not loaded in full.
    #[test]
    fn decodes_real_prspk001_prefix() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        // find any in-repo real spike binary
        let mut found: Option<PathBuf> = None;
        let runs = root.join("runs");
        if runs.exists() {
            for entry in walk(&runs, 6) {
                if entry
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.ends_with("_spikes.bin"))
                    .unwrap_or(false)
                {
                    found = Some(entry);
                    break;
                }
            }
        }
        let Some(path) = found else {
            eprintln!("skip: no in-repo *_spikes.bin fixture");
            return;
        };
        // read only a 64 KiB prefix (header + plenty of records).
        let mut f = std::fs::File::open(&path).unwrap();
        let mut buf = vec![0u8; 64 * 1024];
        let n = f.read(&mut buf).unwrap();
        buf.truncate(n);
        assert_eq!(&buf[0..8], PRSPK001_MAGIC, "real fixture must be PRSPK001");

        let d = decode_prspk001_head(&buf, 5).unwrap();
        assert_eq!(d.stride, 96, "real GpuSpikeEvent stride");
        assert!(d.record_count > 0, "real file declares records");
        assert_eq!(
            d.events.len(),
            5.min(d.record_count),
            "decoded prefix records"
        );
        // sanity on a real record: spike_source is a small engine code, wavelength a UV line/0.
        let e0 = d.events[0];
        assert!(
            (0..=5).contains(&e0.spike_source),
            "real spike_source code in range, got {}",
            e0.spike_source
        );
    }

    /// OPERATOR-POINTABLE REAL-CORPUS PROOF — decode an actual engine
    /// `*_spikes.bin` named by `PRISM_TOKENIZER_SPIKE_FIXTURE` (a file outside
    /// the repo, e.g. under a governed `/var/lib/prismos` run root) and assert
    /// it declares the 96 B record stride and yields intact payload fields.
    #[test]
    fn env_fixture_corpus_decodes_at_the_declared_stride() {
        let Ok(fixture) = std::env::var("PRISM_TOKENIZER_SPIKE_FIXTURE") else {
            eprintln!("skip: PRISM_TOKENIZER_SPIKE_FIXTURE not set");
            return;
        };
        let mut f =
            std::fs::File::open(&fixture).unwrap_or_else(|e| panic!("open fixture {fixture}: {e}"));
        let mut buf = vec![0u8; 256 * 1024];
        let n = f.read(&mut buf).unwrap();
        buf.truncate(n);
        let d = decode_prspk001_head(&buf, 64).expect("PRSPK001 prefix decode");
        assert!(
            known_record_stride(d.stride),
            "fixture stride {} must be the known record layout ({} B)",
            d.stride,
            GPU_SPIKE_EVENT_BYTES
        );
        assert!(!d.events.is_empty(), "fixture yielded no records");
        for ev in &d.events {
            // The payload itself must be intact: spike_source is a small engine code.
            assert!((0..=5).contains(&ev.spike_source));
        }
    }

    /// P1-8 end-to-end on real data: decode a real PRSPK001 prefix → route each spike record into
    /// a TokenStream (the real-spike→tokens path the synthetic E2E only simulated before).
    #[test]
    fn real_prspk001_records_route_to_tokens() {
        use crate::encoder::Encoder;
        use crate::stream::TokenStream;
        use crate::types::ArtifactFamily;

        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../runs");
        if !root.exists() {
            eprintln!("skip: no runs/ fixtures");
            return;
        }
        let Some(path) = walk(&root, 6).into_iter().find(|p| {
            p.to_str()
                .map(|s| s.ends_with("_spikes.bin"))
                .unwrap_or(false)
        }) else {
            eprintln!("skip: no *_spikes.bin");
            return;
        };
        let mut f = std::fs::File::open(&path).unwrap();
        let mut buf = vec![0u8; 64 * 1024];
        let n = f.read(&mut buf).unwrap();
        buf.truncate(n);
        let d = decode_prspk001_head(&buf, 5).unwrap();

        let mut enc = Encoder::new();
        let coords: Vec<[f64; 3]> = d
            .events
            .iter()
            .map(|e| {
                [
                    e.position[0] as f64,
                    e.position[1] as f64,
                    e.position[2] as f64,
                ]
            })
            .collect();
        enc.fit_coords(ArtifactFamily::SpikeEvent, "position", &coords);
        let mut s = TokenStream::new();
        let mut total = 0;
        // PRSPK001 integrity is FNV (not a seal): verified reflects integrity, authenticated stays 0.
        for ev in &d.events {
            total += enc.route_spike(&mut s, ev, true, [0u8; 32]);
        }
        assert_eq!(total, 9 * d.events.len(), "9 fields per real spike record");
        assert_eq!(s.len(), total);
    }

    // tiny recursive walker (no walkdir dep) for the fixture hunt.
    fn walk(dir: &std::path::Path, depth: usize) -> Vec<PathBuf> {
        let mut out = Vec::new();
        if depth == 0 {
            return out;
        }
        if let Ok(rd) = std::fs::read_dir(dir) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    out.extend(walk(&p, depth - 1));
                } else {
                    out.push(p);
                }
                if out.len() > 5000 {
                    break;
                }
            }
        }
        out
    }
}
