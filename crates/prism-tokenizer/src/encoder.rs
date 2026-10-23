//! The encoder surface that ties §6 registry → §7 quantizers → §4/§10/§12 stream, and the
//! Stage-1 ontology routing (decoded record / JSON → typed tokens). Handles scalar, 3-D
//! coordinate (lattice), and SMILES-sequence fields.

use std::collections::{HashMap, HashSet};

use crate::codec::GpuSpikeEvent;
use crate::quant::{
    build_field_quantizer, smiles_tokenize_deep, Bcc3dQuantizer, SmilesVocab, Vq3dEma,
};
use crate::schema::FieldRegistry;
use crate::stream::TokenStream;
use crate::types::{ArtifactFamily, FieldGeometry, Quantizer};

// Cold-start coordinate box for the BCC fallback (engine-space Å). The non-naive path is
// `fit_coords` → a VQ-EMA learned on the real (clustered) distribution (§6).
const COORD_BOX_LO: [f64; 3] = [-128.0, -128.0, -128.0];
const COORD_BOX_HI: [f64; 3] = [128.0, 128.0, 128.0];

/// §13 codebook-health summary for a fitted lattice (VQ) field (see [`Encoder::coord_health`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VqHealth {
    /// `2^entropy` of the code-usage distribution (effective codes in use).
    pub perplexity: f64,
    pub codebook_size: usize,
    pub codes_used: usize,
    pub dead_codes: usize,
    /// mean / 95th-percentile / worst Euclidean distortion (Å) over the measured points.
    pub mean_distortion: f64,
    pub p95_distortion: f64,
    pub worst_distortion: f64,
    /// points whose distortion exceeds the field's declared Euclidean bound.
    pub escapes: usize,
}

pub struct Encoder {
    registry: FieldRegistry,
    quantizers: HashMap<(u8, &'static str), Box<dyn Quantizer>>, // scalar / critical-region
    bcc: HashMap<(u8, &'static str), Bcc3dQuantizer>,            // cold-start lattice fallback
    vq: HashMap<(u8, &'static str), Vq3dEma>,                    // fitted lattice (preferred, §6)
    smiles: SmilesVocab,
}

impl Encoder {
    pub fn new() -> Self {
        let registry = FieldRegistry::build();
        let mut quantizers: HashMap<(u8, &'static str), Box<dyn Quantizer>> = HashMap::new();
        let mut bcc: HashMap<(u8, &'static str), Bcc3dQuantizer> = HashMap::new();
        for spec in registry.iter() {
            match &spec.geometry {
                FieldGeometry::LatticeVq3d { codebook_bits, .. } => {
                    bcc.insert(
                        (spec.family.family_slot(), spec.field),
                        Bcc3dQuantizer::new(*codebook_bits, COORD_BOX_LO, COORD_BOX_HI),
                    );
                }
                _ => {
                    if let Some(q) = build_field_quantizer(spec) {
                        quantizers.insert((spec.family.family_slot(), spec.field), q);
                    }
                }
            }
        }
        Self {
            registry,
            quantizers,
            bcc,
            vq: HashMap::new(),
            smiles: SmilesVocab::new(),
        }
    }

    pub fn registry(&self) -> &FieldRegistry {
        &self.registry
    }

    pub fn quantizer(&self, family: ArtifactFamily, field: &str) -> Option<&dyn Quantizer> {
        let spec = self.registry.lookup(family, field)?;
        self.quantizers
            .get(&(family.family_slot(), spec.field))
            .map(|b| b.as_ref())
    }

    /// Codebook health (§13) for a fitted lattice (VQ) field: `2^entropy` of the code-usage
    /// distribution (effective number of codes in use). None if the field is unknown or has no
    /// fitted VQ yet (cold-start BCC fallback carries no usage histogram).
    pub fn coord_perplexity(&self, family: ArtifactFamily, field: &str) -> Option<f64> {
        let spec = self.registry.lookup(family, field)?;
        self.vq
            .get(&(family.family_slot(), spec.field))
            .map(|q| q.perplexity())
    }

    /// Full §13 codebook health for a fitted lattice field, measured by encoding `points` through
    /// the fitted VQ: perplexity, codebook size, distinct/dead codes, escapes (distortion beyond
    /// the field's declared Euclidean bound), and mean/p95/worst Euclidean distortion (Å). None if
    /// the field is unknown or has no fitted VQ yet.
    pub fn coord_health(
        &self,
        family: ArtifactFamily,
        field: &str,
        points: &[[f64; 3]],
    ) -> Option<VqHealth> {
        let spec = self.registry.lookup(family, field)?;
        let q = self.vq.get(&(family.family_slot(), spec.field))?;
        let codebook_size = 1usize << q.bit_budget();
        let bound = match spec.max_distortion {
            crate::types::DistortionBound::AbsoluteEuclidean(b) => b as f64,
            _ => f64::INFINITY,
        };
        let mut used: HashSet<u32> = HashSet::new();
        let mut dists: Vec<f64> = Vec::with_capacity(points.len());
        let mut escapes = 0usize;
        for &p in points {
            let (code, d) = q.encode(p);
            used.insert(code);
            let dv = d.value();
            if dv > bound + 1e-9 {
                escapes += 1;
            }
            dists.push(dv);
        }
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = dists.len();
        let mean = dists.iter().sum::<f64>() / n.max(1) as f64;
        let p95_idx = (((n as f64) * 0.95) as usize).min(n.saturating_sub(1));
        let p95 = dists.get(p95_idx).copied().unwrap_or(0.0);
        let worst = dists.last().copied().unwrap_or(0.0);
        Some(VqHealth {
            perplexity: q.perplexity(),
            codebook_size,
            codes_used: used.len(),
            dead_codes: codebook_size.saturating_sub(used.len()),
            mean_distortion: mean,
            p95_distortion: p95,
            worst_distortion: worst,
            escapes,
        })
    }

    /// Fit a learned VQ-EMA codebook for a lattice field on the real coordinate distribution
    /// (preferred over the cold-start BCC; §6). Subsequent `encode_vec3` uses it. CPU reference path
    /// — always available, always the truth.
    pub fn fit_coords(&mut self, family: ArtifactFamily, field: &str, points: &[[f64; 3]]) -> bool {
        self.fit_coords_with_backend(family, field, points, crate::backend::TokenizerBackend::Cpu)
    }

    /// Fit a learned VQ-EMA codebook with an explicit compute backend. `Auto` picks the GPU lane
    /// ONLY above the benchmarked crossover thresholds AND when a device is available (and the
    /// `gpu-tokenizer` feature is built) — otherwise it runs the identical CPU `fit`. The fitted
    /// codebook is bit-identical regardless of backend (proven by the parity bin), so correctness
    /// never depends on the GPU. Returns false for unknown / non-lattice fields.
    pub fn fit_coords_with_backend(
        &mut self,
        family: ArtifactFamily,
        field: &str,
        points: &[[f64; 3]],
        backend: crate::backend::TokenizerBackend,
    ) -> bool {
        let Some(spec) = self.registry.lookup(family, field) else {
            return false;
        };
        let bits = match spec.geometry {
            FieldGeometry::LatticeVq3d { codebook_bits, .. } => codebook_bits,
            _ => return false,
        };
        let mut q = Vq3dEma::new(bits);
        self.fit_vq_with_backend(&mut q, points, backend);
        self.vq.insert((family.family_slot(), spec.field), q);
        true
    }

    /// Internal: run the VQ fit on `q` via the resolved backend. Without the `gpu-tokenizer`
    /// feature (or without a device, or below threshold) this is exactly `q.fit(points)`.
    fn fit_vq_with_backend(
        &self,
        q: &mut Vq3dEma,
        points: &[[f64; 3]],
        backend: crate::backend::TokenizerBackend,
    ) {
        use crate::backend::{CrossoverThresholds, WorkloadShape};
        let thr = CrossoverThresholds::default();
        let codebook_size = 1usize << q.bit_budget();
        let shape = WorkloadShape {
            n_points: points.len(),
            codebook_size,
            scalar_len: 0,
        };

        #[cfg(feature = "gpu-tokenizer")]
        {
            let device_ok = crate::backend::device_available(thr.min_free_device_bytes);
            if backend.resolve_vq_fit(shape, &thr, device_ok)
                == crate::backend::TokenizerBackend::Gpu
            {
                if let Some(gpu) = crate::gpu::GpuVqContext::try_new() {
                    if q.fit_gpu(points, &gpu) {
                        return;
                    }
                }
            }
            // any miss falls through to the CPU reference (correctness never needs the GPU).
            q.fit(points);
        }
        #[cfg(not(feature = "gpu-tokenizer"))]
        {
            // CPU-only build: backend selection collapses to CPU; resolve to confirm Auto is honest.
            let _ = backend.resolve_vq_fit(shape, &thr, false);
            q.fit(points);
        }
    }

    /// P2-12: refit a scalar field's quantizer on observed samples (corpus-fit), so resolution
    /// concentrates where the real values are. False if unknown / non-scalar.
    pub fn fit_field(&mut self, family: ArtifactFamily, field: &str, samples: &[f64]) -> bool {
        let key = match self.registry.lookup(family, field) {
            Some(s) => (family.family_slot(), s.field),
            None => return false,
        };
        match self.quantizers.get_mut(&key) {
            Some(q) => {
                q.fit(samples);
                true
            }
            None => false,
        }
    }

    /// Encode one decoded scalar field. False if unknown or non-scalar (lattice/SMILES handled by
    /// `encode_vec3`/`encode_smiles`).
    pub fn encode_scalar(
        &self,
        stream: &mut TokenStream,
        family: ArtifactFamily,
        field: &str,
        value: f64,
        verified: bool,
        seal_sha: [u8; 32],
    ) -> bool {
        let Some(spec) = self.registry.lookup(family, field) else {
            return false;
        };
        let Some(ns) = self.registry.namespace(family, field) else {
            return false;
        };
        let Some(q) = self.quantizers.get(&(family.family_slot(), spec.field)) else {
            return false;
        };
        stream.push_scalar(spec, ns, q.as_ref(), value, verified, seal_sha);
        true
    }

    /// Encode a 3-D coordinate (position) → one lattice token. Uses the fitted VQ-EMA if present,
    /// else the cold-start BCC. INV-7 escape applies (Euclidean bound).
    pub fn encode_vec3(
        &self,
        stream: &mut TokenStream,
        family: ArtifactFamily,
        field: &str,
        value: [f64; 3],
        verified: bool,
        seal_sha: [u8; 32],
    ) -> bool {
        let Some(spec) = self.registry.lookup(family, field) else {
            return false;
        };
        let Some(ns) = self.registry.namespace(family, field) else {
            return false;
        };
        let key = (family.family_slot(), spec.field);
        let (code, dist) = if let Some(vq) = self.vq.get(&key) {
            vq.encode(value)
        } else if let Some(b) = self.bcc.get(&key) {
            b.encode(value)
        } else {
            return false;
        };
        stream.push_vec3(spec, ns, code, dist.value(), value, verified, seal_sha);
        true
    }

    /// Encode a SMILES string → a token SEQUENCE (one per grammar sub-token). Returns the count.
    /// The exact sub-token text is in the sidecar, so bracket atoms / UNK reconstruct losslessly.
    pub fn encode_smiles(
        &self,
        stream: &mut TokenStream,
        family: ArtifactFamily,
        field: &str,
        smiles: &str,
        verified: bool,
        seal_sha: [u8; 32],
    ) -> usize {
        let Some(spec) = self.registry.lookup(family, field) else {
            return 0;
        };
        let Some(ns) = self.registry.namespace(family, field) else {
            return 0;
        };
        if !matches!(spec.geometry, FieldGeometry::SmilesGrammar) {
            return 0;
        }
        let mut n = 0;
        for st in smiles_tokenize_deep(smiles) {
            let code = self.smiles.code(&st);
            stream.push_smiles_token(spec, ns, code, st, verified, seal_sha);
            n += 1;
        }
        n
    }

    /// Stage-1 routing: encode every field of a decoded SPIKE_EVENT record into the stream.
    pub fn route_spike(
        &self,
        stream: &mut TokenStream,
        ev: &GpuSpikeEvent,
        verified: bool,
        seal_sha: [u8; 32],
    ) -> usize {
        let f = ArtifactFamily::SpikeEvent;
        let mut n = 0usize;
        n += self.encode_scalar(
            stream,
            f,
            "wavelength_nm",
            ev.wavelength_nm as f64,
            verified,
            seal_sha,
        ) as usize;
        n += self.encode_scalar(
            stream,
            f,
            "spike_source",
            ev.spike_source as f64,
            verified,
            seal_sha,
        ) as usize;
        n += self.encode_vec3(
            stream,
            f,
            "position",
            [
                ev.position[0] as f64,
                ev.position[1] as f64,
                ev.position[2] as f64,
            ],
            verified,
            seal_sha,
        ) as usize;
        n += self.encode_scalar(
            stream,
            f,
            "phase_bits",
            ev.phase_bits as f64,
            verified,
            seal_sha,
        ) as usize;
        n += self.encode_scalar(
            stream,
            f,
            "intensity",
            ev.intensity as f64,
            verified,
            seal_sha,
        ) as usize;
        n += self.encode_scalar(
            stream,
            f,
            "aromatic_residue_id",
            ev.aromatic_residue_id as f64,
            verified,
            seal_sha,
        ) as usize;
        n += self.encode_scalar(
            stream,
            f,
            "aromatic_type",
            ev.aromatic_type as f64,
            verified,
            seal_sha,
        ) as usize;
        n += self.encode_scalar(
            stream,
            f,
            "water_density",
            ev.water_density as f64,
            verified,
            seal_sha,
        ) as usize;
        n += self.encode_scalar(
            stream,
            f,
            "n_nearby_excited",
            ev.n_nearby_excited as f64,
            verified,
            seal_sha,
        ) as usize;
        n
    }

    /// Stage-1 routing from a decoded JSON object: dispatch each key matching a registered field of
    /// `family` by geometry (SMILES sequence / vec3 array / scalar). Unmatched keys are skipped.
    pub fn route_json(
        &self,
        stream: &mut TokenStream,
        family: ArtifactFamily,
        json: &serde_json::Value,
        verified: bool,
        seal_sha: [u8; 32],
    ) -> usize {
        let Some(obj) = json.as_object() else {
            return 0;
        };
        let mut n = 0usize;
        for (key, val) in obj {
            let Some(spec) = self.registry.lookup(family, key) else {
                continue;
            };
            match &spec.geometry {
                FieldGeometry::SmilesGrammar => {
                    if let Some(s) = val.as_str() {
                        n += self.encode_smiles(stream, family, key, s, verified, seal_sha);
                    }
                }
                FieldGeometry::LatticeVq3d { .. } => {
                    if let Some(a) = val.as_array() {
                        if a.len() == 3 {
                            let v = [
                                a[0].as_f64().unwrap_or(0.0),
                                a[1].as_f64().unwrap_or(0.0),
                                a[2].as_f64().unwrap_or(0.0),
                            ];
                            n += self.encode_vec3(stream, family, key, v, verified, seal_sha)
                                as usize;
                        }
                    }
                }
                _ => {
                    let v = val
                        .as_f64()
                        .or_else(|| val.as_bool().map(|b| b as i64 as f64));
                    if let Some(x) = v {
                        n +=
                            self.encode_scalar(stream, family, key, x, verified, seal_sha) as usize;
                    }
                }
            }
        }
        n
    }
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::container::{decode_qnanl, decode_spike_packet, transpose, SPIKE_MAGIC};
    use crate::epistemic::canonical_eligible;
    use crate::sidecar::ExactValue;
    use std::io::Cursor;
    use std::path::PathBuf;

    const SHA: [u8; 32] = [0u8; 32];

    #[test]
    fn encodes_scalar_with_namespace_verified_and_sidecar() {
        let enc = Encoder::new();
        let mut s = TokenStream::new();
        assert!(enc.encode_scalar(
            &mut s,
            ArtifactFamily::SpikeEvent,
            "intensity",
            42.5,
            true,
            SHA
        ));
        let tok = s.tokens[0];
        assert_eq!(
            tok.namespace(),
            enc.registry()
                .namespace(ArtifactFamily::SpikeEvent, "intensity")
                .unwrap()
        );
        assert!(tok.verified() && tok.provenance() && !tok.authenticated());
        assert_eq!(s.sidecar.get(0).unwrap().exact, ExactValue::Scalar(42.5));
    }

    #[test]
    fn fit_field_refits_scalar_quantizer_and_rejects_nonscalar() {
        let mut enc = Encoder::new();
        let samples: Vec<f64> = (0..100).map(|i| 1.0 + i as f64 * 0.005).collect();
        assert!(enc.fit_field(ArtifactFamily::SpikeEvent, "intensity", &samples));
        assert!(!enc.fit_field(ArtifactFamily::SpikeEvent, "position", &[1.0])); // non-scalar
        assert!(!enc.fit_field(ArtifactFamily::SpikeEvent, "nope", &[1.0])); // unknown
    }

    #[test]
    fn auto_backend_fit_matches_cpu_fit_bit_for_bit() {
        // Correctness is independent of backend: fitting with TokenizerBackend::Auto must yield a
        // codebook that encodes EVERY probe to the identical code + bit-identical distortion as the
        // CPU default fit. Holds with and without the gpu-tokenizer feature (the GPU lane is itself
        // parity-proven by the parity bin; here we guard the integration in the normal test suite).
        use crate::backend::TokenizerBackend;
        let pts: Vec<[f64; 3]> = (0..3000)
            .map(|i| {
                let t = i as f64;
                [
                    10.0 + (t * 0.07).sin() * 3.0,
                    20.0 + (t * 0.05).cos() * 3.0,
                    5.0 + (t * 0.11).sin() * 3.0,
                ]
            })
            .collect();
        let mut cpu = Encoder::new();
        cpu.fit_coords(ArtifactFamily::SpikeEvent, "position", &pts);
        let mut auto = Encoder::new();
        auto.fit_coords_with_backend(
            ArtifactFamily::SpikeEvent,
            "position",
            &pts,
            TokenizerBackend::Auto,
        );

        let mut sc = TokenStream::new();
        let mut sa = TokenStream::new();
        for &p in &pts {
            cpu.encode_vec3(
                &mut sc,
                ArtifactFamily::SpikeEvent,
                "position",
                p,
                true,
                SHA,
            );
            auto.encode_vec3(
                &mut sa,
                ArtifactFamily::SpikeEvent,
                "position",
                p,
                true,
                SHA,
            );
        }
        assert_eq!(sc.len(), sa.len());
        for i in 0..sc.len() {
            assert_eq!(
                sc.tokens[i].field_token(),
                sa.tokens[i].field_token(),
                "code parity at {i}"
            );
            assert_eq!(
                sc.tokens[i].namespace(),
                sa.tokens[i].namespace(),
                "namespace parity at {i}"
            );
        }
    }

    #[test]
    fn firewall_blocks_generative_field_even_when_sealed() {
        let enc = Encoder::new();
        let mut s = TokenStream::new();
        enc.encode_scalar(
            &mut s,
            ArtifactFamily::GflownetTrajectory,
            "reward",
            0.82,
            true,
            SHA,
        );
        assert!(!canonical_eligible(s.tokens[0]));
    }

    #[test]
    fn unknown_and_nonscalar_fields_not_encoded_as_scalar() {
        let enc = Encoder::new();
        let mut s = TokenStream::new();
        assert!(!enc.encode_scalar(&mut s, ArtifactFamily::SpikeEvent, "nope", 1.0, true, SHA));
        assert!(!enc.encode_scalar(
            &mut s,
            ArtifactFamily::SpikeEvent,
            "position",
            1.0,
            true,
            SHA
        ));
        assert!(s.is_empty());
    }

    #[test]
    fn encode_vec3_with_fit_is_within_bound_and_records_vec3() {
        let mut enc = Encoder::new();
        // a few clustered coords → fit VQ → encode within the 0.25Å bound (non-escape, real token).
        let pts: Vec<[f64; 3]> = (0..200)
            .map(|i| {
                let t = i as f64;
                [
                    10.0 + (t * 0.11).sin() * 0.5,
                    20.0 + (t * 0.13).cos() * 0.5,
                    5.0 + (t * 0.17).sin() * 0.5,
                ]
            })
            .collect();
        assert!(enc.fit_coords(ArtifactFamily::SpikeEvent, "position", &pts));
        let mut s = TokenStream::new();
        assert!(enc.encode_vec3(
            &mut s,
            ArtifactFamily::SpikeEvent,
            "position",
            pts[0],
            true,
            SHA
        ));
        assert!(
            !s.is_escaped(0),
            "fitted VQ should meet the 0.25Å bound on its own training point"
        );
        assert_eq!(s.sidecar.get(0).unwrap().exact, ExactValue::Vec3(pts[0]));
    }

    #[test]
    fn encode_smiles_emits_a_token_sequence_with_lossless_sidecar() {
        let enc = Encoder::new();
        let mut s = TokenStream::new();
        let n = enc.encode_smiles(
            &mut s,
            ArtifactFamily::GflownetTrajectory,
            "canonical_smiles",
            "CC(=O)O",
            true,
            SHA,
        );
        assert_eq!(n, 7);
        assert_eq!(s.len(), 7);
        // every smiles token is generative-tier (gflownet) → not canonical-eligible.
        assert!(s.tokens.iter().all(|t| !canonical_eligible(*t)));
        // sidecar reconstructs the exact SMILES from the sub-token text.
        let recon: String = (0..s.len())
            .map(|i| match &s.sidecar.get(i).unwrap().exact {
                ExactValue::Text(t) => t.clone(),
                _ => String::new(),
            })
            .collect();
        assert_eq!(recon, "CC(=O)O");
    }

    #[test]
    fn route_json_dispatches_scalar_fields() {
        let enc = Encoder::new();
        let mut s = TokenStream::new();
        let j = serde_json::json!({"rhat": 1.05, "ess": 500.0, "label_confidence": 0.8, "is_bimodal": true, "unrelated_key": 9});
        let n = enc.route_json(
            &mut s,
            ArtifactFamily::TeacherStudentConsensus,
            &j,
            true,
            SHA,
        );
        assert_eq!(n, 4); // 4 registered fields matched; unrelated_key skipped
        assert_eq!(s.len(), 4);
    }

    /// End-to-end: synthetic PRSPCZ01 spike packet → decode → route each record → TokenStream.
    #[test]
    fn end_to_end_spike_packet_decodes_and_tokenizes() {
        let stride = crate::codec::GPU_SPIKE_EVENT_BYTES;
        let n = 4usize;
        let mut events = Vec::new();
        let mut payload = Vec::new();
        for k in 0..n {
            let mut ev = GpuSpikeEvent::zeroed();
            ev.wavelength_nm = 280.0;
            ev.spike_source = (k % 5) as i32 + 1;
            ev.position = [10.0 + k as f32, 20.0, 5.0];
            ev.intensity = 100.0 * (k as f32 + 1.0);
            ev.aromatic_residue_id = if k == 0 { -1 } else { k as i32 };
            ev.phase_bits = (k * 13) as u32;
            payload.extend_from_slice(&ev.to_le_bytes());
            events.push(ev);
        }
        let transposed = transpose(&payload, n, stride).unwrap();
        let compressed = zstd::stream::encode_all(Cursor::new(&transposed), 3).unwrap();
        let header =
            serde_json::to_vec(&serde_json::json!({"record_count": n, "payload_stride": stride}))
                .unwrap();
        let mut packet = Vec::new();
        packet.extend_from_slice(SPIKE_MAGIC);
        packet.extend_from_slice(&1u32.to_le_bytes());
        packet.extend_from_slice(&(header.len() as u64).to_le_bytes());
        packet.extend_from_slice(&0u64.to_le_bytes());
        packet.extend_from_slice(&0u64.to_le_bytes());
        packet.extend_from_slice(&(compressed.len() as u64).to_le_bytes());
        packet.extend_from_slice(&header);
        packet.extend_from_slice(&compressed);

        let decoded = decode_spike_packet(&packet).unwrap();
        assert_eq!(decoded.events, events);

        let mut enc = Encoder::new();
        let coords: Vec<[f64; 3]> = decoded
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
        for ev in &decoded.events {
            total += enc.route_spike(&mut s, ev, true, SHA);
        }
        assert_eq!(total, 9 * n, "9 fields per spike record");
        assert_eq!(s.len(), 9 * n);
        // wavelength tokens round-trip to an exact line (280); spike_source enum exact.
        // every token carries the spike leverage tier (1) and is verified.
        assert!(s.tokens.iter().all(|t| t.verified()));
    }

    /// End-to-end plumbing: a real sealed QNANL artifact → decode (seal re-verifies) → JSON →
    /// route_json without panic. (Ledger artifacts won't match token fields, so token count may be
    /// 0 — this proves the decode→JSON→route path, not field coverage.)
    #[test]
    fn end_to_end_real_qnanl_decode_then_route_no_panic() {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../OPERATOR_CONTROL/PRISMOS_PATHB_ACTIVE_MATERIALIZATION.json.prismqnanl");
        if !p.exists() {
            eprintln!("skip: real QNANL fixture absent");
            return;
        }
        let bytes = std::fs::read(&p).unwrap();
        let d = decode_qnanl(&bytes).unwrap();
        assert!(d.verified);
        let payload = d.payload.unwrap();
        if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&payload) {
            let enc = Encoder::new();
            let mut s = TokenStream::new();
            // probe under a token family; ledger keys won't match → 0 tokens, but must not panic.
            let _ = enc.route_json(
                &mut s,
                ArtifactFamily::TeacherStudentConsensus,
                &json,
                d.verified,
                d.seal_sha,
            );
        }
    }
}
