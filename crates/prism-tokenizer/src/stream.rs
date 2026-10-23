//! §4/§9 — TokenStream assembly. The model-visible token vector plus the isolated sidecar
//! (INV-4). Segment table (§9 BOCPD boundaries) is added in M4.

use crate::epistemic::leverage_tier;
use crate::sidecar::{ExactValue, Sidecar, SidecarEntry};
use crate::types::{pack_namespace, ArtifactFamily, FieldSpec, Quantizer, Token};

/// A contiguous run of tokens belonging to one decoded record (§9 grouping; full segment table in
/// M4). Lets multi-record streams (e.g. 89k spikes) be grouped + reconstructed per record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Segment {
    pub start: usize,
    pub len: usize,
    pub record_index: usize,
    pub family: ArtifactFamily,
}

#[derive(Default)]
pub struct TokenStream {
    /// Model-visible packed tokens.
    pub tokens: Vec<Token>,
    /// Isolated audit/reconstruction store (NOT model-visible).
    pub sidecar: Sidecar,
    /// Per-record token groupings.
    pub segments: Vec<Segment>,
}

impl TokenStream {
    pub fn new() -> Self {
        Self::default()
    }

    /// Low-level: append a token (epistemic bits from family/verified, INV-2 namespace) and record
    /// its exact value in the sidecar (INV-4). `authenticated` defaults 0 (INV-9). Returns it.
    pub fn push(
        &mut self,
        spec: &FieldSpec,
        namespace: u16,
        field_token: u64,
        verified: bool,
        exact: ExactValue,
        seal_sha: [u8; 32],
    ) -> Token {
        let pos = self.tokens.len();
        let tok = Token::new(
            leverage_tier(spec.family),
            verified,
            true,
            namespace,
            field_token,
        );
        self.tokens.push(tok);
        self.sidecar.record(SidecarEntry {
            position: pos,
            field: spec.field,
            exact,
            seal_sha,
        });
        tok
    }

    /// Quantize one scalar field value → token + sidecar entry. INV-7: if incurred distortion
    /// exceeds the field's declared bound, emit the escape code (the exact value is in the sidecar).
    pub fn push_scalar(
        &mut self,
        spec: &FieldSpec,
        namespace: u16,
        quantizer: &dyn Quantizer,
        value: f64,
        verified: bool,
        seal_sha: [u8; 32],
    ) -> Token {
        let (code, dist) = quantizer.encode(value);
        let field_token = if spec.max_distortion.allows(dist.value(), value) {
            code as u64
        } else {
            Token::ESCAPE_CODE
        };
        self.push(
            spec,
            namespace,
            field_token,
            verified,
            ExactValue::Scalar(value),
            seal_sha,
        )
    }

    /// A 3-D coordinate token (lattice code) + exact Vec3 in the sidecar. `code`/`dist` come from
    /// the field's lattice quantizer; INV-7 escape applies on the Euclidean bound.
    pub fn push_vec3(
        &mut self,
        spec: &FieldSpec,
        namespace: u16,
        code: u32,
        dist: f64,
        value: [f64; 3],
        verified: bool,
        seal_sha: [u8; 32],
    ) -> Token {
        // P2-15: pass the vector magnitude (not 0.0) so a RelativeFraction bound, if ever set on a
        // coordinate field, is evaluated correctly instead of always escaping.
        let mag = (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt();
        let field_token = if spec.max_distortion.allows(dist, mag) {
            code as u64
        } else {
            Token::ESCAPE_CODE
        };
        self.push(
            spec,
            namespace,
            field_token,
            verified,
            ExactValue::Vec3(value),
            seal_sha,
        )
    }

    /// One token of a SMILES sub-token sequence: the vocab `code` is model-visible; the exact
    /// sub-token text is in the sidecar (so bracket atoms / UNK reconstruct losslessly).
    pub fn push_smiles_token(
        &mut self,
        spec: &FieldSpec,
        namespace: u16,
        code: u32,
        sub_token: String,
        verified: bool,
        seal_sha: [u8; 32],
    ) -> Token {
        self.push(
            spec,
            namespace,
            code as u64,
            verified,
            ExactValue::Text(sub_token),
            seal_sha,
        )
    }

    /// Open a record segment; returns its id. Pair with `end_record`.
    pub fn begin_record(&mut self, family: ArtifactFamily, record_index: usize) -> usize {
        self.segments.push(Segment {
            start: self.tokens.len(),
            len: 0,
            record_index,
            family,
        });
        self.segments.len() - 1
    }

    /// Close a record segment (records its token span).
    pub fn end_record(&mut self, seg: usize) {
        if let Some(s) = self.segments.get_mut(seg) {
            s.len = self.tokens.len() - s.start;
        }
    }

    /// Capture an out-of-schema / unknown field losslessly (INV-1/INV-4 — never silently dropped):
    /// an Opaque escape token (leverage tier max, never canonical-eligible) whose exact `key=value`
    /// JSON lives in the sidecar so the whole record reconstructs.
    pub fn push_opaque(
        &mut self,
        key: &str,
        val_json: &str,
        verified: bool,
        seal_sha: [u8; 32],
    ) -> Token {
        let pos = self.tokens.len();
        let ns = pack_namespace(ArtifactFamily::Opaque.family_slot(), 0);
        let tok = Token::new(
            leverage_tier(ArtifactFamily::Opaque),
            verified,
            true,
            ns,
            Token::ESCAPE_CODE,
        );
        self.tokens.push(tok);
        self.sidecar.record(SidecarEntry {
            position: pos,
            field: "__unknown__",
            exact: ExactValue::Text(format!("{key}={val_json}")),
            seal_sha,
        });
        tok
    }

    /// True if the token at `pos` was escaped to the sidecar (INV-7).
    pub fn is_escaped(&self, pos: usize) -> bool {
        self.tokens
            .get(pos)
            .map(|t| t.field_token() == Token::ESCAPE_CODE)
            .unwrap_or(false)
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}
