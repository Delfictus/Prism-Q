//! §12 — reversibility & provenance. The sidecar holds the EXACT decoded value + originating
//! seal sha for every quantized token (INV-4: the token stream is a lossy view; the sidecar is
//! truth). It is ISOLATED: an audit/reconstruction store, never part of the model-visible stream,
//! and the seal/keying material is read-only to the tokenizer (airlock fail-closed posture).

use std::collections::HashMap;

/// The exact decoded value behind a token. Scalars, 3-vectors (coordinates), and text
/// (SMILES sub-tokens, bracket atoms) all round-trip losslessly from here regardless of how
/// lossy the model-visible token is.
#[derive(Clone, Debug, PartialEq)]
pub enum ExactValue {
    Scalar(f64),
    Vec3([f64; 3]),
    Text(String),
}

/// One exact-value record, keyed by its token position in the stream.
#[derive(Clone, Debug, PartialEq)]
pub struct SidecarEntry {
    pub position: usize,
    pub field: &'static str,
    pub exact: ExactValue,
    /// BLAKE3 of the originating seal (provenance; → here, never into vocabulary).
    pub seal_sha: [u8; 32],
}

#[derive(Default)]
pub struct Sidecar {
    entries: Vec<SidecarEntry>,
    by_pos: HashMap<usize, usize>,
}

impl Sidecar {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, entry: SidecarEntry) {
        let idx = self.entries.len();
        self.by_pos.insert(entry.position, idx);
        self.entries.push(entry);
    }

    pub fn get(&self, position: usize) -> Option<&SidecarEntry> {
        self.by_pos.get(&position).map(|&i| &self.entries[i])
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn iter(&self) -> impl Iterator<Item = &SidecarEntry> {
        self.entries.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_and_retrieves_exact_values_by_position() {
        let mut s = Sidecar::new();
        s.record(SidecarEntry {
            position: 0,
            field: "intensity",
            exact: ExactValue::Scalar(42.123),
            seal_sha: [7u8; 32],
        });
        s.record(SidecarEntry {
            position: 1,
            field: "position",
            exact: ExactValue::Vec3([1.0, 2.0, 3.0]),
            seal_sha: [9u8; 32],
        });
        s.record(SidecarEntry {
            position: 2,
            field: "canonical_smiles",
            exact: ExactValue::Text("[C@@H]".into()),
            seal_sha: [9u8; 32],
        });
        assert_eq!(s.len(), 3);
        assert_eq!(s.get(0).unwrap().exact, ExactValue::Scalar(42.123));
        assert_eq!(s.get(1).unwrap().exact, ExactValue::Vec3([1.0, 2.0, 3.0]));
        assert_eq!(s.get(2).unwrap().exact, ExactValue::Text("[C@@H]".into()));
        assert!(s.get(99).is_none());
    }
}
