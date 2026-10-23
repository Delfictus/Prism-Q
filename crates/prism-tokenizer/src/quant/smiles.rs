//! SMILES sub-tokenizer (§7) for `canonical_smiles` — an atom/bond grammar split, NEVER BPE
//! (do-not §18). Produces a token sequence (bracket atoms, two-letter organics, bonds, branches,
//! ring closures), then maps to a fixed vocab (unknown → UNK).

/// Split a SMILES string into grammar tokens (bracketed atoms / two-letter organics / `%nn`
/// ring bonds / single chars). Order-preserving and deterministic.
pub fn tokenize(smiles: &str) -> Vec<String> {
    let b = smiles.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < b.len() {
        let c = b[i] as char;
        if c == '[' {
            // bracketed atom: consume through ']'
            let start = i;
            i += 1;
            while i < b.len() && b[i] != b']' {
                i += 1;
            }
            if i < b.len() {
                i += 1; // include ']'
            }
            out.push(smiles[start..i].to_string());
        } else if c == '%'
            && i + 2 < b.len()
            && b[i + 1].is_ascii_digit()
            && b[i + 2].is_ascii_digit()
        {
            out.push(smiles[i..i + 3].to_string()); // two-digit ring bond
            i += 3;
        } else if (c == 'C' && i + 1 < b.len() && b[i + 1] == b'l')
            || (c == 'B' && i + 1 < b.len() && b[i + 1] == b'r')
        {
            out.push(smiles[i..i + 2].to_string()); // Cl / Br
            i += 2;
        } else {
            out.push(c.to_string());
            i += 1;
        }
    }
    out
}

/// Decompose a bracket atom (`[C@@H+]`, `[nH]`, `[O-]`, `[13C]`) into its grammar components so
/// two different stereocenters/charges become DIFFERENT model-visible code sequences (P2-13),
/// instead of all collapsing to UNK. The concatenation of the components reconstructs the atom.
pub fn decompose_bracket(tok: &str) -> Vec<String> {
    if !(tok.starts_with('[') && tok.ends_with(']')) {
        return vec![tok.to_string()];
    }
    let inner = &tok[1..tok.len() - 1];
    let b = inner.as_bytes();
    let mut out = vec!["[".to_string()];
    let mut i = 0;
    while i < b.len() {
        let c = b[i] as char;
        if c == '@' {
            if i + 1 < b.len() && b[i + 1] == b'@' {
                out.push("@@".into());
                i += 2;
            } else {
                out.push("@".into());
                i += 1;
            }
        } else if c.is_ascii_uppercase() {
            // element: 2-letter (Cl, Br, Se, …) if next is lowercase, else 1-letter.
            if i + 1 < b.len() && (b[i + 1] as char).is_ascii_lowercase() {
                out.push(inner[i..i + 2].to_string());
                i += 2;
            } else {
                out.push(c.to_string());
                i += 1;
            }
        } else {
            // digits (isotope/charge count/Hcount), aromatic lowercase, +, -, :, etc. — one each.
            out.push(c.to_string());
            i += 1;
        }
    }
    out.push("]".to_string());
    out
}

/// Like `tokenize`, but bracket atoms are decomposed into their components (P2-13).
pub fn tokenize_deep(smiles: &str) -> Vec<String> {
    let mut out = Vec::new();
    for t in tokenize(smiles) {
        if t.starts_with('[') {
            out.extend(decompose_bracket(&t));
        } else {
            out.push(t);
        }
    }
    out
}

/// Fixed SMILES vocabulary (organic subset atoms, aromatics, bonds, structure, ring digits,
/// and bracket-internal grammar). Code 0 = UNK; deterministic ordering.
pub struct SmilesVocab {
    table: std::collections::HashMap<String, u32>,
}

const SMILES_TOKENS: &[&str] = &[
    // organic subset
    "C", "N", "O", "S", "P", "F", "I", "B", "Cl", "Br", "H", // aromatic
    "c", "n", "o", "s", "p", "b", // bonds
    "-", "=", "#", "$", ":", "/", "\\", ".", // structure
    "(", ")", // ring digits
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    // bracket-internal grammar (P2-13); "-" already covered by bonds
    "[", "]", "@", "@@", "+",
];

impl SmilesVocab {
    pub fn new() -> Self {
        let mut table = std::collections::HashMap::new();
        // 0 reserved for UNK; real tokens start at 1.
        for (i, t) in SMILES_TOKENS.iter().enumerate() {
            table.insert((*t).to_string(), (i + 1) as u32);
        }
        Self { table }
    }

    pub const UNK: u32 = 0;

    /// Token → code (UNK for anything not in the fixed vocab, e.g. bracketed atoms).
    pub fn code(&self, token: &str) -> u32 {
        *self.table.get(token).unwrap_or(&Self::UNK)
    }

    /// Tokenize + map to codes.
    pub fn encode(&self, smiles: &str) -> Vec<u32> {
        tokenize(smiles).iter().map(|t| self.code(t)).collect()
    }

    pub fn size(&self) -> usize {
        self.table.len() + 1 // + UNK
    }
}

impl Default for SmilesVocab {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_atoms_bonds_branches() {
        assert_eq!(tokenize("CC(=O)O"), vec!["C", "C", "(", "=", "O", ")", "O"]);
    }

    #[test]
    fn two_letter_organics_and_brackets_are_single_tokens() {
        assert_eq!(
            tokenize("[C@@H](Cl)Br"),
            vec!["[C@@H]", "(", "Cl", ")", "Br"]
        );
    }

    #[test]
    fn aromatic_ring_and_ring_digits() {
        assert_eq!(
            tokenize("c1ccccc1"),
            vec!["c", "1", "c", "c", "c", "c", "c", "1"]
        );
    }

    #[test]
    fn two_digit_ring_bond() {
        assert_eq!(tokenize("C%12"), vec!["C", "%12"]);
    }

    #[test]
    fn bracket_decompose_distinguishes_stereo_and_reconstructs() {
        // P2-13: bracket atoms decompose into components → distinct model-visible code sequences.
        assert_eq!(tokenize_deep("[C@@H]"), vec!["[", "C", "@@", "H", "]"]);
        assert_eq!(tokenize_deep("[C@H]"), vec!["[", "C", "@", "H", "]"]);
        let v = SmilesVocab::new();
        let codes = |s: &str| -> Vec<u32> { tokenize_deep(s).iter().map(|t| v.code(t)).collect() };
        assert_ne!(
            codes("[C@@H]"),
            codes("[C@H]"),
            "stereocenters must differ in model codes"
        );
        assert!(
            codes("[C@@H]").iter().all(|&c| c != SmilesVocab::UNK),
            "no component is UNK now"
        );
        assert_eq!(
            tokenize_deep("[C@@H]").concat(),
            "[C@@H]",
            "components reconstruct the atom"
        );
        // non-bracket SMILES is unchanged.
        assert_eq!(tokenize_deep("CC(=O)O"), tokenize("CC(=O)O"));
    }

    #[test]
    fn vocab_maps_known_and_unk() {
        let v = SmilesVocab::new();
        assert_ne!(v.code("C"), SmilesVocab::UNK);
        assert_ne!(v.code("Cl"), v.code("C"));
        assert_eq!(v.code("[C@@H]"), SmilesVocab::UNK); // bracket atom → UNK in the fixed vocab
                                                        // benzene: 8 tokens, all known.
        let codes = v.encode("c1ccccc1");
        assert_eq!(codes.len(), 8);
        assert!(codes.iter().all(|&c| c != SmilesVocab::UNK));
    }
}
