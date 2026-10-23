use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub type NanlDictResult<T> = Result<T, NanlDictError>;

pub const NANL_DICT_MAGIC: &str = "NANLDICT";
pub const NANL_DICT_VERSION: &str = "0.1";

#[derive(Debug, thiserror::Error)]
pub enum NanlDictError {
    #[error("PC_ERR_DICT_FORMAT:{0}")]
    Format(String),
    #[error("PC_ERR_DICT_HASH:{0}")]
    Hash(String),
    #[error("PC_ERR_DICT_ORDERING:{0}")]
    Ordering(String),
    #[error("PC_ERR_UNKNOWN_TOKEN:{0}")]
    UnknownToken(String),
    #[error("PC_ERR_INTERNAL:{0}")]
    Internal(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum NanlDictionaryClass {
    Entity,
    AtomType,
    Residue,
    EventKind,
    EventMotif,
    WaterGate,
    Variant,
    ThermoQuantity,
    UncertaintyModel,
    TeacherLabel,
    GFlowNetAction,
    NrfTile,
    DrfChannel,
    NanlSymbol,
}

impl NanlDictionaryClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Entity => "ENTITY",
            Self::AtomType => "ATOM_TYPE",
            Self::Residue => "RESIDUE",
            Self::EventKind => "EVENT_KIND",
            Self::EventMotif => "EVENT_MOTIF",
            Self::WaterGate => "WATER_GATE",
            Self::Variant => "VARIANT",
            Self::ThermoQuantity => "THERMO_QUANTITY",
            Self::UncertaintyModel => "UNCERTAINTY_MODEL",
            Self::TeacherLabel => "TEACHER_LABEL",
            Self::GFlowNetAction => "GFLOWNET_ACTION",
            Self::NrfTile => "NRF_TILE",
            Self::DrfChannel => "DRF_CHANNEL",
            Self::NanlSymbol => "NANL_SYMBOL",
        }
    }
}

pub const REQUIRED_DICTIONARY_CLASSES: [NanlDictionaryClass; 14] = [
    NanlDictionaryClass::Entity,
    NanlDictionaryClass::AtomType,
    NanlDictionaryClass::Residue,
    NanlDictionaryClass::EventKind,
    NanlDictionaryClass::EventMotif,
    NanlDictionaryClass::WaterGate,
    NanlDictionaryClass::Variant,
    NanlDictionaryClass::ThermoQuantity,
    NanlDictionaryClass::UncertaintyModel,
    NanlDictionaryClass::TeacherLabel,
    NanlDictionaryClass::GFlowNetAction,
    NanlDictionaryClass::NrfTile,
    NanlDictionaryClass::DrfChannel,
    NanlDictionaryClass::NanlSymbol,
];

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NanlDictionarySymbol {
    pub symbol_class: NanlDictionaryClass,
    pub source_token_ref: u64,
    pub token: String,
    pub frequency: u64,
    pub first_seen_ref: u64,
    pub flags: u32,
}

impl NanlDictionarySymbol {
    pub fn new(
        symbol_class: NanlDictionaryClass,
        source_token_ref: u64,
        token: impl Into<String>,
        frequency: u64,
        first_seen_ref: u64,
    ) -> Self {
        Self {
            symbol_class,
            source_token_ref,
            token: token.into(),
            frequency,
            first_seen_ref,
            flags: 0,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NanlDictHeader {
    pub magic: String,
    pub version: String,
    pub dictionary_kind: String,
    pub symbol_count: u64,
    pub canonical_source_hash: String,
    pub dictionary_hash: String,
    pub reverse_lookup_hash: String,
    pub provenance_ref: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NanlDictEntry {
    pub symbol_id: u64,
    pub symbol_class: NanlDictionaryClass,
    pub source_token_ref: u64,
    pub token: String,
    pub encoded_value: u64,
    pub frequency: u64,
    pub first_seen_ref: u64,
    pub flags: u32,
    pub entry_hash: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NanlDictionary {
    pub header: NanlDictHeader,
    pub entries: Vec<NanlDictEntry>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EncodedNanlDictToken {
    pub symbol_id: u64,
    pub encoded_value: u64,
    pub symbol_class: NanlDictionaryClass,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NanlDictValidationReport {
    pub status: String,
    pub dictionary_hash: String,
    pub reverse_lookup_hash: String,
    pub symbol_count: u64,
    pub stable_ordering: bool,
    pub required_classes_present: bool,
    pub lookup_table_valid: bool,
    pub reverse_lookup_valid: bool,
    pub issues: Vec<String>,
}

impl NanlDictionary {
    pub fn build(
        dictionary_kind: impl Into<String>,
        canonical_source_hash: impl Into<String>,
        provenance_ref: u64,
        symbols: &[NanlDictionarySymbol],
    ) -> NanlDictResult<Self> {
        let dictionary_kind = dictionary_kind.into();
        let canonical_source_hash = canonical_source_hash.into();
        if canonical_source_hash.trim().is_empty() {
            return Err(NanlDictError::Format(
                "NANL-DICT missing canonical source hash".to_string(),
            ));
        }
        if symbols.is_empty() {
            return Err(NanlDictError::Format(
                "NANL-DICT requires at least one typed symbol".to_string(),
            ));
        }

        let mut aggregate: BTreeMap<(NanlDictionaryClass, String), AggregateSymbol> =
            BTreeMap::new();
        for symbol in symbols {
            let token = symbol.token.trim();
            if token.is_empty() {
                return Err(NanlDictError::Format(
                    "NANL-DICT refuses empty runtime token".to_string(),
                ));
            }
            let key = (symbol.symbol_class, token.to_string());
            aggregate
                .entry(key)
                .and_modify(|item| {
                    item.source_token_ref = item.source_token_ref.min(symbol.source_token_ref);
                    item.frequency = item.frequency.saturating_add(symbol.frequency.max(1));
                    item.first_seen_ref = item.first_seen_ref.min(symbol.first_seen_ref);
                    item.flags |= symbol.flags;
                })
                .or_insert(AggregateSymbol {
                    source_token_ref: symbol.source_token_ref,
                    frequency: symbol.frequency.max(1),
                    first_seen_ref: symbol.first_seen_ref,
                    flags: symbol.flags,
                });
        }

        let mut entries = Vec::with_capacity(aggregate.len());
        for ((symbol_class, token), item) in aggregate {
            let symbol_id = entries.len() as u64 + 1;
            let mut entry = NanlDictEntry {
                symbol_id,
                symbol_class,
                source_token_ref: item.source_token_ref,
                token,
                encoded_value: symbol_id,
                frequency: item.frequency,
                first_seen_ref: item.first_seen_ref,
                flags: item.flags,
                entry_hash: String::new(),
            };
            entry.entry_hash = compute_entry_hash(&entry)?;
            entries.push(entry);
        }

        let reverse_lookup_hash = compute_reverse_lookup_hash(&entries)?;
        let dictionary_hash = compute_dictionary_hash(
            &dictionary_kind,
            &canonical_source_hash,
            &reverse_lookup_hash,
            provenance_ref,
            &entries,
        )?;
        let header = NanlDictHeader {
            magic: NANL_DICT_MAGIC.to_string(),
            version: NANL_DICT_VERSION.to_string(),
            dictionary_kind,
            symbol_count: entries.len() as u64,
            canonical_source_hash,
            dictionary_hash,
            reverse_lookup_hash,
            provenance_ref,
        };
        let dictionary = Self { header, entries };
        dictionary.fail_if_invalid()?;
        Ok(dictionary)
    }

    pub fn lookup(
        &self,
        symbol_class: NanlDictionaryClass,
        token: &str,
    ) -> NanlDictResult<&NanlDictEntry> {
        self.entries
            .iter()
            .find(|entry| entry.symbol_class == symbol_class && entry.token == token)
            .ok_or_else(|| {
                NanlDictError::UnknownToken(format!("{}:{token}", symbol_class.as_str()))
            })
    }

    pub fn reverse_lookup(&self, symbol_id: u64) -> NanlDictResult<&NanlDictEntry> {
        self.entries
            .iter()
            .find(|entry| entry.symbol_id == symbol_id)
            .ok_or_else(|| NanlDictError::UnknownToken(format!("symbol_id:{symbol_id}")))
    }

    pub fn encode_tokens(
        &self,
        tokens: &[(NanlDictionaryClass, String)],
    ) -> NanlDictResult<Vec<EncodedNanlDictToken>> {
        tokens
            .iter()
            .map(|(symbol_class, token)| {
                let entry = self.lookup(*symbol_class, token)?;
                Ok(EncodedNanlDictToken {
                    symbol_id: entry.symbol_id,
                    encoded_value: entry.encoded_value,
                    symbol_class: entry.symbol_class,
                })
            })
            .collect()
    }

    pub fn decode_tokens(&self, tokens: &[EncodedNanlDictToken]) -> NanlDictResult<Vec<String>> {
        tokens
            .iter()
            .map(|token| {
                let entry = self.reverse_lookup(token.symbol_id)?;
                if entry.encoded_value != token.encoded_value
                    || entry.symbol_class != token.symbol_class
                {
                    return Err(NanlDictError::UnknownToken(format!(
                        "encoded token mismatch for symbol_id {}",
                        token.symbol_id
                    )));
                }
                Ok(entry.token.clone())
            })
            .collect()
    }

    pub fn validate(&self) -> NanlDictResult<NanlDictValidationReport> {
        let mut issues = Vec::new();

        if self.header.magic != NANL_DICT_MAGIC {
            issues.push("NANL-DICT bad magic".to_string());
        }
        if self.header.version != NANL_DICT_VERSION {
            issues.push("NANL-DICT unsupported version".to_string());
        }
        if self.header.canonical_source_hash.trim().is_empty() {
            issues.push("NANL-DICT missing canonical source hash".to_string());
        }
        if self.header.symbol_count != self.entries.len() as u64 {
            issues.push("NANL-DICT symbol count mismatch".to_string());
        }

        let stable_ordering = entries_are_stably_ordered(&self.entries);
        if !stable_ordering {
            issues.push("NANL-DICT entries violate stable ordering".to_string());
        }

        let mut seen_ids = BTreeSet::new();
        let mut seen_tokens = BTreeSet::new();
        let mut lookup_table_valid = true;
        let mut reverse_lookup_valid = true;
        for (idx, entry) in self.entries.iter().enumerate() {
            if entry.symbol_id != idx as u64 + 1 {
                reverse_lookup_valid = false;
                issues.push(format!(
                    "NANL-DICT non-contiguous symbol id {} at index {}",
                    entry.symbol_id, idx
                ));
            }
            if !seen_ids.insert(entry.symbol_id) {
                reverse_lookup_valid = false;
                issues.push(format!("NANL-DICT duplicate symbol id {}", entry.symbol_id));
            }
            if entry.token.trim().is_empty() {
                lookup_table_valid = false;
                issues.push("NANL-DICT empty token in lookup table".to_string());
            }
            if !seen_tokens.insert((entry.symbol_class, entry.token.clone())) {
                lookup_table_valid = false;
                issues.push(format!(
                    "NANL-DICT duplicate token {}:{}",
                    entry.symbol_class.as_str(),
                    entry.token
                ));
            }
            let expected_hash = compute_entry_hash(entry)?;
            if entry.entry_hash != expected_hash {
                issues.push(format!(
                    "NANL-DICT bad entry hash for symbol_id {}",
                    entry.symbol_id
                ));
            }
        }

        let reverse_lookup_hash = compute_reverse_lookup_hash(&self.entries)?;
        if self.header.reverse_lookup_hash != reverse_lookup_hash {
            reverse_lookup_valid = false;
            issues.push("NANL-DICT reverse lookup hash mismatch".to_string());
        }

        let dictionary_hash = compute_dictionary_hash(
            &self.header.dictionary_kind,
            &self.header.canonical_source_hash,
            &self.header.reverse_lookup_hash,
            self.header.provenance_ref,
            &self.entries,
        )?;
        if self.header.dictionary_hash != dictionary_hash {
            issues.push("NANL-DICT dictionary hash mismatch".to_string());
        }

        let required_classes_present = REQUIRED_DICTIONARY_CLASSES.iter().all(|required| {
            self.entries
                .iter()
                .any(|entry| entry.symbol_class == *required)
        });
        if !required_classes_present {
            issues.push(
                "NANL-DICT missing one or more required PRISM dictionary classes".to_string(),
            );
        }

        Ok(NanlDictValidationReport {
            status: if issues.is_empty() { "PASS" } else { "FAIL" }.to_string(),
            dictionary_hash,
            reverse_lookup_hash,
            symbol_count: self.entries.len() as u64,
            stable_ordering,
            required_classes_present,
            lookup_table_valid,
            reverse_lookup_valid,
            issues,
        })
    }

    pub fn fail_if_invalid(&self) -> NanlDictResult<()> {
        let report = self.validate()?;
        if report.status != "PASS" {
            return Err(classify_validation_issues(report.issues));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AggregateSymbol {
    source_token_ref: u64,
    frequency: u64,
    first_seen_ref: u64,
    flags: u32,
}

#[derive(Serialize)]
struct EntryHashMaterial<'a> {
    symbol_id: u64,
    symbol_class: NanlDictionaryClass,
    source_token_ref: u64,
    token: &'a str,
    encoded_value: u64,
    frequency: u64,
    first_seen_ref: u64,
    flags: u32,
}

#[derive(Serialize)]
struct ReverseLookupHashMaterial<'a> {
    symbol_id: u64,
    symbol_class: NanlDictionaryClass,
    token: &'a str,
    encoded_value: u64,
}

#[derive(Serialize)]
struct DictionaryHashMaterial<'a> {
    magic: &'a str,
    version: &'a str,
    dictionary_kind: &'a str,
    symbol_count: u64,
    canonical_source_hash: &'a str,
    reverse_lookup_hash: &'a str,
    provenance_ref: u64,
    entries: &'a [NanlDictEntry],
}

fn compute_entry_hash(entry: &NanlDictEntry) -> NanlDictResult<String> {
    hash_json(&EntryHashMaterial {
        symbol_id: entry.symbol_id,
        symbol_class: entry.symbol_class,
        source_token_ref: entry.source_token_ref,
        token: &entry.token,
        encoded_value: entry.encoded_value,
        frequency: entry.frequency,
        first_seen_ref: entry.first_seen_ref,
        flags: entry.flags,
    })
}

fn compute_reverse_lookup_hash(entries: &[NanlDictEntry]) -> NanlDictResult<String> {
    let material = entries
        .iter()
        .map(|entry| ReverseLookupHashMaterial {
            symbol_id: entry.symbol_id,
            symbol_class: entry.symbol_class,
            token: &entry.token,
            encoded_value: entry.encoded_value,
        })
        .collect::<Vec<_>>();
    hash_json(&material)
}

fn compute_dictionary_hash(
    dictionary_kind: &str,
    canonical_source_hash: &str,
    reverse_lookup_hash: &str,
    provenance_ref: u64,
    entries: &[NanlDictEntry],
) -> NanlDictResult<String> {
    hash_json(&DictionaryHashMaterial {
        magic: NANL_DICT_MAGIC,
        version: NANL_DICT_VERSION,
        dictionary_kind,
        symbol_count: entries.len() as u64,
        canonical_source_hash,
        reverse_lookup_hash,
        provenance_ref,
        entries,
    })
}

fn entries_are_stably_ordered(entries: &[NanlDictEntry]) -> bool {
    entries
        .windows(2)
        .all(|window| stable_entry_key(&window[0]) < stable_entry_key(&window[1]))
}

fn stable_entry_key(entry: &NanlDictEntry) -> (NanlDictionaryClass, &str) {
    (entry.symbol_class, entry.token.as_str())
}

fn classify_validation_issues(issues: Vec<String>) -> NanlDictError {
    let text = issues.join(";");
    if text.contains("stable ordering") || text.contains("non-contiguous") {
        NanlDictError::Ordering(text)
    } else if text.contains("hash") {
        NanlDictError::Hash(text)
    } else {
        NanlDictError::Format(text)
    }
}

fn hash_json<T: Serialize>(value: &T) -> NanlDictResult<String> {
    let bytes =
        serde_json::to_vec(value).map_err(|err| NanlDictError::Internal(err.to_string()))?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}
