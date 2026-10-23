use crate::cse_tables::ScientificStateSource;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CseDictionaries {
    pub schema_version: String,
    pub entity_dictionary: Vec<String>,
    pub atom_type_dictionary: Vec<String>,
    pub residue_dictionary: Vec<String>,
    pub event_kind_dictionary: Vec<String>,
    pub event_motif_dictionary: Vec<String>,
    pub water_gate_dictionary: Vec<String>,
    pub variant_dictionary: Vec<String>,
    pub thermo_quantity_dictionary: Vec<String>,
    pub unit_dictionary: Vec<String>,
    pub uncertainty_model_dictionary: Vec<String>,
    pub teacher_label_dictionary: Vec<String>,
    pub gflownet_action_dictionary: Vec<String>,
    pub nrf_tile_dictionary: Vec<String>,
    pub drf_channel_dictionary: Vec<String>,
    pub nanl_symbol_dictionary: Vec<String>,
    pub provenance_dictionary: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CseDictionaryKind {
    Entity,
    AtomType,
    Residue,
    EventKind,
    EventMotif,
    WaterGate,
    Variant,
    ThermoQuantity,
    Unit,
    UncertaintyModel,
    TeacherLabel,
    GFlowNetAction,
    NrfTile,
    DrfChannel,
    NanlSymbol,
    Provenance,
}

impl CseDictionaries {
    pub fn build(source: &ScientificStateSource) -> Self {
        let mut entity = DictBuilder::default();
        entity.insert(source.molecule.name.clone());
        entity.insert(source.protein.receptor.clone());
        for water in &source.waters {
            entity.insert(format!("water:{}:{}", water.shell, water.gate_state));
        }

        let mut atom = DictBuilder::default();
        for value in &source.molecule.atom_types {
            atom.insert(value.clone());
        }

        let mut residue = DictBuilder::default();
        for value in &source.protein.residues {
            residue.insert(value.clone());
        }

        let mut event_kind = DictBuilder::default();
        let mut motif = DictBuilder::default();
        for event in &source.events {
            event_kind.insert(event.event_kind.clone());
            motif.insert(event.motif.clone());
        }

        let mut water_gate = DictBuilder::default();
        for water in &source.waters {
            water_gate.insert(water.gate_state.clone());
        }

        let mut variant = DictBuilder::default();
        for item in &source.variants {
            variant.insert(format!("variant:{}", item.variant_id));
        }

        let mut thermo_quantity = DictBuilder::default();
        let mut unit = DictBuilder::default();
        for thermo in &source.thermo {
            thermo_quantity.insert(thermo.quantity_kind.clone());
            unit.insert(thermo.unit_ref.clone());
        }

        let mut uncertainty = DictBuilder::default();
        for item in &source.uncertainties {
            uncertainty.insert(item.distribution.clone());
            uncertainty.insert(item.calibration.clone());
        }

        let mut teacher = DictBuilder::default();
        for item in &source.teacher_outputs {
            teacher.insert(item.model_id.clone());
            teacher.insert(item.label.clone());
        }

        let mut gflownet = DictBuilder::default();
        for trace in &source.gflownet_traces {
            for action in &trace.action_sequence {
                gflownet.insert(action.clone());
            }
        }

        let mut nanl = DictBuilder::default();
        for symbol in &source.nanl_symbols {
            nanl.insert(symbol.clone());
        }
        for event in &source.events {
            nanl.insert(event.event_kind.clone());
            nanl.insert(event.motif.clone());
        }

        let mut provenance = DictBuilder::default();
        provenance.insert(source.provenance.label.clone());

        Self {
            schema_version: "PRISM_CSE_DICTIONARIES_V0_1".to_string(),
            entity_dictionary: entity.finish(),
            atom_type_dictionary: atom.finish(),
            residue_dictionary: residue.finish(),
            event_kind_dictionary: event_kind.finish(),
            event_motif_dictionary: motif.finish(),
            water_gate_dictionary: water_gate.finish(),
            variant_dictionary: variant.finish(),
            thermo_quantity_dictionary: thermo_quantity.finish(),
            unit_dictionary: unit.finish(),
            uncertainty_model_dictionary: uncertainty.finish(),
            teacher_label_dictionary: teacher.finish(),
            gflownet_action_dictionary: gflownet.finish(),
            nrf_tile_dictionary: vec![
                "phase_tile".to_string(),
                "water_gate".to_string(),
                "transition_edge".to_string(),
            ],
            drf_channel_dictionary: vec![
                "resonance".to_string(),
                "firing".to_string(),
                "decay".to_string(),
                "uncertainty".to_string(),
                "acquisition".to_string(),
            ],
            nanl_symbol_dictionary: nanl.finish(),
            provenance_dictionary: provenance.finish(),
        }
    }

    pub fn index_entity(&self, value: &str) -> u32 {
        index_of(&self.entity_dictionary, value)
    }

    pub fn index_event_kind(&self, value: &str) -> u32 {
        index_of(&self.event_kind_dictionary, value)
    }

    pub fn index_event_motif(&self, value: &str) -> u32 {
        index_of(&self.event_motif_dictionary, value)
    }

    pub fn index_thermo_quantity(&self, value: &str) -> u32 {
        index_of(&self.thermo_quantity_dictionary, value)
    }

    pub fn index_teacher_label(&self, value: &str) -> u32 {
        index_of(&self.teacher_label_dictionary, value)
    }

    pub fn index_gflownet_action(&self, value: &str) -> u32 {
        index_of(&self.gflownet_action_dictionary, value)
    }

    pub fn id_for(&self, kind: CseDictionaryKind, value: &str) -> Option<u32> {
        let values = match kind {
            CseDictionaryKind::Entity => &self.entity_dictionary,
            CseDictionaryKind::AtomType => &self.atom_type_dictionary,
            CseDictionaryKind::Residue => &self.residue_dictionary,
            CseDictionaryKind::EventKind => &self.event_kind_dictionary,
            CseDictionaryKind::EventMotif => &self.event_motif_dictionary,
            CseDictionaryKind::WaterGate => &self.water_gate_dictionary,
            CseDictionaryKind::Variant => &self.variant_dictionary,
            CseDictionaryKind::ThermoQuantity => &self.thermo_quantity_dictionary,
            CseDictionaryKind::Unit => &self.unit_dictionary,
            CseDictionaryKind::UncertaintyModel => &self.uncertainty_model_dictionary,
            CseDictionaryKind::TeacherLabel => &self.teacher_label_dictionary,
            CseDictionaryKind::GFlowNetAction => &self.gflownet_action_dictionary,
            CseDictionaryKind::NrfTile => &self.nrf_tile_dictionary,
            CseDictionaryKind::DrfChannel => &self.drf_channel_dictionary,
            CseDictionaryKind::NanlSymbol => &self.nanl_symbol_dictionary,
            CseDictionaryKind::Provenance => &self.provenance_dictionary,
        };
        values
            .iter()
            .position(|candidate| candidate == value)
            .map(|idx| idx as u32)
    }
}

#[derive(Default)]
pub struct CseDictionarySymbols {
    entity: DictBuilder,
    atom: DictBuilder,
    residue: DictBuilder,
    event_kind: DictBuilder,
    motif: DictBuilder,
    water_gate: DictBuilder,
    variant: DictBuilder,
    thermo_quantity: DictBuilder,
    unit: DictBuilder,
    uncertainty: DictBuilder,
    teacher: DictBuilder,
    gflownet: DictBuilder,
    nrf_tile: DictBuilder,
    drf_channel: DictBuilder,
    nanl: DictBuilder,
    provenance: DictBuilder,
}

impl CseDictionarySymbols {
    pub fn insert(&mut self, kind: CseDictionaryKind, value: impl Into<String>) {
        let value = value.into();
        match kind {
            CseDictionaryKind::Entity => self.entity.insert(value),
            CseDictionaryKind::Variant => self.variant.insert(value),
            CseDictionaryKind::Provenance => self.provenance.insert(value),
            CseDictionaryKind::AtomType => self.atom.insert(value),
            CseDictionaryKind::Residue => self.residue.insert(value),
            CseDictionaryKind::EventKind => self.event_kind.insert(value),
            CseDictionaryKind::EventMotif => self.motif.insert(value),
            CseDictionaryKind::WaterGate => self.water_gate.insert(value),
            CseDictionaryKind::ThermoQuantity => self.thermo_quantity.insert(value),
            CseDictionaryKind::Unit => self.unit.insert(value),
            CseDictionaryKind::UncertaintyModel => self.uncertainty.insert(value),
            CseDictionaryKind::TeacherLabel => self.teacher.insert(value),
            CseDictionaryKind::GFlowNetAction => self.gflownet.insert(value),
            CseDictionaryKind::NrfTile => self.nrf_tile.insert(value),
            CseDictionaryKind::DrfChannel => self.drf_channel.insert(value),
            CseDictionaryKind::NanlSymbol => self.nanl.insert(value),
        }
    }

    pub fn finish(self) -> CseDictionaries {
        CseDictionaries {
            schema_version: "PRISM_CSE_HD_DICTIONARIES_V0_1".to_string(),
            entity_dictionary: self.entity.finish(),
            atom_type_dictionary: self.atom.finish(),
            residue_dictionary: self.residue.finish(),
            event_kind_dictionary: self.event_kind.finish(),
            event_motif_dictionary: self.motif.finish(),
            water_gate_dictionary: self.water_gate.finish(),
            variant_dictionary: self.variant.finish(),
            thermo_quantity_dictionary: self.thermo_quantity.finish(),
            unit_dictionary: self.unit.finish(),
            uncertainty_model_dictionary: self.uncertainty.finish(),
            teacher_label_dictionary: self.teacher.finish(),
            gflownet_action_dictionary: self.gflownet.finish(),
            nrf_tile_dictionary: self.nrf_tile.finish(),
            drf_channel_dictionary: self.drf_channel.finish(),
            nanl_symbol_dictionary: self.nanl.finish(),
            provenance_dictionary: self.provenance.finish(),
        }
    }
}

#[derive(Default)]
struct DictBuilder {
    values: BTreeMap<String, ()>,
}

impl DictBuilder {
    fn insert(&mut self, value: String) {
        if !value.trim().is_empty() {
            self.values.insert(value, ());
        }
    }

    fn finish(self) -> Vec<String> {
        self.values.into_keys().collect()
    }
}

fn index_of(values: &[String], needle: &str) -> u32 {
    values.iter().position(|value| value == needle).unwrap_or(0) as u32
}
