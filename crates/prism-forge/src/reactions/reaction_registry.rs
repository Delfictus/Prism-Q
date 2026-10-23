use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct ReactionRegistry {
    pub schema_version: String,
    pub registry_name: String,
    pub disclaimer: String,
    pub reactions: Vec<ReactionRule>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ReactionRule {
    pub reaction_id: String,
    pub reaction_name: String,
    pub reaction_class: String,
    pub version: u32,
    pub enabled: bool,
    pub epistemic_status: String,
    pub smarts: String,
    pub reactant_roles: HashMap<String, ReactantRole>,
    pub product_bond: ProductBondSpec,
    pub guards: ReactionGuards,
    pub provenance: ReactionProvenance,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ReactantRole {
    pub required_smarts: String,
    pub reactive_atom_map: u32,
    pub leaving_group_atom_maps: Vec<u32>,
    pub bond_vector_reference: Vec<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProductBondSpec {
    pub atom_map_a: u32,
    pub atom_map_b: u32,
    pub bond_order: u8,
    #[serde(rename = "ideal_bond_length_A")]
    pub ideal_bond_length_a: f32,
    pub ideal_bond_angle_deg: f32,
    pub torsion_policy: TorsionPolicy,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TorsionPolicy {
    pub mode: String,
    pub dihedral_deg: Vec<f32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ReactionGuards {
    pub max_product_heavy_atoms: u32,
    pub allowed_formal_charge_range: [i32; 2],
    pub reject_radicals: bool,
    pub reject_unmapped_reactive_atoms: bool,
    pub require_single_match_or_enumerate_matches: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ReactionProvenance {
    pub source: String,
    pub notes: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReactionMatch {
    pub reaction_id: String,
    pub scaffold_role: String,
    pub synthon_role: String,
    pub scaffold_site: MatchedReactiveSite,
    pub synthon_site: MatchedReactiveSite,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchedReactiveSite {
    pub reactive_atom_idx: usize,
    pub atom_map: u32,
    pub leaving_groups: Vec<LeavingGroupSpec>,
    pub reference_atom_idx: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeavingGroupSpec {
    pub atom_idx: usize,
    pub atom_map: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssemblyPlan {
    pub reaction_id: String,
    pub scaffold_role: String,
    pub synthon_role: String,
    pub scaffold_reactive_atom_idx: usize,
    pub synthon_reactive_atom_idx: usize,
    pub scaffold_reference_atom_1: usize,
    pub scaffold_reference_atom_2: usize,
    pub synthon_reference_atom_idx: usize,
    pub scaffold_leaving_group_atom_indices: Vec<usize>,
    pub synthon_leaving_group_atom_indices: Vec<usize>,
    pub selected_dihedral_deg: f32,
}

impl ReactionRegistry {
    pub fn load(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        Self::from_yaml_str(&text).with_context(|| format!("parse {}", path.display()))
    }

    pub fn from_yaml_str(text: &str) -> Result<Self> {
        let registry: Self = serde_yaml::from_str(text)?;
        registry.validate()?;
        Ok(registry)
    }

    pub fn enabled_reactions(&self) -> impl Iterator<Item = &ReactionRule> {
        self.reactions.iter().filter(|reaction| reaction.enabled)
    }

    pub fn get(&self, reaction_id: &str) -> Result<&ReactionRule> {
        self.reactions
            .iter()
            .find(|reaction| reaction.reaction_id == reaction_id)
            .ok_or_else(|| anyhow!("reaction_id unknown: {reaction_id}"))
    }

    pub fn validate(&self) -> Result<()> {
        if !self
            .disclaimer
            .contains("not guaranteed experimental success")
        {
            return Err(anyhow!(
                "registry disclaimer must constrain experimental synthesis claims"
            ));
        }
        let mut seen_ids = HashSet::new();
        for reaction in &self.reactions {
            if !seen_ids.insert(reaction.reaction_id.clone()) {
                return Err(anyhow!("duplicate reaction_id {}", reaction.reaction_id));
            }
            reaction.validate()?;
        }
        Ok(())
    }
}

impl ReactionRule {
    pub fn validate(&self) -> Result<()> {
        if self.reaction_id.is_empty() {
            return Err(anyhow!("reaction_id missing"));
        }
        if self.enabled && self.guards.max_product_heavy_atoms == 0 {
            return Err(anyhow!(
                "{} enabled reaction lacks guards",
                self.reaction_id
            ));
        }
        if self.product_bond.ideal_bond_length_a <= 0.0
            || !self.product_bond.ideal_bond_length_a.is_finite()
        {
            return Err(anyhow!("{} invalid ideal bond length", self.reaction_id));
        }
        if self.product_bond.torsion_policy.dihedral_deg.is_empty() {
            return Err(anyhow!("{} empty torsion grid", self.reaction_id));
        }
        if !self.reactant_roles.contains_key("scaffold")
            || !self.reactant_roles.contains_key("synthon")
        {
            return Err(anyhow!(
                "{} must define scaffold and synthon roles",
                self.reaction_id
            ));
        }
        let mut role_maps = HashSet::new();
        for (role_name, role) in &self.reactant_roles {
            let maps = atom_maps_from_smarts(&role.required_smarts);
            if maps.is_empty() {
                return Err(anyhow!(
                    "{}:{} required_smarts has no atom maps",
                    self.reaction_id,
                    role_name
                ));
            }
            if !maps.contains(&role.reactive_atom_map) {
                return Err(anyhow!(
                    "{}:{} reactive atom map missing",
                    self.reaction_id,
                    role_name
                ));
            }
            for leaving_map in &role.leaving_group_atom_maps {
                if !maps.contains(leaving_map) {
                    return Err(anyhow!(
                        "{}:{} leaving-group map {} missing",
                        self.reaction_id,
                        role_name,
                        leaving_map
                    ));
                }
            }
            if !role.bond_vector_reference.contains(&role.reactive_atom_map) {
                return Err(anyhow!(
                    "{}:{} bond_vector_reference must include reactive atom",
                    self.reaction_id,
                    role_name
                ));
            }
            role_maps.extend(maps);
        }
        if !role_maps.contains(&self.product_bond.atom_map_a)
            || !role_maps.contains(&self.product_bond.atom_map_b)
        {
            return Err(anyhow!(
                "{} product bond atom maps missing from reactants",
                self.reaction_id
            ));
        }
        Ok(())
    }
}

pub fn atom_maps_from_smarts(smarts: &str) -> HashSet<u32> {
    let mut maps = HashSet::new();
    let bytes = smarts.as_bytes();
    let mut idx = 0;
    while idx < bytes.len() {
        if bytes[idx] == b':' {
            let mut end = idx + 1;
            while end < bytes.len() && bytes[end].is_ascii_digit() {
                end += 1;
            }
            if end > idx + 1 {
                if let Ok(value) = smarts[idx + 1..end].parse::<u32>() {
                    maps.insert(value);
                }
            }
            idx = end;
        } else {
            idx += 1;
        }
    }
    maps
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry_path() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("00_registry/chemistry/reaction_rules.v1.yml")
    }

    #[test]
    fn load_registry() {
        let registry = ReactionRegistry::load(&registry_path()).expect("registry loads");
        assert_eq!(registry.schema_version, "reaction_rules.v1");
        assert!(registry.enabled_reactions().count() >= 3);
    }

    #[test]
    fn parse_three_core_rules() {
        let registry = ReactionRegistry::load(&registry_path()).expect("registry loads");
        assert!(registry.get("RXN_AMIDE_COUPLING").is_ok());
        assert!(registry.get("RXN_SUZUKI_ARYL_ARYL").is_ok());
        assert!(registry.get("RXN_BUCHWALD_HARTWIG").is_ok());
    }

    #[test]
    fn validate_atom_maps() {
        let registry = ReactionRegistry::load(&registry_path()).expect("registry loads");
        let rule = registry.get("RXN_SUZUKI_ARYL_ARYL").expect("suzuki");
        assert_eq!(rule.product_bond.atom_map_a, 1);
        assert_eq!(rule.product_bond.atom_map_b, 3);
        assert!(atom_maps_from_smarts(&rule.smarts).contains(&1));
    }

    #[test]
    fn reject_malformed_reaction_rule() {
        let yaml = r#"
schema_version: reaction_rules.v1
registry_name: bad
disclaimer: "Reaction-rule compatibility indicates virtual synthesis plausibility, not guaranteed experimental success."
reactions:
  - reaction_id: BAD
    reaction_name: Bad
    reaction_class: bad
    version: 1
    enabled: true
    epistemic_status: PROJECTED_REACTION_GRAMMAR
    smarts: "[C:1].[N:2]>>[C:1][N:2]"
    reactant_roles:
      scaffold:
        required_smarts: "[C:1]"
        reactive_atom_map: 1
        leaving_group_atom_maps: [99]
        bond_vector_reference: [1]
      synthon:
        required_smarts: "[N:2]"
        reactive_atom_map: 2
        leaving_group_atom_maps: []
        bond_vector_reference: [2]
    product_bond:
      atom_map_a: 1
      atom_map_b: 2
      bond_order: 1
      ideal_bond_length_A: 1.4
      ideal_bond_angle_deg: 120.0
      torsion_policy:
        mode: discrete_grid
        dihedral_deg: [0]
    guards:
      max_product_heavy_atoms: 1
      allowed_formal_charge_range: [-1, 1]
      reject_radicals: true
      reject_unmapped_reactive_atoms: true
      require_single_match_or_enumerate_matches: enumerate_matches
    provenance:
      source: bad
      notes: bad
"#;
        assert!(ReactionRegistry::from_yaml_str(yaml).is_err());
    }

    #[test]
    fn enumerate_reaction_maps_from_small_test_smarts() {
        let maps = atom_maps_from_smarts("[c:1][Br:2].[c:3][B:4]([O:5])[O:6]>>[c:1]-[c:3]");
        assert_eq!(maps, HashSet::from([1, 2, 3, 4, 5, 6]));
    }
}
