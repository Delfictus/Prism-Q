//! Hash-bound, topology-verified atom support for native ASC steering.
//!
//! The policy carries structural membership only. It deliberately has no
//! causal-role vocabulary: structural/cavity membership is not evidence of a
//! causal role. Residue declarations are expanded against the admitted native
//! topology before any CUDA allocation occurs.

use crate::input::PrismPrepTopology;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::OpenOptions;
use std::io::Read;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

pub const ASC_SELECTED_SUPPORT_SCHEMA: &str = "prism_asc_selected_support_policy";
pub const ASC_SELECTED_SUPPORT_VERSION: &str = "1.0.0";
pub const ASC_STRUCTURAL_SUPPORT_BASIS: &str = "EXPLICIT_STRUCTURAL_ATOM_RESIDUE_MEMBERSHIP";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AscAtomMembership {
    pub atom_index: usize,
    pub atom_name: String,
    pub element: String,
    pub physical_residue_id: usize,
    pub residue_name: String,
    pub chain_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AscResidueMembership {
    pub physical_residue_id: usize,
    pub residue_name: String,
    pub chain_id: String,
}

/// Operator-authored structural support declaration. The exact source-file
/// digest is measured by [`load`](Self::load), never trusted from JSON.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AscSelectedSupportPolicy {
    pub schema_kind: String,
    pub schema_version: String,
    pub topology_sha256: String,
    pub atom_order_sha256: String,
    pub source_kind: String,
    pub source_identity: String,
    pub source_evidence_sha256: String,
    pub support_basis: String,
    #[serde(default)]
    pub causal_role: Option<String>,
    #[serde(default)]
    pub causal_role_evidence_sha256: Option<String>,
    #[serde(default)]
    pub atoms: Vec<AscAtomMembership>,
    #[serde(default)]
    pub residues: Vec<AscResidueMembership>,
    #[serde(skip)]
    source_file_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AscSelectedSupport {
    policy_source_sha256: String,
    source_kind: String,
    source_identity: String,
    source_evidence_sha256: String,
    topology_sha256: String,
    atom_order_sha256: String,
    atom_indices: Vec<u32>,
    physical_residue_ids: Vec<u32>,
    mask: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct AtomOrderIdentity<'a> {
    atom_index: usize,
    atom_name: &'a str,
    element: &'a str,
    physical_residue_id: usize,
    residue_name: &'a str,
    chain_id: &'a str,
}

#[derive(Debug, Clone)]
struct TopologySelectionView<'a> {
    atoms: Vec<AtomOrderIdentity<'a>>,
}

impl<'a> TopologySelectionView<'a> {
    fn from_topology(topology: &'a PrismPrepTopology) -> Result<Self> {
        topology
            .validate()
            .map_err(|error| anyhow::anyhow!("ASC topology validation failed: {error}"))?;
        let physical = topology
            .atom_physical_residue_ids
            .as_deref()
            .unwrap_or(&topology.residue_ids);
        if physical != topology.residue_ids.as_slice() {
            bail!("ASC topology physical residue axis differs from residue_ids");
        }
        let atoms = (0..topology.n_atoms)
            .map(|atom_index| AtomOrderIdentity {
                atom_index,
                atom_name: topology.atom_names[atom_index].as_str(),
                element: topology.elements[atom_index].as_str(),
                physical_residue_id: physical[atom_index],
                residue_name: topology.residue_names[atom_index].as_str(),
                chain_id: topology.chain_ids[atom_index].as_str(),
            })
            .collect();
        Ok(Self { atoms })
    }

    fn atom_order_sha256(&self) -> Result<String> {
        let bytes = serde_json::to_vec(&self.atoms)?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }
}

fn require_sha256(value: &str, field: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("{field} must be a lowercase SHA-256 digest");
    }
    Ok(())
}

impl AscSelectedSupportPolicy {
    /// Load a regular, non-symlink policy file and bind the parsed policy to
    /// its exact bytes. `expected_source_sha256` is mandatory admission data.
    pub fn load(path: impl AsRef<Path>, expected_source_sha256: &str) -> Result<Self> {
        require_sha256(expected_source_sha256, "expected ASC policy source SHA256")?;
        let path = path.as_ref();
        let mut file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(path)
            .with_context(|| format!("open ASC support policy {}", path.display()))?;
        if !file.metadata()?.file_type().is_file() {
            bail!(
                "ASC support policy is not a regular file: {}",
                path.display()
            );
        }
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        Self::from_bytes(&bytes, expected_source_sha256)
    }

    pub fn from_bytes(bytes: &[u8], expected_source_sha256: &str) -> Result<Self> {
        require_sha256(expected_source_sha256, "expected ASC policy source SHA256")?;
        let measured = format!("{:x}", Sha256::digest(bytes));
        if measured != expected_source_sha256 {
            bail!(
                "ASC policy source SHA256 mismatch: expected={expected_source_sha256} measured={measured}"
            );
        }
        let mut policy: Self = serde_json::from_slice(bytes).context("parse ASC support policy")?;
        policy.source_file_sha256 = measured;
        policy.validate_header()?;
        Ok(policy)
    }

    fn validate_header(&self) -> Result<()> {
        if self.schema_kind != ASC_SELECTED_SUPPORT_SCHEMA
            || self.schema_version != ASC_SELECTED_SUPPORT_VERSION
        {
            bail!("unsupported ASC selected-support policy schema");
        }
        require_sha256(&self.topology_sha256, "ASC policy topology_sha256")?;
        require_sha256(&self.atom_order_sha256, "ASC policy atom_order_sha256")?;
        require_sha256(
            &self.source_evidence_sha256,
            "ASC policy source_evidence_sha256",
        )?;
        require_sha256(&self.source_file_sha256, "ASC policy exact source SHA256")?;
        if self.source_kind.trim().is_empty() || self.source_identity.trim().is_empty() {
            bail!("ASC policy source identity is empty");
        }
        if self.support_basis != ASC_STRUCTURAL_SUPPORT_BASIS {
            bail!("unsupported ASC support basis");
        }
        if self.causal_role.is_some() || self.causal_role_evidence_sha256.is_some() {
            bail!("ASC selected support does not admit causal-role claims");
        }
        if self.atoms.is_empty() && self.residues.is_empty() {
            bail!("ASC selected support membership is empty");
        }
        if self.source_kind == "NATIVE_OBSERVED_RECEPTOR_ATOM_SUPPORT" {
            self.verify_generated_observation_selection()?;
        }
        Ok(())
    }

    /// Generated policies bind their actual small selection record at launch.
    /// The full readout was verified by the builder; it is not reread during MD
    /// admission. Legacy operator declarations retain their existing contract.
    fn verify_generated_observation_selection(&self) -> Result<()> {
        let mut file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(&self.source_identity)
            .context("open generated ASC selection evidence")?;
        anyhow::ensure!(
            file.metadata()?.file_type().is_file(),
            "ASC selection evidence is not a regular file"
        );
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        anyhow::ensure!(
            format!("{:x}", Sha256::digest(&bytes)) == self.source_evidence_sha256,
            "generated ASC selection evidence digest mismatch"
        );
        let evidence: serde_json::Value = serde_json::from_slice(&bytes)?;
        anyhow::ensure!(
            evidence["schema_kind"] == "prism.asc_observed_support_selection.v1"
                && evidence["selector"]["topology"]["sha256"] == self.topology_sha256
                && evidence["atom_order_sha256"] == self.atom_order_sha256
                && evidence["causal_role"].is_null()
                && self.residues.is_empty(),
            "generated ASC selection identity mismatch"
        );
        let observed: Vec<usize> =
            serde_json::from_value(evidence["selected_atom_indices"].clone())?;
        let observed_set: BTreeSet<usize> = observed.iter().copied().collect();
        let selected_set: BTreeSet<usize> = self.atoms.iter().map(|atom| atom.atom_index).collect();
        anyhow::ensure!(
            !observed.is_empty()
                && observed.len() == observed_set.len()
                && self.atoms.len() == selected_set.len()
                && selected_set == observed_set,
            "generated ASC policy differs from observed atom selection"
        );
        Ok(())
    }

    /// Expand explicit residue declarations to atoms and validate every atom
    /// identity against the exact topology order before CUDA allocation.
    pub fn validate_and_expand(
        &self,
        topology: &PrismPrepTopology,
        topology_sha256: &str,
    ) -> Result<AscSelectedSupport> {
        self.validate_header()?;
        require_sha256(topology_sha256, "admitted ASC topology SHA256")?;
        if topology_sha256 != self.topology_sha256 {
            bail!("ASC support policy topology SHA256 mismatch");
        }
        let view = TopologySelectionView::from_topology(topology)?;
        self.expand_view(&view, topology_sha256)
    }

    fn expand_view(
        &self,
        view: &TopologySelectionView<'_>,
        topology_sha256: &str,
    ) -> Result<AscSelectedSupport> {
        let measured_atom_order = view.atom_order_sha256()?;
        if measured_atom_order != self.atom_order_sha256 {
            bail!("ASC support policy atom-order SHA256 mismatch");
        }

        let mut selected = BTreeSet::new();
        for member in &self.atoms {
            let atom = view.atoms.get(member.atom_index).with_context(|| {
                format!("ASC atom membership out of bounds: {}", member.atom_index)
            })?;
            if atom.atom_name != member.atom_name
                || atom.element != member.element
                || atom.physical_residue_id != member.physical_residue_id
                || atom.residue_name != member.residue_name
                || atom.chain_id != member.chain_id
            {
                bail!(
                    "ASC atom membership identity mismatch at atom {}",
                    member.atom_index
                );
            }
            if !selected.insert(member.atom_index) {
                bail!(
                    "duplicate ASC atom membership at atom {}",
                    member.atom_index
                );
            }
        }

        let mut declared_residues = BTreeSet::new();
        for member in &self.residues {
            if !declared_residues.insert(member.physical_residue_id) {
                bail!(
                    "duplicate ASC residue membership at residue {}",
                    member.physical_residue_id
                );
            }
            let mut matched = 0usize;
            for atom in &view.atoms {
                if atom.physical_residue_id == member.physical_residue_id {
                    if atom.residue_name != member.residue_name || atom.chain_id != member.chain_id
                    {
                        bail!(
                            "ASC residue membership identity mismatch at residue {}",
                            member.physical_residue_id
                        );
                    }
                    matched += 1;
                    if !selected.insert(atom.atom_index) {
                        bail!(
                            "duplicate expanded ASC membership at atom {}",
                            atom.atom_index
                        );
                    }
                }
            }
            if matched == 0 {
                bail!(
                    "ASC residue membership is absent or out of bounds: {}",
                    member.physical_residue_id
                );
            }
        }
        if selected.is_empty() {
            bail!("ASC selected support expanded to zero atoms");
        }

        let mut mask = vec![0u8; view.atoms.len()];
        let mut atom_indices = Vec::with_capacity(selected.len());
        let mut physical_residue_ids = BTreeSet::new();
        for atom_index in selected {
            mask[atom_index] = 1;
            atom_indices.push(u32::try_from(atom_index).context("ASC atom index exceeds u32")?);
            physical_residue_ids.insert(
                u32::try_from(view.atoms[atom_index].physical_residue_id)
                    .context("ASC physical residue index exceeds u32")?,
            );
        }
        Ok(AscSelectedSupport {
            policy_source_sha256: self.source_file_sha256.clone(),
            source_kind: self.source_kind.clone(),
            source_identity: self.source_identity.clone(),
            source_evidence_sha256: self.source_evidence_sha256.clone(),
            topology_sha256: topology_sha256.to_owned(),
            atom_order_sha256: measured_atom_order,
            atom_indices,
            physical_residue_ids: physical_residue_ids.into_iter().collect(),
            mask,
        })
    }
}

impl AscSelectedSupport {
    pub fn policy_source_sha256(&self) -> &str {
        &self.policy_source_sha256
    }
    pub fn source_kind(&self) -> &str {
        &self.source_kind
    }
    pub fn source_identity(&self) -> &str {
        &self.source_identity
    }
    pub fn source_evidence_sha256(&self) -> &str {
        &self.source_evidence_sha256
    }
    pub fn topology_sha256(&self) -> &str {
        &self.topology_sha256
    }
    pub fn atom_order_sha256(&self) -> &str {
        &self.atom_order_sha256
    }
    pub fn atom_indices(&self) -> &[u32] {
        &self.atom_indices
    }
    pub fn physical_residue_ids(&self) -> &[u32] {
        &self.physical_residue_ids
    }
    pub fn mask(&self) -> &[u8] {
        &self.mask
    }
    pub fn support_identity(&self) -> u64 {
        let identity = (0..4).fold(0u64, |value, word| {
            value
                ^ u64::from_str_radix(&self.policy_source_sha256[word * 16..word * 16 + 16], 16)
                    .expect("validated SHA-256 word")
        });
        if identity == 0 {
            u64::MAX
        } else {
            identity
        }
    }
}

pub fn asc_atom_order_sha256(topology: &PrismPrepTopology) -> Result<String> {
    TopologySelectionView::from_topology(topology)?.atom_order_sha256()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows() -> TopologySelectionView<'static> {
        TopologySelectionView {
            atoms: vec![
                AtomOrderIdentity {
                    atom_index: 0,
                    atom_name: "N",
                    element: "N",
                    physical_residue_id: 0,
                    residue_name: "ALA",
                    chain_id: "A",
                },
                AtomOrderIdentity {
                    atom_index: 1,
                    atom_name: "CA",
                    element: "C",
                    physical_residue_id: 0,
                    residue_name: "ALA",
                    chain_id: "A",
                },
                AtomOrderIdentity {
                    atom_index: 2,
                    atom_name: "O",
                    element: "O",
                    physical_residue_id: 1,
                    residue_name: "HOH",
                    chain_id: "W",
                },
            ],
        }
    }

    fn policy(view: &TopologySelectionView<'_>) -> AscSelectedSupportPolicy {
        AscSelectedSupportPolicy {
            schema_kind: ASC_SELECTED_SUPPORT_SCHEMA.into(),
            schema_version: ASC_SELECTED_SUPPORT_VERSION.into(),
            topology_sha256: "a".repeat(64),
            atom_order_sha256: view.atom_order_sha256().unwrap(),
            source_kind: "physical_cavity_lining_evidence".into(),
            source_identity: "lining:frame:42".into(),
            source_evidence_sha256: "b".repeat(64),
            support_basis: ASC_STRUCTURAL_SUPPORT_BASIS.into(),
            causal_role: None,
            causal_role_evidence_sha256: None,
            atoms: vec![AscAtomMembership {
                atom_index: 2,
                atom_name: "O".into(),
                element: "O".into(),
                physical_residue_id: 1,
                residue_name: "HOH".into(),
                chain_id: "W".into(),
            }],
            residues: vec![AscResidueMembership {
                physical_residue_id: 0,
                residue_name: "ALA".into(),
                chain_id: "A".into(),
            }],
            source_file_sha256: "c".repeat(64),
        }
    }

    #[test]
    fn expands_explicit_atoms_and_residues_to_native_mask() {
        let view = rows();
        let support = policy(&view).expand_view(&view, &"a".repeat(64)).unwrap();
        assert_eq!(support.atom_indices(), &[0, 1, 2]);
        assert_eq!(support.physical_residue_ids(), &[0, 1]);
        assert_eq!(support.mask(), &[1, 1, 1]);
        assert_eq!(support.policy_source_sha256(), "c".repeat(64));
    }

    #[test]
    fn rejects_duplicate_expansion_and_identity_mismatch() {
        let view = rows();
        let mut duplicate = policy(&view);
        duplicate.atoms[0] = AscAtomMembership {
            atom_index: 0,
            atom_name: "N".into(),
            element: "N".into(),
            physical_residue_id: 0,
            residue_name: "ALA".into(),
            chain_id: "A".into(),
        };
        assert!(duplicate
            .expand_view(&view, &"a".repeat(64))
            .unwrap_err()
            .to_string()
            .contains("duplicate expanded"));

        let mut mismatch = policy(&view);
        mismatch.atoms[0].atom_name = "CA".into();
        assert!(mismatch
            .expand_view(&view, &"a".repeat(64))
            .unwrap_err()
            .to_string()
            .contains("identity mismatch"));
    }

    #[test]
    fn rejects_empty_out_of_bounds_hash_and_causal_role() {
        let view = rows();
        let mut empty = policy(&view);
        empty.atoms.clear();
        empty.residues.clear();
        assert!(empty.validate_header().is_err());

        let mut bounds = policy(&view);
        bounds.residues.clear();
        bounds.atoms[0].atom_index = 99;
        assert!(bounds
            .expand_view(&view, &"a".repeat(64))
            .unwrap_err()
            .to_string()
            .contains("out of bounds"));

        let mut hash = policy(&view);
        hash.atom_order_sha256 = "d".repeat(64);
        assert!(hash
            .expand_view(&view, &"a".repeat(64))
            .unwrap_err()
            .to_string()
            .contains("atom-order"));

        let mut causal = policy(&view);
        causal.causal_role = Some("driver".into());
        assert!(causal
            .validate_header()
            .unwrap_err()
            .to_string()
            .contains("causal-role"));
    }

    #[test]
    fn generated_observation_evidence_is_checked_at_policy_load() {
        let dir = std::env::temp_dir().join(format!(
            "prism-asc-selection-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&dir).unwrap();
        let mut p = policy(&rows());
        p.residues.clear();
        p.source_kind = "NATIVE_OBSERVED_RECEPTOR_ATOM_SUPPORT".into();
        p.source_identity = dir.join("selection.json").to_string_lossy().into_owned();
        let evidence = serde_json::json!({"schema_kind":"prism.asc_observed_support_selection.v1",
            "selector":{"topology":{"sha256":p.topology_sha256}},"atom_order_sha256":p.atom_order_sha256,
            "causal_role":null,"selected_atom_indices":[2]});
        let source = serde_json::to_vec(&evidence).unwrap();
        std::fs::write(&p.source_identity, &source).unwrap();
        p.source_evidence_sha256 = format!("{:x}", Sha256::digest(&source));
        let load = |v: &AscSelectedSupportPolicy| {
            let bytes = serde_json::to_vec(v).unwrap();
            AscSelectedSupportPolicy::from_bytes(&bytes, &format!("{:x}", Sha256::digest(&bytes)))
        };
        load(&p).unwrap();
        let mut wrong_atom = p.clone();
        wrong_atom.atoms[0].atom_index = 1;
        assert!(load(&wrong_atom).is_err());
        let mut wrong_topology = p.clone();
        wrong_topology.topology_sha256 = "f".repeat(64);
        assert!(load(&wrong_topology).is_err());
        let mut wrong_order = p.clone();
        wrong_order.atom_order_sha256 = "f".repeat(64);
        assert!(load(&wrong_order).is_err());
        std::fs::write(&p.source_identity, b"changed").unwrap();
        assert!(load(&p).is_err());
        std::fs::remove_file(&p.source_identity).unwrap();
        assert!(load(&p).is_err());
        // Existing manual source identities remain declarations, not paths.
        load(&policy(&rows())).unwrap();
        std::fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn exact_policy_source_digest_is_mandatory() {
        let view = rows();
        let mut p = policy(&view);
        p.source_file_sha256.clear();
        let bytes = serde_json::to_vec(&p).unwrap();
        let digest = format!("{:x}", Sha256::digest(&bytes));
        let loaded = AscSelectedSupportPolicy::from_bytes(&bytes, &digest).unwrap();
        assert_eq!(loaded.source_file_sha256, digest);
        assert!(AscSelectedSupportPolicy::from_bytes(&bytes, &"f".repeat(64)).is_err());
    }
}

/// Convert sealed native receptor observations to the existing selected ASC
/// policy. Selection is explicit; no salience threshold or causal label is used.
pub mod observed {
    use super::*;
    use crate::chemical_state_fiber::mechanistic::{
        MeasurementSourcePin, MechanisticMeasurementArtifact,
    };
    use anyhow::ensure;
    use serde_json::{json, Value};
    use std::{
        collections::BTreeMap,
        fs::{self, File},
        io::{BufRead, BufReader, Write},
    };

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct ObservedSupportManifest {
        pub schema_kind: String,
        pub readout_receipt: MeasurementSourcePin,
        pub topology: MeasurementSourcePin,
        pub arm: String,
        /// Zero-based saved frame interval, start included and end excluded.
        pub frame_start: usize,
        pub frame_end: usize,
        /// None selects every observed region separately; never merges regions.
        pub region_ids: Option<Vec<u64>>,
    }

    #[derive(Deserialize)]
    struct MembershipRow {
        schema_kind: String,
        paired_frame_ordinal: usize,
        step: u64,
        phase: String,
        stream_id: u16,
        topology_sha256: String,
        run_ids: BTreeMap<String, String>,
        comparison: Comparison,
    }
    #[derive(Deserialize)]
    struct Comparison {
        region_id: u64,
        native_grids: BTreeMap<String, Vec<Grid>>,
    }
    #[derive(Deserialize)]
    struct Grid {
        grid_ordinal: usize,
        receptor_excluded_volume_face_count: usize,
        boundary_face_count: usize,
        receptor_residue_faces: Vec<ResidueFaces>,
    }
    #[derive(Deserialize)]
    struct ResidueFaces {
        analysis_residue_id: i32,
        atom_indices: Vec<usize>,
        face_indices: Vec<usize>,
    }
    #[derive(Default)]
    struct Support {
        atoms: BTreeSet<usize>,
        selected_rows: usize,
        empty_rows: usize,
        grid_observations: usize,
    }

    fn row_atoms(row: &MembershipRow, arm: &str, analysis_axis: &[i32]) -> Result<BTreeSet<usize>> {
        let grids = row
            .comparison
            .native_grids
            .get(arm)
            .context("selected arm has no native grids")?;
        ensure!(!grids.is_empty(), "missing grid observations");
        let mut atoms = BTreeSet::new();
        for (ordinal, grid) in grids.iter().enumerate() {
            ensure!(ordinal == grid.grid_ordinal, "grid ordinal discontinuity");
            let mut residues = BTreeSet::new();
            let mut faces = BTreeSet::new();
            for group in &grid.receptor_residue_faces {
                ensure!(
                    group.analysis_residue_id >= 0 && residues.insert(group.analysis_residue_id),
                    "invalid or duplicate observed residue"
                );
                ensure!(
                    !group.atom_indices.is_empty() && !group.face_indices.is_empty(),
                    "empty receptor attribution"
                );
                let mut unique_atoms = BTreeSet::new();
                for &atom in &group.atom_indices {
                    ensure!(
                        analysis_axis.get(atom) == Some(&group.analysis_residue_id),
                        "observed atom/residue axis mismatch"
                    );
                    ensure!(unique_atoms.insert(atom), "duplicate receptor atom");
                    atoms.insert(atom);
                }
                for &face in &group.face_indices {
                    ensure!(
                        face < grid.boundary_face_count && faces.insert(face),
                        "invalid/duplicate receptor face"
                    );
                }
            }
            ensure!(
                faces.len() == grid.receptor_excluded_volume_face_count,
                "receptor face population mismatch"
            );
        }
        Ok(atoms)
    }
    fn json_file(path: &Path, value: &impl Serialize) -> Result<MeasurementSourcePin> {
        let bytes = serde_json::to_vec_pretty(value)?;
        let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
        file.write_all(&bytes)?;
        Ok(MeasurementSourcePin {
            path: path.to_path_buf(),
            sha256: format!("{:x}", Sha256::digest(&bytes)),
        })
    }
    fn pin_value(value: &Value) -> Result<MeasurementSourcePin> {
        Ok(serde_json::from_value(value.clone())?)
    }
    fn count(value: &Value, key: &str) -> Result<usize> {
        usize::try_from(
            value[key]
                .as_u64()
                .with_context(|| format!("missing count {key}"))?,
        )
        .context("count overflow")
    }

    /// Reads the complete pinned membership population once, collecting only
    /// the requested frames. Exact atoms are retained; residue expansion is not
    /// implicit. The source file remains authority for complete grid/face data.
    pub fn build_observed_support(manifest_path: &Path, output_dir: &Path) -> Result<Value> {
        let manifest_bytes = fs::read(manifest_path)?;
        let m: ObservedSupportManifest = serde_json::from_slice(&manifest_bytes)?;
        ensure!(
            m.schema_kind == "prism.asc_observed_support_manifest.v1",
            "unsupported ASC observation manifest"
        );
        ensure!(
            m.arm == "treatment" || m.arm == "control",
            "unknown observation arm"
        );
        let selected: Option<BTreeSet<u64>> =
            m.region_ids.as_ref().map(|v| v.iter().copied().collect());
        if let Some(ids) = &selected {
            ensure!(
                !ids.is_empty() && ids.len() == m.region_ids.as_ref().unwrap().len(),
                "empty or duplicate region selection"
            );
        }
        m.readout_receipt.verify()?;
        m.topology.verify()?;
        let receipt: Value = serde_json::from_reader(File::open(&m.readout_receipt.path)?)?;
        ensure!(
            receipt["schema_kind"] == "prism.native_paired_mechanistic_readout_receipt.v1"
                && receipt["status"] == "IMPLEMENTATION_VERIFIED",
            "native readout not sealed"
        );
        let artifact_pin = pin_value(&receipt["source_artifact"])?;
        artifact_pin.verify()?;
        let a: MechanisticMeasurementArtifact =
            serde_json::from_reader(File::open(&artifact_pin.path)?)?;
        ensure!(
            a.schema_kind == "prism.native_fiber_mechanistic_measurements.v1"
                && a.topology_sha256 == m.topology.sha256
                && a.source_pins.contains(&m.topology)
                && receipt["source_manifest_sha256"] == a.manifest_sha256,
            "readout artifact/topology lineage mismatch"
        );
        ensure!(
            m.frame_start < m.frame_end
                && m.frame_end <= a.frame_steps.len()
                && a.frame_steps.len() == a.frame_phases.len()
                && a.frame_steps.windows(2).all(|w| w[0] < w[1]),
            "invalid selected frame interval or clock"
        );
        ensure!(
            a.arm_run_ids.keys().map(String::as_str).collect::<Vec<_>>()
                == ["control", "treatment"]
                && a.arm_run_ids.values().all(|v| !v.trim().is_empty()),
            "incomplete native run identities"
        );
        let frame_count = count(&receipt, "frames")?;
        let region_count = count(&receipt, "regions_per_frame")?;
        let expected_rows = count(&receipt, "paired_region_frames")?;
        ensure!(
            frame_count == a.frame_steps.len()
                && region_count > 0
                && frame_count.checked_mul(region_count) == Some(expected_rows),
            "readout population count mismatch"
        );
        let topology: PrismPrepTopology = serde_json::from_reader(File::open(&m.topology.path)?)?;
        let view = TopologySelectionView::from_topology(&topology)?;
        ensure!(
            a.atom_count == topology.n_atoms,
            "readout atom count mismatch"
        );
        let analysis_axis = topology
            .atom_analysis_residue_ids
            .as_deref()
            .context("observed support requires explicit analysis-residue axis")?;
        ensure!(
            analysis_axis.len() == topology.n_atoms,
            "analysis axis length mismatch"
        );
        let atom_order = asc_atom_order_sha256(&topology)?;
        let member_pin = pin_value(&receipt["outputs"]["paired_receptor_membership.jsonl.zst"])?;
        member_pin.verify()?;
        ensure!(!output_dir.exists(), "ASC output directory already exists");
        fs::create_dir_all(output_dir)?;
        let absolute_output = output_dir.canonicalize()?;
        let output_dir = absolute_output.as_path();
        let mut lineage = zstd::stream::write::Encoder::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(output_dir.join("selected_observation_rows.jsonl.zst"))?,
            1,
        )?;
        let reader = BufReader::new(zstd::stream::read::Decoder::new(File::open(
            &member_pin.path,
        )?)?);
        let mut supports: BTreeMap<u64, Support> = BTreeMap::new();
        let mut seen_regions = BTreeSet::new();
        let mut first_regions: Option<BTreeSet<u64>> = None;
        let mut seen_rows = 0usize;
        let mut current_frame = 0usize;
        for raw in reader.split(b'\n') {
            let raw = raw?;
            ensure!(!raw.is_empty(), "empty native membership record");
            let row: MembershipRow = serde_json::from_slice(&raw)?;
            let frame = row.paired_frame_ordinal;
            ensure!(
                frame < frame_count
                    && row.schema_kind == "prism.native_paired_receptor_membership.v1"
                    && row.step == a.frame_steps[frame]
                    && row.phase == a.frame_phases[frame]
                    && row.topology_sha256 == a.topology_sha256
                    && row.stream_id == a.stream_id
                    && row.run_ids == a.arm_run_ids,
                "native membership endpoint identity mismatch"
            );
            if frame != current_frame {
                ensure!(
                    frame == current_frame + 1 && seen_regions.len() == region_count,
                    "missing/reordered native region frames"
                );
                if let Some(first) = &first_regions {
                    ensure!(&seen_regions == first, "region population changed");
                } else {
                    first_regions = Some(seen_regions.clone());
                }
                seen_regions.clear();
                current_frame = frame;
            }
            let region = row.comparison.region_id;
            ensure!(seen_regions.insert(region), "duplicate native region frame");
            // Validate exact selected-arm atom attribution even outside the selected interval.
            let atoms = row_atoms(&row, &m.arm, analysis_axis)?;
            if frame >= m.frame_start
                && frame < m.frame_end
                && selected.as_ref().map_or(true, |v| v.contains(&region))
            {
                let support = supports.entry(region).or_default();
                support.selected_rows += 1;
                support.empty_rows += usize::from(atoms.is_empty());
                support.grid_observations += row.comparison.native_grids[&m.arm].len();
                support.atoms.extend(atoms);
                serde_json::to_writer(
                    &mut lineage,
                    &json!({"source_line_ordinal":seen_rows,
                    "source_line_sha256":format!("{:x}",Sha256::digest(&raw)),"digest_excludes_newline":true,"region_id":region,
                    "frame_ordinal":frame,"step":row.step,"phase":row.phase,"arm":m.arm}),
                )?;
                lineage.write_all(b"\n")?;
            }
            seen_rows += 1;
        }
        lineage.finish()?.flush()?;
        ensure!(
            seen_rows == expected_rows
                && current_frame + 1 == frame_count
                && seen_regions.len() == region_count,
            "incomplete native membership population"
        );
        if let Some(first) = first_regions {
            ensure!(seen_regions == first, "final region population changed");
        }
        if let Some(ids) = &selected {
            ensure!(
                supports.keys().copied().collect::<BTreeSet<_>>() == *ids,
                "selected region missing"
            );
        }
        ensure!(!supports.is_empty(), "no selected region observations");
        // Recheck authoritative byte sources after the streamed consumer completes.
        member_pin.verify()?;
        artifact_pin.verify()?;
        m.topology.verify()?;
        m.readout_receipt.verify()?;
        let selected_rows_path = output_dir.join("selected_observation_rows.jsonl.zst");
        let selected_rows_pin = MeasurementSourcePin {
            path: selected_rows_path.clone(),
            sha256: format!("{:x}", Sha256::digest(fs::read(&selected_rows_path)?)),
        };
        let mut outputs = Vec::new();
        for (region, support) in supports {
            ensure!(
                support.selected_rows == m.frame_end - m.frame_start,
                "selected support frame gap"
            );
            let provenance = json!({"schema_kind":"prism.asc_observed_support_selection.v1",
                "manifest_sha256":format!("{:x}",Sha256::digest(&manifest_bytes)),"selector":m,
                "readout_artifact":artifact_pin,"membership_source":member_pin,"selected_observation_rows":selected_rows_pin,
                "run_id":a.arm_run_ids[&m.arm],"stream_id":a.stream_id,"region_id":region,
                "atom_order_sha256":atom_order,"selected_atom_indices":support.atoms,
                "source_rows_verified":seen_rows,"selected_region_frames":support.selected_rows,
                "empty_region_frames":support.empty_rows,"selected_grid_observations":support.grid_observations,
                "frame_interval_semantics":"start inclusive; end exclusive; union across all recorded native grids in selected arm only",
                "membership_semantics":"exact observed receptor-face atoms; whole-residue expansion not performed",
                "causal_role":null,"intervention_executed":false});
            let evidence = json_file(
                &output_dir.join(format!("region_{region}_selection.json")),
                &provenance,
            )?;
            if support.atoms.is_empty() {
                outputs.push(json!({"region_id":region,"selection":evidence,"policy":null,
                    "availability":"observed_empty_receptor_support","intervention_executed":false}));
                continue;
            }
            let atoms: Vec<AscAtomMembership> = support
                .atoms
                .iter()
                .map(|&id| {
                    let atom = &view.atoms[id];
                    AscAtomMembership {
                        atom_index: id,
                        atom_name: atom.atom_name.into(),
                        element: atom.element.into(),
                        physical_residue_id: atom.physical_residue_id,
                        residue_name: atom.residue_name.into(),
                        chain_id: atom.chain_id.into(),
                    }
                })
                .collect();
            let policy = AscSelectedSupportPolicy {
                schema_kind: ASC_SELECTED_SUPPORT_SCHEMA.into(),
                schema_version: ASC_SELECTED_SUPPORT_VERSION.into(),
                topology_sha256: m.topology.sha256.clone(),
                atom_order_sha256: atom_order.clone(),
                source_kind: "NATIVE_OBSERVED_RECEPTOR_ATOM_SUPPORT".into(),
                source_identity: evidence.path.to_string_lossy().into_owned(),
                source_evidence_sha256: evidence.sha256.clone(),
                support_basis: ASC_STRUCTURAL_SUPPORT_BASIS.into(),
                causal_role: None,
                causal_role_evidence_sha256: None,
                atoms,
                residues: vec![],
                source_file_sha256: String::new(),
            };
            let bytes = serde_json::to_vec_pretty(&policy)?;
            let digest = format!("{:x}", Sha256::digest(&bytes));
            let validated = AscSelectedSupportPolicy::from_bytes(&bytes, &digest)?;
            let expanded = validated.validate_and_expand(&topology, &m.topology.sha256)?;
            ensure!(
                expanded
                    .atom_indices()
                    .iter()
                    .map(|v| *v as usize)
                    .collect::<BTreeSet<_>>()
                    == support.atoms,
                "ASC policy changed observed membership"
            );
            let p = json_file(
                &output_dir.join(format!("region_{region}_selected_support_policy.json")),
                &policy,
            )?;
            let common = json!([
                "--asc-support-mode",
                "selected-support",
                "--asc-support-policy",
                p.path,
                "--asc-support-policy-sha256",
                p.sha256,
                "--asc-isolated-journal"
            ]);
            let mut control = common.as_array().unwrap().clone();
            control.extend([json!("--asc-control-alpha"), json!("0")]);
            outputs.push(json!({"region_id":region,"selection":evidence,"policy":p,
                "selected_atoms":expanded.atom_indices().len(),"selected_physical_residues":expanded.physical_residue_ids(),
                "common_support_argument_suffix":common,"zero_alpha_control_argument_suffix":control,
                "argument_scope":"append to an explicitly configured native resident V2 campaign; positive gain must be explicitly selected",
                "intervention_executed":false,"nonzero_dose_measured":false}));
        }
        let result = json!({"schema_kind":"prism.asc_observed_support_receipt.v1","status":"IMPLEMENTATION_VERIFIED",
            "selector":m,"source_artifact":artifact_pin,"membership_source":member_pin,"selected_observation_rows":selected_rows_pin,
            "source_rows_verified":seen_rows,"source_frames_verified":frame_count,"policies":outputs,
            "scope":"source-derived structural intervention support preparation; no MD or causal classification",
            "intervention_executed":false});
        json_file(&output_dir.join("EXECUTION_RECEIPT.json"), &result)?;
        Ok(result)
    }
    #[cfg(test)]
    mod tests {
        use super::*;
        fn row() -> Value {
            json!({"schema_kind":"prism.native_paired_receptor_membership.v1",
                "paired_frame_ordinal":1400,"step":14010,"phase":"heating","stream_id":0,
                "topology_sha256":"a".repeat(64),"run_ids":{"treatment":"t","control":"c"},
                "comparison":{"region_id":47,"native_grids":{"treatment":[{
                    "grid_ordinal":0,"boundary_face_count":7,"receptor_excluded_volume_face_count":3,
                    "receptor_residue_faces":[{"analysis_residue_id":0,"atom_indices":[0,2],"face_indices":[1,4]},
                        {"analysis_residue_id":2,"atom_indices":[3],"face_indices":[6]}]}]},
                    "local_window_clipping_observed":{"treatment":true},"scope":"retained source descriptor"}})
        }
        #[test]
        fn observed_support_retains_exact_atoms_without_residue_expansion() {
            let parsed = serde_json::from_value(row()).unwrap();
            assert_eq!(
                row_atoms(&parsed, "treatment", &[0, 0, 0, 2]).unwrap(),
                BTreeSet::from([0, 2, 3])
            );
            assert!(row_atoms(&parsed, "control", &[0, 0, 0, 2]).is_err());
        }
        #[test]
        fn observed_support_rejects_cross_residue_and_out_of_bounds_atoms() {
            let parsed = serde_json::from_value(row()).unwrap();
            assert!(row_atoms(&parsed, "treatment", &[0, 0, 1, 2]).is_err());
            assert!(row_atoms(&parsed, "treatment", &[0, 0, 0]).is_err());
        }
        #[test]
        fn observed_support_checks_native_face_population() {
            for (pointer, value) in [
                (
                    "/comparison/native_grids/treatment/0/grid_ordinal",
                    json!(1),
                ),
                (
                    "/comparison/native_grids/treatment/0/receptor_excluded_volume_face_count",
                    json!(4),
                ),
                (
                    "/comparison/native_grids/treatment/0/receptor_residue_faces/1/face_indices",
                    json!([4]),
                ),
                (
                    "/comparison/native_grids/treatment/0/receptor_residue_faces/1/atom_indices",
                    json!([3, 3]),
                ),
            ] {
                let mut v = row();
                *v.pointer_mut(pointer).unwrap() = value;
                assert!(
                    row_atoms(
                        &serde_json::from_value(v).unwrap(),
                        "treatment",
                        &[0, 0, 0, 2]
                    )
                    .is_err(),
                    "{pointer}"
                );
            }
        }
        #[test]
        fn observed_empty_support_is_retained_as_empty() {
            let mut v = row();
            let grid = &mut v["comparison"]["native_grids"]["treatment"][0];
            grid["receptor_excluded_volume_face_count"] = json!(0);
            grid["receptor_residue_faces"] = json!([]);
            assert!(row_atoms(
                &serde_json::from_value(v).unwrap(),
                "treatment",
                &[0, 0, 0, 2]
            )
            .unwrap()
            .is_empty());
        }
    }
}
