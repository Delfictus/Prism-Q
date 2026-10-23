//! NHS Input Module - PRISM-PREP Topology Loader
//!
//! Loads sanitized topology JSON from PRISM-PREP and prepares it for NHS processing.
//!
//! **CRITICAL**: Only PRISM-PREP sanitized structures are valid input.
//! Raw PDB files must go through `prism-prep` first.
//!
//! ## Pipeline Flow
//!
//! ```text
//! Raw PDB → prism-prep → topology.json → NHS-UV Detection Engine
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use prism_nhs::input::PrismPrepTopology;
//!
//! // Load PRISM-PREP output
//! let topology = PrismPrepTopology::load("1L2Y_topology.json")?;
//!
//! // Get NHS-compatible atom data
//! let (positions, types, charges, residues) = topology.to_nhs_format();
//! ```

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use crate::config::SolventMode;
use crate::coupled_chronology::HmrVerification;
use crate::lambda_fragment::{
    validate_production_admission, LambdaFragmentSystem, LambdaRuntimeCapabilities,
};
use crate::rt_targets::{identify_rt_targets, RtTargets};
use crate::solvate::solvate_protein;

/// Serialize a JSON value exactly as PRISM-PREP's Python
/// `json.dumps(sort_keys=True,separators=(",",":"),ensure_ascii=True)`.
/// This is intentionally local to the topology contract: changing its byte
/// representation would invalidate producer-bound SHA-256 identities.
fn python_canonical_json(value: &serde_json::Value) -> std::result::Result<Vec<u8>, String> {
    fn write_string(value: &str, out: &mut String) {
        out.push('"');
        for ch in value.chars() {
            match ch {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\u{08}' => out.push_str("\\b"),
                '\u{0c}' => out.push_str("\\f"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                ch if ch <= '\u{1f}' || !ch.is_ascii() => {
                    let mut utf16 = [0u16; 2];
                    for unit in ch.encode_utf16(&mut utf16) {
                        out.push_str(&format!("\\u{unit:04x}"));
                    }
                }
                ch => out.push(ch),
            }
        }
        out.push('"');
    }

    fn python_number(number: &serde_json::Number) -> String {
        let encoded = number.to_string();
        if !number.is_f64() {
            return encoded;
        }
        // serde's shortest round-trip digits preserve the binary64 value, but
        // its fixed/scientific notation thresholds differ from Python repr.
        // Normalize notation only; never round, reparse or replace the value.
        let (sign, magnitude) = encoded.strip_prefix('-').map_or(("", encoded.as_str()), |s| ("-", s));
        let (mantissa, exponent) = magnitude.split_once(['e', 'E'])
            .map_or((magnitude, 0i32), |(m, e)| (m, e.parse::<i32>().expect("serde finite float exponent")));
        let point = mantissa.find('.').unwrap_or(mantissa.len()) as i32;
        let digits = mantissa.replace('.', "");
        let Some(first) = digits.bytes().position(|b| b != b'0') else {
            return format!("{sign}0.0");
        };
        let order = exponent + point - first as i32 - 1;
        let significant = digits[first..].trim_end_matches('0');
        if !(-4..16).contains(&order) {
            let fraction = if significant.len() > 1 { format!(".{}", &significant[1..]) } else { String::new() };
            return format!("{sign}{}{fraction}e{}{:02}", &significant[..1], if order < 0 { '-' } else { '+' }, order.unsigned_abs());
        }
        let decimal = order + 1;
        if decimal <= 0 {
            format!("{sign}0.{}{}", "0".repeat((-decimal) as usize), significant)
        } else if decimal as usize >= significant.len() {
            format!("{sign}{}{}.0", significant, "0".repeat(decimal as usize - significant.len()))
        } else {
            let at = decimal as usize;
            format!("{sign}{}.{}", &significant[..at], &significant[at..])
        }
    }

    fn write_value(value: &serde_json::Value, out: &mut String) -> std::result::Result<(), String> {
        match value {
            serde_json::Value::Null => out.push_str("null"),
            serde_json::Value::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
            serde_json::Value::Number(value) => out.push_str(&python_number(value)),
            serde_json::Value::String(value) => write_string(value, out),
            serde_json::Value::Array(values) => {
                out.push('[');
                for (index, value) in values.iter().enumerate() {
                    if index != 0 {
                        out.push(',');
                    }
                    write_value(value, out)?;
                }
                out.push(']');
            }
            serde_json::Value::Object(values) => {
                let mut keys: Vec<&str> = values.keys().map(String::as_str).collect();
                keys.sort_unstable();
                out.push('{');
                for (index, key) in keys.into_iter().enumerate() {
                    if index != 0 {
                        out.push(',');
                    }
                    write_string(key, out);
                    out.push(':');
                    write_value(
                        values
                            .get(key)
                            .ok_or_else(|| format!("canonical JSON key disappeared: {key}"))?,
                        out,
                    )?;
                }
                out.push('}');
            }
        }
        Ok(())
    }

    let mut encoded = String::new();
    write_value(value, &mut encoded)?;
    Ok(encoded.into_bytes())
}

pub(crate) fn python_canonical_sha256(
    value: &serde_json::Value,
) -> std::result::Result<String, String> {
    Ok(format!(
        "{:x}",
        Sha256::digest(python_canonical_json(value)?)
    ))
}

// ============================================================================
// ATOM TYPES FOR NHS
// ============================================================================

/// Atom type classification for NHS exclusion field
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(i32)]
pub enum NhsAtomType {
    /// Hydrophobic atoms (C aliphatic, S in Met/Cys)
    Hydrophobic = 0,
    /// Polar atoms (O, N with H-bond capability)
    Polar = 1,
    /// Positively charged (Lys NZ, Arg NH)
    ChargedPositive = 2,
    /// Negatively charged (Asp OD, Glu OE)
    ChargedNegative = 3,
    /// Aromatic atoms (Trp, Tyr, Phe rings) - UV bias targets
    Aromatic = 4,
    /// Backbone atoms (N, CA, C, O)
    Backbone = 5,
    /// Water oxygen (if present)
    Water = 6,
}

impl NhsAtomType {
    /// Convert to i32 for GPU
    pub fn as_i32(self) -> i32 {
        self as i32
    }
}

// ============================================================================
// PRISM-PREP TOPOLOGY FORMAT
// ============================================================================

/// P5 — PRISM-PREP aromatic chromophore target.
///
/// The prep pipeline computes the exact ring atom set and ring centroid for
/// every aromatic. Before P5 this block was emitted but never deserialized, so
/// the engine re-derived aromatics by residue-name scan and used the first 16
/// atoms of the whole residue. For TRP that put N, H, CA at the head of the
/// list, and `compute_ring_normals` takes the cross product of the first three
/// entries — yielding a BACKBONE AMIDE-PLANE normal rather than a ring normal
/// (measured mean error 65.8 deg against the true ring normal, worse than a
/// random axis), and a centroid displaced 2.196-2.993 A from `ring_center`.
///
/// The ring normal is the anisotropy axis of out-of-plane vibrational
/// dissipation in `compute_thermal_wavefront`, and the centroid is the origin
/// of its arrival-time clock and of the 5 A vibrational shell, the 4 A UV
/// spike-localization radius and the 2 A dewetting halo. Absorption itself is
/// deliberately isotropic in this architecture (`excite_aromatic_wavelength`
/// takes no position or orientation), so this geometry governs where deposited
/// energy GOES, never which chromophore absorbs or how much.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AromaticTargetSpec {
    /// Ordinal of this aromatic within the topology's own target list.
    #[serde(default)]
    pub residue_idx: usize,
    /// Residue name as prepped (TRP / TYR / PHE).
    #[serde(default)]
    pub residue_name: String,
    /// Residue id in the topology's residue numbering.
    #[serde(default)]
    pub residue_id: i64,
    /// Exact ring atom indices: 6 for PHE/TYR, 9 for the TRP indole.
    pub ring_atom_indices: Vec<usize>,
    /// Ring centroid in Angstrom.
    pub ring_center: [f64; 3],
    /// Molar extinction at 280 nm. Carried as EVIDENCE only; the engine's own
    /// per-chromophore constants remain authoritative until reconciled.
    #[serde(default)]
    pub extinction_280: f64,
}

/// Bond parameter structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BondParam {
    /// First atom index
    pub i: usize,
    /// Second atom index
    pub j: usize,
    /// Equilibrium distance
    pub r0: f64,
    /// Force constant
    pub k: f64,
}

/// Angle parameter structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AngleParam {
    /// First atom index
    pub i: usize,
    /// Second atom index
    pub j: usize,
    /// Third atom index (named k_idx to avoid conflict with force constant)
    pub k_idx: usize,
    /// Equilibrium angle
    pub theta0: f64,
    /// Force constant
    pub force_k: f64,
}

/// Coefficient convention, independent of the coefficient's physical units.
/// Both variants use U=p*k*delta^2; equal units do not imply equal p.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HarmonicEnergyConvention {
    #[serde(rename = "OPENMM_FULL_K_HALF_ENERGY")]
    OpenmmFullKHalfEnergy,
    #[serde(rename = "AMBER_NO_HALF_ENERGY")]
    AmberNoHalfEnergy,
}

impl HarmonicEnergyConvention {
    pub fn energy_prefactor(self) -> f64 {
        match self {
            Self::OpenmmFullKHalfEnergy => 0.5,
            Self::AmberNoHalfEnergy => 1.0,
        }
    }
}

/// Explicit source meaning. Raw bond/angle fields are never normalized in place.
/// A missing contract remains readable for historical inspection but cannot be
/// lowered into a new production engine. No provider/filename inference exists.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HarmonicParameterContract {
    pub schema_kind: String,
    pub schema_version: u32,
    pub source_provider: String,
    pub bond_convention: HarmonicEnergyConvention,
    pub angle_convention: HarmonicEnergyConvention,
    pub bond_coefficient_unit: String,
    pub angle_coefficient_unit: String,
    pub bond_equilibrium_unit: String,
    pub angle_equilibrium_unit: String,
    pub bond_uses_periodic_boundary_conditions: bool,
    pub angle_uses_periodic_boundary_conditions: bool,
    pub bond_count: usize,
    pub angle_count: usize,
    /// Canonical JSON SHA256 of this object excluding contract_sha256. Fields
    /// are integers/ASCII strings only, avoiding cross-parser float repr drift.
    pub contract_sha256: String,
}

impl HarmonicParameterContract {
    fn validate_for_topology(&self, topology: &PrismPrepTopology) -> Result<(), String> {
        if self.schema_kind != "prismos_harmonic_parameter_contract"
            || self.schema_version != 1
            || self.source_provider.trim().is_empty()
            || self.bond_coefficient_unit != "kcal/mol/angstrom^2"
            || self.angle_coefficient_unit != "kcal/mol/radian^2"
            || self.bond_equilibrium_unit != "angstrom"
            || self.angle_equilibrium_unit != "radian"
            || self.bond_uses_periodic_boundary_conditions
            || self.angle_uses_periodic_boundary_conditions
            || self.bond_count != topology.bonds.len()
            || self.angle_count != topology.angles.len()
        {
            return Err("harmonic source contract has unsupported schema, units, provider or term counts".into());
        }
        let mut value = serde_json::to_value(self).map_err(|e| e.to_string())?;
        value.as_object_mut().ok_or("harmonic contract is not an object")?
            .remove("contract_sha256");
        let expected = format!("{:x}", Sha256::digest(python_canonical_json(&value)?));
        if self.contract_sha256 != expected {
            return Err("harmonic source contract canonical digest mismatch".into());
        }
        Ok(())
    }
}

/// One boundary-specific lowering. The raw topology remains unchanged and is
/// never itself accepted as an already-lowered buffer for a second conversion.
#[derive(Debug, Clone)]
pub struct HarmonicParameterLowering {
    pub bond_k: Vec<f32>,
    pub angle_k: Vec<f32>,
    pub accounting: serde_json::Value,
}

// A deliberate historical execution policy, not a provider-convention claim.
// It is process-immutable and reachable only by the explicit original-May CLI
// admission before constructors. Its digest is retained in the existing
// harmonic upload/PrismCode/checkpoint identity. No topology field is rewritten.
static HISTORICAL_MAY_LITERAL_ADMISSION: std::sync::OnceLock<serde_json::Value> = std::sync::OnceLock::new();

impl PrismPrepTopology {
    fn historical_harmonic_raw_bytes(&self) -> Vec<u8> {
        let mut raw = Vec::new();
        raw.extend_from_slice(b"PRISMOS_HARMONIC_PARSED_F64_V1\0");
        raw.extend_from_slice(&(self.bonds.len() as u64).to_le_bytes());
        raw.extend_from_slice(&(self.angles.len() as u64).to_le_bytes());
        for b in &self.bonds {
            raw.extend_from_slice(&(b.i as u64).to_le_bytes());
            raw.extend_from_slice(&(b.j as u64).to_le_bytes());
            raw.extend_from_slice(&b.r0.to_bits().to_le_bytes());
            raw.extend_from_slice(&b.k.to_bits().to_le_bytes());
        }
        for a in &self.angles {
            raw.extend_from_slice(&(a.i as u64).to_le_bytes());
            raw.extend_from_slice(&(a.j as u64).to_le_bytes());
            raw.extend_from_slice(&(a.k_idx as u64).to_le_bytes());
            raw.extend_from_slice(&a.theta0.to_bits().to_le_bytes());
            raw.extend_from_slice(&a.force_k.to_bits().to_le_bytes());
        }
        raw
    }

    // Original file identity is admitted before runtime HMR. This fingerprint
    // keeps the interpreted q/LJ/pair/exclusion model immutable across that HMR.
    fn historical_may_nonbonded_bytes(&self) -> Vec<u8> {
        let mut out=b"PRISM_MAY_HISTORICAL_NONBONDED_INPUT_V1\0".to_vec();
        out.extend_from_slice(&(self.n_atoms as u64).to_le_bytes());
        for q in &self.charges {out.extend_from_slice(&q.to_bits().to_le_bytes());}
        for lj in &self.lj_params {out.extend_from_slice(&lj.sigma.to_bits().to_le_bytes());out.extend_from_slice(&lj.epsilon.to_bits().to_le_bytes());}
        out.extend_from_slice(&(self.bonds.len() as u64).to_le_bytes());
        for b in &self.bonds {for i in [b.i,b.j] {out.extend_from_slice(&(i as u64).to_le_bytes());}}
        out.extend_from_slice(&(self.angles.len() as u64).to_le_bytes());
        for a in &self.angles {for i in [a.i,a.k_idx] {out.extend_from_slice(&(i as u64).to_le_bytes());}}
        out.extend_from_slice(&(self.dihedrals.len() as u64).to_le_bytes());
        for d in &self.dihedrals {for i in [d.i,d.l] {out.extend_from_slice(&(i as u64).to_le_bytes());}}
        out.extend_from_slice(&(self.exclusions.len() as u64).to_le_bytes());
        for row in &self.exclusions {out.extend_from_slice(&(row.len() as u64).to_le_bytes());for i in row {out.extend_from_slice(&(*i as u64).to_le_bytes());}}
        out
    }

    pub fn historical_may_nonbonded_execution_contract(&self) -> Result<serde_json::Value,String> {
        let admission=self.historical_may_admission()?.ok_or("historical May nonbonded model lacks exact original input/NMA/profile admission")?;
        if self.lambda_fragment_system.is_some() {return Err("historical May model forbids lambda/provider reinterpretation".into());}
        Ok(serde_json::json!({"model_contract_id":"prism.native.may_historical_nonbonded_model.v1",
            "original_input_sha256":admission["original_input_sha256"],"original_nma_sha256":admission["original_nma_sha256"],
            "model_fingerprint_sha256":admission["nonbonded_model_fingerprint_sha256"],
            "reference_engine_sha256":admission["reference_engine_sha256"],
            "reference_fused_ptx_sha256":"c8308c953191777604885ed59a32b2715612fdf87d21c39f1aa3303101bccb08",
            "pair_selection":"dihedral endpoints minus bond12/angle13; original exclusions minus selected14; original serialized metadata retained",
            "fudge_qq_f32_bits":"3f555326","fudge_lj_f32_bits":"3f000000","coulomb_f32_bits":"43a60824",
            "uv_charge":"ground_q*(1+population*(type_ratio-1))",
            "ordinary_potential":"unshifted LJ+Coulomb inside existing cutoff; zero outside",
            "whole_original_elf_bitwise_equivalence_claim":false}))
    }

    pub fn admit_historical_may_literal_execution(&self, original_input_sha256: &str, original_nma_sha256: &str) -> Result<(), String> {
        const INPUTS: &[(&str,&str)] = &[("17c02ee7c3195383fe7d61062326dc4254d4fb5a4e1f89d0439c3b6fb4f9b46d","0d4124d9bcd4b1adc9afeef21eb06c9e7dbd09c919d006f068d06fe045e35b1c"),("6cd93387505353a873492e66733a423ef030cfe16187e2d4c8055946a5de72c3","3f8dbdc7924b7f35a93b2a4d531cd08784142747bb0162386cfe6f5a87137aa2"),("bc19e5f455552b6619b34e23fff55425c8c60f040168e89f21bbd028d527ff19","5b6876a732fbd626a65ee04c1ae2a25fd4aedb82521b8c04b3d90d02cc26b12d"),("e42415ab62f8fd87cf7d900f1322c80ff51d628a86593d3165758af62cc09b53","76dc026129c93d6b8df4bd9b88c221bed6d0b855edbb7eb5c6cf1c3fb2ff4999")];
        if !INPUTS.contains(&(original_input_sha256,original_nma_sha256)) || self.harmonic_parameter_contract.is_some() {
            return Err("historical May literal execution requires one of the four immutable original input hashes and absent original source contract; no retagging or inference".into());
        }
        self.validate()?;
        let model_fingerprint=format!("{:x}",Sha256::digest(self.historical_may_nonbonded_bytes()));
        let expected=match original_input_sha256 {
            "17c02ee7c3195383fe7d61062326dc4254d4fb5a4e1f89d0439c3b6fb4f9b46d" => "f537c49aec9149dca20d9b3a60b499adc50eb74e0b496de17a65fdd8701ecb58",
            "6cd93387505353a873492e66733a423ef030cfe16187e2d4c8055946a5de72c3" => "a1610fecfc06fb856ecd295c950d9f0777495fcbdd43c903708d264c8d8f6d5e",
            "bc19e5f455552b6619b34e23fff55425c8c60f040168e89f21bbd028d527ff19" => "9ca2389eb5e26b2171c61993844bc1d66fa2a4e3fa06159c78381ce1f86c8589",
            "e42415ab62f8fd87cf7d900f1322c80ff51d628a86593d3165758af62cc09b53" => "1e5d5addf06cd014e81a8ac88b27222597be00f07bf2b7a3a77618d962248b67",
            _=>return Err("historical May model input is not admitted".into()),
        };
        if model_fingerprint!=expected {return Err("historical May q/LJ/pair/exclusion fingerprint differs from exact frozen input".into());}
        let admission=serde_json::json!({"schema":"prism.historical_may.literal_execution.v1",
            "nonbonded_model_fingerprint_sha256":model_fingerprint,
            "reference_engine_sha256":"7e0f5eb00b9180199cc6db698b1e556b9044d0a78102920331aa92380190591a",
            "original_input_sha256":original_input_sha256,"original_nma_sha256":original_nma_sha256,
            "parsed_raw_f64_sha256":format!("{:x}",Sha256::digest(self.historical_harmonic_raw_bytes())),
            "source_provider_convention":"UNKNOWN; retained original input contract is absent",
            "execution_policy":"literal parsed f64 coefficients cast directly to f32; no scale or normalization",
            "historical_upload_evidence":"closest retained 8c858e0a source plus original PTX no-half operator; original Rust build not byte-proven",
            "historical_reference_equivalence_claim":false,"serialized_input_mutated":false});
        if let Some(previous)=HISTORICAL_MAY_LITERAL_ADMISSION.get() {
            if previous!=&admission {return Err("historical May runtime admission cannot change within a process".into());}
        }else{HISTORICAL_MAY_LITERAL_ADMISSION.set(admission).map_err(|_|"historical May admission race")?;}
        Ok(())
    }

    fn historical_may_admission(&self) -> Result<Option<&'static serde_json::Value>, String> {
        let Some(admission)=HISTORICAL_MAY_LITERAL_ADMISSION.get() else{return Ok(None)};
        if admission["nonbonded_model_fingerprint_sha256"].as_str()!=Some(format!("{:x}",Sha256::digest(self.historical_may_nonbonded_bytes())).as_str())
            || self.harmonic_parameter_contract.is_some() || admission["parsed_raw_f64_sha256"].as_str()
            !=Some(format!("{:x}",Sha256::digest(self.historical_harmonic_raw_bytes())).as_str()) {
            return Err("historical May constructor/input scientific identity changed".into());
        }
        Ok(Some(admission))
    }

    pub fn historical_may_literal_execution_enabled(&self) -> Result<bool,String> {
        Ok(self.historical_may_admission()?.is_some())
    }

    fn lower_historical_may_literal(&self,target:HarmonicEnergyConvention,admission:&serde_json::Value)
        ->Result<HarmonicParameterLowering,String> {
        if target!=HarmonicEnergyConvention::AmberNoHalfEnergy {
            return Err("historical May literal execution only supports the original native no-half target operator".into());
        }
        let cast=|value:f64|->Result<f32,String>{let out=value as f32;
            if !value.is_finite() || value<0.0 || !out.is_finite() || (value>0.0&&out==0.0) {
                return Err("historical May coefficient cannot be represented in original f32 upload ABI".into());
            }Ok(out)};
        let bond_k=self.bonds.iter().map(|b|cast(b.k)).collect::<Result<Vec<_>,_>>()?;
        let angle_k=self.angles.iter().map(|a|cast(a.force_k)).collect::<Result<Vec<_>,_>>()?;
        let raw=self.historical_harmonic_raw_bytes();
        let mut lowered=Vec::new();lowered.extend_from_slice(b"PRISMOS_HARMONIC_LOWERED_F32_V1\0");
        lowered.extend_from_slice(&serde_json::to_vec(&target).map_err(|e|e.to_string())?);lowered.push(0);
        lowered.extend_from_slice(&(bond_k.len()as u64).to_le_bytes());lowered.extend_from_slice(&(angle_k.len()as u64).to_le_bytes());
        for value in bond_k.iter().chain(&angle_k){lowered.extend_from_slice(&value.to_bits().to_le_bytes());}
        Ok(HarmonicParameterLowering{bond_k,angle_k,accounting:serde_json::json!({
            "schema_kind":"prismos_harmonic_parameter_lowering","schema_version":2,
            "source_contract":null,"source_convention":"UNKNOWN; no provider normalization claimed",
            "target_convention":target,"literal_execution_contract":admission,
            "bond_scale_f64":1.0,"angle_scale_f64":1.0,
            "conversion_order":"original parsed binary64 coefficient cast directly to float32; no multiply",
            "parsed_raw_f64_sha256":format!("{:x}",Sha256::digest(raw)),
            "lowered_coefficient_f32_sha256":format!("{:x}",Sha256::digest(lowered)),
            "bond_count":self.bonds.len(),"angle_count":self.angles.len(),"raw_topology_coefficients_mutated":false})})
    }

    pub fn lower_harmonic_parameters(
        &self,
        target: HarmonicEnergyConvention,
    ) -> Result<HarmonicParameterLowering, String> {
        self.validate()?;
        if let Some(admission)=self.historical_may_admission()? {
            return self.lower_historical_may_literal(target,admission);
        }
        let source = self.harmonic_parameter_contract.as_ref().ok_or(
            "harmonic source convention is absent: refuse production lowering; preserve the historical input/ELF or provide an explicitly reviewed source contract",
        )?;
        source.validate_for_topology(self)?;
        if self.bonds.iter().any(|b| !(b.r0 as f32).is_finite() || b.r0 as f32 <= 0.0)
            || self.angles.iter().any(|a| !(a.theta0 as f32).is_finite() || a.theta0 as f32 <= 0.0)
        {
            return Err("harmonic equilibrium coordinate is not representable in the target float32 ABI".into());
        }
        let bond_scale = source.bond_convention.energy_prefactor() / target.energy_prefactor();
        let angle_scale = source.angle_convention.energy_prefactor() / target.energy_prefactor();
        let cast = |raw: f64, scale: f64, label: &str, index: usize| -> Result<f32, String> {
            let mapped = raw * scale;
            let device = mapped as f32;
            if !raw.is_finite() || raw < 0.0 || !mapped.is_finite() || !device.is_finite()
                || (raw > 0.0 && device == 0.0)
            {
                return Err(format!("{label} coefficient {index} is invalid or not representable in the target float32 ABI"));
            }
            Ok(device)
        };
        let bond_k = self.bonds.iter().enumerate()
            .map(|(i, b)| cast(b.k, bond_scale, "bond", i)).collect::<Result<Vec<_>, _>>()?;
        let angle_k = self.angles.iter().enumerate()
            .map(|(i, a)| cast(a.force_k, angle_scale, "angle", i)).collect::<Result<Vec<_>, _>>()?;
        let raw = self.historical_harmonic_raw_bytes();
        let mut lowered = Vec::new();
        lowered.extend_from_slice(b"PRISMOS_HARMONIC_LOWERED_F32_V1\0");
        lowered.extend_from_slice(&serde_json::to_vec(&target).map_err(|e| e.to_string())?);
        lowered.push(0);
        lowered.extend_from_slice(&(bond_k.len() as u64).to_le_bytes());
        lowered.extend_from_slice(&(angle_k.len() as u64).to_le_bytes());
        for value in bond_k.iter().chain(angle_k.iter()) {
            lowered.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        let accounting = serde_json::json!({
            "schema_kind":"prismos_harmonic_parameter_lowering",
            "schema_version":1,
            "source_contract":source,
            "target_convention":target,
            "bond_scale_f64":bond_scale,
            "angle_scale_f64":angle_scale,
            "conversion_order":"source_binary64 * exact_power_of_two_scale, then Rust as-f32",
            "parsed_raw_f64_sha256":format!("{:x}",Sha256::digest(&raw)),
            "parsed_raw_f64_identity_scope":"actual parsed native values, not the source JSON numeric text or provider binary64 identity",
            "lowered_coefficient_f32_sha256":format!("{:x}",Sha256::digest(&lowered)),
            "bond_count":bond_k.len(),"angle_count":angle_k.len(),
            "raw_topology_coefficients_mutated":false,
            "source_first_bond":self.bonds.first().map(|b| serde_json::json!({
                "i":b.i,"j":b.j,"r0_f64_bits":b.r0.to_bits(),"k_f64_bits":b.k.to_bits(),
                "r0_decimal":format!("{:.17e}",b.r0),"k_decimal":format!("{:.17e}",b.k)})),
            "source_first_angle":self.angles.first().map(|a| serde_json::json!({
                "i":a.i,"j":a.j,"k_idx":a.k_idx,"theta0_f64_bits":a.theta0.to_bits(),"force_k_f64_bits":a.force_k.to_bits(),
                "theta0_decimal":format!("{:.17e}",a.theta0),"force_k_decimal":format!("{:.17e}",a.force_k)})),
            "lowered_first_bond_k_f32_bits":bond_k.first().map(|v|v.to_bits()),
            "lowered_first_angle_k_f32_bits":angle_k.first().map(|v|v.to_bits()),
            "lowered_first_bond_k_decimal":bond_k.first().map(|v|format!("{:.9e}",v)),
            "lowered_first_angle_k_decimal":angle_k.first().map(|v|format!("{:.9e}",v)),
            "component_energy_force_parity":"NOT_MEASURED",
        });
        Ok(HarmonicParameterLowering { bond_k, angle_k, accounting })
    }
}

/// Dihedral parameter structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DihedralParam {
    /// First atom index
    pub i: usize,
    /// Second atom index
    pub j: usize,
    /// Third atom index (named k_idx to avoid conflict with force constant)
    pub k_idx: usize,
    /// Fourth atom index
    pub l: usize,
    /// Periodicity
    #[serde(default)]
    pub periodicity: i32,
    /// Phase
    #[serde(default)]
    pub phase: f64,
    /// Force constant
    #[serde(default)]
    pub force_k: f64,
    /// True for an improper torsion. This identity bit is part of the
    /// canonical interaction key; omitting it can alias proper/improper
    /// endpoint terms during alchemical union construction.
    #[serde(default)]
    pub improper: bool,
}

/// LJ parameter structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LjParam {
    /// Sigma (distance)
    pub sigma: f64,
    /// Epsilon (well depth)
    pub epsilon: f64,
}

/// Exact AMBER/OpenMM exception parameters for a single scaled nonbonded pair.
///
/// These values are authoritative when present. They preserve heterogeneous
/// SCEE/SCNB behavior such as GLYCAM's unscaled 1-4 interactions instead of
/// forcing every topology through the ff14SB 0.8333/0.5 convention.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NonbondedExceptionParam {
    pub i: usize,
    pub j: usize,
    pub charge_product_e2: f64,
    #[serde(rename = "sigma_A")]
    pub sigma_a: f64,
    pub epsilon_kcal_mol: f64,
}

/// Completeness and byte-level identity contract for exact scaled pairs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NonbondedExceptionContract {
    pub schema_kind: String,
    pub schema_version: u32,
    pub semantics: String,
    pub dynamic_charge_scaling_semantics: String,
    pub exact_scaled_pair_count: usize,
    pub strict_exclusion_pair_count: usize,
    pub total_openmm_exception_count: usize,
    pub pair_digest_algorithm: String,
    pub pair_digest_sha256: String,
    pub strict_exclusion_digest_algorithm: String,
    pub strict_exclusion_digest_sha256: String,
    pub source_prmtop_sha256: String,
    pub hardcoded_uniform_scaling_allowed: bool,
    pub complete: bool,
    pub fallback_used: bool,
}

/// Scientific admission policy for the nonbonded exception plane.
///
/// Production callers use `RequireExactV2`. The legacy uniform-FUDGE path is
/// retained only for explicitly named non-production compatibility runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum NonbondedExecutionPolicy {
    RequireExactV2,
    /// Exact immutable original May input/NMA admission; retained historical operator.
    RequireHistoricalMayModel,
    AllowNonProductionLegacyUniformFudge,
}

/// Hydrogen cluster for SHAKE/RATTLE constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HCluster {
    /// Cluster type (1=CH, 2=CH2, 3=CH3, 4=NH, 5=OH, etc.)
    #[serde(rename = "type")]
    pub cluster_type: i32,
    /// Central atom index
    pub central_atom: usize,
    /// Hydrogen atom indices (-1 for unused slots)
    pub hydrogen_atoms: Vec<i32>,
    /// Bond lengths to each hydrogen
    pub bond_lengths: Vec<f64>,
    /// Number of hydrogens in cluster
    pub n_hydrogens: usize,
    /// Inverse mass of central atom
    pub inv_mass_central: f64,
    /// Inverse mass of hydrogen
    pub inv_mass_h: f64,
}

/// Metadata for a single residue from the topology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidueMetadata {
    pub residue_idx: usize,
    pub residue_name: String,
    pub residue_id: i32,
}

/// PRISM-PREP v3 periodic-cell provenance. Legacy `box_vectors` never
/// satisfies lambda-fragment admission.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PeriodicCellContract {
    pub periodic: bool,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub vectors_angstrom: Option<[[f64; 3]; 3]>,
    #[serde(default)]
    pub volume_angstrom3: Option<f64>,
    #[serde(default)]
    pub orthorhombic: Option<bool>,
    #[serde(default)]
    pub minimum_image_required: Option<bool>,
    #[serde(default)]
    pub coordinate_wrapping_required: Option<bool>,
    #[serde(default)]
    pub reciprocal_method: Option<String>,
    #[serde(default)]
    pub real_space_cutoff_angstrom: Option<f64>,
    #[serde(default)]
    pub contract_sha256: Option<String>,
    // Exact preparation-layer PME authority. Absence preserves historical cell serialization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_implementation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub boundary_condition: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pme_interpolation_order: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relative_force_tolerance: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ewald_alpha_a_inv: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ewald_alpha_derivation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pme_grid_dimensions_lower_bound: Option<[usize; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pme_grid_dimensions: Option<[usize; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pme_grid_formula: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pme_grid_rounding_policy: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_exact_parameter_lock_required: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub independent_openmm_force_energy_validation_required: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pme_parameterization_hash_projection: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pme_parameterization_sha256: Option<String>,

}
impl PeriodicCellContract {
    /// Validate an expanded preparation cell without synthesizing missing PME parameters.
    /// Historical cells lacking the entire expansion retain their previous contract.
    pub fn validated_recorded_pme(&self) -> std::result::Result<Option<crate::lambda_fragment::PmeContract>, String> {
        let value = serde_json::to_value(self).map_err(|e| e.to_string())?;
        const EXPANDED: &[&str] = &["reference_implementation", "boundary_condition", "pme_interpolation_order", "relative_force_tolerance", "ewald_alpha_a_inv", "ewald_alpha_derivation", "pme_grid_dimensions_lower_bound", "pme_grid_dimensions", "pme_grid_formula", "pme_grid_rounding_policy", "runtime_exact_parameter_lock_required", "independent_openmm_force_energy_validation_required", "pme_parameterization_hash_projection", "pme_parameterization_sha256"];
        if !EXPANDED.iter().any(|key| value.get(*key).is_some()) {
            return Ok(None);
        }
        if !self.periodic || self.orthorhombic != Some(true)
            || self.minimum_image_required != Some(true)
            || self.coordinate_wrapping_required != Some(true)
            || self.reciprocal_method.as_deref() != Some("PME")
            || self.pme_parameterization_hash_projection.as_deref() != Some("CANONICAL_JSON_SORTED_KEYS_COMPACT_WITH_FLOAT64_IEEE754_BITS_HEX_V1") {
            return Err("expanded periodic cell has inconsistent boundary/PME authority".into());
        }
        let vectors = self.vectors_angstrom.ok_or("expanded periodic cell lacks vectors")?;
        if vectors.iter().flatten().any(|v| !v.is_finite()) {
            return Err("expanded periodic cell has nonfinite vectors".into());
        }
        let hash = self.contract_sha256.as_ref().ok_or("expanded periodic cell lacks original contract hash")?;
        let periodic_box = crate::lambda_fragment::PeriodicBoxContract {
            vectors_a: vectors, minimum_image_enabled: true, source_sha256: hash.clone(),
        };
        let volume = self.volume_angstrom3.ok_or("expanded periodic cell lacks original volume")?;
        if !volume.is_finite() || volume <= 0.0 || (volume-periodic_box.signed_volume_a3()).abs() > 1.0e-6 {
            return Err("expanded periodic cell volume differs from original vectors".into());
        }
        let mut projection = serde_json::Map::new();
        for key in EXPANDED.iter().copied().filter(|key| *key != "pme_parameterization_hash_projection").chain(std::iter::once("real_space_cutoff_angstrom")) {
            projection.insert(key.into(), value.get(key).cloned().ok_or_else(|| format!("expanded periodic cell lacks {key}"))?);
        }
        let pme: crate::lambda_fragment::PmeContract = serde_json::from_value(serde_json::Value::Object(projection)).map_err(|e| format!("expanded periodic PME projection: {e}"))?;
        crate::lambda_fragment::validate_pme(&pme, &periodic_box)?;
        let mut cell = value;
        let object = cell.as_object_mut().ok_or("periodic cell serialization is not an object")?;
        object.remove("contract_sha256");
        object.retain(|_, value| !value.is_null());
        if python_canonical_sha256(&cell)? != *hash {
            return Err("expanded periodic cell canonical SHA256 mismatch".into());
        }
        Ok(Some(pme))
    }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WaterReferenceParameters {
    pub oh_distances_angstrom: [f64; 2],
    pub hh_distance_angstrom: f64,
    pub hoh_angle_radian: f64,
    /// Historical single runtime-mass authority; never synthesized for split records.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub masses_dalton: Option<[f64; 3]>,
    /// Original force-field masses before optional repartitioning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_masses_dalton: Option<[f64; 3]>,
    /// Actual integrator/SETTLE masses after optional repartitioning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_masses_dalton: Option<[f64; 3]>,
    pub charges_e: [f64; 3],
    pub oxygen_lj: LjParam,
    pub hydrogen_lj: [LjParam; 2],
}

/// One exact O-H-H water instance from PRISM-PREP v3.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WaterInstance {
    pub residue_index: usize,
    pub oxygen_atom_index: usize,
    pub hydrogen_atom_indices: [usize; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WaterModelContract {
    pub schema_kind: String,
    pub schema_version: u32,
    pub model_id: String,
    pub forcefield_resource: String,
    pub n_waters: usize,
    pub atom_order: String,
    pub rigid_water_in_openmm_system: bool,
    pub runtime_settle_rattle_required: bool,
    pub reference_parameters: WaterReferenceParameters,
    pub reference_parameters_sha256: String,
    pub water_instances: Vec<WaterInstance>,
    pub water_instance_map_sha256: String,
    pub water_instance_parameters_sha256: String,
    pub contract_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mass_repartitioning_applied: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_mass_semantics: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_mass_semantics: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub per_water_mass_conservation_tolerance_dalton: Option<f64>,

}

impl WaterModelContract {
    /// Select the explicitly recorded runtime mass authority without conflating
    /// force-field model masses with integrator masses or rewriting source fields.
    pub fn validated_runtime_masses_dalton(&self) -> std::result::Result<[f64; 3], String> {
        let r = &self.reference_parameters;
        let positive = |m: &[f64; 3]| m.iter().all(|v| v.is_finite() && *v > 0.0);
        match (r.masses_dalton, r.model_masses_dalton, r.runtime_masses_dalton) {
            (Some(legacy), None, None) => {
                if self.model_mass_semantics.is_some() || self.runtime_mass_semantics.is_some()
                    || self.mass_repartitioning_applied.is_some()
                    || self.per_water_mass_conservation_tolerance_dalton.is_some()
                    || !positive(&legacy) {
                    return Err("legacy water masses have mixed or invalid split authority".into());
                }
                Ok(legacy)
            }
            (None, Some(model), Some(runtime)) => {
                let tolerance = self.per_water_mass_conservation_tolerance_dalton
                    .ok_or("split water masses lack declared conservation tolerance")?;
                if self.model_mass_semantics.as_deref() != Some("OPENMM_ELEMENT_MASS_BEFORE_HMR")
                    || self.runtime_mass_semantics.as_deref() != Some("TOPOLOGY_MASS_AFTER_OPTIONAL_HMR")
                    || !positive(&model) || !positive(&runtime)
                    || !tolerance.is_finite() || tolerance < 0.0
                    || (model.iter().sum::<f64>() - runtime.iter().sum::<f64>()).abs() > tolerance {
                    return Err("split water mass semantics/population conservation mismatch".into());
                }
                // Exact predicate emitted by stage2_topology::_build_water_model_contract.
                let repartitioned = model.iter().zip(runtime).any(|(m,r)| (*m-r).abs() > 1.0e-12);
                if self.mass_repartitioning_applied != Some(repartitioned) {
                    return Err("split water repartitioning flag differs from recorded masses".into());
                }
                Ok(runtime)
            }
            _ => Err("water requires legacy-only masses or complete separate model/runtime masses".into()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MoleculeIdentityRow {
    pub molecule_id: u64,
    pub atom_indices: Vec<usize>,
    pub residue_indices: Vec<usize>,
    pub residue_names: Vec<String>,
    pub chain_ids: Vec<String>,
    pub kind: String,
    pub identity: String,
    pub component_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MoleculeIdentityTable {
    pub schema_kind: String,
    pub schema_version: u32,
    pub n_molecules: usize,
    pub n_atoms: usize,
    pub molecules: Vec<MoleculeIdentityRow>,
    pub table_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TopologyProvenanceContract {
    pub schema_kind: String,
    pub schema_version: u32,
    pub producer: String,
    pub producer_sha256: String,
    pub source_pdb: String,
    pub source_pdb_sha256: String,
    pub openmm_version: String,
    pub forcefield_resources: serde_json::Value,
    pub protein_forcefield: String,
    pub water_model: Option<String>,
    pub nonbonded_method: String,
    pub nonbonded_cutoff_angstrom: Option<f64>,
    pub openmm_constraints: String,
    pub openmm_rigid_water: bool,
    pub runtime_constraint_authority_required: bool,
    pub preparation_seed: u64,
    pub random_generator: String,
    /// Preparation pH when emitted by the v3 topology producer.
    #[serde(default)]
    pub ph: Option<f64>,
    #[serde(default)]
    pub explicit_solvent_requested: Option<bool>,
    #[serde(default)]
    pub terminal_caps_requested: Option<bool>,
    #[serde(default)]
    pub hydrogen_mass_repartitioning_applied: Option<bool>,
    #[serde(default)]
    pub minimization_contract_sha256: Option<String>,
    #[serde(default)]
    pub minimization_platform_name: Option<String>,
    #[serde(default)]
    pub minimization_platform_plugin_sha256: Option<String>,
    /// Present only after the authenticated NPT -> frozen-NVT preparation
    /// pipeline has replaced the source coordinates and periodic cell.
    #[serde(default)]
    pub npt_equilibrated: Option<bool>,
    #[serde(default)]
    pub npt_population_authority_sha256: Option<String>,
    #[serde(default)]
    pub npt_seed_authority_sha256: Option<String>,
    #[serde(default)]
    pub npt_frozen_cell_authority_sha256: Option<String>,
    #[serde(default)]
    pub base_minimization_contract_sha256: Option<String>,
    #[serde(default)]
    pub energy_minimization_applied: Option<bool>,
    pub contract_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_topology_scientific_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_environment_minimization_contract_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_environment_minimization_platform_plugin_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fragment_slots_start_at_lambda_zero: Option<bool>,

}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisResidueRecord {
    pub analysis_residue_id: usize,
    pub physical_residue_id: usize,
    pub chain_id: String,
    pub pdb_residue_id: i64,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub insertion_code: String,
    pub residue_name: String,
    pub role: String,
}

/// Hash-bound separation of physical molecular residues from the sole
/// protein-only KCC/Ghost analysis axis.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisResidueContract {
    pub schema_kind: String,
    pub schema_version: u32,
    pub analysis_authority: String,
    pub excluded_atom_sentinel: i32,
    pub n_atoms: usize,
    pub total_physical_residue_count: usize,
    pub n_analysis_residues: usize,
    pub physical_residue_roles: Vec<String>,
    pub physical_to_analysis_residue_ids: Vec<i32>,
    pub analysis_residues: Vec<AnalysisResidueRecord>,
    pub atom_analysis_residue_ids_sha256: String,
    pub physical_molecule_table_sha256: String,
    pub lambda_fragment_system_sha256: Option<String>,
    pub role_counts: BTreeMap<String, usize>,
    pub excluded_roles: Vec<String>,
    pub must_recompute_after_fragment_materialization: bool,
    pub contract_sha256: String,
}

impl AnalysisResidueContract {
    /// Validate author identity without guessing a numbering offset. Blank chains
    /// and signed/zero author numbers are valid; the full tuple must be unique.
    pub fn validate_author_mapping(&self) -> std::result::Result<(), String> {
        if self.schema_kind != "prismos_analysis_residue_mapping"
            || self.schema_version != 1
            || self.analysis_authority != "KCC_GHOST_PROTEIN_RESIDUE_AXIS"
            || self.analysis_residues.len() != self.n_analysis_residues
            || self.physical_to_analysis_residue_ids.len() != self.total_physical_residue_count
        {
            return Err("invalid author residue mapping authority or dimensions".into());
        }
        let mut identities = std::collections::HashSet::new();
        let mut physical = std::collections::HashSet::new();
        for (analysis, row) in self.analysis_residues.iter().enumerate() {
            if row.analysis_residue_id != analysis || row.role != "RECEPTOR"
                || row.residue_name.is_empty()
                || self.physical_to_analysis_residue_ids.get(row.physical_residue_id).copied() != Some(analysis as i32)
                || !physical.insert(row.physical_residue_id)
                || !identities.insert((row.chain_id.as_str(), row.pdb_residue_id, row.insertion_code.as_str()))
            {
                return Err(format!("invalid or duplicate author residue mapping at analysis residue {analysis}"));
            }
        }
        let mut value = serde_json::to_value(self).map_err(|e| e.to_string())?;
        value.as_object_mut().ok_or("author contract is not an object")?.remove("contract_sha256");
        if python_canonical_sha256(&value)? != self.contract_sha256 {
            return Err("author residue mapping contract SHA256 mismatch".into());
        }
        Ok(())
    }

    pub fn author_references(&self, ids: &[i32]) -> std::result::Result<Vec<AnalysisResidueRecord>, String> {
        ids.iter().map(|&id| {
            usize::try_from(id).ok().and_then(|i| self.analysis_residues.get(i))
                .filter(|row| row.analysis_residue_id as i32 == id).cloned()
                .ok_or_else(|| format!("author residue mapping missing analysis residue {id}"))
        }).collect()
    }

    pub fn author_mapping_surface(&self) -> std::result::Result<serde_json::Value, String> {
        self.validate_author_mapping()?;
        Ok(serde_json::json!({
            "status": "available",
            "source": "topology.analysis_residue_contract",
            "contract_sha256": self.contract_sha256,
            "dense_coordinate_system": "zero_based_analysis_residue_id",
            "author_identity_fields": ["chain_id", "pdb_residue_id", "insertion_code"],
            "uniprot_mapping_status": "not_supplied",
            "records": self.analysis_residues,
        }))
    }
}


/// PRISM-PREP topology JSON structure
#[derive(Debug, Clone, Serialize, Deserialize)]

pub struct PrismPrepTopology {
    /// PRISM-PREP schema discriminator.
    #[serde(default)]
    pub schema_kind: Option<String>,
    /// PRISM-PREP schema version (v3 is required for lambda execution).
    #[serde(default)]
    pub schema_version: Option<serde_json::Value>,
    /// P5 — exact aromatic ring geometry from PRISM-PREP. Optional: a topology
    /// prepped before this block existed deserializes to an empty vector and the
    /// engine falls back to its derived path, which is reachable and logged
    /// rather than silently preferred.
    #[serde(default)]
    pub aromatic_targets: Vec<AromaticTargetSpec>,
    /// Source PDB file
    pub source_pdb: String,
    /// Number of atoms
    pub n_atoms: usize,
    /// Number of residues
    pub n_residues: usize,
    /// Physical OpenMM residue count. Kept distinct from the analysis axis.
    #[serde(default)]
    pub total_physical_residue_count: Option<usize>,
    /// Dense receptor/protein-only residue count used by KCC and Ghost.
    #[serde(default)]
    pub n_analysis_residues: Option<usize>,
    /// Number of chains
    pub n_chains: usize,
    /// Atom positions [x0, y0, z0, x1, y1, z1, ...]
    pub positions: Vec<f32>,
    /// Element symbols
    pub elements: Vec<String>,
    /// Atom names (CA, CB, N, O, etc.)
    pub atom_names: Vec<String>,
    /// Residue names (ALA, GLY, etc.)
    pub residue_names: Vec<String>,
    /// Residue IDs for each atom
    pub residue_ids: Vec<usize>,
    /// Explicit physical residue axis; must equal legacy `residue_ids`.
    #[serde(default)]
    pub atom_physical_residue_ids: Option<Vec<usize>>,
    /// Sole KCC/Ghost axis; `-1` excludes solvent, ions and fragments.
    #[serde(default)]
    pub atom_analysis_residue_ids: Option<Vec<i32>>,
    /// Hash-bound mapping and role contract for the analysis axis.
    #[serde(default)]
    pub analysis_residue_contract: Option<AnalysisResidueContract>,
    /// Chain IDs
    pub chain_ids: Vec<String>,
    /// Partial charges
    pub charges: Vec<f32>,
    /// Masses
    pub masses: Vec<f32>,
    /// CA atom indices
    pub ca_indices: Vec<usize>,
    /// Bond parameters
    pub bonds: Vec<BondParam>,
    /// Angle parameters
    pub angles: Vec<AngleParam>,
    /// Explicit source coefficient meaning; absent is historical-inspection-only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub harmonic_parameter_contract: Option<HarmonicParameterContract>,
    /// Dihedral parameters
    pub dihedrals: Vec<DihedralParam>,
    /// LJ parameters per atom
    pub lj_params: Vec<LjParam>,
    /// Exclusions per atom (list of atom indices to exclude from nonbonded for each atom)
    pub exclusions: Vec<Vec<usize>>,
    /// Exact scaled nonbonded parameters, normally the prmtop/OpenMM 1-4 exceptions.
    #[serde(default)]
    pub nonbonded_exceptions: Vec<NonbondedExceptionParam>,
    /// Required completeness/digest contract whenever exact exceptions are present.
    #[serde(default)]
    pub nonbonded_exception_contract: Option<NonbondedExceptionContract>,
    /// SHA-256 of the Amber prmtop that was read by OpenMM for exact force extraction.
    #[serde(default)]
    pub generated_prmtop_sha256: Option<String>,
    /// Hydrogen clusters for SHAKE/RATTLE constraints
    #[serde(default)]
    pub h_clusters: Vec<HCluster>,
    /// TRI-STATE hydrogen-mass-repartitioning authority.
    ///
    /// * `Some(true)`  — the serialized masses and H-cluster inverse masses
    ///   have already undergone HMR. This is runtime authority: applying HMR
    ///   twice corrupts the mass model.
    /// * `Some(false)` — the producer explicitly recorded that HMR was NOT
    ///   applied.
    /// * `None`        — the key was ABSENT from the parsed document. Nothing
    ///   has been established either way.
    ///
    /// **Why this is not a `bool`.** The physical governor's HMR hard gate
    /// (`physical_governor::gate_hmr_hard`, mirrored by
    /// `PRISM_GEAR_PERMISSION_*` in `cuda/gearbox.cuh`) is a THREE-valued
    /// permission: `GRANTED` / `DENIED` / `UNKNOWN`, and only `GRANTED`
    /// unlocks gear 2 (4.0 fs). While this field was a plain `bool` with
    /// `#[serde(default)]`, an absent key deserialized to `false`, so
    /// "unverified" was structurally indistinguishable from "verified not
    /// applied" and any bridge into the governor collapsed `UNKNOWN` into
    /// `DENIED`. The presence of the key IS the witness that the topology
    /// producer made a statement, so `Option<bool>` carries the witness and
    /// the value in one field with no second source of truth to drift.
    ///
    /// `skip_serializing_if` keeps the on-disk shape byte-identical for
    /// topologies that never carried the key: absence round-trips to absence
    /// instead of being promoted to an explicit `null` or `false`.
    ///
    /// Read through [`PrismPrepTopology::hmr_authority`] rather than matching
    /// on the `Option` at call sites; that accessor is the single mapping into
    /// the governor's permission encoding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hmr_applied: Option<bool>,
    /// Timestep recommended by the topology producer, in femtoseconds.
    #[serde(default)]
    pub recommended_timestep_fs: Option<f64>,
    /// Water oxygen indices
    #[serde(default)]
    pub water_oxygens: Vec<usize>,
    /// List of residue metadata (mapping indices to PDB IDs)
    #[serde(default)]
    pub residues: Vec<ResidueMetadata>,
    /// G28 SISR dimer-dyad metadata (Amendment 3.4 substrate-aware path).
    /// Present iff the substrate is a homodimer with a known C2 axis;
    /// when set, the V2 pipeline auto-enables the SISR symmetry-consensus
    /// kernel using the supplied axis + center + tolerance.  Optional —
    /// monomer topologies omit this block entirely.
    #[serde(default)]
    pub dimer_dyad: Option<DimerDyad>,
    /// Optional grand-canonical multisite lambda-fragment topology contract.
    /// When present, runtime admission is fail-closed on verified PME and
    /// SETTLE capabilities; an implicit/vacuum compatibility path is forbidden.
    #[serde(default)]
    pub lambda_fragment_system: Option<LambdaFragmentSystem>,
    /// Full PRISM-PREP v3 periodic-cell contract.
    #[serde(default)]
    pub periodic_cell: Option<PeriodicCellContract>,
    /// Exact TIP3P-FB parameter and instance contract.
    #[serde(default)]
    pub water_model_contract: Option<WaterModelContract>,
    /// Exact connected-component molecule partition.
    #[serde(default)]
    pub molecule_identity_table: Option<MoleculeIdentityTable>,
    /// Producer/source/OpenMM/XML closure provenance.
    #[serde(default)]
    pub topology_provenance: Option<TopologyProvenanceContract>,
    /// Hash-bound authority proving that the serialized positions and cell
    /// were transferred exactly from one admitted member of the NPT
    /// checkpoint population and then frozen for NVT production.
    #[serde(default)]
    pub npt_frozen_cell_contract: Option<serde_json::Value>,
}

/// G28 SISR dyad parameters embedded in topology JSON.
/// `axis` is the rotation axis (unit vector); `center` is the rotation
/// center (Å); `epsilon` is the partner-search tolerance (Å, typically 1.5).
/// The C2 reflection sends `c → 2·center + R(axis,π)·(c − center)`. For the
/// canonical Z-axis dyad at the origin this simplifies to (x,y,z) → (-x,-y,z).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimerDyad {
    /// Rotation axis (unit 3-vector).
    pub axis: [f32; 3],
    /// Rotation center in Å.
    pub center: [f32; 3],
    /// Partner-search tolerance in Å.
    pub epsilon: f32,
}

impl PrismPrepTopology {
    /// Preserve and validate complete recorded PME authority, including no-probe cells.
    /// Runtime capabilities and chemical calibration remain independently required.
    pub fn validate_recorded_periodic_authority(&self) -> std::result::Result<(), String> {
        if let Some(cell) = &self.periodic_cell {
            if let Some(pme) = cell.validated_recorded_pme()? {
                if let Some(system) = &self.lambda_fragment_system {
                    if serde_json::to_value(&pme).map_err(|e| e.to_string())?
                        != serde_json::to_value(&system.pme).map_err(|e| e.to_string())?
                        || Some(system.periodic_box.vectors_a) != cell.vectors_angstrom
                        || !system.periodic_box.minimum_image_enabled
                        || Some(system.periodic_box.source_sha256.as_str()) != cell.contract_sha256.as_deref() {
                        return Err("lambda system PME/box differs from complete recorded periodic authority".into());
                    }
                }
            }
        }
        Ok(())
    }

    /// Verify the physical/analysis residue-axis separation emitted by
    /// PRISM-PREP. KCC and Ghost may consume only `atom_analysis_residue_ids`.
    pub fn validate_analysis_residue_axis(&self) -> std::result::Result<(), String> {
        const PROTEIN: &[&str] = &[
            "ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY", "HIS", "HID", "HIE",
            "HIP", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        ];
        const WATER: &[&str] = &["HOH", "WAT", "TIP3"];
        const ION: &[&str] = &["NA", "K", "CL", "MG", "CA", "ZN"];

        let total_physical = self
            .total_physical_residue_count
            .ok_or_else(|| "v3 topology lacks total_physical_residue_count".to_string())?;
        let n_analysis = self
            .n_analysis_residues
            .ok_or_else(|| "v3 topology lacks n_analysis_residues".to_string())?;
        let physical_ids = self
            .atom_physical_residue_ids
            .as_ref()
            .ok_or_else(|| "v3 topology lacks atom_physical_residue_ids".to_string())?;
        let analysis_ids = self
            .atom_analysis_residue_ids
            .as_ref()
            .ok_or_else(|| "v3 topology lacks atom_analysis_residue_ids".to_string())?;
        let contract = self
            .analysis_residue_contract
            .as_ref()
            .ok_or_else(|| "v3 topology lacks analysis_residue_contract".to_string())?;
        if total_physical != self.n_residues
            || physical_ids != &self.residue_ids
            || physical_ids.len() != self.n_atoms
            || analysis_ids.len() != self.n_atoms
            || contract.schema_kind != "prismos_analysis_residue_mapping"
            || contract.schema_version != 1
            || contract.analysis_authority != "KCC_GHOST_PROTEIN_RESIDUE_AXIS"
            || contract.excluded_atom_sentinel != -1
            || contract.n_atoms != self.n_atoms
            || contract.total_physical_residue_count != total_physical
            || contract.n_analysis_residues != n_analysis
            || contract.physical_residue_roles.len() != total_physical
            || contract.physical_to_analysis_residue_ids.len() != total_physical
            || contract.analysis_residues.len() != n_analysis
            || contract.excluded_roles != ["WATER", "ION", "LAMBDA_FRAGMENT", "OTHER"]
            || !contract.must_recompute_after_fragment_materialization
        {
            return Err("invalid analysis residue contract header or top-level projection".into());
        }
        let observed_physical: Vec<usize> = {
            let mut values: Vec<usize> = physical_ids.clone();
            values.sort_unstable();
            values.dedup();
            values
        };
        if observed_physical != (0..total_physical).collect::<Vec<_>>() {
            return Err("physical residue IDs are not dense from zero".into());
        }

        let table = self.molecule_identity_table.as_ref().ok_or_else(|| {
            "analysis residue contract lacks molecule-table authority".to_string()
        })?;
        if contract.physical_molecule_table_sha256 != table.table_sha256 {
            return Err("analysis axis molecule-table hash mismatch".into());
        }
        let mut atom_roles = vec![None::<String>; self.n_atoms];
        for molecule in &table.molecules {
            let role = match molecule.kind.as_str() {
                "WATER" => "WATER",
                "ION" => "ION",
                _ => "OTHER",
            };
            for &atom in &molecule.atom_indices {
                if atom >= self.n_atoms || atom_roles[atom].replace(role.to_string()).is_some() {
                    return Err(format!(
                        "analysis molecule role partition invalid at atom {atom}"
                    ));
                }
            }
        }
        if atom_roles.iter().any(Option::is_none) {
            return Err("analysis molecule roles do not cover every atom".into());
        }
        let lambda_digest = match &self.lambda_fragment_system {
            Some(system) => {
                let value = serde_json::to_value(system).map_err(|error| {
                    format!("serialize lambda system for analysis digest: {error}")
                })?;
                for molecule in &system.molecules {
                    let role = match molecule.role {
                        crate::lambda_fragment::MoleculeRole::Receptor => "RECEPTOR",
                        crate::lambda_fragment::MoleculeRole::Water => "WATER",
                        crate::lambda_fragment::MoleculeRole::Ion => "ION",
                        crate::lambda_fragment::MoleculeRole::LambdaFragment => "LAMBDA_FRAGMENT",
                        crate::lambda_fragment::MoleculeRole::Other => "OTHER",
                    };
                    for &atom in &molecule.atom_indices {
                        if atom >= self.n_atoms {
                            return Err(format!("lambda molecule atom out of range: {atom}"));
                        }
                        atom_roles[atom] = Some(role.to_string());
                    }
                }
                Some(python_canonical_sha256(&value)?)
            }
            None => None,
        };
        if contract.lambda_fragment_system_sha256 != lambda_digest {
            return Err("analysis axis lambda-fragment-system hash mismatch".into());
        }

        let mut expected_physical_roles = vec!["OTHER".to_string(); total_physical];
        let mut physical_names = vec![None::<String>; total_physical];
        let mut physical_chains = vec![None::<String>; total_physical];
        for atom in 0..self.n_atoms {
            let physical = physical_ids[atom];
            if physical_names[physical].is_none() {
                physical_names[physical] = Some(self.residue_names[atom].clone());
                physical_chains[physical] = Some(self.chain_ids[atom].clone());
            } else if physical_names[physical].as_deref() != Some(self.residue_names[atom].as_str())
                || physical_chains[physical].as_deref() != Some(self.chain_ids[atom].as_str())
            {
                return Err(format!(
                    "physical residue {physical} has inconsistent atom identity"
                ));
            }
            let molecule_role = atom_roles[atom].as_deref().unwrap_or("OTHER");
            let name = self.residue_names[atom].as_str();
            let role = if molecule_role == "LAMBDA_FRAGMENT" {
                "LAMBDA_FRAGMENT"
            } else if PROTEIN.contains(&name) {
                "RECEPTOR"
            } else if WATER.contains(&name) || molecule_role == "WATER" {
                "WATER"
            } else if ION.contains(&name.to_ascii_uppercase().as_str()) || molecule_role == "ION" {
                "ION"
            } else {
                "OTHER"
            };
            let prior = expected_physical_roles[physical].as_str();
            if prior != "OTHER" && prior != role {
                return Err(format!(
                    "physical residue {physical} spans conflicting roles"
                ));
            }
            expected_physical_roles[physical] = role.to_string();
        }
        if contract.physical_residue_roles != expected_physical_roles {
            return Err("analysis physical residue roles differ from molecular identity".into());
        }

        let mut expected_physical_to_analysis = vec![-1i32; total_physical];
        let mut next_analysis = 0i32;
        for (physical, role) in expected_physical_roles.iter().enumerate() {
            if role == "RECEPTOR" {
                expected_physical_to_analysis[physical] = next_analysis;
                next_analysis += 1;
            }
        }
        if next_analysis as usize != n_analysis
            || contract.physical_to_analysis_residue_ids != expected_physical_to_analysis
        {
            return Err("analysis physical-to-dense mapping mismatch".into());
        }
        let expected_atom_analysis: Vec<i32> = physical_ids
            .iter()
            .map(|&physical| expected_physical_to_analysis[physical])
            .collect();
        if analysis_ids != &expected_atom_analysis {
            return Err("atom analysis map differs from receptor-only mapping".into());
        }
        let map_hash = python_canonical_sha256(
            &serde_json::to_value(analysis_ids).map_err(|error| error.to_string())?,
        )?;
        if contract.atom_analysis_residue_ids_sha256 != map_hash {
            return Err("atom analysis map SHA256 mismatch".into());
        }
        for (analysis, row) in contract.analysis_residues.iter().enumerate() {
            if row.analysis_residue_id != analysis
                || row.role != "RECEPTOR"
                || expected_physical_to_analysis
                    .get(row.physical_residue_id)
                    .copied()
                    != Some(analysis as i32)
                || physical_names[row.physical_residue_id].as_deref()
                    != Some(row.residue_name.as_str())
                || physical_chains[row.physical_residue_id].as_deref()
                    != Some(row.chain_id.as_str())
            {
                return Err(format!("analysis residue row {analysis} is inconsistent"));
            }
        }
        contract.validate_author_mapping()?;
        let expected_role_counts = expected_physical_roles.iter().fold(
            BTreeMap::<String, usize>::new(),
            |mut counts, role| {
                *counts.entry(role.clone()).or_default() += 1;
                counts
            },
        );
        if contract.role_counts != expected_role_counts {
            return Err("analysis role counts mismatch".into());
        }
        let mut contract_value = serde_json::to_value(contract)
            .map_err(|error| format!("serialize analysis contract: {error}"))?;
        contract_value
            .as_object_mut()
            .ok_or_else(|| "analysis contract is not an object".to_string())?
            .remove("contract_sha256");
        let recomputed = python_canonical_sha256(&contract_value)?;
        if contract.contract_sha256 != recomputed {
            return Err(format!(
                "analysis residue contract SHA256 mismatch: declared={} recomputed={recomputed}",
                contract.contract_sha256
            ));
        }
        Ok(())
    }

    pub fn validate_for_nonbonded_execution(
        &self,
        policy: NonbondedExecutionPolicy,
    ) -> std::result::Result<(), String> {
        self.validate()?;
        match policy {
            NonbondedExecutionPolicy::RequireHistoricalMayModel => {
                self.historical_may_nonbonded_execution_contract()?;
            }
            NonbondedExecutionPolicy::RequireExactV2 => {
                if self.nonbonded_exceptions.is_empty()
                    || self.nonbonded_exception_contract.is_none()
                {
                    return Err(
                        "production nonbonded execution requires exact OpenMM exception parameters and a complete V2 contract; use --non-production-legacy-nonbonded only for explicitly partial compatibility runs"
                            .to_string(),
                    );
                }
            }
            NonbondedExecutionPolicy::AllowNonProductionLegacyUniformFudge => {
                if !self.nonbonded_exceptions.is_empty()
                    || self.nonbonded_exception_contract.is_some()
                {
                    return Err(
                        "non-production legacy nonbonded policy is valid only for an exception-free legacy topology"
                            .to_string(),
                    );
                }
            }
        }
        Ok(())
    }

    /// Validate a lambda-fragment experiment against capabilities of the
    /// selected runtime. Source-file presence is not capability evidence.
    pub fn validate_lambda_fragment_execution(
        &self,
        runtime: &LambdaRuntimeCapabilities,
    ) -> std::result::Result<(), String> {
        match &self.lambda_fragment_system {
            Some(system) => {
                self.validate_lambda_preparation_contract(system, Some(runtime))?;
                validate_production_admission(self, system, runtime)
            }
            None => Ok(()),
        }
    }

    /// Validate immutable PRISM-PREP lambda topology content before a CUDA
    /// runtime exists. Runtime hash/capability promotion remains a separate
    /// post-construction gate and is never inferred from source presence.
    pub fn validate_lambda_fragment_preparation(&self) -> std::result::Result<(), String> {
        self.validate_recorded_periodic_authority()?;
        match &self.lambda_fragment_system {
            Some(system) => self.validate_lambda_preparation_contract(system, None),
            None => Ok(()),
        }
    }

    fn validate_lambda_preparation_contract(
        &self,
        system: &LambdaFragmentSystem,
        runtime: Option<&LambdaRuntimeCapabilities>,
    ) -> std::result::Result<(), String> {
        self.validate_recorded_periodic_authority()?;
        let valid_sha =
            |value: &str| value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit());
        if self.schema_kind.as_deref() != Some("prismos_openmm_topology")
            || self
                .schema_version
                .as_ref()
                .and_then(serde_json::Value::as_str)
                != Some("3.0.0")
        {
            return Err(
                "lambda execution requires PRISM-PREP prismos_openmm_topology schema 3.0.0".into(),
            );
        }
        self.validate_lambda_preparation_digests()?;
        let periodic = self
            .periodic_cell
            .as_ref()
            .ok_or_else(|| "lambda topology lacks periodic_cell v3 contract".to_string())?;
        let periodic_hash = periodic
            .contract_sha256
            .as_deref()
            .ok_or_else(|| "periodic_cell lacks contract_sha256".to_string())?;
        let vectors = periodic
            .vectors_angstrom
            .ok_or_else(|| "periodic_cell lacks full 3x3 vectors".to_string())?;
        let volume = periodic
            .volume_angstrom3
            .ok_or_else(|| "periodic_cell lacks volume".to_string())?;
        if !periodic.periodic
            || periodic.minimum_image_required != Some(true)
            || periodic.coordinate_wrapping_required != Some(true)
            || periodic.reciprocal_method.as_deref() != Some("PME")
            || !valid_sha(periodic_hash)
            || system.periodic_box.source_sha256 != periodic_hash
            || vectors != system.periodic_box.vectors_a
            || !volume.is_finite()
            || (volume - system.periodic_box.volume_a3()).abs() > 1.0e-6
        {
            return Err("lambda periodic-cell contract mismatch".into());
        }
        if let Some(runtime) = runtime {
            if runtime.periodic_cell_contract_sha256.as_deref() != Some(periodic_hash) {
                return Err(
                    "selected runtime is not hash-bound to the prepared periodic cell".into(),
                );
            }
        }

        let water = self
            .water_model_contract
            .as_ref()
            .ok_or_else(|| "lambda topology lacks water_model_contract".to_string())?;
        if water.schema_kind != "prismos_water_model_contract"
            || water.schema_version != 1
            || water.model_id != "TIP3P-FB"
            || water.atom_order != "OHH"
            || !water.runtime_settle_rattle_required
            || water.rigid_water_in_openmm_system
            || water.n_waters != self.water_oxygens.len()
            || !valid_sha(&water.reference_parameters_sha256)
            || !valid_sha(&water.water_instance_map_sha256)
            || !valid_sha(&water.contract_sha256)
            || system.water.model != water.model_id
            || system.water.parameterization_sha256 != water.contract_sha256
            || (system.water.oh_distance_a - water.reference_parameters.oh_distances_angstrom[0])
                .abs()
                > 1.0e-8
            || (water.reference_parameters.oh_distances_angstrom[0]
                - water.reference_parameters.oh_distances_angstrom[1])
                .abs()
                > 1.0e-8
            || (system.water.hoh_angle_rad - water.reference_parameters.hoh_angle_radian).abs()
                > 1.0e-8
        {
            return Err("lambda water model is not the prepared TIP3P-FB contract".into());
        }
        if let Some(runtime) = runtime {
            if runtime.rigid_water_model_id.as_deref() != Some(water.model_id.as_str())
                || runtime.rigid_water_parameterization_sha256.as_deref()
                    != Some(water.contract_sha256.as_str())
            {
                return Err(
                    "selected SETTLE/RATTLE runtime does not match water model/hash".into(),
                );
            }
        }

        let table = self
            .molecule_identity_table
            .as_ref()
            .ok_or_else(|| "lambda topology lacks molecule_identity_table".to_string())?;
        if table.schema_kind != "prismos_molecule_identity_table"
            || table.schema_version != 1
            || table.n_atoms != self.n_atoms
            || table.n_molecules != table.molecules.len()
            || !valid_sha(&table.table_sha256)
        {
            return Err("invalid molecule identity table header".into());
        }
        let mut owners = vec![None; self.n_atoms];
        let table_by_id: HashMap<u64, &MoleculeIdentityRow> = table
            .molecules
            .iter()
            .map(|row| (row.molecule_id, row))
            .collect();
        if table_by_id.len() != table.molecules.len() {
            return Err("duplicate molecule id in preparation table".into());
        }
        for row in &table.molecules {
            if !valid_sha(&row.component_sha256) || row.atom_indices.is_empty() {
                return Err(format!("invalid prepared molecule {}", row.molecule_id));
            }
            for &atom in &row.atom_indices {
                if atom >= self.n_atoms || owners[atom].replace(row.molecule_id).is_some() {
                    return Err(format!(
                        "prepared molecule atom partition invalid at {atom}"
                    ));
                }
            }
        }
        if owners.iter().any(Option::is_none) {
            return Err("prepared molecule table is not an exact atom partition".into());
        }
        for molecule in &system.molecules {
            let prepared = table_by_id.get(&molecule.molecule_id).ok_or_else(|| {
                format!(
                    "lambda molecule {} absent from prep table",
                    molecule.molecule_id
                )
            })?;
            if prepared.atom_indices != molecule.atom_indices {
                return Err(format!(
                    "lambda molecule {} differs from prep connected component",
                    molecule.molecule_id
                ));
            }
        }

        let provenance = self
            .topology_provenance
            .as_ref()
            .ok_or_else(|| "lambda topology lacks topology_provenance".to_string())?;
        let resource_hash = provenance
            .forcefield_resources
            .get("resource_closure_sha256")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| "topology provenance lacks force-field hash closure".to_string())?;
        let resources = provenance
            .forcefield_resources
            .get("resources")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| "topology provenance lacks force-field resources".to_string())?;
        if provenance.schema_kind != "prismos_topology_provenance"
            || provenance.schema_version != 1
            || provenance.water_model.as_deref() != Some("TIP3P-FB")
            || provenance.nonbonded_method != "PME"
            || provenance.openmm_constraints != "None"
            || provenance.openmm_rigid_water
            || !provenance.runtime_constraint_authority_required
            || !valid_sha(&provenance.producer_sha256)
            || !valid_sha(&provenance.source_pdb_sha256)
            || !valid_sha(resource_hash)
            || !valid_sha(&provenance.contract_sha256)
            || resources.is_empty()
            || resources.iter().any(|row| {
                row.get("sha256")
                    .and_then(serde_json::Value::as_str)
                    .map(|value| !valid_sha(value))
                    .unwrap_or(true)
            })
        {
            return Err("incomplete PRISM-PREP v3/OpenMM force-field provenance".into());
        }
        Ok(())
    }

    /// Recompute every PRISM-PREP v3 canonical digest consumed by lambda
    /// admission. A syntactically valid 64-hex string is not evidence that the
    /// decoded content is the content the producer hashed.
    fn validate_lambda_preparation_digests(&self) -> std::result::Result<(), String> {
        fn to_value<T: Serialize>(
            value: &T,
            label: &str,
        ) -> std::result::Result<serde_json::Value, String> {
            serde_json::to_value(value)
                .map_err(|error| format!("serialize {label} for digest recomputation: {error}"))
        }

        fn without_field(
            value: &impl Serialize,
            field: &str,
            label: &str,
        ) -> std::result::Result<serde_json::Value, String> {
            let mut value = to_value(value, label)?;
            let object = value
                .as_object_mut()
                .ok_or_else(|| format!("{label} is not a JSON object"))?;
            object.remove(field);
            object.retain(|_, value| !value.is_null());
            Ok(value)
        }

        fn require_digest(
            label: &str,
            value: &serde_json::Value,
            expected: &str,
        ) -> std::result::Result<(), String> {
            let observed = python_canonical_sha256(value)?;
            if observed != expected {
                return Err(format!(
                    "{label} canonical SHA256 mismatch: declared={expected} recomputed={observed}"
                ));
            }
            Ok(())
        }

        let periodic = self
            .periodic_cell
            .as_ref()
            .ok_or_else(|| "lambda topology lacks periodic_cell v3 contract".to_string())?;
        let periodic_hash = periodic
            .contract_sha256
            .as_deref()
            .ok_or_else(|| "periodic_cell lacks contract_sha256".to_string())?;
        require_digest(
            "periodic_cell",
            &without_field(periodic, "contract_sha256", "periodic_cell")?,
            periodic_hash,
        )?;

        let water = self
            .water_model_contract
            .as_ref()
            .ok_or_else(|| "lambda topology lacks water_model_contract".to_string())?;
        let reference_value = to_value(&water.reference_parameters, "water reference parameters")?;
        require_digest(
            "water reference parameters",
            &reference_value,
            &water.reference_parameters_sha256,
        )?;
        let instances_value = to_value(&water.water_instances, "water instance map")?;
        require_digest(
            "water instance map",
            &instances_value,
            &water.water_instance_map_sha256,
        )?;
        if water.water_instances.len() != water.n_waters
            || water.n_waters != self.water_oxygens.len()
        {
            return Err("water instance count does not match topology water count".into());
        }
        let runtime_water_masses = water.validated_runtime_masses_dalton()?;
        let mut covered_water_atoms = HashSet::with_capacity(water.n_waters * 3);
        let mut covered_oxygens = HashSet::with_capacity(water.n_waters);
        let mut water_rows = Vec::with_capacity(water.n_waters);
        let close = |actual: f64, expected: f64| {
            actual.is_finite()
                && expected.is_finite()
                && (actual - expected).abs() <= 2.0e-6_f64.max(expected.abs() * 2.0e-6)
        };
        for instance in &water.water_instances {
            let [h1, h2] = instance.hydrogen_atom_indices;
            let o = instance.oxygen_atom_index;
            if o >= self.n_atoms
                || h1 >= self.n_atoms
                || h2 >= self.n_atoms
                || o == h1
                || o == h2
                || h1 == h2
                || !covered_oxygens.insert(o)
                || !covered_water_atoms.insert(o)
                || !covered_water_atoms.insert(h1)
                || !covered_water_atoms.insert(h2)
            {
                return Err(format!(
                    "invalid or overlapping water OHH instance at residue {}",
                    instance.residue_index
                ));
            }
            if self.elements[o].trim() != "O"
                || self.elements[h1].trim() != "H"
                || self.elements[h2].trim() != "H"
            {
                return Err(format!(
                    "water residue {} is not an exact OHH element triplet",
                    instance.residue_index
                ));
            }
            let reference = &water.reference_parameters;
            for (label, actual, expected) in [
                (
                    "oxygen mass",
                    self.masses[o] as f64,
                    runtime_water_masses[0],
                ),
                (
                    "H1 mass",
                    self.masses[h1] as f64,
                    runtime_water_masses[1],
                ),
                (
                    "H2 mass",
                    self.masses[h2] as f64,
                    runtime_water_masses[2],
                ),
                (
                    "oxygen charge",
                    self.charges[o] as f64,
                    reference.charges_e[0],
                ),
                ("H1 charge", self.charges[h1] as f64, reference.charges_e[1]),
                ("H2 charge", self.charges[h2] as f64, reference.charges_e[2]),
                (
                    "oxygen sigma",
                    self.lj_params[o].sigma,
                    reference.oxygen_lj.sigma,
                ),
                (
                    "oxygen epsilon",
                    self.lj_params[o].epsilon,
                    reference.oxygen_lj.epsilon,
                ),
                (
                    "H1 sigma",
                    self.lj_params[h1].sigma,
                    reference.hydrogen_lj[0].sigma,
                ),
                (
                    "H1 epsilon",
                    self.lj_params[h1].epsilon,
                    reference.hydrogen_lj[0].epsilon,
                ),
                (
                    "H2 sigma",
                    self.lj_params[h2].sigma,
                    reference.hydrogen_lj[1].sigma,
                ),
                (
                    "H2 epsilon",
                    self.lj_params[h2].epsilon,
                    reference.hydrogen_lj[1].epsilon,
                ),
            ] {
                if !close(actual, expected) {
                    return Err(format!(
                        "water residue {} {label} differs from prepared reference",
                        instance.residue_index
                    ));
                }
            }
            // Preserve the producer's exact reference field names in the row hash.
            // Split model/runtime authority must not acquire a synthetic legacy field.
            let mut row = to_value(reference, "water reference row")?;
            let object = row.as_object_mut().ok_or("water reference is not an object")?;
            object.insert("residue_index".into(), serde_json::json!(instance.residue_index));
            object.insert("oxygen_atom_index".into(), serde_json::json!(o));
            object.insert("hydrogen_atom_indices".into(), serde_json::json!([h1,h2]));
            water_rows.push(row);
        }
        let declared_oxygens: HashSet<usize> = self.water_oxygens.iter().copied().collect();
        if covered_oxygens != declared_oxygens {
            return Err("water instance map does not exactly cover water_oxygens".into());
        }
        require_digest(
            "water instance parameters",
            &serde_json::Value::Array(water_rows),
            &water.water_instance_parameters_sha256,
        )?;
        require_digest(
            "water model contract",
            &without_field(water, "contract_sha256", "water model contract")?,
            &water.contract_sha256,
        )?;

        let table = self
            .molecule_identity_table
            .as_ref()
            .ok_or_else(|| "lambda topology lacks molecule_identity_table".to_string())?;
        for molecule in &table.molecules {
            let lineage = serde_json::json!({
                "atom_indices": molecule.atom_indices,
                "residue_indices": molecule.residue_indices,
                "residue_names": molecule.residue_names,
                "chain_ids": molecule.chain_ids,
                "kind": molecule.kind,
                "identity": molecule.identity,
            });
            require_digest(
                &format!("molecule {} component", molecule.molecule_id),
                &lineage,
                &molecule.component_sha256,
            )?;
        }
        require_digest(
            "molecule identity table",
            &without_field(table, "table_sha256", "molecule identity table")?,
            &table.table_sha256,
        )?;

        let provenance = self
            .topology_provenance
            .as_ref()
            .ok_or_else(|| "lambda topology lacks topology_provenance".to_string())?;
        if provenance.random_generator.trim().is_empty() {
            return Err("topology provenance random_generator is empty".into());
        }
        let resources = provenance
            .forcefield_resources
            .get("resources")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| "topology provenance lacks force-field resources".to_string())?;
        let resource_names: Vec<&str> = resources
            .iter()
            .map(|row| {
                row.get("resource")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| "force-field resource row lacks resource name".to_string())
            })
            .collect::<std::result::Result<_, _>>()?;
        if resource_names.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err("force-field resources are not strictly resource-sorted".into());
        }
        let resource_hash = provenance
            .forcefield_resources
            .get("resource_closure_sha256")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| "topology provenance lacks force-field hash closure".to_string())?;
        require_digest(
            "force-field resource closure",
            &serde_json::Value::Array(resources.clone()),
            resource_hash,
        )?;
        require_digest(
            "topology provenance",
            &without_field(provenance, "contract_sha256", "topology provenance")?,
            &provenance.contract_sha256,
        )?;
        Ok(())
    }

    /// Stable digest over sorted `(i,j,qprod,sigma,epsilon)` tuples.
    pub fn nonbonded_exception_digest_sha256(&self) -> String {
        let mut rows: Vec<&NonbondedExceptionParam> = self.nonbonded_exceptions.iter().collect();
        rows.sort_by_key(|row| (row.i, row.j));
        let mut digest = Sha256::new();
        digest.update(b"PRISMOS_NBEX_ACTIVE_V1\0");
        digest.update((rows.len() as u64).to_le_bytes());
        for row in rows {
            digest.update((row.i as u64).to_le_bytes());
            digest.update((row.j as u64).to_le_bytes());
            digest.update(row.charge_product_e2.to_le_bytes());
            digest.update(row.sigma_a.to_le_bytes());
            digest.update(row.epsilon_kcal_mol.to_le_bytes());
        }
        format!("{:x}", digest.finalize())
    }

    /// Stable digest over the complete canonical strict-exclusion pair set.
    pub fn strict_exclusion_digest_sha256(&self) -> String {
        let mut pairs = HashSet::new();
        for (index, partners) in self.exclusions.iter().enumerate() {
            for &partner in partners {
                pairs.insert((index.min(partner), index.max(partner)));
            }
        }
        let mut pairs = pairs.into_iter().collect::<Vec<_>>();
        pairs.sort_unstable();
        let mut digest = Sha256::new();
        digest.update(b"PRISMOS_NBEX_STRICT_V1\0");
        digest.update((pairs.len() as u64).to_le_bytes());
        for (i, j) in pairs {
            digest.update((i as u64).to_le_bytes());
            digest.update((j as u64).to_le_bytes());
        }
        format!("{:x}", digest.finalize())
    }

    /// Load topology from PRISM-PREP JSON file
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        log::info!("Loading PRISM-PREP topology from: {}", path.display());

        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read topology file: {}", path.display()))?;

        let topology: Self = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse topology JSON: {}", path.display()))?;

        log::info!(
            "Loaded topology: {} atoms, {} residues, {} chains",
            topology.n_atoms,
            topology.n_residues,
            topology.n_chains
        );

        topology
            .validate()
            .map_err(|e| anyhow::anyhow!("Topology validation failed: {}", e))?;
        Ok(topology)
    }

    /// Validate topology data consistency.
    pub fn validate(&self) -> Result<(), String> {
        if let Some(contract) = &self.harmonic_parameter_contract {
            contract.validate_for_topology(self)?;
        }
        self.validate_recorded_periodic_authority()?;
        if let Some(water) = &self.water_model_contract { water.validated_runtime_masses_dalton()?; }
        for (name, len, expected) in [
            ("positions", self.positions.len(), self.n_atoms * 3),
            ("elements", self.elements.len(), self.n_atoms),
            ("atom_names", self.atom_names.len(), self.n_atoms),
            ("residue_names", self.residue_names.len(), self.n_atoms),
            ("residue_ids", self.residue_ids.len(), self.n_atoms),
            ("chain_ids", self.chain_ids.len(), self.n_atoms),
            ("charges", self.charges.len(), self.n_atoms),
            ("masses", self.masses.len(), self.n_atoms),
            ("lj_params", self.lj_params.len(), self.n_atoms),
            ("exclusions", self.exclusions.len(), self.n_atoms),
        ] {
            if len != expected {
                return Err(format!("{} length {} != expected {}", name, len, expected));
            }
        }
        if self.residue_names.len() != self.n_atoms {
            return Err(format!(
                "residue_names length {} != n_atoms {}",
                self.residue_names.len(),
                self.n_atoms
            ));
        }
        if self.residue_ids.len() != self.n_atoms {
            return Err(format!(
                "residue_ids length {} != n_atoms {}",
                self.residue_ids.len(),
                self.n_atoms
            ));
        }
        let max_res_id = self.residue_ids.iter().max().copied().unwrap_or(0);
        if max_res_id >= self.n_residues {
            return Err(format!(
                "max residue_id {} >= n_residues {}",
                max_res_id, self.n_residues
            ));
        }
        for (index, &value) in self.positions.iter().enumerate() {
            if !value.is_finite() {
                return Err(format!("position value {index} is non-finite"));
            }
        }
        for (index, &value) in self.charges.iter().enumerate() {
            if !value.is_finite() {
                return Err(format!("charge value {index} is non-finite"));
            }
        }
        for (index, &value) in self.masses.iter().enumerate() {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("mass value {index} must be finite and positive"));
            }
        }
        for (index, value) in self.lj_params.iter().enumerate() {
            if !value.sigma.is_finite()
                || !value.epsilon.is_finite()
                || value.sigma < 0.0
                || value.epsilon < 0.0
            {
                return Err(format!("invalid LJ parameters for atom {index}"));
            }
        }
        let mut ca_indices = HashSet::new();
        for &index in &self.ca_indices {
            if index >= self.n_atoms || !ca_indices.insert(index) {
                return Err(format!("invalid or duplicate CA atom index {index}"));
            }
        }
        for (index, bond) in self.bonds.iter().enumerate() {
            if bond.i >= self.n_atoms
                || bond.j >= self.n_atoms
                || bond.i == bond.j
                || !bond.r0.is_finite()
                || !bond.k.is_finite()
                || bond.r0 <= 0.0
                || bond.k < 0.0
            {
                return Err(format!("invalid bond term at index {index}"));
            }
        }
        for (index, angle) in self.angles.iter().enumerate() {
            if angle.i >= self.n_atoms
                || angle.j >= self.n_atoms
                || angle.k_idx >= self.n_atoms
                || angle.i == angle.j
                || angle.j == angle.k_idx
                || angle.i == angle.k_idx
                || !angle.theta0.is_finite()
                || !angle.force_k.is_finite()
                || angle.theta0 <= 0.0
                || angle.theta0 > std::f64::consts::PI
                || angle.force_k < 0.0
            {
                return Err(format!("invalid angle term at index {index}"));
            }
        }
        for (index, dihedral) in self.dihedrals.iter().enumerate() {
            if [dihedral.i, dihedral.j, dihedral.k_idx, dihedral.l]
                .iter()
                .any(|&atom| atom >= self.n_atoms)
                || dihedral.periodicity <= 0
                || !dihedral.phase.is_finite()
                || !dihedral.force_k.is_finite()
            {
                return Err(format!("invalid dihedral term at index {index}"));
            }
        }
        let mut strict_exclusion_pairs = HashSet::new();
        for (index, partners) in self.exclusions.iter().enumerate() {
            let mut local = HashSet::new();
            for &partner in partners {
                if partner >= self.n_atoms || partner == index || !local.insert(partner) {
                    return Err(format!(
                        "invalid, self, or duplicate strict exclusion ({index}, {partner})"
                    ));
                }
                if !self.exclusions[partner].contains(&index) {
                    return Err(format!("asymmetric strict exclusion ({index}, {partner})"));
                }
                strict_exclusion_pairs.insert((index.min(partner), index.max(partner)));
            }
        }
        let mut exception_pairs = HashSet::new();
        for exception in &self.nonbonded_exceptions {
            if exception.i >= self.n_atoms || exception.j >= self.n_atoms {
                return Err(format!(
                    "nonbonded exception index out of range: ({}, {}) for {} atoms",
                    exception.i, exception.j, self.n_atoms
                ));
            }
            if exception.i >= exception.j {
                return Err(format!(
                    "nonbonded exception pair must be canonical i < j: ({}, {})",
                    exception.i, exception.j
                ));
            }
            if !exception.charge_product_e2.is_finite()
                || !exception.sigma_a.is_finite()
                || !exception.epsilon_kcal_mol.is_finite()
                || exception.sigma_a < 0.0
                || exception.epsilon_kcal_mol < 0.0
            {
                return Err(format!(
                    "invalid nonbonded exception parameters for pair ({}, {})",
                    exception.i, exception.j
                ));
            }
            for (field, value) in [
                ("charge_product_e2", exception.charge_product_e2),
                ("sigma_A", exception.sigma_a),
                ("epsilon_kcal_mol", exception.epsilon_kcal_mol),
            ] {
                let narrowed = value as f32;
                if !narrowed.is_finite()
                    || (value != 0.0 && narrowed == 0.0)
                    || (value == 0.0 && value.is_sign_negative())
                {
                    return Err(format!(
                        "nonbonded exception {field} is not safely representable as f32 for pair ({}, {})",
                        exception.i, exception.j
                    ));
                }
                let error = ((narrowed as f64) - value).abs();
                let tolerance = 1.0e-12_f64.max(value.abs() * 1.0e-6);
                if error > tolerance {
                    return Err(format!(
                        "nonbonded exception {field} f32 conversion error {error} exceeds {tolerance} for pair ({}, {})",
                        exception.i, exception.j
                    ));
                }
            }
            if !exception_pairs.insert((exception.i, exception.j)) {
                return Err(format!(
                    "duplicate nonbonded exception pair: ({}, {})",
                    exception.i, exception.j
                ));
            }
            if strict_exclusion_pairs.contains(&(exception.i, exception.j)) {
                return Err(format!(
                    "nonbonded exception overlaps strict exclusion: ({}, {})",
                    exception.i, exception.j
                ));
            }
        }
        match (
            self.nonbonded_exceptions.is_empty(),
            self.nonbonded_exception_contract.as_ref(),
        ) {
            (true, None) => {}
            (true, Some(_)) => {
                return Err(
                    "nonbonded exception contract is present but the exact pair vector is empty"
                        .to_string(),
                )
            }
            (false, None) => {
                return Err(
                    "exact nonbonded exceptions require a versioned completeness/digest contract"
                        .to_string(),
                )
            }
            (false, Some(contract)) => {
                if contract.schema_kind != "prismos_exact_nonbonded_exceptions_v2"
                    || contract.schema_version != 2
                {
                    return Err(format!(
                        "unsupported nonbonded exception contract: {} v{}",
                        contract.schema_kind, contract.schema_version
                    ));
                }
                if contract.semantics
                    != "OPENMM_NONBONDEDFORCE_ABSOLUTE_EXCEPTION_PARAMETERS_V1"
                    || contract.dynamic_charge_scaling_semantics
                        != "EXCITED_STATE_MULTIPLIES_GROUND_STATE_CHARGE_PRODUCT_BY_PER_ATOM_SCALE_PRODUCT_V1"
                {
                    return Err("unsupported exact nonbonded exception semantics".to_string());
                }
                if contract.pair_digest_algorithm
                    != "SHA256_PRISMOS_NBEX_ACTIVE_V1_LE_U64_U64_F64_F64_F64"
                    || contract.strict_exclusion_digest_algorithm
                        != "SHA256_PRISMOS_NBEX_STRICT_V1_LE_U64_U64"
                {
                    return Err(format!(
                        "unsupported nonbonded exception digest algorithms: active={} strict={}",
                        contract.pair_digest_algorithm, contract.strict_exclusion_digest_algorithm
                    ));
                }
                if !contract.complete
                    || contract.fallback_used
                    || contract.hardcoded_uniform_scaling_allowed
                {
                    return Err(
                        "exact nonbonded exception contract is incomplete or records fallback"
                            .to_string(),
                    );
                }
                if contract.exact_scaled_pair_count != self.nonbonded_exceptions.len() {
                    return Err(format!(
                        "nonbonded exception count mismatch: contract={} actual={}",
                        contract.exact_scaled_pair_count,
                        self.nonbonded_exceptions.len()
                    ));
                }
                if contract.strict_exclusion_pair_count != strict_exclusion_pairs.len() {
                    return Err(format!(
                        "strict exclusion count mismatch: contract={} actual={}",
                        contract.strict_exclusion_pair_count,
                        strict_exclusion_pairs.len()
                    ));
                }
                if contract.total_openmm_exception_count
                    != contract.exact_scaled_pair_count + contract.strict_exclusion_pair_count
                {
                    return Err(format!(
                        "OpenMM exception completeness mismatch: total={} active={} strict={}",
                        contract.total_openmm_exception_count,
                        contract.exact_scaled_pair_count,
                        contract.strict_exclusion_pair_count
                    ));
                }
                let actual_digest = self.nonbonded_exception_digest_sha256();
                if contract.pair_digest_sha256 != actual_digest {
                    return Err(format!(
                        "nonbonded exception digest mismatch: contract={} actual={}",
                        contract.pair_digest_sha256, actual_digest
                    ));
                }
                let actual_strict_digest = self.strict_exclusion_digest_sha256();
                if contract.strict_exclusion_digest_sha256 != actual_strict_digest {
                    return Err(format!(
                        "strict exclusion digest mismatch: contract={} actual={}",
                        contract.strict_exclusion_digest_sha256, actual_strict_digest
                    ));
                }
                let source_prmtop_sha256 =
                    self.generated_prmtop_sha256.as_deref().ok_or_else(|| {
                        "exact nonbonded exceptions require generated_prmtop_sha256 provenance"
                            .to_string()
                    })?;
                if contract.source_prmtop_sha256 != source_prmtop_sha256
                    || source_prmtop_sha256.len() != 64
                    || !source_prmtop_sha256
                        .bytes()
                        .all(|byte| byte.is_ascii_hexdigit())
                {
                    return Err(
                        "exact exception source prmtop hash is absent, malformed, or mismatched"
                            .to_string(),
                    );
                }
            }
        }
        if self.schema_kind.as_deref() == Some("prismos_openmm_topology")
            && self
                .schema_version
                .as_ref()
                .and_then(serde_json::Value::as_str)
                == Some("3.0.0")
        {
            self.validate_analysis_residue_axis()?;
        }
        Ok(())
    }

    /// Build a map from residue_id to residue name.
    /// residue_names is atom-indexed; this gives the correct per-residue name.
    pub fn residue_id_to_name(&self) -> Vec<String> {
        let mut map = vec!["UNK".to_string(); self.n_residues];
        for (atom_idx, &res_id) in self.residue_ids.iter().enumerate() {
            if res_id < self.n_residues && map[res_id] == "UNK" {
                map[res_id] = self.residue_names[atom_idx].clone();
            }
        }
        map
    }

    /// Classify atoms for NHS exclusion field
    ///
    /// Returns atom type for each atom based on:
    /// - Element (C, N, O, S, H)
    /// - Residue type (aromatic: TRP, TYR, PHE)
    /// - Atom name (backbone vs sidechain)
    /// - Partial charge (charged residues)
    pub fn classify_atoms(&self) -> Vec<NhsAtomType> {
        let mut types = Vec::with_capacity(self.n_atoms);

        // Aromatic residues (including histidine protonation states)
        let aromatic_residues = ["TRP", "TYR", "PHE", "HIS", "HID", "HIE", "HIP"];

        // Backbone atom names
        let backbone_atoms = ["N", "CA", "C", "O", "OXT"];

        for i in 0..self.n_atoms {
            let element = &self.elements[i];
            let atom_name = &self.atom_names[i];
            let res_id = self.residue_ids[i];
            let res_name = &self.residue_names[i]; // atom-indexed, NOT res_id-indexed
            let charge = self.charges[i];

            let atom_type = if self.water_oxygens.contains(&i) {
                NhsAtomType::Water
            } else if backbone_atoms.contains(&atom_name.as_str()) {
                NhsAtomType::Backbone
            } else if aromatic_residues.contains(&res_name.as_str())
                && Self::is_ring_atom(atom_name, res_name)
            {
                NhsAtomType::Aromatic
            } else if charge > 0.3 {
                NhsAtomType::ChargedPositive
            } else if charge < -0.3 {
                NhsAtomType::ChargedNegative
            } else if Self::is_polar_atom(element, atom_name, res_name) {
                NhsAtomType::Polar
            } else {
                NhsAtomType::Hydrophobic
            };

            types.push(atom_type);
        }

        types
    }

    /// Check if atom is part of aromatic ring
    fn is_ring_atom(atom_name: &str, res_name: &str) -> bool {
        match res_name {
            "PHE" => matches!(atom_name, "CG" | "CD1" | "CD2" | "CE1" | "CE2" | "CZ"),
            "TYR" => matches!(
                atom_name,
                "CG" | "CD1" | "CD2" | "CE1" | "CE2" | "CZ" | "OH"
            ),
            "TRP" => matches!(
                atom_name,
                "CG" | "CD1" | "CD2" | "NE1" | "CE2" | "CE3" | "CZ2" | "CZ3" | "CH2"
            ),
            "HIS" => matches!(atom_name, "CG" | "ND1" | "CD2" | "CE1" | "NE2"),
            _ => false,
        }
    }

    /// Check if atom is polar
    fn is_polar_atom(element: &str, atom_name: &str, res_name: &str) -> bool {
        match element {
            "O" => true,
            "N" => {
                // Amide nitrogen is polar, but aromatic N in His handled separately
                !matches!(res_name, "PRO") || atom_name != "N"
            }
            "S" => {
                // Cys SG can be polar (H-bond acceptor)
                res_name == "CYS" && atom_name == "SG"
            }
            _ => false,
        }
    }

    /// Convert to NHS GPU format
    ///
    /// Returns (positions, types, charges, residue_ids) ready for NhsGpuEngine
    pub fn to_nhs_format(&self) -> (Vec<f32>, Vec<i32>, Vec<f32>, Vec<i32>) {
        let types = self.classify_atoms();
        let types_i32: Vec<i32> = types.iter().map(|t| t.as_i32()).collect();
        let residues_i32: Vec<i32> = self.residue_ids.iter().map(|&r| r as i32).collect();

        (
            self.positions.clone(),
            types_i32,
            self.charges.clone(),
            residues_i32,
        )
    }

    /// Get atom positions as array of [x,y,z] triples
    pub fn positions_as_xyz(&self) -> Vec<[f32; 3]> {
        self.positions
            .chunks(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect()
    }

    /// Compute bounding box for grid origin
    pub fn bounding_box(&self) -> ([f32; 3], [f32; 3]) {
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];

        for chunk in self.positions.chunks(3) {
            for i in 0..3 {
                min[i] = min[i].min(chunk[i]);
                max[i] = max[i].max(chunk[i]);
            }
        }

        (min, max)
    }

    /// Apply Hydrogen Mass Repartitioning (HMR).
    ///
    /// Redistributes mass from central atoms to bound hydrogens by a given factor.
    /// This allows using a larger timestep (4fs instead of 2fs) while maintaining
    /// SHAKE constraint accuracy, because the fastest motions (X-H stretches) are
    /// already constrained and heavier H atoms reduce remaining vibration frequencies.
    ///
    /// Standard HMR factor = 3.0: H mass goes from 1.008 to 3.024 amu.
    /// The stolen mass is subtracted from the central atom to conserve total mass.
    /// TRI-STATE HMR authority for the physical governor.
    ///
    /// This is the ONLY sanctioned mapping from the parsed topology into the
    /// governor's permission encoding. `None` (key absent) becomes
    /// [`crate::coupled_chronology::HmrVerification::Unverified`], which the
    /// governor treats exactly like a refusal for gear 2 — an unestablished
    /// permission is never a grant.
    ///
    /// See [`Self::hmr_applied`] for why the underlying field is an
    /// `Option<bool>` and not a `bool`.
    pub fn hmr_authority(&self) -> HmrVerification {
        match self.hmr_applied {
            Some(true) => HmrVerification::Applied,
            Some(false) => HmrVerification::NotApplied,
            None => HmrVerification::Unverified,
        }
    }

    /// The tri-state permission in the shape
    /// `gearbox::TopologyPermissions::hmr_applied` expects.
    ///
    /// Pass-through of [`Self::hmr_applied`]; declared as its own accessor so
    /// the bridge cannot accidentally read the raw field and re-collapse the
    /// third state on the way to `resolve_governor_policy`.
    pub fn hmr_topology_permission(&self) -> Option<bool> {
        self.hmr_applied
    }

    /// Whether HMR has been VERIFIED applied.
    ///
    /// `Unverified` answers `false` here — this predicate exists for the
    /// "must I still repartition?" question, not for permission decisions.
    /// Permission decisions go through [`Self::hmr_authority`].
    pub fn hmr_verified_applied(&self) -> bool {
        self.hmr_applied == Some(true)
    }

    pub fn apply_hmr(&mut self, factor: f64) {
        self.apply_hmr_with_historical_repeat(factor,false);
    }

    /// Explicit old execution repeats HMR on the actual serialized mass model.
    /// The original hmr_applied flag is never cleared or rewritten to fake a
    /// clean input. Normal apply_hmr remains idempotent.
    pub fn apply_hmr_historical_may_execution(&mut self) -> Result<(),String> {
        if self.historical_may_admission()?.is_none(){return Err("historical HMR lacks explicit immutable admission".into());}
        self.apply_hmr_with_historical_repeat(3.0,true);Ok(())
    }

    fn apply_hmr_with_historical_repeat(&mut self, factor:f64, repeat_verified:bool) {
        // Only a VERIFIED prior application suppresses repartitioning.
        // `Unverified` must not suppress it: skipping the repartition on an
        // unestablished claim would leave the mass model unrepartitioned while
        // the run believed otherwise. The conservative action under an
        // unestablished claim is to perform the documented repartition, which
        // then establishes the fact.
        if self.hmr_verified_applied() && !repeat_verified {
            log::info!(
                "HMR: topology already records repartitioned masses; preserving serialized mass model"
            );
            return;
        }
        if self.hmr_applied.is_none() {
            log::info!(
                "HMR: topology carried NO hmr_applied key (authority=Unverified); applying the \
                 documented repartition and recording the result as verified-applied"
            );
        }
        let mut n_repartitioned = 0usize;
        let mut total_mass_before = 0.0f64;
        let mut total_mass_after = 0.0f64;

        for cluster in &mut self.h_clusters {
            // Original masses from inverse masses
            let m_h_orig = 1.0 / cluster.inv_mass_h;
            let m_central_orig = 1.0 / cluster.inv_mass_central;

            total_mass_before += m_central_orig + cluster.n_hydrogens as f64 * m_h_orig;

            // New hydrogen mass = original * factor
            let m_h_new = m_h_orig * factor;
            let delta_m = m_h_new - m_h_orig;

            // Steal mass from central atom (conserve total)
            let m_central_new = m_central_orig - cluster.n_hydrogens as f64 * delta_m;

            // Safety: central atom must retain at least 1.0 amu
            if m_central_new < 1.0 {
                log::warn!(
                    "HMR: central atom {} would drop to {:.2} amu, skipping cluster",
                    cluster.central_atom,
                    m_central_new
                );
                total_mass_after += m_central_orig + cluster.n_hydrogens as f64 * m_h_orig;
                continue;
            }

            cluster.inv_mass_h = 1.0 / m_h_new;
            cluster.inv_mass_central = 1.0 / m_central_new;
            n_repartitioned += 1;

            total_mass_after += m_central_new + cluster.n_hydrogens as f64 * m_h_new;

            // Also update the masses array for the atoms
            if cluster.central_atom < self.masses.len() {
                self.masses[cluster.central_atom] = m_central_new as f32;
            }
            for &h_idx in &cluster.hydrogen_atoms {
                if h_idx >= 0 && (h_idx as usize) < self.masses.len() {
                    self.masses[h_idx as usize] = m_h_new as f32;
                }
            }
        }

        log::info!("HMR: repartitioned {}/{} clusters (factor {:.1}x), total mass {:.2} → {:.2} amu (delta={:.4})",
            n_repartitioned, self.h_clusters.len(), factor,
            total_mass_before, total_mass_after,
            (total_mass_after - total_mass_before).abs());
        self.hmr_applied = Some(true);
        self.recommended_timestep_fs = Some(4.0);
    }

    /// Existing native multistream coverage policy, shared with saved replay.
    /// Preserves the engine's f32 arithmetic,5A padding and1.02 coverage margin.
    pub fn grid_spacing_for_coverage(&self, grid_dim:usize, floor_spacing:f32)->(f32,f32) {
        let (min_pos,max_pos)=self.bounding_box();let mut extent=0.0f32;
        for axis in 0..3 {
            let axis_extent=max_pos[axis]-min_pos[axis]+2.0*5.0;
            if axis_extent.is_finite(){extent=extent.max(axis_extent);}
        }
        let padded_extent=extent.max(0.0);
        if grid_dim==0 || padded_extent<=0.0{return (floor_spacing,padded_extent);}
        let required_spacing=(padded_extent/grid_dim as f32)*1.02;
        (floor_spacing.max(required_spacing),padded_extent)
    }

    /// Get grid origin with padding
    pub fn grid_origin(&self, padding: f32) -> [f32; 3] {
        let (min, _) = self.bounding_box();
        [min[0] - padding, min[1] - padding, min[2] - padding]
    }

    /// Get required grid dimension for given spacing
    pub fn required_grid_dim(&self, spacing: f32, padding: f32) -> usize {
        let (min, max) = self.bounding_box();
        let size = [
            max[0] - min[0] + 2.0 * padding,
            max[1] - min[1] + 2.0 * padding,
            max[2] - min[2] + 2.0 * padding,
        ];
        let max_size = size[0].max(size[1]).max(size[2]);
        ((max_size / spacing).ceil() as usize).max(32)
    }

    /// Get aromatic residue indices for UV bias targeting
    pub fn aromatic_residues(&self) -> Vec<usize> {
        let aromatic = ["TRP", "TYR", "PHE"];
        let mut residues = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for (i, res_name) in self.residue_names.iter().enumerate() {
            let res_id = self.residue_ids[i];
            if aromatic.contains(&res_name.as_str()) && !seen.contains(&res_id) {
                residues.push(res_id);
                seen.insert(res_id);
            }
        }

        residues
    }

    /// Get atom type statistics
    pub fn atom_type_stats(&self) -> HashMap<NhsAtomType, usize> {
        let types = self.classify_atoms();
        let mut stats = HashMap::new();

        for t in types {
            *stats.entry(t).or_insert(0) += 1;
        }

        stats
    }
}

// ============================================================================
// NHS INPUT PREPARED FOR GPU
// ============================================================================

/// Prepared input data for NHS GPU processing
#[derive(Debug, Clone)]
pub struct NhsPreparedInput {
    /// Source topology
    pub topology: PrismPrepTopology,
    /// Flat positions array
    pub positions: Vec<f32>,
    /// Atom types as i32
    pub types: Vec<i32>,
    /// Partial charges
    pub charges: Vec<f32>,
    /// Residue IDs as i32
    pub residues: Vec<i32>,
    /// Grid origin
    pub grid_origin: [f32; 3],
    /// Required grid dimension
    pub grid_dim: usize,

    // === RT Integration Fields [STAGE-1-PREP] ===
    /// Solvent mode (implicit/explicit/hybrid)
    pub solvent_mode: SolventMode,
    /// Water atom indices (if explicit/hybrid)
    pub water_atoms: Option<Vec<usize>>,
    /// RT probe targets for spatial sensing
    pub rt_targets: RtTargets,
    /// Total atom count (protein + waters)
    pub total_atoms: usize,
}

impl NhsPreparedInput {
    /// Create prepared input from topology with RT support
    ///
    /// # Arguments
    /// * `topology` - PRISM-PREP topology
    /// * `grid_spacing` - Grid spacing for detection
    /// * `padding` - Padding around protein
    /// * `solvent_mode` - Solvent mode for RT integration
    ///
    /// # RT Integration [STAGE-1-PREP]
    ///
    /// This method now supports explicit solvation and RT target identification:
    /// 1. If Explicit/Hybrid: Generate water box around protein
    /// 2. Identify RT targets (protein atoms, water O, aromatic centers)
    /// 3. Compute total atom count (protein + waters)
    ///
    /// # Returns
    /// * `Result<Self>` - Prepared input with RT targets
    pub fn from_topology(
        mut topology: PrismPrepTopology,
        grid_spacing: f32,
        padding: f32,
        solvent_mode: &SolventMode,
    ) -> Result<Self> {
        log::info!("Preparing system with {:?} solvent mode", solvent_mode);

        // Step 1: Validate solvent mode
        solvent_mode
            .validate()
            .context("Invalid solvent mode configuration")?;

        // Step 2: Solvate if explicit mode (hybrid starts implicit, adds waters later in Stage 2b)
        let water_atoms = if solvent_mode.starts_explicit() {
            let padding_angstroms = match solvent_mode {
                SolventMode::Explicit { padding_angstroms } => *padding_angstroms,
                SolventMode::Hybrid { .. } => padding, // Use grid padding
                SolventMode::Implicit => 0.0, // Won't reach here due to requires_water check
            };

            let protein_coords = topology.positions.clone();
            let (water_coords, water_indices) =
                solvate_protein(&topology, &protein_coords, padding_angstroms)
                    .context("Failed to solvate protein")?;

            let n_waters = water_indices.len();
            log::info!(
                "Added {} water molecules ({} atoms)",
                n_waters,
                water_coords.len() / 3
            );

            // Append water positions to topology
            topology.positions.extend_from_slice(&water_coords);

            // Append water metadata to all arrays
            for _ in 0..n_waters {
                topology.elements.push("O".to_string()); // Water oxygen
                topology.atom_names.push("O".to_string());
                topology.residue_ids.push(topology.n_residues); // All waters in same "residue"
                topology.chain_ids.push("W".to_string()); // Water chain
                topology.charges.push(-0.834); // TIP3P oxygen charge
                topology.masses.push(15.9994); // Oxygen mass
            }

            topology.n_atoms += n_waters;
            // residue_names is atom-indexed: one entry per atom
            for _ in 0..n_waters {
                topology.residue_names.push("HOH".to_string());
            }
            topology.n_residues += 1; // All waters count as one "residue" for simplicity
            topology.water_oxygens = water_indices.clone();

            Some(water_indices)
        } else {
            log::info!("Implicit mode: no explicit waters added");
            None
        };

        // Step 3: Identify RT targets
        let rt_targets = identify_rt_targets(&topology, solvent_mode)
            .context("Failed to identify RT targets")?;

        log::info!("{}", rt_targets.summary());

        // Step 4: Compute total atom count
        let total_atoms = topology.n_atoms;

        // Step 5: Convert to NHS format
        let (positions, types, charges, residues) = topology.to_nhs_format();
        let grid_origin = topology.grid_origin(padding);
        let grid_dim = topology.required_grid_dim(grid_spacing, padding);

        log::info!(
            "Prepared NHS input: {} atoms ({} protein{}), grid {}³ at origin {:?}",
            total_atoms,
            total_atoms - water_atoms.as_ref().map_or(0, |w| w.len()),
            if water_atoms.is_some() {
                format!(" + {} waters", water_atoms.as_ref().unwrap().len())
            } else {
                String::new()
            },
            grid_dim,
            grid_origin
        );

        Ok(Self {
            topology,
            positions,
            types,
            charges,
            residues,
            grid_origin,
            grid_dim,
            solvent_mode: solvent_mode.clone(),
            water_atoms,
            rt_targets,
            total_atoms,
        })
    }

    /// Load and prepare from file with RT support
    ///
    /// # Arguments
    /// * `path` - Path to PRISM-PREP topology JSON
    /// * `grid_spacing` - Grid spacing for detection
    /// * `padding` - Padding around protein
    /// * `solvent_mode` - Solvent mode for RT integration
    ///
    /// # Returns
    /// * `Result<Self>` - Prepared input with RT targets
    pub fn load(
        path: impl AsRef<Path>,
        grid_spacing: f32,
        padding: f32,
        solvent_mode: &SolventMode,
    ) -> Result<Self> {
        let topology = PrismPrepTopology::load(path)?;
        Self::from_topology(topology, grid_spacing, padding, solvent_mode)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atom_type_values() {
        assert_eq!(NhsAtomType::Hydrophobic.as_i32(), 0);
        assert_eq!(NhsAtomType::Aromatic.as_i32(), 4);
    }

    #[test]
    fn test_is_ring_atom() {
        assert!(PrismPrepTopology::is_ring_atom("CG", "PHE"));
        assert!(PrismPrepTopology::is_ring_atom("NE1", "TRP"));
        assert!(!PrismPrepTopology::is_ring_atom("CB", "PHE"));
    }

    #[test]
    fn test_topology_load() {
        // This test requires an actual file
        let path = "../../data/curated_14/topologies/1L2Y_topology.json";
        if std::path::Path::new(path).exists() {
            let topology = PrismPrepTopology::load(path).expect("Failed to load topology");
            assert!(topology.n_atoms > 0);
            assert!(topology.n_residues > 0);

            let stats = topology.atom_type_stats();
            println!("Atom type stats: {:?}", stats);
        }
    }

    // === RT Integration Tests [STAGE-1-PREP] ===

    fn create_minimal_topology() -> PrismPrepTopology {
        PrismPrepTopology {
            aromatic_targets: Vec::new(),
            source_pdb: "test.pdb".into(),
            n_atoms: 5,
            n_residues: 2,
            total_physical_residue_count: None,
            n_analysis_residues: None,
            n_chains: 1,
            positions: vec![
                0.0, 0.0, 0.0, // Atom 0
                1.0, 0.0, 0.0, // Atom 1
                2.0, 0.0, 0.0, // Atom 2
                3.0, 0.0, 0.0, // Atom 3
                4.0, 0.0, 0.0, // Atom 4
            ],
            elements: vec!["C".into(), "N".into(), "O".into(), "C".into(), "C".into()],
            atom_names: vec![
                "CA".into(),
                "N".into(),
                "O".into(),
                "CB".into(),
                "CG".into(),
            ],
            residue_names: vec![
                "ALA".into(),
                "ALA".into(),
                "ALA".into(),
                "PHE".into(),
                "PHE".into(),
            ],
            residue_ids: vec![0, 0, 0, 1, 1],
            atom_physical_residue_ids: None,
            atom_analysis_residue_ids: None,
            analysis_residue_contract: None,
            chain_ids: vec!["A".into(); 5],
            charges: vec![0.0; 5],
            masses: vec![12.0, 14.0, 16.0, 12.0, 12.0],
            ca_indices: vec![0],
            bonds: Vec::new(),
            angles: Vec::new(),
            dihedrals: Vec::new(),
            lj_params: vec![
                LjParam {
                    sigma: 1.5,
                    epsilon: 0.1,
                };
                5
            ],
            exclusions: vec![Vec::new(); 5],
            nonbonded_exceptions: Vec::new(),
            nonbonded_exception_contract: None,
            generated_prmtop_sha256: None,
            h_clusters: Vec::new(),
            hmr_applied: Some(false),
            recommended_timestep_fs: None,
            water_oxygens: Vec::new(),
            residues: Vec::new(),
            dimer_dyad: None,
            lambda_fragment_system: None,
            harmonic_parameter_contract: None,
            schema_kind: None,
            schema_version: None,
            periodic_cell: None,
            water_model_contract: None,
            molecule_identity_table: None,
            topology_provenance: None,
            npt_frozen_cell_contract: None,
        }
    }

    fn create_exact_topology() -> PrismPrepTopology {
        let mut topology = create_minimal_topology();
        topology.exclusions[0].push(1);
        topology.exclusions[1].push(0);
        topology.nonbonded_exceptions = vec![NonbondedExceptionParam {
            i: 0,
            j: 3,
            charge_product_e2: -0.125,
            sigma_a: 3.5,
            epsilon_kcal_mol: 0.025,
        }];
        topology.generated_prmtop_sha256 = Some("a".repeat(64));
        topology.nonbonded_exception_contract = Some(NonbondedExceptionContract {
            schema_kind: "prismos_exact_nonbonded_exceptions_v2".to_string(),
            schema_version: 2,
            semantics: "OPENMM_NONBONDEDFORCE_ABSOLUTE_EXCEPTION_PARAMETERS_V1".to_string(),
            dynamic_charge_scaling_semantics:
                "EXCITED_STATE_MULTIPLIES_GROUND_STATE_CHARGE_PRODUCT_BY_PER_ATOM_SCALE_PRODUCT_V1"
                    .to_string(),
            exact_scaled_pair_count: 1,
            strict_exclusion_pair_count: 1,
            total_openmm_exception_count: 2,
            pair_digest_algorithm: "SHA256_PRISMOS_NBEX_ACTIVE_V1_LE_U64_U64_F64_F64_F64"
                .to_string(),
            pair_digest_sha256: topology.nonbonded_exception_digest_sha256(),
            strict_exclusion_digest_algorithm: "SHA256_PRISMOS_NBEX_STRICT_V1_LE_U64_U64"
                .to_string(),
            strict_exclusion_digest_sha256: topology.strict_exclusion_digest_sha256(),
            source_prmtop_sha256: "a".repeat(64),
            hardcoded_uniform_scaling_allowed: false,
            complete: true,
            fallback_used: false,
        });
        topology
    }

    fn seal_contract<T: Serialize>(value: &T, digest_field: &str) -> String {
        let mut value = serde_json::to_value(value).unwrap();
        value.as_object_mut().unwrap().remove(digest_field);
        value
            .as_object_mut()
            .unwrap()
            .retain(|_, value| !value.is_null());
        python_canonical_sha256(&value).unwrap()
    }

    fn create_v3_digest_fixture() -> PrismPrepTopology {
        let mut topology = create_minimal_topology();
        topology.schema_kind = Some("prismos_openmm_topology".into());
        topology.schema_version = Some(serde_json::Value::String("3.0.0".into()));
        topology.n_atoms = 3;
        topology.n_residues = 1;
        topology.positions = vec![0.0, 0.0, 0.0, 1.0118108, 0.0, 0.0, -0.316, 0.961, 0.0];
        topology.elements = vec!["O".into(), "H".into(), "H".into()];
        topology.atom_names = vec!["O".into(), "H1".into(), "H2".into()];
        topology.residue_names = vec!["HOH".into(); 3];
        topology.residue_ids = vec![0; 3];
        topology.chain_ids = vec!["W".into(); 3];
        topology.charges = vec![-0.8484487, 0.42422435, 0.42422435];
        topology.masses = vec![15.99943, 1.007947, 1.007947];
        topology.ca_indices.clear();
        topology.bonds.clear();
        topology.angles.clear();
        topology.dihedrals.clear();
        topology.lj_params = vec![
            LjParam {
                sigma: 3.178305966129,
                epsilon: 0.17809824,
            },
            LjParam {
                sigma: 1.0,
                epsilon: 0.0,
            },
            LjParam {
                sigma: 1.0,
                epsilon: 0.0,
            },
        ];
        topology.exclusions = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        topology.water_oxygens = vec![0];

        let mut periodic = PeriodicCellContract {
            periodic: true,
            reason: None,
            vectors_angstrom: Some([[30.0, 0.0, 0.0], [0.0, 31.0, 0.0], [0.0, 0.0, 32.0]]),
            volume_angstrom3: Some(29_760.0),
            orthorhombic: Some(true),
            minimum_image_required: Some(true),
            coordinate_wrapping_required: Some(true),
            reciprocal_method: Some("PME".into()),
            real_space_cutoff_angstrom: Some(10.0),
            contract_sha256: Some(String::new()),
            ..Default::default()
        };
        periodic.contract_sha256 = Some(seal_contract(&periodic, "contract_sha256"));
        topology.periodic_cell = Some(periodic);

        let reference = WaterReferenceParameters {
            oh_distances_angstrom: [1.0118108, 1.0118108],
            hh_distance_angstrom: 1.621,
            hoh_angle_radian: 1.8875464,
            masses_dalton: Some([15.99943, 1.007947, 1.007947]),
            model_masses_dalton: None,
            runtime_masses_dalton: None,
            charges_e: [-0.8484487, 0.42422435, 0.42422435],
            oxygen_lj: topology.lj_params[0].clone(),
            hydrogen_lj: [topology.lj_params[1].clone(), topology.lj_params[2].clone()],
        };
        let instances = vec![WaterInstance {
            residue_index: 0,
            oxygen_atom_index: 0,
            hydrogen_atom_indices: [1, 2],
        }];
        let water_rows = serde_json::json!([{
            "residue_index": 0,
            "oxygen_atom_index": 0,
            "hydrogen_atom_indices": [1, 2],
            "oh_distances_angstrom": reference.oh_distances_angstrom,
            "hh_distance_angstrom": reference.hh_distance_angstrom,
            "hoh_angle_radian": reference.hoh_angle_radian,
            "masses_dalton": reference.masses_dalton,
            "charges_e": reference.charges_e,
            "oxygen_lj": reference.oxygen_lj,
            "hydrogen_lj": reference.hydrogen_lj,
        }]);
        let mut water = WaterModelContract {
            schema_kind: "prismos_water_model_contract".into(),
            schema_version: 1,
            model_id: "TIP3P-FB".into(),
            forcefield_resource: "amber14/tip3pfb.xml".into(),
            n_waters: 1,
            atom_order: "OHH".into(),
            rigid_water_in_openmm_system: false,
            runtime_settle_rattle_required: true,
            reference_parameters_sha256: python_canonical_sha256(
                &serde_json::to_value(&reference).unwrap(),
            )
            .unwrap(),
            reference_parameters: reference,
            water_instance_map_sha256: python_canonical_sha256(
                &serde_json::to_value(&instances).unwrap(),
            )
            .unwrap(),
            water_instances: instances,
            water_instance_parameters_sha256: python_canonical_sha256(&water_rows).unwrap(),
            contract_sha256: String::new(),
            mass_repartitioning_applied: None,
            model_mass_semantics: None,
            runtime_mass_semantics: None,
            per_water_mass_conservation_tolerance_dalton: None,
        };
        water.contract_sha256 = seal_contract(&water, "contract_sha256");
        topology.water_model_contract = Some(water);

        let lineage = serde_json::json!({
            "atom_indices": [0, 1, 2],
            "residue_indices": [0],
            "residue_names": ["HOH"],
            "chain_ids": ["W"],
            "kind": "WATER",
            "identity": "TIP3P-FB",
        });
        let molecule = MoleculeIdentityRow {
            molecule_id: 0,
            atom_indices: vec![0, 1, 2],
            residue_indices: vec![0],
            residue_names: vec!["HOH".into()],
            chain_ids: vec!["W".into()],
            kind: "WATER".into(),
            identity: "TIP3P-FB".into(),
            component_sha256: python_canonical_sha256(&lineage).unwrap(),
        };
        let mut table = MoleculeIdentityTable {
            schema_kind: "prismos_molecule_identity_table".into(),
            schema_version: 1,
            n_molecules: 1,
            n_atoms: 3,
            molecules: vec![molecule],
            table_sha256: String::new(),
        };
        table.table_sha256 = seal_contract(&table, "table_sha256");
        topology.molecule_identity_table = Some(table);
        topology.total_physical_residue_count = Some(1);
        topology.n_analysis_residues = Some(0);
        topology.atom_physical_residue_ids = Some(vec![0, 0, 0]);
        topology.atom_analysis_residue_ids = Some(vec![-1, -1, -1]);
        let mut role_counts = BTreeMap::new();
        role_counts.insert("WATER".to_string(), 1);
        let mut analysis_contract = AnalysisResidueContract {
            schema_kind: "prismos_analysis_residue_mapping".into(),
            schema_version: 1,
            analysis_authority: "KCC_GHOST_PROTEIN_RESIDUE_AXIS".into(),
            excluded_atom_sentinel: -1,
            n_atoms: 3,
            total_physical_residue_count: 1,
            n_analysis_residues: 0,
            physical_residue_roles: vec!["WATER".into()],
            physical_to_analysis_residue_ids: vec![-1],
            analysis_residues: Vec::new(),
            atom_analysis_residue_ids_sha256: python_canonical_sha256(&serde_json::json!([
                -1, -1, -1
            ]))
            .unwrap(),
            physical_molecule_table_sha256: topology
                .molecule_identity_table
                .as_ref()
                .unwrap()
                .table_sha256
                .clone(),
            lambda_fragment_system_sha256: None,
            role_counts,
            excluded_roles: vec![
                "WATER".into(),
                "ION".into(),
                "LAMBDA_FRAGMENT".into(),
                "OTHER".into(),
            ],
            must_recompute_after_fragment_materialization: true,
            contract_sha256: String::new(),
        };
        analysis_contract.contract_sha256 = seal_contract(&analysis_contract, "contract_sha256");
        topology.analysis_residue_contract = Some(analysis_contract);

        let resources = serde_json::json!([{
            "resource": "amber14-all.xml",
            "sha256": "a".repeat(64),
            "size_bytes": 1234,
        }, {
            "resource": "amber14/tip3pfb.xml",
            "sha256": "b".repeat(64),
            "size_bytes": 5678,
        }]);
        let forcefield_resources = serde_json::json!({
            "openmm_data_root": "/opt/openmm/data",
            "resources": resources,
            "resource_closure_sha256": python_canonical_sha256(&resources).unwrap(),
        });
        let mut provenance = TopologyProvenanceContract {
            schema_kind: "prismos_topology_provenance".into(),
            schema_version: 1,
            producer: "/opt/prismos/stage2_topology.py".into(),
            producer_sha256: "c".repeat(64),
            source_pdb: "/data/input.pdb".into(),
            source_pdb_sha256: "d".repeat(64),
            openmm_version: "8.2.0".into(),
            forcefield_resources,
            protein_forcefield: "AMBER ff14SB".into(),
            water_model: Some("TIP3P-FB".into()),
            nonbonded_method: "PME".into(),
            nonbonded_cutoff_angstrom: Some(10.0),
            openmm_constraints: "None".into(),
            openmm_rigid_water: false,
            runtime_constraint_authority_required: true,
            preparation_seed: 20_260_723,
            random_generator: "python_stdlib_random_MersenneTwister".into(),
            ph: Some(7.0),
            explicit_solvent_requested: Some(true),
            terminal_caps_requested: Some(false),
            hydrogen_mass_repartitioning_applied: Some(true),
            minimization_contract_sha256: None,
            minimization_platform_name: None,
            minimization_platform_plugin_sha256: None,
            npt_equilibrated: None,
            npt_population_authority_sha256: None,
            npt_seed_authority_sha256: None,
            npt_frozen_cell_authority_sha256: None,
            base_minimization_contract_sha256: None,
            energy_minimization_applied: None,
            contract_sha256: String::new(),
            parent_topology_scientific_sha256: None,
            base_environment_minimization_contract_sha256: None,
            base_environment_minimization_platform_plugin_sha256: None,
            fragment_slots_start_at_lambda_zero: None,
        };
        provenance.contract_sha256 = seal_contract(&provenance, "contract_sha256");
        topology.topology_provenance = Some(provenance);
        topology
    }

    #[test]
    fn v3_lambda_preparation_digests_recompute_and_reject_tampering() {
        let fixture = create_v3_digest_fixture();
        fixture.validate_lambda_preparation_digests().unwrap();

        let mut periodic_tamper = fixture.clone();
        periodic_tamper
            .periodic_cell
            .as_mut()
            .unwrap()
            .vectors_angstrom
            .as_mut()
            .unwrap()[0][0] += 0.25;
        assert!(periodic_tamper
            .validate_lambda_preparation_digests()
            .unwrap_err()
            .contains("periodic_cell canonical SHA256 mismatch"));

        let mut water_tamper = fixture.clone();
        water_tamper
            .water_model_contract
            .as_mut()
            .unwrap()
            .reference_parameters
            .charges_e[0] += 0.01;
        assert!(water_tamper
            .validate_lambda_preparation_digests()
            .unwrap_err()
            .contains("water reference parameters canonical SHA256 mismatch"));

        let mut molecule_tamper = fixture.clone();
        molecule_tamper
            .molecule_identity_table
            .as_mut()
            .unwrap()
            .molecules[0]
            .identity = "TIP3P".into();
        assert!(molecule_tamper
            .validate_lambda_preparation_digests()
            .unwrap_err()
            .contains("molecule 0 component canonical SHA256 mismatch"));

        let mut provenance_tamper = fixture;
        provenance_tamper
            .topology_provenance
            .as_mut()
            .unwrap()
            .random_generator = "unbound_rng".into();
        assert!(provenance_tamper
            .validate_lambda_preparation_digests()
            .unwrap_err()
            .contains("topology provenance canonical SHA256 mismatch"));
    }

    #[test]
    fn hmr_respects_serialized_preapplied_authority() {
        let mut topology = create_minimal_topology();
        topology.masses[0] = 9.0;
        topology.masses[1] = 3.0;
        topology.h_clusters = vec![HCluster {
            cluster_type: 1,
            central_atom: 0,
            hydrogen_atoms: vec![1],
            bond_lengths: vec![1.0],
            n_hydrogens: 1,
            inv_mass_central: 1.0 / 9.0,
            inv_mass_h: 1.0 / 3.0,
        }];
        topology.hmr_applied = Some(true);
        topology.recommended_timestep_fs = Some(4.0);

        let masses_before = topology.masses.clone();
        let inverse_before = (
            topology.h_clusters[0].inv_mass_central,
            topology.h_clusters[0].inv_mass_h,
        );
        topology.apply_hmr(3.0);

        assert_eq!(topology.masses, masses_before);
        assert_eq!(
            (
                topology.h_clusters[0].inv_mass_central,
                topology.h_clusters[0].inv_mass_h,
            ),
            inverse_before
        );
    }

    #[test]
    fn exact_nonbonded_v2_contract_accepts_complete_topology() {
        create_exact_topology().validate().unwrap();
    }

    #[test]
    fn exact_nonbonded_json_preserves_force_parameter_bits() {
        let row: NonbondedExceptionParam = serde_json::from_str(
            r#"{"i":1,"j":14,"charge_product_e2":0.09932701675511141,"sigma_A":2.234373985145219,"epsilon_kcal_mol":0.018372533883399215}"#,
        )
        .unwrap();

        assert_eq!(row.charge_product_e2.to_bits(), 0x3fb9_6d7e_d092_8bec);
        assert_eq!(row.sigma_a.to_bits(), 0x4001_dfff_77c9_daf1);
        assert_eq!(row.epsilon_kcal_mol.to_bits(), 0x3f92_d03f_e0b2_0621);
    }

    #[test]
    fn production_nonbonded_policy_rejects_exception_free_topology() {
        let error = create_minimal_topology()
            .validate_for_nonbonded_execution(NonbondedExecutionPolicy::RequireExactV2)
            .unwrap_err();
        assert!(error.contains("production nonbonded execution requires exact"));
    }

    #[test]
    fn explicit_nonproduction_policy_admits_only_exception_free_topology() {
        create_minimal_topology()
            .validate_for_nonbonded_execution(
                NonbondedExecutionPolicy::AllowNonProductionLegacyUniformFudge,
            )
            .unwrap();
        let error = create_exact_topology()
            .validate_for_nonbonded_execution(
                NonbondedExecutionPolicy::AllowNonProductionLegacyUniformFudge,
            )
            .unwrap_err();
        assert!(error.contains("valid only for an exception-free legacy topology"));
    }

    #[test]
    fn exact_nonbonded_v2_rejects_active_pair_mutation() {
        let mut topology = create_exact_topology();
        topology.nonbonded_exceptions[0].epsilon_kcal_mol += 0.001;
        assert!(topology.validate().unwrap_err().contains("digest mismatch"));
    }

    #[test]
    fn exact_nonbonded_v2_rejects_asymmetric_exclusion() {
        let mut topology = create_exact_topology();
        topology.exclusions[1].clear();
        assert!(topology
            .validate()
            .unwrap_err()
            .contains("asymmetric strict exclusion"));
    }

    #[test]
    fn exact_nonbonded_v2_rejects_invalid_exclusion_partners() {
        for invalid_partner in [0, 5] {
            let mut topology = create_exact_topology();
            topology.exclusions[0].push(invalid_partner);
            assert!(topology.validate().is_err());
        }
        let mut duplicate = create_exact_topology();
        duplicate.exclusions[0].push(1);
        assert!(duplicate.validate().is_err());
    }

    #[test]
    fn exact_nonbonded_v2_rejects_strict_set_mutation() {
        let mut topology = create_exact_topology();
        topology.exclusions[2].push(4);
        topology.exclusions[4].push(2);
        assert!(topology
            .validate()
            .unwrap_err()
            .contains("strict exclusion count mismatch"));
    }

    #[test]
    fn exact_nonbonded_v2_rejects_prmtop_identity_mismatch() {
        let mut topology = create_exact_topology();
        topology.generated_prmtop_sha256 = Some("b".repeat(64));
        assert!(topology
            .validate()
            .unwrap_err()
            .contains("source prmtop hash"));
    }

    #[test]
    fn test_prepared_input_implicit_mode() {
        let topology = create_minimal_topology();
        let solvent_mode = SolventMode::Implicit;

        let prepared = NhsPreparedInput::from_topology(
            topology,
            0.5,  // grid_spacing
            10.0, // padding
            &solvent_mode,
        )
        .expect("Failed to prepare input");

        // Check basic fields
        assert_eq!(prepared.total_atoms, 5, "Should have 5 protein atoms");
        assert!(
            prepared.water_atoms.is_none(),
            "Implicit mode should have no waters"
        );

        // Check RT targets
        assert_eq!(
            prepared.rt_targets.protein_atoms.len(),
            5,
            "Should have 5 protein heavy atoms"
        );
        assert!(
            prepared.rt_targets.water_atoms.is_none(),
            "Should have no water targets"
        );
        // Minimal topology lacks real aromatic ring geometry (CD1/CD2/CE1/CE2/CZ),
        // so aromatic detection may find 0 centers despite PHE residue name.
        // This is acceptable — aromatic detection requires ring atom positions.
        let n_aro = prepared.rt_targets.aromatic_centers.len();
        assert!(
            n_aro <= 1,
            "Should have 0 or 1 aromatic center(s), got {}",
            n_aro
        );

        // Total targets: 5 protein + n_aro aromatics
        assert_eq!(prepared.rt_targets.total_targets, 5 + n_aro);

        println!("Implicit mode: {}", prepared.rt_targets.summary());
    }

    #[test]
    fn test_prepared_input_explicit_mode() {
        let topology = create_minimal_topology();
        let solvent_mode = SolventMode::Explicit {
            padding_angstroms: 5.0,
        };

        let prepared = NhsPreparedInput::from_topology(
            topology,
            0.5,  // grid_spacing
            10.0, // padding
            &solvent_mode,
        )
        .expect("Failed to prepare input");

        // Check that waters were added
        assert!(
            prepared.water_atoms.is_some(),
            "Explicit mode should have waters"
        );
        let n_waters = prepared.water_atoms.as_ref().unwrap().len();
        assert!(n_waters > 0, "Should have added some waters");

        // Total atoms = protein + waters
        assert_eq!(prepared.total_atoms, 5 + n_waters);

        // Check RT targets include waters
        assert!(
            prepared.rt_targets.water_atoms.is_some(),
            "Should have water targets"
        );
        assert_eq!(
            prepared.rt_targets.water_atoms.as_ref().unwrap().len(),
            n_waters
        );

        println!("Explicit mode: {}", prepared.rt_targets.summary());
        println!("Added {} waters", n_waters);
    }

    #[test]
    fn test_prepared_input_fields_populated() {
        let topology = create_minimal_topology();
        let solvent_mode = SolventMode::Implicit;

        let prepared = NhsPreparedInput::from_topology(topology.clone(), 0.5, 10.0, &solvent_mode)
            .expect("Failed to prepare input");

        // Check all standard fields
        assert_eq!(
            prepared.positions.len(),
            15,
            "Should have 5 atoms * 3 coords"
        );
        assert_eq!(prepared.types.len(), 5);
        assert_eq!(prepared.charges.len(), 5);
        assert_eq!(prepared.residues.len(), 5);
        assert_eq!(prepared.grid_dim, topology.required_grid_dim(0.5, 10.0));

        // Check RT fields
        assert!(matches!(prepared.solvent_mode, SolventMode::Implicit));
        assert_eq!(prepared.total_atoms, 5);
    }
}

#[cfg(test)]
mod recorded_periodic_authority_tests {
    use super::*;
    fn actual_fixtures() -> [&'static str; 2] {
        ["mycmax_no_probe_2fs.topology.json", "mycmax_lambda_species_001_precal.topology.json"]
    }
    fn actual_raw(index: usize) -> Result<serde_json::Value> {
        let root=std::env::var_os("PRISM_CONTROL_ROOT").map(std::path::PathBuf::from).ok_or_else(|| anyhow::anyhow!("PRISM_CONTROL_ROOT unset: set it to the root that holds MYC_MAX_ADAPTIVE_PROBE_GHOST_FIBER_20260723"))?.join("MYC_MAX_ADAPTIVE_PROBE_GHOST_FIBER_20260723/LAMBDA_TOPOLOGIES_20260723_R8_PRODUCTION_DERIVED");
        let bytes=std::fs::read(root.join(actual_fixtures()[index]))?;
        let expected=["7c2e87ef3a2964662f55d9c5e163625132c54c9f5228c001e832a03ba11f7ca3", "9392e3ba8accc6a36144595dce3d29e4e6741cd5135c1ce5fbecc493e99bcc35"];
        anyhow::ensure!(format!("{:x}",Sha256::digest(&bytes))==expected[index],"original periodic fixture changed");
        Ok(serde_json::from_slice(&bytes)?)
    }
    #[test]
    fn python_float_notation_thresholds_keep_exact_values_and_integer_kind() -> Result<()> {
        for (value, expected) in [(1e-4,"0.0001"),(-1e-4,"-0.0001"),(1e-5,"1e-05"),(-1e-5,"-1e-05"),(1e15,"1000000000000000.0"),(-1e15,"-1000000000000000.0"),(1e16,"1e+16"),(-1e16,"-1e+16"),(0.0,"0.0"),(-0.0,"-0.0"),(1.0,"1.0"),(1.2345e-5,"1.2345e-05"),(f64::MIN_POSITIVE,"2.2250738585072014e-308")] {
            let v=serde_json::Value::from(value);
            let encoded=python_canonical_json(&v).map_err(anyhow::Error::msg)?;
            assert_eq!(String::from_utf8(encoded.clone())?,expected);
            let decoded:serde_json::Value=serde_json::from_slice(&encoded)?;
            assert_eq!(decoded.as_f64().unwrap().to_bits(),value.to_bits());
        }
        assert_eq!(python_canonical_json(&serde_json::json!(10000000000000000u64)).map_err(anyhow::Error::msg)?,b"10000000000000000");
        Ok(())
    }
    #[test]
    fn nonperiodic_serialization_has_no_new_fields_and_unknown_fields_fail() -> Result<()> {
        let original=serde_json::json!({"periodic":false,"reason":"no_periodic_cell","vectors_angstrom":null,"volume_angstrom3":null,"orthorhombic":null,"minimum_image_required":null,"coordinate_wrapping_required":null,"reciprocal_method":null,"real_space_cutoff_angstrom":null,"contract_sha256":null});
        let cell:PeriodicCellContract=serde_json::from_value(original.clone())?;
        assert_eq!(serde_json::to_value(&cell)?,original);
        assert!(cell.validated_recorded_pme().map_err(anyhow::Error::msg)?.is_none());
        let mut unknown=original.clone();unknown["unrecognized_boundary"]=serde_json::json!(true);
        assert!(serde_json::from_value::<PeriodicCellContract>(unknown).is_err());
        let mut partial=original;partial["boundary_condition"]=serde_json::json!("ORTHORHOMBIC_TINFOIL_3D");
        assert!(serde_json::from_value::<PeriodicCellContract>(partial)?.validated_recorded_pme().is_err());
        Ok(())
    }
    #[test]
    #[ignore="reads unchanged original periodic no-probe and fragment sources; no simulation"]
    fn actual_expanded_cells_preserve_all_fields_hashes_and_reject_tampering() -> Result<()> {
        for index in 0..2 {
            let raw=actual_raw(index)?;
            let cell:PeriodicCellContract=serde_json::from_value(raw["periodic_cell"].clone())?;
            let pme=cell.validated_recorded_pme().map_err(anyhow::Error::msg)?.context("actual expanded PME absent")?;
            let mut serialized=serde_json::to_value(&cell)?;
            serialized.as_object_mut().unwrap().retain(|_,v| !v.is_null());
            assert_eq!(serialized,raw["periodic_cell"],"original expanded cell projection changed");
            assert_eq!(pme.pme_grid_dimensions,[270,270,270]);
            assert_eq!(pme.parameterization_digest_sha256().map_err(anyhow::Error::msg)?,pme.pme_parameterization_sha256);
            let mut bad=cell.clone();bad.pme_grid_dimensions=Some([269,270,270]);assert!(bad.validated_recorded_pme().is_err());
            let mut bad=cell.clone();bad.boundary_condition=None;assert!(bad.validated_recorded_pme().is_err());
            let mut bad=cell.clone();bad.contract_sha256=Some("0".repeat(64));assert!(bad.validated_recorded_pme().is_err());
            let mut bad=cell;bad.vectors_angstrom.as_mut().unwrap()[0][0]=f64::NAN;assert!(bad.validated_recorded_pme().is_err());
        }
        Ok(())
    }
    #[test]
    #[ignore="checks full current typed load of unchanged periodic input; any remaining schema contradiction is a failure"]
    fn actual_periodic_topologies_load_preserving_signed_and_pme_identity() -> Result<()> {
        for index in 0..2 {
            let raw=actual_raw(index)?;
            let mut topology:PrismPrepTopology=serde_json::from_value(raw.clone())?;
            topology.validate().map_err(anyhow::Error::msg)?;
            topology.validate_recorded_periodic_authority().map_err(anyhow::Error::msg)?;
            let water=topology.water_model_contract.as_ref().context("actual water contract absent")?;
            assert_eq!(python_canonical_sha256(&serde_json::to_value(&water.reference_parameters)?).map_err(anyhow::Error::msg)?,water.reference_parameters_sha256);
            let runtime=water.validated_runtime_masses_dalton().map_err(anyhow::Error::msg)?;
            assert_eq!(Some(runtime),water.reference_parameters.runtime_masses_dalton);
            let mut mixed=water.clone();mixed.reference_parameters.masses_dalton=Some(runtime);assert!(mixed.validated_runtime_masses_dalton().is_err());
            let mut missing=water.clone();missing.reference_parameters.model_masses_dalton=None;assert!(missing.validated_runtime_masses_dalton().is_err());
            let mut tampered=water.clone();tampered.reference_parameters.runtime_masses_dalton.as_mut().unwrap()[0]+=1.0;assert!(tampered.validated_runtime_masses_dalton().is_err());
            for (original, mut typed) in [(&raw["water_model_contract"],serde_json::to_value(&topology.water_model_contract)?),(&raw["topology_provenance"],serde_json::to_value(&topology.topology_provenance)?)] {
                typed.as_object_mut().context("typed preparation metadata absent")?.retain(|_,v| !v.is_null());
                assert_eq!(&typed,original,"original preparation authority fields changed");
                let hash=typed["contract_sha256"].as_str().context("original metadata hash absent")?.to_owned();
                typed.as_object_mut().unwrap().remove("contract_sha256");
                assert_eq!(python_canonical_sha256(&typed).map_err(anyhow::Error::msg)?,hash,"original preparation authority hash changed");
            }
            assert_eq!(serde_json::to_value(topology.atom_analysis_residue_ids.as_ref().context("signed axis absent")?)?,raw["atom_analysis_residue_ids"]);
            if let Some(system)=topology.lambda_fragment_system.as_mut() {
                assert_eq!(serde_json::to_value(&system.pme)?,raw["lambda_fragment_system"]["pme"]);
                system.pme.pme_grid_dimensions[0]+=1;
                assert!(topology.validate_recorded_periodic_authority().is_err(),"conflicting lambda authority accepted");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod author_residue_mapping_tests {
    use super::*;

    fn contract() -> AnalysisResidueContract {
        let mut value = serde_json::json!({
            "schema_kind": "prismos_analysis_residue_mapping", "schema_version": 1,
            "analysis_authority": "KCC_GHOST_PROTEIN_RESIDUE_AXIS", "excluded_atom_sentinel": -1,
            "n_atoms": 6, "total_physical_residue_count": 6, "n_analysis_residues": 5,
            "physical_residue_roles": ["RECEPTOR", "WATER", "RECEPTOR", "RECEPTOR", "RECEPTOR", "RECEPTOR"],
            "physical_to_analysis_residue_ids": [0, -1, 1, 2, 3, 4],
            "analysis_residues": [
                {"analysis_residue_id":0,"physical_residue_id":0,"chain_id":"A","pdb_residue_id":-2,"residue_name":"MET","role":"RECEPTOR"},
                {"analysis_residue_id":1,"physical_residue_id":2,"chain_id":"A","pdb_residue_id":0,"residue_name":"ALA","role":"RECEPTOR"},
                {"analysis_residue_id":2,"physical_residue_id":3,"chain_id":"A","pdb_residue_id":42,"residue_name":"GLY","role":"RECEPTOR"},
                {"analysis_residue_id":3,"physical_residue_id":4,"chain_id":"A","pdb_residue_id":42,"insertion_code":"A","residue_name":"GLY","role":"RECEPTOR"},
                {"analysis_residue_id":4,"physical_residue_id":5,"chain_id":"B","pdb_residue_id":42,"residue_name":"SER","role":"RECEPTOR"}
            ],
            "atom_analysis_residue_ids_sha256":"test-map", "physical_molecule_table_sha256":"test-molecules",
            "lambda_fragment_system_sha256":null, "role_counts":{"RECEPTOR":5,"WATER":1},
            "excluded_roles":["WATER","ION","LAMBDA_FRAGMENT","OTHER"],
            "must_recompute_after_fragment_materialization":true
        });
        let digest = python_canonical_sha256(&value).unwrap();
        value["contract_sha256"] = digest.into();
        serde_json::from_value(value).unwrap()
    }

    fn reseal(c: &mut AnalysisResidueContract) {
        let mut value = serde_json::to_value(&*c).unwrap();
        value.as_object_mut().unwrap().remove("contract_sha256");
        c.contract_sha256 = python_canonical_sha256(&value).unwrap();
    }

    #[test]
    fn signed_gapped_insertion_and_chain_author_identity_is_preserved() {
        let c = contract();
        c.validate_author_mapping().unwrap();
        let refs = c.author_references(&[4, 3, 2, 1, 0]).unwrap();
        assert_eq!(refs[0].chain_id, "B");
        assert_eq!(refs[1].insertion_code, "A");
        assert_eq!(refs[2].pdb_residue_id, 42);
        assert_eq!(refs[3].pdb_residue_id, 0);
        assert_eq!(refs[4].pdb_residue_id, -2);
        assert_eq!(refs[3].physical_residue_id, 2);
    }

    #[test]
    fn historical_empty_insertion_code_does_not_change_hash_payload() {
        let c = contract();
        let v = serde_json::to_value(&c.analysis_residues[0]).unwrap();
        assert!(v.get("insertion_code").is_none());
        c.validate_author_mapping().unwrap();
    }

    #[test]
    fn duplicate_missing_and_stale_author_mappings_are_rejected() {
        let mut c = contract();
        c.analysis_residues[3].insertion_code.clear();
        reseal(&mut c);
        assert!(c.validate_author_mapping().unwrap_err().contains("duplicate"));
        let mut c = contract();
        c.analysis_residues[1].analysis_residue_id = 8;
        reseal(&mut c);
        assert!(c.validate_author_mapping().is_err());
        let mut c = contract();
        c.analysis_residues[0].pdb_residue_id = 1;
        assert!(c.validate_author_mapping().unwrap_err().contains("SHA256"));
        assert!(contract().author_references(&[-1]).is_err());
        assert!(contract().author_references(&[5]).is_err());
    }
}
