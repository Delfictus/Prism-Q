//! PRISM-Therm output pipeline: JSON reports, druggability PDB, ranked pocket table.
//!
//! Converts `PrismThermAnalysis` into publication-ready output files and
//! a human-readable ranked pocket summary for the log.

use serde::{Serialize, Serializer};
use std::io::Write;
use std::path::Path;

use crate::input::PrismPrepTopology;
use crate::sdst_bridge::{
    validate_materialized_sites, AnalyzedSiteIdentity, MaterializedSite, MaterializedSiteId,
    PrismThermAnalysis, PrismThermSiteResult, SiteUnmatchedReason, TideResidueResult,
};

/// Serialize an `f32` honestly: finite values become numbers; NaN/Inf become
/// JSON `null`. Producer-repair contract: NaN means "not honestly computable"
/// (e.g. saturated TE/Fisher trains, undefined causal lag), and the on-disk
/// representation must distinguish that from a real `0.0`.
fn serialize_finite_or_null<S: Serializer>(v: &f32, s: S) -> Result<S::Ok, S::Error> {
    if v.is_finite() {
        s.serialize_f32(*v)
    } else {
        s.serialize_none()
    }
}

// ---------------------------------------------------------------------------
// ResidueRole classification
// ---------------------------------------------------------------------------

/// Functional role of a TIDE-identified residue with respect to a pocket.
#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub enum ResidueRole {
    /// High TE + high Fisher: causally drives pocket AND high leverage for perturbation.
    /// Prime drug design target.
    Trigger,
    /// High TE + low Fisher: causally connected but not a leverage point.
    /// Conformationally important but harder to exploit.
    Stabilizer,
    /// High KL divergence: large heating/cooling asymmetry.
    /// Gatekeeper residue controlling pocket access.
    Gateway,
    /// Low everything: minimal causal influence on pocket.
    Spectator,
}

impl std::fmt::Display for ResidueRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResidueRole::Trigger => write!(f, "TRIGGER"),
            ResidueRole::Stabilizer => write!(f, "STABIL."),
            ResidueRole::Gateway => write!(f, "GATEWAY"),
            ResidueRole::Spectator => write!(f, "SPECTAT"),
        }
    }
}

/// Classify a TIDE residue based on transfer entropy, Fisher info, and KL divergence.
fn classify_residue(r: &TideResidueResult) -> ResidueRole {
    let high_te = r.transfer_entropy > 0.005;
    let high_fisher = r.fisher_info > 0.002;
    let high_kl = r.kl_divergence > 0.1;

    if high_te && high_fisher {
        ResidueRole::Trigger
    } else if high_te {
        ResidueRole::Stabilizer
    } else if high_kl {
        ResidueRole::Gateway
    } else {
        ResidueRole::Spectator
    }
}

// ---------------------------------------------------------------------------
// PocketReport for standalone JSON
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct ResidueContribution {
    pub residue_id: u32,
    pub residue_name: String,
    /// Transfer entropy in bits. NaN ⇒ undefined (saturated trains) ⇒ JSON `null`.
    #[serde(serialize_with = "serialize_finite_or_null")]
    pub transfer_entropy: f32,
    /// Causal ΔG in kcal/mol. NaN ⇒ undefined (TE undefined) ⇒ JSON `null`.
    #[serde(serialize_with = "serialize_finite_or_null")]
    pub causal_dg: f32,
    /// Fisher information. NaN ⇒ undefined ⇒ JSON `null`.
    #[serde(serialize_with = "serialize_finite_or_null")]
    pub fisher_info: f32,
    /// KL divergence between heating/cooling Bernoulli rates.
    #[serde(serialize_with = "serialize_finite_or_null")]
    pub kl_divergence: f32,
    pub role: String,
    pub n_causal_spikes: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct PocketReport {
    pub pocket_id: i32,
    pub analyzed_site_identity: AnalyzedSiteIdentity,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub materialized_site_id: Option<MaterializedSiteId>,
    /// Real centroid of the site this pocket was materialized from.
    ///
    /// `None` when the centroid could not be resolved. It is deliberately not
    /// defaulted: writing `[0,0,0]` on a failed join produced a spatially
    /// meaningless pocket at the coordinate origin that read as a real result
    /// in both the report JSON and the druggability PDB.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub centroid: Option<[f32; 3]>,
    /// Typed reason this pocket has no centroid, when it has none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub centroid_unmatched_reason: Option<SiteUnmatchedReason>,
    pub ccns_tau: f32,
    pub ccns_class: String,
    pub druggability_score: f32,
    pub hysteresis_asymmetry: f32,
    pub relative_asymmetry: f32,
    pub is_cryptic: bool,
    pub therm_class: String,
    pub top_residues: Vec<ResidueContribution>,
}

/// Typed geometry resolution used by lattice reports and their regression
/// tests. `centroid=None` never carries a substitute coordinate.
#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub struct MaterializedSiteGeometryResolution {
    pub centroid: Option<[f32; 3]>,
    pub aabb_min: Option<[f32; 3]>,
    pub aabb_max: Option<[f32; 3]>,
    pub unmatched_reason: Option<SiteUnmatchedReason>,
}

/// Resolve report geometry through exact materialized and analyzed identities.
pub fn resolve_materialized_site_geometry(
    result: &PrismThermSiteResult,
    materialized_sites: &[MaterializedSite],
) -> Result<MaterializedSiteGeometryResolution, anyhow::Error> {
    validate_materialized_sites(materialized_sites)?;
    let Some(materialized_site_id) = result.materialized_site_id else {
        return Ok(MaterializedSiteGeometryResolution {
            centroid: None,
            aabb_min: None,
            aabb_max: None,
            unmatched_reason: Some(SiteUnmatchedReason::NoMaterializationMapping),
        });
    };
    let Some(site) = materialized_sites
        .iter()
        .find(|site| site.materialized_site_id == materialized_site_id)
    else {
        return Ok(MaterializedSiteGeometryResolution {
            centroid: None,
            aabb_min: None,
            aabb_max: None,
            unmatched_reason: Some(SiteUnmatchedReason::MissingCompanionData),
        });
    };
    if !result.matches_analyzed(site.analyzed_site_identity) {
        return Ok(MaterializedSiteGeometryResolution {
            centroid: None,
            aabb_min: None,
            aabb_max: None,
            unmatched_reason: Some(SiteUnmatchedReason::ConsumerDomainMismatch),
        });
    }
    Ok(MaterializedSiteGeometryResolution {
        centroid: Some(site.centroid),
        aabb_min: Some(site.aabb_min),
        aabb_max: Some(site.aabb_max),
        unmatched_reason: None,
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct PrismThermReport {
    pub structure: String,
    pub sdst_event_count: u32,
    pub total_pockets: usize,
    pub cryptic_pockets: usize,
    pub tide_residues_mapped: usize,
    pub pockets: Vec<PocketReport>,
}

// ---------------------------------------------------------------------------
// Output functions
// ---------------------------------------------------------------------------

/// Build the standalone PRISM-Therm report from analysis results.
pub fn build_report(
    analysis: &PrismThermAnalysis,
    topology: &PrismPrepTopology,
    structure_name: &str,
    site_centroids: &[([f32; 3], i32)], // (centroid, site_id) pairs
) -> PrismThermReport {
    // This compatibility entry point is explicitly CCL-only. A lattice report
    // must use `build_report_for_materialized_sites`; allowing lattice rows to
    // join through this integer tuple would recreate the cross-domain defect.
    let resolutions: Vec<MaterializedSiteGeometryResolution> = analysis
        .sites
        .iter()
        .map(|site| {
            let centroid = match site.analyzed_site_identity {
                AnalyzedSiteIdentity::Cluster(cluster_id) => {
                    i32::try_from(cluster_id.0).ok().and_then(|cluster_id| {
                        site_centroids
                            .iter()
                            .find(|(_, id)| *id == cluster_id)
                            .map(|(centroid, _)| *centroid)
                    })
                }
                AnalyzedSiteIdentity::LatticeComponent(_)
                | AnalyzedSiteIdentity::MaterializerRegion(_) => None,
            };
            MaterializedSiteGeometryResolution {
                centroid,
                aabb_min: None,
                aabb_max: None,
                unmatched_reason: if centroid.is_some() {
                    None
                } else if matches!(
                    site.analyzed_site_identity,
                    AnalyzedSiteIdentity::Cluster(_)
                ) {
                    Some(SiteUnmatchedReason::MissingCompanionData)
                } else {
                    Some(SiteUnmatchedReason::ConsumerDomainMismatch)
                },
            }
        })
        .collect();
    build_report_with_geometry(analysis, topology, structure_name, &resolutions)
}

/// Build a lattice report from canonical materialized-site records.
pub fn build_report_for_materialized_sites(
    analysis: &PrismThermAnalysis,
    topology: &PrismPrepTopology,
    structure_name: &str,
    materialized_sites: &[MaterializedSite],
) -> Result<PrismThermReport, anyhow::Error> {
    validate_materialized_sites(materialized_sites)?;
    let resolutions = analysis
        .sites
        .iter()
        .map(|site| resolve_materialized_site_geometry(site, materialized_sites))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(build_report_with_geometry(
        analysis,
        topology,
        structure_name,
        &resolutions,
    ))
}

fn build_report_with_geometry(
    analysis: &PrismThermAnalysis,
    topology: &PrismPrepTopology,
    structure_name: &str,
    resolutions: &[MaterializedSiteGeometryResolution],
) -> PrismThermReport {
    debug_assert_eq!(analysis.sites.len(), resolutions.len());
    let mut pockets: Vec<PocketReport> = Vec::new();

    for (site, resolution) in analysis.sites.iter().zip(resolutions) {
        let top_residues: Vec<ResidueContribution> = site
            .tide_decomposition
            .iter()
            .map(|r| {
                let role = classify_residue(r);
                let res_name = residue_label(topology, r.residue_id);
                ResidueContribution {
                    residue_id: r.residue_id,
                    residue_name: res_name,
                    transfer_entropy: r.transfer_entropy,
                    causal_dg: r.causal_dg,
                    fisher_info: r.fisher_info,
                    kl_divergence: r.kl_divergence,
                    role: role.to_string(),
                    n_causal_spikes: r.n_causal_spikes,
                }
            })
            .collect();

        pockets.push(PocketReport {
            pocket_id: site.site_id,
            analyzed_site_identity: site.analyzed_site_identity,
            materialized_site_id: site.materialized_site_id,
            centroid: resolution.centroid,
            centroid_unmatched_reason: resolution.unmatched_reason,
            ccns_tau: site.tau,
            ccns_class: site.ccns_classification.clone(),
            druggability_score: if site.druggability.is_finite() {
                site.druggability
            } else {
                0.0
            },
            hysteresis_asymmetry: site.asymmetry_score,
            is_cryptic: site.therm_class == crate::sdst_bridge::ThermClass::Cryptic,
            therm_class: site.therm_class.to_string(),
            relative_asymmetry: site.relative_asymmetry,
            top_residues,
        });
    }

    // Sort by druggability descending
    pockets.sort_by(|a, b| {
        b.druggability_score
            .partial_cmp(&a.druggability_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let cryptic_count = pockets.iter().filter(|p| p.is_cryptic).count();

    PrismThermReport {
        structure: structure_name.to_string(),
        sdst_event_count: analysis.sdst_event_count,
        total_pockets: pockets.len(),
        cryptic_pockets: cryptic_count,
        tide_residues_mapped: analysis.tide_residues_mapped,
        pockets,
    }
}

/// Write the standalone `.prism_therm.json` file.
pub fn write_json(
    report: &PrismThermReport,
    output_dir: &Path,
    base_name: &str,
) -> std::io::Result<()> {
    let path = output_dir.join(format!("{}.prism_therm.json", base_name));
    let json = serde_json::to_string_pretty(report)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    std::fs::write(&path, json)?;
    log::info!("  PRISM-Therm JSON: {}", path.display());
    Ok(())
}

/// Write a druggability PDB where B-factor = per-residue druggability score.
///
/// The druggability score for each residue is computed as:
/// - For residues in the TIDE top-20 for any site: max(TE across sites) × 100
/// - For other residues: 0.0
///
/// This produces a heatmap viewable in PyMOL/ChimeraX/3Dmol.js:
///   `spectrum b, blue_white_red, minimum=0, maximum=50`
pub fn write_druggability_pdb(
    report: &PrismThermReport,
    topology: &PrismPrepTopology,
    output_dir: &Path,
    base_name: &str,
) -> std::io::Result<()> {
    // Build per-residue druggability: max TE × 100 across all pockets
    let n_residues = topology.n_residues;
    let mut res_score = vec![0.0f32; n_residues];

    for pocket in &report.pockets {
        for r in &pocket.top_residues {
            let rid = r.residue_id as usize;
            if rid < n_residues {
                let score = r.transfer_entropy * 100.0;
                if score > res_score[rid] {
                    res_score[rid] = score;
                }
            }
        }
    }

    let path = output_dir.join(format!("{}.druggability.pdb", base_name));
    let mut f = std::fs::File::create(&path)?;

    writeln!(f, "REMARK   PRISM-Therm druggability PDB")?;
    writeln!(
        f,
        "REMARK   B-factor = Transfer Entropy × 100 (per-residue max across pockets)"
    )?;
    writeln!(
        f,
        "REMARK   Visualize: spectrum b, blue_white_red, minimum=0, maximum=50"
    )?;

    for i in 0..topology.n_atoms {
        let x = topology.positions[i * 3];
        let y = topology.positions[i * 3 + 1];
        let z = topology.positions[i * 3 + 2];
        let res_id = topology.residue_ids[i];
        let b_factor = if res_id < n_residues {
            res_score[res_id]
        } else {
            0.0
        };

        let atom_name = &topology.atom_names[i];
        let res_name = if i < topology.residue_names.len() {
            &topology.residue_names[i] // atom-indexed, NOT res_id-indexed
        } else {
            "UNK"
        };
        let chain_id = if i < topology.chain_ids.len() {
            topology.chain_ids[i].chars().next().unwrap_or('A')
        } else {
            'A'
        };

        // PDB ATOM record format (fixed-width columns)
        write!(
            f,
            "ATOM  {:>5} {:>4} {:>3} {}{:>4}    {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}\n",
            (i + 1) % 100000,
            format_atom_name(atom_name),
            res_name,
            chain_id,
            (res_id + 1) % 10000,
            x,
            y,
            z,
            1.00, // occupancy
            b_factor.min(99.99),
        )?;
    }
    writeln!(f, "END")?;

    log::info!("  PRISM-Therm druggability PDB: {}", path.display());
    Ok(())
}

/// Print the ranked pocket summary table to the log.
pub fn print_summary_table(report: &PrismThermReport) {
    log::info!("");
    log::info!("╔═════╦══════════════════════╦═══════╦════════╦══════════╦═════════╗");
    log::info!("║  #  ║ Centroid (Å)         ║  tau  ║ Drug.  ║ Asymm.   ║ Type    ║");
    log::info!("╠═════╬══════════════════════╬═══════╬════════╬══════════╬═════════╣");

    for (i, p) in report.pockets.iter().enumerate() {
        let ptype = match p.therm_class.as_str() {
            "CRYPTIC" => "CRYPTIC ",
            "DYNAMIC" => "DYNAMIC ",
            "RESPONSIVE" => "RESPOND.",
            "INERT" => "INERT   ",
            _ => "?       ",
        };
        // An unresolved centroid is shown as unmatched, never as (0.0,0.0,0.0).
        let centroid_cell = match p.centroid {
            Some(c) => format!("({:>5.1},{:>5.1},{:>5.1})", c[0], c[1], c[2]),
            None => p
                .centroid_unmatched_reason
                .map(|reason| format!("{:>20}", format!("{reason:?}")))
                .unwrap_or_else(|| format!("{:>20}", "UNMATCHED")),
        };
        log::info!(
            "║ {:>3} ║ {} ║ {:>5.2} ║ {:>5.2}  ║ {:>7.3}  ║ {} ║",
            i + 1,
            centroid_cell,
            p.ccns_tau,
            p.druggability_score,
            p.hysteresis_asymmetry,
            ptype,
        );
    }

    log::info!("╚═════╩══════════════════════╩═══════╩════════╩══════════╩═════════╝");

    // Print top causal residues for CRYPTIC pockets
    for p in &report.pockets {
        if !p.is_cryptic || p.top_residues.is_empty() {
            continue;
        }

        log::info!("");
        log::info!(
            "  TIDE — Pocket {} (CRYPTIC, tau={:.2}, drug={:.2}):",
            p.pocket_id,
            p.ccns_tau,
            p.druggability_score
        );
        log::info!(
            "    {:>4}  {:>6}  {:>8}  {:>8}  {:>8}  {:>7}  {:>5}",
            "Res",
            "Name",
            "TE",
            "dG",
            "Fisher",
            "KL",
            "Role"
        );
        log::info!("    {}", "─".repeat(60));

        for r in p.top_residues.iter().take(10) {
            log::info!(
                "    {:>4}  {:>6}  {:>8.5}  {:>8.5}  {:>8.5}  {:>7.4}  {:>7}",
                r.residue_id,
                r.residue_name,
                r.transfer_entropy,
                r.causal_dg,
                r.fisher_info,
                r.kl_divergence,
                r.role,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Get a human-readable residue label like "ALA42" from topology.
/// Uses residue_id_to_name() to correctly map residue_id → name
/// (residue_names is atom-indexed, not residue-indexed).
fn residue_label(topology: &PrismPrepTopology, res_id: u32) -> String {
    let rid = res_id as usize;
    let res_names = topology.residue_id_to_name();
    if rid < res_names.len() {
        format!("{}{}", res_names[rid], rid + 1)
    } else {
        format!("?{}", rid + 1)
    }
}

/// Format atom name for PDB (right-justify in 4-char field, with leading space for 1-3 char names).
fn format_atom_name(name: &str) -> String {
    if name.len() >= 4 {
        name[..4].to_string()
    } else {
        format!(" {:<3}", name)
    }
}
