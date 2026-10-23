//! # PRISM-TWIN site detection
//!
//! First-class TWIN-aware site detection that uses the interferometric data
//! produced by `coupled_md::run_coupled_twin`. This is NOT a wrapper around
//! single-stream detection — it consumes signals that only coupled observation
//! can produce:
//!
//! 1. **Consensus** — per-residue agreement between Stream A (scout) and
//!    Stream B (observer). High consensus = reproducible pocket dynamics,
//!    detectable only because we have two independent observations.
//! 2. **Differential** — residues that light up in Stream B but not Stream A.
//!    Stream B carries NMA-biased perturbation during warm_hold; residues
//!    active only in B are BARRIER-GATED (invisible to thermal-only MD).
//! 3. **Cross-stream CCF** — mean-centered cross-correlation between the two
//!    streams' per-residue spike matrices. High distant-pair CCF = allosteric
//!    coupling. One site opening causes another to open.
//! 4. **Phase structure** — per-phase spike counts per stream from the CCNS
//!    protocol (cold_hold, heating, warm_hold, cooling, cold_return) for
//!    thermal vs barrier-gated classification.
//!
//! Pipeline:
//!
//! ```text
//! spikes_a + spikes_b + topology
//!   │
//!   ├─ map spikes to nearest CA (per stream, per phase)
//!   │
//!   ├─ consensus vector (both streams agree)
//!   ├─ differential vector (B-only: NMA-exclusive)
//!   ├─ mean-centered CCF matrix (allosteric)
//!   │
//!   ├─ candidate residues = union of
//!   │    - top consensus
//!   │    - top differential
//!   │    - top CCF centrality
//!   │
//!   ├─ spatial cluster candidates → raw sites
//!   │
//!   ├─ classify each site:
//!   │    CONSENSUS_CRYPTIC | BARRIER_GATED | ALLOSTERIC_HUB
//!   │    COOPERATIVE_NETWORK | NMA_RESPONSIVE | THERMAL_TRANSIENT
//!   │    PREFORMED_STABLE
//!   │
//!   ├─ twin rank score (6 signals, interferometric)
//!   │
//!   └─ write {prefix}.binding_sites.json (twin schema)
//!         {prefix}.kcc_visualization.json (CCF-derived)
//!         {prefix}.topology.prism_therm.json (phase-derived)
//!         {prefix}.ensemble_trajectory.json (stub)
//!         {prefix}.twin_per_residue.json (48+ TWIN fields per residue)
//!         {prefix}.twin_ccf_matrix.json (full CCF matrix for downstream)
//! ```

#![cfg(feature = "gpu")]

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::fused_engine::GpuSpikeEvent;
use crate::input::PrismPrepTopology;

// ─────────────────────────────────────────────────────────────────────────────
// Constants — tunable detection parameters
// ─────────────────────────────────────────────────────────────────────────────

/// Spatial clustering radius for site construction (Å).
const CLUSTER_EPS_ANGSTROM: f32 = 8.0;

/// Minimum candidate residues per cluster to be emitted as a site.
const CLUSTER_MIN_PTS: usize = 3;

/// Nearest-CA cutoff for spike→residue assignment (Å).
const SPIKE_TO_CA_CUTOFF: f32 = 12.0;

/// Number of time bins for CCF matrix.
const CCF_TIME_BINS: usize = 200;

/// Maximum spikes per stream to use for CCF (downsampled if exceeded).
const CCF_SAMPLE_LIMIT: usize = 600_000;

/// Allosteric threshold: pairs with |i-j| > this AND ccf > threshold are allosteric.
const ALLOSTERIC_MIN_SEPARATION: usize = 20;

/// Protocol phase boundaries as fractions of total timesteps (--fast --hysteresis).
/// Matches the engine's CCNS 5-phase protocol exactly.
const PHASE_BOUNDS_FAST_HYSTERESIS: [(f32, f32, &str); 5] = [
    (0.000, 0.311, "cold_hold"),
    (0.311, 0.444, "heating"),
    (0.444, 0.778, "warm_hold"),
    (0.778, 0.889, "cooling"),
    (0.889, 1.000, "cold_return"),
];

/// TWIN rank score weights. Base components sum to 1.0.
/// The interferometric multiplier is applied AFTER the base score.
const RANK_W_CONSENSUS: f32 = 0.25;
const RANK_W_DIFFERENTIAL: f32 = 0.15;
const RANK_W_ALLOSTERIC: f32 = 0.15;
const RANK_W_DRUGGABILITY: f32 = 0.15;
const RANK_W_HYSTERESIS: f32 = 0.10;
const RANK_W_SPIKE_DENSITY: f32 = 0.10;
const RANK_W_PHASE_QUALITY: f32 = 0.10; // warm_hold spike enrichment

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Summary of the twin detection run. Written into coupled_twin_result.json
/// by the caller to keep a single source of truth.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwinDetectionSummary {
    pub n_sites: usize,
    pub n_consensus_sites: usize,
    pub n_barrier_gated_sites: usize,
    pub n_allosteric_hub_sites: usize,
    pub n_cooperative_network_sites: usize,
    pub n_nma_responsive_sites: usize,
    pub n_thermal_transient_sites: usize,
    pub n_preformed_stable_sites: usize,
    pub ccf_distant_pair_mean: f32,
    pub ccf_max_offdiag: f32,
    pub n_candidate_residues: usize,
    pub n_ca_atoms: usize,
    pub elapsed_seconds: f64,
    pub files_written: Vec<String>,
}

/// The full twin detection entry point. Call AFTER `run_coupled_twin` has
/// returned, passing the two spike streams plus metadata. Writes all twin
/// detection artifacts to `output_dir` under `{prefix}.*` naming.
///
/// This function does NOT require a GPU (all CPU), but is gated on the gpu
/// feature because it consumes `GpuSpikeEvent`.
pub fn detect_and_write_twin_sites(
    spikes_a: &[GpuSpikeEvent],
    spikes_b: &[GpuSpikeEvent],
    topology: &PrismPrepTopology,
    // LINK #1: OPTIONAL second system (holo / variant). When Some it drives the
    // ResidueAlignment built below; the second ca_xyz that projects B into the
    // reference CA-ordinal frame is wired in a later stage. Sourcing of n_res /
    // ca_xyz stays on the primary topology (=A, N_ref) so no-flag behavior is
    // unchanged.
    topology_b: Option<&PrismPrepTopology>,
    output_dir: &Path,
    prefix: &str,
    nma_modes_path: Option<&str>,
) -> Result<TwinDetectionSummary> {
    let t0 = std::time::Instant::now();
    std::fs::create_dir_all(output_dir)?;

    log::info!("╔═══════════════════════════════════════════════════════════════╗");
    log::info!("║  PRISM-TWIN INTERFEROMETRIC SITE DETECTION                     ║");
    log::info!("║  Consumes: consensus + differential + CCF + phase structure   ║");
    log::info!("╚═══════════════════════════════════════════════════════════════╝");

    // ── Structure ──
    let ca_xyz = extract_ca_positions(topology);
    let n_res = ca_xyz.len();
    log::info!("  Structure: {} CA atoms", n_res);
    log::info!("  Stream A: {} spikes (scout)", spikes_a.len());
    log::info!(
        "  Stream B: {} spikes (observer, thermal+NMA)",
        spikes_b.len()
    );

    let max_ts = spikes_a
        .iter()
        .chain(spikes_b.iter())
        .map(|s| s.timestep)
        .max()
        .unwrap_or(1) as f32;

    // ── Step 1: per-residue per-phase spike counts ──
    //
    // The original implementation scanned every CA atom once for total counts
    // and again for phase counts, per stream. On 170M+ spike coupled-TWIN runs
    // that made post-MD CPU-bound. The spatial index is lossless for the same
    // nearest-CA cutoff: with cell_size == cutoff, any CA inside cutoff must be
    // in the spike cell or one of its 26 neighboring cells.
    let ca_index = CaSpatialIndex::new(&ca_xyz, SPIKE_TO_CA_CUTOFF);
    log::info!(
        "  Spike→CA assignment: exact spatial index ({} occupied cells, cutoff={:.1}Å)",
        ca_index.occupied_cells(),
        SPIKE_TO_CA_CUTOFF
    );

    // ── LINK #1: sequence-position residue alignment (two-system contrast) ──
    //
    // When a second topology is supplied it is a CONTROLLED DERIVATIVE of the
    // reference protein frame (holo / design variant). The alignment is built
    // BEFORE stream B is assigned so B can be projected into the reference
    // CA-ordinal frame. `ResidueAlignment::build` HARD-ERRORS on any
    // protein-frame mismatch (missing / reordered / count-changed protein
    // residues) — never a silent `.min()` truncation.
    let two_system = topology_b.is_some();
    let alignment = match topology_b {
        Some(topo_b) => {
            let a = ResidueAlignment::build(topology, topo_b)?;
            log::info!(
                "  LINK #1 alignment: {} shared residues ({} CA), {} B-only ligand/ion residue(s), {} design mutation(s)",
                a.n_ref_residues,
                a.n_ref_ca,
                a.b_only_residues.len(),
                a.n_mutations
            );
            Some(a)
        }
        None => None,
    };

    // Stream A: ALWAYS assigned against the reference ca_xyz (=A). Byte-identical
    // to the historical single-stream path.
    let (counts_a, phase_counts_a) =
        assign_spikes_to_residues_and_phases(spikes_a, &ca_xyz, &ca_index, max_ts);

    // Stream B assignment (Path-1):
    //   single-topology  → assign against the SAME reference ca_xyz(A)
    //                       (byte-identical; both streams share one frame).
    //   two-system LINK #1 → (a) PRE-FILTER: drop any spike whose PRIMARY nearby
    //     residue is a B-only residue (ordinal >= N_ref) into the ligand-contact
    //     localization channel so ligand spikes are NOT snapped onto pocket-lining
    //     CAs (mirrors the Path-2 GPU filter at twin_kernels.rs:833);
    //     (b) assign the retained PROTEIN spikes against a SECOND ca_xyz_b built
    //     from topology_b's own coordinate frame;
    //     (c) PROJECT those B CA-ordinal counts into the reference CA-ordinal
    //     frame via the alignment (exact length n_res, 0 where B has no
    //     counterpart) so every consumer indexed by A's CA-ordinal stays in
    //     bounds.
    // `ca_xyz_b_store` / `spikes_b_protein` outlive the branch so the CCF B-frame
    // reference (Step-7) into them stays valid for `compute_ccf_matrix`.
    let ca_xyz_b_store: Vec<[f32; 3]>;
    let spikes_b_protein: Vec<GpuSpikeEvent>;
    #[allow(clippy::type_complexity)]
    let (counts_b, phase_counts_b, ccf_b_frame): (
        Vec<u32>,
        [Vec<u32>; 5],
        Option<(&[[f32; 3]], &[Option<usize>])>,
    ) = if let (Some(al), Some(topo_b)) = (alignment.as_ref(), topology_b) {
        let n_ref = al.n_ref_residues as i32;

        // (a) ligand pre-filter → ligand-contact localization channel + counter
        let mut retained: Vec<GpuSpikeEvent> = Vec::with_capacity(spikes_b.len());
        let mut ligand = LigandContactChannel::new(al.n_ref_residues);
        for s in spikes_b {
            let primary = if s.n_residues > 0 {
                s.nearby_residues[0]
            } else {
                -1
            };
            if primary >= n_ref {
                ligand.record(s, n_ref);
            } else {
                retained.push(*s);
            }
        }
        log::info!(
            "  LINK #1 Path-1 ligand filter: {} B ligand-region spike(s) → ligand-contact channel, {} protein spike(s) retained",
            ligand.spike_count,
            retained.len()
        );
        spikes_b_protein = retained;

        // (b) second CA frame for B; assign the protein spikes there.
        ca_xyz_b_store = extract_ca_positions(topo_b);
        let ca_index_b = CaSpatialIndex::new(&ca_xyz_b_store, SPIKE_TO_CA_CUTOFF);
        let (counts_b_native, phase_b_native) = assign_spikes_to_residues_and_phases(
            &spikes_b_protein,
            &ca_xyz_b_store,
            &ca_index_b,
            max_ts,
        );

        // (c) project B CA-ordinal counts into the reference CA-ordinal frame.
        assert_eq!(
            al.ca_a_to_b.len(),
            n_res,
            "LINK #1: CA-ordinal alignment length {} != reference CA count {}",
            al.ca_a_to_b.len(),
            n_res
        );
        let counts_b_ref = project_ca_ordinal(&counts_b_native, &al.ca_a_to_b, n_res);
        let phase_b_ref: [Vec<u32>; 5] = [
            project_ca_ordinal(&phase_b_native[0], &al.ca_a_to_b, n_res),
            project_ca_ordinal(&phase_b_native[1], &al.ca_a_to_b, n_res),
            project_ca_ordinal(&phase_b_native[2], &al.ca_a_to_b, n_res),
            project_ca_ordinal(&phase_b_native[3], &al.ca_a_to_b, n_res),
            project_ca_ordinal(&phase_b_native[4], &al.ca_a_to_b, n_res),
        ];

        // Ligand-contact localization channel (additive; two-system only).
        let lc_path = output_dir.join(format!("{prefix}.twin_ligand_contact.json"));
        match ligand.write_json(&lc_path, topology, topo_b) {
            Ok(()) => log::info!(
                "  LINK #1 ligand-contact channel: {} spike(s), {} contacted protein residue(s) → {}",
                ligand.spike_count,
                ligand.contact_residues.len(),
                lc_path.display()
            ),
            Err(e) => log::warn!("  LINK #1 ligand-contact channel write failed: {}", e),
        }

        (
            counts_b_ref,
            phase_b_ref,
            Some((ca_xyz_b_store.as_slice(), al.ca_a_to_b.as_slice())),
        )
    } else {
        // Single-topology: ca_xyz_b_store stays empty (never borrowed — ccf_b_frame
        // is None). spikes_b_protein must be initialized because it is textually
        // read by the two_system-gated ccf branch below.
        ca_xyz_b_store = Vec::new();
        let _ = &ca_xyz_b_store;
        spikes_b_protein = Vec::new();
        let (cb, pb) = assign_spikes_to_residues_and_phases(spikes_b, &ca_xyz, &ca_index, max_ts);
        (cb, pb, None)
    };

    // ── Step 2: consensus vector ──
    let consensus = compute_consensus_vector(&counts_a, &counts_b, two_system);
    let n_consensus = consensus.iter().filter(|&&c| c > 0.0).count();
    log::info!(
        "  Consensus residues (both streams active): {}/{}",
        n_consensus,
        n_res
    );

    // ── Step 3: differential vector ──
    let differential = compute_differential_vector(&counts_a, &counts_b, two_system);
    let n_differential = differential.iter().filter(|&&d| d > 0.0).count();
    log::info!(
        "  Differential residues (B-only, NMA-exclusive): {}/{}",
        n_differential,
        n_res
    );

    // ── Step 4: mean-centered CCF matrix ──
    //
    // LINK #1 (Path-1 CCF): single-topology assigns both streams to the one
    // ca_xyz(A). Two-system passes the retained protein spikes plus the B CA
    // frame + CA-ordinal map so `compute_ccf_matrix` accumulates B on ITS OWN
    // frame and remaps rows into the reference CA-ordinal frame before
    // correlating — never snapping B onto arbitrary A CAs across two
    // independently integrated coordinate frames.
    let ccf_spikes_b: &[GpuSpikeEvent] = if two_system {
        &spikes_b_protein
    } else {
        spikes_b
    };
    let (ccf, ccf_stats) = compute_ccf_matrix(spikes_a, ccf_spikes_b, &ca_xyz, n_res, ccf_b_frame);
    log::info!(
        "  CCF: distant-pair mean={:.4} max_offdiag={:.4} (< 0.15 ok)",
        ccf_stats.distant_pair_mean,
        ccf_stats.max_offdiag
    );

    // ── Step 5: candidate residues (union) ──
    let candidate_residues = build_candidate_residues(&consensus, &differential, &ccf);
    log::info!(
        "  Candidate residues (union top-consensus / top-differential / top-CCF-centrality): {}",
        candidate_residues.len()
    );

    // ── Step 6: spatial cluster → raw sites ──
    let raw_sites = cluster_candidates(&candidate_residues, &ca_xyz);
    log::info!("  Raw spatial clusters: {}", raw_sites.len());

    // ── LINK #1 Path-1 OOB guard ──
    // Every Path-1 consumer (build_twin_sites, build_per_residue_twin_features)
    // indexes counts_a/counts_b/phase/consensus/differential by A's CA-ordinal in
    // [0, n_res). In two-system mode B was projected into that exact frame; assert
    // equal length here so a projection bug is a HARD abort, not a silent OOB
    // panic deep inside build_twin_sites. Single-topology counts are always equal
    // length so this never fires on the no-flag path.
    if two_system {
        assert_eq!(counts_a.len(), n_res, "LINK #1: counts_a.len() != n_res");
        assert_eq!(
            counts_b.len(),
            n_res,
            "LINK #1: counts_b_ref.len() != n_res"
        );
        assert_eq!(consensus.len(), n_res, "LINK #1: consensus.len() != n_res");
        assert_eq!(
            differential.len(),
            n_res,
            "LINK #1: differential.len() != n_res"
        );
        for p in 0..5 {
            assert_eq!(
                phase_counts_b[p].len(),
                n_res,
                "LINK #1: phase_counts_b[{}].len() != n_res",
                p
            );
        }
    }

    // ── Step 7: per-site aggregate + classify + rank ──
    let mut sites = build_twin_sites(
        &raw_sites,
        &ca_xyz,
        topology,
        &counts_a,
        &counts_b,
        &phase_counts_a,
        &phase_counts_b,
        &consensus,
        &differential,
        &ccf,
    );

    // Classification, NMA responsive modes, twin rank
    for site in sites.iter_mut() {
        classify_twin_site(site);
    }
    compute_twin_ranks(&mut sites);
    sites.sort_by(|a, b| {
        b.twin_rank_score
            .partial_cmp(&a.twin_rank_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (rank, site) in sites.iter_mut().enumerate() {
        site.rank = (rank + 1) as u32;
    }

    let class_counts = tally_classifications(&sites);
    log::info!("  Sites detected: {} total", sites.len());
    log::info!(
        "    CONSENSUS_CRYPTIC:    {}",
        class_counts.consensus_cryptic
    );
    log::info!("    BARRIER_GATED:        {}", class_counts.barrier_gated);
    log::info!("    ALLOSTERIC_HUB:       {}", class_counts.allosteric_hub);
    log::info!(
        "    COOPERATIVE_NETWORK:  {}",
        class_counts.cooperative_network
    );
    log::info!("    NMA_RESPONSIVE:       {}", class_counts.nma_responsive);
    log::info!(
        "    THERMAL_TRANSIENT:    {}",
        class_counts.thermal_transient
    );
    log::info!(
        "    PREFORMED_STABLE:     {}",
        class_counts.preformed_stable
    );

    // ── Step 8: Build per-residue twin features ──
    let per_residue = build_per_residue_twin_features(
        &ca_xyz,
        topology,
        &counts_a,
        &counts_b,
        &phase_counts_a,
        &phase_counts_b,
        &consensus,
        &differential,
        &ccf,
    );

    // ── Step 9: Write outputs ──
    let mut files_written = Vec::new();

    // 1. binding_sites.json (twin schema, drop-in compatible with postprocess_twin.py)
    let bs_path = output_dir.join(format!("{prefix}.binding_sites.json"));
    write_binding_sites_json(&bs_path, &sites, topology, two_system)?;
    files_written.push(bs_path.display().to_string());

    // 2. kcc_visualization.json (CCF-derived)
    let kcc_path = output_dir.join(format!("{prefix}.kcc_visualization.json"));
    write_kcc_json(&kcc_path, &sites, &ccf, topology, two_system)?;
    files_written.push(kcc_path.display().to_string());

    // 3. prism_therm.json (phase-derived)
    let therm_path = output_dir.join(format!("{prefix}.topology.prism_therm.json"));
    write_prism_therm_json(&therm_path, &sites, two_system)?;
    files_written.push(therm_path.display().to_string());

    // 4. ensemble_trajectory.json (stub — empty list, twin doesn't track per-replica ensemble)
    let ens_path = output_dir.join(format!("{prefix}.ensemble_trajectory.json"));
    std::fs::write(
        &ens_path,
        r#"{"replicas": [], "note": "TWIN mode: ensemble trajectory is captured in coupled_spikes.json"}"#,
    )?;
    files_written.push(ens_path.display().to_string());

    // 5. Per-residue twin features
    let pr_path = output_dir.join(format!("{prefix}.twin_per_residue.json"));
    let pr_json = serde_json::to_string_pretty(&per_residue)?;
    std::fs::write(&pr_path, pr_json)?;
    files_written.push(pr_path.display().to_string());

    // 6. CCF matrix (flat row-major JSON for downstream consumption)
    let ccf_path = output_dir.join(format!("{prefix}.twin_ccf_matrix.json"));
    let ccf_flat: Vec<f32> = ccf.iter().flatten().copied().collect();
    let ccf_wire = serde_json::json!({
        "n_residues": n_res,
        "shape": [n_res, n_res],
        "ccf_row_major": ccf_flat,
        "distant_pair_mean": ccf_stats.distant_pair_mean,
        "max_offdiag": ccf_stats.max_offdiag,
        "mean_centered": true,
        "method": "mean_centered_per_residue_unit_norm_double_centering",
    });
    std::fs::write(&ccf_path, serde_json::to_string(&ccf_wire)?)?;
    files_written.push(ccf_path.display().to_string());

    let summary = TwinDetectionSummary {
        n_sites: sites.len(),
        n_consensus_sites: class_counts.consensus_cryptic,
        n_barrier_gated_sites: class_counts.barrier_gated,
        n_allosteric_hub_sites: class_counts.allosteric_hub,
        n_cooperative_network_sites: class_counts.cooperative_network,
        n_nma_responsive_sites: class_counts.nma_responsive,
        n_thermal_transient_sites: class_counts.thermal_transient,
        n_preformed_stable_sites: class_counts.preformed_stable,
        ccf_distant_pair_mean: ccf_stats.distant_pair_mean,
        ccf_max_offdiag: ccf_stats.max_offdiag,
        n_candidate_residues: candidate_residues.len(),
        n_ca_atoms: n_res,
        elapsed_seconds: t0.elapsed().as_secs_f64(),
        files_written: files_written
            .iter()
            .map(|p| {
                Path::new(p)
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default()
            })
            .collect(),
    };

    log::info!(
        "  Twin detection: {} sites in {:.1}s",
        sites.len(),
        summary.elapsed_seconds
    );

    let _ = nma_modes_path; // Reserved for future per-mode attribution (Gate 4)

    Ok(summary)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal types
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TwinSiteClass {
    ConsensusCryptic,
    BarrierGated,
    AllostericHub,
    CooperativeNetwork,
    NmaResponsive,
    ThermalTransient,
    PreformedStable,
}

impl TwinSiteClass {
    fn as_str(self) -> &'static str {
        match self {
            Self::ConsensusCryptic => "CONSENSUS_CRYPTIC",
            Self::BarrierGated => "BARRIER_GATED",
            Self::AllostericHub => "ALLOSTERIC_HUB",
            Self::CooperativeNetwork => "COOPERATIVE_NETWORK",
            Self::NmaResponsive => "NMA_RESPONSIVE",
            Self::ThermalTransient => "THERMAL_TRANSIENT",
            Self::PreformedStable => "PREFORMED_STABLE",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwinSite {
    pub id: u32,
    pub rank: u32,
    pub classification: String,  // stringified TwinSiteClass
    pub twin_site_class: String, // same, redundant for compat
    pub centroid: [f32; 3],
    pub volume_angstrom3: f32, // convex hull estimate
    pub n_lining_residues: usize,
    pub residue_ids: Vec<i32>, // topology resids (0-based in topology space)
    pub lining_residues: Vec<LiningResidue>,

    // Raw totals (over both streams)
    pub spike_count: u64,
    pub spikes_a: u64,
    pub spikes_b: u64,

    // Twin signals (in [0,1] unless noted)
    pub consensus_fraction: f32, // fraction of lining residues active in BOTH streams
    pub differential_signal: f32, // fraction of lining residues active ONLY in B (NMA-exclusive)
    pub ccf_centrality: f32,     // mean CCF to residues outside the site
    pub ccf_allosteric_score: f32, // normalized count of strong distant CCF pairs
    pub hysteresis_asymmetry: f32, // |warm_hold_frac - cold_hold_frac|
    pub druggability: f32,       // heuristic from lining residue chemistry
    pub spike_density: f32,      // spikes per lining residue per 1000 steps, normalized

    // Phase structure
    pub warm_hold_fraction: f32,
    pub cold_hold_fraction: f32,
    pub heating_fraction: f32,
    pub cooling_fraction: f32,
    pub cold_return_fraction: f32,

    // Engine compat fields (read by postprocess_twin.py)
    pub quality_score: f32,            // alias for twin_rank_score, for compat
    pub druggability_score: f32,       // duplicate of druggability for compat
    pub classification_engine: String, // LIGSITE-like label: "Cryptic", "Druggable", etc
    pub therm_class: String,           // "CRYPTIC", "RESPONSIVE", "DYNAMIC", "INERT"
    pub aromatic_score: f32,

    // Twin ranking output
    pub twin_rank_score: f32,
    pub rank_components: RankComponents,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiningResidue {
    pub resid: i32,
    pub resname: String,
    pub min_distance_angstrom: f32,
    pub aa1: String,
    pub in_consensus: bool,
    pub in_differential: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankComponents {
    pub consensus_component: f32,
    pub differential_component: f32,
    pub allosteric_component: f32,
    pub druggability_component: f32,
    pub hysteresis_component: f32,
    pub spike_density_component: f32,
    pub phase_quality_component: f32,
    /// Multiplicative boost (1.0-2.0) for twin-validated sites.
    /// High when BOTH streams agree AND the signal is phase-enriched.
    pub interferometric_multiplier: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwinPerResidue {
    pub resid: i32,
    pub resname: String,
    pub aa1: String,
    pub ca_xyz: [f32; 3],

    // Consensus / differential
    pub spikes_a: u32,
    pub spikes_b: u32,
    pub total_spikes: u32,
    pub consensus_score: f32,
    pub differential_score: f32,
    pub b_over_a_ratio: f32,

    // CCF
    pub ccf_mean: f32,
    pub ccf_max: f32,
    pub ccf_n_distant_above_0p5: u32,
    pub ccf_centrality: f32,

    // Phase structure
    pub phase_cold_hold: u32,
    pub phase_heating: u32,
    pub phase_warm_hold: u32,
    pub phase_cooling: u32,
    pub phase_cold_return: u32,
    pub warm_hold_fraction: f32,

    // Barrier
    pub barrier_class: String, // LOW / MEDIUM / HIGH / UNKNOWN
}

struct CcfStats {
    distant_pair_mean: f32,
    max_offdiag: f32,
}

struct ClassTally {
    consensus_cryptic: usize,
    barrier_gated: usize,
    allosteric_hub: usize,
    cooperative_network: usize,
    nma_responsive: usize,
    thermal_transient: usize,
    preformed_stable: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Structure helpers
// ─────────────────────────────────────────────────────────────────────────────

fn extract_ca_positions(topo: &PrismPrepTopology) -> Vec<[f32; 3]> {
    topo.ca_indices
        .iter()
        .map(|&ai| {
            let base = ai * 3;
            [
                topo.positions[base],
                topo.positions[base + 1],
                topo.positions[base + 2],
            ]
        })
        .collect()
}

#[derive(Debug, Clone)]
struct CaSpatialIndex {
    cell_size: f32,
    cutoff2: f32,
    cells: HashMap<(i32, i32, i32), Vec<usize>>,
}

impl CaSpatialIndex {
    fn new(ca_xyz: &[[f32; 3]], cutoff: f32) -> Self {
        let cell_size = cutoff.max(1.0);
        let cutoff2 = cutoff * cutoff;
        let mut cells: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
        for (i, &ca) in ca_xyz.iter().enumerate() {
            let key = Self::cell_for(cell_size, ca);
            cells.entry(key).or_default().push(i);
        }
        Self {
            cell_size,
            cutoff2,
            cells,
        }
    }

    fn occupied_cells(&self) -> usize {
        self.cells.len()
    }

    fn cell(&self, pos: [f32; 3]) -> (i32, i32, i32) {
        Self::cell_for(self.cell_size, pos)
    }

    fn cell_for(cell_size: f32, pos: [f32; 3]) -> (i32, i32, i32) {
        (
            (pos[0] / cell_size).floor() as i32,
            (pos[1] / cell_size).floor() as i32,
            (pos[2] / cell_size).floor() as i32,
        )
    }

    fn nearest(&self, pos: [f32; 3], ca_xyz: &[[f32; 3]]) -> Option<usize> {
        let (cx, cy, cz) = self.cell(pos);
        let mut best_idx: Option<usize> = None;
        let mut best_d2 = self.cutoff2;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let key = (cx + dx, cy + dy, cz + dz);
                    let Some(indices) = self.cells.get(&key) else {
                        continue;
                    };
                    for &i in indices {
                        let ca = ca_xyz[i];
                        let ddx = pos[0] - ca[0];
                        let ddy = pos[1] - ca[1];
                        let ddz = pos[2] - ca[2];
                        let d2 = ddx * ddx + ddy * ddy + ddz * ddz;
                        let replace = match best_idx {
                            None => d2 < best_d2,
                            Some(best_i) => d2 < best_d2 || (d2 == best_d2 && i < best_i),
                        };
                        if replace {
                            best_d2 = d2;
                            best_idx = Some(i);
                        }
                    }
                }
            }
        }
        best_idx
    }
}

fn assign_spikes_to_residues_and_phases(
    spikes: &[GpuSpikeEvent],
    ca_xyz: &[[f32; 3]],
    ca_index: &CaSpatialIndex,
    max_ts: f32,
) -> (Vec<u32>, [Vec<u32>; 5]) {
    let n_res = ca_xyz.len();
    let mut counts = vec![0u32; n_res];
    let mut phase_counts: [Vec<u32>; 5] = [
        vec![0; n_res],
        vec![0; n_res],
        vec![0; n_res],
        vec![0; n_res],
        vec![0; n_res],
    ];
    if n_res == 0 || max_ts <= 0.0 {
        return (counts, phase_counts);
    }
    for s in spikes {
        let pos = s.position;
        let ts_f = s.timestep as f32;
        let frac = ts_f / max_ts;
        let phase_idx = PHASE_BOUNDS_FAST_HYSTERESIS
            .iter()
            .position(|&(lo, hi, _)| frac >= lo && frac < hi)
            .unwrap_or(4);
        if let Some(best_idx) = ca_index.nearest(pos, ca_xyz) {
            counts[best_idx] += 1;
            phase_counts[phase_idx][best_idx] += 1;
        }
    }
    (counts, phase_counts)
}

// ─────────────────────────────────────────────────────────────────────────────
// LINK #1 — sequence-position residue alignment (two-system molecular contrast)
// ─────────────────────────────────────────────────────────────────────────────

/// Sequence-position residue alignment between the reference topology A and a
/// controlled-derivative topology B (holo / design variant).
///
/// Keys on `(chain_id, residue_id)` SEQUENCE POSITION — never `residue_name`. A
/// residue_name that differs at an aligned position is a design mutation and is
/// ACCEPTED: that difference IS the molecular-contrast signal, never a rejection
/// reason. The shared protein frame is `[0, n_ref_residues)` sized by topology A.
/// Residues `[n_ref_residues, n_b)` present only in B are B-only (ligand / ion /
/// appended) and feed the ligand-contact localization channel.
///
/// TWO maps are built because the two analysis paths index different spaces and
/// diverge whenever the topology carries any non-CA residue (a holo ligand):
/// - `residue_a_to_b` — residue-ordinal space (Path-2: nearby_residues,
///   n_residues incl. ligand).
/// - `ca_a_to_b` — CA-ordinal space (Path-1: ca_xyz, protein-only).
///
/// `build` HARD-ERRORS (returns `Err`) on any protein-frame mismatch — missing,
/// reordered, or count-changed protein residues — never a silent `.min()`
/// truncation.
#[derive(Debug, Clone)]
pub struct ResidueAlignment {
    /// Shared reference residue count (residue-ordinal space) = `A.n_residues`.
    pub n_ref_residues: usize,
    /// Shared reference CA count (CA-ordinal space) = `A.ca_indices.len()`.
    pub n_ref_ca: usize,
    /// Total residue count of B (always `>= n_ref_residues`).
    pub n_b_residues: usize,
    /// Path-2 residue-ordinal map: A residue-ordinal → B residue-ordinal.
    /// Length `n_ref_residues`; `Some` for every shared residue (a controlled
    /// derivative guarantees a counterpart), position-for-position.
    pub residue_a_to_b: Vec<Option<usize>>,
    /// Path-1 CA-ordinal map: A CA-ordinal → B CA-ordinal. Length `n_ref_ca`.
    /// `None` only if B carries no CA for that shared residue (should not happen
    /// for a controlled derivative).
    pub ca_a_to_b: Vec<Option<usize>>,
    /// B-only residue ordinals `[n_ref_residues, n_b)`: ligand / ion / appended.
    pub b_only_residues: Vec<usize>,
    /// Count of aligned positions whose `residue_name` differs (design
    /// mutations). Informational — NEVER a rejection reason.
    pub n_mutations: usize,
}

impl ResidueAlignment {
    /// Build the alignment from reference topology A and controlled-derivative B.
    ///
    /// Errors (hard) when B does not share A's protein `(chain_id, residue_id)`
    /// sequence over `[0, n_ref)`: count change, chain reorder, or residue_id
    /// mismatch. `residue_name` differences are ACCEPTED (design-mutation signal).
    pub fn build(topology_a: &PrismPrepTopology, topology_b: &PrismPrepTopology) -> Result<Self> {
        let n_ref = topology_a.n_residues;
        let n_b = topology_b.n_residues;

        // B must contain at least the whole shared protein frame; fewer residues
        // means the protein frame was truncated (missing loop / count change).
        if n_b < n_ref {
            anyhow::bail!(
                "ResidueAlignment: topology_b has {} residues < reference {} — protein frame \
                 truncated (missing / count-changed residues). topology_b must be a controlled \
                 derivative sharing A's protein frame, differing only by appended ligand/ion \
                 residues and residue_name mutations",
                n_b,
                n_ref
            );
        }

        // Per-residue (chain_id, residue_id, residue_name) descriptors keyed by
        // residue ordinal, reconstructed from the always-present atom-indexed
        // arrays (residues[] metadata is #[serde(default)] and may be empty).
        let desc_a = per_residue_descriptor(topology_a)?;
        let desc_b = per_residue_descriptor(topology_b)?;

        // Validate the shared frame position-for-position on (chain_id,
        // residue_id). residue_name is NOT compared — it is allowed to differ.
        let mut n_mutations = 0usize;
        for r in 0..n_ref {
            let (chain_a, rid_a, name_a) = &desc_a[r];
            let (chain_b, rid_b, name_b) = &desc_b[r];
            if chain_a != chain_b || rid_a != rid_b {
                anyhow::bail!(
                    "ResidueAlignment: protein-frame mismatch at ordinal {} — \
                     A=(chain={}, residue_id={}) B=(chain={}, residue_id={}). topology_b must \
                     share A's (chain_id, residue_id) sequence; reorder / renumber is rejected \
                     (mutations in residue_name are allowed)",
                    r,
                    chain_a,
                    rid_a,
                    chain_b,
                    rid_b
                );
            }
            if name_a != name_b {
                n_mutations += 1; // design mutation — accepted, counted only
            }
        }

        // Residue-ordinal map: index-for-index identity over the shared range.
        let residue_a_to_b: Vec<Option<usize>> = (0..n_ref).map(Some).collect();

        // B-only residues [n_ref, n_b): ligand / ion / appended.
        let b_only_residues: Vec<usize> = (n_ref..n_b).collect();

        // CA-ordinal map: for each A CA-ordinal take its residue ordinal, map it
        // through the (identity) residue map, then locate B's CA-ordinal for that
        // residue. A ligand residue in B has no CA, so it never appears here.
        let ca_res_a: Vec<usize> = topology_a
            .ca_indices
            .iter()
            .map(|&ai| topology_a.residue_ids[ai])
            .collect();
        let mut b_res_to_ca: HashMap<usize, usize> = HashMap::new();
        for (k, &ai) in topology_b.ca_indices.iter().enumerate() {
            b_res_to_ca.insert(topology_b.residue_ids[ai], k);
        }
        let ca_a_to_b: Vec<Option<usize>> = ca_res_a
            .iter()
            .map(|&ra| {
                residue_a_to_b
                    .get(ra)
                    .copied()
                    .flatten()
                    .and_then(|rb| b_res_to_ca.get(&rb).copied())
            })
            .collect();

        Ok(Self {
            n_ref_residues: n_ref,
            n_ref_ca: ca_res_a.len(),
            n_b_residues: n_b,
            residue_a_to_b,
            ca_a_to_b,
            b_only_residues,
            n_mutations,
        })
    }
}

/// Per-residue `(chain_id, residue_id, residue_name)` keyed by residue ordinal,
/// reconstructed from the always-present atom-indexed arrays (first atom of each
/// residue wins). Errors if a residue ordinal in `[0, n_residues)` has no atom
/// (malformed topology).
fn per_residue_descriptor(topo: &PrismPrepTopology) -> Result<Vec<(String, i32, String)>> {
    let mut out: Vec<Option<(String, i32, String)>> = vec![None; topo.n_residues];
    for atom_idx in 0..topo.n_atoms {
        let rid = topo.residue_ids[atom_idx];
        if rid < topo.n_residues && out[rid].is_none() {
            out[rid] = Some((
                topo.chain_ids[atom_idx].clone(),
                rid as i32,
                topo.residue_names[atom_idx].clone(),
            ));
        }
    }
    out.into_iter()
        .enumerate()
        .map(|(r, d)| {
            d.ok_or_else(|| {
                anyhow::anyhow!(
                    "per_residue_descriptor: residue ordinal {} has no atoms in topology \
                     (malformed)",
                    r
                )
            })
        })
        .collect()
}

/// LINK #1 Path-1 projection: scatter a B-native CA-ordinal count vector into the
/// reference (A) CA-ordinal frame. Output length is EXACTLY `n_ref_ca`; entries
/// are 0 where the reference CA has no B counterpart (`ca_a_to_b[k_a] == None`).
/// This is what guarantees counts_b_ref.len() == counts_a.len() == n_res so no
/// downstream A-ordinal index goes out of bounds.
fn project_ca_ordinal(native_b: &[u32], ca_a_to_b: &[Option<usize>], n_ref_ca: usize) -> Vec<u32> {
    let mut out = vec![0u32; n_ref_ca];
    for (k_a, slot) in out.iter_mut().enumerate() {
        if let Some(Some(k_b)) = ca_a_to_b.get(k_a) {
            if let Some(&v) = native_b.get(*k_b) {
                *slot = v;
            }
        }
    }
    out
}

/// LINK #1 ligand-contact localization channel (decision 4). Collects the
/// stream-B spikes Path-1 drops — those whose PRIMARY nearby residue is a B-only
/// (ligand / ion / appended) residue at ordinal `>= N_ref` — so they are never
/// snapped onto pocket-lining CAs. It localizes WHERE the ligand sits on the
/// protein by tallying, per dropped spike, the PROTEIN residues (`< N_ref`) that
/// appear in its `nearby_residues` set, and keeps the raw drop counter. This is
/// a QUALITATIVE perturbation-localization channel, NOT a binding affinity / ΔG.
#[derive(Default)]
struct LigandContactChannel {
    n_ref_residues: usize,
    /// Total dropped ligand-region spikes (the raw drop counter).
    spike_count: u64,
    /// Protein residue ordinal (`< N_ref`, resides in topology A) → contact count.
    contact_residues: std::collections::BTreeMap<i32, u64>,
    /// B-only ligand/ion residue ordinal (`>= N_ref`, resides in topology B) →
    /// spike count.
    ligand_residues: std::collections::BTreeMap<i32, u64>,
}

impl LigandContactChannel {
    fn new(n_ref_residues: usize) -> Self {
        Self {
            n_ref_residues,
            ..Default::default()
        }
    }

    /// Record one dropped ligand-region spike. `n_ref` is `N_ref` as `i32`.
    fn record(&mut self, s: &GpuSpikeEvent, n_ref: i32) {
        self.spike_count += 1;
        let n = s.n_residues.clamp(0, 8) as usize;
        for r in 0..n {
            let resid = s.nearby_residues[r];
            if resid < 0 {
                continue;
            }
            if resid >= n_ref {
                *self.ligand_residues.entry(resid).or_insert(0) += 1;
            } else {
                *self.contact_residues.entry(resid).or_insert(0) += 1;
            }
        }
    }

    /// Write the ligand-contact channel JSON. Protein residue names resolve
    /// against topology A (`< N_ref`); ligand residue names against topology B
    /// (`>= N_ref`). Carries the machine-readable qualitative caveat.
    fn write_json(
        &self,
        path: &Path,
        topology_a: &PrismPrepTopology,
        topology_b: &PrismPrepTopology,
    ) -> Result<()> {
        let desc_a = per_residue_descriptor(topology_a)?;
        let desc_b = per_residue_descriptor(topology_b)?;
        let name_of = |resid: i32, desc: &[(String, i32, String)]| -> String {
            usize::try_from(resid)
                .ok()
                .and_then(|r| desc.get(r))
                .map(|(_, _, name)| name.clone())
                .unwrap_or_else(|| "UNK".to_string())
        };

        let contacts: Vec<serde_json::Value> = self
            .contact_residues
            .iter()
            .map(|(&resid, &count)| {
                serde_json::json!({
                    "residue_id": resid,
                    "residue_name": name_of(resid, &desc_a),
                    "contact_spike_count": count,
                })
            })
            .collect();
        let ligands: Vec<serde_json::Value> = self
            .ligand_residues
            .iter()
            .map(|(&resid, &count)| {
                serde_json::json!({
                    "residue_id": resid,
                    "residue_name": name_of(resid, &desc_b),
                    "ligand_spike_count": count,
                })
            })
            .collect();

        let doc = serde_json::json!({
            "channel": "twin_ligand_contact_localization",
            "quantitative": false,
            "interpretation": "qualitative_perturbation_localization_not_binding_affinity_not_dG",
            "differential_sign": "B_excess_over_A",
            "n_ref_residues": self.n_ref_residues,
            "ligand_region_spike_count": self.spike_count,
            "protein_contact_residues": contacts,
            "ligand_residues": ligands,
        });
        std::fs::write(path, serde_json::to_string_pretty(&doc)?)?;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Consensus / differential / CCF
// ─────────────────────────────────────────────────────────────────────────────

fn compute_consensus_vector(counts_a: &[u32], counts_b: &[u32], two_system: bool) -> Vec<f32> {
    // LINK #1: in two-system (molecular-contrast) mode the B counts MUST have
    // been projected into the reference CA-ordinal frame, i.e. exactly A's
    // length — a silent `.min()` truncation would drop the tail of the longer
    // vector and corrupt the differential. Single-topology mode always produces
    // equal-length counts (both assigned against the one ca_xyz), so this assert
    // never fires on the no-flag path.
    if two_system {
        assert_eq!(
            counts_a.len(),
            counts_b.len(),
            "LINK #1 two-system consensus requires B counts projected into the reference \
             CA-ordinal frame (equal length); got A={} B={}",
            counts_a.len(),
            counts_b.len()
        );
    }
    let n = counts_a.len().min(counts_b.len());
    let mut out = vec![0.0; n];
    for i in 0..n {
        let a = counts_a[i] as f32;
        let b = counts_b[i] as f32;
        if a > 0.0 && b > 0.0 {
            let mn = a.min(b);
            let mx = a.max(b);
            // Consensus = (geometric mean of counts) * (agreement ratio)
            out[i] = (a * b).sqrt() * (mn / mx);
        }
    }
    out
}

fn compute_differential_vector(counts_a: &[u32], counts_b: &[u32], two_system: bool) -> Vec<f32> {
    // LINK #1: see compute_consensus_vector — B counts must be reference-frame
    // projected (equal length) in two-system mode; no-flag path is always equal.
    if two_system {
        assert_eq!(
            counts_a.len(),
            counts_b.len(),
            "LINK #1 two-system differential requires B counts projected into the reference \
             CA-ordinal frame (equal length); got A={} B={}",
            counts_a.len(),
            counts_b.len()
        );
    }
    let n = counts_a.len().min(counts_b.len());
    let mut out = vec![0.0; n];
    for i in 0..n {
        let a = counts_a[i] as f32;
        let b = counts_b[i] as f32;
        // Differential = Stream B excess over Stream A (NMA-exclusive signal)
        if b > a {
            // A small a with large b → large differential.
            // A large a with slightly larger b → small differential.
            let excess = b - a;
            let normalization = 1.0 + a; // penalize sites that are already active in A
            out[i] = excess / normalization;
        }
    }
    out
}

fn compute_ccf_matrix(
    spikes_a: &[GpuSpikeEvent],
    spikes_b: &[GpuSpikeEvent],
    ca_xyz: &[[f32; 3]],
    n_res: usize,
    // LINK #1 Path-1 CCF second frame. `None` ⇒ single-topology: stream B is
    // accumulated on the SAME ca_xyz(A) exactly as before (byte-identical).
    // `Some((ca_xyz_b, ca_a_to_b))` ⇒ two-system: stream B is accumulated on its
    // OWN CA frame `ca_xyz_b` then REMAPPED row-for-row into the reference
    // CA-ordinal frame via `ca_a_to_b` (A CA-ordinal → B CA-ordinal) before the
    // A×B correlation. This never correlates raw positions across two
    // independently integrated coordinate frames.
    b_frame: Option<(&[[f32; 3]], &[Option<usize>])>,
) -> (Vec<Vec<f32>>, CcfStats) {
    // Time-binned per-residue spike matrices for A and B, then mean-center
    // along BOTH axes (remove the global CCNS protocol envelope AND the
    // per-residue mean), then unit-normalize rows, then correlate A with B.
    let mut mat_a = vec![vec![0.0f32; CCF_TIME_BINS]; n_res];
    let mut mat_b = vec![vec![0.0f32; CCF_TIME_BINS]; n_res];

    if n_res == 0 {
        let empty = vec![vec![0.0f32; 0]; 0];
        return (
            empty,
            CcfStats {
                distant_pair_mean: 0.0,
                max_offdiag: 0.0,
            },
        );
    }

    let max_ts = spikes_a
        .iter()
        .chain(spikes_b.iter())
        .map(|s| s.timestep)
        .max()
        .unwrap_or(1)
        .max(1) as f32;
    let bin_size = (max_ts / CCF_TIME_BINS as f32).max(1.0);

    // Downsample if needed
    let stride_a = (spikes_a.len() / CCF_SAMPLE_LIMIT).max(1);
    let stride_b = (spikes_b.len() / CCF_SAMPLE_LIMIT).max(1);

    let ca_index = CaSpatialIndex::new(ca_xyz, SPIKE_TO_CA_CUTOFF);
    // Accumulate against an EXPLICIT (index, frame) so the two-system path can
    // bin stream B on its own CA frame. Numerically identical to the historical
    // closure for the A frame.
    let accumulate = |spikes: &[GpuSpikeEvent],
                      stride: usize,
                      mat: &mut [Vec<f32>],
                      index: &CaSpatialIndex,
                      frame: &[[f32; 3]]| {
        for (k, s) in spikes.iter().enumerate() {
            if k % stride != 0 {
                continue;
            }
            let pos = s.position;
            if let Some(best_idx) = index.nearest(pos, frame) {
                let bin = ((s.timestep as f32 / bin_size) as usize).min(CCF_TIME_BINS - 1);
                mat[best_idx][bin] += 1.0;
            }
        }
    };
    accumulate(spikes_a, stride_a, &mut mat_a, &ca_index, ca_xyz);
    match b_frame {
        // Single-topology: stream B shares the reference frame (byte-identical).
        None => accumulate(spikes_b, stride_b, &mut mat_b, &ca_index, ca_xyz),
        // Two-system: bin B on its own frame, then remap rows into the reference
        // CA-ordinal frame via ca_a_to_b (0-row where B has no counterpart).
        Some((ca_xyz_b, ca_a_to_b)) => {
            let n_res_b = ca_xyz_b.len();
            let ca_index_b = CaSpatialIndex::new(ca_xyz_b, SPIKE_TO_CA_CUTOFF);
            let mut mat_b_native = vec![vec![0.0f32; CCF_TIME_BINS]; n_res_b];
            accumulate(spikes_b, stride_b, &mut mat_b_native, &ca_index_b, ca_xyz_b);
            for (k_a, slot) in ca_a_to_b.iter().enumerate().take(n_res) {
                if let Some(k_b) = slot {
                    if *k_b < n_res_b {
                        mat_b[k_a].clone_from(&mat_b_native[*k_b]);
                    }
                }
            }
        }
    }

    // Double-centering: subtract column mean (across residues, per time bin)
    // then row mean (across time, per residue). This kills the CCNS envelope.
    double_center(&mut mat_a);
    double_center(&mut mat_b);

    // Unit-normalize rows
    unit_normalize_rows(&mut mat_a);
    unit_normalize_rows(&mut mat_b);

    // CCF[i][j] = <mat_a[i], mat_b[j]> (dot product of unit-normalized rows)
    // Symmetrize: (A⋅B^T + B⋅A^T) / 2
    let mut ccf = vec![vec![0.0f32; n_res]; n_res];
    for i in 0..n_res {
        for j in 0..n_res {
            let mut dot_ab = 0.0f32;
            for t in 0..CCF_TIME_BINS {
                dot_ab += mat_a[i][t] * mat_b[j][t];
            }
            ccf[i][j] = dot_ab;
        }
    }
    // Symmetrize
    for i in 0..n_res {
        for j in (i + 1)..n_res {
            let avg = 0.5 * (ccf[i][j] + ccf[j][i]);
            ccf[i][j] = avg;
            ccf[j][i] = avg;
        }
    }
    // Set diagonal to self-correlation = 1 (unit-normalized)
    for i in 0..n_res {
        ccf[i][i] = 1.0;
    }

    // Stats
    let mut distant_sum = 0.0f64;
    let mut distant_n = 0u64;
    let mut max_offdiag = 0.0f32;
    for i in 0..n_res {
        for j in (i + 1)..n_res {
            let v = ccf[i][j];
            if v.abs() > max_offdiag.abs() {
                max_offdiag = v;
            }
            if (j - i) > ALLOSTERIC_MIN_SEPARATION {
                distant_sum += v as f64;
                distant_n += 1;
            }
        }
    }
    let distant_pair_mean = if distant_n > 0 {
        (distant_sum / distant_n as f64) as f32
    } else {
        0.0
    };

    (
        ccf,
        CcfStats {
            distant_pair_mean,
            max_offdiag,
        },
    )
}

fn double_center(mat: &mut [Vec<f32>]) {
    let n_res = mat.len();
    if n_res == 0 {
        return;
    }
    let n_bins = mat[0].len();
    if n_bins == 0 {
        return;
    }
    // Column mean (per time bin, across residues)
    let mut col_mean = vec![0.0f32; n_bins];
    for row in mat.iter() {
        for (t, v) in row.iter().enumerate() {
            col_mean[t] += *v;
        }
    }
    for v in col_mean.iter_mut() {
        *v /= n_res as f32;
    }
    for row in mat.iter_mut() {
        for (t, v) in row.iter_mut().enumerate() {
            *v -= col_mean[t];
        }
    }
    // Row mean (per residue)
    for row in mat.iter_mut() {
        let m: f32 = row.iter().copied().sum::<f32>() / n_bins as f32;
        for v in row.iter_mut() {
            *v -= m;
        }
    }
}

fn unit_normalize_rows(mat: &mut [Vec<f32>]) {
    for row in mat.iter_mut() {
        let norm_sq: f32 = row.iter().map(|v| v * v).sum();
        let norm = norm_sq.sqrt();
        if norm > 1e-12 {
            for v in row.iter_mut() {
                *v /= norm;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Candidate residues + clustering
// ─────────────────────────────────────────────────────────────────────────────

fn build_candidate_residues(
    consensus: &[f32],
    differential: &[f32],
    ccf: &[Vec<f32>],
) -> Vec<usize> {
    let n = consensus.len();
    if n == 0 {
        return Vec::new();
    }

    // Top 20% consensus residues
    let mut sorted_c: Vec<(usize, f32)> =
        consensus.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    sorted_c.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_c: Vec<usize> = sorted_c
        .iter()
        .take(n / 5 + 1)
        .filter(|&(_, v)| *v > 0.0)
        .map(|&(i, _)| i)
        .collect();

    // Top 20% differential residues
    let mut sorted_d: Vec<(usize, f32)> = differential
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    sorted_d.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_d: Vec<usize> = sorted_d
        .iter()
        .take(n / 5 + 1)
        .filter(|&(_, v)| *v > 0.0)
        .map(|&(i, _)| i)
        .collect();

    // Top 10% CCF centrality (mean off-diagonal CCF)
    let mut centrality: Vec<(usize, f32)> = (0..n)
        .map(|i| {
            let mut s = 0.0f32;
            for j in 0..n {
                if i != j {
                    s += ccf[i][j].abs();
                }
            }
            let mean = if n > 1 { s / (n as f32 - 1.0) } else { 0.0 };
            (i, mean)
        })
        .collect();
    centrality.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_ccf: Vec<usize> = centrality
        .iter()
        .take(n / 10 + 1)
        .filter(|&(_, v)| *v > 0.05)
        .map(|&(i, _)| i)
        .collect();

    let mut set: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    for i in &top_c {
        set.insert(*i);
    }
    for i in &top_d {
        set.insert(*i);
    }
    for i in &top_ccf {
        set.insert(*i);
    }
    set.into_iter().collect()
}

fn cluster_candidates(candidates: &[usize], ca_xyz: &[[f32; 3]]) -> Vec<Vec<usize>> {
    // Simple DBSCAN over candidate CA positions
    let n = candidates.len();
    let eps2 = CLUSTER_EPS_ANGSTROM * CLUSTER_EPS_ANGSTROM;
    let mut visited = vec![false; n];
    let mut cluster_id = vec![-1i32; n];
    let mut next_cluster = 0i32;

    let neighbors = |idx: usize| -> Vec<usize> {
        let mut neigh = Vec::new();
        let p1 = ca_xyz[candidates[idx]];
        for j in 0..n {
            if j == idx {
                continue;
            }
            let p2 = ca_xyz[candidates[j]];
            let dx = p1[0] - p2[0];
            let dy = p1[1] - p2[1];
            let dz = p1[2] - p2[2];
            if dx * dx + dy * dy + dz * dz <= eps2 {
                neigh.push(j);
            }
        }
        neigh
    };

    for i in 0..n {
        if visited[i] {
            continue;
        }
        visited[i] = true;
        let neigh = neighbors(i);
        if neigh.len() + 1 < CLUSTER_MIN_PTS {
            continue;
        }
        let cid = next_cluster;
        next_cluster += 1;
        cluster_id[i] = cid;
        let mut stack = neigh;
        while let Some(j) = stack.pop() {
            if !visited[j] {
                visited[j] = true;
                let nj = neighbors(j);
                if nj.len() + 1 >= CLUSTER_MIN_PTS {
                    for k in nj {
                        stack.push(k);
                    }
                }
            }
            if cluster_id[j] == -1 {
                cluster_id[j] = cid;
            }
        }
    }

    let mut clusters: Vec<Vec<usize>> = (0..next_cluster).map(|_| Vec::new()).collect();
    for (i, &cid) in cluster_id.iter().enumerate() {
        if cid >= 0 {
            clusters[cid as usize].push(candidates[i]);
        }
    }
    clusters.retain(|c| c.len() >= CLUSTER_MIN_PTS);
    clusters
}

// ─────────────────────────────────────────────────────────────────────────────
// Site construction
// ─────────────────────────────────────────────────────────────────────────────

fn build_twin_sites(
    raw_clusters: &[Vec<usize>],
    ca_xyz: &[[f32; 3]],
    topology: &PrismPrepTopology,
    counts_a: &[u32],
    counts_b: &[u32],
    phase_counts_a: &[Vec<u32>; 5],
    phase_counts_b: &[Vec<u32>; 5],
    consensus: &[f32],
    differential: &[f32],
    ccf: &[Vec<f32>],
) -> Vec<TwinSite> {
    let mut sites = Vec::with_capacity(raw_clusters.len());
    let n_res = ca_xyz.len();
    // Percentile thresholds so "in_consensus"/"in_differential" mean
    // "strongly consensus/differential" — not "had at least one spike"
    // (which is trivially true for most residues in a 30M spike run).
    let consensus_thresh = percentile_gt0(consensus, 0.75);
    let differential_thresh = percentile_gt0(differential, 0.75);

    for (ci, cluster) in raw_clusters.iter().enumerate() {
        if cluster.is_empty() {
            continue;
        }
        // Centroid
        let mut cx = 0.0f32;
        let mut cy = 0.0f32;
        let mut cz = 0.0f32;
        for &ri in cluster {
            cx += ca_xyz[ri][0];
            cy += ca_xyz[ri][1];
            cz += ca_xyz[ri][2];
        }
        let n_c = cluster.len() as f32;
        let centroid = [cx / n_c, cy / n_c, cz / n_c];

        // Lining residues: any CA within 8 Å of the centroid
        let mut lining: Vec<LiningResidue> = Vec::new();
        let mut lining_ids: Vec<i32> = Vec::new();
        let lining_cutoff2 = 8.0 * 8.0;
        for i in 0..n_res {
            let dx = ca_xyz[i][0] - centroid[0];
            let dy = ca_xyz[i][1] - centroid[1];
            let dz = ca_xyz[i][2] - centroid[2];
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 < lining_cutoff2 {
                let resname = topology
                    .residue_names
                    .get(topology.ca_indices[i])
                    .cloned()
                    .unwrap_or_else(|| "UNK".to_string());
                let aa1 = aa3_to_aa1(&resname);
                lining.push(LiningResidue {
                    resid: i as i32,
                    resname,
                    aa1,
                    min_distance_angstrom: d2.sqrt(),
                    in_consensus: consensus[i] > consensus_thresh,
                    in_differential: differential[i] > differential_thresh,
                });
                lining_ids.push(i as i32);
            }
        }

        // Spike totals over lining residues
        let mut sa: u64 = 0;
        let mut sb: u64 = 0;
        for lr in &lining {
            sa += counts_a[lr.resid as usize] as u64;
            sb += counts_b[lr.resid as usize] as u64;
        }
        let total = sa + sb;

        // Consensus / differential fractions over lining
        let n_consensus = lining.iter().filter(|r| r.in_consensus).count();
        let n_differential = lining.iter().filter(|r| r.in_differential).count();
        let consensus_fraction = if lining.is_empty() {
            0.0
        } else {
            n_consensus as f32 / lining.len() as f32
        };
        let differential_signal = if lining.is_empty() {
            0.0
        } else {
            n_differential as f32 / lining.len() as f32
        };

        // CCF centrality: mean |CCF| from lining residues to residues OUTSIDE the site
        let lining_set: std::collections::HashSet<usize> =
            lining.iter().map(|r| r.resid as usize).collect();
        let mut ccf_sum = 0.0f32;
        let mut ccf_n = 0u32;
        let mut strong_distant_pairs: u32 = 0;
        for &li in &lining_set {
            for j in 0..n_res {
                if lining_set.contains(&j) {
                    continue;
                }
                let v = ccf[li][j].abs();
                ccf_sum += v;
                ccf_n += 1;
                if v > 0.5
                    && (j as i32 - li as i32).unsigned_abs() as usize > ALLOSTERIC_MIN_SEPARATION
                {
                    strong_distant_pairs += 1;
                }
            }
        }
        let ccf_centrality = if ccf_n > 0 {
            ccf_sum / ccf_n as f32
        } else {
            0.0
        };
        let ccf_allosteric_score =
            (strong_distant_pairs as f32 / lining.len().max(1) as f32).min(1.0);

        // Phase fractions over lining residues
        let mut p_sum = [0u64; 5];
        for lr in &lining {
            for phase in 0..5 {
                p_sum[phase] += phase_counts_a[phase][lr.resid as usize] as u64;
                p_sum[phase] += phase_counts_b[phase][lr.resid as usize] as u64;
            }
        }
        let phase_total = p_sum.iter().sum::<u64>().max(1) as f32;
        let cold_hold_fraction = p_sum[0] as f32 / phase_total;
        let heating_fraction = p_sum[1] as f32 / phase_total;
        let warm_hold_fraction = p_sum[2] as f32 / phase_total;
        let cooling_fraction = p_sum[3] as f32 / phase_total;
        let cold_return_fraction = p_sum[4] as f32 / phase_total;
        let hysteresis_asymmetry = (warm_hold_fraction - cold_hold_fraction).abs();

        // Druggability heuristic from lining residue chemistry
        let druggability = simple_druggability(&lining);

        // Aromatic score
        let aromatic_score = lining.iter().filter(|r| is_aromatic(&r.aa1)).count() as f32
            / lining.len().max(1) as f32;

        // Spike density: (sa + sb) / n_lining, normalized to [0, 1] by a soft ceiling
        let raw_density = (total as f32 / lining.len().max(1) as f32) / 1000.0;
        let spike_density = (raw_density / (1.0 + raw_density)).min(1.0);

        // Convex-hull volume estimate (bounding sphere proxy — cheap)
        let mut rmax = 0.0f32;
        for lr in &lining {
            let ca = ca_xyz[lr.resid as usize];
            let dx = ca[0] - centroid[0];
            let dy = ca[1] - centroid[1];
            let dz = ca[2] - centroid[2];
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 > rmax {
                rmax = d2;
            }
        }
        let r = rmax.sqrt();
        let volume = (4.0 / 3.0) * std::f32::consts::PI * r * r * r * 0.5; // × 0.5 for pocket-like

        let site = TwinSite {
            id: ci as u32,
            rank: 0,                       // filled after ranking
            classification: String::new(), // filled after classify
            twin_site_class: String::new(),
            centroid,
            volume_angstrom3: volume,
            n_lining_residues: lining.len(),
            residue_ids: lining_ids,
            lining_residues: lining,
            spike_count: total,
            spikes_a: sa,
            spikes_b: sb,
            consensus_fraction,
            differential_signal,
            ccf_centrality,
            ccf_allosteric_score,
            hysteresis_asymmetry,
            druggability,
            spike_density,
            warm_hold_fraction,
            cold_hold_fraction,
            heating_fraction,
            cooling_fraction,
            cold_return_fraction,
            quality_score: 0.0, // alias of twin_rank_score, filled later
            druggability_score: druggability,
            classification_engine: String::new(),
            therm_class: String::new(),
            aromatic_score,
            twin_rank_score: 0.0,
            rank_components: RankComponents {
                consensus_component: 0.0,
                differential_component: 0.0,
                allosteric_component: 0.0,
                druggability_component: 0.0,
                hysteresis_component: 0.0,
                spike_density_component: 0.0,
                phase_quality_component: 0.0,
                interferometric_multiplier: 1.0,
            },
        };
        sites.push(site);
    }

    sites
}

/// Returns the p-th percentile of nonzero values in the vector (p in [0, 1]).
/// Falls back to 0.0 if the vector has no positive values.
fn percentile_gt0(values: &[f32], p: f32) -> f32 {
    let mut nonzero: Vec<f32> = values.iter().copied().filter(|&v| v > 0.0).collect();
    if nonzero.is_empty() {
        return 0.0;
    }
    nonzero.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((nonzero.len() as f32) * p).floor() as usize;
    nonzero[idx.min(nonzero.len() - 1)]
}

fn simple_druggability(lining: &[LiningResidue]) -> f32 {
    if lining.is_empty() {
        return 0.0;
    }
    let n = lining.len() as f32;
    let hydrophobic = lining
        .iter()
        .filter(|r| {
            matches!(
                r.aa1.as_str(),
                "A" | "V" | "L" | "I" | "M" | "F" | "W" | "Y" | "P" | "C"
            )
        })
        .count() as f32
        / n;
    let aromatic = lining.iter().filter(|r| is_aromatic(&r.aa1)).count() as f32 / n;
    let polar = lining
        .iter()
        .filter(|r| matches!(r.aa1.as_str(), "S" | "T" | "N" | "Q" | "H"))
        .count() as f32
        / n;
    // Simple heuristic: prefer hydrophobic + aromatic, modest polar
    (0.5 * hydrophobic + 0.3 * aromatic + 0.2 * polar).min(1.0)
}

fn is_aromatic(aa1: &str) -> bool {
    matches!(aa1, "F" | "W" | "Y" | "H")
}

fn aa3_to_aa1(resname: &str) -> String {
    match resname {
        "ALA" => "A",
        "ARG" => "R",
        "ASN" => "N",
        "ASP" | "ASH" => "D",
        "CYS" | "CYX" | "CYM" => "C",
        "GLN" => "Q",
        "GLU" | "GLH" => "E",
        "GLY" => "G",
        "HIS" | "HID" | "HIE" | "HIP" => "H",
        "ILE" => "I",
        "LEU" => "L",
        "LYS" | "LYN" => "K",
        "MET" => "M",
        "PHE" => "F",
        "PRO" => "P",
        "SER" => "S",
        "THR" => "T",
        "TRP" => "W",
        "TYR" => "Y",
        "VAL" => "V",
        _ => "X",
    }
    .to_string()
}

// ─────────────────────────────────────────────────────────────────────────────
// Classification + ranking
// ─────────────────────────────────────────────────────────────────────────────

fn classify_twin_site(site: &mut TwinSite) {
    // The classification order matters: earlier branches are more specific.
    let class = if site.consensus_fraction > 0.7 && site.hysteresis_asymmetry > 0.15 {
        TwinSiteClass::ConsensusCryptic
    } else if site.differential_signal > 0.5 && site.consensus_fraction < 0.4 {
        TwinSiteClass::BarrierGated
    } else if site.ccf_allosteric_score > 0.3 && site.ccf_centrality > 0.15 {
        TwinSiteClass::AllostericHub
    } else if site.differential_signal > 0.25 && site.ccf_allosteric_score > 0.1 {
        TwinSiteClass::NmaResponsive
    } else if site.consensus_fraction > 0.5 && site.cold_hold_fraction > 0.25 {
        TwinSiteClass::PreformedStable
    } else if site.ccf_centrality > 0.1 {
        TwinSiteClass::CooperativeNetwork
    } else {
        TwinSiteClass::ThermalTransient
    };

    let class_str = class.as_str().to_string();
    site.classification = class_str.clone();
    site.twin_site_class = class_str;

    // Compat fields for postprocess_twin.py (which expects LIGSITE-like labels)
    site.classification_engine = match class {
        TwinSiteClass::ConsensusCryptic => "Cryptic".to_string(),
        TwinSiteClass::BarrierGated => "Cryptic".to_string(),
        TwinSiteClass::AllostericHub => "Allosteric".to_string(),
        TwinSiteClass::CooperativeNetwork => "Allosteric".to_string(),
        TwinSiteClass::NmaResponsive => "Cryptic".to_string(),
        TwinSiteClass::PreformedStable => "Druggable".to_string(),
        TwinSiteClass::ThermalTransient => "Unknown".to_string(),
    };

    site.therm_class = match class {
        TwinSiteClass::ConsensusCryptic => "CRYPTIC",
        TwinSiteClass::BarrierGated => "DYNAMIC",
        TwinSiteClass::AllostericHub => "RESPONSIVE",
        TwinSiteClass::CooperativeNetwork => "RESPONSIVE",
        TwinSiteClass::NmaResponsive => "DYNAMIC",
        TwinSiteClass::PreformedStable => "INERT",
        TwinSiteClass::ThermalTransient => "INERT",
    }
    .to_string();
}

fn compute_twin_ranks(sites: &mut [TwinSite]) {
    // Compute max spike count across all sites for normalization
    let max_spikes = sites
        .iter()
        .map(|s| s.spike_count)
        .max()
        .unwrap_or(1)
        .max(1) as f32;

    for site in sites.iter_mut() {
        let c = RANK_W_CONSENSUS * site.consensus_fraction;
        let d = RANK_W_DIFFERENTIAL * site.differential_signal;
        let a = RANK_W_ALLOSTERIC * site.ccf_allosteric_score;
        let dr = RANK_W_DRUGGABILITY * site.druggability;
        let h = RANK_W_HYSTERESIS * site.hysteresis_asymmetry.min(1.0);
        let sd = RANK_W_SPIKE_DENSITY * site.spike_density;

        // Phase quality: reward sites enriched during warm_hold + heating
        // (where cryptic pockets open). Penalize cold-only signals.
        let warm_enrichment = (site.warm_hold_fraction + site.heating_fraction)
            - (site.cold_hold_fraction + site.cold_return_fraction);
        let pq = RANK_W_PHASE_QUALITY * (0.5 + warm_enrichment).clamp(0.0, 1.0);

        let base_score = c + d + a + dr + h + sd + pq;

        // ── Interferometric Multiplier ──
        // Boosts sites where BOTH twin streams independently produce strong,
        // correlated signal. Three factors:
        //
        // 1. Twin agreement: min(A,B)/max(A,B) — 1.0 when perfectly balanced
        // 2. Absolute evidence: spike_count/max_spikes — rewards high total signal
        // 3. CCF validation: ccf_centrality — rewards correlated dynamics
        //
        // Multiplier range: 1.0 (no boost) to 2.0 (maximum twin validation)
        let twin_agreement = if site.spikes_a.max(site.spikes_b) > 0 {
            site.spikes_a.min(site.spikes_b) as f32 / site.spikes_a.max(site.spikes_b) as f32
        } else {
            0.0
        };
        let evidence_strength = (site.spike_count as f32 / max_spikes).sqrt(); // sqrt to compress
        let ccf_validation = site.ccf_centrality.min(1.0);

        // Geometric mean of the three factors, scaled to [1.0, 2.0]
        let raw_multiplier = (twin_agreement * evidence_strength * ccf_validation).cbrt();
        let interferometric_multiplier = 1.0 + raw_multiplier.clamp(0.0, 1.0);

        let final_score = base_score * interferometric_multiplier;
        site.twin_rank_score = final_score;
        site.quality_score = final_score;

        site.rank_components = RankComponents {
            consensus_component: c,
            differential_component: d,
            allosteric_component: a,
            druggability_component: dr,
            hysteresis_component: h,
            spike_density_component: sd,
            phase_quality_component: pq,
            interferometric_multiplier,
        };
    }
}

fn tally_classifications(sites: &[TwinSite]) -> ClassTally {
    let mut t = ClassTally {
        consensus_cryptic: 0,
        barrier_gated: 0,
        allosteric_hub: 0,
        cooperative_network: 0,
        nma_responsive: 0,
        thermal_transient: 0,
        preformed_stable: 0,
    };
    for s in sites {
        match s.classification.as_str() {
            "CONSENSUS_CRYPTIC" => t.consensus_cryptic += 1,
            "BARRIER_GATED" => t.barrier_gated += 1,
            "ALLOSTERIC_HUB" => t.allosteric_hub += 1,
            "COOPERATIVE_NETWORK" => t.cooperative_network += 1,
            "NMA_RESPONSIVE" => t.nma_responsive += 1,
            "THERMAL_TRANSIENT" => t.thermal_transient += 1,
            "PREFORMED_STABLE" => t.preformed_stable += 1,
            _ => {}
        }
    }
    t
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-residue features
// ─────────────────────────────────────────────────────────────────────────────

fn build_per_residue_twin_features(
    ca_xyz: &[[f32; 3]],
    topology: &PrismPrepTopology,
    counts_a: &[u32],
    counts_b: &[u32],
    phase_counts_a: &[Vec<u32>; 5],
    phase_counts_b: &[Vec<u32>; 5],
    consensus: &[f32],
    differential: &[f32],
    ccf: &[Vec<f32>],
) -> Vec<TwinPerResidue> {
    let n = ca_xyz.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let sa = counts_a[i];
        let sb = counts_b[i];
        let total = sa + sb;
        let b_over_a = if sa > 0 {
            sb as f32 / sa as f32
        } else if sb > 0 {
            f32::INFINITY
        } else {
            0.0
        };

        let p_cold_hold = phase_counts_a[0][i] + phase_counts_b[0][i];
        let p_heating = phase_counts_a[1][i] + phase_counts_b[1][i];
        let p_warm_hold = phase_counts_a[2][i] + phase_counts_b[2][i];
        let p_cooling = phase_counts_a[3][i] + phase_counts_b[3][i];
        let p_cold_return = phase_counts_a[4][i] + phase_counts_b[4][i];
        let warm_hold_fraction = if total > 0 {
            p_warm_hold as f32 / total as f32
        } else {
            0.0
        };

        // CCF features
        let mut ccf_sum = 0.0f32;
        let mut ccf_max = 0.0f32;
        let mut n_distant_strong: u32 = 0;
        for j in 0..n {
            if i == j {
                continue;
            }
            let v = ccf[i][j];
            let abs_v = v.abs();
            ccf_sum += abs_v;
            if abs_v > ccf_max {
                ccf_max = abs_v;
            }
            if abs_v > 0.5
                && (j as i32 - i as i32).unsigned_abs() as usize > ALLOSTERIC_MIN_SEPARATION
            {
                n_distant_strong += 1;
            }
        }
        let ccf_mean = if n > 1 { ccf_sum / (n - 1) as f32 } else { 0.0 };
        let ccf_centrality = ccf_mean;

        let barrier_class = classify_barrier(sa, sb, warm_hold_fraction);
        let resname = topology
            .residue_names
            .get(topology.ca_indices[i])
            .cloned()
            .unwrap_or_else(|| "UNK".to_string());
        let aa1 = aa3_to_aa1(&resname);

        out.push(TwinPerResidue {
            resid: i as i32,
            resname,
            aa1,
            ca_xyz: ca_xyz[i],
            spikes_a: sa,
            spikes_b: sb,
            total_spikes: total,
            consensus_score: consensus[i],
            differential_score: differential[i],
            b_over_a_ratio: b_over_a,
            ccf_mean,
            ccf_max,
            ccf_n_distant_above_0p5: n_distant_strong,
            ccf_centrality,
            phase_cold_hold: p_cold_hold,
            phase_heating: p_heating,
            phase_warm_hold: p_warm_hold,
            phase_cooling: p_cooling,
            phase_cold_return: p_cold_return,
            warm_hold_fraction,
            barrier_class,
        });
    }
    out
}

fn classify_barrier(sa: u32, sb: u32, warm_hold_frac: f32) -> String {
    if sa == 0 && sb == 0 {
        return "UNKNOWN".to_string();
    }
    let b_over_a = if sa > 0 {
        sb as f32 / sa as f32
    } else {
        f32::INFINITY
    };
    if b_over_a > 1.5 && warm_hold_frac > 0.4 {
        "LOW".to_string()
    } else if b_over_a < 0.67 {
        "HIGH".to_string()
    } else {
        "MEDIUM".to_string()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Output writers — JSON formats compatible with postprocess_twin.py
// ─────────────────────────────────────────────────────────────────────────────

/// LINK #1: machine-readable "NOT a binding affinity / dG" caveat inserted at the root of every
/// two-system (apo-vs-holo / design-vs-reference) emission, so a qualitative
/// perturbation-localization differential can never be read as a validated binding score. Gated on
/// `two_system` at the call sites, so single-topology (no --topology-b) output is byte-identical.
fn insert_two_system_caveat(root: &mut serde_json::Value) {
    if let Some(obj) = root.as_object_mut() {
        obj.insert(
            "contrast".into(),
            serde_json::json!({
                "quantitative": false,
                "interpretation": "qualitative_perturbation_localization_not_binding_affinity_not_dG",
                "differential_sign": "B_excess_over_A",
                "note": "Two-system TWIN contrast (--topology-b): where engine_b's (holo/designed) \
                         spike dynamics diverge from engine_a's (apo/reference). druggability / \
                         druggability_score / twin_rank_score are heuristic pocket descriptors, NOT a \
                         validated binding affinity or dG — do NOT use them to gate wet-lab spend.",
            }),
        );
    }
}

fn write_binding_sites_json(
    path: &Path,
    sites: &[TwinSite],
    topology: &PrismPrepTopology,
    two_system: bool,
) -> Result<()> {
    // Mirror the non-twin binding_sites.json schema that postprocess_twin.py consumes,
    // with additional twin_* fields appended to each site record.
    let sites_json: Vec<serde_json::Value> = sites
        .iter()
        .map(|s| {
            let lining_resids: Vec<i32> = s.lining_residues.iter().map(|r| r.resid).collect();
            let lining_for_json: Vec<serde_json::Value> = s
                .lining_residues
                .iter()
                .map(|lr| {
                    serde_json::json!({
                        "resid": lr.resid,
                        "resname": lr.resname,
                        "chain": "A",
                        "is_catalytic": false,
                        "min_distance": lr.min_distance_angstrom,
                        "n_atoms": 8,  // placeholder; postprocess_twin.py doesn't need exact value
                    })
                })
                .collect();

            serde_json::json!({
                "id": s.id,
                "rank": s.rank,
                "rank_score": s.twin_rank_score,   // alias expected by prism-postflight.py
                "centroid": s.centroid,
                "volume": s.volume_angstrom3,
                "classification": s.classification_engine,
                "therm_class": s.therm_class,
                "quality_score": s.quality_score,
                "druggability": s.druggability_score,
                "is_druggable": s.druggability_score > 0.4,
                "aromatic_score": s.aromatic_score,
                "spike_count": s.spike_count,
                "residue_ids": lining_resids,
                "lining_residues": lining_for_json,
                "ccns_tau": 0.0,
                "hysteresis_asymmetry": s.hysteresis_asymmetry,
                "relative_asymmetry": s.cold_return_fraction - s.cold_hold_fraction,
                "catalytic_residue_count": 0,
                // TWIN-exclusive fields
                "twin_site_class": s.twin_site_class,
                "twin_rank_score": s.twin_rank_score,
                "twin_consensus_fraction": s.consensus_fraction,
                "twin_differential_signal": s.differential_signal,
                "twin_ccf_centrality": s.ccf_centrality,
                "twin_ccf_allosteric_score": s.ccf_allosteric_score,
                "twin_spike_density": s.spike_density,
                "twin_spikes_a": s.spikes_a,
                "twin_spikes_b": s.spikes_b,
                "twin_phase_warm_hold_fraction": s.warm_hold_fraction,
                "twin_phase_cold_hold_fraction": s.cold_hold_fraction,
                "twin_rank_components": s.rank_components,
            })
        })
        .collect();

    // cryptic_sites and all_pockets MUST be lists (not ints) — prism-postflight.py
    // iterates over them and takes len(), so emitting counts here crashes the wrapper.
    let cryptic_sites_list: Vec<serde_json::Value> = sites
        .iter()
        .filter(|s| s.therm_class == "CRYPTIC" || s.therm_class == "DYNAMIC")
        .map(|s| {
            serde_json::json!({
                "id": s.id,
                "centroid": s.centroid,
                "classification": s.classification_engine,
                "twin_site_class": s.twin_site_class,
            })
        })
        .collect();

    let mut root = serde_json::json!({
        "mode": "prism-twin",
        "detection_method": "twin-aware: consensus + differential + CCF + phase",
        "n_streams": 2,
        "structure": {
            "n_residues": topology.n_residues,
            "n_atoms": topology.n_atoms,
        },
        "sites": sites_json,
        "binding_sites": sites.len(),
        "druggable_sites": sites.iter().filter(|s| s.druggability_score > 0.4).count(),
        "cryptic_sites": cryptic_sites_list,
        "all_pockets": sites_json.iter().map(|s| serde_json::json!({
            "id": s.get("id"),
            "centroid": s.get("centroid"),
        })).collect::<Vec<_>>(),
    });

    if two_system {
        insert_two_system_caveat(&mut root);
    }
    std::fs::write(path, serde_json::to_string_pretty(&root)?)?;
    Ok(())
}

fn write_kcc_json(
    path: &Path,
    sites: &[TwinSite],
    ccf: &[Vec<f32>],
    topology: &PrismPrepTopology,
    two_system: bool,
) -> Result<()> {
    // KCC visualization from CCF: top CCF pairs are "causally coupled".
    let n = ccf.len();
    let mut edges: Vec<(usize, usize, f32)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let v = ccf[i][j].abs();
            if v > 0.5 && (j - i) > ALLOSTERIC_MIN_SEPARATION {
                edges.push((i, j, v));
            }
        }
    }
    edges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    edges.truncate(500);

    let mut residue_rows: Vec<serde_json::Value> = Vec::with_capacity(n);
    for i in 0..n {
        let resname = topology
            .residue_names
            .get(topology.ca_indices[i])
            .cloned()
            .unwrap_or_else(|| "UNK".to_string());
        let degree = edges.iter().filter(|(a, b, _)| *a == i || *b == i).count();
        residue_rows.push(serde_json::json!({
            "resid": i,
            "resname": resname,
            "ca_position": [0.0, 0.0, 0.0],  // not used by postprocess_twin.py
            "degree": degree,
            "ccf_centrality": degree as f32 / n.max(1) as f32,
        }));
    }

    let sites_rows: Vec<serde_json::Value> = sites
        .iter()
        .map(|s| {
            serde_json::json!({
                "id": s.id,
                "centroid": s.centroid,
                "gtck_rank": s.rank,
                "rank_score": s.twin_rank_score,
                "kcc_confidence": s.ccf_centrality,
                "driver_residue_topo": s.residue_ids.first().copied().unwrap_or(-1),
                "site_causal_lag_steps": 0.0,
                "site_burst_motion": s.differential_signal,
                "site_lag_corr_peak": s.ccf_centrality,
                "site_local_cov": s.consensus_fraction,
            })
        })
        .collect();

    let mut root = serde_json::json!({
        "mode": "prism-twin",
        "semantics": {
            "kcc": "Cross-stream CCF from TWIN interferometric observation",
            "vector": "Allosteric coupling direction",
            "color_mapping": "CCF magnitude",
            "weight": "rank by TWIN score",
            "site": "TWIN-detected pocket",
        },
        "interpretation_guidelines": {
            "persistent_site": "high consensus_fraction",
            "transient_site": "THERMAL_TRANSIENT class",
            "high_value_residue": "high ccf_centrality",
            "noise_pattern": "low consensus with differential signal",
        },
        "residues": residue_rows,
        "sites": sites_rows,
        "vector_field_definition": "TWIN CCF-derived coupling",
        "pdb_source": "from topology",
        "ccf_edges_top500": edges.iter().map(|(i, j, v)| serde_json::json!([i, j, v])).collect::<Vec<_>>(),
    });

    if two_system {
        insert_two_system_caveat(&mut root);
    }
    std::fs::write(path, serde_json::to_string_pretty(&root)?)?;
    Ok(())
}

fn write_prism_therm_json(path: &Path, sites: &[TwinSite], two_system: bool) -> Result<()> {
    let sites_rows: Vec<serde_json::Value> = sites
        .iter()
        .map(|s| {
            serde_json::json!({
                "centroid_angstrom": s.centroid,
                "asymmetry_score": s.hysteresis_asymmetry,
                "ccns_classification": s.therm_class,
                "druggability": s.druggability,
                "cooling_spike_count": 0,
                "cooling_spike_rate": 0.0,
                "heating_spike_count": 0,
                "heating_spike_rate": 0.0,
                "twin_site_class": s.twin_site_class,
                "twin_rank_score": s.twin_rank_score,
                "tide_trigger_residues": s.residue_ids.iter().take(5).copied().collect::<Vec<_>>(),
            })
        })
        .collect();

    let hysteretic_sites = sites
        .iter()
        .filter(|s| s.hysteresis_asymmetry > 0.15)
        .count();

    let mut root = serde_json::json!({
        "mode": "prism-twin",
        "source": "TWIN phase-resolved spike counts per stream per residue",
        "global_pockets": [],
        "hysteresis_threshold": 0.15,
        "hysteretic_site_count": hysteretic_sites,
        "sdst_event_count": sites.iter().map(|s| s.spike_count).sum::<u64>(),
        "total_avalanches": 0,
        "sites": sites_rows,
        "tide_residues_mapped": sites.iter().flat_map(|s| s.residue_ids.iter().take(3).copied()).collect::<Vec<_>>(),
    });

    if two_system {
        insert_two_system_caveat(&mut root);
    }
    std::fs::write(path, serde_json::to_string_pretty(&root)?)?;
    Ok(())
}

// Silence unused imports helper
#[allow(dead_code)]
fn _silence() -> HashMap<String, String> {
    HashMap::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spike(timestep: i32, position: [f32; 3]) -> GpuSpikeEvent {
        GpuSpikeEvent {
            timestep,
            position,
            ..GpuSpikeEvent::default()
        }
    }

    // LINK #1 — build a minimal topology from a residue list. Each residue is
    // (chain_id, residue_name, has_ca) and contributes exactly one atom; the
    // residue ordinal (and atom-indexed residue_id) is the slice position. A CA
    // atom is emitted when has_ca is true (protein), omitted otherwise (ligand).
    fn align_topo(residues: &[(&str, &str, bool)]) -> PrismPrepTopology {
        let mut positions = Vec::new();
        let mut elements = Vec::new();
        let mut atom_names = Vec::new();
        let mut residue_names = Vec::new();
        let mut residue_ids = Vec::new();
        let mut chain_ids = Vec::new();
        let mut ca_indices = Vec::new();
        for (rid, (chain, name, has_ca)) in residues.iter().enumerate() {
            if *has_ca {
                ca_indices.push(rid); // one atom per residue ⇒ atom_idx == rid
                atom_names.push("CA".into());
            } else {
                atom_names.push("X".into());
            }
            positions.extend_from_slice(&[rid as f32, 0.0, 0.0]);
            elements.push("C".into());
            residue_names.push((*name).to_string());
            residue_ids.push(rid);
            chain_ids.push((*chain).to_string());
        }
        let n_atoms = residues.len();
        let n_residues = residues.len();
        PrismPrepTopology {
            aromatic_targets: Vec::new(),
            source_pdb: "align_test.pdb".into(),
            n_atoms,
            n_residues,
            total_physical_residue_count: None,
            n_analysis_residues: None,
            n_chains: 1,
            positions,
            elements,
            atom_names,
            residue_names,
            residue_ids,
            atom_physical_residue_ids: None,
            atom_analysis_residue_ids: None,
            analysis_residue_contract: None,
            chain_ids,
            charges: vec![0.0; n_atoms],
            masses: vec![12.0; n_atoms],
            ca_indices,
            bonds: Vec::new(),
            angles: Vec::new(),
            dihedrals: Vec::new(),
            lj_params: Vec::new(),
            exclusions: Vec::new(),
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

    #[test]
    fn alignment_identity_maps_a_to_a() {
        let a = align_topo(&[
            ("A", "MET", true),
            ("A", "ALA", true),
            ("A", "GLY", true),
            ("A", "LEU", true),
        ]);
        let al = ResidueAlignment::build(&a, &a).expect("identity align must succeed");
        assert_eq!(al.n_ref_residues, 4);
        assert_eq!(al.n_ref_ca, 4);
        assert_eq!(al.n_b_residues, 4);
        assert!(al.b_only_residues.is_empty());
        assert_eq!(al.n_mutations, 0);
        assert_eq!(al.residue_a_to_b, (0..4).map(Some).collect::<Vec<_>>());
        assert_eq!(al.ca_a_to_b, (0..4).map(Some).collect::<Vec<_>>());
    }

    #[test]
    fn alignment_accepts_appended_ligand() {
        // A = 4 protein residues; B = same protein frame + 1 ligand (no CA).
        let a = align_topo(&[
            ("A", "MET", true),
            ("A", "ALA", true),
            ("A", "GLY", true),
            ("A", "LEU", true),
        ]);
        let b = align_topo(&[
            ("A", "MET", true),
            ("A", "ALA", true),
            ("A", "GLY", true),
            ("A", "LEU", true),
            ("A", "ALN", false), // appended ligand, no CA
        ]);
        let al = ResidueAlignment::build(&a, &b).expect("ligand-append align must succeed");
        assert_eq!(al.n_ref_residues, 4);
        assert_eq!(al.n_ref_ca, 4);
        assert_eq!(al.n_b_residues, 5);
        assert_eq!(al.b_only_residues, vec![4]); // ligand ordinal classified B-only
        assert_eq!(al.n_mutations, 0);
        // CA map still covers only the 4 shared protein residues.
        assert_eq!(al.ca_a_to_b, (0..4).map(Some).collect::<Vec<_>>());
    }

    #[test]
    fn alignment_accepts_mutation_and_stays_index_aligned() {
        // Design variant: residue 1 ALA→PHE, same chain + residue_id.
        let a = align_topo(&[
            ("A", "MET", true),
            ("A", "ALA", true),
            ("A", "GLY", true),
            ("A", "LEU", true),
        ]);
        let b = align_topo(&[
            ("A", "MET", true),
            ("A", "PHE", true), // mutation — name differs, position identical
            ("A", "GLY", true),
            ("A", "LEU", true),
        ]);
        let al = ResidueAlignment::build(&a, &b).expect("mutation must be accepted");
        assert_eq!(al.n_mutations, 1);
        assert!(al.b_only_residues.is_empty());
        // Mutation does NOT break the index alignment.
        assert_eq!(al.residue_a_to_b, (0..4).map(Some).collect::<Vec<_>>());
        assert_eq!(al.ca_a_to_b, (0..4).map(Some).collect::<Vec<_>>());
    }

    #[test]
    fn alignment_rejects_frame_mismatch() {
        let a = align_topo(&[
            ("A", "MET", true),
            ("A", "ALA", true),
            ("A", "GLY", true),
            ("A", "LEU", true),
        ]);

        // (a) chain reorder at ordinal 2 → hard error.
        let reordered = align_topo(&[
            ("A", "MET", true),
            ("A", "ALA", true),
            ("B", "GLY", true), // chain flips mid-sequence
            ("A", "LEU", true),
        ]);
        assert!(
            ResidueAlignment::build(&a, &reordered).is_err(),
            "chain reorder must hard-error"
        );

        // (b) truncated protein frame (count change) → hard error.
        let truncated = align_topo(&[("A", "MET", true), ("A", "ALA", true), ("A", "GLY", true)]);
        assert!(
            ResidueAlignment::build(&a, &truncated).is_err(),
            "protein-frame truncation must hard-error"
        );
    }

    fn brute_nearest(pos: [f32; 3], ca_xyz: &[[f32; 3]]) -> Option<usize> {
        let cutoff2 = SPIKE_TO_CA_CUTOFF * SPIKE_TO_CA_CUTOFF;
        let mut best_idx: Option<usize> = None;
        let mut best_d2 = cutoff2;
        for (i, ca) in ca_xyz.iter().enumerate() {
            let dx = pos[0] - ca[0];
            let dy = pos[1] - ca[1];
            let dz = pos[2] - ca[2];
            let d2 = dx * dx + dy * dy + dz * dz;
            let replace = match best_idx {
                None => d2 < best_d2,
                Some(best_i) => d2 < best_d2 || (d2 == best_d2 && i < best_i),
            };
            if replace {
                best_d2 = d2;
                best_idx = Some(i);
            }
        }
        best_idx
    }

    #[test]
    fn two_system_caveat_shape_and_purely_additive() {
        // Stage D safety: the not-a-dG caveat is well-formed AND purely additive, so a
        // single-topology (no --topology-b) emission that never inserts it is byte-identical to
        // the pre-change output.
        let base = serde_json::json!({"mode": "prism-twin", "sites": [], "binding_sites": 0});
        // no-flag path never carries the caveat
        assert!(base.get("contrast").is_none());
        // two-system path: caveat inserted, correctly shaped
        let mut two = base.clone();
        insert_two_system_caveat(&mut two);
        let c = &two["contrast"];
        assert_eq!(c["quantitative"], serde_json::json!(false));
        assert_eq!(c["differential_sign"], serde_json::json!("B_excess_over_A"));
        assert!(c["interpretation"]
            .as_str()
            .unwrap()
            .contains("not_binding_affinity_not_dG"));
        // removing ONLY the inserted key restores the exact no-flag structure (byte-identical)
        two.as_object_mut().unwrap().remove("contrast");
        assert_eq!(
            two, base,
            "two_system emission must differ from no-flag output ONLY by the additive contrast key"
        );
    }

    #[test]
    fn twin_spatial_index_matches_bruteforce_nearest_ca() {
        let ca_xyz = vec![
            [0.0, 0.0, 0.0],
            [11.9, 0.0, 0.0],
            [24.0, 0.0, 0.0],
            [-12.0, -12.0, 0.0],
            [0.0, 12.0, 12.0],
        ];
        let index = CaSpatialIndex::new(&ca_xyz, SPIKE_TO_CA_CUTOFF);
        let probes = [
            [0.1, 0.1, 0.1],
            [6.0, 0.0, 0.0],
            [12.1, 0.1, 0.0],
            [23.9, 0.0, 0.0],
            [-11.9, -11.8, 0.0],
            [0.0, 12.1, 11.9],
            [100.0, 100.0, 100.0],
        ];
        for pos in probes {
            assert_eq!(index.nearest(pos, &ca_xyz), brute_nearest(pos, &ca_xyz));
        }
    }

    #[test]
    fn twin_combined_assignment_matches_bruteforce_counts_and_phases() {
        let ca_xyz = vec![
            [0.0, 0.0, 0.0],
            [11.9, 0.0, 0.0],
            [24.0, 0.0, 0.0],
            [0.0, 12.0, 12.0],
        ];
        let spikes = vec![
            spike(0, [0.1, 0.1, 0.1]),
            spike(311, [12.1, 0.0, 0.0]),
            spike(444, [23.9, 0.0, 0.0]),
            spike(778, [0.0, 12.1, 11.9]),
            spike(889, [0.0, 12.1, 11.9]),
            spike(999, [100.0, 100.0, 100.0]),
        ];
        let max_ts = 1000.0;
        let index = CaSpatialIndex::new(&ca_xyz, SPIKE_TO_CA_CUTOFF);
        let (counts, phase_counts) =
            assign_spikes_to_residues_and_phases(&spikes, &ca_xyz, &index, max_ts);

        let mut expected_counts = vec![0u32; ca_xyz.len()];
        let mut expected_phases: [Vec<u32>; 5] = [
            vec![0; ca_xyz.len()],
            vec![0; ca_xyz.len()],
            vec![0; ca_xyz.len()],
            vec![0; ca_xyz.len()],
            vec![0; ca_xyz.len()],
        ];
        for s in &spikes {
            let Some(idx) = brute_nearest(s.position, &ca_xyz) else {
                continue;
            };
            expected_counts[idx] += 1;
            let frac = s.timestep as f32 / max_ts;
            let phase_idx = PHASE_BOUNDS_FAST_HYSTERESIS
                .iter()
                .position(|&(lo, hi, _)| frac >= lo && frac < hi)
                .unwrap_or(4);
            expected_phases[phase_idx][idx] += 1;
        }

        assert_eq!(counts, expected_counts);
        assert_eq!(phase_counts, expected_phases);
    }
}
