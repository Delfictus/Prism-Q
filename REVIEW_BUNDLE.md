# Prism-Q engine review bundle

> GENERATED FILE. Do not edit. Regenerate with `python3 scripts/build_review_bundle.py`.

| Field | Value |
|---|---|
| Source commit | `5173c7110c4c1f6e08868386ef66626a6a79727f` (engine subdir `engine`) |
| Generator | `build_review_bundle.py` v1.0.0 |
| Spec | `review_bundle.toml` sha256 `4f0101e9c7d0067ba5a9c549c58f05841ba0618b80c1949a6f3e16e639b2ab52` |
| Included source files | 23 |
| Source digest (sha256 over sorted path, sha256 pairs) | `941409d8051d85d1aa60dd0f6a9574fc92f08a9ae5900832cb54aaac35788679` |
| Sections | 15 |
| Open defects | 35 |

The source commit above is the commit checked out when the bundle was generated. The commit that contains this file is that commit's child when the bundle is regenerated in the same change. Staleness is decided by the file hashes and the rendered content (`--check`), not by the commit id.

## Reviewer contract

1. Read this file first. It is generated from the source at the commit shown above; it is never hand-edited.
2. Review the verbatim source excerpts, not documentation, summaries or comments. Where an excerpt and
   its prose disagree, the source wins and the disagreement is a finding.
3. Answer every mandatory question in every section. Each answer cites `path:line` anchors from the excerpts
   (or from the repository at the same commit).
4. Re-run `python3 scripts/build_review_bundle.py --check` (staleness) and `--verify` (reference checks)
   and report the exit codes. Do not trust the embedded results without re-running them.
5. Report every disagreement with an equation, unit, invariant, status or defect entry as a finding with
   severity, `path:line`, the expected value, the observed value and a reproduction.
6. Status words mean exactly: QUALIFIED (passed its stated gate with a receipt), EXECUTED_UNQUALIFIED
   (runs, gate not passed), NOT_BUILT (does not exist yet). Never upgrade a status in a review.
7. Claim scope: computational evidence only.

## Claim scope

Computational evidence only. Nothing in this repository or bundle is wet-lab, clinical,
regulatory, synthesis, safety or ADMET validation. The BIPRISM electronic layer is a
phenomenological quantum analogue with uncalibrated parameters (label: analogue_uncalibrated)
until the tight-binding grounding package passes its gates.

## Layer status

| Section | Layer | Status | Open defects | NOT_BUILT parts | Note |
|---|---|---|---|---|---|
| S01 | Topology and preparation (HMR, residue identity) | **EXECUTED_UNQUALIFIED** | 2 | 0 |  |
| S02 | Bonded and nonbonded forces | **EXECUTED_UNQUALIFIED** | 1 | 0 |  |
| S03 | Integrator (BAOAB/OVRVO phases, SHAKE/RATTLE/SETTLE, dt, MTS) | **EXECUTED_UNQUALIFIED** | 3 | 1 |  |
| S04 | Thermostat, RNG and the friction ramp | **EXECUTED_UNQUALIFIED** | 3 | 0 |  |
| S05 | Neighbour list | **EXECUTED_UNQUALIFIED** | 1 | 0 |  |
| S06 | Solvent (MVS-DS) | **NOT_BUILT** | 1 | 3 |  |
| S07 | UV excited state and the W_uv[lambda] ledger | **EXECUTED_UNQUALIFIED** | 2 | 1 |  |
| S08 | BIPRISM mesh and DEC operator (simplicial tile edge weights, PCG) | **EXECUTED_UNQUALIFIED** | 4 | 0 |  |
| S09 | Electronic propagator (Cayley/Magnus) and the tight-binding coupling | **NOT_BUILT** | 6 | 4 |  |
| S10 | LIF / Voronoi detector and event extraction | **EXECUTED_UNQUALIFIED** | 4 | 0 |  |
| S11 | Chronicle: CLOCK, leaves and restart closure | **EXECUTED_UNQUALIFIED** | 3 | 2 |  |
| S12 | QNANL encode / seal / decode | **QUALIFIED** | 1 | 2 | Packet codec (PRSQNB01) qualified as a post-run seal (253 round-trip receipts). In-loop sealing, families 52-59 and the out-of-process decoder are NOT_BUILT. |
| S13 | md_evidence_manifest | **EXECUTED_UNQUALIFIED** | 1 | 1 |  |
| S14 | Path B materialization | **EXECUTED_UNQUALIFIED** | 2 | 0 |  |
| S15 | DSTW export | **EXECUTED_UNQUALIFIED** | 1 | 0 |  |

## Data flow (input to Path B / DSTW)

```text
 prepared topology (AMBER params, residue identity, HMR)                         [S01]
        |
        v
 +---------------------------- one MD step (captured CUDA graph) ----------------------------+
 |  neighbour list [S05] -> bonded + nonbonded forces [S02] (+ solvent field [S06] NOT_BUILT) |
 |        -> integrator phases OU/kick/drift/SHAKE/RATTLE [S03] <- thermostat + RNG [S04]     |
 |        -> UV excited state + intervention work W_uv [S07]                                  |
 |        -> BIPRISM mesh/DEC operator [S08] -> electronic analogue [S09] (in-loop NOT_BUILT) |
 |        -> LIF / Voronoi detector -> spike events [S10]                                     |
 +--------------------------------------------------------------------------------------------+
        |  every datum bound to clock key + atom/residue/mesh anchor
        v
 Chronicle L0..L5: CLOCK, leaves, checkpoints, restart closure                    [S11]
        |
        v
 QNANL encode / seal / durable publish (post-run today)                            [S12]
        |
        v
 md_evidence_manifest.json (replay contract)                                       [S13]
        |                                  (no MD rerun past this line)
        v
 Path B materialization (prism_materialize_sites)                                  [S14]
        |
        v
 DSTW export (dstw_export_wt_pathb) -> WT physical profile JSON + Parquet         [S15]
```

## Build and tree hazards

- PTX resolution: generated PTX is never committed. The nhs_rt_full path resolves PTX through prism_gpu::ptx_authority (env dir, beside the executable, then this build's OUT_DIR/ptx) and records kernel_set_sha256 (e6d7c4f). Still open: the nhs-analyze-* binaries and 6 prism-gpu modules resolve PTX outside the authority, and the baked OUT_DIR fallback can bind a mutable build tree for an installed binary that has no env and no ptx/ beside it (review findings S2-F1..F10).
- The v10/v9 adjudicator include now resolves to the real crates/prism-nhs/src/cuda/adjudicator.cuh, and a missing header is an #error (67d6730; PTX byte-identical). Builds before 67d6730 compiled an opaque 256-byte stand-in struct. Open: prism-gpu build.rs does not list that header as rerun-if-changed (S1-F1).
- Workspace .cargo/config.toml replaces crates-io with vendor/crates-io, which is not published; a public build needs `cargo vendor` or removal of the source replacement.
- The SDST library is a prebuilt binary kept outside this repository; the build reads it through SDST_LIB_DIR pinned by PRISM_SDST_LIB_SHA256 and fails closed when missing or mismatched.

## Included files

| File | sha256 |
|---|---|
| `crates/prism-gpu/src/kernels/nhs_amber_fused.cu` | `46287f27d2c49231de8376c1a58cc60e18fff3fb1cf44a6b0f818f2f643f8f30` |
| `crates/prism-gpu/src/kernels/nhs_biprism_bowyer_watson_seed.cuh` | `255d925e1c37fb03d880a51365b43f1bf331adb130c2825171d7e4f0274f4a8a` |
| `crates/prism-gpu/src/kernels/nhs_biprism_dec_pcg.cuh` | `18dfce17d400d9b04389575948783ebabff949c127b6faa0ff89e068e9015c01` |
| `crates/prism-gpu/src/kernels/nhs_biprism_electronic_discrete.cuh` | `a416c303a31602bc1ae077cc17874a6b59d5048a792726a334b30265ef19eedc` |
| `crates/prism-gpu/src/kernels/nhs_biprism_simplicial_tile.cuh` | `849da5dd1333fffd796546f1b732dc8317490835d6af46326893354a02afd79e` |
| `crates/prism-gpu/src/kernels/nhs_biprism_unit_bridge.cuh` | `de47488e2cd3e678cf3af21c8b493b5bc926fcfc4e938fdc3f108d2e71272f30` |
| `crates/prism-gpu/src/kernels/nhs_biprism_voronoi_lif.cuh` | `306032ad98cdf52cc84364484c07c5a8ec2b7647dd4aa55f3ae10575e8f29760` |
| `crates/prism-gpu/src/kernels/nhs_excited_state.cuh` | `de02a444844fbbd1961175364a82b67c5717794039b1c1b47f6d4f61fa14e2fa` |
| `crates/prism-gpu/src/kernels/nhs_neighbor_warp.cuh` | `f7c489740080c04e04818a6686f13c35d14cddc68c14b03b9077db03638bc771` |
| `crates/prism-gpu/src/kernels/settle.cu` | `92087d513c25993f062f93e60c00209df35c1f5cdb1203874d03905e5ed76abc` |
| `crates/prism-nhs/src/bin/dstw_export_wt_pathb.rs` | `7cd848565353d36450c5e6a8a463e1345dd23f28b0d69fe4779c777ac195d4ec` |
| `crates/prism-nhs/src/bin/nhs_rt_full.rs` | `fe8181ee63e2a511f6861e181ea4b203b8484f2afc00e199251f306ebded87aa` |
| `crates/prism-nhs/src/bin/nhs_rt_full/chronicle_raw.rs` | `046809bdc19eca4739a0c24d7d5d84ab1da7a37a21d53ed32ef57d8efe8544a1` |
| `crates/prism-nhs/src/bin/prism_materialize_sites.rs` | `2e7e8271afd5e98c58264356bc5219197548e7e172c1b1db6afbebaccd22e4c2` |
| `crates/prism-nhs/src/biprism_mesh.rs` | `814e718daaa4e6655925eb24a79139ac54d8d8d65ec4905f53af04d9e1c142b4` |
| `crates/prism-nhs/src/coupled_chronology.rs` | `5a5734db642c02806eb58e23db63d4c1c36026ac476f2a1bf5ce6d96028f6be5` |
| `crates/prism-nhs/src/cuda/gearbox.cuh` | `d09f87ea71c709e8ba000212a0fed833f9e1a90cfb56b299ad8bb1169342d3da` |
| `crates/prism-nhs/src/fused_engine.rs` | `59a75928d85b580bd3ee9fd7ba0bd53ab2bfa0d75e3b2ffd0923fe940f80e309` |
| `crates/prism-nhs/src/input.rs` | `5547e11a20811dd3afb8c4d227b93489fd780589e925721f1923ea9d926dd3f1` |
| `crates/prism-nhs/src/scientific_checkpoint/restart_transaction.rs` | `7e300c9105fe8c96023b841405bf0c455416aec0d9233c6da02c69efb6fb2a20` |
| `crates/prism-nhs/src/sdst_bridge.rs` | `3639178df54984f061750cc1fdedc12d0f620492758baa3070b53356eb119fd3` |
| `crates/prismcode/src/md_native_runtime.rs` | `627c438845fb83319f833b7fdfaf49bcd9cdf0a3bab166768a9a7c7137d41499` |
| `crates/prismcode/src/qnanl_artifact.rs` | `b279575a69792e714fd8372203a7b94b06ed9cf93ac28baf753dfb7138fd8c91` |

## S01. Topology and preparation (HMR, residue identity)

**Layer status:** `EXECUTED_UNQUALIFIED`

Reads a prepared AMBER-parameterised topology (ff14SB/ff19SB style prmtop/JSON), builds per-atom
masses, charges, LJ parameters, bonded terms, exclusions and residue identity. Hydrogen mass
repartitioning (HMR) moves mass from heavy atoms to bonded hydrogens. Residue identity is carried
as residue index plus name and chain; insertion codes and PDB numbering are not carried end to end.

### Source excerpts (verbatim)

**`crates/prism-nhs/src/input.rs:2482-2484`** (selector: symbol='pub fn apply_hmr\\('; sha256 `5547e11a20811dd3…`)

```rust
2482 |     pub fn apply_hmr(&mut self, factor: f64) {
2483 |         self.apply_hmr_with_historical_repeat(factor,false);
2484 |     }
```

**`crates/prism-nhs/src/input.rs:2494-2565`** (selector: symbol='fn apply_hmr_with_historical_repeat'; sha256 `5547e11a20811dd3…`)

```rust
2494 |     fn apply_hmr_with_historical_repeat(&mut self, factor:f64, repeat_verified:bool) {
2495 |         // Only a VERIFIED prior application suppresses repartitioning.
2496 |         // `Unverified` must not suppress it: skipping the repartition on an
2497 |         // unestablished claim would leave the mass model unrepartitioned while
2498 |         // the run believed otherwise. The conservative action under an
2499 |         // unestablished claim is to perform the documented repartition, which
2500 |         // then establishes the fact.
2501 |         if self.hmr_verified_applied() && !repeat_verified {
2502 |             log::info!(
2503 |                 "HMR: topology already records repartitioned masses; preserving serialized mass model"
2504 |             );
2505 |             return;
2506 |         }
2507 |         if self.hmr_applied.is_none() {
2508 |             log::info!(
2509 |                 "HMR: topology carried NO hmr_applied key (authority=Unverified); applying the \
2510 |                  documented repartition and recording the result as verified-applied"
2511 |             );
2512 |         }
2513 |         let mut n_repartitioned = 0usize;
2514 |         let mut total_mass_before = 0.0f64;
2515 |         let mut total_mass_after = 0.0f64;
2516 | 
2517 |         for cluster in &mut self.h_clusters {
2518 |             // Original masses from inverse masses
2519 |             let m_h_orig = 1.0 / cluster.inv_mass_h;
2520 |             let m_central_orig = 1.0 / cluster.inv_mass_central;
2521 | 
2522 |             total_mass_before += m_central_orig + cluster.n_hydrogens as f64 * m_h_orig;
2523 | 
2524 |             // New hydrogen mass = original * factor
2525 |             let m_h_new = m_h_orig * factor;
2526 |             let delta_m = m_h_new - m_h_orig;
2527 | 
2528 |             // Steal mass from central atom (conserve total)
2529 |             let m_central_new = m_central_orig - cluster.n_hydrogens as f64 * delta_m;
2530 | 
2531 |             // Safety: central atom must retain at least 1.0 amu
2532 |             if m_central_new < 1.0 {
2533 |                 log::warn!(
2534 |                     "HMR: central atom {} would drop to {:.2} amu, skipping cluster",
2535 |                     cluster.central_atom,
2536 |                     m_central_new
2537 |                 );
2538 |                 total_mass_after += m_central_orig + cluster.n_hydrogens as f64 * m_h_orig;
2539 |                 continue;
2540 |             }
2541 | 
2542 |             cluster.inv_mass_h = 1.0 / m_h_new;
2543 |             cluster.inv_mass_central = 1.0 / m_central_new;
2544 |             n_repartitioned += 1;
2545 | 
2546 |             total_mass_after += m_central_new + cluster.n_hydrogens as f64 * m_h_new;
2547 | 
2548 |             // Also update the masses array for the atoms
2549 |             if cluster.central_atom < self.masses.len() {
2550 |                 self.masses[cluster.central_atom] = m_central_new as f32;
2551 |             }
2552 |             for &h_idx in &cluster.hydrogen_atoms {
2553 |                 if h_idx >= 0 && (h_idx as usize) < self.masses.len() {
2554 |                     self.masses[h_idx as usize] = m_h_new as f32;
2555 |                 }
2556 |             }
2557 |         }
2558 | 
2559 |         log::info!("HMR: repartitioned {}/{} clusters (factor {:.1}x), total mass {:.2} → {:.2} amu (delta={:.4})",
2560 |             n_repartitioned, self.h_clusters.len(), factor,
2561 |             total_mass_before, total_mass_after,
2562 |             (total_mass_after - total_mass_before).abs());
2563 |         self.hmr_applied = Some(true);
2564 |         self.recommended_timestep_fs = Some(4.0);
2565 |     }
```

**`crates/prism-nhs/src/input.rs:1057-1077`** (selector: anchor='pub residue_ids: Vec<usize>'; sha256 `5547e11a20811dd3…`)

```rust
1057 |     pub elements: Vec<String>,
1058 |     /// Atom names (CA, CB, N, O, etc.)
1059 |     pub atom_names: Vec<String>,
1060 |     /// Residue names (ALA, GLY, etc.)
1061 |     pub residue_names: Vec<String>,
1062 |     /// Residue IDs for each atom
1063 |     pub residue_ids: Vec<usize>,
1064 |     /// Explicit physical residue axis; must equal legacy `residue_ids`.
1065 |     #[serde(default)]
1066 |     pub atom_physical_residue_ids: Option<Vec<usize>>,
1067 |     /// Sole KCC/Ghost axis; `-1` excludes solvent, ions and fragments.
1068 |     #[serde(default)]
1069 |     pub atom_analysis_residue_ids: Option<Vec<i32>>,
1070 |     /// Hash-bound mapping and role contract for the analysis axis.
1071 |     #[serde(default)]
1072 |     pub analysis_residue_contract: Option<AnalysisResidueContract>,
1073 |     /// Chain IDs
1074 |     pub chain_ids: Vec<String>,
1075 |     /// Partial charges
1076 |     pub charges: Vec<f32>,
1077 |     /// Masses
```

### Mathematics

HMR: for each H bonded to heavy atom X, \(m_H' = f\,m_H\), \(m_X' = m_X - (f-1)\,m_H\), total mass conserved: \(\sum_i m_i' = \sum_i m_i\).

### Units

| Quantity | Unit |
|---|---|
| mass | amu (g/mol) |
| charge | e |
| length | Angstrom |
| LJ epsilon | kcal/mol |
| LJ sigma | Angstrom |

### Invariants

- Total mass is unchanged by HMR (to FP64 round-off).
- No hydrogen mass exceeds its declared target; no heavy atom mass becomes non-positive.
- Each atom maps to exactly one (chain, residue index, residue name); this map is the spatial anchor for every downstream datum.

### Known defects

| ID | Defect | Where | Status | Fix package | Check |
|---|---|---|---|---|---|
| D-RESID-OFFSET | Residue index converted with lr.resid - 1 in site lining / phase-manifold features | crates/prism-nhs/src/bin/nhs_rt_full.rs (lr.resid - 1, four sites) | **OPEN** | P3 |  |
| D-GB-RADII-UNREAD | gb_radii emitted by prep and read by nothing | prep output; no engine consumer | **OPEN** | P9 |  |

### Qualifying gate

Topology round trip against the prep tool output: per-atom mass/charge/type identity, residue identity (chain, PDB number, insertion code) preserved, HMR total-mass check.

### Mandatory reviewer questions

1. Derive the heavy-atom mass after apply_hmr for a CH3 group with factor f and show the total-mass invariant holds in the excerpt (cite path:line).
2. Where is the PDB residue number and insertion code carried? If it is not carried, list every downstream consumer that converts residue index to residue number and show the conversion (path:line).
3. Does the historical-May HMR path (apply_hmr_historical_may_execution) apply HMR a second time to an already repartitioned topology? Show the guard.
4. gb_radii is emitted by prep: find any reader in the engine. If there is none, state it as a finding.

## S02. Bonded and nonbonded forces

**Layer status:** `EXECUTED_UNQUALIFIED`

Production nonbonded is plain Coulomb with epsilon_r = 1 (vacuum) plus 12-6 LJ, cut at 10 Angstrom
with a potential shift on the ordinary (non-exception) pairs. This is vacuum, not implicit solvent.
1-4 pairs use ff14SB fudge factors or exact prmtop exceptions. A second kernel,
compute_nonbonded_neighborlist, carries a distance-dependent dielectric (epsilon = 4r); which kernel
the production route launches must be established from the host launch sites (reviewer question 5).

### Source excerpts (verbatim)

**`crates/prism-gpu/src/kernels/nhs_amber_fused.cu:66-66`** (selector: anchor='#define COULOMB_CONSTANT'; sha256 `46287f27d2c49231…`)

```cuda
66 | #define COULOMB_CONSTANT 332.0636f  // kcal/mol * Angstrom / e^2
```

**`crates/prism-gpu/src/kernels/nhs_amber_fused.cu:165-166`** (selector: anchor='#define NB_CUTOFF 10'; sha256 `46287f27d2c49231…`)

```cuda
165 | #define NB_CUTOFF 10.0f             // Nonbonded cutoff (Angstroms)
166 | #define NB_CUTOFF_SQ 100.0f         // Cutoff squared
```

**`crates/prism-gpu/src/kernels/nhs_amber_fused.cu:632-640`** (selector: symbol='__device__ __forceinline__ double ordinary_cutoff_potential_offset_v1'; sha256 `46287f27d2c49231…`)

```cuda
632 | __device__ __forceinline__ double ordinary_cutoff_potential_offset_v1(
633 |     float sigma,float epsilon,float charge_product,float cutoff_sq,
634 |     float scale_lj,float scale_qq
635 | ) {
636 |     const GeometryReal inv_rc=bp_native_rsqrt((GeometryReal)cutoff_sq);
637 |     const GeometryReal sr=sigma*inv_rc,sr2=sr*sr,sr6=sr2*sr2*sr2,sr12=sr6*sr6;
638 |     return (double)scale_lj*4.0*(double)epsilon*((double)sr12-(double)sr6)
639 |         +(double)scale_qq*(double)COULOMB_CONSTANT*(double)charge_product*(double)inv_rc;
640 | }
```

**`crates/prism-gpu/src/kernels/nhs_amber_fused.cu:644-717`** (selector: symbol='__device__ __forceinline__ void compute_nonbonded_force\\('; sha256 `46287f27d2c49231…`)

```cuda
644 | __device__ __forceinline__ void compute_nonbonded_force(
645 |     const GeoVec& pi, const GeoVec& pj,
646 |     float qi, float qj,
647 |     float sigma_i, float epsilon_i,
648 |     float sigma_j, float epsilon_j,
649 |     float3& fi, float3& fj,
650 |     float cutoff_sq,
651 |     // M1.2.17 — per-atom PE accumulator + 2 atom indices.  pe_components
652 |     // = nullptr disables PE accumulation (legacy callers).
653 |     double* pe_components,
654 |     int nbi, int nbj,
655 |     // Legacy 1-4 pairs use ff14SB FUDGE values. Exact exception pairs use
656 |     // the prmtop/OpenMM charge product and LJ values passed alongside them.
657 |     bool is_14,
658 |     bool exact_14,
659 |     float exact_charge_product,
660 |     float exact_sigma,
661 |     float exact_epsilon,
662 |     bool historical_may_model = false
663 | ) {
664 |     using GeoReal = typename BpNativeGeometryVector<GeoVec>::Real;
665 |     GeoVec rij = bp_native_vec<GeoVec>(pj.x - pi.x, pj.y - pi.y, pj.z - pi.z);
666 |     GeoReal r_sq = rij.x*rij.x + rij.y*rij.y + rij.z*rij.z;
667 | 
668 |     // OpenMM NonbondedForce exceptions are explicit bonded corrections and
669 |     // remain active outside the ordinary nonbonded cutoff.
670 |     if ((!exact_14 && r_sq > cutoff_sq) || r_sq < 1e-8f) return;
671 | 
672 |     // OPTIMIZATION: Use rsqrtf (hardware intrinsic) instead of sqrt + divide
673 |     GeoReal inv_r = bp_native_rsqrt(r_sq);  // 1/sqrt(r_sq) in single instruction
674 |     GeoReal inv_r2 = inv_r * inv_r;
675 | 
676 |     float sigma = exact_14 ? exact_sigma : 0.5f * (sigma_i + sigma_j);
677 |     float epsilon = exact_14 ? exact_epsilon : sqrtf(epsilon_i * epsilon_j);
678 |     float charge_product = exact_14 ? exact_charge_product : qi * qj;
679 | 
680 |     // LJ force - precompute powers efficiently
681 |     GeoReal sigma_r = sigma * inv_r;
682 |     GeoReal sigma_r2 = sigma_r * sigma_r;
683 |     GeoReal sigma_r6 = sigma_r2 * sigma_r2 * sigma_r2;  // (sigma/r)^6
684 |     GeoReal sigma_r12 = sigma_r6 * sigma_r6;
685 |     GeoReal lj_force = 24.0f * epsilon * inv_r2 * (2.0f * sigma_r12 - sigma_r6);
686 | 
687 |     // Coulomb force
688 |     GeoReal elec_force = COULOMB_CONSTANT * charge_product * inv_r2 * inv_r;
689 | 
690 |     // M1.2.18-P2.3 — 1-4 FUDGE applied to BOTH force and PE.
691 |     // Branchless select.
692 |     const float scale_lj = (is_14 && !exact_14) ? FUDGE_LJ : 1.0f;
693 |     const float scale_qq = (is_14 && !exact_14) ? FUDGE_QQ : 1.0f;
694 |     GeoReal total_force = scale_lj * lj_force + scale_qq * elec_force;
695 | 
696 |     GeoVec f = bp_native_vec<GeoVec>(total_force * rij.x, total_force * rij.y, total_force * rij.z);
697 |     fi.x -= f.x; fi.y -= f.y; fi.z -= f.z;
698 |     fj.x += f.x; fj.y += f.y; fj.z += f.z;
699 | 
700 |     // M1.2.17 — accumulate V_pair = V_LJ + V_elec (with 1-4 FUDGE
701 |     // applied per M1.2.18-P2.3) split half-half across atoms i, j.
702 |     //   V_LJ   = 4·ε·(σ_r¹² − σ_r⁶)         (Lorentz-Berthelot ε,σ form)
703 |     //   V_elec = COULOMB_CONSTANT·q_i·q_j / r
704 |     if (pe_components != nullptr) {
705 |         const double v_lj   = 4.0 * (double)epsilon
706 |                                 * ((double)sigma_r12 - (double)sigma_r6);
707 |         const double v_elec = (double)COULOMB_CONSTANT * (double)charge_product
708 |                                 * (double)inv_r;
709 |         const double v_pair = (double)scale_lj * v_lj
710 |                             + (double)scale_qq * v_elec
711 |                             - ((exact_14 || historical_may_model) ? 0.0 : ordinary_cutoff_potential_offset_v1<GeoReal>(
712 |                                 sigma,epsilon,charge_product,cutoff_sq,scale_lj,scale_qq));
713 |         const double v_pair_half = v_pair * 0.5;
714 |         atomicAdd(pe_components + nbi, v_pair_half);
715 |         atomicAdd(pe_components + nbj, v_pair_half);
716 |     }
717 | }
```

**`crates/prism-gpu/src/kernels/nhs_amber_fused.cu:12014-12030`** (selector: anchor='Coulomb with implicit solvent \\(ε = 4r\\)'; sha256 `46287f27d2c49231…`)

```cuda
12014 |         float inv_r2_soft = 1.0f / r2_soft;
12015 |         float inv_r6_soft = inv_r2_soft * inv_r2_soft * inv_r2_soft;
12016 |         float sigma6_r6 = sigma6 * inv_r6_soft;
12017 | 
12018 |         float lj_force = 24.0f * eps_ij * (2.0f * sigma6_r6 * sigma6_r6 - sigma6_r6) / r2_soft;
12019 | 
12020 |         // Coulomb with implicit solvent (ε = 4r)
12021 |         float coul_force = COULOMB_CONSTANT * my_charge * other_charge * 0.25f * inv_r * inv_r * inv_r;
12022 | 
12023 |         // Total force with capping
12024 |         float total_force = lj_force + coul_force;
12025 |         float max_force = 500.0f;
12026 |         if (fabsf(total_force) > max_force) {
12027 |             total_force = copysignf(max_force, total_force);
12028 |         }
12029 | 
12030 |         // Accumulate force on my atom
```

### Mathematics

\(F_{ij} = \left[24\,\varepsilon_{ij}\,r^{-2}\left(2(\sigma/r)^{12}-(\sigma/r)^{6}\right) + k_C\,q_iq_j\,r^{-3}\right]\mathbf r_{ij}\), \(k_C = 332.0636\) kcal mol\(^{-1}\) Å e\(^{-2}\)

Shifted potential for \(r<r_c\): \(V(r) = V_{LJ}(r)+V_C(r) - [V_{LJ}(r_c)+V_C(r_c)]\); force is not shifted (discontinuous force derivative at \(r_c\)).

Lorentz-Berthelot: \(\sigma_{ij}=\tfrac12(\sigma_i+\sigma_j)\), \(\varepsilon_{ij}=\sqrt{\varepsilon_i\varepsilon_j}\).

### Units

| Quantity | Unit |
|---|---|
| r, sigma, cutoff | Angstrom |
| epsilon, V | kcal/mol |
| q | e |
| k_C (COULOMB_CONSTANT) | kcal mol^-1 Angstrom e^-2 |
| F | kcal mol^-1 Angstrom^-1 |

### Invariants

- Newton's third law per pair: fi -= f, fj += f.
- The potential offset applies to ordinary pairs only; exact exception pairs are never shifted and stay active beyond the cutoff.
- Net force on an isolated system with no external field is zero to round-off.

### Known defects

| ID | Defect | Where | Status | Fix package | Check |
|---|---|---|---|---|---|
| D-VACUUM | Production electrostatics are vacuum (epsilon_r = 1, 10 Angstrom potential-shift cutoff); temperature-dependent dielectric is computed and unused | nhs_amber_fused.cu COULOMB_CONSTANT / compute_nonbonded_force; fused_engine.rs compute_cryo_dielectric (_effective_dielectric unused) | **OPEN** | P9 (MVS-DS solvent) |  |

### Qualifying gate

Per-term energy and force agreement against an independent reference (OpenMM 8.6.1 with identical cutoff/shift settings) on frozen snapshots; finite-difference force check of the composed potential.

### Mandatory reviewer questions

1. Show the units of each factor in elec_force and confirm the result is kcal mol^-1 Angstrom^-2 before multiplication by r_ij (path:line).
2. ordinary_cutoff_potential_offset_v1 shifts the potential but not the force. Quantify the energy-conservation consequence at r_c = 10 Angstrom for a +1/-1 pair.
3. Which value of the Coulomb constant does the engine use, and how does it compare with AMBER (332.0522173), CHARMM (332.0716) and the CODATA 2022 value? (See the unit_conversions check.)
4. Is there any dielectric, reaction-field or GB term on the production path? Cite the line where it would be applied or state that it is absent.
5. compute_nonbonded_neighborlist uses epsilon = 4r while compute_nonbonded_force uses epsilon = 1. Find every host launch of each kernel (fused_engine.rs) and state which one the production route executes.

## S03. Integrator (BAOAB/OVRVO phases, SHAKE/RATTLE/SETTLE, dt, MTS)

**Layer status:** `EXECUTED_UNQUALIFIED`

Langevin splitting executed as separate phase kernels (OU half, kick, drift, SHAKE position,
RATTLE velocity) inside a captured step graph. Gears are 0.5 / 2 / 4 fs. There is no 5 or 6 fs gear
and no multiple-time-step (MTS) integrator. X-H bonds are constrained (SHAKE/RATTLE); SETTLE exists
for rigid water but is used only on the lambda path and a self-test.

### Source excerpts (verbatim)

**`crates/prism-nhs/src/cuda/gearbox.cuh:393-399`** (selector: anchor='#define PRISM_GEAR_DEFAULT_DT_FS_G0'; sha256 `d09f87ea71c709e8…`)

```cuda
393 | /// Canonical gear timesteps in fs, mirroring `d_gearbox_table` slot 0
394 | /// (0.0005 / 0.0020 / 0.0040 ps).  Gear 3 is the abort sentinel and has no
395 | /// timestep; 0.0 keeps the chunk-latency arithmetic finite.
396 | #define PRISM_GEAR_DEFAULT_DT_FS_G0 0.5f
397 | #define PRISM_GEAR_DEFAULT_DT_FS_G1 2.0f
398 | #define PRISM_GEAR_DEFAULT_DT_FS_G2 4.0f
399 | #define PRISM_GEAR_DEFAULT_DT_FS_G3 0.0f
```

**`crates/prism-gpu/src/kernels/nhs_amber_fused.cu:3656-3683`** (selector: anchor='Post-UV original second half: same unrecomputed, unclamped force'; sha256 `46287f27d2c49231…`)

```cuda
3656 |     if (execution_phase == MD_PHASE_OU_HALF_N || execution_phase == MD_PHASE_OU_HALF_NP1) {
3657 |         if (tid < n_atoms && gamma != 0.0f) {
3658 |             double* work_component = d_external_work != nullptr ? &d_external_work[tid] : nullptr;
3659 |             curandState local_rng = rng_states[tid];
3660 |             langevin_thermostat(
3661 |                 velocities[tid], masses[tid], target_temp, gamma, 0.5f * dt,
3662 |                 &local_rng, work_component);
3663 |             rng_states[tid] = local_rng;
3664 |         }
3665 |         return;
3666 |     }
3667 |     if (execution_phase == MD_PHASE_KICK_N || execution_phase == MD_PHASE_KICK_NP1) {
3668 |         if (exact_nonbonded_exceptions_enabled < 0) {
3669 |             // Post-UV original second half: same unrecomputed, unclamped force.
3670 |             // This opcode is admitted only by the historical post-step caller.
3671 |             if(execution_phase!=MD_PHASE_KICK_NP1){PRISM_PHYSICS_TRAP();return;}
3672 |             if(tid<n_atoms) {
3673 |                 const float inverse=1.0f/masses[tid];
3674 |                 velocities[tid].x+=0.5f*dt*forces[tid].x*inverse;
3675 |                 velocities[tid].y+=0.5f*dt*forces[tid].y*inverse;
3676 |                 velocities[tid].z+=0.5f*dt*forces[tid].z*inverse;
3677 |                 if(!isfinite(velocities[tid].x)||!isfinite(velocities[tid].y)||!isfinite(velocities[tid].z)){
3678 |                     PRISM_PHYSICS_TRAP();return;
3679 |                 }
3680 |             }
3681 |             return;
3682 |         }
3683 |         if (tid < n_atoms) {
```

**`crates/prism-gpu/src/kernels/settle.cu:53-132`** (selector: symbol='__global__ void settle_constraints\\('; sha256 `92087d513c25993f…`)
  
_Excerpt truncated at max_lines=80; the full symbol spans lines 53-180._

```cuda
 53 | __global__ void settle_constraints(
 54 |     float* __restrict__ new_pos,
 55 |     const float* __restrict__ old_pos,
 56 |     const int* __restrict__ water_idx,
 57 |     int n_waters,
 58 |     float mO, float mH,
 59 |     float ra, float rb, float rc,
 60 |     float rOH2, float rHH2,
 61 |     int* __restrict__ failure_count
 62 | ) {
 63 |     int w = blockIdx.x * blockDim.x + threadIdx.x;
 64 |     if (w >= n_waters) return;
 65 | 
 66 |     // Get atom indices for this water
 67 |     int idxO = water_idx[w * 3];
 68 |     int idxH1 = water_idx[w * 3 + 1];
 69 |     int idxH2 = water_idx[w * 3 + 2];
 70 | 
 71 |     // Load old positions (reference configuration for frame definition)
 72 |     float3 oldO = make_float3(
 73 |         old_pos[idxO * 3], old_pos[idxO * 3 + 1], old_pos[idxO * 3 + 2]
 74 |     );
 75 |     float3 oldH1 = make_float3(
 76 |         old_pos[idxH1 * 3], old_pos[idxH1 * 3 + 1], old_pos[idxH1 * 3 + 2]
 77 |     );
 78 |     float3 oldH2 = make_float3(
 79 |         old_pos[idxH2 * 3], old_pos[idxH2 * 3 + 1], old_pos[idxH2 * 3 + 2]
 80 |     );
 81 | 
 82 |     // Load new positions (unconstrained)
 83 |     float3 newO = make_float3(
 84 |         new_pos[idxO * 3], new_pos[idxO * 3 + 1], new_pos[idxO * 3 + 2]
 85 |     );
 86 |     float3 newH1 = make_float3(
 87 |         new_pos[idxH1 * 3], new_pos[idxH1 * 3 + 1], new_pos[idxH1 * 3 + 2]
 88 |     );
 89 |     float3 newH2 = make_float3(
 90 |         new_pos[idxH2 * 3], new_pos[idxH2 * 3 + 1], new_pos[idxH2 * 3 + 2]
 91 |     );
 92 | 
 93 |     float3 b0 = make_float3(oldH1.x-oldO.x, oldH1.y-oldO.y, oldH1.z-oldO.z);
 94 |     float3 c0 = make_float3(oldH2.x-oldO.x, oldH2.y-oldO.y, oldH2.z-oldO.z);
 95 |     float3 dpO = make_float3(newO.x-oldO.x, newO.y-oldO.y, newO.z-oldO.z);
 96 |     float3 dpH1 = make_float3(newH1.x-oldH1.x, newH1.y-oldH1.y, newH1.z-oldH1.z);
 97 |     float3 dpH2 = make_float3(newH2.x-oldH2.x, newH2.y-oldH2.y, newH2.z-oldH2.z);
 98 |     float invmT = 1.0f/(mO+2.0f*mH);
 99 | 
100 |     // Unconstrained new COM relative to old O.  Constraint impulses preserve it.
101 |     float3 com;
102 |     com.x=(mO*dpO.x+mH*(b0.x+dpH1.x)+mH*(c0.x+dpH2.x))*invmT;
103 |     com.y=(mO*dpO.y+mH*(b0.y+dpH1.y)+mH*(c0.y+dpH2.y))*invmT;
104 |     com.z=(mO*dpO.z+mH*(b0.z+dpH1.z)+mH*(c0.z+dpH2.z))*invmT;
105 |     float3 a1=make_float3(dpO.x-com.x, dpO.y-com.y, dpO.z-com.z);
106 |     float3 b1=make_float3(b0.x+dpH1.x-com.x, b0.y+dpH1.y-com.y, b0.z+dpH1.z-com.z);
107 |     float3 c1=make_float3(c0.x+dpH2.x-com.x, c0.y+dpH2.y-com.y, c0.z+dpH2.z-com.z);
108 | 
109 |     // ez is the old plane normal.  ex depends on the unconstrained oxygen,
110 |     // which carries the incremental rotation discarded by the old shortcut.
111 |     float3 ez=make_float3(b0.y*c0.z-b0.z*c0.y, b0.z*c0.x-b0.x*c0.z, b0.x*c0.y-b0.y*c0.x);
112 |     float3 ex=make_float3(a1.y*ez.z-a1.z*ez.y, a1.z*ez.x-a1.x*ez.z, a1.x*ez.y-a1.y*ez.x);
113 |     float3 ey=make_float3(ez.y*ex.z-ez.z*ex.y, ez.z*ex.x-ez.x*ex.z, ez.x*ex.y-ez.y*ex.x);
114 |     float ex2=ex.x*ex.x+ex.y*ex.y+ex.z*ex.z;
115 |     float ey2=ey.x*ey.x+ey.y*ey.y+ey.z*ey.z;
116 |     float ez2=ez.x*ez.x+ez.y*ez.y+ez.z*ez.z;
117 |     if (ex2<1.0e-16f || ey2<1.0e-16f || ez2<1.0e-16f) {
118 |         atomicAdd(failure_count, 1);
119 |         return;
120 |     }
121 |     float ix=rsqrtf(ex2), iy=rsqrtf(ey2), iz=rsqrtf(ez2);
122 |     ex.x*=ix; ex.y*=ix; ex.z*=ix;
123 |     ey.x*=iy; ey.y*=iy; ey.z*=iy;
124 |     ez.x*=iz; ez.y*=iz; ez.z*=iz;
125 | 
126 | #define DOT_FRAME(axis,v) (axis.x*v.x+axis.y*v.y+axis.z*v.z)
127 |     float xb0d=DOT_FRAME(ex,b0), yb0d=DOT_FRAME(ey,b0);
128 |     float xc0d=DOT_FRAME(ex,c0), yc0d=DOT_FRAME(ey,c0);
129 |     float za1d=DOT_FRAME(ez,a1);
130 |     float xb1d=DOT_FRAME(ex,b1), yb1d=DOT_FRAME(ey,b1), zb1d=DOT_FRAME(ez,b1);
131 |     float xc1d=DOT_FRAME(ex,c1), yc1d=DOT_FRAME(ey,c1), zc1d=DOT_FRAME(ez,c1);
132 | 
```

**`crates/prism-gpu/src/kernels/nhs_amber_fused.cu:1889-1978`** (selector: symbol='__device__ bool shake_cluster_positions\\('; sha256 `46287f27d2c49231…`)
  
_Excerpt truncated at max_lines=90; the full symbol spans lines 1889-2173._

```cuda
1889 | template<bool Compensated=false>
1890 | __device__ bool shake_cluster_positions(
1891 |     float3* positions,
1892 |     const float3* positions_step_start,
1893 |     float3* velocities,
1894 |     const HCluster cluster,
1895 |     float dt,
1896 |     double* nonham_work,
1897 |     int max_iter,
1898 |     int diagnostic_native_step
1899 | ,
1900 |     const BiprismCoordinateOwnerFfi* coordinate_owner=nullptr,
1901 |     uint32_t* precision_fault=nullptr
1902 | ) {
1903 |     if constexpr(!Compensated) {
1904 |     // The coordinate state is f32 and protein coordinates are commonly O(100 A).
1905 |     // An absolute 1e-6 A^2 criterion is below the representable correction at
1906 |     // that scale and deterministically stalls otherwise valid constraints.
1907 |     // Evaluate a dimensionless squared-distance residual instead.
1908 |     const float relative_tolerance = 1.0e-4f;
1909 |     const float3 central_velocity_before = velocities[cluster.central_atom];
1910 |     const float3 central_position_before = positions[cluster.central_atom];
1911 |     float3 hydrogen_positions_before[3];
1912 |     float3 hydrogen_velocities_before[3];
1913 |     #pragma unroll
1914 |     for (int h = 0; h < 3; h++) {
1915 |         const int h_atom = cluster.hydrogen_atoms[h];
1916 |         hydrogen_positions_before[h] = h_atom >= 0 ? positions[h_atom] : make_float3(0, 0, 0);
1917 |         hydrogen_velocities_before[h] = h_atom >= 0 ? velocities[h_atom] : make_float3(0, 0, 0);
1918 |     }
1919 |     bool converged = false;
1920 |     for (int iter = 0; iter < max_iter; iter++) {
1921 |         float max_relative_error = 0.0f;
1922 |         for (int h = 0; h < cluster.n_hydrogens; h++) {
1923 |             const int h_atom = cluster.hydrogen_atoms[h];
1924 |             if (h_atom < 0) continue;
1925 |             float3& pos_central = positions[cluster.central_atom];
1926 |             float3& pos_h = positions[h_atom];
1927 |             const float3 r = make_float3(
1928 |                 pos_h.x - pos_central.x,
1929 |                 pos_h.y - pos_central.y,
1930 |                 pos_h.z - pos_central.z);
1931 |             const float3 r0 = make_float3(
1932 |                 positions_step_start[h_atom].x - positions_step_start[cluster.central_atom].x,
1933 |                 positions_step_start[h_atom].y - positions_step_start[cluster.central_atom].y,
1934 |                 positions_step_start[h_atom].z - positions_step_start[cluster.central_atom].z);
1935 |             const float r_sq = r.x*r.x + r.y*r.y + r.z*r.z;
1936 |             const float target_sq = cluster.bond_lengths[h] * cluster.bond_lengths[h];
1937 |             const float diff = r_sq - target_sq;
1938 |             if (!isfinite(r_sq) || !isfinite(target_sq) || target_sq <= 0.0f) {
1939 |                 printf("PRISM_SHAKE_INVALID_LENGTH central=%d hydrogen=%d r2=%.9g target2=%.9g dt=%.9g\n",
1940 |                     cluster.central_atom,h_atom,(double)r_sq,(double)target_sq,(double)dt);
1941 |                 return false;
1942 |             }
1943 |             const float relative_error = fabsf(diff) / target_sq;
1944 |             max_relative_error = fmaxf(max_relative_error, relative_error);
1945 |             if (relative_error <= relative_tolerance) continue;
1946 |             const float r_dot_r0 = r.x*r0.x + r.y*r0.y + r.z*r0.z;
1947 |             const float r0_sq = r0.x*r0.x + r0.y*r0.y + r0.z*r0.z;
1948 |             const float direction_scale = sqrtf(r_sq * r0_sq);
1949 |             if (!isfinite(r_dot_r0) || !isfinite(direction_scale)
1950 |                     || direction_scale <= 0.0f
1951 |                     || fabsf(r_dot_r0) / direction_scale <= 1.0e-6f
1952 |                     || !isfinite(dt) || dt <= 0.0f) {
1953 |                 printf("PRISM_SHAKE_INVALID_DIRECTION central=%d hydrogen=%d r2=%.9g dot=%.9g scale=%.9g dt=%.9g\n",
1954 |                     cluster.central_atom,h_atom,(double)r_sq,(double)r_dot_r0,(double)direction_scale,(double)dt);
1955 |                 return false;
1956 |             }
1957 |             const float inv_mass_sum = cluster.inv_mass_central + cluster.inv_mass_h;
1958 |             const float lambda = diff / (2.0f * inv_mass_sum * r_dot_r0);
1959 |             const float3 correction = make_float3(lambda * r0.x, lambda * r0.y, lambda * r0.z);
1960 |             pos_central.x += cluster.inv_mass_central * correction.x;
1961 |             pos_central.y += cluster.inv_mass_central * correction.y;
1962 |             pos_central.z += cluster.inv_mass_central * correction.z;
1963 |             pos_h.x -= cluster.inv_mass_h * correction.x;
1964 |             pos_h.y -= cluster.inv_mass_h * correction.y;
1965 |             pos_h.z -= cluster.inv_mass_h * correction.z;
1966 |             velocities[cluster.central_atom].x += cluster.inv_mass_central * correction.x / dt;
1967 |             velocities[cluster.central_atom].y += cluster.inv_mass_central * correction.y / dt;
1968 |             velocities[cluster.central_atom].z += cluster.inv_mass_central * correction.z / dt;
1969 |             velocities[h_atom].x -= cluster.inv_mass_h * correction.x / dt;
1970 |             velocities[h_atom].y -= cluster.inv_mass_h * correction.y / dt;
1971 |             velocities[h_atom].z -= cluster.inv_mass_h * correction.z / dt;
1972 |         }
1973 |         if (max_relative_error <= relative_tolerance) {
1974 |             converged = true;
1975 |             break;
1976 |         }
1977 |     }
1978 | 
```

### Planned components (NOT_BUILT)

| Component | Package | Status |
|---|---|---|
| MTS integrator (inner <= 2.5 fs for 5 fs outer, <= 2 fs for 6 fs outer) | Package G | NOT_BUILT |

### Mathematics

OVRVO / BAOAB-type splitting with \(c_1 = e^{-\gamma\,\Delta t/2}\): \(v \leftarrow c_1 v + \sqrt{(1-c_1^2)k_BT/m}\,\xi\); \(v \leftarrow v + \tfrac{\Delta t}{2} F/m\); \(x \leftarrow x + \Delta t\,v\); constraints; force; kick; OU.

Linear stability of velocity Verlet for the fastest mode: \(\omega\Delta t < 2\). Nonlinear \(n{:}1\) resonance: \(\omega\Delta t = 2\sin(\pi/n)\) (\(n=3\): \(\sqrt3\); \(n=4\): \(\sqrt2\)).
Effective frequency: \(\omega_{\rm eff} = \tfrac{2}{\Delta t}\arcsin(\omega\Delta t/2)\).

### Units

| Quantity | Unit |
|---|---|
| dt | ps (gears 0.0005 / 0.002 / 0.004) |
| gamma | ps^-1 |
| v | Angstrom/ps |
| F/m conversion | AMBER_KCAL_TO_AMU_A2_PS2 (kcal/mol -> amu Angstrom^2 ps^-2) |

### Invariants

- Constraint residuals below tolerance after every SHAKE/RATTLE phase.
- Each phase kernel runs in the declared order; the post-UV historical second half (exact_nonbonded_exceptions_enabled < 0) is admitted only for the historical caller and traps otherwise.
- Gear dt is read from one table (gearbox.cuh); no other dt source.

### Known defects

| ID | Defect | Where | Status | Fix package | Check |
|---|---|---|---|---|---|
| D-RESONANCE-4FS | 4 fs single-rate sits on the 4:1 nonlinear resonance (omega*dt = 1.4164, ~11% blue shift of the fastest mode); the existing 'qualified' level 1.4272 is circular | gearbox.cuh gear 2 | **OPEN** | Package G / MTS |  |
| D-6FS-REUSE | May '6 fs holds' reuse an unrecomputed second-half force | nhs_amber_fused.cu KICK_NP1 historical branch | **OPEN** | contradiction graph; historical only |  |
| D-NO-MTS | No 5/6 fs gear and no MTS integrator | gearbox.cuh | **OPEN** | Package G |  |

### Qualifying gate

Measured omega_max resonance scan, NVE drift per gear, dt-convergence of dihedral distributions; 4 fs single-rate only after the gates pass; 5/6 fs only as MTS (inner <= 2.5 / 2 fs).

### Mandatory reviewer questions

1. Using omega_max = 354.1 ps^-1, compute omega*dt at 4 fs and identify which n:1 resonance it sits on. Compare with the omega_dt_resonance check output.
2. In the KICK_NP1 branch with exact_nonbonded_exceptions_enabled < 0, is the force recomputed before the second half kick? Explain the consequence for any '6 fs holds' claim (path:line).
3. Show where the constraint tolerance is set and whether SHAKE iterations are bounded; what happens on non-convergence?
4. List the gear dt values from gearbox.cuh and confirm no 5 or 6 fs gear exists.

## S04. Thermostat, RNG and the friction ramp

**Layer status:** `EXECUTED_UNQUALIFIED`

Langevin OU half-steps use cuRAND XORWOW per-atom states (not counter-based Philox). Friction starts
at 1000 ps^-1 and decays exponentially over the first 10,000 steps to gamma_base, then switches
discontinuously. The UV vibrational-transfer kick uses an in-kernel LCG seeded only by
(timestep, aromatic index), so every stream draws identical kick directions.

### Source excerpts (verbatim)

**`crates/prism-gpu/src/kernels/nhs_amber_fused.cu:1713-1744`** (selector: symbol='__device__ void langevin_thermostat\\('; sha256 `46287f27d2c49231…`)

```cuda
1713 | __device__ void langevin_thermostat(
1714 |     float3& velocity,
1715 |     float mass,
1716 |     float target_temp,
1717 |     float gamma,
1718 |     float dt,
1719 |     curandState* rng_state,
1720 |     double* nonham_work_component
1721 | ) {
1722 |     const float3 velocity_before = velocity;
1723 |     // Friction coefficient
1724 |     float c1 = expf(-gamma * dt);
1725 |     // AMBER stores energy in kcal/mol, mass in amu, and velocity in A/ps.
1726 |     // Convert kBT to amu*A^2/ps^2 before evaluating fluctuation-dissipation.
1727 |     float c2 = sqrtf((1.0f - c1*c1) * AMBER_KCAL_TO_AMU_A2_PS2
1728 |                      * KB_KCAL_MOL_K * target_temp / mass);
1729 | 
1730 |     // Apply Langevin dynamics
1731 |     velocity.x = c1 * velocity.x + c2 * curand_normal(rng_state);
1732 |     velocity.y = c1 * velocity.y + c2 * curand_normal(rng_state);
1733 |     velocity.z = c1 * velocity.z + c2 * curand_normal(rng_state);
1734 |     if (nonham_work_component != nullptr) {
1735 |         const double before_sq = (double)velocity_before.x * (double)velocity_before.x
1736 |                                + (double)velocity_before.y * (double)velocity_before.y
1737 |                                + (double)velocity_before.z * (double)velocity_before.z;
1738 |         const double after_sq = (double)velocity.x * (double)velocity.x
1739 |                               + (double)velocity.y * (double)velocity.y
1740 |                               + (double)velocity.z * (double)velocity.z;
1741 |         *nonham_work_component += 0.5 * (double)mass * (after_sq - before_sq)
1742 |                                 / (double)AMBER_KCAL_TO_AMU_A2_PS2;
1743 |     }
1744 | }
```

**`crates/prism-nhs/src/fused_engine.rs:15366-15396`** (selector: symbol='fn compute_cryo_friction\\('; sha256 `59a75928d85b580b…`)

```rust
15366 |     fn compute_cryo_friction(&self, temperature: f32) -> f32 {
15367 |         // Equilibration boost: EXTREME friction for first 10000 steps
15368 |         // This is CRITICAL for structures that haven't been energy-minimized
15369 |         // gamma=1000 ps⁻¹ with dt=0.002 ps gives c1=exp(-2)≈0.135 (86.5% damping per step!)
15370 |         const EQUILIBRATION_STEPS: i32 = 10000;
15371 |         const EQUILIBRATION_GAMMA: f32 = 1000.0; // Extreme damping (ps⁻¹)
15372 | 
15373 |         let base_gamma = if self.timestep < EQUILIBRATION_STEPS {
15374 |             // Exponential decay from EQUILIBRATION_GAMMA to gamma_base
15375 |             // This provides strong damping early, then gradually relaxes
15376 |             let progress = self.timestep as f32 / EQUILIBRATION_STEPS as f32;
15377 |             let decay = (-3.0 * progress).exp(); // Exponential decay factor
15378 |             EQUILIBRATION_GAMMA * decay + self.gamma_base * (1.0 - decay)
15379 |         } else {
15380 |             self.gamma_base
15381 |         };
15382 | 
15383 |         // Additional cryogenic scaling at low temperatures
15384 |         if !self.cryo_enabled || temperature >= T_REF {
15385 |             return base_gamma;
15386 |         }
15387 | 
15388 |         // Scale friction inversely with temperature
15389 |         // At T_REF (300K): gamma = base_gamma
15390 |         // At T_MIN (10K): gamma = base_gamma * sqrt(30) (much slower dynamics)
15391 |         let t_clamped = temperature.max(T_MIN);
15392 |         let scale = T_REF / t_clamped;
15393 | 
15394 |         // Use sqrt scaling to prevent extreme values
15395 |         base_gamma * scale.sqrt()
15396 |     }
```

**`crates/prism-gpu/src/kernels/nhs_excited_state.cuh:502-611`** (selector: symbol='__device__ __forceinline__ void apply_vibrational_transfer\\('; sha256 `de02a444844fbbd1…`)
  
_Excerpt truncated at max_lines=110; the full symbol spans lines 502-635._

```cuda
502 | __device__ __forceinline__ void apply_vibrational_transfer(
503 |     int aromatic_idx,
504 |     float energy_to_transfer,
505 |     float3 ring_normal,
506 |     float3* velocities,
507 |     const float* masses,         // Atom masses in amu
508 |     const int* neighbor_list,    // Atoms within ~5Å of aromatic
509 |     int n_neighbors,
510 |     unsigned int seed,           // For random direction component
511 |     int aromatic_type,           // ChromophoreType enum (0=TRP, 1=TYR, 2=PHE, 3=SS)
512 |     float uv_wavelength_nm,      // Current UV wavelength for debug output
513 |     // M1.2.18.5 — Total External Work COMPONENTS (Hamiltonian audit).
514 |     // One f64 component per atom. The caller is responsible for invoking this
515 |     // function from an ordered single-thread UV phase, so overlapping aromatic
516 |     // neighborhoods accumulate in aromatic-index order without velocity or
517 |     // work atomics. A downstream fixed-order reducer writes the scalar total to
518 |     // d_external_work_components[0] for the SFA stability fuse.
519 |     double* d_external_work_components
520 | ) {
521 |     if (energy_to_transfer < 0.001f || n_neighbors == 0) return;
522 | 
523 |     float energy_per_neighbor = energy_to_transfer / (float)n_neighbors;
524 | 
525 |     // Simple LCG for random numbers (no curand needed)
526 |     unsigned int rng = seed;
527 | 
528 | #ifdef DEBUG_UV_PHYSICS
529 |     float total_ke_added = 0.0f;
530 | #endif
531 | 
532 |     for (int i = 0; i < n_neighbors; i++) {
533 |         int neighbor = neighbor_list[i];
534 |         if (neighbor < 0) continue;
535 | 
536 |         // Get neighbor mass for proper velocity conversion
537 |         float mass = masses[neighbor];
538 | 
539 |         // v = sqrt(2 * E * 418.4 / m) = ENERGY_VEL_PREFACTOR * sqrt(E / m)
540 |         // where ENERGY_VEL_PREFACTOR = sqrt(2 * 418.4) = 28.93
541 |         float vel_magnitude = ENERGY_VEL_PREFACTOR * sqrtf(energy_per_neighbor / mass);
542 | 
543 |         // Mix of directed (ring normal) and random
544 |         rng = rng * 1103515245 + 12345;
545 |         float r = (float)(rng & 0x7FFFFFFF) / (float)0x7FFFFFFF;
546 | 
547 |         float3 kick_dir;
548 |         if (r < 0.6f) {
549 |             // 60% along ring normal (strongest coupling)
550 |             kick_dir = ring_normal;
551 |             if ((rng >> 16) & 1) kick_dir = make_float3(-kick_dir.x, -kick_dir.y, -kick_dir.z);
552 |         } else {
553 |             // 40% random
554 |             rng = rng * 1103515245 + 12345;
555 |             float rx = (float)((rng >> 0) & 0xFF) / 127.5f - 1.0f;
556 |             float ry = (float)((rng >> 8) & 0xFF) / 127.5f - 1.0f;
557 |             float rz = (float)((rng >> 16) & 0xFF) / 127.5f - 1.0f;
558 |             float len = sqrtf(rx*rx + ry*ry + rz*rz);
559 |             kick_dir = make_float3(rx/len, ry/len, rz/len);
560 |         }
561 | 
562 |         // M1.2.18.5 — Exact ΔK accumulation (operator UV KINETIC PERFECTION).
563 |         //
564 |         // Read v_old as a float3, compose v_new = v_old + kick·vel_mag in
565 |         // float, then accumulate
566 |         //   ΔK = 0.5 · m · (v_new·v_new − v_old·v_old)
567 |         // in bit-accurate f64 BEFORE applying the velocity update.
568 |         // v_old is read non-atomically (acceptable: the captured fused
569 |         // step's UV path is the only writer to this neighbor in-frame —
570 |         // SISR/G28 ensure no cross-frame race, and the SFA is a coarse
571 |         // First-Law drift detector, not a bit-exact integrator).
572 |         //
573 |         // Conversion: AMBER velocities are in Å/ps and masses in amu, so
574 |         // ½·m·v² is in amu·Å²/ps² ⇒ ÷418.4 gives kcal/mol.
575 |         const float3 v_old = make_float3(
576 |             velocities[neighbor].x,
577 |             velocities[neighbor].y,
578 |             velocities[neighbor].z
579 |         );
580 |         const float3 v_new = make_float3(
581 |             v_old.x + kick_dir.x * vel_magnitude,
582 |             v_old.y + kick_dir.y * vel_magnitude,
583 |             v_old.z + kick_dir.z * vel_magnitude
584 |         );
585 | 
586 |         if (d_external_work_components != nullptr) {
587 |             const double m_d = (double)mass;
588 |             const double vnx = (double)v_new.x;
589 |             const double vny = (double)v_new.y;
590 |             const double vnz = (double)v_new.z;
591 |             const double vox = (double)v_old.x;
592 |             const double voy = (double)v_old.y;
593 |             const double voz = (double)v_old.z;
594 |             const double delta_k_amu_a2_ps2 = 0.5 * m_d * (
595 |                 vnx * vnx + vny * vny + vnz * vnz
596 |               - vox * vox - voy * voy - voz * voz
597 |             );
598 |             const double delta_k_kcal_mol = delta_k_amu_a2_ps2 / 418.4;
599 |             d_external_work_components[neighbor] += delta_k_kcal_mol;
600 |         }
601 | 
602 |         velocities[neighbor] = v_new;
603 | 
604 | #ifdef DEBUG_UV_PHYSICS
605 |         // Verify energy conservation: KE = 0.5 * m * v² / 418.4 kcal/mol
606 |         float ke_added = 0.5f * mass * vel_magnitude * vel_magnitude / 418.4f;
607 |         total_ke_added += ke_added;
608 |         if (i == 0) {
609 |             printf("[UV ENERGY] neighbor=%d mass=%.1f E_in=%.5f kcal/mol v=%.4f A/ps KE_out=%.5f kcal/mol\n",
610 |                 neighbor, mass, energy_per_neighbor, vel_magnitude, ke_added);
611 |         }
```

**`crates/prism-gpu/src/kernels/nhs_amber_fused.cu:11054-11072`** (selector: anchor='timestep \\* n_aromatics \\+ arom_idx'; sha256 `46287f27d2c49231…`)

```cuda
11054 |             d_franck_condon_progress,
11055 |             &energy_to_transfer
11056 |         );
11057 | 
11058 |         if (energy_to_transfer > 0.001f && d_aromatic_neighbors != nullptr) {
11059 |             AromaticNeighbors neighbors = d_aromatic_neighbors[arom_idx];
11060 |             apply_vibrational_transfer(
11061 |                 arom_idx,
11062 |                 energy_to_transfer,
11063 |                 d_ring_normals[arom_idx],
11064 |                 velocities,
11065 |                 masses,
11066 |                 neighbors.atom_indices,
11067 |                 neighbors.n_neighbors,
11068 |                 timestep * n_aromatics + arom_idx,
11069 |                 d_aromatic_type[arom_idx],
11070 |                 uv_wavelength_nm,
11071 |                 d_external_work
11072 |             );
```

**`crates/prism-gpu/src/kernels/nhs_amber_fused.cu:11308-11317`** (selector: symbol='extern "C" __global__ void init_rng_states\\('; sha256 `46287f27d2c49231…`)

```cuda
11308 | extern "C" __global__ void init_rng_states(
11309 |     curandState* states,
11310 |     unsigned long long seed,
11311 |     int n
11312 | ) {
11313 |     int tid = blockIdx.x * blockDim.x + threadIdx.x;
11314 |     if (tid < n) {
11315 |         curand_init(seed, tid, 0, &states[tid]);
11316 |     }
11317 | }
```

### Mathematics

\(c_1 = e^{-\gamma\Delta t}\); \(\gamma(t) = \gamma_{eq}e^{-3s} + \gamma_0(1-e^{-3s})\), \(s = \text{step}/10^4\) for step \(<10^4\), then \(\gamma_0\).
At \(s\to1^-\): \(\gamma = 1000e^{-3} + \gamma_0(1-e^{-3}) \approx 49.8 + 0.95\gamma_0\) ps\(^{-1}\); at \(s=1\): \(\gamma_0\) (jump).
LCG: \(r_{k+1} = 1103515245\,r_k + 12345 \pmod{2^{32}}\).

### Units

| Quantity | Unit |
|---|---|
| gamma | ps^-1 |
| T | K |
| k_B T / m | Angstrom^2 ps^-2 |
| Q_bath (heat) | kcal/mol (separate owner from intervention work) |

### Invariants

- Fluctuation-dissipation: OU noise amplitude matches c1 for the same gamma and dt.
- RNG state is part of the restart closure and restored exactly.
- Thermostat heat (Q_bath) is never booked into intervention work (W_uv, W_lif).

### Known defects

| ID | Defect | Where | Status | Fix package | Check |
|---|---|---|---|---|---|
| D-FRICTION-RAMP | Friction ramp discontinuous at step 10,000 (~59 -> 10 ps^-1 for gamma_base=10); first ~40 ps overdamped | fused_engine.rs compute_cryo_friction | **OPEN** | P1/P2 protocol phases |  |
| D-RNG-UV-LCG-STREAM | UV LCG seeded by (timestep*n_aromatics + arom_idx): all streams receive identical kick directions | nhs_amber_fused.cu UV loop; nhs_excited_state.cuh apply_vibrational_transfer | **OPEN** | H0 (counter-based Philox4x32-10) |  |
| D-XORWOW | Engine RNG is cuRAND XORWOW (stateful), not counter-based; determinism gate not passed | nhs_amber_fused.cu curandState | **OPEN** | H0 |  |

### Qualifying gate

H0: two capture-off launches are bitwise identical over 10k steps including Chronicle leaf hashes; until then every on/off comparison is STATISTICAL_ONLY. Kinetic/causal windows exclude the overdamped first ~40 ps.

### Mandatory reviewer questions

1. Compute gamma just before and at step 10,000 for gamma_base = 10 ps^-1 from compute_cryo_friction and state the size of the discontinuity.
2. Show that the LCG seed in apply_vibrational_transfer contains no stream or replica identifier, and state the consequence for 8 concurrent streams.
3. Is the LCG kick momentum-conserving? Show whether centre-of-mass momentum is removed after the kick (path:line).
4. Where is the per-atom curandState saved and restored for restart, and is the restore bitwise?

## S05. Neighbour list

**Layer status:** `EXECUTED_UNQUALIFIED`

Warp-cooperative cell list, non-periodic, rebuilt every 20 steps with no displacement trigger.
Atoms outside the grid are clamped to face cells, and pairs between 10 and 12 Angstrom that are two
cells apart can be missed.

### Source excerpts (verbatim)

**`crates/prism-gpu/src/kernels/nhs_neighbor_warp.cuh:10-119`** (selector: symbol='nhs_build_neighbor_list_warp\\('; sha256 `f7c489740080c04e…`)

```cuda
 10 | extern "C" __global__ void __launch_bounds__(256) nhs_build_neighbor_list_warp(
 11 |     const float3* __restrict__ positions,
 12 |     const int* __restrict__ cell_list,
 13 |     const int* __restrict__ cell_counts,
 14 |     const int* __restrict__ atom_cell,
 15 |     const int* __restrict__ excl_list,
 16 |     const int* __restrict__ excl_offsets,
 17 |     int* __restrict__ neighbor_list,
 18 |     int* __restrict__ n_neighbors,
 19 |     int nx, int ny, int nz,
 20 |     int n_atoms,
 21 |     float cutoff_sq
 22 | ) {
 23 |     if (blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1) {
 24 |         if (threadIdx.x == 0) PRISM_PHYSICS_TRAP();
 25 |         return;
 26 |     }
 27 |     constexpr unsigned FULL_WARP = 0xffffffffu;
 28 |     constexpr int PAD = 0x7fffffff;
 29 |     __shared__ int rows[8][NEIGHBOR_LIST_SIZE]; // 32 KiB, no global sort traffic
 30 |     const int lane = threadIdx.x & 31;
 31 |     const int warp = threadIdx.x >> 5;
 32 |     const int atom = blockIdx.x * 8 + warp;
 33 |     // Entire warps return together; remaining warps need no block barriers.
 34 |     if (atom >= n_atoms) return;
 35 |     int* row = rows[warp];
 36 |     const float3 pi = positions[atom];
 37 |     const int cell = atom_cell[atom];
 38 |     const int ix = cell % nx, iy = (cell / nx) % ny, iz = cell / (nx * ny);
 39 |     const int excl_begin = excl_offsets[atom], excl_end = excl_offsets[atom+1];
 40 |     int count = 0;
 41 | 
 42 |     for (int dz=-1; dz<=1; ++dz) {
 43 |         const int z = iz+dz;
 44 |         if (z<0 || z>=nz) continue;
 45 |         for (int dy=-1; dy<=1; ++dy) {
 46 |             const int y = iy+dy;
 47 |             if (y<0 || y>=ny) continue;
 48 |             for (int dx=-1; dx<=1; ++dx) {
 49 |                 const int x = ix+dx;
 50 |                 if (x<0 || x>=nx) continue;
 51 |                 const int other_cell = x+y*nx+z*nx*ny;
 52 |                 const int cell_count = cell_counts[other_cell];
 53 |                 if (cell_count > MAX_ATOMS_PER_CELL) {
 54 |                     if (lane==0) PRISM_PHYSICS_TRAP();
 55 |                     return;
 56 |                 }
 57 |                 for (int base=0; base<cell_count; base+=32) {
 58 |                     const int slot = base+lane;
 59 |                     int j = PAD;
 60 |                     bool keep = false;
 61 |                     if (slot<cell_count) {
 62 |                         j = cell_list[other_cell*MAX_ATOMS_PER_CELL+slot];
 63 |                         if (j!=atom) {
 64 |                             const float dx_ij=positions[j].x-pi.x;
 65 |                             const float dy_ij=positions[j].y-pi.y;
 66 |                             const float dz_ij=positions[j].z-pi.z;
 67 |                             const float r2=dx_ij*dx_ij+dy_ij*dy_ij+dz_ij*dz_ij;
 68 |                             // Keep the scalar builder's exact comparison,
 69 |                             // including equality at the buffered cutoff.
 70 |                             keep = !(r2>cutoff_sq);
 71 |                             if (keep) {
 72 |                                 for (int e=excl_begin; e<excl_end; ++e) {
 73 |                                     if (excl_list[e]==j) { keep=false; break; }
 74 |                                 }
 75 |                             }
 76 |                         }
 77 |                     }
 78 |                     const unsigned mask = __ballot_sync(FULL_WARP, keep);
 79 |                     const int added = __popc(mask);
 80 |                     if (count+added>NEIGHBOR_LIST_SIZE) {
 81 |                         if (lane==0) PRISM_PHYSICS_TRAP();
 82 |                         return;
 83 |                     }
 84 |                     if (keep) {
 85 |                         const unsigned lower_lanes = (1u<<lane)-1u;
 86 |                         row[count+__popc(mask & lower_lanes)] = j;
 87 |                     }
 88 |                     count += added;
 89 |                 }
 90 |             }
 91 |         }
 92 |     }
 93 | 
 94 |     // Pad only the next power-of-two extent; all live atom IDs precede PAD.
 95 |     int sort_count=1;
 96 |     while (sort_count<count) sort_count<<=1;
 97 |     for (int i=count+lane; i<sort_count; i+=32) row[i]=PAD;
 98 |     __syncwarp(FULL_WARP);
 99 |     for (int width=2; width<=sort_count; width<<=1) {
100 |         for (int stride=width>>1; stride>0; stride>>=1) {
101 |             for (int i=lane; i<sort_count; i+=32) {
102 |                 const int partner=i^stride;
103 |                 // Each pair has exactly one writer in this network stage.
104 |                 if (partner>i) {
105 |                     const int a=row[i], b=row[partner];
106 |                     const bool ascending=(i & width)==0;
107 |                     if ((ascending && a>b) || (!ascending && a<b)) {
108 |                         row[i]=b;row[partner]=a;
109 |                     }
110 |                 }
111 |             }
112 |             __syncwarp(FULL_WARP);
113 |         }
114 |     }
115 |     int* out=neighbor_list+atom*NEIGHBOR_LIST_SIZE;
116 |     for (int i=lane; i<count; i+=32) out[i]=row[i];
117 |     __syncwarp(FULL_WARP);
118 |     if (lane==0) n_neighbors[atom]=count;
119 | }
```

**`crates/prism-nhs/src/fused_engine.rs:13364-13368`** (selector: anchor='neighbor_list_rebuild_interval: 20'; sha256 `59a75928d85b580b…`)

```rust
13364 |             d_neighbor_list,
13365 |             d_n_neighbors,
13366 |             neighbor_list_rebuild_interval: 20, // Rebuild every 20 steps
13367 |             steps_since_rebuild: 0,
13368 |             use_neighbor_list,
```

### Mathematics

Verlet-buffer requirement: a fixed rebuild interval \(N\) is safe only if \(2\,\max_i|\Delta x_i| < r_{list} - r_c\) over \(N\) steps.

### Units

| Quantity | Unit |
|---|---|
| cell size, cutoff, skin | Angstrom |
| rebuild interval | steps (should be physical time or displacement) |

### Invariants

- Every pair with r < r_c at any step between rebuilds is in the list.
- No atom is silently clamped; out-of-grid atoms fault.

### Known defects

| ID | Defect | Where | Status | Fix package | Check |
|---|---|---|---|---|---|
| D-NBL-MISS-10-12 | Non-periodic list clamps atoms at grid faces and can miss 10-12 Angstrom pairs two cells apart; fixed 20-step rebuild, no displacement trigger | nhs_neighbor_warp.cuh; fused_engine.rs neighbor_list_rebuild_interval | **OPEN** | P3 / Package G |  |

### Qualifying gate

Brute-force O(N^2) pair-set comparison at every step of a recorded trajectory segment; zero missed pairs with r < r_c.

### Mandatory reviewer questions

1. From nhs_build_neighbor_list_warp, derive which cell offsets are searched and show the geometric case in which a pair at 10-12 Angstrom is missed.
2. Show how atom_cell is computed for an atom outside the grid bounds (clamp vs fault).
3. Is there any displacement-based rebuild trigger? Cite the rebuild-interval source.

## S06. Solvent (MVS-DS)

**Layer status:** `NOT_BUILT`

The planned implicit/mesh solvent (MVS-DS) does not exist. Today the engine runs in vacuum.
SolventMode::Explicit appends oxygen-only -0.834 e pseudo-waters, which must never satisfy a solvent
claim. The explicit-water reference (SETTLE + periodic list + PME) is not on the production path;
the reference for qualification comes from an external explicit-water engine run.

### Source excerpts (verbatim)

**`crates/prism-nhs/src/input.rs:2708-2734`** (selector: anchor='topology\\.charges\\.push\\(-0\\.834\\)'; sha256 `5547e11a20811dd3…`)

```rust
2708 |                 "Added {} water molecules ({} atoms)",
2709 |                 n_waters,
2710 |                 water_coords.len() / 3
2711 |             );
2712 | 
2713 |             // Append water positions to topology
2714 |             topology.positions.extend_from_slice(&water_coords);
2715 | 
2716 |             // Append water metadata to all arrays
2717 |             for _ in 0..n_waters {
2718 |                 topology.elements.push("O".to_string()); // Water oxygen
2719 |                 topology.atom_names.push("O".to_string());
2720 |                 topology.residue_ids.push(topology.n_residues); // All waters in same "residue"
2721 |                 topology.chain_ids.push("W".to_string()); // Water chain
2722 |                 topology.charges.push(-0.834); // TIP3P oxygen charge
2723 |                 topology.masses.push(15.9994); // Oxygen mass
2724 |             }
2725 | 
2726 |             topology.n_atoms += n_waters;
2727 |             // residue_names is atom-indexed: one entry per atom
2728 |             for _ in 0..n_waters {
2729 |                 topology.residue_names.push("HOH".to_string());
2730 |             }
2731 |             topology.n_residues += 1; // All waters count as one "residue" for simplicity
2732 |             topology.water_oxygens = water_indices.clone();
2733 | 
2734 |             Some(water_indices)
```

**`crates/prism-nhs/src/solvent/mvs_ds.rs`**: **NOT_BUILT**. Planned location of the MVS-DS solvent module.

### Planned components (NOT_BUILT)

| Component | Package | Status |
|---|---|---|
| MVS-DS mesh solvent backbone (S0-S7) | P9 | NOT_BUILT |
| Learned bounded solvent corrector (W_solv_learned plane) | P9 / D3 | NOT_BUILT |

### Mathematics

(design target, not implemented) \(\Delta G_{solv} = \Delta G_{polar}[\text{mesh DEC Poisson}] + \Delta G_{np} + \text{bounded learned correction}\); conservative force \(F = -\nabla(\ldots)\).

### Units

| Quantity | Unit |
|---|---|
| (not built) | - |

### Invariants

- No solvent claim is made for any observable that has not passed the explicit-water qualification panel.
- Pseudo-water (oxygen-only) mode never satisfies a solvent claim.

### Known defects

| ID | Defect | Where | Status | Fix package | Check |
|---|---|---|---|---|---|
| D-PSEUDO-WATER | SolventMode::Explicit appends oxygen-only -0.834 e pseudo-waters | crates/prism-nhs/src/input.rs | **OPEN** | P9 |  |

### Qualifying gate

Frozen observable panel vs explicit-water reference: dihedral distributions, RMSF, contacts, salt bridges, pocket opening probability and volume, pocket water occupancy, hydration free energies, kinetics.

### Mandatory reviewer questions

1. Confirm from input.rs that the explicit mode adds oxygen-only point charges with no hydrogens and no water model geometry.
2. Which code path, if any, would apply a dielectric or solvent field to the production nonbonded kernel today?
3. State which observables could be claimed under the current vacuum model and which cannot.

## S07. UV excited state and the W_uv[lambda] ledger

**Layer status:** `EXECUTED_UNQUALIFIED`

Per-aromatic excited-state model (Trp/Tyr/Phe/His chromophores) with a prescribed excitation rate,
Franck-Condon progress, vibrational energy transfer to ring neighbours and a charge-state rescale.
Intervention work is published into a mixed d_external_work scalar plane today; the per-wavelength
W_uv[lambda] plane is planned. UV excitations are protocol inputs, not discoveries.

### Source excerpts (verbatim)

**`crates/prism-gpu/src/kernels/nhs_excited_state.cuh:364-417`** (selector: symbol='__device__ __forceinline__ void update_excited_state_inline\\('; sha256 `de02a444844fbbd1…`)

```cuda
364 | __device__ __forceinline__ void update_excited_state_inline(
365 |     int aromatic_idx,
366 |     int residue_type,
367 |     float dt,
368 |     // Excited state arrays
369 |     int* is_excited,
370 |     float* time_since_excitation,
371 |     float* electronic_population,
372 |     float* vibrational_energy,
373 |     float* franck_condon_progress,
374 |     // Output: energy to transfer to environment this step
375 |     float* energy_to_transfer_out
376 | ) {
377 |     if (!is_excited[aromatic_idx]) {
378 |         *energy_to_transfer_out = 0.0f;
379 |         return;
380 |     }
381 |     
382 |     // Update timer
383 |     time_since_excitation[aromatic_idx] += dt;
384 |     
385 |     // --- Franck-Condon relaxation (50 fs) ---
386 |     float fc = franck_condon_progress[aromatic_idx];
387 |     fc += (1.0f - fc) * INV_TAU_FRANCK_CONDON * dt;
388 |     franck_condon_progress[aromatic_idx] = fc;
389 |     
390 |     // --- Vibrational relaxation (2 ps) ---
391 |     float vib = vibrational_energy[aromatic_idx];
392 |     float vib_decay = fast_expf(-INV_TAU_VIBRATIONAL * dt);
393 |     float energy_out = vib * (1.0f - vib_decay);
394 |     vibrational_energy[aromatic_idx] = vib * vib_decay;
395 |     *energy_to_transfer_out = energy_out;
396 |     
397 |     // --- Electronic decay (ns timescale) ---
398 |     float inv_tau_fluor;
399 |     // CANONICAL ordering: 0=TRP, 1=TYR, 2=PHE, 3=S-S
400 |     switch (residue_type) {
401 |         case 0: inv_tau_fluor = INV_TAU_FLUOR_TRP; break;  // TRP: 2600 ps
402 |         case 1: inv_tau_fluor = INV_TAU_FLUOR_TYR; break;  // TYR: 3400 ps
403 |         case 2: inv_tau_fluor = INV_TAU_FLUOR_PHE; break;  // PHE: 6800 ps
404 |         default: inv_tau_fluor = INV_TAU_FLUOR_TRP;
405 |     }
406 |     float total_decay_rate = inv_tau_fluor * (1.0f + INV_TAU_IC_FACTOR);
407 |     float pop = electronic_population[aromatic_idx];
408 |     pop *= fast_expf(-total_decay_rate * dt);
409 |     electronic_population[aromatic_idx] = pop;
410 |     
411 |     // Check for complete decay
412 |     if (pop < 0.01f) {
413 |         is_excited[aromatic_idx] = 0;
414 |         electronic_population[aromatic_idx] = 0.0f;
415 |         vibrational_energy[aromatic_idx] = 0.0f;
416 |     }
417 | }
```

**`crates/prism-gpu/src/kernels/nhs_amber_fused.cu:2692-2716`** (selector: anchor='P5 Step 1 .{1,3}charge-neutral excited-state charge rescale'; sha256 `46287f27d2c49231…`)

```cuda
2692 | // ── P5 Step 1 — charge-neutral excited-state charge rescale ───────────────
2693 | //
2694 | // An electronic excitation changes a chromophore's DIPOLE, not its net charge.
2695 | // The historical form `q_ground * scale` is a uniform multiplicative rescale,
2696 | // which preserves the set's monopole only when that set already sums to zero.
2697 | //
2698 | // That held BY ACCIDENT for the legacy atom mapping (the first 16 atoms of each
2699 | // aromatic residue sum to about +0.003 e). It FAILS for the true ring atom set,
2700 | // which sums to about -0.725 e per aromatic. Swapping the mapping to ring atoms
2701 | // under the multiplicative form would inject roughly -16 e across a 30-aromatic
2702 | // system directly into the Coulomb sum — and every geometry check would still
2703 | // pass while it happened.
2704 | //
2705 | // The affine form rescales about the set's own mean charge:
2706 | //     q_new = q_mean + (q_ground - q_mean) * scale
2707 | // so sum(q_new) == sum(q_ground) for ANY scale. The dipole is rescaled, the
2708 | // monopole is conserved exactly, and the mapping can safely become ring-local.
2709 | __device__ __forceinline__ float uv_excited_charge(
2710 |     float q_ground,
2711 |     const float* d_aromatic_charge_set_mean,
2712 |     int aromatic_idx,
2713 |     float scale
2714 | ) {
2715 |     if (d_aromatic_charge_set_mean == nullptr) {
2716 |         return q_ground * scale;
```

**`crates/prism-gpu/src/kernels/nhs_amber_fused.cu:10793-10810`** (selector: symbol='__device__ __forceinline__ float uv_charge_for_state\\('; sha256 `46287f27d2c49231…`)

```cuda
10793 | __device__ __forceinline__ float uv_charge_for_state(
10794 |     int atom, const float* charges, const float* ground_charges,
10795 |     const int* atom_to_aromatic, const int* aromatic_type, const float* charge_set_mean,
10796 |     const int* is_excited, const float* population, float& scale
10797 | ) {
10798 |     scale = 1.0f;
10799 |     const int arom = atom_to_aromatic[atom];
10800 |     if (arom < 0 || !is_excited[arom]) return charges[atom];
10801 |     float ratio_sqrt;
10802 |     switch (aromatic_type[arom]) {
10803 |         case 0: ratio_sqrt = TRP_DIPOLE_RATIO_SQRT; break;
10804 |         case 1: ratio_sqrt = TYR_DIPOLE_RATIO_SQRT; break;
10805 |         case 2: ratio_sqrt = PHE_DIPOLE_RATIO_SQRT; break;
10806 |         default: ratio_sqrt = 1.0f;
10807 |     }
10808 |     scale = 1.0f + (ratio_sqrt - 1.0f) * population[arom];
10809 |     return uv_excited_charge(ground_charges[atom], charge_set_mean, arom, scale);
10810 | }
```

**`crates/prism-gpu/src/kernels/nhs_amber_fused.cu:10963-10977`** (selector: symbol='extern "C" __global__ void nhs_uv_charge_parameter_work\\('; sha256 `46287f27d2c49231…`)

```cuda
10963 | extern "C" __global__ void nhs_uv_charge_parameter_work(
10964 |     const float3* positions, const float* charges, const float* ground_charges,
10965 |     const int* atom_to_aromatic, const int* aromatic_type, const float* charge_set_mean,
10966 |     const int* excited_before, const float* population_before,
10967 |     const int* excited_after, const float* population_after,
10968 |     const int* neighbor_list, const int* n_neighbors,
10969 |     const int* pairs14_list, const int* pairs14_offsets, const float* pairs14_charge_product,
10970 |     int exact_exceptions, int n_atoms, float cutoff,
10971 |     const int* lambda_atom_slot, const float* lambda_slot_values, int n_lambda_slots,
10972 |     float alpha_q_a2, float softcore_power, float coupling_power, int lambda_v2_ewald_enabled,
10973 |     double* pe_components, double* external_work,
10974 |     double* lambda_dudl_atom, double* lambda_energy_atom
10975 | ) {
10976 |     bp_native_uv_parameter_work_body<false>(positions, charges, ground_charges, atom_to_aromatic, aromatic_type, charge_set_mean, excited_before, population_before, excited_after, population_after, neighbor_list, n_neighbors, pairs14_list, pairs14_offsets, pairs14_charge_product, exact_exceptions, n_atoms, cutoff, lambda_atom_slot, lambda_slot_values, n_lambda_slots, alpha_q_a2, softcore_power, coupling_power, lambda_v2_ewald_enabled, pe_components, external_work, lambda_dudl_atom, lambda_energy_atom, nullptr, nullptr);
10977 | }
```

**`crates/prism-gpu/src/kernels/nhs_amber_fused.cu:11089-11126`** (selector: symbol='extern "C" __global__ void nhs_reduce_external_work_components\\('; sha256 `46287f27d2c49231…`)

```cuda
11089 | // ACCUMULATES into it (atomicAdd, not assignment), so across a replay the cell
11090 | // holds the true window integral. The adjudication graph zeroes that cell at
11091 | // replay head, which closes the window exactly where `cruise->v_prev` latches —
11092 | // so the W_ext window and the v_prev window finally describe the same interval.
11093 | //
11094 | // `d_wext_window` may be null on legacy call paths; the reduction then degrades
11095 | // to a no-op rather than writing a plausible value into the atom plane. The SFA
11096 | // reads UNMEASURED in that case instead of a fabricated zero.
11097 | extern "C" __global__ void nhs_reduce_external_work_components(
11098 |     double* d_external_work,
11099 |     int n_atoms,
11100 |     double* d_wext_window
11101 | ) {
11102 |     __shared__ double partial[256];
11103 |     const int lane = threadIdx.x;
11104 |     double sum = 0.0;
11105 | 
11106 |     if (d_external_work != nullptr) {
11107 |         for (int i = lane; i < n_atoms; i += blockDim.x) {
11108 |             sum += d_external_work[i];
11109 |         }
11110 |     }
11111 |     partial[lane] = sum;
11112 |     __syncthreads();
11113 | 
11114 |     for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
11115 |         if (lane < stride) {
11116 |             partial[lane] += partial[lane + stride];
11117 |         }
11118 |         __syncthreads();
11119 |     }
11120 | 
11121 |     if (lane == 0 && d_external_work != nullptr && d_wext_window != nullptr) {
11122 |         // Accumulate, never assign: the window spans a whole replay, while this
11123 |         // kernel runs once per step. Atom 0's slot is left untouched.
11124 |         atomicAdd(d_wext_window, partial[0]);
11125 |     }
11126 | }
```

### Planned components (NOT_BUILT)

| Component | Package | Status |
|---|---|---|
| Typed W_uv[lambda] work plane (per wavelength, per chromophore) replacing the mixed d_external_work scalar | P7.5 | NOT_BUILT |

### Mathematics

\(E_{photon}(\lambda) = hc/\lambda\); at 280 nm: 4.428 eV = 102.1 kcal/mol.
Charge-neutral rescale: \(q_i' = \bar q + s\,(q_i - \bar q)\) over the chromophore set, so \(\sum q' = \sum q\).
Work of a parameter switch at fixed coordinates: \(W = \sum_{pairs} k_C (q_i'q_j' - q_iq_j)/r_{ij}\).

### Units

| Quantity | Unit |
|---|---|
| lambda | nm |
| E_photon, W_uv | kcal/mol |
| velocity kick | Angstrom/ps |

### Invariants

- Energy deposited by the kick equals the booked W_uv for that event (first-law closure).
- The excited-state charge rescale conserves total charge.
- Only declared chromophore atoms change state under a UV burst.

### Known defects

| ID | Defect | Where | Status | Fix package | Check |
|---|---|---|---|---|---|
| D-UV-MIXED-LEDGER | UV work booked into the mixed d_external_work scalar; no per-wavelength plane | nhs_amber_fused.cu / nhs_excited_state.cuh | **OPEN** | P7.5 |  |
| D-UV-PRESCRIBED | UV excitations come from a prescribed rate model (inputs, not discoveries); 280 nm can couple to the analogue carbon s->p gap | nhs_excited_state.cuh | **OPEN** | H3 |  |

### Qualifying gate

First-law balance per UV event (deposited vs booked), charge conservation <= 1e-12 e, non-chromophore invariance under a 280 nm burst, per-wavelength ledger separation.

### Mandatory reviewer questions

1. Show the units of each term in the energy deposited by apply_vibrational_transfer (E_dep) and where it is added to d_external_work.
2. Is the deposited kinetic energy computed from the actual velocity change (1/2 m (v'^2 - v^2)) or from the requested energy? Cite the line.
3. Derive the charge-neutral rescale and confirm sum(q') = sum(q) in uv_excited_charge / uv_charge_for_state.
4. How would the ledger distinguish two wavelengths applied in the same run today?

## S08. BIPRISM mesh and DEC operator (simplicial tile edge weights, PCG)

**Layer status:** `EXECUTED_UNQUALIFIED`

Per-tile Bowyer-Watson Delaunay seed (<= 256 atoms per tile, <= 8 leaves, cold seeding), peeling
by circumradius, exact Rust re-admission of the mesh, then a P1-FEM / DEC cotangent Laplacian with
regularised volume and a PCG solve. CRITICAL: the diagonal DEC edge weights are assigned to the
opposite edge (01<->23, 02<->13, 03<->12). Every DEC, Phase-2, LIF and v10 result used that
operator. The dual is barycentric, not Voronoi.

### Source excerpts (verbatim)

**`crates/prism-gpu/src/kernels/nhs_biprism_simplicial_tile.cuh:217-274`** (selector: symbol='__device__ __forceinline__ void evaluate_diagonal_dec_tet_weights\\('; sha256 `849da5dd1333fffd…`)

```cuda
217 | __device__ __forceinline__ void evaluate_diagonal_dec_tet_weights(
218 |     float4 x0, float4 x1, float4 x2, float4 x3,
219 |     float4 X0, float4 X1, float4 X2, float4 X3,
220 |     float sigma_0,
221 |     float& out_warped_vol,
222 |     float& out_w01, float& out_w02, float& out_w03,
223 |     float& out_w12, float& out_w13, float& out_w23
224 | ) {
225 |     const float3 e01 = {x1.x - x0.x, x1.y - x0.y, x1.z - x0.z};
226 |     const float3 e02 = {x2.x - x0.x, x2.y - x0.y, x2.z - x0.z};
227 |     const float3 e03 = {x3.x - x0.x, x3.y - x0.y, x3.z - x0.z};
228 |     const float3 e12 = {x2.x - x1.x, x2.y - x1.y, x2.z - x1.z};
229 |     const float3 e13 = {x3.x - x1.x, x3.y - x1.y, x3.z - x1.z};
230 |     const float3 e23 = {x3.x - x2.x, x3.y - x2.y, x3.z - x2.z};
231 | 
232 |     const float3 E01 = {X1.x - X0.x, X1.y - X0.y, X1.z - X0.z};
233 |     const float3 E02 = {X2.x - X0.x, X2.y - X0.y, X2.z - X0.z};
234 |     const float3 E03 = {X3.x - X0.x, X3.y - X0.y, X3.z - X0.z};
235 | 
236 |     const float3 N3 = {e01.y * e02.z - e01.z * e02.y, e01.z * e02.x - e01.x * e02.z, e01.x * e02.y - e01.y * e02.x};
237 |     const float3 N2 = {e03.y * e01.z - e03.z * e01.y, e03.z * e01.x - e03.x * e01.z, e03.x * e01.y - e03.y * e01.x};
238 |     const float3 N1 = {e02.y * e03.z - e02.z * e03.y, e02.z * e03.x - e02.x * e03.z, e02.x * e03.y - e02.y * e03.x};
239 |     const float3 N0 = {e13.y * e12.z - e13.z * e12.y, e13.z * e12.x - e13.x * e12.z, e13.x * e12.y - e13.y * e12.x};
240 | 
241 |     const float det6V = __fmaf_rn(e01.x, N1.x, __fmaf_rn(e01.y, N1.y, e01.z * N1.z));
242 |     const float det6V_ref = E01.x * (E02.y * E03.z - E02.z * E03.y)
243 |                           - E01.y * (E02.x * E03.z - E02.z * E03.x)
244 |                           + E01.z * (E02.x * E03.y - E02.y * E03.x);
245 | 
246 |     const float V_T     = fabsf(det6V) * (1.0f / 6.0f);
247 |     const float V_ref   = fmaxf(fabsf(det6V_ref) * (1.0f / 6.0f), 1e-5f);
248 |     const float J_piola = fminf(fmaxf(V_T / V_ref, 0.05f), 20.0f);
249 | 
250 |     const float l2_sum = (e01.x*e01.x + e01.y*e01.y + e01.z*e01.z) + (e02.x*e02.x + e02.y*e02.y + e02.z*e02.z)
251 |                        + (e03.x*e03.x + e03.y*e03.y + e03.z*e03.z) + (e12.x*e12.x + e12.y*e12.y + e12.z*e12.z)
252 |                        + (e13.x*e13.x + e13.y*e13.y + e13.z*e13.z) + (e23.x*e23.x + e23.y*e23.y + e23.z*e23.z);
253 |     const float l_rms2    = l2_sum * (1.0f / 6.0f);
254 |     const float reg_denom = __fmaf_rn(V_T, V_T, (sigma_0 * sigma_0) * (l_rms2 * l_rms2 * l_rms2));
255 | 
256 |     const float inv_36V_reg = (V_T / fmaxf(reg_denom, 1e-16f)) * (1.0f / 36.0f) * J_piola;
257 |     out_warped_vol = fmaxf(V_T * J_piola, 1e-5f);
258 | 
259 |     // Positive weight floor gives a semidefinite graph Laplacian; SPD additionally needs boundary/gauge conditions.
260 |     const float spd_floor = 1e-3f * sqrtf(l_rms2);
261 |     const float dot_N2_N3 = -(N2.x * N3.x + N2.y * N3.y + N2.z * N3.z);
262 |     const float dot_N1_N3 = -(N1.x * N3.x + N1.y * N3.y + N1.z * N3.z);
263 |     const float dot_N1_N2 = -(N1.x * N2.x + N1.y * N2.y + N1.z * N2.z);
264 |     const float dot_N0_N3 = -(N0.x * N3.x + N0.y * N3.y + N0.z * N3.z);
265 |     const float dot_N0_N2 = -(N0.x * N2.x + N0.y * N2.y + N0.z * N2.z);
266 |     const float dot_N0_N1 = -(N0.x * N1.x + N0.y * N1.y + N0.z * N1.z);
267 | 
268 |     out_w01 = fmaxf(dot_N2_N3 * inv_36V_reg, spd_floor);
269 |     out_w02 = fmaxf(dot_N1_N3 * inv_36V_reg, spd_floor);
270 |     out_w03 = fmaxf(dot_N1_N2 * inv_36V_reg, spd_floor);
271 |     out_w12 = fmaxf(dot_N0_N3 * inv_36V_reg, spd_floor);
272 |     out_w13 = fmaxf(dot_N0_N2 * inv_36V_reg, spd_floor);
273 |     out_w23 = fmaxf(dot_N0_N1 * inv_36V_reg, spd_floor);
274 | }
```

**`crates/prism-gpu/src/kernels/nhs_biprism_simplicial_tile.cuh:286-340`** (selector: symbol='__device__ __forceinline__ void evaluate_diagonal_dec_tet_weights_compensated\\('; sha256 `849da5dd1333fffd…`)

```cuda
286 | __device__ __forceinline__ void evaluate_diagonal_dec_tet_weights_compensated(
287 |     const BiprismCoordinateOwnerFfi& owner,const ContactShellSimplicialStateFfi* tile,
288 |     uint32_t v0,uint32_t v1,uint32_t v2,uint32_t v3,float sigma_input,
289 |     float& out_warped_vol,float& out_w01,float& out_w02,float& out_w03,
290 |     float& out_w12,float& out_w13,float& out_w23
291 | ) {
292 |     const double sigma_0=double(sigma_input);
293 |     const auto e01=bp_p2_coordinate_difference(owner,tile,v0,v1,BiprismCoordinatePlane::Current);
294 |     const auto e02=bp_p2_coordinate_difference(owner,tile,v0,v2,BiprismCoordinatePlane::Current);
295 |     const auto e03=bp_p2_coordinate_difference(owner,tile,v0,v3,BiprismCoordinatePlane::Current);
296 |     const auto e12=bp_p2_coordinate_difference(owner,tile,v1,v2,BiprismCoordinatePlane::Current);
297 |     const auto e13=bp_p2_coordinate_difference(owner,tile,v1,v3,BiprismCoordinatePlane::Current);
298 |     const auto e23=bp_p2_coordinate_difference(owner,tile,v2,v3,BiprismCoordinatePlane::Current);
299 |     const auto E01=bp_p2_coordinate_difference(owner,tile,v0,v1,BiprismCoordinatePlane::MeshReference);
300 |     const auto E02=bp_p2_coordinate_difference(owner,tile,v0,v2,BiprismCoordinatePlane::MeshReference);
301 |     const auto E03=bp_p2_coordinate_difference(owner,tile,v0,v3,BiprismCoordinatePlane::MeshReference);
302 |     const BiprismCoordinate3F64 N3 = {e01.y * e02.z - e01.z * e02.y, e01.z * e02.x - e01.x * e02.z, e01.x * e02.y - e01.y * e02.x};
303 |     const BiprismCoordinate3F64 N2 = {e03.y * e01.z - e03.z * e01.y, e03.z * e01.x - e03.x * e01.z, e03.x * e01.y - e03.y * e01.x};
304 |     const BiprismCoordinate3F64 N1 = {e02.y * e03.z - e02.z * e03.y, e02.z * e03.x - e02.x * e03.z, e02.x * e03.y - e02.y * e03.x};
305 |     const BiprismCoordinate3F64 N0 = {e13.y * e12.z - e13.z * e12.y, e13.z * e12.x - e13.x * e12.z, e13.x * e12.y - e13.y * e12.x};
306 | 
307 |     const double det6V = __fma_rn(e01.x, N1.x, __fma_rn(e01.y, N1.y, e01.z * N1.z));
308 |     const double det6V_ref = E01.x * (E02.y * E03.z - E02.z * E03.y)
309 |                           - E01.y * (E02.x * E03.z - E02.z * E03.x)
310 |                           + E01.z * (E02.x * E03.y - E02.y * E03.x);
311 | 
312 |     const double V_T     = fabs(det6V) * double(1.0f / 6.0f);
313 |     const double V_ref   = fmax(fabs(det6V_ref) * double(1.0f / 6.0f), double(1e-5f));
314 |     const double J_piola = fmin(fmax(V_T / V_ref, double(0.05f)), double(20.0f));
315 | 
316 |     const double l2_sum = (e01.x*e01.x + e01.y*e01.y + e01.z*e01.z) + (e02.x*e02.x + e02.y*e02.y + e02.z*e02.z)
317 |                        + (e03.x*e03.x + e03.y*e03.y + e03.z*e03.z) + (e12.x*e12.x + e12.y*e12.y + e12.z*e12.z)
318 |                        + (e13.x*e13.x + e13.y*e13.y + e13.z*e13.z) + (e23.x*e23.x + e23.y*e23.y + e23.z*e23.z);
319 |     const double l_rms2    = l2_sum * double(1.0f / 6.0f);
320 |     const double reg_denom = __fma_rn(V_T, V_T, (sigma_0 * sigma_0) * (l_rms2 * l_rms2 * l_rms2));
321 | 
322 |     const double inv_36V_reg = (V_T / fmax(reg_denom, double(1e-16f))) * double(1.0f / 36.0f) * J_piola;
323 |     out_warped_vol = fmax(V_T * J_piola, double(1e-5f));
324 | 
325 |     // Positive weight floor gives a semidefinite graph Laplacian; SPD additionally needs boundary/gauge conditions.
326 |     const double spd_floor = double(1e-3f) * sqrt(l_rms2);
327 |     const double dot_N2_N3 = -(N2.x * N3.x + N2.y * N3.y + N2.z * N3.z);
328 |     const double dot_N1_N3 = -(N1.x * N3.x + N1.y * N3.y + N1.z * N3.z);
329 |     const double dot_N1_N2 = -(N1.x * N2.x + N1.y * N2.y + N1.z * N2.z);
330 |     const double dot_N0_N3 = -(N0.x * N3.x + N0.y * N3.y + N0.z * N3.z);
331 |     const double dot_N0_N2 = -(N0.x * N2.x + N0.y * N2.y + N0.z * N2.z);
332 |     const double dot_N0_N1 = -(N0.x * N1.x + N0.y * N1.y + N0.z * N1.z);
333 | 
334 |     out_w01 = fmax(dot_N2_N3 * inv_36V_reg, spd_floor);
335 |     out_w02 = fmax(dot_N1_N3 * inv_36V_reg, spd_floor);
336 |     out_w03 = fmax(dot_N1_N2 * inv_36V_reg, spd_floor);
337 |     out_w12 = fmax(dot_N0_N3 * inv_36V_reg, spd_floor);
338 |     out_w13 = fmax(dot_N0_N2 * inv_36V_reg, spd_floor);
339 |     out_w23 = fmax(dot_N0_N1 * inv_36V_reg, spd_floor);
340 | }
```

**`crates/prism-gpu/src/kernels/nhs_biprism_bowyer_watson_seed.cuh:318-347`** (selector: symbol='BPBW_HD uint32_t bp_bw_peel\\('; sha256 `255d925e1c37fb03…`)

```cuda
318 | BPBW_HD uint32_t bp_bw_peel(BiprismBwScratchFfi& s,BiprismBwStatusFfi& r,uint32_t n,uint32_t target,uint32_t max_edges) {
319 |     uint32_t live=n,edges=0;for(uint32_t e=0;e<65536;++e)if(s.edge_degree[e])++edges;
320 |     r.unpeeled_tets=n;r.unpeeled_edges=edges;
321 |     for(uint32_t i=0;i<n;++i){s.order[i]=i;s.tetrahedra[0][i].radius2=bp_bw_radius2(s,s.tetrahedra[0][i]);
322 |         if(!isfinite(s.tetrahedra[0][i].radius2))return bp_bw_fail(r,BPBW_NONFINITE,i);}
323 |     // Stable descending radius with original scratch ID as the secondary key.
324 |     for(uint32_t i=1;i<n;++i){const uint32_t key=s.order[i];uint32_t j=i;
325 |         while(j>0&&s.tetrahedra[0][s.order[j-1]].radius2<s.tetrahedra[0][key].radius2){s.order[j]=s.order[j-1];--j;}s.order[j]=key;}
326 |     while(live>target||edges>max_edges){
327 |         // Refuse before starting another complete ordered scan at the ceiling.
328 |         if(r.peel_passes>=BPBW_MAX_PEEL_PASSES)return bp_bw_fail(r,BPBW_PEEL_BUDGET_UNREACHABLE);
329 |         bool progress=false;++r.peel_passes;
330 |         for(uint32_t j=0;j<n;++j){const uint32_t i=s.order[j];if(!s.active[i]||!bp_bw_can_peel(s,i))continue;
331 |             const BiprismBwTet t=s.tetrahedra[0][i];s.active[i]=0;--live;++r.removed_tets;progress=true;
332 |             for(uint32_t a=0;a<4;++a){--s.vertex_degree[bp_bw_vertex(t,a)];for(uint32_t b=a+1;b<4;++b){const uint32_t e=bp_bw_edge(bp_bw_vertex(t,a),bp_bw_vertex(t,b));if(--s.edge_degree[e]==0)--edges;}}
333 |             for(uint32_t f=0;f<4;++f){uint16_t a,b,c;bp_bw_face(t,f,a,b,c);const uint32_t slot=bp_bw_face_slot(s,bp_bw_face_key(a,b,c),false);
334 |                 if(slot==BPBW_NONE||s.faces[slot].count==0)return bp_bw_fail(r,BPBW_INVALID_COMPLEX,i);
335 |                 BiprismBwFace& z=s.faces[slot];const uint32_t opp=s.adjacency[i][f];
336 |                 if(z.count==1){bp_bw_boundary_change(s,z.key,-1);z.count=0;}
337 |                 else{bp_bw_boundary_change(s,z.key,1);z.count=1;z.owner0=(z.owner0>>2)==i?z.owner1:z.owner0;z.owner1=BPBW_NONE;
338 |                     if(opp==BPBW_NONE)return bp_bw_fail(r,BPBW_INVALID_COMPLEX,i);
339 |                     for(uint32_t of=0;of<4;++of)if(s.adjacency[opp][of]==i)s.adjacency[opp][of]=BPBW_NONE;}
340 |                 s.adjacency[i][f]=BPBW_NONE;
341 |             }
342 |             if(live<=target&&edges<=max_edges)break;
343 |         }
344 |         if(!progress)return bp_bw_fail(r,BPBW_PEEL_BUDGET_UNREACHABLE);
345 |     }
346 |     r.final_tets=live;r.final_edges=edges;return BPBW_OK;
347 | }
```

**`crates/prism-gpu/src/kernels/nhs_biprism_dec_pcg.cuh:77-177`** (selector: symbol='__device__ __forceinline__ bool biprism_dec_pcg_256_traversal\\('; sha256 `18dfce17d400d9b0…`)

```cuda
 77 | __device__ __forceinline__ bool biprism_dec_pcg_256_traversal(
 78 |     const uint8_t* edge_u, const uint8_t* edge_v, uint32_t edge_stride,
 79 |     const EdgeWeight& weights, uint32_t num_edges, const EdgeTraversal& traversal, double charge,
 80 |     PotentialScalar* phi, float* residual, float* preconditioned, float* direction,
 81 |     float* product, float* inverse_diagonal, float* reduction,
 82 |     float& out_residual, uint32_t& out_iters,
 83 |     double internal_tol_sq = static_cast<double>(BIPRISM_DEC_TOL_SQ)
 84 | ) {
 85 |     using PotentialValue = typename std::remove_cv<PotentialScalar>::type;
 86 |     static_assert(std::is_same<PotentialValue,float>::value
 87 |         ||std::is_same<PotentialValue,double>::value,"DEC potential must be FP32 or retained FP64");
 88 |     const uint32_t tid = threadIdx.x;
 89 |     const bool core = tid < TILE_CORE_VERTICES;
 90 |     out_residual = CUDART_INF_F;
 91 |     out_iters = 0u;
 92 |     // An internal solve target may be tighter than the external admission
 93 |     // ceiling. It may never relax that ceiling or change the iteration cap.
 94 |     bool valid = isfinite(internal_tol_sq) && internal_tol_sq > 0.0
 95 |         && internal_tol_sq <= static_cast<double>(BIPRISM_DEC_TOL_SQ)
 96 |         && isfinite(phi[tid]) && (!core || isfinite(charge));
 97 |     #pragma unroll 1
 98 |     for (uint32_t e = tid; e < num_edges; e += 256u) {
 99 |         valid = valid && isfinite(weights(e)) && weights(e) > 0.0f
100 |             && edge_u[e * edge_stride] != edge_v[e * edge_stride];
101 |     }
102 |     if (!__syncthreads_and(valid)) return false;
103 | 
104 |     double diagonal = 1e-4;
105 |     if (core) {
106 |         #pragma unroll 1
107 |         for (uint32_t cursor = traversal.begin(tid); cursor < traversal.end(tid); ++cursor) {
108 |             const uint32_t e = traversal.edge(cursor);
109 |             if (edge_u[e * edge_stride] == tid || edge_v[e * edge_stride] == tid)
110 |                 diagonal += static_cast<double>(weights(e));
111 |         }
112 |     }
113 |     inverse_diagonal[tid] = core ? static_cast<float>(1.0 / diagonal) : 0.0f;
114 |     double true_r = core ? static_cast<double>(charge)
115 |         - biprism_dec_row_f64_traversal(edge_u, edge_v, edge_stride, weights, traversal, phi, tid) : 0.0;
116 |     if (!__syncthreads_and(isfinite(true_r) && fabs(true_r) <= static_cast<double>(FLT_MAX))) return false;
117 |     double true_rr = biprism_dec_sum_f64_256(true_r * true_r, product);
118 |     if (!isfinite(true_rr)) return false;
119 |     out_residual = static_cast<float>(sqrt(true_rr));
120 |     if (true_rr <= internal_tol_sq) return true;
121 |     residual[tid] = static_cast<float>(true_r);
122 |     preconditioned[tid] = inverse_diagonal[tid] * residual[tid];
123 |     direction[tid] = preconditioned[tid];
124 |     __syncthreads();
125 |     float rr = cta_dot_product_256(residual[tid] * residual[tid], reduction);
126 |     float rz = cta_dot_product_256(residual[tid] * preconditioned[tid], reduction);
127 |     if (!isfinite(rr) || !isfinite(rz)) return false;
128 | 
129 |     #pragma unroll 1
130 |     for (uint32_t iter = 0u; iter < BIPRISM_DEC_MAX_ITERS; ++iter) {
131 |         out_iters = iter + 1u;
132 |         const double ap = biprism_dec_row_f64_traversal(edge_u, edge_v, edge_stride, weights, traversal, direction, tid);
133 |         if (!__syncthreads_and(isfinite(ap) && fabs(ap) <= static_cast<double>(FLT_MAX))) return false;
134 |         product[tid] = static_cast<float>(ap);
135 |         const float pap = cta_dot_product_256(direction[tid] * product[tid], reduction);
136 |         if (!(pap > 0.0f) || !(rz > 0.0f) || !isfinite(pap) || !isfinite(rz)) return false;
137 |         const float alpha = rz / pap;
138 |         if (!isfinite(alpha)) return false;
139 |         if (core) {
140 |             if constexpr (std::is_same<PotentialValue,float>::value) {
141 |                 // Keep the historical FP32 update and its contraction exact.
142 |                 phi[tid] = __fmaf_rn(alpha, direction[tid], phi[tid]);
143 |             } else {
144 |                 // Retain all potential bits through residual/work and commit.
145 |                 phi[tid] = __fma_rn(static_cast<double>(alpha),
146 |                     static_cast<double>(direction[tid]),static_cast<double>(phi[tid]));
147 |             }
148 |             residual[tid] = __fmaf_rn(-alpha, product[tid], residual[tid]);
149 |         }
150 |         __syncthreads();
151 |         if (!__syncthreads_and(isfinite(phi[tid]) && isfinite(residual[tid]))) return false;
152 |         rr = cta_dot_product_256(residual[tid] * residual[tid], reduction);
153 |         if (!isfinite(rr)) return false;
154 |         const bool restart = (out_iters % BIPRISM_DEC_RESTART_INTERVAL == 0u)
155 |             || static_cast<double>(rr) <= internal_tol_sq || out_iters == BIPRISM_DEC_MAX_ITERS;
156 |         if (restart) {
157 |             true_r = core ? static_cast<double>(charge)
158 |                 - biprism_dec_row_f64_traversal(edge_u, edge_v, edge_stride, weights, traversal, phi, tid) : 0.0;
159 |             if (!__syncthreads_and(isfinite(true_r) && fabs(true_r) <= static_cast<double>(FLT_MAX))) return false;
160 |             true_rr = biprism_dec_sum_f64_256(true_r * true_r, product);
161 |             if (!isfinite(true_rr)) return false;
162 |             residual[tid] = static_cast<float>(true_r);
163 |             rr = cta_dot_product_256(residual[tid] * residual[tid], reduction);
164 |             if (!isfinite(rr)) return false;
165 |             out_residual = static_cast<float>(sqrt(true_rr));
166 |             if (true_rr <= internal_tol_sq) return true;
167 |         }
168 |         preconditioned[tid] = inverse_diagonal[tid] * residual[tid];
169 |         const float next_rz = cta_dot_product_256(residual[tid] * preconditioned[tid], reduction);
170 |         if (!(next_rz > 0.0f) || !isfinite(next_rz)) return false;
171 |         const float beta = restart ? 0.0f : next_rz / rz;
172 |         direction[tid] = core ? __fmaf_rn(beta, direction[tid], preconditioned[tid]) : 0.0f;
173 |         rz = next_rz;
174 |         __syncthreads();
175 |     }
176 |     return false;
177 | }
```

**`crates/prism-nhs/src/biprism_mesh.rs:612-671`** (selector: symbol='pub fn validate_tile_geometry_compensated\\('; sha256 `814e718daaa4e665…`)
  
_Excerpt truncated at max_lines=60; the full symbol spans lines 612-819._

```rust
612 | pub fn validate_tile_geometry_compensated(bytes:&[u8],points:&[[f32;3];NV],low:&[[f32;3];NV],
613 |     atom_count:u32,constraints:&[ConstraintGroup],target_tets:u32)->Result<MeshAdmission,String>{
614 |     if bytes.len() != TILE_BYTES
615 |         || target_tets == 0
616 |         || target_tets > 1024
617 |         || atom_count < 256
618 |         || atom_count > 0x0100_0000
619 |     {
620 |         return Err("tile/input shape outside supported ABI".into());
621 |     }
622 |     let nt = u32_at(bytes, COUNTS_OFFSET) as usize;
623 |     let ne = u32_at(bytes, COUNTS_OFFSET + 4) as usize;
624 |     if nt == 0 || nt > target_tets as usize || nt > 1024 || ne == 0 || ne > 1536 {
625 |         return Err("active tile capacity/count refusal".into());
626 |     }
627 |     let ids: [u32; NV] = std::array::from_fn(|v| u32_at(bytes, v * 4) & 0x00ff_ffff);
628 |     if ids.iter().any(|&a| a >= atom_count)
629 |         || ids.iter().copied().collect::<BTreeSet<_>>().len() != NV
630 |     {
631 |         return Err("tile atom identity duplicated or out of bounds".into());
632 |     }
633 |     check_closure(&ids, atom_count as usize, constraints)?;
634 |     for v in 0..NV {
635 |         for k in 0..3 {
636 |             if u32_at(bytes, 1024 + v * 16 + k * 4) != points[v][k].to_bits() {
637 |                 return Err(format!(
638 |                     "tile reference coordinate differs from captured input at{v}/{k}"
639 |                 ));
640 |             }
641 |         }
642 |     }
643 |     let p = exact_points_compensated(points,low)?;
644 |     let mut edge_ids = BTreeMap::new();
645 |     for e in 0..ne {
646 |         let ep = [
647 |             bytes[EDGE_OFFSET + 2 * e] as u16,
648 |             bytes[EDGE_OFFSET + 2 * e + 1] as u16,
649 |         ];
650 |         if ep[0] >= ep[1] || edge_ids.insert(ep, e).is_some() {
651 |             return Err(format!("invalid or duplicate edge{e}"));
652 |         }
653 |     }
654 |     let mut tets = Vec::with_capacity(nt);
655 |     let mut unique = BTreeSet::new();
656 |     let mut faces: BTreeMap<Face, Vec<(usize, usize)>> = BTreeMap::new();
657 |     let mut links: BTreeMap<Edge, Vec<Edge>> = BTreeMap::new();
658 |     let mut stars: Vec<Vec<Face>> = vec![Vec::new(); NV];
659 |     let mut geometry = Vec::with_capacity(nt);
660 |     let mut sphere_tests = 0usize;
661 |     let mut volume = 0.0;
662 |     for ti in 0..nt {
663 |         let t: Tet = std::array::from_fn(|k| bytes[TET_OFFSET + ti * 4 + k] as u16);
664 |         let mut key = t;
665 |         key.sort();
666 |         if key.windows(2).any(|w| w[0] == w[1]) || !unique.insert(key) {
667 |             return Err(format!("degenerate or duplicate tetrahedron{ti}"));
668 |         }
669 |         let (det, numer) = sphere_data(&p, t);
670 |         if det.sign <= 0 {
671 |             return Err(format!("nonpositive exact orientation at tetrahedron{ti}"));
```

### Mathematics

Face area normals of tetrahedron \(T=(p_0,p_1,p_2,p_3)\): \(N_i\) is the inward area normal (\(|N_i| = 2A_i\)) of the face opposite \(p_i\); \(\sum_i N_i = 0\).
P1 basis gradients: \(\nabla\phi_i = N_i/(6V)\). Stiffness: \(K_{ij} = V\,\nabla\phi_i\cdot\nabla\phi_j = N_i\cdot N_j/(36V)\).
Edge weight (cotangent form): \(w_{ij} = -K_{ij} = -\dfrac{N_i\cdot N_j}{36V} = \tfrac16\,\ell_{kl}\cot\theta_{kl}\), where \(\{k,l\}\) is the opposite edge and \(\theta_{kl}\) the dihedral angle there.
Correct mapping: \(w_{01}\leftarrow N_0\cdot N_1,\ w_{02}\leftarrow N_0\cdot N_2,\ w_{03}\leftarrow N_0\cdot N_3,\ w_{12}\leftarrow N_1\cdot N_2,\ w_{13}\leftarrow N_1\cdot N_3,\ w_{23}\leftarrow N_2\cdot N_3\).
Regularisation in source: \(1/(36V) \to V/(V^2+\sigma^2\ell_{rms}^6)/36 \cdot J\), with Piola ratio \(J = \mathrm{clamp}(V/V_{ref}, 0.05, 20)\) and floor \(w \ge 10^{-3}\ell_{rms}\).

### Units

| Quantity | Unit |
|---|---|
| coordinates, edge length | Angstrom (bohr in v10 working planes) |
| V | Angstrom^3 |
| w_ij | Angstrom (length; dual-area / primal-length) |
| PCG potential phi | no physical scale today (no 4 pi, vacuum) |

### Invariants

- Row sums of the assembled Laplacian are zero before the SPD floor and gauge are applied.
- Each edge weight uses the two face normals whose faces share that edge.
- Mesh admission is exact (1536-bit Rust re-admission) and refuses negative controls.

### Known defects

| ID | Defect | Where | Status | Fix package | Check |
|---|---|---|---|---|---|
| D-DEC-OPP-EDGE | CRITICAL: DEC edge weight assigned to the opposite edge (01<->23, 02<->13, 03<->12) | nhs_biprism_simplicial_tile.cuh evaluate_diagonal_dec_tet_weights (FP32 and compensated): out_w01..out_w23 assignments | **OPEN** | P3 (first, under change control) | dec_p1fem_edge_weight_mapping |
| D-DEC-NO-SCALE | PCG solves the permuted operator with no physical scale (no 4 pi, vacuum, halo q/4pi carries units of charge) | nhs_biprism_dec_pcg.cuh | **OPEN** | P3 / P9 |  |
| D-BW-PEEL-CAVITY | BW mesh is not a whole-protein lattice: 256-atom single-thread tiles, <= 8 leaves, <= 1024 exported tets; the peel removes large-circumradius cavity tets | nhs_biprism_bowyer_watson_seed.cuh bp_bw_peel | **OPEN** | P7.3 |  |
| D-DUAL-NAMING | The dual is barycentric, not Voronoi; naming is wrong | nhs_biprism_simplicial_tile.cuh | **OPEN** | P3 |  |

### Qualifying gate

Independent exact star0/star1 recompute on non-regular tetrahedra (dec_p1fem_edge_weight_mapping check) under change control; requalification of every DEC/LIF/v10 receipt on the corrected operator; observables invariant across three mesh resolutions.

### Mandatory reviewer questions

1. Derive w01 from the face normals N0..N3 as defined in evaluate_diagonal_dec_tet_weights and compare it with the out_w01 assignment line. Which edge does dot_N2_N3 belong to?
2. Show that the faces opposite vertices 2 and 3 share edge 01 or not, and conclude which edge's dihedral angle dot_N2_N3 measures.
3. Confirm that the FP32 and the compensated (FP64) variants carry the same mapping, and list every other kernel that consumes out_w01..out_w23.
4. State the units of inv_36V_reg and of the spd_floor; are they the same?
5. What does the circumradius peel in bp_bw_peel remove, and what does that imply for cavity (pocket) tetrahedra?

## S09. Electronic propagator (Cayley/Magnus) and the tight-binding coupling

**Layer status:** `NOT_BUILT`

The in-loop hybrid electronic layer (H1-H3) is NOT BUILT on the run-once path. A two-level s/p
pseudo-spin per atom propagated by a Cayley map exists only inside the v10 192-atom replicated tile
(about 7 days/ns; stalls at epoch 5; refuted as the run-once integrator). The Cayley map with 4
substeps at 4 fs compresses a 0.38 Ha onsite gap to an effective ~0.070 Ha. Parameters have no
provenance and onsite/edge terms mix Hartree*bohr^3 with Hartree. The tight-binding (GFN2-xTB form)
parameter and validation source (P8/H4) and the charge-conserving coupling (H2) are not built.

### Source excerpts (verbatim)

**`crates/prism-gpu/src/kernels/nhs_biprism_electronic_discrete.cuh:191-231`** (selector: symbol='bp_ed_hermitian_cayley\\(BiprismElectronicSpinor'; sha256 `a416c303a31602bc…`)

```cuda
191 | }
192 | // Scaled direct solution of (omega I+i dt H/2)psi1=(omega I-i dt H/2)psi0.
193 | // No eigenbasis rotation, renormalization, determinant floor or hidden retry.
194 | // false means unsupported/nonfinite arithmetic; the owner must fault/refuse.
195 | // A true result still carries equation/norm diagnostics for owner admission.
196 | template<class R> __host__ __device__ __forceinline__ bool
197 | bp_ed_hermitian_cayley(BiprismElectronicSpinor<R> p0,BiprismElectronicHermitian<R> H,
198 |     R omega,R dt,BiprismElectronicCayleyResult<R>& out) {
199 |     out={p0,R(0),R(0)};
200 |     if(!bp_ed_finite(p0)||!isfinite(H.h00)||!isfinite(H.h11)||!isfinite(H.h01_re)||!isfinite(H.h01_im)
201 |         ||!isfinite(omega)||!(omega>R(0))||!isfinite(dt)) return false;
202 |     const R n0=bp_ed_norm2(p0.s0)+bp_ed_norm2(p0.s1);
203 |     if(!isfinite(n0))return false;
204 |     const R a=R(.5)*dt*H.h00,d=R(.5)*dt*H.h11,tr=R(.5)*dt*H.h01_re,ti=R(.5)*dt*H.h01_im;
205 |     if(!isfinite(a)||!isfinite(d)||!isfinite(tr)||!isfinite(ti))return false;
206 |     const R scale=bp_ed_max(omega,bp_ed_max(bp_ed_abs(a),bp_ed_max(bp_ed_abs(d),bp_ed_max(bp_ed_abs(tr),bp_ed_abs(ti)))));
207 |     const R w=omega/scale,aa=a/scale,dd=d/scale,rr=tr/scale,ii=ti/scale;
208 |     const BiprismElectronicComplex<R> A={w,aa},B={-ii,rr},C={ii,rr},D={w,dd};
209 |     const BiprismElectronicComplex<R> det=bp_ed_sub(bp_ed_mul(A,D),bp_ed_mul(B,C));
210 |     const R ds=bp_ed_max(bp_ed_abs(det.re),bp_ed_abs(det.im));
211 |     if(!isfinite(ds)||!(ds>R(0)))return false;
212 |     const R dr=det.re/ds,di=det.im/ds,den=bp_ed_fma(dr,dr,di*di);
213 |     const auto b0=bp_ed_scale(p0.s0,w),b1=bp_ed_scale(p0.s1,w);
214 |     const auto n_0=bp_ed_sub(bp_ed_mul(D,b0),bp_ed_mul(B,b1));
215 |     const auto n_1=bp_ed_sub(bp_ed_mul(A,b1),bp_ed_mul(C,b0));
216 |     const R n0r=n_0.re/ds,n0i=n_0.im/ds,n1r=n_1.re/ds,n1i=n_1.im/ds;
217 |     const BiprismElectronicComplex<R> y0={bp_ed_fma(n0r,dr,n0i*di)/den,bp_ed_fma(n0i,dr,-n0r*di)/den};
218 |     const BiprismElectronicComplex<R> y1={bp_ed_fma(n1r,dr,n1i*di)/den,bp_ed_fma(n1i,dr,-n1r*di)/den};
219 |     const BiprismElectronicSpinor<R> p1={bp_ed_sub(bp_ed_scale(y0,R(2)),p0.s0),bp_ed_sub(bp_ed_scale(y1,R(2)),p0.s1)};
220 |     if(!bp_ed_finite(p1))return false;
221 |     const auto sum0=bp_ed_add(p1.s0,p0.s0),sum1=bp_ed_add(p1.s1,p0.s1);
222 |     const auto delta0=bp_ed_sub(p1.s0,p0.s0),delta1=bp_ed_sub(p1.s1,p0.s1);
223 |     const auto hs0=bp_ed_add(bp_ed_scale(sum0,aa),bp_ed_mul(BiprismElectronicComplex<R>{rr,ii},sum1));
224 |     const auto hs1=bp_ed_add(bp_ed_mul(BiprismElectronicComplex<R>{rr,-ii},sum0),bp_ed_scale(sum1,dd));
225 |     const auto res0=bp_ed_add(bp_ed_scale(delta0,w),BiprismElectronicComplex<R>{-hs0.im,hs0.re});
226 |     const auto res1=bp_ed_add(bp_ed_scale(delta1,w),BiprismElectronicComplex<R>{-hs1.im,hs1.re});
227 |     out.psi_np1=p1;
228 |     out.norm_defect=bp_ed_abs(bp_ed_norm2(p1.s0)+bp_ed_norm2(p1.s1)-n0);
229 |     out.scaled_equation_residual=bp_ed_max(bp_ed_max(bp_ed_abs(res0.re),bp_ed_abs(res0.im)),bp_ed_max(bp_ed_abs(res1.re),bp_ed_abs(res1.im)));
230 |     return isfinite(out.norm_defect)&&isfinite(out.scaled_equation_residual);
231 | }
```

**`crates/prism-gpu/src/kernels/nhs_biprism_unit_bridge.cuh:1-18`** (selector: lines=[1, 18]; sha256 `de47488e2cd3e678…`)

```cuda
 1 | #pragma once
 2 | #include "nhs_biprism_variational_so3g_v10.cuh"
 3 | #include "nhs_biprism_coordinate_precision.cuh"
 4 | 
 5 | // Native engine: Angstrom, ps, kcal/mol/Angstrom. v10 working planes:
 6 | // bohr, atomic time, Hartree/bohr. Coefficients are explicit numerical-model
 7 | // inputs; this bridge does not establish a GFN parameterization or an energy gate.
 8 | // CODATA2022 factors are independently bound in MASS_AND_UNIT_PREFLIGHT.json.
 9 | static constexpr double BIPRISM_ANGSTROM_TO_BOHR = 1.8897261259077824;
10 | static constexpr double BIPRISM_NATIVE_VELOCITY_TO_AU = 4.571028907499172e-5;
11 | static constexpr double BIPRISM_NATIVE_FORCE_TO_AU = 0.0008432975634897884;
12 | static constexpr double BIPRISM_PS_TO_ATOMIC_TIME = 41341.37333517015;
13 | static constexpr uint32_t BIPRISM_BRIDGE_MAGIC = 0x42554231u;
14 | static constexpr uint32_t BIPRISM_BRIDGE_DOMAIN = 1u;
15 | static constexpr uint32_t BIPRISM_BRIDGE_EPOCH = 2u;
16 | static constexpr uint32_t BIPRISM_BRIDGE_NONFINITE = 4u;
17 | static constexpr uint32_t BIPRISM_BRIDGE_CONVERGENCE = 8u;
18 | static constexpr uint32_t BIPRISM_BRIDGE_TERMINAL = 0x40000000u;
```

### Planned components (NOT_BUILT)

| Component | Package | Status |
|---|---|---|
| H1 in-loop analogue node in the AMBER BpNativeGeometry<true> step graph (closed-form SU(2), commutator-free Magnus, interaction picture) | H1 | NOT_BUILT |
| H2 charge-conserving electronic coupling contract (observe default; W_electronic planes) | H2 | NOT_BUILT |
| H3 physical electronic event ontology | H3 | NOT_BUILT |
| P8/H4 GPU SCC tight-binding (GFN2-xTB form) with tblite reference | P8 / H4 | NOT_BUILT |

### Mathematics

Cayley step (source): \((\omega I + \tfrac{i h}{2}H)\psi_1 = (\omega I - \tfrac{i h}{2}H)\psi_0\). For \(\omega=1\) this is unitary with eigenphase \(\theta = 2\arctan(Eh/2)\) instead of the exact \(Eh\).
Effective gap after \(n\) substeps of \(h=\Delta t/n\): \(\Delta_{eff} = \tfrac{2}{h}\arctan(\Delta h/2)\).
At \(\Delta=0.38\) Ha, \(\Delta t = 4\) fs \(=165.36\) a.u., \(n=4\): \(\Delta_{eff}\approx 0.070\) Ha.
Unit bridge: Angstrom -> bohr \(=1/a_0\); ps -> a.u. time \(=10^{-12}/t_{au}\).

### Units

| Quantity | Unit |
|---|---|
| H, onsite, couplings | Hartree (onsite currently Hartree*bohr^3: unit defect) |
| h (electronic substep) | atomic time units (1 ps = 41341.37 a.u.) |
| psi | dimensionless, normalised |

### Invariants

- Norm conservation |psi_1| = |psi_0| (unitarity) to FP64 round-off.
- Electronic substeps are decoupled from nuclear dt (interaction picture) once H1 lands; until then dt changes the effective Hamiltonian.
- Total charge is conserved by the coupling (dq_u = -dq_v).

### Known defects

| ID | Defect | Where | Status | Fix package | Check |
|---|---|---|---|---|---|
| D-ELEC-CAYLEY-GAP | Cayley with 4 substeps compresses the 0.38 Ha gap to ~0.070 Ha at 4 fs (5.4x, ~29x inflated off-resonant mixing); dt changes the effective Hamiltonian | nhs_biprism_electronic_discrete.cuh bp_ed_hermitian_cayley (v10 callers) | **OPEN** | H1 (interaction-picture Cayley, decoupled microsteps) | cayley_vs_su2_phase |
| D-ANALOGUE-UNITS | Analogue parameters have no provenance; onsite energy in Hartree*bohr^3 against edge terms in Hartree | nhs_biprism_unit_bridge.cuh; nhs_biprism_variational_so3g_v10.cuh | **OPEN** | P8 / H1 |  |
| D-CHARGE-CREATE | Charge is created (q = q_amber + 0.15*rho11), not moved; electrostatics double-counted against AMBER | v10 / native coupling | **OPEN** | H2 | coupling_charge_conservation |
| D-CHARGE-UV-NULL-FALLBACK | uv_excited_charge falls back silently when the excited-charge set is null instead of refusing | nhs_amber_fused.cu uv_excited_charge / uv_charge_for_state | **OPEN** | H2 | charge.uv_excited_charge_null_fallback |
| D-PME-BYPASS | The PME/Ewald path deliberately bypasses the electronic charge rescale; any Ewald solvent decouples the electronics until a charge-plane writer with Hamiltonian-work accounting exists | nhs_amber_fused.cu (lambda_v2_ewald_enabled branches) | **OPEN** | H2 / P8 |  |
| D-V10-NONDET | v10 H assembly uses nondeterministic FP64 atomicAdd; identical replicas diverge in epoch 1 | nhs_biprism_variational_so3g_v10.cuh | **OPEN** | H0 / P4 |  |

### Qualifying gate

H1: Rabi phase <= 1e-3 rad/ps and Landau-Zener populations <= 1e-4 at 0.5/2/4 fs (FP64 analytic controls); observe mode bitwise-identical nuclear trajectory analogue on/off. H2: charge conserved <= 1e-12 e over 1e5 steps; coupled NVE drift no worse than MM-only. H4/P8: force RMSE and energy MAE vs GFN2-xTB (tblite) on a non-target panel.

### Mandatory reviewer questions

1. From bp_ed_hermitian_cayley, derive the eigenphase of the update for a diagonal H with omega = 1 and compare it with the exact phase at h = 41.34 a.u. (cayley_vs_su2_phase check).
2. What value is passed as omega in the v10 call sites, and what are its units? Is the map still unitary when omega != 1?
3. Check BIPRISM_ANGSTROM_TO_BOHR and BIPRISM_PS_TO_ATOMIC_TIME against CODATA 2022 (unit_conversions check). Which CODATA vintage do they match?
4. Show the line that forms q = q_amber + 0.15*rho11 (or its successor) and state whether total charge is conserved.

## S10. LIF / Voronoi detector and event extraction

**Layer status:** `EXECUTED_UNQUALIFIED`

Leaky integrate-and-fire neurons on mesh vertices driven by DEC edge flux; spikes are the event
substrate. Flux accumulation uses FP32 shared-memory atomicAdd (order nondeterministic), the leaf
count is hard-coded to 8 (r < 8u), and it consumes the DEC weights from S08 (opposite-edge defect).
Refractory period and UV LIF window are counted in steps, so causal windows change with dt.

### Source excerpts (verbatim)

**`crates/prism-gpu/src/kernels/nhs_biprism_voronoi_lif.cuh:134-148`** (selector: symbol='__device__ __forceinline__ float biprism_live_diagonal_edge\\('; sha256 `306032ad98cdf52c…`)

```cuda
134 | __device__ __forceinline__ float biprism_live_diagonal_edge(
135 |     const ContactShellSimplicialStateFfi* tile, const BiprismCoordinateOwnerFfi& coordinates,
136 |     uint32_t e, float3& grad_u
137 | ) {
138 |     const uint8_t u = tile->edge_u8[e][0], v = tile->edge_u8[e][1];
139 |     const uint32_t gu = UNPACK_GLOBAL_ATOM_ID(tile->global_atom_idx[u]);
140 |     const uint32_t gv = UNPACK_GLOBAL_ATOM_ID(tile->global_atom_idx[v]);
141 |     const auto d=bp_coord_difference(bp_coord_load(coordinates,BiprismCoordinatePlane::Current,gv),
142 |         bp_coord_load(coordinates,BiprismCoordinatePlane::Current,gu));
143 |     const auto ref=bp_coord_difference(bp_coord_load(coordinates,BiprismCoordinatePlane::MeshReference,gv),
144 |         bp_coord_load(coordinates,BiprismCoordinatePlane::MeshReference,gu));
145 |     const float w_live=bp_geometry_live_weight(tile->star1_wt[e],d,ref);
146 |     grad_u=bp_geometry_live_weight_gradient(w_live,d);
147 |     return w_live;
148 | }
```

**`crates/prism-gpu/src/kernels/nhs_biprism_voronoi_lif.cuh:675-685`** (selector: anchor='const float flux = w_live \\* \\(s_v_mem\\[v\\] - s_v_mem\\[u\\]\\);'; sha256 `306032ad98cdf52c…`)

```cuda
675 |     for (uint32_t e = tid; e < num_edges; e += 256) {
676 |         const uint8_t u = tile->edge_u8[e][0];
677 |         const uint8_t v = tile->edge_u8[e][1];
678 |         float3 grad_unused;
679 |         const float w_live = biprism_live_diagonal_edge(tile, coordinates, e, grad_unused);
680 |         const float flux = w_live * (s_v_mem[v] - s_v_mem[u]);
681 |         if (!isfinite(flux)) atomicOr(&s_input_fault, 1u);
682 |         if (u < TILE_CORE_VERTICES) atomicAdd(&s_syn_flux[u],  flux);
683 |         if (v < TILE_CORE_VERTICES) atomicAdd(&s_syn_flux[v], -flux);
684 |     }
685 |     __syncthreads();
```

**`crates/prism-gpu/src/kernels/nhs_amber_fused.cu:105-105`** (selector: anchor='#define REFRACTORY_STEPS'; sha256 `46287f27d2c49231…`)

```cuda
105 | #define REFRACTORY_STEPS 250       // 250 steps * 0.004ps (HMR) = 1.0ps refractory period
```

**`crates/prism-nhs/src/sdst_bridge.rs:55-94`** (selector: symbol='pub fn select_native_percentile_indices\\('; sha256 `3639178df54984f0…`)
  
_Excerpt truncated at max_lines=40; the full symbol spans lines 55-123._

```rust
55 | pub fn select_native_percentile_indices(
56 |     spikes: &[GpuSpikeEvent],
57 |     spike_percentile: u32,
58 | ) -> (Vec<usize>, Vec<NativeChannelSelectionReceipt>) {
59 |     let whole_bypass = spikes.len() <= 1000;
60 |     let pct_f = (spike_percentile.min(99) as f32) / 100.0;
61 |     let channel = |source| match source {
62 |         1 => 0,
63 |         3 => 2,
64 |         4 | 5 => 3,
65 |         6 => 4,
66 |         _ => 1,
67 |     };
68 |     let names = ["UV", "LIF", "EFP", "LADD_COFIRE", "VORONOI_LIF"];
69 |     if whole_bypass {
70 |         let mut counts = [0usize; 5];
71 |         for spike in spikes {
72 |             counts[channel(spike.spike_source)] += 1;
73 |         }
74 |         let receipts = (0..5)
75 |             .map(|i| NativeChannelSelectionReceipt {
76 |                 channel: names[i],
77 |                 input_count: counts[i],
78 |                 selected_count: counts[i],
79 |                 threshold: None,
80 |                 threshold_raw_f32_bits: None,
81 |                 rule: "whole_population_len_le_1000_preserved",
82 |             })
83 |             .collect();
84 |         return ((0..spikes.len()).collect(), receipts);
85 |     }
86 |     let mut channels: [Vec<usize>; 5] = std::array::from_fn(|_| Vec::new());
87 |     for (index, spike) in spikes.iter().enumerate() {
88 |         channels[channel(spike.spike_source)].push(index);
89 |     }
90 |     let mut selected = Vec::new();
91 |     let mut receipts = Vec::new();
92 |     for (i, indices) in channels.into_iter().enumerate() {
93 |         let input_count = indices.len();
94 |         let (threshold, rule) = if i == 3 {
```

### Mathematics

\(\tau\,\dot V_u = -(V_u - V_{rest}) + \sum_{v\sim u} w_{uv}(V_v - V_u) + I_u\); spike when \(V_u \ge \theta\), reset, refractory \(t_{ref}\).
Edge flux: \(f_{uv} = w_{uv}(V_v - V_u)\), antisymmetric: \(f_{vu} = -f_{uv}\).

### Units

| Quantity | Unit |
|---|---|
| V (membrane) | model units (dimensionless) |
| w_uv | Angstrom (from S08) |
| refractory | steps today (must be ps) |

### Invariants

- Flux antisymmetry: the sum of flux over all core vertices is zero (closed tile).
- Spikes are invariant when solver tolerance is tightened 10x.
- Observer mode (lif_kick = 0) leaves the nuclear trajectory bitwise unchanged.

### Known defects

| ID | Defect | Where | Status | Fix package | Check |
|---|---|---|---|---|---|
| D-LIF-ATOMIC | FP32 atomicAdd flux accumulation (nondeterministic order) | nhs_biprism_voronoi_lif.cuh | **OPEN** | H0 / P4 |  |
| D-LIF-8-LEAVES | Leaf count hard-coded to 8 (r < 8u) | nhs_biprism_voronoi_lif.cuh; nhs_biprism_unit_bridge.cuh; v10 | **OPEN** | H1 |  |
| D-STEP-WINDOWS | REFRACTORY_STEPS and UV_LIF_WINDOW in steps; causal windows change with dt | nhs_amber_fused.cu | **OPEN** | P7.2 |  |
| D-SDST-FOLD | Unknown spike sources are folded into the LIF channel | sdst_bridge.rs select_native_percentile_indices (`_ => 1`) | **OPEN** | P11 |  |

### Qualifying gate

LIF 10k steps with zero fault latches; deterministic scan-ordered flux; spikes invariant under 10x tolerance tightening and across dt = 4/2/0.5 fs (physical-time windows); BY-FDR against surrogates.

### Mandatory reviewer questions

1. Show that the flux accumulation order is nondeterministic and state which spike outcomes can therefore differ between identical runs.
2. Where is the 8-leaf limit (r < 8u) and what happens for a protein that needs more than 8 tiles?
3. Express REFRACTORY_STEPS in ps at 2 fs and at 4 fs and state the effect on causal lag windows.
4. Which DEC weight function feeds biprism_live_diagonal_edge, and does it inherit D-DEC-OPP-EDGE?

## S11. Chronicle: CLOCK, leaves and restart closure

**Layer status:** `EXECUTED_UNQUALIFIED`

Raw Chronicle (L0) writes spike bytes with exact clocks; the coupled chronology record (L1, CCR)
stamps step, time, dt, gear and emitter on the GPU; scientific checkpoints carry excited-state,
RNG, protocol and LIF state. A fresh-process resume has never completed end to end, coordinates
are not captured on qualified routes, and restart equivalence has never been run.

### Source excerpts (verbatim)

**`crates/prism-nhs/src/bin/nhs_rt_full/chronicle_raw.rs:22-33`** (selector: anchor='^const CLOCK: u32 = 1;'; sha256 `046809bdc19eca47…`)

```rust
22 | 
23 | const REPLICA_EPOCH: u32 = 1; // Fresh constructor0, then the one actual reset_for_replica.
24 | const RAW: u32 = 0;
25 | const CLOCK: u32 = 1;
26 | const SELECTION: u32 = 2;
27 | const DRAIN: u32 = 3;
28 | const QUEUE: u32 = 4;
29 | const QUEUE_BYTES:usize=80;
30 | const DRAIN_BYTES: usize = 112;
31 | const SOURCE_DECLARED_QUEUE_CAPACITY: u64 = 2_000_000;
32 | const PINNED_STORE_MANIFEST_LIMIT: u64 = 16 * 1024 * 1024;
33 | const MAX_SLAB_BYTES: usize = 4 * 1024 * 1024;
```

**`crates/prism-nhs/src/bin/nhs_rt_full/chronicle_raw.rs:499-513`** (selector: symbol='fn read_physical_clock'; sha256 `046809bdc19eca47…`)

```rust
499 | pub fn read_physical_clock(owner:&NhsAmberFusedEngine,authority:RawClockAuthorityV1,start:u64,end:u64,capacity:u64,
500 |     paired_time_rows:&[prism_nhs::governor_device::PhysicalTimeRecord])->Result<Vec<PhysicalTimeRecord>> {
501 |     let rows=match authority {
502 |         RawClockAuthorityV1::StandaloneObserver=>owner.observer_physical_time_records(start,end)?,
503 |         RawClockAuthorityV1::UnarmedPhysicalGuard=>owner.chronicle_unarmed_guard_time_records(start,end,capacity)?,
504 |     };
505 |     if authority==RawClockAuthorityV1::UnarmedPhysicalGuard {
506 |         ensure!(paired_time_rows.len() as u64==end,"raw clock and original time/energy writer frontier differ");
507 |         let paired=paired_time_rows.get(usize::try_from(start)?..usize::try_from(end)?).context("raw clock writer prefix missing")?;
508 |         ensure!(rows.len()==paired.len() && rows.iter().zip(paired).all(|(raw,native)|raw.step==native.step
509 |             && raw.applied_dt_ps.to_bits()==native.applied_dt_ps.to_bits()
510 |             && raw.cumulative_simulated_time_ps.to_bits()==native.cumulative_simulated_time_ps.to_bits()),
511 |             "raw Chronicle and original guard time/energy records are not bit-identical");
512 |     }Ok(rows)
513 | }
```

**`crates/prism-nhs/src/coupled_chronology.rs:457-536`** (selector: symbol='pub struct CoupledChronologyRecord'; sha256 `5a5734db642c0280…`)
  
_Excerpt truncated at max_lines=80; the full symbol spans lines 457-550._

```rust
457 | pub struct CoupledChronologyRecord {
458 |     // ── chronology core ──
459 |     /// Global MD step index this record describes. [`CCR_U64_UNMEASURED`]
460 |     /// when not established.
461 |     pub global_md_step: u64, // @0
462 |     /// Cumulative simulated time in picoseconds at `global_md_step`.
463 |     ///
464 |     /// **Mandatory once the gear varies.** Step index stops being
465 |     /// proportional to elapsed physics the moment `dt` changes.
466 |     /// [`CCR_F64_UNMEASURED`] when not established.
467 |     pub cumulative_simulated_time_ps: f64, // @8
468 |     /// The timestep in picoseconds that was ACTUALLY integrated over the
469 |     /// interval ending at `global_md_step`. Not the requested dt, not the
470 |     /// authorized dt. [`CCR_F64_UNMEASURED`] when not established.
471 |     pub dt_applied_ps: f64, // @16
472 |     /// Monotonic gear epoch; increments on every authorized transition.
473 |     pub gear_epoch_id: u64, // @24
474 |     /// Identifier of the tensor snapshot bound to this coordinate epoch.
475 |     pub tensor_snapshot_id: u64, // @32
476 | 
477 |     // ── TSH demand summary (scientific DEMAND — never authority) ──
478 |     /// Integrated salience NORMALISED PER SIMULATED PICOSECOND.
479 |     ///
480 |     /// Never per record, per chunk, or per invocation. Per-record
481 |     /// normalisation creates positive feedback into blindness: low salience →
482 |     /// coarser gear → fewer snapshots → less evidence → lower measured
483 |     /// salience → coarser still, which is stable in the WRONG equilibrium.
484 |     pub tsh_integrated_salience_per_ps: f64, // @40
485 |     /// TSH memory state scalar.
486 |     pub tsh_memory_state: f64, // @48
487 |     /// TSH confidence in its own resolution request.
488 |     pub tsh_request_confidence: f64, // @56
489 |     /// Anti-blindness observation confidence. Coarsening is prohibited while
490 |     /// this is below the policy threshold, regardless of salience.
491 |     pub observation_confidence: f64, // @64
492 | 
493 |     // ── physical governor summary (SOLE safety authority) ──
494 |     /// Constraint residual observed by the governor.
495 |     pub constraint_residual: f64, // @72
496 |     /// Hamiltonian drift `|V_t - (V_{t-1} + W_ext)| / |V_{t-1} + W_ext|`.
497 |     pub hamiltonian_drift: f64, // @80
498 |     /// `||F||_2` of the integrator force buffer.
499 |     pub force_norm: f64, // @88
500 | 
501 |     // ── 4-byte block ──
502 |     /// [`CCR_SCHEMA_VERSION`].
503 |     pub schema_version: u32, // @96
504 |     /// `CCR_FLAG_*` bitmask.
505 |     pub record_flags: u32, // @100
506 |     /// Gear the TSH arm DEMANDED. Not a measurement of what ran.
507 |     pub requested_gear: u32, // @104
508 |     /// Gear the governor AUTHORIZED = `min(demanded, permitted)`.
509 |     pub authorized_gear: u32, // @108
510 |     /// Gear the DEVICE actually ran. The only field that states the
511 |     /// resolution the integrator held.
512 |     pub device_measured_gear: u32, // @112
513 |     /// TSH active-branch bitmask. `0` is legitimate (no branches active);
514 |     /// absence is [`CCR_U32_UNMEASURED`].
515 |     pub tsh_active_branches: u32, // @116
516 |     /// `CCR_GUARD_*` bitmask. `0` is legitimate (all guards clear); absence
517 |     /// is [`CCR_U32_UNMEASURED`].
518 |     pub physical_guard_status: u32, // @120
519 |     /// `CCR_REASON_*`. `0` is [`CCR_REASON_NOT_RECORDED`].
520 |     pub transition_reason: u32, // @124
521 |     /// HMR tri-state: applied / not-applied / unverified.
522 |     pub hmr_verified: u32, // @128
523 |     /// Producing stream. `0` is a legitimate stream id.
524 |     pub stream_id: u32, // @132
525 |     /// Producing replica. `0` is a legitimate replica id.
526 |     pub replica_id: u32, // @136
527 |     /// `CCR_EMITTER_*`. `0` is [`CCR_EMITTER_UNSPECIFIED`].
528 |     pub emitter_tag: u32, // @140
529 |     /// `CCR_FENCE_DIRTY` / `CCR_FENCE_PUBLISHED`. Raised LAST by the
530 |     /// producer, after a system-scope threadfence.
531 |     pub publication_fence: u32, // @144
532 |     /// Reserved; producers write 0.
533 |     pub _reserved_u32: u32, // @148
534 |     /// Per-emitter monotonic record sequence, incremented by exactly 1 on
535 |     /// every emitted record. [`CCR_U64_UNMEASURED`] when not stamped.
536 |     ///
```

**`crates/prism-nhs/src/scientific_checkpoint/restart_transaction.rs:868-888`** (selector: symbol='pub fn publish_branch_metadata\\('; sha256 `7e300c9105fe8c96…`)

```rust
868 | pub fn publish_branch_metadata(root:&Path,relative:&Path,bytes:&[u8])->Result<()> {
869 |     ensure!(bytes.len()<=64*1024*1024 && !relative.as_os_str().is_empty()
870 |         && relative.components().all(|c|matches!(c,std::path::Component::Normal(_))),"invalid bounded branch metadata");
871 |     verify_directory_chain(root)?;
872 |     let mut parent=root.to_path_buf();
873 |     for part in relative.parent().context("metadata relative parent missing")?.components() {
874 |         parent.push(part.as_os_str());
875 |         match fs::create_dir(&parent){Ok(())=>sync_dir(parent.parent().context("metadata parent missing")?)?,
876 |             Err(e)if e.kind()==std::io::ErrorKind::AlreadyExists=>{},Err(e)=>return Err(e.into())}
877 |         verify_directory_chain(&parent)?;
878 |     }
879 |     let path=root.join(relative);
880 |     if path.try_exists()? {
881 |         let existing=read_bounded(&path,64*1024*1024)?;
882 |         ensure!(existing==bytes,"existing historical metadata differs");return Ok(());
883 |     }
884 |     let temporary=parent.join(format!(".metadata-{}.branch.next",hash(bytes)));
885 |     if temporary.try_exists()?{ensure!(read_bounded(&temporary,64*1024*1024)?==bytes,"branch metadata temporary differs");}
886 |     else{spend_branch_copy_capacity(root,bytes.len() as u64)?;write_new(&temporary,bytes)?;}
887 |     fs::hard_link(&temporary,&path)?;sync_dir(&parent)?;fs::remove_file(&temporary)?;sync_dir(&parent)?;Ok(())
888 | }
```

### Planned components (NOT_BUILT)

| Component | Package | Status |
|---|---|---|
| L3 coordinate frame datum (observer-only, per-frame step/time bits/hash, fsync) | P2.4 | NOT_BUILT |
| Offline reader prism_chronicle_sdst_replay | P2.5 | NOT_BUILT |

### Mathematics

Clock key: (run, stream_id, replica_id, global_md_step, cumulative_simulated_time_ps, gear_epoch_id, protocol phase, segment).
\(t_{k+1} = t_k + \Delta t_{gear(k)}\) exactly (no nominal-dt reconstruction).

### Units

| Quantity | Unit |
|---|---|
| global_md_step | steps |
| cumulative_simulated_time | ps (exact sum of applied dt) |
| leaf hash | sha256 |

### Invariants

- Every leaf carries a clock key; ordinals per (stream, field) are gap-free or the gap is explicit.
- Restart closure: resumed run reproduces the uninterrupted run's subsequent output sequence (bitwise after H0).
- Publish is atomic and durable (fsync file + parent directory) before acknowledgement.

### Known defects

| ID | Defect | Where | Status | Fix package | Check |
|---|---|---|---|---|---|
| D-RESUME-NEVER | Fresh-process resume has never completed once ('Bounded JSON owner exceeded' on the reservation-retirement record) | scientific_checkpoint/restart_transaction.rs | **OPEN** | P2.1 |  |
| D-FRAMES-NOT-CAPTURED | Coordinates never captured on qualified routes; frames.bin single-dt header, no fsync; snapshot time hard-coded 2 fs | fused_engine.rs; chronicle_raw.rs | **OPEN** | P2.4 |  |
| D-NOMINAL-DT | Nominal dt written to ghost_time_map.json | nhs_rt_full.rs | **OPEN** | P3 |  |

### Qualifying gate

P2: native run -> stop -> fresh-process resume -> complete -> offline materialize -> DSTW, reproduced from the immutable record; omission, corruption, stale-head refusal and kill-at-every-boundary tests.

### Mandatory reviewer questions

1. List every field in the restart closure and identify any engine state that is NOT serialized (thermostat, adaptive-dt controller, BOCPD, CUDA graph phase).
2. How does chronicle_raw.rs bind the CLOCK field to RAW leaves, and what happens if a CLOCK ordinal is missing?
3. Show the fsync/rename sequence for a committed generation and identify any window in which a crash leaves an acknowledged but unpublished leaf.

## S12. QNANL encode / seal / decode

**Layer status:** `QUALIFIED`
  
**Status note:** Packet codec (PRSQNB01) qualified as a post-run seal (253 round-trip receipts). In-loop sealing, families 52-59 and the out-of-process decoder are NOT_BUILT.

PRSQNB01 packet codec: fixed header, section payloads (SectionZstd only in use), seal over header
material, durable publish with lock, staging, fsync and recovery. Delta-time, event dictionary,
event RLE and tensor-chunked codecs are registered but never selected.

### Source excerpts (verbatim)

**`crates/prismcode/src/qnanl_artifact.rs:993-1082`** (selector: symbol='pub fn encode_bytes\\('; sha256 `b279575a69792e71…`)
  
_Excerpt truncated at max_lines=90; the full symbol spans lines 993-1158._

```rust
 993 | pub fn encode_bytes(
 994 |     raw: &[u8],
 995 |     packet_path: &Path,
 996 |     receipt_path: &Path,
 997 |     options: QnanlEncodeOptions,
 998 | ) -> QnanlArtifactResult<QnanlArtifactReceipt> {
 999 |     let _publication_lock = lock_artifact_publication(packet_path)?;
1000 |     let _receipt_lock = lock_artifact_publication(receipt_path)?;
1001 |     recover_packet_publication(packet_path, receipt_path)?;
1002 | 
1003 |     let source_path_str = options
1004 |         .source_path
1005 |         .as_ref()
1006 |         .map(|path| path.display().to_string())
1007 |         .unwrap_or_else(|| "memory".to_string());
1008 |     let source_file_name = options.source_path.as_ref().and_then(|path| {
1009 |         path.file_name()
1010 |             .and_then(|name| name.to_str())
1011 |             .map(str::to_string)
1012 |     });
1013 | 
1014 |     let registry = RuntimeCodecRegistry::with_builtin_codecs();
1015 |     registry.validate_required_codecs()?;
1016 | 
1017 |     let encode_start = Instant::now();
1018 |     let section = registry.encode_section(CompressionCodecKind::SectionZstd, raw)?;
1019 |     let encode_ns = encode_start.elapsed().as_nanos();
1020 | 
1021 |     let raw_hash = hash32(raw);
1022 |     let compressed_hash = hash32(&section.compressed_bytes);
1023 | 
1024 |     let source_path_for_header = options
1025 |         .source_path
1026 |         .as_ref()
1027 |         .map(|path| path.display().to_string())
1028 |         .unwrap_or_default();
1029 |     let source_file_for_header = source_file_name.unwrap_or_default();
1030 |     let pre_seal_header = BinaryHeaderMaterial {
1031 |         artifact_family: options.artifact_family,
1032 |         transform_code: TRANSFORM_CANONICAL_BYTES,
1033 |         flags: 0,
1034 |         raw_bytes: raw.len() as u64,
1035 |         compressed_body_bytes: section.compressed_bytes.len() as u64,
1036 |         qnanl_seal_bytes: 0,
1037 |         artifact_id: options.artifact_id.clone(),
1038 |         source_path: source_path_for_header.clone(),
1039 |         source_file_name: source_file_for_header.clone(),
1040 |         raw_hash,
1041 |         compressed_hash,
1042 |         qnanl_seal_hash: [0u8; 32],
1043 |     };
1044 |     let seal_payload = build_seal_payload(&pre_seal_header);
1045 | 
1046 |     let qnanl_start = Instant::now();
1047 |     let q4_packet = nanl_q4::encode_payload(&seal_payload)?;
1048 |     let decoded_seal = nanl_q4::decode_packet(q4_packet.bytes())?;
1049 |     if decoded_seal != seal_payload {
1050 |         return Err(QnanlArtifactError::Hash(
1051 |             "QNANL seal roundtrip mismatch".to_string(),
1052 |         ));
1053 |     }
1054 |     let qnanl_report = nanl_q4::validate_packet(q4_packet.bytes())?;
1055 |     let qnanl_seal_ns = qnanl_start.elapsed().as_nanos();
1056 |     let qnanl_hash = hash32(q4_packet.bytes());
1057 | 
1058 |     let header = BinaryHeaderMaterial {
1059 |         qnanl_seal_bytes: q4_packet.bytes().len() as u64,
1060 |         qnanl_seal_hash: qnanl_hash,
1061 |         ..pre_seal_header
1062 |     };
1063 |     let header_bytes = encode_fixed_header(&header)?;
1064 | 
1065 |     if let Some(parent) = packet_path.parent() {
1066 |         durable_create_dir_all(parent)?;
1067 |     }
1068 |     let mut packet = Vec::with_capacity(
1069 |         header_bytes.len() + section.compressed_bytes.len() + q4_packet.bytes().len(),
1070 |     );
1071 |     packet.extend_from_slice(&header_bytes);
1072 |     packet.extend_from_slice(&section.compressed_bytes);
1073 |     packet.extend_from_slice(q4_packet.bytes());
1074 | 
1075 |     let decode_start = Instant::now();
1076 |     let decoded = decode_packet_bytes(&packet)?;
1077 |     let codec_roundtrip = registry.validate_encoded_by_kind(
1078 |         CompressionCodecKind::SectionZstd,
1079 |         &section.compressed_bytes,
1080 |         section.hashes,
1081 |     )?;
1082 |     if decoded != raw {
```

**`crates/prismcode/src/qnanl_artifact.rs:1170-1197`** (selector: symbol='pub fn decode_packet_bytes\\('; sha256 `b279575a69792e71…`)

```rust
1170 | pub fn decode_packet_bytes(packet: &[u8]) -> QnanlArtifactResult<Vec<u8>> {
1171 |     let (header, compressed, qnanl_seal) = parse_packet(packet)?;
1172 |     if hash32(qnanl_seal) != header.qnanl_seal_hash {
1173 |         return Err(QnanlArtifactError::Hash(
1174 |             "QNANL seal hash mismatch".to_string(),
1175 |         ));
1176 |     }
1177 |     let decoded_seal = nanl_q4::decode_packet(qnanl_seal)?;
1178 |     let expected_seal = build_seal_payload(&BinaryHeaderMaterial {
1179 |         qnanl_seal_bytes: 0,
1180 |         qnanl_seal_hash: [0u8; 32],
1181 |         ..header.clone()
1182 |     });
1183 |     if decoded_seal != expected_seal {
1184 |         return Err(QnanlArtifactError::Hash(
1185 |             "QNANL seal payload mismatch".to_string(),
1186 |         ));
1187 |     }
1188 |     let registry = RuntimeCodecRegistry::with_builtin_codecs();
1189 |     let codec = registry.codec_by_kind(CompressionCodecKind::SectionZstd)?;
1190 |     let decoded = codec.decode(compressed)?;
1191 |     if hash32(&decoded) != header.raw_hash {
1192 |         return Err(QnanlArtifactError::Hash(
1193 |             "decoded raw hash mismatch".to_string(),
1194 |         ));
1195 |     }
1196 |     Ok(decoded)
1197 | }
```

**`crates/prismcode/src/qnanl_artifact.rs:1492-1527`** (selector: symbol='pub fn publish_immutable_artifact\\('; sha256 `b279575a69792e71…`)

```rust
1492 | pub fn publish_immutable_artifact(path: &Path, bytes: &[u8]) -> QnanlArtifactResult<()> {
1493 |     durable_create_dir_all(parent_directory(path))?;
1494 |     let sequence = DURABLE_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
1495 |     let temporary = sidecar_path(
1496 |         path,
1497 |         &format!(".tmp.{}.{}.{}", std::process::id(), now_unix_ns(), sequence),
1498 |     );
1499 |     let result = (|| {
1500 |         let mut file = OpenOptions::new()
1501 |             .write(true)
1502 |             .create_new(true)
1503 |             .open(&temporary)?;
1504 |         file.write_all(bytes)?;
1505 |         file.sync_all()?;
1506 |         match std::fs::hard_link(&temporary, path) {
1507 |             Ok(()) => {}
1508 |             Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
1509 |                 if std::fs::read(path)? != bytes {
1510 |                     return Err(durability_error(&format!(
1511 |                         "conflicting immutable artifact {}",
1512 |                         path.display()
1513 |                     )));
1514 |                 }
1515 |                 File::open(path)?.sync_all()?;
1516 |             }
1517 |             Err(error) => return Err(error.into()),
1518 |         }
1519 |         sync_parent(path)?;
1520 |         std::fs::remove_file(&temporary)?;
1521 |         sync_parent(path)
1522 |     })();
1523 |     if result.is_err() {
1524 |         let _ = std::fs::remove_file(&temporary);
1525 |     }
1526 |     result
1527 | }
```

**`crates/prismcode/src/md_native_runtime.rs:539-576`** (selector: symbol='pub fn record_qnanl_inline_seal_verify\\('; sha256 `627c438845fb8331…`)

```rust
539 |     pub fn record_qnanl_inline_seal_verify(
540 |         &mut self,
541 |         step: u64,
542 |         inner_step: u32,
543 |         event_byte_source_sha256: &str,
544 |         qnanl_sealed_from_sha256: &str,
545 |         byte_exact_decode_pass: bool,
546 |     ) -> Result<(), String> {
547 |         if !is_lower_sha256(event_byte_source_sha256)
548 |             || !is_lower_sha256(qnanl_sealed_from_sha256)
549 |             || event_byte_source_sha256 != qnanl_sealed_from_sha256
550 |             || !byte_exact_decode_pass
551 |         {
552 |             return Err(
553 |                 "EVIDENCE_NOT_FOUND: inline QNANL event-byte/seal/decode binding failed"
554 |                     .to_string(),
555 |             );
556 |         }
557 |         let note = json!({
558 |             "runtime_origin": "RUNTIME_INLINE",
559 |             "event_byte_source_sha256": event_byte_source_sha256,
560 |             "qnanl_sealed_from_sha256": qnanl_sealed_from_sha256,
561 |             "byte_exact_decode_pass": byte_exact_decode_pass,
562 |         })
563 |         .to_string();
564 |         self.record(
565 |             step,
566 |             inner_step,
567 |             "step_qnanl_inline",
568 |             "prismcode_qnanl_inline_seal_verify",
569 |             &[OP_MD_QNANL_SEAL, OP_MD_QNANL_VERIFY],
570 |             (1, 1, 1),
571 |             (1, 1, 1),
572 |             0,
573 |             Some(&note),
574 |         );
575 |         Ok(())
576 |     }
```

**`crates/prism-nhs/src/bin/nhs_rt_full.rs:8906-8940`** (selector: symbol='fn write_qnanl_md_evidence_artifact'; sha256 `fe8181ee63e2a511…`)

```rust
8906 | fn write_qnanl_md_evidence_artifact(
8907 |     raw_path: &Path,
8908 |     family: QnanlArtifactFamily,
8909 | ) -> anyhow::Result<QnanlArtifactReceipt> {
8910 |     let packet_path = append_qnanl_suffix(raw_path, ".prismqnanl");
8911 |     let receipt_path = append_qnanl_suffix(raw_path, ".prismqnanl.receipt.json");
8912 |     if !raw_path.exists() && packet_path.exists() && receipt_path.exists() {
8913 |         return read_qnanl_md_evidence_receipt(&receipt_path, family)
8914 |             .with_context(|| format!("import direct QNANL receipt {}", receipt_path.display()));
8915 |     }
8916 |     let artifact_id = raw_path
8917 |         .file_name()
8918 |         .and_then(|value| value.to_str())
8919 |         .map(ToOwned::to_owned)
8920 |         .unwrap_or_else(|| raw_path.display().to_string());
8921 |     // The full CA atom population can be many GiB per stream. Reuse the
8922 |     // existing lossless streaming encoder instead of allocating the whole file.
8923 |     let encode = if raw_path.file_name().and_then(|n|n.to_str())
8924 |         .is_some_and(|name|name.ends_with("_ca_coordinate_work.f64le") || name.ends_with("_nma_force_work.f64le")) {
8925 |         qnanl_artifact::encode_file_streaming
8926 |     } else { qnanl_artifact::encode_file };
8927 |     let receipt = encode(
8928 |         raw_path,
8929 |         &packet_path,
8930 |         &receipt_path,
8931 |         QnanlEncodeOptions::runtime_subprocess_orchestrated_by_prismos(
8932 |             artifact_id,
8933 |             family,
8934 |             Some(raw_path.to_path_buf()),
8935 |             "nhs_rt_full::write_qnanl_md_evidence_artifact",
8936 |         ),
8937 |     )
8938 |     .with_context(|| format!("QNANL encode {}", raw_path.display()))?;
8939 |     Ok(receipt)
8940 | }
```

### Planned components (NOT_BUILT)

| Component | Package | Status |
|---|---|---|
| In-loop QNANL seal per chunk; families 52-59 (ElectronicEventStream ... LearnedSolventCorrectorWeights) | P11 | NOT_BUILT |
| Independent out-of-process decoder (G-Q2) | P11 | NOT_BUILT |

### Mathematics

\(\text{decode}(\text{encode}(x)) = x\) byte-exact; \(\text{seal} = \mathrm{SHA256}(\text{header material})\); a single flipped bit is refused.

### Units

| Quantity | Unit |
|---|---|
| sizes | bytes |
| hashes | sha256 hex |

### Invariants

- Round trip is byte-exact for every family.
- A tampered packet (bit flip in header or body) is refused, never repaired.
- RUNTIME_INLINE is stamped only on seals made in the MD loop at the producing step.

### Known defects

| ID | Defect | Where | Status | Fix package | Check |
|---|---|---|---|---|---|
| D-QNANL-POSTRUN | Packet is a post-run seal only; the source of truth requires pre-run run authority; a post-run sweep is stamped RUNTIME_INLINE | qnanl_artifact.rs; prismos.rs sweep (not in this repo) | **OPEN** | P11 | qnanl_roundtrip |

### Qualifying gate

G-Q1..G-Q8: in-process round trip; independent out-of-process decoder; crash matrix over publish steps; bit-flip refusal; tokenizer decode with hashes; ratio/throughput receipts per family; RUNTIME_INLINE honesty; consumer ack carries packet hash.

### Mandatory reviewer questions

1. Trace encode_bytes -> parse_packet -> decode_packet_bytes and list every integrity check that runs on decode.
2. In publish_immutable_artifact, show the fsync of the file and of the parent directory and the rename order.
3. Where is RUNTIME_INLINE asserted, and can a post-run seal ever be stamped RUNTIME_INLINE (record_qnanl_inline_seal_verify)?

## S13. md_evidence_manifest

**Layer status:** `EXECUTED_UNQUALIFIED`

md_evidence_manifest.json enumerates every artifact a run wrote (spikes, Chronicle, checkpoints,
telemetry) with hashes; it is the Path B replay contract. Several protocol profiles refuse to write
it, and some status fields are literals rather than computed values.

### Source excerpts (verbatim)

**`crates/prism-nhs/src/bin/nhs_rt_full.rs:7559-7579`** (selector: anchor='md_evidence_manifest.json. enumerates every artifact'; sha256 `fe8181ee63e2a511…`)

```rust
7559 |     ///        {stem}_streamN_spikes.bin
7560 |     ///        {stem}_streamN_signal_grid.bin
7561 |     ///        {stem}_streamN_kcc_v2full.bin
7562 |     ///     each with a fixed schema header (magic, schema_version,
7563 |     ///     run_id, stem, stream_id, record_count, byte_stride, endian
7564 |     ///     marker, payload bytes, FNV-1a checksum).
7565 |     ///   * `md_evidence_manifest.json` enumerates every artifact
7566 |     ///     (existing per-stream + new).
7567 |     ///   * `path_a_completion.json` records honest evidence-handoff
7568 |     ///     status (completion_status="md_evidence_complete_postmd_materialization_required",
7569 |     ///     post_md_aggregation_status="materialization_required_path_b", path_b_required=true,
7570 |     ///     site_materialization_status="not_materialized",
7571 |     ///     binding_sites_status="not_materialized_md_only").
7572 |     ///   * `field_completeness_report.json` + `post_md_required_inputs.json`
7573 |     ///     define the Path B replay contract.
7574 |     ///   * the run RETURNS Ok(()) before entering the legacy post-MD
7575 |     ///     CPU pipeline (spatial hash clustering, find_neighbors_union,
7576 |     ///     LIGSITE, PH refinement, XGBoost rerank). Path A bounded
7577 |     ///     evidence handoff only — site materialization belongs to
7578 |     ///     Path B.
7579 |     ///
```

**`crates/prism-nhs/src/bin/nhs_rt_full.rs:8068-8093`** (selector: symbol='fn md_evidence_write_header'; sha256 `fe8181ee63e2a511…`)

```rust
8068 | fn md_evidence_write_header(
8069 |     w: &mut impl std::io::Write,
8070 |     magic: &[u8; 8],
8071 |     schema_version: u32,
8072 |     stream_id: u32,
8073 |     run_id: &str,
8074 |     stem: &str,
8075 |     record_count: u64,
8076 |     byte_stride: u64,
8077 |     payload_size: u64,
8078 | ) -> std::io::Result<()> {
8079 |     w.write_all(magic)?;
8080 |     w.write_all(&schema_version.to_le_bytes())?;
8081 |     w.write_all(&0x01020304u32.to_le_bytes())?;
8082 |     w.write_all(&stream_id.to_le_bytes())?;
8083 |     let rid = run_id.as_bytes();
8084 |     w.write_all(&(rid.len() as u64).to_le_bytes())?;
8085 |     w.write_all(rid)?;
8086 |     let st = stem.as_bytes();
8087 |     w.write_all(&(st.len() as u64).to_le_bytes())?;
8088 |     w.write_all(st)?;
8089 |     w.write_all(&record_count.to_le_bytes())?;
8090 |     w.write_all(&byte_stride.to_le_bytes())?;
8091 |     w.write_all(&payload_size.to_le_bytes())?;
8092 |     Ok(())
8093 | }
```

**`crates/prism-nhs/src/bin/nhs_rt_full.rs:15812-15832`** (selector: anchor='let expected_manifest_path = args.output.join\\("md_evidence_manifest.json"\\);'; sha256 `fe8181ee63e2a511…`)

```rust
15812 |         // map binds the committed result plus the engine-owned, hash-bound
15813 |         // per-stream Ghost artifacts; it deliberately does not depend on the
15814 |         // manifest, avoiding a hash cycle.  The manifest is the final commit
15815 |         // and includes this aggregate map in its required artifact inventory.
15816 |         let expected_manifest_path = args.output.join("md_evidence_manifest.json");
15817 |         if expected_manifest_path.exists() {
15818 |             bail!(
15819 |                 "strict coupled-TWIN commit-last invariant failed: manifest appeared before aggregate Path-B map commit: {}",
15820 |                 expected_manifest_path.display()
15821 |             );
15822 |         }
15823 |         let ghost_site_map_path = emit_coupled_twin_native_ghost_site_map(
15824 |             &args.output,
15825 |             config,
15826 |             &result_path,
15827 |             &result_readback,
15828 |         )?;
15829 |         let ghost_site_map_sha256 = prism_nhs::ensemble::manifest::sha256_file(
15830 |             &ghost_site_map_path,
15831 |         )
15832 |         .with_context(|| format!("hash Ghost site map {}", ghost_site_map_path.display()))?;
```

### Planned components (NOT_BUILT)

| Component | Package | Status |
|---|---|---|
| Shared manifest structs crates/prism-nhs/src/pathb_manifest.rs + contract test | P1 | NOT_BUILT |

### Mathematics

Manifest completeness: \(\{\text{artifacts on disk}\} = \{\text{artifacts in manifest}\}\), each with \(\mathrm{sha256}\) and byte length.

### Units

| Quantity | Unit |
|---|---|
| hash | sha256 |
| size | bytes |

### Invariants

- Every artifact listed exists with the recorded hash; no unlisted artifact is consumed downstream.
- path_b_required with no evidence plan is refused before launch (planned, P1).

### Known defects

| ID | Defect | Where | Status | Fix package | Check |
|---|---|---|---|---|---|
| D-MANIFEST-LITERALS | streams_completed:0 scopes V2 instantiation; path_b_required is a literal; static 'Accepted' audit stamps | nhs_rt_full.rs | **OPEN** | P1 / P3 |  |

### Qualifying gate

P1 contract test (md_evidence_manifest_contract) shared by writer and every reader; recorder enforced before GPU work.

### Mandatory reviewer questions

1. Find where md_evidence_manifest.json is written and list the fields that are literals rather than computed (e.g. path_b_required, streams_completed).
2. Which run profiles refuse to write the manifest, and where is that refusal?
3. Does any reader accept an artifact missing from the manifest?

## S14. Path B materialization

**Layer status:** `EXECUTED_UNQUALIFIED`

prism_materialize_sites is offline: it consumes the manifest and the persisted spikes/KCC artifacts
(no MD rerun), validates envelopes and hashes, and materializes candidate sites. Known output
defects: stale pre-consensus site views, residue offset (lr.resid - 1), uncertainty defaulting to
0, NaN lag mapped to 0, 'raw_spikes' meaning filtered spikes.

### Source excerpts (verbatim)

**`crates/prism-nhs/src/bin/prism_materialize_sites.rs:561-680`** (selector: symbol='fn validate_artifact\\('; sha256 `2e7e8271afd5e98c…`)
  
_Excerpt truncated at max_lines=120; the full symbol spans lines 561-700._

```rust
561 | fn validate_artifact(
562 |     artifact_path: &Path,
563 |     expected_magic: &[u8; 8],
564 |     expected_trailer: &[u8; 8],
565 |     expected_kind: &str,
566 |     manifest_record_count: Option<u64>,
567 |     manifest_checksum_hex: Option<&str>,
568 |     manifest_run_id: &str,
569 |     manifest_stem: &str,
570 |     manifest_filename_stream_id: Option<u32>,
571 |     headers_only: bool,
572 | ) -> Result<(ArtifactValidation, Option<EnvelopeHeader>)> {
573 |     let on_disk_size = std::fs::metadata(artifact_path)
574 |         .with_context(|| format!("stat {}", artifact_path.display()))?
575 |         .len();
576 |     let f =
577 |         File::open(artifact_path).with_context(|| format!("open {}", artifact_path.display()))?;
578 |     let mut r = BufReader::with_capacity(1 << 20, f);
579 |     let header = read_envelope_header(&mut r)?;
580 |     let header_size_so_far: u64 =
581 |         8 + 4 + 4 + 4 + 8 + header.run_id.len() as u64 + 8 + header.stem.len() as u64 + 8 + 8 + 8;
582 | 
583 |     let declared_total = header_size_so_far + header.payload_size + 8 /* checksum */ + 8 /* trailer */;
584 |     let size_consistency = declared_total == on_disk_size;
585 | 
586 |     let magic_match = &header.magic == expected_magic;
587 |     let endian_ok_le = header.endian_marker == 0x01020304u32;
588 | 
589 |     // Compute payload checksum (streaming) — unless headers_only.
590 |     let computed_checksum: u64 =
591 |         if headers_only || !magic_match || !endian_ok_le || !size_consistency {
592 |             0
593 |         } else {
594 |             fnv1a_64_streamed(&mut r, header.payload_size)?
595 |         };
596 | 
597 |     // Read trailing 8 bytes checksum + 8 bytes trailer.
598 |     let mut tail_checksum_bytes = [0u8; 8];
599 |     let mut trailer_bytes = [0u8; 8];
600 |     if !headers_only && magic_match && endian_ok_le && size_consistency {
601 |         r.read_exact(&mut tail_checksum_bytes)?;
602 |         r.read_exact(&mut trailer_bytes)?;
603 |     }
604 |     let stored_checksum = if headers_only {
605 |         0
606 |     } else {
607 |         u64::from_le_bytes(tail_checksum_bytes)
608 |     };
609 |     let trailer_match = !headers_only && &trailer_bytes == expected_trailer;
610 | 
611 |     let checksum_matches_manifest = if headers_only {
612 |         None
613 |     } else {
614 |         manifest_checksum_hex
615 |             .map(|hex| format!("{:016x}", computed_checksum) == hex.to_ascii_lowercase())
616 |     };
617 | 
618 |     let record_count_matches_manifest = manifest_record_count.map(|n| n == header.record_count);
619 | 
620 |     let stream_id_matches_filename = manifest_filename_stream_id.map(|id| id == header.stream_id);
621 | 
622 |     let identity_matches = header.run_id == manifest_run_id
623 |         && header.stem == manifest_stem
624 |         && stream_id_matches_filename.unwrap_or(true);
625 |     let valid = identity_matches
626 |         && magic_match
627 |         && endian_ok_le
628 |         && size_consistency
629 |         && (headers_only || trailer_match)
630 |         && (headers_only || stored_checksum == computed_checksum)
631 |         && record_count_matches_manifest.unwrap_or(true)
632 |         && checksum_matches_manifest.unwrap_or(true);
633 | 
634 |     let failure_reason: Option<String> = if !identity_matches {
635 |         Some("envelope run/stem/filename stream identity mismatch".to_string())
636 |     } else if !magic_match {
637 |         Some(format!(
638 |             "magic mismatch: expected {:?}, got {:?}",
639 |             std::str::from_utf8(expected_magic).unwrap_or("?"),
640 |             std::str::from_utf8(&header.magic).unwrap_or("?")
641 |         ))
642 |     } else if !endian_ok_le {
643 |         Some(format!(
644 |             "endian marker not LE: 0x{:08x}",
645 |             header.endian_marker
646 |         ))
647 |     } else if !size_consistency {
648 |         Some(format!(
649 |             "size mismatch: declared total {}, on-disk {}",
650 |             declared_total, on_disk_size
651 |         ))
652 |     } else if !headers_only && !trailer_match {
653 |         Some(format!(
654 |             "trailer mismatch: expected {:?}, got {:?}",
655 |             std::str::from_utf8(expected_trailer).unwrap_or("?"),
656 |             std::str::from_utf8(&trailer_bytes).unwrap_or("?")
657 |         ))
658 |     } else if !headers_only && stored_checksum != computed_checksum {
659 |         Some(format!(
660 |             "checksum mismatch: stored 0x{:016x}, computed 0x{:016x}",
661 |             stored_checksum, computed_checksum
662 |         ))
663 |     } else if record_count_matches_manifest == Some(false) {
664 |         Some("record_count mismatch with manifest".to_string())
665 |     } else if checksum_matches_manifest == Some(false) {
666 |         Some("checksum mismatch with manifest".to_string())
667 |     } else {
668 |         None
669 |     };
670 | 
671 |     let v = ArtifactValidation {
672 |         artifact_path: artifact_path.display().to_string(),
673 |         expected_kind: expected_kind.to_string(),
674 |         magic_observed: String::from_utf8_lossy(&header.magic).to_string(),
675 |         magic_match,
676 |         schema_version: header.schema_version,
677 |         endian_marker_hex: format!("{:08x}", header.endian_marker),
678 |         endian_ok_le,
679 |         stream_id: header.stream_id,
680 |         stream_id_matches_filename,
```

**`crates/prism-nhs/src/bin/prism_materialize_sites.rs:1636-1641`** (selector: symbol='fn adaptive_dt_time_axis\\('; sha256 `2e7e8271afd5e98c…`)

```rust
1636 | fn adaptive_dt_time_axis(
1637 |     path: &Path,
1638 |     schedule: Option<&ThermalSchedule>,
1639 | ) -> Result<serde_json::Value> {
1640 |     adaptive_dt_time_axis_bytes(&std::fs::read(path)?, schedule)
1641 | }
```

**`crates/prism-nhs/src/bin/nhs_rt_full.rs:51273-51285`** (selector: anchor='let topo_idx = \\(lr\\.resid - 1\\)\\.max\\(0\\) as usize;'; sha256 `fe8181ee63e2a511…`)

```rust
51273 |             // residues.  Resids are PDB-author 1-indexed; topology
51274 |             // ca_indices is 0-indexed.  Map resid → resid-1 with bounds
51275 |             // check.
51276 |             let mut lining_centroid: Option<[f32; 3]> = None;
51277 |             if !site_ref.lining_residues.is_empty() {
51278 |                 let mut acc = [0.0_f32; 3];
51279 |                 let mut n = 0usize;
51280 |                 for lr in &site_ref.lining_residues {
51281 |                     let topo_idx = (lr.resid - 1).max(0) as usize;
51282 |                     if topo_idx < phase3_n_residues {
51283 |                         let ca = phase3_ca_pos[topo_idx];
51284 |                         acc[0] += ca[0];
51285 |                         acc[1] += ca[1];
```

### Mathematics

Adaptive time axis: \(t_k = \sum_{j<k}\Delta t_j\) (not \(k\,\Delta t_{nominal}\)).

### Units

| Quantity | Unit |
|---|---|
| time | ps |
| lag | ps (NaN must stay null with reason) |
| site coordinates | Angstrom |

### Invariants

- Materialization reads only manifest-listed artifacts and verifies their hashes.
- Missing values are null with a reason, never 0.
- Site views are generated after the final rerank.

### Known defects

| ID | Defect | Where | Status | Fix package | Check |
|---|---|---|---|---|---|
| D-STALE-SITE-VIEWS | PDB/PML/CXC/MD site views are a stale pre-consensus stage (10 sites vs 18) | nhs_rt_full.rs site view writers | **OPEN** | P3 |  |
| D-NULL-AS-ZERO | Uncertainty defaults to 0; NaN lag -> 0; raw_spikes means filtered spikes; ClusterToCausome instantiated but never applied | nhs_rt_full.rs; prism_materialize_sites.rs | **OPEN** | P3 |  |

### Qualifying gate

P3 offline validator validate_output_integrity.py; P2 milestone (offline materialize from the immutable record).

### Mandatory reviewer questions

1. In validate_artifact, list every check performed on an input artifact before it is consumed.
2. Show adaptive_dt_time_axis and confirm it does not reconstruct time from a nominal dt.
3. Find where uncertainty defaults to 0 or NaN lag becomes 0 and cite the lines.

## S15. DSTW export

**Layer status:** `EXECUTED_UNQUALIFIED`

dstw_export_wt_pathb aggregates materialized Path B output into the WT physical-profile response
(JSON plus physics/contact Parquet). The DSTW pre-stage uses nominal dt and a shared baseline
across streams; contact features are proxies.

### Source excerpts (verbatim)

**`crates/prism-nhs/src/bin/dstw_export_wt_pathb.rs:204-238`** (selector: symbol='fn load_residue_map\\('; sha256 `7cd848565353d364…`)

```rust
204 | fn load_residue_map(path: &Path, receptor_chain_id: &str) -> Result<BTreeMap<i32, MappedResidue>> {
205 |     let text = std::fs::read_to_string(path)
206 |         .with_context(|| format!("read residue map {}", path.display()))?;
207 |     let env: ResidueMapEnvelope = serde_json::from_str(&text)
208 |         .with_context(|| format!("parse residue map {}", path.display()))?;
209 |     let mut out = BTreeMap::new();
210 |     for row in env.residues {
211 |         if row.chain != receptor_chain_id {
212 |             continue;
213 |         }
214 |         if row.pdb_resid < 1 {
215 |             bail!(
216 |                 "{}: residue map row has non-positive receptor residue index: {:?}",
217 |                 path.display(),
218 |                 row.pdb_resid
219 |             );
220 |         }
221 |         out.insert(
222 |             row.topology_index,
223 |             MappedResidue {
224 |                 residue_index: row.pdb_resid,
225 |                 residue_name: row.resname,
226 |                 chain_id: row.chain,
227 |             },
228 |         );
229 |     }
230 |     if out.is_empty() {
231 |         bail!(
232 |             "{}: residue map contains no residues for receptor chain {:?}",
233 |             path.display(),
234 |             receptor_chain_id
235 |         );
236 |     }
237 |     Ok(out)
238 | }
```

**`crates/prism-nhs/src/bin/dstw_export_wt_pathb.rs:289-299`** (selector: symbol='fn convert_topology_indices\\('; sha256 `7cd848565353d364…`)

```rust
289 | fn convert_topology_indices(
290 |     topology_indices: Vec<i32>,
291 |     residue_map: &BTreeMap<i32, MappedResidue>,
292 | ) -> Vec<i32> {
293 |     let mut seen = BTreeSet::new();
294 |     topology_indices
295 |         .into_iter()
296 |         .filter_map(|idx| residue_map.get(&idx).map(|r| r.residue_index))
297 |         .filter(|idx| seen.insert(*idx))
298 |         .collect()
299 | }
```

**`crates/prism-nhs/src/bin/dstw_export_wt_pathb.rs:316-455`** (selector: symbol='fn aggregate_pathb\\('; sha256 `7cd848565353d364…`)
  
_Excerpt truncated at max_lines=140; the full symbol spans lines 316-480._

```rust
316 | fn aggregate_pathb(
317 |     root: &Value,
318 |     residue_map: &BTreeMap<i32, MappedResidue>,
319 | ) -> Result<(
320 |     BTreeMap<i32, ResidueAccumulator>,
321 |     BTreeMap<(i32, i32), (u32, f64)>,
322 |     usize,
323 | )> {
324 |     let mut per_residue: BTreeMap<i32, ResidueAccumulator> = BTreeMap::new();
325 |     for mapped in residue_map.values() {
326 |         per_residue
327 |             .entry(mapped.residue_index)
328 |             .or_insert_with(ResidueAccumulator::default);
329 |     }
330 | 
331 |     let sites = root
332 |         .get("binding_sites")
333 |         .and_then(|v| v.as_array())
334 |         .context("binding_sites must be an array")?;
335 |     if sites.is_empty() {
336 |         bail!("materialized file has zero binding_sites");
337 |     }
338 | 
339 |     let mut pair_counts: BTreeMap<(i32, i32), (u32, f64)> = BTreeMap::new();
340 | 
341 |     for site in sites {
342 |         let lining = convert_topology_indices(v_i32_vec(site, "lining_residues"), residue_map);
343 |         if lining.is_empty() {
344 |             continue;
345 |         }
346 |         let drivers: BTreeSet<i32> =
347 |             convert_topology_indices(v_i32_vec(site, "driver_residues"), residue_map)
348 |                 .into_iter()
349 |                 .collect();
350 | 
351 |         let density = v_f64(site, &["score_components", "density_score"]);
352 |         let temporal = v_f64(site, &["score_components", "temporal_persistence_factor"]);
353 |         let stream_balance = v_f64(site, &["score_components", "stream_balance_factor"]);
354 |         let residue_shell = v_f64(
355 |             site,
356 |             &["score_components", "residue_shell_plausibility_factor"],
357 |         );
358 |         let manifold = v_f64(
359 |             site,
360 |             &["score_components", "centroid_manifold_consistency_factor"],
361 |         )
362 |         .max(v_f64(
363 |             site,
364 |             &["manifold_consistency", "manifold_consistency_score"],
365 |         ));
366 |         let kcc_factor = v_f64(site, &["score_components", "kcc_driver_factor"]).max(1.0);
367 |         let kcc_score = v_f64(site, &["kcc_driver", "kcc_driver_score"]);
368 |         let rayleigh = v_f64(site, &["phase_support", "rayleigh_r_stat"]);
369 |         let n_spikes = v_i64(site, &["n_spikes"]).unwrap_or(0).max(0) as f64;
370 |         let spread = v_f64(site, &["centroid_spread_a"]);
371 |         let physiology_te_out = v_f64(
372 |             site,
373 |             &["binding_site_physiology_surface", "tide_te_out_proxy"],
374 |         );
375 |         let physiology_te_in = v_f64(
376 |             site,
377 |             &["binding_site_physiology_surface", "tide_te_in_proxy"],
378 |         );
379 |         let physiology_delta_hc = v_f64(
380 |             site,
381 |             &["binding_site_physiology_surface", "tide_delta_hc_proxy"],
382 |         );
383 |         let physiology_signal = v_f64(
384 |             site,
385 |             &[
386 |                 "binding_site_physiology_surface",
387 |                 "signal_preservation_factor",
388 |             ],
389 |         );
390 |         let physiology_scale = v_f64(
391 |             site,
392 |             &[
393 |                 "binding_site_physiology_surface",
394 |                 "scale_normalized_dynamic_support",
395 |             ],
396 |         );
397 |         let morphology_rank =
398 |             v_f64(site, &["pocket_morphology_surface", "morphology_rank"]) / 1000.0;
399 |         let morphology_enclosure = v_f64(
400 |             site,
401 |             &["pocket_morphology_surface", "enclosure_wall_blocking_score"],
402 |         );
403 |         let morphology_access = v_f64(
404 |             site,
405 |             &["pocket_morphology_surface", "mouth_accessibility_proxy"],
406 |         );
407 | 
408 |         // These are conservative, documented channel lifts from materialized MD
409 |         // evidence. Canonical physiology/morphology fields take precedence when
410 |         // present; legacy-compatible fallbacks remain for older materializations:
411 |         //   te_out   := density × temporal persistence × stream balance,
412 |         //               scaled by log spike support and KCC driver support.
413 |         //   te_in    := residue-shell plausibility × manifold consistency,
414 |         //               i.e. local sensitivity/enclosure support.
415 |         //   delta_hc := phase coherence × manifold consistency.
416 |         //   sigma    := centroid spread variance proxy; downstream EiV paths
417 |         //               can down-weight high-spread sites.
418 |         let spike_scale = n_spikes.ln_1p().max(1.0);
419 |         let fallback_te_out =
420 |             density.max(0.0) * temporal.max(0.0) * stream_balance.max(0.0) * spike_scale;
421 |         let fallback_te_in = residue_shell.max(0.0)
422 |             * manifold.max(0.0)
423 |             * stream_balance.max(0.0)
424 |             * morphology_rank.max(0.0).max(0.5);
425 |         let fallback_delta_hc = rayleigh.max(0.0) * manifold.max(0.0);
426 |         let physiology_scale_factor = if physiology_scale > 0.0 {
427 |             physiology_scale.clamp(0.5, 1.25)
428 |         } else {
429 |             1.0
430 |         };
431 |         let morphology_enclosure_factor = if morphology_enclosure > 0.0 {
432 |             morphology_enclosure.clamp(0.5, 1.25)
433 |         } else {
434 |             1.0
435 |         };
436 |         let morphology_access_factor = if morphology_access > 0.0 {
437 |             (1.0 + (1.0 - morphology_access.clamp(0.0, 1.0))).clamp(1.0, 2.0)
438 |         } else {
439 |             1.0
440 |         };
441 |         let base_te_out = physiology_te_out.max(fallback_te_out) * physiology_scale_factor;
442 |         let base_te_in = physiology_te_in.max(fallback_te_in) * morphology_enclosure_factor;
443 |         let site_delta_hc = physiology_delta_hc
444 |             .max(fallback_delta_hc)
445 |             .max(physiology_signal.max(0.0) * rayleigh.max(0.0));
446 |         let site_sigma = (spread.max(0.0) / 10.0).powi(2) * morphology_access_factor;
447 | 
448 |         for &rid in &lining {
449 |             let is_driver = drivers.contains(&rid);
450 |             let driver_boost = if is_driver {
451 |                 kcc_factor.max(1.0) + kcc_score.max(0.0).ln_1p()
452 |             } else {
453 |                 1.0
454 |             };
455 |             let acc = per_residue.entry(rid).or_default();
```

### Mathematics

Per-residue aggregation over streams \(s\): \(x_r = f(\{x_{r,s}\})\) with per-stream baselines \(b_s\) (planned; shared baseline today).

### Units

| Quantity | Unit |
|---|---|
| distances | Angstrom |
| time | ps |
| residue id | PDB number + chain (via residue map) |

### Invariants

- Every exported row maps to a residue through the residue map (chain + PDB number).
- No MD propagation is invoked.
- Proxy features are named *_proxy.

### Known defects

| ID | Defect | Where | Status | Fix package | Check |
|---|---|---|---|---|---|
| D-DSTW-PRESTAGE | DSTW pre-stage: shared baseline across streams, nominal dt, *_contact_proxy naming | dstw_export_wt_pathb.rs | **OPEN** | P2.5 |  |

### Qualifying gate

P2 milestone: DSTW result reproduced offline from the immutable record; per-stream baselines; adaptive dt.

### Mandatory reviewer questions

1. In aggregate_pathb, show how per-stream values are combined and whether baselines are per stream.
2. In convert_topology_indices / load_residue_map, show the index-to-residue mapping and compare it with the lr.resid - 1 convention in S14.
3. List the columns written by write_contact_parquet that are proxies and whether they are named as such.

## Executable reference checks (`--verify`)

Independent CPU reference implementations, deterministic. Host paths in records are shown as `<engine_root>`, `<home>`, `<host-mnt>`. `KNOWN_DEFECT_OPEN` means the check reproduces a defect that is still recorded as OPEN above; `NOT_BUILT` means the source under test does not exist yet. Neither is silently skipped.

| Check | Verdict | Defect | Module | Expected | Observed | Tolerance |
|---|---|---|---|---|---|---|
| cayley_vs_su2_phase | **KNOWN_DEFECT_OPEN** | D-ELEC-CAYLEY-GAP | builtin_fallback.py (fallback) | {"effective_gap_Ha": 0.38, "phase_error_rad_per_ps_max": 0.001} | {"reproduces_plan_0p070_Ha_at_4fs": true, "table": [{"cayley_phase_rad": 1.552488, "compression": 1.2649, "dt_fs": 0.5,… | relative 1e-3 on the effective gap (H1 gate: <= 1e-3 rad/ps) |
| charge.biprism_rho11_charge_creation | **KNOWN_DEFECT_OPEN** | D-CHARGE-CREATE | charge_conservation.py | Sum q conserved (\|dQ\| <= 1e-12 e) when rho11 changes | {"Q_amber_core": -5.567789047656737, "cases": {"all_excited_rho1": {"dQ_e": 28.80000114440918, "predicted_0.15_sum_rho1… | 1e-12 |
| charge.edge_transfer_reference | **PASS** |  | charge_conservation.py | dq_u = -dq_v keeps \|Sum q - Q0\| <= 1e-12 e over 1e5 steps (H2 gate) | {"atoms": 192, "edges": 600, "final_dQ_e": -1.509903313490213e-13, "flux_sigma_e": 0.001, "max_abs_dQ_e": 1.86517468137… | 1e-12 |
| charge.uv_excited_charge_null_fallback | **KNOWN_DEFECT_OPEN** | D-CHARGE-UV-NULL-FALLBACK | charge_conservation.py | Sum q over the rescaled set unchanged by the UV rescale on every reachable branch | {"cases": {"pop_0.25": {"dQ_affine": 2.7755575615628914e-17, "dQ_null_fallback": -0.005727000000000038, "scale": 1.1724… | 1e-06 |
| coupling_charge_conservation | **NOT_BUILT** |  | builtin_fallback.py (fallback) | sum(dq) = 0 to 1e-12 e over 1e5 steps | null | 1e-12 |
| dec.engine_edge_weight_mapping | **KNOWN_DEFECT_OPEN** | D-DEC-OPP-EDGE | dec_weights.py | engine w_ij == -N_i.N_j/(36V) (opposite-face normals of i and j) | {"max_rel_err_engine_vs_correct": 1.2276869126025787, "max_rel_err_engine_vs_correct_of_opposite_edge": 1.7189645844021… | 1e-06 |
| dec.reference_p1_selftest | **PASS** |  | dec_weights.py | three independent P1 forms agree; sum_e w_e (du_e)^2 == V\|a\|^2 for linear u | {"max_rel_err_cotangent_vs_gradients": 8.182176897894583e-16, "max_rel_err_linear_dirichlet_energy": 3.0663530809072114… | 1e-10 |
| dec.star0_star1_exact_recompute | **KNOWN_DEFECT_OPEN** | D-DEC-OPP-EDGE | dec_weights.py | engine star0 and star1 equal exact rational values; interior linear precision == 0 | {"exact_negative_star1_edges": 17, "interior_linear_precision_engine_as_is": [-1.141703402468945, -0.3835189861169255, … | 1e-06 |
| dec_p1fem_edge_weight_mapping | **KNOWN_DEFECT_OPEN** | D-DEC-OPP-EDGE | builtin_fallback.py (fallback) | {"mapping": {"w01": "N0.N1", "w02": "N0.N2", "w03": "N0.N3", "w12": "N1.N2", "w13": "N1.N3", "w23": "N2.N3"}, "weights_… | {"face_normal_identity_holds": true, "mapping_per_function": [{"w01": "N2.N3", "w02": "N1.N3", "w03": "N1.N2", "w12": "… | exact (rational arithmetic) |
| electronic.cayley_effective_gap | **KNOWN_DEFECT_OPEN** | D-ELEC-CAYLEY-GAP | electronic_propagator.py | effective splitting == exact gap within 1e-3 rad/ps | {"max_phase_error_rad_per_ps": 15136.198737420838, "plan_claim_0p38Ha_4fs": {"compression": 5.439022288461997, "engine_… | 0.001 |
| electronic.cayley_transcription_selftest | **PASS** |  | electronic_propagator.py | transcription == (I + i dt H/2)^-1 (I - i dt H/2); unitary; H=omega*h makes omega cancel | {"max_abs_diff_omega_scaling": 4.965068306494546e-16, "max_abs_diff_vs_dense_solve": 2.424763686264159e-15, "max_norm_d… | 1e-12 |
| electronic.driven_offresonant_mixing | **KNOWN_DEFECT_OPEN** | D-ELEC-CAYLEY-GAP | electronic_propagator.py | mean rho11 under Cayley within 5% of exact for drives in the nuclear band | {"H_engine_abs_eff_gap": {"Ha": 0.013872861099558844, "cm-1": 3044.7410757785547, "rad_per_ps": 573.5231299438212}, "ma… | 0.05 |
| electronic.landau_zener | **KNOWN_DEFECT_OPEN** | D-ELEC-CAYLEY-GAP | electronic_propagator.py | H1 gate: Landau-Zener transition probability within 1e-4 of the converged reference at 0.5/2/4 fs | {"P_LZ_analytic_infinite_time": 0.3659313069412933, "P_reference_cf4_h1au": 0.3606709318608189, "finite_window_minus_an… | 0.0001 |
| electronic.rabi_phase_and_population | **KNOWN_DEFECT_OPEN** | D-ELEC-CAYLEY-GAP | electronic_propagator.py | H1 gate: Rabi phase <= 1e-3 rad/ps and populations <= 1e-4 at 0.5/2/4 fs | {"max_phase_err_rad_per_ps": 15156.318720668227, "max_population_err": 0.014308093111729599, "rows": [{"V_Ha": 0.01, "d… | {"phase_rad_per_ps": 0.001, "population": 0.0001} |
| neighbor.cell_scheme_vs_bruteforce | **KNOWN_DEFECT_OPEN** | D-NBL-MISS-10-12 | neighbor_list.py | cell-scheme pair set == brute-force pair set for r <= 12 A (the list cutoff) | {"missed_in_buffer_total": 304, "missed_within_potential_cutoff_total": 0, "real_systems_evaluated": [], "systems": {"s… | 0 |
| omega_dt_resonance | **PASS** |  | builtin_fallback.py (fallback) | {"blue_shift_4fs_pct": 11.1, "dt_3to1_fs": 4.89, "gears_fs": [0.5, 2.0, 4.0, 0.0], "omega_dt_4fs": 1.4164} | {"gears_fs_in_source": [0.5, 2.0, 4.0, 0.0], "nearest_resonance_4fs": 4, "omega_dt_4fs": 1.4164, "resonance_dt_fs": {"2… | 1e-3 in omega*dt; 0.2% shift |
| qnanl_roundtrip | **NOT_BUILT** |  | builtin_fallback.py (fallback) | byte-exact round trip + bit-flip refusal | null | exact |
| rng.philox4x32_10_known_answer | **PASS** |  | rng_streams.py | bit-exact Random123 outputs | [{"ctr": ["0x0", "0x0", "0x0", "0x0"], "expected": ["0x6627e8d5", "0xe169c58d", "0xbc57ac4c", "0x9b00dbd8"], "key": ["0… | 0 |
| rng.philox_stream_independence | **PASS** |  | rng_streams.py | cross-stream \|corr\| < 5/sqrt(N); chi2 < mean+5sd; purpose keys decorrelate; bitwise replay | {"corr_bound_5sigma": 0.0390625, "deterministic_replay": true, "max_abs_cross_stream_corr": 0.015156207463668706, "purp… | {"chi2": 119.12486080160912, "corr": 0.0390625} |
| rng.uv_lcg_stream_identity | **KNOWN_DEFECT_OPEN** | D-RNG-UV-LCG-STREAM | rng_streams.py | kick sequences differ across streams (fraction identical == 0) | {"aromatics": 12, "first_draw_increment_between_consecutive_seeds": {"mean": 0.5138643295504526, "std": 2.3274214343643… | 0.0 |
| timestep.resonance_table | **PASS** |  | timestep_resonance.py | plan statements: 4 fs on 4:1 with ~11% blue shift; 3:1 at 4.9 fs; 5 fs past 3:1; 6 fs unstable | {"baoab_mean_map_spectral_radius_6fs": {"gamma_10_per_ps": 1.9603089841393793, "gamma_1_per_ps": 2.012478789184189, "ga… | {"rel_to_limit": 0.01, "w_dt_abs": 0.0005} |
| unit_conversions | **PASS** |  | builtin_fallback.py (fallback) | {"BIPRISM_ANGSTROM_TO_BOHR": 1.8897261259077822, "BIPRISM_PS_TO_ATOMIC_TIME": 41341.37333517015, "COULOMB_CONSTANT": 33… | {"relative_error": {"BIPRISM_ANGSTROM_TO_BOHR": 1.175009446505726e-16, "BIPRISM_PS_TO_ATOMIC_TIME": 0.0, "COULOMB_CONST… | {"relative": {"BIPRISM_ANGSTROM_TO_BOHR": 1e-09, "BIPRISM_PS_TO_ATOMIC_TIME": 1e-09, "COULOMB_CONSTANT": 2e-06}} |
| units.constants_codata2022 | **PASS** |  | units.py | engine constants equal CODATA 2022 derivations within per-constant tolerance | {"Ha_to_kcal_mol": 627.5094740628974, "atomic_time_fs": 0.024188843265863295, "constants": {"BIPRISM_ANGSTROM_TO_BOHR":… | {"BIPRISM_ANGSTROM_TO_BOHR": 1e-12, "BIPRISM_NATIVE_FORCE_TO_AU": 1e-10, "BIPRISM_NATIVE_VELOCITY_TO_AU": 1e-10, "BIPRI… |
| units.photon_energy_and_edep | **PASS** |  | units.py | E_dep units kcal/mol; TRP@280 dT ~20 K with N_eff=9 as the source comment states | {"211nm_sigma_all_chromophores_A2": {"PHE": 1.9492572412386667e-12, "SS": 0.00025592975431099817, "TRP": 1.383314907205… | {"dT_K": 1.0} |

<details><summary>Full check records (JSON)</summary>

```json
{"builtin_fallback": true, "defect_id": "D-ELEC-CAYLEY-GAP", "expected": {"effective_gap_Ha": 0.38, "phase_error_rad_per_ps_max": 0.001}, "inputs": {"inner_substeps": 4, "omega": 1.0, "onsite_gap_Ha": 0.38, "source_present": true}, "module": "builtin_fallback.py", "name": "cayley_vs_su2_phase", "notes": "Eigenphase of the Cayley map (I + i h H/2)^-1 (I - i h H/2) is 2 atan(E h/2) versus exact E h. Expected to fail until the interaction-picture propagator (H1) lands.", "observed": {"reproduces_plan_0p070_Ha_at_4fs": true, "table": [{"cayley_phase_rad": 1.552488, "compression": 1.2649, "dt_fs": 0.5, "effective_gap_Ha": 0.300423, "exact_phase_rad": 1.963715, "substep_au": 5.1677}, {"cayley_phase_rad": 2.642949, "compression": 2.972, "dt_fs": 2.0, "effective_gap_Ha": 0.12786, "exact_phase_rad": 7.854861, "substep_au": 20.6707}, {"cayley_phase_rad": 2.888336, "compression": 5.439, "dt_fs": 4.0, "effective_gap_Ha": 0.069865, "exact_phase_rad": 15.709722, "substep_au": 41.3414}]}, "pass": false, "tolerance": "relative 1e-3 on the effective gap (H1 gate: <= 1e-3 rad/ps)", "verdict": "KNOWN_DEFECT_OPEN"}
{"builtin_fallback": false, "defect_id": "D-CHARGE-CREATE", "expected": "Sum q conserved (|dQ| <= 1e-12 e) when rho11 changes", "inputs": {"source_anchors": [{"cited_lines": "263-263", "fragments_found_nearby": true, "located_lines": "263-263", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_biprism_mixed_energy.cuh", "sha256": "eb52f1fc5b6c32add51f4191d202044a2831003117c9518cd586c1aee18e151b", "transcription_current": true}, {"cited_lines": "140-141", "fragments_found_nearby": true, "located_lines": "140-141", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_biprism_electronic_discrete.cuh", "sha256": "a416c303a31602bc1ae077cc17874a6b59d5048a792726a334b30265ef19eedc", "transcription_current": true}], "tile_atoms": "first 192 IL2 atoms"}, "module": "charge_conservation.py", "name": "charge.biprism_rho11_charge_creation", "notes": "Sum of engine charges = Q_amber + 0.15*Sum rho11 exactly (dQ equals the prediction): every rise in excited population creates net charge, e.g. +0.15 e per fully excited atom; across 192 atoms at rho11=1 that is +28.8 e. Also the factor is double(0.15f)=0.15000000596. H2 gate requires <=1e-12 e.", "observed": {"Q_amber_core": -5.567789047656737, "cases": {"all_excited_rho1": {"dQ_e": 28.80000114440918, "predicted_0.15_sum_rho11": 28.80000114440918, "sum_rho11": 192.0}, "thermal_like_rho11_1e-3": {"dQ_e": 0.028951353884602682, "predicted_0.15_sum_rho11": 0.028951353884603098, "sum_rho11": 0.19300901822786468}, "uv_excited_10pct_atoms_rho0.5": {"dQ_e": 1.3500000536441803, "predicted_0.15_sum_rho11": 1.3500000536441803, "sum_rho11": 9.0}}, "charges": {"path": "unset (set $PRISM_REFS_TOPOLOGY_IL2)", "source": "synthetic neutral (IL2 topology not found)"}, "double(.15f)": 0.15000000596046448, "tile_core_atoms": 192}, "pass": false, "tolerance": 1e-12, "verdict": "KNOWN_DEFECT_OPEN"}
{"builtin_fallback": false, "expected": "dq_u = -dq_v keeps |Sum q - Q0| <= 1e-12 e over 1e5 steps (H2 gate)", "inputs": {"charges": {"path": "unset (set $PRISM_REFS_TOPOLOGY_IL2)", "source": "synthetic neutral (IL2 topology not found)"}}, "module": "charge_conservation.py", "name": "charge.edge_transfer_reference", "notes": "Reference-only: the conserving form the H2 change adopts. FP64 per-atom accumulation, exactly-rounded total (math.fsum); residual is round-off of the per-atom adds.", "observed": {"atoms": 192, "edges": 600, "final_dQ_e": -1.509903313490213e-13, "flux_sigma_e": 0.001, "max_abs_dQ_e": 1.865174681370263e-13, "steps": 100000}, "pass": true, "tolerance": 1e-12, "verdict": "PASS"}
{"builtin_fallback": false, "defect_id": "D-CHARGE-UV-NULL-FALLBACK", "expected": "Sum q over the rescaled set unchanged by the UV rescale on every reachable branch", "inputs": {"dipole_ratio_sqrt": 1.69, "residue": "IL2 TRP", "source_anchors": [{"cited_lines": "2709-2720", "fragments_found_nearby": true, "located_lines": "2709-2720", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_amber_fused.cu", "sha256": "46287f27d2c49231de8376c1a58cc60e18fff3fb1cf44a6b0f818f2f643f8f30", "transcription_current": true}, {"cited_lines": "3118-3119", "fragments_found_nearby": true, "located_lines": "3118-3119", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_amber_fused.cu", "sha256": "46287f27d2c49231de8376c1a58cc60e18fff3fb1cf44a6b0f818f2f643f8f30", "transcription_current": true}]}, "module": "charge_conservation.py", "name": "charge.uv_excited_charge_null_fallback", "notes": "Ring set = IL2 TRP indole atoms incl. ring H (an assumption about the mapped set). The affine form conserves the monopole for any scale; the null-pointer fallback scales the monopole by `scale` and creates charge whenever the set's net charge is nonzero. Plan H2: fallback becomes a refusal.", "observed": {"cases": {"pop_0.25": {"dQ_affine": 2.7755575615628914e-17, "dQ_null_fallback": -0.005727000000000038, "scale": 1.1724999999999999}, "pop_1.0": {"dQ_affine": -5.551115123125783e-17, "dQ_null_fallback": -0.02290799999999993, "scale": 1.69}}, "charges": {"source": "embedded AMBER TRP indole charges (IL2 topology not found)"}, "max_abs_dQ_affine": 5.551115123125783e-17, "max_abs_dQ_null_fallback": 0.02290799999999993, "sum_ring_charge": -0.03319999999999995, "trp_ring_atoms": 15}, "pass": false, "tolerance": 1e-06, "verdict": "KNOWN_DEFECT_OPEN"}
{"builtin_fallback": true, "expected": "sum(dq) = 0 to 1e-12 e over 1e5 steps", "inputs": {}, "module": "builtin_fallback.py", "name": "coupling_charge_conservation", "notes": "NOT_BUILT: the H2 charge-conserving electronic coupling is not built; the current form q = q_amber + 0.15*rho11 creates charge by construction (D-CHARGE-CREATE).", "observed": null, "pass": null, "tolerance": 1e-12, "verdict": "NOT_BUILT"}
{"builtin_fallback": false, "defect_id": "D-DEC-OPP-EDGE", "expected": "engine w_ij == -N_i.N_j/(36V) (opposite-face normals of i and j)", "inputs": {"sigma0": 0.0001, "source_anchors": [{"cited_lines": "268-273", "fragments_found_nearby": true, "located_lines": "268-273", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_biprism_simplicial_tile.cuh", "sha256": "849da5dd1333fffd796546f1b732dc8317490835d6af46326893354a02afd79e", "transcription_current": true}, {"cited_lines": "334-339", "fragments_found_nearby": true, "located_lines": "334-339", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_biprism_simplicial_tile.cuh", "sha256": "849da5dd1333fffd796546f1b732dc8317490835d6af46326893354a02afd79e", "transcription_current": true}, {"cited_lines": "236-239", "fragments_found_nearby": true, "located_lines": "236-239", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_biprism_simplicial_tile.cuh", "sha256": "849da5dd1333fffd796546f1b732dc8317490835d6af46326893354a02afd79e", "transcription_current": true}, {"cited_lines": "688-689", "fragments_found_nearby": true, "located_lines": "688-689", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_biprism_simplicial_tile.cuh", "sha256": "849da5dd1333fffd796546f1b732dc8317490835d6af46326893354a02afd79e", "transcription_current": true}, {"cited_lines": "440-440", "fragments_found_nearby": true, "located_lines": "440-440", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_biprism_bowyer_watson_seed.cuh", "sha256": "255d925e1c37fb03d880a51365b43f1bf331adb130c2825171d7e4f0274f4a8a", "transcription_current": true}], "tets": "64 seeded non-regular tetrahedra"}, "module": "dec_weights.py", "name": "dec.engine_edge_weight_mapping", "notes": "Engine weight of edge ij equals the correct weight of the opposite edge kl to 1.7e-05 (only the sigma0=1e-4 regulariser differs): an exact 01<->23, 02<->13, 03<->12 swap. Invisible on a regular tetrahedron (all six weights equal), so regular-mesh tests cannot catch it. Floor disabled here to isolate the mapping; see star1 check.", "observed": {"max_rel_err_engine_vs_correct": 1.2276869126025787, "max_rel_err_engine_vs_correct_of_opposite_edge": 1.7189645844021923e-05, "max_rel_err_linear_dirichlet_energy_engine": 25.630081572586803, "median_rel_err_engine_vs_correct": 0.8514345208911955, "regular_tet_rel_err": 7.199994815354493e-07, "worked_example_tet0": {"coords": [[2.6942672284191955, 3.3654702526370164, 1.203934345732862], [-0.6303397770770001, 1.2490051733547036, 0.19400216087355115], [1.748973572923655, 1.323645857469982, -0.6022903615883466], [1.4052244984596787, 0.8463359742478008, 0.4644712462935071]], "weights": {"w01": {"correct": -0.06022174433887227, "engine": 0.7511032461883917}, "w02": {"correct": 0.08089111420079366, "engine": 0.35639466444223006}, "w03": {"correct": 0.1348469820408208, "engine": 0.0027566743135817302}, "w12": {"correct": -0.007792815209414853, "engine": 0.13484653619858195}, "w13": {"correct": 0.35639584278762626, "engine": 0.08089084675190529}, "w23": {"correct": 0.7511057295562438, "engine": 0.0027566743135817302}}}}, "pass": false, "tolerance": 1e-06, "verdict": "KNOWN_DEFECT_OPEN"}
{"builtin_fallback": false, "expected": "three independent P1 forms agree; sum_e w_e (du_e)^2 == V|a|^2 for linear u", "inputs": {"linear_field_a": [0.3, -1.1, 0.7], "tets": "64 seeded non-regular tetrahedra (seed 20261023, shape V/lmax^3>0.01)"}, "module": "dec_weights.py", "name": "dec.reference_p1_selftest", "notes": "Reference-only. w_ij = -N_i.N_j/(36V), N_i = normal of face opposite vertex i.", "observed": {"max_rel_err_cotangent_vs_gradients": 8.182176897894583e-16, "max_rel_err_linear_dirichlet_energy": 3.0663530809072114e-15, "max_rel_err_normals_vs_gradients": 9.367541754659327e-16, "n_tets": 64}, "pass": true, "tolerance": 1e-10, "verdict": "PASS"}
{"builtin_fallback": false, "defect_id": "D-DEC-OPP-EDGE", "expected": "engine star0 and star1 equal exact rational values; interior linear precision == 0", "inputs": {"mesh": "3x3x3 grid, spacing 3/2, rational perturbation k/64 (|k|<=12), Kuhn split", "source_anchors": [{"cited_lines": "268-273", "fragments_found_nearby": true, "located_lines": "268-273", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_biprism_simplicial_tile.cuh", "sha256": "849da5dd1333fffd796546f1b732dc8317490835d6af46326893354a02afd79e", "transcription_current": true}, {"cited_lines": "334-339", "fragments_found_nearby": true, "located_lines": "334-339", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_biprism_simplicial_tile.cuh", "sha256": "849da5dd1333fffd796546f1b732dc8317490835d6af46326893354a02afd79e", "transcription_current": true}, {"cited_lines": "236-239", "fragments_found_nearby": true, "located_lines": "236-239", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_biprism_simplicial_tile.cuh", "sha256": "849da5dd1333fffd796546f1b732dc8317490835d6af46326893354a02afd79e", "transcription_current": true}, {"cited_lines": "688-689", "fragments_found_nearby": true, "located_lines": "688-689", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_biprism_simplicial_tile.cuh", "sha256": "849da5dd1333fffd796546f1b732dc8317490835d6af46326893354a02afd79e", "transcription_current": true}, {"cited_lines": "440-440", "fragments_found_nearby": true, "located_lines": "440-440", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_biprism_bowyer_watson_seed.cuh", "sha256": "255d925e1c37fb03d880a51365b43f1bf331adb130c2825171d7e4f0274f4a8a", "transcription_current": true}, {"cited_lines": "1022-1022", "fragments_found_nearby": true, "located_lines": "1022-1022", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_biprism_simplicial_tile.cuh", "sha256": "849da5dd1333fffd796546f1b732dc8317490835d6af46326893354a02afd79e", "transcription_current": true}]}, "module": "dec_weights.py", "name": "dec.star0_star1_exact_recompute", "notes": "Exact rational star0/star1 on a perturbed Kuhn mesh (48 tets). star0 (barycentric V/4) matches; star1 is wrong as-is. Linear precision at the interior vertex is exact (0) for the correct operator and fails for the engine operator. The SPD floor fmax(w,1e-3*l_rms) is a separate defect whenever exact weights are negative (non-Delaunay / obtuse dihedrals): it breaks linear precision even after the mapping fix. The dual is barycentric (V/4), not Voronoi.", "observed": {"exact_negative_star1_edges": 17, "interior_linear_precision_engine_as_is": [-1.141703402468945, -0.3835189861169255, 2.2705805450661085], "interior_linear_precision_exact": [0.0, 0.0, 0.0], "interior_linear_precision_mapping_fixed_no_floor": [-8.494861322705383e-07, -3.563784390370195e-07, 1.024485138390041e-06], "interior_linear_precision_mapping_fixed_with_floor": [-0.45445549253252304, 0.09647475192708033, 0.27141999119091453], "max_rel_err_star0_engine": 4.0503701243815805e-16, "max_rel_err_star1_engine_as_is": 0.8701751717466565, "max_rel_err_star1_mapping_fixed_no_floor": 1.7922634803379063e-06, "max_rel_err_star1_mapping_fixed_with_floor": 0.10679438071407807, "n_edges": 98, "n_tets": 48, "n_vertices": 27, "sum_star0_engine": 26.729012807210285, "sum_star0_exact": "21020551/786432"}, "pass": false, "tolerance": 1e-06, "verdict": "KNOWN_DEFECT_OPEN"}
{"builtin_fallback": true, "defect_id": "D-DEC-OPP-EDGE", "expected": {"mapping": {"w01": "N0.N1", "w02": "N0.N2", "w03": "N0.N3", "w12": "N1.N2", "w13": "N1.N3", "w23": "N2.N3"}, "weights_exact_unregularised": {"w01": 0.12646604938271605, "w02": 0.48564814814814816, "w03": 0.2111111111111111, "w12": 0.0837962962962963, "w13": 0.12222222222222222, "w23": 0.06666666666666667}}, "inputs": {"assignment_blocks_found": 2, "source": "crates/prism-gpu/src/kernels/nhs_biprism_simplicial_tile.cuh", "tet_vertices": [["0", "0", "0"], ["3", "0", "0"], ["1/2", "2", "0"], ["1", "1/3", "5/2"]]}, "module": "builtin_fallback.py", "name": "dec_p1fem_edge_weight_mapping", "notes": "Independent P1 stiffness from the inverse Jacobian; source mapping parsed from out_wXY = fmax(dot_Na_Nb ...). A mismatch reproduces the opposite-edge assignment (01<->23, 02<->13, 03<->12).", "observed": {"face_normal_identity_holds": true, "mapping_per_function": [{"w01": "N2.N3", "w02": "N1.N3", "w03": "N1.N2", "w12": "N0.N3", "w13": "N0.N2", "w23": "N0.N1"}, {"w01": "N2.N3", "w02": "N1.N3", "w03": "N1.N2", "w12": "N0.N3", "w13": "N0.N2", "w23": "N0.N1"}], "signed_6V": "15", "weights_from_source_mapping": {"w01": 0.06666666666666667, "w02": 0.12222222222222222, "w03": 0.0837962962962963, "w12": 0.2111111111111111, "w13": 0.48564814814814816, "w23": 0.12646604938271605}}, "pass": false, "tolerance": "exact (rational arithmetic)", "verdict": "KNOWN_DEFECT_OPEN"}
{"builtin_fallback": false, "defect_id": "D-ELEC-CAYLEY-GAP", "expected": "effective splitting == exact gap within 1e-3 rad/ps", "inputs": {"dts_fs": [0.5, 2.0, 4.0], "lut_Ha": {"C": [-0.558, -0.392], "H": [-0.5, -0.12], "N": [-0.682, -0.445], "O": [-0.825, -0.51]}, "omega": 1.0, "phi_bar": 0.0, "source_anchors": [{"cited_lines": "197-231", "fragments_found_nearby": true, "located_lines": "197-231", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_biprism_electronic_discrete.cuh", "sha256": "a416c303a31602bc1ae077cc17874a6b59d5048a792726a334b30265ef19eedc", "transcription_current": true}, {"cited_lines": "148-150", "fragments_found_nearby": true, "located_lines": "148-150", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_biprism_electronic_discrete.cuh", "sha256": "a416c303a31602bc1ae077cc17874a6b59d5048a792726a334b30265ef19eedc", "transcription_current": true}, {"cited_lines": "4363-4363", "fragments_found_nearby": true, "located_lines": "4363-4363", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_biprism_variational_so3g_v10.cuh", "sha256": "6cffbf23921bb20cb2161b4b3145e0080659b1a6aa60729fd30eaf5d748da16b", "transcription_current": true}, {"cited_lines": "22-25", "fragments_found_nearby": true, "located_lines": "22-25", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_biprism_xtb_lut.cuh", "sha256": "256849480644cce560e6bec72f661185819bb4df40be88c808da2fe1a9fd6d7e", "transcription_current": true}], "substeps": 4}, "module": "electronic_propagator.py", "name": "electronic.cayley_effective_gap", "notes": "Plan figure reproduced only in the ground-at-zero convention (0.38 Ha -> 0.0699 Ha at 4 fs, 5.44x). The engine passes ABSOLUTE onsite energies (eps_s, eps_p both negative) and Cayley is not shift-invariant, so the realised splitting is smaller still (0.0139 Ha, 27.4x for H at 4 fs). Even 0.5 fs fails the 1e-3 rad/ps gate by orders of magnitude.", "observed": {"max_phase_error_rad_per_ps": 15136.198737420838, "plan_claim_0p38Ha_4fs": {"compression": 5.439022288461997, "engine_absolute_onsite_Ha": 0.013872861099558844, "engine_compression": 27.39160994065485, "plan_states_Ha": 0.07, "reproduced_ground_at_zero_Ha": 0.06986549784988165}, "table": [{"analytic_engine": 0.0667993824902456, "analytic_ground_at_zero": 0.15681471718815948, "compression_engine_absolute": 2.4850529123415197, "compression_ground_at_zero": 1.0585741120255905, "dt_fs": 0.5, "eff_gap_engine_absolute_onsite_Ha": 0.06679938249024564, "eff_gap_ground_at_zero_Ha": 0.15681471718815948, "element": "C", "gap_Ha": 0.166, "h_substep_au": 5.167671666896269, "phase_error_rad_per_ps_engine": 4101.089763550173}, {"analytic_engine": 0.006801760171971466, "analytic_ground_at_zero": 0.10092285484998072, "compression_engine_absolute": 24.405447384641548, "compression_ground_at_zero": 1.6448206924660906, "dt_fs": 2.0, "eff_gap_engine_absolute_onsite_Ha": 0.006801760171971465, "eff_gap_ground_at_zero_Ha": 0.10092285484998072, "element": "C", "gap_Ha": 0.166, "h_substep_au": 20.670686667585077, "phase_error_rad_per_ps_engine": 6581.473867032484}, {"analytic_engine": 0.0017565680609723045, "analytic_ground_at_zero": 0.06227265993331929, "compression_engine_absolute": 94.50245833805886, "compression_ground_at_zero": 2.665696313241647, "dt_fs": 4.0, "eff_gap_engine_absolute_onsite_Ha": 0.0017565680609723045, "eff_gap_ground_at_zero_Ha": 0.062272659933319284, "element": "C", "gap_Ha": 0.166, "h_substep_au": 41.341373335170154, "phase_error_rad_per_ps_engine": 6790.049037640955}, {"analytic_engine": 0.07727353869684442, "analytic_ground_at_zero": 0.21265469200704196, "compression_engine_absolute": 3.067026617349392, "compression_ground_at_zero": 1.1144828160770226, "dt_fs": 0.5, "eff_gap_engine_absolute_onsite_Ha": 0.0772735386968444, "eff_gap_ground_at_zero_Ha": 0.21265469200704196, "element": "N", "gap_Ha": 0.237, "h_substep_au": 5.167671666896269, "phase_error_rad_per_ps_engine": 6603.311268239367}, {"analytic_engine": 0.007079200504946986, "analytic_ground_at_zero": 0.11448073774967227, "compression_engine_absolute": 33.47835674867284, "compression_ground_at_zero": 2.070217266753057, "dt_fs": 2.0, "eff_gap_engine_absolute_onsite_Ha": 0.007079200504946985, "eff_gap_ground_at_zero_Ha": 0.11448073774967227, "element": "N", "gap_Ha": 0.237, "h_substep_au": 20.670686667585077, "phase_error_rad_per_ps_engine": 9505.24160944579}, {"analytic_engine": 0.0018128194162539736, "analytic_ground_at_zero": 0.06625021017480308, "compression_engine_absolute": 130.73558120297443, "compression_ground_at_zero": 3.577347141611608, "dt_fs": 4.0, "eff_gap_engine_absolute_onsite_Ha": 0.0018128194162539734, "eff_gap_ground_at_zero_Ha": 0.06625021017480308, "element": "N", "gap_Ha": 0.237, "h_substep_au": 41.341373335170154, "phase_error_rad_per_ps_engine": 9722.961036158727}, {"analytic_engine": 0.08147331181411785, "analytic_ground_at_zero": 0.2643992500234692, "compression_engine_absolute": 3.8662967416701504, "compression_ground_at_zero": 1.191380081343042, "dt_fs": 0.5, "eff_gap_engine_absolute_onsite_Ha": 0.08147331181411784, "eff_gap_ground_at_zero_Ha": 0.26439925002346915, "element": "O", "gap_Ha": 0.315, "h_substep_au": 5.167671666896269, "phase_error_rad_per_ps_engine": 9654.314000018421}, {"analytic_engine": 0.006844697523240879, "analytic_ground_at_zero": 0.12314867297993559, "compression_engine_absolute": 46.02102560857231, "compression_ground_at_zero": 2.557883835673344, "dt_fs": 2.0, "eff_gap_engine_absolute_onsite_Ha": 0.006844697523240879, "eff_gap_ground_at_zero_Ha": 0.12314867297993558, "element": "O", "gap_Ha": 0.315, "h_substep_au": 20.670686667585077, "phase_error_rad_per_ps_engine": 12739.563404903978}, {"analytic_engine": 0.0017417261583254158, "analytic_ground_at_zero": 0.06861925166664377, "compression_engine_absolute": 180.85506639164046, "compression_ground_at_zero": 4.59054845905764, "dt_fs": 4.0, "eff_gap_engine_absolute_onsite_Ha": 0.0017417261583254156, "eff_gap_ground_at_zero_Ha": 0.06861925166664377, "element": "O", "gap_Ha": 0.315, "h_substep_au": 41.341373335170154, "phase_error_rad_per_ps_engine": 12950.527249219633}, {"analytic_engine": 0.23663413380903392, "analytic_ground_at_zero": 0.3004231838615719, "compression_engine_absolute": 1.6058545480453121, "compression_ground_at_zero": 1.264882407261536, "dt_fs": 0.5, "eff_gap_engine_absolute_onsite_Ha": 0.23663413380903386, "eff_gap_ground_at_zero_Ha": 0.30042318386157185, "element": "H", "gap_Ha": 0.38, "h_substep_au": 5.167671666896269, "phase_error_rad_per_ps_engine": 5926.9417977207795}, {"analytic_engine": 0.04716043369054473, "analytic_ground_at_zero": 0.1278597812870144, "compression_engine_absolute": 8.057601897672683, "compression_ground_at_zero": 2.9720057094966523, "dt_fs": 2.0, "eff_gap_engine_absolute_onsite_Ha": 0.04716043369054474, "eff_gap_ground_at_zero_Ha": 0.1278597812870144, "element": "H", "gap_Ha": 0.38, "h_substep_au": 20.670686667585077, "phase_error_rad_per_ps_engine": 13760.044771515311}, {"analytic_engine": 0.013872861099558844, "analytic_ground_at_zero": 0.06986549784988166, "compression_engine_absolute": 27.39160994065485, "compression_ground_at_zero": 5.439022288461997, "dt_fs": 4.0, "eff_gap_engine_absolute_onsite_Ha": 0.013872861099558844, "eff_gap_ground_at_zero_Ha": 0.06986549784988165, "element": "H", "gap_Ha": 0.38, "h_substep_au": 41.341373335170154, "phase_error_rad_per_ps_engine": 15136.198737420838}]}, "pass": false, "tolerance": 0.001, "verdict": "KNOWN_DEFECT_OPEN"}
{"builtin_fallback": false, "expected": "transcription == (I + i dt H/2)^-1 (I - i dt H/2); unitary; H=omega*h makes omega cancel", "inputs": {"cases": "200 seeded Hermitian 2x2, omega in [0.2,5], dt in [1,60] au"}, "module": "electronic_propagator.py", "name": "electronic.cayley_transcription_selftest", "notes": "Reference-only: validates the transcription that the gap/Rabi/LZ checks use.", "observed": {"max_abs_diff_omega_scaling": 4.965068306494546e-16, "max_abs_diff_vs_dense_solve": 2.424763686264159e-15, "max_norm_defect": 8.881784197001252e-16}, "pass": true, "tolerance": 1e-12, "verdict": "PASS"}
{"builtin_fallback": false, "defect_id": "D-ELEC-CAYLEY-GAP", "expected": "mean rho11 under Cayley within 5% of exact for drives in the nuclear band", "inputs": {"V0_Ha": 0.01, "drive_fracs_of_w_max": [0.25, 0.5, 1.0], "dt_fs": 4.0, "source_anchors": [{"cited_lines": "197-231", "fragments_found_nearby": true, "located_lines": "197-231", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_biprism_electronic_discrete.cuh", "sha256": "a416c303a31602bc1ae077cc17874a6b59d5048a792726a334b30265ef19eedc", "transcription_current": true}, {"cited_lines": "148-150", "fragments_found_nearby": true, "located_lines": "148-150", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_biprism_electronic_discrete.cuh", "sha256": "a416c303a31602bc1ae077cc17874a6b59d5048a792726a334b30265ef19eedc", "transcription_current": true}, {"cited_lines": "4363-4363", "fragments_found_nearby": true, "located_lines": "4363-4363", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_biprism_variational_so3g_v10.cuh", "sha256": "6cffbf23921bb20cb2161b4b3145e0080659b1a6aa60729fd30eaf5d748da16b", "transcription_current": true}, {"cited_lines": "22-25", "fragments_found_nearby": true, "located_lines": "22-25", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_biprism_xtb_lut.cuh", "sha256": "256849480644cce560e6bec72f661185819bb4df40be88c808da2fe1a9fd6d7e", "transcription_current": true}], "window_ps": 0.25}, "module": "electronic_propagator.py", "name": "electronic.driven_offresonant_mixing", "notes": "Coupling V0*sin(w t) with w in the nuclear band (<= 354.1 rad/ps), 0.25 ps, start in |0>. Exact: closed-form SU(2) at 0.5 au. In the engine convention the H-atom Cayley gap at 4 fs is 3045 cm-1, inside the vibrational band, so nuclear motion can drive spurious near-resonant electronic population. The plan's '29x inflated mixing' is not a fixed factor; it is frequency dependent: ~1.00-1.03 in ground-at-zero, 0.14-26x in the engine convention across the nuclear band, and ~10^3x when the drive hits gap_eff (spurious resonance).", "observed": {"H_engine_abs_eff_gap": {"Ha": 0.013872861099558844, "cm-1": 3044.7410757785547, "rad_per_ps": 573.5231299438212}, "max_rel_dev_engine_abs_in_nuclear_band": 25.359081627727033, "plan_29x_claim": {"(0.38/gap_eff)^2": 29.58296345438638, "verdict": "not a uniform factor: inflation depends on drive frequency vs gap_eff; see rows and spurious_resonance"}, "rows": [{"drive_rad_per_ps": 88.525, "element": "C", "mean_rho11_cayley_engine_abs": 0.047184368777242185, "mean_rho11_cayley_ground_at_zero": 0.0017937079144773151, "mean_rho11_exact": 0.001790061180568943, "ratio_engine_abs": 26.359081627727033, "ratio_ground_at_zero": 1.0020372118830112}, {"drive_rad_per_ps": 177.05, "element": "C", "mean_rho11_cayley_engine_abs": 0.0013663215885513387, "mean_rho11_cayley_ground_at_zero": 0.0018097686528785082, "mean_rho11_exact": 0.0017950226758209834, "ratio_engine_abs": 0.7611723277681876, "ratio_ground_at_zero": 1.008214925224151}, {"drive_rad_per_ps": 354.1, "element": "C", "mean_rho11_cayley_engine_abs": 0.00025350487534030154, "mean_rho11_cayley_ground_at_zero": 0.0018755575372413817, "mean_rho11_exact": 0.0018147881544159233, "ratio_engine_abs": 0.13968841196337337, "ratio_ground_at_zero": 1.033485662046883}, {"drive_rad_per_ps": 88.525, "element": "H", "mean_rho11_cayley_engine_abs": 0.0003844368425033593, "mean_rho11_cayley_ground_at_zero": 0.00034432451514459193, "mean_rho11_exact": 0.00034361704805317716, "ratio_engine_abs": 1.1187944389879776, "ratio_ground_at_zero": 1.0020588823966188}, {"drive_rad_per_ps": 177.05, "element": "H", "mean_rho11_cayley_engine_abs": 0.000537808088354881, "mean_rho11_cayley_ground_at_zero": 0.0003466911309817007, "mean_rho11_exact": 0.0003438647651810524, "ratio_engine_abs": 1.5640104564703312, "ratio_ground_at_zero": 1.0082194109046332}, {"drive_rad_per_ps": 354.1, "element": "H", "mean_rho11_cayley_engine_abs": 0.0018988797304236769, "mean_rho11_cayley_ground_at_zero": 0.00035633003383373255, "mean_rho11_exact": 0.0003448387446615646, "ratio_engine_abs": 5.50657302817668, "ratio_ground_at_zero": 1.033323660261685}], "spurious_resonance_at_eff_gap": {"drive_rad_per_ps": 2888.335629859485, "mean_rho11_cayley": 0.4769931477575873, "mean_rho11_exact": 0.0004078629341696228, "ratio": 1169.4937386960848}}, "pass": false, "tolerance": 0.05, "verdict": "KNOWN_DEFECT_OPEN"}
{"builtin_fallback": false, "defect_id": "D-ELEC-CAYLEY-GAP", "expected": "H1 gate: Landau-Zener transition probability within 1e-4 of the converged reference at 0.5/2/4 fs", "inputs": {"T_au": 16000.0, "dts_fs": [0.5, 2.0, 4.0], "g_Ha": 0.002, "source_anchors": [{"cited_lines": "197-231", "fragments_found_nearby": true, "located_lines": "197-231", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_biprism_electronic_discrete.cuh", "sha256": "a416c303a31602bc1ae077cc17874a6b59d5048a792726a334b30265ef19eedc", "transcription_current": true}, {"cited_lines": "148-150", "fragments_found_nearby": true, "located_lines": "148-150", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_biprism_electronic_discrete.cuh", "sha256": "a416c303a31602bc1ae077cc17874a6b59d5048a792726a334b30265ef19eedc", "transcription_current": true}, {"cited_lines": "4363-4363", "fragments_found_nearby": true, "located_lines": "4363-4363", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_biprism_variational_so3g_v10.cuh", "sha256": "6cffbf23921bb20cb2161b4b3145e0080659b1a6aa60729fd30eaf5d748da16b", "transcription_current": true}, {"cited_lines": "22-25", "fragments_found_nearby": true, "located_lines": "22-25", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_biprism_xtb_lut.cuh", "sha256": "256849480644cce560e6bec72f661185819bb4df40be88c808da2fe1a9fd6d7e", "transcription_current": true}], "substeps": 4, "v_Ha_per_au": 2.5e-05}, "module": "electronic_propagator.py", "name": "electronic.landau_zener", "notes": "Reference: 4th-order commutator-free Magnus with exact SU(2) exponentials, h=1 au (converged vs h=2 au as reported); analytic P=exp(-2 pi g^2/v) differs from the finite +-0.2 Ha window by finite-time oscillations only. The -0.5 Ha offset case reproduces the engine convention (absolute onsite energies); exact dynamics are offset-invariant, Cayley is not. su2_midpoint is the H1 fix preview (closed-form SU(2) exponential at the same substep).", "observed": {"P_LZ_analytic_infinite_time": 0.3659313069412933, "P_reference_cf4_h1au": 0.3606709318608189, "finite_window_minus_analytic": -0.005260375080474389, "max_cayley_err": 0.6298332732555675, "reference_convergence_h1_vs_h2": 1.6978468631023702e-10, "rows": [{"P_cayley_engine_offset_-0.5Ha": 0.6856779531434105, "P_cayley_traceless": 0.37350068219407917, "P_su2_midpoint_fix_preview": 0.360670967515477, "dt_fs": 0.5, "err_cayley_engine_offset": 0.3250070212825916, "err_cayley_traceless": 0.012829750333260281, "err_su2_midpoint": 3.565465811483648e-08, "h_au": 5.167671666896269}, {"P_cayley_engine_offset_-0.5Ha": 0.9679814408732091, "P_cayley_traceless": 0.37162436146334216, "P_su2_midpoint_fix_preview": 0.36023657252612545, "dt_fs": 2.0, "err_cayley_engine_offset": 0.6073105090123903, "err_cayley_traceless": 0.010953429602523268, "err_su2_midpoint": 0.0004343593346934349, "h_au": 20.670686667585077}, {"P_cayley_engine_offset_-0.5Ha": 0.9905042051163864, "P_cayley_traceless": 0.35689001482113536, "P_su2_midpoint_fix_preview": 0.36204942643679217, "dt_fs": 4.0, "err_cayley_engine_offset": 0.6298332732555675, "err_cayley_traceless": 0.003780917039683529, "err_su2_midpoint": 0.001378494575973277, "h_au": 41.341373335170154}]}, "pass": false, "tolerance": 0.0001, "verdict": "KNOWN_DEFECT_OPEN"}
{"builtin_fallback": false, "defect_id": "D-ELEC-CAYLEY-GAP", "expected": "H1 gate: Rabi phase <= 1e-3 rad/ps and populations <= 1e-4 at 0.5/2/4 fs", "inputs": {"V_Ha": 0.01, "duration_ps": 1.0, "elements": ["C", "H"], "source_anchors": [{"cited_lines": "197-231", "fragments_found_nearby": true, "located_lines": "197-231", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_biprism_electronic_discrete.cuh", "sha256": "a416c303a31602bc1ae077cc17874a6b59d5048a792726a334b30265ef19eedc", "transcription_current": true}, {"cited_lines": "148-150", "fragments_found_nearby": true, "located_lines": "148-150", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_biprism_electronic_discrete.cuh", "sha256": "a416c303a31602bc1ae077cc17874a6b59d5048a792726a334b30265ef19eedc", "transcription_current": true}, {"cited_lines": "4363-4363", "fragments_found_nearby": true, "located_lines": "4363-4363", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_biprism_variational_so3g_v10.cuh", "sha256": "6cffbf23921bb20cb2161b4b3145e0080659b1a6aa60729fd30eaf5d748da16b", "transcription_current": true}, {"cited_lines": "22-25", "fragments_found_nearby": true, "located_lines": "22-25", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_biprism_xtb_lut.cuh", "sha256": "256849480644cce560e6bec72f661185819bb4df40be88c808da2fe1a9fd6d7e", "transcription_current": true}]}, "module": "electronic_propagator.py", "name": "electronic.rabi_phase_and_population", "notes": "Static two-level H=[[eps_s,V],[V,eps_p]] for 1 ps. Cayley preserves eigenvectors, so the mixing AMPLITUDE of a static H is exact; the error is the Rabi phase (effective splitting), which then corrupts populations after dephasing. Closed-form SU(2) exponential is exact here.", "observed": {"max_phase_err_rad_per_ps": 15156.318720668227, "max_population_err": 0.014308093111729599, "rows": [{"V_Ha": 0.01, "dt_fs": 0.5, "element": "C", "exact_su2_phase_err": 0.0, "max_pop_err_1ps_engine": 0.014308093111729599, "max_pop_err_1ps_ground_at_zero": 0.014306729678645234, "rabi_phase_err_rad_per_ps_engine": 4130.406713149912, "rabi_phase_err_rad_per_ps_ground_at_zero": 383.59980915634264}, {"V_Ha": 0.01, "dt_fs": 2.0, "element": "C", "exact_su2_phase_err": 0.0, "max_pop_err_1ps_engine": 0.014304324800066855, "max_pop_err_1ps_ground_at_zero": 0.014291090518486815, "rabi_phase_err_rad_per_ps_engine": 6628.952716176774, "rabi_phase_err_rad_per_ps_ground_at_zero": 2708.9179888014805}, {"V_Ha": 0.01, "dt_fs": 4.0, "element": "C", "exact_su2_phase_err": 0.0, "max_pop_err_1ps_engine": 0.014285907329987083, "max_pop_err_1ps_ground_at_zero": 0.014307805330716482, "rabi_phase_err_rad_per_ps_engine": 6839.1206964927605, "rabi_phase_err_rad_per_ps_ground_at_zero": 4311.110544275502}, {"V_Ha": 0.01, "dt_fs": 0.5, "element": "H", "exact_su2_phase_err": 0.0, "max_pop_err_1ps_engine": 0.0027622005280319604, "max_pop_err_1ps_ground_at_zero": 0.0027613114200483735, "rabi_phase_err_rad_per_ps_engine": 5934.693257129836, "rabi_phase_err_rad_per_ps_ground_at_zero": 3295.153124384455}, {"V_Ha": 0.01, "dt_fs": 2.0, "element": "H", "exact_su2_phase_err": 0.0, "max_pop_err_1ps_engine": 0.0027511238467940874, "max_pop_err_1ps_ground_at_zero": 0.002758135136861148, "rabi_phase_err_rad_per_ps_engine": 13777.107162553444, "rabi_phase_err_rad_per_ps_ground_at_zero": 10434.033239944098}, {"V_Ha": 0.01, "dt_fs": 4.0, "element": "H", "exact_su2_phase_err": 0.0, "max_pop_err_1ps_engine": 0.0027493389522054764, "max_pop_err_1ps_ground_at_zero": 0.0027606385214648654, "rabi_phase_err_rad_per_ps_engine": 15156.318720668227, "rabi_phase_err_rad_per_ps_ground_at_zero": 12832.084850724705}]}, "pass": false, "tolerance": {"phase_rad_per_ps": 0.001, "population": 0.0001}, "verdict": "KNOWN_DEFECT_OPEN"}
{"builtin_fallback": false, "defect_id": "D-NBL-MISS-10-12", "expected": "cell-scheme pair set == brute-force pair set for r <= 12 A (the list cutoff)", "inputs": {"1BTL": "INPUT_NOT_FOUND: unset (set $PRISM_REFS_TOPOLOGY_1BTL)", "IL2": "INPUT_NOT_FOUND: unset (set $PRISM_REFS_TOPOLOGY_IL2)", "source_anchors": [{"cited_lines": "165-171", "fragments_found_nearby": true, "located_lines": "165-171", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_amber_fused.cu", "sha256": "46287f27d2c49231de8376c1a58cc60e18fff3fb1cf44a6b0f818f2f643f8f30", "transcription_current": true}, {"cited_lines": "11704-11717", "fragments_found_nearby": true, "located_lines": "11704-11717", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_amber_fused.cu", "sha256": "46287f27d2c49231de8376c1a58cc60e18fff3fb1cf44a6b0f818f2f643f8f30", "transcription_current": true}, {"cited_lines": "42-51", "fragments_found_nearby": true, "located_lines": "42-51", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_neighbor_warp.cuh", "sha256": "f7c489740080c04e04818a6686f13c35d14cddc68c14b03b9077db03638bc771", "transcription_current": true}, {"cited_lines": "172-181", "fragments_found_nearby": true, "located_lines": "172-181", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_neighbor_warp.cuh", "sha256": "f7c489740080c04e04818a6686f13c35d14cddc68c14b03b9077db03638bc771", "transcription_current": true}, {"cited_lines": "12355-12388", "fragments_found_nearby": true, "located_lines": "12272-12305", "path": "<engine_root>/crates/prism-nhs/src/fused_engine.rs", "sha256": "59a75928d85b580bd3ee9fd7ba0bd53ab2bfa0d75e3b2ffd0923fe940f80e309", "transcription_current": false}, {"cited_lines": "16741-16744", "fragments_found_nearby": true, "located_lines": "16671-16674", "path": "<engine_root>/crates/prism-nhs/src/fused_engine.rs", "sha256": "59a75928d85b580bd3ee9fd7ba0bd53ab2bfa0d75e3b2ffd0923fe940f80e309", "transcription_current": false}], "synthetic_600": "seeded uniform sphere r=11.3 A, dmin 1 A, seed 20261023"}, "module": "neighbor_list.py", "name": "neighbor.cell_scheme_vs_bruteforce", "notes": "INPUT_NOT_FOUND for IL2/1BTL (synthetic system still evaluated). Brute force uses the engine's FP32 r^2 and the same <= comparison. No pair within the 10 A potential cutoff is missed at build time; pairs in the 10-12 A buffer two cells apart are, so a pair that closes inside 10 A during the 20-step reuse window (no displacement trigger) is absent from the force sum until the next rebuild. Clamping is not a miss mechanism (it makes cells closer); after drift past the fixed 10 A padding it piles atoms into face cells (occupancy reported vs the 512 trap) and the list is over-inclusive. Fix options: cell size >= list cutoff (12 A), or a 5x5x5 stencil, plus a half-skin displacement trigger.", "observed": {"missed_in_buffer_total": 304, "missed_within_potential_cutoff_total": 0, "real_systems_evaluated": [], "systems": {"synthetic_600": {"cutoff_sq": 144.0, "grid": [5, 5, 5], "max_cell_occupancy": 81, "missed_10_10p25A": 0, "missed_10_10p5A": 6, "missed_10_11A": 33, "missed_10_12A": 304, "missed_fraction_of_buffer_pairs": 0.01037471844925261, "missed_le_10A": 0, "missed_total": 304, "n_atoms": 600, "pairs_in_buffer_10_12A": 29302, "pairs_kept_cell_scheme": 93912, "pairs_le_12A_bruteforce": 94216, "spurious_pairs": 0}}}, "pass": false, "tolerance": 0, "verdict": "KNOWN_DEFECT_OPEN"}
{"builtin_fallback": true, "expected": {"blue_shift_4fs_pct": 11.1, "dt_3to1_fs": 4.89, "gears_fs": [0.5, 2.0, 4.0, 0.0], "omega_dt_4fs": 1.4164}, "inputs": {"omega_max_ps^-1": 354.1, "period_fs": 17.74}, "module": "builtin_fallback.py", "name": "omega_dt_resonance", "notes": "Velocity-Verlet n:1 resonance at omega*dt = 2 sin(pi/n). 4 fs sits on 4:1 (sqrt 2). A PASS reproduces the plan's resonance table; it does not qualify any timestep.", "observed": {"gears_fs_in_source": [0.5, 2.0, 4.0, 0.0], "nearest_resonance_4fs": 4, "omega_dt_4fs": 1.4164, "resonance_dt_fs": {"2:1": 5.6481, "3:1": 4.8914, "4:1": 3.9938, "5:1": 3.3199, "6:1": 2.8241}, "table": [{"blue_shift_pct": 0.131, "dt_fs": 0.5, "linearly_stable": true, "omega_dt": 0.17705}, {"blue_shift_pct": 2.217, "dt_fs": 2.0, "linearly_stable": true, "omega_dt": 0.7082}, {"blue_shift_pct": 11.119, "dt_fs": 4.0, "linearly_stable": true, "omega_dt": 1.4164}, {"blue_shift_pct": 22.794, "dt_fs": 5.0, "linearly_stable": true, "omega_dt": 1.7705}, {"blue_shift_pct": null, "dt_fs": 6.0, "linearly_stable": false, "omega_dt": 2.1246}]}, "pass": true, "tolerance": "1e-3 in omega*dt; 0.2% shift", "verdict": "PASS"}
{"builtin_fallback": true, "expected": "byte-exact round trip + bit-flip refusal", "inputs": {}, "module": "builtin_fallback.py", "name": "qnanl_roundtrip", "notes": "NOT_BUILT: no independent (out-of-process, non-Rust) PRSQNB01 decoder or committed fixture exists yet (G-Q2). The Rust codec's own round-trip tests are not independent references.", "observed": null, "pass": null, "tolerance": "exact", "verdict": "NOT_BUILT"}
{"builtin_fallback": false, "expected": "bit-exact Random123 outputs", "inputs": {"vectors": "Random123 kat_vectors philox4x32_10"}, "module": "rng_streams.py", "name": "rng.philox4x32_10_known_answer", "notes": "Reference implementation used by the stream-independence test and the H0 target keying.", "observed": [{"ctr": ["0x0", "0x0", "0x0", "0x0"], "expected": ["0x6627e8d5", "0xe169c58d", "0xbc57ac4c", "0x9b00dbd8"], "key": ["0x0", "0x0"], "observed": ["0x6627e8d5", "0xe169c58d", "0xbc57ac4c", "0x9b00dbd8"], "ok": true}, {"ctr": ["0xffffffff", "0xffffffff", "0xffffffff", "0xffffffff"], "expected": ["0x408f276d", "0x41c83b0e", "0xa20bc7c6", "0x6d5451fd"], "key": ["0xffffffff", "0xffffffff"], "observed": ["0x408f276d", "0x41c83b0e", "0xa20bc7c6", "0x6d5451fd"], "ok": true}, {"ctr": ["0x243f6a88", "0x85a308d3", "0x13198a2e", "0x3707344"], "expected": ["0xd16cfe09", "0x94fdcceb", "0x5001e420", "0x24126ea1"], "key": ["0xa4093822", "0x299f31d0"], "observed": ["0xd16cfe09", "0x94fdcceb", "0x5001e420", "0x24126ea1"], "ok": true}], "pass": true, "tolerance": 0, "verdict": "PASS"}
{"builtin_fallback": false, "expected": "cross-stream |corr| < 5/sqrt(N); chi2 < mean+5sd; purpose keys decorrelate; bitwise replay", "inputs": {"atoms": 64, "counter": "(atom, step_lo32, step_hi16<<16|purpose<<8|replica_epoch, 0)", "key": "(seed, stream)", "steps": 64, "streams": 8}, "module": "rng_streams.py", "name": "rng.philox_stream_independence", "notes": "Reference keying for H0 (counter-based, order-independent, stream/replica/purpose separated).", "observed": {"corr_bound_5sigma": 0.0390625, "deterministic_replay": true, "max_abs_cross_stream_corr": 0.015156207463668706, "purpose_decorrelation_abs_corr": 0.00692524704012072, "samples_per_stream": 16384, "uniformity_chi2_63dof": 57.8232421875}, "pass": true, "tolerance": {"chi2": 119.12486080160912, "corr": 0.0390625}, "verdict": "PASS"}
{"builtin_fallback": false, "defect_id": "D-RNG-UV-LCG-STREAM", "expected": "kick sequences differ across streams (fraction identical == 0)", "inputs": {"model": "8 streams, 12 aromatics, 20 neighbours, 200 steps", "source_anchors": [{"cited_lines": "11060-11072", "fragments_found_nearby": true, "located_lines": "11060-11072", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_amber_fused.cu", "sha256": "46287f27d2c49231de8376c1a58cc60e18fff3fb1cf44a6b0f818f2f643f8f30", "transcription_current": true}, {"cited_lines": "11018-11018", "fragments_found_nearby": true, "located_lines": "11018-11018", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_amber_fused.cu", "sha256": "46287f27d2c49231de8376c1a58cc60e18fff3fb1cf44a6b0f818f2f643f8f30", "transcription_current": true}, {"cited_lines": "530-560", "fragments_found_nearby": true, "located_lines": "530-560", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_excited_state.cuh", "sha256": "de02a444844fbbd1961175364a82b67c5717794039b1c1b47f6d4f61fa14e2fa", "transcription_current": true}]}, "module": "rng_streams.py", "name": "rng.uv_lcg_stream_identity", "notes": "seed = timestep*n_aromatics + arom_idx has no stream/replica/purpose term, so all streams sharing current_step draw identical kick directions (fraction 1.0): stream-to-stream UV responses are not independent samples. Consecutive seeds give first draws that advance by a constant 0.5139 mod 1 (std 0), a deterministic lattice. Philox keyed by (seed, stream) gives no cross-stream collisions.", "observed": {"aromatics": 12, "first_draw_increment_between_consecutive_seeds": {"mean": 0.5138643295504526, "std": 2.3274214343643085e-10}, "fraction_identical_kick_sequences_across_streams": 1.0, "neighbors": 20, "philox_keyed_collisions_across_streams": 0, "steps": 200, "streams": 8}, "pass": false, "tolerance": 0.0, "verdict": "KNOWN_DEFECT_OPEN"}
{"builtin_fallback": false, "expected": "plan statements: 4 fs on 4:1 with ~11% blue shift; 3:1 at 4.9 fs; 5 fs past 3:1; 6 fs unstable", "inputs": {"dts_fs": [2, 4, 5, 6], "source_anchors": [{"cited_lines": "396-399", "fragments_found_nearby": true, "located_lines": "396-399", "path": "<engine_root>/crates/prism-nhs/src/cuda/gearbox.cuh", "sha256": "d09f87ea71c709e8ba000212a0fed833f9e1a90cfb56b299ad8bb1169342d3da", "transcription_current": true}], "w_max_per_ps": 354.1}, "module": "timestep_resonance.py", "name": "timestep.resonance_table", "notes": "All plan statements reproduced. 4 fs sits 0.16% above the 4:1 limit sqrt(2); a 1-DOF cubic anharmonic Verlet test (A=0.5, alpha=-0.15, unit frequency, 5000 steps) escapes in a narrow band at w dt ~1.415-1.42, i.e. exactly where 4 fs puts the fastest mode, and is unstable from ~1.685 (amplitude-softened 3:1). The 4:1 limit for w_max is dt=3.994 fs. BAOAB friction does not rescue 6 fs (mean-map spectral radius 1.73-2.01 for gamma 59-1/ps). The circular 1.4272 level corresponds to w=356.8/ps or dt=4.03 fs at w_max.", "observed": {"baoab_mean_map_spectral_radius_6fs": {"gamma_10_per_ps": 1.9603089841393793, "gamma_1_per_ps": 2.012478789184189, "gamma_59_per_ps": 1.7345491037997252}, "cubic_oscillator_escape_w_dt_near_3to1": [1.685, 1.7675], "cubic_oscillator_escape_w_dt_near_4to1": [1.4175, 1.42], "dt_fs_at_limits": {"2:1": 5.648121999435188, "3:1": 4.8914171351846285, "4:1": 3.993825366769542, "5:1": 3.3198828144166797, "6:1": 2.8240609997175934}, "limits_w_dt": {"2:1": 2.0, "3:1": 1.7320508075688772, "4:1": 1.414213562373095, "5:1": 1.1755705045849463, "6:1": 0.9999999999999999}, "period_fs": 17.74409858000448, "plan_statements": {"3to1_at_4p9fs": {"observed_fs": 4.8914171351846285, "ok": true, "plan": 4.9}, "4fs_blue_shift": {"observed": 0.11119091480768728, "ok": true, "plan": 0.11}, "4fs_on_4to1": {"observed_rel_offset": 0.0015460448726259182, "ok": true, "plan": "on 4:1 (sqrt2=1.41421)"}, "4fs_w_dt": {"observed": 1.4164, "ok": true, "plan": 1.4164}, "5fs_past_3to1": {"observed_w_dt": 1.7705, "ok": true}, "6fs_linearly_unstable": {"observed_w_dt": 2.1246000000000005, "ok": true, "spectral_radius": 2.018510298478797}, "mts_inner_2fs_w_dt": {"observed": 0.7082, "ok": true}, "mts_inner_2p5fs_w_dt": {"observed": 0.8852500000000001, "ok": true}, "mts_outer_6fs_near_T_over_3": {"T_over_2_fs": 8.87204929000224, "T_over_3_fs": 5.9146995266681595, "ok": true, "rel": 0.014421776279122603}, "nyquist_4fs_cm-1": {"observed": 4169.5511899769, "ok": true, "plan": 4170}, "period_17p75fs": {"observed": 17.74409858000448, "ok": true, "plan": 17.75}, "qualified_1p4272_is_not_w_max_at_4fs": {"1.4272/0.004ps": 356.8, "implied_dt_fs_at_w_max": 4.03049985879695, "ok": true}}, "rows": [{"dt_fs": 2.0, "linearly_stable": true, "nearest_limit": "6:1", "past_limits": [], "rel_dist_to_nearest": 0.2917999999999999, "verlet_spectral_radius": 1.0, "w_dt": 0.7082, "w_eff_over_w": 1.0221732237237502}, {"dt_fs": 4.0, "linearly_stable": true, "nearest_limit": "4:1", "past_limits": ["4:1", "5:1", "6:1"], "rel_dist_to_nearest": 0.0015460448726260755, "verlet_spectral_radius": 1.0, "w_dt": 1.4164, "w_eff_over_w": 1.1111909148076873}, {"dt_fs": 5.0, "linearly_stable": true, "nearest_limit": "3:1", "past_limits": ["3:1", "4:1", "5:1", "6:1"], "rel_dist_to_nearest": 0.022198651600232454, "verlet_spectral_radius": 1.0, "w_dt": 1.7705, "w_eff_over_w": 1.2279370091018913}, {"dt_fs": 6.0, "linearly_stable": false, "nearest_limit": "2:1", "past_limits": ["2:1", "3:1", "4:1", "5:1", "6:1"], "rel_dist_to_nearest": 0.062300000000000244, "verlet_spectral_radius": 2.018510298478797, "w_dt": 2.1246000000000005, "w_eff_over_w": null}]}, "pass": true, "tolerance": {"rel_to_limit": 0.01, "w_dt_abs": 0.0005}, "verdict": "PASS"}
{"builtin_fallback": true, "expected": {"BIPRISM_ANGSTROM_TO_BOHR": 1.8897261259077822, "BIPRISM_PS_TO_ATOMIC_TIME": 41341.37333517015, "COULOMB_CONSTANT": 332.0637130741707, "E_photon_280nm_eV": 4.428007086900009, "E_photon_280nm_kcal_per_mol": 102.11226922177768, "hartree_kcal_per_mol": 627.5094740628974}, "inputs": {"codata": "2022", "kcal": "4184 J"}, "module": "builtin_fallback.py", "name": "unit_conversions", "notes": "k_C reference: AMBER uses 332.0522173, CHARMM 332.0716; the CODATA 2022 value is ~332.0637. Bridge constants are compared with CODATA 2022 a0 and atomic time; a mismatch at ~1e-10 relative indicates an older CODATA vintage (record it, do not silently change physics).", "observed": {"relative_error": {"BIPRISM_ANGSTROM_TO_BOHR": 1.175009446505726e-16, "BIPRISM_PS_TO_ATOMIC_TIME": 0.0, "COULOMB_CONSTANT": 3.4051950351972965e-07}, "source_constants": {"BIPRISM_ANGSTROM_TO_BOHR": 1.8897261259077824, "BIPRISM_PS_TO_ATOMIC_TIME": 41341.37333517015, "COULOMB_CONSTANT": 332.0636}}, "pass": true, "tolerance": {"relative": {"BIPRISM_ANGSTROM_TO_BOHR": 1e-09, "BIPRISM_PS_TO_ATOMIC_TIME": 1e-09, "COULOMB_CONSTANT": 2e-06}}, "verdict": "PASS"}
{"builtin_fallback": false, "expected": "engine constants equal CODATA 2022 derivations within per-constant tolerance", "inputs": {"profile": "CODATA 2022", "source_anchors": [{"cited_lines": "66-66", "fragments_found_nearby": true, "located_lines": "66-66", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_amber_fused.cu", "sha256": "46287f27d2c49231de8376c1a58cc60e18fff3fb1cf44a6b0f818f2f643f8f30", "transcription_current": true}, {"cited_lines": "37-40", "fragments_found_nearby": true, "located_lines": "37-40", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_excited_state.cuh", "sha256": "de02a444844fbbd1961175364a82b67c5717794039b1c1b47f6d4f61fa14e2fa", "transcription_current": true}, {"cited_lines": "67-87", "fragments_found_nearby": true, "located_lines": "67-87", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_excited_state.cuh", "sha256": "de02a444844fbbd1961175364a82b67c5717794039b1c1b47f6d4f61fa14e2fa", "transcription_current": true}, {"cited_lines": "90-98", "fragments_found_nearby": true, "located_lines": "90-98", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_excited_state.cuh", "sha256": "de02a444844fbbd1961175364a82b67c5717794039b1c1b47f6d4f61fa14e2fa", "transcription_current": true}, {"cited_lines": "108-108", "fragments_found_nearby": true, "located_lines": "108-108", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_excited_state.cuh", "sha256": "de02a444844fbbd1961175364a82b67c5717794039b1c1b47f6d4f61fa14e2fa", "transcription_current": true}, {"cited_lines": "209-231", "fragments_found_nearby": true, "located_lines": "209-231", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_excited_state.cuh", "sha256": "de02a444844fbbd1961175364a82b67c5717794039b1c1b47f6d4f61fa14e2fa", "transcription_current": true}, {"cited_lines": "9-12", "fragments_found_nearby": true, "located_lines": "9-12", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_biprism_unit_bridge.cuh", "sha256": "de47488e2cd3e678cf3af21c8b493b5bc926fcfc4e938fdc3f108d2e71272f30", "transcription_current": true}]}, "module": "units.py", "name": "units.constants_codata2022", "notes": "332.0636 is alpha*hbar*c*N_A/(1 A * 4184 J/kcal) to 2e-7 (CODATA 2022 gives 332.06371); it is not the AMBER 332.0522 (18.2223^2) value, so energies differ from AMBER/pmemd by 3.5e-5 relative. EV_TO_KCAL_MOL=23.06 is truncated (2.4e-5 rel). Bridge constants match CODATA 2022.", "observed": {"Ha_to_kcal_mol": 627.5094740628974, "atomic_time_fs": 0.024188843265863295, "constants": {"BIPRISM_ANGSTROM_TO_BOHR": {"codata2022": 1.8897261259077824, "engine": 1.8897261259077824, "ok": true, "rel": 0.0, "tol": 1e-12}, "BIPRISM_NATIVE_FORCE_TO_AU": {"codata2022": 0.0008432975634897883, "engine": 0.0008432975634897884, "ok": true, "rel": 1.285669755760694e-16, "tol": 1e-10}, "BIPRISM_NATIVE_VELOCITY_TO_AU": {"codata2022": 4.571028907499039e-05, "engine": 4.571028907499172e-05, "ok": true, "rel": 2.9055770334679776e-14, "tol": 1e-10}, "BIPRISM_PS_TO_ATOMIC_TIME": {"codata2022": 41341.37333517135, "engine": 41341.37333517015, "ok": true, "rel": 2.903950472586034e-14, "tol": 1e-12}, "COULOMB_CONSTANT": {"amber_332.0522173_rel": 3.4619177633225297e-05, "charmm_332.0716_rel": 2.375124718868988e-05, "codata2022": 332.0637130726684, "engine": 332.0636, "engine_float32": 332.0635986328125, "ok": true, "rel": 3.4051497934516204e-07, "rel_float32": 3.446322238806868e-07, "tol": 1e-06}, "ENERGY_VEL_PREFACTOR": {"codata2022": 28.92749557082327, "engine": 28.93, "ok": true, "rel": 8.657608020702637e-05, "tol": 0.0005}, "EPSILON_TO_SIGMA_FACTOR": {"codata2022": 3.823532502408737e-05, "engine": 3.823e-05, "ok": true, "rel": 0.00013926974817186075, "tol": 0.0005}, "EV_TO_KCAL_MOL": {"codata2022": 23.06054783061903, "engine": 23.06, "ok": true, "rel": 2.3756184070489822e-05, "tol": 5e-05}, "HC_EV_NM": {"codata2022": 1239.8419843320025, "engine": 1239.84, "ok": true, "rel": 1.6004716953429106e-06, "tol": 2e-06}, "UV_PHOTON_ENERGY_EV": {"codata2022": 4.428007086900009, "engine": 4.428, "ok": true, "rel": 1.6004716952741398e-06, "tol": 0.0005}}, "eV_to_kcal_mol": 23.06054783061903, "kcal_mol_to_amu_A2_ps2": 418.4}, "pass": true, "tolerance": {"BIPRISM_ANGSTROM_TO_BOHR": 1e-12, "BIPRISM_NATIVE_FORCE_TO_AU": 1e-10, "BIPRISM_NATIVE_VELOCITY_TO_AU": 1e-10, "BIPRISM_PS_TO_ATOMIC_TIME": 1e-12, "COULOMB_CONSTANT": 1e-06, "ENERGY_VEL_PREFACTOR": 0.0005, "EPSILON_TO_SIGMA_FACTOR": 0.0005, "EV_TO_KCAL_MOL": 5e-05, "HC_EV_NM": 2e-06, "UV_PHOTON_ENERGY_EV": 0.0005}, "verdict": "PASS"}
{"builtin_fallback": false, "expected": "E_dep units kcal/mol; TRP@280 dT ~20 K with N_eff=9 as the source comment states", "inputs": {"F_photons_per_A2": 0.024, "bands": {"PHE": [258.0, 200.0, 10.0], "SS": [250.0, 300.0, 20.0], "TRP": [280.0, 5600.0, 15.0], "TYR": [274.0, 1490.0, 12.0]}, "source_anchors": [{"cited_lines": "66-66", "fragments_found_nearby": true, "located_lines": "66-66", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_amber_fused.cu", "sha256": "46287f27d2c49231de8376c1a58cc60e18fff3fb1cf44a6b0f818f2f643f8f30", "transcription_current": true}, {"cited_lines": "37-40", "fragments_found_nearby": true, "located_lines": "37-40", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_excited_state.cuh", "sha256": "de02a444844fbbd1961175364a82b67c5717794039b1c1b47f6d4f61fa14e2fa", "transcription_current": true}, {"cited_lines": "67-87", "fragments_found_nearby": true, "located_lines": "67-87", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_excited_state.cuh", "sha256": "de02a444844fbbd1961175364a82b67c5717794039b1c1b47f6d4f61fa14e2fa", "transcription_current": true}, {"cited_lines": "90-98", "fragments_found_nearby": true, "located_lines": "90-98", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_excited_state.cuh", "sha256": "de02a444844fbbd1961175364a82b67c5717794039b1c1b47f6d4f61fa14e2fa", "transcription_current": true}, {"cited_lines": "108-108", "fragments_found_nearby": true, "located_lines": "108-108", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_excited_state.cuh", "sha256": "de02a444844fbbd1961175364a82b67c5717794039b1c1b47f6d4f61fa14e2fa", "transcription_current": true}, {"cited_lines": "209-231", "fragments_found_nearby": true, "located_lines": "209-231", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_excited_state.cuh", "sha256": "de02a444844fbbd1961175364a82b67c5717794039b1c1b47f6d4f61fa14e2fa", "transcription_current": true}, {"cited_lines": "9-12", "fragments_found_nearby": true, "located_lines": "9-12", "path": "<engine_root>/crates/prism-gpu/src/kernels/nhs_biprism_unit_bridge.cuh", "sha256": "de47488e2cd3e678cf3af21c8b493b5bc926fcfc4e938fdc3f108d2e71272f30", "transcription_current": true}], "wavelengths_nm": [280, 274, 258, 254, 211]}, "module": "units.py", "name": "units.photon_energy_and_edep", "notes": "E_dep = E_photon*sigma*F*eta is dimensionally consistent iff F is photons/A^2 (as the comment states); 0.024 has no recorded provenance (it is calibrated to the dT~20 K target, reproduced here). Single Gaussian bands: at 211 nm every chromophore sigma is ~0 (TRP 1.4e-10 A^2, TYR 6e-14 A^2) although Trp has a ~33,000 M^-1cm^-1 B-band near 218 nm and the peptide bond absorbs strongly below 230 nm, so the 211/254/258 nm channels deposit essentially nothing for TRP/TYR: a calibration defect for multi-wavelength action spectra, not a unit defect.", "observed": {"211nm_sigma_all_chromophores_A2": {"PHE": 1.9492572412386667e-12, "SS": 0.00025592975431099817, "TRP": 1.3833149072056041e-10, "TYR": 6.100267350067099e-14}, "E_dep": {"PHE": {"211": {"E_dep_Ha": 1.0101889229829655e-14, "E_dep_kcal_mol": 6.339031197652055e-12, "dT_K_over_Neff": 2.362906905271548e-10, "p_absorb": 4.6782173789728e-14, "sigma_A2": 1.9492572412386667e-12}, "254": {"E_dep_Ha": 2.804977363613264e-05, "E_dep_kcal_mol": 0.01760149870199292, "dT_K_over_Neff": 0.6561050344960005, "p_absorb": 0.00015637179376648864, "sigma_A2": 0.006515491406937027}, "258": {"E_dep_Ha": 3.240637765259013e-05, "E_dep_kcal_mol": 0.020335308997060465, "dT_K_over_Neff": 0.7580092375595572, "p_absorb": 0.000183504, "sigma_A2": 0.007646}, "274": {"E_dep_Ha": 2.358879491377379e-06, "E_dep_kcal_mol": 0.001480219229011974, "dT_K_over_Neff": 0.05517594295550432, "p_absorb": 1.4185728968430282e-05, "sigma_A2": 0.0005910720403512617}, "280": {"E_dep_Ha": 2.3610591318830735e-07, "E_dep_kcal_mol": 0.00014815869740793486, "dT_K_over_Neff": 0.005522692636548576, "p_absorb": 1.4509760231148982e-06, "sigma_A2": 6.045733429645409e-05}}, "SS": {"211": {"E_dep_Ha": 1.3263380399317288e-06, "E_dep_kcal_mol": 0.0008322896858671733, "dT_K_over_Neff": 0.1396081424859711, "p_absorb": 6.142314103463956e-06, "sigma_A2": 0.00025592975431099817}, "254": {"E_dep_Ha": 4.743904721248562e-05, "E_dep_kcal_mol": 0.02976845156635181, "dT_K_over_Neff": 4.9933554367333866, "p_absorb": 0.0002644630578278097, "sigma_A2": 0.011019294076158736}, "258": {"E_dep_Ha": 4.1422340137079605e-05, "E_dep_kcal_mol": 0.02599291087387327, "dT_K_over_Neff": 4.360046828075167, "p_absorb": 0.000234557690649733, "sigma_A2": 0.009773237110405541}, "274": {"E_dep_Ha": 1.0844432627076378e-05, "E_dep_kcal_mol": 0.0068049842143272235, "dT_K_over_Neff": 1.1414669939334035, "p_absorb": 6.521578682863577e-05, "sigma_A2": 0.0027173244511931567}, "280": {"E_dep_Ha": 4.720856886480806e-06, "E_dep_kcal_mol": 0.002962382421961778, "dT_K_over_Neff": 0.49690956680818327, "p_absorb": 2.9011768736928545e-05, "sigma_A2": 0.0012088236973720227}}, "TRP": {"211": {"E_dep_Ha": 7.168932692374241e-13, "E_dep_kcal_mol": 4.4985731833840714e-10, "dT_K_over_Neff": 1.676866591668585e-08, "p_absorb": 3.31995577729345e-12, "sigma_A2": 1.3833149072056041e-10}, "254": {"E_dep_Ha": 4.568365695644573e-05, "E_dep_kcal_mol": 0.028666927550009085, "dT_K_over_Neff": 1.0685746598932213, "p_absorb": 0.000254677113504053, "sigma_A2": 0.010611546396002206}, "258": {"E_dep_Ha": 0.00010557785432012783, "E_dep_kcal_mol": 0.06625110383711262, "dT_K_over_Neff": 2.4695444123474153, "p_absorb": 0.0005978440042530408, "sigma_A2": 0.0249101668438767}, "274": {"E_dep_Ha": 0.00072806565729991, "E_dep_kcal_mol": 0.4568680976955242, "dT_K_over_Neff": 17.029996369837797, "p_absorb": 0.004378410225461682, "sigma_A2": 0.18243375939423676}, "280": {"E_dep_Ha": 0.0008360845434368253, "E_dep_kcal_mol": 0.52465097212416, "dT_K_over_Neff": 19.556638329036577, "p_absorb": 0.005138112, "sigma_A2": 0.214088}}, "TYR": {"211": {"E_dep_Ha": 3.1614208601613224e-16, "E_dep_kcal_mol": 1.9838215412513042e-13, "dT_K_over_Neff": 6.655318336888559e-12, "p_absorb": 1.4640641640161039e-15, "sigma_A2": 6.100267350067099e-14}, "254": {"E_dep_Ha": 1.5247519098463785e-05, "E_dep_kcal_mol": 0.009567962690240994, "dT_K_over_Neff": 0.32098571476777793, "p_absorb": 8.500182364552973e-05, "sigma_A2": 0.003541742651897072}, "258": {"E_dep_Ha": 4.080446424541846e-05, "E_dep_kcal_mol": 0.02560518789806084, "dT_K_over_Neff": 0.8590020472806945, "p_absorb": 0.00023105891337697214, "sigma_A2": 0.009627454724040505}, "274": {"E_dep_Ha": 0.00022732955651840684, "E_dep_kcal_mol": 0.1426514504498172, "dT_K_over_Neff": 4.785666423218621, "p_absorb": 0.0013671048, "sigma_A2": 0.0569627}, "280": {"E_dep_Ha": 0.00017325062713005455, "E_dep_kcal_mol": 0.10871640991144768, "dT_K_over_Neff": 3.647214738619953, "p_absorb": 0.0010647022878873744, "sigma_A2": 0.0443625953286406}}}, "TRP_280_dT_K_with_Neff9": 19.556638329036577, "dimensional_analysis": {"E_dep": "E_photon[kcal/mol photons] * p * eta -> kcal/mol per chromophore (expected molar energy per pulse)", "consistent": true, "p_absorb": "sigma[A^2] * F[photons/A^2] -> dimensionless; requires F in photons/A^2 (declared in comment :33,:39)", "sigma": "eps[M^-1 cm^-1] * (1000 cm^3/L) * ln10 / N_A -> cm^2; *1e16 -> A^2  (factor 3.8235e-5)"}, "engine_target_comment": "TRP @ 280nm -> dT ~ 20K (:34)", "photon_energies": {"211": {"Ha": 0.21594005937977057, "eV": 5.876028361763045, "engine_kcal_mol": 135.50099715639809, "kcal_mol": 135.50443309051067}, "254": {"Ha": 0.1793832776737464, "eV": 4.8812676548504035, "engine_kcal_mol": 112.56185196850393, "kcal_mol": 112.56470622873131}, "258": {"Ha": 0.17660214158578136, "eV": 4.805589086558149, "engine_kcal_mol": 110.81670697674419, "kcal_mol": 110.81951698487501}, "274": {"Ha": 0.16628960777055324, "eV": 4.524970745737235, "engine_kcal_mol": 104.34565839416057, "kcal_mol": 104.34830431422537}, "280": {"Ha": 0.16272625903261284, "eV": 4.428007086900009, "engine_kcal_mol": 102.10968, "kcal_mol": 102.1122692217777}}}, "pass": true, "tolerance": {"dT_K": 1.0}, "verdict": "PASS"}
```

</details>
