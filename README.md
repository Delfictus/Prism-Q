# Prism-Q

Prism-Q is a GPU molecular-dynamics engine for proteins. It is designed as a **hybrid**: classical MD
propagation with a BIPRISM quantum-analogue layer (mesh/DEC geometry, a per-atom discrete electronic
state, a LIF event detector and intervention work such as multi-wavelength UV) in the same step loop.
Each run is written once to an exact-clock record, the **Chronicle**. All later analysis (Path B site
materialization and DSTW export) replays that record offline and never reruns MD.

> **Status in one line:** the classical MD path, the Chronicle and the Path B/DSTW consumers run but are
> not qualified. The in-loop quantum-analogue layer (H1–H3) is **not built**. Electrostatics are vacuum
> today. A critical defect in the DEC operator (opposite-edge weights) is open. The per-layer table is below.

**Reviewers (human or AI): start with [`REVIEW_BUNDLE.md`](REVIEW_BUNDLE.md) and [`AGENTS.md`](AGENTS.md).**

## Architecture

```text
 prepared topology (AMBER parameters, residue identity, HMR)
        |
        v
 +----------------------------- one MD step (captured CUDA graph) -----------------------------+
 |  neighbour list -> bonded + nonbonded forces (+ solvent field: NOT BUILT)                    |
 |    -> Langevin integrator phases (OU / kick / drift / SHAKE / RATTLE) <- thermostat + RNG    |
 |    -> UV excited state + intervention work (W_uv)                                            |
 |    -> BIPRISM mesh / DEC operator -> discrete electronic analogue (in-loop: NOT BUILT)       |
 |    -> LIF / Voronoi detector -> spike events                                                 |
 +----------------------------------------------------------------------------------------------+
        |   every datum bound to a clock key and an atom / residue / mesh anchor
        v
 Chronicle L0-L5  (raw spikes, accepted-step chronology, epochs/events, frames, interventions, restart anchors)
        |
        v
 QNANL packet encode / seal / durable publish          (post-run today; in-loop sealing planned)
        |
        v
 md_evidence_manifest.json                              (replay contract; no MD rerun past this point)
        |
        v
 Path B materialization (prism_materialize_sites)  ->  DSTW export (dstw_export_wt_pathb)
```

## Per-layer status

Status words are used strictly:
- **QUALIFIED** means the layer passed its stated gate and a receipt exists.
- **EXECUTED_UNQUALIFIED** means it runs, but its gate has not passed.
- **NOT_BUILT** means it does not exist yet.

`REVIEW_BUNDLE.md` is generated from the source and is authoritative where this table and it differ.

| # | Layer | Status | Main open items |
|---|---|---|---|
| S01 | Topology / prep (HMR, residue identity) | EXECUTED_UNQUALIFIED | residue index converted with `lr.resid - 1` downstream; PDB number / insertion code not carried end to end |
| S02 | Bonded + nonbonded forces | EXECUTED_UNQUALIFIED | **vacuum electrostatics** (ε = 1, 10 Å potential-shift cutoff); no solvent |
| S03 | Integrator (BAOAB/OVRVO phases, SHAKE/RATTLE) | EXECUTED_UNQUALIFIED | 4 fs sits on the 4:1 nonlinear resonance; no MTS; no 5/6 fs gear |
| S04 | Thermostat / RNG / friction ramp | EXECUTED_UNQUALIFIED | XORWOW (not counter-based); UV LCG gives identical kicks across streams; friction ramp discontinuity |
| S05 | Neighbour list | EXECUTED_UNQUALIFIED | non-periodic, clamps at grid faces, can miss 10–12 Å pairs; fixed 20-step rebuild |
| S06 | Solvent (MVS-DS) | **NOT_BUILT** | the engine is vacuum until this lands; pseudo-water mode never satisfies a solvent claim |
| S07 | UV excited state + `W_uv[λ]` ledger | EXECUTED_UNQUALIFIED | work booked in a mixed scalar plane; excitations are prescribed inputs |
| S08 | BIPRISM mesh / DEC operator | EXECUTED_UNQUALIFIED | **CRITICAL: DEC edge weights assigned to the opposite edge**; no physical scale in the PCG solve |
| S09 | Electronic propagator + TB coupling | **NOT_BUILT** (in-loop H1–H3) | the Cayley map exists only in a 192-atom reference tile, where it compresses the onsite gap about 5.4× at 4 fs; uncalibrated parameters |
| S10 | LIF / Voronoi detector, event extraction | EXECUTED_UNQUALIFIED | nondeterministic FP32 flux atomics; 8-leaf limit; windows counted in steps |
| S11 | Chronicle CLOCK / leaves / restart | EXECUTED_UNQUALIFIED | fresh-process resume has never completed; coordinates not captured on qualified routes |
| S12 | QNANL encode / seal / decode | QUALIFIED (post-run codec only) | in-loop sealing, new families and an independent decoder are NOT_BUILT |
| S13 | `md_evidence_manifest` | EXECUTED_UNQUALIFIED | some status fields are literals |
| S14 | Path B materialization | EXECUTED_UNQUALIFIED | stale site views, residue offset, null-as-zero defaults |
| S15 | DSTW export | EXECUTED_UNQUALIFIED | shared baseline across streams, nominal dt, proxy naming |

Until H1–H3 pass their gates, **no run of this engine is hybrid evidence**. The electronic layer is a
phenomenological analogue labelled `analogue_uncalibrated` until tight-binding (GFN2-xTB form) grounding
passes.

## Build

Requirements:
- NVIDIA GPU with compute capability 12.0 (`sm_120`, Blackwell); other architectures are untested.
- CUDA toolkit **13.2**.
- Rust 1.97.1, pinned by `rust-toolchain.toml`.
- Python ≥ 3.10 for the review tooling.

```bash
# Release builds fail closed unless PRISM_GPU_NVCC names an absolute nvcc; `nvcc --version` is recorded in the binary.
export PRISM_GPU_NVCC=/usr/local/cuda/bin/nvcc        # CUDA 13.2
export PRISM_GPU_CUDA_ARCH=sm_120
export PRISM_GHOST_TOPOLOGY_POLICY_PATH="$PWD/BUILD_INPUTS/GHOST_TOPOLOGY_POLICY.json"
export PRISM_GHOST_TOPOLOGY_POLICY_SHA256=75183a61df6ff6b36d1145336cc9476d1a8cb86150e85ef166fe3ae4fb656ce6
export SDST_LIB_DIR=/path/to/sdst/lib                 # prebuilt libsdst.so (required even for cargo check)
export PRISM_SDST_LIB_SHA256=83d0e71e9da31b99bc2ef29bfa1845f265466a975f57d654417dc22333e00854
cargo build --release --locked -p prism-nhs --bin nhs_rt_full \
  --no-default-features --features gpu,ensemble,v2_ignition
```

Notes:
- `.cargo/config.toml` replaces crates.io with `vendor/crates-io`. Vendored crates are **not** published
  here, so run `cargo vendor vendor/crates-io` first, or remove the source replacement for an online build.
- Generated PTX/cubin are never committed. They are rebuilt by `build.rs`, and the kernel-set hash is
  recorded in the build receipt.
- The ghost topology policy is committed as `BUILD_INPUTS/GHOST_TOPOLOGY_POLICY.json`. The build binds it by
  path and sha256.
- The prebuilt SDST library is not in this repository. Point `SDST_LIB_DIR` at it and pin
  `PRISM_SDST_LIB_SHA256` (the hash is listed in `manifests/THIRD_PARTY.json`). The build fails closed when it
  is missing or mismatched. Its sources are under `.archive/crates/sdst`.
- Build outputs, runs and evidence never live in the repository. See `.gitignore`.

There is no GPU in CI. CI runs the review-bundle staleness check, the CPU reference checks and the
CPU-only tests. GPU gates (native runs, determinism, sanitizer, profiling) are recorded as receipts
outside this repository and are cited by hash.

## Runtime configuration

Nothing in the source names a host path. Roots that used to be workstation defaults now come from the
environment or CLI flags, and they fail closed when unset:

| Setting | Used by |
|---|---|
| `PRISM_WP4_CONTROL_ROOT` | `nhs_rt_full` WP4 self-test and proposal outputs (absolute admitted control root) |
| `--raw-root`, `--topology-root` | GLP-1R provenance binaries (`aromatic_kinematics`, `mechanical_load`, `signal_grid_differential`, `warp_jacobian`) |
| `PRISM_BOUNDED_NAS`, `PRISM_ACCEPTANCE_TMP`, `PRISM_DEV_SOURCE_ROOTS`, `PRISM_CHAMPION_DEVBOX_ROOT` | `prismcode` hermeticity, acceptance and model-runtime checks |
| `PRISM_BIPRISM_RESOLUTION_ROOT`, `PRISM_BIPRISM_TASK_ROOT`, `PRISM_SINGLE_CHAIN10_ROOT`, `PRISM_REFERENCE_PYTHON`, `PRISM_BIPRISM_RECORDED_ENGINE_ROOT` | BIPRISM QC harnesses under `cuda/tests/` and `scripts/` |
| `PRISM_CONTROL_ROOT`, `PRISM_INPUTS_ROOT`, `PRISM_MD_RUNS_ROOT`, `PRISM_CONSTRAINT_RECORDED_ROOT`, `PRISM_APO_CRYPTIC_MATRIX_DIR` | `#[ignore]`d tests that read sha256-pinned host data |
| `data/` (untracked) | inputs for `crates/prism-nhs/examples`, resolved through `CARGO_MANIFEST_DIR` |

`/opt/prismos/...` and `/var/lib/prismos/...` paths in the source are the documented PrismOS appliance install
layout (installed binaries, PTX and governed run roots), not workstation paths.

## Layout

| Path | Contents |
|---|---|
| `crates/prism-nhs/` | engine crate: host driver (`fused_engine.rs`), `nhs_rt_full` binary, Chronicle, checkpoints, Path B and DSTW binaries |
| `crates/prism-gpu/` | CUDA kernels (`src/kernels/*.cu`, `*.cuh`) and GPU host bindings |
| `crates/prismcode/` | QNANL packet codec and runtime evidence records |
| `crates/prism-tokenizer/`, `crates/prism-forge/`, `crates/prismos-ui/` | tokenizer, generator/scoring support, workbench UI |
| `cuda/` | standalone CUDA tests and fixtures |
| `scripts/` | review-bundle generator, reference checks (`scripts/physics_refs/`), release gate |
| `manifests/` | third-party inventory, hashes of excluded build inputs and evidence |
| `LICENSE`, `NOTICE` | all-rights-reserved terms and third-party notices |
| `review_bundle.toml` | the review bundle specification |
| `REVIEW_BUNDLE.md` | the generated review bundle (never hand-edited) |

## Review bundle

```bash
python3 scripts/build_review_bundle.py            # regenerate REVIEW_BUNDLE.md
python3 scripts/build_review_bundle.py --check    # exit 1 if the bundle is stale or an anchor is broken
python3 scripts/build_review_bundle.py --verify   # run the independent CPU reference checks
```

Every change that touches a source file listed in `review_bundle.toml` regenerates the bundle in the
same commit. CI fails if the bundle is stale.

## Claim scope

Everything here is computational evidence only. Nothing in this repository is wet-lab, clinical,
regulatory, synthesis, safety or ADMET validation. Novelty, "events others miss" and state-of-the-art
claims are not made until the falsification benchmark gates pass.

## Embedded site rankers

`crates/prism-nhs/assets/xgb_ranker_v3.onnx` and `tokenized_ranker_v4.bin` (plus `.json`) are first-party
trained weights, embedded at compile time. Their training receipts (model and evaluation records) are
**missing**: the v3 training script is known, but the receipt directories no longer exist, and no v4 training
script has been found. The bytes therefore cannot be bound to a training run. Provenance is recorded as
PARTIAL in `manifests/THIRD_PARTY.json`, and LOTO figures quoted in source comments are unverified claims.

## License

Copyright (c) 2026 Delfictus. **All rights reserved.** You may view this repository on GitHub. No
license to use, copy, modify or distribute it is granted (see [`LICENSE`](LICENSE)). Third-party components
keep their own licenses (see [`NOTICE`](NOTICE) and `manifests/THIRD_PARTY.json`).
