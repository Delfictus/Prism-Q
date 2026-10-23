//! Typed command adapters for the installed PrismOS backend binaries.
//!
//! Every backend invocation produced here is an explicit `argv` vector — a
//! binary plus a `Vec<String>` of arguments. There is **no shell**, no string
//! interpolation, and no `sh -c`: arguments are passed straight to
//! `std::process::Command`, so paths containing spaces, quotes, or shell
//! metacharacters can never be reinterpreted. This is the "typed command
//! backend" contract from the Beta2.0 brief.
//!
//! The argv shapes below are derived from the clap-derive surface of each
//! binary in `crates/prismcode/src/bin/*.rs` and `crates/prism-nhs/src/bin/*.rs`.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// Presentation-only output selection; native request admission remains authoritative.
pub fn governed_output_for_topology(topology: &Path) -> Result<PathBuf, String> {
    let profile = std::env::var_os("PRISMOS_EXECUTION_PROFILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/opt/prismos/profiles/hyper-runtime-md-v1.json"));
    let bytes = std::fs::read(&profile).map_err(|error| format!("execution profile unavailable: {error}"))?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
    let root = value.pointer("/output_and_corpus_policy/run_output_root")
        .and_then(serde_json::Value::as_str).ok_or("execution profile lacks governed run root")?;
    let name = topology.file_name().and_then(|value| value.to_str()).ok_or("topology name must be UTF-8")?;
    let target = name.strip_suffix(".topology.json").or_else(|| name.strip_suffix(".json")).unwrap_or(name);
    if !Path::new(root).is_absolute() || target.is_empty() || target.len() > 128
        || !target.as_bytes()[0].is_ascii_alphanumeric()
        || !target.bytes().all(|byte| byte.is_ascii_alphanumeric() || b"_.-".contains(&byte)) {
        return Err("topology requires an explicit governed target ID".into());
    }
    Ok(Path::new(root).join(target))
}

/// Logical PrismOS backend binaries the GUI can drive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrismBin {
    PrismOs,
    PrismCtl,
    PrismHw,
    PrismQnanl,
    PrismSeal,
    PrismAttest,
    PrismRun,
    PrismReplay,
    PrismVal,
    PrismDis,
    PrismUi,
    PrismSec,
    PrismBench,
    PrismC,
    PrismFw,
    PrismKey,
    PrismRedact,
    PrismModeld,
    PrismAcceptance,
    PrismMaterializeSites,
    DstwExportWtPathb,
    PrismCrypticSiteAnalysis,
    PrismScore,
    PrismCompanion,
    PrismPocketAudit,
    PrismPhysics,
    PrismPocketEvo,
    /// The installed Rust/CUDA physics engine. Direct execution remains subject
    /// to the engine's native governance and compatibility gates.
    NhsRtFull,
    /// Deterministic `nhs_rt_full --help` inventory/equality verifier.
    PrismEngineCapabilities,
    /// Governed preflight → engine → postflight launch adapter. This is the
    /// desktop run path, and it forwards the complete engine argv unchanged.
    PrismValidateRun,
    /// `scripts/prism_replicate.py` — the CANONICAL PRODUCTION entry point:
    /// N-replicate consensus (CLAUDE.md "Production = replicate consensus"). It
    /// runs N single engine passes through the wrapper internally, then
    /// consensus-clusters the metastable pockets. A `scripts/` python tool
    /// resolved the same way as `prism-clean.py` / `prism-merge-chains.py`.
    PrismReplicate,
    /// `scripts/prism-mol2smi` — molecule file (.sdf/.mol/.pdb) → SMILES via an
    /// installed chem toolkit (openbabel or rdkit). Degrades honestly (emits a
    /// JSON `error` + install hint) when neither toolkit is present.
    PrismMol2Smi,
    /// `scripts/prism-clean.py` — raw PDB cleaner: strips altconfs (keeps A),
    /// keeps ONE chain, drops ALL HETATM (⇒ apo), residue-type diversity ≥15
    /// hard gate. Positional argv `<raw.pdb> <clean.pdb> [chain]`.
    PrismClean,
    /// `scripts/prism-prep` — the ONLY valid topology prep tool. Positional argv
    /// `<in.pdb> <out.topology.json>` + optional `--mode/--use-amber/--strict/
    /// --hmr/--nma-modes/--nma-amplification/--check-deps`.
    PrismPrep,
    /// `scripts/prism-merge-chains.py` — multichain merge: renumbers every
    /// residue to chain A sequential from 1, emits a back-translation chain map.
    /// argv `<chainA.pdb> <chainB.pdb>… -o <merged.pdb> --chain-map <map.json>`.
    PrismMergeChains,
}

impl PrismBin {
    /// Every backend the adapter layer knows about.
    pub const ALL: [PrismBin; 35] = [
        PrismBin::PrismOs,
        PrismBin::PrismCtl,
        PrismBin::PrismHw,
        PrismBin::PrismQnanl,
        PrismBin::PrismSeal,
        PrismBin::PrismAttest,
        PrismBin::PrismRun,
        PrismBin::PrismReplay,
        PrismBin::PrismVal,
        PrismBin::PrismDis,
        PrismBin::PrismUi,
        PrismBin::PrismSec,
        PrismBin::PrismBench,
        PrismBin::PrismC,
        PrismBin::PrismFw,
        PrismBin::PrismKey,
        PrismBin::PrismRedact,
        PrismBin::PrismModeld,
        PrismBin::PrismAcceptance,
        PrismBin::PrismMaterializeSites,
        PrismBin::DstwExportWtPathb,
        PrismBin::PrismCrypticSiteAnalysis,
        PrismBin::PrismScore,
        PrismBin::PrismCompanion,
        PrismBin::PrismPocketAudit,
        PrismBin::PrismPhysics,
        PrismBin::PrismPocketEvo,
        PrismBin::NhsRtFull,
        PrismBin::PrismEngineCapabilities,
        PrismBin::PrismValidateRun,
        PrismBin::PrismReplicate,
        PrismBin::PrismMol2Smi,
        PrismBin::PrismClean,
        PrismBin::PrismPrep,
        PrismBin::PrismMergeChains,
    ];

    /// On-disk executable name (the command `clap` registers).
    pub fn file_name(self) -> &'static str {
        match self {
            PrismBin::PrismOs => "prismos",
            PrismBin::PrismCtl => "prismctl",
            PrismBin::PrismHw => "prismhw",
            PrismBin::PrismQnanl => "prismqnanl",
            PrismBin::PrismSeal => "prismseal",
            PrismBin::PrismAttest => "prismattest",
            PrismBin::PrismRun => "prismrun",
            PrismBin::PrismReplay => "prismreplay",
            PrismBin::PrismVal => "prismval",
            PrismBin::PrismDis => "prismdis",
            PrismBin::PrismUi => "prismui",
            PrismBin::PrismSec => "prismsec",
            PrismBin::PrismBench => "prismbench",
            PrismBin::PrismC => "prismc",
            PrismBin::PrismFw => "prismfw",
            PrismBin::PrismKey => "prismkey",
            PrismBin::PrismRedact => "prismredact",
            PrismBin::PrismModeld => "prism-modeld",
            PrismBin::PrismAcceptance => "prism-acceptance",
            PrismBin::PrismMaterializeSites => "prism-materialize-sites",
            PrismBin::DstwExportWtPathb => "dstw_export_wt_pathb",
            PrismBin::PrismCrypticSiteAnalysis => "prism-cryptic-site-analysis",
            PrismBin::PrismScore => "prism-score",
            PrismBin::PrismCompanion => "prism-companion",
            PrismBin::PrismPocketAudit => "prism-pocket-audit",
            PrismBin::PrismPhysics => "prism-physics",
            PrismBin::PrismPocketEvo => "prism-pocket-evo",
            PrismBin::NhsRtFull => "nhs_rt_full",
            PrismBin::PrismEngineCapabilities => "prismos-engine-capabilities",
            PrismBin::PrismValidateRun => "prism-validate-and-run.sh",
            PrismBin::PrismReplicate => "prism_replicate.py",
            PrismBin::PrismMol2Smi => "prism-mol2smi",
            PrismBin::PrismClean => "prism-clean.py",
            PrismBin::PrismPrep => "prism-prep",
            PrismBin::PrismMergeChains => "prism-merge-chains.py",
        }
    }

    /// Short human label for menus / observability.
    pub fn label(self) -> &'static str {
        self.file_name()
    }

    fn required_install_path(self) -> PathBuf {
        let root = match self {
            PrismBin::PrismValidateRun
            | PrismBin::PrismReplicate
            | PrismBin::PrismMol2Smi
            | PrismBin::PrismClean
            | PrismBin::PrismPrep
            | PrismBin::PrismMergeChains => "/opt/prismos/source/copilot_executor_cfe5ea89/scripts",
            _ => "/opt/prismos/bin",
        };
        PathBuf::from(root).join(self.file_name())
    }
}

/// Resolves a [`PrismBin`] to a concrete executable path.
///
/// Resolution order, first existing wins:
/// 1. an explicit per-binary override (`PRISMOS_UI_BIN_<UPPER>`),
/// 2. `PRISM_ENGINE_BIN` for [`PrismBin::NhsRtFull`],
/// 3. the appliance-owned executable directories,
/// 4. any explicit deployment search directory,
/// 5. otherwise the required installed path, which fails closed when absent.
#[derive(Debug, Clone)]
pub struct BinaryResolver {
    search_dirs: Vec<PathBuf>,
}

impl BinaryResolver {
    /// Build a resolver using only installed-image and explicit deployment dirs.
    pub fn with_defaults() -> Self {
        let mut search_dirs = vec![
            PathBuf::from("/opt/prismos/bin"),
            PathBuf::from("/usr/local/bin"),
            PathBuf::from("/opt/prismos/source/copilot_executor_cfe5ea89/scripts"),
        ];
        // Explicit per-deployment override directories.
        if let Ok(custom) = std::env::var("PRISMOS_UI_BIN_DIR") {
            for dir in std::env::split_paths(&custom) {
                search_dirs.push(dir);
            }
        }
        Self { search_dirs }
    }

    /// Construct a resolver from an explicit list of directories (testing).
    pub fn from_dirs(search_dirs: Vec<PathBuf>) -> Self {
        Self { search_dirs }
    }

    /// Register an additional directory to search (e.g. a user-chosen build dir).
    pub fn push_dir(&mut self, dir: impl Into<PathBuf>) {
        self.search_dirs.push(dir.into());
    }

    /// Resolve a backend binary to an executable path.
    ///
    /// Returns the required installed path when no candidate exists. That path
    /// is intentionally not resolved through `PATH`, so a missing payload
    /// cannot silently select a host or development-tree executable.
    pub fn resolve(&self, bin: PrismBin) -> PathBuf {
        let name = bin.file_name();
        if let Ok(explicit) = std::env::var(env_override_key(bin)) {
            if !explicit.is_empty() {
                return PathBuf::from(explicit);
            }
        }
        if bin == PrismBin::NhsRtFull {
            if let Ok(explicit) = std::env::var("PRISM_ENGINE_BIN") {
                if !explicit.is_empty() {
                    return PathBuf::from(explicit);
                }
            }
        }
        for dir in &self.search_dirs {
            let candidate = dir.join(name);
            if candidate.is_file() {
                return candidate;
            }
        }
        bin.required_install_path()
    }
}

fn env_override_key(bin: PrismBin) -> String {
    let mut key = String::from("PRISMOS_UI_BIN_");
    for ch in bin.file_name().chars() {
        if ch.is_ascii_alphanumeric() {
            key.push(ch.to_ascii_uppercase());
        } else {
            key.push('_');
        }
    }
    key
}

/// A fully-typed backend invocation: a binary plus an explicit argv tail.
///
/// The argv tail never contains shell syntax. Build one with a typed
/// constructor (e.g. [`PrismCommand::qnanl_encode`]) and run it with
/// [`PrismCommand::to_std_command`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrismCommand {
    pub bin: PrismBin,
    /// argv tail as `OsString` so path arguments are passed to the backend
    /// byte-for-byte (no lossy UTF-8 round-trip); non-path args are plain text.
    pub args: Vec<OsString>,
}

impl PrismCommand {
    /// Generic constructor (escape hatch for the long tail of subcommands).
    pub fn new(bin: PrismBin, args: Vec<OsString>) -> Self {
        Self { bin, args }
    }

    /// The argv that will be executed, *with the binary resolved* — exactly
    /// what the observability log shows the operator. Quoting here is for
    /// display only; execution never goes through a shell.
    pub fn display(&self, resolver: &BinaryResolver) -> String {
        let mut out = display_token(resolver.resolve(self.bin).to_string_lossy().as_ref());
        for arg in &self.args {
            out.push(' ');
            out.push_str(&display_token(&arg.to_string_lossy()));
        }
        out
    }

    /// The argv tail without the binary, for compact display.
    pub fn args_line(&self) -> String {
        self.args
            .iter()
            .map(|a| display_token(&a.to_string_lossy()))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Build a ready-to-spawn [`std::process::Command`].
    pub fn to_std_command(&self, resolver: &BinaryResolver) -> std::process::Command {
        let mut cmd = std::process::Command::new(resolver.resolve(self.bin));
        cmd.args(&self.args);
        cmd.env("PRISMOS_CLIENT_SURFACE", "gui");
        cmd
    }

    // --------------------------------------------------------------------
    // Typed constructors. Argv shapes mirror the clap-derive surface 1:1.
    // --------------------------------------------------------------------

    // ---- prismui : operator/science dashboard text --------------------
    pub fn ui_dashboard(
        input: Option<&Path>,
        status: Option<&Path>,
        report: Option<&Path>,
        open_panel: Option<&str>,
    ) -> Self {
        let mut args = Vec::new();
        if let Some(input) = input {
            args.push(path_arg(input));
        }
        push_opt_path(&mut args, "--status", status);
        push_opt_path(&mut args, "--report", report);
        push_opt_str(&mut args, "--open-panel", open_panel);
        Self::new(PrismBin::PrismUi, args)
    }

    // ---- prismqnanl : QNANL receipt encode / decode / header ----------
    pub fn qnanl_encode(
        input: &Path,
        packet: &Path,
        receipt: &Path,
        family: &str,
        artifact_id: Option<&str>,
    ) -> Self {
        let mut args = vec![
            "encode".into(),
            "--input".into(),
            path_arg(input),
            "--packet".into(),
            path_arg(packet),
            "--receipt".into(),
            path_arg(receipt),
            "--family".into(),
            family.into(),
        ];
        push_opt_str(&mut args, "--artifact-id", artifact_id);
        Self::new(PrismBin::PrismQnanl, args)
    }

    pub fn qnanl_decode(packet: &Path, output: &Path) -> Self {
        Self::new(
            PrismBin::PrismQnanl,
            vec![
                "decode".into(),
                "--packet".into(),
                path_arg(packet),
                "--output".into(),
                path_arg(output),
            ],
        )
    }

    pub fn qnanl_header(packet: &Path) -> Self {
        Self::new(
            PrismBin::PrismQnanl,
            vec!["header".into(), "--packet".into(), path_arg(packet)],
        )
    }

    // ---- prismseal : dev sealing + verification -----------------------
    pub fn seal_seal(input: &Path, out: &Path) -> Self {
        Self::new(
            PrismBin::PrismSeal,
            vec![
                "seal".into(),
                path_arg(input),
                "--out".into(),
                path_arg(out),
            ],
        )
    }

    pub fn seal_verify(input: &Path) -> Self {
        Self::new(PrismBin::PrismSeal, vec!["verify".into(), path_arg(input)])
    }

    // ---- prismhw : hardware inspection --------------------------------
    pub fn hw_inspect(cuda: bool, json: Option<&Path>) -> Self {
        let mut args = vec!["inspect".into()];
        if cuda {
            args.push("--cuda".into());
        }
        push_opt_path(&mut args, "--json", json);
        Self::new(PrismBin::PrismHw, args)
    }

    pub fn hw_cuda_selftest(json: Option<&Path>) -> Self {
        let mut args = vec!["cuda-selftest".into()];
        push_opt_path(&mut args, "--json", json);
        Self::new(PrismBin::PrismHw, args)
    }

    // ---- prismctl : installed OS control / release verify -------------
    pub fn ctl_status(bundle: &Path) -> Self {
        Self::ctl_simple("status", bundle)
    }

    /// `prism-modeld query --model-id prism_champion_pllm_qwen3_8b --smiles <s> --top-k <k>`.
    /// Governed one-shot physics query against the champion Oracle (advisory; ADR-0016).
    pub fn champion_query(smiles: &str, top_k: u32) -> Self {
        Self::new(
            PrismBin::PrismModeld,
            vec![
                OsString::from("query"),
                OsString::from("--model-id"),
                OsString::from("prism_champion_pllm_qwen3_8b"),
                OsString::from("--smiles"),
                OsString::from(smiles),
                OsString::from("--top-k"),
                OsString::from(top_k.to_string()),
            ],
        )
    }

    /// `prism-score --smiles <s>` : Oracle-aligned binding score (Mamba-2 student + reward head).
    pub fn score_smiles(smiles: &str) -> Self {
        Self::new(
            PrismBin::PrismScore,
            vec![OsString::from("--smiles"), OsString::from(smiles)],
        )
    }

    /// `prism-score --smiles-list <csv>` : batch score + rank/percentile table.
    pub fn score_batch(smiles_csv: &str) -> Self {
        Self::new(
            PrismBin::PrismScore,
            vec![OsString::from("--smiles-list"), OsString::from(smiles_csv)],
        )
    }

    /// `prism-score --report --smiles <s>` : full lead-optimization report
    /// (grounded chemistry + Oracle physics + gpt-oss Companion hypotheses).
    pub fn score_report(smiles: &str) -> Self {
        Self::new(
            PrismBin::PrismScore,
            vec![
                OsString::from("--report"),
                OsString::from("--smiles"),
                OsString::from(smiles),
            ],
        )
    }

    /// `prism-companion --message <text>` : conversational GPT-OSS Companion turn.
    /// gpt-oss-120b (CF Workers AI) that tool-calls the local Qwen champion Oracle.
    pub fn companion_chat(message: &str) -> Self {
        Self::new(
            PrismBin::PrismCompanion,
            vec![OsString::from("--message"), OsString::from(message)],
        )
    }

    /// `prism-companion --reset` : clear the Companion's rolling conversation session.
    pub fn companion_reset() -> Self {
        Self::new(PrismBin::PrismCompanion, vec![OsString::from("--reset")])
    }

    /// `prism-companion --message <m> [--attach <file>]` : one Prism CoPilot turn,
    /// optionally with an attached file (PDB, trajectory, …) for it to consume.
    pub fn copilot_chat(message: &str, attach: Option<&Path>) -> Self {
        let mut args = vec![OsString::from("--message"), OsString::from(message)];
        if let Some(p) = attach {
            args.push(OsString::from("--attach"));
            args.push(p.as_os_str().to_owned());
        }
        Self::new(PrismBin::PrismCompanion, args)
    }

    /// `prism-pocket-audit` : coordinate-backed audit of predicted pockets vs target residues.
    /// `prism-physics --smiles <s>` : full PrismAI physics envelope (JSON) from :8711/predict.
    pub fn physics_query(smiles: &str, top_k: u32) -> Self {
        Self::new(
            PrismBin::PrismPhysics,
            vec![
                OsString::from("--smiles"),
                OsString::from(smiles),
                OsString::from("--top-k"),
                OsString::from(top_k.to_string()),
            ],
        )
    }

    /// `prism-pocket-evo --target <t> [--refresh]` : per-POCKET occupancy + hydration
    /// evolution graph as JSON (cache-first; `refresh` forces the full ~90s spike decode).
    pub fn pocket_evolution_graph(target: &str, refresh: bool) -> Self {
        let mut args = vec![OsString::from("--target"), OsString::from(target)];
        if refresh {
            args.push(OsString::from("--refresh"));
        }
        Self::new(PrismBin::PrismPocketEvo, args)
    }

    pub fn pocket_audit(
        structure: &Path,
        pockets: &Path,
        target_residues: &str,
        radii: &str,
        output: &Path,
    ) -> Self {
        Self::new(
            PrismBin::PrismPocketAudit,
            vec![
                OsString::from("--structure"),
                structure.as_os_str().to_owned(),
                OsString::from("--pockets"),
                pockets.as_os_str().to_owned(),
                OsString::from("--target-residues"),
                OsString::from(target_residues),
                OsString::from("--radii"),
                OsString::from(radii),
                OsString::from("--output"),
                output.as_os_str().to_owned(),
            ],
        )
    }

    pub fn ctl_audit(bundle: &Path) -> Self {
        Self::ctl_simple("audit", bundle)
    }

    pub fn ctl_market_ready(bundle: &Path, mode: &str) -> Self {
        Self::new(
            PrismBin::PrismCtl,
            vec![
                "market-ready".into(),
                "--bundle".into(),
                path_arg(bundle),
                "--mode".into(),
                mode.into(),
            ],
        )
    }

    fn ctl_simple(sub: &str, bundle: &Path) -> Self {
        Self::new(
            PrismBin::PrismCtl,
            vec![sub.into(), "--bundle".into(), path_arg(bundle)],
        )
    }

    // ---- prismattest : runtime / artifact attestation -----------------
    pub fn attest_runtime(out: Option<&Path>) -> Self {
        let mut args = vec!["runtime".into()];
        push_opt_path(&mut args, "--out", out);
        Self::new(PrismBin::PrismAttest, args)
    }

    pub fn attest_verify(input: &Path) -> Self {
        Self::new(
            PrismBin::PrismAttest,
            vec!["verify".into(), path_arg(input)],
        )
    }

    // ---- prismval : packet validation ---------------------------------
    pub fn val_validate(input: &Path, level: &str, out: Option<&Path>) -> Self {
        let mut args = vec![
            "validate".into(),
            path_arg(input),
            "--level".into(),
            level.into(),
        ];
        push_opt_path(&mut args, "--out", out);
        Self::new(PrismBin::PrismVal, args)
    }

    // ---- prismdis : packet disassembly --------------------------------
    pub fn dis(input: &Path, out: Option<&Path>) -> Self {
        let mut args = vec![path_arg(input)];
        push_opt_path(&mut args, "--out", out);
        Self::new(PrismBin::PrismDis, args)
    }

    // ---- prismrun : execute a packet ----------------------------------
    pub fn run_packet(input: &Path, backend: &str, out: &Path) -> Self {
        Self::new(
            PrismBin::PrismRun,
            vec![
                "run".into(),
                path_arg(input),
                "--backend".into(),
                backend.into(),
                "--out".into(),
                path_arg(out),
            ],
        )
    }

    // ---- prismreplay : deterministic replay verification --------------
    pub fn replay_verify(input: &Path, out: Option<&Path>) -> Self {
        let mut args = vec!["verify".into(), path_arg(input)];
        push_opt_path(&mut args, "--out", out);
        Self::new(PrismBin::PrismReplay, args)
    }

    // ---- prismbench : packet benchmark --------------------------------
    pub fn bench_run(input: &Path, backend: &str, out: &Path) -> Self {
        Self::new(
            PrismBin::PrismBench,
            vec![
                "run".into(),
                path_arg(input),
                "--backend".into(),
                backend.into(),
                "--out".into(),
                path_arg(out),
            ],
        )
    }

    pub fn bench_suite(backend: &str, corpus: Option<&Path>, out: &Path) -> Self {
        let mut args = vec!["suite".into(), "--backend".into(), backend.into()];
        push_opt_path(&mut args, "--corpus", corpus);
        args.push("--out".into());
        args.push(path_arg(out));
        Self::new(PrismBin::PrismBench, args)
    }

    // ---- prismos : run-root browser / doctor / view -------------------
    pub fn os_doctor(run_dir: Option<&Path>, strict: bool) -> Self {
        let mut args = vec!["doctor".into()];
        push_opt_path(&mut args, "--run-dir", run_dir);
        if strict {
            args.push("--strict".into());
        }
        Self::new(PrismBin::PrismOs, args)
    }

    pub fn os_view(surface: &str, run_dir: Option<&Path>) -> Self {
        let mut args = vec!["view".into(), surface.into(), "--no-server".into()];
        push_opt_path(&mut args, "--run-dir", run_dir);
        Self::new(PrismBin::PrismOs, args)
    }

    pub fn os_manifest(surface: &str, run_dir: Option<&Path>, output: Option<&Path>) -> Self {
        let mut args = vec!["manifest".into(), surface.into()];
        push_opt_path(&mut args, "--run-dir", run_dir);
        push_opt_path(&mut args, "--output", output);
        Self::new(PrismBin::PrismOs, args)
    }

    // ---- prism-materialize-sites : Path-B offline materializer --------
    pub fn materialize_sites(
        manifest: &Path,
        output_dir: Option<&Path>,
        headers_only: bool,
    ) -> Self {
        let mut args = vec!["--manifest".into(), path_arg(manifest)];
        if let Some(dir) = output_dir {
            args.push("--output-dir".into());
            args.push(path_arg(dir));
        }
        if headers_only {
            args.push("--headers-only".into());
        }
        Self::new(PrismBin::PrismMaterializeSites, args)
    }

    // ---- nhs_rt_full capability discovery + governed launch ------------
    /// Ask the installed engine for its live clap surface. `--help` exits
    /// before physics execution and therefore remains safe for inventory use.
    pub fn nhs_rt_full_help() -> Self {
        Self::new(PrismBin::NhsRtFull, vec!["--help".into()])
    }

    /// Emit a deterministic JSON inventory of the installed engine's main and
    /// subcommand clap surfaces. When `expected` is set the inventory command
    /// performs exact option/subcommand equality validation.
    pub fn engine_capabilities(expected: Option<&Path>, pretty: bool) -> Self {
        let mut args = Vec::new();
        if let Some(expected) = expected {
            args.push("--expected".into());
            args.push(path_arg(expected));
        }
        if pretty {
            args.push("--pretty".into());
        }
        Self::new(PrismBin::PrismEngineCapabilities, args)
    }

    /// Complete governed engine argv mode. Topology/output remain typed so the
    /// wrapper can run its evidence gates; every other token is forwarded in
    /// order without a wrapper allowlist. The engine's clap parser and native
    /// runtime compatibility checks remain authoritative.
    pub fn validate_and_run_raw(
        topology: &Path,
        output_dir: &Path,
        chain_map: Option<&Path>,
        engine_argv: &EngineArgv,
    ) -> Self {
        let mut args = vec![
            "-t".into(),
            path_arg(topology),
            "-o".into(),
            path_arg(output_dir),
        ];
        if let Some(chain_map) = chain_map {
            args.push("--chain-map".into());
            args.push(path_arg(chain_map));
        }
        args.extend(engine_argv.tokens().iter().cloned());
        Self::new(PrismBin::PrismValidateRun, args)
    }

    // ---- prism-validate-and-run.sh : typed scientific presets ----------
    /// Build the calibrated canonical production run invocation.
    ///
    /// The argv below is the canonical production shape. Operators can instead
    /// select [`validate_and_run_raw`](Self::validate_and_run_raw) for complete
    /// access to the live engine clap surface without changing this preset:
    ///
    /// The three calibrated-evidence flags — `--md-only-evidence`,
    /// `--path-a-production-profile`, `--path-a-max-wall-seconds` — plus
    /// `--site-ranker phase-manifold` are unconditional defaults for this preset.
    ///
    /// `multi_stream` follows the sizing rule (script header line 17):
    /// <200 residues → 8, 200-400 → 8, >400 → 20. The caller derives it from the
    /// topology residue count; this constructor renders whatever it is handed.
    /// Complete primary1010 execution; the panel has already been production-prepared.
    /// No raw topology or MD-only flags are inferred or appended.
    pub fn primary_1010_cycle(panel: &Path) -> Self {
        Self::new(PrismBin::PrismValidateRun, vec!["run-cycle".into(), "--root".into(), path_arg(panel)])
    }

    pub fn primary_1010_inspect(panel: &Path) -> Self {
        Self::new(PrismBin::PrismValidateRun, vec!["inspect-panel".into(), "--root".into(), path_arg(panel)])
    }

    pub fn primary_1010_result(panel: &Path) -> Self {
        Self::new(PrismBin::PrismValidateRun, vec!["verify-result".into(), "--root".into(), path_arg(panel)])
    }

    pub fn validate_and_run(topology: &Path, output_dir: &Path, multi_stream: u32) -> Self {
        Self::new(
            PrismBin::PrismValidateRun,
            canonical_base_args(topology, output_dir, multi_stream, 180),
        )
    }

    /// Calibrated canonical run with a typed **surface tail** appended.
    ///
    /// The base is emitted FIRST (with `spike_percentile` threaded straight into
    /// the canonical base, so exactly ONE `--spike-percentile` token is emitted —
    /// the widget value no longer duplicates through the surface tail), then an
    /// optional `--chain-map` (set when a multichain merge produced one), then the
    /// six fundamental physics/capture/detection tokens from `params`
    /// ([`push_run_params`]), then each [`SurfaceFlag`]'s tokens in order. The
    /// complete-argv mode covers every additional current or future engine flag.
    ///
    /// With `chain_map = None`, `surface = &[]`, `spike_percentile = 70`, and
    /// `params = &RunParams::default()` this is the locked base (identical to
    /// [`validate_and_run`] when `max_wall = 180`) followed by the six default
    /// run-param tokens. Used by the C-canonical command-class preset.
    #[allow(clippy::too_many_arguments)]
    pub fn validate_and_run_ext(
        topology: &Path,
        output_dir: &Path,
        multi_stream: u32,
        max_wall: u64,
        chain_map: Option<&Path>,
        surface: &[SurfaceFlag],
        spike_percentile: u32,
        params: &RunParams,
    ) -> Self {
        // Standard protocol: drop `--fast` (include_fast = false) AND
        // `--multi-differential` (DiffMode::None) so the engine takes its
        // user-configurable else-branch and honors --temperature/--cryo-temp/--steps
        // directly (nhs_rt_full.rs:5094-5112 / :9052-9080). Fast keeps the calibrated
        // `--fast --multi-differential` base.
        let (diff, include_fast) = if params.standard_protocol {
            (DiffMode::None, false)
        } else {
            (DiffMode::MultiDifferential, true)
        };
        let mut args = canonical_base_args_full(
            topology,
            output_dir,
            multi_stream,
            6,
            spike_percentile,
            "3.0",
            max_wall,
            diff,
            false,
            include_fast,
        );
        if let Some(map) = chain_map {
            args.push("--chain-map".into());
            args.push(path_arg(map));
        }
        push_run_params(&mut args, params);
        for flag in surface {
            args.extend(flag.to_args());
        }
        Self::new(PrismBin::PrismValidateRun, args)
    }

    /// C-twin-graph preset: the calibrated base with `--coupled-twin`
    /// `--graph-coupling` **substituted for** `--multi-differential`.
    ///
    /// This is NOT append-only: the base hard-codes `--multi-differential`, which
    /// the engine dispatches FIRST and which the docs say "Overrides
    /// --coupled-twin" (nhs_rt_full.rs:1014, dispatch :4191-4194). Appending
    /// `--coupled-twin` on top of `--multi-differential` silently no-ops TWIN, so
    /// this constructor drops the multi-differential token entirely.
    ///
    /// The advanced-surface tail (`surface`) IS threaded here: TWIN coupling is
    /// orthogonal to the ghost-lattice backend / its tuning knobs and to
    /// REST2 / multi-temp / stepped-holds, so the operator can run TWIN *with* a
    /// tuned ghost lattice — the full tail layers on with no conflict (no filter
    /// needed, unlike the named-surface arm).
    pub fn validate_and_run_twin(
        topology: &Path,
        output_dir: &Path,
        multi_stream: u32,
        max_wall: u64,
        chain_map: Option<&Path>,
        surface: &[SurfaceFlag],
    ) -> Self {
        let mut args = canonical_base_args_full(
            topology,
            output_dir,
            multi_stream,
            6,
            70,
            "3.0",
            max_wall,
            DiffMode::Twin,
            false,
            true,
        );
        if let Some(map) = chain_map {
            args.push("--chain-map".into());
            args.push(path_arg(map));
        }
        // Advanced-surface tail (ghost lattice backend + tuning knobs, REST2 /
        // multi-temp / stepped-holds). TWIN coupling is ORTHOGONAL to clustering
        // and enhanced sampling, so the full tail layers on with no conflict — no
        // filtering (unlike the named-surface arm, which owns backend selection).
        for flag in surface {
            args.extend(flag.to_args());
        }
        Self::new(PrismBin::PrismValidateRun, args)
    }

    /// Calibrated canonical run driven by a first-class named engine surface.
    ///
    /// Emits the canonical base with NO `--multi-differential` (`DiffMode::None`),
    /// then an optional `--chain-map`, then the SINGLE `--engine-surface <value>`
    /// token. The engine expands the named value UPSTREAM of every guard/dispatch
    /// (`nhs_rt_full.rs:1043` field + `:3666-3699` preset match), so this one
    /// token fully determines the differential / twin / ghost config with NO
    /// conflicting base flag:
    ///   - `captured-graph-twin` needs `coupled_twin`+`graph_coupling` and MUST
    ///     NOT carry `--multi-differential` — which the engine dispatches first
    ///     and which "Overrides --coupled-twin" (`:1026`, `:3681-3683`), silently
    ///     no-oping TWIN. This is the SAME reason [`validate_and_run_twin`] drops
    ///     it. So the base here uses `DiffMode::None`.
    ///   - `interferometric-twin` / `twin16` re-set `multi_differential`
    ///     themselves (`:3681-3690`), so a `None` base is equivalent.
    ///   - `ghost-tiles` sets the ghost backend + its two prereqs itself
    ///     (`:3691-3695`).
    ///
    /// The UI never hand-assembles the backend-selection tokens a named surface
    /// OWNS (`--coupled-twin`, `--clustering-backend`, `--m1-monolithic-discovery`,
    /// `--mar-v2-telemetry`) — the engine's expansion emits those, and `surface_tail`
    /// is filtered by [`SurfaceFlag::owned_by_named_surface`] so they are never
    /// duplicated. It DOES layer the additive advanced-surface tail — the
    /// `--ghost-phase-lattice-*` tuning knobs and orthogonal REST2 / multi-temp /
    /// stepped-holds — so a `ghost-tiles` run can be fine-tuned. Every
    /// Calibrated evidence flags + `--site-ranker phase-manifold` remain preset
    /// defaults. Complete argv mode can select a different valid engine profile.
    ///
    /// `surface = EngineSurface::None` renders the literal `none` value, which is
    /// the engine's no-op arm (`:3696-3698`); callers should instead use the
    /// preset constructors for the un-named path.
    #[allow(clippy::too_many_arguments)]
    pub fn validate_and_run_engine_surface(
        topology: &Path,
        output_dir: &Path,
        multi_stream: u32,
        max_wall: u64,
        chain_map: Option<&Path>,
        surface: EngineSurface,
        params: &RunParams,
        surface_tail: &[SurfaceFlag],
    ) -> Self {
        let mut args = canonical_base_args_full(
            topology,
            output_dir,
            multi_stream,
            6,
            70,
            "3.0",
            max_wall,
            DiffMode::None,
            false,
            true,
        );
        if let Some(map) = chain_map {
            args.push("--chain-map".into());
            args.push(path_arg(map));
        }
        // The six fundamental physics/capture/detection tokens are NOT surface
        // tokens — they apply to every run path, a named surface never overrides
        // them. They precede the single --engine-surface token.
        push_run_params(&mut args, params);
        // Advanced-surface tail, layered on top of the named surface. A named
        // surface OWNS backend selection (`--clustering-backend`,
        // `--m1-monolithic-discovery`, `--mar-v2-telemetry`) via the engine's own
        // upstream expansion, so those are filtered out here to avoid
        // duplicate/conflicting tokens (`owned_by_named_surface`). The ADDITIVE
        // ghost tuning knobs (`--ghost-phase-lattice-*`) and orthogonal
        // enhanced-sampling flags survive, so a `ghost-tiles` run can be fine-tuned.
        for flag in surface_tail {
            if !flag.owned_by_named_surface() {
                args.extend(flag.to_args());
            }
        }
        args.extend(SurfaceFlag::EngineSurface(surface).to_args());
        Self::new(PrismBin::PrismValidateRun, args)
    }

    /// C-smoke-min preset: bounded init check, `--fast-25k --multi-stream 1`, the
    /// full calibrated evidence set, and no `--multi-differential` (multi-diff
    /// floors streams at `.max(8)`, nhs_rt_full.rs:4192, defeating a 1-stream
    /// smoke). NOT a production run.
    pub fn smoke_min(topology: &Path, output_dir: &Path) -> Self {
        Self::new(
            PrismBin::PrismValidateRun,
            canonical_base_args_full(
                topology,
                output_dir,
                1,
                6,
                70,
                "3.0",
                180,
                DiffMode::None,
                true,
                true,
            ),
        )
    }

    /// C-smoke-eng preset: multi-stream emission exercise, `--fast-25k
    /// --multi-stream 2`, calibrated evidence set, no `--multi-differential`.
    pub fn smoke_eng(topology: &Path, output_dir: &Path) -> Self {
        Self::new(
            PrismBin::PrismValidateRun,
            canonical_base_args_full(
                topology,
                output_dir,
                2,
                6,
                70,
                "3.0",
                180,
                DiffMode::None,
                true,
                true,
            ),
        )
    }

    // ---- prism_replicate.py : N-replicate consensus (PRODUCTION entry) ---
    /// CLAUDE.md canonical production entry point — "Production = replicate
    /// consensus". Runs N single engine passes (through the wrapper, internally)
    /// then consensus-clusters the metastable pockets. Argv mirrors the CLAUDE.md
    /// canonical block: `--topology <t> --target-name <name> --n-replicates <N>
    /// --output-dir <o>` (scripts/prism_replicate.py argparse:123-127). The engine
    /// flag-set is selected by the replicate driver, so no per-run engine flags
    /// are threaded through this convenience constructor.
    pub fn replicate_consensus(
        topology: &Path,
        target_name: &str,
        n_replicates: usize,
        output_dir: &Path,
    ) -> Self {
        Self::new(
            PrismBin::PrismReplicate,
            vec![
                "--topology".into(),
                path_arg(topology),
                "--target-name".into(),
                target_name.into(),
                "--n-replicates".into(),
                n_replicates.to_string().into(),
                "--output-dir".into(),
                path_arg(output_dir),
            ],
        )
    }

    // ---- prism-clean.py : raw PDB → cleaned single-chain PDB (apo) ------
    /// `prism-clean.py <raw.pdb> <clean.pdb> <chain>` — positional argv
    /// (scripts/prism-clean.py:80-88). The chain is ALWAYS passed explicitly
    /// (script default is "A" at :87). Strips altconfs + ALL HETATM (:26 ATOM
    /// filter ⇒ apo), residue-type diversity ≥15 is a hard `exit(1)` (:70-73).
    pub fn clean_pdb(raw: &Path, clean: &Path, chain: &str) -> Self {
        Self::new(
            PrismBin::PrismClean,
            vec![path_arg(raw), path_arg(clean), chain.into()],
        )
    }

    // ---- prism-prep : cleaned PDB → topology.json (the ONLY prep tool) --
    /// `prism-prep <in.pdb> <out.topology.json> [--mode M] [--use-amber]
    /// [--strict] [--hmr] [--nma-modes N] [--nma-amplification f]`
    /// (argparse scripts/prism-prep:899-931). Uses POSITIONAL in/out — `-o` is
    /// `--output-dir` for `--batch`, NOT the topology output (:905). Only set
    /// optionals are pushed.
    #[allow(clippy::too_many_arguments)]
    pub fn prep_topology(
        clean_pdb: &Path,
        topo_out: &Path,
        mode: Option<&str>,
        use_amber: bool,
        strict: bool,
        hmr: bool,
        nma_modes: Option<u32>,
        nma_amp: Option<f32>,
    ) -> Self {
        let mut args = vec![path_arg(clean_pdb), path_arg(topo_out)];
        if let Some(m) = mode {
            args.push("--mode".into());
            args.push(m.into());
        }
        if use_amber {
            args.push("--use-amber".into());
        }
        if strict {
            args.push("--strict".into());
        }
        if hmr {
            args.push("--hmr".into());
        }
        if let Some(n) = nma_modes {
            args.push("--nma-modes".into());
            args.push(n.to_string().into());
        }
        if let Some(a) = nma_amp {
            args.push("--nma-amplification".into());
            args.push(a.to_string().into());
        }
        Self::new(PrismBin::PrismPrep, args)
    }

    /// `prism-prep --check-deps` — probe OpenMM/PDBFixer availability and exit
    /// (scripts/prism-prep:931). Non-destructive dependency check.
    pub fn prep_check_deps() -> Self {
        Self::new(PrismBin::PrismPrep, vec!["--check-deps".into()])
    }

    // ---- prism-merge-chains.py : per-chain cleaned PDBs → merged PDB ----
    /// `prism-merge-chains.py <chainA.pdb> <chainB.pdb>… -o <merged.pdb>
    /// --chain-map <map.json>` (argparse scripts/prism-merge-chains.py:47-49).
    /// Renumbers every residue to chain A sequential from 1 (:114-123); the
    /// chain map is the ONLY back-translation to author residue numbers.
    pub fn merge_chains(chain_pdbs: &[PathBuf], merged: &Path, chain_map: &Path) -> Self {
        let mut args: Vec<OsString> = chain_pdbs.iter().map(|p| path_arg(p)).collect();
        args.push("-o".into());
        args.push(path_arg(merged));
        args.push("--chain-map".into());
        args.push(path_arg(chain_map));
        Self::new(PrismBin::PrismMergeChains, args)
    }

    // ---- prism-mol2smi : molecule file → SMILES (honest degradation) ----
    /// `prism-mol2smi <file>` : derive a SMILES string from a `.sdf`/`.mol`/`.pdb`
    /// molecule file using an installed chem toolkit. Emits JSON with either
    /// `{"smiles":...,"method":...}` or `{"error":...,"install":...}` — it never
    /// fabricates a parse. (`.smi` files are read in pure Rust by the panel and
    /// never routed here.)
    pub fn mol_to_smiles(path: &Path) -> Self {
        Self::new(PrismBin::PrismMol2Smi, vec![path_arg(path)])
    }

    // ---- prism-cryptic-site-analysis : cryptic-vs-orthosteric ranking --
    pub fn cryptic_site_analysis(run_root: &Path, out: &Path) -> Self {
        Self::new(
            PrismBin::PrismCrypticSiteAnalysis,
            vec![
                "--run-root".into(),
                path_arg(run_root),
                "--out".into(),
                path_arg(out),
            ],
        )
    }
}

// ------------------------------------------------------------------------
// Complete engine argv + typed preset layer.
// ------------------------------------------------------------------------

/// Exact, shell-free argv tokens for the complete engine surface.
///
/// The desktop editor uses one non-empty line per token. This avoids shell
/// interpolation while preserving spaces inside a path or value. `-t`/`--topology`
/// and `-o`/`--output` are reserved because those two values are selected by typed
/// file widgets and are required by the governed preflight/postflight adapter.
/// Every other token is admitted; native clap parsing and runtime compatibility
/// checks decide whether the requested combination is valid.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EngineArgv {
    tokens: Vec<OsString>,
}

impl EngineArgv {
    /// Parse one exact argv token per non-empty line.
    pub fn from_token_lines(text: &str) -> Result<Self, String> {
        let mut tokens = Vec::new();
        for (index, raw_line) in text.lines().enumerate() {
            let token = raw_line.strip_suffix('\r').unwrap_or(raw_line);
            if token.is_empty() {
                continue;
            }
            if token.contains('\0') {
                return Err(format!("line {} contains a NUL byte", index + 1));
            }
            if is_reserved_engine_transport_token(token) {
                return Err(format!(
                    "line {} repeats typed topology/output transport token {token:?}",
                    index + 1
                ));
            }
            tokens.push(OsString::from(token));
        }
        Ok(Self { tokens })
    }

    pub fn tokens(&self) -> &[OsString] {
        &self.tokens
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

fn is_reserved_engine_transport_token(token: &str) -> bool {
    matches!(token, "-t" | "--topology" | "-o" | "--output")
        || (token.starts_with("-t") && !token.starts_with("--"))
        || (token.starts_with("-o") && !token.starts_with("--"))
        || token.starts_with("--topology=")
        || token.starts_with("--output=")
}

// The calibrated typed argv is emitted by `canonical_base_args_full`. It remains
// a convenience profile; it is not an allowlist and does not constrain
// [`EngineArgv`].

/// Which "differential / observation" tokens the canonical base carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DiffMode {
    /// `--multi-differential` (C-canonical; nhs_rt_full.rs:1016).
    MultiDifferential,
    /// `--coupled-twin --graph-coupling` (C-twin-graph; :986,:1002).
    Twin,
    /// Neither (smoke presets — multi-diff would floor streams at `.max(8)`).
    None,
}

impl DiffMode {
    fn tokens(self) -> Vec<OsString> {
        match self {
            DiffMode::MultiDifferential => vec!["--multi-differential".into()],
            DiffMode::Twin => vec!["--coupled-twin".into(), "--graph-coupling".into()],
            DiffMode::None => vec![],
        }
    }
}

/// The calibrated canonical base argv (pin-stable 4-arg surface). Byte-for
/// byte identical to the original `validate_and_run` body when `max_wall = 180`.
pub(crate) fn canonical_base_args(
    topology: &Path,
    output_dir: &Path,
    multi_stream: u32,
    max_wall: u64,
) -> Vec<OsString> {
    canonical_base_args_full(
        topology,
        output_dir,
        multi_stream,
        6,
        70,
        "3.0",
        max_wall,
        DiffMode::MultiDifferential,
        false,
        true,
    )
}

/// Fully-parametrized canonical base. `fused_steps`, `spike_percentile`,
/// `nma_amplification`, `multi_stream`, and `max_wall` are the only values that
/// vary across presets; the evidence flags and `--site-ranker phase-manifold`
/// are unconditional defaults of this preset.
#[allow(clippy::too_many_arguments)]
fn canonical_base_args_full(
    topology: &Path,
    output_dir: &Path,
    multi_stream: u32,
    fused_steps: u32,
    spike_percentile: u32,
    nma_amplification: &str,
    max_wall: u64,
    diff: DiffMode,
    fast_25k: bool,
    include_fast: bool,
) -> Vec<OsString> {
    let mut args: Vec<OsString> = vec![
        "-t".into(),
        path_arg(topology),
        "-o".into(),
        path_arg(output_dir),
    ];
    // The Standard protocol drops `--fast` so the engine runs its user-configurable
    // else-branch (nhs_rt_full.rs:5094-5112 / :9052-9068) and honors
    // --temperature/--cryo-temp/--steps 1:1. Every other path keeps the calibrated
    // `--fast` (the fast/twin-differential protocol fixes the thermal cycle + budget).
    if include_fast {
        args.push("--fast".into());
    }
    if fast_25k {
        args.push("--fast-25k".into());
    }
    args.push("--hysteresis".into());
    args.push("--prism-therm".into());
    args.push("--multi-stream".into());
    args.push(multi_stream.to_string().into());
    args.push("--spike-percentile".into());
    args.push(spike_percentile.to_string().into());
    args.push("--fused-steps".into());
    args.push(fused_steps.to_string().into());
    args.push("--hmr".into());
    args.push("--adaptive-dt".into());
    args.extend(diff.tokens());
    args.push("--closed-loop-steering".into());
    args.push("--asymmetric-steering".into());
    args.push("--site-ranker".into());
    args.push("phase-manifold".into());
    args.push("--md-only-evidence".into());
    args.push("--path-a-production-profile".into());
    args.push("--path-a-max-wall-seconds".into());
    args.push(max_wall.to_string().into());
    args.push("--uv-wavelengths".into());
    args.push("280,274,258,254,211".into());
    args.push("--nma-amplification".into());
    args.push(nma_amplification.into());
    args.push("--nma-scan-fraction".into());
    args.push("0.3".into());
    args.push("--replica-seed".into());
    args.push("42".into());
    args.push("-v".into());
    args
}

/// Tunable run parameters exposed by the Run-Engine panel's parameter tabs. They
/// apply to BOTH the Canonical (ext) and named-engine-surface run paths (a named
/// surface never overrides them).
///
/// PROTOCOL-AWARE SET (2026-07-09): which tokens are emitted depends on
/// `standard_protocol`.
///   - `standard_protocol = false` (Fast, default): the engine runs the calibrated
///     `--fast` cryo-UV protocol, which FIXES the thermal cycle (50 K → 310 K) and
///     the step budget. On this path `--temperature`/`--cryo-temp` are NOT emitted
///     (the fast/twin-differential protocol does not read them), and `--steps` is
///     honored only when paired with `--wall-bound-hysteresis` — the 5-stage
///     protocol then allocates it (`steps_per_stream`, nhs_rt_full.rs:9080). So a
///     non-default `steps` emits `--steps <n> --wall-bound-hysteresis`; the default
///     budget emits neither, so the default Fast run is byte-identical to the locked
///     canonical base.
///   - `standard_protocol = true` (Standard): the base drops `--fast` and
///     `--multi-differential`, so the engine takes its USER-CONFIGURABLE else-branch
///     (nhs_rt_full.rs:5094-5112 `start_temp: args.cryo_temp` / `end_temp:
///     args.temperature`, and :9080 `steps_per_stream = args.steps`). On this path
///     `--temperature <t>`, `--cryo-temp <c>` and `--steps <n>` are all emitted and
///     honored 1:1 — NO `--wall-bound-hysteresis` is needed.
///
/// `--cluster-threshold` is NEVER emitted on either path: it is not consumed by the
/// engine (`nhs_rt_full.rs:434` decl + `:3051` command-echo only), and
/// `--md-only-evidence` replaces in-engine CCL clustering with phase-manifold ranking.
///
/// Always-honored knobs (1:1 with a VERIFIED clap flag, both protocols):
///   `save_trajectory_interval` → `--save-trajectory-interval` (`:480`, 0 = off) —
///       genuinely gates frame writing on the multi-stream path.
///   `lining_cutoff` → `--lining-cutoff` (`:444`, default 8.0) — genuinely drives
///       the post-MD site-lining / residue-assignment geometry.
#[derive(Debug, Clone, Copy)]
pub struct RunParams {
    /// Standard (user-configurable thermodynamics) vs Fast (calibrated) protocol.
    pub standard_protocol: bool,
    /// `--steps` total MD steps (`nhs_rt_full.rs:399`, engine default 500000).
    /// Honored directly in Standard mode; in Fast mode only when a non-default
    /// value pairs it with `--wall-bound-hysteresis`.
    pub steps: i32,
    /// `--temperature` warm / end-of-ramp target K (`:404`, engine default 300.0).
    /// Emitted ONLY in Standard mode.
    pub temperature: f32,
    /// `--cryo-temp` cryo start K (`:408`, engine default 50.0). Emitted ONLY in
    /// Standard mode.
    pub cryo_temp: f32,
    pub save_trajectory_interval: u32,
    pub lining_cutoff: f32,
}

impl Default for RunParams {
    fn default() -> Self {
        Self {
            standard_protocol: false,
            steps: 500_000,
            temperature: 300.0,
            cryo_temp: 50.0,
            save_trajectory_interval: 0,
            lining_cutoff: 8.0,
        }
    }
}

/// Append the protocol-aware run-parameter tokens for `p` to `args`.
///
/// - Standard protocol (`p.standard_protocol`): emit `--temperature <t>
///   --cryo-temp <c> --steps <n>`; the engine's user-configurable else-branch honors
///   all three 1:1 (nhs_rt_full.rs:5094-5112 / :9080). NO `--wall-bound-hysteresis`
///   (the standard path reads `args.steps` directly).
/// - Fast protocol: `--steps` alone is a no-op, so a NON-default budget is emitted
///   ONLY paired with `--wall-bound-hysteresis` (which makes the 5-stage protocol
///   allocate it, :9080; it also skips the base `--hysteresis`, :9070, so the pairing
///   cannot conflict). A default (500000) budget emits NEITHER token, so the default
///   Fast run stays byte-identical to the locked canonical base. `--temperature` /
///   `--cryo-temp` are never emitted on this path.
///
/// `--cluster-threshold` is never emitted on either path (see [`RunParams`]).
fn push_run_params(args: &mut Vec<OsString>, p: &RunParams) {
    if p.standard_protocol {
        args.push("--temperature".into());
        args.push(p.temperature.to_string().into());
        args.push("--cryo-temp".into());
        args.push(p.cryo_temp.to_string().into());
        args.push("--steps".into());
        args.push(p.steps.to_string().into());
    } else if p.steps != 500_000 {
        args.push("--steps".into());
        args.push(p.steps.to_string().into());
        args.push("--wall-bound-hysteresis".into());
    }
    args.push("--save-trajectory-interval".into());
    args.push(p.save_trajectory_interval.to_string().into());
    args.push("--lining-cutoff".into());
    args.push(p.lining_cutoff.to_string().into());
}

/// A first-class NAMED engine surface, 1:1 with the `--engine-surface` clap
/// `value_parser` set in `nhs_rt_full.rs:1041` (`["none", "captured-graph-twin",
/// "interferometric-twin", "twin16", "ghost-tiles"]`).
///
/// The engine EXPANDS each named value into the SAME real underlying fields a
/// knowledgeable operator sets by hand — UPSTREAM of every guard/dispatch
/// (`nhs_rt_full.rs:3666-3699`). The UI therefore emits ONLY the single
/// `--engine-surface <value>` token and lets the engine expand it; it never
/// hand-assembles the low-level tokens. The set is closed and typed, so only the
/// engine's own `value_parser` strings can ever be emitted (an invalid value is
/// unrepresentable — clap would otherwise reject the run).
///
/// HONEST MAPPING (grounded at `nhs_rt_full.rs:1032-1034`): there is deliberately
/// NO value for "always-on sensor" (the spike-stream writer is unconditional
/// baseline infra — nothing to toggle) nor a bare interferometric/sensor toggle
/// (no distinct code path). Those are surfaced in the panel as "not a flag", not
/// fabricated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineSurface {
    /// `none` — no named surface; the command-class preset drives the run
    /// (engine no-op arm, `nhs_rt_full.rs:3696-3698`).
    None,
    /// `captured-graph-twin` → engine sets `coupled_twin` + `graph_coupling`
    /// (`nhs_rt_full.rs:3677-3680`); dispatch takes the coupled-twin branch.
    CapturedGraphTwin,
    /// `interferometric-twin` → engine sets `multi_differential`
    /// (`nhs_rt_full.rs:3681-3683`).
    InterferometricTwin,
    /// `twin16` → engine sets `multi_differential` + bumps `multi_stream` to ≥16
    /// (only bumps; a larger even operator value is respected;
    /// `nhs_rt_full.rs:3684-3690`).
    Twin16,
    /// `ghost-tiles` → engine sets `clustering_backend=ghost-phase-lattice-4d`
    /// + `mar_v2_telemetry` + `m1_monolithic_discovery`
    /// (`nhs_rt_full.rs:3691-3695`).
    GhostTiles,
}

impl EngineSurface {
    /// The exact clap value string (`nhs_rt_full.rs:1041` value_parser). Emitted
    /// verbatim as the argument to `--engine-surface`.
    pub fn as_str(self) -> &'static str {
        match self {
            EngineSurface::None => "none",
            EngineSurface::CapturedGraphTwin => "captured-graph-twin",
            EngineSurface::InterferometricTwin => "interferometric-twin",
            EngineSurface::Twin16 => "twin16",
            EngineSurface::GhostTiles => "ghost-tiles",
        }
    }
}

/// Typed convenience toggles for common engine surfaces. Each variant maps 1:1
/// to a clap field in `crates/prism-nhs/src/bin/nhs_rt_full.rs`; [`EngineArgv`]
/// supplies complete access beyond this convenience set.
#[derive(Debug, Clone, PartialEq)]
pub enum SurfaceFlag {
    /// `--coupled-twin` (nhs_rt_full.rs:986). Requires even `--multi-stream`.
    CoupledTwin,
    /// `--graph-coupling` (:1002).
    GraphCoupling,
    /// `--rest2` (:1150).
    Rest2,
    /// `--multi-temp` (:1161).
    MultiTemp,
    /// `--stepped-holds` (:1170).
    SteppedHolds,
    /// `--m1-monolithic-discovery` (:529) — ghost-lattice-4d prereqisite.
    M1MonolithicDiscovery,
    /// `--mar-v2-telemetry` (:1241) — ghost-lattice-4d prerequisite.
    MarV2Telemetry,
    /// `--clustering-backend <b>` (:1039). Engine-native compatibility checks
    /// reject invalid combinations (for example grid + multi-differential).
    ClusteringBackend(String),
    /// `--ghost-phase-lattice-so3-threshold <f>` (:1049).
    GhostSo3(f32),
    /// `--ghost-phase-lattice-cell-a <f>` (:1055).
    GhostCellA(f32),
    /// `--ghost-phase-lattice-max-temporal-edge-steps <n>` (:1061).
    GhostMaxTemporalEdge(u64),
    /// `--ghost-phase-lattice-step-bucket-size <n>` (:1066).
    GhostStepBucket(u64),
    /// `--captured-graph-body-unroll 1` (:1093), the calibrated typed-preset value.
    CapturedGraphBodyUnroll,
    /// `--spike-percentile <n>` (:695). Override of the base default (70).
    SpikePercentile(u32),
    /// `--nma-perturb <path>` (:966).
    NmaPerturb(PathBuf),
    /// `--nma-amplification <f>` (:971). Override of the base default (3.0).
    NmaAmplification(f32),
    /// `--engine-surface <named>` (:1043). A first-class NAMED engine surface
    /// the engine EXPANDS upstream of every guard/dispatch (:3666-3699). The
    /// value is a closed [`EngineSurface`], never a free string, so only the
    /// engine's own value_parser set can be emitted.
    EngineSurface(EngineSurface),
}

impl SurfaceFlag {
    /// Constructor for a clustering backend. It deliberately does not duplicate
    /// the engine's compatibility matrix; invalid combinations fail at the
    /// native engine gate with the engine's own diagnostic.
    pub fn clustering_backend(backend: &str) -> Option<Self> {
        Some(SurfaceFlag::ClusteringBackend(backend.to_string()))
    }

    /// True for flags a first-class NAMED `--engine-surface` expansion emits
    /// ITSELF (backend selection + its two prerequisites), so they must NOT be
    /// hand-assembled alongside a named surface token — the engine would then
    /// receive duplicate/conflicting flags (`ghost-tiles` already expands to
    /// `--clustering-backend ghost-phase-lattice-4d --m1-monolithic-discovery
    /// --mar-v2-telemetry`, nhs_rt_full.rs:3691-3695). The additive tuning knobs
    /// (`--ghost-phase-lattice-*`) and the orthogonal enhanced-sampling flags
    /// (`--rest2` / `--multi-temp` / `--stepped-holds`) are NOT owned by any
    /// surface and layer on top of a named surface safely.
    pub fn owned_by_named_surface(&self) -> bool {
        matches!(
            self,
            SurfaceFlag::M1MonolithicDiscovery
                | SurfaceFlag::MarV2Telemetry
                | SurfaceFlag::ClusteringBackend(_)
        )
    }

    /// The argv tokens this surface flag contributes.
    pub fn to_args(&self) -> Vec<OsString> {
        match self {
            SurfaceFlag::CoupledTwin => vec!["--coupled-twin".into()],
            SurfaceFlag::GraphCoupling => vec!["--graph-coupling".into()],
            SurfaceFlag::Rest2 => vec!["--rest2".into()],
            SurfaceFlag::MultiTemp => vec!["--multi-temp".into()],
            SurfaceFlag::SteppedHolds => vec!["--stepped-holds".into()],
            SurfaceFlag::M1MonolithicDiscovery => vec!["--m1-monolithic-discovery".into()],
            SurfaceFlag::MarV2Telemetry => vec!["--mar-v2-telemetry".into()],
            SurfaceFlag::ClusteringBackend(b) => {
                vec!["--clustering-backend".into(), b.clone().into()]
            }
            SurfaceFlag::GhostSo3(v) => {
                vec![
                    "--ghost-phase-lattice-so3-threshold".into(),
                    v.to_string().into(),
                ]
            }
            SurfaceFlag::GhostCellA(v) => {
                vec!["--ghost-phase-lattice-cell-a".into(), v.to_string().into()]
            }
            SurfaceFlag::GhostMaxTemporalEdge(v) => vec![
                "--ghost-phase-lattice-max-temporal-edge-steps".into(),
                v.to_string().into(),
            ],
            SurfaceFlag::GhostStepBucket(v) => {
                vec![
                    "--ghost-phase-lattice-step-bucket-size".into(),
                    v.to_string().into(),
                ]
            }
            SurfaceFlag::CapturedGraphBodyUnroll => {
                vec!["--captured-graph-body-unroll".into(), "1".into()]
            }
            SurfaceFlag::SpikePercentile(v) => {
                vec!["--spike-percentile".into(), v.to_string().into()]
            }
            SurfaceFlag::NmaPerturb(p) => vec!["--nma-perturb".into(), path_arg(p)],
            SurfaceFlag::NmaAmplification(v) => {
                vec!["--nma-amplification".into(), v.to_string().into()]
            }
            SurfaceFlag::EngineSurface(s) => {
                vec!["--engine-surface".into(), s.as_str().into()]
            }
        }
    }
}

// ------------------------------------------------------------------------
// Argument helpers. Paths are stored as OsString via `path_arg` so the exact
// on-disk bytes reach the backend (Linux filenames are arbitrary byte
// sequences); `to_std_command` passes them to `Command::args` verbatim — no
// shell, no lossy UTF-8 round-trip. Only the human-facing `display`/`args_line`
// renderings apply a lossy conversion, which is acceptable for display.
// ------------------------------------------------------------------------

fn path_arg(path: &Path) -> OsString {
    path.as_os_str().to_os_string()
}

fn push_opt_path(args: &mut Vec<OsString>, flag: &str, value: Option<&Path>) {
    if let Some(value) = value {
        args.push(flag.into());
        args.push(path_arg(value));
    }
}

fn push_opt_str(args: &mut Vec<OsString>, flag: &str, value: Option<&str>) {
    if let Some(value) = value {
        args.push(flag.into());
        args.push(value.into());
    }
}

/// Quote a token for *display* only when it contains characters that would be
/// ambiguous in a copy-pasted command line. Execution never uses this.
fn display_token(token: &str) -> String {
    if token.is_empty() {
        return "''".to_owned();
    }
    let needs_quote = token.chars().any(|c| {
        c.is_whitespace()
            || matches!(
                c,
                '"' | '\'' | '$' | '`' | '\\' | '|' | '&' | ';' | '<' | '>' | '(' | ')'
            )
    });
    if needs_quote {
        format!("'{}'", token.replace('\'', "'\\''"))
    } else {
        token.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `Vec<OsString>` from string literals for argv assertions.
    fn os(parts: &[&str]) -> Vec<OsString> {
        parts.iter().map(OsString::from).collect()
    }

    #[test]
    fn all_binaries_have_unique_names() {
        let mut names: Vec<&str> = PrismBin::ALL.iter().map(|b| b.file_name()).collect();
        names.sort_unstable();
        let count = names.len();
        names.dedup();
        assert_eq!(count, names.len(), "duplicate backend file names");
        assert_eq!(count, 35);
    }

    #[test]
    fn path_b_exporter_uses_established_underscore_identity() {
        assert_eq!(
            PrismBin::DstwExportWtPathb.file_name(),
            "dstw_export_wt_pathb"
        );
        assert_eq!(
            BinaryResolver::from_dirs(Vec::new()).resolve(PrismBin::DstwExportWtPathb),
            PathBuf::from("/opt/prismos/bin/dstw_export_wt_pathb")
        );
    }

    #[test]
    fn validate_and_run_builds_exact_canonical_argv() {
        let cmd = PrismCommand::validate_and_run(
            Path::new("/data/1btl.topology.json"),
            Path::new("/runs/1btl"),
            8,
        );
        assert_eq!(cmd.bin, PrismBin::PrismValidateRun);
        assert_eq!(
            cmd.args,
            os(&[
                "-t",
                "/data/1btl.topology.json",
                "-o",
                "/runs/1btl",
                "--fast",
                "--hysteresis",
                "--prism-therm",
                "--multi-stream",
                "8",
                "--spike-percentile",
                "70",
                "--fused-steps",
                "6",
                "--hmr",
                "--adaptive-dt",
                "--multi-differential",
                "--closed-loop-steering",
                "--asymmetric-steering",
                "--site-ranker",
                "phase-manifold",
                "--md-only-evidence",
                "--path-a-production-profile",
                "--path-a-max-wall-seconds",
                "180",
                "--uv-wavelengths",
                "280,274,258,254,211",
                "--nma-amplification",
                "3.0",
                "--nma-scan-fraction",
                "0.3",
                "--replica-seed",
                "42",
                "-v",
            ])
        );
        // The calibrated evidence flags + phase-manifold preset must be present.
        let has = |flag: &str| cmd.args.iter().any(|a| a == flag);
        assert!(has("--md-only-evidence"));
        assert!(has("--path-a-production-profile"));
        assert!(has("--path-a-max-wall-seconds"));
        assert!(has("--site-ranker"));
        assert!(has("phase-manifold"));
    }

    #[test]
    fn validate_and_run_honors_multi_stream_sizing() {
        let big = PrismCommand::validate_and_run(
            Path::new("/data/large.topology.json"),
            Path::new("/runs/large"),
            20,
        );
        // --multi-stream value is whatever the caller derived from residue count.
        let idx = big
            .args
            .iter()
            .position(|a| a == "--multi-stream")
            .expect("--multi-stream present");
        assert_eq!(big.args[idx + 1].to_str(), Some("20"));
    }

    #[test]
    fn complete_engine_argv_forwards_every_non_transport_token_exactly() {
        let raw = EngineArgv::from_token_lines(
            "--boltzmann-rank\n--cascade\n--replicas\n40\n--site-ranker\nxgb-v3\n\
             --clustering-backend\ngrid\n--persistent-coupling\n--captured-graph-body-unroll\n8\n\
             /data/value with spaces.json\n",
        )
        .unwrap();
        let cmd = PrismCommand::validate_and_run_raw(
            Path::new("/data/a.topology.json"),
            Path::new("/runs/a"),
            None,
            &raw,
        );
        assert_eq!(cmd.bin, PrismBin::PrismValidateRun);
        assert_eq!(
            &cmd.args[4..],
            os(&[
                "--boltzmann-rank",
                "--cascade",
                "--replicas",
                "40",
                "--site-ranker",
                "xgb-v3",
                "--clustering-backend",
                "grid",
                "--persistent-coupling",
                "--captured-graph-body-unroll",
                "8",
                "/data/value with spaces.json",
            ])
            .as_slice()
        );
    }

    #[test]
    fn complete_engine_argv_reserves_typed_topology_and_output_only() {
        for text in [
            "--topology\n/other.json",
            "--topology=/other.json",
            "-t\n/other.json",
            "-t/other.json",
            "--output\n/other",
            "--output=/other",
            "-o\n/other",
            "-o/other",
        ] {
            assert!(EngineArgv::from_token_lines(text).is_err(), "{text}");
        }
        assert!(EngineArgv::from_token_lines("--topology-b\n/b.json").is_ok());
    }

    #[test]
    fn installed_engine_and_capability_commands_are_first_class_backends() {
        assert_eq!(
            PrismBin::NhsRtFull.required_install_path(),
            PathBuf::from("/opt/prismos/bin/nhs_rt_full")
        );
        assert_eq!(PrismCommand::nhs_rt_full_help().args, os(&["--help"]));
        assert_eq!(
            PrismCommand::engine_capabilities(Some(Path::new("/expected.json")), true).args,
            os(&["--expected", "/expected.json", "--pretty"])
        );
    }

    /// The calibrated evidence tokens + ranker that must appear
    /// in EVERY canonical/ext/twin argv.
    fn assert_mandatory_present(args: &[OsString]) {
        let has = |flag: &str| args.iter().any(|a| a == flag);
        assert!(has("--md-only-evidence"), "missing --md-only-evidence");
        assert!(
            has("--path-a-production-profile"),
            "missing --path-a-production-profile"
        );
        assert!(
            has("--path-a-max-wall-seconds"),
            "missing --path-a-max-wall-seconds"
        );
        assert!(has("--site-ranker"), "missing --site-ranker");
        assert!(has("phase-manifold"), "missing phase-manifold ranker value");
    }

    /// Every SurfaceFlag variant we can construct, for exhaustive checks.
    fn all_surface_flags() -> Vec<SurfaceFlag> {
        vec![
            SurfaceFlag::CoupledTwin,
            SurfaceFlag::GraphCoupling,
            SurfaceFlag::Rest2,
            SurfaceFlag::MultiTemp,
            SurfaceFlag::SteppedHolds,
            SurfaceFlag::M1MonolithicDiscovery,
            SurfaceFlag::MarV2Telemetry,
            SurfaceFlag::clustering_backend("ghost-phase-lattice-4d").unwrap(),
            SurfaceFlag::clustering_backend("gpu-hash").unwrap(),
            SurfaceFlag::clustering_backend("auto").unwrap(),
            SurfaceFlag::clustering_backend("optix").unwrap(),
            SurfaceFlag::clustering_backend("lbvh").unwrap(),
            SurfaceFlag::GhostSo3(0.75),
            SurfaceFlag::GhostCellA(5.0),
            SurfaceFlag::GhostMaxTemporalEdge(500),
            SurfaceFlag::GhostStepBucket(500),
            SurfaceFlag::CapturedGraphBodyUnroll,
            SurfaceFlag::SpikePercentile(70),
            SurfaceFlag::NmaPerturb(PathBuf::from("/data/x_nma_modes.json")),
            SurfaceFlag::NmaAmplification(3.0),
        ]
    }

    #[test]
    fn ext_empty_surface_is_base_plus_default_run_params() {
        // With spike_percentile = 70 the base is byte-for-byte the locked
        // `validate_and_run` base; ext then appends exactly the six default
        // physics/capture/detection tokens (no surface tail).
        let base = PrismCommand::validate_and_run(
            Path::new("/data/1btl.topology.json"),
            Path::new("/runs/1btl"),
            8,
        );
        let ext = PrismCommand::validate_and_run_ext(
            Path::new("/data/1btl.topology.json"),
            Path::new("/runs/1btl"),
            8,
            180,
            None,
            &[],
            70,
            &RunParams::default(),
        );
        assert!(
            ext.args.starts_with(&base.args),
            "the locked base must be a strict prefix of ext"
        );
        let tail = &ext.args[base.args.len()..];
        // Honest default tail: only the two genuinely-honored knobs. The step
        // budget is OFF by default (no --steps / --wall-bound-hysteresis).
        assert_eq!(
            tail,
            os(&["--save-trajectory-interval", "0", "--lining-cutoff", "8"]).as_slice()
        );
        // Proven fidelity-trap flags must NEVER be emitted, and --steps only appears
        // when the step budget is explicitly honored (default off).
        for forbidden in [
            "--temperature",
            "--cryo-temp",
            "--cluster-threshold",
            "--steps",
            "--wall-bound-hysteresis",
        ] {
            assert!(
                !ext.args.iter().any(|a| a == forbidden),
                "default ext must not emit {forbidden}"
            );
        }
        // Exactly ONE --spike-percentile (threaded through the base, never
        // duplicated by the surface tail).
        assert_eq!(
            ext.args
                .iter()
                .filter(|a| *a == "--spike-percentile")
                .count(),
            1
        );
    }

    #[test]
    fn ext_fast_nondefault_step_budget_pairs_steps_with_wall_bound_hysteresis() {
        // FAST protocol (default), a NON-default step budget: --steps is a no-op on
        // its own, so it is emitted together with --wall-bound-hysteresis (the only
        // way the engine honors it under --fast; steps_per_stream,
        // nhs_rt_full.rs:9080). --fast stays; --temperature/--cryo-temp are NOT
        // emitted on the fast path.
        let params = RunParams {
            steps: 1_000_000,
            ..RunParams::default() // standard_protocol = false
        };
        let ext = PrismCommand::validate_and_run_ext(
            Path::new("/data/t.topology.json"),
            Path::new("/runs/t"),
            8,
            180,
            None,
            &[],
            70,
            &params,
        );
        let has = |f: &str| ext.args.iter().any(|a| a == f);
        assert!(has("--steps") && has("--wall-bound-hysteresis"));
        assert!(has("--fast"), "fast protocol keeps --fast");
        assert!(!has("--temperature") && !has("--cryo-temp"));
        let si = ext.args.iter().position(|a| a == "--steps").unwrap();
        assert_eq!(ext.args[si + 1].to_str(), Some("1000000"));
        assert_mandatory_present(&ext.args);
    }

    #[test]
    fn ext_standard_protocol_emits_temp_cryo_steps_and_drops_fast_and_multidiff() {
        // STANDARD protocol: the base drops --fast AND --multi-differential so the
        // engine takes its user-configurable else-branch (nhs_rt_full.rs:5094-5112 /
        // :9052-9080), and push_run_params emits --temperature/--cryo-temp/--steps
        // directly (NO --wall-bound-hysteresis needed — honored 1:1).
        let params = RunParams {
            standard_protocol: true,
            steps: 250_000,
            temperature: 320.0,
            cryo_temp: 40.0,
            ..RunParams::default()
        };
        let ext = PrismCommand::validate_and_run_ext(
            Path::new("/data/t.topology.json"),
            Path::new("/runs/t"),
            8,
            180,
            None,
            &[],
            70,
            &params,
        );
        let has = |f: &str| ext.args.iter().any(|a| a == f);
        // Dropped by the standard base.
        assert!(!has("--fast"), "standard drops --fast");
        assert!(
            !has("--multi-differential"),
            "standard drops --multi-differential"
        );
        assert!(
            !has("--wall-bound-hysteresis"),
            "standard honors --steps directly, no wall-bound"
        );
        // Emitted + honored 1:1.
        let ti = ext.args.iter().position(|a| a == "--temperature").unwrap();
        assert_eq!(ext.args[ti + 1].to_str(), Some("320"));
        let ci = ext.args.iter().position(|a| a == "--cryo-temp").unwrap();
        assert_eq!(ext.args[ci + 1].to_str(), Some("40"));
        let si = ext.args.iter().position(|a| a == "--steps").unwrap();
        assert_eq!(ext.args[si + 1].to_str(), Some("250000"));
        // Calibrated evidence defaults remain intact on the standard path.
        assert_mandatory_present(&ext.args);
    }

    #[test]
    fn replicate_consensus_builds_canonical_entry_argv() {
        // CLAUDE.md canonical production entry point shape.
        let cmd = PrismCommand::replicate_consensus(
            Path::new("/data/1btl.topology.json"),
            "1btl",
            5,
            Path::new("/runs/1btl_consensus"),
        );
        assert_eq!(cmd.bin, PrismBin::PrismReplicate);
        assert_eq!(
            cmd.args,
            os(&[
                "--topology",
                "/data/1btl.topology.json",
                "--target-name",
                "1btl",
                "--n-replicates",
                "5",
                "--output-dir",
                "/runs/1btl_consensus",
            ])
        );
    }

    #[test]
    fn ext_appends_only_surface_tail_in_order() {
        let base = PrismCommand::validate_and_run(
            Path::new("/data/t.topology.json"),
            Path::new("/runs/t"),
            8,
        );
        let surface = vec![
            SurfaceFlag::Rest2,
            SurfaceFlag::clustering_backend("ghost-phase-lattice-4d").unwrap(),
            SurfaceFlag::M1MonolithicDiscovery,
        ];
        let ext = PrismCommand::validate_and_run_ext(
            Path::new("/data/t.topology.json"),
            Path::new("/runs/t"),
            8,
            180,
            None,
            &surface,
            70,
            &RunParams::default(),
        );
        // The base is a strict prefix; the tail is the six run-param tokens
        // followed by exactly the surface tokens, in order.
        assert!(ext.args.starts_with(&base.args), "base must be a prefix");
        let tail = &ext.args[base.args.len()..];
        let mut expected: Vec<OsString> = Vec::new();
        push_run_params(&mut expected, &RunParams::default());
        for f in &surface {
            expected.extend(f.to_args());
        }
        assert_eq!(tail, expected.as_slice());
    }

    #[test]
    fn ext_chain_map_precedes_surface_tail() {
        let ext = PrismCommand::validate_and_run_ext(
            Path::new("/data/m_merged.topology.json"),
            Path::new("/runs/m"),
            8,
            180,
            Some(Path::new("/data/m_merged.chain_map.json")),
            &[SurfaceFlag::Rest2],
            70,
            &RunParams::default(),
        );
        let cm = ext
            .args
            .iter()
            .position(|a| a == "--chain-map")
            .expect("--chain-map present");
        let r2 = ext.args.iter().position(|a| a == "--rest2").unwrap();
        assert!(cm < r2, "--chain-map must precede surface tail");
        assert_eq!(
            ext.args[cm + 1].to_str(),
            Some("/data/m_merged.chain_map.json")
        );
    }

    #[test]
    fn typed_surface_flags_remain_order_preserving() {
        // Every single typed convenience flag remains representable.
        for f in all_surface_flags() {
            let cmd = PrismCommand::validate_and_run_ext(
                Path::new("/t.topology.json"),
                Path::new("/o"),
                8,
                180,
                None,
                std::slice::from_ref(&f),
                70,
                &RunParams::default(),
            );
            assert_mandatory_present(&cmd.args);
        }
        let all = all_surface_flags();
        let cmd = PrismCommand::validate_and_run_ext(
            Path::new("/t.topology.json"),
            Path::new("/o"),
            8,
            180,
            None,
            &all,
            70,
            &RunParams::default(),
        );
        assert_mandatory_present(&cmd.args);
    }

    #[test]
    fn clustering_backend_preserves_engine_values_for_native_validation() {
        assert!(SurfaceFlag::clustering_backend("grid").is_some());
        assert!(SurfaceFlag::clustering_backend("ghost-phase-lattice-4d").is_some());
        assert!(SurfaceFlag::clustering_backend("auto").is_some());
    }

    #[test]
    fn twin_drops_multi_differential_and_pairs_coupled_graph() {
        let twin = PrismCommand::validate_and_run_twin(
            Path::new("/t.topology.json"),
            Path::new("/o"),
            8,
            180,
            None,
            &[],
        );
        assert!(
            !twin.args.iter().any(|a| a == "--multi-differential"),
            "twin must DROP --multi-differential"
        );
        assert!(twin.args.iter().any(|a| a == "--coupled-twin"));
        assert!(twin.args.iter().any(|a| a == "--graph-coupling"));
        assert_mandatory_present(&twin.args);
    }

    #[test]
    fn twin_threads_full_surface_tail() {
        // Gap-(c): TWIN coupling is orthogonal to clustering + enhanced sampling,
        // so the FULL ghost bundle layers on top with no conflict and no filter.
        let tail = vec![
            SurfaceFlag::M1MonolithicDiscovery,
            SurfaceFlag::MarV2Telemetry,
            SurfaceFlag::clustering_backend("ghost-phase-lattice-4d").unwrap(),
            SurfaceFlag::GhostStepBucket(250),
            SurfaceFlag::Rest2,
        ];
        let cmd = PrismCommand::validate_and_run_twin(
            Path::new("/t.topology.json"),
            Path::new("/o"),
            8,
            180,
            None,
            &tail,
        );
        assert!(cmd.args.iter().any(|a| a == "--coupled-twin"));
        assert!(cmd.args.iter().any(|a| a == "--graph-coupling"));
        // Full ghost bundle survives on the twin arm (no owned-flag filter).
        assert!(cmd.args.iter().any(|a| a == "--clustering-backend"));
        assert!(cmd.args.iter().any(|a| a == "--m1-monolithic-discovery"));
        assert!(cmd.args.iter().any(|a| a == "--mar-v2-telemetry"));
        assert!(cmd
            .args
            .iter()
            .any(|a| a == "--ghost-phase-lattice-step-bucket-size"));
        assert!(cmd.args.iter().any(|a| a == "--rest2"));
        assert!(!cmd.args.iter().any(|a| a == "--multi-differential"));
        assert_mandatory_present(&cmd.args);
    }

    #[test]
    fn named_surface_keeps_ghost_knobs_but_filters_owned_backend_flags() {
        // Gap-(c): a `ghost-tiles` surface OWNS backend selection (engine expands
        // it), so those flags must NOT be duplicated — but the additive tuning
        // knobs and enhanced-sampling flags MUST survive so the run is tunable.
        let tail = vec![
            SurfaceFlag::M1MonolithicDiscovery,
            SurfaceFlag::MarV2Telemetry,
            SurfaceFlag::clustering_backend("ghost-phase-lattice-4d").unwrap(),
            SurfaceFlag::GhostSo3(0.9),
            SurfaceFlag::GhostCellA(6.0),
            SurfaceFlag::GhostMaxTemporalEdge(400),
            SurfaceFlag::GhostStepBucket(250),
            SurfaceFlag::Rest2,
        ];
        let cmd = PrismCommand::validate_and_run_engine_surface(
            Path::new("/t.topology.json"),
            Path::new("/o"),
            8,
            180,
            None,
            EngineSurface::GhostTiles,
            &RunParams::default(),
            &tail,
        );
        // Backend-selection flags the named surface owns are NOT hand-emitted.
        for owned in [
            "--clustering-backend",
            "--m1-monolithic-discovery",
            "--mar-v2-telemetry",
        ] {
            assert!(
                !cmd.args.iter().any(|a| a == owned),
                "{owned} must not be duplicated alongside a named surface"
            );
        }
        // Additive tuning knobs + enhanced sampling survive.
        assert!(cmd
            .args
            .iter()
            .any(|a| a == "--ghost-phase-lattice-so3-threshold"));
        assert!(cmd.args.iter().any(|a| a == "--ghost-phase-lattice-cell-a"));
        assert!(cmd
            .args
            .iter()
            .any(|a| a == "--ghost-phase-lattice-max-temporal-edge-steps"));
        assert!(cmd
            .args
            .iter()
            .any(|a| a == "--ghost-phase-lattice-step-bucket-size"));
        assert!(cmd.args.iter().any(|a| a == "--rest2"));
        // Exactly one named token, right value, mandatory set intact.
        assert_eq!(
            cmd.args.iter().filter(|a| *a == "--engine-surface").count(),
            1
        );
        let es = cmd
            .args
            .iter()
            .position(|a| a == "--engine-surface")
            .unwrap();
        assert_eq!(cmd.args[es + 1].to_str(), Some("ghost-tiles"));
        assert_mandatory_present(&cmd.args);
    }

    // --------------------------------------------------------------------
    // First-class NAMED engine surface (`--engine-surface`).
    // --------------------------------------------------------------------

    fn engine_surface_cmd(surface: EngineSurface) -> PrismCommand {
        PrismCommand::validate_and_run_engine_surface(
            Path::new("/t.topology.json"),
            Path::new("/o"),
            8,
            180,
            None,
            surface,
            &RunParams::default(),
            &[],
        )
    }

    #[test]
    fn engine_surface_values_map_to_exact_clap_strings() {
        // 1:1 with the engine value_parser (nhs_rt_full.rs:1041).
        assert_eq!(EngineSurface::None.as_str(), "none");
        assert_eq!(
            EngineSurface::CapturedGraphTwin.as_str(),
            "captured-graph-twin"
        );
        assert_eq!(
            EngineSurface::InterferometricTwin.as_str(),
            "interferometric-twin"
        );
        assert_eq!(EngineSurface::Twin16.as_str(), "twin16");
        assert_eq!(EngineSurface::GhostTiles.as_str(), "ghost-tiles");
    }

    #[test]
    fn engine_surface_flag_emits_single_named_token() {
        assert_eq!(
            SurfaceFlag::EngineSurface(EngineSurface::Twin16).to_args(),
            os(&["--engine-surface", "twin16"])
        );
        assert_eq!(
            SurfaceFlag::EngineSurface(EngineSurface::GhostTiles).to_args(),
            os(&["--engine-surface", "ghost-tiles"])
        );
    }

    #[test]
    fn engine_surface_cmd_emits_named_flag_directly_not_low_level() {
        for s in [
            EngineSurface::CapturedGraphTwin,
            EngineSurface::InterferometricTwin,
            EngineSurface::Twin16,
            EngineSurface::GhostTiles,
        ] {
            let cmd = engine_surface_cmd(s);
            assert_eq!(cmd.bin, PrismBin::PrismValidateRun);
            // The one named token appears EXACTLY once, with the right value.
            let hits: Vec<usize> = cmd
                .args
                .iter()
                .enumerate()
                .filter(|(_, a)| *a == "--engine-surface")
                .map(|(i, _)| i)
                .collect();
            assert_eq!(hits.len(), 1, "--engine-surface must appear exactly once");
            assert_eq!(cmd.args[hits[0] + 1].to_str(), Some(s.as_str()));
            // The base DROPS --multi-differential so captured-graph-twin is NOT
            // no-op'd by the multi_differential-first dispatch precedence; the
            // engine re-adds it for the multi_differential-based surfaces via its
            // own upstream expansion (nhs_rt_full.rs:3681-3690).
            assert!(
                !cmd.args.iter().any(|a| a == "--multi-differential"),
                "engine-surface base must drop --multi-differential"
            );
            // The UI emits ONLY the named flag — never the hand-assembled
            // low-level tokens (the whole point of a first-class named surface).
            for low in [
                "--coupled-twin",
                "--graph-coupling",
                "--clustering-backend",
                "--mar-v2-telemetry",
                "--m1-monolithic-discovery",
            ] {
                assert!(
                    !cmd.args.iter().any(|a| a == low),
                    "named surface must not hand-emit {low}"
                );
            }
            // v5 golden-master: body-unroll is never emitted here.
            assert!(!cmd.args.iter().any(|a| a == "--captured-graph-body-unroll"));
            assert_mandatory_present(&cmd.args);
        }
    }

    #[test]
    fn engine_surface_cmd_chain_map_precedes_named_flag() {
        let cmd = PrismCommand::validate_and_run_engine_surface(
            Path::new("/m_merged.topology.json"),
            Path::new("/o"),
            8,
            180,
            Some(Path::new("/m_merged.chain_map.json")),
            EngineSurface::CapturedGraphTwin,
            &RunParams::default(),
            &[],
        );
        let cm = cmd
            .args
            .iter()
            .position(|a| a == "--chain-map")
            .expect("--chain-map present");
        let es = cmd
            .args
            .iter()
            .position(|a| a == "--engine-surface")
            .unwrap();
        assert!(cm < es, "--chain-map must precede --engine-surface");
        assert_eq!(cmd.args[cm + 1].to_str(), Some("/m_merged.chain_map.json"));
        assert_eq!(cmd.args[es + 1].to_str(), Some("captured-graph-twin"));
    }

    #[test]
    fn engine_surface_twin16_passes_sizing_multi_stream_verbatim() {
        // The UI still passes the sizing-derived --multi-stream; the ENGINE bumps
        // it to >=16 for twin16 (nhs_rt_full.rs:3684-3690). A 20-stream large
        // target is left at 20 (the engine only bumps when < 16).
        let cmd = PrismCommand::validate_and_run_engine_surface(
            Path::new("/t.topology.json"),
            Path::new("/o"),
            20,
            180,
            None,
            EngineSurface::Twin16,
            &RunParams::default(),
            &[],
        );
        let i = cmd.args.iter().position(|a| a == "--multi-stream").unwrap();
        assert_eq!(cmd.args[i + 1].to_str(), Some("20"));
    }

    #[test]
    fn engine_surface_none_equals_named_none_value() {
        // None renders the literal engine no-op value; not routed by the UI.
        let cmd = engine_surface_cmd(EngineSurface::None);
        let es = cmd
            .args
            .iter()
            .position(|a| a == "--engine-surface")
            .unwrap();
        assert_eq!(cmd.args[es + 1].to_str(), Some("none"));
        assert_mandatory_present(&cmd.args);
    }

    #[test]
    fn smoke_has_fast_25k_no_multi_differential() {
        for cmd in [
            PrismCommand::smoke_min(Path::new("/t.topology.json"), Path::new("/o")),
            PrismCommand::smoke_eng(Path::new("/t.topology.json"), Path::new("/o")),
        ] {
            assert!(cmd.args.iter().any(|a| a == "--fast-25k"));
            assert!(!cmd.args.iter().any(|a| a == "--multi-differential"));
            assert_mandatory_present(&cmd.args);
        }
        // stream counts 1 and 2 respectively
        let mmin = PrismCommand::smoke_min(Path::new("/t.topology.json"), Path::new("/o"));
        let i = mmin
            .args
            .iter()
            .position(|a| a == "--multi-stream")
            .unwrap();
        assert_eq!(mmin.args[i + 1].to_str(), Some("1"));
    }

    #[test]
    fn max_wall_override_is_single_token_not_duplicated() {
        let ext = PrismCommand::validate_and_run_ext(
            Path::new("/t.topology.json"),
            Path::new("/o"),
            8,
            240,
            None,
            &[],
            70,
            &RunParams::default(),
        );
        let hits: Vec<usize> = ext
            .args
            .iter()
            .enumerate()
            .filter(|(_, a)| *a == "--path-a-max-wall-seconds")
            .map(|(i, _)| i)
            .collect();
        assert_eq!(hits.len(), 1, "max-wall must appear exactly once");
        assert_eq!(ext.args[hits[0] + 1].to_str(), Some("240"));
    }

    #[test]
    fn surface_flags_map_to_real_flag_strings() {
        assert_eq!(SurfaceFlag::CoupledTwin.to_args(), os(&["--coupled-twin"]));
        assert_eq!(
            SurfaceFlag::clustering_backend("ghost-phase-lattice-4d")
                .unwrap()
                .to_args(),
            os(&["--clustering-backend", "ghost-phase-lattice-4d"])
        );
        assert_eq!(
            SurfaceFlag::CapturedGraphBodyUnroll.to_args(),
            os(&["--captured-graph-body-unroll", "1"])
        );
        assert_eq!(
            SurfaceFlag::GhostSo3(0.75).to_args(),
            os(&["--ghost-phase-lattice-so3-threshold", "0.75"])
        );
    }

    #[test]
    fn clean_pdb_is_positional_with_explicit_chain() {
        let cmd = PrismCommand::clean_pdb(
            Path::new("/data/1btl_raw.pdb"),
            Path::new("/data/1btl_clean.pdb"),
            "A",
        );
        assert_eq!(cmd.bin, PrismBin::PrismClean);
        assert_eq!(
            cmd.args,
            os(&["/data/1btl_raw.pdb", "/data/1btl_clean.pdb", "A"])
        );
    }

    #[test]
    fn prep_topology_positional_in_out_and_only_set_optionals() {
        let bare = PrismCommand::prep_topology(
            Path::new("/data/1btl_clean.pdb"),
            Path::new("/data/1btl_clean.topology.json"),
            None,
            false,
            false,
            false,
            None,
            None,
        );
        assert_eq!(bare.bin, PrismBin::PrismPrep);
        assert_eq!(
            bare.args,
            os(&["/data/1btl_clean.pdb", "/data/1btl_clean.topology.json"])
        );
        // -o must NOT appear (that is --output-dir for --batch).
        assert!(!bare.args.iter().any(|a| a == "-o"));

        let full = PrismCommand::prep_topology(
            Path::new("/data/c.pdb"),
            Path::new("/data/c.topology.json"),
            Some("cryptic"),
            true,
            true,
            true,
            Some(20),
            Some(3.0),
        );
        assert_eq!(
            full.args,
            os(&[
                "/data/c.pdb",
                "/data/c.topology.json",
                "--mode",
                "cryptic",
                "--use-amber",
                "--strict",
                "--hmr",
                "--nma-modes",
                "20",
                "--nma-amplification",
                "3",
            ])
        );
    }

    #[test]
    fn merge_chains_inputs_then_output_then_map() {
        let cmd = PrismCommand::merge_chains(
            &[
                PathBuf::from("/data/x_chainA.pdb"),
                PathBuf::from("/data/x_chainB.pdb"),
            ],
            Path::new("/data/x_merged.pdb"),
            Path::new("/data/x_merged.chain_map.json"),
        );
        assert_eq!(cmd.bin, PrismBin::PrismMergeChains);
        assert_eq!(
            cmd.args,
            os(&[
                "/data/x_chainA.pdb",
                "/data/x_chainB.pdb",
                "-o",
                "/data/x_merged.pdb",
                "--chain-map",
                "/data/x_merged.chain_map.json",
            ])
        );
    }

    #[test]
    fn mol_to_smiles_is_single_path_arg() {
        let cmd = PrismCommand::mol_to_smiles(Path::new("/mols/aspirin.sdf"));
        assert_eq!(cmd.bin, PrismBin::PrismMol2Smi);
        assert_eq!(cmd.args, os(&["/mols/aspirin.sdf"]));
    }

    #[test]
    fn qnanl_encode_builds_exact_argv() {
        let cmd = PrismCommand::qnanl_encode(
            Path::new("in.json"),
            Path::new("out.prismqnanl"),
            Path::new("out.receipt.json"),
            "SpikeEvent",
            Some("art-123"),
        );
        assert_eq!(cmd.bin, PrismBin::PrismQnanl);
        assert_eq!(
            cmd.args,
            os(&[
                "encode",
                "--input",
                "in.json",
                "--packet",
                "out.prismqnanl",
                "--receipt",
                "out.receipt.json",
                "--family",
                "SpikeEvent",
                "--artifact-id",
                "art-123",
            ])
        );
    }

    #[test]
    fn qnanl_encode_omits_optional_artifact_id() {
        let cmd = PrismCommand::qnanl_encode(
            Path::new("in"),
            Path::new("p"),
            Path::new("r"),
            "BocpdLog",
            None,
        );
        assert!(!cmd.args.iter().any(|a| a.to_str() == Some("--artifact-id")));
    }

    #[test]
    fn qnanl_decode_and_header() {
        assert_eq!(
            PrismCommand::qnanl_decode(Path::new("p.prismqnanl"), Path::new("o.json")).args,
            os(&["decode", "--packet", "p.prismqnanl", "--output", "o.json"])
        );
        assert_eq!(
            PrismCommand::qnanl_header(Path::new("p.prismqnanl")).args,
            os(&["header", "--packet", "p.prismqnanl"])
        );
    }

    #[test]
    fn hw_inspect_cuda_flag_is_presence_only() {
        assert_eq!(
            PrismCommand::hw_inspect(true, None).args,
            os(&["inspect", "--cuda"])
        );
        assert_eq!(PrismCommand::hw_inspect(false, None).args, os(&["inspect"]));
        assert_eq!(
            PrismCommand::hw_inspect(false, Some(Path::new("hw.json"))).args,
            os(&["inspect", "--json", "hw.json"])
        );
    }

    #[test]
    fn seal_verify_is_positional() {
        assert_eq!(
            PrismCommand::seal_verify(Path::new("a.pcode")).args,
            os(&["verify", "a.pcode"])
        );
    }

    #[test]
    fn val_validate_carries_level() {
        assert_eq!(
            PrismCommand::val_validate(Path::new("a.pcode"), "L5", None).args,
            os(&["validate", "a.pcode", "--level", "L5"])
        );
    }

    #[test]
    fn ui_dashboard_optional_input_and_panel() {
        let cmd =
            PrismCommand::ui_dashboard(None, Some(Path::new("s.json")), None, Some("nanl-hd"));
        assert_eq!(
            cmd.args,
            os(&["--status", "s.json", "--open-panel", "nanl-hd"])
        );
    }

    #[test]
    fn cryptic_site_analysis_carries_run_root_and_output_dir() {
        let cmd = PrismCommand::cryptic_site_analysis(
            Path::new("/runs/current"),
            Path::new("/out/cryptic"),
        );
        assert_eq!(cmd.bin, PrismBin::PrismCrypticSiteAnalysis);
        assert_eq!(
            cmd.args,
            os(&["--run-root", "/runs/current", "--out", "/out/cryptic",])
        );
    }

    #[test]
    fn resolver_fails_closed_to_required_install_path() {
        let resolver = BinaryResolver::from_dirs(vec![PathBuf::from("/nonexistent/dir")]);
        assert_eq!(
            resolver.resolve(PrismBin::PrismHw),
            PathBuf::from("/opt/prismos/bin/prismhw")
        );
    }

    #[test]
    fn resolver_fails_closed_to_installed_copilot_script_path() {
        let resolver = BinaryResolver::from_dirs(vec![PathBuf::from("/nonexistent/dir")]);
        assert_eq!(
            resolver.resolve(PrismBin::PrismValidateRun),
            PathBuf::from(
                "/opt/prismos/source/copilot_executor_cfe5ea89/scripts/prism-validate-and-run.sh"
            )
        );
    }

    #[test]
    fn script_override_key_is_a_valid_environment_identifier() {
        assert_eq!(
            env_override_key(PrismBin::PrismValidateRun),
            "PRISMOS_UI_BIN_PRISM_VALIDATE_AND_RUN_SH"
        );
    }

    #[test]
    fn resolver_prefers_existing_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("prismqnanl");
        std::fs::write(&bin, b"#!/bin/sh\n").unwrap();
        let resolver = BinaryResolver::from_dirs(vec![tmp.path().to_path_buf()]);
        assert_eq!(resolver.resolve(PrismBin::PrismQnanl), bin);
    }

    #[test]
    fn display_quotes_spaces_but_execution_uses_argv() {
        let cmd = PrismCommand::qnanl_header(Path::new("/runs/with space/p.prismqnanl"));
        let resolver = BinaryResolver::from_dirs(vec![]);
        let line = cmd.display(&resolver);
        assert!(line.contains("'/runs/with space/p.prismqnanl'"));
        // The argv itself is unquoted — Command::args passes it verbatim.
        assert_eq!(cmd.args[2].to_str(), Some("/runs/with space/p.prismqnanl"));
    }
}

#[cfg(test)]
mod primary_1010_request_tests {
    use super::*;
    #[test]
    fn normal_primary_request_is_exact_complete_controller_request() {
        let cmd = PrismCommand::primary_1010_cycle(Path::new("/prepared/panel"));
        assert_eq!(cmd.bin, PrismBin::PrismValidateRun);
        assert_eq!(cmd.args, vec!["run-cycle", "--root", "/prepared/panel"]);
        assert!(!cmd.args.iter().any(|v| v == "--md-only-evidence" || v == "--path-a-max-wall-seconds"));
    }
    #[test]
    fn inspection_and_completion_are_distinct_read_only_requests() {
        assert_eq!(PrismCommand::primary_1010_inspect(Path::new("/p")).args, vec!["inspect-panel", "--root", "/p"]);
        assert_eq!(PrismCommand::primary_1010_result(Path::new("/p")).args, vec!["verify-result", "--root", "/p"]);
    }
}
