//! Run-root branch discovery (GTK-free, unit-testable).
//!
//! PrismOS native runs are organized as a *run root* containing one directory
//! per **branch** (one modality / seed replicate of the run). Each branch
//! directory carries its own modality receipts and a `pathb_runtime_stage/`
//! subdirectory with the materialized binding sites.
//!
//! The desktop panels used to hardcode a fixed list of four bare modality
//! names (`apo_orthosteric`, `apo_cryptic`, `holo_gdp_orthosteric`,
//! `holo_gdp_cryptic`) and join them straight under the run root. Real run
//! roots do not name their branch directories that way — a high-fidelity
//! chronology matrix names them `row01_apo_orthosteric_seed7101` …
//! `row12_holo_cryptic_seed7103` (a `rowNN_` prefix, a `_seedNNNN` suffix, and
//! `holo_*` rather than `holo_gdp_*`). The hardcoded joins therefore never
//! resolved and the panels reported "unavailable" even though the evidence was
//! present.
//!
//! This module replaces the hardcoded list with real filesystem discovery: it
//! walks the run root (a few nesting levels deep, symlink- and cycle-safe) and
//! returns every directory that actually contains the marker file(s) that
//! define a branch. Discovery keys off the on-disk evidence, not off a naming
//! convention, so it is robust to the concrete branch names, to the number of
//! branches, and to reasonable grouping/nesting under the run root.

use std::path::{Path, PathBuf};

/// Supported relative locations, inside a branch directory, of the
/// materialized Path-B binding sites. The installed launcher writes the file
/// directly into its output directory; retained multi-branch science runs use
/// the historical `pathb_runtime_stage/` child directory.
///
/// Direct installed-run output is intentionally first. If both layouts are
/// present in one branch, the current launcher result is authoritative and a
/// retained nested artifact cannot silently shadow it.
pub const BINDING_SITES_MARKERS: &[&[&str]] = &[
    &["binding_sites.materialized.json"],
    &["pathb_runtime_stage", "binding_sites.materialized.json"],
];

/// Any one of these relative paths, present inside a directory, marks it as a
/// native *modality branch* for the Scientific Results surface. A branch is a
/// modality branch if it carries at least one science-bearing evidence file.
const MODALITY_MARKERS: &[&[&str]] = &[
    &["prismos_modality_receipt.json"],
    &["TWIN16_INTERFEROMETRIC_TRANSITION_VIEW_RESIDUALS.json"],
    &["thermodynamic_phase_2x2x2x2_condition_ledger.json"],
    &["dense_tensor_semantic_summaries.json"],
    &["prismos_runtime_tokenizer_export_receipt.json"],
    &["binding_sites.materialized.json"],
    &["pathb_runtime_stage", "binding_sites.materialized.json"],
];

/// Maximum number of directory levels below the run root at which a branch may
/// be found. Level 1 is an immediate child of the run root (the common case);
/// deeper levels tolerate a run root that groups its branches one or two
/// directories down (e.g. `<root>/<matrix>/<branch>/…`).
const MAX_BRANCH_DEPTH: usize = 3;

/// A discovered branch directory: its display name (the directory basename,
/// used as the branch label) and its absolute path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchDir {
    /// Directory basename, used verbatim as the branch label in the UI.
    pub name: String,
    /// Absolute path to the branch directory.
    pub path: PathBuf,
}

impl BranchDir {
    fn new(path: &Path) -> Self {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.to_string_lossy().into_owned());
        BranchDir {
            name,
            path: path.to_path_buf(),
        }
    }

    /// Join a relative marker path (a slice of path components) under this
    /// branch directory.
    pub fn join_rel(&self, rel: &[&str]) -> PathBuf {
        let mut p = self.path.clone();
        for component in rel {
            p.push(component);
        }
        p
    }

    /// Absolute path to this branch's materialized binding-sites document.
    pub fn binding_sites_materialized(&self) -> PathBuf {
        BINDING_SITES_MARKERS
            .iter()
            .map(|marker| self.join_rel(marker))
            .find(|candidate| candidate.is_file())
            .unwrap_or_else(|| self.join_rel(BINDING_SITES_MARKERS[0]))
    }
}

fn join_rel(dir: &Path, rel: &[&str]) -> PathBuf {
    let mut p = dir.to_path_buf();
    for component in rel {
        p.push(component);
    }
    p
}

/// True when `dir` is a binding-site branch (contains the materialized Path-B
/// binding-sites file).
pub fn is_binding_site_branch(dir: &Path) -> bool {
    if !native_binding_site_publications(dir).is_empty() { return true; }
    BINDING_SITES_MARKERS
        .iter()
        .any(|marker| join_rel(dir, marker).is_file())
}

/// True when `dir` is a native modality branch (carries at least one
/// science-bearing modality artifact).
pub fn is_modality_branch(dir: &Path) -> bool {
    if !native_binding_site_publications(dir).is_empty() { return true; }
    MODALITY_MARKERS
        .iter()
        .any(|marker| join_rel(dir, marker).is_file())
}

/// Native publication is a distinct source format; no materialized file is created.
#[derive(Debug, Clone)]
pub struct NativeBindingPublication {
    pub publication: PathBuf,
    pub source: PathBuf,
    pub source_sha256: String,
    pub structure: String,
    pub identity_domain: String,
    /// Exact final native array order and values, with no rank or score conversion.
    pub sites: Vec<serde_json::Value>,
    pub visual_files: Vec<PathBuf>,
}

/// Discover only regular native publication files at this branch depth.
pub fn native_binding_site_publications(dir: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            if entry.file_type().map(|t| t.is_file()).unwrap_or(false)
                && entry.file_name().to_string_lossy().ends_with(".binding_sites.publication.json") {
                paths.push(entry.path());
            }
        }
    }
    paths.sort();
    paths
}

fn publication_sha256(path: &Path) -> Result<String, String> {
    // Hashing is read-only; no native process, script or molecular provider executes.
    let output = std::process::Command::new("/usr/bin/sha256sum")
        .arg("--").arg(path).output().map_err(|e| e.to_string())?;
    if !output.status.success() { return Err(format!("Cannot hash {}", path.display())); }
    let digest = String::from_utf8_lossy(&output.stdout).split_whitespace()
        .next().unwrap_or("").to_string();
    if digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("Invalid source hash output".to_string());
    }
    Ok(digest)
}

fn publication_bytes_sha256(bytes: &[u8]) -> Result<String, String> {
    use std::io::Write;
    use std::process::Stdio;
    let mut child = std::process::Command::new("/usr/bin/sha256sum").arg("-")
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped())
        .spawn().map_err(|e| e.to_string())?;
    child.stdin.take().ok_or("Hash stdin unavailable")?.write_all(bytes).map_err(|e| e.to_string())?;
    let output = child.wait_with_output().map_err(|e| e.to_string())?;
    let digest = String::from_utf8_lossy(&output.stdout).split_whitespace().next().unwrap_or("").to_string();
    if !output.status.success() || digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("Native source byte hash unavailable".to_string());
    }
    Ok(digest)
}

/// Read the producer's native JSON/publication pair and its complete visual inventory.
/// This verifies source-format consistency; it is not MD/scientific acceptance.
pub fn load_native_binding_publication(publication: &Path) -> Result<NativeBindingPublication, String> {
    use serde_json::Value;
    let filename = publication.file_name().and_then(|n| n.to_str()).ok_or("Non-UTF8 publication name")?;
    let prefix = filename.strip_suffix(".binding_sites.publication.json")
        .filter(|s| !s.is_empty()).ok_or("Native publication suffix required")?;
    let parent = publication.parent().ok_or("Publication parent missing")?;
    let source = parent.join(format!("{prefix}.binding_sites.json"));
    for p in [publication, source.as_path()] {
        if !std::fs::symlink_metadata(p).map_err(|e| e.to_string())?.file_type().is_file() {
            return Err("Native source/publication must be regular files".to_string());
        }
    }
    let parse = |p: &Path| -> Result<Value, String> {
        serde_json::from_str(&std::fs::read_to_string(p).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
    };
    let published = parse(publication)?;
    let native_bytes = std::fs::read(&source).map_err(|e| e.to_string())?;
    let native: Value = serde_json::from_slice(&native_bytes).map_err(|e| e.to_string())?;
    if published["schema"] != "prism.canonical_binding_site_publication.v1"
        || native["schema_version"].as_u64() != Some(2) {
        return Err("Unsupported native prediction/publication schema".to_string());
    }
    let source_sha256 = publication_bytes_sha256(&native_bytes)?;
    if published["source_json_sha256"].as_str() != Some(source_sha256.as_str()) {
        return Err("Native publication source SHA256 differs".to_string());
    }
    let sites = native["sites"].as_array().ok_or("Native sites array missing")?;
    if published["site_count"].as_u64() != Some(sites.len() as u64)
        || native["binding_sites"].as_u64() != Some(sites.len() as u64)
        || published["sites"].as_array() != Some(sites)
        || published["site_identity_domain"] != native["binding_site_identity_domain"]
        || published["order"] != "final native JSON sites array; no new scientific rank" {
        return Err("Published native count/identity/order/site values differ".to_string());
    }
    let domain = native["binding_site_identity_domain"].as_str()
        .filter(|v| matches!(*v, "ccl_cluster" | "materialized_site"))
        .ok_or("Unsupported native identity domain")?.to_string();
    let mut ids = std::collections::BTreeSet::new();
    for site in sites {
        let id = &site["id"];
        if !(id.is_i64() || id.is_u64() || id.as_str().map(|v| !v.is_empty()).unwrap_or(false))
            || !ids.insert(id.to_string()) { return Err("Missing/duplicate native site identity".to_string()); }
        let xyz = site["centroid"].as_array().filter(|v| v.len() == 3)
            .ok_or("Native centroid geometry missing")?;
        if !xyz.iter().all(|v| v.as_f64().map(|n| n.is_finite()).unwrap_or(false)) {
            return Err("Nonfinite native centroid".to_string());
        }
    }
    // Available native ordinal fields retain their own names and population.
    for key in ["rank", "cryptic_rank"] {
        let available = sites.iter().filter(|r| !r[key].is_null()).count();
        if available != 0 {
            let mut ranks = sites.iter().map(|r| r[key].as_u64().filter(|n| *n > 0)
                .ok_or_else(|| format!("Invalid/incomplete native {key}")))
                .collect::<Result<Vec<_>, _>>()?;
            ranks.sort_unstable();
            if ranks != (1..=sites.len() as u64).collect::<Vec<_>>() {
                return Err(format!("Native {key} ordinal population differs"));
            }
        }
    }
    let files = published["files"].as_array().ok_or("Native visual inventory missing")?;
    let expected = ["pdb", "pml", "cxc", "md"].map(|ext| format!("{prefix}.binding_sites.{ext}"));
    if files.len() != expected.len() { return Err("Native visual inventory count differs".to_string()); }
    let mut seen = std::collections::BTreeSet::new();
    let mut visual_files = Vec::new();
    for row in files {
        let name = row["file"].as_str().filter(|n| expected.iter().any(|e| e.as_str() == *n))
            .ok_or("Unexpected native visual filename")?;
        if !seen.insert(name.to_string()) { return Err("Duplicate native visual filename".to_string()); }
        let path = parent.join(name);
        let stat = std::fs::symlink_metadata(&path).map_err(|e| e.to_string())?;
        if !stat.file_type().is_file() || row["bytes"].as_u64() != Some(stat.len())
            || row["sha256"].as_str() != Some(publication_sha256(&path)?.as_str()) {
            return Err(format!("Native visual publication changed: {name}"));
        }
        visual_files.push(path);
    }
    Ok(NativeBindingPublication {
        publication: publication.to_path_buf(), source,
        source_sha256, structure: native["structure"].as_str().unwrap_or("unknown").to_string(),
        identity_domain: domain, sites: sites.clone(), visual_files,
    })
}

/// Discover every binding-site branch under `run_root`.
pub fn discover_binding_site_branches(run_root: &Path) -> Vec<BranchDir> {
    discover_branches(run_root, &is_binding_site_branch)
}

/// Discover every native modality branch under `run_root`.
pub fn discover_modality_branches(run_root: &Path) -> Vec<BranchDir> {
    discover_branches(run_root, &is_modality_branch)
}

/// Discover branch directories under `run_root` for which `is_branch` holds.
///
/// The run root itself is checked first, so a single-run layout (branch content
/// directly under the run root, no per-branch subdirectory) is handled. When
/// the run root is not itself a branch, its subtree is walked up to
/// [`MAX_BRANCH_DEPTH`] levels deep; a directory that satisfies `is_branch` is
/// recorded and not descended into further. Results are de-duplicated by path
/// and sorted by name for deterministic display.
pub fn discover_branches(run_root: &Path, is_branch: &dyn Fn(&Path) -> bool) -> Vec<BranchDir> {
    let mut out: Vec<BranchDir> = Vec::new();

    // Single-run layout: the run root itself is the branch.
    if is_branch(run_root) {
        out.push(BranchDir::new(run_root));
        return out;
    }

    walk(run_root, 0, is_branch, &mut out);

    out.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| a.path.cmp(&b.path)));
    out.dedup_by(|a, b| a.path == b.path);
    out
}

/// Bounded, symlink-safe directory walk. Symlinked directories report
/// `is_dir() == false` on the dirent file type and are skipped, which makes the
/// walk cycle-safe. Never panics: unreadable directories are skipped.
fn walk(dir: &Path, depth: usize, is_branch: &dyn Fn(&Path) -> bool, out: &mut Vec<BranchDir>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    let mut child_dirs: Vec<PathBuf> = Vec::new();
    for entry in entries.flatten() {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        // `is_dir()` on the dirent type is false for symlinks -> never followed.
        if file_type.is_dir() {
            child_dirs.push(entry.path());
        }
    }
    child_dirs.sort();

    for child in child_dirs {
        if is_branch(&child) {
            out.push(BranchDir::new(&child));
        } else if depth + 1 < MAX_BRANCH_DEPTH {
            walk(&child, depth + 1, is_branch, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Write an empty file, creating parent directories as needed.
    fn touch(path: &Path) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, b"{}").unwrap();
    }

    /// The real-world high-fidelity chronology matrix layout: 12 branch
    /// directories named `rowNN_<mode>_seedNNNN`, each with its own
    /// `pathb_runtime_stage/binding_sites.materialized.json` and modality
    /// receipts — none matching the old hardcoded `apo_orthosteric` names.
    fn build_matrix_root(root: &Path) {
        let branches = [
            "row01_apo_orthosteric_seed7101",
            "row02_apo_orthosteric_seed7102",
            "row03_apo_orthosteric_seed7103",
            "row04_apo_cryptic_seed7101",
            "row05_apo_cryptic_seed7102",
            "row06_apo_cryptic_seed7103",
            "row07_holo_orthosteric_seed7101",
            "row08_holo_orthosteric_seed7102",
            "row09_holo_orthosteric_seed7103",
            "row10_holo_cryptic_seed7101",
            "row11_holo_cryptic_seed7102",
            "row12_holo_cryptic_seed7103",
        ];
        for branch in branches {
            let dir = root.join(branch);
            touch(
                &dir.join("pathb_runtime_stage")
                    .join("binding_sites.materialized.json"),
            );
            touch(&dir.join("prismos_modality_receipt.json"));
        }
        // Ledger/summary files at the run-root top level must not be mistaken
        // for branches.
        touch(&root.join("MATRIX_ACCEPTANCE_SUMMARY.json"));
    }

    #[test]
    fn discovers_all_real_named_binding_site_branches() {
        let tmp = tempfile::tempdir().unwrap();
        build_matrix_root(tmp.path());

        let branches = discover_binding_site_branches(tmp.path());
        assert_eq!(branches.len(), 12, "expected all 12 matrix branches");
        // Sorted by name; first is row01, last is row12.
        assert_eq!(
            branches.first().unwrap().name,
            "row01_apo_orthosteric_seed7101"
        );
        assert_eq!(branches.last().unwrap().name, "row12_holo_cryptic_seed7103");
        // The marker file resolves for every discovered branch.
        for branch in &branches {
            assert!(branch.binding_sites_materialized().is_file());
        }
    }

    #[test]
    fn discovers_all_real_named_modality_branches() {
        let tmp = tempfile::tempdir().unwrap();
        build_matrix_root(tmp.path());
        let branches = discover_modality_branches(tmp.path());
        assert_eq!(branches.len(), 12);
    }

    #[test]
    fn old_hardcoded_names_are_absent_but_discovery_still_finds_branches() {
        // Proves the regression: the four old bare names do not exist on disk,
        // yet discovery finds the real branches anyway.
        let tmp = tempfile::tempdir().unwrap();
        build_matrix_root(tmp.path());
        for bare in [
            "apo_orthosteric",
            "apo_cryptic",
            "holo_gdp_orthosteric",
            "holo_gdp_cryptic",
        ] {
            assert!(!tmp.path().join(bare).exists());
        }
        assert_eq!(discover_binding_site_branches(tmp.path()).len(), 12);
    }

    #[test]
    fn modality_branch_needs_only_one_marker() {
        let tmp = tempfile::tempdir().unwrap();
        let branch = tmp.path().join("row01_apo_orthosteric_seed7101");
        // Only the tokenizer receipt is present — no pathb stage.
        touch(&branch.join("prismos_runtime_tokenizer_export_receipt.json"));
        assert!(is_modality_branch(&branch));
        assert!(!is_binding_site_branch(&branch));
        assert_eq!(discover_modality_branches(tmp.path()).len(), 1);
        assert_eq!(discover_binding_site_branches(tmp.path()).len(), 0);
    }

    #[test]
    fn ignores_directories_without_markers() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("logs")).unwrap();
        fs::create_dir_all(tmp.path().join("p2rank")).unwrap();
        touch(&tmp.path().join("logs").join("run.json"));
        assert_eq!(discover_binding_site_branches(tmp.path()).len(), 0);
        assert_eq!(discover_modality_branches(tmp.path()).len(), 0);
    }

    #[test]
    fn tolerates_one_level_of_grouping_nesting() {
        // Branches grouped one directory deeper than usual.
        let tmp = tempfile::tempdir().unwrap();
        let group = tmp.path().join("BLACKWELL_MATRIX_20260702");
        for branch in [
            "row01_apo_orthosteric_seed7101",
            "row02_apo_cryptic_seed7102",
        ] {
            touch(
                &group
                    .join(branch)
                    .join("pathb_runtime_stage")
                    .join("binding_sites.materialized.json"),
            );
        }
        let branches = discover_binding_site_branches(tmp.path());
        assert_eq!(branches.len(), 2);
    }

    #[test]
    fn single_run_root_is_itself_a_branch() {
        // No per-branch subdirectory: the marker lives directly under the root.
        let tmp = tempfile::tempdir().unwrap();
        touch(
            &tmp.path()
                .join("pathb_runtime_stage")
                .join("binding_sites.materialized.json"),
        );
        let branches = discover_binding_site_branches(tmp.path());
        assert_eq!(branches.len(), 1);
        assert_eq!(branches[0].path, tmp.path());
    }

    #[test]
    fn installed_launcher_output_root_is_itself_a_branch() {
        let tmp = tempfile::tempdir().unwrap();
        let direct = tmp.path().join("binding_sites.materialized.json");
        touch(&direct);

        let branches = discover_binding_site_branches(tmp.path());
        assert_eq!(branches.len(), 1);
        assert_eq!(branches[0].path, tmp.path());
        assert_eq!(branches[0].binding_sites_materialized(), direct);
        assert!(is_modality_branch(tmp.path()));
    }

    #[test]
    fn direct_installed_result_precedes_retained_nested_result() {
        let tmp = tempfile::tempdir().unwrap();
        let direct = tmp.path().join("binding_sites.materialized.json");
        touch(&direct);
        touch(
            &tmp.path()
                .join("pathb_runtime_stage")
                .join("binding_sites.materialized.json"),
        );

        let branches = discover_binding_site_branches(tmp.path());
        assert_eq!(branches.len(), 1);
        assert_eq!(branches[0].binding_sites_materialized(), direct);
    }
}


#[cfg(test)]
mod native_publication_tests {
    use super::*;
    use serde_json::{json, Value};
    fn fixture(root: &Path) -> (PathBuf, Value) {
        // Source-format fixture only; never scientific/native execution evidence.
        let native = json!({"schema_version":2,"structure":"unit_fixture","binding_sites":2,
            "binding_site_identity_domain":"ccl_cluster","sites":[
                {"id":4003,"rank":2,"cryptic_rank":1,"centroid":[1.0,2.0,3.0],"quality_score":0.3,"druggability":null,"lining_residues":[]},
                {"id":4091,"rank":1,"cryptic_rank":2,"centroid":[4.0,5.0,6.0],"quality_score":0.2,"druggability":0.4,"lining_residues":[]}]});
        for ext in ["pdb","pml","cxc","md"] {
            std::fs::write(root.join(format!("fixture.binding_sites.{ext}")),b"unit fixture only\n").unwrap();
        }
        let path=root.join("fixture.binding_sites.publication.json");
        write_pair(root,&native,&path);
        (path,native)
    }
    fn write_pair(root: &Path, native: &Value, publication: &Path) {
        let source=root.join("fixture.binding_sites.json");
        let bytes=serde_json::to_vec_pretty(native).unwrap();std::fs::write(&source,&bytes).unwrap();
        let files:Vec<Value>=["pdb","pml","cxc","md"].iter().map(|ext| {
            let name=format!("fixture.binding_sites.{ext}");let p=root.join(&name);
            json!({"file":name,"bytes":p.metadata().unwrap().len(),"sha256":publication_sha256(&p).unwrap()})
        }).collect();
        let p=json!({"schema":"prism.canonical_binding_site_publication.v1","source_json_sha256":publication_bytes_sha256(&bytes).unwrap(),
            "site_count":native["sites"].as_array().unwrap().len(),"site_identity_domain":native["binding_site_identity_domain"],
            "order":"final native JSON sites array; no new scientific rank","sites":native["sites"],"files":files});
        std::fs::write(publication,serde_json::to_vec(&p).unwrap()).unwrap();
    }
    #[test]
    fn native_order_values_and_separate_rank_fields_are_preserved() {
        let root=tempfile::tempdir().unwrap();let(p,native)=fixture(root.path());
        let result=load_native_binding_publication(&p).unwrap();
        assert_eq!(result.sites,native["sites"].as_array().unwrap().clone());
        assert_eq!(result.sites[0]["id"],4003);assert_eq!(result.sites[0]["rank"],2);
        assert_eq!(result.sites[0]["cryptic_rank"],1);assert!(result.sites[0]["druggability"].is_null());
        assert!(is_binding_site_branch(root.path()));assert_eq!(native_binding_site_publications(root.path()),vec![p]);
    }
    #[test]
    fn source_hash_count_identity_rank_geometry_and_visual_drift_are_refused() {
        for defect in ["source_hash","count","identity","rank","geometry","visual"] {
            let root=tempfile::tempdir().unwrap();let(p,mut native)=fixture(root.path());
            match defect {
                "source_hash"=>{std::fs::write(root.path().join("fixture.binding_sites.json"),b"{}\n").unwrap();},
                "count"=>{native["binding_sites"]=json!(99);write_pair(root.path(),&native,&p);},
                "identity"=>{native["sites"][1]["id"]=native["sites"][0]["id"].clone();write_pair(root.path(),&native,&p);},
                "rank"=>{native["sites"][1]["cryptic_rank"]=json!(1);write_pair(root.path(),&native,&p);},
                "geometry"=>{native["sites"][0]["centroid"]=json!([null,2,3]);write_pair(root.path(),&native,&p);},
                "visual"=>{std::fs::write(root.path().join("fixture.binding_sites.pml"),b"changed").unwrap();},
                _=>unreachable!(),
            }
            assert!(load_native_binding_publication(&p).is_err(),"{defect}");
        }
    }
}
