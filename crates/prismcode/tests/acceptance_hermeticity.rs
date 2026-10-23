use prismcode::acceptance_hermeticity::{begin, finish};
use std::fs;

#[test]
fn hermeticity_detects_primary_bundle_unchanged() {
    let tmp = tempfile::tempdir().unwrap();
    let bundle = tmp.path().join("bundle");
    fs::create_dir_all(bundle.join("bin")).unwrap();
    fs::create_dir_all(bundle.join("release")).unwrap();
    fs::create_dir_all(bundle.join("firmware")).unwrap();
    fs::write(bundle.join("bin/prism-acceptance"), b"binary").unwrap();
    fs::write(bundle.join("release/release_manifest.json"), b"{}").unwrap();
    let run = begin(&bundle).unwrap();
    let report = finish(&run, true).unwrap();
    assert_eq!(report["primary_bundle_unchanged"], true);
    assert_eq!(report["status"], "PASS");
}

#[test]
fn hermeticity_rejects_source_tree_as_bundle() {
    let tmp = tempfile::tempdir().unwrap();
    fs::write(tmp.path().join("Cargo.toml"), b"[workspace]\n").unwrap();
    assert!(begin(tmp.path()).is_err());
}
