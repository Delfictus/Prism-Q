use prismcode::bundle_hash_tree::hash_bundle_tree;
use std::fs;
use std::os::unix::fs::symlink;
use std::path::PathBuf;

#[test]
fn bundle_hash_tree_rejects_symlink_escape() {
    let tmp_parent = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let tmp = tempfile::tempdir_in(tmp_parent).unwrap();
    let bundle = tmp.path().join("bundle");
    fs::create_dir_all(bundle.join("release")).unwrap();
    symlink("/etc/passwd", bundle.join("release/escape")).unwrap();
    assert!(hash_bundle_tree(&bundle).is_err());
}
