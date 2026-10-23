use prismcode::bundle_copy::copy_bundle_for_tamper;
use prismcode::bundle_hash_tree::hash_bundle_tree;
use std::fs;
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;

#[test]
fn tamper_copy_does_not_mutate_primary_bundle() {
    let tmp = tempfile::tempdir().unwrap();
    let primary = tmp.path().join("primary");
    let copy = tmp.path().join("copy");
    fs::create_dir_all(primary.join("release")).unwrap();
    fs::write(
        primary.join("release/release_manifest.json"),
        b"{\"status\":\"PASS\"}",
    )
    .unwrap();
    let before = hash_bundle_tree(&primary).unwrap().merkle_root;
    copy_bundle_for_tamper(&primary, &copy).unwrap();
    fs::write(
        copy.join("release/release_manifest.json"),
        b"{\"status\":\"TAMPERED\"}",
    )
    .unwrap();
    let after = hash_bundle_tree(&primary).unwrap().merkle_root;
    assert_eq!(before, after);
}

#[test]
fn tamper_copy_hardlinks_model_artifacts_only() {
    let tmp = tempfile::tempdir().unwrap();
    let primary = tmp.path().join("primary");
    let copy = tmp.path().join("copy");
    fs::create_dir_all(primary.join("release")).unwrap();
    fs::create_dir_all(primary.join("models/openai_gpt_oss_20b")).unwrap();
    fs::write(
        primary.join("release/release_manifest.json"),
        b"{\"status\":\"PASS\"}",
    )
    .unwrap();
    fs::write(
        primary.join("models/openai_gpt_oss_20b/model.safetensors"),
        b"large-model-placeholder",
    )
    .unwrap();

    copy_bundle_for_tamper(&primary, &copy).unwrap();
    fs::write(
        copy.join("release/release_manifest.json"),
        b"{\"status\":\"TAMPERED\"}",
    )
    .unwrap();
    assert_eq!(
        fs::read(primary.join("release/release_manifest.json")).unwrap(),
        b"{\"status\":\"PASS\"}"
    );

    #[cfg(unix)]
    {
        let primary_model =
            fs::metadata(primary.join("models/openai_gpt_oss_20b/model.safetensors")).unwrap();
        let copy_model =
            fs::metadata(copy.join("models/openai_gpt_oss_20b/model.safetensors")).unwrap();
        assert_eq!(primary_model.ino(), copy_model.ino());
    }
}
