use prismcode::hermetic_process::HermeticCommand;
use std::fs;
use std::os::unix::fs::PermissionsExt;

#[test]
fn hermetic_process_uses_absolute_bundled_binary_and_scrubs_env() {
    let tmp = tempfile::tempdir().unwrap();
    let bundle = tmp.path().join("bundle");
    let bin_dir = bundle.join("bin");
    let acc_tmp = tmp.path().join("acc_tmp");
    fs::create_dir_all(&bin_dir).unwrap();
    fs::create_dir_all(&acc_tmp).unwrap();
    let script = bin_dir.join("prismseal");
    fs::write(
        &script,
        b"#!/bin/sh\nif [ -n \"$LD_LIBRARY_PATH\" ]; then exit 7; fi\nif [ -n \"$PRISM_TRUST_ROOT\" ]; then exit 8; fi\nif [ \"$PRISM_HERMETIC\" = \"1\" ]; then exit 0; fi\nexit 9\n",
    )
    .unwrap();
    let mut perms = fs::metadata(&script).unwrap().permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&script, perms).unwrap();
    std::env::set_var("PATH", "/tmp/malicious");
    std::env::set_var("LD_LIBRARY_PATH", "/tmp/malicious");
    std::env::set_var("PRISM_TRUST_ROOT", "/tmp/fake.pem");
    let report = HermeticCommand::new(&bundle, &acc_tmp, "prismseal", &[])
        .unwrap()
        .run();
    std::env::remove_var("PATH");
    std::env::remove_var("LD_LIBRARY_PATH");
    std::env::remove_var("PRISM_TRUST_ROOT");
    assert_eq!(report["success"], true);
    assert_eq!(report["environment_scrubbed"], true);
    assert_eq!(report["path_inherited"], false);
}
