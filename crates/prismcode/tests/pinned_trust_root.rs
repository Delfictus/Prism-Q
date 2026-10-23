use prismcode::trust_root;

#[test]
fn pinned_root_report_is_machine_readable() {
    let report = trust_root::report();
    assert!(report["pinned_root_hash"].as_str().is_some());
    assert_eq!(report["editable_config_root_used"], false);
}

#[test]
fn editable_env_root_is_rejected() {
    std::env::set_var("PRISM_TRUST_ROOT", "/tmp/fake.pem");
    let result = trust_root::reject_editable_root_inputs();
    std::env::remove_var("PRISM_TRUST_ROOT");
    assert!(result.is_err());
}
