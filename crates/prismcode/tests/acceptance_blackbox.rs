use prismcode::acceptance::AcceptancePhase;

#[test]
fn acceptance_phase_rejects_unknown_modes() {
    assert!(AcceptancePhase::parse("release-security").is_ok());
    assert!(AcceptancePhase::parse("full").is_ok());
    assert!(AcceptancePhase::parse("demo").is_err());
}
