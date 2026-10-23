#[test]
fn escaped_physical_replay_retires_observed_generation() {
    let domain=PhysicalGraphLaunchDomain::default();
    let observed=Some(*domain.lease().unwrap());
    {let mut launch=domain.lease().unwrap();PhysicalGraphLaunchDomain::record_launch(&mut launch).unwrap();}
    assert!(PhysicalGraphLaunchDomain::verify(*domain.lease().unwrap(),observed).is_err());
}
#[test]
fn endpoint_publication_lease_excludes_graph_launch() {
    let domain=PhysicalGraphLaunchDomain::default();
    let endpoint=domain.lease().unwrap();
    assert!(domain.lease().is_err());
    drop(endpoint);
    assert!(domain.lease().is_ok());
}
#[test]
fn uncertain_launch_generation_never_certifies_prior_clock() {
    let domain=PhysicalGraphLaunchDomain::default();
    let before=Some(*domain.lease().unwrap());
    {let mut launch=domain.lease().unwrap();PhysicalGraphLaunchDomain::record_launch(&mut launch).unwrap();}
    // CUDA failure after issue still advanced host generation; no native claim.
    assert!(PhysicalGraphLaunchDomain::verify(*domain.lease().unwrap(),before).is_err());
}
#[test]
fn launch_generation_overflow_is_refused() {
    let mut generation=u64::MAX;
    assert!(PhysicalGraphLaunchDomain::record_launch(&mut generation).is_err());
    assert_eq!(generation,u64::MAX);
}
#[test]
fn independent_owners_have_independent_leases() {
    let first=PhysicalGraphLaunchDomain::default();let second=PhysicalGraphLaunchDomain::default();
    let held=first.lease().unwrap();assert!(second.lease().is_ok());drop(held);
}
