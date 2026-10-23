// Exercises the actual owner clock state, including failure poisoning.
#[test]
fn six_warmup_then_repeated_exact_five_hundred_windows() {
    let mut clock = AutonomousChunkClock::default();
    assert!(clock.endpoint().is_err());
    clock.observe(6);
    for (start,end) in [(6,506),(506,1006),(1006,1506)] {
        assert_eq!(clock.begin(500,500),Ok((start,end)));
        assert!(clock.endpoint().is_err());
        assert_eq!(clock.complete(end,end),Ok(()));
        assert_eq!(clock.endpoint(),Ok(end));
    }
}
#[test]
fn exact_one_and_irregular_tail_do_not_round_up() {
    for steps in [1,37,499] {
        let mut clock=AutonomousChunkClock::default();clock.observe(506);
        assert_eq!(clock.begin(steps,steps),Ok((506,506+steps)));
        assert_eq!(clock.complete(506+steps,506+steps),Ok(()));
    }
}
#[test]
fn empty_unobserved_and_multiple_launch_windows_refuse() {
    let mut clock=AutonomousChunkClock::default();
    assert!(clock.begin(500,500).is_err());
    clock.observe(6);
    for (steps,body) in [(0,0),(0,500),(499,500),(1000,500)] {
        assert!(clock.begin(steps,body).is_err());
        assert_eq!(clock.endpoint(),Ok(6));
    }
}
#[test]
fn failed_or_incomplete_launch_cannot_supply_an_endpoint() {
    let mut clock=AutonomousChunkClock::default();clock.observe(6);
    assert_eq!(clock.begin(500,500),Ok((6,506)));
    // No completion call: launch/fence/readback failures leave no authority.
    assert!(clock.endpoint().is_err());
    assert!(clock.begin(500,500).is_err());
}
#[test]
fn off_by_one_latch_and_stale_endpoint_poison_authority() {
    for observed in [6,505,507,1006] {
        let mut clock=AutonomousChunkClock::default();clock.observe(6);
        assert!(clock.begin(500,500).is_ok());
        assert!(clock.complete(506,observed).is_err());
        assert!(clock.endpoint().is_err());
        assert!(clock.begin(500,500).is_err());
    }
}
#[test]
fn reset_or_mutable_exposure_requires_a_new_observation() {
    let mut clock=AutonomousChunkClock::default();clock.observe(506);
    clock.invalidate();
    assert!(clock.endpoint().is_err());
    assert!(clock.begin(500,500).is_err());
    clock.observe(6);
    assert_eq!(clock.begin(500,500),Ok((6,506)));
}
#[test]
fn clock_overflow_is_refused_before_launch() {
    let mut clock=AutonomousChunkClock::default();clock.observe(u32::MAX-1);
    assert!(clock.begin(500,500).is_err());
    assert!(clock.endpoint().is_err());
}
