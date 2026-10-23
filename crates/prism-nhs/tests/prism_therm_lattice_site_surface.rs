//! Regression: PRISM-Therm samples the GhostPhaseLattice4D site surface.
//!
//! The claim under test: lattice component selection drops under-supported
//! components, returns the survivors largest-first, and breaks ties by
//! ascending component index so the run is reproducible.
//!
//! # Why this matters
//!
//! The CCL consensus-cluster producer percolates on dense spike fields. A 6G6K
//! calibration run collapsed 712,856 spikes to `clusters=1`; the downstream
//! k-means / dual-centroid / consensus recovery stages then emitted 8 centroids
//! spanning 0.7 Å in total. Every SDST region built from those was empty, so
//! hysteresis AND CCNS returned identically zero for every site regardless of
//! run length — a physics-shaped null that was really an empty-region artifact.
//! The lattice, computed in the same run and previously discarded, resolved
//! 4,831 components across a 45×41×82 Å envelope.
//!
//! # Why this test has teeth
//!
//! Asserting "the selector agrees with itself" is worthless if the fixture is
//! already sorted and already above threshold — every implementation, including
//! a no-op, would pass. So each phase is guarded:
//!
//!   1. `fixture_is_adversarial` PROVES the fixture is NOT already in the
//!      required order and DOES contain components below threshold. If that
//!      ever stops holding, the rest of the file is vacuous and this test fails
//!      loudly rather than passing quietly.
//!   2. Only then are the filter, the ordering, and the tie-break asserted.
//!   3. A no-op counter-example (identity ordering) is asserted to DIFFER from
//!      the selector's output on the same fixture, so a regression that
//!      silently reverts to "return everything in input order" is caught.

use prism_nhs::sdst_bridge::select_lattice_components;

/// Node counts chosen so that the correct answer is NOT the input order and
/// NOT the full set: index 3 is below any sane threshold, and the largest
/// component sits in the middle rather than first.
const FIXTURE: [u32; 7] = [12, 40, 9, 1, 40, 8, 3];
const MIN_NODES: u32 = 8;

#[test]
fn fixture_is_adversarial() {
    // Guard 1: the fixture must contain components that the filter has to drop.
    let below: Vec<u32> = FIXTURE.iter().copied().filter(|n| *n < MIN_NODES).collect();
    assert!(
        !below.is_empty(),
        "vacuity guard: fixture has nothing below min_nodes={MIN_NODES}, so the \
         filter assertion would pass for any implementation. Fixture: {FIXTURE:?}"
    );

    // Guard 2: the fixture must not already be in descending order, or the
    // ordering assertion would pass for a no-op that preserves input order.
    let already_sorted = FIXTURE.windows(2).all(|w| w[0] >= w[1]);
    assert!(
        !already_sorted,
        "vacuity guard: fixture is already descending, so the ordering \
         assertion is vacuous. Fixture: {FIXTURE:?}"
    );

    // Guard 3: the fixture must contain a tie, or the tie-break is untested.
    let mut counts = FIXTURE.to_vec();
    counts.sort_unstable();
    let has_tie = counts.windows(2).any(|w| w[0] == w[1]);
    assert!(
        has_tie,
        "vacuity guard: fixture has no duplicate node count, so the tie-break \
         assertion is vacuous. Fixture: {FIXTURE:?}"
    );
}

#[test]
fn drops_components_below_min_nodes() {
    let order = select_lattice_components(&FIXTURE, MIN_NODES);
    for &i in &order {
        assert!(
            FIXTURE[i] >= MIN_NODES,
            "component {i} has {} nodes, below min_nodes={MIN_NODES}",
            FIXTURE[i]
        );
    }
    // indices 3 (1 node) and 6 (3 nodes) must be gone; 5 (8 nodes) must remain
    // because the filter is >=, not >.
    assert!(!order.contains(&3), "1-node singleton survived: {order:?}");
    assert!(!order.contains(&6), "3-node component survived: {order:?}");
    assert!(
        order.contains(&5),
        "8-node component was dropped, but min_nodes is inclusive: {order:?}"
    );
    assert_eq!(order.len(), 5, "expected 5 survivors, got {order:?}");
}

#[test]
fn returns_survivors_largest_first() {
    let order = select_lattice_components(&FIXTURE, MIN_NODES);
    let counts: Vec<u32> = order.iter().map(|&i| FIXTURE[i]).collect();
    assert!(
        counts.windows(2).all(|w| w[0] >= w[1]),
        "not largest-first: {counts:?} (order {order:?})"
    );
}

#[test]
fn ties_break_by_ascending_index() {
    let order = select_lattice_components(&FIXTURE, MIN_NODES);
    // 1 and 4 both carry 40 nodes and are the two largest; 1 must precede 4.
    let p1 = order.iter().position(|&i| i == 1).expect("index 1 kept");
    let p4 = order.iter().position(|&i| i == 4).expect("index 4 kept");
    assert!(
        p1 < p4,
        "tie between equal 40-node components 1 and 4 did not break by \
         ascending index: {order:?}"
    );
    assert_eq!(
        order,
        vec![1, 4, 0, 2, 5],
        "full expected selection changed: {order:?}"
    );
}

#[test]
fn selector_differs_from_input_order_noop() {
    // Counter-example: a regression that reverts to "keep everything, input
    // order" must not be able to pass the assertions above.
    let identity: Vec<usize> = (0..FIXTURE.len()).collect();
    let order = select_lattice_components(&FIXTURE, MIN_NODES);
    assert_ne!(
        order, identity,
        "selector returned the input order unchanged, which is the exact \
         pre-repair behaviour this regression exists to catch"
    );
}

#[test]
fn empty_lattice_selects_nothing() {
    // An explicit lattice caller records the empty selection and fails with a
    // typed unavailable surface. It never substitutes the CCL surface.
    assert!(select_lattice_components(&[], MIN_NODES).is_empty());
    assert!(select_lattice_components(&[1, 2, 3], MIN_NODES).is_empty());
}
