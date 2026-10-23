//! Regression: the post-MD scientific stack is reachable, and its bypass is
//! an explicit operator-visible mode rather than an invisible hard-return.
//!
//! # What was broken
//!
//! `nhs_rt_full` contained an UNCONDITIONAL `return Ok(())` after the V2
//! teardown whenever `--md-only-evidence` was off. Because V2 is live on
//! essentially every governed run (GPU primacy), that return preempted the
//! entire post-MD scientific stack — spatial-hash clustering, LIGSITE,
//! persistent homology, XGBoost rerank, PRISM-Therm — on every run since
//! 2026-05-21, while `--prism-therm` was still being accepted on the command
//! line and silently doing nothing. The run reported success. Nothing in the
//! output said the science had been skipped.
//!
//! # The contract this pins
//!
//! 1. The bypass still EXISTS and is reachable — it is deprecated, not denied.
//!    `--post-md-aggregation=skip` reproduces the historical behaviour verbatim,
//!    which is required to replay pre-repair runs and to A/B against them.
//! 2. The DEFAULT is `full`, so the science runs unless an operator explicitly
//!    asks for the bypass.
//! 3. The two modes are distinguishable — a single-variant enum, or a default
//!    of `skip`, would silently restore the defect.
//!
//! This suite asserts the source-level contract. It deliberately does NOT
//! assert runtime behaviour: proving the post-MD stack actually executes needs
//! a GPU, a real topology and a full MD run, which is a production validation,
//! not a unit test. Claiming otherwise here would be the same class of error
//! the bypass itself represented.

use std::fs;
use std::path::PathBuf;

fn bin_source() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("bin")
        .join("nhs_rt_full.rs");
    fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

#[test]
fn post_md_aggregation_enum_has_both_modes() {
    let src = bin_source();
    let decl = src
        .split("enum PostMdAggregationPolicy {")
        .nth(1)
        .expect("PostMdAggregation enum is absent — the bypass is unconditional again")
        .split('}')
        .next()
        .expect("unterminated PostMdAggregation enum");
    assert!(
        decl.contains("Full"),
        "PostMdAggregation lost its Full variant, so the post-MD stack is \
         unreachable again: {decl}"
    );
    assert!(
        decl.contains("Skip"),
        "PostMdAggregation lost its Skip variant, converting a deprecated \
         capability into a deny: {decl}"
    );
}

#[test]
fn default_is_full_not_skip() {
    let src = bin_source();
    let idx = src
        .find("post_md_aggregation: PostMdAggregation")
        .expect("post_md_aggregation arg is absent");
    // The #[arg(...)] attribute sits immediately above the field.
    let window = &src[idx.saturating_sub(200)..idx];
    assert!(
        window.contains("default_value = \"full\""),
        "post_md_aggregation no longer defaults to full; the post-MD science \
         would be skipped unless explicitly requested. Attribute window: {window}"
    );
}

#[test]
fn full_continuation_retains_the_md_evidence_manifest() {
    let src = bin_source();
    let handoff = src.split("let multi_post_md_route = resolve_post_md_route(")
        .nth(1).expect("generic multi route resolution");
    let handoff = handoff.split("// ── Extract ASC consensus").next().unwrap();
    assert!(handoff.contains("if args.md_only_evidence"));
    assert!(handoff.contains("|| args.require_prismcode_md_native"));
    assert!(handoff.contains("|| v2_was_live_for_lattice"));
    let manifest = handoff.find("write_json_pretty_atomic(&manifest_path, &manifest)")
        .expect("durable terminal MD evidence manifest is still emitted");
    let conditional_exit = handoff.rfind(
        "if multi_post_md_route.effective == EffectivePostMdAggregation::Skip {")
        .expect("only resolved Skip returns after the handoff");
    assert!(manifest < conditional_exit);
    assert!(handoff[conditional_exit..].contains("return Ok(());"));
    assert!(handoff[conditional_exit..].contains("continuing native Full post-MD analysis"));
    assert!(handoff.contains("terminal_closeout_controls"));
    assert!(handoff.contains("pending_native_full_post_md"));
}

#[test]
fn evidence_shortfall_does_not_unconditionally_veto() {
    let src = bin_source();
    let branch = src
        .split("if !evidence_artifacts_accepted {")
        .nth(1)
        .expect("evidence-shortfall branch not found");
    let head = &branch[..branch.len().min(400)];
    assert!(
        head.contains("match args.post_md_aggregation"),
        "an evidence bookkeeping shortfall again vetoes the run unconditionally, \
         discarding sound physics: {head}"
    );
    assert!(
        head.contains("anyhow::bail!"),
        "the Skip arm must still fail hard — the strict path was not preserved: {head}"
    );
}

#[test]
fn skip_mode_is_still_reachable_from_the_cli() {
    // Guard against "fixing" the defect by deleting the deprecated mode.
    let src = bin_source();
    assert!(
        src.contains("post_md_aggregation: PostMdAggregation"),
        "the CLI arg was removed; the bypass would become unreachable rather \
         than deprecated"
    );
}
