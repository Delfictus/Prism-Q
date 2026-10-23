# AGENTS.md: reviewer and contributor contract for Prism-Q

This file binds every AI agent (and human) that reviews or changes this repository.
`CLAUDE.md` imports it, and other agents read it directly.

## 1. Review contract (mandatory for every review)

1. **Read `REVIEW_BUNDLE.md` first.** It is the flattened, generated view of the engine in data-flow order:
   - verbatim source excerpts with `path:line`;
   - the mathematics each excerpt implements;
   - units, invariants and known defects with status;
   - the qualifying gate and the mandatory reviewer questions.
2. **Review the actual source, not descriptions of it.** Prose, comments, docs, commit messages and
   summaries (including this file and the README) are claims to be checked against the excerpts.
   Where they disagree, the source is what the engine does and the disagreement is a finding.
3. **Answer every mandatory question in every section.** Each answer cites `path:line` anchors at the
   commit named in the bundle header. Write "cannot determine from source" when that is true; do not
   infer it from docs.
4. **Re-run the tooling yourself** and report the exit codes:
   ```bash
   python3 scripts/build_review_bundle.py --check    # bundle is current and every anchor resolves
   python3 scripts/build_review_bundle.py --verify   # independent CPU reference checks
   ```
   Do not trust embedded results you have not reproduced.
5. **Report disagreements as findings.** Each finding has:
   - severity (critical / major / minor);
   - `path:line`;
   - the claim being contradicted (section id + field);
   - the expected value with a derivation;
   - the observed value;
   - a minimal reproduction or calculation.

   Disagreements with the bundle's equations, units, invariants, statuses or defect entries are findings too.
6. **Never upgrade a status in a review.** The only status words are:
   - `QUALIFIED`: passed its stated gate with a receipt;
   - `EXECUTED_UNQUALIFIED`: runs, but its gate has not passed;
   - `NOT_BUILT`: does not exist yet.

   A review can recommend a downgrade, with evidence.
7. **Known critical items to verify independently, not to take on trust:**
   - the DEC opposite-edge weight assignment (S08);
   - the Cayley gap compression (S09);
   - vacuum electrostatics (S02/S06);
   - the 4 fs resonance (S03);
   - identical UV kicks across streams (S04).

## 2. Engineering rules (for any change)

- **Change gate.** Any change to equations, parameters, numerical methods, estimators, CUDA code or
  compiler flags, provider mappings, state continuity, scientific tests or acceptance rules needs all of:
  - a written classification of the change;
  - the invariant it preserves or changes;
  - the gate that will qualify it;
  - a reference check, added or updated under `scripts/physics_refs/`.

  Physics-changing changes stop at their qualification result for review before promotion.
- **Regenerate the bundle in the same commit.** Every commit that touches a source listed in
  `review_bundle.toml` also commits the regenerated `REVIEW_BUNDLE.md`. When code moves, update the anchors.
  `--check` fails on stale bundles and broken anchors.
- **No proxies or substitutes.** Never substitute any of the following for the real path, and never
  present it as engine evidence:
  - proxy, surrogate, mock, toy or sample data;
  - wrapper-only fixes;
  - smoke tests;
  - Python re-implementations.

  Reference checks are independent references, never runtime replacements. Name proxies `*_proxy`.
- **No silent fallbacks.** Missing values are `null` with a reason, never `0` or `NaN`.
  - Missing sources render `NOT_BUILT`.
  - Unsupported combinations fail closed.
  - Deprecated capabilities stay reachable and are labelled deprecated.
- **Determinism and units.**
  - Physical windows are expressed in physical time, not in steps.
  - Every datum carries units and a clock key.
  - Until the determinism gate passes, on/off comparisons are `STATISTICAL_ONLY`.
- **Never commit:**
  - build outputs, vendored crates, or generated PTX/cubin;
  - run outputs, Chronicle data or large evidence (record hashes in `manifests/` instead);
  - credentials, tokens or keys;
  - host-specific absolute paths, usernames or private network addresses.

  Scripts take their roots from arguments or configuration.
- **Third-party material.** Every vendored or copied file (force-field parameters, structures, fonts,
  models, copied code) is listed in `manifests/THIRD_PARTY.json` with its license. Anything that is not
  redistributable is excluded.
- **Completion language.** Report work only as one of:
  - `IMPLEMENTATION_VERIFIED`: files written and parsed, verification commands passed;
  - `IMPLEMENTATION_PARTIAL`: a named step remains;
  - `IMPLEMENTATION_BLOCKED`: required evidence or permission is missing, and the blocker is named.

  A successful build is not runtime validation, and a smoke test is not acceptance.
- **Claim scope.** Computational evidence only. Never claim wet-lab, clinical, regulatory, synthesis,
  safety or ADMET validation. Never claim "DFT-accurate" without DFT references. The electronic layer is
  `analogue_uncalibrated` until its grounding gates pass.
