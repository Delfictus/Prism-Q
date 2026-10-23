#!/usr/bin/env bash
#
# M1 lane atomic-or grep gate.
#
# Enforces blueprint §M2 / M1 contract §3.2 HALT predicate at the
# source level: a CUDA kernel performing cluster-survivor election
# using `atomicOr` MUST NOT use a tie-breaker that depends on
# z-coordinate, geometry-jitter, or any spatial / floating-point
# quantity. Authorized tie-breakers are identity tuples only:
# (chain_id, residue_id, atom_index) per §M6.
#
# This script greps the M1 lane's .cu / .cuh source files for the
# co-occurrence pattern and exits non-zero if any forbidden pattern
# is found. Per the M1 contract C2, the gate must:
#   (a) FIRE on a synthetic test case that introduces the pattern, AND
#   (b) NOT fire on the actual M1 kernel source.
# Both directions are exercised by the unit tests in
# `crates/prism-nhs/src/spike_to_cluster_4d.rs`.
#
# Usage:
#   scripts/m1_atomic_or_grep.sh                 — gate the live source
#   scripts/m1_atomic_or_grep.sh <file.cu>...    — gate specified files
#
# Exit codes:
#   0  — no forbidden patterns found
#   1  — at least one forbidden pattern found (HALT)
#   2  — invocation error (file not found, etc.)

set -euo pipefail

# Default scope: M1 lane CUDA sources under prism-nhs.
DEFAULT_SCOPE=(
    "crates/prism-nhs/src/cuda/spike_to_cluster_4d.cu"
    "crates/prism-nhs/src/cuda/spike_to_cluster_4d.cuh"
)

# Forbidden tie-breaker symbols co-occurring with atomicOr.
# These are the geometric / floating-point quantities §M2 forbids.
FORBIDDEN_SYMBOLS='(\bz\b|coord|position|jitter|distance|centroid)'

if [[ $# -gt 0 ]]; then
    targets=("$@")
else
    targets=("${DEFAULT_SCOPE[@]}")
fi

# Verify each target exists. A missing file is invocation error
# (exit 2), not a clean pass — this prevents the gate from silently
# accepting a target that has been moved/renamed.
for f in "${targets[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "[m1-atomic-or-gate] ERROR: target not found: $f" >&2
        exit 2
    fi
done

violations=0

for f in "${targets[@]}"; do
    # An "atomicOr" line, OR a line within the next 5 lines after
    # an atomicOr line, that mentions one of the forbidden symbols
    # is a violation. Use grep --max-count=0 to count rather than
    # short-circuit.
    #
    # `awk` is used here because the co-occurrence is a small-window
    # search that grep's -A flag does not directly express as a
    # boolean predicate.
    while IFS= read -r line; do
        ((violations += 1))
        echo "[m1-atomic-or-gate] VIOLATION in $f: $line" >&2
    done < <(
        awk -v sym="$FORBIDDEN_SYMBOLS" '
            BEGIN { window = 0 }
            /atomicOr/ { window = 5; print FILENAME ":" NR ": " $0; next }
            window > 0 {
                if ($0 ~ sym) { print FILENAME ":" NR ": " $0 }
                window--
            }
        ' "$f"
    )
done

if [[ $violations -gt 0 ]]; then
    echo "[m1-atomic-or-gate] HALT: $violations forbidden atomicOr-co-occurrence pattern(s) found." >&2
    echo "[m1-atomic-or-gate] Per blueprint §M2 / M1 contract §3.2: authorized tie-breakers are" >&2
    echo "[m1-atomic-or-gate] identity tuples (chain_id, residue_id, atom_index) only. Spatial / FP" >&2
    echo "[m1-atomic-or-gate] tie-breakers are forbidden." >&2
    exit 1
fi

echo "[m1-atomic-or-gate] PASS: no forbidden atomicOr-co-occurrence patterns found in:"
for f in "${targets[@]}"; do
    echo "  $f"
done
exit 0
