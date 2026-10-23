#!/usr/bin/env python3
"""Bounded one-pivot FP32 norm completion, independent of the CUDA search.

This selects a represented endpoint; it never rescales a spinor.  For each of
four pivot components, the other three take their two original FP32 brackets.
The pivot ranges over all finite FP32 magnitudes with its original sign.  The
source-order FP64 norm is monotone on this ordered magnitude lattice.  Binary
search therefore finds the closest admissible pivot in at most 31 iterations
for each of 32 slices.  Completeness is only for these slices, not the entire
four-dimensional lattice or the nonlinear equation/work/energy constraints.

All candidate decisions use the actual norm subtraction predicates.  No sqrt
or rounded analytical interval is substituted at an acceptance boundary.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from check_biprism_norm_refusal import TOL, bits, bracket, distance, f32, norm

MAX_MAGNITUDE_BITS = 0x7F7FFFFF


def magnitude_from_bits(value: int) -> float:
    if not 0 <= value <= MAX_MAGNITUDE_BITS:
        raise ValueError("finite nonnegative FP32 magnitude index required")
    return struct.unpack("<f", struct.pack("<I", value))[0]


def norm_class(value: float, targets: tuple[float, ...]) -> tuple[bool, bool]:
    """Below/above the intersection under source-order subtraction/rounding."""
    deltas = tuple(value - target for target in targets)
    return any(d < -TOL for d in deltas), any(d > TOL for d in deltas)


@dataclass(frozen=True)
class SliceResult:
    candidate: tuple[float, float, float, float] | None
    magnitude_bits: int
    iterations: int
    direction: str


def closest_in_slice(trial, fixed, pivot, targets) -> SliceResult:
    """Nearest pivot on one fixed-three-component slice, or exact refusal."""
    if not 0 <= pivot < 4:
        raise ValueError("pivot must be 0..3")
    if not all(math.isfinite(x) for x in tuple(trial) + tuple(fixed) + tuple(targets)):
        raise ValueError("finite inputs required")
    negative = math.copysign(1.0, trial[pivot]) < 0.0

    def candidate(index):
        values = list(fixed)
        x = magnitude_from_bits(index)
        values[pivot] = -x if negative else x
        return tuple(values)

    try:
        nearest = f32(abs(trial[pivot]))
    except OverflowError:
        return SliceResult(None, MAX_MAGNITUDE_BITS, 0, "input_unrepresentable")
    if not math.isfinite(nearest):
        return SliceResult(None, MAX_MAGNITUDE_BITS, 0, "input_unrepresentable")
    index = bits(nearest)
    below, above = norm_class(norm(candidate(index)), targets)
    count = 0
    direction = "nearest"
    if below:
        direction = "up"
        left, right = index, MAX_MAGNITUDE_BITS
        while left < right:
            middle = left + (right - left) // 2
            low, _ = norm_class(norm(candidate(middle)), targets)
            if low:
                left = middle + 1
            else:
                right = middle
            count += 1
        index = left
    elif above:
        direction = "down"
        left, right = 0, index
        while left < right:
            middle = left + (right - left + 1) // 2
            _, high = norm_class(norm(candidate(middle)), targets)
            if high:
                right = middle - 1
            else:
                left = middle
            count += 1
        index = left
    assert count <= 31
    result = candidate(index)
    q = norm(result)
    admitted = math.isfinite(q) and all(abs(q - t) <= TOL for t in targets)
    return SliceResult(result if admitted else None, index, count, direction)


def complete_norm_lattice(trial, norm_n, norm_outer, *, absolute_one=True):
    """Return the nearest admitted member of 32 one-pivot slices and metadata.

    Callers separately refuse invalid physical input norms.  Including the
    absolute-one target here qualifies candidates; it does not repair or admit
    an invalid input state.  Native nonlinear equation/work gates remain required.
    """
    trial = tuple(trial)
    if len(trial) != 4 or not all(math.isfinite(x) for x in trial + (norm_n, norm_outer)):
        raise ValueError("four finite components and finite norm targets required")
    targets = (norm_n, norm_outer, 1.0) if absolute_one else (norm_n, norm_outer)
    try:
        bounds = [bracket(x) for x in trial]
    except OverflowError:
        return None, math.inf, ()
    if not all(math.isfinite(x) for pair in bounds for x in pair):
        return None, math.inf, ()
    best, best_error = None, math.inf
    slices = []
    for pivot in range(4):
        others = [j for j in range(4) if j != pivot]
        for mask in range(8):
            fixed = list(trial)
            for bit, j in enumerate(others):
                fixed[j] = bounds[j][(mask >> bit) & 1]
            result = closest_in_slice(trial, fixed, pivot, targets)
            slices.append((pivot, mask, result))
            if result.candidate is not None:
                error = distance(result.candidate, trial)
                if error < best_error:
                    best, best_error = result.candidate, error
    return best, best_error, tuple(slices)
