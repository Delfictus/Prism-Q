#!/usr/bin/env python3
"""Read RRF1/RRF2/RRF3 representation diagnostics; no trajectory admission.

The score is local to each leaf. Its global minimum is compared with the next
fully evaluated absolute error. All identities and recorded winning leaves are
checked before reporting that difference. RRF2 retains round zero and the latest
sixteen evaluated rounds. A missing interval is explicitly reported and never
treated as one update. RRF3 marks charge-changing equation repairs explicitly;
their NaN energy predictions are never compared as linear energy-rank scores.
No format establishes native trajectory acceptance.
"""
import argparse
import hashlib
import math
from pathlib import Path
import struct

LEAVES, STRIDE, OFFSET = 8, 38656, 13056
MAGIC_V1, MAGIC_V2, UINT_MAX = 0x52524631, 0x52524632, 0xffffffff
MAGIC_V3, PRECISE_STRIDE, REPAIR_BIT = 0x52524633, 40704, 0x80000000


def same_double(a, b):
    return struct.pack("<d", a) == struct.pack("<d", b)


def inspect(raw):
    if len(raw) not in (LEAVES * STRIDE, LEAVES * PRECISE_STRIDE, LEAVES * 43008, LEAVES * 66048, LEAVES * 70656, LEAVES * 72960, LEAVES * 97536, LEAVES * 106752, LEAVES * 143872, LEAVES * 196992, LEAVES * 198144, LEAVES * 235136, LEAVES * 349056, LEAVES * 496768, LEAVES * 510720, LEAVES * 1680768, LEAVES * 1697920, LEAVES * 1784448, LEAVES * 1837696, LEAVES * 1997440):
        raise ValueError("complete eight-bank numerical readback required")
    stride = len(raw)//LEAVES
    traces = []
    for leaf in range(LEAVES):
        off = leaf * stride + OFFSET
        header = struct.unpack_from("<4I", raw, off)
        error = struct.unpack_from("<17d", raw, off + 16)
        score = struct.unpack_from("<17d", raw, off + 152)
        winner = struct.unpack_from("<17I", raw, off + 288)
        magic, stage, count, terminal = header
        stages = range(4, 13) if magic == MAGIC_V3 else range(6, 13)
        if magic not in (MAGIC_V1, MAGIC_V2, MAGIC_V3) or not (1 <= count <= 17) or stage not in stages:
            raise ValueError(f"leaf {leaf}: missing evaluated RRF1/2/3 trace")
        if magic == MAGIC_V3 and stride not in (PRECISE_STRIDE,43008,66048,70656,72960,97536,106752,143872,196992,198144,235136,349056,496768,510720,1680768,1697920,1784448,1837696,1997440):
            raise ValueError(f"leaf {leaf}: RRF3 requires retained-potential numerical layout")
        # A failed H build records its attempted round before an error sample
        # exists. All other post-evaluation exits retain the evaluated round.
        last = terminal - 1 if stage == 6 else terminal
        if last < 0 or last == UINT_MAX or count != min(last + 1, 17):
            raise ValueError(f"leaf {leaf}: invalid terminal/evaluated round identity")
        if last > (16 if magic == MAGIC_V1 else (6164 if magic == MAGIC_V3 else 6144)):
            raise ValueError(f"leaf {leaf}: finite represented search bound exceeded")
        if magic == MAGIC_V1:
            if last + 1 != count:
                raise ValueError(f"leaf {leaf}: nonconsecutive original RRF1 trace")
            rounds = tuple(range(count)) + (UINT_MAX,) * (17 - count)
        else:
            rounds = struct.unpack_from("<17I", raw, off + 356)
            expected = {0} | set(range(max(1, last - 15), last + 1))
            recorded = [r for r in rounds if r != UINT_MAX]
            if len(recorded) != count or len(set(recorded)) != count or set(recorded) != expected:
                raise ValueError(f"leaf {leaf}: incomplete or stale RRF2 retained-round set")
            for slot, r in enumerate(rounds):
                if r != UINT_MAX and slot != (0 if r == 0 else 1 + ((r - 1) & 15)):
                    raise ValueError(f"leaf {leaf}: RRF2 round occupies the wrong ring slot")
        traces.append((header, error, score, winner, rounds))
    if any(t[0] != traces[0][0] for t in traces):
        raise ValueError("leaf trace headers disagree")
    if any(t[4] != traces[0][4] for t in traces):
        raise ValueError("leaf retained-round identities disagree")
    header = traces[0][0]
    slots = {r: slot for slot, r in enumerate(traces[0][4]) if r != UINT_MAX}
    ordered = sorted(slots)
    last = ordered[-1]
    version = {MAGIC_V1: "RRF1", MAGIC_V2: "RRF2", MAGIC_V3: "RRF3"}[header[0]]
    lines = [f"SCOPE=offline {version} search diagnostic; no native acceptance inferred",
             f"STAGE={header[1]} COUNT={header[2]} TERMINAL_ROUND={header[3]} LAST_EVALUATED_ROUND={last}",
             "round\tsigned_evaluated_error\tglobal_min_predicted_abs_error\twinner\tnext_abs_error_minus_prediction\tupdate_kind"]
    maximum = 0.
    monotone = True
    pairs = 0
    repairs = 0
    for r in ordered:
        slot = slots[r]
        error = traces[0][1][slot]
        winner = traces[0][3][slot]
        if not math.isfinite(error) or any(not same_double(t[1][slot], error) or t[3][slot] != winner for t in traces):
            raise ValueError(f"round {r}: global error/winner disagreement or nonfinite value")
        scores = [t[2][slot] for t in traces]
        repair = winner != UINT_MAX and bool(winner & REPAIR_BIT)
        if repair:
            selected = winner & ~REPAIR_BIT
            if header[0] != MAGIC_V3 or selected >= 8*192*4 or selected % 4 or not all(math.isnan(v) for v in scores):
                raise ValueError(f"round {r}: invalid equation-repair identity or fabricated energy prediction")
            if r == last and header[1] not in (4, 5, 6):
                raise ValueError(f"round {r}: repair missing its following full evaluation")
            best = math.nan
            repairs += 1
        elif all(math.isnan(v) for v in scores):
            # The full gate may refuse, or the strict target may accept, before
            # candidate ranking runs. Those NaNs are intentional reset values.
            if r != last or header[1] not in (7, 11, 12) or winner != UINT_MAX:
                raise ValueError(f"round {r}: missing score outside a terminal pre-ranking exit")
            best = math.nan
        else:
            if any(not math.isfinite(v) or v < 0 or v > abs(error) for v in scores):
                raise ValueError(f"round {r}: invalid local scores")
            best = min(scores)
            if winner != UINT_MAX:
                if not (0 <= winner < 8*192*4) or scores[winner//(192*4)] != best or best >= abs(error):
                    raise ValueError(f"round {r}: winner does not own a strictly improved minimum score")
        if winner == UINT_MAX and r < last:
            raise ValueError(f"round {r}: a missing winner cannot produce another round")
        delta = math.nan
        if r + 1 in slots and not repair:
            after = abs(traces[0][1][slots[r + 1]])
            delta = after - best
            maximum = max(maximum, abs(delta))
            monotone = monotone and after < abs(error)
            pairs += 1
        lines.append(f"{r}\t{error:.17g}\t{best:.17g}\t{winner}\t{delta:.17g}\t{'equation_repair' if repair else 'energy_rank_or_terminal'}")
    if repairs > 4:
        raise ValueError("retained equation-repair count exceeds the four-proposal bound")
    if header[1] in (4, 5) and not (traces[0][3][slots[last]] != UINT_MAX and traces[0][3][slots[last]] & REPAIR_BIT):
        raise ValueError("post-proposal DEC refusal has no recorded equation repair")
    gaps = [(a + 1, b - 1) for a, b in zip(ordered, ordered[1:]) if b > a + 1]
    lines.extend((f"COMPARED_ADJACENT_UPDATES={pairs}",
                  f"EQUATION_REPAIR_ROWS_RETAINED={repairs}",
                  f"UNRETAINED_ROUND_INTERVALS={gaps}",
                  f"MAX_ACTUAL_VS_PREDICTED_ERROR={maximum:.17g}" if pairs else "MAX_ACTUAL_VS_PREDICTED_ERROR=not_evaluated",
                  f"OBSERVED_ADJACENT_STRICTLY_IMPROVING={str(monotone).lower()}" if pairs else "OBSERVED_ADJACENT_STRICTLY_IMPROVING=not_evaluated",
                  "FULL_SEARCH_HISTORY_RETAINED=" + str(not gaps).lower(),
                  "NEW_ACCEPTANCE_TOLERANCE_DEFINED=false"))
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--numerical", type=Path, required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    data = args.numerical.read_bytes()
    text = (f"NUMERICAL_PATH={args.numerical.resolve()}\nNUMERICAL_SHA256={hashlib.sha256(data).hexdigest()}\n"
            f"CHECKER_SHA256={hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}\n" + inspect(data))
    if args.output:
        with args.output.open("x") as output:
            output.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
