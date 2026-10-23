"""Run every PRISM physics reference check.

    python3 -m prism_refs.run_all --json out.json [--engine-root PATH] [--only substr] [--timing]

Output is deterministic for a fixed engine tree, inputs and numpy version
(no wall-clock fields unless --timing). Exit code 0 unless a check raised or
an UNEXPECTED failure occurred (a failing check without a defect_id).
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import pathlib
import platform
import sys
import time
import traceback

MODULES = ("dec_weights", "electronic_propagator", "timestep_resonance", "units",
           "charge_conservation", "rng_streams", "neighbor_list")


def _load(name):
    if __package__:
        return importlib.import_module(f"{__package__}.{name}")
    p = pathlib.Path(__file__).with_name(f"{name}.py")
    s = importlib.util.spec_from_file_location(f"prism_refs_{name}", p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


def verdict(r):
    if r.get("pass") is True:
        return "PASS"
    if r.get("pass") is None:
        n = str(r.get("notes", ""))
        return n.split(":", 1)[0] if n.startswith(("NOT_BUILT", "INPUT_NOT_FOUND", "TRANSCRIPTION_STALE")) else "UNDETERMINED"
    return "KNOWN_DEFECT_OPEN" if r.get("defect_id") else "FAIL"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True)
    ap.add_argument("--engine-root", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--timing", action="store_true")
    a = ap.parse_args(argv)
    import numpy
    C = _load("_common")
    root = C.resolve_engine_root(a.engine_root)
    results, timing = [], {}
    for mod in MODULES:
        m = _load(mod)
        for fn in m.CHECKS:
            t0 = time.perf_counter()
            try:
                r = fn(engine_root=root)
            except Exception as ex:  # a crashing check is a FAIL, never silently dropped
                r = {"name": f"{mod}.{fn.__name__}", "inputs": {}, "expected": None, "observed": None,
                     "pass": False, "tolerance": None, "notes": f"EXCEPTION: {ex!r}\n{traceback.format_exc()}"}
            if a.only and a.only not in r["name"]:
                continue
            r["module"] = mod
            r["verdict"] = verdict(r)
            results.append(r)
            timing[r["name"]] = round(time.perf_counter() - t0, 3)
    summary = {}
    for r in results:
        summary[r["verdict"]] = summary.get(r["verdict"], 0) + 1
    out = {"schema": "prism.physics_refs.results.v1", "engine_root": str(root) if root else None,
           "environment": {"python": platform.python_version(), "numpy": numpy.__version__},
           "summary": summary, "checks": results}
    if a.timing:
        out["timing_s"] = timing
    with open(a.json, "w") as f:
        json.dump(out, f, indent=1, sort_keys=False, default=str, allow_nan=False)
        f.write("\n")
    w = max(len(r["name"]) for r in results)
    for r in results:
        print(f"{r['name']:<{w}}  {r['verdict']:<18} {r.get('defect_id', '')}")
    print(json.dumps(summary))
    bad = sum(1 for r in results if r["verdict"] == "FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
