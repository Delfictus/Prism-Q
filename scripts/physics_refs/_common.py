"""Shared helpers for the PRISM physics reference checks.

Every check returns the review-bundle record:
    {name, inputs, expected, observed, pass, tolerance, notes[, defect_id]}
pass=None with notes starting "NOT_BUILT:" or "INPUT_NOT_FOUND:" means the
check could not run against the engine tree it was given.

Source anchors: a transcription is only trusted when the cited engine lines
still contain the transcribed text. Each anchor records path, line range,
sha256 of the whole file and whether the expected fragments were found in the
cited range ("transcription_current").
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

KERNELS = Path("crates/prism-gpu/src/kernels")
NHS_SRC = Path("crates/prism-nhs/src")

# Optional workstation defaults live in _site.py (NOT shipped to the public repo):
#   ENGINE_ROOT_CANDIDATES = (Path, ...); TOPOLOGY_CANDIDATES = {"IL2": Path, ...}
try:
    try:
        from . import _site as _SITE  # type: ignore
    except ImportError:
        import importlib.util as _ilu
        _sp = Path(__file__).with_name("_site.py")
        if _sp.is_file():
            _ss = _ilu.spec_from_file_location("prism_refs_site", _sp)
            _SITE = _ilu.module_from_spec(_ss); _ss.loader.exec_module(_SITE)
        else:
            _SITE = None
except Exception:  # never let site defaults break a check
    _SITE = None


def resolve_engine_root(engine_root=None):
    """argument -> $PRISM_ENGINE_ROOT -> in-repo (engine/scripts/physics_refs -> engine) -> _site.py."""
    if engine_root is not None:
        return Path(engine_root)
    env = os.environ.get("PRISM_ENGINE_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    if len(here.parents) > 2 and (here.parents[2] / KERNELS).is_dir():
        return here.parents[2]
    for cand in getattr(_SITE, "ENGINE_ROOT_CANDIDATES", ()):
        if (Path(cand) / KERNELS).is_dir():
            return Path(cand)
    return None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class AnchorMissing(Exception):
    pass


def anchor(engine_root, rel, first, last, fragments, search_window=400):
    """Bind a transcription to engine source.

    rel: path relative to engine_root. first/last: cited 1-based lines.
    fragments: strings that must appear (whitespace-insensitive) in the cited
    range. If they are not in the cited range but are found within
    +-search_window lines, the anchor reports the drifted line range.
    """
    root = resolve_engine_root(engine_root)
    if root is None:
        raise AnchorMissing("no engine root found (pass engine_root or set PRISM_ENGINE_ROOT)")
    path = root / rel
    if not path.is_file():
        raise AnchorMissing(f"{path} not present")
    lines = path.read_text(errors="replace").splitlines()
    squash = lambda s: "".join(s.split())

    def found_in(a, b):
        blob = squash("\n".join(lines[max(a - 1, 0):b]))
        return all(squash(fr) in blob for fr in fragments)

    current = found_in(first, last)
    located = (first, last) if current else None
    if not current:
        span = last - first
        for a in range(max(1, first - search_window), min(len(lines), first + search_window) + 1):
            if found_in(a, a + span):
                located = (a, a + span)
                break
    return {
        "path": str(path),
        "cited_lines": f"{first}-{last}",
        "located_lines": None if located is None else f"{located[0]}-{located[1]}",
        "sha256": sha256_file(path),
        "transcription_current": bool(current),
        "fragments_found_nearby": located is not None,
    }


def record(name, inputs, expected, observed, passed, tolerance, notes, defect_id=None, anchors=None):
    if passed is not None:
        passed = bool(passed)
    rec = {
        "name": name,
        "inputs": inputs,
        "expected": expected,
        "observed": observed,
        "pass": passed,
        "tolerance": tolerance,
        "notes": notes,
    }
    if anchors is not None:
        rec["inputs"] = dict(inputs or {})
        rec["inputs"]["source_anchors"] = anchors
    if defect_id:
        rec["defect_id"] = defect_id
    return rec


def not_built(name, why):
    return record(name, {}, None, None, None, None, f"NOT_BUILT: {why}")


def anchors_ok(anchors):
    return all(a.get("fragments_found_nearby") for a in anchors)


# Real coordinate/charge inputs are read in place (never copied) from
# $PRISM_REFS_TOPOLOGY_<NAME> or _site.TOPOLOGY_CANDIDATES, and hashed.
_TOPO_CACHE = {}


def load_topology(name):
    """Return (dict, path, sha256) or (None, path, None). Override with $PRISM_REFS_TOPOLOGY_<NAME>."""
    import json
    default = getattr(_SITE, "TOPOLOGY_CANDIDATES", {}).get(name, "")
    path = Path(os.environ.get(f"PRISM_REFS_TOPOLOGY_{name.upper()}", default))
    if name in _TOPO_CACHE:
        return _TOPO_CACHE[name]
    if not str(default) and f"PRISM_REFS_TOPOLOGY_{name.upper()}" not in os.environ:
        out = (None, f"unset (set $PRISM_REFS_TOPOLOGY_{name.upper()})", None)
    elif not path.is_file():
        out = (None, str(path), None)
    else:
        out = (json.loads(path.read_text()), str(path), sha256_file(path))
    _TOPO_CACHE[name] = out
    return out
