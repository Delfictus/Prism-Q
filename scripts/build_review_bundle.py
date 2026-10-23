#!/usr/bin/env python3
"""Build, check and verify REVIEW_BUNDLE.md for the Prism-Q engine.

The bundle is the single flattened file AI and human reviewers read first. It
contains verbatim source excerpts (by symbol or line range) in data-flow order,
the equations each excerpt implements, units, invariants, known defects with
status, qualifying gates, mandatory reviewer questions, the sha256 of every
included file, and the results of independent executable reference checks.

Modes
  (default)  render REVIEW_BUNDLE.md (runs the reference checks and embeds them)
  --check    render in memory; exit 1 if REVIEW_BUNDLE.md is stale or an anchor is broken
  --verify   run the reference checks; exit 1 on any unexpected failure

Stdlib only (Python 3.10+; tomllib on 3.11+, falls back to the `tomli` package on 3.10).
Reference checks may use numpy/mpmath (pinned in CI).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

GENERATOR_NAME = "build_review_bundle.py"
GENERATOR_VERSION = "1.0.0"
LAYER_STATUSES = ("QUALIFIED", "EXECUTED_UNQUALIFIED", "NOT_BUILT")
DEFECT_STATUSES = ("OPEN", "FIX_IN_REVIEW", "FIXED", "WONTFIX_DOCUMENTED")
VOLATILE_PREFIX = "<!-- volatile:"  # lines excluded from the --check comparison
FENCE_LANG = {
    ".rs": "rust", ".cu": "cuda", ".cuh": "cuda", ".h": "c", ".hpp": "cpp", ".cpp": "cpp",
    ".c": "c", ".py": "python", ".toml": "toml", ".json": "json", ".sh": "bash", ".md": "markdown",
}

REVIEWER_CONTRACT = """\
1. Read this file first. It is generated from the source at the commit shown above; it is never hand-edited.
2. Review the verbatim source excerpts, not documentation, summaries or comments. Where an excerpt and
   its prose disagree, the source wins and the disagreement is a finding.
3. Answer every mandatory question in every section. Each answer cites `path:line` anchors from the excerpts
   (or from the repository at the same commit).
4. Re-run `python3 scripts/build_review_bundle.py --check` (staleness) and `--verify` (reference checks)
   and report the exit codes. Do not trust the embedded results without re-running them.
5. Report every disagreement with an equation, unit, invariant, status or defect entry as a finding with
   severity, `path:line`, the expected value, the observed value and a reproduction.
6. Status words mean exactly: QUALIFIED (passed its stated gate with a receipt), EXECUTED_UNQUALIFIED
   (runs, gate not passed), NOT_BUILT (does not exist yet). Never upgrade a status in a review.
7. Claim scope: computational evidence only."""

DATA_FLOW = """\
 prepared topology (AMBER params, residue identity, HMR)                         [S01]
        |
        v
 +---------------------------- one MD step (captured CUDA graph) ----------------------------+
 |  neighbour list [S05] -> bonded + nonbonded forces [S02] (+ solvent field [S06] NOT_BUILT) |
 |        -> integrator phases OU/kick/drift/SHAKE/RATTLE [S03] <- thermostat + RNG [S04]     |
 |        -> UV excited state + intervention work W_uv [S07]                                  |
 |        -> BIPRISM mesh/DEC operator [S08] -> electronic analogue [S09] (in-loop NOT_BUILT) |
 |        -> LIF / Voronoi detector -> spike events [S10]                                     |
 +--------------------------------------------------------------------------------------------+
        |  every datum bound to clock key + atom/residue/mesh anchor
        v
 Chronicle L0..L5: CLOCK, leaves, checkpoints, restart closure                    [S11]
        |
        v
 QNANL encode / seal / durable publish (post-run today)                            [S12]
        |
        v
 md_evidence_manifest.json (replay contract)                                       [S13]
        |                                  (no MD rerun past this line)
        v
 Path B materialization (prism_materialize_sites)                                  [S14]
        |
        v
 DSTW export (dstw_export_wt_pathb) -> WT physical profile JSON + Parquet         [S15]"""


# --------------------------------------------------------------------------- utils

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_text(path: Path) -> str:
    return path.read_bytes().decode("utf-8", errors="replace")


def git(engine_root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(engine_root), *args], capture_output=True, text=True,
                             timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def source_commit(engine_root: Path, override: str | None) -> dict[str, Any]:
    if override:
        return {"commit": override, "source": "--commit"}
    head = git(engine_root, "rev-parse", "HEAD")
    if head is None:
        return {"commit": "UNKNOWN (not a git checkout)", "source": "none"}
    prefix = git(engine_root, "rev-parse", "--show-prefix") or ""
    return {"commit": head, "source": "git", "subdir": prefix.rstrip("/") or "."}


def strip_c_like(line: str, in_block: bool) -> tuple[str, bool]:
    """Remove comments and string/char literals so brace counting is not fooled."""
    out: list[str] = []
    i, n = 0, len(line)
    while i < n:
        if in_block:
            end = line.find("*/", i)
            if end < 0:
                return "".join(out), True
            i, in_block = end + 2, False
            continue
        two = line[i:i + 2]
        if two == "//":
            break
        if two == "/*":
            in_block, i = True, i + 2
            continue
        ch = line[i]
        if ch == '"':
            j = i + 1
            while j < n and line[j] != '"':
                j += 2 if line[j] == "\\" else 1
            i = j + 1
            continue
        if ch == "'":
            # char literal ('{', '\n') vs Rust lifetime ('a): only skip short literals
            m = re.match(r"'(\\.|[^\\'])'", line[i:])
            if m:
                i += m.end()
                continue
        out.append(ch)
        i += 1
    return "".join(out), in_block


def brace_end(lines: list[str], start: int, max_scan: int = 20000) -> int | None:
    """Return the 0-based index of the line closing the first brace block at/after start."""
    depth, opened, in_block = 0, False, False
    for idx in range(start, min(len(lines), start + max_scan)):
        code, in_block = strip_c_like(lines[idx], in_block)
        for ch in code:
            if ch == "{":
                depth += 1
                opened = True
            elif ch == "}":
                depth -= 1
                if opened and depth == 0:
                    return idx
        if not opened and idx > start and code.rstrip().endswith(";"):
            return idx  # declaration/prototype without a body
    return None


# --------------------------------------------------------------------------- excerpts

@dataclass
class Excerpt:
    path: str
    selector: str
    state: str  # OK | NOT_BUILT | ANCHOR_BROKEN | SPEC_ERROR
    start: int = 0  # 1-based
    end: int = 0
    full_end: int = 0
    text: str = ""
    note: str = ""


def extract(engine_root: Path, spec: dict[str, Any], default_max: int) -> Excerpt:
    rel = spec.get("path", "")
    selectors = [k for k in ("symbol", "anchor", "lines") if k in spec]
    sel_desc = ", ".join(f"{k}={spec[k]!r}" for k in selectors)
    if len(selectors) != 1:
        return Excerpt(rel, sel_desc, "SPEC_ERROR", note="exactly one of symbol/anchor/lines required")
    path = engine_root / rel
    if not path.is_file():
        return Excerpt(rel, sel_desc, "NOT_BUILT", note=spec.get("note", "source file does not exist at this commit"))
    lines = read_text(path).splitlines()
    max_lines = int(spec.get("max_lines", default_max))
    kind = selectors[0]
    if kind == "lines":
        a, b = spec["lines"]
        if not (1 <= a <= b <= len(lines)):
            return Excerpt(rel, sel_desc, "ANCHOR_BROKEN", note=f"line range outside file (1..{len(lines)})")
        s, e = a - 1, b - 1
        full_e = e
    else:
        try:
            rx = re.compile(spec[kind])
        except re.error as exc:
            return Excerpt(rel, sel_desc, "SPEC_ERROR", note=f"bad regex: {exc}")
        occurrence = int(spec.get("occurrence", 1))
        hits = [i for i, ln in enumerate(lines) if rx.search(ln)]
        if len(hits) < occurrence:
            return Excerpt(rel, sel_desc, "ANCHOR_BROKEN",
                           note=f"pattern matched {len(hits)} time(s); occurrence {occurrence} requested")
        hit = hits[occurrence - 1]
        if kind == "anchor":
            s = max(0, hit - int(spec.get("before", 3)))
            e = min(len(lines) - 1, hit + int(spec.get("after", 12)))
            full_e = e
        else:
            s = max(0, hit - int(spec.get("before", 0)))
            end = brace_end(lines, hit)
            full_e = end if end is not None else min(len(lines) - 1, hit + max_lines - 1)
            e = full_e
    if e - s + 1 > max_lines:
        e = s + max_lines - 1
    width = len(str(e + 1))
    body = "\n".join(f"{str(i + 1).rjust(width)} | {lines[i]}" for i in range(s, e + 1))
    note = spec.get("note", "")
    if full_e > e:
        note = (note + " " if note else "") + (
            f"Excerpt truncated at max_lines={max_lines}; the full symbol spans lines {s + 1}-{full_e + 1}.")
    return Excerpt(rel, sel_desc, "OK", s + 1, e + 1, full_e + 1, body, note)


# --------------------------------------------------------------------------- checks

def load_checks(refs_dirs: list[Path], engine_root: Path) -> list[dict[str, Any]]:
    """Discover CHECKS in every *.py under the refs dirs and run them.

    Contract: fn(engine_root: Path | None = None) -> dict with keys
    name, inputs, expected, observed, pass (bool | None), tolerance, notes; optional defect_id.
    pass=None with notes starting 'NOT_BUILT' renders NOT_BUILT. Modules whose filename starts
    with 'builtin_' are fallbacks; a check of the same name from any other module supersedes them.
    """
    results: dict[str, dict[str, Any]] = {}
    origin: dict[str, bool] = {}  # name -> is_builtin
    files: list[Path] = []
    for d in refs_dirs:
        if d.is_dir():
            files.extend(sorted(p for p in d.glob("*.py") if not p.name.startswith("_")))
    for f in files:
        is_builtin = f.name.startswith("builtin_")
        mod_name = f"prismq_refs_{sha256_bytes(str(f).encode())[:12]}"
        try:
            spec = importlib.util.spec_from_file_location(mod_name, f)
            assert spec and spec.loader
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)
            checks = list(getattr(mod, "CHECKS", []))
        except Exception as exc:  # noqa: BLE001 - report, never hide
            name = f"import:{f.name}"
            results[name] = {"name": name, "module": f.name, "pass": False, "inputs": {}, "expected": "module imports",
                             "observed": f"{type(exc).__name__}: {exc}", "tolerance": None,
                             "notes": "Reference module failed to import.", "verdict": "ERROR"}
            origin[name] = is_builtin
            continue
        for fn in checks:
            try:
                try:
                    r = fn(engine_root=engine_root)
                except TypeError:
                    r = fn()
                r = dict(r)
            except Exception as exc:  # noqa: BLE001
                r = {"name": getattr(fn, "__name__", "unnamed"), "inputs": {}, "expected": "no exception",
                     "observed": f"{type(exc).__name__}: {exc}", "pass": False, "tolerance": None,
                     "notes": traceback.format_exc(limit=3)}
            r.setdefault("tolerance", None)
            r["module"] = f.name
            name = str(r.get("name", getattr(fn, "__name__", "unnamed")))
            if name in results and not origin[name] and is_builtin:
                continue  # a non-builtin check already owns this name
            results[name] = r
            origin[name] = is_builtin
    out = []
    for name in sorted(results):
        r = redact_host_paths(results[name], engine_root)
        r["builtin_fallback"] = origin[name]
        r["verdict"] = verdict(r)
        out.append(r)
    return out


HOST_PATH_RX = [(re.compile(r"/home/[^/\s\"']+"), "<home>"), (re.compile(r"/Users/[^/\s\"']+"), "<home>"),
                (re.compile(r"/mnt/[^/\s\"']+"), "<host-mnt>"), (re.compile(r"/tmp/[^\s\"']+"), "<tmp>")]


def redact_host_paths(r: dict[str, Any], engine_root: Path) -> dict[str, Any]:
    """Check records are published: replace the engine root and host-specific absolute paths with tokens."""
    s = json.dumps(r, sort_keys=True, default=str)
    for root in {str(engine_root), str(engine_root.resolve())}:
        s = s.replace(json.dumps(root)[1:-1], "<engine_root>")
    for rx, tok in HOST_PATH_RX:
        s = rx.sub(tok, s)
    return json.loads(s)


def verdict(r: dict[str, Any]) -> str:
    if r.get("verdict") == "ERROR":
        return "ERROR"
    p = r.get("pass")
    notes = str(r.get("notes", ""))
    if p is None:
        if notes.startswith("NOT_BUILT"):
            return "NOT_BUILT"
        if notes.startswith("INPUT_NOT_FOUND"):
            return "INPUT_NOT_FOUND"  # rendered explicitly; the check could not run on this host
        return "INCONCLUSIVE"
    if p is True:
        return "FIXED_UPDATE_SPEC" if r.get("defect_id") else "PASS"
    return "KNOWN_DEFECT_OPEN" if r.get("defect_id") else "FAIL"


def check_failed(r: dict[str, Any], open_defects: set[str]) -> bool:
    v = r["verdict"]
    if v in ("FAIL", "ERROR", "INCONCLUSIVE", "FIXED_UPDATE_SPEC"):
        return True
    if v == "KNOWN_DEFECT_OPEN" and r.get("defect_id") not in open_defects:
        return True  # failing against a defect the spec says is closed (or unknown)
    return False


# --------------------------------------------------------------------------- render

def jdump(v: Any) -> str:
    return json.dumps(v, sort_keys=True, default=str, ensure_ascii=False)


def md_cell(s: Any) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def render(spec: dict[str, Any], engine_root: Path, commit: dict[str, Any], checks: list[dict[str, Any]],
           spec_sha: str) -> tuple[str, dict[str, Any]]:
    b = spec.get("bundle", {})
    default_max = int(b.get("default_max_lines", 140))
    sections = spec.get("section", [])
    problems: list[str] = []
    file_hashes: dict[str, str] = {}
    rendered_sections: list[str] = []
    status_rows: list[str] = []

    for sec in sections:
        sid, title = sec["id"], sec["title"]
        st = sec.get("status", "")
        if st not in LAYER_STATUSES:
            problems.append(f"{sid}: invalid status {st!r}")
        excerpts = [extract(engine_root, e, default_max) for e in sec.get("excerpt", [])]
        for ex in excerpts:
            if ex.state == "OK":
                p = engine_root / ex.path
                file_hashes[ex.path] = sha256_bytes(p.read_bytes())
            elif ex.state in ("ANCHOR_BROKEN", "SPEC_ERROR"):
                problems.append(f"{sid}: {ex.state} {ex.path} [{ex.selector}] {ex.note}")
        open_ids = [d["id"] for d in sec.get("defect", []) if d.get("status") == "OPEN"]
        for d in sec.get("defect", []):
            if d.get("status") not in DEFECT_STATUSES:
                problems.append(f"{sid}: defect {d.get('id')} invalid status {d.get('status')!r}")
        planned = sec.get("planned", [])
        missing_src = [ex for ex in excerpts if ex.state == "NOT_BUILT"]
        status_rows.append(
            f"| {sid} | {md_cell(title)} | **{st}** | {len(open_ids)} | {len(planned) + len(missing_src)} | "
            f"{md_cell(sec.get('status_note', ''))} |")

        out: list[str] = [f"## {sid}. {title}", "", f"**Layer status:** `{st}`"]
        if sec.get("status_note"):
            out.append(f"  \n**Status note:** {sec['status_note']}")
        out += ["", sec.get("summary", "").strip(), ""]
        out += ["### Source excerpts (verbatim)", ""]
        if not excerpts:
            out += ["_No excerpt declared._", ""]
        for ex in excerpts:
            if ex.state == "OK":
                lang = FENCE_LANG.get(Path(ex.path).suffix, "")
                out.append(f"**`{ex.path}:{ex.start}-{ex.end}`** (selector: {ex.selector}; "
                           f"sha256 `{file_hashes[ex.path][:16]}…`)")
                if ex.note:
                    out.append(f"  \n_{ex.note}_")
                out += ["", f"```{lang}", ex.text, "```", ""]
            elif ex.state == "NOT_BUILT":
                out += [f"**`{ex.path}`**: **NOT_BUILT**. {ex.note}", ""]
            else:
                out += [f"**`{ex.path}`**: **{ex.state}** ({ex.selector}). {ex.note}", ""]
        if planned:
            out += ["### Planned components (NOT_BUILT)", "", "| Component | Package | Status |", "|---|---|---|"]
            out += [f"| {md_cell(p['name'])} | {md_cell(p.get('package', ''))} | NOT_BUILT |" for p in planned]
            out.append("")
        out += ["### Mathematics", "", sec.get("equations", "").strip() or "_none declared_", ""]
        out += ["### Units", "", "| Quantity | Unit |", "|---|---|"]
        out += [f"| {md_cell(q)} | {md_cell(u)} |" for q, u in sec.get("units", [])]
        out += ["", "### Invariants", ""]
        out += [f"- {i}" for i in sec.get("invariants", [])] or ["- _none declared_"]
        out += ["", "### Known defects", ""]
        defects = sec.get("defect", [])
        if defects:
            out += ["| ID | Defect | Where | Status | Fix package | Check |", "|---|---|---|---|---|---|"]
            out += [f"| {d['id']} | {md_cell(d['title'])} | {md_cell(d.get('where', ''))} | **{d['status']}** | "
                    f"{md_cell(d.get('fix_package', ''))} | {md_cell(d.get('check', ''))} |" for d in defects]
        else:
            out.append("_None recorded._")
        out += ["", "### Qualifying gate", "", sec.get("gate", "_none declared_"), ""]
        qs = sec.get("questions", [])
        if len(qs) < 3:
            problems.append(f"{sid}: fewer than 3 mandatory reviewer questions")
        out += ["### Mandatory reviewer questions", ""]
        out += [f"{n}. {q}" for n, q in enumerate(qs, 1)]
        out.append("")
        rendered_sections.append("\n".join(out))

    all_open = {d["id"] for s in sections for d in s.get("defect", []) if d.get("status") == "OPEN"}
    tree_digest = sha256_bytes("\n".join(f"{p}\t{h}" for p, h in sorted(file_hashes.items())).encode())

    # ---- checks section
    chk: list[str] = ["## Executable reference checks (`--verify`)", "",
                      "Independent CPU reference implementations, deterministic. Host paths in records are shown as `<engine_root>`, `<home>`, `<host-mnt>`. `KNOWN_DEFECT_OPEN` means the check "
                      "reproduces a defect that is still recorded as OPEN above; `NOT_BUILT` means the source under "
                      "test does not exist yet. Neither is silently skipped.", ""]
    if checks:
        chk += ["| Check | Verdict | Defect | Module | Expected | Observed | Tolerance |", "|---|---|---|---|---|---|---|"]
        for r in checks:
            chk.append(f"| {md_cell(r.get('name'))} | **{r['verdict']}** | {md_cell(r.get('defect_id', ''))} | "
                       f"{md_cell(r.get('module'))}{' (fallback)' if r.get('builtin_fallback') else ''} | "
                       f"{md_cell(_short(r.get('expected')))} | {md_cell(_short(r.get('observed')))} | "
                       f"{md_cell(_short(r.get('tolerance')))} |")
        chk += ["", "<details><summary>Full check records (JSON)</summary>", "", "```json"]
        chk += [jdump({k: r[k] for k in sorted(r) if k != "verdict"} | {"verdict": r["verdict"]}) for r in checks]
        chk += ["```", "", "</details>", ""]
    else:
        chk += ["**NOT_BUILT**: no reference check modules were found under `scripts/physics_refs/`.", ""]

    # ---- header
    hdr: list[str] = [
        f"# {b.get('title', 'Review bundle')}", "",
        "> GENERATED FILE. Do not edit. Regenerate with `python3 scripts/build_review_bundle.py`.", "",
        "| Field | Value |", "|---|---|",
        f"| Source commit | `{commit.get('commit')}` (engine subdir `{commit.get('subdir', '.')}`) |",
        f"| Generator | `{GENERATOR_NAME}` v{GENERATOR_VERSION} |",
        f"| Spec | `review_bundle.toml` sha256 `{spec_sha}` |",
        f"| Included source files | {len(file_hashes)} |",
        f"| Source digest (sha256 over sorted path, sha256 pairs) | `{tree_digest}` |",
        f"| Sections | {len(sections)} |",
        f"| Open defects | {len(all_open)} |", "",
        "The source commit above is the commit checked out when the bundle was generated. The commit that "
        "contains this file is that commit's child when the bundle is regenerated in the same change. "
        "Staleness is decided by the file hashes and the rendered content (`--check`), not by the commit id.", "",
        "## Reviewer contract", "", REVIEWER_CONTRACT, "",
        "## Claim scope", "", b.get("claim_scope", "").strip(), "",
        "## Layer status", "",
        "| Section | Layer | Status | Open defects | NOT_BUILT parts | Note |", "|---|---|---|---|---|---|",
        *status_rows, "",
        "## Data flow (input to Path B / DSTW)", "", "```text", DATA_FLOW, "```", "",
        "## Build and tree hazards", "",
        *([f"- {h}" for h in b.get("hazards", [])] or ["_None recorded._"]), "",
        "## Included files", "", "| File | sha256 |", "|---|---|",
        *[f"| `{p}` | `{h}` |" for p, h in sorted(file_hashes.items())], "",
    ]
    if problems:
        hdr += ["## Bundle integrity problems", "", *[f"- {p}" for p in problems], ""]
    doc = "\n".join(hdr) + "\n" + "\n".join(rendered_sections) + "\n" + "\n".join(chk)
    if not doc.endswith("\n"):
        doc += "\n"
    meta = {"problems": problems, "file_hashes": file_hashes, "tree_digest": tree_digest, "open_defects": sorted(all_open),
            "bytes": len(doc.encode("utf-8"))}
    max_bytes = int(b.get("max_bytes", 2_200_000))
    if meta["bytes"] > max_bytes:
        problems.append(f"bundle is {meta['bytes']} bytes > max_bytes {max_bytes}")
    return doc, meta


def _short(v: Any, n: int = 120) -> str:
    s = v if isinstance(v, str) else jdump(v)
    return s if len(s) <= n else s[: n - 1] + "…"


def normalize(doc: str) -> str:
    """Reduce a bundle to what --check compares.

    Dropped: the source-commit line (the regenerating commit is the child of the recorded one) and
    volatile lines. In the reference-check section only (check name, verdict) is compared, because
    float digits can differ across machines/BLAS builds; --verify re-runs the values themselves.
    """
    out: list[str] = []
    in_checks = in_details = False
    for ln in doc.splitlines():
        if ln.startswith("| Source commit |") or ln.startswith(VOLATILE_PREFIX):
            continue
        if ln.startswith("## "):
            in_checks = ln.startswith("## Executable reference checks")
        if in_checks:
            if ln.startswith("<details>"):
                in_details = True
            if in_details:
                if ln.startswith("</details>"):
                    in_details = False
                continue
            if ln.startswith("| ") and not ln.startswith("|---"):
                cells = ln.split("|")
                ln = "|".join(cells[:3])
        out.append(ln)
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent.parent
    ap.add_argument("--engine-root", type=Path, default=here, help="engine root (default: repo root)")
    ap.add_argument("--spec", type=Path, default=None, help="review_bundle.toml (default: <engine-root>/review_bundle.toml)")
    ap.add_argument("--out", type=Path, default=None, help="output (default: <engine-root>/REVIEW_BUNDLE.md)")
    ap.add_argument("--refs-dir", type=Path, action="append", default=None,
                    help="reference-check module dir (repeatable; default: <engine-root>/scripts/physics_refs)")
    ap.add_argument("--commit", default=None, help="override the recorded source commit")
    ap.add_argument("--json", type=Path, default=None, help="also write check results + metadata as JSON")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="fail if REVIEW_BUNDLE.md is stale or anchors are broken")
    mode.add_argument("--verify", action="store_true", help="run reference checks; fail on unexpected failures")
    a = ap.parse_args(argv)

    root = a.engine_root.resolve()
    spec_path = (a.spec or root / "review_bundle.toml").resolve()
    out_path = (a.out or root / "REVIEW_BUNDLE.md").resolve()
    refs = a.refs_dir or [root / "scripts" / "physics_refs"]
    spec_bytes = spec_path.read_bytes()
    spec = tomllib.loads(spec_bytes.decode("utf-8"))
    spec_sha = sha256_bytes(spec_bytes)
    commit = source_commit(root, a.commit)

    checks = load_checks([p.resolve() for p in refs], root)
    all_open = {d["id"] for s in spec.get("section", []) for d in s.get("defect", []) if d.get("status") == "OPEN"}

    if a.verify:
        bad = [r for r in checks if check_failed(r, all_open)]
        for r in checks:
            print(f"{r['verdict']:<20} {r.get('name')}  [{r.get('module')}]"
                  + (f"  defect={r['defect_id']}" if r.get("defect_id") else ""))
        if a.json:
            a.json.write_text(json.dumps({"checks": checks}, indent=2, sort_keys=True, default=str) + "\n")
        if not checks:
            print("NOT_BUILT: no reference checks found", file=sys.stderr)
        print(f"verify: {len(checks)} checks, {len(bad)} unexpected failure(s)")
        return 1 if bad else 0

    doc, meta = render(spec, root, commit, checks, spec_sha)
    if a.json:
        a.json.write_text(json.dumps({"meta": meta, "checks": checks, "commit": commit}, indent=2,
                                     sort_keys=True, default=str) + "\n")
    if a.check:
        rc = 0
        if meta["problems"]:
            print("bundle integrity problems:", *meta["problems"], sep="\n  ", file=sys.stderr)
            rc = 1
        if not out_path.is_file():
            print(f"STALE: {out_path} does not exist", file=sys.stderr)
            return 1
        if normalize(out_path.read_text(encoding="utf-8")) != normalize(doc):
            print(f"STALE: {out_path} differs from a fresh render; run scripts/build_review_bundle.py", file=sys.stderr)
            rc = 1
        if rc == 0:
            print(f"check: {out_path.name} is current ({meta['bytes']} bytes, {len(meta['file_hashes'])} files)")
        return rc

    out_path.write_text(doc, encoding="utf-8")
    print(f"wrote {out_path} ({meta['bytes']} bytes, {len(meta['file_hashes'])} files, "
          f"{len(checks)} checks, {len(meta['problems'])} problem(s))")
    for p in meta["problems"]:
        print(f"  problem: {p}", file=sys.stderr)
    return 1 if meta["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
