#!/usr/bin/env python3
"""Reopen a native PRISM-Therm result with its original typed joins.

This is a saved-result reader. It never propagates MD or reruns SDST. Native
estimator values keep their recorded availability and computational meaning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def read(path: Path) -> tuple[dict, dict]:
    raw = path.read_bytes()
    value = json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    require(isinstance(value, dict), f"object required: {path}")
    return value, {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def identity(value: dict) -> tuple[str, int]:
    require(isinstance(value, dict), "typed site identity absent")
    domain, index = value.get("domain"), value.get("id")
    require(domain == "cluster" and type(index) is int and index >= 0,
            "this reader requires the recorded original cluster identity")
    return domain, index


def same_float(a: object, b: object, field: str) -> None:
    if a is None or b is None:
        require(a is None and b is None, f"availability mismatch: {field}")
        return
    require(type(a) in (int, float) and type(b) in (int, float), f"non-numeric {field}")
    require(math.isfinite(a) and math.isfinite(b), f"nonfinite {field}")
    # Rust f32 and serde_json::Value can print different decimal precisions.
    import struct
    require(struct.pack("<f", a) == struct.pack("<f", b), f"native f32 mismatch: {field}")


def inspect(report_path: Path, status_path: Path, binding_path: Path, pocket: int | None = None,
            acceptance_path: Path | None = None) -> dict:
    report, report_pin = read(report_path)
    status, status_pin = read(status_path)
    binding, binding_pin = read(binding_path)
    acceptance_pin = None
    if acceptance_path is not None:
        acceptance, acceptance_pin = read(acceptance_path)
        expected = acceptance["binding_sites"]
        require(Path(expected["path"]).resolve() == binding_path.resolve()
                and expected["sha256"] == binding_pin["sha256"], "accepted binding-site artifact mismatch")
        require(acceptance.get("status") == "IMPLEMENTATION_VERIFIED", "source acceptance is not verified")
    require(status.get("schema_kind") == "prismos.therm_surface_status.v1"
            and status.get("schema_version") == 1, "unknown Therm status schema")
    require(status.get("status") == "observed" and status.get("effective") == "clusters",
            "original cluster Therm analysis was not observed")
    require(status.get("failure") is None, "native analysis records a failure")
    require(report.get("structure") == binding.get("structure"), "structure identity mismatch")
    pockets = report.get("pockets")
    require(isinstance(pockets, list) and report.get("total_pockets") == len(pockets), "report population mismatch")
    expected = [identity(x) for x in status["analyzed_site_ids"]]
    observed = [identity(x["analyzed_site_identity"]) for x in pockets]
    require(len(set(expected)) == len(expected) and set(expected) == set(observed)
            and len(set(observed)) == len(observed), "typed pocket population mismatch")
    therm = binding["prism_therm"]
    require(report["sdst_event_count"] == therm["sdst_event_count"]
            == sum(x["events"] for x in status["group_outcomes"]), "native event-count mismatch")
    require(all(x["status"] == "observed" for x in status["group_outcomes"]), "unobserved protocol group")
    raw_sites = {identity(x["analyzed_site_identity"]): x for x in therm["sites"]}
    require(len(raw_sites) == len(therm["sites"]) and set(raw_sites) == set(expected), "analysis population mismatch")
    output_sites = {x["id"]: x for x in binding["sites"]}
    require(len(output_sites) == len(binding["sites"]), "duplicate PathB output site identity")
    checks = 0
    for row in pockets:
        key = identity(row["analyzed_site_identity"])
        require(row["pocket_id"] == key[1], "report numeric/typed identity mismatch")
        raw = raw_sites[key]
        out = output_sites.get(key[1])
        require(out is not None, "reported pocket missing from PathB binding-site output")
        for field, source in (("ccns_tau", "tau"), ("hysteresis_asymmetry", "asymmetry_score"),
                              ("relative_asymmetry", "relative_asymmetry")):
            same_float(row[field], raw[source], field)
            same_float(row[field], out[field], field)
            checks += 2
        require(row["therm_class"].upper() == str(raw["therm_class"]).upper()
                == str(out["therm_class"]).upper(), "classification mismatch")
        checks += 1
    require(report["cryptic_pockets"] == sum(bool(x["is_cryptic"]) for x in pockets), "cryptic count mismatch")
    chosen = pockets if pocket is None else [x for x in pockets if x["pocket_id"] == pocket]
    require(chosen or pocket is None, "requested pocket is outside recorded population")
    return {
        "schema": "prism.native_therm_saved_result_reader.v1",
        "status": "IMPLEMENTATION_VERIFIED", "scope": "saved original native SDST report reader and cross-file identity/value joins",
        "md_executed": False, "native_sdst_reexecuted": False, "restart_qualified": False,
        "source_artifacts": [report_pin, status_pin, binding_pin],
        "acceptance_receipt": acceptance_pin,
        "source_identity_scope": "supplied report/status/binding-site coherence; report and status contain no run_id; accepted binding-site hash checked when acceptance receipt is supplied",
        "run_id": binding["run_id"], "structure": report["structure"],
        "recorded_sdst_event_count": report["sdst_event_count"], "recorded_pocket_count": len(pockets),
        "cross_file_value_checks": checks, "protocol_groups": status["provenance"]["source_streams_by_group"],
        "query": {"kind": "recorded_cluster_readout", "pocket_id": pocket, "supported": True},
        "pockets": chosen,
        "measurement_scope": {
            "ccns_tau": "native fitted avalanche exponent",
            "hysteresis_asymmetry": "native protocol-conditioned heating/cooling event-rate asymmetry",
            "transfer_entropy": "native TIDE binned event estimator; predictive association does not establish intervention effect",
            "causal_dg": "preserved upstream field name; native event-derived proxy, not binding free energy",
            "fisher_info": "preserved upstream field name; native windowed transfer-entropy variance",
            "null": "recorded unavailable or undefined estimate; never replaced with zero",
        },
        "unsupported_query": {"kind": "equilibrium_pocket_opening_rate_or_mutation_affinity",
                              "supported": False, "reason": "requires an applicable physical model, trajectory population, and separately validated estimator"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--binding-sites", type=Path, required=True)
    parser.add_argument("--pocket-id", type=int)
    parser.add_argument("--query", choices=("recorded_cluster_readout", "natural_opening_rate"),
                        default="recorded_cluster_readout")
    parser.add_argument("--acceptance", type=Path, help="Existing production acceptance receipt pinning the binding-site artifact")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.report, args.status, args.binding_sites, args.pocket_id, args.acceptance)
    if args.query == "natural_opening_rate":
        result["query"] = {"kind": args.query, "pocket_id": args.pocket_id, "supported": False,
                           "value": None, "reason": "requires an unperturbed reference population, complete physical exposure and controller context, and a validated kinetic sampling law"}
        result["pockets"] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "recorded_sdst_event_count", "recorded_pocket_count", "cross_file_value_checks")}))


if __name__ == "__main__":
    main()
