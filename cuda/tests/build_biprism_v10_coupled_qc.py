#!/usr/bin/env python3
"""Build the actual-input coupled diagnostic against the active source tree.

No header overlays, selected candidate copies, or GPU dispatch occur here.
The executable and raw compiler output are retained under the supplied prefix.
"""
import argparse
import datetime
import hashlib
import json
import re
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--initial-guess-diagnostic", action="store_true",
                        help="Separate non-qualifying CPU-initial-guess experiment; never an official attempt")
    parser.add_argument("--tangent-model-capture", action="store_true",
                        help="Separate non-qualifying stop retaining the first failed continuous tangent action")
    args = parser.parse_args()
    if args.initial_guess_diagnostic and args.tangent_model_capture:
        parser.error("diagnostic modes are mutually exclusive")
    if not re.fullmatch(r"[0-9]{3}", args.attempt):
        parser.error("attempt must contain three decimal digits")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    role = Path(__file__).resolve().parents[2]
    kernels = role / "crates/prism-gpu/src/kernels"
    source = role / "cuda/tests/biprism_v10_coupled_qc.cu"
    prefix = output / (("failure_2_tangent_capture_build_" if args.tangent_model_capture else "failure_2_initial_guess_diagnostic_build_" if args.initial_guess_diagnostic
                        else "failure_1_2_3_coupled_native_build_") + args.attempt)
    receipt_path = prefix.with_suffix(".json")
    binary = output / (("biprism_tangent_capture_" if args.tangent_model_capture else "biprism_initial_guess_diagnostic_" if args.initial_guess_diagnostic
                        else "biprism_coupled_native_qc_") + args.attempt)
    if list(output.glob(prefix.name + ".*")) or binary.exists():
        parser.error("attempt already exists; retained builds are immutable")
    flags = [
        "/usr/local/cuda/bin/nvcc", "-arch=sm_120", "-O3", "--restrict",
        "-std=c++17", "-lineinfo", "-I" + str(kernels),
        "-I" + str(role / "cuda"), "-I" + str(source.parent),
        "-include", "stdint.h",
    ]
    if args.initial_guess_diagnostic:
        flags.append("-DBIPRISM_V10_DIAGNOSTIC_INITIAL_GUESS=1")
    if args.tangent_model_capture:
        flags.append("-DBIPRISM_V10_DIAGNOSTIC_TANGENT_CAPTURE=1")
    dependencies = subprocess.run(
        flags + ["-M", str(source)], capture_output=True, text=True, timeout=60
    )
    prefix.with_suffix(".dependencies").write_text(dependencies.stdout)
    prefix.with_suffix(".dependencies.stderr").write_text(dependencies.stderr)
    if dependencies.returncode:
        print(dependencies.stderr)
        return dependencies.returncode
    paths = dependencies.stdout.replace("\\\n", " ").split(":", 1)[1].split()
    pins = {str(Path(p).resolve()): digest(p) for p in paths}
    pins[str(Path(__file__).resolve())] = digest(__file__)
    command = flags + ["-Xptxas=-v", str(source), "-lcuda", "-o", str(binary)]
    receipt = {
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "canonical_source_tree": str(role), "source": str(source),
        "input_hashes": pins, "command": command, "header_overlays": False,
        "native_execution": False, "scientific_acceptance": False,
        "diagnostic_initial_guess": args.initial_guess_diagnostic,
        "diagnostic_tangent_capture": args.tangent_model_capture,
        "autonomous_qualification_eligible": not (args.initial_guess_diagnostic or args.tangent_model_capture),
        "selected_sources": {
            name: str(kernels / name) for name in (
                "nhs_biprism_variational_so3g_v10.cuh",
                "nhs_biprism_dec_pcg.cuh", "nhs_biprism_electronic_discrete.cuh",
            )
        },
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    with prefix.with_suffix(".stdout").open("x") as stdout, prefix.with_suffix(".stderr").open("x") as stderr:
        try:
            code = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=300).returncode
        except subprocess.TimeoutExpired:
            code = 124
    stable = all(digest(p) == h for p, h in pins.items())
    receipt.update(exit_code=code, sources_unchanged=stable)
    if code == 0:
        receipt["binary_sha256"] = digest(binary)
    receipt["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"exit_code": code, "sources_unchanged": stable, "receipt": str(receipt_path)}), flush=True)
    return code if code else (0 if stable else 3)


if __name__ == "__main__":
    raise SystemExit(main())
