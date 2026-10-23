#!/usr/bin/env python3
"""Compile actual potential-owner/layout/readback predicates without CUDA dispatch.

The C++ header and Rust function bodies are consumed from the canonical tree.
Only CUDA/serde imports and derives are omitted from the isolated Rust drivers;
the numerical and acceptance predicates are copied verbatim. This checks source
and host contracts; a full Cargo build and native acceptance remain separate.
"""
from pathlib import Path
import argparse
import hashlib
import re
import subprocess
import tempfile

ROLE = Path(__file__).resolve().parents[2]
KERNELS = ROLE / "crates/prism-gpu/src/kernels"
NHS = ROLE / "crates/prism-nhs/src"


def between(text, first, last):
    begin = text.index(first)
    return text[begin:text.index(last, begin)]


def host_declarations(text):
    text = text.replace(",serde::Serialize", "").replace(", Serialize", "")
    return re.sub(r"^unsafe impl DeviceRepr for \w+ \{\}\n", "", text, flags=re.M)


def run(command):
    subprocess.run([str(x) for x in command], check=True)


def controls(output):
    output.mkdir(parents=True, exist_ok=True)
    sources = [KERNELS / "nhs_biprism_potential_precision.cuh",
               KERNELS / "nhs_biprism_coordinate_precision.cuh",
               NHS / "biprism_v10_runtime.rs", NHS / "biprism_phase2_work.rs",
               NHS / "biprism_mixed_energy.rs",
               ROLE / "cuda/tests/biprism_potential_owner_control.cpp",
               ROLE / "cuda/tests/biprism_mixed_validator_control.rs"]
    identities = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    for path, digest in identities.items():
        print(f"source {digest} {path}", flush=True)
    cpp = output / "potential_owner"
    run(["g++", "-std=c++17", "-O2", "-I", KERNELS,
         ROLE / "cuda/tests/biprism_potential_owner_control.cpp", "-o", cpp])
    run([cpp])

    runtime = (NHS / "biprism_v10_runtime.rs").read_text()
    constants = between(runtime, "pub const BIPRISM_LEAVES", "/// Exact CUDA ABI")
    table = between(runtime, "/// Exact CUDA ABI", "/// Persistent FP64 potential")
    owner = between(runtime, "/// Persistent FP64 potential", "impl BiprismPotentialReadback")
    predicates = between(runtime, "fn potential_bytes", "/// Device-built traversal scratch")
    scratch = between(runtime, "#[repr(C,align(128))]\npub struct BiprismNuclearQrFfi", "/// This relation is stronger")
    abi = between(runtime, "const _: () = {\n    assert!(core::mem::size_of::<FoliatedStreamPtrTableFfi>", "#[derive(Clone, Debug, PartialEq, Eq)]\npub struct BiprismIdentity")
    assertions = "const _:()={\n" + "\n".join(line for line in abi.splitlines()
        if "assert!" in line and ("FoliatedStreamPtrTableFfi" in line or "BiprismPotentialOwnerFfi" in line)) + "\n};\n"
    drivers = {"potential_readback": host_declarations(constants + table + owner + predicates + scratch + assertions)}

    phase2 = (NHS / "biprism_phase2_work.rs").read_text()
    drivers["phase2_readback"] = host_declarations(
        "const BIPRISM_TILE_BYTES:usize=225408;\n"
        + between(phase2, "pub const PHASE2_WORK_MAGIC", "impl Phase2WorkReadback")
        + between(phase2, "fn bytes_le", "pub(crate) struct BiprismPhase2WorkOwner")
        + between(phase2, "fn completed<", "fn upload<")
        + phase2[phase2.index("fn close("):])
    for name, source in drivers.items():
        path = output / f"{name}.rs"
        path.write_text("#![allow(dead_code)]\n" + source)
        binary = output / name
        run(["rustc", "--edition=2021", "--test", "-O", path, "-o", binary])
        run([binary, "--test-threads=1"])

    mixed = (NHS / "biprism_mixed_energy.rs").read_text()
    driver = output / "mixed_readback.rs"
    driver.write_text("#![allow(dead_code)]\n" + host_declarations(
        between(mixed, "pub const MIXED_MAGIC", "fn bytes<T>"))
        + between(mixed, "fn close(", "pub(crate) struct MixedGuardPointers")
        + (ROLE / "cuda/tests/biprism_mixed_validator_control.rs").read_text())
    binary = output / "mixed_readback"
    run(["rustc", "--edition=2021", "-O", driver, "-o", binary])
    run([binary])
    if identities != {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}:
        raise RuntimeError("source changed during controls")
    print("PASS actual-source C++/Rust controls; source identities unchanged; no GPU execution", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir:
        controls(args.output_dir.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="biprism-potential-controls-") as directory:
            controls(Path(directory))
