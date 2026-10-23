#!/usr/bin/env python3
"""Run exact ordered-edge phases and actual DEC row templates on the CPU.

The extracted device row functions contain no CUDA collectives. Erasing only
their device/inlining qualifiers lets the actual arithmetic run on the host.
This does not execute or qualify the cooperative GPU builder or PCG solver.
"""
import argparse
import hashlib
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]


def declaration(text, marker, terminator="}"):
    begin = text.index(marker)
    opening = text.index("{", begin)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    if terminator == "};":
        assert text[end] == ";"
        end += 1
    return text[begin:end]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tiles", nargs="*", type=pathlib.Path)
    args = parser.parse_args()
    dec = ROOT / "crates/prism-gpu/src/kernels/nhs_biprism_dec_pcg.cuh"
    runtime = ROOT / "crates/prism-nhs/src/biprism_v10_runtime.rs"
    cpp = ROOT / "cuda/tests/biprism_ordered_edges_control.cpp"
    ordered = dec.with_name("nhs_biprism_ordered_edges.cuh")
    paths = [dec, runtime, cpp, ordered, pathlib.Path(__file__), *args.tiles]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    source = dec.read_text()
    rows = [declaration(source, marker, end) for marker, end in [
        ("template<class EdgeWeight, class EdgeTraversal>\n__device__ __forceinline__ double biprism_dec_row_f64_traversal(", "}"),
        ("template<class EdgeWeight>\n__device__ __forceinline__ double biprism_dec_row_f64_accessor(", "}"),
        ("struct BiprismDecArrayWeights {", "};"),
    ]]
    exact = "\n\n".join(rows).replace("__device__ ", "").replace("__forceinline__", "inline")
    with tempfile.TemporaryDirectory(prefix="biprism_ordered_edges_") as work:
        directory = pathlib.Path(work)
        (directory / "biprism_actual_dec_rows.hpp").write_text(
            "constexpr uint32_t TILE_CORE_VERTICES=192u;\n" + exact)
        binary = directory / "ordered_edges_control"
        subprocess.run(["g++", "-std=c++17", "-O3", "-ffp-contract=off",
                        "-DBIPRISM_TEST_ACTUAL_DEC_ROWS", "-I", str(directory),
                        str(cpp), "-o", str(binary)], check=True)
        subprocess.run([str(binary), *map(str, args.tiles)], check=True)
        rust = runtime.read_text()
        ffi = []
        for name in ("FoliatedStreamPtrTableFfi", "BiprismOrderedEdgesFfi"):
            ffi.append("#[repr(C, align(128))]\n" + declaration(rust, f"pub struct {name} {{"))
        assertions = [line.strip() for line in rust.splitlines() if
                      line.strip().startswith("assert!(core::mem::") and
                      any(name in line for name in ("FoliatedStreamPtrTableFfi", "BiprismOrderedEdgesFfi"))]
        rust_control = directory / "layout.rs"
        rust_control.write_text(
            "const BIPRISM_LEAVES:usize=8;const BIPRISM_ORDERED_EDGES_BYTES:usize=6656;\n"
            + "\n".join(ffi) + "\nfn main(){\n" + "\n".join(assertions)
            + '\nprintln!("actual Rust table640/ordered6656 layout controls PASS");}\n')
        subprocess.run(["rustc", "--edition=2021", str(rust_control), "-o", str(directory / "layout")], check=True)
        subprocess.run([str(directory / "layout")], check=True)
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    if before != after:
        raise RuntimeError("input changed during focused control")
    for path, digest in before.items():
        print(digest, path)


if __name__ == "__main__":
    main()
