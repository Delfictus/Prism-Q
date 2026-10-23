#!/usr/bin/env python3
"""Check the actual three-body connected main/LIF native component readback.

This adapter binds the canonical native build, dispatch, boot, physical inputs
and raw outputs, derives the work/time indices from retained device windows,
then calls the Fraction-based check_buffers numerical oracle. It
does not launch CUDA or fabricate a host-produced accepted journal. The native
executable itself is the bound device fatbinary; no separate PTX is substituted.

The diagnostic's fixed requests and component descriptor are explicit inputs
and control records. Passing this checker does not qualify the full AMBER
schedule, mixed-energy guard, terminal-control continuation or complete C4
physics. The guarded runner separately requires its independent DEC checks.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import io
import re
from pathlib import Path
import struct
import sys
import uuid

import check_biprism_native_lif_work_rows as W

ENGINE = Path(__file__).resolve().parents[2]
KERNELS = "crates/prism-gpu/src/kernels/"
REQUIRED_SOURCES = (
    "cuda/tests/biprism_v10_coupled_qc.cu",
    "cuda/tests/biprism_connected_work_control.hpp",
    "cuda/tests/biprism_v10_reciprocal_parity.cuh",
    *(KERNELS + name for name in (
        "nhs_biprism_variational_so3g_v10.cuh", "nhs_biprism_dec_pcg.cuh",
        "nhs_biprism_electronic_discrete.cuh", "nhs_biprism_voronoi_lif.cuh",
        "nhs_biprism_lif_work.cuh", "nhs_biprism_shared_step.cuh",
        "nhs_biprism_shared_step_abi.cuh", "nhs_biprism_unit_bridge.cuh",
        "nhs_biprism_coordinate_precision.cuh", "nhs_biprism_coordinate_geometry.cuh")),
)
PHYSICAL_HASHES = (
    "eff6c982d8cdbd9f77e6e4ecf37852fb33ca486712747079e3a0e42efaf5bd54",
    "b32acda57f813d8b24b013826a3f8cc635917e1c02a95c6e1004a63ca40d69f9",
    "37e070f483d07f186cddc1d8ac2cdf2a8339b57d0bf6884742864b54bfedc772",
)
TERMINAL_KEYS = (
    "abi_magic", "leaf_id", "terminal_step", "captured_step", "temperature",
    "effective_gamma", "requested_dt_ps", "uv_wavelength_nm", "uv_burst_active",
    "uv_target_idx", "phase_bits", "reserved",
)
CARRY_BYTES = 5152
require = W.require


def integer(value, expected, label):
    require(type(value) is int and value == expected, label)


def digest_valid(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


class Files:
    """Keep all executed-source/input/output identities and recheck on exit."""
    def __init__(self):
        self.identities = {}

    def verify(self, path, digest=None, size=None):
        path = Path(path)
        require(path.is_absolute(), f"unqualified relative evidence path: {path}")
        path = path.resolve()
        require(path.is_file(), f"required evidence absent: {path}")
        actual = {"sha256": W.sha256_file(path), "bytes": path.stat().st_size}
        if digest is not None:
            require(digest_valid(digest) and actual["sha256"] == digest, f"hash mismatch: {path}")
        if size is not None:
            require(type(size) is int and size >= 0 and actual["bytes"] == size, f"byte extent mismatch: {path}")
        require(str(path) not in self.identities or self.identities[str(path)] == actual,
                f"evidence changed during read: {path}")
        self.identities[str(path)] = actual
        return path

    def read(self, path, digest=None, size=None):
        path = self.verify(path, digest, size)
        data = path.read_bytes()
        require(hashlib.sha256(data).hexdigest() == self.identities[str(path)]["sha256"],
                f"evidence changed while reading: {path}")
        return data

    def document(self, path, digest=None):
        return W.strict_json(self.read(path, digest))

    def unchanged(self):
        for path, identity in list(self.identities.items()):
            self.verify(path, identity["sha256"], identity["bytes"])


def receipt_fields(execution, build):
    for name in ("exit_code", "native_exit_code"):
        integer(execution.get(name), 0, f"native dispatch did not pass {name}")
    for name in ("native_execution", "sources_binary_inputs_unchanged", "gpu_recovery_none_after"):
        require(execution.get(name) is True, f"missing successful native dispatch field: {name}")
    require(execution.get("new_kernel_fault_records") == [], "native dispatch recorded a new kernel fault")
    require(execution.get("diagnostic_mode") == "--connected-work" and
            execution.get("baseline_repeat") is False and execution.get("full_trajectory_accepted") is False,
            "wrong diagnostic mode or promoted trajectory claim")
    integer(execution.get("expected_epochs"), 3, "connected diagnostic must contain three bodies")
    require(isinstance(execution.get("boot_id"), str), "missing boot identity")
    uuid.UUID(execution["boot_id"])
    started, ended = (datetime.datetime.fromisoformat(execution[k]) for k in ("started_utc", "ended_utc"))
    require(started.tzinfo is not None and ended.tzinfo is not None and ended >= started,
            "invalid native dispatch time interval")
    resources = execution.get("main_compiler_resources", {})
    for key in ("stack", "cumulative_stack", "spill_stores", "spill_loads"):
        integer(resources.get(key), 0, "main/callee resource gate: " + key)
    require(type(resources.get("registers")) is int and 0 < resources["registers"] <= 128,
            "main register gate")
    integer(build.get("exit_code"), 0, "native build failed")
    require(build.get("sources_unchanged") is True and build.get("header_overlays") is False,
            "build is not an unchanged canonical-source build")
    require(build.get("canonical_source_tree") == str(ENGINE), "canonical build root mismatch")


def load_dispatch(files, execution_path, build_path):
    execution = files.document(execution_path)
    binding = execution["build_receipt"]
    require(Path(binding["path"]).resolve() == Path(build_path).resolve(), "dispatch points to a different build")
    build = files.document(build_path, binding["sha256"])
    receipt_fields(execution, build)
    command = execution["command"]
    require(isinstance(command, list) and len(command) == 6 and command[-1] == "--connected-work"
            and all(isinstance(x, str) for x in command), "unexpected native command")
    binary, case, selected, inverse, prefix = command[:5]
    require(all(Path(p).is_absolute() for p in command[:5]), "relative native command path")
    require(prefix.endswith("_state"), "connected output prefix identity")
    hashes, build_hashes = execution["input_hashes"], build["input_hashes"]
    require(isinstance(hashes, dict) and isinstance(build_hashes, dict) and build_hashes,
            "source/input hash population missing")
    for source, digest in build_hashes.items():
        require(hashes.get(source) == digest, "compiled input absent from dispatch: " + source)
    for rel in REQUIRED_SOURCES:
        require(str(ENGINE / rel) in build_hashes, "required compiled native source absent: " + rel)
    for name in ("nhs_biprism_variational_so3g_v10.cuh", "nhs_biprism_dec_pcg.cuh",
                 "nhs_biprism_electronic_discrete.cuh"):
        require(build["selected_sources"].get(name) == str(ENGINE / KERNELS / name),
                "selected scientific source mismatch: " + name)
    for p, expected in zip((case, selected, inverse), PHYSICAL_HASHES):
        require(hashes.get(p) == expected, "physical input differs from the retained KRAS input: " + p)
    require(hashes.get(binary) == build["binary_sha256"], "dispatch/build fatbinary mismatch")
    build_command = build["command"]
    require(build_command.count("-o") == 1 and
            Path(build_command[build_command.index("-o") + 1]).resolve() == Path(binary).resolve(),
            "build output is not the executed fatbinary")
    require("-arch=sm_120" in build_command, "build target is not sm_120")
    for path, digest in hashes.items():
        files.verify(path, digest)
    artifacts = execution["artifacts"]
    require(isinstance(artifacts, dict), "missing dispatch output bindings")

    def raw(suffix):
        path = prefix + suffix
        require(path in artifacts, "dispatch does not bind raw output: " + path)
        entry = artifacts[path]
        return files.read(path, entry["sha256"], entry["bytes"])

    return execution, build, command, raw


def window(raw, magic):
    require(len(raw) == 32, "window byte extent")
    u = struct.unpack("<8I", raw)
    require(u[0] == magic and u[1:5] == (3, 0, 3, 1) and u[5:] == (0, 0, 0),
            "raw window is not the cold three-body connected diagnostic")
    return u


def derive_indices(rows, ledgers, shared, raw_work_window, raw_time_window, terminal, final, component):
    w, t = window(raw_work_window, W.WORK_MAGIC), window(raw_time_window, W.SHARED_MAGIC)
    require(w[1:] == t[1:], "work/shared window identity mismatch")
    n = w[3]
    require(len(rows) == n*8*256*464 and len(ledgers) == n*8*288 and len(shared) == n*896,
            "incomplete actual work/time journal")
    require(len(terminal) == 8*48 and len(final) == 128 and len(component) == 64,
            "terminal/final/component extent mismatch")
    for body in range(n):
        step = w[4] + body
        u = struct.unpack_from("<8I", shared, body*896)
        require(u[:6] == (W.SHARED_MAGIC, 2, 0, step, step, body+1),
                "raw shared journal is not accepted")
        for leaf in range(8):
            lu = struct.unpack_from("<6I", ledgers, (body*8+leaf)*288)
            require(lu == (W.WORK_MAGIC, leaf, 3, 0, step, 4*(body+1)),
                    "raw work journal is not sealed")
    require(final == bytes(shared[(n-1)*896:(n-1)*896+128]), "final time owner differs from the last journal")
    control = struct.unpack("<16I", component)
    require(control == (0x42434331, 0, w[4]+n-1, n, 4*n, 0, *([0]*10)),
            "component descriptor mismatch or false native-AMBER clock claim")
    snapshots = [dict(zip(TERMINAL_KEYS, struct.unpack_from("<4I4f2i2I", terminal, leaf*48)))
                 for leaf in range(8)]
    require(all(s["terminal_step"] == n+1 for s in snapshots), "diagnostic terminal-step population mismatch")
    generation = struct.unpack_from("<Q", shared, 56)[0]
    require(generation == 23, "diagnostic allocation generation mismatch")
    before = struct.unpack_from("<d", shared, 48)[0] - float(struct.unpack_from("<f", shared, 32)[0])
    require(before == 0., "cold raw accepted-time origin mismatch")
    wi = {"schema": "biprism.lif.actual_work.v2", "attempted_current_body": False,
          "row_bytes": 464, "ledger_bytes": 288, "summary_fields": list(W.NAMES),
          "window": {"abi_magic": w[0], "capacity": w[1], "first_body": w[2],
                     "requested_steps": w[3], "first_scheduled_step": w[4], "reserved": list(w[5:])},
          "reserved_row_slots": len(rows)//464, "ledger_slots": len(ledgers)//288}
    si = {"schema": "biprism.shared_physical_time.v1", "record_bytes": 896,
          "window": {"magic": t[0], "capacity": t[1], "first_body": t[2], "requested_bodies": t[3],
                     "first_scheduled_step": t[4], "reserved": list(t[5:])},
          "records": len(shared)//896, "generation": generation, "before_time_ps": before,
          "terminal_steps": [s["terminal_step"] for s in snapshots], "terminal_snapshots": snapshots}
    return wi, si


def selected_population(raw_ids, tsv, atoms):
    require(len(raw_ids) == 8*256*4, "selected raw ID extent")
    flat = struct.unpack("<2048I", raw_ids)
    ids = [flat[i*256:(i+1)*256] for i in range(8)]
    source = [[None]*256 for _ in range(8)]
    reader = csv.DictReader(io.StringIO(tsv.decode()), delimiter="\t")
    require({"leaf", "vertex", "atom", "core"} <= set(reader.fieldnames or []), "selection TSV columns")
    count = 0
    for row in reader:
        leaf, vertex, atom, core = (int(row[k]) for k in ("leaf", "vertex", "atom", "core"))
        require(0 <= leaf < 8 and 0 <= vertex < 256 and 0 <= atom < atoms and
                core == int(vertex < 192) and source[leaf][vertex] is None, "selection TSV identity/duplicate")
        source[leaf][vertex] = atom
        count += 1
    require(count == 2048 and all(tuple(s) == actual and len(set(actual)) == 256
                                 for s, actual in zip(source, ids)), "raw selected IDs differ from the dispatched selection")
    require(all(row == ids[0] for row in ids), "connected diagnostic physical correspondence changed")
    return ids


def check_input_and_publication(rows, ids, case, inverse, retained_hmr, atom_work):
    require(case[:8] == b"BPCASE02" and len(case) >= 32, "physical input transport ABI")
    atoms = struct.unpack_from("<I", case, 8)[0]
    require(256 <= atoms <= 100000 and len(case) >= 32+20*atoms, "physical input extent")
    require(len(inverse) == len(retained_hmr) == 4*atoms and len(atom_work) == 8*atoms*8,
            "true/HMR mass or native atom-work extent")
    require(retained_hmr == case[32+16*atoms:32+20*atoms], "retained HMR masses differ from BPCASE02")
    expected_work = [0.]*(8*atoms)
    for i in range(len(rows)//464):
        body, rem = divmod(i, 8*256)
        leaf, vertex = divmod(rem, 256)
        atom = ids[leaf][vertex]
        require(atom < atoms, "selected atom exceeds physical input")
        off = i*464
        require(bytes(rows[off+32:off+36]) == inverse[4*atom:4*atom+4] and
                bytes(rows[off+44:off+48]) == retained_hmr[4*atom:4*atom+4],
                f"body={body} leaf={leaf} vertex={vertex}: recorded mass differs from actual input")
        expected_work[leaf*atoms+atom] += struct.unpack_from("<d", rows, off+248+7*8)[0]
    actual = struct.unpack("<"+str(8*atoms)+"d", atom_work)
    require(all(W.f64bits(a) == W.f64bits(b) for a, b in zip(actual, expected_work)),
            "native atom-work plane differs from the once-per-body actual row publication")


def check_endpoint_primitives(raw, rows, ids, atoms, numerical_bytes=38656):
    """Join recorded work primitives to the actual accepted C4 readback.

    These are provenance/bit-identity checks. They do not replace the main's
    independent equation, DEC, energy or represented-state acceptance gates.
    """
    for body in range(3):
        epoch = body+1
        numerical = raw(f".epoch{epoch}.numerical.bin")
        coords = raw(f".epoch{epoch}.au_coords.bin")
        low = raw(f".epoch{epoch}.au_coords_low.bin")
        native = raw(f".epoch{epoch}.post_bridge.native_coords_high.bin")
        native_low = raw(f".epoch{epoch}.post_bridge.native_coords_low.bin")
        require(numerical_bytes in (38656, 40704, 43008, 66048, 70656, 72960, 97536, 106752, 143872, 196992,198144,235136,349056,496768,510720,1680768,1697920,1784448,1837696,1997440), "unsupported compiled numerical schema")
        require(len(numerical) == 8*numerical_bytes and
                all(len(x) == 8*atoms*12 for x in (coords, low, native, native_low)),
                "accepted C4 numerical/coordinate extent")
        owner_intervals = []
        for frame, label in ((1, "native"), (2, "au")):
            owners = raw(f".epoch{epoch}.{label}_coordinate_owners.96b")
            require(owners == raw(f".epoch{epoch}.entry.{label}_coordinate_owners.96b") ==
                    raw(f".epoch{epoch}.post_bridge.{label}_coordinate_owners.96b"),
                    "coordinate allocation authority changed during the body")
            owner_intervals.extend(check_coordinate_owners(owners, atoms, frame, 23))
        owner_intervals.sort()
        require(all(b <= c for (_, b), (c, _) in zip(owner_intervals, owner_intervals[1:])),
                "native and AU coordinate planes alias")
        for leaf in range(8):
            for vertex in range(256):
                off = ((body*8+leaf)*256+vertex)*464
                co = (leaf*atoms+ids[leaf][vertex])*12
                require(bytes(rows[off+48:off+60]) == coords[co:co+12],
                        f"body={body} leaf={leaf} vertex={vertex}: work coordinate differs from actual accepted endpoint")
                require(bytes(rows[off+384:off+396]) == low[co:co+12],
                        f"body={body} leaf={leaf} vertex={vertex}: work coordinate low differs from accepted endpoint")
                require(bytes(rows[off+60:off+72]) == native[co:co+12] and
                        bytes(rows[off+396:off+408]) == native_low[co:co+12],
                        f"body={body} leaf={leaf} vertex={vertex}: native coordinate expansion differs from actual export")
                if vertex < 192:
                    no = leaf*numerical_bytes+vertex*4
                    velocity = b"".join(numerical[no+11520+axis*768:no+11524+axis*768] for axis in range(3))
                    metric = b"".join(numerical[no+34048+axis*768:no+34052+axis*768] for axis in range(6))
                    require(bytes(rows[off+72:off+84]) == velocity,
                            f"body={body} leaf={leaf} vertex={vertex}: recorded main velocity differs from accepted C4")
                    require(bytes(rows[off+148:off+172]) == metric,
                            f"body={body} leaf={leaf} vertex={vertex}: recorded metric differs from accepted C4")
                    if numerical_bytes>=106752:
                        precise=b"".join(numerical[leaf*numerical_bytes+97536+axis*1536+vertex*8:leaf*numerical_bytes+97544+axis*1536+vertex*8] for axis in range(6))
                        require(bytes(rows[off+416:off+464])==precise,
                                f"body={body} leaf={leaf} vertex={vertex}: FP64 metric differs from accepted C4")


def check_coordinate_owners(raw, atoms, frame, generation):
    require(len(raw) == 8*96, "coordinate owner extent")
    intervals = []
    for leaf in range(8):
        u = struct.unpack_from("<4IQ2I8Q", raw, leaf*96)
        require(u[:7] == (0x31504342, 1, leaf, atoms, generation, frame, 7)
                and u[-2:] == (0, 0), "coordinate owner version/frame/generation identity")
        for pointer in u[7:13]:
            require(pointer != 0 and pointer % 4 == 0 and pointer + atoms*12 <= 2**64,
                    "coordinate owner pointer domain")
            intervals.append((pointer, pointer+atoms*12))
    intervals.sort()
    require(all(b <= c for (_, b), (c, _) in zip(intervals, intervals[1:])),
            "coordinate planes alias within or across leaves")
    return intervals


def check_carry_and_publication(raw, rows, ledgers, shared, ids):
    previous = bytes(8*CARRY_BYTES)
    nonzero = False
    for body in range(3):
        epoch = body+1
        before = raw(f".epoch{epoch}.carry_before_delivery.bin")
        require(before == previous, "pending carry changed between actual producer and next capture")
        require(raw(f".epoch{epoch}.published_work.464b") == bytes(rows[body*8*256*464:(body+1)*8*256*464]) and
                raw(f".epoch{epoch}.published_work.288b") == bytes(ledgers[body*8*288:(body+1)*8*288]),
                "published work differs from retained journal")
        selected = bytearray(shared[body*896:body*896+128])
        struct.pack_into("<I", selected, 4, 1)
        struct.pack_into("<II", selected, 16, epoch-1, body)
        prior_time = 0. if body == 0 else struct.unpack_from("<d", shared, (body-1)*896+48)[0]
        struct.pack_into("<d", selected, 48, prior_time)
        require(raw(f".epoch{epoch}.selected_time.128b") == selected, "selected-time input differs from accepted-time journal")
        conv = raw(f".epoch{epoch}.conv.bin")
        require(len(conv) == 8*128, "accepted-main convergence extent")
        for leaf in range(8):
            require(struct.unpack_from("<6I", conv, leaf*128) == (0x42505631, leaf, 0, 0, 0, 4*epoch),
                    "work is not bound to this actual accepted-main commit identity")
            for vertex in range(256):
                force = before[leaf*CARRY_BYTES+1056+vertex*16:leaf*CARRY_BYTES+1068+vertex*16]
                off = ((body*8+leaf)*256+vertex)*464+136
                require(bytes(rows[off:off+12]) == force, "row requested force differs from actual pending carry")
        produced = raw(f".epoch{epoch}.produced_carry.bin")
        require(len(produced) == 8*CARRY_BYTES, "actual produced carry extent")
        for leaf in range(8):
            off = leaf*CARRY_BYTES
            u = struct.unpack_from("<8I", produced, off)
            require(u == (1, epoch, epoch if body else 0, 0, 4*epoch, 0, 0, 0), "actual carry publication chronology")
            require(struct.unpack_from("<256I", produced, off+32) == ids[leaf], "actual carry atom correspondence")
            sums = [W.F(0)]*3
            for vertex in range(256):
                f = struct.unpack_from("<4f", produced, off+1056+vertex*16)
                W.finite(f, "actual carry force")
                require(W.f32bits(f[3]) == 0, "carry reserved force component is not zero")
                for axis in range(3):
                    sums[axis] += W.F(f[axis])
                    nonzero = nonzero or f[axis] != 0.
            require(sums == [0, 0, 0], "actual represented producer force is not exactly balanced")
        previous = produced
    require(nonzero, "actual producer has no nonzero force coverage")
    require(raw(".lif_work.carry.final.bin") == previous, "final carry differs from the last actual producer")


def check(execution_path, build_path):
    files = Files()
    # Retain the exact independently executed oracle and adapter identities too.
    files.verify(Path(__file__).resolve())
    files.verify(Path(W.__file__).resolve())
    execution, build, command, raw = load_dispatch(files, execution_path, build_path)
    _, case_path, selection_path, inverse_path, _ = command[:5]
    rows, ledgers, shared = (raw(s) for s in (".lif_work.rows.464b", ".lif_work.ledgers.288b", ".shared_time.records.896b"))
    wi, si = derive_indices(rows, ledgers, shared, raw(".lif_work.window.32b"), raw(".shared_time.window.32b"),
                            raw(".shared_time.terminal.48b"), raw(".shared_time.final.128b"), raw(".component_control.64b"))
    case, inverse = files.read(case_path), files.read(inverse_path)
    require(len(case) >= 32, "truncated physical input")
    atoms = struct.unpack_from("<I", case, 8)[0]
    ids = selected_population(raw(".lif_work.selected_ids.u32le"), files.read(selection_path), atoms)
    check_input_and_publication(rows, ids, case, inverse, raw(".lif_work.hmr_masses.f32le"),
                                raw(".lif_work.atom_work.kcal.f64le"))
    main_path = build["selected_sources"]["nhs_biprism_variational_so3g_v10.cuh"]
    main_source = files.read(main_path, build["input_hashes"][main_path]).decode()
    sizes = re.findall(r"static_assert\(sizeof\(BiprismV10NumericalScratchFfi\)\s*==\s*(\d+)", main_source)
    require(len(sizes) == 1 and int(sizes[0]) in (38656, 40704, 43008, 66048, 70656, 72960, 97536, 106752, 143872, 196992,198144,235136,349056,496768,510720,1680768,1697920,1784448,1837696,1997440), "compiled numerical ABI assertion")
    numerical_bytes = int(sizes[0])
    if numerical_bytes in (40704, 43008, 66048, 70656, 72960, 97536, 106752, 143872, 196992,198144,235136,349056,496768,510720,1680768,1697920,1784448,1837696,1997440):
        potential_path = str(ENGINE / KERNELS / "nhs_biprism_potential_precision.cuh")
        require(potential_path in build["input_hashes"], "missing compiled full-potential owner")
        files.verify(potential_path, build["input_hashes"][potential_path])
    check_endpoint_primitives(raw, rows, ids, atoms, numerical_bytes)
    check_carry_and_publication(raw, rows, ledgers, shared, ids)
    result = W.check_buffers(rows, ledgers, shared, wi, si, ids)
    require(any(struct.unpack_from("<d", ledgers, i*288+72+7*8)[0] != 0. or
                struct.unpack_from("<d", ledgers, i*288+72+1*8)[0] != 0. for i in range(24)),
            "actual work journal has no nonzero delivery-work coverage")
    files.unchanged()
    result.update(boot_id=execution["boot_id"], binary_sha256=build["binary_sha256"],
                  bound_files=len(files.identities), full_AMBER_schedule_inferred=False,
                  mixed_energy_acceptance_inferred=False, terminal_hold_qualification_inferred=False,
                  independent_DEC_requirement="separate guarded-runner result")
    identities = [(str(Path(p).resolve()), files.identities[str(Path(p).resolve())]["sha256"])
                  for p in (execution_path, build_path, __file__, W.__file__)]
    identities += [(str(ENGINE/rel), build["input_hashes"][str(ENGINE/rel)]) for rel in REQUIRED_SOURCES]
    return result, identities


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", required=True, type=Path)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="New plain-text result path; existing evidence is never overwritten.")
    args = parser.parse_args(argv)
    if args.output:
        require(args.output.suffix == ".txt" and not args.output.exists(), "output must be a new plain-text .txt file")
    try:
        result, identities = check(args.execution.resolve(), args.build.resolve())
        lines = ["PASS: independent connected main/LIF work readback",
                 *(f"{key}={value}" for key, value in result.items()),
                 *(f"source_or_receipt={path} sha256={digest}" for path, digest in identities),
                 "scope=three actual main/LIF component bodies; complete raw work/time population; exact primitive work and lab momentum",
                 "not_inferred=full AMBER trajectory, mixed-energy admission, terminal hold, full C4 qualification or model validity"]
        code = 0
    except (W.Refusal, KeyError, TypeError, ValueError, OSError, struct.error, OverflowError) as exc:
        lines = [f"REFUSED: {exc}", "No connected-work qualification was issued."]
        code = 1
    output = "\n".join(lines)+"\n"
    sys.stdout.write(output)
    if args.output:
        with args.output.open("x") as stream:
            stream.write(output)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
