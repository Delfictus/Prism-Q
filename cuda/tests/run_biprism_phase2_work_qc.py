#!/usr/bin/env python3
"""Build or run the actual seven-case Phase2 fixed-state work control.

Builds use the active canonical headers, without overlays. Runs use the same
nonprivileged GPU lock and boot/Xid checks as run_biprism_v10_coupled_qc.py.
The independent primitive-row checker consumes the retained native outputs.
These controls do not advance an accepted physical clock or launch v10/AMBER.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import struct
import subprocess
import sys

def _require_root(name):
    """Absolute root from the environment; there is no compiled-in host default."""
    value = os.environ.get(name, '')
    if not os.path.isabs(value):
        raise SystemExit(f'{name} must name an absolute path (no compiled-in host default)')
    return Path(value)



ROLE = Path(__file__).resolve().parents[2]
OUTPUT = _require_root("PRISM_BIPRISM_RESOLUTION_ROOT")
TASK = _require_root("PRISM_BIPRISM_TASK_ROOT")
SOURCE = ROLE / "cuda/tests/biprism_phase2_work_native_qc.cu"
CHECKER = ROLE / "cuda/tests/check_biprism_phase2_work_native.py"
MESH_CHECKER = ROLE / "cuda/tests/check_biprism_mesh_sensitivity.py"
KERNELS = ROLE / "crates/prism-gpu/src/kernels"
XTB = KERNELS / "nhs_biprism_xtb_lut.cuh"
PYTHON = _require_root("PRISM_REFERENCE_PYTHON")
CASE = TASK / "INTERNAL_QC/v10_multiplicity_revision/native/host_controls/KRAS_INPUT.bpqc"
SEED = TASK / "INTERNAL_QC/v10_engine_integration/reduction_synchronization/native_qualification_001/state.seed_tiles.bin"
INPUTS = {
    str(CASE): "eff6c982d8cdbd9f77e6e4ecf37852fb33ca486712747079e3a0e42efaf5bd54",
    str(SEED): "5ed733573d17f43fde83884016698271f9316791f725d1e15584a762de7e52db",
}
CASES = ("cold_placeholder", "cold_leaf_refusal", "unchanged", "deformed", "boundary", "identity_refusal", "low_plane")
ENTRIES = (
    "nhs_biprism_import_native_state_v1", "nhs_biprism_guard_unit_import_v1",
    "nhs_biprism_simplicial_tile_compensated_v1", "nhs_biprism_phase2_work_initialize_v1",
    "nhs_biprism_phase2_work_initialize_publish_v1",
    "nhs_biprism_phase2_work_capture_v1", "nhs_biprism_phase2_work_measure_v1",
    "nhs_biprism_phase2_work_publish_v1",
)
MESH_ENTRIES = (
    "nhs_biprism_import_native_state_v1", "nhs_biprism_guard_unit_import_v1",
    "nhs_seed_bowyer_watson_256_compensated_v1",
    "nhs_biprism_simplicial_tile_compensated_v1", "nhs_biprism_phase2_work_initialize_v1",
    "nhs_biprism_phase2_work_initialize_publish_v1",
)
BOUNDARY = (
    "Seven actual Phase2 fixed-state operator/boundary-work controls, including a low-plane-only displacement "
    "and a single-leaf cold-publication refusal, on the "
    "retained KRAS BPCASE02 and cold 960-tetrahedron/1461-edge seed. No v10 "
    "propagation, accepted physical-time update, or full AMBER trajectory."
)
MESH_BOUNDARY = (
    "Three diagnostic BW tetrahedron budgets on identical KRAS selected atoms and imported hi+lo "
    "coordinates, each followed by actual Phase2/cold DEC initialization. Quantifies changes in "
    "the retained domain and graph operator; no fixed-domain refinement, convergence order, "
    "physical trajectory, production-target change or new tolerance is asserted."
)


def profile(args):
    if args.mesh_budgets is not None:
        return ("failure_5_mesh_sensitivity", "biprism_mesh_sensitivity_qc", MESH_CHECKER,
                MESH_ENTRIES, MESH_BOUNDARY)
    return ("failure_7_phase2_native", "biprism_phase2_work_native_qc", CHECKER, ENTRIES, BOUNDARY)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def stable(pins):
    return all(Path(p).is_file() and sha(p) == h for p, h in pins.items())


def run_logged(command, prefix, timeout, *, env=None):
    with Path(str(prefix) + ".stdout").open("x") as out, Path(str(prefix) + ".stderr").open("x") as err:
        try:
            return subprocess.run(command, cwd=ROLE, stdout=out, stderr=err,
                                  timeout=timeout, env=env).returncode
        except subprocess.TimeoutExpired:
            return 124
        except OSError as exc:
            err.write(repr(exc) + "\n")
            return 127


def resources(text):
    pattern = (r"Function properties for (\S+)\s*\n\s*"
               r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads"
               r"\s*\nptxas info\s*: Used (\d+) registers([^\n]*)")
    result = {}
    for name, stack, stores, loads, regs, tail in re.findall(pattern, text):
        cumulative = re.search(r"(\d+) bytes cumulative stack size", tail)
        result[name] = dict(registers=int(regs), stack=int(stack),
                            spill_stores=int(stores), spill_loads=int(loads),
                            cumulative_stack=int(cumulative.group(1)) if cumulative else int(stack))
    return result


def build(args):
    stem, binary_stem, checker, entries, boundary = profile(args)
    prefix = args.output / (stem + "_build_" + args.attempt)
    binary = args.output / (binary_stem + "_" + args.attempt)
    require(not list(args.output.glob(prefix.name + ".*")) and not binary.exists(),
            "Build attempt already exists; retained builds are immutable")
    flags = ["/usr/local/cuda/bin/nvcc", "-arch=sm_120", "-O3", "--restrict",
             "-std=c++17", "-lineinfo", "-I" + str(KERNELS),
             "-I" + str(ROLE / "cuda"), "-I" + str(SOURCE.parent), "-include", "stdint.h"]
    dependency_command = flags + ["-M", str(SOURCE)]
    dependency_prefix = Path(str(prefix) + ".dependencies")
    code = run_logged(dependency_command, dependency_prefix, 60)
    receipt = dict(started_utc=now(), canonical_source_tree=str(ROLE), source=str(SOURCE),
                   dependency_command=dependency_command, dependency_exit_code=code,
                   header_overlays=False, native_execution=False, scientific_acceptance=False,
                   boundary=boundary, mesh_budgets=args.mesh_budgets)
    receipt_path = Path(str(prefix) + ".json")
    if code:
        receipt.update(exit_code=code, ended_utc=now())
        write_json(receipt_path, receipt)
        return code
    dependency_text = Path(str(dependency_prefix) + ".stdout").read_text()
    paths = shlex.split(dependency_text.replace("\\\n", " ").split(":", 1)[1])
    paths = [Path(p) if Path(p).is_absolute() else ROLE / p for p in paths]
    pins = {str(p.resolve()): sha(p) for p in paths}
    for p in (SOURCE, XTB, KERNELS / "nhs_biprism_phase2_work.cuh"):
        require(str(p) in pins, "Actual compilation dependency missing: " + str(p))
    pins.update({str(Path(__file__).resolve()): sha(__file__), str(checker): sha(checker)})
    if args.mesh_budgets is not None:
        for name in ("check_biprism_v10_dec_readback.py", "check_biprism_boundary_identity.py",
                     "check_biprism_v10_accepted_prefix.py"):
            p = SOURCE.parent / name
            pins[str(p)] = sha(p)
    command = flags + ["-Xptxas=-v", str(SOURCE), "-lcuda", "-o", str(binary)]
    receipt.update(command=command, input_hashes=pins, binary=str(binary),
                   checker=str(checker), xtb_source=str(XTB),
                   compiler_sha256=sha(Path(flags[0]).resolve()))
    write_json(receipt_path, receipt)
    code = run_logged(command, prefix, 300)
    raw_resources = resources(Path(str(prefix) + ".stderr").read_text())
    present = all(name in raw_resources for name in entries)
    receipt.update(exit_code=code, sources_unchanged=stable(pins),
                   invoked_entry_resources={k: raw_resources.get(k) for k in entries},
                   invoked_entry_reports_present=present, ended_utc=now(),
                   compiler_outputs={str(prefix) + suffix: sha(str(prefix) + suffix)
                                     for suffix in (".stdout", ".stderr")})
    if code == 0 and binary.is_file():
        receipt["binary_sha256"] = sha(binary)
    write_json(receipt_path, receipt)
    print(json.dumps({"exit_code": code, "sources_unchanged": receipt["sources_unchanged"],
                      "invoked_entry_reports_present": present, "receipt": str(receipt_path)}))
    return code or (0 if receipt["sources_unchanged"] and present else 3)


def expected_outputs(prefix, atoms):
    common = {"seed_tiles.bin": 8 * 225408, "initial_tiles.bin": 8 * 225408, "provisional_tiles.bin": 8 * 225408,
              "initial_ledgers.240b": 8 * 240, "initial_convergence.128b": 8 * 128,
              "initial_step.u32le": 4, "initial_vertices.256b": 8 * 256 * 256,
              "initial_potential_owners.128b": 8 * 128,
              "initial_phi_accepted.f64le": 8 * 256 * 8, "initial_phi_prepared.f64le": 8 * 256 * 8}
    measured = {"before_tiles.bin": 8 * 225408, "coordinates_au.f32le": 8 * atoms * 12,
                "after_tiles.bin": 8 * 225408, "ledgers.240b": 8 * 240,
                "vertices.256b": 8 * 256 * 256, "edges.64b": 8 * 1536 * 64,
                "before_potential_owners.128b": 8 * 128, "after_potential_owners.128b": 8 * 128,
                "before_phi_accepted.f64le": 8 * 256 * 8, "after_phi_accepted.f64le": 8 * 256 * 8,
                "before_phi_prepared.f64le": 8 * 256 * 8, "after_phi_prepared.f64le": 8 * 256 * 8,
                "coordinates_au_low.f32le": 8 * atoms * 12,
                "reference_au.f32le": 8 * atoms * 12, "reference_au_low.f32le": 8 * atoms * 12,
                "coordinates_native.f32le": 8 * atoms * 12, "coordinates_native_low.f32le": 8 * atoms * 12,
                "journal.240b": 8 * 240, "convergence.128b": 8 * 128,
                "shared.128b": 128, "window.48b": 48, "scheduled_step.u32le": 4}
    return {str(prefix) + "." + case + "." + suffix: size for case in CASES
            for suffix, size in (common if case in ("cold_placeholder", "cold_leaf_refusal") else {**common, **measured}).items()}


def mesh_expected_outputs(prefix, atoms, budgets):
    extents = {"requested_target.u32le": 4, "selected_atoms.u32le": 8*256*4,
               "bw_status.80b": 8*80, "seed_tiles.bin": 8*225408,
               "phase2_tiles.bin": 8*225408, "initial_tiles.bin": 8*225408, "provisional_tiles.bin": 8*225408,
               "initial_ledgers.240b": 8*240, "initial_vertices.256b": 8*256*256,
               "initial_potential_owners.128b": 8*128,
               "initial_phi_accepted.f64le": 8*256*8, "initial_phi_prepared.f64le": 8*256*8,
               "initial_convergence.128b": 8*128, "initial_step.u32le": 4,
               "coordinate_owners_native.96b": 8*96, "coordinate_owners_au.96b": 8*96,
               "bridge.64b": 8*64}
    for name in ("coordinates_au", "coordinates_au_low", "reference_au", "reference_au_low",
                 "coordinates_native", "coordinates_native_low"):
        extents[name+".f32le"] = 8*atoms*12
    return {str(prefix)+".target"+str(target)+"."+suffix: size
            for target in budgets for suffix, size in extents.items()}


def dispatch(args):
    stem, binary_stem, checker, entries, boundary = profile(args)
    prefix = args.output / (stem + "_" + args.attempt)
    require(not list(args.output.glob(prefix.name + "*")), "Run attempt already exists")
    build_path = args.output / (stem + "_build_" + args.attempt + ".json")
    built = json.loads(build_path.read_text())
    require(built.get("exit_code") == 0 and built.get("sources_unchanged") is True
            and built.get("invoked_entry_reports_present") is True, "Actual source build is not qualified")
    require(built["canonical_source_tree"] == str(ROLE) and built["source"] == str(SOURCE)
            and built["header_overlays"] is False and built["checker"] == str(checker)
            and built["xtb_source"] == str(XTB), "Build does not bind the actual canonical path")
    require(built.get("mesh_budgets") == args.mesh_budgets, "Build/run diagnostic target population differs")
    binary = args.output / (binary_stem + "_" + args.attempt)
    require(built["binary"] == str(binary), "Build binary path differs")
    pins = {**built["input_hashes"], **built["compiler_outputs"], **INPUTS,
            str(binary): built["binary_sha256"], str(build_path): sha(build_path),
            str(args.python): sha(args.python)}
    require(stable(pins), "Source, binary, checker, interpreter or input pin differs")
    with CASE.open("rb") as stream:
        header = stream.read(12)
    require(header[:8] == b"BPCASE02", "Actual input transport identity differs")
    atoms, = struct.unpack_from("<I", header, 8)
    state = Path(str(prefix) + "_state")
    command = [str(binary), str(CASE), str(SEED), str(state)]
    if args.mesh_budgets is not None:
        command.append("--mesh-budgets="+",".join(map(str, args.mesh_budgets)))
    receipt_path = Path(str(prefix) + "_execution.json")
    receipt = dict(started_utc=now(), command=command, input_hashes=pins,
                   build_receipt={"path": str(build_path), "sha256": sha(build_path)},
                   boundary=boundary, expected_cases=(list(CASES) if args.mesh_budgets is None else
                       ["target"+str(t) for t in args.mesh_budgets]),
                   mesh_budgets=args.mesh_budgets, native_execution=False,
                   physical_trajectory_acceptance=False, accepted_physical_time_not_advanced=None,
                   installed_state_mutation=False, production_admission_reset=False,
                   privileged_lease_claim=False, historical_faults_preserved=True,
                   authorization="Operator-authorized nonprivileged native diagnostic; root dispatch lane.",
                   invoked_entry_resources=built["invoked_entry_resources"])
    lock_path = "/tmp/prism-foundation-gpu-%d.lock" % os.getuid()
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    locked = False
    exit_code = 3
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        receipt["cooperative_gpu_lock"] = lock_path
        receipt["boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        write_json(receipt_path, receipt)

        def capture(name, cmd):
            p = Path(str(prefix) + "_" + name)
            code = run_logged(cmd, p, 30)
            return code, Path(str(p) + ".stdout").read_text()

        def journal(name):
            code, text = capture(name, ["journalctl", "-k", "-b", "--grep", "NVRM|Xid", "-o", "json", "--no-pager"])
            require(code == 0, "Kernel journal unavailable: " + name)
            return [json.loads(s) for s in text.splitlines() if s.startswith("{")]

        def health(name):
            code, text = capture(name, ["nvidia-smi", "--query-gpu=name,uuid,driver_version,gpu_recovery_action,memory.used,utilization.gpu", "--format=csv,noheader"])
            return code == 0 and bool(text.strip()) and all(", None," in row for row in text.splitlines() if row.strip())

        before = journal("journal_before")
        require(health("gpu_before"), "GPU recovery state is not None")
        code, apps = capture("compute_before", ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"])
        require(code == 0 and all("/usr/libexec/gnome-remote-desktop-daemon" in row
                                for row in apps.splitlines() if row.strip()), "GPU has an unrelated compute owner")
        code, _ = capture("checker_environment", [str(args.python), "-I", "-B", "-c",
            "import numpy, sys; print(sys.version); print(numpy.__version__); assert not sys.flags.optimize"])
        require(code == 0, "Independent checker environment unavailable")
        require(stable(pins), "Inputs changed before dispatch")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        receipt["native_started_utc"] = now()
        write_json(receipt_path, receipt)
        native_code = run_logged(command, prefix, 180, env=env)
        receipt.update(native_execution=native_code != 127, native_dispatch_attempted=True,
                       native_exit_code=native_code, native_ended_utc=now())
        after = journal("journal_after")
        old = {row.get("__CURSOR") for row in before}
        new = [row for row in after if row.get("__CURSOR") not in old]
        receipt.update(new_kernel_fault_records=new, historical_kernel_records_retained=len(before),
                       gpu_recovery_none_after=health("gpu_after"),
                       boot_unchanged=receipt["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                       sources_binary_inputs_unchanged=stable(pins),
                       artifacts={str(p): {"sha256": sha(p), "bytes": p.stat().st_size}
                                  for p in args.output.glob(prefix.name + "*")
                                  if p.is_file() and p != receipt_path})
        dispatch_path = Path(str(prefix) + "_dispatch.json")
        write_json(dispatch_path, receipt)
        receipt["dispatch_receipt"] = {"path": str(dispatch_path), "sha256": sha(dispatch_path)}
        exit_code = native_code or 2
        require(native_code == 0 and not new and receipt["gpu_recovery_none_after"]
                and receipt["boot_unchanged"] and receipt["sources_binary_inputs_unchanged"],
                "Native execution or post-dispatch identity/health check refused")
        expected = (expected_outputs(state, atoms) if args.mesh_budgets is None else
                    mesh_expected_outputs(state, atoms, args.mesh_budgets))
        for path, size in expected.items():
            require(Path(path).is_file() and Path(path).stat().st_size == size,
                    "Missing or wrong-sized native output: " + path)
        if args.mesh_budgets is None:
            for case in CASES:
                require(sha(str(state) + "." + case + ".seed_tiles.bin") == INPUTS[str(SEED)],
                        "Native case did not retain the actual cold seed: " + case)
            checker_command = [str(args.python), "-I", "-B", str(checker), "--native-prefix", str(state),
                               "--xtb-source", str(XTB)]
        else:
            checker_command = [str(args.python), "-I", "-B", str(checker), "--execution", str(dispatch_path),
                               "--budgets", *map(str, args.mesh_budgets)]
        check_prefix = Path(str(prefix) + "_independent")
        check_code = run_logged(checker_command, check_prefix, 180)
        unchanged = stable(pins) and all(Path(p).is_file() and sha(p) == a["sha256"]
                                        for p, a in receipt["artifacts"].items())
        receipt["independent_check"] = {"command": checker_command, "exit_code": check_code,
            "checker_sha256": sha(checker), "outputs_unchanged": unchanged,
            "stdout_sha256": sha(str(check_prefix) + ".stdout"),
            "stderr_sha256": sha(str(check_prefix) + ".stderr")}
        key = "phase2_measurement_controls_passed" if args.mesh_budgets is None else "mesh_sensitivity_quantified"
        receipt[key] = check_code == 0 and unchanged
        if receipt[key]:
            receipt["accepted_physical_time_not_advanced"] = True
        exit_code = 0 if receipt[key] else 2
    except Exception as exc:
        receipt["error"] = repr(exc)
        receipt["phase2_measurement_controls_passed" if args.mesh_budgets is None else "mesh_sensitivity_quantified"] = False
    finally:
        receipt.update(exit_code=exit_code, ended_utc=now())
        if locked:
            try:
                write_json(receipt_path, receipt)
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    print(json.dumps({"exit_code": exit_code, "receipt": str(receipt_path) if locked else None,
                      "error": receipt.get("error")}))
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "run"))
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--python", type=Path, default=PYTHON,
                        help="Independent checker's NumPy-enabled interpreter (run only)")
    parser.add_argument("--mesh-budgets", type=int, nargs=3,
                        help="Diagnostic fresh native targets below960,960,above960; production settings stay unchanged")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9]{3}", args.attempt):
        parser.error("attempt must contain three decimal digits")
    if args.mesh_budgets is not None and not (0 < args.mesh_budgets[0] < 960
            and args.mesh_budgets[1] == 960 < args.mesh_budgets[2] <= 1024):
        parser.error("mesh budgets must bracket960 within the native1..1024 capacity")
    args.output = args.output.resolve()
    args.python = args.python.absolute()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        return build(args) if args.action == "build" else dispatch(args)
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
