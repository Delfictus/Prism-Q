#!/usr/bin/env python3
"""Replay retained017 acceptance inputs through the actual build019 export.

Only the CPU driver is compiled. cuobjdump extracts the CUDA module from the
retained executable; the exported instruction words must match that executable.
--prepare-only performs extraction, host compilation and CPU controls, with no
CUDA initialization or GPU launch. A later invocation reuses the immutable
preparation and dispatches 133 cases under the existing nonprivileged GPU lock.
"""
import argparse
import csv
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import struct
import subprocess
import sys
import tempfile

def _require_root(name):
    """Absolute root from the environment; there is no compiled-in host default."""
    value = os.environ.get(name, '')
    if not os.path.isabs(value):
        raise SystemExit(f'{name} must name an absolute path (no compiled-in host default)')
    return Path(value)



ROOT = Path(__file__).resolve().parents[2]
OUTPUT = _require_root("PRISM_BIPRISM_RESOLUTION_ROOT")
KERNELS = ROOT / "crates/prism-gpu/src/kernels"
BUILD = OUTPUT / "failure_1_2_3_coupled_native_build_019.json"
BINARY = OUTPUT / "biprism_coupled_native_qc_019"
BINARY_SHA256 = "8b70727c4d8a564646b0b9b9086e3501f276d6cf2f2dca7623b8289f012934e2"
HISTORICAL_DISPATCH = OUTPUT / "failure_1_2_3_coupled_native_017_connected_work_dispatch.json"
FIXTURE = OUTPUT / "failure_1_2_3_coupled_native_017_connected_work_state.epoch1.time_accept.before"
DRIVER_SOURCE = ROOT / "cuda/tests/biprism_shared_accept_control.cpp"
CUOBJDUMP = Path("/usr/local/cuda/bin/cuobjdump")
CXX = Path("/usr/bin/g++")
CUDA_INCLUDE = Path("/usr/local/cuda/include")
EXPORT = "nhs_biprism_shared_step_accept_v1"
EXPECTED_CASES = ["retained017", *[f"journal_byte_{i}" for i in range(128)],
                  "continuation", "nonincreasing_time", "accepted_counter_mismatch",
                  "journal_generation_mismatch"]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(path):
    path = Path(path)
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def write_json(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def unchanged(pins):
    return all(Path(path).is_file() and sha(path) == digest for path, digest in pins.items())


def capture(prefix, name, command, *, cwd=ROOT, timeout=30, env=None):
    stdout_path = Path(str(prefix) + "_" + name + ".stdout")
    stderr_path = Path(str(prefix) + "_" + name + ".stderr")
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        try:
            result = subprocess.run([str(arg) for arg in command], cwd=cwd, env=env,
                                    stdout=stdout, stderr=stderr, timeout=timeout)
            code, timed_out = result.returncode, False
        except subprocess.TimeoutExpired:
            code, timed_out = 124, True
    return {"command": [str(arg) for arg in command], "exit_code": code,
            "timed_out": timed_out, "stdout": identity(stdout_path),
            "stderr": identity(stderr_path)}


def output_text(result):
    return Path(result["stdout"]["path"]).read_text()


def successful(result, label):
    require(result["exit_code"] == 0, f"{label} failed; inspect {result['stderr']['path']}")
    return output_text(result)


def export_words(text):
    """Bind encoded SASS words, independent of printed headers/whitespace."""
    markers = list(re.finditer(r"Function\s*:\s*" + re.escape(EXPORT) + r"\b", text))
    require(len(markers) == 1, "Missing or ambiguous actual acceptance export")
    body = text[markers[0].end():]
    body = re.split(r"\n\s*(?:Function\s*:|Fatbin elf code:)", body, maxsplit=1)[0]
    words = re.findall(r"/\*\s*(0x[0-9a-fA-F]+)\s*\*/", body)
    require(words and "EF_CUDA_SM120" in body, "Actual export is not an sm120 SASS body")
    return b"".join(struct.pack("<Q", int(word, 16)) for word in words)


def source_and_fixture_pins():
    build = json.loads(BUILD.read_text())
    require(build.get("exit_code") == 0 and build.get("sources_unchanged") is True,
            "Build019 did not complete with stable sources")
    require(build.get("canonical_source_tree") == str(ROOT)
            and build.get("header_overlays") is False, "Build019 is not the canonical source build")
    require(build.get("binary_sha256") == BINARY_SHA256 and sha(BINARY) == BINARY_SHA256,
            "Actual019 executable identity mismatch")
    require(build["command"][-2:] == ["-o", str(BINARY)]
            and "-arch=sm_120" in build["command"], "Unexpected build019 output/architecture")
    pins = dict(build["input_hashes"])
    for name in ("nhs_biprism_shared_step.cuh", "nhs_biprism_shared_step_abi.cuh"):
        require(str(KERNELS / name) in pins, f"Build019 is missing {name}")
    require(unchanged(pins), "A build019 source/dependency changed")

    historical = json.loads(HISTORICAL_DISPATCH.read_text())
    require(historical.get("native_execution") is True
            and historical.get("native_exit_code") == 1
            and historical.get("sources_binary_inputs_unchanged") is True,
            "Retained017 tuple lacks its actual stable failed dispatch")
    lineage = historical["build_receipt"]
    require(sha(lineage["path"]) == lineage["sha256"], "Historical017 build receipt changed")
    historical_build = json.loads(Path(lineage["path"]).read_text())
    require(historical_build.get("exit_code") == 0 and historical_build.get("sources_unchanged") is True,
            "Historical017 build was not source stable")
    historical_binary = Path(historical["command"][0])
    require(sha(historical_binary) == historical_build["binary_sha256"],
            "Historical017 executable lineage changed")
    expected_sizes = {".shared.128b": 128, ".window.32b": 32, ".control.64b": 64,
                      ".convergence.128b": 1024, ".scheduled.u32le": 4}
    window = Path(str(FIXTURE) + ".window.32b").read_bytes()
    require(len(window) == 32, "Retained017 window extent")
    capacity = struct.unpack_from("<I", window, 4)[0]
    require(0 < capacity <= 500, "Retained017 journal capacity")
    expected_sizes[".journal.896b"] = capacity * 896
    fixture_records = {}
    for suffix, size in expected_sizes.items():
        path = Path(str(FIXTURE) + suffix)
        recorded = historical["artifacts"][str(path)]
        require(path.stat().st_size == size == recorded["bytes"]
                and sha(path) == recorded["sha256"], f"Retained017 tuple changed: {path}")
        pins[str(path)] = recorded["sha256"]
        fixture_records[str(path)] = identity(path)
    for path in (BUILD, BINARY, HISTORICAL_DISPATCH, Path(lineage["path"]), historical_binary,
                 DRIVER_SOURCE, Path(__file__).resolve(), CUOBJDUMP, CXX):
        pins[str(path)] = sha(path)
    return pins, fixture_records


def prepare(prefix):
    receipt_path = Path(str(prefix) + "_prepare.json")
    if receipt_path.exists():
        prepared = json.loads(receipt_path.read_text())
        require(prepared["status"] == "prepared" and unchanged(prepared["input_hashes"]),
                "Prepared source/binary/input identity changed")
        require(all(sha(path) == item["sha256"] and Path(path).stat().st_size == item["bytes"]
                    for path, item in prepared["artifacts"].items()), "Prepared artifact changed")
        return prepared
    require(not list(OUTPUT.glob(prefix.name + "*")), "Attempt output already exists; choose a fresh attempt")
    pins, fixture_records = source_and_fixture_pins()
    commands = []
    receipt = {"started_utc": now(), "status": "preparing", "input_hashes": pins,
               "build_receipt": identity(BUILD), "fixture_dispatch": identity(HISTORICAL_DISPATCH),
               "fixture_prefix": str(FIXTURE), "fixture_raw_files": fixture_records,
               "script_sha256": sha(__file__), "native_execution": False,
               "cuda_source_compilation": False, "header_overlays": False, "commands": commands}
    try:
        listed = capture(prefix, "elf_listing", [CUOBJDUMP, "--list-elf", BINARY])
        commands.append(listed)
        require("sm_120.cubin" in successful(listed, "ELF listing"), "No actual sm120 image")
        binary_dump = capture(prefix, "binary_export_sass", [CUOBJDUMP, "--gpu-architecture", "sm_120",
                              "--dump-sass", "--function", EXPORT, BINARY])
        commands.append(binary_dump)
        binary_words = export_words(successful(binary_dump, "Executable export SASS"))
        candidates = []
        # cuobjdump may list duplicate names; verify the one exported function
        # against the whole executable, rather than trusting extraction order.
        with tempfile.TemporaryDirectory(prefix="biprism_accept_extract_") as directory:
            extracted = capture(prefix, "extract", [CUOBJDUMP, "--extract-elf", "sm_120", BINARY], cwd=directory)
            commands.append(extracted)
            successful(extracted, "Actual ELF extraction")
            images = sorted(Path(directory).glob("*.cubin"))
            require(images, "cuobjdump extracted no cubin")
            for index, image in enumerate(images):
                retained = Path(str(prefix) + f"_image_{index:03d}.sm_120.cubin")
                with retained.open("xb") as stream:
                    stream.write(image.read_bytes())
                dump = capture(prefix, f"image_{index:03d}_export_sass", [CUOBJDUMP,
                               "--gpu-architecture", "sm_120", "--dump-sass", "--function", EXPORT, retained])
                commands.append(dump)
                text = successful(dump, "Extracted export SASS")
                if re.search(r"Function\s*:\s*" + re.escape(EXPORT) + r"\b", text):
                    require(export_words(text) == binary_words, "Extracted export instruction words differ")
                    candidates.append({**identity(retained), "cuobjdump_original_name": image.name})
        require(len(candidates) == 1, "Missing or ambiguous extracted acceptance module")
        receipt["actual_module"] = candidates[0]
        receipt["export"] = EXPORT
        receipt["export_instruction_words_sha256"] = hashlib.sha256(binary_words).hexdigest()
        receipt["export_instruction_word_bytes"] = len(binary_words)

        flags = ["-std=c++17", "-O2", "-DBIPRISM_ACCEPT_NATIVE", "-I" + str(KERNELS),
                 "-I" + str(CUDA_INCLUDE)]
        dependencies = capture(prefix, "driver_dependencies", [CXX, *flags, "-M", "-MT", "driver", DRIVER_SOURCE])
        commands.append(dependencies)
        dep_text = successful(dependencies, "CPU driver dependencies").replace("\\\n", " ")
        for name in shlex.split(dep_text.split(":", 1)[1]):
            path = Path(name).resolve()
            digest = sha(path)
            if str(path) in pins:
                require(pins[str(path)] == digest, f"CPU/build019 dependency mismatch: {path}")
            pins[str(path)] = digest
        driver = Path(str(prefix) + "_driver")
        compiled = capture(prefix, "driver_compile", [CXX, *flags, DRIVER_SOURCE, "-lcuda", "-o", driver], timeout=120)
        commands.append(compiled)
        successful(compiled, "CPU native-mode driver compilation")
        receipt["driver"] = identity(driver)
        cpu = capture(prefix, "cpu_controls", [driver, FIXTURE])
        commands.append(cpu)
        require("PASS CPU controls=1284 " in successful(cpu, "CPU acceptance controls"),
                "Expected actual-source CPU controls absent")
        require(unchanged(pins), "Source/binary/input changed during preparation")
        receipt.update(status="prepared", ended_utc=now(), sources_binary_inputs_unchanged=True)
        receipt["artifacts"] = {str(path): identity(path) for path in OUTPUT.glob(prefix.name + "*") if path.is_file()}
        write_json(receipt_path, receipt)
        return receipt
    except Exception as error:
        receipt.update(status="prepare_failed", ended_utc=now(), error=str(error))
        receipt["artifacts"] = {str(path): identity(path) for path in OUTPUT.glob(prefix.name + "*") if path.is_file()}
        write_json(Path(str(prefix) + "_prepare_failure.json"), receipt)
        raise


def parse_cases(text):
    rows = []
    pattern = re.compile(r"PASS native_accept (\S+) state=(\d+) sticky=(\d+) accepted=(\d+) time=(\S+)")
    for line in text.splitlines():
        match = pattern.fullmatch(line)
        if match:
            name, state, sticky, accepted, time = match.groups()
            rows.append({"name": name, "state": int(state), "sticky_fault": int(sticky),
                         "accepted_bodies": int(accepted), "accepted_time_text": time})
    return rows


def recovery_none(text):
    rows = list(csv.reader(text.splitlines(), skipinitialspace=True))
    return bool(rows) and all(len(row) == 6 and row[3].strip() == "None" for row in rows)


def run_native(prefix, prepared):
    started_path = Path(str(prefix) + "_dispatch_start.json")
    receipt_path = Path(str(prefix) + "_execution.json")
    require(not started_path.exists() and not receipt_path.exists(), "Native attempt already exists")
    pins = dict(prepared["input_hashes"])
    for item in (prepared["driver"], prepared["actual_module"]):
        pins[item["path"]] = item["sha256"]
    preparation_path = Path(str(prefix) + "_prepare.json")
    pins[str(preparation_path)] = sha(preparation_path)
    require(unchanged(pins), "Native source/module/driver/input pins changed")
    lock_path = f"/tmp/prism-foundation-gpu-{os.getuid()}.lock"
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    commands = []
    receipt = {"started_utc": now(), "authorization": "Operator-authorized exact exported-kernel replay; nonprivileged diagnostic.",
               "scope": "133 shared-step acceptance cases from retained017; actual019 extracted export. No molecular trajectory or production admission claim.",
               "preparation": identity(preparation_path), "input_hashes": pins,
               "fixture_prefix": str(FIXTURE), "fixture_raw_files": prepared["fixture_raw_files"],
               "actual_module": prepared["actual_module"], "driver": prepared["driver"],
               "export": EXPORT, "expected_case_names": EXPECTED_CASES,
               "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
               "cooperative_gpu_lock": lock_path, "privileged_lease_claim": False,
               "installed_state_mutation": False, "production_admission_reset": False,
               "historical_faults_preserved": True, "full_trajectory_accepted": False,
               "cuda_source_compilation": False, "native_execution": False, "commands": commands}
    acquired = False
    try:
        info = os.fstat(lock_fd)
        require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid(), "Unexpected GPU lock owner/type")
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        acquired = True
        health_args = ["nvidia-smi", "--query-gpu=name,uuid,driver_version,gpu_recovery_action,memory.used,utilization.gpu",
                       "--format=csv,noheader"]

        def checked(name, args):
            result = capture(prefix, name, args)
            commands.append(result)
            return successful(result, name)

        def journal(name):
            text = checked(name, ["journalctl", "-k", "-b", "--grep", "NVRM|Xid", "-o", "json", "--no-pager"])
            records = [json.loads(line) for line in text.splitlines() if line.startswith("{")]
            require(all(record.get("__CURSOR") for record in records), "Kernel journal cursor missing")
            return records

        before = journal("journal_before")
        require(recovery_none(checked("gpu_before", health_args)), "GPU recovery action is not None")
        apps = checked("compute_before", ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"])
        require(all(len(row) == 3 and row[1].strip() == "/usr/libexec/gnome-remote-desktop-daemon"
                    for row in csv.reader(apps.splitlines(), skipinitialspace=True) if row), "Another compute client is active")
        require(unchanged(pins), "Inputs changed during GPU preflight")
        command = [prepared["driver"]["path"], str(FIXTURE), prepared["actual_module"]["path"]]
        receipt.update(command=command, historical_kernel_records_retained=len(before), native_dispatch_attempted=True)
        write_json(started_path, receipt)
        native = capture(prefix, "native", command, timeout=120,
                         env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
        commands.append(native)
        rows = parse_cases(output_text(native))
        receipt.update(native_exit_code=native["exit_code"], timed_out=native["timed_out"],
                       native_execution=bool(rows), observed_case_count=len(rows), case_results=rows)
        # Always retain post-dispatch health, including a failed or timed-out run.
        after = journal("journal_after")
        old = {record["__CURSOR"] for record in before}
        new = [record for record in after if record["__CURSOR"] not in old]
        healthy = recovery_none(checked("gpu_after", health_args))
        stable = unchanged(pins)
        text = output_text(native)
        cases_match = [row["name"] for row in rows] == EXPECTED_CASES
        summary = "PASS native acceptance cases=133; exact supplied module, no molecular trajectory claim" in text
        cpu_pass = "PASS CPU controls=1284 " in text
        passed = native["exit_code"] == 0 and cases_match and summary and cpu_pass and not new and healthy and stable
        receipt.update(status="passed" if passed else "failed", passed=passed,
                       expected_cases_observed=cases_match, native_summary_present=summary,
                       cpu_controls_passed=cpu_pass, new_kernel_fault_records=new,
                       gpu_recovery_none_after=healthy, sources_binary_inputs_unchanged=stable)
        return 0 if passed else (native["exit_code"] or 1)
    except Exception as error:
        receipt.update(status="failed", passed=False, error=str(error))
        raise
    finally:
        try:
            if acquired:
                receipt["ended_utc"] = now()
                receipt["artifacts"] = {str(path): identity(path) for path in OUTPUT.glob(prefix.name + "*")
                                        if path.is_file() and path != receipt_path}
                write_json(receipt_path, receipt)
        finally:
            if acquired:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", default="001", help="Fresh three-digit replay attempt; build019 remains fixed.")
    parser.add_argument("--prepare-only", action="store_true", help="No CUDA initialization or GPU dispatch.")
    options = parser.parse_args()
    require(re.fullmatch(r"[0-9]{3}", options.attempt), "Attempt must contain three decimal digits")
    prefix = OUTPUT / ("failure_7_shared_accept_native_019_replay_" + options.attempt)
    prepared = prepare(prefix)
    if options.prepare_only:
        print(json.dumps({"status": "prepared", "native_execution": False,
                          "receipt": str(prefix) + "_prepare.json"}))
        return 0
    code = run_native(prefix, prepared)
    print(json.dumps({"exit_code": code, "receipt": str(prefix) + "_execution.json"}))
    return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"FAIL {error}", file=sys.stderr)
        sys.exit(1)
