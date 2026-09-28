"""Validated, queued, bounded SERVER-only launcher; --dry-run is static only."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
QUEUE_POLL_SECONDS = 30.0
CLEANUP_SECONDS = 10.0
ENGINES = {
    "relu": ("code/scaling_run.py", "DONE.json"),
    "swiglu": ("swiglu/code/run_one.py", "result.json"),
}
THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
)
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}\Z")


def utc(timestamp=None):
    return datetime.fromtimestamp(time.time() if timestamp is None else timestamp,
                                  timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Nonfinite JSON number {value} in {path}")

    return json.loads(path.read_text(), object_pairs_hook=unique,
                      parse_constant=invalid)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def relative_file(root, value):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"Expected a relative file path: {value!r}")
    if any(part in (".", "..") for part in value.split("/")):
        raise ValueError(f"Noncanonical relative path: {value!r}")
    path = (root / value).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"File escapes bundle or is not regular: {value!r}")
    return path


def positive_number(value, name, *, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"Invalid {name}: {value!r}")
    return float(value)


def validate(root, expected_manifest_hash=None):
    """Read JSON and source bytes only. Never import any scientific code."""
    manifest_path = root / "MANIFEST.json"
    manifest_hash = sha256(manifest_path)
    if expected_manifest_hash is not None and manifest_hash != expected_manifest_hash:
        raise ValueError("MANIFEST.json changed since this invocation began")
    manifest = load_json(manifest_path)
    runtime = manifest["runtime"]
    workers = runtime["workers"]
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 4:
        raise ValueError("runtime.workers must be an integer from 1 to 4")
    total = positive_number(runtime["global_seconds"], "global_seconds")
    arm = positive_number(runtime["per_arm_seconds"], "per_arm_seconds")
    reserve = positive_number(runtime["diagnostic_reserve_seconds"],
                              "diagnostic_reserve_seconds", allow_zero=True)
    if total > 3300 or arm > 480 or arm > total or arm <= reserve + CLEANUP_SECONDS:
        raise ValueError("Require reserve + 10 < per_arm <= 480 and per_arm <= global <= 3300 seconds")
    hashes = manifest["source_sha256"]
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("source_sha256 must be a nonempty relative-path/hash mapping")
    for name, expected in hashes.items():
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"Invalid SHA-256 for {name}")
        if sha256(relative_file(root, name)) != expected:
            raise ValueError(f"Source changed: {name}")
    configs = manifest["configs"]
    if not isinstance(configs, list) or not configs:
        raise ValueError("configs must be a nonempty list")
    ids, paths = set(), set()
    for entry in configs:
        cid = entry["id"]
        if not isinstance(cid, str) or not NAME.fullmatch(cid) or cid in ids:
            raise ValueError(f"Invalid or duplicate config id: {cid!r}")
        ids.add(cid)
        engine = entry["engine"]
        if engine not in ENGINES:
            raise ValueError(f"Unsupported engine {engine!r}; no command is guessed")
        source, _ = ENGINES[engine]
        if source not in hashes:
            raise ValueError(f"Engine entrypoint lacks declared source hash: {source}")
        path = relative_file(root, entry["config_path"])
        if path in paths:
            raise ValueError(f"Config path reused: {entry['config_path']}")
        paths.add(path)
        cfg = load_json(path)
        if canonical(cfg) != canonical(entry["config"]):
            raise ValueError(f"Config JSON differs from manifest: {cid}")
        if engine == "relu" and not isinstance(cfg, dict):
            raise ValueError(f"ReLU config must be an object: {cid}")
        if engine == "swiglu" and not (
                isinstance(cfg, list) and len(cfg) == 1 and isinstance(cfg[0], dict)):
            raise ValueError(f"SwiGLU config must be a singleton list of objects: {cid}")
    dependency = manifest.get("dependency")
    if dependency is not None:
        if not isinstance(dependency, dict) or set(dependency) != {"run", "max_wait_seconds"}:
            raise ValueError("dependency must contain only run and max_wait_seconds")
        if not isinstance(dependency["run"], str) or not NAME.fullmatch(dependency["run"]):
            raise ValueError("Invalid dependency run name")
        positive_number(dependency["max_wait_seconds"], "dependency.max_wait_seconds")
    return manifest, manifest_hash


def require_server():
    if not sys.platform.startswith("linux") or os.environ.get("AGOP_EXECUTION_SITE") != "SERVER":
        raise RuntimeError("Execution requires Linux and AGOP_EXECUTION_SITE=SERVER")


def pid_alive(pid):
    """Read only the named PID; zombies are no longer executing."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0] not in ("Z", "X")
    except (OSError, IndexError):
        return True


def dependency_state(dependency, home=None):
    directory = (Path.home() if home is None else home) / "compute/runs" / dependency["run"] / ".lrun"
    state = {"run": dependency["run"], "directory": str(directory), "ready": False}
    try:
        pid_text = (directory / "pid").read_text().strip()
        if not re.fullmatch(r"[0-9]+", pid_text) or int(pid_text) <= 1:
            return {**state, "reason": "missing_or_invalid_recorded_pid"}
        pid = int(pid_text)
        alive = pid_alive(pid)
        state.update(pid=pid, pid_alive=alive)
        exit_text = (directory / "exit_code").read_text().strip()
        if not re.fullmatch(r"-?[0-9]+", exit_text):
            return {**state, "reason": "missing_or_invalid_exit_code"}
        if ((directory / "pid").read_text().strip() != pid_text or
                (directory / "exit_code").read_text().strip() != exit_text):
            return {**state, "reason": "dependency_records_changed_while_reading"}
        state["exit_code"] = int(exit_text)
        state["ready"] = not alive
        state["reason"] = "completed_and_pid_not_live" if not alive else "recorded_pid_still_live"
    except OSError as exc:
        state["reason"] = f"dependency_record_unavailable:{type(exc).__name__}"
    return state


def build_command(root, output, entry, runtime, seconds_available, deadline_utc):
    """Engine contracts are explicit; max-seconds includes ReLU diagnostics."""
    arm_seconds = min(float(runtime["per_arm_seconds"]), seconds_available)
    reserve = float(runtime["diagnostic_reserve_seconds"])
    usable = arm_seconds - CLEANUP_SECONDS
    if usable <= reserve:
        raise ValueError("Insufficient remaining budget for diagnostics and cleanup")
    source, _ = ENGINES[entry["engine"]]
    common = [sys.executable, str(root / source), "--config", str(root / entry["config_path"])]
    dest = str(output / "data" / entry["id"])
    if entry["engine"] == "relu":
        return common + ["--out", dest, "--max-seconds", str(usable),
                         "--diagnostic-reserve", str(reserve), "--deadline-utc", deadline_utc]
    if entry["engine"] == "swiglu":
        return common + ["--tag", entry["id"], "--out-dir", dest,
                         "--wall-seconds", str(usable - reserve),
                         "--diagnostic-seconds", str(reserve), "--deadline-utc", deadline_utc]
    raise ValueError(f"Unsupported engine: {entry['engine']}")


def child_environment():
    env = os.environ.copy()
    env.update({key: "1" for key in THREAD_VARIABLES})
    env.update(AGOP_EXECUTION_SITE="SERVER", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    return env


@dataclass
class Child:
    entry: dict
    process: object
    command: list
    stdout: object
    stderr: object
    started: float
    deadline: float
    stop_reason: str | None = None
    term_sent: bool = False
    kill_sent: bool = False
    cleanup_status: str = "running"


def signal_child(child, sig):
    # Only PIDs of this launcher's start_new_session children enter this function.
    try:
        os.killpg(child.process.pid, sig)
    except ProcessLookupError:
        pass


def stop_child(child, reason, hard=False):
    child.stop_reason = child.stop_reason or reason
    if hard and not child.kill_sent:
        signal_child(child, signal.SIGKILL)
        child.kill_sent = True
        child.cleanup_status = "kill_requested"
    elif not hard and not child.term_sent:
        signal_child(child, signal.SIGTERM)
        child.term_sent = True
        child.cleanup_status = "term_requested"


def reap_child(child, timeout):
    """Wait only within a caller-provided cap; a kill request is not a reap."""
    try:
        code = child.process.wait(timeout=max(0.0, timeout))
    except subprocess.TimeoutExpired:
        child.cleanup_status = "kill_requested_unreaped" if child.kill_sent else "unreaped"
        return None
    child.cleanup_status = "reaped"
    return code


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def execute(root, output, manifest, manifest_hash):
    require_server()
    # Never resume, truncate or overwrite an existing attempt.
    output.mkdir(exist_ok=False)
    (output / "logs").mkdir()
    (output / "data").mkdir()
    (output / "MANIFEST.snapshot.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    atomic_json(output / "PROVENANCE.json", {
        "manifest_sha256": manifest_hash, "source_sha256": manifest["source_sha256"],
        "config_sha256": {c["config_path"]: sha256(root / c["config_path"]) for c in manifest["configs"]},
        "created_utc": utc(), "launcher_pid": os.getpid(), "python": sys.executable,
        "thread_environment": {key: "1" for key in THREAD_VARIABLES},
        "resource_exclusivity": "not established; only the named dependency is awaited",
    })
    history = (output / "STATE_HISTORY.jsonl").open("x")
    runtime = manifest["runtime"]
    pending, active, completed = list(manifest["configs"]), {}, []
    queue_started, started, deadline = time.monotonic(), None, None
    dependency, queue_receipt = manifest.get("dependency"), None
    interrupted, previous, wake = [None], {}, threading.Event()

    def signal_handler(signum, frame):
        if signum != signal.SIGCHLD:
            interrupted[0] = signum
        wake.set()

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGCHLD):
        previous[sig] = signal.signal(sig, signal_handler)

    def save(status, error=None):
        now = time.monotonic()
        value = {
            "status": status, "updated_utc": utc(), "launcher_pid": os.getpid(),
            "compute_elapsed_seconds": None if started is None else now - started,
            "queue_elapsed_seconds": (now if started is None else started) - queue_started,
            "global_seconds": runtime["global_seconds"], "dependency": queue_receipt,
            "pending": [c["id"] for c in pending], "completed": completed,
            "active": [{"id": key, "engine": c.entry["engine"], "pid": c.process.pid,
                        "command": c.command, "elapsed_seconds": now - c.started,
                        "stop_reason": c.stop_reason, "term_sent": c.term_sent,
                        "kill_sent": c.kill_sent, "cleanup_status": c.cleanup_status,
                        "returncode": getattr(c.process, "returncode", None),
                        "reap_acknowledged": False} for key, c in active.items()],
            "interrupted_signal": interrupted[0], "error": error,
            "scientific_success": "not_assessed",
        }
        history.write(json.dumps(value, allow_nan=False) + "\n")
        history.flush()
        atomic_json(output / "STATUS.json", value)

    def finish(cid, child, returncode):
        child.stdout.close()
        child.stderr.close()
        indicator = ENGINES[child.entry["engine"]][1]
        completed.append({"id": cid, "engine": child.entry["engine"], "pid": child.process.pid,
                          "returncode": returncode, "elapsed_seconds": time.monotonic() - child.started,
                          "stop_reason": child.stop_reason, "term_sent": child.term_sent,
                          "kill_sent": child.kill_sent, "reap_acknowledged": True,
                          "cleanup_status": "reaped", "completion_indicator": indicator,
                          "completion_indicator_exists": (output / "data" / cid / indicator).is_file()})
        del active[cid]

    outcome, error = "failed", None
    try:
        if dependency is not None:
            queue_deadline = queue_started + float(dependency["max_wait_seconds"])
            while True:
                validate(root, manifest_hash)
                queue_receipt = dependency_state(dependency)
                save("queued")
                if interrupted[0]:
                    outcome = "interrupted"
                    return 128 + interrupted[0]
                if queue_receipt["ready"]:
                    break
                remaining = queue_deadline - time.monotonic()
                if remaining <= 0:
                    outcome = "queue_timeout"
                    return 124
                wake.wait(min(QUEUE_POLL_SECONDS, remaining))
                wake.clear()
        validate(root, manifest_hash)
        started = time.monotonic()
        deadline = started + float(runtime["global_seconds"])
        save("starting")
        exhausted = False
        while active or pending:
            now = time.monotonic()
            stopping = bool(interrupted[0]) or now >= deadline - 8
            for cid, child in list(active.items()):
                code = child.process.poll()
                if code is not None:
                    # Any surviving descendants still belong to this child's own group.
                    signal_child(child, signal.SIGKILL)
                    finish(cid, child, code)
                    continue
                if stopping or now >= child.deadline - 8:
                    reason = "launcher_signal" if interrupted[0] else (
                        "global_wall_cap" if now >= deadline - 8 else "per_arm_wall_cap")
                    stop_child(child, reason)
                if (child.term_sent and (interrupted[0] or now >= min(deadline, child.deadline) - 2)):
                    stop_child(child, child.stop_reason, hard=True)
                    code = reap_child(child, min(0.25, min(deadline, child.deadline) - time.monotonic()))
                    if code is not None:
                        finish(cid, child, code)
                        continue
                if now >= min(deadline, child.deadline):
                    stop_child(child, child.stop_reason or "wall_cap", hard=True)
                    code = child.process.poll()
                    if code is None:
                        exhausted = True
                        stopping = True
                        break
            while pending and len(active) < runtime["workers"] and not stopping:
                validate(root, manifest_hash)
                if dependency is not None:
                    queue_receipt = dependency_state(dependency)
                    if not queue_receipt["ready"]:
                        raise RuntimeError("Named dependency became nonterminal before an arm launch")
                now = time.monotonic()
                available = min(float(runtime["per_arm_seconds"]), deadline - now)
                if available <= float(runtime["diagnostic_reserve_seconds"]) + CLEANUP_SECONDS:
                    exhausted = True
                    break
                entry = pending[0]
                cutoff = utc(time.time() + available - CLEANUP_SECONDS)
                command = build_command(root, output, entry, runtime, available, cutoff)
                stdout = (output / "logs" / (entry["id"] + ".stdout.log")).open("xb")
                stderr = (output / "logs" / (entry["id"] + ".stderr.log")).open("xb")
                try:
                    process = subprocess.Popen(command, cwd=root, env=child_environment(),
                                               stdout=stdout, stderr=stderr, start_new_session=True)
                except BaseException:
                    stdout.close()
                    stderr.close()
                    raise
                active[entry["id"]] = Child(entry, process, command, stdout, stderr,
                                             now, min(deadline, now + float(runtime["per_arm_seconds"])))
                pending.pop(0)
                save("running")
            save("stopping" if stopping else "running")
            if (not active and (stopping or exhausted)) or time.monotonic() >= deadline:
                break
            if exhausted and any(c.process.poll() is None and time.monotonic() >= c.deadline
                                 for c in active.values()):
                break
            if not active and not pending:
                break
            now = time.monotonic()
            next_time = min([t for t in (deadline - 8, deadline - 2, deadline) if t > now] +
                            [t for c in active.values() for t in (c.deadline - 8, c.deadline - 2, c.deadline)
                             if t > now] + [now + 5])
            # Future deadlines prevent spinning after a previously reached boundary.
            delay = min(5.0, max(0.05, next_time - now))
            wake.wait(delay)
            wake.clear()
        if interrupted[0]:
            outcome = "interrupted"
            return 128 + interrupted[0]
        if active or pending or any(c["stop_reason"] for c in completed):
            outcome = "capped"
            return 124
        outcome = "completed" if all(c["returncode"] == 0 and c["completion_indicator_exists"]
                                     for c in completed) else "failed"
        return 0 if outcome == "completed" else 1
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        save("failed", error)
        raise
    finally:
        for cid, child in list(active.items()):
            stop_child(child, child.stop_reason or "launcher_cleanup", hard=True)
            cleanup_deadline = min(time.monotonic() + 2.0, child.deadline,
                                   deadline if deadline is not None else float("inf"))
            code = reap_child(child, cleanup_deadline - time.monotonic())
            if code is not None:
                finish(cid, child, code)
            else:
                # Keep the still-unacknowledged child in final active state.
                child.stdout.close()
                child.stderr.close()
        save(outcome, error)
        history.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(argv=None, root=ROOT):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execution-dir", default="execution",
                        help="New single directory name inside the bundle; existing names are refused")
    args = parser.parse_args(argv)
    if not NAME.fullmatch(args.execution_dir):
        parser.error("--execution-dir must be a single safe directory name")
    manifest, manifest_hash = validate(root)
    output = root / args.execution_dir
    if output.exists():
        raise FileExistsError(f"Execution directory already exists: {output}")
    if args.dry_run:
        print(json.dumps({"valid": True, "arms": len(manifest["configs"]),
                          "runtime": manifest["runtime"], "dependency": manifest.get("dependency"),
                          "manifest_sha256": manifest_hash, "model_evaluations": 0,
                          "execution_output_created": False}))
        return 0
    return execute(root, output, manifest, manifest_hash)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        print(f"Launcher error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
