"""Bounded original-rank2 diagnostics completion; SERVER-only; dry-run is static.

Fork of the reviewed scheduler; frozen scientific kernels and diagnostics-only adapter.
"""
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
ENGINES = {"diagnostics": ("adapter/complete_diagnostics.py", "DIAGNOSTICS_COMPLETION.json")}
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
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 1:
        raise ValueError("runtime.workers must be an integer from 1 to 1")
    total = positive_number(runtime["global_seconds"], "global_seconds")
    arm = positive_number(runtime["per_arm_seconds"], "per_arm_seconds")
    reserve = positive_number(runtime["diagnostic_reserve_seconds"],
                              "diagnostic_reserve_seconds", allow_zero=True)
    if total > 1300 or arm > 1300 or arm > total or arm <= reserve + CLEANUP_SECONDS:
        raise ValueError("Require reserve + 10 < per_arm <= 1300 and per_arm <= global <= 1300 seconds")
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
        if not (isinstance(cfg, list) and len(cfg) == 1 and isinstance(cfg[0], dict)):
            raise ValueError(f"Diagnostic config must be a singleton list of objects: {cid}")
    dependency = manifest.get("dependency")
    if dependency is not None:
        if not isinstance(dependency, dict) or set(dependency) != {"run", "max_wait_seconds"}:
            raise ValueError("dependency must contain only run and max_wait_seconds")
        if not isinstance(dependency["run"], str) or not NAME.fullmatch(dependency["run"]):
            raise ValueError("Invalid dependency run name")
        positive_number(dependency["max_wait_seconds"], "dependency.max_wait_seconds")
    return manifest, manifest_hash


def require_server():
    """Diagnostics completion has no local execution mode."""
    if sys.platform.startswith('linux') and os.environ.get('AGOP_EXECUTION_SITE') == 'SERVER':
        return 'SERVER'
    raise RuntimeError('Execution requires Linux and AGOP_EXECUTION_SITE=SERVER')


def resource_snapshot():
    """Read allocated CPU/memory limits; never resize a machine or start work."""
    cpus = float(len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1)
    report = {"affinity_cpus": cpus}
    quota = Path('/sys/fs/cgroup/cpu.max')
    if quota.exists():
        fields = quota.read_text().split()
        if fields[0] != 'max':
            quota_cpus = int(fields[0]) / int(fields[1])
            report['cgroup_quota_cpus'] = quota_cpus
            cpus = min(cpus, quota_cpus)
    else:
        q = Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')
        p = Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us')
        if q.exists() and p.exists() and int(q.read_text()) > 0:
            quota_cpus = int(q.read_text()) / int(p.read_text())
            report['cgroup_quota_cpus'] = quota_cpus
            cpus = min(cpus, quota_cpus)
    mem = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        fields = line.split()
        if fields[0] in ('MemAvailable:', 'MemTotal:'):
            mem[fields[0][:-1]] = int(fields[1]) * 1024
    available = mem['MemAvailable']
    limit = Path('/sys/fs/cgroup/memory.max')
    current = Path('/sys/fs/cgroup/memory.current')
    if limit.exists() and current.exists() and limit.read_text().strip() != 'max':
        cgroup_available = max(0, int(limit.read_text()) - int(current.read_text()))
        available = min(available, cgroup_available)
        report['cgroup_available_memory_bytes'] = cgroup_available
    elif not limit.exists():
        legacy_limit = Path('/sys/fs/cgroup/memory/memory.limit_in_bytes')
        legacy_current = Path('/sys/fs/cgroup/memory/memory.usage_in_bytes')
        if legacy_limit.exists() and legacy_current.exists():
            cgroup_available = max(0, int(legacy_limit.read_text()) - int(legacy_current.read_text()))
            available = min(available, cgroup_available)
            report['cgroup_v1_available_memory_bytes'] = cgroup_available
    report.update(effective_cpus=cpus, available_memory_gib=available / 1024**3,
                  physical_memory_gib=mem['MemTotal'] / 1024**3,
                  free_disk_gib=__import__('shutil').disk_usage(ROOT).free / 1024**3)
    return report


def require_capacity(runtime):
    report = resource_snapshot()
    if report['effective_cpus'] < 32:
        raise RuntimeError('Insufficient allocated CPUs: SERVER requires the authorized allocation of at least32 CPUs')
    reserved = runtime.get('reserved_cpus', 0)
    if isinstance(reserved, bool) or not isinstance(reserved, int) or reserved < 0:
        raise ValueError('reserved_cpus must be a nonnegative integer')
    if runtime['workers'] + reserved > report['effective_cpus']:
        raise RuntimeError(f"Insufficient allocated CPUs: need {runtime['workers']} workers + {reserved} free, have {report['effective_cpus']}")
    if report['available_memory_gib'] < runtime.get('min_available_memory_gib', 0):
        raise RuntimeError('Insufficient available memory for the declared worker count')
    if report['free_disk_gib'] < runtime.get('min_free_disk_gib', 0):
        raise RuntimeError('Insufficient free disk for the declared run')
    report.update(workers=runtime['workers'], reserved_cpus=reserved, checked_utc=utc())
    return report


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
    return common + ["--tag", entry["id"], "--out-dir", dest,
                     "--diagnostic-seconds", str(min(reserve, usable)), "--deadline-utc", deadline_utc]


def child_environment():
    env = os.environ.copy()
    env.update({key: "1" for key in THREAD_VARIABLES})
    env.update(AGOP_EXECUTION_SITE=require_server(), PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               CUDA_VISIBLE_DEVICES="", AGOP_DIAGNOSTIC_LAUNCHER_PID=str(os.getpid()))
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
    require_review(root, manifest_hash)
    require_priority()
    resource_receipt = require_capacity(manifest["runtime"])
    resource_receipt.update(execution_site=require_server(),
                            effective_nice=os.nice(0), expected_child_nice='inherits parent >=10')
    # Never resume, truncate or overwrite an existing attempt.
    output.mkdir(exist_ok=False)
    atomic_json(output / "RESOURCES.json", resource_receipt)
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
        final_error = None
        try:
            validate(root, manifest_hash)
        except Exception as exc:
            final_error = exc
            outcome, error = 'failed', f'Final source/config validation failed: {exc}'
        save(outcome, error)
        history.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if final_error is not None:
            raise RuntimeError(error) from final_error


def require_priority():
    # Fail before reserving an attempt; children inherit the verified nice value.
    if os.nice(0) < 10:
        os.nice(10 - os.nice(0))
    if os.nice(0) < 10:
        raise RuntimeError('Reduced priority could not be established; do not launch')


def validate_protocol(manifest, root=ROOT):
    """One completed training trajectory, original grid, no new training."""
    expected = dict(workers=1,reserved_cpus=31,min_available_memory_gib=8,
                    min_free_disk_gib=5,global_seconds=1300,per_arm_seconds=1300,
                    diagnostic_reserve_seconds=1200)
    if manifest['runtime'] != expected or manifest.get('dependency') is not None:
        raise ValueError('Runtime must equal the frozen one-worker diagnostics bounds')
    if len(manifest['configs']) != 1:
        raise ValueError('Exactly one original-rank2 diagnostics case is allowed')
    entry = manifest['configs'][0]
    spec = load_json(root/'INPUT_SPEC.json')
    original = spec['source_job_config']
    if entry['id'] != spec['tag'] or entry['config'] != original['config']:
        raise ValueError('Original scientific configuration must remain unchanged')
    if entry['engine'] != 'diagnostics' or entry['config'][0]['args'] != spec['cfg']:
        raise ValueError('Require the pinned diagnostic recipe')
    cfg=spec['cfg']
    if (spec['tag'] != 'v10_rank_swiglu_h2_r2_d64_m64_seed642' or
            cfg['link'] != 'h2' or cfg['seed'] != 642 or cfg['c'] != [1.,1.] or
            [cfg[k] for k in ('n_pair','n_diag','n_z','n_x')] != [32,96,48,24]):
        raise ValueError('Only the reviewed original rank2 case is authorized')
    if manifest['external_input_spec_sha256'] != sha256(root/'INPUT_SPEC.json'):
        raise ValueError('Pinned external input specification changed')
    if spec['termination']['reason'] != 'loss_stop' or spec['termination']['step'] != 1244:
        raise ValueError('Require original completed training trajectory')
    plan=load_json(root/'COMPLETION_PLAN.json')
    if (plan['checkpoint_count'] != 269 or plan['preserved_success_indices'] != [0,25,26,30,31]
            or len(plan['full_compute_order']) != 264 or plan['new_training_updates'] != 0):
        raise ValueError('Require the original diagnostic grid and exact reuse set')


def require_review(root, manifest_hash):
    review = load_json(root / 'REVIEW.json')
    if (review.get('status') != 'approved_for_execution' or
            review.get('manifest_sha256') != manifest_hash or
            not review.get('reviewer') or not review.get('checks')):
        raise RuntimeError('Independent review of this exact manifest is required before execution')


class attempt_lock:
    """Kernel lock plus conservative restart journal; no automatic stale cleanup."""
    def __init__(self, root, output):
        self.root, self.output, self.stream = root, output, None

    def __enter__(self):
        import fcntl
        self.stream = (self.root / '.diagnostics_completion.lock').open('a+')
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.stream.seek(0)
            previous = self.stream.read().strip()
            if previous:
                record = json.loads(previous)
                directory = self.root / record['execution_dir']
                if directory.parent != self.root or not NAME.fullmatch(record['execution_dir']):
                    raise RuntimeError('Invalid restart journal; independent inspection required')
                status = load_json(directory / 'STATUS.json')
                terminal = ('completed', 'failed', 'capped', 'interrupted', 'queue_timeout')
                if status['status'] not in terminal or status['active']:
                    raise RuntimeError('Prior attempt not acknowledged terminal; preserve and inspect it')
                for child in status['completed']:
                    try:
                        os.killpg(child['pid'], 0)
                    except ProcessLookupError:
                        continue
                    raise RuntimeError('Prior process group remains live or ambiguous; do not restart')
            self.stream.seek(0)
            self.stream.truncate()
            json.dump(dict(execution_dir=self.output.name, launcher_pid=os.getpid(),
                           started_utc=utc()), self.stream)
            self.stream.flush()
            os.fsync(self.stream.fileno())
            return self
        except BaseException:
            self.stream.close()
            raise

    def __exit__(self, *exc):
        self.stream.close()


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
    validate_protocol(manifest, root)
    if args.dry_run:
        print(json.dumps({"valid": True, "arms": len(manifest["configs"]),
                          "runtime": manifest["runtime"], "dependency": manifest.get("dependency"),
                          "manifest_sha256": manifest_hash, "model_evaluations": 0,
                          "execution_output_created": False}))
        return 0
    require_server()
    require_review(root, manifest_hash)
    require_capacity(manifest["runtime"])
    require_priority()
    with attempt_lock(root, output):
        return execute(root, output, manifest, manifest_hash)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        print(f"Launcher error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
