# Bounded SERVER batch launcher

`launch.py` uses only the Python standard library (Python 3.10+). It never imports a scientific runner. All changes for this task are confined to this `launcher/` directory.

From the bundle directory, static validation is:

```sh
PYTHONDONTWRITEBYTECODE=1 python launcher/launch.py --dry-run
```

This reads JSON and source bytes, checks the requested output name, and prints a validation receipt. It does not check remote processes, create an execution directory, import NumPy, evaluate a model, or run a subprocess. Normal execution additionally requires Linux and `AGOP_EXECUTION_SITE=SERVER`; the interpreter running the launcher also runs its children, so a batch-private virtual environment is supported.

## Manifest contract

The launcher reads the parent directory's `MANIFEST.json`:

```json
{
  "runtime": {
    "workers": 4,
    "global_seconds": 3300,
    "per_arm_seconds": 480,
    "diagnostic_reserve_seconds": 140
  },
  "source_sha256": {
    "code/scaling_run.py": "<64 lowercase hexadecimal characters>",
    "swiglu/code/run_one.py": "<64 lowercase hexadecimal characters>"
  },
  "configs": [
    {
      "id": "a_unique_arm_id",
      "engine": "relu",
      "config_path": "configs/a_unique_arm_id.json",
      "config": {"id": "a_unique_arm_id"}
    }
  ],
  "dependency": {
    "run": "modarith-p61-decay-grid-20260928-v2",
    "max_wait_seconds": 7200
  }
}
```

The example omits scientific fields and actual hashes. Source paths and config paths are relative to the bundle root; traversal, absolute paths and resolved files outside the bundle are rejected. The hash map must include each used engine entrypoint; the coordinator should list every source dependency, including launcher files. Every declared hash is verified. Config contents must match `entry.config` in canonical JSON, including number/boolean representation; duplicate JSON keys and nonfinite values are rejected. ReLU configs are objects; SwiGLU configs are singleton lists of objects. IDs and config paths must be unique. The manifest itself is hashed at entry and cannot change during the invocation.

Validation runs before queueing, at each queue check, immediately before starting the compute budget, and before every arm launch. No shell command is constructed: runners receive argument lists directly.

## Dependency and resource scope

The optional dependency is exactly one run name. Its only observed files are:

```text
Path.home()/compute/runs/<run>/.lrun/pid
Path.home()/compute/runs/<run>/.lrun/exit_code
```

This matches the Studio home `/teamspace/studios/this_studio`; it is not absolute `/compute`. Readiness requires a valid recorded PID, an integer exit code, stable records during the read, and that the recorded PID is no longer executing. Linux zombies count as non-running; missing/invalid records, permission errors and uncertain liveness keep the dependency pending. A nonzero exit is accepted as terminal and preserved in the receipt. The launcher does not alter, signal, restart or remove the dependency or any other existing job. It rechecks the dependency immediately before each child launch and aborts its own batch if that run became nonterminal again.

Queue checks are 30 seconds apart, with a shorter final wait only to respect the finite queue deadline or an incoming termination signal. The 7200-second queue allowance is separate from the 3300-second compute budget. Timeout leaves every unstarted arm recorded. This queue does **not** establish exclusive use of the server; unrelated jobs are not scanned or controlled.

## Runner budgets and shutdown

The launcher enforces `1 <= workers <= 4`, `diagnostic_reserve + 10 < per_arm <= 480`, and `per_arm <= global <= 3300`. For the planned runtime:

| Engine | Explicit child flags | Completion indicator |
|---|---|---|
| ReLU | `code/scaling_run.py --config PATH --out execution/data/ID --max-seconds 470 --diagnostic-reserve 140 --deadline-utc CUTOFF` | `DONE.json` |
| SwiGLU | `swiglu/code/run_one.py --config PATH --tag ID --out-dir execution/data/ID --wall-seconds 330 --diagnostic-seconds 140 --deadline-utc CUTOFF` | `result.json` |

SwiGLU's wall-seconds allowance is training-only, so it is `per_arm - reserve - 10`. ReLU's max-seconds allowance includes its diagnostics and is `per_arm - 10`. Each UTC cutoff is the earlier of that arm's cap and the global cap, minus ten seconds. If less global time remains, both engine budgets shrink accordingly; an arm is not started if its diagnostic reserve and cleanup allowance no longer fit.

All arms retain manifest order. Up to `workers` children run concurrently. Every child is started in a new session/process group, with OpenBLAS, OpenMP, MKL, NumExpr, vecLib and BLIS thread counts set to one. Standard output and standard error are separate preserved files.

The parent requests SIGTERM eight seconds before an arm/global cap and SIGKILL two seconds before the cap, then performs a bounded wait for reaping within the remaining cap. The monitor sleeps between child exits, status updates and upcoming deadlines; it is not a busy loop. SIGINT/SIGTERM, spawn errors and source/config changes trigger cleanup of only this invocation's recorded child groups. Any descendants left after an engine exits are also killed within that same recorded group. A kill request is never treated as a reap acknowledgment: an uninterruptible or otherwise unreaped child remains in the final `active` list with null return code, `reap_acknowledged: false` and `cleanup_status: kill_requested_unreaped`. Completed children have `reap_acknowledged: true`. No later arm is started after an unreaped child reaches its cap. Final cleanup waits are bounded by both the individual and global remaining budgets.

The planned worst case is `28 × 480 / 4 = 3360` seconds, exceeding the 3300-second global allowance. Partial diagnostic output and global censoring are therefore expected possibilities and must remain explicit in analysis. A timeout or missing diagnostic is not evidence against the scientific hypothesis.

## Preserved outputs and statuses

The default is a fresh `execution/` directory. Existing directories are refused, including under `--dry-run`; there is no overwrite or resume. An explicitly named new attempt can use `--execution-dir execution_attempt2`. This must be a single safe directory name inside the bundle.

- `MANIFEST.snapshot.json`: complete manifest used for the invocation.
- `PROVENANCE.json`: manifest/source/config hashes, interpreter, thread environment and resource-scope statement.
- `STATE_HISTORY.jsonl`: append-only full status history; `STATUS.json` is its latest atomically replaced snapshot.
- `logs/ID.stdout.log` and `logs/ID.stderr.log`: exclusive-created logs.
- `data/ID/`: unchanged runner output destination; every checkpoint/state/partial file remains in place.

Status includes pending IDs, active children and completed child receipts. Each completed receipt contains engine, PID, return code, elapsed seconds, stop reason, TERM/KILL flags, the engine-specific completion indicator and whether it exists. Engine exit zero plus its indicator is execution completion only; every status records `scientific_success: not_assessed`.

Final statuses are `completed` (exit 0), `failed` (exit 1), `capped` or `queue_timeout` (exit 124), and `interrupted` (128 + signal). Runtime snapshots also use `queued`, `starting`, `running` and `stopping`. Pending IDs are retained even when the overall batch stops.

## Tests

```sh
cd launcher
PYTHONDONTWRITEBYTECODE=1 python -m unittest -v test_launch.py
```

Tests use temporary fixture bundles within this directory and mocked processes, signals and clocks. They cover static dry-run behavior, source/config/manifest integrity, path containment, duplicate IDs, engine contracts, server/overwrite guards, dependency liveness, queue limits, compute-budget separation, hard caps, pending-arm preservation, output receipts and one-thread child environments. No scientific runner or remote job is executed by these tests.
