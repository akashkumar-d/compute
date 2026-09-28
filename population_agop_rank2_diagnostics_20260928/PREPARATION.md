# Original rank-two diagnostic completion

Prepared only. No training, model evaluation, network access, publication, or dispatch occurred during preparation. Independent review of the final manifest is required before the parent launches this pass.

This is one diagnostics-only pass on `v10_rank_swiglu_h2_r2_d64_m64_seed642`, from the original rank-sweep job `agop-ranks-2to16-20260928-a01`. Training reached `loss_stop` at update 1244, time 115.84333122304338, normalized loss 0.009839304411645156. The original result reports `completed=true`, but only 5 of its 269 saved-state diagnostic records are successful. The original result, raw history, snapshots, partial diagnostics, and launch receipt remain untouched.

The frozen configuration is copied exactly from the original job. The eleven scientific/support files under `swiglu/code` remain byte-identical to the reviewed rank-sweep/v8 source. The adapter calls only `diagnostics.make_engine(cfg)` and `diagnostics.metrics` at multiplier 1 (orders 32/96/48/24). It performs no initialization, update, integration, new quadrature comparison, or success classification. The teacher, head rate, step controls, Gaussian seed, recorded trajectory, and scientific success criteria are unchanged.

`INPUT_SPEC.json` pins the five original input files by SHA-256 and size, the source hashes, the exact configuration, the termination state, the local recovered location, and the remote inventory hash. All five local recovered files matched `rank_sweep/REMOTE_INVENTORY_AFTER_STOP.json`. The approximately 17.3 MB snapshot archive remains in the original remote job directory; no original result or NPZ payload is bundled for publication. Runtime hashes are checked before importing the scientific backend and again after evaluation. The array shapes/timestamps and original all-update/checkpoint correspondence are also checked. Arrays are loaded with `allow_pickle=False` and made read-only.

`COMPLETION_PLAN.json` is derived using the original all-update maximum/minimum loss prefixes and the original saved grid. It preserves diagnostic indices 0, 25, 26, 30, 31 (updates 0, 222, 223, 296, 297) verbatim. Provenance is recorded in a separate mapping, without adding fields to those five records. The order is 24 missing initial-1% states, three additional missing initial-5% states, then the remaining 237 states in ascending original order. No state is selected based on its alignment or refit outcome. The planner/row-completion core can be reused for another separately pinned and reviewed case; this launcher's production protocol admits only this one case.

The new artifact is `execution_remote_a01/data/v10_rank_swiglu_h2_r2_d64_m64_seed642/DIAGNOSTICS_COMPLETION.json`. It contains all 269 diagnostic records with a separate origin map, input/source/config hashes, original termination, attempted/missing indices, and independent completion-pass status. It is atomically refreshed before work and after every attempted state. A complete pass returns zero only if every row is successful and final hashes match. Budget exhaustion, interruption, failed states, or changed inputs cannot be reported as complete. There is no new `result.json` and no retroactive claim that the original process completed its diagnostics. Analyses must explicitly consume this derived sidecar and retain its provenance rather than replacing the original result.

The scheduler is a minimal derivative of the independently reviewed exact-state continuation scheduler, whose bytes are retained in `reference/`; the scheduler diff is in `launcher/CHANGES.diff`. It allows one single-thread CPU worker on Linux with `AGOP_EXECUTION_SITE=SERVER`, at least 32 effective CPUs, 8 GiB available memory, 5 GiB free disk, and verified niceness of at least 10. It has a 1200-second soft diagnostics ceiling and a 1300-second hard global/per-case process budget, including setup and preservation. The scheduler sends TERM and then KILL only to its own process group, preserves unacknowledged/capped states, refuses existing output directories, and retains its restart journal. The child requires the reviewed scheduler parent, review/resource/source/config/input gates, all six thread variables set to 1, and GPUs hidden. No local scientific execution mode exists.

The parent must verify external concurrency immediately before launch: at most 29 other task workers, at most 30 total, with two cores free on the authorized 32-CPU Studio. The scheduler does not discover or stop unrelated jobs, resize machines, dispatch automatically, or await a guessed dependency.

From this bundle's directory, the exact static checks are:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 launcher/launch.py --dry-run
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s launcher -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s adapter -p 'test_*.py'
```

After the parent publishes the code-only bundle and independent review approves its exact manifest, the remote-only command is:

```sh
AGOP_EXECUTION_SITE=SERVER PYTHONDONTWRITEBYTECODE=1 nice -n 10 python3 launcher/launch.py --execution-dir execution_remote_a01
```

Use the existing remote Python environment containing the frozen requirements; do not install or upgrade scientific dependencies silently. The launch must refuse unavailable original inputs, hash changes, insufficient capacity, failed niceness, absent/wrong review, uncertain prior process state, or existing output. If interrupted, preserve this attempt and inspect its journal/output before deciding on another separately named attempt. This bundle always reuses the five pinned original rows; it does not silently adopt an unreviewed completion sidecar from a previous attempt.

Independent review should verify external input/inventory pins; byte-identical kernels and configuration; exact reuse of all five rows; the 24/3/237 loss-only ordering and full 264-state coverage; no training calls or source writes; source provenance recorded before the first evaluation; final hash checks; all 33 mocked/synthetic tests; and the one-worker, SERVER-only, nice/resource/time/restart gates. Tests evaluate only artificial data with a mock backend. Real recovered arrays are inspected only by their archive headers and hashes during preparation.
