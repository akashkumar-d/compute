# Matched higher-order seed642 trajectory validation — prepared, not launched

This bundle contains exactly two from-initialization numerical repetitions of the existing development seed642 h2/slower-head SwiGLU recipe. They refine all four quadrature families by2 and4. They are two numerical recipes for one seed, not two independent seeds. The Gaussian initialization, all trained blocks, head_lr0.001, h0.01, fixed teacher normalization, profiled intercept, and adaptive Euler rule are unchanged. Scientific files and their existing frozen SOURCE_MANIFEST are byte-identical to v8. Prior trajectories are read-only controls.

The original step380/t33.95 candidate is near the5% initial loss-window edge and has quadrature-sensitive geometry. The parent’s saved-state audit found original-to-doubled rate-weighted direction error10.9005% and step error10.883%; doubled-to-quadrupled errors0.067199% and0.06773%. These are endpoint comparisons, not a verified trajectory. Seed641/t18.44 remains a separate previous observation and is not rerun here.

## Predeclared pair

| Numerical recipe | Pair / diagonal / conditional / teacher orders | Seed | h | head_lr | Physical horizon | Stop loss | Step cap |
|---|---|---|---|---|---|---|---|
| order2 |32 /96 /48 /24|642|0.01|0.001|100|0.8|3000|
| order4 |64 /192 /96 /48|642|0.01|0.001|100|0.8|3000|

Relative to v8, only identifiers, the four quadrature orders, t_max, L_stop and max_steps change. Every config records separate cell/recipe metadata; do not compute a median or independent-seed success count across orders. Source/config hashes, v8 lineage, and exact changes are in MANIFEST.json and CONFIG_COMPARISONS.json.

## Execution support and bounds

The unchanged v8 launcher is preserved under reference/ and remains server-only with its900s arm gate. It cannot represent the requested budget. launcher/launch.py is a separate small fork of that already-tested scheduler; LAUNCHER.diff lists every change. No source gate is patched or spoofed. Its LOCAL_AUTHORIZED mode uses the runner’s already-supported local gate; SERVER requires Linux and at least32 allocated CPUs. Both modes launch exactly2 workers, leaving at least2 CPUs available; the32CPU remote authorization is a ceiling/resource precondition, not a demand to run30 arms. GPU visibility is disabled.

Each worker is single-threaded, with verified inherited nice>=10. Priority-setting errors are fatal, never ignored. Admission requires8GiB currently available memory and5GiB free disk. This is a conservative initial guard, not a measured peak-memory claim. No memory stress/model test was run. Both arms execute concurrently in one wave, with1800s soft training +180s soft diagnostics +10s cleanup,1990s hard arm cap,2150s global cap. A current model evaluation may overrun a soft budget; TERM begins8s before a hard deadline, KILL2s before, and only process groups created by this invocation are signaled. Unreaped processes remain active in the receipt. Completion/exit0 is not scientific success. Training or diagnostics may be censored.

Only static JSON/hash validation and standard-library mocked process tests have run. REVIEW.json is deliberately pending. An independent reviewer must record status=approved_for_execution, the exact MANIFEST.json SHA256, reviewer identity, and a nonempty checks list before launch. This is the scientific prelaunch review record. The v8 SOURCE_MANIFEST status/hash remains unchanged.

From this directory, exact static commands (no model evaluation):

```sh
PYTHONDONTWRITEBYTECODE=1 python3 launcher/launch.py --dry-run --execution-dir execution_local_order24_a01
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s launcher -p test_launch.py -v
```

After independent review, the exact local compute command, using an interpreter with the pinned dependencies installed, is:

```sh
AGOP_EXECUTION_SITE=LOCAL_AUTHORIZED PYTHONDONTWRITEBYTECODE=1 python3 launcher/launch.py --execution-dir execution_local_order24_a01
```

This command must be run with permissions that permit nice>=10; the wrapper verifies it. The parent may substitute its already-verified NumPy/SciPy interpreter path for python3 without changing scientific files, and must record that interpreter in provenance (the wrapper does so).

The exact **server-side** compute command after this reviewed bundle has been staged is:

```sh
AGOP_EXECUTION_SITE=SERVER PYTHONDONTWRITEBYTECODE=1 python3 launcher/launch.py --execution-dir execution_server_order24_a01
```

REMOTE code support is prepared and mocked, not deployed or integration-tested. Lightning was reported stopped. No network, machine switch, repository publication, dependency install or training was performed. Follow the project lrun flow for a future remote launch: check current machine; commit/push reviewed new code to the shared compute repo; use lrun repo with that actual branch/run name; monitor and pull results. There is intentionally no invented remote URL/ref/path in this preparation. No GPU authorization is implied.

## Preservation and restart

The launcher refuses any existing execution directory, including completed outputs. All raw JSON, all-update losses, snapshots, periodic partials, diagnostics (including missing/failed states), per-arm logs, provenance, status history and termination records stay inside the attempt directory. It never resumes or overwrites an earlier arm. Every retry starts again from the same seed in a new attempt directory, retaining the earlier attempt and its censoring.

A bundle-level kernel lock prevents concurrent launcher attempts; its durable journal survives parent death. Before accepting a new attempt, the last attempt must have a terminal status, no unreaped active children, and no remaining recorded process group. Ambiguous/missing status or reused/live process-group IDs block restart without killing anything. A reviewer must inspect and document any recovery; do not delete old data or automatically clear the journal. Never launch a second copy of this bundle concurrently to evade this lock.

After a normal terminal attempt, rerun the same exact local/server command with a unique suffix, such as execution_local_order24_a02. That is a restart from initialization and must be reported as a new attempt; it cannot silently replace a capped attempt. Source/config hashes are checked before each arm and again at shutdown. REVIEW.json itself is excluded from immutable source hashes so the parent can complete review without changing the reviewed manifest.

## Independent review and post-run analysis

Before execution, independently verify: unchanged science and frozen source manifest; config changes only as declared; same seed and RNG initialization path; budgets/priority/CPU/site gates; exact engine command arguments; no overwrite/restart hazards; source/config hash coverage; and mock tests. Review the new wrapper diff, not only the old source review. The parent owns review and any launch.

After execution, compare each new initial P,V,a snapshot bitwise against v8 seed642 and the other numerical recipe. Recompute each trajectory’s1.01 and1.05 all-update initial max/min prefix using that trajectory’s own losses. Never apply the old step380 or old window endpoint to the new trajectory. Apply the unchanged first-qualified-candidate rule at one saved checkpoint: all-direction Amin gain>=0.5, the existing AGOP resolution/gap/PSD guards, cutoff-envelope refit gain>=0.1VarY, then later raw loss drop>=0.1VarY relative to that same candidate. Keep numerical refit-envelope language; it is not a formal bound on unrestricted optimal risk. Report incomplete diagnostics and censored endpoints explicitly.

Use existing summarize_coverage.py on this manifest and each execution directory; its configs schema is retained. Plot order2/order4 and original-order seed642 separately against physical time and loss-window position. Review full histories before any claim of trajectory convergence. Fixed teacher tail[-10,10], covariance floors, cancellation, and Euler discretization are unchanged; h-halving and a seed641 higher-order trajectory remain separate future checks. This bounded pair cannot establish14-teacher success or fresh-seed confirmation.
