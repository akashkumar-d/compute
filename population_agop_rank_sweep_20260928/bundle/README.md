# Remote rank-sweep execution bundle

Prepared for32 arms: raw h2 and raw ReLU teachers, ReLU and SwiGLU students, ranks2/4/8/16, reused seeds641/642, d64. Widths are256 and64. All32 configs are copied exactly from the independently authored DESIGN.snapshot.json; fresh rank8 runs are matched numerical controls using reused seeds, not fresh-seed confirmation. Success criteria remain byte-equivalent JSON to the design and unchanged from the current goal. The source kernels and runners are unchanged from reviewed v7 ReLU and v8 SwiGLU bundles.

The queue is fixed before outcomes: descending rank16,8,4,2; within rank, ascending seed641/642, teacher h2/relu, student relu/swiglu. Architecture therefore alternates. Separate rank cells retain paired seeds. Teacher axes are nested; initial arrays must be checked bitwise across within-architecture same-seed cells after execution before asserting a matched-initialization result. The same normalized training stop0.01VarY applies to every rank, following the independent design; this is a stopping rule, not a changed scientific success criterion.

## Admission and time bounds

Execution is SERVER-only on Linux. LOCAL_AUTHORIZED is rejected. Effective allocation must be at least32 CPUs, with28 single-thread rank workers and4 reserved slots:2 concurrent v9 validation workers and2 free CPUs. Rank and validation launchers do not share a global kernel lock, so the parent must ensure no additional workstream job exceeds the total30-worker authorization. Available-memory admission is32GiB and free disk admission10GiB. No peak-memory claim is made; source preparation included no model evaluation.

Every child inherits verified nice>=10 and six BLAS/OpenMP thread limits set to1; GPU visibility is disabled. Training has a1800s soft ceiling, frozen diagnostics180s, cleanup10s; the hard arm cap is1990s. Global compute is capped at3600s, not4200s.32 arms at28 workers occupy at most two launch waves; second-wave arms can receive less than their full per-arm budget. Current model evaluations can overrun soft caps; TERM starts8s before the hard cutoff, KILL2s before, with bounded reaping. Capped/unstarted/failed/incomplete-diagnostic arms remain in receipts and analyses. Exit0 does not imply scientific success.

The launcher descends from the independently reviewed v9 wrapper; its diff is saved as LAUNCHER.diff. The changes are execution bounds, remote-only gates, rank-design validation, and the dedicated rank-sweep lock. Original v9 launcher/tests/review are preserved under reference/. No scientific gate is disabled.

## Static checks and reviewed execution

Static validation and mock tests are safe to run locally and do not import scientific modules:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 launcher/launch.py --dry-run --execution-dir execution_remote_a01
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s launcher -p test_launch.py -v
```

Before launch, REVIEW.json must record status=approved_for_execution, this exact MANIFEST.json SHA256, reviewer identity, and nonempty checks. Review the final32 design/config matches, source hashes, scheduler diff/tests,32CPU/32GiB admission, external2-worker reservation, total30-worker ceiling, and3600s global cap. No deployment or training is performed by this preparation.

After the parent stages the reviewed bundle in the shared compute repository and follows the project lrun workflow, this is the exact server-side command from the bundle directory:

```sh
AGOP_EXECUTION_SITE=SERVER PYTHONDONTWRITEBYTECODE=1 python3 launcher/launch.py --execution-dir execution_remote_a01
```

Use the reviewed runtime interpreter containing the pinned NumPy/SciPy dependencies; the wrapper records its actual path. Never run the scientific entrypoints directly to circumvent the wrapper, source freeze, capacity, time, or review gates. A different manifest or config requires review of the new hash.

## Preservation, restarts and results

Existing execution directories are refused. A bundle-level kernel lock plus durable journal prevents concurrent attempts and blocks ambiguous restart after parent death. Restart requires the prior attempt to have a terminal receipt, no unreaped active child, and no surviving recorded process group. No arbitrary PID is killed. Preserve all earlier attempts; a permitted retry uses a new directory and starts again from the same seed. Never clear the journal automatically or duplicate the bundle to bypass its lock.

Keep logs, all-update losses, initial/saved states, periodic partials, diagnostics including failures/missing rows, status history and provenance. Source/config hashes are rechecked before each arm and at shutdown. All frozen inputs, exact configs, design snapshot, source lineage and prior scheduler references are covered by MANIFEST.json; REVIEW.json is deliberately mutable so the parent can record review against the frozen manifest.

Use the existing saved-data analyzer with this manifest and execution/data layout. Group by explicit rank cells; never merge ranks into independent-seed counts or promote a process completion to a numerical candidate. Recompute1%/5% whole-update initial max/min windows from each new trajectory and apply the unchanged first-qualified same-checkpoint rule. Retain raw and variance-normalized loss/refit values, weakest-direction gain, resolution guards, later-loss release, and all uncertainty/censoring. Higher-order saved-state audits and rank-wise random-baseline interpretation remain as declared by the design owner.

Native ReLU result flags retain the historical raw >1e-5 screen in unchanged source. They do not decide this sweep’s primary outcome. The canonical saved-data summarizer must apply the frozen0.1VarY refit/later-loss thresholds and all same-checkpoint requirements.
