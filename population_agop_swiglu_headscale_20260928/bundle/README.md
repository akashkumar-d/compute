# Six-arm cubic Gaussian-head scaling — prepared, not dispatched

This bundle executes the six configurations previously fixed in DESIGN.snapshot.json and reviewed in INDEPENDENT_SCALING_REVIEW.snapshot.md. Their config bytes are unchanged. Raw additive h3 uses r8,d64,m64,s0.3; reused Gaussian seeds641/642; head ratios q=1,0.3,0.1; head_lr=0.01*q^2; common quadrature32/96/48/24. Physical horizon3000/q, dt_max50/q and checkpoint time floor0.5/q match the planned normalized clock tau=q*t. The15000-step cap and loss stop0.01 remain fixed.

This is an exploratory paired change in dynamics, not a time-rescaling equivalence, fresh-seed confirmation, or evidence of success. Fixed-state q^2 AGOP scaling does not change its eigenspace. Actual normalized head dynamics can change AGOP even with fixed hidden features, so the original separate material refit-gain requirement remains necessary. Smaller heads may worsen neuron imbalance or delay release. Preserve negative and censored outcomes.

All scientific files and the frozen SOURCE_MANIFEST are copied unchanged from v8. The only executable modification is a small resource/protocol adaptation of the tested SERVER-only rank scheduler; LAUNCHER.diff shows it. Its reviewed predecessor source/tests/review are preserved in reference/. The previously reviewed scaling rationale is distinct from the runtime review still required here.

## Dispatch prerequisites and resource bounds

No automatic dispatch or dependency polling is configured. The parent must first review the v9 higher-order trajectory validation, review this exact bundle/runtime, and check all task-owned live training workers immediately before launch. This batch adds6 workers, so other task training must be<=24 and the combined total<=30, leaving2 CPUs free on32CPU. The bundle-level lock does not coordinate other bundle copies or the active rank/v9 jobs. Record the external concurrency and v9 review checks in REVIEW.json; do not launch based only on this bundle's own worker limit.

Execution requires Linux and AGOP_EXECUTION_SITE=SERVER; LOCAL_AUTHORIZED is rejected. Admission requires at least32 effective allocated CPUs,12GiB currently available memory and5GiB free disk. Runtime reserves26 CPU slots outside this batch: up to24 other task workers and2 free. These are reservation/admission rules, not proof that other launchers are idle. No machine change, network operation, package installation or model evaluation is part of preparation.

Exactly6 children run in one wave, single-threaded with all six BLAS/OpenMP controls set to1, GPU visibility disabled, and inherited nice>=10 verified before any launch. Failure to establish low priority is fatal. Per arm:1800s soft training,180s soft frozen diagnostics,10s cleanup,1990s hard cap. Global compute cap2150s. A current evaluation may overrun a soft budget; TERM begins8s before the hard cutoff, KILL2s before, with bounded reaping. Unreaped children remain active in receipts. A process exit0 is not scientific success.

## Static validation and eventual server command

These commands perform only hash/JSON or mocked scheduler checks and may run locally:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 launcher/launch.py --dry-run --execution-dir execution_remote_a01
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s launcher -p test_launch.py -v
```

REVIEW.json must be completed by an independent reviewer with status=approved_for_execution, this exact MANIFEST.json hash, reviewer identity and nonempty checks. Include verification of six unchanged config hashes, source lineage, scaling review, runtime limits, v9 prerequisite review, current external concurrency, and available CPU/memory/disk. The parent owns review, any shared-repository publication and any lrun dispatch.

After those checks and staging, the exact server-side command from this bundle directory is:

```sh
AGOP_EXECUTION_SITE=SERVER PYTHONDONTWRITEBYTECODE=1 python3 launcher/launch.py --execution-dir execution_remote_a01
```

Use the parent-verified runtime interpreter; its actual path is recorded in provenance. Source requirements remain preserved as historical inputs; this preparation does not install or change dependencies. Never invoke the runner directly to bypass the source/review/resource/time gates.

## Preservation and interpretation

Existing execution directories are refused. A dedicated kernel lock and persistent restart journal prevent overlapping attempts in this bundle and block ambiguous restart after parent death. A restart requires a terminal prior receipt, no unreaped active child and no remaining recorded process group; no arbitrary process is killed. Use a fresh attempt directory and preserve prior attempts. Do not clear the journal automatically or duplicate the bundle to bypass it.

Keep per-arm logs, all-update loss histories, initial/saved states, periodic partials, all diagnostic statuses, resource/provenance receipts and status history. Source/config hashes are checked before each arm and at shutdown. Initial P,V and normalized heads a/q should be checked against the paired q=1 initialization before claiming matched draws; do not require raw heads to match across q.

Use the existing canonical saved-data analyzer with the manifest/configs and execution/data layout. Keep q-specific cells and two reused seeds separate. Plot physical t and normalized tau=q*t, but preserve raw loss and loss/VarY unchanged. The original initial1%/5% whole-update max/min windows, same-checkpoint minimum alignment gain>=0.5, numerical cutoff-envelope refit gain>=0.1VarY, later raw-loss release>=0.1VarY and numerical resolution guards remain mandatory. Do not divide loss by q or weaken the loss window. Censoring, missing diagnostics and finite-order uncertainty remain visible. Promising states require block-rate-weighted update and higher-order geometry/refit checks, with full trajectory order confirmation if sensitivity warrants it.
