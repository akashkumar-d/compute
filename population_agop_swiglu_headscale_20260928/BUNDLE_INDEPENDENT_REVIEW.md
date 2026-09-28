# Independent review of the concrete cubic head-scaling bundle

**PASS for source/configuration integrity and the reviewed runtime implementation; dispatch remains pending.** No scientifically material discrepancy was found. This review does not establish numerical validity of a new cubic trajectory, completion of the v9 recovery/review, or current external capacity. `bundle/REVIEW.json` remains unchanged at `pending_independent_runtime_review`, with `execution_launched=false`.

Reviewed exact manifest SHA256: `7042a39550d2499d656dd6ae5a3d8a7795d40df83641ae603518a795f98477e0`.

Reviewed design SHA256: `f7769bd338ee43de91d4cdc53fcb76a34e7ad76623a2fe7615b6156c02d3fa5d`.

The review read the complete launcher, launcher diff, all 24 mock-test definitions, runner budget/recording code, manifest, lineage, protocol/design snapshots, preparation/runtime receipts, and six configurations. All verification used file reads and standard-library arithmetic; no scientific module was imported, no model was evaluated, and no training, network operation, or manuscript edit occurred.

## Integrity and scientific design

- **All 34 frozen file hashes pass.** The two manifest source maps agree; the recorded launcher diff exactly reproduces the change from the preserved rank scheduler. Manifest, design and prior runtime-review hashes match the preparation receipts.
- **All 11 scientific files, including `SOURCE_MANIFEST.json`, are byte-identical to v8.** This includes `swpop.py`, `swsmall_periodic.py`, diagnostics and `run_one.py`. The historical source review is preserved as lineage evidence; it is not approval to dispatch this new experiment.
- **All six bundled configuration files are byte-identical to the previously reviewed originals.** Their embedded manifest objects, configuration hashes, labels, seed/q identities and v8 change lists agree. The design, protocol and independent algebra-review snapshots also match their current parent files.
- The experiment remains raw h3, r8/d64/m64, s0.3, alpha1, seeds641/642, q in {1,0.3,0.1}, head ratio q, head rate0.01q², common orders32/96/48/24, physical horizon3000/q, `dt_max=50/q`, checkpoint floor0.5/q, 15000 updates and normalized loss stop0.01. The initial-window, same-state alignment/refit and later-release criteria are unchanged.

The previous algebra qualifications still apply. q changes the normalized vector field; it does not make exact copies of the q=1 trajectory. The adaptive step metric includes the raw head. Relative normalized heads may change AGOP geometry even with fixed hidden features. Small q can permit fast-neuron growth or worsen directional imbalance; it provides no all-direction-plateau guarantee. The common quadrature orders are a provisional matched recipe, and promising states still require actual block-weighted update and higher-order geometry/refit checks. No automatic higher-order comparison is added by this bundle.

## Runtime semantics

The executable entry point validates the frozen hashes and exact six-arm design before execution. Normal execution requires Linux plus `AGOP_EXECUTION_SITE=SERVER`; local execution is refused. The manifest-specific review gate requires `approved_for_execution`, the exact manifest hash, reviewer identity and nonempty checks. The pending record currently fails this gate. The static dry-run path creates no execution directory and does not import the scientific code.

Admission requires at least32 effective allocated CPUs, six workers plus26 reserved slots, 12GiB available memory and5GiB free disk. CPU affinity and available cgroup quotas/limits enter the resource read. Nice>=10 is established and verified before launching children; all six thread controls are set to1 and GPU visibility is disabled. The child inherits the low priority. The six configurations are intended to start in one batch; process-launch overhead still occurs sequentially.

`build_command` passes **1800 seconds soft training and180 seconds soft diagnostics** to the unchanged SwiGLU runner. Each child has a1990-second hard deadline, and the scheduler has a2150-second global compute deadline. The runner's deadline excludes10 seconds for cleanup; diagnostics also stop before that soft deadline. A current training/diagnostic evaluation may overrun its soft allocation, so a complete180-second diagnostic phase is not guaranteed. The scheduler starts TERM8 seconds before the applicable hard cutoff, KILL2 seconds before, and bounds reaping by the remaining deadline. Unreaped children stay listed as active rather than being falsely declared finished. A zero exit code or existing `result.json` is not a scientific success label.

Source/configuration validation is repeated before each child and at shutdown. A kernel lock and persistent attempt journal prevent a second overlapping attempt in this bundle, refuse an ambiguous prior attempt, and do not automatically kill a prior process group. Existing output directories are refused. The unchanged runner redirects periodic partials into each job's own output directory; it does not put this batch's partials into another attempt's shared recording directory. Logs, raw histories, snapshots and incomplete-diagnostic statuses remain available for later recovery.

**External concurrency and v9 validation are manual gates.** The launcher does not count other bundles' workers, coordinate their locks, or inspect scientific v9 results. Its26-slot reservation is an admission rule, not evidence that those slots are idle. `dependency=null` is mandatory, so this bundle neither waits on v9 automatically nor dispatches when another run changes state. The parent must establish the prerequisites and record them before changing the review status. Generic nonempty review checks are not machine verification of those external facts.

## Existing verification and outstanding dispatch conditions

The checked receipts report24 standard-library/mock tests passing during preparation and again in the parent's independent runtime review; both receipts refer to this same manifest and frozen test source. I inspected the24 definitions and did not repeat already-passing tests or dry runs. Coverage meaningfully includes source/config/manifest mutation, path escape and duplicate IDs, fixed design/order/criteria, six-worker scheduling, command budget arithmetic, hard caps and pending preservation, unreaped-child reporting, SERVER-only admission, fatal priority failure, review rejection and restart-journal preservation. These mocks do not certify real-server performance, live capacity, or numerical population accuracy.

No execution directory exists in the reviewed local bundle. No code change is required by this review. Before dispatch, the parent must still:

1. Recover and review the full relevant v9 higher-order trajectory evidence; a stopped process alone does not satisfy this scientific prerequisite.
2. Check immediately before launch that other task-owned training workers are at most24, combined workers will remain at most30, and current CPU/memory/disk admission is satisfied.
3. Complete the exact-manifest review record and the required shared-repository publication/staging, then explicitly dispatch the reviewed bundle.

Until those conditions are met, this document records **reviewed preparation, not permission to dispatch**. No pending gate, source file, configuration, launcher, test, prior result or manuscript was changed. Only this review file was created.
