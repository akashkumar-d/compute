# Selected development-state order audit

Prepared for review only. No model evaluation, training, network access or dispatch occurred locally. This package adds a small fixed-state numerical check to three designated development positives; it is neither a fresh replication nor a full-trajectory or eigenspace certificate.

The final combined32 summary is pinned in SELECTION.json. Each case selects initialization, the first qualified saved candidate in its whole initial1% loss prefix, and its exact all-update initial5% endpoint. All nine states are retained regardless of outcome:

| Teacher/student | Rank/seed | Saved update indices |
|---|---|---|
| quadratic/SwiGLU | 8/641 | 0,102,350 |
| quadratic/SwiGLU | 16/641 | 0,247,372 |
| raw-ReLU/SwiGLU | 2/641 | 0,111,298 |

Every case uses its original head_lr=.01, width64/dimension64 and all unchanged configuration fields. Native orders32/96/48/24 are compared with64/192/96/48. Raw gradient comparisons and actual directions m*(gP,gV,head_lr*ga) are both reported, together with the original full-neuron relative cap and hypothetical dt-scaled update. No update is applied. Relative differences use the higher-order norm (or higher-order dt) as denominator and are fractions, not percentages. All bias coordinates enter the cap denominator.

The v9 fixed-state comparator and frozen update_directions are reused. The original eleven scientific/support files remain unchanged. A thin cache lets unchanged diagnostics.metrics reuse the already computed evaluation and AGOP, avoiding duplicate model calls. Full four-cutoff refit diagnostics, numerical guards, rank eigengaps, learned-projector differences, principal angles and perturbation/high-gap ratios are retained. The latter is a diagnostic, not an eigenvector-error certificate. Same-order initialization is used for both orders' gain tests via the existing cell-aware canonical criteria. The mini criteria record has no trajectory history, deliberately preventing a new later-release claim. Native-versus-original diagnostic differences are exposed rather than silently treating recomputation as exact. Any nonfinite or absolute normalized-loss/alignment/cutoff-risk difference above1e-9 produces a conspicuous NATIVE_REPRODUCTION_MISMATCH flag and native_reproduction_all_pass=false. This reproduction flag is not a new scientific success criterion. Native and doubled qualifications are separately reported; thresholds_survive_doubling requires both.

Original result/raw/snapshot files remain external in their existing server jobs. All nine input file sizes and hashes are pinned against saved remote inventories. Retry paths come from the saved scheduler command receipts. SELECTION.json stores the exact states, original diagnostic records and original canonical candidate metadata. No NPZ is bundled. Both the parent scheduler and each child verify inputs before scientific imports/evaluation; source/selection hashes and the input receipt are recorded before computation and checked at completion. Snapshot arrays are read without pickle and made read-only. Original files are never written.

The inherited reviewed scheduler is retained with a narrow protocol delta in launcher/CHANGES.diff. It runs at most two single-thread CPU workers, one case per worker, with250s soft case time,270s hard case time and600s global time. It requires Linux/SERVER, at least32 effective CPUs,8GiB available memory,5GiB disk, verified nice>=10, all six thread controls=1 and no GPU. The child also requires its scheduler-parent PID. An exact-manifest REVIEW.json is required; no local scientific mode or auto-dispatch exists. Root must verify at most28 other live task workers immediately before dispatch, so the combined allocation remains at most30 workers on32 CPUs. It must use the existing numerical environment without silent dependency upgrades.

Each child atomically saves FIXED_STATE_AUDIT.json after every evaluated order. ALL_STATES.json is created before worker launch and refreshed after the scheduler ends, retaining all nine slots even if a worker is missing, failed, interrupted or capped. Successful, partial and unevaluated slots must all be reported. No retries or outcome filtering occur. Existing output and uncertain restart state are refused; TERM/KILL applies only to this scheduler's own process groups. The pre-existing scheduler tests are not repeated; four focused synthetic checks cover the new weighted-cap/cache path, same-order baseline qualification, runtime bounds and nine-slot preservation.

Safe local checks (the test command requires NumPy but imports no scientific model module):

```sh
PYTHONDONTWRITEBYTECODE=1 python3 launcher/launch.py --dry-run
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m unittest discover -s adapter -p test_delta.py
```

After independent approval, publication and fresh capacity checks, root may run remotely:

```sh
AGOP_EXECUTION_SITE=SERVER PYTHONDONTWRITEBYTECODE=1 nice -n 10 python3 launcher/launch.py --execution-dir execution_remote_a01
```

No scientific result is yet available. Fixed-candidate threshold survival at doubled order would support those selected saved-state claims only; it would not validate unsaved states, an entire doubled-order trajectory, unselected arms, or independent-seed confirmation.
