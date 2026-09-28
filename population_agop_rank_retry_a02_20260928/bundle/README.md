# Rank sweep attempt 2: unfinished-arm retry only

Prepared at the user's explicit request to rerun stopped jobs. This bundle reruns exactly twelve unfinished SwiGLU arms from their unchanged initialization: ranks16/8/4 × h2/raw ReLU × seeds641/642. It retains original IDs and config bytes. Each manifest entry adds attempt2 provenance; no seed, precision, horizon, update cap, teacher, loss stop, or success criterion changes. The original32-arm study remains the denominator. These retries add zero independent seeds.

The frozen remote file inventory, not the stale scheduler count, identifies20 completed markers: all16 ReLU DONE.json files and all4 rank2 SwiGLU result.json files. Those20 arms are excluded from execution. A marker denotes a completed artifact for retry selection, not scientific success. All old interrupted/partial artifacts remain in the original attempt. No partial local download was used to select jobs. Reference manifest and inventory hashes are enforced by the launcher, as are the exact original config bytes and scientific sources.

Runtime: Linux SERVER only;12 single-thread workers,20 reserved slots;32 effective CPUs,32GiB available memory,10GiB free disk. Per arm1990s=1800s soft training+180s soft diagnostics+10s cleanup, global2150s. The existing priority, deadline, TERM/KILL/reaping, source revalidation and persistent restart-journal rules are retained. A current evaluation may overrun a soft budget; pending, capped and incomplete-diagnostic outcomes remain explicit. The reservation does not inspect other jobs: the parent must verify that the combined task-owned worker count stays<=30, leaving2CPUs free. No GPU use, local training, auto-dispatch, network call or machine change is part of preparation.

Use a separate shared-repository leaf `population_agop_rank_retry_a02_20260928`, a new remote job `agop-ranks-retry-20260928-a02`, and the new output `execution_remote_a02`; the parent owns publication and dispatch. Do not write into or resume the old execution directory, and do not rename the original scientific IDs. `MANIFEST.snapshot.json` plus `PROVENANCE.json` links all original-ID result/status records to this attempt2 bundle. Preserve both attempts; do not count retries as new study arms or splice their clocks into one trajectory.

`REVIEW.json` is deliberately pending. The parent independently reviews this exact manifest, verifies old jobs are stopped and current capacity is sufficient, then records approval and publishes before manually dispatching. The previous rank bundle's review is historical lineage, not current approval.

Static and mocked checks, which never import scientific runners:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 launcher/launch.py --dry-run --execution-dir execution_remote_a02
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s launcher -p test_launch.py -v
```

After review and staging, the parent invokes its verified server interpreter on:

```sh
AGOP_EXECUTION_SITE=SERVER PYTHONDONTWRITEBYTECODE=1 python3 launcher/launch.py --execution-dir execution_remote_a02
```

This is a bounded attempt to complete interrupted observations. Common order32/96/48/24 remains provisional; candidate update/AGOP/refit precision checks and trajectory sensitivity review remain required. Apply the original canonical same-state .1Var criterion and initial1%/5% windows, retain all32 original IDs and both attempt records, and report which attempt supplies each displayed trajectory. Keep old unresolved observations and do not convert process completion into scientific success.
