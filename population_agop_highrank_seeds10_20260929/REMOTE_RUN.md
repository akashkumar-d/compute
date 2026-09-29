# Dispatch: ten-seed high-rank cohort (three chained jobs)

Prerequisites:

1. Each `bundle_a0*/REVIEW.json` says `approved_for_execution` for that bundle's current manifest hash.
2. This leaf is committed and pushed to `codex/agop-breadth14-20260928`.
3. `lrun check` and `lrun status` show at most 2 other task workers active, so the aggregate stays at 30 or fewer.

Do not dispatch if another job would push the total above 30 workers.

```sh
lrun repo https://github.com/akashkumar-d/compute -n agop-highrank-s10-20260929-a01 -b codex/agop-breadth14-20260928 -- 'AGOP_EXECUTION_SITE=SERVER PYTHONDONTWRITEBYTECODE=1 ../agop-breadth14-r8-cpu64-20260928-v1/agop_runtime/bin/python population_agop_highrank_seeds10_20260929/bundle_a01/launcher/launch.py --execution-dir execution_a01'
lrun repo https://github.com/akashkumar-d/compute -n agop-highrank-s10-20260929-a02 -b codex/agop-breadth14-20260928 -- 'AGOP_EXECUTION_SITE=SERVER PYTHONDONTWRITEBYTECODE=1 ../agop-breadth14-r8-cpu64-20260928-v1/agop_runtime/bin/python population_agop_highrank_seeds10_20260929/bundle_a02/launcher/launch.py --execution-dir execution_a02'
lrun repo https://github.com/akashkumar-d/compute -n agop-highrank-s10-20260929-a03 -b codex/agop-breadth14-20260928 -- 'AGOP_EXECUTION_SITE=SERVER PYTHONDONTWRITEBYTECODE=1 ../agop-breadth14-r8-cpu64-20260928-v1/agop_runtime/bin/python population_agop_highrank_seeds10_20260929/bundle_a03/launcher/launch.py --execution-dir execution_a03'
lrun status agop-highrank-s10-20260929-a01
lrun logs agop-highrank-s10-20260929-a01 -n 20
lrun pull agop-highrank-s10-20260929-a01 <fresh local dir> --only population_agop_highrank_seeds10_20260929/bundle_a01/execution_a01
```

Jobs a02 and a03 wait on the named previous run. Their queue receipts are in `STATUS.json` with status `queued`. If a dependency never resolves, the job stops with `queue_timeout` without running any arm. Launch it manually after checking the previous job is terminal.

Preserve failed, capped and timed-out attempts. Never reuse an execution directory, and never relaunch a completed job.
