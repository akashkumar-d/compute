# Prepared pilot bundle

Not launched. Read PROTOCOL.md and SETTINGS.json. No model evaluation or training
was performed locally. Parent owns source approval, code-only publication and
all dispatch. Do not publish runtime_inputs/, outputs, hidden attempt journals,
or any NPZ. The final MANIFEST.json enumerates every pinned code/config file.

Pure-array/scalar tests: `python -B -m unittest -v test_pilot` with BLAS pools1.
Source-only check (no model imports):
`python -B run.py smoke --output smoke_a01 --cpu 31 --other-workers 0 --dry-run`.

After independent REVIEW.json, parent resource/account/concurrency checks and
same-Studio input preparation, from the published checkout root:

```
AGOP_EXECUTION_SITE=SERVER ../../agop-breadth14-r8-cpu64-20260928-v1/agop_runtime/bin/python -B prepare_runtime.py
bash run_server.sh smoke --output smoke_a01 --cpu 31 --other-workers 0
```

These two examples assume the current directory is the published bundle;
replace the CPU/other-worker count with verified live values. Preparation only
copies already-pinned remote arrays. Source REVIEW.json must contain
status=approved_for_execution, manifest_sha256, reviewer. It is excluded from
the self-referential manifest. Smoke completion does not dispatch training.

Only after independent saved-output review creates SMOKE_REVIEW.json with
status=approved_for_pilot, manifest_sha256, smoke_result_sha256 and reviewer:

```
bash run_server.sh arm --arm h3_scale_0.3_seed641 --output pilot_s03_seed641_a01 --cpu 31 --other-workers 0 --smoke-output smoke_a01
```

All four IDs are in MATCHED_SCALE_PROTOCOL.json. Every arm is a separate bounded
invocation; no automatic batch/retry is provided. A failed or capped smoke
retains all arms and authorizes no training.
