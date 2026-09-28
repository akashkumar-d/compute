# Remote execution after Studio restart

Prepared and independently reviewed; no training has started. This bundle contains two numerical-order recipes for the same development seed642, not two independent seeds. Manifest39503531b9f5cd0df5fa51d421e48ce797c33bfdbfd3fe5f926511bf44e19a30.

Use lrun check first. The reviewed scheduler admits Linux SERVER only with>=32effectiveCPUs and8GiBavailableRAM, but uses only2single-threaded nice>=10workers. NoGPU. Scientific source files are unchanged from v8; runtime bounds1800s training/180sdiagnostics perarm,1990hardarm/2150global. Confirm no earlier attempt is active before launching a new attempt name.

From the local project, after this source is pushed on codex/agop-breadth14-20260928:

```sh
lrun repo https://github.com/akashkumar-d/compute -n agop-swiglu-ordercheck-20260928-a01 -b codex/agop-breadth14-20260928 -- 'AGOP_EXECUTION_SITE=SERVER PYTHONDONTWRITEBYTECODE=1 ../agop-breadth14-r8-cpu64-20260928-v1/agop_runtime/bin/python population_agop_swiglu_ordercheck_20260928/launcher/launch.py --execution-dir execution_server_order24_a01'
lrun status agop-swiglu-ordercheck-20260928-a01
lrun logs agop-swiglu-ordercheck-20260928-a01
lrun pull agop-swiglu-ordercheck-20260928-a01 /absolute/local/new-output --only population_agop_swiglu_ordercheck_20260928/execution_server_order24_a01
```

The interpreter path points to the persistent pinned environment prepared for v7 (NumPy1.26.4/SciPy1.11.4). Check its existence and versions with lrun sh before dispatch; do not silently substitute a different runtime. Code is shared here; saved-state reports refer to preserved local v8 inputs and are evidence, not bundled training data. Recompute each trajectory's own all-update initial1%/5%windows and preserve capped/missing/negative outcomes. At<=2%weeklyallowance, checkpoint and pause only task-owned jobs/agents. Do not change manuscripts.
