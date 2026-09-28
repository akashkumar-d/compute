# Verified32CPU rank-comparison dispatch

Use the current shared branch codex/agop-breadth14-20260928 after this leaf is committed and pushed. The user-designated Studio account was verified privately before dispatch. The rank bundle uses28single-thread workers while the separate v9 validation uses2; total30leaves2CPUsfree. Runtime32GiB/10GiB admission,1990s perarm,3600s global. The review approves execution of the exact manifest, not numerical or scientific outcomes.

```sh
lrun repo https://github.com/akashkumar-d/compute -n agop-ranks-2to16-20260928-a01 -b codex/agop-breadth14-20260928 -- 'AGOP_EXECUTION_SITE=SERVER PYTHONDONTWRITEBYTECODE=1 ../agop-breadth14-r8-cpu64-20260928-v1/agop_runtime/bin/python population_agop_rank_sweep_20260928/bundle/launcher/launch.py --execution-dir execution_remote_a01'
lrun logs agop-ranks-2to16-20260928-a01
lrun status agop-ranks-2to16-20260928-a01
lrun pull agop-ranks-2to16-20260928-a01 /absolute/local/fresh-output --only population_agop_rank_sweep_20260928/bundle/execution_remote_a01
```

Do not launch again if dispatch is pending or an existing process is active. Preserve failed and capped runs. Use the original canonical analyzer in analysis/, passing the new bundle/MANIFEST.json and downloaded execution directory. The copied numeric analysis functions are unchanged; explicit rank cell IDs prevent pooling ranks. Paired development plot mode is needed for this16-cell subset. Sort/present rank numerically, and keep losses raw with target variance specified; any reporting-format adjustment is separate from training. No manuscript changes are part of this run.
