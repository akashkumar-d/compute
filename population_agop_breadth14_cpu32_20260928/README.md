# Fourteen teachers: 32-CPU execution variant

This resource-only variant implements the user request to use30 of32 allocated CPU cores, leaving2 free. It contains the same60 attempts as the reviewed64CPU preparation:14 canonical teachers x2 students x2 fresh seeds, plus4 supplementary pure-sine scale runs. All executable scientific inputs, configurations, seeds and per-arm budgets are byte-identical. The parent package remains preserved and unlaunched.

Use an isolated interpreter with the pinned dependencies in bundle/requirements.txt. On the verified32CPU Lightning machine:

```sh
AGOP_EXECUTION_SITE=SERVER PYTHONDONTWRITEBYTECODE=1 python start_cpu.py
```

Resource guards require32effective CPUs,24GiB available memory and10GiB free disk. At most30 single-thread children run concurrently; each has900seconds including diagnostics and cleanup. The60 attempts take two waves, with a2100second dispatcher cap and120second setup cap. Terminal/capped receipts do not imply scientific success.

See bundle/PROTOCOL.md for the frozen scientific design and interpretation rules. RESOURCE_VARIANT.json records provenance; RESOURCE_REVIEW.json records the independent resource-only check. The new bundle manifest SHA256 is66e7723363b080b7bb535531f51f7faeb0d0acb88139f8626848e2df7d9ad194. Source is published in the sharedcompute branch codex/agop-breadth14-20260928, subdirectory population_agop_breadth14_cpu32_20260928.

The old supplementary configuration suffix _b64 is a preserved identifier from the parent preparation, not an assertion about the current worker count. No old data, plots or manuscript were edited.
