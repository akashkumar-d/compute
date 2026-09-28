# Retry reporting with a fixed attempt map

The saved-data report is in [report_a02/REPORT.md](report_a02/REPORT.md). It provides the canonical [retry-only summary](report_a02/RETRY12_SUMMARY.json), a separately labeled [derived combined summary](report_a02/COMBINED32_SUMMARY.json), [attempt map](report_a02/ATTEMPT_MAP.json), and [hash/provenance record](report_a02/REPORT_PROVENANCE.json). The original a01 summary and every original/retry result remain unchanged.

`PLAN.json` was frozen before retry outcomes were summarized. Its rule uses original terminal-marker availability only: original a01 records for 20 arms, retry a02 records for exactly the 12 original arms without terminal markers. The denominator stays 32. The original source summaries remain accessible, including the original incomplete attempts. These retries use the same predeclared Gaussian seeds and configurations; no fresh-seed or best-attempt claim is made.

`report.py` imports the unchanged canonical `rank_sweep/analysis/summarize_coverage.py` for the 12-arm summary. The combined summary copies each selected per-arm record verbatim and calls the canonical `build_cells` helper for cell tables. Its `derivation` section explicitly records the combination; it does not claim to be one execution. Per-arm process receipts remain unmodified, including the original rank-two seed642 historical `active_snapshot` label despite the separately verified original job shutdown. No synthesized training data, completion markers, or process receipts are created.

The parent supplied a verified transfer receipt: 113 files, 276,045,471 bytes, exact remote-inventory hashes/sizes, all 12 child processes exit zero. The report verifies that audit's inventory hash, rechecks original and retry canonical input hashes, requires the saved 12/12 terminal status, and preserves the source/summary/configuration hash chain.

Current results, before any diagnostic-completion sidecar:

- Retry12: four qualified complete sequences at each tolerance; all are quadratic SwiGLU (rank4 seeds641/642, rank8 seed641, rank16 seed641).
- Combined32: 15 qualified complete sequences at each tolerance (all32 denominator).
- Three incomplete full diagnostic grids remain: original h2/rank2/seed642 (264 of269 missing), retry h2/rank4/seed642 (293 of462 missing), and retry raw-ReLU/rank4/seed642 (123 of340 missing). Prefix assessments are retained exactly from the canonical summaries; a later-grid gap does not automatically imply an initial-window gap.
- Ten retry trajectories are wall-censored. Process exit zero, horizon completion, diagnostic coverage and scientific outcomes remain separate fields.

The rank-two diagnostics sidecar is deliberately not included in these artifacts. Final combined figures are held pending its status and an explicit separately labeled derived reporting step. The existing rank/canonical plot helpers remain unchanged. The optional plotting path only binds their output directory guard to this authorized analysis folder; it does not change numerical aggregation, interpolation, unit conversion, criteria or drawing. No figures have been rendered here.

Four synthetic selection tests pass. They check the exact20/12 partition, outcome-independent selection even when a retry outcome is worse, verbatim records and immutable sources, missing/duplicate/swapped IDs, scientific-config/source/criteria mismatches, integrity flags, and frozen plan hashes. No tests or reports import model kernels or perform training/evaluation.

Commands used from the repository root (output directories must be fresh):

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s plateau_reset_v1/efficient_200_v1/population_agop_higher_rank_v1/goal_followup_v10/rank_retry_a02/analysis -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 plateau_reset_v1/efficient_200_v1/population_agop_higher_rank_v1/goal_followup_v10/rank_retry_a02/analysis/report.py --execution plateau_reset_v1/efficient_200_v1/population_agop_higher_rank_v1/goal_followup_v10/rank_retry_a02/remote_results/population_agop_rank_retry_a02_20260928/bundle/execution_remote_a02 --download-audit plateau_reset_v1/efficient_200_v1/population_agop_higher_rank_v1/goal_followup_v10/rank_retry_a02/TRANSFER_QA.json --output plateau_reset_v1/efficient_200_v1/population_agop_higher_rank_v1/goal_followup_v10/rank_retry_a02/analysis/report_a02
```

The existing output is preserved and rerunning that exact output name is refused. Read `REPORT_PROVENANCE.json` and require `pass_check=true` before using an output. Independent review is pending; the parent owns review, publication and any later figure/sidecar integration. No network, model calls, training, or manuscript changes occurred.
