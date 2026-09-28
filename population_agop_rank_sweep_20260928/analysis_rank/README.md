# Rank-specific saved-data reporting

This folder presents the declared **32-arm paired rank study**: ranks 2, 4, 8, 16; quadratic and raw-ReLU teachers; ReLU and SwiGLU students; reused seeds 641 and 642. It imports no model or training code. The live bundle, canonical summary code, and existing v8 plotter remain unchanged.

`plot_rank.py` checks the exact manifest grid and all 32 summary IDs, then calls the existing v8 `aggregate`, `metric`, `interpolate`, `normalized_units_check`, `draw_metric` and `cell_axes` functions. The local canonical plot copy must remain byte-identical to the v8 source. Presentation adds natural rank order and a teacher-specific layout: four rank columns and separate ReLU/SwiGLU row groups. Each group contains loss, minimum/mean AGOP alignment, and refit rows.

Each teacher gets three sheets: full trajectories, the initial 1% window, and the initial 5% window. A zoom ends at the smaller of both seeds' loss-only prefix endpoints and their shared history endpoint. Selection does not inspect alignment/refit. Both initial windows are shaded on all sheets. Two PDFs contain the three modes per teacher; matching PNGs/SVGs and aggregate arrays are also saved.

## Run on a downloaded, stable result snapshot

Use the unmodified canonical summarizer first. Set `RANK_EXECUTION` to the **downloaded execution directory containing STATUS/COMPLETE and data/**, not to an individual run or the entire repository. The expected remote leaf is `population_agop_rank_sweep_20260928/bundle/execution_remote_a01`; the local download parent is chosen when pulling the run. This guide does not launch or download anything.

```sh
export RANK_ROOT='/Users/legendkiller/Downloads/Local generalization runs/Gptcode_consolidated_experiments copy/plateau_reset_v1/efficient_200_v1/population_agop_higher_rank_v1/goal_followup_v10/rank_sweep'
export RANK_PY=/Users/legendkiller/anaconda3/bin/python3
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLCONFIGDIR=/tmp/agop-rank-plots
export RANK_EXECUTION='/absolute/path/to/download/population_agop_rank_sweep_20260928/bundle/execution_remote_a01'
export RANK_REVIEW="$(mktemp -d "$RANK_ROOT/analysis_rank/report_XXXXXXXX")"
"$RANK_PY" "$RANK_ROOT/analysis/summarize_coverage.py" \
  --manifest "$RANK_ROOT/bundle/MANIFEST.json" \
  --execution "$RANK_EXECUTION" --output "$RANK_REVIEW/SUMMARY.json"
"$RANK_PY" "$RANK_ROOT/analysis_rank/plot_rank.py" \
  --summary "$RANK_REVIEW/SUMMARY.json" \
  --manifest "$RANK_ROOT/bundle/MANIFEST.json" \
  --output "$RANK_REVIEW/plots" \
  --title 'Rank comparison: paired development seeds 641 and 642'
```

Replace the example `RANK_EXECUTION` path with the actual download. A missing execution directory produces unavailable arms, not evidence that a remote run has finished. Read the summary's process states, reader errors, missing files, censoring and execution receipts before interpreting curves. Stable partial snapshots may be rendered, but all missing arms remain visible and the report does not infer process liveness.

In particular, the inherited `not_started` state can mean **no matching record was available locally**. It cannot establish that the remote arm never started. For the actual batch, wait for completion or obtain an explicitly frozen receipt/data snapshot; reconcile remote receipts and the transfer inventory before assigning a scientific denominator or interpreting nonobservations.

The plot output must be a **new** child directory of `analysis_rank/`; existing directories are refused. The canonical summary records must be stable (`snapshot_usable` and `inputs_unchanged` true), have the current canonical summarizer hash, and hash this exact manifest. Existing source hashes are rechecked before/after rendering. Unavailable original paths are disclosed; a local unavailable path is not claimed as reverified.

## What is preserved

- All 32 planned arms and 16 cells are required. Unknown, duplicated or silently dropped IDs are rejected. Rank/teacher/student/seed/cell, supplied config, clock and recipe mismatches are rejected.
- Canonical error-fallback arms omit some rank/recipe metadata. Only absent **planning metadata** is recovered from the manifest, on a copied record, with every recovered field logged. Histories, metrics, uncertainty flags and outcomes are never filled in.
- Medians/ranges require both prescribed seeds and use only shared finite support. Longer individual tails remain faint. Missing diagnostics remain NaN gaps; no interpolation crosses them. Finite unresolved scores remain visible with warnings, including in medians; a median is not a certificate.
- All plot and axis labels are black; existing curve and window colors are reused. Seed ranges are not confidence intervals.
- Prediction and refit rows show **raw MSE**. Raw-ReLU target variance depends on rank; h2 variance is one. Canonical normalization multiplies saved SwiGLU normalized values by target variance exactly once. The variance appears in every cell. The profiled output intercept fits the unchanged raw teacher mean.
- Original and numerically qualified same-checkpoint candidates, censoring, later-release fields, process statuses and issues pass through unchanged. The canonical `0.1 Var(Y)` refit/later-loss criteria govern; old native ReLU `>1e-5` flags do not replace them.
- Cell headers separately count **loss-prefix censored/unavailable** arms and **diagnostic gaps/warnings** in each initial window. A closed loss window does not imply diagnostic coverage. The latter count includes missing/unresolved prefix records, failed or absent screens, and recorded integrity warnings; it does not alter any candidate, curve, or canonical assessment.
- ReLU accepted-force time/bounded refit and SwiGLU adaptive-flow time/cutoff-sensitive unrestricted refit remain separate. The two reused seeds are development evidence; rank-8 reruns are matched controls, not fresh-seed confirmation. Matching initialization must be independently verified from stored arrays.

Outputs: `RANK_REPORT.md` (32-row table), `RANK_RECORDS.json` (complete per-arm records grouped by rank), `PLOT_QA.json`, `PLOT_PROVENANCE.json`, six PNG/SVG sheets, two three-page PDFs, and an aggregate NPZ for each cell with paired support. Plot provenance records every output hash. `visual_review` remains pending until someone inspects the actual-run figures; fixture review does not certify future figures.

## Recovered original attempt a01

Start with [the recovery report](recovery_a01/RECOVERY_RESULTS.md), [unchanged canonical summary](recovery_a01/SUMMARY.json), and [separate stopped-process reconciliation](recovery_a01/RECONCILIATION.json). The [download audit](recovery_a01/DOWNLOAD_AUDIT.json) verifies all 1,572 inventory files. The original denominator remains 32 arms: 20 terminal result markers and 12 without a terminal marker; only 19 child exit codes were recorded. Terminal markers do not guarantee complete diagnostics.

The current six-sheet gallery is [plots_v2](recovery_a01/plots_v2/), with explicit diagnostic-coverage counts. The earlier [plots](recovery_a01/plots/) gallery is retained as the initial presentation version. Its source data and numerical curves are unchanged. Review [visual QA](recovery_a01/VISUAL_QA.json) and [final verification](recovery_a01/FINAL_QA.json) alongside the figures. Retry attempts are separate data and must not silently replace these original records.

## Synthetic checks and visual fixture

`make_fixture.py` generates clearly labeled artificial trajectories, unequal endpoints, warning flags, missing diagnostic gaps, and four unavailable seed arms. It reads the manifest for labels only. It does not read rank-sweep results or perform model evaluations.

The checked-in [final visual fixture](fixture/FINAL_VISUAL.png) and [six-sheet fixture gallery](fixture/gallery_v1/) are **synthetic layout checks, never actual rank-study findings**. [VERIFICATION.json](VERIFICATION.json) and [test output](fixture/TEST_RESULTS.txt) record the original 12-test validation. These historical artifacts predate the explicit diagnostic-coverage header addition; the recovery report records the current 13-test verification. Their earlier source hashes are historical, not hashes of the current wrapper.

```sh
"$RANK_PY" -m unittest discover -s "$RANK_ROOT/analysis_rank" -p test_rank.py -v
export RANK_FIXTURE="$(mktemp -d "$RANK_ROOT/analysis_rank/fixture_XXXXXXXX")"
"$RANK_PY" "$RANK_ROOT/analysis_rank/make_fixture.py" \
  --manifest "$RANK_ROOT/bundle/MANIFEST.json" \
  --output "$RANK_FIXTURE/SYNTHETIC_SUMMARY.json"
"$RANK_PY" "$RANK_ROOT/analysis_rank/plot_rank.py" \
  --summary "$RANK_FIXTURE/SYNTHETIC_SUMMARY.json" \
  --manifest "$RANK_ROOT/bundle/MANIFEST.json" \
  --output "$RANK_FIXTURE/gallery" \
  --title 'SYNTHETIC rank presentation fixture' --synthetic-fixture
```

Synthetic mode requires both an explicit fixture marker in the input and a SYNTHETIC title. Omitting the flag does not allow synthetic input to masquerade as live results. Conversely, real summaries are rejected in synthetic mode.

Tests exercise grid/identity failures, clock/recipe mismatch, manifest-only fallback preservation, unstable snapshots, missing-seed no-median behavior, shared support, missing-point gaps, warning propagation, raw-unit checks, loss-only zoom selection, diagnostic coverage independent of loss censoring, and all-arm report retention. One test invokes the unchanged canonical summarizer against an explicitly nonexistent fixture execution path to verify its real missing-input schema; no training or numerical model evaluation occurs.
