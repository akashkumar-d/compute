# Cubic head-scaling pilot: saved-data reporting

**Preparation only:** no real pilot outcomes have been analyzed here. [The fixture gallery](preparation/synthetic_gallery_v1/) contains artificial data and is labelled **SYNTHETIC FIXTURE ONLY**. It checks presentation and clock handling, not scientific success. The frozen [protocol](../bundle/PROTOCOL.snapshot.md) and [six-arm manifest](../bundle/MANIFEST.json) define q = 1, 0.3, 0.1, each with reused seeds 641/642. All arms use the same quadrature orders 32/96/48/24.

The wrapper uses unchanged canonical metrics, two-seed medians, interpolation, warning marks and colors. Each pair is aggregated **once in physical time**. The normalized view changes only copies of the horizontal coordinates and prefix boundaries to τ = q t; metric/flag arrays are identical between views. Original source histories, raw losses, refits, thresholds and candidate membership are untouched. Recorded candidate times remain physical in the output JSON.

## Analyze a verified download

Wait for a completed transfer and its inventory/hash check before reading results. A frozen partial snapshot may be analyzed with all six arms retained; it is not a process-liveness check. Missing local data, inherited `not_started` labels or zero diagnosed candidates do not establish that a remote job never started or that learning failed. Preserve incomplete, capped, missing and numerically unresolved arms.

```sh
export HEAD_ROOT='/Users/legendkiller/Downloads/Local generalization runs/Gptcode_consolidated_experiments copy/plateau_reset_v1/efficient_200_v1/population_agop_higher_rank_v1/goal_followup_v10/headscale'
export HEAD_PY=/Users/legendkiller/anaconda3/bin/python3
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLCONFIGDIR=/tmp/agop-headscale-plots
export HEAD_EXECUTION="$HEAD_ROOT/remote_results/population_agop_swiglu_headscale_20260928/bundle/execution_remote_a01"
export HEAD_REVIEW="$(mktemp -d "$HEAD_ROOT/analysis/report_XXXXXXXX")"
"$HEAD_PY" "$HEAD_ROOT/../rank_sweep/analysis/summarize_coverage.py" \
  --manifest "$HEAD_ROOT/bundle/MANIFEST.json" \
  --execution "$HEAD_EXECUTION" --output "$HEAD_REVIEW/SUMMARY.json"
"$HEAD_PY" "$HEAD_ROOT/analysis/plot_headscale.py" \
  --summary "$HEAD_REVIEW/SUMMARY.json" \
  --manifest "$HEAD_ROOT/bundle/MANIFEST.json" --output "$HEAD_REVIEW/plots"
```

Set `HEAD_EXECUTION` to the actual verified local download if the coordinating task chose a different parent directory. Its remote leaf above comes from the launch receipt. Use a fresh output for every snapshot; plot outputs refuse overwrites. To redraw saved results, reuse a stable existing `SUMMARY.json` and choose a fresh plot directory. NumPy and Matplotlib are available in the stated interpreter. No training, model evaluation or network call is needed.

`plot_headscale.py` also runs the read-only saved-initialization audit. It follows recorded result/raw-file links to the step-0 state, checks P and V against q=1 within each seed, and checks both the raw head scaling and a/q agreement. Missing or inconsistent initial records stay explicit. A planned shared seed does not establish matching initialization; inspect `INITIALIZATION_AUDIT.json` before making that claim. The standalone audit command is:

```sh
"$HEAD_PY" "$HEAD_ROOT/analysis/paired_initialization.py" \
  --manifest "$HEAD_ROOT/bundle/MANIFEST.json" \
  --execution "$HEAD_EXECUTION" --output "$HEAD_REVIEW/INITIALIZATION_AUDIT.json"
```

## Outputs and interpretation

Each clock has full, initial-1% and initial-5% sheets, in PNG/SVG and a three-page PDF. Columns follow q = 1, 0.3, 0.1. Rows show raw MSE, AGOP minimum/mean and raw refit MSE. Every median requires both prescribed seeds on shared finite support; bands are seed ranges, not confidence intervals. Missing values stay gaps and finite unresolved values retain their flags. The zoom depends only on loss-prefix endpoints and shared history support, never alignment or refit.

`RECORDS.json` keeps all six physical-time arm records and original/qualified/later-release counts. `aggregate_q*.npz` stores both `x_t` and `x_tau`, with one unchanged set of metric/flag arrays. `PLOT_QA.json`, `INITIALIZATION_AUDIT.json` and `PROVENANCE.json` record units, support, initialization status and input/output hashes. Plotting does not resolve scientific warnings. Qualified later-release counts use `first_qualified_complete_sequence`, not the canonical cell-level original-criterion count.

The fixed h3 target has Var(Y)=1, so raw and variance-normalized risk coincide here; risk is never divided by q. The same 1.01/1.05 whole-update loss windows, ΔA_min ≥ 0.5, cutoff-envelope refit gain ≥ 0.1 Var(Y), and later within-run loss decrease ≥ 0.1 Var(Y) remain in force. No displayed interpolation decides a candidate.

Smaller q changes the dynamics and may delay or prevent success. A stretched physical clock alone is not a longer plateau. AGOP can change through head dynamics even with fixed hidden features; the separate material feature-refit criterion is necessary. Common-order screens and two reused seeds do not establish a formal population or high-probability claim. Any promising results require separate higher-order block-weighted update and geometry/refit validation.

## Preparation checks only

[Clock tests](test_clock_display.py) exercise copy isolation, exact metric/flag preservation, shared support and missing gaps under positive clock scaling, no one-seed median, fixed six-arm identity and invalid scales. The paired-initialization helper has its own bounded synthetic checks. These do not repeat the canonical numerical test suite or read actual pilot results.

```sh
"$HEAD_PY" -m unittest discover -s "$HEAD_ROOT/analysis" -p 'test_*.py' -v
```

Synthetic rendering requires both a synthetic input marker and `--synthetic-fixture`; real summaries cannot be passed as fixtures. [Preparation QA](preparation/PREPARATION_QA.json) and [independent clock review](preparation/INDEPENDENT_CLOCK_REVIEW.md) record the checks. They do not certify future result figures.
