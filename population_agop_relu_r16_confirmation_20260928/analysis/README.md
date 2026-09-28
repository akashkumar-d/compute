# Four fresh-seed confirmation: saved-data analysis

This cohort contains exactly four ReLU/h2 rank-16 runs, seeds **9351–9354**, fixed before their outcomes. Development seeds 641/642 remain separate. Read the frozen [protocol](../bundle/PROTOCOL.md) and [manifest](../bundle/MANIFEST.json) for the selected recipe and unchanged criteria. Historical “unlaunched” language in the frozen protocol describes preparation, not current process status.

Start with [the result report](report_a01/RESULTS.md), [four-seed median figure](report_a01/four_seed_median.png), and [canonical per-run summary](SUMMARY.json). The [independent arithmetic review](independent_review/) and [median-adapter review](median_review/) address saved-data correctness separately from [visual QA](VISUAL_QA.json). [FINAL_QA.json](FINAL_QA.json) records hashes and final checks.

The confirmation denominator is **4** for every outcome. All four process exit codes are zero and all four planned diagnostic grids are complete. All four meet original and additional numerical qualification in both initial 1%/5% windows, followed by the required later loss release. Each stops at the force-time horizon 1500; final raw MSE is approximately 0.0193–0.0194, above the 0.01 loss target. Numerical qualification is not a formal population certificate or a high-probability claim.

## Reproduce from the completed download

The coordinating task verified the complete transfer in [COMPLETED_TRANSFER_QA_20260928_2308.json](../../COMPLETED_TRANSFER_QA_20260928_2308.json): 657 files, 90,050,223 bytes, no missing or mismatched files. Use a completed, stable snapshot and a **fresh** output directory. These commands read saved data only and import no model or training code.

```sh
export CONFIRM_ROOT='/Users/legendkiller/Downloads/Local generalization runs/Gptcode_consolidated_experiments copy/plateau_reset_v1/efficient_200_v1/population_agop_higher_rank_v1/goal_followup_v10/relu_rank16_confirmation'
export CONFIRM_PY=/Users/legendkiller/anaconda3/bin/python3
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLCONFIGDIR=/tmp/agop-confirmation-plots
export CONFIRM_EXECUTION="$CONFIRM_ROOT/remote_results/population_agop_relu_r16_confirmation_20260928/bundle/execution_confirmation_a01"
export CONFIRM_REVIEW="$(mktemp -d "$CONFIRM_ROOT/analysis/review_XXXXXXXX")"
"$CONFIRM_PY" "$CONFIRM_ROOT/../rank_sweep/analysis/summarize_coverage.py" \
  --manifest "$CONFIRM_ROOT/bundle/MANIFEST.json" \
  --execution "$CONFIRM_EXECUTION" --output "$CONFIRM_REVIEW/SUMMARY.json"
"$CONFIRM_PY" "$CONFIRM_ROOT/analysis/report_confirmation.py" \
  --summary "$CONFIRM_REVIEW/SUMMARY.json" \
  --manifest "$CONFIRM_ROOT/bundle/MANIFEST.json" \
  --output "$CONFIRM_REVIEW/plots"
```

To redraw without repeating saved-data summarization, point `--summary` at the existing `analysis/SUMMARY.json` and choose a new output directory. The wrapper refuses an existing output directory. It checks the exact four planned IDs/seeds, the unchanged recipe against the frozen development reference, stable summary provenance, source hashes and all locally available input hashes before and after rendering. No completed run is overwritten.

The runtime needs NumPy and Matplotlib, already present in the interpreter above. Canonical summarization also uses the dependencies of the existing rank saved-data reader; it performs no training or model evaluation. No remote command, GPU or additional run is needed for these reports.

## Plot and criterion conventions

- Three individual-seed sheets show full trajectories and initial 1%/5% loss-prefix zooms. Every column is one fixed seed and uses denominator 1. No seed is selected from the outcomes.
- The separate four-seed median sheet shows full, 1% and 5% views with denominator 4. Its bold curve exists only where **all four** seeds have finite interpolable values; shading is their min–max range, **not a confidence interval**. Each median zoom ends at the smallest of the four loss-only prefix endpoints and shared history support. Individual-seed sheets use each seed’s own loss-only window.
- `aggregate_four` is an explicit adapter. It reuses canonical metric extraction and gap-preserving interpolation, then takes min/median/max over all four seeds. A separate order-statistic calculation checks the median as the mean of the two middle sorted values. The canonical two-seed aggregate is untouched. There is no fallback to a three-seed or one-seed median.
- Colors, raw-MSE units, numerical-warning marks and curve drawing reuse the canonical rank plotter. Finite unresolved scores remain visible with warnings; missing values remain gaps. Diagnostic gaps/warnings and loss-prefix censoring are distinct header counts.
- Same-state criteria come directly from the unchanged canonical summary: ΔA_min ≥ 0.5, conservative bounded-refit gain ≥ 0.1 Var(Y), passing initial/current guards and additional numerical screens. Later loss release is a separate within-run decrease ≥ 0.1 Var(Y) after that candidate. No criterion is inferred from an interpolated median.
- Qualified later-release totals explicitly use `first_qualified_complete_sequence`. The canonical cell-level `complete_sequences` field tracks the original criterion and must not be relabeled as qualified.
- h2 has E[Y²]=Var(Y)=1, so raw and variance-normalized risk coincide here. The unchanged raw teacher uses a profiled intercept. Refit budgets remain l2 ≤ 32 and l1 ≤ 256; the accepted-force clock is used.

Outputs include four PNG/SVG sheets, a four-page PDF, per-seed plotted arrays, `aggregate_four_fresh_seeds.npz` with support counts and screens, all four complete records, exact candidate records, plot QA and hash provenance. No fresh quadrature, Jacobian or model-state audit is implied. This confirms only the selected rank-16 quadratic/ReLU recipe with these four fresh seeds.
