# Explicit rank-two diagnostic overlay and updated figures

The updated view is [report_with_sidecar/REPORT.md](report_with_sidecar/REPORT.md). Its [COMBINED32_WITH_SIDECAR.json](report_with_sidecar/COMBINED32_WITH_SIDECAR.json) retains the fixed20-original/12-retry rule and replaces only the diagnostic-derived fields of original h2/rank2/SwiGLU/seed642, using the separately reviewed, completed diagnostic sidecar. The pre-sidecar [report_a02](report_a02/) remains unchanged.

No synthetic result.json or process receipt is created. `complete_with_sidecar.py` supplies a transient Reader subclass that reads/hashes the original result normally, then overlays only its `diagnostics` list in memory. The unchanged canonical `load_arm` and `evaluate_arm` functions perform all metric conversion and scientific qualification. Original history, configuration, completion_record, training_record, process_state and process_receipt are checked for exact equality. The sidecar process receipt and origin map are separate provenance. All other31 arm records are preserved exactly.

The verification checks the completed sidecar's eight transferred files against their inventory; its reviewed source manifest; all five original input hashes; exact typed values and float bits in the five reused diagnostic records; every original snapshot's identity/time and archive shape; all269 successful rows; exactly264 new diagnostics in the declared order; zero training updates; and unchanged source inputs. There is no new seed, training run, quadrature comparison, or local model evaluation.

The completed grid reveals an earlier first qualified complete sequence for the target arm at update56/time7.0228243607 (the sparse original grid's first diagnosed candidate was update222/time12.341342729). The gain in minimum-direction alignment is0.5422169483; the cutoff-envelope refit gain is0.4327099717 Var(Y); later raw loss decreases by0.9899538225 Var(Y). Both initial tolerances retain the same15/32 combined qualifying count. These are unchanged-rule saved numerical observations, not formal certificates.

Full diagnostic coverage is now30/32. The two rank-four seed642 retry gaps remain: quadratic teacher293/462 missing; raw-ReLU teacher123/340 missing. Both original and retry attempts, wall censoring, numerical warnings and the full32 denominator remain available.

Updated figures:

- [Quadratic full](report_with_sidecar/plots/h2_full.png), [initial1%](report_with_sidecar/plots/h2_initial_1pct.png), [initial5%](report_with_sidecar/plots/h2_initial_5pct.png), [three-page PDF](report_with_sidecar/plots/h2_rank_sheets.pdf).
- [Raw-ReLU full](report_with_sidecar/plots/relu_full.png), [initial1%](report_with_sidecar/plots/relu_initial_1pct.png), [initial5%](report_with_sidecar/plots/relu_initial_5pct.png), [three-page PDF](report_with_sidecar/plots/relu_rank_sheets.pdf).

SVGs, all16 aggregate NPZs, full rank records, plot QA and complete hash provenance accompany the figures. All six PNG sheets were visually inspected. The existing rank/canonical plot helpers are unchanged; the wrapper changes only the permitted output root. Medians/ranges use both prescribed seeds on shared finite support; missing diagnostics stay gaps; clocks/refit classes and raw target variance remain explicit. The original target's stale process-state label is a historical original receipt, not the state of the completed sidecar.

Eight synthetic tests pass (four fixed-attempt tests and four sidecar tests). [FINAL_QA.json](report_with_sidecar/FINAL_QA.json) and [VISUAL_QA.json](report_with_sidecar/VISUAL_QA.json) record checks; the separate [final independent review](independent_review/FINAL_SIDECAR_REVIEW.md) approves the derived report and six PNG sheets.

To reproduce into a new output directory:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /Users/legendkiller/anaconda3/bin/python3 plateau_reset_v1/efficient_200_v1/population_agop_higher_rank_v1/goal_followup_v10/rank_retry_a02/analysis/complete_with_sidecar.py --output plateau_reset_v1/efficient_200_v1/population_agop_higher_rank_v1/goal_followup_v10/rank_retry_a02/analysis/report_with_sidecar_reproduced
```

The existing output name is intentionally refused. No input files are overwritten. This command reads saved data and renders figures only; it neither trains nor evaluates models.

Source clarification: this rank report uses the complete v8/v10 cell-aware summarizer (SHA256 `57ca1a05…`), not a wrapper around the v7 reader (`fe0aedb…`). Only cell grouping differs; per-arm readers, thresholds and scientific qualification are unchanged, as checked in the final independent review.
