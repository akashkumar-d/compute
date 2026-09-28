# Saved-data coverage summarizer

`summarize_coverage.py` reads saved JSON/JSONL/NPY records only. It does not import scientific models, evaluate states, train, or contact a server. It preserves every manifest arm, including unstarted attempts, and separates the56 canonical runs into28 teacher/student cells from the4 pure-sine supplements. The immutable bundle and prior outcomes are read-only.

Run from any directory with a Python environment containing NumPy:

```sh
python summarize_coverage.py --manifest ../bundle/MANIFEST.json --execution /path/to/pulled/execution --output /path/to/new/SUMMARY.json
```

The execution argument is the directory containing STATUS.json and data/<arm_id>. Historical execution/<arm_id> layouts are also supported. JSON and a readable Markdown report are produced; output inside the immutable bundle or execution input directory is refused. `summarize(manifest_path,execution_path)` returns the same report without writing files. Every read input is hashed with its first-read digest; reread conflicts remain recorded even if a file changes back; a changed input snapshot invalidates candidate reporting rather than silently mixing versions.

## Stable plotting and audit schema

Every `arms` entry has id,cell,student,teacher,seed,r,d,m,scale,head_lr,profiled_intercept,supplemental,config,process_state,target_mean,target_variance,comparison_class and certificate. `history` contains `{step,time,loss_raw,loss_over_target_variance}` for every update. `time_unit` is `accepted_force_time` (cumulative accepted h) for ReLU and `adaptive_population_flow_time` for SwiGLU. Their clocks and algorithms are not interchangeable.

`checkpoints` contains every required/saved checkpoint plus each exact loss-prefix endpoint. Missing required diagnostics are explicit rows with null scores and diagnostic_status=`not_evaluated`; plots must keep those as gaps. Available rows contain A_min,A_mean,A_top,raw refit and cutoff/bound endpoints, original AGOP and additional refit screens, raw gains, candidate flags, later loss minima and all warnings. High mean/top1 alignment never substitutes for A_min. Sparse/dense losses, initial-time origin, times, metadata configuration, normalization, available source hashes and dense counts are checked. ReLU state files and output receipts are hash-verified; SwiGLU raw files and snapshot archives are hash-verified and snapshot counts, shapes and times checked. Missing snapshot verification blocks extra-qualified evidence. Missing authoritative required-grid metadata cannot support a resolved grid negative. Unknown variances are never silently replaced by1.

`prefixes['1.01']` and `prefixes['1.05']` recompute the maximal initial positive-loss max/min windows from every update, with exact last-admissible step/time, first crossing, ratio, right-censoring and diagnosed-boundary status. There is no interpolation, tolerance expansion of the loss ratio, or feature-dependent window selection. No secondary later-window search is performed here.

## Original criterion and numerical qualification

The original frozen criterion requires the initial/current AGOP guards, delta A_min>=0.5 and initial-low minus current-high numerical refit improvement>=0.1Var(Y) at one saved checkpoint inside its loss window. `same_checkpoint_material_candidate_original`, `first_material_candidate`, `candidate_count`, and `original_criterion_assessment` retain this criterion. The historical pre-refit-check flag also remains explicitly named `legacy_material_candidate_before_refit_qualification`.

Additional qualification is separate: `same_checkpoint_material_candidate_numerically_qualified`, `first_numerically_qualified_candidate`, `qualified_candidate_count`, and `numerical_qualification_assessment`. ReLU qualification checks success, first-order optimizer tolerance and total numerical gap<=1e-7. SwiGLU qualification checks all four cutoff solves, normalized spread<=1e-7, positive equilibrated Gram ratio and normal-equation residual<=1e-7. Neither extra check overwrites original results. The original1e-9 relative spectral/gap guards remain unchanged. A subsequent raw loss drop>=0.1Var(Y) is measured from that same candidate at a strictly later update; `first_complete_sequence` and `first_qualified_complete_sequence` remain separate.

All raw refit risks must be nonnegative within numerical tolerance. The ReLU tolerance is1e-10 raw; SwiGLU uses the existing1e-8 variance-normalized tolerance across all cutoff risks. Raw values are retained. A tiny negative current endpoint is floored tozero only for conservative gain arithmetic, with `roundoff_risk_floor_for_criterion` and the unadjusted gain preserved; it cannot manufacture a threshold crossing. This does not change the0.1Var(Y) threshold.

`effect_size_eligibility='initial_refit_already_below_material_threshold'` applies only when the actual valid saved initial feasible upper risk is below0.1Var(Y). The arm remains in the denominator. The teacher's oracle affine residual is not used as a bound on the actual student's refit class. Unknown initial refit/guards produce unknown eligibility, not exclusion.

ReLU uses its fixed normalized L2=32/L1=64sqrt(r) bounded readout and saved numerical lower bounds. SwiGLU uses unrestricted cutoff solves: `cutoff_envelope_gain_raw` is conservative across those declared numerical solves, **not a bound on improvement of unrestricted optimal-head risk**. Every SwiGLU arm/cell carries certificate=`notformaloptimalriskbound`; comparisons must retain these different classes. A separate old-quadratic same-budget SwiGLU audit is preserved in `../audit/BOUNDED_SWIGLU_REFIT_OLD_H2.json`; it does not change the frozen primary comparator.

Missing or failed guards, unfinished windows, inconsistent metadata, optimizer/sensitivity warnings, invalid losses and absent diagnostics cannot become resolved negative evidence. A no-candidate result with a closed and fully diagnosed grid is explicitly limited to that saved grid. A candidate inside an open window is observable even though the endpoint remains censored.

Quadrature is a separate limitation. This frozen launcher does not request per-state SwiGLU doubled quadrature; an empty list means unchecked. ReLU's smooth-teacher preflight is not a candidate-specific doubled teacher-cross check. `quadrature_followup_audit_status` reports this explicitly. A saved doubled check also needs its tolerance review; presence alone is not a pass or a formal integration certificate.

## Validation

`test_summarize_coverage.py` has32 tests with synthetic edge fixtures for nonmonotone loss ratios, exact boundary exclusion, missing diagnostics/guards, open windows, no cross-checkpoint splicing, minimum-vs-top1 distinction, normalization, negative risk, low initial refit, criterion/qualification separation, process receipts and all60 unstarted arms. It also compares16 completed v6 runs to their original saved summaries, covering both engines,32 exact prefixes, candidate counts/steps and cutoff/lower gains. No models are evaluated.

```sh
python -m unittest discover -s . -p test_summarize_coverage.py -v
```

Historical comparison JSON/Markdown files live under `validation/`. They are labeled historical and do not imply the new60-arm batch ran. The two old smaller-scale cubic cases retain original5% positives while separately failing the stricter total-gap qualification; this difference is expected and tested. The root source/dispatch review and eventual input receipts remain necessary before publishing or interpreting new results.

Portable tests: the 25 self-contained synthetic edge tests run without old datasets. Two additional manifest integration tests skip with a clear reason when `../bundle/MANIFEST.json` is absent. The five historical regression methods in `SavedV6Regression` skip as a class if the original v6 inputs are unavailable; they continue to run locally against all 16 preserved runs. No old data need be copied into the shared repository.
