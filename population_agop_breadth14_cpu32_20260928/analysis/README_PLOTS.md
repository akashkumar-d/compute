# Saved-data scientific figures

`plot_coverage.py` renders the unified summary written by `summarize_coverage.py`. It imports no model or training code and never evaluates a model. It writes only below this `analysis/` directory and refuses to overwrite a nonempty output directory.

From `goal_followup_v7/breadth14/`, after creating a stable summary:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/legendkiller/anaconda3/bin/python3 analysis/plot_coverage.py \
  --summary analysis/SUMMARY.json --output analysis/rendered_final
```

The Python executable can be replaced by any environment containing NumPy and Matplotlib. No remote paths, machine allocation, or worker count are assumed. Summaries with `snapshot_usable=false` or `inputs_unchanged=false` are rejected. Each new invocation needs a fresh output directory.

## Figures and interpretation

Each student receives two seven-teacher pages for the full observed trajectories and two pages for the initial prefix zoom, saved as PNG and editable SVG, plus one multipage PDF per mode. Each teacher has three aligned rows: raw prediction MSE; minimum and mean AGOP alignment; and actual refit MSE in the student's saved readout class. Supplemental mixtures/products stay in the summary but are excluded from the canonical fourteen-link atlas and listed in provenance.

The page uses serif text, black row labels, navy loss (`#25395B`), teal minimum alignment (`#007C83`), dashed ochre mean alignment (`#BB7518`), and purple refit MSE (`#814B9C`). No top-one alignment is shown. Gaussian initialization scale, head learning-rate multiplier, dimensions, profiled intercept, and target variance are explicit.

Bold curves are the median of exactly the two planned seeds. Bands are the observed seed range, **not confidence intervals**. Faint curves and hollow endpoint markers show individual observed trajectories, including their tails. Each bold curve ends at its own cell's common observed support; there is no extrapolation, one-seed median, pooling of teachers, or pooling of students. An arm with no recorded history remains planned and prevents a paired median. Recipe or clock mismatches within a cell are rejected.

For plotting on different saved grids, interpolation is linear only between adjacent finite saved points. An expected but uncomputed diagnostic is NaN, with a red cross; neither an individual curve nor a median bridges that gap. Red ticks retain unresolved AGOP or refit numerical screens without silently deleting the observed values. Conflicting values at one physical clock time become NaN, rather than inventing an interpolation through a zero-clock update.

The initial zoom is selected solely from the initial loss histories: it ends at the smaller of the two saved 5% prefix endpoints and their common observed horizon. It never depends on alignment or refit performance. The darker and lighter backgrounds respectively mark the common observed 1% and 5% initial prefixes. Right-censored prefixes remain explicitly counted; shading does not imply their true exit was observed.

Each cell displays **original saved per-run material-criterion counts** and, when available, the separate numerically qualified counts. These are passed through from the summary, never inferred from median curves. Numerical qualification adds the summary's solver/sensitivity screens and does not replace an original positive result. Criterion counts appear in a dedicated header outside each data panel. Per-seed stop reasons, final clocks, complete prefix records, and process states remain in `PLOT_QA.json`; no text box obscures the plotted data.

Raw MSE is not variance-normalized. The renderer checks `loss_raw = target_variance × loss_over_target_variance` for every available loss pair. Because the canonical teacher normalization retains its raw mean and sets `E[Y²]=1`, `Var(Y)` can differ from one. Both students profile the intercept. ReLU uses the accepted force clock `Σh_k` and its bounded normalized readout class. SwiGLU uses adaptive population-flow time `ΣΔt_k`, the declared block-specific head rate, and unrestricted cutoff refits. The two refit classes and two clocks are not asserted to be equivalent. AGOP and refit numerical qualifications are not formal population certificates.

## Provenance and checks

`PLOT_PROVENANCE.json` records the summary, renderer, available underlying input, and every PNG/SVG/PDF/aggregate hash. Relocated summaries remain usable: original paths that cannot be rechecked locally are listed explicitly rather than claimed verified. The summary and renderer are checked again after rendering. `PLOT_QA.json` records each planned pair, common support, fixed zoom endpoint, finite diagnostic support, source criteria, unit checks, and visual review status. The saved `aggregate_*.npz` files expose the exact plotted arrays.

Pure saved-data tests run with:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/legendkiller/anaconda3/bin/python3 analysis/test_plot_coverage.py
```

These test missing-diagnostic gaps, duplicate clocks, common support, independent interpolation/median/range calculations, no extrapolation or one-seed median, raw/normalized units, loss-only zoom selection, retained stop reasons, and rejection of incompatible recipes or clocks. They perform zero model evaluations and zero training updates.

`plot_validation/` contains historical validation and explicitly labeled synthetic layout tests. Historical fixtures select the first two ascending seed identifiers in each saved cell, independently of metrics; exclusions and source hashes are recorded. Synthetic pages repeat historical curves only to stress the seven-cell layout and insert diagnostic gaps. They are **not new breadth14 results**. Earlier `*_v1` renders are retained development artifacts. The final refreshed render, review receipt, and exact hashes are identified in `plot_validation/PLOT_VALIDATION_RECEIPT.json` once visual review is complete.

The final historical validation is `plot_validation/historical_render_v3/`; the final seven-cell synthetic layout check is `plot_validation/synthetic_layout_v3/`. Historical v2 is explicitly invalid: the input-hash guard detected that its summaries were finalized during rendering. Its partial output is retained with `INVALID_RENDER.json`. The final versions use the frozen summaries and final renderer.

The annotation-placement revision is recorded in `plot_validation/annotation_fix/RECEIPT.json`. It supersedes the earlier plotting freeze and preserves the earlier source and all earlier renders.
