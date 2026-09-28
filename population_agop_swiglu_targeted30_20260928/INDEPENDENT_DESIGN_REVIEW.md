# Separate SwiGLU follow-up: design review

Recommendation: a fixed **20-arm maximum development study**, reusing seeds 641/642 as paired controls, not a fresh confirmation cohort. Preserve the 60-arm screen and its original 1%/5% verdicts. This review uses saved data and source inspection only; no model evaluation, training, or network access.

## What the completed screen establishes

For each additive teacher, the affine residual fraction is `rho = (E[q²] − E[q]² − E[Zq]²)/(E[q²] − E[q]²)`. Frozen canonical moments give:

| Teacher | Var(Y) | Affine residual / Var(Y) | Observed stopping time, seeds 641/642 |
|---|---:|---:|---:|
| softplus | 0.0496427765 | 0.0792388681 | 125.012 / 114.480 |
| tanh | 1 | 0.0695300764 | 24.991 / 22.383 |
| erf | 1 | 0.0864171572 | 25.998 / 23.847 |

All six stopped at normalized loss approximately 0.1, above these residual energies. Thus they do not establish failure to learn the nonlinear residual. Terminal tanh/erf AGOP screens fail; their tiny reported A_min is unresolved, not evidence of a missing direction. Softplus passes the original guard but has small gaps (1.9–4.4e-9).

Cubic is a different case: initial 1% windows exited at t=238.40/315.94 and 5% windows at 290.89/419.45, without material candidates. The 5% assessments remain numerically unresolved. Terminal losses were 0.53555/0.33916 with screened A_min 0.00368/0.00379 and mean alignment 0.649/0.752. Seed641 hit the 710-second training cap at t=516.45; seed642 reached t=600.06 in 501 seconds. Extending time cannot turn their already closed initial windows into successes.

Both h4/h5 seeds reached t=600 in only 12 dt=50 updates (3.7–4.2 seconds), with loss approximately one and negligible refit change. Their initial windows are genuinely right-censored; weak small-scale high-order coupling is a plausible hypothesis, not established numerical evidence of an exact zero gradient.

## Fixed configuration grid

Every row uses both seeds 641/642; r=8, d=64, m=64, alpha=1, head_ratio=1, the same Gaussian draws, all P/V/a trained, and the same profiled intercept.

| Purpose | Teachers | Scale s | head_lr | Arms |
|---|---|---:|---:|---:|
| Lower stopping threshold, unchanged dynamics | softplus, tanh, erf | 0.1 | 0.01 | 6 |
| Matched long-horizon control | h3, h4, h5 | 0.1 | 0.01 | 6 |
| Test stronger initial high-order coupling | h3, h4, h5 | 0.3 | 0.01 | 6 |
| Isolate head-rate change at fixed larger scale | h3 | 0.3 | 0.001 | 2 |

All arms: t_max=3000, L_stop=0.01 (loss/Var(Y), not raw MSE), max_steps=20000, h=0.01, dt_max=50; retain checkpoint and quadrature settings 1.06/0.5/1.5 and 16/48/24/12, plus existing diagnostic priorities. Use fresh deterministic reruns; saved checkpoints do not contain a supported complete adaptive-run resume state. Verify unchanged-recipe loss/update histories through their original stopping point. Terminal checkpoint scheduling may differ after the old stop; do not assert identical post-stop diagnostic grids.

Bound each arm to **1800 seconds training + 240 diagnostics + 10 cleanup**, hard2050 seconds; global2300 seconds if all20 run in one wave on the verified32-CPU allocation, single-thread workers. This bounds cost; it does not promise t=3000. Cubic already needed 500–710 seconds near t=600, and later dt can shrink. A 5000-step cap could independently censor the extension; 20000 makes the explicit wall cap the likely limiter. Retain every cap and unfinished diagnostic.

## Interpretation and numerical constraints

Do not include s=0.5 in this first grid. Moving 0.1→0.3 already changes P, V, and head amplitudes together, the initial feature distribution and initial refit risk; it is not simply a faster clock. A 0.5 arm adds another mechanism without a matched necessity. Reassess only if the bounded 0.3 results motivate it.

head_lr=0.001 is a separate block-rate flow, not common-rate GD. The weighted invariant remains `a²/head_lr − ||V||²`. Its scaled head direction participates in the relative-update cap; neither physical speed nor step count scales predictably by ten. Larger gates and evolved norms can worsen finite-order quadrature and Gram conditioning. Before interpreting a new positive or stalled near-zero signal, use separately budgeted saved-state order doubling at initialization, a candidate, and terminal state; report raw loss/gradient and AGOP/refit sensitivity. Existing guards remain unchanged. A finite-order agreement is not a certificate; no training quadrature is silently changed mid-run.

Primary analysis remains the original initial 1%/5% all-update windows, same-checkpoint A_min gain≥0.5 and refit gain≥0.1Var(Y), with original and extra-qualified outcomes separate. Compare paired arms on common observed support, with raw and normalized losses; softplus uses raw-MSE gradients despite its smaller variance, so do not pool clocks across teachers.

Predeclare secondary physical windows **[150,1500]** for the stop-extension teachers and **[600,3000]** for Hermites. Use the first saved state at/after the left boundary, disclose its lag, and do not slide the window to favor alignment. Retain original initial-reference criteria as one separate column. For genuinely post-linear gains, 0.1Var(Y) exceeds the entire affine residual for softplus/tanh/erf: an additional gain of that size is structurally unavailable after affine fitting. Report later gain raw and divided by the fixed affine-residual energy; if a binary secondary test is needed, predeclare gain≥0.1×that residual energy plus A_min gain≥0.5 relative to the secondary anchor, with the same numerical screens and loss-only 1%/5% prefix rule inside that fixed window. Label it a different estimand, never an original-window success. Empty, exited, or censored secondary windows stay explicit.

Source: `cpu32/server_analysis/population_agop_breadth14_cpu32_20260928/analysis/cpu32_SUMMARY.json`; canonical moments and frozen runner under `cpu32/bundle/`. Summary SHA256: a3ae03952b380f6b56682e39711ba6bdd06ae50a23d6f674555e50bef8063516.

## Independent review of the actual v8 targeted30 bundle

**No blocking configuration or source issue found.** This actual 30-arm protocol supersedes the advisory 20-arm grid and proposed secondary physical windows above; use the already implemented longest loss-band rule unchanged instead of mixing the two definitions. The new study remains explicitly exploratory.

Independently checked the actual JSON configs against the v7 manifest, not only CONFIG_COMPARISONS.json: all30 configurations exactly match their embedded manifest records and per-file hashes; all15 cells contain both seeds641/642. Every scientific argument difference equals the declared comparison and belongs to {t_max,L_stop,max_steps,s,head_lr}. Outer rank and other non-label configuration fields are unchanged. Every SwiGLU Python source is byte-identical to v7. All68 source hashes verify; their config-path set is exactly the30 new config paths, with no stale parent paths. Reviewed manifest SHA256 `ce71d19034ba2c20b6d47b9a7ba78350cabb44ec9916c93dce028912e588681f`; CONFIG_COMPARISONS SHA256 `66f52cc892119812abd9ecaf84dc28a241ed50bccc7788d069607d0f9d7f6034`.

The scale0.5 arms have no demonstrated numerical defect requiring removal; retaining them as exploratory scale comparisons is reasonable. This review performed no numerical model evaluation and cannot validate their quadrature. Saved-state order/sensitivity checks are required before interpreting positives or apparent near-zero gradients; finite-order agreement remains non-certifying.

Two interpretation limits are material. First, unchanged710-second training budgets make t=3000 an allowed maximum, not a likely attained horizon: cubic seed641 already exhausted that budget at t≈516. Thus the new long-control may again be wall-censored before its old declared horizon. Second, h4/h5 lack new s=0.1 long controls; scale0.3 versus0.5 is matched within v8, while causal comparisons with s=0.1 beyond v7's t=600 are unavailable. The h2/abs/RBF head-rate comparisons likewise have only historical common observed support for their old-rate controls. These limits do not block a bounded screen, but they preclude claiming isolated long-horizon improvement from those historical comparisons.

Retain primary initial-prefix failures, every wall cap, and all numerical flags. Freeze the secondary existing rule (loss bands, longest-window selection, ties, minimum50 updates,10% eligible-clock and50% saved-clock coverage) before new outcomes; do not reuse its optimized endpoints across configurations. Its absolute0.1Var refit threshold remains unchanged, so a small affine-residual regime can be structurally ineligible. Continuous residual-relative effects must remain separately labeled, without converting them into primary successes. The protocol's 30 single-thread workers, verified32-CPU requirement,900-second per-arm accounting and1100-second global ceiling define a bounded one-wave search; the user's usage-triggered pause instruction still takes precedence.

This addendum is readiness advice, not an integration or quadrature certificate. No source/config changes, training, network calls, or extra agents were used for the review.
