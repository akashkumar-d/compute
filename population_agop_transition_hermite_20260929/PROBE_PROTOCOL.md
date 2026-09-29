# Fixed-state Hermite comparison protocol

This is a numerical-method diagnostic, not a learning experiment. It preserves
the original rank8, dimension64, width64 pure-h3 SwiGLU configuration, all neurons,
and the saved states at updates472 and2260. No training updates, hyperparameter
sweep, new random initialization or paper edits occur.

The comparison reference is the stored transition-quadrature base evaluation at
order16 and Gaussian cutoff12. Its directional checks and order/cutoff refinement
are documented in `../NUMERICAL_REPAIR_STATUS.md`; these do not prove exact
Gaussian integration. This diagnostic compares numerical approximations.

At both states, evaluate Hermite degree64/scalar order16, degree128/order16, and
degree128/order24. The **fixed final candidate is degree128/order24**; the other
two settings only assess numerical convergence. Retain every outcome. Its
diagonal and teacher calculations use the unchanged transition order16/cutoff12.

Before finite differences, require at both states:

- Loss discrepancy at most1e-7 times teacher variance.
- Weighted update discrepancy at most1e-4 of the reference update norm, plus
  1e-10 times width times teacher variance divided by max(1,parameter norm).
- Common-direction slope discrepancy at most1e-4 of the reference slope magnitude,
  plus1e-10 times teacher variance divided by max(1,parameter norm).
- No materially negative scalar residual-energy estimate under the core's fixed
  roundoff rule. This is only an inconsistency screen, not an error bound.

The weighted direction uses the original width and head learning-rate factor;
both candidates are compared along the stored reference unit direction. When
both final-candidate base comparisons pass, use the existing two central-
difference scales1e-4 and1e-5 times max(1,parameter norm) at each state. Each
relative slope mismatch must be at most0.001, and scalar-energy screens must
remain valid on the stencil. Degrees and orders do not adapt within the stencil.

Maximum14 full evaluations: six base calls and eight conditional stencil calls.
One server CPU, single-thread numerical libraries, niceness10; admission deadline
170seconds and external TERM175/KILL180. No automatic follow-on training. Save
returned losses/times, every base gradient and Gram array, ordered moment fields,
raw signed scalar-energy residuals and per-pair tail estimates. A failed base
screen completes the diagnostic with its failure recorded; it is not repaired
by weakening tolerances or choosing a different outcome-dependent setting.

Report runtime separately from accuracy. A subsecond evaluation is the engineering
target, not a measured result or scientific success criterion. This probe does
not validate the AGOP matrix, refit solve, neighboring states or a full trajectory.
Near-collinear pairs and Hermite truncation may still require a reviewed fallback.
