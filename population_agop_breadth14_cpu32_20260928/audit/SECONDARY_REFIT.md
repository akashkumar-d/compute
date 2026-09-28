# Fixed-budget SwiGLU refit check

This is a secondary audit of two **previously completed** quadratic-teacher development runs, not a new training experiment or a changed primary criterion. It checks the exact initialization and original first material1%-plateau candidate in each run; no checkpoint was reselected.

For each saved gate/value pair, let the centered feature be

`phi_j(X) = [SiLU(P_j·X_aug)(V_j·X_aug) - E[SiLU(P_j·X_aug)(V_j·X_aug)]] / (||P_j|| ||V_j||)`.

The augmented parameter norms include biases. Refit `sum_j beta_j phi_j(X)` with the same constraints at both times: `||beta||_2 <= 32` and `||beta||_1 <= 64 sqrt(r)`. The output intercept fits the teacher mean. These are the same feasible coefficient budgets applied to the changing normalized representation. The feature definition and budgets remain fixed.

The existing convex quadratic solver supplies feasible prediction risks and numerical lower optimization bounds. Both the original and doubled quadrature orders were evaluated. The reported gain uses the smallest initial lower bound and the largest candidate feasible risk across those orders.

|Previous run|Original candidate step|Minimum-alignment gain|Fixed-budget raw refit gain, numerical lower value|
|---|---:|---:|---:|
|Quadratic, slower-head SwiGLU, seed501|97|0.541920|0.62139134937|
|Quadratic, slower-head SwiGLU, seed502|58|0.517663|0.52365364568|

Both targets have variance1, so raw and variance-normalized gains coincide. Their loss and AGOP claims remain those of the prior saved-state audits. This refit check alone does not prove the alignment or loss-window claims anew.

The original cutoff-sensitive unrestricted refit values remain unchanged and separately labeled. A cutoff sensitivity envelope is not a formal lower bound on improvement of the unrestricted population optimum. The new controlled comparison gives stronger numerical same-budget evidence for these two examples, but still does not certify Gaussian integration error, imply high-probability success, or establish other teacher cases.

`BOUNDED_SWIGLU_REFIT_OLD_H2.json` contains every risk, coefficient norm, solver gap, quadrature order and input hash. `bounded_swiglu_refit.py` reproduces the audit under a60-second cap, with an analytic one-feature scaling/variance test. It completed in17.93seconds without training and preserved all original data. Apply any future bounded refit audit as a separately labeled measurement; do not overwrite the frozen primary outputs.
